"""Stitch a ride's candidate clips into one file you can watch straight through.

Phase 1 was validated by watching footage against the curve, and clip selection
deserves the same treatment before anything is built on top of it. Reading a
table of in and out points tells you nothing about whether a clip starts too
late or ends mid-corner.

Labels are drawn with PIL and composited, not with ffmpeg's `drawtext`. drawtext
needs a font file at a path that differs on every machine and fails at render
time when it is missing; PIL is already a dependency here for the overlay
playhead, and a pre-rendered PNG cannot fail halfway through a two-minute encode.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config

BAR_H = 34
INK = (17, 16, 14, 220)
TEXT = (244, 239, 227, 255)
ACCENT = {"speed": (226, 103, 58), "turn": (192, 132, 74),
          "rough": (150, 187, 111), "jump": (244, 239, 227),
          "unknown": (138, 146, 139)}


def _font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def label_png(text: str, kind: str, width: int, out: Path) -> Path:
    img = Image.new("RGBA", (width, BAR_H), INK)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 6, BAR_H], fill=(*ACCENT.get(kind, ACCENT["unknown"]), 255))
    d.text((14, BAR_H // 2), text, font=_font(), fill=TEXT, anchor="lm")
    img.save(out)
    return out


def _segment(proxy: str, t_in: float, dur: float, label: Path,
             out: Path, has_audio: bool) -> None:
    """Cut one clip and burn its label in.

    The trim lives in the filter graph rather than in `-ss`/`-t`, which is not a
    style choice. ffmpeg binds those to whichever input follows them, so with a
    second input for the label PNG they silently became options for the *image*
    and every clip came out as the whole ride: a 58-second reel rendered as 720
    seconds and reported success. `trim` attaches to a named stream and cannot
    be misread that way.

    `setpts=PTS-STARTPTS` rebases each part to zero so the concat demuxer does
    not see three clips all claiming to start at their original timestamps.
    """
    graph = (f"[0:v]trim=start={t_in:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS[v0];"
             f"[v0][1:v]overlay=0:H-h:eof_action=repeat[v]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", proxy, "-i", str(label)]
    maps = ["-map", "[v]"]
    if has_audio:
        graph += (f";[0:a]atrim=start={t_in:.3f}:duration={dur:.3f},"
                  f"asetpts=PTS-STARTPTS[a]")
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-filter_complex", graph, *maps,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clip render failed: {r.stderr.strip()[:300]}")


def _probe(path: str) -> tuple[int, int, bool]:
    """Width, height, and whether there is an audio stream to trim alongside."""
    import json
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_streams", path], capture_output=True, text=True)
    streams = json.loads(r.stdout or "{}").get("streams", [])
    size = next(((int(s["width"]), int(s["height"])) for s in streams
                 if s.get("codec_type") == "video" and s.get("width")), None)
    if size is None:
        raise RuntimeError(f"cannot read dimensions from {path}")
    return size[0], size[1], any(s.get("codec_type") == "audio" for s in streams)


def build(proxy_path: str, clips: list[dict], content_hash: str,
          ride: str) -> str:
    """Render every candidate back to back. Returns the output path."""
    if not clips:
        raise ValueError("no candidates to render")
    out_dir = config.derived_dir(content_hash)
    work = out_dir / "reel_parts"
    work.mkdir(exist_ok=True)
    width, _h, has_audio = _probe(proxy_path)

    parts = []
    for i, c in enumerate(clips, 1):
        text = (f"{i}/{len(clips)}  {ride}  {c['t_in']:.0f}-{c['t_out']:.0f}s"
                f"   {c['dominant']}   score {c['score']:.2f}")
        png = label_png(text, c.get("dominant", "unknown"), width,
                        work / f"label{i:02d}.png")
        part = work / f"clip{i:02d}.mp4"
        _segment(proxy_path, c["t_in"], c["t_out"] - c["t_in"], png, part, has_audio)
        parts.append(part)

    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    out = out_dir / "reel.mp4"
    # Concat demuxer rather than the concat filter: every part was encoded with
    # identical settings, so this is a stream copy and costs nothing.
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True, cwd=str(work))
    if r.returncode != 0:
        raise RuntimeError(f"concat failed: {r.stderr.strip()[:300]}")

    # Verify the length rather than trusting it. The -ss/-t bug produced a reel
    # 12x too long and reported success, so this is the cheapest possible guard
    # against the same class of mistake returning.
    want = sum(c["t_out"] - c["t_in"] for c in clips)
    got = _duration(out)
    if got and abs(got - want) > max(2.0, 0.1 * want):
        raise RuntimeError(
            f"reel is {got:.0f}s but the clips total {want:.0f}s — the trim did "
            f"not apply. Refusing to hand you a file that does not match its "
            f"own description.")
    return str(out)


def _duration(path: Path) -> float | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except (TypeError, ValueError):
        return None
