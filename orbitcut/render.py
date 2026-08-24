"""Stage 5 — approved clips out as 9:16 Reels, from the originals.

Two outputs, because a clip is worth having on its own and worth having in a
set: every approved clip renders standalone, and each ride's approved clips also
render in time order as one compilation.

**Rendered from the originals, never the proxies.** The proxies are 540p and
exist so that scoring and review are cheap; a Reel made from one would be a
540p source upscaled to 1080 wide. The proxy's only job here is to have told you
which seconds are worth the full decode.

**The crop is centred.** Subject-aware framing needs the dog detector, which is
phase 4. A centre crop is the honest version of "we do not know where the
subject is yet", and it is what the 8:7 capture setting was chosen for — the
sensor is tall so that the sides can go.

It can be rotated, though, because phase 0 measured roll suppression at +0.11
and so the mount's tilt is still in the pixels. `--level` picks what to do about
it and `level.py` measures it; the crop then shrinks by exactly as much as the
angle demands, which is why an unlevelled clip loses nothing at all.

Both source shapes work but not equally:

    8:7  3956x3460  ->  crop 1946x3460   2010 px of pan left over
    16:9 3840x2160  ->  crop 1214x2160   2626 px of pan, far less trail ahead

The 16:9 rides still clear 1080 wide, so they render — but a 9:16 slice of a
wide frame shows much less of what is coming, and most of the approved clips are
from those rides. Worth looking at before rendering all of them.

Instagram accepts 1080x1920, H.264/AAC in MP4, 24-60 fps, and a Reel over three
minutes is out of scope here by choice, so compilations are split into parts
rather than truncated.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import config, proxy as proxy_mod

TARGET_W, TARGET_H = 1080, 1920
MAX_ROTATE_DEG = 25.0
MAX_REEL_S = 175.0          # a little under three minutes, for safety
CRF = 20                    # visually lossless enough at 1080; Instagram re-encodes
AUDIO_KBPS = "128k"


def probe(path: str) -> dict[str, Any]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()[:200]}")
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams
              if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if not v:
        raise RuntimeError(f"no video stream in {path}")
    return {
        "width": int(v["width"]), "height": int(v["height"]),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "fps": v.get("r_frame_rate", "30/1"),
    }


def safe_scale(cw: int, ch: int, width: int, height: int, max_deg: float) -> float:
    """How far the crop must shrink so rotation never exposes a black corner.

    Rotating the frame and then taking the same crop pulls the frame's corners
    inward across the crop's edges. The crop's own corners, rotated by θ, span
    `w/2·sin + h/2·cos` vertically and `w/2·cos + h/2·sin` horizontally, and both
    must still fit. For an 8:7 source the crop is already full height, so even a
    5 degree correction overruns it — this is not a rounding allowance.
    """
    if max_deg <= 0.01:
        return 1.0
    a = np.radians(min(abs(max_deg), MAX_ROTATE_DEG))
    need_h = (cw / 2) * np.sin(a) + (ch / 2) * np.cos(a)
    need_w = (cw / 2) * np.cos(a) + (ch / 2) * np.sin(a)
    return float(min(1.0, (height / 2) / need_h, (width / 2) / need_w))


def crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Centred 9:16 window inside a frame of this shape.

    Dimensions are forced even: H.264 with yuv420p cannot encode an odd width or
    height, and the failure arrives from the encoder rather than from here.
    """
    if width * TARGET_H > height * TARGET_W:      # wider than 9:16 — trim sides
        cw, ch = round(height * TARGET_W / TARGET_H), height
    else:                                         # taller — trim top and bottom
        cw, ch = width, round(width * TARGET_H / TARGET_W)
    cw = min(width, cw - (cw % 2))
    ch = min(height, ch - (ch % 2))
    return cw, ch, (width - cw) // 2, (height - ch) // 2


def _encoder(hwaccel: str) -> list[str]:
    if hwaccel.startswith("videotoolbox"):
        # Quality-based rate control on VideoToolbox, which does not take -crf.
        return ["-c:v", "h264_videotoolbox", "-q:v", "55", "-profile:v", "high"]
    if hwaccel == "cuda":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(CRF)]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(CRF)]


def rotation_budget(cw: int, ch: int, width: int, height: int) -> float:
    """Largest rotation whose shrunken crop still delivers 1080 wide.

    Levelling trades pan margin for horizon. On a 16:9 source there is much less
    to trade — past about 15 degrees the crop drops under 1080 and the render
    would be upscaling. Better to level less than to invent pixels, so the angle
    is clamped here and the clamp is reported rather than applied silently.
    """
    lo, hi = 0.0, MAX_ROTATE_DEG
    for _ in range(24):
        mid = (lo + hi) / 2
        if int(cw * safe_scale(cw, ch, width, height, mid)) >= TARGET_W:
            lo = mid
        else:
            hi = mid
    return lo


_ROLL: dict[str, tuple] = {}


def roll_for(src: str, telemetry: str | None) -> tuple:
    """(t, roll) for a source file, with the sign settled against its own pixels.

    Cached per source: the sign check decodes a dozen frames, and every approved
    clip from a ride would otherwise repeat it. Returns empty arrays when the
    sign cannot be established — levelling the wrong way doubles the tilt
    instead of removing it, and that failure is silent, so it is declined here
    rather than guessed at.
    """
    if src in _ROLL:
        return _ROLL[src]
    import pandas as pd

    from . import level as lv

    empty = (np.array([]), np.array([]))
    if not telemetry or not Path(telemetry).exists():
        _ROLL[src] = empty
        return empty
    t, roll = lv.roll_series(pd.read_parquet(telemetry))
    if not len(roll):
        print("    ! no usable GRAV arc — rendering unlevelled")
        _ROLL[src] = empty
        return empty
    v = lv.verify_sign(src, t, roll)
    if not v.get("usable"):
        print(f"    ! roll sign unverified ({v.get('reason', '')}) — "
              f"rendering unlevelled")
        _ROLL[src] = empty
        return empty
    if v["sign"] < 0:
        roll = -roll
    print(f"    roll sign {v['sign']:+d} (corr {v['corr']:+.2f} "
          f"over {v['frames']} frames)")
    _ROLL[src] = (t, roll)
    return _ROLL[src]


def clip(src: str, t_in: float, t_out: float, out: Path,
         hwaccel: str | None = None, level: str | None = None,
         telemetry: str | None = None) -> Path:
    """One approved clip as a 1080x1920 Reel.

    `level` is None, "constant" (one rotation for the clip) or "dynamic"
    (per-frame). Both need telemetry; without it the clip renders unlevelled
    rather than failing, because an unlevelled Reel is still a Reel.
    """
    hwaccel = hwaccel or config.HWACCEL
    meta = probe(src)
    cw, ch, _x, _y = crop_box(meta["width"], meta["height"])
    dur = max(t_out - t_in, 0.1)

    rot_graph, cmd_file, applied = "", None, 0.0
    if level:
        from . import level as lv
        tt, roll = roll_for(src, telemetry)
        if len(roll):
            span = (tt >= t_in) & (tt < t_out) & np.isfinite(roll)
            if span.sum() > 4:
                budget = rotation_budget(cw, ch, meta["width"], meta["height"])
                if level == "constant":
                    ang = float(np.clip(np.median(roll[span]), -budget, budget))
                    # Below the threshold the tilt is not visible but the crop
                    # cost is real, so a near-square mount pays nothing.
                    if abs(ang) < lv.MIN_CONSTANT_DEG:
                        ang = 0.0
                    applied = abs(ang)
                    if applied:
                        rot_graph = f"rotate={-np.radians(ang):.6f}:ow=iw:oh=ih,"
                else:
                    sm = np.clip(lv.smoothed(tt, roll), -budget, budget)
                    applied = float(np.max(np.abs(sm[span])))
                    cmd_file = out.with_suffix(".cmds")
                    cmd_file.write_text(lv.sendcmd(tt, sm, t_in, t_out))
                    rot_graph = (f"sendcmd=f='{cmd_file.as_posix()}',"
                                 f"rotate=0:ow=iw:oh=ih,")

    # Shrink the crop to whatever the applied rotation demands. Doing this after
    # the angle is known, rather than reserving a fixed margin, means an
    # unlevelled clip loses nothing at all.
    k = safe_scale(cw, ch, meta["width"], meta["height"], applied)
    cw2, ch2 = int(cw * k) & ~1, int(ch * k) & ~1
    x2, y2 = (meta["width"] - cw2) // 2, (meta["height"] - ch2) // 2

    # trim in the filter graph, not via -ss/-t: with more than one input those
    # bind to whichever input follows them, which once produced a reel twelve
    # times too long that reported success.
    graph = (f"[0:v]trim=start={t_in:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS,"
             f"{rot_graph}"
             f"crop={cw2}:{ch2}:{x2}:{y2},"
             f"scale={TARGET_W}:{TARGET_H}:flags=lanczos,"
             f"setsar=1[v]")
    maps = ["-map", "[v]"]
    if meta["has_audio"]:
        graph += (f";[0:a]atrim=start={t_in:.3f}:duration={dur:.3f},"
                  f"asetpts=PTS-STARTPTS[a]")
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", AUDIO_KBPS, "-ar", "48000"]

    cmd = ["ffmpeg", "-y", "-v", "error",
           *proxy_mod._decode_args(hwaccel if hwaccel != "videotoolbox_vt"
                                   else "videotoolbox"),
           "-i", src, "-filter_complex", graph, *maps, *_encoder(hwaccel),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and hwaccel != "none":
        # Same ladder as proxy generation: step down rather than abandoning
        # hardware entirely on the first complaint.
        nxt = proxy_mod.FALLBACK.get(hwaccel) or "none"
        first = (r.stderr.strip().splitlines() or ["no stderr"])[0]
        print(f"  ! {hwaccel} failed ({first[:110]}) — retrying with {nxt}")
        return clip(src, t_in, t_out, out, nxt, level, telemetry)
    if r.returncode != 0:
        raise RuntimeError(f"render failed: {r.stderr.strip()[:300]}")
    if cmd_file:
        cmd_file.unlink(missing_ok=True)
    _check(out, dur)
    return out


def compile_reel(parts: list[Path], out: Path) -> Path:
    """Join rendered clips. Every part shares an encode, so this is a copy."""
    listing = out.parent / f"{out.stem}_parts.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listing.name, "-c", "copy", "-movflags", "+faststart", out.name],
        capture_output=True, text=True, cwd=str(out.parent))
    if r.returncode != 0:
        raise RuntimeError(f"concat failed: {r.stderr.strip()[:300]}")
    return out


def split_by_budget(clips: list[dict], budget: float = MAX_REEL_S) -> list[list[dict]]:
    """Group clips into reels that each fit the length limit.

    Truncating to the limit would cut a clip mid-action, so a clip that does not
    fit starts the next reel instead. A single clip longer than the budget still
    gets its own reel rather than being dropped — that is a selection problem,
    and silently discarding approved footage would be worse than a long file.
    """
    reels: list[list[dict]] = []
    cur: list[dict] = []
    used = 0.0
    for c in clips:
        d = c["t_out"] - c["t_in"]
        if cur and used + d > budget:
            reels.append(cur)
            cur, used = [], 0.0
        cur.append(c)
        used += d
    if cur:
        reels.append(cur)
    return reels


def _check(path: Path, want: float) -> None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=width,height",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        data = json.loads(r.stdout or "{}")
        got = float(data["format"]["duration"])
        v = next(s for s in data["streams"] if s.get("width"))
    except (KeyError, ValueError, StopIteration, json.JSONDecodeError):
        return
    if abs(got - want) > max(1.0, 0.15 * want):
        raise RuntimeError(f"{path.name} is {got:.1f}s but should be {want:.1f}s")
    if (int(v["width"]), int(v["height"])) != (TARGET_W, TARGET_H):
        raise RuntimeError(f"{path.name} is {v['width']}x{v['height']}, "
                           f"expected {TARGET_W}x{TARGET_H}")
