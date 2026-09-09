"""Watch a ride with its own score curve running underneath it.

This is the whole point of phase 1. Every number in `score.py` is a hypothesis
about what looks exciting, and the only way to test a hypothesis about taste is
to sit and watch footage against it. If the curve peaks where you would have
reached for the scrubber, the premise holds. If it peaks on a fire road, the
weights are wrong and you have found that out for the price of one render.

The strip is drawn once as an image and composited by ffmpeg with a moving
playhead, so this costs one pass over a 540p proxy rather than a render per
frame.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config

STRIP_H = 150
INK = "#11100E"
COMPOSITE = "#E2673A"
TAN = "#C0844A"
MOSS = "#96BB6F"
WHITE = "#F4EFE3"
MUTED = "#8A928B"


def _rgba(hex_colour: str, alpha: int) -> tuple[int, int, int, int]:
    h = hex_colour.lstrip("#")
    return (*(int(h[i:i + 2], 16) for i in (0, 2, 4)), alpha)


def draw_strip(scored: pd.DataFrame, events: pd.DataFrame | None,
               width: int, out: Path) -> Path:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, STRIP_H / dpi), dpi=dpi)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)

    t = scored["t"].to_numpy()
    comp = scored["composite_s"].to_numpy()

    # sub-scores, faint, behind — so you can see *which* one is driving a peak
    for col, colour in (("s_rough", MOSS), ("s_turn", TAN), ("s_speed", MUTED)):
        if col in scored and np.isfinite(scored[col]).any():
            ax.plot(t, scored[col], color=colour, lw=0.8, alpha=0.45)

    ax.fill_between(t, 0, comp, color=COMPOSITE, alpha=0.22)
    ax.plot(t, comp, color=COMPOSITE, lw=1.8)

    # airtime gets its own marker: it is an event, not a level
    if events is not None and len(events):
        for _, e in events.iterrows():
            ax.axvline(e["t_start"], color=WHITE, lw=1.2, alpha=0.8)
            ax.text(e["t_start"], 1.03, f"{e['duration']:.2f}s",
                    color=WHITE, fontsize=7, ha="center", va="bottom")

    if np.isfinite(comp).any():
        thresh = float(np.nanpercentile(comp, 85))
        ax.axhline(thresh, color=WHITE, lw=0.7, ls=(0, (4, 4)), alpha=0.5)

    ax.set_xlim(t[0], t[-1] if len(t) > 1 else 1)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelsize=7, length=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color=MUTED, alpha=0.15, lw=0.5)
    fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0.18)
    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    return out


def _unreadable(path: str, detail: str) -> str:
    return (
        f"cannot read the proxy: {path}\n"
        f"  {detail}\n"
        f"  A truncated proxy does this, and the ffmpeg 8 `-vsync` failure left\n"
        f"  some behind before that was fixed. Rebuild it:\n"
        f"      orbitcut ingest <the original .MP4> --force"
    )


def _video_size(path: str) -> tuple[int, int]:
    """Dimensions of the first real video stream.

    Deliberately JSON rather than `-of csv=p=0`. Positional CSV looks simpler
    and is a liability: any second section — cover art, a thumbnail, a stream
    reporting only some of the requested fields — turns the output into
    something `split(",")` mis-parses several fields later, where the error
    names an int conversion rather than the actual problem. JSON is
    self-describing, so a surprising stream is skipped rather than shifting
    everything after it.
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-select_streams", "v", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(_unreadable(path, r.stderr.strip()[:200]))
    try:
        streams = json.loads(r.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise RuntimeError(_unreadable(path, str(exc))) from exc

    for st in streams:
        # Skip cover art: it is a video stream by codec type but not by intent.
        if st.get("disposition", {}).get("attached_pic"):
            continue
        if st.get("width") and st.get("height"):
            return int(st["width"]), int(st["height"])
    raise RuntimeError(_unreadable(path, f"{len(streams)} video stream(s), none usable"))


def render(proxy_path: str, scored: pd.DataFrame, events: pd.DataFrame | None,
           content_hash: str, duration_s: float) -> str:
    out_dir = config.derived_dir(content_hash)
    w, h = _video_size(proxy_path)

    strip = draw_strip(scored, events, w, out_dir / "strip.png")
    out = out_dir / "overlay.mp4"

    # The playhead is a 3px image swept across by `overlay`, not a drawbox.
    # drawbox looks like the obvious tool and is a trap: its `t` is thickness,
    # not time, so an x expression using `t` silently produces no box at all.
    # overlay's x/y expressions do expose the timestamp.
    bar = out_dir / "playhead.png"
    Image.new("RGBA", (3, STRIP_H), _rgba(COMPOSITE, 245)).save(bar)
    sweep = f"t/{max(duration_s, 0.001):.4f}*{w}"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", proxy_path, "-i", str(strip), "-i", str(bar),
        "-filter_complex",
        f"[0:v]pad=iw:ih+{STRIP_H}:0:0:color={INK}[bg];"
        f"[bg][1:v]overlay=0:{h}[stacked];"
        f"[stacked][2:v]overlay=x='{sweep}':y={h}[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy", "-movflags", "+faststart",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"overlay render failed: {r.stderr.strip()[:400]}")
    return str(out)


# --------------------------------------------------------------- subject framing
def render_track(row, track: "pd.DataFrame", t_in: float, t_out: float,
                 out: Path | None = None, step_s: float = 0.05) -> str:
    """Watch the boxes and the solved crop window, drawn on the proxy.

    This exists to be looked at *before* `render.py` learns to pan, and that
    ordering is the point: if the detector cannot find Orbit, or the crop path
    lurches, one pass over a 540p proxy says so — where finding the same thing
    from a finished Reel costs a full-resolution decode per attempt and tells
    you less, because the crop has already thrown away everything outside it.

    Three boxes, and the third is the one worth watching. The detection is where
    the model says he is. The crop window is what the Reel would keep. The band
    is the dead zone around the crop's centre: while he is inside it the camera
    is entitled to sit perfectly still, so a crop that moves anyway, or one that
    sits still while he is outside it, is visible as such rather than inferred
    from a number.

    Everything is driven from one `sendcmd` script, because `drawbox`'s x, y, w,
    h, colour and thickness all carry ffmpeg's runtime-command flag. So this is
    still one pass and one encode, whatever the path does.
    """
    from . import reframe as rf, render as rn, track as tk

    proxy = row["proxy_path"]
    pw, ph = _video_size(proxy)
    delta, _ = tk.orientation_delta(row)

    meta = rn.probe(row["source_path"]) if row["source_path"] else None
    deg = (int(row["rotation"] or 0) + delta) % 360
    fw, fh = (rn.display_size(meta, deg) if meta
              else (int(row["width"]), int(row["height"])))
    cw, ch, _, _ = rn.crop_box(fw, fh)

    sl = track[(track["t"] >= t_in - rf.PAD_S) & (track["t"] <= t_out + rf.PAD_S)]
    grid, x, report = rf.crop_path(sl["t"].to_numpy(), sl["u"].to_numpy(),
                                   sl["v"].to_numpy(), t_in, t_out, fw, fh, cw)

    # Proxy pixels, not source pixels. The two differ by one scale factor and
    # nothing else, because the crop is full height in every shape here.
    k = pw / fw
    band_px = rf.DEAD_ZONE * cw * k

    steps = np.arange(0.0, max(t_out - t_in, step_s), step_s)
    have = sl.dropna(subset=["u"])
    lines = []
    for g in steps:
        t = t_in + g
        cx = float(np.interp(t, grid, x)) * k          # crop left, proxy px
        cwp = cw * k
        # The detection, when there is one close enough in time to be this
        # frame's rather than a neighbour's.
        near = have.iloc[(have["t"] - t).abs().to_numpy().argmin()] if len(have) else None
        fresh = near is not None and abs(float(near["t"]) - t) <= 1.0 / 5.0
        if fresh:
            bx, by = float(near["x0"]) * pw, float(near["y0"]) * ph
            bw, bh = (float(near["x1"]) - float(near["x0"])) * pw, \
                     (float(near["y1"]) - float(near["y0"])) * ph
        else:
            bx = by = bw = bh = 0                       # w=0 draws nothing
        # One interval per step. Within an interval commands are separated by
        # commas and the interval itself is closed by a semicolon — a semicolon
        # between commands instead reads as "next interval starts at
        # `drawbox@dog`", which ffmpeg reports as an invalid start time.
        cmds = [f"drawbox@dog x {bx:.0f}", f"drawbox@dog y {by:.0f}",
                f"drawbox@dog w {bw:.0f}", f"drawbox@dog h {bh:.0f}",
                f"drawbox@crop x {cx:.0f}", f"drawbox@crop w {cwp:.0f}",
                f"drawbox@band x {cx + cwp / 2 - band_px:.0f}",
                f"drawbox@band w {2 * band_px:.0f}",
                # Thick while tracking, thin while holding through a gap, so a
                # dropout is visible without reading a log.
                f"drawbox@crop t {4 if fresh else 2}"]
        lines.append(f"{g:.3f} " + ", ".join(cmds) + ";\n")

    # Named for the clip, not the ride: two clips from one ride would otherwise
    # write to the same file and the second would silently replace the first.
    out = (Path(out) if out else config.derived_dir(row["content_hash"])
           / f"track_{int(t_in):04d}_{int(t_out):04d}.mp4")
    cmds = out.with_suffix(".cmds")
    cmds.write_text("".join(lines))
    graph = (f"[0:v]trim=start={t_in:.3f}:duration={max(t_out - t_in, 0.1):.3f},"
             f"setpts=PTS-STARTPTS,{rn._orient_graph(delta)}"
             f"sendcmd=f='{cmds.as_posix()}',"
             f"drawbox@dog=x=0:y=0:w=0:h=0:color={MOSS}@0.9:t=3,"
             f"drawbox@crop=x=0:y=0:w=0:h={ph}:color={COMPOSITE}@0.8:t=4,"
             f"drawbox@band=x=0:y=0:w=0:h={ph}:color={MUTED}@0.35:t=2[v]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", proxy,
           "-filter_complex", graph, "-map", "[v]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    cmds.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"track overlay failed: {r.stderr.strip()[:400]}")
    return str(out), report
