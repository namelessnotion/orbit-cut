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
