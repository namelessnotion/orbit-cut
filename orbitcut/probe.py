"""ffprobe wrapper. Answers 'what is this file' without decoding a single frame."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import config


def _aspect_label(w: int, h: int) -> str:
    if not w or not h:
        return "unknown"
    r = w / h
    for label, target in (("8:7", 8 / 7), ("16:9", 16 / 9), ("4:3", 4 / 3), ("9:16", 9 / 16)):
        if abs(r - target) < 0.02:
            return label
    return f"{r:.2f}:1"


def _bit_depth(pix_fmt: str) -> int:
    if not pix_fmt:
        return 8
    if "p10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt:
        return 10
    if "p12" in pix_fmt:
        return 12
    return 8


def probe(path: str | Path) -> dict[str, Any]:
    config.which_or_die("ffprobe")
    out = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    meta = json.loads(out)

    fmt = meta.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    streams = meta.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    # GoPro writes telemetry as a data stream tagged 'gpmd'. Its presence is the
    # single most important thing this function reports.
    gpmd = next(
        (s for s in streams
         if s.get("codec_tag_string") == "gpmd"
         or "gopro met" in (s.get("tags", {}).get("handler_name", "") or "").lower()),
        None,
    )

    fps = 0.0
    if video.get("r_frame_rate"):
        try:
            fps = float(Fraction(video["r_frame_rate"]))
        except (ZeroDivisionError, ValueError):
            fps = 0.0

    # An upside-down mount is either baked into the pixels or corrected by a
    # display matrix that players honour and ffmpeg applies by default. Which
    # one it is changes what every later stage sees, so record it.
    rotation = 0
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    pix_fmt = video.get("pix_fmt") or ""

    model = (
        tags.get("com.apple.quicktime.model")
        or tags.get("model")
        or tags.get("major_brand")
        or None
    )

    return {
        "bytes": int(fmt.get("size") or 0),
        "duration_s": float(fmt.get("duration") or 0.0),
        "container": fmt.get("format_name"),
        "bitrate_bps": int(fmt.get("bit_rate") or 0),
        "recorded_at": tags.get("creation_time"),
        "camera_model": model,
        "vcodec": video.get("codec_name"),
        "width": width,
        "height": height,
        "aspect": _aspect_label(width, height),
        "fps": round(fps, 3),
        "pix_fmt": pix_fmt,
        "bit_depth": _bit_depth(pix_fmt),
        "rotation": rotation,
        "has_gpmd": 1 if gpmd else 0,
    }
