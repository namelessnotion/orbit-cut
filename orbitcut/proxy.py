"""Proxy generation — the slow half of ingest, and the reason everything
downstream is cheap. A 10-15 GB original becomes ~50 MB that every later stage
reads instead."""
from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path

from . import config

# Errors that mean the command line is wrong, not that the hardware is. Retrying
# these in software wastes a full decode and — worse — reports a hardware failure
# that never happened, burying the real message.
#
# Kept deliberately narrow: only errors that mean ffmpeg could not parse the
# command at all. A missing encoder or filter ("Encoder not found",
# "No such filter: scale_cuda") is a hardware-availability problem and *should*
# fall back — classifying those as fatal is how you turn a working software path
# into a hard failure on any machine without the accelerator.
_ARG_ERRORS = (
    "unrecognized option",
    "option not found",
    "error splitting the argument list",
)


@functools.lru_cache(maxsize=1)
def _fps_mode_flag() -> str:
    """`-vsync` was deprecated in ffmpeg 5 and removed in 8; `-fps_mode` replaced
    it. Pick whichever this build accepts rather than assuming a version."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        ).stdout
        m = re.search(r"ffmpeg version n?(\d+)", out)
        if m and int(m.group(1)) >= 5:
            return "-fps_mode"
    except (OSError, subprocess.SubprocessError):
        pass
    return "-vsync"


# Each backend degrades to the next one on failure rather than straight to
# software. That distinction matters: `videotoolbox_vt` only adds a hardware
# scaler to a path that already works, so if the scaler is missing the right
# answer is to lose the scaler, not to lose the GPU as well.
FALLBACK = {
    "videotoolbox_vt": "videotoolbox",
    "videotoolbox": "none",
    "cuda": "none",
    "none": None,
}


def _decode_args(hwaccel: str) -> list[str]:
    if hwaccel == "videotoolbox_vt":
        # Keep decoded frames in GPU memory so the scaler can run there too.
        # The format is `videotoolbox_vld`, not `videotoolbox` — the latter is
        # the hwaccel's name, not the pixel format's, and ffmpeg rejects it with
        # "Unrecognised hwaccel output format", which reads like the build lacks
        # support when in fact the argument was simply wrong.
        return ["-hwaccel", "videotoolbox",
                "-hwaccel_output_format", "videotoolbox_vld"]
    if hwaccel == "videotoolbox":
        # No output format: frames come back to system memory, where a software
        # `scale` runs. Costs CPU, but works on every build.
        return ["-hwaccel", "videotoolbox"]
    if hwaccel == "cuda":
        # Keep frames on the GPU: decode, scale and encode without a PCIe round trip.
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []


def _filter_and_encode(hwaccel: str, height: int) -> list[str]:
    if hwaccel == "cuda":
        return [
            "-vf", f"scale_cuda=-2:{height}",
            "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", config.PROXY_BITRATE,
        ]
    if hwaccel == "videotoolbox_vt":
        # scale_vt arrived in ffmpeg 6.1. On an older build this filter does not
        # exist and the run fails immediately — cheaply, before any decoding.
        return [
            "-vf", f"scale_vt=-2:{height}",
            "-c:v", "h264_videotoolbox", "-b:v", config.PROXY_BITRATE,
        ]
    if hwaccel == "videotoolbox":
        return [
            "-vf", f"scale=-2:{height}",
            "-c:v", "h264_videotoolbox", "-b:v", config.PROXY_BITRATE,
        ]
    return [
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
    ]


def build_command(src: Path, out: Path, hwaccel: str, height: int) -> list[str]:
    return [
        "ffmpeg", "-y", "-v", "error",
        *_decode_args(hwaccel),
        "-i", str(src),
        *_filter_and_encode(hwaccel, height),
        # Timing must survive intact — every clip in/out point is a timestamp
        # measured on the proxy and applied back to the original.
        _fps_mode_flag(), "passthrough",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(out),
    ]


def make_proxy(
    path: str | Path,
    content_hash: str,
    hwaccel: str | None = None,
    height: int | None = None,
) -> dict[str, str]:
    config.which_or_die("ffmpeg")
    hwaccel = hwaccel or config.HWACCEL
    height = height or config.PROXY_HEIGHT
    out = config.derived_dir(content_hash) / "proxy.mp4"

    cmd = build_command(Path(path), out, hwaccel, height)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return {"proxy_path": str(out)}

    stderr = result.stderr.strip()
    low = stderr.lower()

    # A malformed command fails identically on every backend, so falling back
    # would only produce a second identical failure under a misleading label.
    if any(marker in low for marker in _ARG_ERRORS):
        raise RuntimeError(f"ffmpeg rejected the command: {stderr[:400]}")

    nxt = FALLBACK.get(hwaccel)
    if nxt is not None:
        # Hardware paths fail for unglamorous reasons — an unsupported pixel
        # format, a missing filter, a busy GPU. Say what actually happened, and
        # step down one rung rather than abandoning the GPU entirely.
        first_line = stderr.splitlines()[0] if stderr else "no stderr"
        print(f"  ! {hwaccel} failed ({first_line[:120]}) — retrying with {nxt}")
        return make_proxy(path, content_hash, hwaccel=nxt, height=height)

    raise RuntimeError(f"ffmpeg failed: {stderr[:400]}")
