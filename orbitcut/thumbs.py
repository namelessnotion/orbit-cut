"""Contact sheet — fifteen frames spread across the ride, tiled into one image.

Cheap because it reads the proxy, not the original. Useful three times over:
you can see at a glance whether a file is worth anything, whether the rotation
landed the right way up, and later it gives the review UI something to show
before a clip is scrubbed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config

COLS, ROWS = 5, 3
TILE_WIDTH = 320


def make_contact_sheet(
    proxy_path: str | Path,
    content_hash: str,
    duration_s: float,
) -> dict[str, str]:
    config.which_or_die("ffmpeg")
    out = config.derived_dir(content_hash) / "contact.jpg"
    frames = COLS * ROWS

    # Sample evenly across the whole file rather than the first N frames, which
    # on a ride would all be the trailhead. Nudge inward so the last tile is not
    # the moment the camera was already being switched off.
    span = max(duration_s * 0.98, 1.0)
    rate = f"{frames}/{span:.3f}"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(proxy_path),
        "-vf", f"fps={rate},scale={TILE_WIDTH}:-2,tile={COLS}x{ROWS}",
        "-frames:v", "1", "-q:v", "4",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"contact sheet failed: {result.stderr.strip()[:300]}")

    return {"contact_path": str(out)}
