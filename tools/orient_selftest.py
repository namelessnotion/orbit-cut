"""Planted-orientation checks on rendering.

The bug this exists for was invisible in three separate ways at once, which is
why it survived 97 files and reached a finished reel.

The crop box is 9:16 by construction, so it scales to 1080x1920 with no
distortion whichever frame it lands on — the output looks clean and passes the
dimension check while showing the wrong part of the wrong-way-up picture. The
proxy stage and the render stage disagreed on the same file, so there was never
a single orientation to notice was wrong. And ffmpeg's autorotation of
complex-filtergraph inputs is version-dependent, so the failure does not
reproduce everywhere.

Nothing about that is caught by asserting on numbers. So plant a picture with
an unmistakable top — a white bar — stamp a display matrix on it, render it,
and look at where the white ends up.

    python tools/orient_selftest.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitcut import render as rn  # noqa: E402

W, H = 640, 360          # coded shape, landscape like the real files
BAR = 40


def make_source(path: Path, stamped: int, upside_down: bool) -> None:
    """A 3 s clip whose top edge is white — or whose bottom is, if inverted.

    `stamped` is the display matrix written into the container, which is the
    thing under test: it is allowed to be a lie.
    """
    top, bottom = ("white", "black") if not upside_down else ("black", "white")
    plain = path.with_name("plain_" + path.name)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=gray:s={W}x{H}:d=3:r=30",
         "-vf", (f"drawbox=x=0:y=0:w={W}:h={BAR}:color={top}:t=fill,"
                 f"drawbox=x=0:y={H - BAR}:w={W}:h={BAR}:color={bottom}:t=fill"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", str(stamped),
         "-i", str(plain), "-c", "copy", str(path)], check=True)
    plain.unlink()


def make_telemetry(path: Path, upside_down: bool) -> None:
    """Gravity for a camera in that attitude, at the library's strap angle."""
    n = 300
    g1 = -0.94 if upside_down else +0.94
    pd.DataFrame({
        "t": np.linspace(0, 3, n),
        "grav_0": np.full(n, 0.01),
        "grav_1": np.full(n, g1),
        "grav_2": np.full(n, 0.26),
    }).to_parquet(path)


def top_and_bottom(clip: Path) -> tuple[float, float]:
    """Mean luma of the top and bottom eighths of a rendered frame."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    img = np.frombuffer(raw, dtype=np.uint8).reshape(rn.TARGET_H, rn.TARGET_W)
    band = rn.TARGET_H // 8
    return float(img[:band].mean()), float(img[-band:].mean())


def check(name: str, ok: bool, detail: str) -> int:
    print(f"  {name:<52}{detail:<30}{'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    fails = 0
    tmp = Path(tempfile.mkdtemp(prefix="orient_"))

    # (label, what the container claims, whether the camera was really inverted)
    #
    # The last row is GX010600: a matrix that disagrees with the accelerometer.
    # The two rows above it are the shapes the rest of the library actually
    # takes, and they are here so a fix for 0600 cannot quietly break them.
    cases = [
        ("right way up, no matrix", 0, False),
        ("inverted, matrix ok (180)", 180, True),
        ("inverted, matrix lies (90)", 90, True),
    ]
    for label, stamped, inverted in cases:
        src = tmp / f"src_{stamped}_{int(inverted)}.mp4"
        tel = tmp / f"tel_{stamped}_{int(inverted)}.parquet"
        out = tmp / f"out_{stamped}_{int(inverted)}.mp4"
        make_source(src, stamped, inverted)
        make_telemetry(tel, inverted)

        meta = rn.probe(str(src))
        deg, why = rn.orientation(meta, str(tel))
        want = 180 if inverted else 0
        fails += check(f"{label}: orientation chosen", deg == want,
                       f"{deg}° from {why.split(' (')[0]}")

        rn.clip(str(src), 0.5, 2.5, out, hwaccel="none", telemetry=str(tel))
        top, bot = top_and_bottom(out)
        # Upright means the white bar is at the top of the rendered frame,
        # whatever the file claimed on the way in.
        fails += check(f"{label}: white bar ends up at the top", top > bot + 60,
                       f"top {top:.0f} vs bottom {bot:.0f}")

    # A 9:16 crop of a 16:9 frame keeps full height, so both bars survive the
    # crop and the test above is meaningful. Assert that rather than assume it.
    meta = rn.probe(str(tmp / "src_0_0.mp4"))
    cw, ch, _, _ = rn.crop_box(*rn.display_size(meta, 0))
    fails += check("the crop keeps full height, so both bars show", ch == H,
                   f"crop {cw}x{ch} of {W}x{H}")

    # Gravity that cannot answer must say so rather than guess. A sideways mount
    # is a real possibility and there is no file here to fix its sign against.
    lateral = tmp / "lateral.parquet"
    pd.DataFrame({"grav_0": [0.95] * 10, "grav_1": [0.05] * 10}).to_parquet(lateral)
    fails += check("a sideways mount declines to guess",
                   rn.upright_from_gravity(str(lateral)) is None, "returned None")
    flat = tmp / "flat.parquet"
    pd.DataFrame({"grav_0": [0.01] * 10, "grav_1": [0.02] * 10}).to_parquet(flat)
    fails += check("gravity with no direction declines too",
                   rn.upright_from_gravity(str(flat)) is None, "returned None")
    fails += check("no telemetry falls back to the container",
                   rn.orientation({"rotation": 90}, None) == (90, "container"),
                   "90° from container")

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
