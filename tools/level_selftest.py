"""End-to-end check on horizon levelling, against footage with a known tilt.

Levelling has two ways to be wrong that both look like success. It can measure
the tilt correctly and rotate the wrong way, doubling it. And it can rotate the
right way but not shrink the crop enough, pulling the frame's own corners into
view as black wedges. Neither shows up in an exit code, and both are obvious the
moment somebody watches the Reel, which is too late.

So this plants a tilt rather than trusting one. A striped scene is rotated by a
known angle to make the source, telemetry is synthesised from the same angle,
and then the pipeline is asked to undo it. Three things get measured on the
result: what came back out of `roll_series`, what tilt is left in the pixels,
and whether any corner went black.

    python tools/level_selftest.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitcut import level as lv, render as rn   # noqa: E402

FPS = 6
DUR = 5.0
PITCH_DEG = 13.0      # chest mount, the case that broke the first attempt


def scene(w: int, h: int) -> np.ndarray:
    """Trees and a horizon: vertical bars over a band, nothing near black.

    Bright everywhere on purpose — after this, any black pixel in a rendered
    frame came from a rotation exposing a corner, so the corner test needs no
    tolerance and no judgement.
    """
    img = np.full((h, w), 210, dtype=np.uint8)
    img[int(h * 0.45):, :] = 150                      # ground
    xs = np.arange(w)
    img[:, (xs // 40) % 2 == 0] = np.minimum(img[:, (xs // 40) % 2 == 0], 90)
    img[int(h * 0.44):int(h * 0.46), :] = 40          # horizon line
    return img


def make_source(path: Path, w: int, h: int, expr: str) -> None:
    """A video of the scene tilted by `expr` radians, with no black introduced.

    The scene is drawn oversized and rotated before cropping down to the target
    frame, so the source itself never has an exposed corner to confuse the test.
    """
    big = int(max(w * 0.94 + h * 0.35, h * 0.94 + w * 0.35)) | 1
    raw = scene(big, big)
    png = path.with_suffix(".png")
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-s", f"{big}x{big}", "-i", "-",
                        "-frames:v", "1", str(png)], input=raw.tobytes(),
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[:300])
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-r", str(FPS), "-i", str(png),
         "-t", str(DUR),
         "-vf", f"rotate={expr}:ow=iw:oh=ih,crop={w}:{h},format=yuv420p",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(path)],
        capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:300])
    png.unlink(missing_ok=True)


def telemetry(roll_deg: np.ndarray, t: np.ndarray, order: tuple[int, int, int]
              ) -> pd.DataFrame:
    """Gravity as a camera with this pitch and roll would report it.

    `order` says which raw column is down, right and forward, because the point
    of the SVD was that the answer must not depend on that — the test runs every
    permutation and expects the same number back.
    """
    d, rgt, fwd = order
    p = np.radians(PITCH_DEG)
    a = np.radians(roll_deg)
    g = np.zeros((len(a), 3))
    g[:, fwd] = np.sin(p)
    g[:, d] = np.cos(p) * np.cos(a)
    g[:, rgt] = np.cos(p) * np.sin(a)
    return pd.DataFrame({"t": t, "grav_x": g[:, 0], "grav_y": g[:, 1],
                         "grav_z": g[:, 2]})


def corners(path: Path, when: float, box: int = 24) -> int:
    """Darkest pixel found in the four corners of a frame."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{when:.2f}", "-i", str(path),
                        "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    w, h = rn.TARGET_W, rn.TARGET_H
    img = np.frombuffer(r.stdout[:w * h], dtype=np.uint8).reshape(h, w)
    return int(min(img[:box, :box].min(), img[:box, -box:].min(),
                   img[-box:, :box].min(), img[-box:, -box:].min()))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="orbitcut_level_"))
    # The encoder is not what is under test; a slow preset just makes the
    # check too tedious to run, which is how checks stop being run.
    rn._encoder = lambda _hw: ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"]
    t = np.arange(0.0, DUR, 1 / 30)
    fails = 0

    # 1. Measurement, under every axis assignment. A number that moves when the
    #    columns are relabelled is not a measurement.
    print("  measuring a planted 4.0 deg offset with a +/-12 deg swing")
    # Whole periods, or the median of the swing is not zero and the offset the
    # test is checking comes back shifted by the leftover part of a cycle.
    t1 = np.arange(120) / 30.0
    planted = 4.0 + 12.0 * np.sin(2 * np.pi * t1 / 2.0)
    for order in [(0, 1, 2), (1, 0, 2), (2, 0, 1), (0, 2, 1)]:
        _tt, roll = lv.roll_series(telemetry(planted, t1, order))
        s = lv.summarise(_tt, roll)
        if not s["usable"]:
            print(f"    down={order[0]} right={order[1]} fwd={order[2]}: unusable")
            fails += 1
            continue
        corr = float(np.corrcoef(planted, roll)[0, 1])
        ok = abs(abs(s["constant_deg"]) - 4.0) < 0.5 and abs(corr) > 0.999
        fails += 0 if ok else 1
        print(f"    down={order[0]} right={order[1]} fwd={order[2]}: "
              f"offset {s['constant_deg']:+.2f} deg  swing {s['spread_deg']:.1f} deg "
              f"corr {corr:+.3f}  {'ok' if ok else 'FAIL'}")

    # 2. The full path, on both source shapes: does the tilt come out of the
    #    pixels, and does anything go black doing it?
    for name, (w, h) in (("8:7", (3956, 3460)), ("16:9", (3840, 2160))):
        src = tmp / f"src_{name.replace(':', '')}.mp4"
        # A constant lean plus a swing, so constant and dynamic differ.
        make_source(src, w, h, "6*PI/180 + 9*PI/180*sin(2*PI*t/4.0)")
        roll = 6.0 + 9.0 * np.sin(2 * np.pi * t / 4.0)
        tel = tmp / f"tel_{name.replace(':', '')}.parquet"
        telemetry(roll, t, (1, 0, 2)).to_parquet(tel)

        probes = np.linspace(0.35, DUR - 0.45, 9)
        # Judged as a fraction of what the source reads, not in degrees.
        # `frame_tilt` is an orientation estimator, not a protractor: on these
        # hard-edged synthetic stripes it returns about 1.33x the planted angle
        # (5 deg reads 6.7, 10 reads 13.3, -8 reads -10.6 — a gain, not an
        # offset, from aliasing along near-vertical edges). Levelling is judged
        # by how much of the tilt is gone, which that gain divides out of.
        before = np.array([lv.frame_tilt(str(src), s)[0] for s in probes])
        print(f"\n  {name} {w}x{h} — source reads {before.mean():+.2f} deg mean, "
              f"{np.max(np.abs(before)):.2f} worst")

        for mode in ("constant", "dynamic"):
            rn._ROLL.clear()
            out = tmp / f"out_{name.replace(':', '')}_{mode}.mp4"
            try:
                rn.clip(str(src), 0.2, DUR - 0.2, out, hwaccel="none",
                        level=mode, telemetry=str(tel))
            except Exception as exc:
                print(f"    {mode:<9} FAILED: {exc}")
                fails += 1
                continue
            tilts = [lv.frame_tilt(str(out), s)[0] for s in probes - 0.2]
            dark = max(corners(out, s) for s in probes - 0.2)
            left = float(np.mean(tilts))
            # Constant only removes the lean, so a swing survives it by design
            # and only the average has to come out flat; dynamic is judged frame
            # by frame, which is the whole claim it makes.
            if mode == "constant":
                metric = abs(left) / max(abs(before.mean()), 1e-6)
            else:
                metric = float(np.max(np.abs(tilts))) / np.max(np.abs(before))
            ok = metric < 0.25 and dark > 16
            fails += 0 if ok else 1
            print(f"    {mode:<9} residual {left:+.2f} deg "
                  f"(worst {np.max(np.abs(tilts)):.2f}) — "
                  f"{metric:.0%} of the source's tilt left  "
                  f"darkest corner {dark:>3}  {'ok' if ok else 'FAIL'}")

        # 3. The same footage with the telemetry's handedness reversed — the
        #    convention this code refuses to assume. Getting it wrong does not
        #    fail, it doubles the tilt, so the check is that the *rendered
        #    result* is the same either way.
        rn._ROLL.clear()
        flip = tmp / f"tel_{name.replace(':', '')}_flipped.parquet"
        telemetry(-roll, t, (1, 0, 2)).to_parquet(flip)
        out = tmp / f"out_{name.replace(':', '')}_flipped.mp4"
        rn.clip(str(src), 0.2, DUR - 0.2, out, hwaccel="none",
                level="dynamic", telemetry=str(flip))
        tilts = [lv.frame_tilt(str(out), s)[0] for s in probes - 0.2]
        share = float(np.max(np.abs(tilts))) / np.max(np.abs(before))
        ok = share < 0.25
        fails += 0 if ok else 1
        print(f"    flipped   worst {np.max(np.abs(tilts)):.2f} deg — "
              f"{share:.0%} left  {'ok' if ok else 'FAIL — sign not caught'}")

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    print(f"  artefacts in {tmp}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
