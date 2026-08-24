"""End-to-end check on horizon levelling, against footage with a known tilt.

Levelling has three ways to be wrong that all look like success. It can measure
the tilt and rotate the wrong way, doubling it. It can rotate the right way but
not shrink the crop enough, pulling the frame's own corners in as black wedges.
And it can be right about the direction and wrong about the amount — which is
what happens if you feed it the body's roll without asking how much of that roll
the camera already removed. None shows up in an exit code.

So this plants a tilt rather than trusting one. A striped scene is rotated by a
known angle to build the source, telemetry is synthesised from that same angle
divided by a chosen suppression factor, and the pipeline is asked to work out
the factor and undo the tilt. What gets measured on the result: what `calibrate`
recovered, what tilt is left in the pixels, and whether any corner went black.

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
DUR = 12.0
PITCH_DEG = 13.0      # chest mount, the case that broke the first attempt
CONST_DEG = 6.0       # a mount sitting off square
SWING_DEG = 9.0       # and the lean on top of it
PERIOD_S = 4.0        # slower than the 0.7 s smoothing window, as a corner is


def scene(w: int, h: int) -> np.ndarray:
    """Trees and a horizon: vertical bars over a band, nothing near black.

    Bright everywhere on purpose — after this, any black pixel in a rendered
    frame came from a rotation exposing a corner, so the corner test needs no
    tolerance and no judgement. The bars are wide and soft-edged rather than
    fine and hard: a fine pattern aliases under rotation, and an aliased edge
    reports the wrong orientation, which would make the test's own ground truth
    the least reliable thing in it.
    """
    img = np.full((h, w), 210.0, dtype=np.float32)
    img[int(h * 0.45):, :] = 150.0
    xs = np.arange(w)
    img -= 60.0 * (0.5 + 0.5 * np.cos(2 * np.pi * xs / 90.0))[None, :]
    img[int(h * 0.43):int(h * 0.47), :] -= 60.0
    return np.clip(img, 40, 255).astype(np.uint8)


def make_source(path: Path, w: int, h: int, expr: str) -> None:
    """A video of the scene tilted by `expr` radians, with no black introduced.

    The scene is drawn oversized and rotated before cropping to the target frame,
    so the source itself never has an exposed corner to confuse the test.
    """
    big = int(max(w * 0.94 + h * 0.35, h * 0.94 + w * 0.35)) | 1
    png = path.with_suffix(".png")
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-s", f"{big}x{big}", "-i", "-",
                        "-frames:v", "1", str(png)], input=scene(big, big).tobytes(),
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

    `order` says which raw column is down, right and forward. The pipeline is not
    told — it works the forward axis out by fitting each candidate against the
    frames — so the test runs several assignments and expects the same answer.
    """
    d, rgt, fwd = order
    p, a = np.radians(PITCH_DEG), np.radians(roll_deg)
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
    # The encoder is not what is under test; a slow preset just makes the check
    # too tedious to run, which is how checks stop being run.
    rn._encoder = lambda _hw: ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"]
    t = np.arange(0.0, DUR, 1 / 30)
    visible = CONST_DEG + SWING_DEG * np.sin(2 * np.pi * t / PERIOD_S)
    fails = 0

    for name, (w, h) in (("8:7", (3956, 3460)), ("16:9", (3840, 2160))):
        src = tmp / f"src_{name.replace(':', '')}.mp4"
        make_source(src, w, h,
                    f"{CONST_DEG}*PI/180 + {SWING_DEG}*PI/180*sin(2*PI*t/{PERIOD_S})")
        # A 540p stand-in for the proxy. Calibration reads dozens of frames and
        # a rotation angle is the same at any scale, so real runs read the proxy
        # too — decoding 4K frames to measure an angle is pure waste.
        prox = tmp / f"proxy_{name.replace(':', '')}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
                        "scale=-2:540", "-c:v", "libx264", "-preset", "ultrafast",
                        "-crf", "22", str(prox)], check=True, capture_output=True)
        probes = np.linspace(0.35, DUR - 0.45, 9)
        before = np.array([lv.frame_tilt(str(src), s)[0] for s in probes])
        print(f"\n  {name} {w}x{h} — planted {CONST_DEG:.0f}° lean +/-{SWING_DEG:.0f}°, "
              f"frames read {before.mean():+.2f}° mean, {np.max(np.abs(before)):.2f}° worst")

        # 1. Calibration. The pipeline is given telemetry whose roll is larger
        #    than what reached the picture — the camera's own stabilisation — on
        #    an axis assignment it has to work out, sometimes with the handedness
        #    reversed. All three have to come back.
        print(f"    {'telemetry':<26}{'axis':>5}{'gain':>7}{'corr':>7}"
              f"{'offset':>8}   recovered")
        cases = [("straight (down,right,fwd)=(1,0,2)", (1, 0, 2), 1.0),
                 ("axes permuted   =(0,2,1)", (0, 2, 1), 1.0),
                 ("handedness reversed", (1, 0, 2), -1.0),
                 ("stabilised, 40% reaches frame", (1, 0, 2), 0.4)]
        for label, order, share in cases:
            body = visible / share      # what the rider did, before suppression
            tel = telemetry(body, t, order)
            cal = lv.calibrate(str(prox), tel, n_frames=24)
            if not cal.get("usable"):
                print(f"    {label:<26}{cal.get('reason', '')}   FAIL")
                fails += 1
                continue
            # 20% on the gain, because the planted angle is only as good as the
            # estimator reading it, and on this synthetic pattern that is worth
            # about 10% — it reads 0.86-0.91 where 1.00 was planted. On real
            # frames, rotated by known angles, it measures 0.970 +/- 0.045. Both
            # err low, which under-corrects: the safe direction.
            ok = (abs(cal["gain"] - share) < 0.20 * max(abs(share), 0.4)
                  and abs(cal["constant_deg"] - CONST_DEG) < 2.0)
            fails += 0 if ok else 1
            print(f"    {label:<26}{cal['axis']:>5}{cal['gain']:>+7.2f}"
                  f"{cal['corr']:>+7.2f}{cal['constant_deg']:>+7.1f}°   "
                  f"{'ok' if ok else f'FAIL (wanted gain {share:+.2f})'}")

        # 2. The render. Suppression on, so getting the amount wrong shows up as
        #    residual tilt rather than as an obviously broken picture.
        tel_path = tmp / f"tel_{name.replace(':', '')}.parquet"
        telemetry(visible / 0.4, t, (1, 0, 2)).to_parquet(tel_path)
        for mode in ("constant", "dynamic"):
            rn._ROLL.clear()
            out = tmp / f"out_{name.replace(':', '')}_{mode}.mp4"
            try:
                rn.clip(str(src), 0.2, DUR - 0.2, out, hwaccel="none",
                        level=mode, telemetry=str(tel_path), preview=str(prox))
            except Exception as exc:
                print(f"    {mode:<9} FAILED: {exc}")
                fails += 1
                continue
            tilts = [lv.frame_tilt(str(out), s)[0] for s in probes - 0.2]
            dark = max(corners(out, s) for s in probes - 0.2)
            left = float(np.mean(tilts))
            # Constant only removes the lean, so the swing survives it by design
            # and only the average has to come out flat; dynamic is judged frame
            # by frame, which is the whole claim it makes.
            if mode == "constant":
                share = abs(left) / max(abs(before.mean()), 1e-6)
            else:
                share = float(np.max(np.abs(tilts))) / np.max(np.abs(before))
            ok = share < 0.25 and dark > 16
            fails += 0 if ok else 1
            print(f"    {mode:<9} residual {left:+.2f}° (worst "
                  f"{np.max(np.abs(tilts)):.2f}°) — {share:.0%} of the tilt left, "
                  f"darkest corner {dark:>3}  {'ok' if ok else 'FAIL'}")

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    print(f"  artefacts in {tmp}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
