"""Horizon levelling, measured against the pixels rather than assumed from the axes.

Phase 0 established that levelling applies at all: roll suppression came out at
+0.11, so the camera did not level in body and the tilt is still in the picture.
What took three attempts was measuring it.

Two things get called levelling and they are not the same:

    constant   one rotation for the whole clip, removing a mounting offset —
               a chest strap sitting a few degrees off square
    dynamic    per-frame counter-rotation, removing the lean as well — what
               HyperSmooth's Horizon Lock does

Dynamic is not strictly better. On a bike the lean *is* the riding; locking it
out can read as floaty and detached, and it costs crop margin on every frame. So
this module measures both and reports what each would do, and `render` takes the
mode as a choice.

## Why the arc-fitting version was wrong

Rolling rotates gravity about the optical axis, so gravity traces an arc whose
plane normal is that axis, and an SVD finds the plane without naming a single
axis. That reasoning is correct and the implementation matched it: on planted
telemetry it recovered a 4.0 degree offset as 4.00 with correlation 1.000, under
every axis order.

On a real ride it returned a constant of **-1536 degrees** and a swing of 3957.

The mistake is specific and worth keeping. Gravity vectors are *unit* vectors, so
they have no radial variation at all — the direction of least variance is always
the mean direction, not the optical axis. On the real ride `mean_gravity · normal`
came out **-0.978**: the SVD had returned the gravity direction itself. It only
looked right in the test because the test planted roll and nothing else, which
makes the arc exactly planar; real riding pitches as well as rolls, so gravity
wanders over a patch of the sphere and there is no plane to find. The singular
values say so plainly — 11.8, 10.6, 2.6, not two-and-a-remainder.

**A synthetic test can only refute the errors its generator can express.** Mine
planted roll alone, so it could prove axis-order invariance and could not see the
planarity assumption underneath.

## What it does instead

The optical axis is not derived; it is *chosen by fit*. Each of the three axes is
tried as the optical one, each gives a roll series, and each is regressed against
the tilt visible in the frames — gradient orientations summed at four times their
angle, so that vertical trees and a horizontal horizon make the same claim. The
axis that matches the picture wins. On real rides that is consistently axis 2,
the fore-aft one, which is the physically expected answer; the difference is that
now it is a measurement with a correlation attached.

The regression's slope carries something no axis assignment could: **how much of
the body's roll actually reaches the picture.** Measured over three rides it is
0.14, 0.42 and 0.12 — the camera removes most of the dynamic roll before it ever
lands in a frame, and how much varies by ride. Feeding raw body roll to the first
would have overcorrected it sevenfold. The slope carries the sign too, which is a
GoPro convention this code refuses to assume, because getting it backwards
doubles the tilt rather than failing.

This also settles what phase 0's "+0.11 roll suppression" meant, which looked
like it contradicted the above. It does not: phase 0 compared a *steady* body
tilt of 13.2 degrees against 11.7 on screen, and a constant offset is exactly
what stabilisation passes through. The dynamic swing is what it removes. Both
numbers are right about different things.

The constant offset is not taken from telemetry at all. Where zero is depends on
which way the camera is screwed to the strap, so it is read straight off the
frames. Across those three rides it is +0.3, -0.9 and -0.4 degrees: **the mount
is square**, constant levelling has nothing to do, and what is left for dynamic
is between 3 and 11 degrees of visible swing.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

# Below this, a mounting offset is not worth a rotation and its crop cost.
MIN_CONSTANT_DEG = 1.5
# Dynamic levelling has to be smoothed or it fights the suspension. A trail
# corner lasts seconds; frame-rate wobble is not horizon movement.
SMOOTH_S = 0.7
# Rotating by θ shrinks the usable rectangle. Beyond this the crop starts eating
# real content, so the correction is clamped rather than silently zooming in.
MAX_DEG = 25.0
# Low-pass before the gradient. Measured, not chosen — see `frame_tilt`.
BLUR_SIGMA = 1.0
# How well the frames have to agree with the telemetry before either is trusted.
MIN_CORR = 0.35
# A plausible share of body roll reaching the picture. Above 1.5 the telemetry is
# not describing this camera; below 0.08 there is nothing left to correct.
GAIN_RANGE = (0.08, 1.5)


def roll_series(tel: pd.DataFrame, optical: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(t, body roll in degrees) with `optical` treated as the fore-aft axis.

    Zero is gravity along the camera's own down axis. This is the *body's* roll —
    what the rider did — not what survived into the frame; `calibrate` measures
    the difference.
    """
    grav = [c for c in tel.columns if c.startswith("grav_")]
    if len(grav) != 3 or "t" not in tel:
        return np.array([]), np.array([])
    g = np.array(tel[grav].to_numpy(dtype=float), copy=True)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    good = (n > 1e-6).ravel() & np.isfinite(g).all(axis=1)
    if good.sum() < 10:
        return np.array([]), np.array([])
    g = np.divide(g, n, out=np.zeros_like(g), where=n > 1e-6)

    mean_g = g[good].mean(axis=0)
    p, q = [i for i in (0, 1, 2) if i != optical]
    # Down is whichever of the two carries gravity; on an inverted chest mount
    # that component is negative, and the sign puts zero where level is.
    dn, rt = (p, q) if abs(mean_g[p]) >= abs(mean_g[q]) else (q, p)
    ang = np.degrees(np.arctan2(g[:, rt], np.sign(mean_g[dn]) * g[:, dn]))
    ang[~good] = np.nan
    # No unwrapping. Roll on a bike stays inside a quarter turn, and unwrapping
    # a signal that hovers near the +/-180 boundary — which is what the broken
    # reference produced — turns noise into thousands of degrees of drift.
    return np.array(tel["t"].to_numpy(dtype=float), copy=True), ang


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    r = int(3 * sigma)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma * sigma))
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)


def frame_tilt(src: str, when: float, width: int = 480) -> tuple[float, float]:
    """(tilt in degrees, how strongly the frame says so) from one frame.

    Rotating an image rotates every gradient in it by the same angle, so a
    frame's tilt is readable without rotating anything: sum the gradient
    directions as unit vectors at four times their angle and the result points at
    whatever orientation the picture is built on. Four times, because vertical
    and horizontal are the same claim about a tilt — a tree leaning 3 degrees and
    a horizon leaning 3 degrees agree, and this is a forest full of trees.

    **The blur is not cosmetic.** A central-difference gradient is only accurate
    on content well below Nyquist; on raw frames it pushes orientations toward
    the diagonal, which amplifies small tilts. Measured by rotating real frames
    by known angles and reading the shift back: unsmoothed it returns **1.244x**
    the rotation applied, and worse at lower resolution (1.545 at 240 px wide) —
    a bias that would have been invisible in any correlation and would have
    overcorrected every clip by a quarter. At sigma = 1 it returns **0.970** at
    480 px and **0.973** at 240, near unity and stable across scale, which is
    what says the estimator is band-limited rather than merely tuned.

    The magnitude comes back too: a frame of blurred dirt has no tilt to report
    and averaging it in as though it did would be noise. On this library the
    strength runs 0.07 to 0.32, so 0.02 is a floor for "nothing there", not a
    quality bar.
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(when, 0):.3f}", "-i", src,
         "-frames:v", "1", "-vf", f"scale={width}:-2,format=gray",
         "-f", "rawvideo", "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return float("nan"), 0.0
    h = len(r.stdout) // width
    if h < 8:
        return float("nan"), 0.0
    img = np.frombuffer(r.stdout[:h * width], dtype=np.uint8).reshape(h, width)
    gy, gx = np.gradient(_blur(img.astype(np.float32), BLUR_SIGMA))
    mag = np.hypot(gx, gy)
    # Drop the flat majority: sky and dirt contribute direction-free noise that
    # would otherwise outvote the few strong edges carrying the tilt.
    m = mag >= np.percentile(mag, 80)
    if m.sum() < 200:
        return float("nan"), 0.0
    z = np.sum(mag[m] * np.exp(4j * np.arctan2(gy[m], gx[m]))) / np.sum(mag[m])
    return float(np.degrees(np.angle(z)) / 4.0), float(np.abs(z))


def _square_pixels(src: str) -> bool:
    """Angles measured on an anamorphic frame are sheared, not rotated."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=sample_aspect_ratio", "-of", "json", src],
        capture_output=True, text=True)
    try:
        sar = json.loads(r.stdout or "{}")["streams"][0]["sample_aspect_ratio"]
        a, b = (int(x) for x in sar.split(":"))
        return abs(a / b - 1.0) < 0.02
    except Exception:
        return True


def calibrate(video: str, tel: pd.DataFrame, n_frames: int = 60,
              t_in: float = 0.0, t_out: float | None = None) -> dict:
    """Fit the telemetry's roll to the tilt actually visible in `video`.

    Returns the optical axis, the share of body roll that reaches the picture,
    the constant tilt the frames show, and the correlation the whole thing rests
    on. `video` should be the proxy — it is 540p, this decodes dozens of frames,
    and rotation angles are the same at any scale.
    """
    if "t" not in tel:
        return {"usable": False, "reason": "no telemetry"}
    if not _square_pixels(video):
        return {"usable": False, "reason": "non-square pixels; angles are sheared"}
    t_all = np.array(tel["t"].to_numpy(dtype=float), copy=True)
    lo = max(t_in, float(t_all[0]))
    hi = min(t_out if t_out is not None else float(t_all[-1]), float(t_all[-1]))
    if hi - lo < 4.0:
        return {"usable": False, "reason": "too short to calibrate"}
    # Skip the ends: mounting, dismounting, and a camera being picked up are not
    # the ride. Proportional rather than fixed, so calibrating over a single
    # twelve-second clip does not trim away most of it.
    edge = min(2.0, 0.1 * (hi - lo))
    when = np.linspace(lo + edge, hi - edge, n_frames)

    seen, wts, keep = [], [], []
    for s in when:
        tilt, conf = frame_tilt(video, float(s))
        if np.isfinite(tilt) and conf >= 0.02:
            seen.append(tilt)
            wts.append(conf)
            keep.append(s)
    if len(seen) < 12:
        return {"usable": False, "reason": f"only {len(seen)} frames have structure"}
    y, w, ts = np.array(seen), np.array(wts), np.array(keep)

    best = None
    for axis in (0, 1, 2):
        tt, roll = roll_series(tel, optical=axis)
        if not len(roll):
            continue
        ok = np.isfinite(roll)
        if ok.sum() < 10:
            continue
        x = np.interp(ts, tt[ok], roll[ok])
        x = x - np.median(x)
        if np.ptp(x) < 1.0:
            continue                      # this axis sees no roll to explain
        corr = float(np.corrcoef(x, y)[0, 1])
        gain = float(np.polyfit(x, y, 1)[0])
        if best is None or abs(corr) > abs(best["corr"]):
            best = {"axis": axis, "corr": corr, "gain": gain,
                    "swing_deg": float(np.percentile(x, 95) - np.percentile(x, 5))}
    if best is None:
        return {"usable": False, "reason": "no axis produced a roll series"}

    # The constant comes from the frames, not the telemetry: where zero is
    # depends on how the camera sits on the strap, which gravity cannot know.
    best["constant_deg"] = float(np.average(y, weights=w))
    best["frames"] = int(len(y))
    best["worth_constant"] = abs(best["constant_deg"]) >= MIN_CONSTANT_DEG
    lo_g, hi_g = GAIN_RANGE
    if abs(best["corr"]) < MIN_CORR:
        best["usable"] = False
        best["reason"] = f"frames and telemetry agree only {best['corr']:+.2f}"
    elif not lo_g <= abs(best["gain"]) <= hi_g:
        best["usable"] = False
        best["reason"] = f"gain {best['gain']:+.2f} is outside {lo_g}–{hi_g}"
    else:
        best["usable"] = True
        best["reason"] = ""
    return best


def visible_roll(tel: pd.DataFrame, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    """(t, the tilt in the picture) — body roll scaled by what reaches the frame.

    The constant is added back from the frames, so the series is what a viewer
    sees rather than what the rider did.
    """
    if not cal.get("usable"):
        return np.array([]), np.array([])
    t, roll = roll_series(tel, optical=cal["axis"])
    if not len(roll):
        return np.array([]), np.array([])
    return t, cal["gain"] * (roll - np.nanmedian(roll)) + cal["constant_deg"]


def summarise(t: np.ndarray, roll: np.ndarray) -> dict:
    if not len(roll):
        return {"usable": False}
    r = roll[np.isfinite(roll)]
    if not len(r):
        return {"usable": False}
    med = float(np.median(r))
    centred = r - med
    return {
        "usable": True,
        "n": int(len(r)),
        "constant_deg": med,
        "spread_deg": float(np.percentile(centred, 95) - np.percentile(centred, 5)),
        "max_deg": float(np.max(np.abs(centred))),
        "worth_constant": abs(med) >= MIN_CONSTANT_DEG,
    }


def smoothed(t: np.ndarray, roll: np.ndarray, window_s: float = SMOOTH_S) -> np.ndarray:
    """Roll with frame-rate wobble removed, so levelling tracks the horizon
    rather than the suspension."""
    if len(t) < 3:
        return roll
    rate = len(t) / max(t[-1] - t[0], 1e-6)
    w = max(3, int(window_s * rate) | 1)
    return (pd.Series(roll).rolling(w, center=True, min_periods=1)
            .median().interpolate(limit_direction="both").to_numpy())


def sendcmd(t: np.ndarray, roll: np.ndarray, t_in: float, t_out: float,
            step_s: float = 0.02) -> str:
    """A `sendcmd` script driving the rotate filter over the clip.

    ffmpeg's `rotate` takes an expression in `t`, but not an arbitrary measured
    series, and no closed form describes a trail. sendcmd is the supported way to
    steer a filter parameter from outside — one command per step, timestamps
    rebased to the clip because the trim resets them to zero.

    The step has to beat the frame interval, not merely look fine. A frame holds
    the last command before it, so a 0.1 second grid leaves each frame up to a
    tenth of a second stale, and at the 14 deg/s a corner reaches that is 1.4
    degrees of lag — which measured as 2.7 degrees of tilt still in the picture
    after levelling. At 50 Hz the same clip came out flat.
    """
    if not len(t):
        return ""
    grid = np.arange(0.0, max(t_out - t_in, step_s), step_s)
    # Sample the middle of each step rather than its start: the command holds
    # until the next one, so the midpoint halves the error at both ends.
    vals = np.interp(grid + t_in + step_s / 2, t, roll)
    vals = np.clip(np.nan_to_num(vals), -MAX_DEG, MAX_DEG)
    # Counter-rotate: a frame tilted +5 deg needs -5 deg applied.
    return "".join(f"{g:.3f} rotate angle '{-v * np.pi / 180:.6f}';\n"
                   for g, v in zip(grid, vals))
