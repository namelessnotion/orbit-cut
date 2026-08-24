"""Horizon levelling, measured before it is applied.

Phase 0 established this is needed: roll suppression came out at +0.11, meaning
the camera did *not* level in-body, so the tilt is still in the pixels. Applying
levelling on top of footage the camera had already levelled would double-correct
and look worse than doing nothing, which is why that measurement came first.

What is not settled is *which* levelling. Two quite different things get called
by the same name:

    constant   one rotation for the whole clip, removing a mounting offset —
               a chest strap sitting a few degrees off square
    dynamic    per-frame counter-rotation, removing the lean as well — what
               HyperSmooth's Horizon Lock does

Dynamic is not strictly better. On a bike the lean *is* the riding; locking it
out can read as floaty and detached, and it costs crop margin on every frame.
Constant costs almost nothing and fixes the thing nobody wants. So this module
measures the roll first and reports what each mode would do, and `render` takes
the mode as a choice rather than assuming one.

**Roll comes from GRAV, not from a Euler convention.** GoPro's quaternion axis
order is not something to assume — that assumption has already been wrong twice
here — but gravity is a direction, and a direction's angle inside the image
plane is well defined however the axes are labelled.

Finding that plane is the trick, and the obvious way is wrong. Rolling rotates
gravity *about the optical axis*, so the gravity vectors trace an arc, and the
plane of that arc has the optical axis as its normal. Take it with an SVD.
Choosing axes by "gravity is largest along vertical, smallest along forward"
instead looks equivalent and fails on exactly this camera: a chest mount's
forward pitch parks a constant 0.22 on the optical axis while the rolling
component averages below it, so the two swap and a planted 4 degree offset with
a +/-12 degree swing reads back as 12.5 degrees with a 0.5 degree swing —
identically under every axis order, because axis order was never the problem.

**Two things the plane alone cannot tell you, and how each is settled.**

*Where zero is.* The SVD returns a basis for the image plane, but nothing in it
points at the image's own "down", so measuring the angle against it makes the
median zero by construction — a mounting offset would be defined away, and
constant levelling would forever apply 0.0 degrees while looking like it worked.
Zero has to come from the camera's own axes: the raw axis nearest the plane
normal is forward, and of the two left, the one that lines up with average
gravity is down. That "the camera is upright on average" is an assumption, but a
weak one — a mount a few degrees off is still nearly down, while the other axis
is ninety degrees away — and it is the assumption that makes the offset a
measurement rather than a definition.

*Which way is positive.* Whether roll +5 means the image tilts left or right is
a GoPro axis-sign convention, and getting it backwards does not fail loudly: it
doubles the tilt instead of removing it. So it is not assumed either. Gradient
orientations in the frame carry the answer — trees are vertical, horizons are
horizontal — and `verify_sign` reads the structural tilt straight off the pixels
and correlates it with the measured roll. Positive correlation means the sign is
right. `render` runs that check per ride and declines to level when it comes back
inconclusive, because unlevelled is a far cheaper mistake than double-tilted.
"""
from __future__ import annotations

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
# The arc has to be an arc. With almost no roll in a ride the gravity vectors
# collapse toward a point and the plane through them is whatever noise says, so
# the second singular value has to stand clear of the third.
MIN_PLANE_RATIO = 3.0
# Below this correlation between measured roll and the tilt visible in the
# frames, the sign is not established and levelling is declined.
MIN_SIGN_CORR = 0.35


def roll_series(tel: pd.DataFrame, sign: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(t, roll in degrees) — how far the horizon is tilted in frame.

    Zero means gravity points along the camera's own down axis, so the median is
    the mounting offset and not an artefact of the reference. `sign` flips the
    handedness, which is a convention `verify_sign` resolves from the pixels.
    """
    grav = [c for c in tel.columns if c.startswith("grav_")]
    if len(grav) != 3 or "t" not in tel:
        return np.array([]), np.array([])
    g = tel[grav].to_numpy(dtype=float)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    good = (n > 1e-6).ravel() & np.isfinite(g).all(axis=1)
    if good.sum() < 10:
        return np.array([]), np.array([])
    g = np.divide(g, n, out=np.zeros_like(g), where=n > 1e-6)

    # 1. The image plane, geometrically rather than by picking axes.
    #
    # The first attempt guessed: vertical is where gravity is largest, forward
    # is where it is smallest. That is false for a chest mount, whose forward
    # pitch parks a constant 0.22 on the optical axis while the rolling
    # component averages *below* it — so the roll axis and the horizontal axis
    # swapped, and a planted 4 degree offset with a +/-12 degree swing came back
    # as 12.5 degrees with a 0.5 degree swing. Confidently, and identically
    # under every axis order, because the bug was not about axis order.
    #
    # Rolling rotates gravity about the optical axis, so the gravity vectors
    # trace an arc in a plane whose normal IS that axis.
    gg = g[good]
    centred = gg - gg.mean(axis=0)
    _u, sv, vt = np.linalg.svd(centred, full_matrices=True)
    if sv[2] > 1e-12 and sv[1] / sv[2] < MIN_PLANE_RATIO:
        return np.array([]), np.array([])   # too little roll to find the plane
    axis = vt[2]

    # 2. Where zero is. The plane's own basis has no relation to the image, so
    # the reference comes from the camera's axes: nearest raw axis to the normal
    # is forward, and of the remaining two the one aligned with mean gravity is
    # down. Measuring against the plane basis instead would subtract the
    # mounting offset by construction — the thing constant levelling exists to
    # find.
    fwd = int(np.argmax(np.abs(axis)))
    rest = [i for i in (0, 1, 2) if i != fwd]

    def in_plane(i: int) -> np.ndarray:
        e = np.zeros(3)
        e[i] = 1.0
        e = e - (e @ axis) * axis
        m = np.linalg.norm(e)
        return e / m if m > 1e-9 else e

    mean_g = gg.mean(axis=0)
    proj = [in_plane(i) for i in rest]
    k = int(np.argmax([abs(p @ mean_g) for p in proj]))
    down = proj[k] * np.sign(proj[k] @ mean_g or 1.0)
    right = proj[1 - k]
    right = right - (right @ down) * down
    right /= max(np.linalg.norm(right), 1e-9)

    ang = np.degrees(np.arctan2(sign * (g @ right), g @ down))
    ang = (ang + 180.0) % 360.0 - 180.0
    # Unwrap before use: an arc straddling +/-180 would otherwise show a 360
    # degree jump in the middle of an ordinary corner.
    ang = np.degrees(np.unwrap(np.radians(ang)))
    ang[~good] = np.nan
    return tel["t"].to_numpy(dtype=float), ang


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
    series, and no closed form describes a trail. sendcmd is the supported way
    to steer a filter parameter from outside — one command per step, timestamps
    rebased to the clip because the trim resets them to zero.

    The step has to beat the frame interval, not merely look fine. A frame holds
    the last command before it, so a 0.1 second grid leaves each frame up to a
    tenth of a second stale, and at the 14 deg/s a corner reaches that is 1.4
    degrees of lag — which measured as 2.7 degrees of tilt still in the picture
    after levelling. At 50 Hz the same clip came out flat. The file is a few
    hundred lines either way.
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


# --------------------------------------------------------------- sign, from pixels
def frame_tilt(src: str, when: float, width: int = 480) -> tuple[float, float]:
    """(tilt in degrees, how strongly the frame says so) from one frame.

    Rotating an image rotates every gradient in it by the same angle, so the
    frame's structural tilt is readable without rotating anything: sum the
    gradient directions as unit vectors at four times their angle, and the
    result points at whatever orientation the picture is built on. Four times,
    because vertical and horizontal are the same claim about a tilt — a tree
    leaning 3 degrees and a horizon leaning 3 degrees agree, and this is a
    forest full of vertical trees.

    The magnitude comes back as well: a frame of blurred dirt has no structural
    tilt to report, and averaging it in as though it did would be noise.
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
    img = img.astype(np.float32)
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    # Drop the flat majority: sky and dirt contribute direction-free noise that
    # would otherwise outvote the few strong edges that carry the tilt.
    thr = np.percentile(mag, 80)
    m = mag >= thr
    if m.sum() < 200:
        return float("nan"), 0.0
    th = np.arctan2(gy[m], gx[m])
    w = mag[m]
    z = np.sum(w * np.exp(4j * th)) / np.sum(w)
    # arg/4 is the gradient direction; the structure is perpendicular, and the
    # quarter-turn ambiguity is already folded in by the factor of four.
    tilt = np.degrees(np.angle(z)) / 4.0
    return float(tilt), float(np.abs(z))


def verify_sign(src: str, t: np.ndarray, roll: np.ndarray,
                t_in: float = 0.0, t_out: float | None = None,
                n_frames: int = 14) -> dict:
    """Does the tilt in the pixels move with the measured roll, or against it?

    Frames are chosen where the roll is largest in both directions, because a
    frame shot level says nothing about which way positive points. The answer is
    a correlation, not a vote, so a weak one can be recognised as weak — and
    when it is, the caller is expected to skip levelling rather than pick.
    """
    ok = np.isfinite(roll)
    if t_out is None:
        t_out = float(t[-1]) if len(t) else 0.0
    ok &= (t >= t_in) & (t <= t_out)
    if ok.sum() < 20:
        return {"usable": False, "reason": "not enough roll samples"}
    tt, rr = t[ok], roll[ok]
    rr = rr - np.median(rr)
    # Extremes of both signs, spread out in time so one corner does not decide.
    order = np.argsort(rr)
    picks = np.concatenate([order[:n_frames // 2], order[-(n_frames // 2):]])
    picks = np.unique(picks)

    meas, seen, wts = [], [], []
    for i in picks:
        tilt, conf = frame_tilt(src, float(tt[i]))
        if not np.isfinite(tilt) or conf < 0.02:
            continue
        # Both are angles mod 90; compare them on the same wrapped scale.
        meas.append(((rr[i] + 45.0) % 90.0) - 45.0)
        seen.append(tilt)
        wts.append(conf)
    if len(meas) < 6:
        return {"usable": False, "reason": "frames have no usable structure"}

    a, b, w = np.array(meas), np.array(seen), np.array(wts)
    aw, bw = a - np.average(a, weights=w), b - np.average(b, weights=w)
    denom = np.sqrt(np.sum(w * aw ** 2) * np.sum(w * bw ** 2))
    corr = float(np.sum(w * aw * bw) / denom) if denom > 1e-12 else 0.0
    return {
        "usable": abs(corr) >= MIN_SIGN_CORR,
        "corr": corr,
        "sign": 1 if corr >= 0 else -1,
        "frames": int(len(meas)),
        "reason": "" if abs(corr) >= MIN_SIGN_CORR else
                  f"correlation {corr:+.2f} is under {MIN_SIGN_CORR}",
    }
