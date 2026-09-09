"""Where the 9:16 crop should sit, second by second, so that Orbit is in it.

`track.py` says where the dog was. This says where the camera should be, and
they are not the same question: a crop pinned to the dog's centre every frame
reproduces the detector's jitter as camera shake and makes the background swim.
What reads as a camera operator is **hold, smooth pan, hold** — and that is a
property of the whole clip at once, not of any one frame.

Rendering is offline, so solve it offline. Minimise, over the whole clip,

    (how far the crop is from where the dog is)
  + (how fast the crop is moving)
  + (how hard the crop is accelerating)

with a **dead zone**: inside it, the dog moving produces no camera movement at
all. The dead zone is what separates this from a low-pass filter. A filtered
path is always moving a little, which is exactly the thing a viewer notices; a
dead zone gives genuine stillness, and stillness is what makes the pans that do
happen look deliberate.

## Units, and why they are not pixels

Position is in **crop widths** and time is in **seconds**. So `DEAD_ZONE = 0.12`
means "an eighth of the frame's width" on a 1680-wide crop and on a 2614-wide
one alike, and the same constants describe every source shape in the library
without a table. Pixels only appear at the very end, in `sendcmd`.

The penalties are written on **derivatives** — divided by `dt` and `dt**2`, with
the whole sum multiplied by `dt` — which makes the objective a Riemann sum of a
continuous functional rather than a sum over samples. That is not decoration:
written on raw differences instead, the effective smoothing strength becomes a
silent function of the sample rate, and the day `TRACK_HZ` changes every clip
reframes differently with nothing in the diff to explain it. `grid invariance`
in `tools/reframe_selftest.py` is the check that this stayed true.

## The dead zone does not make this an optimisation problem

A dead zone makes the tracking term piecewise quadratic rather than quadratic,
which looks like it demands an iterative optimiser and a new dependency. It does
not, and the reason is worth stating precisely, because the first version of
this module got it wrong in a way that produced entirely plausible output.

Outside the band the penalty is `(|e| - d)**2`, and that is *exactly* a plain
quadratic centred on the near edge of the band, at `c + d*sign(e)`. Inside the
band it is flat. So for any fixed guess at which samples are outside, the whole
problem is an ordinary weighted least squares against a modified target — one
symmetric pentadiagonal solve through `scipy.linalg.solveh_banded`, with no
tolerance to tune. Take the answer, recompute which samples are outside, repeat.
That is semismooth Newton on a convex piecewise-quadratic; each iteration is
exact for its active set, and on real clips it converges in two. scipy is
already a dependency for the roughness band-pass, and cvxpy is not needed.

**What does not work, and looked like it did.** The tidier-seeming route is

    psi_d(e) = min over |s| <= d of (e - s)**2

which makes the problem jointly convex in `(z, s)` and invites alternating the
blocks: clip the slack, then solve for the path. Each step really is the exact
minimiser of its block, so it is genuine block coordinate descent and the
objective really does decrease. It is also unusably slow, and it fails quietly.
With the slack absorbing the error exactly, the path-step's right-hand side
collapses to `w*dt*z_old`, so each iteration only smooths the previous answer a
little — and once the smoothness weights are small, `diag(w*dt)` dominates and
it barely moves at all. Measured on a planted half-band step, where the correct
answer is a flat line costing exactly zero: after 24 iterations the objective
was 2.0e-5, after 500 it was 4.4e-6, and it took 8000 to reach 6.3e-9. Every one
of those runs returned a smooth, thoroughly reasonable-looking path. The
24-iteration one had followed a step it was supposed to ignore — the dead zone
not working, reported as success.

**A second shortcut that does not work**, and the first thing anyone tries:
precompute a deadbanded target from the raw dog track, then do a single
quadratic solve. The dead zone is a band around *where the camera currently is*,
not around where the dog was, so it has to be recomputed against the solution.
The one-shot version produces a slow residual ramp where there should be a flat
hold, which again looks very nearly right.

## The dead zone needs an anchor, or the problem is not well posed

Inside the band the tracking term is flat, and a constant offset has zero first
and second derivative. So sliding the whole path sideways within the band costs
nothing in any term of the objective at all. That is a genuine null direction:
the minimiser is not unique, the matrix is singular wherever no sample is
outside its band, and which member of the family comes back is an artefact of
the linear algebra rather than an answer. `ANCHOR` is what removes it.

## Dropouts

The detector loses Orbit constantly — behind brush, into shadow, or simply too
far down the trail to resolve at proxy scale. The policy is hold, then ease:

    tracked                     aim at the dog
    gap up to HOLD_S            hold the last aim, exactly
    the next EASE_S             raised-cosine back to the middle of the frame
    beyond that                 the middle of the frame

Every grid point gets a target and a positive weight, including the ones in a
gap. Leaving a gap unweighted instead is tempting and it fails at the end of a
clip: with nothing pulling on it, the acceleration penalty drives the crop to
constant velocity and it sails off the edge of the frame.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded

# How far the dog may wander before the camera moves at all, in crop widths.
DEAD_ZONE = 0.12
# Weights on pan velocity (crop widths/s) and acceleration (crop widths/s^2).
#
# **Measured — these want to be small, and the first draft had them 25x too
# large.** The intuition is that smoothness is what makes the crop read as a
# camera rather than a tracker, so it should be weighted hard. That is exactly
# backwards, and the reason is the dead zone: *the band is what provides the
# stillness*, and the smoothness terms only have to round the corners at its
# edges. Weighted hard, they instead fight the band. Planted a dog sitting
# still for five seconds and then lost, with lam_v=3, lam_a=12: the crop never
# reached the aim at all and slid continuously for the whole clip, because
# spreading the eventual return to centre over eight seconds cost 0.04 in
# velocity while leaving the subject early cost 0.0007 in tracking. The solver
# was right; the objective was wrong.
#
# Two further things that sweep settled, both worth not rediscovering:
#
#   `lam_a` is very nearly inert — 0.02 to 0.4 moved the settled path by under
#   0.002 crop widths. The curvature it exists to penalise has already been
#   absorbed by the band, which the path is free to round its own corners
#   inside. It is kept small and non-zero to damp the corners, not to shape the
#   path, and tuning it is not where an improvement will come from.
#
#   What survives at these values is a *deliberate* one-second anticipation
#   before a give-up: the crop starts moving inside the band shortly before the
#   hold expires, because inside the band that movement is free. It reads the
#   way an operator who has lost the subject reads, and removing it would cost
#   the stillness everywhere else.
#
# At 0.02/0.01 a planted steady dog holds at 0.000 crop widths/s for four
# seconds, the ease peaks at 0.34, and the crop is back within the band of
# centre 3.1 s after the loss against a nominal HOLD_S + EASE_S of 3.0.
LAMBDA_V = 0.02
LAMBDA_A = 0.01

HOLD_S = 1.0        # a gap shorter than this is a blink; hold the aim
EASE_S = 2.0        # ...and this long to give up and return to centre
PAD_S = 2.0         # context solved either side of the clip, then discarded
SOLVE_HZ = 10.0     # the grid the path is solved on, before sendcmd resamples

# How hard each kind of target pulls. Only the ratios matter, and they encode
# how much the policy believes its own aim at that moment:
#
#   tracked   a detection — the strongest claim there is
#   hold      a blink. The aim was right a quarter-second ago and nothing says
#             it stopped being right, so this is nearly as good as a detection.
#             It has to be: the solve is offline, so a weak hold lets the path
#             start drifting toward the ease *before the hold has expired*,
#             which is a camera that reacts to the future.
#   ease      the least certain moment there is — the old aim is stale and
#             centre is not yet earned. Let smoothness win here.
#   lost      the dog has been gone for seconds. Centre is genuinely where the
#             frame should be, and there is no competing evidence, so this
#             pulls hard enough to actually arrive.
W_TRACK, W_HOLD, W_EASE, W_LOST = 1.0, 0.8, 0.2, 0.6

# A weak pull toward the subject, acting inside the dead zone as well as
# outside it, as a share of the tracking weight. See the module docstring for
# why the problem is not well posed without one.
#
# **Its size is not a taste setting, and it is not free.** The same number
# decides two things that pull against each other: how completely the crop
# finishes a large move, and whether it ignores a small one. Swept on planted
# steps of half a dead zone and three dead zones, measuring how far the crop
# moved for the first and how close it ended up for the second:
#
#     anchor    half-band step moves      three-band step ends
#     1e-5      0.01 dead zones           0.51 dead zones from the dog
#     1e-4      0.11                      0.38
#     1e-3      0.39                      0.07
#     1e-2      0.50                      0.00
#
# Read the right-hand column as the band doing its job, not as error: a crop
# that stops once the dog is inside the band is correct, and "0.51 dead zones
# from the dog" means he is sitting comfortably within it. The left-hand column
# is the one with a wrong end. At 1e-2 the crop follows a step it was supposed
# to ignore — and it is not a bug when it does, it is arithmetic: declining a
# sub-band step of size `delta` costs `anchor * w * delta**2 * T` while
# following it costs `lam_v * delta**2 / tau`, and `delta**2` cancels out of
# both. So the anchor does not discriminate by size at all. It either follows
# every sub-band step or none, and 1e-5 is on the right side of that.
ANCHOR = 1e-5

# Penalty stiffness for a crop that wants to leave the frame. Large enough that
# the bound is met to well under a pixel, small enough not to wreck the
# conditioning of a matrix whose other entries are order 1-100.
BOUND_MU = 1.0e4

# Two consecutive track samples further apart than this are not one continuous
# observation, and interpolating across them would invent a dog. Expressed as a
# multiple of the track's own sample interval, so it follows `--hz`.
MAX_INTERP_STEPS = 1.6


def _normal_matrix(n: int, w: np.ndarray, dt: float,
                   lam_v: float, lam_a: float) -> np.ndarray:
    """The banded form of `diag(w*dt) + lam_v*D1'D1/dt + lam_a*D2'D2/dt**3`.

    Upper form for `solveh_banded`: `ab[2 + i - j, j] = A[i, j]` for `i <= j`,
    so row 2 is the main diagonal, row 1 the first superdiagonal, row 0 the
    second. The two `D` matrices are the first and second difference operators,
    and their normal matrices are written out term by term rather than assembled
    densely, because a whole ride would otherwise be a 7680-square matrix to
    hold a pentadiagonal one. `banded matches dense` in the selftest builds both
    for a small `n` and compares them, which is the only reason to trust this.
    """
    ab = np.zeros((3, n), dtype=float)
    ab[2] = w * dt

    if n >= 2:
        # D1'D1: main diag 1, 2, 2, ..., 2, 1; first superdiag all -1.
        kv = lam_v / dt
        ab[2] += kv * 2.0
        ab[2, 0] -= kv
        ab[2, -1] -= kv
        ab[1, 1:] -= kv

    if n >= 3:
        # D2'D2, from rows i = 0..n-3 with entries (1, -2, 1) at i, i+1, i+2.
        ka = lam_a / dt**3
        j = np.arange(n)
        ok = lambda i: (i >= 0) & (i <= n - 3)          # noqa: E731 — row exists
        ab[2] += ka * (ok(j) * 1.0 + ok(j - 1) * 4.0 + ok(j - 2) * 1.0)
        if n >= 2:
            # A[j-1, j] takes -2 from the window starting at j-1 and -2 from
            # the one starting at j-2 — indices of the *window*, not the column.
            jj = np.arange(1, n)
            ab[1, 1:] += ka * (-2.0 * ok(jj - 1) - 2.0 * ok(jj - 2))
        if n >= 3:
            jj = np.arange(2, n)
            ab[0, 2:] += ka * ok(jj - 2) * 1.0
    return ab


def targets(t: np.ndarray, z: np.ndarray, grid: np.ndarray, *,
            centre: float, hold_s: float = HOLD_S, ease_s: float = EASE_S,
            w_track: float = W_TRACK, w_hold: float = W_HOLD,
            w_ease: float = W_EASE, w_lost: float = W_LOST,
            max_interp_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(target, weight) on the solve grid, applying the hold-then-ease policy.

    `z` is where the crop would sit to centre the dog, in crop widths, with NaN
    wherever the tracker had no box. `centre` is the same quantity for a crop in
    the middle of the frame — which is what a long dropout returns to, and what
    the whole clip is when there is no track at all.
    """
    c = np.full(grid.shape, float(centre))
    w = np.full(grid.shape, float(w_lost))
    good = np.isfinite(z) if len(z) else np.zeros(0, dtype=bool)
    if not good.any():
        return c, w

    tg, zg = np.asarray(t, float)[good], np.asarray(z, float)[good]
    if max_interp_s is None:
        step = np.median(np.diff(tg)) if len(tg) > 1 else np.inf
        max_interp_s = MAX_INTERP_STEPS * float(step)

    # Which observed interval each grid point falls in. `side="right"` then -1
    # gives the index of the sample at or before the grid time; -1 means the
    # grid point precedes every observation.
    idx = np.searchsorted(tg, grid, side="right") - 1
    before = idx < 0
    after = idx >= len(tg) - 1
    inner = ~before & ~after

    # Tracked: strictly between two samples close enough to be one observation.
    live = np.zeros(grid.shape, dtype=bool)
    if inner.any():
        i = idx[inner]
        span = tg[i + 1] - tg[i]
        near = span <= max_interp_s
        j = np.flatnonzero(inner)[near]
        if len(j):
            live[j] = True
            c[j] = np.interp(grid[j], tg, zg)
            w[j] = w_track
    # A grid point landing exactly on the final sample is tracked too, and the
    # interval test above cannot see it because there is no interval after it.
    on_last = np.isclose(grid, tg[-1])
    live |= on_last
    c[on_last] = zg[-1]
    w[on_last] = w_track

    # Everything else is a gap. Age it from the last observation before it; a
    # grid point preceding every observation has no aim to hold and starts from
    # the centre, which is also what an empty track gives.
    gap = ~live
    if gap.any():
        g = np.flatnonzero(gap)
        held = idx[g]
        seen = held >= 0
        # Before the first observation there is no aim to hold *forward* from,
        # and centre is the wrong guess: it drags the opening of every clip away
        # from the dog and then spends seconds travelling back, which is the
        # part of a Reel a viewer is most likely to watch. The best estimate of
        # where the dog was a moment before he was first seen is where he was
        # first seen. So hold the first aim backwards, and let the clip open
        # already framed on him.
        lead = g[~seen]
        c[lead] = zg[0]
        w[lead] = w_hold
        gi, hi = g[seen], held[seen]
        elapsed = grid[gi] - tg[hi]
        last = zg[hi]

        hold = elapsed <= hold_s
        c[gi[hold]] = last[hold]
        w[gi[hold]] = w_hold

        easing = (elapsed > hold_s) & (elapsed <= hold_s + ease_s)
        if easing.any():
            # Raised cosine: zero slope at both ends, so neither the moment the
            # hold expires nor the moment the crop arrives at centre shows a
            # corner in the path.
            k = (elapsed[easing] - hold_s) / max(ease_s, 1e-9)
            blend = 0.5 * (1.0 - np.cos(np.pi * k))
            c[gi[easing]] = last[easing] + (centre - last[easing]) * blend
            # Confidence recovers across the ease as the old aim goes stale and
            # centre becomes the only claim left, so the weight walks back up
            # with it rather than stepping at the far end.
            w[gi[easing]] = w_ease + (w_lost - w_ease) * blend
        # Past hold + ease both target and weight are already the lost case.
    return c, w


def solve(c: np.ndarray, w: np.ndarray, dt: float, *,
          dead: float = DEAD_ZONE, lam_v: float = LAMBDA_V,
          lam_a: float = LAMBDA_A, anchor: float = ANCHOR,
          lo: float = -np.inf, hi: float = np.inf,
          iters: int = 24, tol: float = 1e-5) -> tuple[np.ndarray, dict]:
    """The crop path, in the same units as `c`, and what the solve cost.

    Semismooth Newton: guess which samples have left their dead zone, solve the
    ordinary least-squares problem that guess implies, and repeat until the
    guess stops changing. The frame-edge bounds ride in the same loop as a
    second active set, because the two interact — pinning the path at an edge
    changes which of the samples either side of it are outside their band.

    `iters` is a safety net rather than a budget. Two iterations is the usual
    count and the loop exits on the active sets settling, not on running out.
    """
    n = len(c)
    if n == 0:
        return np.zeros(0), {"iters": 0, "max_violation": 0.0, "saturated": 0.0}
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo_, hi_ = -np.inf, np.inf
    elif hi <= lo:
        # The crop is the whole frame — there is nowhere to pan to, and that is
        # a property of the source, not a failure. Say so in the report.
        z = np.full(n, lo)
        return z, {"iters": 0, "max_violation": 0.0, "saturated": 1.0}
    else:
        lo_, hi_ = lo, hi

    wd = w * dt
    z = np.clip(c.copy(), lo_, hi_)
    active = np.zeros(n, dtype=bool)       # pressing against a frame edge
    pinned = np.zeros(n)
    outside = np.abs(z - c) > dead         # outside its dead zone
    edge = c + dead * np.sign(z - c)

    it = 0
    for it in range(1, iters + 1):
        # Where the dog has left the band the penalty is an ordinary quadratic
        # centred on the near edge of it; where he has not, only the anchor
        # pulls. Both are diagonal, so the matrix stays banded and the solve
        # stays exact for this choice of active set.
        pull = w * (outside + anchor)
        rhs = wd * (outside * edge + anchor * c)
        ab = _normal_matrix(n, pull, dt, lam_v, lam_a)
        if active.any():
            ab[2, active] += BOUND_MU
            rhs = rhs + np.where(active, BOUND_MU * pinned, 0.0)
        nxt = solveh_banded(ab, rhs, lower=False)

        e = nxt - c
        new_outside = np.abs(e) > dead
        below, above = nxt < lo_ - 1e-12, nxt > hi_ + 1e-12
        new_active = below | above
        pinned = np.where(below, lo_, np.where(above, hi_, 0.0))

        step = float(np.max(np.abs(nxt - z))) if n else 0.0
        settled = (np.array_equal(new_outside, outside)
                   and np.array_equal(new_active, active))
        z, outside, active = nxt, new_outside, new_active
        edge = c + dead * np.sign(e)
        # The two active sets are what the iteration is really solving for.
        # Once neither changes, the solve just performed was already exact.
        if settled and step < tol:
            break

    # The clip should be a no-op to well under a pixel; report it rather than
    # let a real violation be silently tidied away.
    clipped = np.clip(z, lo_, hi_)
    report = {
        "iters": it,
        "max_violation": float(np.max(np.abs(clipped - z))) if n else 0.0,
        "saturated": float(np.mean(np.isclose(clipped, lo_) | np.isclose(clipped, hi_))),
    }
    return clipped, report


def rotate_offset(du: float | np.ndarray, dv: float | np.ndarray,
                  deg: float) -> float | np.ndarray:
    """Where a point at frame offset (du, dv) lands horizontally after `rotate`.

    `render.clip` counter-rotates by `-radians(applied)` to take out a tilt of
    `+applied`, so a feature the tracker measured on an unrotated proxy is not
    where the crop will find it. Both offsets are from the frame centre and in
    the same units; the answer is the new horizontal offset in those units.

    The sign is measured, not derived — `rotation sign` in
    `tools/reframe_selftest.py` puts a mark at a known offset, runs it through
    ffmpeg's own `rotate` at the angle `render` would apply, and checks that
    this function predicted where it came out. Getting it backwards would double
    the error rather than remove it, which is precisely the failure `level.py`
    refuses to guess at.
    """
    a = np.radians(-deg)
    return du * np.cos(a) - dv * np.sin(a)


def crop_path(t: np.ndarray, u: np.ndarray, v: np.ndarray,
              t_in: float, t_out: float, frame_w: int, frame_h: int,
              crop_w: int, *, applied_deg: float = 0.0,
              dead: float = DEAD_ZONE, lam_v: float = LAMBDA_V,
              lam_a: float = LAMBDA_A, anchor: float = ANCHOR,
              hold_s: float = HOLD_S,
              ease_s: float = EASE_S, pad_s: float = PAD_S,
              solve_hz: float = SOLVE_HZ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop-left in pixels over one clip, on a grid of times in source seconds.

    `t`, `u`, `v` come straight from `track.parquet` — normalised box centres in
    the **render-oriented** frame, NaN where the tracker had nothing — and should
    be passed in already covering `[t_in - pad_s, t_out + pad_s]` where the ride
    has that much footage.

    The clip is solved with `pad_s` of context either side and the pad then
    thrown away. Without it the smoothness terms' natural boundary condition is
    zero derivative, which freezes the first second of every clip into a hold it
    did not earn. Twenty extra grid points is a cheap way not to have that.
    """
    dt = 1.0 / float(solve_hz)
    lo_t, hi_t = t_in - pad_s, t_out + pad_s
    grid = np.arange(lo_t, hi_t + dt * 0.5, dt)

    t = np.asarray(t, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float) if v is not None else np.full(u.shape, 0.5)

    # Where the crop's left edge must be for the dog to be centred, in crop
    # widths. The rotation correction moves the dog, not the crop, so it is
    # applied to the measured position before that conversion.
    du = (u - 0.5) * frame_w
    dv = (v - 0.5) * frame_h
    x_dog = frame_w / 2.0 + (rotate_offset(du, dv, applied_deg)
                             if applied_deg else du)
    z = (x_dog - crop_w / 2.0) / crop_w
    centre = (frame_w - crop_w) / 2.0 / crop_w

    c, w = targets(t, z, grid, centre=centre, hold_s=hold_s, ease_s=ease_s)
    lo, hi = 0.0, (frame_w - crop_w) / crop_w
    zz, report = solve(c, w, dt, dead=dead, lam_v=lam_v, lam_a=lam_a,
                       anchor=anchor, lo=lo, hi=hi)

    x = zz * crop_w
    keep = (grid >= t_in - 1e-9) & (grid <= t_out + 1e-9)
    tracked = np.isfinite(z)
    inside = tracked & (t >= t_in) & (t <= t_out)
    n_in = int(((t >= t_in) & (t <= t_out)).sum())

    pan = np.abs(np.diff(zz[keep])) / dt if keep.sum() > 1 else np.zeros(1)
    report.update({
        "hits": float(inside.sum() / n_in) if n_in else 0.0,
        "n_samples": n_in,
        "held": float(np.mean(np.abs(zz[keep] - c[keep]) <= dead + 1e-9))
                if keep.any() else 0.0,
        "pan_p50": float(np.percentile(pan, 50)),
        "pan_p95": float(np.percentile(pan, 95)),
        "max_violation_px": report.pop("max_violation") * crop_w,
        "pan_range_px": float(frame_w - crop_w),
    })
    return grid[keep], x[keep], report


def sendcmd(t: np.ndarray, x: np.ndarray, t_in: float, t_out: float,
            frame_w: int, crop_w: int, step_s: float = 0.02) -> str:
    """A `sendcmd` script driving the crop filter's `x` over the clip.

    The same shape as `level.sendcmd`, for the same reasons and with the same
    grid: a frame holds the last command before it, so the step has to beat the
    frame interval rather than merely look fine, and the midpoint of each step
    is sampled so the error is halved at both ends. `crop`'s `x` carries
    ffmpeg's runtime-command flag, which is what makes this possible at all.

    Values come out **even integers**. `crop` snaps `x` to the chroma
    subsampling on yuv420p regardless, so emitting what will actually be used
    means a check can assert an exact number instead of a tolerance, and nobody
    later wonders why a smooth path arrived in two-pixel stairsteps.
    """
    if not len(t):
        return ""
    grid = np.arange(0.0, max(t_out - t_in, step_s), step_s)
    vals = np.interp(grid + t_in + step_s / 2, t, x)
    vals = np.clip(np.nan_to_num(vals), 0, max(frame_w - crop_w, 0))
    vals = (np.rint(vals).astype(int)) & ~1
    return "".join(f"{g:.3f} crop x {v:d};\n" for g, v in zip(grid, vals))
