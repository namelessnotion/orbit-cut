"""Stage 3 — turn a per-second curve into a handful of clips worth watching.

Naive top-N is the obvious approach and it fails in a specific way: the best
second, the second-best second and the third-best second are usually the same
ten seconds of trail, so you get five near-identical clips from one rowdy
section and nothing from the rest of the ride. Everything here exists to avoid
that.

Four ideas, in the order they are applied:

**Grow from peaks, do not slide fixed windows.** A clip's length should come
from how long the good bit lasts, not from a constant. Each peak is extended
outwards while the curve stays above a fraction of that peak, then clamped to a
watchable length. A 9 s burst and a 19 s sustained section both come out at
their natural length.

**Lead in.** The run-in is what makes a jump land, so every clip starts earlier
than its action by `LEAD_S`. Nobody wants to arrive mid-air.

**Suppress neighbours.** Candidates must be separated, and the separation is
measured between clip *edges*, not peaks: two peaks 10 s apart whose clips are
14 s long still overlap, and a gap rule written on peaks would happily return
both.

**Then diversify.** A penalty for repeating a dominant feature type is applied
*after* suppression, not folded into the score before it. Fold it in early and
the penalty changes which clips suppress which, so a single mediocre turn-clip
can knock out the best turn-clip in the ride and the result depends on
evaluation order. Applied afterwards it only reorders what survived.

Hard constraints from the plan: never cut inside a detected freefall window —
a clip that starts mid-air is unusable — and prefer in-points where the
composite is rising, which growing from a peak gives for free.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# A clip has to be long enough to establish and short enough to hold attention.
CLIP_MIN_S = 10.0
CLIP_MAX_S = 30.0
LEAD_S = 1.5  # run-in before the action starts
GAP_S = 9.0  # minimum quiet between one clip ending and the next
GROW_FRACTION = 0.55  # extend while the curve holds this share of the peak
PEAK_PERCENTILE = 70.0  # a peak below this is not worth considering at all
# ...and a peak has to stand out from the ride, not merely sit above its median.
# Without this the floor lands in the noise on a quiet ride and the selector
# pads its output with flat sections, which read as confident recommendations.
# `limit` is a cap, not a quota: four good clips beats six with two duds.
MIN_PROMINENCE = 0.30  # share of (p95 - median) a peak must clear
# An absolute floor as well, which is only meaningful because `calibrate.py`
# makes the composite a percentile against the whole library rather than a raw
# physical quantity. 0.55 is "better than the median second you have ever shot".
# A purely relative gate cannot tell a flat ride from a varied one — every ride
# has a best 30%, including the dull ones — so without this a boring ride still
# produces six confident-looking recommendations.
MIN_SCORE = 0.55
DIVERSITY = 0.75  # score multiplier per earlier clip of the same type
MAX_CANDIDATES = 6

# Which sub-score was loudest at the peak. `s_air` is included even though
# airtime is not in the weighted level: a jump is what a clip is *about* when
# there is one, whatever the arithmetic says.
TYPES = {"s_speed": "speed", "s_turn": "turn", "s_rough": "rough", "s_air_d": "jump"}
STAGE_VERSION = 1


def _peaks(y: np.ndarray, floor: float) -> list[int]:
    """Indices of local maxima above `floor`, strongest first.

    Plateaus matter here — a sustained rowdy section is flat at the top, and a
    strict `>` on both sides finds nothing in it. Comparing one side loosely
    picks the first index of a plateau, which is where the clip should start
    growing from anyway.
    """
    idx = [
        i
        for i in range(1, len(y) - 1)
        if y[i] >= floor and y[i] > y[i - 1] and y[i] >= y[i + 1]
    ]
    if len(y) and y[0] >= floor and (len(y) == 1 or y[0] > y[1]):
        idx.append(0)
    return sorted(idx, key=lambda i: -y[i])


def _grow(y: np.ndarray, peak: int, duration: float) -> tuple[float, float]:
    """Extend outward from a peak while the curve holds up, then clamp."""
    thresh = y[peak] * GROW_FRACTION
    lo = hi = peak
    while lo > 0 and y[lo - 1] >= thresh:
        lo -= 1
    while hi < len(y) - 1 and y[hi + 1] >= thresh:
        hi += 1

    t_in, t_out = float(lo), float(hi + 1)
    t_in -= LEAD_S
    # Too short: grow symmetrically. Too long: keep the part around the peak,
    # since that is the bit that earned the clip.
    if t_out - t_in < CLIP_MIN_S:
        short = CLIP_MIN_S - (t_out - t_in)
        t_in -= short / 2
        t_out += short / 2
    if t_out - t_in > CLIP_MAX_S:
        centre = float(peak)
        t_in = max(t_in, centre - (CLIP_MAX_S - LEAD_S))
        t_out = min(t_out, t_in + CLIP_MAX_S)
    return max(0.0, t_in), min(duration, t_out)


def _protect_air(
    t_in: float, t_out: float, events: pd.DataFrame | None
) -> tuple[float, float]:
    """Never start or end inside a freefall window; snap outward instead."""
    if events is None or not len(events):
        return t_in, t_out
    for _, e in events.iterrows():
        s, t = float(e["t_start"]), float(e["t_end"])
        if s < t_in < t:
            t_in = s - 0.5
        if s < t_out < t:
            t_out = t + 0.5
    return max(0.0, t_in), t_out


def _dominant(scored: pd.DataFrame, lo: int, hi: int) -> tuple[str, dict[str, float]]:
    """Which feature carried this window, and the full vector for the log."""
    vec: dict[str, float] = {}
    for col, name in TYPES.items():
        if col in scored:
            v = scored[col].to_numpy(dtype=float)[lo:hi]
            v = v[np.isfinite(v)]
            vec[name] = float(v.max()) if len(v) else float("nan")
    live = {k: v for k, v in vec.items() if np.isfinite(v)}
    return (max(live, key=live.get) if live else "unknown"), vec


def candidates(
    scored: pd.DataFrame,
    events: pd.DataFrame | None,
    duration: float,
    limit: int = MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Clips worth reviewing, best first."""
    if "composite" not in scored or not len(scored):
        return []
    y = scored["composite"].to_numpy(dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    if not y.any():
        return []

    median = float(np.median(y))
    span = float(np.percentile(y, 95)) - median
    floor = max(
        float(np.percentile(y, PEAK_PERCENTILE)), median + MIN_PROMINENCE * span
    )
    picked: list[dict[str, Any]] = []

    for p in _peaks(y, floor):
        t_in, t_out = _grow(y, p, duration)
        t_in, t_out = _protect_air(t_in, t_out, events)
        if t_out - t_in < CLIP_MIN_S * 0.8:
            continue
        # Edge-to-edge separation, not peak-to-peak — see the module docstring.
        if any(t_in < c["t_out"] + GAP_S and c["t_in"] - GAP_S < t_out for c in picked):
            continue

        lo, hi = int(max(0, t_in)), int(min(len(y), np.ceil(t_out)))
        score = float(np.mean(y[lo : max(hi, lo + 1)]))
        # The window's own average must clear the bar too. A tall spike sitting
        # in a flat stretch grows into a mostly-flat clip, and the clip is what
        # gets watched, not the spike.
        if score < max(MIN_SCORE, median + MIN_PROMINENCE * span * 0.5):
            continue
        kind, vec = _dominant(scored, lo, max(hi, lo + 1))
        picked.append(
            {
                "t_in": round(t_in, 2),
                "t_out": round(t_out, 2),
                "duration": round(t_out - t_in, 2),
                "peak_t": float(p),
                "peak": float(y[p]),
                "score": score,
                "dominant": kind,
                "features": vec,
            }
        )
        if len(picked) >= limit * 3:  # room for diversity to reorder
            break

    # Diversity last: a penalty applied before suppression would change which
    # clips knock out which, making the result depend on evaluation order.
    seen: dict[str, int] = {}
    for c in sorted(picked, key=lambda c: -c["score"]):
        n = seen.get(c["dominant"], 0)
        c["adjusted"] = c["score"] * (DIVERSITY**n)
        seen[c["dominant"]] = n + 1

    out = sorted(picked, key=lambda c: -c["adjusted"])[:limit]
    for i, c in enumerate(out, 1):
        c["rank"] = i
    return out
