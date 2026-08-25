"""Turn raw physical features into a 0-1 composite, calibrated on your library.

Why calibration is separate from scoring: "how exciting" is not a property of a
second of footage, it is a comparison against the rest of what you shoot. A
5 m/s^2 rock garden is unremarkable if every ride you own is rocky and it is the
highlight of the year if none of them are. So `score.py` stores physics, this
module stores the corpus distribution, and the composite is computed where the
two meet — which also means changing a weight costs nothing to re-apply.

One feature deliberately escapes percentile ranking. Airtime is zero for well
over 99% of seconds, so its percentile is meaningless: a 0.2 s hop and a 0.9 s
send would both land at "the 99.7th percentile". It gets an absolute saturating
scale instead, because a second of air is impressive on its own terms and does
not need the rest of the library to say so.

The level is then ranked a second time, and that is not redundant. Renormalising
a weighted mean over "whatever features are present" sounds fair and is not: a
mean of two terms is more variable than a mean of four, so it reaches high
values more often even when every underlying feature is identically
distributed. Rides shot without GPS have two features instead of four, and the
effect is not subtle — in a simulation with no real difference between the
rides at all, the two-feature rides took all six of the top six places, exactly
as they did on the real library. So each level is ranked against other rows
carrying the *same* information, which asks "how good is this for what we can
measure here" and makes the answer comparable across rides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

# Percentile-ranked against the corpus.
RANKED = ("speed_ms", "rough", "yaw_rate", "lat_accel", "grade")
# Absolute scale: seconds of air that counts as a full-marks jump.
AIR_FULL_S = 0.80

# Weights, set by ear against rides Anthony remembers. Continuous features only
# — see AIR_GAIN for why airtime is not in here.
#
# These moved a long way once the turn feature stopped being a vibration meter.
# The earlier note here described speed carrying half and said the point was
# "a ride with one real sprint beats a ride that is merely brisk throughout";
# that rationale outlived the numbers it justified, which is worse than no
# rationale, so it is gone. Roughness now carries most of the composite.
#
# Worth being clear about what that means, because it looks like a regression
# and is not. `turn` used to correlate 0.62-0.81 with `rough`, so a roughness-
# heavy result was arriving through a feature that claimed to measure
# cornering. Now the two are close to independent and the weighting says
# outright that terrain is what makes this footage worth watching — which is a
# plausible thing for bikejoring on Michigan singletrack, where the dog sets a
# fairly even pace and the trail supplies the variation.
#
# Still taste, not correctness, and still provisional: the approve/reject log
# is what eventually replaces both these and SHARPNESS with fitted numbers.
# `orbitcut fit` already knows how to check whether it can beat them.
WEIGHTS = {
    "speed": 0.1,
    "turn": 0.3,
    "rough": 0.6,
    # Descent is switched off, and the reason is measured rather than assumed.
    # Checked against a Trailforks profile of the trail actually ridden — 85% of
    # seconds within 20 m of it — GoPro altitude produced descents peaking at
    # 5.45 m/s where the trail's entire relief is 23 m and its own profile
    # allows 0.41. Twelve times too large, and uncorrelated with the terrain.
    #
    # The feature is still computed and stored, because the fault is the
    # *source*, not the idea: rate of descent is worth having. When stage 1's
    # map-matching lands, grade should come from trail elevation as
    # `gradient(s) * speed` — smooth in position, so it survives the ~10 m of
    # GPS error that makes differencing altitudes hopeless. Re-enable it then,
    # and confirm with tools/verify_grade.py rather than by assuming.
    "descent": 0.0,
}

# Airtime combines with the level instead of averaging into it.
#
# Averaging was the first attempt and it was wrong: a 0.71 s jump is one second
# out of ninety, so a three-second rolling mean — which is right for everything
# else, because a clip is seconds long — halved it and dropped both planted
# jumps out of the top five. A rare event and a sustained level are different
# kinds of evidence and must not be averaged.
#
# So: probabilistic OR. Each source independently lifts the score toward 1, and
# nothing can dilute anything else. A jump on a quiet second still scores; a
# jump during hard riding scores higher still.
AIR_GAIN = 0.75
AIR_DILATE_S = 1  # a jump's take-off and landing belong to it too
CALIBRATION = "calibration.json"
GRID = np.linspace(0, 100, 101)


# How much a peak in one feature counts against being decent at all of them.
#
# The composite used to be a weighted arithmetic mean, and that quietly encoded
# the wrong taste: averaging punishes specialisation, and every exciting second
# is specialised. Fast means straight, so a sprint scores near zero on turn.
# Twisty means slow. A rock garden is neither fast nor flowing. With the mean,
# a second that is 0.62/0.68/0.64 — brisk, never special — beat a sprint at
# 0.99/0.20/0.55, a switchback section at 0.30/0.97/0.50, and a rock garden at
# 0.25/0.45/0.98. All three of the memorable seconds lost to the forgettable
# one, which is exactly what the ranking looked like on the real library.
#
# A power mean with p > 1 leans toward the maximum without ignoring the rest:
# p = 1 is the arithmetic mean, p -> infinity is the maximum. At p = 3 all three
# specialised seconds beat the merely-brisk one, while a second that is strong
# everywhere still beats a second that is strong at one thing.
#
# This is the same lesson AIR_GAIN already encodes for airtime — different kinds
# of evidence must not be averaged into each other — applied one level up.
# Chosen against the real library rather than a simulation. High p leans hard
# toward the single best feature in a second, which is the point — a highlight
# is a moment that is outstanding at one thing, not adequate at everything.
#
# The original 8.0 was picked when speed carried half the weight, to stop
# sprint rides interleaving with merely-consistent ones. With one feature now
# carrying most of the composite, sharpness does less work than it did: a power
# mean over weights of 0.2/0.2/0.6 is already close to "whatever rough says".
# It is kept high because it still separates a specialised second from an
# adequate one on the two minor features.
SHARPNESS = 12

FEATURES = ("speed", "turn", "rough", "descent")
SUB = {"speed": "s_speed", "turn": "s_turn", "rough": "s_rough", "descent": "s_descent"}
MIN_BUCKET = 200  # below this, a bucket's percentiles are noise


def _level(
    frame: pd.DataFrame, weights: dict[str, float], sharpness: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean of the present sub-scores, plus a per-row availability key.

    The key is a bitmask of which features contributed, and it is what makes
    the second ranking possible.
    """
    stack = np.vstack([frame[SUB[k]].to_numpy(dtype=float) for k in FEATURES])
    w = np.array([weights.get(k, 0.0) for k in FEATURES])[:, None]
    # A feature weighted zero is absent, not present-and-ignored. It must drop
    # out of the availability key too, or turning one off splits the corpus into
    # buckets that differ only by a feature nothing is using.
    present = np.isfinite(stack) & (w > 0)
    wsum = (w * present).sum(axis=0)
    # Power mean, not arithmetic — see SHARPNESS. Sub-scores are already 0-1 so
    # the exponent is well behaved; clipping guards against a percentile rank
    # landing a hair outside the range.
    p = float(sharpness or SHARPNESS)
    powered = np.clip(np.where(present, stack, 0.0), 0.0, 1.0) ** p
    total = np.nansum(powered * w, axis=0)
    level = np.where(wsum > 0, (total / np.maximum(wsum, 1e-9)) ** (1.0 / p), np.nan)
    key = (present * (1 << np.arange(len(FEATURES)))[:, None]).sum(axis=0)
    return level, key.astype(int)


def _sub_scores(frame: pd.DataFrame, feats: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()

    def ranked(col: str) -> np.ndarray:
        if col not in out or col not in feats:
            return np.full(len(out), np.nan)
        return np.where(
            np.isfinite(out[col]),
            _rank(out[col].to_numpy(dtype=float), feats[col]["breaks"]),
            np.nan,
        )

    out["s_speed"] = ranked("speed_ms")
    out["s_rough"] = ranked("rough")
    # Cornering force where GPS allows it, bare turn rate where it does not.
    lat, yaw = ranked("lat_accel"), ranked("yaw_rate")
    out["s_turn"] = np.where(np.isfinite(lat), lat, yaw)
    out["s_descent"] = ranked("grade")
    out["s_air"] = (
        np.clip(out["air_s"].to_numpy(dtype=float) / AIR_FULL_S, 0, 1)
        if "air_s" in out
        else np.nan
    )
    return out


def fit(
    score_paths: list[str],
    weights: dict[str, float] | None = None,
    sharpness: float | None = None,
) -> dict[str, Any]:
    """Build the corpus distribution for each ranked feature."""
    frames = []
    for p in score_paths:
        if Path(p).exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise ValueError("no scores.parquet found — run `orbitcut score` first")
    allrows = pd.concat(frames, ignore_index=True)

    table: dict[str, Any] = {
        "n_seconds": int(len(allrows)),
        "n_assets": len(frames),
        "features": {},
        "missing": {},
    }
    # A feature that is absent everywhere used to drop out of this table in
    # silence, which is how a GPS column-naming bug survived a whole library:
    # every ride scored on turn and roughness alone and nothing said so. An
    # empty feature is now reported, because it is nearly always a plumbing
    # failure rather than a fact about the riding.
    for col in RANKED:
        if col not in allrows:
            table["missing"][col] = "column absent from every scores.parquet"
            continue
        v = allrows[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            table["missing"][col] = (
                f"only {len(v)} finite samples in " f"{len(allrows)} seconds"
            )
            continue
        # A feature with no spread is not a feature. Percentile-ranking a
        # constant assigns every second the same score, which then dilutes the
        # composite for every ride that "has" it — worse than not having it at
        # all, and invisible in a table of percentiles that are all the same
        # number. This is what mis-scaled GPS speed looked like: 21,000 samples,
        # p50 = p90 = p99 = 0.00, and a bucket claiming four features.
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        if not np.isfinite(hi - lo) or hi - lo <= 1e-9:
            table["missing"][col] = (
                f"{len(v)} samples but no spread "
                f"(p1 = p99 = {lo:.4g}) — constant, not usable"
            )
            continue
        table["features"][col] = {
            "breaks": [float(x) for x in np.percentile(v, GRID)],
            "n": int(len(v)),
        }

    # Second pass: the distribution of the level itself, per availability
    # bucket. This is what removes the fewer-features-scores-higher bias.
    weights = weights or WEIGHTS
    sharpness = float(sharpness or SHARPNESS)
    table["weights"] = dict(weights)
    table["sharpness"] = sharpness
    levels, keys = [], []
    for f in frames:
        lv, k = _level(_sub_scores(f, table["features"]), weights, sharpness)
        good = np.isfinite(lv)
        levels.append(lv[good])
        keys.append(k[good])
    lv_all = np.concatenate(levels) if levels else np.array([])
    k_all = np.concatenate(keys) if keys else np.array([])

    table["levels"] = {}
    for key in np.unique(k_all):
        v = lv_all[k_all == key]
        if len(v) < MIN_BUCKET:
            continue
        table["levels"][str(int(key))] = {
            "breaks": [float(x) for x in np.percentile(v, GRID)],
            "n": int(len(v)),
            "features": "+".join(
                f for i, f in enumerate(FEATURES) if int(key) >> i & 1
            ),
        }
    if len(lv_all):
        table["levels"]["global"] = {
            "breaks": [float(x) for x in np.percentile(lv_all, GRID)],
            "n": int(len(lv_all)),
            "features": "any",
        }
    return table


def weights_match(
    table: dict[str, Any] | None, weights: dict[str, float] | None
) -> bool:
    """Were the table's level buckets built with these weights?

    Only the ratios matter, since the level divides by the sum of whatever is
    present. A mismatch is not an error — the sub-score percentiles are still
    valid and re-weighting them is the whole point of trying weights out — but
    the level buckets are then a distribution of a slightly different quantity,
    so cross-ride comparison is approximate until you recalibrate.
    """
    fitted = (table or {}).get("weights")
    if not fitted or not weights:
        return True

    def ratios(w: dict[str, float]) -> np.ndarray:
        v = np.array([float(w.get(k, 0.0)) for k in FEATURES])
        return v / max(v.sum(), 1e-9)

    return bool(np.allclose(ratios(fitted), ratios(weights), atol=1e-6))


def save(table: dict[str, Any]) -> Path:
    config.ensure_dirs()
    p = config.ROOT / CALIBRATION
    p.write_text(json.dumps(table, indent=2))
    return p


def load() -> dict[str, Any] | None:
    p = config.ROOT / CALIBRATION
    return json.loads(p.read_text()) if p.exists() else None


def _rank(values: np.ndarray, breaks: list[float]) -> np.ndarray:
    """Percentile rank of each value against the corpus, as 0-1."""
    b = np.asarray(breaks, dtype=float)
    # np.interp needs an increasing x; ties in the corpus can flatten it.
    b = np.maximum.accumulate(b)
    return np.clip(np.interp(values, b, GRID / 100.0), 0.0, 1.0)


def apply(
    scores: pd.DataFrame,
    table: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    sharpness: float | None = None,
) -> pd.DataFrame:
    """Add normalised sub-scores and the composite. Returns a new frame."""
    table = table or load() or {}
    # Default to the weights the table was fitted with, not the module's. The
    # level distribution stored in the table is a distribution *of levels*, and
    # a level computed with different weights is a different quantity.
    weights = weights or table.get("weights") or WEIGHTS
    sharpness = float(sharpness or table.get("sharpness") or SHARPNESS)
    out = _sub_scores(scores, table.get("features", {}))

    level, key = _level(out, weights, sharpness)
    out["level_raw"] = level

    # Rank the level against rows that had the same features available.
    buckets = table.get("levels", {})
    ranked = np.full(len(out), np.nan)
    if buckets:
        for k in np.unique(key):
            sel = key == k
            b = buckets.get(str(int(k))) or buckets.get("global")
            if b:
                ranked[sel] = _rank(level[sel], b["breaks"])
    else:
        ranked = level  # uncalibrated: fall back to the raw level
    out["level"] = ranked

    # Smooth the level — a clip is seconds long and single-second wobble is not
    # what anyone watches — but smooth it *before* air is folded in.
    level_s = pd.Series(ranked).rolling(3, center=True, min_periods=1).mean().to_numpy()

    air = out["s_air"].to_numpy(dtype=float)
    if np.isfinite(air).any():
        # Dilate rather than average: max over a small window, so the seconds
        # either side of a jump inherit it without the jump being reduced.
        air = (
            pd.Series(np.nan_to_num(air))
            .rolling(2 * AIR_DILATE_S + 1, center=True, min_periods=1)
            .max()
            .to_numpy()
        )
    else:
        air = np.zeros(len(out))
    out["s_air_d"] = air

    base = np.where(np.isfinite(level_s), level_s, 0.0)
    out["composite"] = 1.0 - (1.0 - base) * (1.0 - AIR_GAIN * air)
    out["composite_s"] = out["composite"]
    return out
