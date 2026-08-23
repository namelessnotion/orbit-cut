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

# Starting weights. Replace these with values fitted to your own approve/reject
# decisions once phase 4's log has a hundred or so in it — that is the whole
# point of recording them.
# Continuous features only — see AIR_GAIN for why airtime is not in here.
WEIGHTS = {
    "speed": 0.27,
    "turn": 0.34,
    "rough": 0.25,
    "descent": 0.14,
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
AIR_DILATE_S = 1          # a jump's take-off and landing belong to it too
CALIBRATION = "calibration.json"
GRID = np.linspace(0, 100, 101)


def fit(score_paths: list[str]) -> dict[str, Any]:
    """Build the corpus distribution for each ranked feature."""
    frames = []
    for p in score_paths:
        if Path(p).exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise ValueError("no scores.parquet found — run `orbitcut score` first")
    allrows = pd.concat(frames, ignore_index=True)

    table: dict[str, Any] = {"n_seconds": int(len(allrows)), "n_assets": len(frames),
                             "features": {}}
    for col in RANKED:
        if col not in allrows:
            continue
        v = allrows[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        table["features"][col] = {
            "breaks": [float(x) for x in np.percentile(v, GRID)],
            "n": int(len(v)),
        }
    return table


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


def apply(scores: pd.DataFrame, table: dict[str, Any] | None = None,
          weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Add normalised sub-scores and the composite. Returns a new frame."""
    table = table or load()
    weights = weights or WEIGHTS
    out = scores.copy()
    feats = (table or {}).get("features", {})

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
    out["s_air"] = (np.clip(out["air_s"].to_numpy(dtype=float) / AIR_FULL_S, 0, 1)
                    if "air_s" in out else np.nan)

    # Renormalise over whatever is actually present. A ride with no GPS is
    # scored on roughness, turning and air rather than being penalised for the
    # camera settings it was shot with.
    cols = {"speed": "s_speed", "turn": "s_turn",
            "rough": "s_rough", "descent": "s_descent"}
    stack = np.vstack([out[c].to_numpy(dtype=float) for c in cols.values()])
    w = np.array([weights[k] for k in cols])[:, None]
    present = np.isfinite(stack)
    wsum = (w * present).sum(axis=0)
    total = np.nansum(np.where(present, stack * w, 0.0), axis=0)
    level = np.where(wsum > 0, total / np.maximum(wsum, 1e-9), np.nan)

    # Smooth the level — a clip is seconds long and single-second wobble is not
    # what anyone watches — but smooth it *before* air is folded in.
    level_s = (pd.Series(level).rolling(3, center=True, min_periods=1)
               .mean().to_numpy())
    out["level"] = level_s

    air = out["s_air"].to_numpy(dtype=float)
    if np.isfinite(air).any():
        # Dilate rather than average: max over a small window, so the seconds
        # either side of a jump inherit it without the jump being reduced.
        air = (pd.Series(np.nan_to_num(air))
               .rolling(2 * AIR_DILATE_S + 1, center=True, min_periods=1)
               .max().to_numpy())
    else:
        air = np.zeros(len(out))
    out["s_air_d"] = air

    base = np.where(np.isfinite(level_s), level_s, 0.0)
    out["composite"] = 1.0 - (1.0 - base) * (1.0 - AIR_GAIN * air)
    out["composite_s"] = out["composite"]
    return out
