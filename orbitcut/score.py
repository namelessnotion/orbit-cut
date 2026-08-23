"""Phase 1 — turn telemetry into a per-second picture of how exciting a ride is.

Five features, all from sensors, none from pixels:

    air_s       seconds of freefall in this second      ACCL @ 200 Hz
    rough       high-band vibration, m/s^2 RMS          ACCL @ 200 Hz
    yaw_rate    heading change, rad/s                   GYRO . GRAV
    lat_accel   cornering force, m/s^2                  yaw_rate x speed
    speed_ms    ground speed                            GPS
    grade       descent, m/s vertical                   GPS altitude

Two things here are worth understanding before changing anything.

**Axis order never comes into it.** GoPro's accelerometer axis order varies by
generation and this parser does not normalise it, so every feature below is
built to be invariant to it: roughness and airtime use |accel|, which is a
magnitude; yaw rate is the component of angular velocity *about the gravity
axis*, obtained by projecting GYRO onto GRAV. Rotating about gravity is exactly
what turning is, and a dot product does not care which axis is which.

**Raw units, not scores.** This module stores physical quantities. Converting
them to a 0-1 "how exciting" number needs the rest of the library for context,
so that lives in `calibrate.py` and happens at read time. Rescoring after a
weight change then costs nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from . import config

# --- airtime ---------------------------------------------------------------
# In freefall an accelerometer reads ~0: the sensor and its case fall together.
# On the ground it reads ~9.8 plus whatever the trail is doing. The gap between
# those two states is enormous, which is why this detector needs no tuning to
# speak of and no model at all.
AIR_THRESHOLD = 3.5      # m/s^2 — comfortably below 1 g, above sensor noise
AIR_MIN_S = 0.10         # shorter than this is a wheel skipping, not a jump
AIR_LANDING_G = 18.0     # m/s^2 — a landing spike must follow, or it wasn't air
AIR_LANDING_WINDOW = 0.6 # seconds after touchdown to look for that spike

ROUGH_BAND = (5.0, 40.0)  # Hz — chatter and impacts, above body movement

# --- descent ---------------------------------------------------------------
# GPS altitude is the least accurate thing a GoPro records: metres of noise per
# sample, plus outright jumps when the receiver re-locks. Differencing adjacent
# seconds — the obvious way to get a rate of descent — amplifies exactly that,
# and measured against a simulated Michigan profile it correlated 0.26 with the
# real descent while reporting peaks 24x too large.
#
# The separation that makes this fixable: altitude noise is per-sample, and a
# trail descent lasts tens of seconds. So take the slope over a window instead
# of the difference between neighbours, after a median filter to remove the
# jumps. That lifts the correlation to ~0.9 on the same profile.
#
# The window is a trade. Longer is cleaner and blurs short descents; 21 s keeps
# a useful signal-to-noise ratio on an 8 s drop and roughly triples it on a
# long one.
ALT_MEDIAN_S = 9         # kills re-lock jumps without touching real slopes
GRADE_WINDOW_S = 21      # seconds of slope fit
GPS_DOP_MAX = 6.0        # dilution of precision above this is not a fix worth using

# Even after the slope fit, altitude keeps producing impossible descents: a
# median filter removes spikes, but a receiver re-lock is a *step*, and a 44 m
# step through a 21 s slope window reads as 3.2 m/s of descent. There is no
# filter that separates a real step from a fake one.
#
# So bound it by physics instead. You cannot descend faster than you are
# travelling, and a rideable trail tops out near a 30% gradient. Seconds that
# violate that are seconds where the altitude is wrong, and the honest value
# for them is "unknown", not a clipped number that piles up at the limit.
#
# This bites hardest exactly where the altitude is worst — stationary, under
# canopy, receiver re-acquiring — because the bound scales with speed.
MAX_GRADIENT = 0.60      # rise/run; ~31 degrees, far past anything rideable
MAX_DESCENT_MS = 3.0     # absolute fallback when speed is unknown
STAGE_VERSION = 3


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end indices of each True run."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def detect_air(imu: pd.DataFrame) -> list[dict[str, float]]:
    """Freefall events. Returns t_start, t_end, duration, landing peak."""
    cols = [c for c in imu.columns if c.startswith("accl_")]
    if len(cols) != 3 or len(imu) < 10:
        return []
    t = imu["t"].to_numpy()
    mag = np.linalg.norm(imu[cols].to_numpy(), axis=1)
    rate = len(t) / max(t[-1] - t[0], 1e-6)

    events = []
    for i0, i1 in _runs(mag < AIR_THRESHOLD):
        dur = t[min(i1, len(t) - 1)] - t[i0]
        if dur < AIR_MIN_S:
            continue
        j = min(i1 + int(AIR_LANDING_WINDOW * rate), len(mag))
        peak = float(mag[i1:j].max()) if j > i1 else 0.0
        if peak < AIR_LANDING_G:
            continue          # went quiet but never came down: not a jump
        events.append({
            "t_start": float(t[i0]),
            "t_end": float(t[min(i1, len(t) - 1)]),
            "duration": float(dur),
            "landing": peak,
        })
    return events


def roughness(imu: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """RMS of the 5-40 Hz band of |accel|, per second of the grid."""
    cols = [c for c in imu.columns if c.startswith("accl_")]
    if len(cols) != 3 or len(imu) < 64:
        return np.full(len(grid), np.nan)
    t = imu["t"].to_numpy()
    mag = np.linalg.norm(imu[cols].to_numpy(), axis=1)
    rate = len(t) / max(t[-1] - t[0], 1e-6)
    if rate < 2 * ROUGH_BAND[1]:
        return np.full(len(grid), np.nan)

    sos = butter(4, ROUGH_BAND, btype="band", fs=rate, output="sos")
    band = sosfiltfilt(sos, mag - np.nanmean(mag))

    out = np.full(len(grid), np.nan)
    idx = np.searchsorted(t, grid)
    for i, start in enumerate(idx):
        end = np.searchsorted(t, grid[i] + 1.0)
        if end - start > 8:
            out[i] = float(np.sqrt(np.mean(band[start:end] ** 2)))
    return out


def yaw_rate(tel: pd.DataFrame) -> np.ndarray:
    """Angular velocity about the gravity axis — i.e. how fast heading changes.

    Projecting GYRO onto the unit gravity vector sidesteps the axis-order
    problem entirely, and is the physically correct definition of turning.
    """
    gyro = [c for c in tel.columns if c.startswith("gyro_")]
    grav = [c for c in tel.columns if c.startswith("grav_")]
    if len(gyro) != 3 or len(grav) != 3:
        return np.full(len(tel), np.nan)
    g = tel[grav].to_numpy(dtype=float)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    g = np.divide(g, n, out=np.zeros_like(g), where=n > 1e-6)
    w = tel[gyro].to_numpy(dtype=float)
    return np.abs((w * g).sum(axis=1))


def descent_rate(alt: np.ndarray) -> np.ndarray:
    """Metres per second of descent, positive downhill.

    Least-squares slope over a centred window. With evenly spaced samples that
    reduces to one convolution with k / sum(k^2), which is both exact and cheap
    — no loop over windows and no curve fitting.
    """
    good = np.isfinite(alt)
    if good.sum() < max(GRADE_WINDOW_S, 5):
        return np.full(len(alt), np.nan)

    # Bridge short gaps so the filter sees an evenly spaced series; seconds that
    # had no altitude are put back to NaN at the end.
    idx = np.arange(len(alt), dtype=float)
    filled = np.interp(idx, idx[good], alt[good])
    filled = median_filter(filled, size=ALT_MEDIAN_S, mode="nearest")

    half = GRADE_WINDOW_S // 2
    k = np.arange(-half, half + 1, dtype=float)
    kernel = (k / (k ** 2).sum())[::-1]
    slope = np.convolve(np.pad(filled, half, mode="edge"), kernel, mode="valid")

    out = -slope[:len(alt)]        # descending is positive
    out[~good] = np.nan
    return out


def bound_by_speed(grade: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, float]:
    """Blank descents the rider's own speed says are impossible.

    Returns the bounded grade and the fraction of known seconds rejected. That
    fraction is worth surfacing: a few percent is ordinary GPS, while a large
    number means the altitude is unusable on that ride and `grade` should not be
    trusted there at all.
    """
    out = np.asarray(grade, dtype=float).copy()
    known = np.isfinite(out)
    if not known.any():
        return out, 0.0

    limit = np.where(np.isfinite(speed), np.abs(speed) * MAX_GRADIENT,
                     MAX_DESCENT_MS)
    # A stationary rider can still drift a little; never bound below a floor, or
    # every second of a pause becomes NaN on GPS jitter alone.
    limit = np.maximum(limit, 0.35)
    bad = known & (np.abs(out) > limit)
    out[bad] = np.nan
    return out, float(bad.sum() / max(known.sum(), 1))


def compute(telemetry_path: str, imu_path: str | None, duration_s: float) -> pd.DataFrame:
    """One row per second of the ride. Raw physical units throughout."""
    tel = pd.read_parquet(telemetry_path)
    grid = np.arange(0.0, max(duration_s, 1.0), 1.0)
    out = pd.DataFrame({"t": grid})

    def on_grid(col: str) -> np.ndarray:
        if col not in tel.columns:
            return np.full(len(grid), np.nan)
        v = tel[col].to_numpy(dtype=float)
        good = np.isfinite(v)
        if good.sum() < 2:
            return np.full(len(grid), np.nan)
        return np.interp(grid, tel["t"].to_numpy()[good], v[good])

    # --- turning ------------------------------------------------------------
    yr = yaw_rate(tel)
    if np.isfinite(yr).any():
        tt = tel["t"].to_numpy()
        good = np.isfinite(yr)
        # Mean over each second, not a sample: a turn is a second-long event.
        out["yaw_rate"] = [
            float(np.nanmean(yr[good][(tt[good] >= s) & (tt[good] < s + 1)]))
            if ((tt[good] >= s) & (tt[good] < s + 1)).any() else np.nan
            for s in grid
        ]
    else:
        out["yaw_rate"] = np.nan

    # --- speed and descent (GPS only) --------------------------------------
    speed = on_grid("gps_speed2d")
    # A stationary GoPro still reports a metre or two per second of GPS noise.
    speed = np.where(np.isfinite(speed) & (speed >= 0) & (speed < 30), speed, np.nan)

    # Drop the seconds before the receiver had a lock — they read as a slow
    # crawl across the county, which is exactly the wrong thing to average in.
    #
    # Only when the stream actually reports a lock somewhere. GPS5's fix is
    # sticky metadata and can be absent altogether, in which case the parser
    # reports 0 for the whole ride: a fix field that never says yes is a field
    # we cannot trust, not a ride that never got GPS. Failing open here means a
    # missing fix costs nothing; failing closed would silently discard the GPS
    # of every camera that does not record GPSF.
    fix = on_grid("gps_fix")
    if np.isfinite(fix).any() and np.nanmax(fix) >= 2:
        speed = np.where(fix >= 1.5, speed, np.nan)     # 2 = 2D lock, 3 = 3D
    out["speed_ms"] = speed

    alt = on_grid("gps_alt")
    if np.isfinite(fix).any() and np.nanmax(fix) >= 2:
        # Altitude is the least trustworthy GPS field there is; without a lock
        # it is not noisy, it is fiction, and `grade` differentiates it.
        alt = np.where(fix >= 1.5, alt, np.nan)
    # GPS9 reports dilution of precision per sample. A high DOP is the receiver
    # telling you the fix is geometrically weak, which is where the altitude
    # jumps come from — so drop those seconds rather than filtering them later.
    dop = on_grid("gps_dop")
    if np.isfinite(dop).any():
        alt = np.where(np.isfinite(dop) & (dop > GPS_DOP_MAX), np.nan, alt)
    grade, rejected = bound_by_speed(descent_rate(alt), speed)
    out["grade"] = grade
    out.attrs["grade_rejected"] = rejected

    # Cornering force is what actually looks fast on screen: a hard turn at
    # speed, not a slow one. Without GPS this stays NaN and the composite
    # renormalises over whatever it does have.
    out["lat_accel"] = out["yaw_rate"] * out["speed_ms"]

    # --- accelerometer features --------------------------------------------
    if imu_path and Path(imu_path).exists():
        imu = pd.read_parquet(imu_path)
        out["rough"] = roughness(imu, grid)
        air = np.zeros(len(grid))
        for ev in detect_air(imu):
            # Credit the whole event to the second it starts in, so a jump
            # never gets split across two rows and diluted.
            i = int(ev["t_start"])
            if 0 <= i < len(air):
                air[i] = max(air[i], ev["duration"])
        out["air_s"] = air
    else:
        out["rough"] = np.nan
        out["air_s"] = np.nan

    return out


def summarise(scores: pd.DataFrame, events: list[dict[str, float]]) -> dict[str, Any]:
    def q(col: str, p: float) -> float | None:
        if col not in scores or not np.isfinite(scores[col]).any():
            return None
        return float(np.nanpercentile(scores[col], p))

    return {
        "seconds": int(len(scores)),
        "air_events": len(events),
        "air_total_s": round(sum(e["duration"] for e in events), 2),
        "air_longest_s": round(max((e["duration"] for e in events), default=0.0), 2),
        "rough_p50": q("rough", 50),
        "rough_p95": q("rough", 95),
        "yaw_p95": q("yaw_rate", 95),
        "speed_p50": q("speed_ms", 50),
        "speed_p95": q("speed_ms", 95),
        "has_gps": bool(np.isfinite(scores["speed_ms"]).any()),
    }


def score_asset(row: Any) -> tuple[pd.DataFrame, list[dict], dict]:
    scores = compute(row["telemetry_path"], row["imu_path"], row["duration_s"] or 1.0)
    events = []
    if row["imu_path"] and Path(row["imu_path"]).exists():
        events = detect_air(pd.read_parquet(row["imu_path"]))
    out_dir = config.derived_dir(row["content_hash"])
    scores.to_parquet(out_dir / "scores.parquet", index=False)
    if events:
        pd.DataFrame(events).to_parquet(out_dir / "air_events.parquet", index=False)
    return scores, events, summarise(scores, events)
