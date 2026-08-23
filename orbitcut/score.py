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
STAGE_VERSION = 1


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
    out["speed_ms"] = speed

    alt = on_grid("gps_alt")
    if np.isfinite(alt).any():
        smooth = pd.Series(alt).rolling(5, center=True, min_periods=2).mean().to_numpy()
        out["grade"] = -np.gradient(smooth)     # positive = descending
    else:
        out["grade"] = np.nan

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
