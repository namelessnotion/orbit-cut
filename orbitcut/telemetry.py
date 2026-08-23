"""GPMF telemetry -> parquet.

Two outputs, deliberately:

  telemetry_10hz.parquet   every stream resampled onto one common grid. This is
                           what the scorers read.
  imu_raw.parquet          ACCL at its native ~200 Hz. The freefall detector
                           needs to resolve a 100 ms window, so it cannot use
                           the 10 Hz grid.

This stage decodes no video and finishes in seconds — which is why it is split
from proxy generation. Score a whole ride while the proxies are still churning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

# Streams whose component order is documented and stable.
NAMED_COLUMNS = {
    "GPS5": ["gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d"],
    "CORI": ["cori_w", "cori_x", "cori_y", "cori_z"],
    "IORI": ["iori_w", "iori_x", "iori_y", "iori_z"],
}
# Everything else gets positional names. ACCL/GYRO/GRAV axis order varies by
# camera generation, and the features that matter most — |accel| for roughness
# and airtime — are rotation-invariant, so this costs nothing. Run `verify` on
# your first file to see the real mapping before writing anything axis-specific.
SCALAR_STREAMS = {"SHUT": "shut", "ISOE": "isoe", "TMPC": "tmpc"}


def _timeline(stream: Any) -> tuple[np.ndarray, list[Any]]:
    """Seconds since start, plus values. Prefer PTS; fall back to STMP ms."""
    if stream.pts_data:
        pairs = stream.pts_data
        times = np.array([p[0] for p in pairs], dtype=float)
    else:
        pairs = stream.data
        times = np.array([p[0] for p in pairs], dtype=float) / 1000.0
    return times, [p[1] for p in pairs]


def _columns_for(key: str, values: list[Any]) -> dict[str, np.ndarray]:
    first = values[0]
    if isinstance(first, (list, tuple)):
        arity = len(first)
        names = NAMED_COLUMNS.get(key)
        if not names or len(names) != arity:
            names = [f"{key.lower()}_{i}" for i in range(arity)]
        arr = np.array(
            [v if isinstance(v, (list, tuple)) and len(v) == arity else [np.nan] * arity
             for v in values],
            dtype=float,
        )
        return {n: arr[:, i] for i, n in enumerate(names)}
    name = SCALAR_STREAMS.get(key, key.lower())
    return {name: np.array([float(v) if v is not None else np.nan for v in values])}


def extract(path: str | Path, content_hash: str) -> dict[str, Any]:
    import telemetrik

    from . import gpmf_compat
    gpmf_compat.apply()      # 64-bit MP4 boxes and co64 — see that module

    out_dir = config.derived_dir(content_hash)
    streams = telemetrik.extract_all_telemetry(str(path), streams=config.STREAMS)
    present = sorted(streams)

    # --- build the common 10 Hz grid -----------------------------------------
    end = 0.0
    for s in streams.values():
        if s.sample_count:
            t, _ = _timeline(s)
            end = max(end, float(t[-1]) if len(t) else 0.0)
    if end <= 0:
        raise ValueError("telemetry present but no usable timestamps")

    grid = np.arange(0.0, end, 1.0 / config.RESAMPLE_HZ)
    frame: dict[str, np.ndarray] = {"t": grid}
    rates: dict[str, float] = {}

    for key, stream in streams.items():
        if not stream.sample_count:
            continue
        try:
            times, values = _timeline(stream)
            if len(times) < 2:
                continue
            rates[key] = round(len(times) / max(times[-1], 1e-6), 1)
            for name, col in _columns_for(key, values).items():
                good = np.isfinite(col)
                if good.sum() < 2:
                    continue
                frame[name] = np.interp(grid, times[good], col[good])
        except Exception as exc:                      # one bad stream must not
            print(f"  ! {key}: {exc}")                # take the whole file down
            continue

    df = pd.DataFrame(frame)
    telemetry_path = out_dir / "telemetry_10hz.parquet"
    df.to_parquet(telemetry_path, index=False)

    # --- native-rate IMU for the freefall detector ---------------------------
    imu_path = None
    if "ACCL" in streams and streams["ACCL"].sample_count:
        at, av = _timeline(streams["ACCL"])
        acc = _columns_for("ACCL", av)
        imu = {"t": at, **acc}
        if "GYRO" in streams and streams["GYRO"].sample_count:
            gt, gv = _timeline(streams["GYRO"])
            for name, col in _columns_for("GYRO", gv).items():
                good = np.isfinite(col)
                if good.sum() >= 2:
                    imu[name] = np.interp(at, gt[good], col[good])
        imu_path = out_dir / "imu_raw.parquet"
        pd.DataFrame(imu).to_parquet(imu_path, index=False)

    return {
        "telemetry_path": str(telemetry_path),
        "imu_path": str(imu_path) if imu_path else None,
        "streams": present,
        "rates": rates,
        "duration_s": float(end),
        **_diagnostics(df),
    }


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (N,4) quaternion arrays, (w,x,y,z)."""
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def _qconj(q: np.ndarray) -> np.ndarray:
    out = q.astype(float).copy()
    out[:, 1:] *= -1
    return out


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate (N,3) vectors by (N,4) quaternions."""
    w, x, y, z = q.T
    u = np.stack([x, y, z], axis=1)
    return (2 * (u * v).sum(1, keepdims=True) * u
            + (w[:, None] ** 2 - (u * u).sum(1, keepdims=True)) * v
            + 2 * w[:, None] * np.cross(u, v))


def _vspread_deg(v: np.ndarray) -> float:
    """Mean angular distance of unit vectors from their mean direction."""
    v = v[np.isfinite(v).all(axis=1)]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = v[(n > 1e-9).ravel()] / n[n > 1e-9][:, None]
    if len(v) < 2:
        return float("nan")
    m = v.mean(axis=0)
    m /= np.linalg.norm(m)
    return float(np.degrees(np.arccos(np.clip(v @ m, -1, 1))).mean())


def _spread_deg(q: np.ndarray) -> float:
    """Mean angular distance from the mean orientation, in degrees.

    Convention-free: it never has to know which axis is roll, which matters
    because GoPro's quaternion axis order is not something to assume.
    """
    q = q[np.isfinite(q).all(axis=1)]
    if len(q) < 2:
        return float("nan")
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = q * np.sign(q @ q[0])[:, None]          # resolve double-cover sign flips
    m = q.mean(axis=0)
    m /= np.linalg.norm(m)
    return float(np.degrees(2 * np.arccos(np.clip(np.abs(q @ m), 0, 1))).mean())


def _diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Sanity numbers plus the two derived facts phase 1 wants."""
    out: dict[str, Any] = {
        "accl_mag_mean": None,
        "grav_mag_mean": None,
        "gps_fix_fraction": None,
        "gps_lat": None,
        "gps_lon": None,
        "horizon_locked": None,
    }

    accl = [c for c in df.columns if c.startswith("accl_")]
    if len(accl) == 3:
        out["accl_mag_mean"] = float(np.nanmean(np.linalg.norm(df[accl].to_numpy(), axis=1)))

    grav = [c for c in df.columns if c.startswith("grav_")]
    if len(grav) == 3:
        out["grav_mag_mean"] = float(np.nanmean(np.linalg.norm(df[grav].to_numpy(), axis=1)))

    if {"gps_lat", "gps_lon"} <= set(df.columns):
        lat, lon = df["gps_lat"].to_numpy(), df["gps_lon"].to_numpy()
        valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 0.01) & (np.abs(lat) <= 90)
        out["gps_fix_fraction"] = float(valid.mean())
        if valid.any():
            out["gps_lat"] = float(np.median(lat[valid]))
            out["gps_lon"] = float(np.median(lon[valid]))

    # Horizon Lock is detected here, never read from a capture setting.
    #
    # Two earlier attempts at this were wrong, and it is worth recording why.
    # "Is IORI non-identity" cannot separate real leveling from HyperSmooth's
    # ordinary rotational correction. Comparing the spread of CORI against
    # CORI x IORI fixes that but is swamped by yaw: over a ten-minute ride your
    # heading changes by more than a hundred degrees, which dwarfs the ~25 deg
    # of roll that leveling actually removes. Validated against a synthetic
    # ride with realistic yaw, full Horizon Lock scored +0.025 on that metric —
    # indistinguishable from no leveling at all.
    #
    # Gravity is the way out. Rotating about the gravity axis does not change
    # the gravity direction, so yaw cannot contaminate it. GRAV is gravity in
    # camera coordinates; apply IORI to express it in image coordinates. If the
    # camera is levelling, gravity stops moving in the frame while it still
    # swings in the body. Same synthetic ride: +1.00 locked, +0.60 partial,
    # -0.01 HyperSmooth-only. It is also immune to an inverted mount, because
    # it measures the variance of the direction and never its sign.
    grav_cols = [c for c in ("grav_0", "grav_1", "grav_2") if c in df.columns]
    iori_cols = ["iori_w", "iori_x", "iori_y", "iori_z"]
    if len(grav_cols) == 3:
        g = df[grav_cols].to_numpy(dtype=float)
        out["gravity_spread_deg"] = _vspread_deg(g)
        out["gravity_mean"] = [round(float(x), 3) for x in np.nanmean(g, axis=0)]

        if set(iori_cols) <= set(df.columns):
            iori = df[iori_cols].to_numpy(dtype=float)
            body = out["gravity_spread_deg"]
            # GoPro's composition convention is not something to assume, so try
            # both and keep whichever reduces the spread. Leveling can only ever
            # reduce it; the wrong convention cannot manufacture a reduction.
            candidates = [_vspread_deg(_qrot(_qconj(iori), g)),
                          _vspread_deg(_qrot(iori, g))]
            frame = min(candidates)
            out["gravity_spread_image_deg"] = frame
            if body and np.isfinite(body) and body > 0.5 and np.isfinite(frame):
                supp = 1.0 - frame / body
                out["roll_suppression"] = float(supp)
                out["horizon_locked"] = 1 if supp > 0.5 else 0

    return out


def lighting_from_exposure(df: pd.DataFrame) -> str | None:
    """Fallback for files with no GPS: the camera's own exposure response.

    Less precise than solar elevation, but it needs no location fix — and a
    ride with no GPS still records how dark the camera thought it was.
    """
    if "isoe" not in df.columns:
        return None
    iso = float(np.nanmedian(df["isoe"]))
    if not np.isfinite(iso):
        return None
    if iso < 400:
        return "day"
    if iso > 1600:
        return "night"
    # 400-1600 is genuinely ambiguous: dense canopy at midday and open sky at
    # dusk land in the same band. Solar elevation separates them and needs GPS.
    return "twilight"


def sun_elevation(lat: float, lon: float, when: str) -> float | None:
    """Day/night without a model: solar position is a closed-form function of
    latitude, longitude and UTC time. Optional dependency; returns None if absent."""
    try:
        from datetime import datetime
        from astral import Observer
        from astral.sun import elevation
    except ImportError:
        return None
    try:
        dt = datetime.fromisoformat(when.replace("Z", "+00:00"))
        return float(elevation(Observer(latitude=lat, longitude=lon), dt))
    except Exception:
        return None


def lighting_label(sun_elev: float | None) -> str:
    if sun_elev is None:
        return "unknown"
    if sun_elev > 0:
        return "day"
    if sun_elev > -6:
        return "twilight"
    return "night"
