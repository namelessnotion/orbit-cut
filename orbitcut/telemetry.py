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

# Streams whose component order is documented and stable, keyed by arity.
#
# Keying by arity is not over-engineering, it is the fix for a bug that cost a
# whole library's worth of GPS. GPS5 is five fields in the GPMF spec, but
# telemetrik appends the sticky GPSF (fix) and GPSP (precision) values to every
# sample, so what arrives here is seven. The old table listed five names, the
# arity did not match, and the positional fallback below quietly produced
# `gps5_0 … gps5_6` instead. Nothing crashed. `gps_speed2d` simply never
# existed, so speed, grade and cornering force were NaN on every ride in the
# library while `inventory` cheerfully reported GPS present — it was reading
# the stream list, which was correct.
NAMED_COLUMNS = {
    "GPS5": {
        5: ["gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d"],
        7: ["gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d",
            "gps_fix", "gps_dop"],
    },
    # HERO11 and later. Same first five fields, then a timestamp split into
    # days-since-2000 and seconds-since-midnight, then per-sample DOP and fix.
    # Per-sample is a real gain over GPS5, whose fix is sticky across a whole
    # payload and so cannot show a lock coming and going.
    "GPS9": {
        9: ["gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d",
            "gps_days", "gps_secs", "gps_dop", "gps_fix"],
    },
    "CORI": {4: ["cori_w", "cori_x", "cori_y", "cori_z"]},
    "IORI": {4: ["iori_w", "iori_x", "iori_y", "iori_z"]},
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
        known = NAMED_COLUMNS.get(key)
        names = (known or {}).get(arity)
        if names is None:
            # Falling back is right for ACCL/GYRO/GRAV and wrong for a stream we
            # thought we knew: it renames the fields everything downstream looks
            # up by name, and does it silently. Say so.
            if known:
                print(f"  ! {key}: {arity} fields, expected "
                      f"{' or '.join(str(a) for a in sorted(known))} — "
                      f"falling back to positional names, so nothing that reads "
                      f"{known[min(known)][0]} will find it. Check the parser version.")
            names = [f"{key.lower()}_{i}" for i in range(arity)]
        arr = np.array(
            [v if isinstance(v, (list, tuple)) and len(v) == arity else [np.nan] * arity
             for v in values],
            dtype=float,
        )
        return {n: arr[:, i] for i, n in enumerate(names)}
    name = SCALAR_STREAMS.get(key, key.lower())
    return {name: np.array([float(v) if v is not None else np.nan for v in values])}


MAX_GPS_GAP_S = 5.0
# How long a GPS dropout may be before the bridge across it stops counting as a
# measurement. `np.interp` will happily draw a straight line across a two-minute
# canopy dropout and hold the first and last values flat beyond the ends, and
# the parquet keeps no record of which grid seconds were actually measured — so
# a fabricated speed is indistinguishable from a real one downstream. Five
# seconds is short enough that nothing real is lost and long enough to bridge
# the odd missed sample.


GPS_EPOCH = "2000-01-01T00:00:00Z"
# GPS9 carries the only clock in the file that is not the camera's own. Its
# `gps_days` counts days since 2000-01-01 UTC and `gps_secs` is seconds into
# that day, both straight from the satellites.
#
# This matters because the camera's clock was 53 to 95 days behind reality
# across this library, drifting further between sessions, and `recorded_at`
# came from the MP4 container's creation_time — the camera's clock. Sun
# elevation computed from it labelled eighteen daylight rides as night. The
# humble exposure fallback got every one of them right; the "exact, closed-form"
# solar calculation was wrong because its input was fiction.
CLOCK_DRIFT_WARN_S = 300.0


def gps_start_utc(df: pd.DataFrame) -> str | None:
    """True UTC of the first sample, from the satellites rather than the camera.

    `gps_time - t` rather than the timestamp itself, because the GPS stream does
    not necessarily start at the same instant as the video.

    Only locked samples count. An unlocked receiver still emits a timestamp,
    and it is a placeholder: five files in this library report a constant
    `gps_days` of 7736 with fix 0 and DOP 100 throughout, which decodes to
    2021-03-07T00:00:00 and read as the camera clock being 1795 days *fast*.
    Taking the first sample believed it. The median of the locked samples is
    both gated and robust to the odd bad row.
    """
    if "gps_days" not in df.columns or "gps_secs" not in df.columns:
        return None
    days = df["gps_days"].to_numpy(dtype=float)
    secs = df["gps_secs"].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    ok = np.isfinite(days) & np.isfinite(secs) & np.isfinite(t) & (days > 3650)
    if "gps_fix" in df.columns:
        fix = df["gps_fix"].to_numpy(dtype=float)
        ok &= np.isfinite(fix) & (fix >= 2)
    if ok.sum() < 10:
        return None
    start = (pd.Timestamp(GPS_EPOCH)
             + pd.to_timedelta(days[ok], unit="D")
             + pd.to_timedelta(secs[ok] - t[ok], unit="s"))
    return pd.Series(start).median().strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def clock_drift_s(container_time: str | None, gps_time: str | None) -> float | None:
    """How far the camera's clock is from the satellites', in seconds."""
    if not container_time or not gps_time:
        return None
    try:
        a = pd.Timestamp(container_time)
        b = pd.Timestamp(gps_time)
    except (ValueError, TypeError):
        return None
    if a.tzinfo is None:
        a = a.tz_localize("UTC")
    if b.tzinfo is None:
        b = b.tz_localize("UTC")
    return float((b - a).total_seconds())


def _gap_aware_interp(grid: np.ndarray, t: np.ndarray,
                      v: np.ndarray) -> np.ndarray:
    """Interpolate onto `grid`, but leave NaN where nothing was measured.

    Absence has to stay visible. This project has twice paid for a pipeline that
    degraded quietly — the GPS column naming and the corpus that renormalised
    over missing features — and an interpolated dropout is the same failure in a
    new place: it does not look like missing data, it looks like a slow steady
    crawl in a straight line.
    """
    out = np.interp(grid, t, v)
    # Outside the measured span entirely: np.interp clamps, which invents a
    # stationary receiver for as long as the ride keeps going.
    out[(grid < t[0]) | (grid > t[-1])] = np.nan
    if len(t) > 1:
        i = np.searchsorted(t, grid, side="right")
        prev = np.clip(i - 1, 0, len(t) - 1)
        nxt = np.clip(i, 0, len(t) - 1)
        out[np.where(t[nxt] - t[prev] > MAX_GPS_GAP_S, True, False)] = np.nan
    return out


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

    # GPS is parsed here rather than by telemetrik, for both stream types and
    # for different reasons — see gps.py. This deliberately runs last and

    # overwrites any gps_* columns already in `frame`: telemetrik's GPS5 values
    # are scaled by a single divisor, which leaves latitude and longitude right
    # and everything else wrong by orders of magnitude.
    try:
        from . import gps
        fix = gps.extract(path)
    except Exception as exc:
        print(f"  ! GPS: {exc}")
        fix = None
    if fix:
        for name, col in fix["columns"].items():
            good = np.isfinite(col)
            if good.sum() >= 2:
                frame[name] = _gap_aware_interp(grid, fix["t"][good], col[good])
        if fix["key"] not in present:
            present = sorted(present + [fix["key"]])
        span = float(fix["t"][-1] - fix["t"][0]) if fix["n"] > 1 else 0.0
        rates[fix["key"]] = round(fix["n"] / span, 1) if span > 0 else 0.0

    df = pd.DataFrame(frame)

    # Speed is the field most likely to be wrong without looking wrong, because
    # a scaling error leaves position perfectly plausible. Check it against what
    # a bicycle does rather than against zero.
    if "gps_speed2d" in df:
        sp = df["gps_speed2d"].to_numpy(dtype=float)
        sp = sp[np.isfinite(sp)]
        if len(sp):
            p95 = float(np.percentile(sp, 95))
            if p95 < 0.5:
                print(f"  ! GPS speed p95 is {p95:.5g} m/s — too slow to be riding. "
                      f"Suspect a SCAL divisor, not a slow ride.")
            elif p95 > 30:
                print(f"  ! GPS speed p95 is {p95:.5g} m/s ({p95 * 3.6:.0f} km/h) — "
                      f"too fast for a bike. Suspect a SCAL divisor.")

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
        "gps_start_utc": gps_start_utc(df),
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
        "gps_speed_p50": None,
        "gps_speed_p95": None,
        "gps_locked_fraction": None,
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

    # Speed is reported here for one reason: it is the number that makes a
    # broken GPS extraction obvious. A stream list saying GPS5 tells you the
    # camera wrote location data, not that this pipeline read it.
    if "gps_speed2d" in df.columns:
        sp = df["gps_speed2d"].to_numpy(dtype=float)
        sp = sp[np.isfinite(sp)]
        if len(sp):
            out["gps_speed_p50"] = float(np.median(sp))
            out["gps_speed_p95"] = float(np.percentile(sp, 95))

    if "gps_fix" in df.columns:
        fx = df["gps_fix"].to_numpy(dtype=float)
        if np.isfinite(fx).any():
            out["gps_locked_fraction"] = float(np.mean(fx[np.isfinite(fx)] >= 1.5))

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
