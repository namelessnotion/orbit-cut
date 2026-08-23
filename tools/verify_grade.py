"""Check `grade` against a trail's own elevation profile.

Every fix to GPS altitude so far has been an argument from physics rather than a
measurement: descents that cannot happen at the measured speed, steps too large
to be terrain. Those bound the error; they do not measure it.

A GPX closes that gap, because the position fix is good even when the altitude
is not — latitude and longitude survived even the SCAL bug that zeroed speed.

    python tools/verify_grade.py <ride> <trail.gpx>

**How the truth channel works, and why not the obvious way.** The obvious method
is to take each second's nearest point on the trail, read its elevation, and
difference that. It is too fragile to convict anyone: matching quantises to a
vertex and differencing amplifies the error, so at the ~10 m of horizontal error
a real ride carries, a *perfect* elevation source still only reproduces itself
at a correlation of 0.42. The measurement would be the thing under test.

Instead the trail is treated as a function of distance along it. Its gradient is
smooth and slowly varying, so a few metres of positional error barely moves it,
and descent rate is `gradient(position) * speed` — using the rider's own speed,
which is now trustworthy. Same fixture, same 10 m of error: 0.70 rather than
0.42.

**Read the magnitude line before the correlation.** Magnitude does not depend on
matching at all: a trail with 23 m of total relief cannot produce 5 m/s of
descent however badly the positions line up. Correlation is the more informative
number when the match is good and the weaker one when it is not.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_RADIUS_M = 20.0      # beyond this, the rider was not on the mapped trail


def read_gpx(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lat, lon, ele from every trkpt/rtept/wpt that has an elevation."""
    root = ET.parse(path).getroot()
    lat, lon, ele = [], [], []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag not in ("trkpt", "rtept", "wpt"):
            continue
        e = next((c for c in el if c.tag.rsplit("}", 1)[-1] == "ele"), None)
        if e is None or e.text is None:
            continue
        try:
            lat.append(float(el.attrib["lat"]))
            lon.append(float(el.attrib["lon"]))
            ele.append(float(e.text))
        except (KeyError, ValueError):
            continue
    return np.array(lat), np.array(lon), np.array(ele)


def to_metres(lat: np.ndarray, lon: np.ndarray,
              lat0: float, lon0: float) -> np.ndarray:
    """Local equirectangular projection — exact enough over a few km."""
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
    return np.column_stack([(lon - lon0) * m_per_deg_lon,
                            (lat - lat0) * m_per_deg_lat])


def implied_gradient(ele: np.ndarray, s: np.ndarray) -> float:
    """p99 of |d(elevation)/d(distance)| along the trail.

    Replaces a raw jaggedness figure, which was misleading: second differences
    scale with point spacing, so a perfectly good profile sampled every 8 m
    looked "noisy" next to one sampled every 2 m. Gradient is spacing-invariant
    and physically interpretable — a trail cannot sustain much past 30%, so a
    profile implying more than that is carrying noise, whatever its source.
    """
    if len(ele) < 3:
        return float("nan")
    step = np.gradient(np.asarray(ele, float), np.asarray(s, float))
    return float(np.percentile(np.abs(step[np.isfinite(step)]), 99))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    ride, gpx_path = sys.argv[1], sys.argv[2]

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scipy.spatial import cKDTree
    from orbitcut import db, score as score_mod

    conn = db.connect()
    rows = [r for r in db.assets(conn)
            if ride in (r["ride_id"] or "") or ride in (r["filename"] or "")
            or (r["content_hash"] or "").startswith(ride)]
    rows = [r for r in rows if r["telemetry_path"]]
    if not rows:
        print(f"no ingested asset matching {ride!r}")
        return 1

    glat, glon, gele = read_gpx(gpx_path)
    if len(glat) < 10:
        print(f"{gpx_path}: found {len(glat)} points with elevation — need a track "
              f"with an elevation profile")
        return 1

    lat0, lon0 = float(np.median(glat)), float(np.median(glon))
    tpts = to_metres(glat, glon, lat0, lon0)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(tpts, axis=0), axis=1))])

    # Gradient along the trail, lightly smoothed over ~25 m of distance. This is
    # the truth channel, and it is deliberately *not* a difference of matched
    # elevations: matching quantises to a vertex and differencing amplifies
    # that, so at the ~10 m of GPS error a real ride carries, a perfect signal
    # degrades to a correlation of 0.42 through the measurement alone. Gradient
    # varies slowly along the trail, so the same error barely moves it.
    spacing = max(float(np.median(np.diff(along))), 0.5)
    win = max(3, int(25 / spacing) | 1)
    smooth = pd.Series(gele).rolling(win, center=True, min_periods=1).mean().to_numpy()
    tgrad = np.gradient(smooth, np.maximum(along, np.arange(len(along)) * 1e-6))

    g99 = implied_gradient(smooth, along)
    print(f"\ntrail: {Path(gpx_path).name}")
    print(f"  {len(glat)} points ~{spacing:.1f} m apart, elevation "
          f"{gele.min():.1f}-{gele.max():.1f} m ({gele.max() - gele.min():.1f} m of relief)")
    print(f"  steepest 1% of the profile: {g99 * 100:.0f}% gradient — "
          + ("plausible terrain" if g99 < 0.35 else
             "too steep to be terrain; this profile carries GPS noise, so "
             "treat the comparison below as a magnitude check only"))

    tree = cKDTree(tpts)

    for r in sorted(rows, key=lambda r: r["chapter"] or 0):
        tel = pd.read_parquet(r["telemetry_path"])
        if not {"gps_lat", "gps_lon"} <= set(tel.columns):
            print(f"\n  {r['filename']}: no GPS columns")
            continue

        dur = r["duration_s"] or 1.0
        grid = np.arange(0.0, max(dur, 1.0), 1.0)
        t = tel["t"].to_numpy()

        def on_grid(col):
            v = tel[col].to_numpy(dtype=float)
            g = np.isfinite(v)
            return np.interp(grid, t[g], v[g]) if g.sum() > 1 else np.full(len(grid), np.nan)

        rlat, rlon = on_grid("gps_lat"), on_grid("gps_lon")
        good = np.isfinite(rlat) & np.isfinite(rlon) & (np.abs(rlat) > 0.01)
        if good.sum() < 60:
            print(f"\n  {r['filename']}: only {good.sum()} located seconds")
            continue

        dist, idx = tree.query(to_metres(rlat[good], rlon[good], lat0, lon0))
        on_trail = dist <= MATCH_RADIUS_M

        print(f"\n  {r['filename']}")
        print(f"    {good.sum()} located seconds, {on_trail.sum()} within "
              f"{MATCH_RADIUS_M:.0f} m of the trail "
              f"({on_trail.sum() / good.sum() * 100:.0f}%), "
              f"median offset {np.median(dist):.1f} m")
        if on_trail.sum() < 60:
            print("    too little overlap — is this the right trail for this ride?")
            continue

        # Truth: the trail's own gradient where he was, times how fast he was
        # going there. Needs his speed, which is finally trustworthy.
        speed = on_grid("gps_speed2d") if "gps_speed2d" in tel.columns else None
        if speed is None:
            print("    no speed column — re-extract telemetry first")
            continue
        truth = np.full(len(grid), np.nan)
        sel = np.flatnonzero(good)[on_trail]
        truth[sel] = -tgrad[idx[on_trail]] * np.abs(speed[sel])

        mine = pd.read_parquet(r["scores_path"])["grade"].to_numpy() \
            if r["scores_path"] and Path(r["scores_path"]).exists() else None
        if mine is None:
            print("    not scored yet")
            continue
        n = min(len(mine), len(truth))
        both = np.isfinite(mine[:n]) & np.isfinite(truth[:n])
        if both.sum() < 60:
            print(f"    only {both.sum()} seconds comparable")
            continue

        a, b = mine[:n][both], truth[:n][both]
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"    {'':16}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}")
        for name, v in (("from GoPro alt", a), ("from trail", b)):
            print(f"    {name:16}{np.percentile(v,50):>8.2f}{np.percentile(v,90):>8.2f}"
                  f"{np.percentile(v,99):>8.2f}{v.max():>8.2f}")
        print(f"    correlation {corr:>.3f}   RMS error "
              f"{np.sqrt(np.mean((a - b) ** 2)):.3f} m/s   over {both.sum()} s")
        # Magnitude is the robust half of this test: it does not depend on
        # matching precision at all. The trail's total relief bounds what any
        # descent rate on it can be, however badly the positions line up.
        ratio = np.percentile(a, 99) / max(np.percentile(b, 99), 1e-6)
        if ratio > 3:
            print(f"    p99 is {ratio:.0f}x what this trail's relief allows — the "
                  f"altitude is wrong regardless of how well positions matched")
        if corr > 0.7:
            print("    -> grade is measuring real descent")
        elif corr > 0.4:
            print("    -> partly real; usable but noisy")
        else:
            print("    -> grade is not tracking the terrain on this ride")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
