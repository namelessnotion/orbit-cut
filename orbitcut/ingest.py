"""Phase 0 orchestration.

Every stage is a plain function taking a path and returning a dict. That is not
an accident — when you eventually want these on a Celery worker, wrapping them
is a decorator rather than a rewrite.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from . import (config, db, hashing, naming, probe as probe_mod, proxy as proxy_mod,
               telemetry as tel_mod, thumbs as thumbs_mod)


def find_videos(root: str | Path) -> Iterator[Path]:
    p = Path(root).expanduser()
    if p.is_file():
        yield p
        return
    for f in sorted(p.rglob("*")):
        # Skip GoPro's own low-res companions — we make our own proxies.
        if f.is_file() and f.suffix in config.VIDEO_SUFFIXES and f.suffix.lower() != ".lrv":
            yield f


def ingest_one(path: Path, conn, force: bool = False) -> dict[str, Any]:
    ch = hashing.content_hash(path)
    summary: dict[str, Any] = {"file": path.name, "hash": ch, "stages": {}}

    ride_id, chapter = naming.parse(path.name)
    db.upsert_asset(
        conn, ch,
        filename=path.name,
        source_path=str(path.resolve()),
        host=_hostname(),
        ride_id=ride_id,
        chapter=chapter,
    )

    # --- probe ---------------------------------------------------------------
    if force or not db.stage_done(conn, ch, "probe"):
        started = db.now()
        try:
            meta = probe_mod.probe(path)
            db.upsert_asset(conn, ch, **meta)
            db.record_stage(conn, ch, "probe", "ok", started)
            summary["stages"]["probe"] = "ok"
        except Exception as exc:
            db.record_stage(conn, ch, "probe", "error", started, str(exc))
            summary["stages"]["probe"] = f"error: {exc}"
            return summary
    else:
        summary["stages"]["probe"] = "cached"

    row = conn.execute("SELECT * FROM asset WHERE content_hash = ?", (ch,)).fetchone()

    # --- telemetry (fast: no video decode) -----------------------------------
    if not row["has_gpmd"]:
        summary["stages"]["telemetry"] = "skipped: no gpmd track"
    elif force or not db.stage_done(conn, ch, "telemetry"):
        started = db.now()
        try:
            tel = tel_mod.extract(path, ch)
            fields = {
                "telemetry_path": tel["telemetry_path"],
                "imu_path": tel["imu_path"],
                "streams": tel["streams"],
                "gps_lat": tel["gps_lat"],
                "gps_lon": tel["gps_lon"],
                "horizon_locked": tel["horizon_locked"],
            }
            if tel["gps_lat"] is not None and row["recorded_at"]:
                elev = tel_mod.sun_elevation(tel["gps_lat"], tel["gps_lon"], row["recorded_at"])
                fields["sun_elevation"] = elev
                fields["lighting"] = tel_mod.lighting_label(elev)
                fields["lighting_source"] = "sun"
            else:
                # No GPS, so no solar elevation. The camera's own exposure
                # response still says how dark it thought the scene was.
                import pandas as _pd
                fields["lighting"] = tel_mod.lighting_from_exposure(
                    _pd.read_parquet(tel["telemetry_path"])
                ) or "unknown"
                fields["lighting_source"] = "exposure"
            db.upsert_asset(conn, ch, **fields)
            db.record_stage(conn, ch, "telemetry", "ok", started)
            summary["stages"]["telemetry"] = f"ok ({len(tel['streams'])} streams)"
            summary["telemetry"] = tel
        except Exception as exc:
            db.record_stage(conn, ch, "telemetry", "error", started, str(exc))
            summary["stages"]["telemetry"] = f"error: {exc}"
    else:
        summary["stages"]["telemetry"] = "cached"

    # --- proxy (slow: full decode) -------------------------------------------
    if force or not db.stage_done(conn, ch, "proxy"):
        started = db.now()
        try:
            result = proxy_mod.make_proxy(path, ch)
            db.upsert_asset(conn, ch, **result)
            db.record_stage(conn, ch, "proxy", "ok", started)
            summary["stages"]["proxy"] = "ok"
        except Exception as exc:
            db.record_stage(conn, ch, "proxy", "error", started, str(exc))
            summary["stages"]["proxy"] = f"error: {exc}"
    else:
        summary["stages"]["proxy"] = "cached"

    # --- contact sheet (reads the proxy, so it is nearly free) ---------------
    row = conn.execute("SELECT * FROM asset WHERE content_hash = ?", (ch,)).fetchone()
    if not row["proxy_path"]:
        summary["stages"]["thumbs"] = "skipped: no proxy"
    elif force or not db.stage_done(conn, ch, "thumbs"):
        started = db.now()
        try:
            result = thumbs_mod.make_contact_sheet(
                row["proxy_path"], ch, row["duration_s"] or 1.0
            )
            db.upsert_asset(conn, ch, **result)
            db.record_stage(conn, ch, "thumbs", "ok", started)
            summary["stages"]["thumbs"] = "ok"
        except Exception as exc:
            db.record_stage(conn, ch, "thumbs", "error", started, str(exc))
            summary["stages"]["thumbs"] = f"error: {exc}"
    else:
        summary["stages"]["thumbs"] = "cached"

    return summary


def _hostname() -> str:
    import socket
    return socket.gethostname()
