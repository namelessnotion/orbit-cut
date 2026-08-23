"""SQLite store. One writer, one machine — this carries you to phase 5."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS asset (
    content_hash   TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    host           TEXT,
    ride_id        TEXT,
    chapter        INTEGER,
    bytes          INTEGER,

    duration_s     REAL,
    container      TEXT,
    vcodec         TEXT,
    width          INTEGER,
    height         INTEGER,
    aspect         TEXT,
    fps            REAL,
    pix_fmt        TEXT,
    bit_depth      INTEGER,
    rotation       INTEGER,
    bitrate_bps    INTEGER,
    camera_model   TEXT,
    recorded_at    TEXT,

    has_gpmd       INTEGER,
    streams        TEXT,          -- json list of FourCCs actually present
    gps_lat        REAL,
    gps_lon        REAL,
    sun_elevation  REAL,
    lighting       TEXT,          -- day | twilight | night | unknown
    lighting_source TEXT,         -- sun | exposure
    horizon_locked INTEGER,       -- derived from IORI, not from a setting

    proxy_path     TEXT,
    contact_path   TEXT,
    scores_path    TEXT,
    air_events     INTEGER,
    air_total_s    REAL,
    air_longest_s  REAL,
    telemetry_path TEXT,
    imu_path       TEXT,
    archived_path  TEXT,
    archived_at    TEXT,

    first_seen     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_run (
    content_hash  TEXT NOT NULL,
    stage         TEXT NOT NULL,
    stage_version INTEGER NOT NULL,
    status        TEXT NOT NULL,       -- ok | error
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT,
    PRIMARY KEY (content_hash, stage)
);

"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_asset_recorded ON asset(recorded_at);
CREATE INDEX IF NOT EXISTS idx_asset_ride     ON asset(ride_id, chapter);
CREATE INDEX IF NOT EXISTS idx_stage_status   ON stage_run(stage, status);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)          # must precede the indexes: they reference columns
    conn.executescript(INDEXES)
    _backfill_rides(conn)
    return conn


def _backfill_rides(conn: sqlite3.Connection) -> None:
    """Populate ride_id/chapter for rows ingested before those columns existed.

    Both are a pure function of the filename, which is already stored — so this
    costs one cheap UPDATE rather than a re-ingest of the originals. Anything
    derivable from data already in the database should be repaired here, not by
    making the user reprocess terabytes.
    """
    from . import naming

    rows = conn.execute(
        "SELECT content_hash, filename FROM asset WHERE ride_id IS NULL"
    ).fetchall()
    if not rows:
        return
    updates = []
    for r in rows:
        ride_id, chapter = naming.parse(r["filename"] or "")
        if ride_id:
            updates.append((ride_id, chapter, r["content_hash"]))
    if updates:
        conn.executemany(
            "UPDATE asset SET ride_id = ?, chapter = ? WHERE content_hash = ?", updates
        )
        conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any columns SCHEMA gained since this database was created.

    `CREATE TABLE IF NOT EXISTS` silently does nothing on an existing table, so
    a new column in SCHEMA would otherwise surface as an OperationalError on the
    next insert. Additive-only, which covers every change so far.
    """
    for table in ("asset", "stage_run"):
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not have:
            continue
        for line in SCHEMA.splitlines():
            line = line.split("--")[0].strip().rstrip(",").strip()
            if not line or line.startswith(("CREATE", "PRAGMA", ")", "(")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, decl = parts[0], parts[1]
            if name.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK"}:
                continue
            if name not in have and _column_belongs(table, name):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()


def _column_belongs(table: str, name: str) -> bool:
    body = SCHEMA.split(f"CREATE TABLE IF NOT EXISTS {table}")[1].split(");")[0]
    return any(line.strip().startswith(name + " ") for line in body.splitlines())


def upsert_asset(conn: sqlite3.Connection, content_hash: str, **fields: Any) -> None:
    """Insert or update. Only the keys you pass are touched."""
    if "streams" in fields and isinstance(fields["streams"], (list, tuple)):
        fields["streams"] = json.dumps(list(fields["streams"]))

    row = conn.execute(
        "SELECT 1 FROM asset WHERE content_hash = ?", (content_hash,)
    ).fetchone()

    if row is None:
        fields.setdefault("first_seen", now())
        fields.setdefault("filename", "")
        fields.setdefault("source_path", "")
        cols = ["content_hash"] + list(fields)
        marks = ",".join("?" * len(cols))
        conn.execute(
            f"INSERT INTO asset ({','.join(cols)}) VALUES ({marks})",
            [content_hash] + list(fields.values()),
        )
    elif fields:
        sets = ",".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE asset SET {sets} WHERE content_hash = ?",
            list(fields.values()) + [content_hash],
        )
    conn.commit()


def stage_done(conn: sqlite3.Connection, content_hash: str, stage: str) -> bool:
    """True when this stage already succeeded at the current stage_version.

    This is the idempotency check the whole design rests on: re-running ingest
    is free, and bumping a stage_version in config makes reprocessing selective.
    """
    row = conn.execute(
        "SELECT stage_version, status FROM stage_run WHERE content_hash = ? AND stage = ?",
        (content_hash, stage),
    ).fetchone()
    if row is None:
        return False
    return row["status"] == "ok" and row["stage_version"] >= config.STAGE_VERSIONS[stage]


def record_stage(
    conn: sqlite3.Connection,
    content_hash: str,
    stage: str,
    status: str,
    started_at: str,
    error: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO stage_run
               (content_hash, stage, stage_version, status, started_at, finished_at, error)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(content_hash, stage) DO UPDATE SET
               stage_version = excluded.stage_version,
               status        = excluded.status,
               started_at    = excluded.started_at,
               finished_at   = excluded.finished_at,
               error         = excluded.error""",
        (
            content_hash,
            stage,
            config.STAGE_VERSIONS[stage],
            status,
            started_at,
            now(),
            error,
        ),
    )
    conn.commit()


def assets(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    return conn.execute("SELECT * FROM asset ORDER BY recorded_at, filename").fetchall()
