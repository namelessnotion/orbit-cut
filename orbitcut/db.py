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

CREATE TABLE IF NOT EXISTS segment (
    id            INTEGER PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    t_in          REAL NOT NULL,
    t_out         REAL NOT NULL,
    rank          INTEGER,
    score         REAL,
    dominant      TEXT,             -- speed | turn | rough | jump
    features      TEXT,             -- json: the full vector, for the decision log
    status        TEXT DEFAULT 'candidate',   -- candidate|approved|rejected|rendered
    reason        TEXT,             -- reject reason chip, when there is one
    t_in_user     REAL,             -- where you moved the in-point to
    t_out_user    REAL,             -- ...and the out-point
    decided_at    TEXT,
    stage_version INTEGER DEFAULT 1,
    -- One row per (asset, in-point). Re-running selection replaces candidates
    -- rather than accumulating duplicates, but an ON CONFLICT update must never
    -- clobber a decision you already made — see replace_candidates().
    UNIQUE (content_hash, t_in)
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
CREATE INDEX IF NOT EXISTS idx_segment_asset  ON segment(content_hash, rank);
CREATE INDEX IF NOT EXISTS idx_segment_status ON segment(status);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL lets a reader run while a writer holds the file, which matters as soon
    # as ingest runs more than one file at a time; busy_timeout turns the
    # remaining write collisions into a short wait instead of an exception.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
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


def replace_candidates(conn: sqlite3.Connection, content_hash: str,
                       clips: list[dict]) -> tuple[int, int]:
    """Store a fresh set of candidates, preserving anything already decided.

    Re-running selection after a weight change is expected and should be cheap,
    but the decision log is the one thing in this system that cannot be
    regenerated — every approve and reject is a hand-made label. So undecided
    candidates are cleared and rewritten, and decided ones are left exactly
    where they are, even when the new selection disagrees about the in-point.

    Returns (written, kept).
    """
    kept = conn.execute(
        "SELECT COUNT(*) FROM segment WHERE content_hash = ? AND status != 'candidate'",
        (content_hash,)).fetchone()[0]
    conn.execute("DELETE FROM segment WHERE content_hash = ? AND status = 'candidate'",
                 (content_hash,))

    written = 0
    for c in clips:
        try:
            conn.execute(
                """INSERT INTO segment
                       (content_hash, t_in, t_out, rank, score, dominant, features)
                   VALUES (?,?,?,?,?,?,?)""",
                (content_hash, c["t_in"], c["t_out"], c.get("rank"), c.get("score"),
                 c.get("dominant"), json.dumps(c.get("features", {}))))
            written += 1
        except sqlite3.IntegrityError:
            # A decided segment already owns this in-point. Leave it alone.
            continue
    conn.commit()
    return written, kept


def segments(conn: sqlite3.Connection, content_hash: str | None = None,
             status: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM segment WHERE 1=1"
    args: list[Any] = []
    if content_hash:
        sql += " AND content_hash = ?"; args.append(content_hash)
    if status:
        sql += " AND status = ?"; args.append(status)
    return conn.execute(sql + " ORDER BY content_hash, rank, t_in", args).fetchall()


def stale_stage(conn: sqlite3.Connection, stage: str) -> list[str]:
    """Assets whose newest successful run of `stage` predates the current version.

    Bumping a stage_version makes `ingest` redo that stage — but only for files
    it actually visits, and it visits a directory, not the database. An original
    that has been archived off since it was ingested is never revisited, so it
    keeps its old derived data indefinitely while everything else moves on. That
    is a corpus quietly built from two different extractions, which is worse than
    either, so it needs to be visible rather than inferred.
    """
    rows = conn.execute(
        "SELECT content_hash FROM stage_run WHERE stage = ? AND status = 'ok' "
        "AND stage_version < ?",
        (stage, config.STAGE_VERSIONS[stage]),
    ).fetchall()
    return [r["content_hash"] for r in rows]


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
