"""Stage: archive. Copy inbox originals to the archive drive, verify, record.

Copy, re-hash at the destination, record `archived_path`, reclaim locally —
and never let it delete on a transfer's exit code alone (README.md). A
transfer reporting success and a file being byte-identical are different
claims, so every copy in this module goes through `_atomic_copy`: written to
a `.partial` name, verified against the expected content hash, and only then
renamed onto the real path. A crash or a bad USB read mid-copy leaves either
nothing or a `.partial` file — never a corrupt file sitting at a name that
looks like a good one.

Restoring is the same primitive run in reverse: `ensure_original` is what
`render` and `cut` call so a missing original triggers a verified copy back
from the archive instead of a hard failure pointing at `relink` (relink finds
files that moved by hand; it cannot find one that was deliberately archived
off and correctly no longer sits anywhere `relink` would scan).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from . import config, db, hashing


def _atomic_copy(src: Path, dest: Path, expected_hash: str) -> None:
    partial = dest.with_name(dest.name + ".partial")
    try:
        shutil.copy2(src, partial)
        if not hashing.verify(partial, expected_hash):
            raise RuntimeError(
                f"copy of {src.name} to {dest.parent} did not verify — "
                f"expected {expected_hash}, got {hashing.content_hash(partial)}"
            )
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def index_archive(root: Path) -> dict[str, Path]:
    """Hash every video file under `root`. Three 8 MiB reads per file, not a
    full read — the same cost `relink` pays to re-identify a library."""
    index: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in config.VIDEO_SUFFIXES:
            index[hashing.content_hash(p)] = p
    return index


def archive_one(source: Path, content_hash: str, archive_root: Path,
                 existing: dict[str, Path]) -> dict[str, Any]:
    """Get `source` onto the archive drive, copying only if it isn't there
    already. Returns fields for `db.upsert_asset`."""
    found = existing.get(content_hash)
    if found is not None:
        return {"archived_path": str(found), "archived_at": db.now()}

    dest = archive_root / source.name
    if dest.exists():
        raise RuntimeError(
            f"{dest} already exists but its hash isn't in the archive index — "
            f"refusing to overwrite a file that might be a different asset"
        )
    _atomic_copy(source, dest, content_hash)
    return {"archived_path": str(dest), "archived_at": db.now()}


def restore_one(archived_path: Path, content_hash: str, dest_dir: Path) -> Path:
    """Copy an archived original back to a local working directory, verified.
    Reuses a prior restore if one is already there and still correct."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / archived_path.name
    if dest.exists() and hashing.verify(dest, content_hash):
        return dest
    _atomic_copy(archived_path, dest, content_hash)
    return dest


def ensure_original(conn, asset_row: dict[str, Any]) -> Path:
    """Resolve the original for rendering: local if it's there, restored from
    the archive if it isn't, or a clear error naming which of those failed."""
    source_path = asset_row.get("source_path")
    if source_path and Path(source_path).exists():
        return Path(source_path)

    archived_path = asset_row.get("archived_path")
    filename = asset_row.get("filename") or asset_row["content_hash"]
    if not archived_path:
        raise RuntimeError(
            f"no original on disk for {filename}, and it has never been "
            f"archived — run `orbitcut archive` first"
        )
    if not Path(archived_path).exists():
        raise RuntimeError(
            f"the original for {filename} is archived at {archived_path}, "
            f"but that path isn't reachable right now — is the drive mounted?"
        )

    restored = restore_one(Path(archived_path), asset_row["content_hash"], config.RESTORED)
    db.upsert_asset(conn, asset_row["content_hash"], source_path=str(restored))
    return restored
