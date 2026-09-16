"""Planted-file checks on the archive stage.

`archive.py`'s whole job is to survive a bad USB transfer without lying about
it: never delete an original until the copy re-hashes correctly, and never
leave a half-written file sitting at a name that looks like a good one. That
second property doesn't show up by reading the output — a `.partial` file
that silently became the real file on a bad run would look identical to a
correct one until someone tried to render from it — so it is checked here
directly rather than trusted by inspection, the same reasoning `qr_selftest.py`
gives for checking the QR encoder's matrices instead of eyeballing them.

No real footage and no `orbitcut.db` file: the stage functions are plain
path-in/path-out, and `ensure_original`'s one piece of database interaction is
exercised against a throwaway in-memory table rather than the real catalog.

    python tools/archive_selftest.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitcut import archive as arch_mod, config, hashing   # noqa: E402


def check(name: str, ok: bool, detail: str) -> int:
    print(f"  {name:<52}{detail:<30}{'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def plant(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashing.content_hash(path)


def memory_asset(content_hash: str, **fields) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE asset (content_hash TEXT PRIMARY KEY, source_path TEXT)")
    conn.execute("INSERT INTO asset (content_hash, source_path) VALUES (?, ?)",
                 (content_hash, fields.get("source_path", "")))
    conn.commit()
    return conn


def main() -> int:
    fails = 0

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inbox, drive, restored = root / "inbox", root / "drive", root / "restored"
        inbox.mkdir(); drive.mkdir(); restored.mkdir()
        config.RESTORED = restored  # ensure_original() reads this directly

        # 1. A file not yet on the drive gets copied there and the copy verifies.
        src = inbox / "GX010001.MP4"
        ch = plant(src, b"first ride" * 1000)
        result = arch_mod.archive_one(src, ch, drive, existing={})
        dest = Path(result["archived_path"])
        fails += check("archive_one copies a new file",
                       dest.exists() and hashing.verify(dest, ch), str(dest))

        # 2. Already present elsewhere on the drive (under a different name, the
        #    way this library's actual passport drive is organised): no second
        #    copy is made at the flat destination.
        src2 = inbox / "GX010002.MP4"
        ch2 = plant(src2, b"second ride" * 1000)
        scattered = drive / "new" / "3"
        scattered.mkdir(parents=True)
        found_at = scattered / "GX010002.MP4"
        found_at.write_bytes(src2.read_bytes())
        result2 = arch_mod.archive_one(src2, ch2, drive, existing={ch2: found_at})
        flat_dest = drive / "GX010002.MP4"
        fails += check("archive_one dedups against an existing copy",
                       result2["archived_path"] == str(found_at) and not flat_dest.exists(),
                       result2["archived_path"])

        # 3. A copy that doesn't verify (simulating a bad transfer) leaves
        #    neither a corrupt file at the real name nor a stray .partial.
        src3 = inbox / "GX010003.MP4"
        plant(src3, b"third ride" * 1000)
        bad_dest = drive / "GX010003.MP4"
        raised = False
        try:
            arch_mod._atomic_copy(src3, bad_dest, "s3b2_0000000000000000")
        except RuntimeError:
            raised = True
        partial = bad_dest.with_name(bad_dest.name + ".partial")
        fails += check("a failed verify leaves no file at all",
                       raised and not bad_dest.exists() and not partial.exists(),
                       f"raised={raised}")

        # 4. Restoring copies the archived original back locally and verifies.
        restored_path = arch_mod.restore_one(dest, ch, restored)
        fails += check("restore_one copies back and verifies",
                       restored_path.exists() and hashing.verify(restored_path, ch),
                       str(restored_path))

        # 5. Restoring again, with a good copy already sitting there, is a
        #    no-op rather than a second multi-GB transfer.
        before = restored_path.stat().st_mtime_ns
        arch_mod.restore_one(dest, ch, restored)
        after = restored_path.stat().st_mtime_ns
        fails += check("restore_one skips a copy that's already correct",
                       before == after, f"mtime unchanged={before == after}")

        # 6. ensure_original: local file present -> returned untouched, no DB write.
        conn = memory_asset(ch, source_path=str(src))
        row = {"content_hash": ch, "source_path": str(src), "archived_path": str(dest),
              "filename": src.name}
        got = arch_mod.ensure_original(conn, row)
        fails += check("ensure_original is a no-op when the original is local",
                       got == src, str(got))

        # 7. ensure_original: local copy gone, archived copy reachable -> restores
        #    and records the new source_path.
        gone = inbox / "GX010004.MP4"
        ch4 = plant(gone, b"fourth ride" * 1000)
        archived4 = drive / "GX010004.MP4"
        archived4.write_bytes(gone.read_bytes())
        gone.unlink()
        conn4 = memory_asset(ch4, source_path=str(gone))
        row4 = {"content_hash": ch4, "source_path": str(gone), "archived_path": str(archived4),
               "filename": "GX010004.MP4"}
        got4 = arch_mod.ensure_original(conn4, row4)
        db_row = conn4.execute("SELECT source_path FROM asset WHERE content_hash = ?",
                               (ch4,)).fetchone()
        fails += check("ensure_original restores from the archive and updates the row",
                       got4.exists() and hashing.verify(got4, ch4)
                       and db_row["source_path"] == str(got4),
                       str(got4))

        # 8. ensure_original: neither local nor archived -> a clear error, not a
        #    silent failure or a pointer at `relink` (which can't find this).
        conn5 = memory_asset("s3b2_missing", source_path="")
        row5 = {"content_hash": "s3b2_missing", "source_path": "",
                "archived_path": None, "filename": "GX019999.MP4"}
        try:
            arch_mod.ensure_original(conn5, row5)
            never_archived_raised = False
        except RuntimeError as exc:
            never_archived_raised = "never been archived" in str(exc)
        fails += check("ensure_original explains a file that was never archived",
                       never_archived_raised, "")

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
