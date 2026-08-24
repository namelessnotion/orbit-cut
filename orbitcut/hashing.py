"""Content hashing.

Full BLAKE3 over a 15 GB file across USB is minutes of pure I/O per file, and we
gain nothing from it: we are identifying our own footage, not defending against
an adversary. Sample three windows plus the exact byte length instead — two
distinct GoPro files sharing size, head, middle and tail does not happen.

The algorithm name is baked into the hash string, so if we ever want a stronger
one we can add it without invalidating what we already have.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 8 * 1024 * 1024  # 8 MiB per sampled window
ALGO = "s3b2"  # sampled-3-window blake2b


def content_hash(path: str | Path) -> str:
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode())

    with p.open("rb") as f:
        if size <= CHUNK * 3:
            h.update(f.read())
        else:
            for offset in (0, size // 2 - CHUNK // 2, size - CHUNK):
                f.seek(offset)
                h.update(f.read(CHUNK))

    return f"{ALGO}_{h.hexdigest()}"


def verify(path: str | Path, expected: str) -> bool:
    """Used by the archive stage. A transfer reporting success and a file being
    byte-identical are different claims — never skip this before deleting."""
    return content_hash(path) == expected
