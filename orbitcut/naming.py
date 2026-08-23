"""GoPro filename parsing — chapters, and the rides they belong to.

GoPro splits long recordings into chapters and encodes that in the filename:

    GX 01 0674 .MP4
    ^^ ^^ ^^^^
    |  |  +--- recording number: the ride
    |  +------ chapter number within that recording
    +--------- encoding (GX = HEVC, GH = AVC, GP = older)

So GX010674 and GX020674 are not two clips, they are one continuous ride cut in
half by the camera. That matters more than it looks: a chapter boundary falls
wherever the file hit its size limit, which is to say at a completely arbitrary
moment — quite possibly mid-jump. Scored as separate assets, the action either
side of the seam is invisible and no clip can ever span it.

Phase 0 only records the grouping. Phase 1 is where the score series for a ride
should be stitched across chapters before any peak-finding runs.
"""
from __future__ import annotations

import re
from pathlib import Path

# GX/GH/GP + 2-digit chapter + 4-digit recording number
_CHAPTERED = re.compile(r"^(G[XHP])(\d{2})(\d{4})$", re.IGNORECASE)
# Older HERO naming: GOPR1234.MP4 is chapter 1, GP011234.MP4 is chapter 2+
_LEGACY_FIRST = re.compile(r"^GOPR(\d{4})$", re.IGNORECASE)


def parse(filename: str) -> tuple[str | None, int | None]:
    """Return ``(ride_id, chapter)``, or ``(None, None)`` if unrecognised."""
    stem = Path(filename).stem

    m = _CHAPTERED.match(stem)
    if m:
        return m.group(3), int(m.group(2))

    m = _LEGACY_FIRST.match(stem)
    if m:
        return m.group(1), 1

    return None, None
