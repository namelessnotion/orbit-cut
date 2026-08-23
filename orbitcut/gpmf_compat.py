"""Large-file patches for telemetrik's MP4 walker.

telemetrik assumes 32-bit MP4 structures throughout. GoPro writes files well
past 4 GB on exFAT cards, and those use two extensions the shipped parser does
not implement. Both are fixed here and applied at import time.

1.  **64-bit box sizes.** A box whose 32-bit size field reads 1 carries its real
    length in the following 8 bytes. The shipped walker advances by 1 byte
    instead, desynchronises, and then reads arbitrary data as the next box type
    — which surfaces as::

        UnicodeDecodeError: 'ascii' codec can't decode byte 0xe3

    several frames below the actual problem. A file over 4 GB always has a
    64-bit ``mdat``, so this fires on essentially every long ride.

2.  **64-bit chunk offsets.** Offsets past 4 GB live in ``co64`` rather than
    ``stco``, and a large file has no ``stco`` at all — so the lookup raises
    IndexError immediately after (1) is fixed.

Worth sending upstream; until then this module keeps the dependency unmodified.
"""
from __future__ import annotations

from typing import BinaryIO, List

from telemetrik import parser as _p
from telemetrik.parser import Box, Sample, _from_bytes


def get_boxes(f: BinaryIO, offset: int, size: int, box_path: List[str]) -> List[Box]:
    """Walk the MP4 box tree, honouring 64-bit and to-end-of-file sizes."""
    boxes: List[Box] = []
    end = offset + size
    f.seek(offset)

    while offset < end:
        header = f.read(8)
        if len(header) < 8:
            break

        box_size = _from_bytes(header[0:4])
        try:
            key = header[4:8].decode("ascii")
        except UnicodeDecodeError:
            # We have lost sync. Stop rather than walk further into noise —
            # a truncated result is debuggable, a garbage one is not.
            break

        header_size = 8
        if box_size == 1:
            large = f.read(8)
            if len(large) < 8:
                break
            box_size = _from_bytes(large)
            header_size = 16
        elif box_size == 0:
            box_size = end - offset          # extends to the end of its parent

        if box_size < header_size:
            break

        if key == box_path[0]:
            if len(box_path) == 1:
                boxes.append(Box(key, offset, box_size))
            else:
                boxes += get_boxes(
                    f, offset + header_size, box_size - header_size, box_path[1:]
                )

        offset += box_size
        f.seek(offset)

    return boxes


def get_samples(f: BinaryIO, stbl: Box) -> List[Sample]:
    """Parse the sample table, accepting either ``stco`` or ``co64`` offsets."""
    # --- sizes (stsz) --------------------------------------------------------
    stsz = get_boxes(f, stbl.offset, stbl.size, ["stbl", "stsz"])[0]
    f.seek(stsz.offset + 12)
    uniform_size = _from_bytes(f.read(4))
    num_entries = _from_bytes(f.read(4))
    if uniform_size != 0:
        sample_sizes = [uniform_size] * num_entries
    else:
        sample_sizes = [_from_bytes(f.read(4)) for _ in range(num_entries)]

    # --- offsets (stco 32-bit, or co64 64-bit) -------------------------------
    sample_offsets: List[int] = []
    stco = get_boxes(f, stbl.offset, stbl.size, ["stbl", "stco"])
    co64 = get_boxes(f, stbl.offset, stbl.size, ["stbl", "co64"])
    if stco:
        f.seek(stco[0].offset + 12)
        count = _from_bytes(f.read(4))
        sample_offsets = [_from_bytes(f.read(4)) for _ in range(count)]
    elif co64:
        f.seek(co64[0].offset + 12)
        count = _from_bytes(f.read(4))
        sample_offsets = [_from_bytes(f.read(8)) for _ in range(count)]
    else:
        raise ValueError("sample table has neither stco nor co64")

    # --- durations (stts) -> DTS --------------------------------------------
    sample_durations: List[int] = []
    stts = get_boxes(f, stbl.offset, stbl.size, ["stbl", "stts"])
    if stts:
        f.seek(stts[0].offset + 12)
        for _ in range(_from_bytes(f.read(4))):
            count = _from_bytes(f.read(4))
            delta = _from_bytes(f.read(4))
            sample_durations.extend([delta] * count)

    # --- composition offsets (ctts) -> PTS -----------------------------------
    composition_offsets = [0] * len(sample_sizes)
    ctts = get_boxes(f, stbl.offset, stbl.size, ["stbl", "ctts"])
    if ctts:
        f.seek(ctts[0].offset + 12)
        index = 0
        for _ in range(_from_bytes(f.read(4))):
            count = _from_bytes(f.read(4))
            value = _from_bytes(f.read(4))
            for _ in range(count):
                if index < len(composition_offsets):
                    composition_offsets[index] = value
                    index += 1

    # GoPro's gpmd track writes one sample per chunk, so chunk offsets are
    # sample offsets. stsc is not consulted — same assumption as upstream.
    n = min(len(sample_sizes), len(sample_offsets))
    samples: List[Sample] = []
    dts = 0
    for i in range(n):
        samples.append(
            Sample(sample_offsets[i], sample_sizes[i],
                   pts=dts + composition_offsets[i], dts=dts)
        )
        if i < len(sample_durations):
            dts += sample_durations[i]
    return samples


_applied = False


def apply() -> None:
    """Install the patched walkers. Idempotent."""
    global _applied
    if _applied:
        return
    _p.get_boxes = get_boxes
    _p.get_samples = get_samples
    _applied = True
