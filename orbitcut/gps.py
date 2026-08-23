"""Parse GPS5 and GPS9 out of GPMF directly, because the upstream parser
mishandles both — in different ways, each of which looks like working code.

**GPS9** (HERO11 and later) is a GPMF *complex* type: its field layout lives in
a sibling `TYPE` element instead of being one repeated primitive. telemetrik
parses primitives only, so it returns the raw 32-byte struct and the failure
surfaces far downstream as `could not convert string to float`.

**GPS5** is worse, because it fails silently. Its `SCAL` element carries one
divisor per field — `(1e7, 1e7, 1000, 1000, 100)` — and telemetrik reads only
the first and applies it to all five. Latitude and longitude come out correct,
which is exactly why nothing looked wrong: the position was right, the map
would have plotted, and speed was 10,000x too small. A whole library
calibrated with `speed_ms` at p50 = 0.00, p90 = 0.00, p99 = 0.00.

The lesson worth keeping: a partly-correct decode is more dangerous than a
failed one. GPS9 announced itself. GPS5 produced plausible coordinates and
quietly zeroed three of the five scoring features.

Everything here is read from the file — layout from `TYPE`, divisors from
`SCAL`, timing from `STMP`. The only fixed knowledge is what the fields *mean*,
which the GPMF spec defines and a camera cannot change without changing arity.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = {
    "GPS5": ("gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d"),
    "GPS9": ("gps_lat", "gps_lon", "gps_alt", "gps_speed2d", "gps_speed3d",
             "gps_days", "gps_secs", "gps_dop", "gps_fix"),
}
# Used only when a file omits SCAL entirely, which no GoPro does.
FALLBACK_SCALE = {
    "GPS5": (1e7, 1e7, 1000.0, 1000.0, 100.0),
    "GPS9": (1e7, 1e7, 1000.0, 1000.0, 1000.0, 1.0, 1000.0, 100.0, 1.0),
}
TYPE_CHARS = {"b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
              "j": "q", "J": "Q", "f": "f", "d": "d"}
PREFERENCE = ("GPS9", "GPS5")      # GPS9 first: per-sample fix and DOP


def _fmt(layout: str) -> str | None:
    try:
        return ">" + "".join(TYPE_CHARS[c] for c in layout)
    except KeyError:
        return None


def struct_format(key: str, type_char: str | None, item_size: int,
                  layout: str | None) -> str:
    """Byte layout of one sample, from TYPE where present, else the box header."""
    if layout:
        fmt = _fmt(layout)
        if fmt and struct.calcsize(fmt) == item_size:
            return fmt
    if type_char and type_char in TYPE_CHARS:
        one = struct.calcsize(">" + TYPE_CHARS[type_char])
        if one and item_size % one == 0:
            return ">" + TYPE_CHARS[type_char] * (item_size // one)
    fallback = ">7i2h" if key == "GPS9" else ">5i"
    if struct.calcsize(fallback) == item_size:
        return fallback
    raise ValueError(
        f"{key}: {item_size}-byte samples, TYPE {layout!r}, type char "
        f"{type_char!r} — no layout fits. Refusing to guess a field order.")


def divisors_for(key: str, scales: list[int], n_fields: int) -> np.ndarray:
    """One divisor per field.

    A single SCAL value legitimately means "same scale for every field"; the bug
    being fixed here is treating a *list* of divisors as though it were that.
    """
    if len(scales) == n_fields:
        d = np.array(scales, dtype=float)
    elif len(scales) == 1:
        d = np.full(n_fields, float(scales[0]))
    else:
        d = np.array(FALLBACK_SCALE[key][:n_fields], dtype=float)
    d[d == 0] = 1.0
    return d


def decode(blob: bytes, fmt: str, divisors: np.ndarray) -> np.ndarray:
    """Packed samples -> scaled values, one row per sample."""
    item = struct.calcsize(fmt)
    rows = [struct.unpack(fmt, blob[i:i + item])
            for i in range(0, len(blob) - item + 1, item)]
    if not rows:
        return np.empty((0, len(divisors)))
    return np.array(rows, dtype=float) / divisors


def _read_scal(f, box) -> list[int]:
    f.seek(box.offset + 8)
    raw = f.read(box.struct_size * box.repeat)
    # SCAL may be stored as one wide struct or as `repeat` narrow ones.
    for size in (box.struct_size, 4, 2, 8):
        if size in (2, 4, 8) and len(raw) % size == 0 and len(raw) // size >= 1:
            code = {2: ">h", 4: ">i", 8: ">q"}[size]
            vals = [struct.unpack(code, raw[i:i + size])[0]
                    for i in range(0, len(raw), size)]
            if all(v != 0 for v in vals):
                return vals
    return []


def _payloads(path: str, key: str) -> list[dict]:
    import telemetrik.parser as P

    total = Path(path).stat().st_size
    found = []
    with open(path, "rb") as f:
        stbl = None
        for box in P.get_boxes(f, 0, total, ["moov", "trak", "mdia", "minf"]):
            if P.get_boxes(f, box.offset, box.size, ["minf", "gmhd", "gpmd"]):
                stbl = P.get_boxes(f, box.offset, box.size, ["minf", "stbl"])[0]
                break
        if stbl is None:
            return []

        for sample in P.get_samples(f, stbl):
            strm = next(
                (c for c in P.get_gpmf_boxes(f, sample.offset, sample.size,
                                             ["DEVC", "STRM"])
                 if P.get_gpmf_boxes(f, c.offset, c.size, ["STRM", key])), None)
            if strm is None:
                continue
            box = P.get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", key])[0]

            layout = None
            tb = P.get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "TYPE"])
            if tb:
                f.seek(tb[0].offset + 8)
                layout = f.read(tb[0].struct_size * tb[0].repeat).decode(
                    "ascii", "ignore").rstrip("\x00")

            scal = P.get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "SCAL"])
            scales = _read_scal(f, scal[0]) if scal else []

            stmp_us = None
            st = P.get_gpmf_boxes(f, strm.offset, strm.size, ["STRM", "STMP"])
            if st:
                f.seek(st[0].offset + 8)
                stmp_us = int.from_bytes(f.read(st[0].struct_size * st[0].repeat),
                                         "big")

            f.seek(box.offset + 8)
            found.append({
                "stmp_us": stmp_us, "item": box.struct_size, "repeat": box.repeat,
                "blob": f.read(box.struct_size * box.repeat),
                "layout": layout, "scales": scales, "type": box.type,
            })
    return found


def extract(path: str | Path, key: str | None = None) -> dict[str, Any] | None:
    """Best available GPS stream as {"key", "t", "columns", "n"}."""
    path = str(path)
    for candidate in ([key] if key else PREFERENCE):
        payloads = _payloads(path, candidate)
        if not payloads:
            continue

        layout = next((p["layout"] for p in payloads if p["layout"]), None)
        fmt = struct_format(candidate, payloads[0]["type"],
                            payloads[0]["item"], layout)
        names = FIELDS[candidate]
        arity = len(struct.unpack(fmt, b"\x00" * struct.calcsize(fmt)))
        if arity != len(names):
            raise ValueError(f"{candidate}: {arity} fields, expected {len(names)}")

        scales = next((p["scales"] for p in payloads
                       if len(p["scales"]) in (1, arity)), [])
        divisors = divisors_for(candidate, scales, arity)

        times, chunks = [], []
        for i, p in enumerate(payloads):
            start = (p["stmp_us"] / 1e6) if p["stmp_us"] is not None else float(i)
            nxt = payloads[i + 1]["stmp_us"] if i + 1 < len(payloads) else None
            if nxt is not None and p["stmp_us"] is not None:
                step = (nxt / 1e6 - start) / max(p["repeat"], 1)
            else:
                step = times[-1][1] if times else 1.0 / max(p["repeat"], 1)
            vals = decode(p["blob"], fmt, divisors)
            chunks.append(vals)
            times.extend((start + j * step, step) for j in range(len(vals)))

        values = np.vstack([c for c in chunks if len(c)]) if chunks else None
        if values is None or not len(values):
            continue
        return {
            "key": candidate,
            "t": np.array([t for t, _ in times], dtype=float),
            "columns": {n: values[:, i] for i, n in enumerate(names)},
            "n": len(values),
            "layout": layout,
            "scales": [float(d) for d in divisors],
        }
    return None
