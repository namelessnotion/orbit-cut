"""Dump exactly what the GPMF track contains and what the parser makes of it.

The GPS fields have now survived two confident explanations that turned out to
be wrong, so this deliberately reports observations rather than conclusions, at
three levels:

  1. which GPS FourCCs are physically in the file's metadata track
  2. what the parser returns for them — stream keys, sample counts, arity
  3. what the extractor turns that into — actual parquet column names

A disagreement between any two of those levels localises the fault exactly.

    python tools/gps_probe.py /path/to/GX010684.MP4
"""
from __future__ import annotations

import sys

import numpy as np
from pathlib import Path

FOURCC = (b"GPS5", b"GPS9", b"GPSF", b"GPSP", b"GPSU", b"GPSA",
          b"ACCL", b"GYRO", b"GRAV", b"CORI", b"IORI")


def raw_fourccs(path: str, scan_mb: int = 64) -> dict[str, int]:
    """Count GPS-related FourCCs in the raw bytes of the metadata track.

    Crude on purpose: it makes no assumption about box structure, so it still
    answers "is GPS9 in this file at all" when the structured parser returns
    nothing. Scans a bounded slice — payloads repeat every second, so a few
    tens of MB is plenty to establish presence.
    """
    import telemetrik.parser as P

    counts = {k.decode(): 0 for k in FOURCC}
    with open(path, "rb") as f:
        size = Path(path).stat().st_size
        stbl = None
        minf = P.get_boxes(f, 0, size, ["moov", "trak", "mdia", "minf"])
        for box in minf:
            if P.get_boxes(f, box.offset, box.size, ["minf", "gmhd", "gpmd"]):
                stbl = P.get_boxes(f, box.offset, box.size, ["minf", "stbl"])[0]
                break
        if stbl is None:
            return counts

        read = 0
        for s in P.get_samples(f, stbl):
            if read >= scan_mb * 1024 * 1024:
                break
            f.seek(s.offset)
            blob = f.read(s.size)
            read += len(blob)
            for k in FOURCC:
                counts[k.decode()] += blob.count(k)
    return counts


def parser_view(path: str) -> dict:
    import telemetrik

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from orbitcut import config, gpmf_compat
    gpmf_compat.apply()

    out = {}
    # Ask for everything the config asks for, plus GPS9 explicitly in case the
    # installed config predates it.
    want = sorted(set(list(config.STREAMS) + ["GPS5", "GPS9"]))
    streams = telemetrik.extract_all_telemetry(path, streams=want)
    for key, st in sorted(streams.items()):
        first = None
        arity = None
        if st.data:
            first = st.data[0][1]
            arity = len(first) if isinstance(first, (list, tuple)) else 1
        out[key] = {"samples": st.sample_count, "arity": arity, "first": first}
    out["_requested"] = want
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    if not Path(path).is_file():
        print(f"not a file: {path}")
        return 1

    print(f"\n{Path(path).name}\n")

    print("1. FourCCs physically present in the metadata track")
    try:
        counts = raw_fourccs(path)          # one scan, not one per report line
        for k, n in counts.items():
            if n:
                print(f"     {k:6s} {n:>6} occurrences")
        absent = [k for k, n in counts.items() if not n]
        if absent:
            print(f"     absent: {', '.join(absent)}")
    except Exception as exc:
        print(f"     failed: {type(exc).__name__}: {exc}")

    print("\n2. What the parser returns")
    try:
        view = parser_view(path)
        req = view.pop("_requested")
        print(f"     requested: {', '.join(req)}")
        for key, info in view.items():
            print(f"     {key:6s} samples={info['samples']:<8} arity={info['arity']}")
            if key.startswith("GPS"):
                print(f"            first sample: {info['first']}")
        for key in ("GPS5", "GPS9"):
            if key in req and key not in view:
                print(f"     {key}: requested, returned nothing")
    except Exception as exc:
        print(f"     failed: {type(exc).__name__}: {exc}")

    print("\n3. What the in-tree GPS parser makes of it")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from orbitcut import gps
        g = gps.extract(path)
        if not g:
            print("     no GPS5 or GPS9 payloads found")
        else:
            print(f"     stream {g['key']}  samples={g['n']}  "
                  f"TYPE={g['layout']!r}")
            print(f"     divisors {g['scales']}")
            for name, col in g["columns"].items():
                v = col[np.isfinite(col)] if len(col) else col
                if len(v):
                    print(f"       {name:14s} min={v.min():>14.4f} "
                          f"median={np.median(v):>14.4f} max={v.max():>14.4f}")
    except Exception as exc:
        print(f"     failed: {type(exc).__name__}: {exc}")

    print("\n4. Columns the extractor produces")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import pandas as pd
        from orbitcut import telemetry as T
        res = T.extract(path, "gpsprobe")
        df = pd.read_parquet(res["telemetry_path"])
        gps = [c for c in df.columns if "gps" in c.lower()]
        print(f"     all columns: {', '.join(df.columns)}")
        print(f"     gps columns: {', '.join(gps) if gps else '(none)'}")
        for c in gps:
            v = df[c].dropna()
            if len(v):
                print(f"       {c:14s} n={len(v):<7} "
                      f"min={v.min():.4f} median={v.median():.4f} max={v.max():.4f}")
    except Exception as exc:
        print(f"     failed: {type(exc).__name__}: {exc}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
