"""orbitcut — phase 0 CLI.

    orbitcut doctor                 check the toolchain
    orbitcut verify FILE            one file, in detail — run this first
    orbitcut ingest PATH            hash, probe, telemetry, proxy
    orbitcut inventory [--csv OUT]  what you actually have
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, db, hashing, ingest, probe as probe_mod, telemetry as tel_mod


# --------------------------------------------------------------------- doctor
def cmd_doctor(_args) -> int:
    ok = True

    # Which interpreter is actually running matters more than it looks: a venv
    # built with `python3` instead of pyenv's `python` shim lands on the wrong
    # version silently, and every import below would still say ok.
    v = sys.version_info
    print(f"  {'ok':4s}  python {v.major}.{v.minor}.{v.micro}")
    print(f"        {sys.executable}")
    if (v.major, v.minor) < (3, 11):
        print("  MISS  python 3.11+ required")
        ok = False

    for binary in ("ffmpeg", "ffprobe"):
        try:
            config.which_or_die(binary)
            print(f"  ok    {binary}")
        except RuntimeError as exc:
            print(f"  MISS  {exc}")
            ok = False

    for mod, hint in (
        ("telemetrik", "pip install telemetrik"),
        ("pandas", "pip install pandas"),
        ("pyarrow", "pip install pyarrow"),
        ("numpy", "pip install numpy"),
        ("astral", "pip install astral   (optional — day/night from GPS + clock)"),
    ):
        try:
            __import__(mod)
            print(f"  ok    {mod}")
        except ImportError:
            optional = mod == "astral"
            print(f"  {'warn' if optional else 'MISS'}  {mod} — {hint}")
            ok = ok and optional

    print(f"\n  root      {config.ROOT}")
    print(f"  database  {config.DB_PATH}")
    print(f"  hwaccel   {config.HWACCEL}")
    print(f"  archive   {config.ARCHIVE or '(unset — ORBITCUT_ARCHIVE)'}")
    return 0 if ok else 1


# --------------------------------------------------------------------- verify
def cmd_verify(args) -> int:
    """Everything you want to know about one file before trusting a whole library."""
    path = Path(args.file).expanduser()
    if not path.is_file():
        print(f"not a file: {path}")
        return 1

    print(f"\n{path.name}")
    meta = probe_mod.probe(path)
    print(f"  {meta['width']}x{meta['height']} {meta['aspect']} @ {meta['fps']}fps  "
          f"{meta['vcodec']} {meta['bit_depth']}-bit  {meta['duration_s']:.0f}s  "
          f"{meta['bytes'] / 1e9:.1f} GB")
    print(f"  recorded  {meta['recorded_at']}")
    if meta.get("rotation"):
        print(f"  rotation  {meta['rotation']}° display matrix — players and ffmpeg "
              f"correct this automatically")
    else:
        print("  rotation  none — pixels are stored as shot")

    if not meta["has_gpmd"]:
        print("\n  NO GPMF TRACK — telemetry-based scoring will not work on this file.")
        return 1

    ch = hashing.content_hash(path)
    print(f"  hash      {ch}")
    print("\n  extracting telemetry...")
    tel = tel_mod.extract(path, ch)

    print(f"\n  streams   {', '.join(tel['streams'])}")
    for key, rate in sorted(tel["rates"].items()):
        print(f"    {key:5s} {rate:>8.1f} Hz")

    print("\n  sanity checks")
    accl = tel.get("accl_mag_mean")
    if accl is not None:
        # Over a real ride this sits above 9.81: vibration adds to gravity.
        # Suspiciously close to 9.81 can mean the camera barely moved.
        verdict = "ok" if 9.0 < accl < 13.0 else "SUSPECT — expected 9.8-12"
        print(f"    mean |accel|      {accl:.2f} m/s²   {verdict}")
    grav = tel.get("grav_mag_mean")
    if grav is not None:
        verdict = "ok" if 0.9 < grav < 1.1 else "SUSPECT — expected ~1.0"
        print(f"    mean |gravity|    {grav:.3f}        {verdict}")
    fix = tel.get("gps_fix_fraction")
    if fix is not None:
        verdict = "ok" if fix > 0.8 else "sparse — canopy"
        print(f"    GPS fix           {fix * 100:.0f}%          {verdict}")
        if tel.get("gps_lat"):
            print(f"    location          {tel['gps_lat']:.5f}, {tel['gps_lon']:.5f}")

    gm = tel.get("gravity_mean")
    if gm:
        dom = max(range(3), key=lambda i: abs(gm[i]))
        sign = "+" if gm[dom] > 0 else "-"
        print(f"    gravity direction ({gm[0]:+.2f},{gm[1]:+.2f},{gm[2]:+.2f})  "
              f"dominant axis grav_{dom} {sign}")

    body = tel.get("gravity_spread_deg")
    frame = tel.get("gravity_spread_image_deg")
    supp = tel.get("roll_suppression")
    if supp is not None:
        if supp > 0.5:
            state = "LEVELED in camera — do NOT level again in post"
        elif supp > 0.15:
            state = "partially leveled — inspect before trusting"
        else:
            state = "not leveled — post-leveling applies"
        print(f"    gravity in body   {body:.1f}°        how much the camera tilts")
        print(f"    gravity in frame  {frame:.1f}°        how much it tilts on screen")
        print(f"    roll suppression  {supp:+.2f}        {state}")

    if tel.get("gps_lat") and meta.get("recorded_at"):
        elev = tel_mod.sun_elevation(tel["gps_lat"], tel["gps_lon"], meta["recorded_at"])
        if elev is not None:
            print(f"    sun elevation     {elev:+.1f}°        {tel_mod.lighting_label(elev)}")

    # A missing stream is a finding, not an absence of output. GPS in
    # particular takes three sub-scores down with it, so say so loudly.
    missing = [s for s in ("ACCL", "GYRO", "GRAV", "CORI", "IORI") if s not in tel["streams"]]
    if "GPS5" not in tel["streams"] and "GPS9" not in tel["streams"]:
        import pandas as _pd
        est = tel_mod.lighting_from_exposure(_pd.read_parquet(tel["telemetry_path"]))
        print("\n  NO GPS TRACK — the camera wrote no location data at all.")
        print("    Lost:  speed sub-score, trail identification, sun-elevation day/night.")
        print("    Kept:  roughness, airtime, turn rate — none of those need GPS.")
        print("    Fix:   turn GPS on in the camera. It is off, not merely unlocked;")
        print("           an enabled-but-unlocked GPS still writes the stream.")
        if est:
            print(f"    Day/night falls back to exposure response: {est}")
    if missing:
        print(f"\n  MISSING: {', '.join(missing)} — expected on HERO8 and later.")

    print("\n  wrote")
    print(f"    {tel['telemetry_path']}")
    if tel["imu_path"]:
        print(f"    {tel['imu_path']}")
    print("\n  Check the axis mapping before writing anything axis-specific:")
    print("  accl_0/1/2 and grav_0/1/2 are positional. Magnitude-based features")
    print("  (roughness, airtime) do not care; gravity-pitch features do.\n")
    return 0


# --------------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    conn = db.connect()
    files = list(ingest.find_videos(args.path))
    if not files:
        print(f"no video files under {args.path}")
        return 1

    print(f"{len(files)} file(s)\n")
    failed = 0
    for i, path in enumerate(files, 1):
        gb = path.stat().st_size / 1e9
        print(f"[{i}/{len(files)}] {path.name}  ({gb:.1f} GB)")
        result = ingest.ingest_one(path, conn, force=args.force)
        for stage, status in result["stages"].items():
            marker = "!" if status.startswith("error") else " "
            print(f"    {marker} {stage:10s} {status}")
            if status.startswith("error"):
                failed += 1
        print()

    print(f"done — {len(files)} file(s), {failed} stage failure(s)")
    print(f"next: orbitcut inventory")
    return 1 if failed else 0


# ------------------------------------------------------------------ inventory
def cmd_inventory(args) -> int:
    conn = db.connect()
    rows = list(db.assets(conn))
    if not rows:
        print("nothing ingested yet")
        return 0

    def has_gps(r):
        streams = json.loads(r["streams"] or "[]")
        return any(k in streams for k in ("GPS5", "GPS9"))

    if args.files:
        _print_files(rows, has_gps)
    else:
        _print_rides(rows, has_gps)

    total_s = sum(r["duration_s"] or 0 for r in rows)
    total_b = sum(r["bytes"] or 0 for r in rows)
    aspects: dict[str, int] = {}
    for r in rows:
        aspects[r["aspect"] or "?"] = aspects.get(r["aspect"] or "?", 0) + 1

    rides = {r["ride_id"] or r["content_hash"] for r in rows}
    print(f"\n{len(rows)} files in {len(rides)} rides · {total_s / 3600:.1f} h · "
          f"{total_b / 1e9:.0f} GB")
    print("aspect: " + ", ".join(f"{k} x{v}" for k, v in sorted(aspects.items())))

    no_tel = [r for r in rows if not r["has_gpmd"]]
    no_gps = [r for r in rows if r["has_gpmd"] and not has_gps(r)]
    tiny = [r for r in rows if (r["duration_s"] or 0) < 30]
    if no_tel:
        print(f"\n  {len(no_tel)} file(s) with no telemetry track — cannot be scored "
              f"from sensors at all.")
    if no_gps:
        print(f"\n  {len(no_gps)} of {len(rows)} file(s) have telemetry but NO GPS.")
        print("    Those lose the speed sub-score, trail identification and")
        print("    sun-elevation day/night. Turn GPS on in the camera.")
    if tiny:
        print(f"\n  {len(tiny)} file(s) under 30 s — accidental starts, safe to ignore.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(rows[0].keys())
            for r in rows:
                w.writerow(list(r))
        print(f"\nwrote {args.csv}")
    return 0


def _print_files(rows, has_gps) -> None:
    hdr = (f"{'file':<22}{'ride':>6}{'ch':>4}{'dur':>7}{'asp':>7}{'fps':>5}"
           f"{'gps':>5}{'light':>10}{'src':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{(r['filename'] or '')[:21]:<22}"
              f"{r['ride_id'] or '-':>6}{r['chapter'] or 0:>4}"
              f"{(r['duration_s'] or 0) / 60:>6.1f}m"
              f"{r['aspect'] or '?':>7}{r['fps'] or 0:>5.0f}"
              f"{'yes' if has_gps(r) else 'NO':>5}"
              f"{r['lighting'] or '-':>10}{_light_src(r):>10}")


def _light_src(r) -> str:
    """Whether `lighting` came from solar elevation or from exposure response.

    Derived when the stored column is absent, because it follows from whether
    the file had a GPS fix — and the distinction matters: solar elevation is
    exact, while exposure cannot tell dense canopy from dusk.
    """
    if r["lighting_source"]:
        return r["lighting_source"]
    if not r["lighting"]:
        return "-"
    return "sun" if r["gps_lat"] is not None else "exposure"


def _print_rides(rows, has_gps) -> None:
    """One line per ride. Chapters are one continuous recording, not separate
    clips, so counting them as files overstates how much footage you have."""
    rides: dict[str, list] = {}
    for r in rows:
        rides.setdefault(r["ride_id"] or r["content_hash"], []).append(r)

    hdr = (f"{'ride':<8}{'ch':>4}{'dur':>8}{'res':>12}{'asp':>7}{'fps':>5}"
           f"{'bit':>5}{'gps':>5}{'light':>10}")
    print(hdr); print("-" * len(hdr))
    for ride, group in sorted(rides.items(), key=lambda kv: kv[1][0]["recorded_at"] or ""):
        group.sort(key=lambda r: r["chapter"] or 0)
        first = group[0]
        dur = sum(r["duration_s"] or 0 for r in group)
        gps = "yes" if all(has_gps(r) for r in group) else "NO"
        lights = {r["lighting"] for r in group if r["lighting"]}
        light = lights.pop() if len(lights) == 1 else ("mixed" if lights else "-")
        print(f"{ride:<8}{len(group):>4}"
              f"{dur / 60:>7.1f}m"
              f"{str(first['width']) + 'x' + str(first['height']):>12}"
              f"{first['aspect'] or '?':>7}{first['fps'] or 0:>5.0f}"
              f"{first['bit_depth'] or 0:>5}{gps:>5}{light:>10}")


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitcut", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the toolchain").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("verify", help="inspect one file in detail")
    p.add_argument("file")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("ingest", help="ingest a file or directory")
    p.add_argument("path")
    p.add_argument("--force", action="store_true", help="redo stages even if cached")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("inventory", help="what you have")
    p.add_argument("--csv", help="also write a CSV here")
    p.add_argument("--files", action="store_true",
                   help="one line per file instead of per ride")
    p.set_defaults(fn=cmd_inventory)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
