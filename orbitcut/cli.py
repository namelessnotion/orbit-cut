"""orbitcut — phase 0 CLI.

    orbitcut doctor                 check the toolchain
    orbitcut verify FILE            one file, in detail — run this first
    orbitcut ingest PATH            hash, probe, telemetry, proxy
    orbitcut inventory [--csv OUT]  what you actually have

    orbitcut score [ASSET]          telemetry features, per second
    orbitcut calibrate              fit the 0-1 scale to your library
    orbitcut overlay ASSET          watch a ride with its score curve
    orbitcut rank                   rank rides against each other
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import (calibrate as cal_mod, config, db, hashing, ingest, overlay as ov_mod,
               probe as probe_mod, score as score_mod, telemetry as tel_mod)


def _dependencies() -> list[tuple[str, bool, str]]:
    """(name, installed, note) for every declared dependency."""
    from importlib.metadata import PackageNotFoundError, requires, version

    out = []
    try:
        reqs = requires("orbitcut") or []
    except PackageNotFoundError:
        return [("orbitcut", False, " — not installed; run pip install -e .")]

    for raw in reqs:
        spec, _, marker = raw.partition(";")
        name = re.split(r"[<>=!\[ ]", spec.strip(), 1)[0]
        optional = "extra" in marker
        try:
            out.append((name, True, f" {version(name)}"))
        except PackageNotFoundError:
            out.append((name, False, " (optional)" if optional else ""))
    return out


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

    # Read the dependency list from package metadata rather than repeating it
    # here. A hand-maintained list drifts the moment a module gains an import —
    # which is exactly how scipy and matplotlib shipped undeclared, and why
    # doctor said "ok" right up until `score` crashed on the missing module.
    for name, ok_, note in _dependencies():
        if ok_:
            print(f"  ok    {name}{note}")
        elif note == " (optional)":
            print(f"  warn  {name} missing — optional: pip install {name}")
        else:
            print(f"  MISS  {name} — pip install -e .")
            ok = False

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
    weights = _weights(args.weights)
    if weights:
        _show_weights(weights)
        print()

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


def _weights(spec: str | None) -> dict[str, float] | None:
    """Parse `speed=0.5,turn=0.2` into an override on the default weights.

    Only the named keys change; the rest keep their defaults. Values are
    relative — `apply()` divides by the sum of whatever is present — so there
    is no need to make them add up to anything.
    """
    if not spec:
        return None
    w = dict(cal_mod.WEIGHTS)
    for part in spec.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in w:
            raise SystemExit(f"unknown weight {key!r}; expected one of "
                             f"{', '.join(cal_mod.WEIGHTS)}")
        try:
            w[key] = float(value)
        except ValueError:
            raise SystemExit(f"weight {key!r} needs a number, got {value!r}")
    return w


def _show_weights(w: dict[str, float]) -> None:
    total = sum(w.values())
    parts = ", ".join(f"{k} {v / total:.0%}" for k, v in w.items())
    print(f"  weights   {parts}")


# ---------------------------------------------------------------------- score
def _find(conn, needle: str):
    """Match an asset by hash prefix, filename fragment, or ride number."""
    rows = conn.execute("SELECT * FROM asset").fetchall()
    hits = [r for r in rows
            if r["content_hash"].startswith(needle)
            or needle.lower() in (r["filename"] or "").lower()
            or needle == (r["ride_id"] or "")]
    if not hits:
        print(f"no asset matching {needle!r}")
        return None
    if len(hits) > 1:
        print(f"{needle!r} matches {len(hits)} assets:")
        for r in hits[:10]:
            print(f"  {r['filename']}  {r['content_hash'][:20]}")
        return None
    return hits[0]


def cmd_score(args) -> int:
    conn = db.connect()
    if args.asset:
        row = _find(conn, args.asset)
        if row is None:
            return 1
        rows = [row]
    else:
        rows = [r for r in db.assets(conn) if r["telemetry_path"]]
    if not rows:
        print("nothing to score — run `orbitcut ingest` first")
        return 1

    print(f"{len(rows)} asset(s)\n")
    hdr = f"{'file':<22}{'air':>5}{'total':>8}{'longest':>9}{'rough p95':>11}{'spd p95':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        try:
            _, _ev, summ = score_mod.score_asset(r)
        except Exception as exc:
            print(f"{(r['filename'] or '')[:21]:<22}  error: {exc}")
            continue
        db.upsert_asset(conn, r["content_hash"],
                        scores_path=str(config.derived_dir(r["content_hash"]) / "scores.parquet"),
                        air_events=summ["air_events"], air_total_s=summ["air_total_s"],
                        air_longest_s=summ["air_longest_s"])
        db.record_stage(conn, r["content_hash"], "score", "ok", db.now())
        sp = summ["speed_p95"]
        print(f"{(r['filename'] or '')[:21]:<22}"
              f"{summ['air_events']:>5}"
              f"{summ['air_total_s']:>7.2f}s"
              f"{summ['air_longest_s']:>8.2f}s"
              f"{summ['rough_p95'] or 0:>11.2f}"
              f"{(sp * 3.6 if sp else 0):>8.1f}k")
    print("\nnext: orbitcut calibrate")
    return 0


def cmd_calibrate(args) -> int:
    conn = db.connect()
    paths = [r["scores_path"] for r in db.assets(conn) if r["scores_path"]]
    if not paths:
        print("nothing scored yet — run `orbitcut score`")
        return 1
    table = cal_mod.fit(paths)
    p = cal_mod.save(table)
    print(f"calibrated on {table['n_assets']} assets, "
          f"{table['n_seconds'] / 3600:.1f} h of footage\n")
    hdr = f"{'feature':<12}{'p50':>10}{'p90':>10}{'p99':>10}{'samples':>10}"
    print(hdr); print("-" * len(hdr))
    for name, f in table["features"].items():
        b = f["breaks"]
        print(f"{name:<12}{b[50]:>10.2f}{b[90]:>10.2f}{b[99]:>10.2f}{f['n']:>10}")
    print(f"\nwrote {p}")
    return 0


def cmd_overlay(args) -> int:
    import pandas as pd
    conn = db.connect()
    row = _find(conn, args.asset)
    if row is None:
        return 1
    if not row["scores_path"] or not row["proxy_path"]:
        print("that asset needs `orbitcut score` and a proxy first")
        return 1

    table = cal_mod.load()
    if table is None:
        print("no calibration yet — run `orbitcut calibrate` first")
        return 1

    weights = _weights(args.weights)
    scored = cal_mod.apply(pd.read_parquet(row["scores_path"]), table, weights)
    if weights:
        _show_weights(weights)
    ev_path = config.derived_dir(row["content_hash"]) / "air_events.parquet"
    events = pd.read_parquet(ev_path) if ev_path.exists() else None

    print(f"rendering {row['filename']}...")
    out = ov_mod.render(row["proxy_path"], scored, events,
                        row["content_hash"], row["duration_s"] or 1.0)

    comp = scored["composite_s"].to_numpy()
    top = scored.nlargest(5, "composite_s")
    print(f"\n  composite  p50 {pd.Series(comp).median():.2f}   "
          f"p95 {pd.Series(comp).quantile(.95):.2f}   max {pd.Series(comp).max():.2f}")
    print("\n  best seconds")
    for _, r in top.iterrows():
        m, sec = divmod(int(r["t"]), 60)
        drivers = sorted(
            ((r.get(f"s_{k}"), k) for k in ("speed", "turn", "rough", "descent")
             if pd.notna(r.get(f"s_{k}"))), reverse=True)[:2]
        why = ", ".join(f"{k} {v:.2f}" for v, k in drivers)
        if r.get("air_s", 0) > 0:
            why = f"AIR {r['air_s']:.2f}s, " + why
        print(f"    {m:02d}:{sec:02d}   {r['composite']:.2f}   {why}")
    print(f"\n  wrote {out}")
    print("  Watch it. If the curve peaks where you would have grabbed the")
    print("  scrubber, the weights are right. If not, try another set:")
    print("      orbitcut overlay <ride> --weights speed=0.5")
    print("  Nothing is rescored — only the composite is recomputed.")
    return 0


# ----------------------------------------------------------------------- rank
def cmd_rank(args) -> int:
    """Rank rides against each other, so cross-ride calibration can be checked.

    Within a ride the curve only has to order its own seconds. Clip selection
    pulls from the whole library, so it also needs a boring ride's peaks to sit
    below a good ride's peaks — a different claim, and one only you can confirm.
    """
    import numpy as np
    import pandas as pd

    conn = db.connect()
    table = cal_mod.load()
    if table is None:
        print("no calibration yet — run `orbitcut calibrate` first")
        return 1

    weights = _weights(args.weights)
    if weights:
        _show_weights(weights)
        print()

    rides: dict[str, list] = {}
    for r in db.assets(conn):
        if r["scores_path"] and Path(r["scores_path"]).exists():
            rides.setdefault(r["ride_id"] or r["content_hash"], []).append(r)
    if not rides:
        print("nothing scored yet — run `orbitcut score`")
        return 1

    rows = []
    for ride, group in rides.items():
        group.sort(key=lambda r: r["chapter"] or 0)
        # Chapters are one continuous recording: concatenate before ranking,
        # or a two-minute tail chapter competes as if it were a whole ride.
        frames = [cal_mod.apply(pd.read_parquet(r["scores_path"]), table, weights)
                  for r in group]
        comp = np.concatenate([f["composite"].to_numpy() for f in frames])
        comp = comp[np.isfinite(comp)]
        if not len(comp):
            continue
        best = np.sort(comp)[::-1][:30]      # about two or three clips' worth
        rows.append({
            "ride": ride,
            "chapters": len(group),
            "dur": sum(r["duration_s"] or 0 for r in group) / 60,
            "top30": float(best.mean()),
            "peak": float(comp.max()),
            "air": sum(r["air_events"] or 0 for r in group),
            "longest": max((r["air_longest_s"] or 0) for r in group),
            "light": group[0]["lighting"] or "-",
        })

    rows.sort(key=lambda r: r["top30"], reverse=True)
    if args.top:
        rows = rows[:args.top]

    hdr = (f"{'ride':<8}{'ch':>3}{'dur':>8}{'top30':>8}{'peak':>7}"
           f"{'air':>5}{'longest':>9}{'light':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['ride']:<8}{r['chapters']:>3}{r['dur']:>7.1f}m"
              f"{r['top30']:>8.2f}{r['peak']:>7.2f}"
              f"{r['air']:>5}{r['longest']:>8.2f}s{r['light']:>10}")
    print()
    print("  top30 = mean of the ride's 30 best seconds, which is roughly what")
    print("  clip selection would take from it. Sorted by that.")
    print()
    print("  Check the order against your memory of these rides. If a ride you")
    print("  remember as dull outranks one you remember as good, the calibration")
    print("  is the problem, not the per-second curve.")
    if not weights:
        print()
        print("  To try a different balance without editing anything:")
        print("      orbitcut rank --weights speed=0.5")
    return 0


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

    p = sub.add_parser("score", help="compute telemetry features")
    p.add_argument("asset", nargs="?", help="hash prefix, filename or ride number")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("calibrate", help="fit the corpus distribution")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("overlay", help="render a proxy with its score curve")
    p.add_argument("asset", help="hash prefix, filename or ride number")
    p.add_argument("--weights", help="override, e.g. speed=0.5,turn=0.2")
    p.set_defaults(fn=cmd_overlay)

    p = sub.add_parser("rank", help="rank rides against each other")
    p.add_argument("--top", type=int, help="only the best N")
    p.add_argument("--weights", help="override, e.g. speed=0.5,turn=0.2")
    p.set_defaults(fn=cmd_rank)

    p = sub.add_parser("inventory", help="what you have")
    p.add_argument("--csv", help="also write a CSV here")
    p.add_argument("--files", action="store_true",
                   help="one line per file instead of per ride")
    p.set_defaults(fn=cmd_inventory)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:
        # `orbitcut inventory | head` closes the pipe under us. Exiting quietly
        # is the correct behaviour; a traceback here just looks like a crash.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
