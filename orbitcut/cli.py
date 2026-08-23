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


# Clip length the ranking assumes, in seconds. Stage 3 targets 7-20 s biased to
# 8-15; 12 sits in the middle of that and is what "best clip" means here.
CLIP_S = 12


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

    # Which code is running, and from where. Trivial to print and it settles a
    # question that is otherwise pure guesswork: a command that reports an
    # option it does not have, or output in an old format, almost always means
    # the edits landed somewhere other than the copy on the path. An editable
    # install points into the repo; a plain one points into site-packages and
    # will happily ignore every change you make.
    import orbitcut
    pkg = Path(orbitcut.__file__).parent
    print(f"  ok    orbitcut {orbitcut.__version__}")
    print(f"        {pkg}")
    if "site-packages" in str(pkg):
        print("  warn  running from site-packages, not your repo — edits to the")
        print("        working tree will not take effect. Reinstall editable:")
        print("            pip install -e .")
    print()

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
        locked = tel.get("gps_locked_fraction")
        if locked is not None:
            print(f"    GPS lock          {locked * 100:.0f}%          "
                  f"{'ok' if locked > 0.8 else 'partial — early seconds dropped'}")
        p50, p95 = tel.get("gps_speed_p50"), tel.get("gps_speed_p95")
        if p50 is not None:
            verdict = "ok" if 0.5 < p95 < 20 else "SUSPECT — not riding speeds"
            print(f"    speed p50/p95     {p50 * 3.6:.1f} / {p95 * 3.6:.1f} km/h  {verdict}")

    # The stream is present but its fields did not survive extraction. This is
    # the exact shape of the GPS5-arity bug, so name it rather than leave three
    # sub-scores quietly empty.
    if any(k in tel["streams"] for k in ("GPS5", "GPS9")) and tel.get("gps_lat") is None:
        print("\n  GPS STREAM PRESENT BUT NO USABLE COLUMNS.")
        print("    The camera wrote location data and this pipeline did not read it,")
        print("    which costs speed, descent and cornering force on every ride.")
        print("    Re-extract:  orbitcut ingest <file> --force")
        print("    If it persists, the parser's field layout has changed — check for")
        print("    a `! GPS5: N fields` warning during ingest.")

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

    jobs = max(1, getattr(args, "jobs", 1) or 1)
    print(f"{len(files)} file(s)" + (f", {jobs} at a time" if jobs > 1 else "") + "\n")
    failed = 0

    def report(i: int, path: Path, result: dict) -> int:
        gb = path.stat().st_size / 1e9
        print(f"[{i}/{len(files)}] {path.name}  ({gb:.1f} GB)")
        bad = 0
        for stage, status in result["stages"].items():
            marker = "!" if status.startswith("error") else " "
            print(f"    {marker} {stage:10s} {status}")
            bad += status.startswith("error")
        print()
        return bad

    if jobs == 1:
        for i, path in enumerate(files, 1):
            failed += report(i, path, ingest.ingest_one(path, conn, force=args.force))
    else:
        # Threads, not processes: the expensive stage is an ffmpeg subprocess,
        # which holds no GIL, and threads can share the database file through
        # WAL without the marshalling a process pool would need. The honest
        # caveat is that telemetry parsing is pure Python and *does* hold the
        # GIL, so if `orbitcut timing` says telemetry dominates, expect less
        # from this than the job count suggests.
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        local = threading.local()

        def run(i: int, path: Path):
            if not hasattr(local, "conn"):
                local.conn = db.connect()
            return i, path, ingest.ingest_one(path, local.conn, force=args.force)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run, i, p) for i, p in enumerate(files, 1)]
            for fut in as_completed(futures):
                i, path, result = fut.result()
                failed += report(i, path, result)

    print(f"done — {len(files)} file(s), {failed} stage failure(s)")
    print(f"next: orbitcut inventory")
    return 1 if failed else 0


def cmd_timing(args) -> int:
    """Where ingest actually spends its time.

    Worth having as a command rather than a guess: "would the GPU help" is
    unanswerable in the abstract and trivial to answer from the record, because
    every stage already stores when it started and finished. A stage that is 5%
    of wall time cannot be made 5% faster no matter what you accelerate.
    """
    from datetime import datetime

    conn = db.connect()
    rows = conn.execute(
        "SELECT stage, started_at, finished_at FROM stage_run WHERE status = 'ok'"
    ).fetchall()
    if not rows:
        print("no completed stages recorded yet")
        return 1

    def seconds(a: str | None, b: str | None) -> float | None:
        if not a or not b:
            return None
        try:
            return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
        except ValueError:
            return None

    per: dict[str, list[float]] = {}
    for r in rows:
        s = seconds(r["started_at"], r["finished_at"])
        if s is not None and s >= 0:
            per.setdefault(r["stage"], []).append(s)
    if not per:
        print("stages recorded but without usable timestamps")
        return 1

    total = sum(sum(v) for v in per.values())
    order = sorted(per.items(), key=lambda kv: -sum(kv[1]))

    # Stage timestamps are stored to the second, so anything faster than that
    # reads as zero. Fine when proxies take minutes; worth saying rather than
    # dividing by it.
    if total <= 0:
        print("every stage completed inside the 1 s timestamp resolution — "
              "nothing here is slow enough to be worth optimising")
        return 0

    hdr = f"{'stage':<12}{'runs':>6}{'total':>10}{'per file':>10}{'share':>8}"
    print(hdr); print("-" * len(hdr))
    for stage, vals in order:
        tot = sum(vals)
        print(f"{stage:<12}{len(vals):>6}{tot / 60:>9.1f}m"
              f"{tot / len(vals):>9.1f}s{tot / total:>8.0%}")
    print("-" * len(hdr))
    print(f"{'all':<12}{sum(len(v) for v in per.values()):>6}{total / 60:>9.1f}m")

    # The interpretation is the point of the command, so make it rather than
    # leaving a table to be read.
    top, vals = order[0]
    share = sum(vals) / total
    print()
    if top == "proxy":
        print(f"  Proxy is {share:.0%} of ingest — it is the decode/encode stage, so")
        print("  it is the only one an encoder or a GPU can touch. Check whether it")
        print("  is already on hardware: `orbitcut doctor` reports the backend.")
        print("  If it is, the remaining lever is running files concurrently, not")
        print("  a faster encoder — see `orbitcut ingest --jobs`.")
    else:
        print(f"  {top} is {share:.0%} of ingest, and it does not decode video.")
        print("  Faster video encoding cannot move it. Concurrency can:")
        print("      orbitcut ingest <dir> --jobs 4")
    return 0


def _proxy_minutes(conn) -> float:
    """Minutes already spent making proxies, for scaling a projection against."""
    from datetime import datetime

    total = 0.0
    for r in conn.execute("SELECT started_at, finished_at FROM stage_run "
                          "WHERE stage = 'proxy' AND status = 'ok'").fetchall():
        try:
            total += (datetime.fromisoformat(r["finished_at"])
                      - datetime.fromisoformat(r["started_at"])).total_seconds()
        except (TypeError, ValueError):
            continue
    return total / 60.0


def cmd_bench(args) -> int:
    """Which of read / decode / encode is the ceiling, measured not guessed."""
    from . import bench as bench_mod

    conn = db.connect()
    path = args.asset
    if not Path(path).exists():
        row = _find(conn, args.asset)
        if row is None:
            return 1
        path = row["source_path"]
        if not path or not Path(path).exists():
            print(f"the original for {row['filename']} is not at {path}")
            print("bench needs an original, not a proxy — point it at a file directly.")
            return 1

    print(f"benchmarking {Path(path).name}")
    res = bench_mod.run(path, height=args.height)
    print(f"  {res['size_gb']:.1f} GB, {res['width']}x{res['height']}, "
          f"{res['duration_s'] / 60:.1f} min — measuring a "
          f"{res['window_s']:.0f} s window from {int(bench_mod.START_FRACTION * 100)}% in\n")

    read = res["read_mbps"]
    if read:
        # The number that matters is not MB/s, it is what MB/s implies for a
        # whole file: a stage cannot go faster than the bytes arrive.
        secs = res["size_gb"] * 1000 / read
        ceiling = res["duration_s"] / secs if secs else 0
        span = f"{secs:.0f} s" if secs < 90 else f"{secs / 60:.1f} min"
        print(f"  disk read        {read:>8.0f} MB/s   "
              f"→ {span} per file, a hard ceiling of {ceiling:.0f}x realtime")
        if read > 2000:
            # Nothing spinning or USB-attached reaches this. A repeat run on the
            # same file is the usual cause, and it measures RAM, not the disk.
            print("                            (cached — run on a file you have not "
                  "just read for a true figure)")
    dx = res["decode_x"]
    if dx:
        how = "" if res.get("decode_hw") == config.HWACCEL else f" ({res['decode_hw']})"
        print(f"  read+decode+scale{dx:>8.1f}x{how:<9} everything except the encoder")
        if res.get("decode_note"):
            print(f"                            {res['decode_note'][:70]}")
    elif res.get("decode_error"):
        print(f"  read+decode+scale  failed   {res['decode_error']}")

    print(f"\n  {'backend':<20}{'speed':>9}   note")
    print("  " + "-" * 52)
    best, best_x = None, 0.0
    current_x = None
    for hw, r in res["backends"].items():
        x = r["x_realtime"]
        tag = "current" if hw == config.HWACCEL else ""
        if x is None:
            print(f"  {hw:<20}{'--':>9}   unavailable: {r['error'][:40]}")
            continue
        if hw == config.HWACCEL:
            current_x = x
        if x > best_x:
            best, best_x = hw, x
        print(f"  {hw:<20}{x:>8.1f}x   {tag}")

    io_ceiling = None
    if read:
        secs = res["size_gb"] * 1000 / read
        io_ceiling = (res["duration_s"] / secs) if secs else None

    # One verdict, not several. Reading, decoding and encoding are in series, so
    # exactly one of them is binding — printing a paragraph about each produces
    # advice that contradicts itself.
    # A stage cannot be slower than a pipeline that contains it. When that shows
    # up, the probe is measuring itself and must not be allowed to reach a
    # verdict — the first version of this command concluded "decoding is the
    # ceiling" from exactly this impossibility.
    if dx and current_x and dx < current_x * 0.98:
        print(f"\n  The read+decode+scale figure ({dx:.1f}x) came out below the full "
              f"pipeline ({current_x:.1f}x),")
        print("  which cannot be true — it is a subset of that work. Treat it as a bad")
        print("  measurement, not a finding, and read only the table below.")
        dx = None

    print()
    if best and current_x and best != config.HWACCEL and best_x > current_x * 1.15:
        print(f"  {best} is {best_x / current_x:.1f}x faster than the backend you are using.")
        print(f"      export ORBITCUT_HWACCEL={best}")
        print("  Then re-ingest with --force to rebuild the proxies, or leave the")
        print("  existing ones alone and let it apply to everything from here.")
    elif io_ceiling and best_x and io_ceiling < best_x * 1.2:
        print(f"  Reading the file caps you at {io_ceiling:.1f}x and the pipeline runs at")
        print(f"  {best_x:.1f}x, so the disk is the constraint. No encoder setting can")
        print("  help; faster storage can.")
    elif dx and best_x and dx < best_x * 1.25:
        print(f"  Everything up to the encoder is {dx:.1f}x and the whole pipeline is "
              f"{best_x:.1f}x,")
        print("  so the encoder is nearly free and the cost is reading, decoding and")
        print("  scaling. No encoder setting will move that. Concurrency is the lever")
        print("  left, and the table below says whether it works.")
    elif dx and best_x and dx > best_x * 1.5:
        print(f"  Everything up to the encoder runs at {dx:.1f}x and the full pass at "
              f"{best_x:.1f}x,")
        print("  so the encoder is the cost — reading, decoding and scaling are not.")
    else:
        print("  The backend you are on is already the fastest available here.")

    conc = res.get("concurrency") or {}
    ok = {n: v for n, v in conc.items() if v}
    if len(ok) > 1:
        one = ok.get(1)
        print(f"\n  {'concurrent':<20}{'total':>9}   {res.get('concurrency_backend', '')}")
        print("  " + "-" * 52)
        for n, v in sorted(ok.items()):
            gain = f"{v / one:.1f}x" if one else ""
            print(f"  {n} at once{'':<11}{v:>8.1f}x   {gain}")
        top = max(ok, key=lambda n: ok[n])
        if one and ok[top] > one * 1.3:
            print(f"\n  Running {top} at once is {ok[top] / one:.1f}x the total throughput —")
            print("  the engine is not saturated by one stream. Use it:")
            print(f"      orbitcut ingest <dir> --jobs {top}")
            spent = _proxy_minutes(conn)
            if spent:
                print(f"  Your recorded proxy time is {spent:.0f} min; at this ratio "
                      f"that is about {spent * one / ok[top]:.0f}.")
        elif one:
            print("\n  Concurrency buys nothing here — one stream already saturates the")
            print("  pipeline, so --jobs would only interleave the same total work.")
    return 0


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

    stale = set(db.stale_stage(conn, "telemetry"))
    if stale:
        names = [r["filename"] or r["content_hash"][:12] for r in db.assets(conn)
                 if r["content_hash"] in stale]
        print(f"  {len(names)} asset(s) still carry telemetry from an older extraction:")
        print(f"    {', '.join(names[:6])}{' …' if len(names) > 6 else ''}")
        print("  Scoring them mixes two extractions in one corpus. Re-ingest the")
        print("  directory holding those originals, or pass each file to `ingest`.\n")

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
    weights = _weights(args.weights) or dict(cal_mod.WEIGHTS)
    table = cal_mod.fit(paths, weights, getattr(args, "sharpness", None))
    p = cal_mod.save(table)
    print(f"calibrated on {table['n_assets']} assets, "
          f"{table['n_seconds'] / 3600:.1f} h of footage")
    _show_weights(weights)
    print()
    hdr = f"{'feature':<12}{'p50':>10}{'p90':>10}{'p99':>10}{'samples':>10}"
    print(hdr); print("-" * len(hdr))
    for name, f in table["features"].items():
        b = f["breaks"]
        print(f"{name:<12}{b[50]:>10.2f}{b[90]:>10.2f}{b[99]:>10.2f}{f['n']:>10}")

    missing = table.get("missing") or {}
    if missing:
        print("\nnot usable:")
        for name, why in missing.items():
            print(f"  {name:<12}{why}")
        if any(m.startswith("gps") or m in ("speed_ms", "grade", "lat_accel")
               for m in missing):
            print("\n  Missing GPS features usually mean extraction, not the ride.\n"
                  "  `orbitcut inventory` reads the stream list, so it can say GPS\n"
                  "  is present while the columns never reached the scorer. Check\n"
                  "  with:  orbitcut verify <one file that should have GPS>")

    # Availability buckets. Rides carrying different features are ranked
    # separately, so it is worth seeing how the library splits — a bucket that
    # is one short ride is not a distribution, and falls back to global.
    buckets = {k: v for k, v in table.get("levels", {}).items() if k != "global"}
    if buckets:
        print(f"\n{'bucket':<30}{'hours':>8}{'share':>8}")
        print("-" * 46)
        total = sum(v["n"] for v in buckets.values())
        for _k, v in sorted(buckets.items(), key=lambda kv: -kv[1]["n"]):
            print(f"{v['features']:<30}{v['n'] / 3600:>8.1f}{v['n'] / total:>8.0%}")
        if len(buckets) > 1:
            print("\n  Rides are compared within their own bucket, so the ones\n"
                  "  without GPS no longer float to the top for lack of features.")
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
    scored = cal_mod.apply(pd.read_parquet(row["scores_path"]), table, weights,
                           getattr(args, "sharpness", None))
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
        if not cal_mod.weights_match(table, weights):
            print("  note      not the weights this calibration was fitted with, so")
            print("            cross-ride comparison is approximate. If you settle on")
            print("            them: orbitcut calibrate --weights ...")
        print()
    if "levels" not in table:
        print("  This calibration predates availability buckets, so rides without\n"
              "  GPS will rank too highly. Re-run `orbitcut calibrate`.\n")

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
        frames = [cal_mod.apply(pd.read_parquet(r["scores_path"]), table, weights,
                                getattr(args, "sharpness", None))
                  for r in group]
        comp = np.concatenate([f["composite"].to_numpy() for f in frames])
        comp = comp[np.isfinite(comp)]
        if not len(comp):
            continue
        # A clip is contiguous. `top30` took the 30 best seconds from anywhere
        # in the ride, which measures how much good footage a ride contains in
        # total — a different question from whether it contains one great clip,
        # and one that favours long even rides. Best contiguous window answers
        # the question clip selection will actually ask.
        best = np.sort(comp)[::-1][:30]
        roll = pd.Series(comp).rolling(CLIP_S).mean().to_numpy()
        clip = float(np.nanmax(roll)) if np.isfinite(roll).any() else float("nan")
        # Which sub-scores this ride actually had — the bucket it was ranked in.
        have = [k for k in cal_mod.FEATURES
                if any(np.isfinite(f[cal_mod.SUB[k]]).any() for f in frames)]
        rows.append({
            "ride": ride,
            "chapters": len(group),
            "dur": sum(r["duration_s"] or 0 for r in group) / 60,
            "feat": "".join(k[0] for k in cal_mod.FEATURES if k in have),
            "clip": clip,
            "top30": float(best.mean()),
            "peak": float(comp.max()),
            "air": sum(r["air_events"] or 0 for r in group),
            "longest": max((r["air_longest_s"] or 0) for r in group),
            "light": group[0]["lighting"] or "-",
        })

    key = "top30" if getattr(args, "by", "clip") == "top30" else "clip"
    rows.sort(key=lambda r: (r[key] if np.isfinite(r[key]) else -1), reverse=True)
    if args.top:
        rows = rows[:args.top]

    hdr = (f"{'ride':<8}{'ch':>3}{'dur':>8}{'feat':>6}{'clip':>7}{'top30':>8}{'peak':>7}"
           f"{'air':>5}{'longest':>9}{'light':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['ride']:<8}{r['chapters']:>3}{r['dur']:>7.1f}m{r['feat']:>6}"
              f"{r['clip']:>7.2f}{r['top30']:>8.2f}{r['peak']:>7.2f}"
              f"{r['air']:>5}{r['longest']:>8.2f}s{r['light']:>10}")
    print()
    print(f"  clip  = best contiguous {CLIP_S} s — does this ride contain one great")
    print("          clip? Sorted by this.")
    print("  top30 = the 30 best seconds from anywhere in the ride — how much good")
    print("          footage it holds in total. A long even ride wins on this and")
    print("          loses on clip; a ride with one real sprint does the reverse.")
    print("          Compare the two columns where they disagree.")
    print()
    print("  feat  = which sub-scores the ride had (s]peed t]urn r]ough d]escent).")
    print("  Rides are ranked against others carrying the same ones, so a ride")
    print("  without GPS is not flattered by having fewer numbers to average.")
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
    p.add_argument("--jobs", type=int, default=1,
                   help="files to process at once (default 1; try 3-4)")
    p.set_defaults(fn=cmd_ingest)

    sub.add_parser("timing", help="where ingest spends its time").set_defaults(fn=cmd_timing)

    p = sub.add_parser("bench", help="what limits proxy speed: disk, decode or encode")
    p.add_argument("asset", help="a file path, or a hash prefix / filename / ride number")
    p.add_argument("--height", type=int, help=f"proxy height (default {config.PROXY_HEIGHT})")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("score", help="compute telemetry features")
    p.add_argument("asset", nargs="?", help="hash prefix, filename or ride number")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("calibrate", help="fit the corpus distribution")
    p.add_argument("--weights", help="bake these in, e.g. speed=0.5,turn=0.2")
    p.add_argument("--sharpness", type=float,
                   help=f"peak emphasis; 1 = average all features, higher favours "
                        f"a standout moment (default {cal_mod.SHARPNESS:g})")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("overlay", help="render a proxy with its score curve")
    p.add_argument("asset", help="hash prefix, filename or ride number")
    p.add_argument("--weights", help="override, e.g. speed=0.5,turn=0.2")
    p.add_argument("--sharpness", type=float, help="peak emphasis; see calibrate")
    p.set_defaults(fn=cmd_overlay)

    p = sub.add_parser("rank", help="rank rides against each other")
    p.add_argument("--top", type=int, help="only the best N")
    p.add_argument("--by", choices=("clip", "top30"), default="clip",
                   help="sort by best contiguous clip (default) or total good footage")
    p.add_argument("--weights", help="override, e.g. speed=0.5,turn=0.2")
    p.add_argument("--sharpness", type=float, help="peak emphasis; see calibrate")
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
