"""Look for a telemetry signature of the dog pulling, using a moment you name.

Nothing in the scorer measures pull. Speed would hint at it, but the rides where
Orbit works hardest are the older ones shot without GPS, so the only signal
available is the accelerometer — and `rough` deliberately band-passes 5-40 Hz to
isolate trail chatter, which throws away everything a body or an animal does.

The 0.5-5 Hz band nobody is using is where the interesting things live:

    ~1.2-1.7 Hz   pedalling, 70-100 rpm
    ~2.0-3.5 Hz   a dog's stride — trot toward the bottom, gallop toward the top

Those are different frequencies, which is what makes this worth testing at all:
a hard pull should show up as stride-band power rising while pedal-band power
falls, since the dog is doing the work.

That is a hypothesis, not a finding. This script does not assume it. Give it a
ride and the second something happens, and it reports what changed in every band
either side of that mark — including the possibility that nothing did.

    python tools/pull_probe.py 0603 41 --profile     # whole ride, per window
    python tools/pull_probe.py 0603 60 --after 120 --baseline 700 730

Prefer --profile when the pull starts early and continues, because the
two-window form needs a stretch of *riding without pull* to compare against and
there often is not one. Comparing a pre-ride standstill against a ride is not a
weak test, it is a meaningless one: the guard in main() refuses it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Above this ratio the two windows are not the same activity — see the check in
# main(). Calibrated against the fixtures: a much rougher trail is 27x.
POWER_RATIO_MAX = 25.0

BANDS = [
    ("0.5-1.2 Hz  sway/terrain", 0.5, 1.2),
    ("1.2-1.7 Hz  pedal cadence", 1.2, 1.7),
    ("1.7-2.2 Hz  ", 1.7, 2.2),
    ("2.2-3.0 Hz  dog trot", 2.2, 3.0),
    ("3.0-4.5 Hz  dog gallop", 3.0, 4.5),
    ("4.5-8 Hz    ", 4.5, 8.0),
    ("8-20 Hz     trail chatter", 8.0, 20.0),
    ("20-40 Hz    chatter/impact", 20.0, 40.0),
]


def components(imu: pd.DataFrame, tel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Split acceleration into along-gravity and across-gravity parts.

    Direction comes from GRAV, so this needs no assumption about which axis is
    which — the same reason `yaw_rate` projects onto gravity. Across-gravity is
    where a forward tug lives; along-gravity is mostly suspension and body.
    """
    acols = [c for c in imu.columns if c.startswith("accl_")]
    a = imu[acols].to_numpy(dtype=float)
    t = imu["t"].to_numpy(dtype=float)

    gcols = [c for c in tel.columns if c.startswith("grav_")]
    if len(gcols) == 3:
        gt = tel["t"].to_numpy(dtype=float)
        g = np.column_stack([np.interp(t, gt, tel[c].to_numpy(dtype=float))
                             for c in gcols])
        n = np.linalg.norm(g, axis=1, keepdims=True)
        g = np.divide(g, n, out=np.zeros_like(g), where=n > 1e-6)
        along = (a * g).sum(axis=1)
        across = np.linalg.norm(a - along[:, None] * g, axis=1)
    else:
        along = np.full(len(a), np.nan)
        across = np.linalg.norm(a, axis=1)
    return {"across gravity (fore-aft + lateral)": across,
            "along gravity (vertical)": along,
            "|accel| (what rough uses)": np.linalg.norm(a, axis=1)}


def band_power(x: np.ndarray, rate: float) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if len(x) < 256:
        return {}
    x = x - x.mean()
    freq = np.fft.rfftfreq(len(x), d=1.0 / rate)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    out = {}
    for name, lo, hi in BANDS:
        sel = (freq >= lo) & (freq < hi)
        out[name] = float(power[sel].sum()) if sel.any() else 0.0
    return out


def peaks(x: np.ndarray, rate: float, n: int = 6,
          lo: float = 0.5, hi: float = 8.0) -> list[tuple[float, float]]:
    """The strongest spectral peaks, and where they actually are.

    BANDS above are guesses — "dog trot", "pedal cadence" are labels imposed on
    the data before looking at it, and a peak that sits between two of them gets
    split in half and attributed to neither. This asks the signal to name its
    own frequencies instead, which is the only way to tell a real periodicity
    from a band boundary in the wrong place.
    """
    x = x[np.isfinite(x)]
    if len(x) < 1024:
        return []
    x = x - x.mean()
    freq = np.fft.rfftfreq(len(x), d=1.0 / rate)
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    sel = (freq >= lo) & (freq <= hi)
    f, p = freq[sel], power[sel]
    if not len(p):
        return []
    # Smooth lightly so one bin of noise is not a "peak", then take local maxima.
    k = max(3, int(0.05 / (f[1] - f[0])) | 1) if len(f) > 1 else 3
    sm = pd.Series(p).rolling(k, center=True, min_periods=1).mean().to_numpy()
    idx = [i for i in range(1, len(sm) - 1) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]]
    idx.sort(key=lambda i: -sm[i])
    total = sm.sum() or 1.0
    return [(float(f[i]), float(sm[i] / total)) for i in idx[:n]]


def profile(imu, tel, t, rate, step, mark, row) -> int:
    """Band shares over the whole ride.

    The two-window test assumes there is a "before" to compare against. When the
    dog pulls from early on and keeps pulling, there is not — so show the shares
    over time and let the footage say which windows were which.
    """
    sig = components(imu, tel)["across gravity (fore-aft + lateral)"]
    stride = [b for b in BANDS if "dog" in b[0]]
    pedal = [b for b in BANDS if "pedal" in b[0]]
    chatter = [b for b in BANDS if "chatter" in b[0]]

    print(f"\n{row['filename']}  band shares per {step:.0f} s "
          f"(mark at {mark:.0f} s)\n")
    print(f"  {'t':>6}{'power':>11}{'stride':>9}{'pedal':>8}{'chatter':>9}   stride share")
    print("  " + "-" * 68)
    starts = np.arange(float(t[0]), float(t[-1]) - step, step)
    rows = []
    for s0 in starts:
        sel = (t >= s0) & (t < s0 + step)
        bp = band_power(sig[sel], rate)
        if not bp:
            continue
        tot = sum(bp.values()) or 1.0
        f = lambda group: sum(bp[n] for n, _, _ in group) / tot
        rows.append((s0, tot, f(stride), f(pedal), f(chatter)))

    if not rows:
        print("  not enough samples")
        return 1
    quiet = np.median([r[1] for r in rows]) / 50      # riding vs standing still
    for s0, tot, st, pe, ch in rows:
        if tot < quiet:
            print(f"  {s0:>5.0f}s{tot:>11.2g}{'':>9}{'':>8}{'':>9}   (not riding)")
            continue
        bar = "#" * int(round(st * 40))
        flag = " <" if abs(s0 - mark) < step else ""
        print(f"  {s0:>5.0f}s{tot:>11.2g}{st:>9.1%}{pe:>8.1%}{ch:>9.1%}   {bar}{flag}")

    st_all = np.array([r[2] for r in rows if r[1] >= quiet])
    print(f"\n  stride share over the ride: min {st_all.min():.1%}  "
          f"median {np.median(st_all):.1%}  max {st_all.max():.1%}")
    print("  If the windows you remember Orbit pulling have a visibly higher")
    print("  stride share than the ones you do not, there is a feature here.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ride")
    ap.add_argument("mark", type=float, help="second the event starts")
    ap.add_argument("--baseline", "--quiet-window", nargs=2, type=float,
                    dest="baseline", metavar=("FROM", "TO"),
                    help="a stretch of riding WITHOUT the thing you are testing "
                         "for. May sit anywhere in the ride — later is fine, and "
                         "often the only option. Defaults to everything before "
                         "the mark, which is usually wrong because the start of "
                         "a ride is not riding.")
    ap.add_argument("--after", type=float, default=None,
                    help="seconds after the mark to analyse (default: to the end)")
    ap.add_argument("--profile", action="store_true",
                    help="band shares over the whole ride instead of two windows")
    ap.add_argument("--step", type=float, default=15.0,
                    help="profile window in seconds (default 15)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from orbitcut import db

    conn = db.connect()
    rows = [r for r in db.assets(conn)
            if args.ride in (r["ride_id"] or "") or args.ride in (r["filename"] or "")]
    rows = [r for r in rows if r["imu_path"] and Path(r["imu_path"]).exists()]
    if not rows:
        print(f"no ingested asset with IMU matching {args.ride!r}")
        return 1
    row = sorted(rows, key=lambda r: r["chapter"] or 0)[0]

    imu = pd.read_parquet(row["imu_path"])
    tel = pd.read_parquet(row["telemetry_path"]) if row["telemetry_path"] else pd.DataFrame()
    t = imu["t"].to_numpy(dtype=float)
    rate = len(t) / max(t[-1] - t[0], 1e-6)

    before_lo, before_hi = (args.baseline if args.baseline
                            else (float(t[0]), args.mark))
    after_lo = args.mark
    after_hi = args.mark + args.after if args.after else float(t[-1])

    if args.profile:
        return profile(imu, tel, t, rate, args.step, args.mark, row)

    pre = (t >= before_lo) & (t < before_hi)
    post = (t >= after_lo) & (t < after_hi)
    print(f"\n{row['filename']}  IMU {rate:.0f} Hz")
    print(f"  baseline {before_lo:.0f}-{before_hi:.0f}s ({pre.sum()} samples)   "
          f"event {after_lo:.0f}-{after_hi:.0f}s ({post.sum()} samples)")
    if pre.sum() < 512 or post.sum() < 512:
        print("  not enough samples either side of the mark")
        return 1

    for label, sig in components(imu, tel).items():
        a, b = band_power(sig[pre], rate), band_power(sig[post], rate)
        if not a or not b:
            continue
        print(f"\n  {label}")
        print(f"    {'band':<28}{'baseline':>12}{'event':>12}{'change':>10}")
        print("    " + "-" * 62)
        # Normalise by total power: absolute power rises whenever the trail gets
        # rougher, so the question is which bands gained *share*, not which grew.
        ta, tb = sum(a.values()) or 1.0, sum(b.values()) or 1.0
        # Shares are only meaningful between comparable windows. A large power
        # ratio means one window is not the same activity as the other — most
        # often that the "before" window is the camera standing still before the
        # ride, whose spectrum has nothing to do with the dog. On the fixtures,
        # a much rougher trail moved total power 27x; 3900x is a different
        # activity, not a harder one.
        if not 1 / POWER_RATIO_MAX < ta / tb < POWER_RATIO_MAX:
            print(f"\n  {label}")
            print(f"    NOT COMPARABLE — total power differs {max(ta, tb) / min(ta, tb):,.0f}x "
                  f"({np.sqrt(max(ta, tb) / min(ta, tb)):,.0f}x amplitude).")
            print("    One of these windows is not the same activity as the other;")
            print("    a pre-ride window of standing still does this. Pick a baseline")
            print("    where you were already riding:")
            print(f"        python tools/pull_probe.py {args.ride} {args.mark:.0f} "
                  f"--baseline <from> <to>")
            print("    or look at the whole ride instead:")
            print(f"        python tools/pull_probe.py {args.ride} {args.mark:.0f} --profile")
            continue
        for name, _, _ in BANDS:
            fa, fb = a[name] / ta, b[name] / tb
            arrow = "  " if abs(fb - fa) < 0.02 else ("UP" if fb > fa else "dn")
            print(f"    {name:<28}{fa:>11.1%}{fb:>12.1%}{arrow:>10}")
        print(f"    {'total power':<28}{ta:>11.3g}{tb:>12.3g}"
              f"{tb / ta:>9.2f}x")

    # Peaks, unlabelled. If a periodicity really belongs to the dog it should
    # appear in one window and not the other, at a frequency the data picks.
    print("\n  strongest periodicities, 0.5-8 Hz (the data's own peaks, no labels)")
    comp = components(imu, tel)
    for label in ("across gravity (fore-aft + lateral)", "along gravity (vertical)"):
        sig = comp[label]
        pa, pb = peaks(sig[pre], rate), peaks(sig[post], rate)
        print(f"\n    {label}")
        print(f"      {'baseline':<34}{'event':<34}")
        for i in range(max(len(pa), len(pb))):
            a = f"{pa[i][0]:5.2f} Hz  {pa[i][1]:5.1%}" if i < len(pa) else ""
            b = f"{pb[i][0]:5.2f} Hz  {pb[i][1]:5.1%}" if i < len(pb) else ""
            print(f"      {a:<34}{b:<34}")

    print("\n  Bands marked UP gained share in the event window vs the baseline.")
    print("  If the stride bands rose while the pedal band fell, there is a pull")
    print("  signature worth building a feature on. If everything moved together,")
    print("  the accelerometer only saw the trail get rougher and this is a dead end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
