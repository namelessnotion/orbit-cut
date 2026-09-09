"""Planted-answer checks on subject-aware framing.

Every check here corresponds to a bug that could ship, and most of them describe
failures that would look fine in the output. A crop path that is slightly wrong
does not throw; it produces a Reel that reads as *almost* right, which is the
hardest kind of defect to find by watching.

The solver checks plant a dog track and assert on the path, so they need no
footage and no model. The pipeline checks build their own video and drive it
through the real frame pipe with a model-free detector, the way
`orient_selftest` builds its own source rather than reading a ride.

    python tools/reframe_selftest.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitcut import detect, reframe as rf  # noqa: E402

# A shape from the library: the 5312x2988 rides, whose crop is 1680 wide and
# which therefore have 3632 px of pan to play with.
FRAME_W, FRAME_H, CROP_W = 5312, 2988, 1680
HZ = 5.0


def check(name: str, ok: bool, detail: str) -> int:
    print(f"  {name:<52}{detail:<30}{'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def track(t_end: float, u, hz: float = HZ):
    """A planted track: times at `hz`, and `u(t)` as a callable or an array."""
    t = np.arange(0.0, t_end + 1e-9, 1.0 / hz)
    return t, (np.asarray([u(x) for x in t], float) if callable(u)
               else np.asarray(u, float))


def path(t, u, t_in, t_out, **kw):
    v = np.full(np.shape(u), 0.5)
    return rf.crop_path(t, u, v, t_in, t_out, FRAME_W, FRAME_H, CROP_W, **kw)


def main() -> int:
    fails = 0
    print("\n  the solver's arithmetic")

    # The banded assembly is written out term by term rather than built from
    # dense difference matrices, because a whole ride would be a 7680-square
    # matrix to hold a pentadiagonal one. An off-by-one in it is invisible: the
    # solve still succeeds and still returns a smooth path, just not the one the
    # objective describes. This caught exactly that in the first draft.
    worst = 0.0
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 7, 12, 50):
        w = rng.uniform(0.2, 1.0, n)
        dt, lv, la = 0.1, 3.0, 12.0
        d1 = np.diff(np.eye(n), axis=0) if n >= 2 else np.zeros((0, n))
        d2 = np.diff(np.eye(n), n=2, axis=0) if n >= 3 else np.zeros((0, n))
        dense = np.diag(w * dt) + lv * (d1.T @ d1) / dt + la * (d2.T @ d2) / dt**3
        ab = rf._normal_matrix(n, w, dt, lv, la)
        got = np.zeros((n, n))
        for j in range(n):
            for i in range(max(0, j - 2), j + 1):
                got[i, j] = ab[2 + i - j, j]
        got = got + np.triu(got, 1).T
        worst = max(worst, float(np.max(np.abs(got - dense))) / max(la / dt**3, 1))
    fails += check("banded assembly matches the dense matrix", worst < 1e-12,
                   f"rel {worst:.1e}")

    print("\n  the dead zone")

    # 1. Inside the dead zone the camera is *stationary*, not merely slow. A
    # low-pass filter is always moving a little, and a viewer notices exactly
    # that. This is also what catches the tempting wrong implementation:
    # deadbanding the target once up front and doing a single quadratic solve
    # leaves a slow residual ramp here instead of a flat hold.
    d = rf.DEAD_ZONE
    amp = 0.5 * d * CROP_W / FRAME_W       # wander half a dead zone, in frame units
    t, u = track(12.0, lambda x: 0.5 + amp * np.sin(2 * np.pi * x / 3.0))
    _, x, _ = path(t, u, 2.0, 10.0)
    moved = float(np.max(x) - np.min(x)) / CROP_W
    fails += check("small wander moves the camera not at all", moved < 0.02 * d,
                   f"{moved / d:.4f} dead zones")

    # 2. ...but a move bigger than the band is followed, and all the way. An
    # earlier draft of this check asserted the path settles *a dead zone short*
    # of the target, which is simply not what a dead zone does: the band makes
    # the tracking term flat, and the smoothest path through a static tube sits
    # on the target. What the band promises is the contrast between these two
    # plants, so assert the contrast rather than either number alone.
    small = 0.5 * d * CROP_W / FRAME_W
    big = 3.0 * d * CROP_W / FRAME_W
    moved = {}
    for label, jump in (("small", small), ("big", big)):
        t, u = track(20.0, lambda x, j=jump: 0.5 + (j if x >= 8.0 else 0.0))
        g, x, _ = path(t, u, 1.0, 19.0)
        want = (0.5 + jump) * FRAME_W - CROP_W / 2
        moved[label] = (float(x.max() - x.min()) / CROP_W,
                        abs(float(x[-1]) - want) / CROP_W)
    fails += check("a step inside the band moves nothing",
                   moved["small"][0] < 0.1 * d,
                   f"{moved['small'][0] / d:.3f} dead zones")
    # Stated as "the dog ends up inside the band", not "the crop centres on
    # him". A crop that stops as soon as he is within the band is the dead zone
    # working; demanding dead centre would be demanding it not work.
    fails += check("a step outside it is followed into the band",
                   moved["big"][1] <= d, f"dog lands {moved['big'][1] / d:.2f} bands out")

    # 3. The penalties are written on derivatives, so the solve rate is a
    # discretisation choice and not a tuning knob. Written on raw differences
    # instead, changing --hz would silently reframe every clip in the library.
    a = path(t, u, 2.0, 18.0, solve_hz=10.0)[1]
    b = path(t, u, 2.0, 18.0, solve_hz=20.0)[1]
    diff = float(np.max(np.abs(a - b[::2][:len(a)]))) / CROP_W
    fails += check("solving twice as fine gives the same path", diff < 0.002,
                   f"{diff:.5f} crop widths")

    print("\n  dropouts")

    # 4. The feature's whole promise. A re-acquisition after a long gap must not
    # arrive as a jump, and the sendcmd grid is where a jump would actually be
    # visible, so measure it there rather than on the solve grid.
    t = np.arange(0.0, 20.0, 1 / HZ)
    u = np.where((t > 5.0) & (t < 9.0), np.nan, 0.5)
    u[t >= 9.0] = 0.5 + 0.8 * CROP_W / FRAME_W
    g, x, _ = path(t, u, 1.0, 19.0)
    script = rf.sendcmd(g, x, 1.0, 19.0, FRAME_W, CROP_W)
    vals = np.array([int(line.split()[-1].rstrip(";"))
                     for line in script.strip().splitlines()])
    jump = float(np.max(np.abs(np.diff(vals)))) / CROP_W
    # Per 0.02 s command step, so 0.015 crop widths is 16 px of the finished
    # Reel between one command and the next — a pan, by any reading.
    fails += check("a re-acquisition is a pan, never a jump", jump < 0.015,
                   f"max step {jump:.4f} crop widths")

    # 5. Hold, then ease. Three separate failures live here: easing on the
    # first missed frame (the camera reads as losing interest at a blink),
    # never easing (a crop frozen on nothing for the rest of the clip), and a
    # step at the moment the gap opens.
    #
    # Both of the first two are stated against the dead zone rather than
    # against zero, and that is not a softened threshold — it is the only
    # well-posed way to ask. Inside the band the tracking term is flat by
    # construction, so "still on the aim" and "back at centre" can only ever
    # mean "within the band of it". An earlier draft demanded 0.06 crop widths
    # of a 0.12 dead zone, which no correct solver could have satisfied.
    off = 0.25
    # 0.25, not more: at 0.35 the dog is past what a 1680-wide crop can reach
    # in a 5312 frame, and the check would be measuring saturation instead.
    gone = 5.0
    t = np.arange(0.0, 16.0, 1 / HZ)
    u = np.where(t <= gone, 0.5 + off, np.nan)
    g, x, _ = path(t, u, 0.0, 16.0)
    at = lambda s: float(np.interp(s, g, x))                      # noqa: E731
    centre = (FRAME_W - CROP_W) / 2
    aim = (0.5 + off) * FRAME_W - CROP_W / 2

    steady = x[(g >= 0.5) & (g <= gone - 0.5)]
    still = float(steady.max() - steady.min()) / CROP_W
    at_hold_end = abs(at(gone + rf.HOLD_S) - aim) / CROP_W
    back = abs(at(gone + rf.HOLD_S + rf.EASE_S) - centre) / CROP_W
    seg = x[(g >= gone + rf.HOLD_S) & (g <= gone + rf.HOLD_S + rf.EASE_S)]
    mono = bool(np.all(np.diff(seg) <= 1e-6) or np.all(np.diff(seg) >= -1e-6))

    fails += check("a dog who stays put moves the camera not at all",
                   still < 0.02, f"{still:.4f} crop widths over 4 s")
    # Both allow half a band beyond the band itself, because the solve is
    # offline and deliberately anticipates: the crop begins moving inside the
    # band shortly before the hold expires, where moving is free. 0.18 crop
    # widths is about 190 px of the finished 1080-wide Reel — the dog is still
    # well inside the middle of the frame, which is what "still framed" has to
    # mean here. Demanding the band exactly would be demanding no anticipation,
    # and removing that would cost the stillness everywhere else.
    fails += check("a blink is still framed when the hold expires",
                   at_hold_end <= 1.5 * rf.DEAD_ZONE,
                   f"{at_hold_end:.3f} cw off aim")
    fails += check("a long gap is back at centre on time",
                   back <= 1.5 * rf.DEAD_ZONE,
                   f"{back:.3f} cw, band is {rf.DEAD_ZONE}")
    fails += check("...and the ease never reverses", mono, "monotone")

    print("\n  the edge of the frame")

    # 6. ffmpeg silently clamps a negative crop x. There is no error anywhere —
    # the crop simply stops tracking, which looks like the detector failing.
    t, u = track(14.0, lambda _: 0.02)
    g, x, rep = path(t, u, 1.0, 13.0)
    fails += check("the crop never leaves the frame",
                   x.min() >= -1e-6 and x.max() <= FRAME_W - CROP_W + 1e-6,
                   f"x in [{x.min():.0f}, {x.max():.0f}]")
    fails += check("...and the bound is met, not clipped after the fact",
                   rep["max_violation_px"] < 0.5,
                   f"{rep['max_violation_px']:.3f} px")
    fails += check("a dog it cannot reach is reported, not hidden",
                   rep["saturated"] > 0.5, f"saturated {rep['saturated']:.2f}")

    # 7. Levelling shrinks the crop, and the pan must be bounded by the
    # *shrunken* one. Using the unshrunk width walks the crop into the black
    # wedge the rotation left behind — and only on levelled clips, which is
    # exactly how it would survive review.
    from orbitcut import render as rn
    k = rn.safe_scale(CROP_W, FRAME_H, FRAME_W, FRAME_H, 5.0)
    cw2 = int(CROP_W * k) & ~1
    g2, x2, _ = rf.crop_path(t, u, np.full(t.shape, 0.5), 1.0, 13.0,
                             FRAME_W, FRAME_H, cw2)
    fails += check("a levelled clip is bounded by the shrunken crop",
                   x2.max() <= FRAME_W - cw2 + 1e-6 and cw2 < CROP_W,
                   f"crop {cw2} of {CROP_W}")

    print("\n  the detector's contract")

    # 10. The wrong answer here is not an exception. It is a bird.
    fails += check("the dog is COCO category 18", detect.DOG_CLASS_ID == 18,
                   detect.COCO_CATEGORY_NAMES[18])
    fails += check("...and index 16 of the name list is the trap",
                   detect.COCO_CLASS_NAMES.index("dog") == 16
                   and detect.COCO_CATEGORY_NAMES[16] == "bird",
                   f"category 16 is {detect.COCO_CATEGORY_NAMES[16]!r}")

    print("\n  the priors, and following one dog")

    # 11. On a chest mount most of what a COCO dog detector finds is the rider.
    # Measured over 90 s of approved footage: 80 boxes at width 0.04 and area
    # 0.006 (Orbit), 20 at width 0.28 and area 0.063, every one of them touching
    # a left or right frame edge (a forearm). The counts have to stay separable
    # too — "saw nothing" and "saw only the rider" are different frames.
    from orbitcut import track as tr
    D = detect.DOG_CLASS_ID
    plant = np.array([
        [0.00, 0.30, 0.30, 0.52, 0.55, D],      # forearm, wide, on the edge
        [0.47, 0.35, 0.51, 0.50, 0.27, D],      # Orbit
        [0.60, 0.40, 0.606, 0.408, 0.30, D],    # a speck at the horizon
        [0.20, 0.10, 0.80, 0.60, 0.40, D],      # something filling the band
        [0.45, 0.35, 0.49, 0.50, 0.30, 1],      # a person-class box, not ours
    ], dtype=np.float32)
    kept, rejected = tr.keep(plant)
    fails += check("the priors keep the dog and drop the rider",
                   len(kept) == 1 and abs((kept[0, 0] + kept[0, 2]) / 2 - 0.49) < 1e-3,
                   f"{len(kept)} kept")
    fails += check("...and say how many they dropped", rejected == 3,
                   f"{rejected} rejected of 4 dog-class")

    def seq(rows_at, n, hz=HZ):
        return [(i / hz, np.asarray(rows_at(i / hz), np.float32).reshape(-1, 6), 0)
                for i in range(n)]

    def box(u, sc=0.6, w=0.04, h=0.15):
        return [u - w / 2, 0.4 - h / 2, u + w / 2, 0.4 + h / 2, sc, D]

    # 12. The one idea worth taking from ByteTrack. A dog blurred to 0.30 while
    # a crisp 0.85 false positive sits half a frame away: a tracker that takes
    # the best score swaps to the distractor and swings the crop across the
    # frame. It must stay on the dog, and it must not call it a new dog.
    df = tr.follow(seq(lambda t: [box(0.5, 0.30), box(0.95, 0.85, w=0.05)]
                       if 2.0 <= t < 2.6 else [box(0.5, 0.70)], 40))
    blur = df[(df.t >= 2.0) & (df.t < 2.6)]
    fails += check("a blurred dog beats a crisp distractor",
                   bool((blur["u"] < 0.6).all()) and df["track_id"].max() == 1,
                   f"u {blur['u'].min():.2f}-{blur['u'].max():.2f}, "
                   f"{df['track_id'].max()} track(s)")

    # 13. Association must be aged from the last *sighting*, not from the last
    # sampled frame. Measured from the frame interval instead — which is what
    # this did first — MAX_GAP_S can never be reached, so a track survives any
    # dropout and re-associates to whatever turns up near a stale prediction.
    # On ride 0598 that showed as one unbroken track across a five-second gap.
    gap_end = 4.0 + tr.MAX_GAP_S + 1.0
    df = tr.follow(seq(lambda t: [box(0.5)] if t < 4.0
                       else ([] if t < gap_end else [box(0.8)]), 60))
    fails += check("a gap past MAX_GAP_S is a new dog, not the same one",
                   df["track_id"].max() == 2, f"{df['track_id'].max()} tracks")
    fails += check("...while a blink is not",
                   tr.follow(seq(lambda t: [] if 2.0 <= t < 2.4 else [box(0.5)],
                                 40))["track_id"].max() == 1, "1 track")

    fails += _orientation_checks()

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


BLOB_U = 0.75           # where the blob sits in the *coded* frame
SW, SH = 640, 360


def _blob_source(path: Path, stamped: int) -> None:
    """A clip with one bright blob at a known coded-frame x, and a display matrix.

    The matrix is the thing under test, so it is allowed to be a lie — which is
    the whole point: on this library it never is, and that is exactly why the
    lying case has to be planted rather than waited for.
    """
    x = int(BLOB_U * SW) - 20
    plain = path.with_name("plain_" + path.name)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s={SW}x{SH}:d=2:r=10",
         "-vf", f"drawbox=x={x}:y=140:w=40:h=40:color=white:t=fill",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", str(stamped),
         "-i", str(plain), "-c", "copy", str(path)], check=True)
    plain.unlink()


def _orientation_checks() -> int:
    """The proxy is in container orientation; the render is in gravity's.

    This is the check the whole module is arranged around, and it cannot be
    replaced by looking at output. Across all 97 assets in this library the two
    agree — rotation 180 on all 71 chest rides, 0 on all 24 helmet rides — so a
    tracker that ignored the mapping entirely would render *correctly on every
    file that exists* and be silently wrong the first time a container lied
    again. It has lied before: `docs/architecture.md` records 77 of 97 files
    rendering upside down when the container was trusted.

    So both directions are planted. The agreeing case is what the library looks
    like today; the disagreeing case is the one with no footage to catch it.
    """
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from orient_selftest import make_telemetry

    from orbitcut import proxy as px, track as tk

    fails = 0
    print("\n  proxy orientation against gravity's")
    with tempfile.TemporaryDirectory(prefix="orbitcut_reframe_") as tmp:
        tmp = Path(tmp)
        for stamped, upside_down, want_delta, label in (
                (180, True, 0, "container 180, gravity 180 — every file here"),
                (0, True, 180, "container 0, gravity 180 — the lying container")):
            src = tmp / f"src_{stamped}.mp4"
            tel = tmp / f"tel_{stamped}.parquet"
            _blob_source(src, stamped)
            make_telemetry(tel, upside_down)

            row = {"rotation": stamped, "telemetry_path": str(tel)}
            delta, _ = tk.orientation_delta(row)
            fails += check(f"delta is {want_delta}° when {label.split(' — ')[0]}",
                           delta == want_delta, f"{delta}°")

            # The real proxy builder, not a copy of it: what makes this check
            # meaningful is that ffmpeg applies the display matrix here exactly
            # as it does during ingest.
            out = tmp / f"proxy_{stamped}"
            out.mkdir()
            cmd = px.build_command(src, out / "proxy.mp4", "none", SH)
            subprocess.run(cmd, check=True, capture_output=True)

            # ...and the real frame pipe, with a detector that needs no model:
            # whatever comes back is where the pipe actually put the blob.
            det = detect.BrightestDetector(input_size=128)
            t, subs, places = next(tk.frames(str(out / "proxy.mp4"), det.input_size,
                                             hz=2.0, delta_deg=delta,
                                             band=(0.0, 1.0), tiles=1,
                                             overlap=0.0))
            rows = det.detect(np.stack(subs))[0]
            ox, oy, sx, sy = places[0]
            u = ox + (rows[0, 0] + rows[0, 2]) / 2 * sx

            # The proxy carries `stamped`; the render will apply gravity's 180.
            # So in the render's frame the blob is at 1 - BLOB_U whenever the
            # coded frame gets flipped on the way to the render, and at BLOB_U
            # when it does not.
            want = 1.0 - BLOB_U if upside_down else BLOB_U
            fails += check("...and the blob lands where render will see it",
                           abs(u - want) < 0.05, f"u {u:.3f}, want {want:.3f}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
