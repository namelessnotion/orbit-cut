"""Planted-curve checks on clip selection.

Selection has no ground truth — which ten seconds are worth watching is taste,
and taste is what the review UI is for. But it has *structure* that can be
checked: a clip should contain its peak, leave room after it, never begin
mid-air, and never be silently dropped because it happens to sit at the end of
a ride. Those are properties, not preferences, and each one below failed at
least once.

    python tools/select_selftest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitcut import select as sel   # noqa: E402


def frame(y: np.ndarray) -> pd.DataFrame:
    """A scored frame carrying just the composite the selector reads."""
    return pd.DataFrame({
        "t": np.arange(len(y), dtype=float),
        "composite": y,
        "s_turn": y,
    })


def check(name: str, ok: bool, detail: str) -> int:
    print(f"  {name:<44}{detail:<40}{'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    fails = 0
    rng = np.random.default_rng(7)

    # 1. A long plateau. The old clamp put the peak LEAD_S before the *end* of
    #    the clip; a 30 s clip could land its best moment at second 28.5.
    y = np.full(200, 0.30)
    y[40:140] = 0.80             # a 100 s rowdy section, far longer than the cap
    y[95] = 0.99                 # with one clear best moment inside it
    t_in, t_out = sel._grow(y, 95, 200.0)
    tail = t_out - 95
    fails += check("long plateau: peak has room after it",
                   tail >= sel.MIN_TAIL_S,
                   f"clip {t_in:.1f}-{t_out:.1f}s, peak at +{95 - t_in:.1f}s, tail {tail:.1f}s")
    fails += check("long plateau: clip contains its peak",
                   t_in <= 95 <= t_out, f"{t_in:.1f} <= 95 <= {t_out:.1f}")
    # And the peak should sit near the front, not merely inside. On a flat
    # plateau every window scores the same, so this is decided entirely by the
    # tie-break — which is exactly why it is worth asserting.
    fails += check("long plateau: peak sits near the front",
                   abs((95 - t_in) - sel.LEAD_S) < 1.5,
                   f"peak at +{95 - t_in:.1f}s, wanted ~{sel.LEAD_S:.1f}s")
    fails += check("long plateau: length within the cap",
                   t_out - t_in <= sel.CLIP_MAX_S + 1e-6,
                   f"{t_out - t_in:.1f}s <= {sel.CLIP_MAX_S:.0f}s")

    # 2. A peak at the very end of the region — the case where the tail has to
    #    be taken from somewhere rather than simply granted.
    y2 = np.full(200, 0.30)
    y2[40:140] = 0.80
    y2[139] = 0.99
    t_in2, t_out2 = sel._grow(y2, 139, 200.0)
    fails += check("peak at the region's end: still gets a tail",
                   t_out2 - 139 >= sel.MIN_TAIL_S - 1e-6,
                   f"clip {t_in2:.1f}-{t_out2:.1f}s, tail {t_out2 - 139:.1f}s")

    # 3. The final second of a ride. `_peaks` skipped it entirely, so a trail
    #    that ends on its best moment offered nothing there.
    y3 = np.concatenate([rng.uniform(0.2, 0.4, 60), [0.95]])
    peaks = sel._peaks(y3, 0.5)
    fails += check("a ride ending on its best second is seen",
                   len(y3) - 1 in peaks, f"peaks found: {peaks}")

    # 4. Two freefall windows half a second apart. One pass out of the first
    #    landed the in-point inside the second.
    events = pd.DataFrame([{"t_start": 20.0, "t_end": 20.4},
                           {"t_start": 19.2, "t_end": 19.6},
                           {"t_start": 18.4, "t_end": 18.8}])
    a, b = sel._protect_air(20.2, 40.0, events)
    inside = any(s < a < e for s, e in
                 zip(events.t_start, events.t_end))
    fails += check("chained jumps: in-point ends up outside all of them",
                   not inside, f"in-point {a:.2f}s")

    # 5. End to end on a curve with three separated bursts: the selector should
    #    find them and not stack three clips on the loudest one.
    y5 = np.full(300, 0.25)
    for centre in (50, 150, 250):
        y5[centre - 6:centre + 6] = 0.85
        y5[centre] = 0.97
    got = sel.candidates(frame(y5), None, 300.0)
    centres = sorted(round((c["t_in"] + c["t_out"]) / 2) for c in got)
    fails += check("three bursts produce three clips",
                   len(got) == 3, f"{len(got)} clips, centred near {centres}")
    if got:
        worst = min(c["t_out"] - c["t_in"] for c in got)
        best = max(c["t_out"] - c["t_in"] for c in got)
        fails += check("every clip is within the length bounds",
                       worst >= sel.CLIP_MIN_S * 0.8 and best <= sel.CLIP_MAX_S,
                       f"{worst:.1f}-{best:.1f}s")

    # 6. A flat, dull ride should produce nothing rather than six confident
    #    recommendations. This one already worked; it is here because it is the
    #    property most likely to break while tuning the others.
    dull = sel.candidates(frame(np.full(300, 0.42) + rng.normal(0, 0.01, 300)),
                          None, 300.0)
    fails += check("a flat ride yields no confident clips",
                   len(dull) == 0, f"{len(dull)} clips")

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
