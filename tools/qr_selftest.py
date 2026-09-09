"""Planted-answer checks on the QR encoder.

A wrong QR encoder produces a picture that looks exactly like a QR code and
scans as nothing, so this cannot be checked by looking at it. It was developed
against two independent implementations — matrices compared bit-for-bit with
`segno` and read back with `zxing-cpp` — and neither is a dependency of this
project, so what is kept here are the answers those runs produced.

Every check below stands for a bug that was actually in this file:

  * the format word was placed LSB-first. Eleven of its fifteen bits are
    palindromic, so only four modules moved and the symbol looked perfect. No
    decoder could read a single one.
  * eight format bits went down the column and seven along the row, which is
    backwards, and the eighth overwrote the module the spec fixes dark forever.
  * the format words for masks 4-7 were typed in from a table that was wrong.
    The BCH check below is what caught it, and it needs no table at all.

`ANCHORS` are digests of matrices confirmed against segno at the time of
writing: the 14-byte one fills version 1 exactly and was bit-identical to
segno's, the URL is a realistic payload that zxing-cpp read back correctly.

    python tools/qr_selftest.py
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbitcut import qr   # noqa: E402

ANCHORS = {
    "ORBITCUT-QR-01": (1, "93fa96eabf8749a1"),
    "http://192.168.1.20:8770/s/Ab3xQrST/g/2": (3, "1fa64d2f6a2ea832"),
}


def digest(m) -> str:
    flat = "".join("".join(map(str, row)) for row in m)
    return hashlib.sha256(flat.encode()).hexdigest()[:16]


def check(name: str, ok: bool, detail: str) -> int:
    print(f"  {name:<48}{detail:<28}{'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def svg_matrix(text: str, n: int, px: int = 4, quiet: int = 4):
    out = [[0] * n for _ in range(n)]
    d = re.search(r'd="([^"]*)"', text).group(1)
    for mv in re.finditer(r"M(\d+) (\d+)h(\d+)v", d):
        x, y, w = (int(g) for g in mv.groups())
        r, c0, span = y // px - quiet, x // px - quiet, w // px
        for c in range(c0, c0 + span):
            out[r][c] = 1
    return out


def main() -> int:
    fails = 0

    # 1. The anchors. A change here means the bytes on screen changed, which
    #    is either a fix or a regression — never something to wave through.
    for text, (want_v, want_d) in ANCHORS.items():
        m = qr.encode(text)
        v = (len(m) - 17) // 4
        fails += check(f"anchor v{want_v}: version", v == want_v, f"version {v}")
        fails += check(f"anchor v{want_v}: matrix unchanged",
                       digest(m) == want_d, digest(m))

    # 2. Format words must be valid BCH codewords. This is self-proving: strip
    #    the 0x5412 mask and the remainder against the generator must be zero.
    #    It is the check that caught a mistyped table.
    bad = [k for k in range(8)
           if qr._bch((qr._format_bits(k) ^ 0x5412) >> 10, 0x537, 10)
           != ((qr._format_bits(k) ^ 0x5412) & 0x3FF)]
    fails += check("all 8 format words are valid BCH codewords",
                   not bad, f"{8 - len(bad)}/8")

    # 3. The always-dark module. Overwriting it is invisible and fatal.
    m = qr.encode("dark module")
    fails += check("the module at (size-8, 8) is dark",
                   m[len(m) - 8][8] == 1, f"value {m[len(m) - 8][8]}")

    # 4. Structure a scanner looks for first.
    ok_finders = all(m[r][c] == int(r in (0, 6) or c in (0, 6)
                                    or (2 <= r <= 4 and 2 <= c <= 4))
                     for r in range(7) for c in range(7))
    fails += check("the top-left finder is a finder", ok_finders, "7x7")
    fails += check("timing alternates",
                   all(m[6][i] == int(i % 2 == 0) for i in range(8, len(m) - 8))
                   and all(m[i][6] == int(i % 2 == 0) for i in range(8, len(m) - 8)),
                   f"{len(m) - 16} modules")

    # 5. Capacity boundaries — one byte over must step up a version, and one
    #    byte over the largest must refuse rather than silently truncate.
    steps = []
    for v in sorted(qr._BLOCKS):
        cap = qr._capacity(v)
        at = (len(qr.encode("x" * cap)) - 17) // 4
        over = ((len(qr.encode("x" * (cap + 1))) - 17) // 4
                if v < max(qr._BLOCKS) else v + 1)
        steps.append(at == v and over == v + 1)
    fails += check("each version fills, then steps up", all(steps),
                   f"{sum(steps)}/{len(steps)}")
    try:
        qr.encode("x" * (qr._capacity(max(qr._BLOCKS)) + 1))
        refused = False
    except ValueError:
        refused = True
    fails += check("too much data is refused, not truncated", refused, "ValueError")

    # 6. The SVG is what actually reaches the browser, so it has to carry the
    #    same modules — and the quiet zone, without which nothing finds the code.
    m = qr.encode("http://192.168.1.20:8770/s/Ab3xQrST/g/2")
    text = qr.svg(m)
    fails += check("the SVG draws the matrix it was given",
                   svg_matrix(text, len(m)) == m, f"{len(m)}x{len(m)}")
    side = int(re.search(r'width="(\d+)"', text).group(1))
    fails += check("the SVG keeps a 4-module quiet zone",
                   side == (len(m) + 8) * 4, f"{side}px for {len(m)} modules")

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
