"""A QR encoder, in-tree, because the alternative is a dependency for one screen.

`shelf` needs to put a URL on screen in a form a phone camera can read. That is
the whole requirement: byte mode, one error-correction level, a URL's worth of
capacity. The QR spec has not moved since 2000, so this is a frozen problem —
write it once, check it against a real decoder, and never touch it again.

Verified by decoding, not by inspection: `tools/qr_selftest.py` encodes strings
of every length up to the version-10 limit, renders each to a bitmap, and reads
them back with an independent decoder. An encoder that is subtly wrong produces
a picture that looks exactly like a QR code and scans as nothing at all, so
"it looks right" is not evidence here.

Level M throughout (~15% recovery). L would hold more and this needs 60 bytes;
M is what survives a phone camera at an angle in the dark, which is the actual
operating condition.
"""
from __future__ import annotations

# (ec codewords per block, blocks in group 1, data codewords each,
#  blocks in group 2, data codewords each) for level M, versions 1-10.
# Versions beyond 10 are not here because a URL never needs them: version 10
# holds 213 bytes and `shelf`'s longest URL is about 60.
_BLOCKS: dict[int, tuple[int, int, int, int, int]] = {
    1:  (10, 1, 16, 0, 0),
    2:  (16, 1, 28, 0, 0),
    3:  (26, 1, 44, 0, 0),
    4:  (18, 2, 32, 0, 0),
    5:  (24, 2, 43, 0, 0),
    6:  (16, 4, 27, 0, 0),
    7:  (18, 4, 31, 0, 0),
    8:  (22, 2, 38, 2, 39),
    9:  (22, 3, 36, 2, 37),
    10: (26, 4, 43, 1, 44),
}

# Centres of the alignment patterns. The corners where they would collide with
# a finder pattern are skipped when the grid is built.
_ALIGN: dict[int, list[int]] = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
    6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
}

_EC_M = 0b00                 # the two format bits meaning "level M"
_PAD = (0xEC, 0x11)          # the alternating pad codewords the spec names


# --------------------------------------------------------------- GF(256)
# The field QR uses: x^8 + x^4 + x^3 + x^2 + 1.
_EXP = [0] * 512
_LOG = [0] * 256


def _build_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_build_tables()


def _mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _generator(n: int) -> list[int]:
    """The degree-n generator polynomial, (x - 2^0)...(x - 2^(n-1))."""
    g = [1]
    for i in range(n):
        g = _poly_mul(g, [1, _EXP[i]])
    return g


def _poly_mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] ^= _mul(ai, bj)
    return out


def _ec_codewords(data: list[int], n: int) -> list[int]:
    """Reed-Solomon remainder — the error correction for one block."""
    gen = _generator(n)
    rem = list(data) + [0] * n
    for i in range(len(data)):
        coef = rem[i]
        if coef:
            for j, g in enumerate(gen):
                rem[i + j] ^= _mul(g, coef)
    return rem[len(data):]


# --------------------------------------------------------------- encoding
def _capacity(version: int) -> int:
    """Bytes of payload this version holds in byte mode at level M."""
    ecw, g1, d1, g2, d2 = _BLOCKS[version]
    data_bits = (g1 * d1 + g2 * d2) * 8
    header = 4 + (8 if version < 10 else 16)
    return (data_bits - header) // 8


def _pick_version(length: int) -> int:
    for v in sorted(_BLOCKS):
        if length <= _capacity(v):
            return v
    raise ValueError(f"{length} bytes is more than a version 10 QR holds "
                     f"({_capacity(max(_BLOCKS))}); this is meant for URLs")


def _bitstream(data: bytes, version: int) -> list[int]:
    ecw, g1, d1, g2, d2 = _BLOCKS[version]
    total = (g1 * d1 + g2 * d2) * 8
    bits: list[int] = []

    def put(value: int, n: int) -> None:
        for i in range(n - 1, -1, -1):
            bits.append((value >> i) & 1)

    put(0b0100, 4)                                   # byte mode
    put(len(data), 8 if version < 10 else 16)
    for byte in data:
        put(byte, 8)
    put(0, min(4, total - len(bits)))                 # terminator, if it fits
    if len(bits) % 8:
        put(0, 8 - len(bits) % 8)
    for i in range((total - len(bits)) // 8):
        put(_PAD[i % 2], 8)
    return bits


def _codewords(data: bytes, version: int) -> list[int]:
    """Data and EC codewords, interleaved the way the spec orders them."""
    ecw, g1, d1, g2, d2 = _BLOCKS[version]
    bits = _bitstream(data, version)
    stream = [int("".join(str(b) for b in bits[i:i + 8]), 2)
              for i in range(0, len(bits), 8)]

    blocks, pos = [], 0
    for count, size in ((g1, d1), (g2, d2)):
        for _ in range(count):
            blocks.append(stream[pos:pos + size])
            pos += size
    ecs = [_ec_codewords(b, ecw) for b in blocks]

    # Interleave: first codeword of every block, then the second, and so on.
    # Short blocks simply run out, which is why this is not a transpose.
    out: list[int] = []
    for i in range(max(len(b) for b in blocks)):
        for b in blocks:
            if i < len(b):
                out.append(b[i])
    for i in range(ecw):
        for e in ecs:
            out.append(e[i])
    return out


# --------------------------------------------------------------- the matrix
def _bch(value: int, generator: int, bits: int) -> int:
    v = value << bits
    top = generator.bit_length() - 1
    while v.bit_length() - 1 >= top:
        v ^= generator << (v.bit_length() - 1 - top)
    return v


def _format_bits(mask: int) -> int:
    v = (_EC_M << 3) | mask
    return ((v << 10) | _bch(v, 0x537, 10)) ^ 0x5412


def _version_bits(version: int) -> int:
    return (version << 12) | _bch(version, 0x1F25, 12)


def _blank(size: int) -> tuple[list[list[int | None]], list[list[bool]]]:
    return ([[None] * size for _ in range(size)],
            [[False] * size for _ in range(size)])


def _place_static(m, fixed, version: int) -> None:
    size = len(m)

    def box(r0: int, c0: int, n: int, pattern) -> None:
        for r in range(n):
            for c in range(n):
                if 0 <= r0 + r < size and 0 <= c0 + c < size:
                    m[r0 + r][c0 + c] = pattern(r, c)
                    fixed[r0 + r][c0 + c] = True

    # Finders, plus the one-module separator around each.
    finder = lambda r, c: int(r in (0, 6) or c in (0, 6)
                              or (2 <= r <= 4 and 2 <= c <= 4))
    for r0, c0 in ((0, 0), (0, size - 7), (size - 7, 0)):
        box(r0, c0, 7, finder)
    for r0, c0 in ((-1, -1), (-1, size - 8), (size - 8, -1)):
        box(r0, c0, 9, lambda r, c: 0 if (r in (0, 8) or c in (0, 8)) else None)
    # The separator pass above writes None inside the finder; put it back.
    for r0, c0 in ((0, 0), (0, size - 7), (size - 7, 0)):
        box(r0, c0, 7, finder)

    # Timing.
    for i in range(8, size - 8):
        m[6][i] = m[i][6] = int(i % 2 == 0)
        fixed[6][i] = fixed[i][6] = True

    # Alignment, except where a finder already sits.
    centres = _ALIGN[version]
    for r in centres:
        for c in centres:
            if (r, c) in ((6, 6), (6, size - 7), (size - 7, 6)):
                continue
            box(r - 2, c - 2, 5,
                lambda rr, cc: int(rr in (0, 4) or cc in (0, 4)
                                   or (rr == 2 and cc == 2)))

    # The one module that is always dark, and the format areas reserved so the
    # data placement below steps over them.
    m[size - 8][8] = 1
    fixed[size - 8][8] = True
    for i in range(9):
        for r, c in ((8, i), (i, 8)):
            if m[r][c] is None:
                m[r][c] = 0
            fixed[r][c] = True
    for i in range(8):
        for r, c in ((8, size - 1 - i), (size - 1 - i, 8)):
            if m[r][c] is None:
                m[r][c] = 0
            fixed[r][c] = True

    if version >= 7:
        bits = _version_bits(version)
        for i in range(18):
            b = (bits >> i) & 1
            m[i // 3][size - 11 + i % 3] = b
            m[size - 11 + i % 3][i // 3] = b
            fixed[i // 3][size - 11 + i % 3] = True
            fixed[size - 11 + i % 3][i // 3] = True


def _place_data(m, fixed, codewords: list[int]) -> None:
    size = len(m)
    bits = [(cw >> i) & 1 for cw in codewords for i in range(7, -1, -1)]
    n, upward, col = 0, True, size - 1
    while col > 0:
        if col == 6:            # the vertical timing column is not data
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for c in (col, col - 1):
                if not fixed[row][c] and m[row][c] is None:
                    m[row][c] = bits[n] if n < len(bits) else 0
                    n += 1
        upward = not upward
        col -= 2


_MASKS = (
    lambda i, j: (i + j) % 2 == 0,
    lambda i, j: i % 2 == 0,
    lambda i, j: j % 3 == 0,
    lambda i, j: (i + j) % 3 == 0,
    lambda i, j: (i // 2 + j // 3) % 2 == 0,
    lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
    lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
    lambda i, j: ((i + j) % 2 + (i * j) % 3) % 2 == 0,
)


def _apply(m, fixed, mask: int) -> list[list[int]]:
    out = [row[:] for row in m]
    for r in range(len(m)):
        for c in range(len(m)):
            if not fixed[r][c] and _MASKS[mask](r, c):
                out[r][c] ^= 1
    size = len(m)
    bits = _format_bits(mask)
    for i in range(15):
        # MSB first: the module at (8, 0) carries bit 14, not bit 0. Reversed,
        # the four bits that happen to be asymmetric come out wrong and nothing
        # reads the symbol — the other eleven are palindromic and hide it.
        b = (bits >> (14 - i)) & 1
        if i < 6:
            out[8][i] = b
        elif i == 6:
            out[8][7] = b
        elif i == 7:
            out[8][8] = b
        elif i == 8:
            out[7][8] = b
        else:
            out[14 - i][8] = b
        # Seven bits down the left of the bottom-left finder, eight along the
        # top of the bottom-right — not eight and seven. Getting that backwards
        # also writes over the module at (size-8, 8), which the spec fixes dark
        # forever, and the result is a picture that looks like a QR code and
        # decodes as nothing.
        if i < 7:
            out[size - 1 - i][8] = b
        else:
            out[8][size - 15 + i] = b
    return out


def _penalty(m: list[list[int]]) -> int:
    size, score = len(m), 0
    lines = [row[:] for row in m] + [list(col) for col in zip(*m)]
    for line in lines:
        run, prev = 1, line[0]
        for v in line[1:]:
            if v == prev:
                run += 1
            else:
                if run >= 5:
                    score += 3 + run - 5
                run, prev = 1, v
        if run >= 5:
            score += 3 + run - 5
        text = "".join(str(v) for v in line)
        score += 40 * (text.count("10111010000") + text.count("00001011101"))
    for r in range(size - 1):
        for c in range(size - 1):
            q = m[r][c] + m[r][c + 1] + m[r + 1][c] + m[r + 1][c + 1]
            if q in (0, 4):
                score += 3
    dark = sum(sum(row) for row in m)
    score += 10 * (abs(dark * 100 // (size * size) - 50) // 5)
    return score


def encode(text: str) -> list[list[int]]:
    """A QR matrix for `text`. 1 is a dark module.

    Byte mode, level M, smallest version that fits. The mask is chosen by the
    spec's penalty score rather than fixed, because a URL full of repeated
    characters can produce a pattern a scanner reads as a finder.
    """
    data = text.encode("utf-8")
    version = _pick_version(len(data))
    size = 17 + 4 * version
    m, fixed = _blank(size)
    _place_static(m, fixed, version)
    _place_data(m, fixed, _codewords(data, version))
    best = min((_apply(m, fixed, k) for k in range(8)), key=_penalty)
    return best


def svg(matrix: list[list[int]], px: int = 4, quiet: int = 4) -> str:
    """The matrix as an SVG, sized in whole modules so nothing blurs.

    Drawn as one `<path>` rather than a rect per module: a version 5 code is
    over a thousand modules, and a thousand elements in a page that lists forty
    clips is a slow page. The quiet zone is not decoration — a scanner needs it
    to find the symbol at all.
    """
    n = len(matrix)
    side = (n + quiet * 2) * px
    parts = []
    for r, row in enumerate(matrix):
        c = 0
        while c < n:
            if row[c]:
                start = c
                while c < n and row[c]:
                    c += 1
                parts.append(f"M{(start + quiet) * px} {(r + quiet) * px}"
                             f"h{(c - start) * px}v{px}h-{(c - start) * px}z")
            else:
                c += 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" '
            f'height="{side}" viewBox="0 0 {side} {side}" '
            f'shape-rendering="crispEdges">'
            f'<rect width="{side}" height="{side}" fill="#fff"/>'
            f'<path fill="#000" d="{"".join(parts)}"/></svg>')
