"""Build the OrbitCut project icon from a photo of Orbit.

Hand-authoring a specific dog in bezier curves went badly — three attempts
produced a generic fox. So the shape comes from the photograph instead:

  1. GrabCut lifts him off the snow.
  2. k-means in Lab space reduces his coat to its three real colours.
  3. the label map is smoothed hard, so the result reads as flat graphic art
     rather than a posterised photograph.
  4. potrace turns each colour layer into actual vector paths.

The palette is sampled from the same photo, then lifted a little: photo medians
sit in shadow, and flat colour needs more separation than a photograph does.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageOps

OUT = Path(__file__).parent
# The source portrait. Repoint this at your own copy — head-on, in even light,
# against a background that contrasts with his coat. GrabCut does the rest.
PHOTO = os.environ.get("ORBIT_PHOTO", "orbit-portrait.jpg")

PALETTE = {
    "ink":    "#1B1815",
    "tan":    "#C0844A",
    "white":  "#F4EFE3",
    # He is tricolour, so the ground has to separate from BOTH ends of him: a
    # dark ground loses his coat, a bone ground loses his blaze and chest. A
    # cool mid-tone is the only thing that holds all three.
    "ground": "#3E4A46",
    "ground_slate": "#48534F",
    "ground_moss": "#4A5C3E",
    "ground_dark": "#11100E",
    "eye":    "#0F0D0B",
}
# Head and a little chest. Tighter than instinct suggests: at 16px the ears are
# the whole logo, and body only steals pixels from them.
CROP = (0.325, 0.088, 0.578, 0.600)
WORK = 900
# Eye centres as a fraction of the working crop, read off the photo.
EYES = ((0.325, 0.415), (0.640, 0.405))


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def segment() -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR image, binary mask) for the subject."""
    im = ImageOps.exif_transpose(Image.open(PHOTO)).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = CROP
    crop = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    crop.thumbnail((WORK, WORK))
    a = np.array(crop)[:, :, ::-1].copy()
    h, w = a.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    rect = (int(.06 * w), int(.005 * h), int(.88 * w), int(.99 * h))
    cv2.grabCut(a, mask, rect, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), 7, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = np.where(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=2)
    return a, m


def posterise(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Label every subject pixel 0=ink 1=tan 2=white, then smooth hard."""
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    pts = lab[m > 0].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centres = cv2.kmeans(pts, 3, None, crit, 8, cv2.KMEANS_PP_CENTERS)

    # Order clusters by lightness so the mapping never depends on k-means luck.
    order = np.argsort(centres[:, 0])
    remap = np.zeros(3, np.uint8)
    for rank, cluster in enumerate(order):
        remap[cluster] = rank

    out = np.full(m.shape, 255, np.uint8)
    out[m > 0] = remap[labels.flatten()]

    # Smoothing is what separates graphic art from a posterised snapshot.
    for _ in range(3):
        out = cv2.medianBlur(out, 21)
    for lvl in (0, 1, 2):
        layer = (out == lvl).astype(np.uint8) * 255
        layer = cv2.morphologyEx(layer, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        out[layer > 0] = lvl
    out[m == 0] = 255
    return out


def trace_layer(binary: np.ndarray, colour: str) -> str:
    """potrace one binary layer into an SVG path group."""
    pbm = OUT / "_layer.pbm"
    Image.fromarray(np.where(binary > 0, 0, 255).astype(np.uint8)).save(pbm)
    svg = subprocess.run(
        ["potrace", str(pbm), "-s", "-o", "-", "--flat",
         "--turdsize", "300", "--alphamax", "1.0", "--opttolerance", "0.6"],
        capture_output=True, check=True,
    ).stdout.decode()
    head = svg.split("<g", 1)[1].split(">", 1)[0]
    inner = svg.split("<g", 1)[1].split(">", 1)[1].rsplit("</g>", 1)[0]
    parts = [t for t in head.split('"') if "translate" in t or "scale" in t]
    return f'<g transform="{parts[0] if parts else ""}" fill="{colour}">{inner}</g>'


def build_mark() -> tuple[str, int, int]:
    a, m = segment()
    labels = posterise(a, m)
    h, w = labels.shape

    preview = np.zeros((h, w, 3), np.uint8)
    for lvl, key in ((0, "ink"), (1, "tan"), (2, "white")):
        preview[labels == lvl] = _hex(PALETTE[key])[::-1]
    preview[m == 0] = _hex(PALETTE["ground"])[::-1]
    cv2.imwrite(str(OUT / "poster_preview.png"), preview)

    eyes = "".join(
        f'<ellipse cx="{fx * w:.0f}" cy="{fy * h:.0f}" rx="{w * 0.030:.0f}" '
        f'ry="{w * 0.034:.0f}" fill="{PALETTE["eye"]}"/>'
        for fx, fy in EYES
    )
    layers = "".join(
        trace_layer((labels == lvl).astype(np.uint8) * 255, PALETTE[key])
        for lvl, key in ((2, "white"), (1, "tan"), (0, "ink"))
    )
    return layers + eyes, w, h


def build() -> None:
    layers, w, h = build_mark()
    side = max(w, h)
    dx, dy = (side - w) / 2, (side - h) / 2

    grounds = (("badge", PALETTE["ground"]), ("slate", PALETTE["ground_slate"]),
               ("moss", PALETTE["ground_moss"]), ("dark", PALETTE["ground_dark"]))
    for name, ground in grounds:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" '
            f'width="{side}" height="{side}">'
            f'<rect width="{side}" height="{side}" rx="{side * 0.22:.0f}" fill="{ground}"/>'
            f'<g transform="translate({dx:.1f},{dy:.1f})">{layers}</g></svg>'
        )
        (OUT / f"orbitcut-{name}.svg").write_text(svg)
        for size in (512, 128, 64, 32, 16):
            cairosvg.svg2png(bytestring=svg.encode(),
                             write_to=str(OUT / f"orbitcut-{name}-{size}.png"),
                             output_width=size, output_height=size)
        print(f"  built {name}")


if __name__ == "__main__":
    build()
