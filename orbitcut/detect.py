"""The dog detector, behind an interface, because which model this is will change.

COCO-pretrained RF-DETR is the starting point and not the destination. The
architecture doc picks RF-DETR for the eventual fine-tune on Orbit specifically
— one dog, a couple of hundred labelled frames, and a model that stops missing
him at distance — so keeping the model behind a protocol means that fine-tune is
a weights swap rather than a rewrite of everything downstream.

Everything above this module sees **normalised** boxes and a class id, and knows
nothing else. Normalising on the way out is the point: a detector's input
resolution is its own business, and `track.parquet` must mean the same thing
whether it was written by nano at 384 or medium at 576.

**torch is imported inside the constructor, never at module scope.** `orbitcut
ingest` must not pay two gigabytes of wheels to make a proxy, and `orbitcut
track --help` must work on a machine that never installed the extra.

Two model-free detectors live here as well, and they are not an afterthought:
`StubDetector` and `BrightestDetector` are what let `tools/reframe_selftest.py`
exercise the real frame pipe, the real orientation handling and the real
tracker on a machine with no vision extra installed — the same reason
`level_selftest` plants telemetry rather than reading a ride.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# COCO's 80 class names, in the order every 80-class checkpoint emits them.
COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# ...and the *category ids* those 80 names carry in the COCO annotations, which
# run to 90 with gaps because eleven categories were dropped after the ids were
# assigned. The two are not interchangeable and this is the whole reason the
# table is here rather than in a comment.
COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)
COCO_CATEGORY_NAMES = dict(zip(COCO_CATEGORY_IDS, COCO_CLASS_NAMES))

DOG_CLASS_ID = 18
# **A COCO category id, not an index into the 80-name list.** RF-DETR's
# pretrained checkpoints emit the raw category id — its own `predict` docstring
# says so — and the obvious-looking `COCO_CLASS_NAMES.index("dog")` returns 16.
# Category 16 is "bird". So the wrong answer here is not an exception and not an
# empty result: it is a bird, and a crop that follows a bird looks exactly like
# a crop that is following nothing in particular. Checked in
# `tools/reframe_selftest.py` rather than trusted.

# Deliberately below rfdetr's own 0.5 default. `track.follow` runs a second
# association pass against low-confidence boxes, which is what carries the track
# through the two or three motion-blurred frames that a corner produces — and it
# cannot do that with boxes the detector already threw away. The floor that
# separates "a weak detection" from "noise" is applied in `track.py`, where it
# is one of the priors and gets counted in the report.
DEFAULT_THRESHOLD = 0.25

# (constructor name, the square side the model was trained at). Frames are
# handed over already resized to that side, so rfdetr's own resize becomes a
# no-op — see `track.frames`.
RFDETR_SIZES = {"nano": ("RFDETRNano", 384),
                "small": ("RFDETRSmall", 512),
                "medium": ("RFDETRMedium", 576)}


@runtime_checkable
class Detector(Protocol):
    """What `track.py` requires of a model, and the whole of it."""

    name: str          # goes into track.json — a track from a different
                       # detector is a different quantity, not a better one
    input_size: int    # the square side frames must be handed over at

    def detect(self, frames: np.ndarray) -> list[np.ndarray]:
        """(B, S, S, 3) uint8 RGB in; one (n, 6) float32 array per frame out.

        Columns are `x0, y0, x1, y1, score, class_id`, with the box normalised
        to 0-1 of the frame that was handed in. `n` may be zero. The order of
        the rows carries no meaning — ranking them is the tracker's job.
        """


class RFDetrDetector:
    """RF-DETR from the `rfdetr` package, COCO-pretrained, on MPS where there is one.

    Weights download once into `~/.roboflow/models` (`RF_HOME` overrides) with an
    md5 check, so the first run needs the network and every run after is offline.

    `optimize_for_inference(compile=True)` is deliberately not called. It makes
    `predict` require exactly `batch_size` images per call, and the last batch of
    a ride is short — a speedup that turns the tail of every ride into an
    exception is not a speedup.
    """

    def __init__(self, size: str = "nano", device: str | None = None,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        if size not in RFDETR_SIZES:
            raise ValueError(f"unknown detector size {size!r} — "
                             f"one of {', '.join(RFDETR_SIZES)}")
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "the dog detector needs the vision extra, which is not "
                "installed: pip install -e '.[vision]'") from exc

        ctor, self.input_size = RFDETR_SIZES[size]
        kw = {"device": device} if device else {}
        self.model = getattr(rfdetr, ctor)(**kw)
        self.threshold = float(threshold)
        self.size = size
        self.name = f"rfdetr-{size}-coco"
        # The package exports no `__version__`; the installed distribution's
        # metadata is the only place the number lives.
        try:
            import importlib.metadata as _md
            self.version = _md.version("rfdetr")
        except Exception:
            self.version = "unknown"

    @property
    def device(self) -> str:
        """Where inference actually runs. Nested one deeper than it looks:
        the wrapper has no `.device`, its inner model does."""
        return str(getattr(getattr(self.model, "model", None), "device", "unknown"))

    def detect(self, frames: np.ndarray) -> list[np.ndarray]:
        # A list of arrays rather than one stacked array: `predict` returns a
        # bare Detections for a single image and a list for several, and
        # normalising that here keeps the branch out of the caller.
        batch = [np.ascontiguousarray(f) for f in frames]
        out = self.model.predict(batch, threshold=self.threshold,
                                 # Otherwise every Detections keeps a copy of
                                 # the frame it came from, and a ride's worth of
                                 # those is gigabytes of nothing.
                                 include_source_image=False)
        if not isinstance(out, list):
            out = [out]
        s = float(self.input_size)
        rows = []
        for det in out:
            xyxy = np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4) / s
            conf = np.asarray(det.confidence, dtype=np.float32).reshape(-1, 1)
            cls = np.asarray(det.class_id, dtype=np.float32).reshape(-1, 1)
            rows.append(np.hstack([xyxy, conf, cls]).astype(np.float32))
        return rows

    def fingerprint(self) -> dict:
        """What went into `track.json` — enough to tell two tracks apart."""
        return {"detector": self.name, "rfdetr_version": self.version,
                "input_size": self.input_size, "threshold": self.threshold,
                "device": self.device}


class StubDetector:
    """Boxes from a planted table, keyed on the frame's index in the sequence.

    No torch, no weights, no network. Frames are counted, not looked at, which
    is exactly what a check of the tracker or the solver wants: the boxes are the
    input under test and the pixels are irrelevant.
    """

    def __init__(self, plan: dict[int, np.ndarray] | list[np.ndarray],
                 input_size: int = 384, name: str = "stub") -> None:
        self.plan = plan
        self.input_size = input_size
        self.name = name
        self.seen = 0

    def detect(self, frames: np.ndarray) -> list[np.ndarray]:
        empty = np.zeros((0, 6), dtype=np.float32)
        rows = []
        for _ in range(len(frames)):
            i = self.seen
            self.seen += 1
            if isinstance(self.plan, dict):
                got = self.plan.get(i, empty)
            else:
                got = self.plan[i] if i < len(self.plan) else empty
            rows.append(np.asarray(got, dtype=np.float32).reshape(-1, 6))
        return rows

    def fingerprint(self) -> dict:
        return {"detector": self.name, "input_size": self.input_size}


class BrightestDetector:
    """The brightest blob in the frame, reported as a dog-class box.

    This is how the orientation check gets an answer out of the *real* frame
    pipe without a model. A synthetic original carries a bright blob at a known
    place; whatever this reports back is where the pipe put it, and the whole
    proxy-orientation-versus-gravity-orientation question reduces to comparing
    two numbers. A stub taking boxes from a table could not do that, because the
    thing under test is what happened to the pixels.
    """

    def __init__(self, input_size: int = 384, half: float = 0.05,
                 name: str = "brightest") -> None:
        self.input_size = input_size
        self.half = half            # half-width of the reported box, normalised
        self.name = name

    def detect(self, frames: np.ndarray) -> list[np.ndarray]:
        rows = []
        for f in frames:
            g = np.asarray(f, dtype=np.float32).mean(axis=2)
            iy, ix = np.unravel_index(int(np.argmax(g)), g.shape)
            u, v = (ix + 0.5) / g.shape[1], (iy + 0.5) / g.shape[0]
            h = self.half
            rows.append(np.array(
                [[u - h, v - h, u + h, v + h, 1.0, DOG_CLASS_ID]],
                dtype=np.float32))
        return rows

    def fingerprint(self) -> dict:
        return {"detector": self.name, "input_size": self.input_size}


def load(size: str = "nano", device: str | None = None,
         threshold: float = DEFAULT_THRESHOLD) -> Detector:
    """The detector `orbitcut track` uses, by name. One place to change later."""
    return RFDetrDetector(size=size, device=device, threshold=threshold)
