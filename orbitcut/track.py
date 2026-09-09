"""Where Orbit is, second by second — the measurement `reframe.py` acts on.

Detection runs on the **proxy**, not the original. It is 540p and already on
disk, the whole ride decodes in about three seconds, and a box centre is a
fraction of the frame at any scale. The originals are for rendering.

## The orientation trap, which is invisible on every file in this library

`proxy.py` builds the proxy with `_decode_args()` and nothing else, so ffmpeg
applies the container's display matrix and **the proxy is in container
orientation**. `render.clip` instead orients from gravity, because
`docs/architecture.md` records the container lying on 77 of 97 files. A box
measured on one and a crop measured in the other are not in the same frame.

Measured across all 97 assets here, container and gravity agree everywhere:
rotation 180 on all 71 chest rides, 0 on all 24 helmet rides. So the delta is
zero on every file you own, and a tracker that ignores the mapping entirely
would render perfectly on the whole library and be silently wrong the first time
a container lied again — which has already happened once in this project.

So the mapping is not arithmetic applied afterwards. The proxy is **decoded
through `render._orient_graph(delta)`**, the renderer's own filter fragment, and
the boxes come out already in the frame the crop is measured in. There is then
nothing left to get wrong, and the one thing that could go wrong is exercised
directly by `tools/reframe_selftest.py`, which plants both directions because
only one of them exists in the footage.

## Detect on a band, in tiles, not on the whole frame

**Measured — framing the detector matters more than the detector.** On 60 frames
of ride 0598, stock COCO RF-DETR found the dog in 4 of them when handed the
whole frame squashed to a square, and in 14 when handed the band he actually
occupies, split into two overlapping tiles. Same model, same weights, 3.5x the
recall. Going up a model size, nano to medium, moved it from 12 to 14 — a fifth
of what the framing was worth.

The reason is arithmetic. The proxy is 960x540; a dog well down the trail is 25
to 60 px wide. Squashing 960 to 384 costs him 2.5x and leaves 10 to 24 px.
Cropping to the band he lives in and cutting that into two overlapping tiles
hands the model a 480-wide piece to fill 576, which is an *upscale*.

The band is where he is, and the bottom of the frame is where he never is: on a
chest mount the lower 40% is bike and forearms in every single frame. `BAND`
was read off the measured distribution — the box centres sit at cy 0.36 to 0.44
between the quartiles, and 0.28 to 0.54 between the 5th and 95th.

## Rejecting the rider, which is most of what a COCO dog detector finds here

Over 90 s of approved footage the geometry came out cleanly bimodal:

    80 boxes   width 0.04   area 0.0064   cx 0.49    — Orbit
    20 boxes   width 0.28   area 0.063    at an edge — the rider's own forearm

Ten times the area and seven times the width, and every one of the 20 touching a
left or right frame edge, because that is how an arm enters a chest-mounted
frame. Separating those needs no model and no confidence threshold, just the
geometry, which is why the priors below are shapes rather than scores.

## One dog, so no multi-object tracker

The architecture doc names ByteTrack. That is a tracker for crowded scenes — a
track pool, Hungarian assignment, identities to keep apart — and there is one
dog. The single idea in it that earns its place here is the **second
association pass against low-confidence boxes**, which is the fix for the two or
three motion-blurred frames a corner produces, and that is about forty lines.
Taking the library would buy the rest of it for a problem that does not have
the rest of it.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from . import (config, detect as detect_mod, overlay, proxy as proxy_mod,
               render)

TRACK_HZ = 5.0
# The band the dog occupies, as a fraction of frame height. Below 0.62 is bike
# and forearm on every chest-mounted frame in the library; above 0.06 is canopy.
BAND = (0.06, 0.62)
TILES = 2
OVERLAP = 0.15          # share of a tile width shared with its neighbour, so a
                        # dog on the seam is whole in one of them
BATCH = 8

# Priors, in fractions of the frame, all read off the measured distribution
# above rather than chosen. Orbit runs 0.003 to 0.009 in area; the forearms run
# 0.063. Anything in between is not a call this has to make well.
MAX_AREA = 0.030
MIN_AREA = 0.0015
EDGE = 0.02             # within this of a left/right frame edge counts as touching
EDGE_MAX_W = 0.12       # ...and that wide as well is an arm, not a dog at distance
MIN_SCORE = 0.10        # below this a box is noise, not a weak detection

# Association. The gate is generous because 5 Hz is a long time for a dog.
GATE = 0.25             # frame widths a box may sit from the prediction
SIZE_PENALTY = 0.35     # weight on disagreeing about how big he is
SCORE_BONUS = 0.20      # ...and on the detector's own confidence
HIGH = 0.30             # the first association pass uses only boxes this good
ALPHA, BETA = 0.6, 0.25  # position and velocity gains of the alpha-beta filter
MAX_GAP_S = 2.0         # longer than this is a new track, not the same dog


def orientation_delta(row: Any) -> tuple[int, str]:
    """Degrees to rotate the proxy by so it matches what `render` will produce.

    The proxy already carries the container's rotation; `render` applies
    gravity's. The difference is what has to happen to the proxy, and it is zero
    on every file in this library — which is exactly why it is computed rather
    than assumed.
    """
    container = int(row["rotation"] or 0) % 360
    deg, source = render.orientation({"rotation": container}, row["telemetry_path"])
    return (deg - container) % 360, source


def frames(proxy: str, size: int, *, hz: float = TRACK_HZ, delta_deg: int = 0,
           band: tuple[float, float] = BAND, tiles: int = TILES,
           overlap: float = OVERLAP) -> Iterator[tuple[float, list, list]]:
    """(t, tile images, tile placements) for one proxy, decoded once.

    One ffmpeg process for the whole ride, always — a 628 s proxy decodes,
    decimates and crops in about three seconds, so windowing the decode would
    save nothing and buy `-ss` timestamp arithmetic, which is a bug class worth
    declining. Window the *detection* instead, by skipping tiles the caller does
    not want before they reach the model.

    Each placement is `(x, y, w, h)` in fractions of the whole frame, so a box
    found in a tile maps back with two multiplies and no knowledge of tiling.
    """
    from PIL import Image

    # overlay's, not a second copy: its docstring records why positional
    # CSV from ffprobe mis-parses a file with more than one video section,
    # which is exactly the error this reached for first.
    w, h = overlay._video_size(proxy)
    if delta_deg in (90, 270):
        # A quarter turn would swap the frame's axes, and there is no file here
        # to check the sign of that against. `upright_from_gravity` declines a
        # lateral mount for the same reason; do not guess where it does not.
        raise RuntimeError(
            f"the proxy needs {delta_deg}° to match what render will produce, "
            f"which swaps the frame's axes. No file in this library does that, "
            f"so the mapping has never been checked — refusing rather than "
            f"guessing at it.")

    y0 = int(band[0] * h) & ~1
    bh = (int(band[1] * h) - y0) & ~1
    tw = int(w / (tiles - (tiles - 1) * overlap)) & ~1
    cmd = ["ffmpeg", "-v", "error", "-i", proxy, "-an", "-sn",
           "-vf", (f"{render._orient_graph(delta_deg)}"
                   f"fps={hz:g},crop={w}:{bh}:0:{y0},format=rgb24"),
           # The muxer must not second-guess the fps filter: everything
           # downstream reads `t = i / hz`, and one duplicated frame shifts
           # every timestamp after it with nothing to complain about it.
           proxy_mod._fps_mode_flag(), "passthrough",
           "-f", "rawvideo", "-"]
    n = w * bh * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=n * 2)
    err: list[bytes] = []
    # Drained on a thread because ffmpeg blocks once a full pipe has nowhere to
    # go — the same reason `render._run` does it.
    threading.Thread(target=lambda: err.extend(proc.stderr or []),
                     daemon=True).start()

    placements = []
    for i in range(tiles):
        x = min(int(i * tw * (1 - overlap)), w - tw)
        placements.append((x / w, y0 / h, tw / w, bh / h))

    i = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if buf is None or len(buf) < n:
                break
            strip = np.frombuffer(buf, np.uint8).reshape(bh, w, 3)
            subs = []
            for k in range(tiles):
                x = min(int(k * tw * (1 - overlap)), w - tw)
                subs.append(np.asarray(Image.fromarray(strip[:, x:x + tw])
                                       # bilinear, not lanczos: the model was
                                       # trained on ordinary resampling, and
                                       # lanczos rings on high-contrast edges.
                                       .resize((size, size), Image.BILINEAR)))
            yield i / hz, subs, placements
            i += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"decoding {Path(proxy).name} failed: "
                           f"{b''.join(err).decode(errors='replace')[:300]}")


def keep(boxes: np.ndarray) -> tuple[np.ndarray, int]:
    """Dog-class boxes that could be a dog at trail distance, and how many could not.

    The count matters as much as the boxes. "Saw nothing" and "saw only the
    rider" are different frames, and a hit rate that cannot tell them apart
    cannot say whether the detector or the framing is what needs work.
    """
    if not len(boxes):
        return boxes.reshape(0, 6), 0
    b = boxes[boxes[:, 5] == detect_mod.DOG_CLASS_ID]
    if not len(b):
        return b.reshape(0, 6), 0
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    area = w * h
    edge = (b[:, 0] < EDGE) | (b[:, 2] > 1.0 - EDGE)
    ok = ((b[:, 4] >= MIN_SCORE) & (area <= MAX_AREA) & (area >= MIN_AREA)
          & ~(edge & (w > EDGE_MAX_W)))
    return b[ok], int((~ok).sum())


def follow(seen: list[tuple[float, np.ndarray, int]], hz: float = TRACK_HZ
           ) -> pd.DataFrame:
    """One dog through a sequence of per-frame box sets.

    Constant-velocity prediction corrected by an alpha-beta filter. Not a Kalman
    filter: its covariance would be fitted to nothing here, and an unfitted
    covariance is a tuning knob wearing a lab coat.

    Two association passes, which is the one idea worth taking from ByteTrack.
    High-confidence boxes first, and only if none of those is inside the gate,
    the low-confidence ones. A dog blurred into a 0.3-confidence smear is still
    the dog; a crisp 0.8 false positive half a frame away is not. A tracker that
    simply takes the best score swaps to the distractor and swings the crop
    across the frame, which is the failure this ordering exists to prevent.
    """
    rows = []
    pos = vel = None            # (u, v) and its rate, in frame widths per second
    size = None                 # (w, h), so a box that changes size implausibly costs
    last_hit = None             # when he was last actually associated
    track_id = 0
    dt_nom = 1.0 / hz

    for t, boxes, rejected in seen:
        # Time since he was last *seen*, not since the last frame was sampled.
        # Measuring this from the frame interval instead is a quiet disaster:
        # `MAX_GAP_S` can then never be reached, so a track survives any dropout
        # and re-associates to whatever turns up near a stale prediction; the
        # prediction never extrapolates across the gap it is meant to cover; and
        # the velocity update divides a residual accumulated over seconds by a
        # single frame interval, inflating it by that ratio. On ride 0598 it
        # showed up as one unbroken track across a five-second dropout.
        since = None if last_hit is None else t - last_hit
        if since is not None and since > MAX_GAP_S:
            pos = vel = size = None          # a re-acquisition, not the same sighting
            since = None
        pred = None if (pos is None or since is None) else pos + vel * since
        # An older prediction deserves a wider gate: uncertainty grows with the
        # gap, and holding the gate fixed makes a re-acquisition after a long
        # dropout impossible for no stated reason.
        gate = GATE * (1.0 + min(since or 0.0, MAX_GAP_S) / MAX_GAP_S)

        chosen, pass2 = None, 0
        if len(boxes):
            cent = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                             (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
            wh = np.stack([boxes[:, 2] - boxes[:, 0],
                           boxes[:, 3] - boxes[:, 1]], axis=1)
            if pred is None:
                # Nothing to associate against: take the most confident box that
                # survived the priors, which is the only claim available.
                chosen = int(np.argmax(boxes[:, 4]))
            else:
                d = np.linalg.norm(cent - pred, axis=1)
                cost = d / gate - SCORE_BONUS * boxes[:, 4]
                if size is not None:
                    cost = cost + SIZE_PENALTY * np.abs(
                        np.log(np.maximum(wh[:, 0], 1e-6) / max(size[0], 1e-6)))
                for lo in (HIGH, 0.0):          # the two passes, in order
                    ok = (d <= gate) & (boxes[:, 4] >= lo)
                    if ok.any():
                        idx = np.flatnonzero(ok)
                        chosen = int(idx[np.argmin(cost[idx])])
                        pass2 = int(lo == 0.0)
                        break

        if chosen is None:
            rows.append((t, len(boxes), rejected, *([np.nan] * 6), track_id, 0))
            continue

        b = boxes[chosen]
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        sz = np.array([b[2] - b[0], b[3] - b[1]])
        if pos is None:
            track_id += 1
            pos, vel, size = c, np.zeros(2), sz
        else:
            step = since if since else dt_nom
            resid = c - pred
            pos = pred + ALPHA * resid
            vel = vel + BETA * resid / step
            size = 0.6 * size + 0.4 * sz
        last_hit = t
        rows.append((t, len(boxes), rejected, b[0], b[1], b[2], b[3],
                     float(b[4]), float(sz[0] * sz[1]), track_id, pass2))

    cols = ["t", "n_boxes", "n_rejected", "x0", "y0", "x1", "y1",
            "score", "area", "track_id", "pass2"]
    df = pd.DataFrame(rows, columns=cols)
    # `u`/`v` are derivable and stored anyway: they are what the solver reads and
    # what the overlay draws, and deriving the same centre in three places is
    # three chances to derive it differently.
    df["u"] = (df["x0"] + df["x1"]) / 2
    df["v"] = (df["y0"] + df["y1"]) / 2
    return df


def run(row: Any, *, size: str = "medium", hz: float = TRACK_HZ,
        spans: list[tuple[float, float]] | None = None,
        detector: Any = None, progress: Any = None) -> dict:
    """Track one asset, writing `track.parquet` and `track.json` beside its proxy.

    `spans` restricts *detection* to those windows — decoding is whole-ride
    regardless, because it is nearly free. A windowed run still writes the
    parquet, and the caller must not record the stage as done for one: a partial
    track that satisfies `stage_done` would make a later full run a no-op.
    """
    proxy = row["proxy_path"]
    if not proxy or not Path(proxy).exists():
        raise RuntimeError("no proxy — run `orbitcut ingest` first")
    delta, source = orientation_delta(row)
    det = detector or detect_mod.load(size)

    def wanted(t: float) -> bool:
        return spans is None or any(a <= t <= b for a, b in spans)

    seen: list[tuple[float, np.ndarray, int]] = []
    batch_t: list[float] = []
    batch_img: list[np.ndarray] = []
    batch_map: list[tuple] = []
    n_frames = 0
    looked = hits_so_far = 0

    def flush() -> None:
        if not batch_t:
            return
        out = det.detect(np.stack(batch_img))
        per = len(batch_map) // len(batch_t)
        for j, t in enumerate(batch_t):
            found = []
            for k in range(per):
                b = out[j * per + k]
                ox, oy, sx, sy = batch_map[j * per + k]
                if len(b):
                    b = b.copy()
                    b[:, [0, 2]] = ox + b[:, [0, 2]] * sx
                    b[:, [1, 3]] = oy + b[:, [1, 3]] * sy
                    found.append(b)
            allb = np.vstack(found) if found else np.zeros((0, 6), np.float32)
            seen.append((t, *keep(allb)))
        batch_t.clear(); batch_img.clear(); batch_map.clear()

    for t, subs, places in frames(proxy, det.input_size, hz=hz, delta_deg=delta):
        n_frames += 1
        if not wanted(t):
            seen.append((t, np.zeros((0, 6), np.float32), 0))
            continue
        batch_t.append(t)
        batch_img.extend(subs)
        batch_map.extend(places)
        if len(batch_t) >= BATCH:
            before = len(seen)
            flush()
            # Frames the detector was actually shown, not frames decoded: with
            # --windows most of a ride is skipped, and counting `seen` here
            # reported 3048 against a true 907 on ride 0600.
            looked += len(seen) - before
            hits_so_far += sum(1 for _, b, _ in seen[before:] if len(b))
            if progress:
                progress(t, looked, hits_so_far)
    flush()

    # Absence has to be loud. A stage that silently mis-times every row is worse
    # than one that failed, and a muxer duplicating a single frame would do
    # exactly that with nothing else noticing.
    want = round(float(row["duration_s"] or 0.0) * hz)
    if want and abs(n_frames - want) > 1:
        raise RuntimeError(
            f"decoded {n_frames} frames but {row['duration_s']:.1f}s at "
            f"{hz:g} Hz should give about {want} — refusing to write a track "
            f"whose timestamps cannot be trusted")

    df = follow(seen, hz)
    d = config.derived_dir(row["content_hash"])
    out = d / "track.parquet"
    df.to_parquet(out, index=False)

    looked = int(sum(1 for t, _, _ in seen if wanted(t)))
    hits = int(df["score"].notna().sum())
    meta = {
        "hz": hz, "band": list(BAND), "tiles": TILES, "overlap": OVERLAP,
        "delta_deg": delta, "delta_source": source,
        "priors": {"max_area": MAX_AREA, "min_area": MIN_AREA,
                   "edge": EDGE, "edge_max_w": EDGE_MAX_W,
                   "min_score": MIN_SCORE},
        "frames": n_frames, "looked_at": looked, "hits": hits,
        "windowed": spans is not None,
        "proxy": str(proxy),
        **(det.fingerprint() if hasattr(det, "fingerprint") else {}),
    }
    (d / "track.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"track_path": str(out), "track_hz": hz,
            "track_detector": getattr(det, "name", "unknown"),
            "track_frames": looked, "track_hits": hits}
