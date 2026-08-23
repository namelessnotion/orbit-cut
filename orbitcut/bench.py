"""Find out what actually limits proxy generation on this machine.

`timing` says proxy is 97% of ingest. That is not the same as saying the
encoder is the problem — a proxy run is three things in series, and any of them
can be the ceiling:

    read the original off disk  ->  decode it  ->  scale and encode

Each has a different fix and two of them are unaffected by a faster encoder, so
guessing is expensive. This measures all three on one real file, plus every
backend the build supports, and reports them in the same unit (multiples of
realtime) so they can be compared directly.

Two details that matter for the numbers to mean anything:

* The window starts a quarter of the way in. The head of a GoPro file is often
  the camera settling — and on a spinning disk the outer tracks are faster —
  so timing the first thirty seconds flatters everything.
* The disk test reads a region the decode test does not touch. Reading the same
  bytes twice measures the page cache, which would make a slow disk look fast
  and send you optimising the wrong thing.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from . import config, proxy as proxy_mod

WINDOW_S = 30.0          # long enough to average out, short enough to iterate
START_FRACTION = 0.25
READ_MB = 512
READ_FRACTION = 0.70     # well clear of the decode window


def _probe(path: str) -> tuple[float, int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", "-select_streams", "v", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:200]}")
    data = json.loads(r.stdout)
    dur = float(data.get("format", {}).get("duration") or 0.0)
    st = next((s for s in data.get("streams", [])
               if not s.get("disposition", {}).get("attached_pic")), {})
    return dur, int(st.get("width") or 0), int(st.get("height") or 0)


def read_throughput(path: str) -> float | None:
    """MB/s reading a window the decode test will not have cached."""
    size = Path(path).stat().st_size
    # Prefer a big window, but never give up on a small file: a quarter of it
    # still measures the disk, and a missing row is worse than an approximate one.
    span = min(READ_MB * 1024 * 1024, size // 4)
    if span < 8 * 1024 * 1024:
        return None
    offset = min(int(size * READ_FRACTION), size - span)

    got, t0 = 0, time.perf_counter()
    with open(path, "rb") as f:
        f.seek(offset)
        while got < span:
            chunk = f.read(min(8 * 1024 * 1024, span - got))
            if not chunk:
                break
            got += len(chunk)
    elapsed = time.perf_counter() - t0
    return (got / 1e6) / elapsed if elapsed > 0 else None


REPEATS = 2


def _time_ffmpeg(cmd: list[str], repeats: int = REPEATS) -> tuple[float | None, str]:
    """Best of N. Fastest rather than mean, deliberately: a slow run means
    something else was competing for the machine, and the question here is what
    the pipeline costs, not what the laptop was doing at the time."""
    best, err = None, ""
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if r.returncode != 0:
            return None, (r.stderr.strip().splitlines() or ["no stderr"])[0][:110]
        best = elapsed if best is None else min(best, elapsed)
    return best, err


def decode_scale(path: str, start: float, hwaccel: str,
                 height: int) -> tuple[float | None, str]:
    """Everything except the encoder: read, decode, scale. Subtract from the
    full pass and what remains is what encoding costs.

    An earlier version of this measured decode with no filter at all, which
    looked purer and was useless: with no `scale` the null sink receives frames
    at full 3840x3360 and pays to handle them, so "decode only" came out slower
    than decode-plus-scale-plus-encode. Any probe that reports a stage as slower
    than a superset of itself is measuring its own scaffolding. Matching the
    real filter chain and changing only the encoder keeps the comparison honest.
    """
    filters = proxy_mod._filter_and_encode(hwaccel, height)
    vf = filters[filters.index("-vf") + 1] if "-vf" in filters else f"scale=-2:{height}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin",
           *proxy_mod._decode_args(hwaccel),
           "-ss", f"{start:.2f}", "-t", f"{WINDOW_S:.2f}", "-i", path,
           "-vf", vf, "-an", "-f", "null", "-"]
    return _time_ffmpeg(cmd)


def full_pass(path: str, start: float, hwaccel: str, height: int) -> tuple[float | None, str]:
    """The real proxy pipeline, discarding the output rather than writing it."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin",
           *proxy_mod._decode_args(hwaccel),
           "-ss", f"{start:.2f}", "-t", f"{WINDOW_S:.2f}", "-i", path,
           *proxy_mod._filter_and_encode(hwaccel, height),
           "-an", "-f", "null", "-"]
    return _time_ffmpeg(cmd)


def concurrency(path: str, start: float, hwaccel: str, height: int,
                n: int) -> float | None:
    """Aggregate throughput with `n` proxies running at once.

    This is the measurement that decides whether `--jobs` is worth anything. If
    the media engine is already saturated by a single stream, n passes take n
    times as long and the aggregate is flat. If one stream leaves it idle
    waiting on per-frame overhead, the aggregate climbs.
    """
    from concurrent.futures import ThreadPoolExecutor

    cmd = ["ffmpeg", "-v", "error", "-nostdin",
           *proxy_mod._decode_args(hwaccel),
           "-ss", f"{start:.2f}", "-t", f"{WINDOW_S:.2f}", "-i", path,
           *proxy_mod._filter_and_encode(hwaccel, height),
           "-an", "-f", "null", "-"]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        codes = list(pool.map(
            lambda _: subprocess.run(cmd, capture_output=True).returncode, range(n)))
    elapsed = time.perf_counter() - t0
    if any(codes) or elapsed <= 0:
        return None
    return (WINDOW_S * n) / elapsed          # seconds of footage per second


def _warm(path: str, start: float, hwaccel: str) -> None:
    """One discarded decode of the measurement window.

    Without this the first timed run pays for a cold page cache and every later
    one does not, which produced a nonsense reading the first time this ran:
    decode-only came out *slower* than decode-plus-scale-plus-encode, purely
    because it went first.
    """
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-nostdin",
         *proxy_mod._decode_args(hwaccel if hwaccel != "videotoolbox_vt" else "videotoolbox"),
         "-ss", f"{start:.2f}", "-t", f"{WINDOW_S:.2f}", "-i", path, "-f", "null", "-"],
        capture_output=True,
    )


def candidates() -> list[str]:
    """Backends worth trying here, current one first."""
    order = [config.HWACCEL]
    for extra in ("videotoolbox_vt", "videotoolbox", "cuda", "none"):
        if extra not in order:
            order.append(extra)
    # Only offer the VideoToolbox pair on a build that has the encoder at all.
    have = subprocess.run(["ffmpeg", "-v", "quiet", "-encoders"],
                          capture_output=True, text=True).stdout
    keep = []
    for hw in order:
        if hw.startswith("videotoolbox") and "h264_videotoolbox" not in have:
            continue
        if hw == "cuda" and "h264_nvenc" not in have:
            continue
        keep.append(hw)
    return keep


def run(path: str, height: int | None = None) -> dict:
    height = height or config.PROXY_HEIGHT
    dur, w, h = _probe(path)
    if dur <= 0:
        raise RuntimeError("could not read a duration from that file")
    start = max(0.0, dur * START_FRACTION)
    window = min(WINDOW_S, max(dur - start, 1.0))

    size_gb = Path(path).stat().st_size / 1e9
    out: dict = {"file": Path(path).name, "size_gb": size_gb, "duration_s": dur,
                 "width": w, "height": h, "window_s": window,
                 "read_mbps": read_throughput(path), "backends": {}}

    # Everything below reads the same window, so warm it once and let every
    # measurement start from the same place.
    _warm(path, start, config.HWACCEL)

    # Decode-only uses the current backend: it is a property of the input, not
    # of which encoder you pick afterwards. If the hardware decoder is not
    # available at all, measure software decode rather than printing nothing —
    # the input-side ceiling is still the number we came here for.
    el, err = decode_scale(path, start, config.HWACCEL, height)
    out["decode_hw"] = config.HWACCEL
    if el is None and config.HWACCEL != "none":
        el, err2 = decode_scale(path, start, "none", height)
        out["decode_hw"] = "none"
        out["decode_note"] = f"{config.HWACCEL} decode unavailable ({err})"
        err = err2
    out["decode_x"] = (window / el) if el else None
    out["decode_error"] = err

    for hw in candidates():
        el, err = full_pass(path, start, hw, height)
        out["backends"][hw] = {"x_realtime": (window / el) if el else None,
                               "error": err}

    # Does running more than one at a time buy anything? Use the fastest
    # backend that actually worked, since that is what ingest would use.
    working = [(hw, r["x_realtime"]) for hw, r in out["backends"].items()
               if r["x_realtime"]]
    if working:
        best_hw = max(working, key=lambda kv: kv[1])[0]
        out["concurrency_backend"] = best_hw
        out["concurrency"] = {
            n: concurrency(path, start, best_hw, height, n) for n in (1, 2, 3)
        }
    return out
