"""Stage 5 — approved clips out as 9:16 Reels, from the originals.

Two outputs, because a clip is worth having on its own and worth having in a
set: every approved clip renders standalone, and each ride's approved clips also
render in time order as one compilation.

**Rendered from the originals, never the proxies.** The proxies are 540p and
exist so that scoring and review are cheap; a Reel made from one would be a
540p source upscaled to 1080 wide. The proxy's only job here is to have told you
which seconds are worth the full decode.

**The crop is centred unless you ask otherwise.** `--frame subject` pans it to
follow Orbit, from the track `orbitcut track` writes; without it, or on a clip
the detector could not follow, the crop sits in the middle as it always has.
Centre remains the default because a change to how every existing clip renders
should be asked for rather than arrived at.

Measured over ride 0598's six approved clips: the median frame barely moves,
because a rider following a dog already points at him — but the 95th percentile
of how far Orbit sits from the crop's centre halves, 0.52 to 0.26 crop widths,
and the share of sightings where he is outside the crop altogether falls from
7% to 1%. This does not improve the typical second. It rescues the ones where a
centre crop had lost him.

It can be rotated, though, because phase 0 measured roll suppression at +0.11
and so the mount's tilt is still in the pixels. `--level` picks what to do about
it and `level.py` measures it; the crop then shrinks by exactly as much as the
angle demands, which is why an unlevelled clip loses nothing at all.

**Which way is up comes from gravity, not from the container.** The rotation is
applied explicitly, because a crop measured in pixels is meaningless until the
frame shape is settled and the container is not a reliable source for it — see
`upright_from_gravity`. Getting the decoder to hand over the unrotated frame so
that can happen is its own problem, and the flag for it silently stopped
working: see `decode_flags`.

The three shapes in this library, and what each leaves to pan with:

    5312x2988  52 files  ->  crop 1680x2988   3632 px spare
    3840x3360  37 files  ->  crop 1890x3360   1950 px spare
    5312x4648   8 files  ->  crop 2614x4648   2698 px spare

Every one crops to **full height**, so the crop is a vertical slice and the only
framing decision there is to make is horizontal. That is what `--frame subject`
spends: `reframe.py` decides where the slice sits, this drives it through
`crop`'s `x` with `sendcmd`, and vertically there is never a choice.

The 16:9 rides still clear 1080 wide, so they render — but a 9:16 slice of a
wide frame shows much less of what is coming, and most of the approved clips are
from those rides. Worth looking at before rendering all of them.

Instagram accepts 1080x1920, H.264/AAC in MP4, 24-60 fps, and a Reel over three
minutes is out of scope here by choice, so compilations are split into parts
rather than truncated.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config, proxy as proxy_mod

TARGET_W, TARGET_H = 1080, 1920
MAX_ROTATE_DEG = 25.0
MAX_REEL_S = 175.0          # a little under three minutes, for safety
CRF = 20                    # visually lossless enough at 1080; Instagram re-encodes
AUDIO_KBPS = "128k"
MAX_FPS = 60.0              # Instagram's ceiling
STOP_POLL_S = 0.25          # how fast a cancel reaches the encoder


class Cancelled(RuntimeError):
    """Raised when a render was stopped on purpose, rather than having failed."""


def _run(cmd: list[str], progress: Callable[[float], None] | None = None,
         stop: Callable[[], bool] | None = None):
    """Run ffmpeg, reporting how many seconds of output exist so far.

    `subprocess.run` is enough when nobody is watching, and that is still the
    path taken when neither hook is given. But a 30-second clip cut from a 5.3K
    HEVC original is most of a minute of decode, and a queue of six is most of
    ten — a progress bar that only moves when a clip finishes is not one. So
    `-progress pipe:1` reports `out_time_us` every few frames.

    stderr is drained on a thread rather than read after the fact: ffmpeg blocks
    once a full pipe has nowhere to go, and the error text is what the fallback
    ladder below prints when hardware encoding refuses a file.

    `stop` is polled on its own thread rather than checked when a progress line
    arrives, and that is the whole point of it. The trim happens inside the
    filter graph (see `clip`), so ffmpeg decodes from the start of the file to
    the in-point before it emits a single output frame — 757 s into a ride, that
    is two minutes during which `-progress` says nothing at all. Cancelling
    against progress lines therefore did nothing during exactly the phase you
    would want to cancel in. A watchdog does not care.
    """
    if progress is None and stop is None:
        return subprocess.run(cmd, capture_output=True, text=True)

    proc = subprocess.Popen([cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err: list[str] = []
    threads = [threading.Thread(target=lambda: err.extend(proc.stderr or []),
                                daemon=True)]
    killed = threading.Event()

    if stop is not None:
        def watch():
            while proc.poll() is None:
                if stop():
                    killed.set()
                    proc.kill()
                    return
                time.sleep(STOP_POLL_S)
        threads.append(threading.Thread(target=watch, daemon=True))

    for t in threads:
        t.start()
    for line in proc.stdout or []:
        # out_time_ms is misnamed and also carries microseconds; read whichever
        # this build prints.
        if progress and line.startswith(("out_time_us=", "out_time_ms=")):
            raw = line.split("=", 1)[1].strip()
            if raw.isdigit():
                progress(int(raw) / 1e6)
    proc.wait()
    for t in threads:
        t.join(timeout=2)
    if killed.is_set():
        raise Cancelled(" ".join(cmd[-1:]))
    return subprocess.CompletedProcess(cmd, proc.returncode, "", "".join(err))


def probe(path: str) -> dict[str, Any]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()[:200]}")
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams
              if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if not v:
        raise RuntimeError(f"no video stream in {path}")
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    return {
        # Coded dimensions — the shape of the frame *before* any rotation.
        "width": int(v["width"]), "height": int(v["height"]),
        "rotation": rot,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "fps": v.get("r_frame_rate", "30/1"),
    }


# ---------------------------------------------------------------------------
# Orientation
#
# This used to be left entirely to ffmpeg's autorotation, and the crop geometry
# was computed from the *coded* dimensions — the shape before rotation. On a
# library where every file is 0 or 180 those two never disagree, so it worked
# for 96 files out of 97 and failed silently on the one that differs.
#
# GX010600 carries a display matrix this ffprobe reads as 90 (its legacy
# `rotate` tag says 270, and the value recorded at ingest was 180 — three
# answers from one file). Its physics says otherwise: mean gravity is
# (+0.01, -0.94, +0.26), which is GX010598's (+0.02, -0.93, +0.22) to within
# noise. Same inverted chest mount, same 15-degree downward strap angle, same
# ride. The camera's orientation sensor caught something at power-on that the
# accelerometer flatly contradicts for the following ten minutes.
#
# The failure was invisible for two reasons worth remembering. The crop box is
# 9:16 by construction, so it scales to 1080x1920 without distortion no matter
# which frame it lands on — the picture comes out clean, just wrong. And the
# proxy stage and the render stage reached different answers on the same file
# from the same ffmpeg, so there was never one orientation to be wrong about.
#
# So do not ask the container. Gravity is measured rather than declared, it is
# recorded 200 times a second for the whole ride, and it cannot be set wrong by
# a sensor glitch at the moment the record button is pressed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# What the decoder actually hands us
#
# `-noautorotate` was supposed to settle this: ask for the coded frame, apply
# the rotation ourselves, and stop the output depending on the machine. On
# ffmpeg 9.0.1 that flag does nothing whatsoever. Measured on a 640x360 clip
# whose coded frame is black at the top and white at the bottom, carrying a 180
# display matrix — upright means the white ends up at the top:
#
#     no flags             white at top      matrix applied
#     -noautorotate        white at top      matrix applied
#     -autorotate 0        white at top      matrix applied
#     -display_rotation 0  white at BOTTOM   the coded frame, as asked for
#
# Both filter forms behave identically; it is not a complex-filtergraph quirk.
# With the flag believed, `clip` applied its own 180 on top of ffmpeg's, and
# every file carrying a 180 matrix rendered upside down — 77 of the 97 here,
# which is every chest-mounted ride. The 20 helmet files have no matrix and
# were unaffected, so it looked like it worked.
#
# The lesson is not "use the other flag". It is that a flag was trusted and
# never checked. So this measures: plant a known frame under a known matrix,
# decode it back with the flags we are about to use for real, and look. Once
# per process, on a 64x64 clip.
# ---------------------------------------------------------------------------

# Preferred first. `-display_rotation` arrived in ffmpeg 6.0, so an older build
# falls through to the second and is compensated for instead of failing.
_ROTATION_FLAGS = (["-display_rotation", "0", "-noautorotate"], ["-noautorotate"])
_PROBE = 64
_DECODE: tuple[list[str], bool] | None = None


def decode_flags() -> tuple[list[str], bool]:
    """(input flags to pass, whether they actually get us the coded frame).

    A False second element means this build applies the display matrix whatever
    it is told, and `clip` then refuses any file that declares a rotation rather
    than correcting for it. Correcting was tried and is unsound: it means
    subtracting the container's number from the gravity answer, and GX010600
    gives three different numbers about itself — display matrix 90, legacy tag
    270, 180 recorded at ingest. Compensation renders it upside down, measured.
    A file declaring 0 has nothing to be applied and renders normally.
    """
    global _DECODE
    if _DECODE is None:
        _DECODE = _measure_decode()
    return _DECODE


def _measure_decode() -> tuple[list[str], bool]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="orbitcut_rot_") as td:
        d = Path(td)
        plain, stamped = d / "plain.mp4", d / "stamped.mp4"
        made = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"color=c=black:s={_PROBE}x{_PROBE}:d=1:r=5",
             "-vf", f"drawbox=x=0:y=0:w={_PROBE}:h={_PROBE // 2}:color=white:t=fill",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(plain)],
            capture_output=True)
        stamp = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-display_rotation", "180",
             "-i", str(plain), "-c", "copy", str(stamped)], capture_output=True)
        if made.returncode or stamp.returncode:
            # Cannot plant the question, so cannot answer it. Assume the flag
            # works, which is what every build before 7.x did; a file with a
            # matrix then still gets the check below rather than a silent flip.
            print("    ! could not check how ffmpeg handles rotation — "
                  "assuming -noautorotate is honoured")
            return list(_ROTATION_FLAGS[-1]), True

        for flags in _ROTATION_FLAGS:
            white_on_top = _probe_upright(stamped, flags)
            if white_on_top is None:
                continue            # this build rejects the flags; try the next
            if white_on_top:
                return list(flags), True       # the coded frame, untouched
        return list(_ROTATION_FLAGS[-1]), False


def _probe_upright(path: Path, flags: list[str]) -> bool | None:
    """Is the planted white band still at the top when this comes back?

    The planted file is white on top *in the coded frame* and carries a 180
    matrix. White still on top means nothing was applied.
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "error", *flags, "-i", str(path),
         "-filter_complex", "[0:v]null[v]", "-map", "[v]", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    if r.returncode != 0 or len(r.stdout) < _PROBE * _PROBE:
        return None
    img = np.frombuffer(r.stdout[:_PROBE * _PROBE], dtype=np.uint8)
    img = img.reshape(_PROBE, _PROBE)
    return bool(img[:_PROBE // 4].mean() > img[-_PROBE // 4:].mean())


# Below this the camera was pointing near-straight up or down and none of the
# lateral axes carry a usable direction.
MIN_GRAVITY = 0.30


def upright_from_gravity(telemetry: str | None) -> int | None:
    """Degrees clockwise that make the coded frame upright, or None.

    `grav_1` is the camera's vertical axis (README, "Axis order"): positive
    when the camera is the right way up, negative when it is inverted on the
    chest strap. That is the whole rule for this library, and it is checkable —
    0609 reads +0.96 and has no display matrix at all; 0598, 0600 and 0603 all
    read about -0.94 and all want 180.

    A genuinely sideways mount would show up as `grav_0` dominating, and this
    deliberately returns None for that case rather than guessing which way round
    it goes. There is no such file here to check a sign against, and a wrong
    sign would put the horizon 180 degrees out while looking entirely plausible
    in the code. When it returns None the caller falls back to the container,
    which is exactly what happened before this function existed.
    """
    if not telemetry or not Path(telemetry).exists():
        return None
    import pandas as pd

    try:
        d = pd.read_parquet(telemetry, columns=["grav_0", "grav_1"])
    except Exception:
        return None
    g0, g1 = float(d["grav_0"].mean()), float(d["grav_1"].mean())
    if not (np.isfinite(g0) and np.isfinite(g1)):
        return None
    if max(abs(g0), abs(g1)) < MIN_GRAVITY:
        return None
    if abs(g1) < abs(g0):
        return None  # lateral: real, but unvalidated. See the docstring.
    return 0 if g1 > 0 else 180


def orientation(meta: dict[str, Any], telemetry: str | None) -> tuple[int, str]:
    """(degrees clockwise to apply, where the number came from)."""
    container = int(meta.get("rotation") or 0) % 360
    measured = upright_from_gravity(telemetry)
    if measured is None:
        return container, "container"
    if measured != container:
        return measured, f"gravity (container says {container})"
    return measured, "gravity"


def _orient_graph(deg: int) -> str:
    """Filter fragment putting the coded frame upright. Trailing comma."""
    if deg == 180:
        return "hflip,vflip,"          # cheaper than two transposes
    if deg == 90:
        return "transpose=1,"          # clockwise
    if deg == 270:
        return "transpose=2,"          # counter-clockwise
    return ""


def display_size(meta: dict[str, Any], deg: int) -> tuple[int, int]:
    """Frame shape after orientation — what the crop must be measured against."""
    w, h = int(meta["width"]), int(meta["height"])
    return (h, w) if deg in (90, 270) else (w, h)


def safe_scale(cw: int, ch: int, width: int, height: int, max_deg: float) -> float:
    """How far the crop must shrink so rotation never exposes a black corner.

    Rotating the frame and then taking the same crop pulls the frame's corners
    inward across the crop's edges. The crop's own corners, rotated by θ, span
    `w/2·sin + h/2·cos` vertically and `w/2·cos + h/2·sin` horizontally, and both
    must still fit. For an 8:7 source the crop is already full height, so even a
    5 degree correction overruns it — this is not a rounding allowance.
    """
    if max_deg <= 0.01:
        return 1.0
    a = np.radians(min(abs(max_deg), MAX_ROTATE_DEG))
    need_h = (cw / 2) * np.sin(a) + (ch / 2) * np.cos(a)
    need_w = (cw / 2) * np.cos(a) + (ch / 2) * np.sin(a)
    return float(min(1.0, (height / 2) / need_h, (width / 2) / need_w))


def crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Centred 9:16 window inside a frame of this shape.

    Dimensions are forced even: H.264 with yuv420p cannot encode an odd width or
    height, and the failure arrives from the encoder rather than from here.
    """
    if width * TARGET_H > height * TARGET_W:      # wider than 9:16 — trim sides
        cw, ch = round(height * TARGET_W / TARGET_H), height
    else:                                         # taller — trim top and bottom
        cw, ch = width, round(width * TARGET_H / TARGET_W)
    cw = min(width, cw - (cw % 2))
    ch = min(height, ch - (ch % 2))
    return cw, ch, (width - cw) // 2, (height - ch) // 2


def _encoder(hwaccel: str) -> list[str]:
    if hwaccel.startswith("videotoolbox"):
        # Quality-based rate control on VideoToolbox, which does not take -crf.
        return ["-c:v", "h264_videotoolbox", "-q:v", "55", "-profile:v", "high"]
    if hwaccel == "cuda":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(CRF)]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(CRF)]


def rotation_budget(cw: int, ch: int, width: int, height: int) -> float:
    """Largest rotation whose shrunken crop still delivers 1080 wide.

    Levelling trades pan margin for horizon. On a 16:9 source there is much less
    to trade — past about 15 degrees the crop drops under 1080 and the render
    would be upscaling. Better to level less than to invent pixels, so the angle
    is clamped here and the clamp is reported rather than applied silently.
    """
    lo, hi = 0.0, MAX_ROTATE_DEG
    for _ in range(24):
        mid = (lo + hi) / 2
        if int(cw * safe_scale(cw, ch, width, height, mid)) >= TARGET_W:
            lo = mid
        else:
            hi = mid
    return lo


_ROLL: dict[str, tuple] = {}
_CAL: dict[str, dict] = {}


def roll_for(src: str, telemetry: str | None, preview: str | None = None) -> tuple:
    """(t, the tilt visible in the picture) for one source, calibrated per ride.

    Calibration runs against `preview` — the 540p proxy — because it decodes
    dozens of frames and a rotation angle is the same at any scale. It settles
    three things telemetry alone cannot: which axis is fore-aft, which way is
    positive, and how much of the body's roll survives the camera's own
    stabilisation, which measured 0.14, 0.42 and 0.12 across three rides.

    Cached per source, since every approved clip from a ride would otherwise
    repeat it. Returns empty arrays when the fit is not good enough — levelling
    the wrong way doubles the tilt instead of removing it, and that failure is
    silent, so it is declined here rather than guessed at.
    """
    cal = calibration_for(src, telemetry, preview)
    if not cal.get("usable"):
        return np.array([]), np.array([])
    import pandas as pd

    from . import level as lv
    return lv.visible_roll(pd.read_parquet(telemetry), cal)


def calibration_for(src: str, telemetry: str | None,
                    preview: str | None = None) -> dict:
    """The per-ride horizon fit, measured once and remembered.

    Two results come out of it and they have different requirements. The
    *constant* tilt is read from the frames alone, so it survives even when the
    telemetry cannot be matched to them — which is most of this library, because
    the camera removes nearly all the roll before it lands in a frame. The
    *dynamic* series needs the fit to hold, and is refused when it does not,
    since levelling the wrong way doubles the tilt rather than failing.
    """
    if src in _CAL:
        return _CAL[src]
    import pandas as pd

    from . import level as lv

    _CAL[src] = {"usable": False, "reason": "no telemetry"}
    if not telemetry or not Path(telemetry).exists():
        return _CAL[src]
    video = preview if preview and Path(preview).exists() else src
    cal = lv.calibrate(video, pd.read_parquet(telemetry))
    _CAL[src] = cal
    if cal.get("usable"):
        print(f"    horizon: axis {cal['axis']}, {cal['gain']:+.2f} of body roll "
              f"reaches the frame (corr {cal['corr']:+.2f}, {cal['frames']} frames)")
    else:
        print(f"    horizon: {cal.get('reason', 'not calibrated')}")
    return cal


MIN_HITS = 0.15
# Share of the sampled frames in a clip that need a box before subject framing
# is worth doing at all. Below it the path is mostly the hold-and-ease policy
# rather than the dog, and a centre crop is the more honest answer. Stock COCO
# RF-DETR runs 0.20-0.35 on this library, so this is a floor for the bad clips,
# not a target — the fine-tune is what moves the number itself.


def _pan(track: str | None, t_in: float, t_out: float, dw: int, dh: int,
         cw2: int, out: Path, applied_deg: float,
         fallback: int) -> tuple[int, str, Path | None]:
    """(crop x at t=0, the sendcmd fragment, the script to delete afterwards).

    Falls back to the centred `fallback` and an empty fragment whenever the
    track cannot carry the clip, saying which of the reasons it was.

    The starting x is the *solved* value rather than the centre, deliberately.
    The first command does fire on the first frame, but seeding the filter with
    the answer means that a sendcmd which somehow did not run shows up as a
    crop sitting at a constant offset — visible and diagnosable — instead of a
    single-frame jump that reads as a stutter.
    """
    import pandas as pd

    from . import reframe as rf

    why = None
    if not track or not Path(track).exists():
        why = "no track for this ride — run `orbitcut track`"
    else:
        df = pd.read_parquet(track)
        # Padded, because the solver wants context either side of the clip and
        # discards it; unpadded, the first second of every clip is a hold it
        # did not earn.
        sl = df[(df["t"] >= t_in - rf.PAD_S) & (df["t"] <= t_out + rf.PAD_S)]
        inside = df[(df["t"] >= t_in) & (df["t"] <= t_out)]
        if not len(inside):
            why = "the track does not cover this clip"
        elif inside["u"].notna().mean() < MIN_HITS:
            why = (f"the detector found Orbit in only "
                   f"{inside['u'].notna().mean():.0%} of this clip")

    if why:
        print(f"    framing: centred — {why}")
        return fallback, "", None

    grid, x, report = rf.crop_path(
        sl["t"].to_numpy(), sl["u"].to_numpy(), sl["v"].to_numpy(),
        t_in, t_out, dw, dh, cw2, applied_deg=applied_deg)
    if not len(x):
        print("    framing: centred — the crop path came out empty")
        return fallback, "", None

    script = rf.sendcmd(grid, x, t_in, t_out, dw, cw2)
    pan_file = out.with_suffix(".pan")
    pan_file.write_text(script)
    x0 = int(round(float(x[0]))) & ~1
    x0 = max(0, min(x0, dw - cw2))
    print(f"    framing: following Orbit — hits {report['hits']:.0%}, "
          f"still {report['held']:.0%} of the clip, "
          f"pan {report['pan_p50']:.2f}/{report['pan_p95']:.2f} cw/s"
          + (f", {report['saturated']:.0%} out of reach"
             if report["saturated"] > 0.01 else ""))
    return x0, f"sendcmd=f='{pan_file.as_posix()}',", pan_file


def clip(src: str, t_in: float, t_out: float, out: Path,
         hwaccel: str | None = None, level: str | None = None,
         telemetry: str | None = None, preview: str | None = None,
         fps: float | None = None,
         frame: str = "centre", track: str | None = None,
         progress: Callable[[float], None] | None = None,
         stop: Callable[[], bool] | None = None) -> Path:
    """One approved clip as a 1080x1920 Reel.

    `level` is None, "constant" (one rotation for the whole clip, removing how
    the camera sits on the strap) or "dynamic" (per-frame, removing the lean as
    well). Constant needs only the frames; dynamic also needs the telemetry to
    match them, and renders unlevelled when it does not, because an unlevelled
    Reel is still a Reel and a double-tilted one is not.

    `fps` forces an output frame rate. Left alone the clip keeps the source's,
    which is right for a standalone clip and wrong for a part of a joined one —
    this library holds both 29.97 and 59.94 rides, and joining those by stream
    copy yields a file whose timing is nonsense while ffmpeg reports success.
    `plan_fps` picks the value; see `join`.

    `frame` is "centre" — the fixed middle slice this has always taken — or
    "subject", which pans the crop to follow Orbit using the `track` parquet
    that `orbitcut track` writes. Centre stays the default: a change to how
    every existing clip renders should be asked for, not arrived at.

    Subject framing falls back to centre, out loud, on a clip the detector could
    not follow — no track, no rows covering it, or too few frames with a box.
    A centre-cropped Reel is still a Reel; one framed on three detections and
    nine seconds of guessing is not, and the fallback is the same shape as the
    one `level="dynamic"` already takes for the same reason.

    `progress` is called with seconds of output written so far, and `stop` is
    asked four times a second whether to give up — see `_run` for why those are
    two hooks rather than one.
    """
    if frame == "subject" and level == "dynamic":
        # The rotation moves the dog as well as the horizon, so the crop target
        # would need correcting per frame against the smoothed roll series. The
        # correction is small — 5 degrees moves a dog 800 px below centre by
        # about 70 px, against 3632 px of pan and a dead zone of 200 — and its
        # sign is the exact hazard `level.py` refuses to guess at, where getting
        # it backwards doubles the error instead of removing it. Refusing is
        # better than a correction nobody has checked.
        raise RuntimeError(
            "subject framing with dynamic levelling is not implemented: the "
            "per-frame rotation moves the dog too, and that correction has not "
            "been measured. Use --level constant or none.")
    hwaccel = hwaccel or config.HWACCEL
    meta = probe(src)
    # Orientation first, and explicitly: everything below measures a crop in
    # pixels, and a crop is meaningless until you know which way up the frame
    # is. See the block above `upright_from_gravity`.
    deg, why = orientation(meta, telemetry)
    if "container says" in why:
        print(f"    orientation: {deg}° from {why} — trusting the accelerometer")

    # Everything below assumes the decoder hands over the frame as stored. That
    # is measured, not assumed — see `decode_flags` — and when it cannot be had,
    # a file that declares a rotation is refused rather than rendered on a guess.
    rot_flags, coded = decode_flags()
    container = int(meta.get("rotation") or 0) % 360
    if not coded and container:
        raise RuntimeError(
            f"this ffmpeg applies the container's display matrix whatever it is "
            f"told, and {Path(src).name} declares {container}°. Correcting for "
            f"that means trusting a number this library has caught lying. "
            f"ffmpeg 6.0 or newer accepts -display_rotation, which does work.")
    orient_graph = _orient_graph(deg)
    dw, dh = display_size(meta, deg)
    cw, ch, _x, _y = crop_box(dw, dh)
    dur = max(t_out - t_in, 0.1)

    rot_graph, cmd_file, applied = "", None, 0.0
    if level:
        from . import level as lv
        budget = rotation_budget(cw, ch, dw, dh)
        cal = calibration_for(src, telemetry, preview)

        if level == "constant":
            # Constant asks only what the frames show, so it does not need the
            # telemetry fit to have held — and on this library it usually has
            # not, because the camera takes the roll out before it reaches a
            # frame. What it cannot take out is how the camera sits on the strap.
            ang = float(np.clip(cal.get("constant_deg", 0.0), -budget, budget))
            # Below the threshold the tilt is not visible but the crop cost is
            # real, so a near-square mount pays nothing.
            if abs(ang) < lv.MIN_CONSTANT_DEG or not cal.get("constant_usable"):
                ang = 0.0
            applied = abs(ang)
            if applied:
                rot_graph = f"rotate={-np.radians(ang):.6f}:ow=iw:oh=ih,"

        else:
            tt, roll = roll_for(src, telemetry, preview)
            span = ((tt >= t_in) & (tt < t_out) & np.isfinite(roll)
                    if len(roll) else np.zeros(0, dtype=bool))
            if span.sum() > 4:
                sm = np.clip(lv.smoothed(tt, roll), -budget, budget)
                applied = float(np.max(np.abs(sm[span])))
                cmd_file = out.with_suffix(".cmds")
                cmd_file.write_text(lv.sendcmd(tt, sm, t_in, t_out))
                rot_graph = (f"sendcmd=f='{cmd_file.as_posix()}',"
                             f"rotate=0:ow=iw:oh=ih,")

    # Shrink the crop to whatever the applied rotation demands. Doing this after
    # the angle is known, rather than reserving a fixed margin, means an
    # unlevelled clip loses nothing at all.
    k = safe_scale(cw, ch, dw, dh, applied)
    cw2, ch2 = int(cw * k) & ~1, int(ch * k) & ~1
    x2, y2 = (dw - cw2) // 2, (dh - ch2) // 2

    # Where the crop sits horizontally. Vertically there is never a choice —
    # every source shape in this library crops to full height — so the whole of
    # subject framing is this one number over time.
    pan_graph, pan_file = "", None
    if frame == "subject":
        from . import reframe as rf
        x2, pan_graph, pan_file = _pan(track, t_in, t_out, dw, dh, cw2, out,
                                       applied if level == "constant" else 0.0,
                                       fallback=x2)

    # trim in the filter graph, not via -ss/-t: with more than one input those
    # bind to whichever input follows them, which once produced a reel twelve
    # times too long that reported success.
    # The frame rate is settled last, after the rotation the levelling filters
    # want every source frame for, and before the scale that is the expensive
    # part of the graph.
    fps_graph = f"fps={fps:.6f}," if fps else ""

    graph = (f"[0:v]trim=start={t_in:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS,"
             f"{orient_graph}"
             f"{rot_graph}"
             f"{fps_graph}"
             # sendcmd emits on frame timestamps, so the pan script goes
             # immediately before the crop it steers and after `fps`, where the
             # command lands on the same frame `crop` then processes.
             f"{pan_graph}"
             f"crop={cw2}:{ch2}:{x2}:{y2},"
             f"scale={TARGET_W}:{TARGET_H}:flags=lanczos,"
             f"setsar=1[v]")
    maps = ["-map", "[v]"]
    if meta["has_audio"]:
        graph += (f";[0:a]atrim=start={t_in:.3f}:duration={dur:.3f},"
                  f"asetpts=PTS-STARTPTS[a]")
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", AUDIO_KBPS, "-ar", "48000"]

    cmd = ["ffmpeg", "-y", "-v", "error",
           *proxy_mod._decode_args(hwaccel if hwaccel != "videotoolbox_vt"
                                   else "videotoolbox"),
           # Orientation is decided above, from gravity, so the decoder is
           # asked for the frame as stored. Which flag does that, and whether
           # any of them does on this build, is measured at `decode_flags` —
           # `-noautorotate` alone silently stopped working and took every
           # chest-mounted ride upside down with it.
           *rot_flags,
           "-i", src, "-filter_complex", graph, *maps, *_encoder(hwaccel),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    r = _run(cmd, progress, stop)
    if r.returncode != 0 and hwaccel != "none":
        # Same ladder as proxy generation: step down rather than abandoning
        # hardware entirely on the first complaint.
        nxt = proxy_mod.FALLBACK.get(hwaccel) or "none"
        first = (r.stderr.strip().splitlines() or ["no stderr"])[0]
        print(f"  ! {hwaccel} failed ({first[:110]}) — retrying with {nxt}")
        # By keyword past `fps`: this call is positional up to there, and
        # inserting an argument into the middle of a positional recursive call
        # is a silent shift rather than an error.
        return clip(src, t_in, t_out, out, nxt, level, telemetry, preview,
                    fps, frame=frame, track=track, progress=progress, stop=stop)
    if r.returncode != 0:
        raise RuntimeError(f"render failed: {r.stderr.strip()[:300]}")
    if cmd_file:
        cmd_file.unlink(missing_ok=True)
    if pan_file:
        pan_file.unlink(missing_ok=True)
    _check(out, dur)
    return out


def source_fps(path: str) -> float:
    """Frame rate of a source, as a number. 0.0 when it cannot be read."""
    try:
        num, _, den = probe(path)["fps"].partition("/")
        return float(num) / float(den or 1)
    except (RuntimeError, ValueError, ZeroDivisionError):
        return 0.0


def plan_fps(rates: list[float], to: str = "fastest") -> float | None:
    """One frame rate for a set of clips about to be joined, or None if they agree.

    This library is 29.97 and 59.94 — 52 rides at one, 37 at the other, and the
    approved clips include both. Stream-copying those together does not fail and
    does not come out the wrong length; it comes out **variable rate**. Measured
    on two two-second parts at 30 and 60: 180 frames over 4.02 s, sixty of them
    33 ms apart and then 119 at 16.7 ms, in a file that declares 60 fps and
    averages 45. Everything local reads that as fine. It is Instagram's
    re-encode, on a phone, that decides what to do with it.

    The default target is the fastest of them. `render` has never set an output
    rate, so a 59.94 ride already renders standalone at 59.94, and conforming a
    joined cut down to 30 would make the same clip look different depending on
    what it was joined to. Duplicating frames in a 29.97 clip costs nothing
    visible; halving a 59.94 one throws away the smoothness it was shot for.

    `to="slowest"` is the other answer, and it is the one the plan's output table
    gives ("conform 60 fps source down"). It is smaller and it is what Instagram
    recommends, so it is offered rather than argued with — the UI asks.

    Instagram takes up to 60, so the ceiling never binds on this library and is
    there for the day a 120 fps clip arrives.
    """
    have = sorted({round(r, 3) for r in rates if r > 0})
    if len(have) < 2:
        return None
    return have[0] if to == "slowest" else min(have[-1], MAX_FPS)


def avg_fps(path: str | Path) -> float:
    """Frames actually delivered per second, which is not always the declared rate.

    A concatenated file carries the rate of whichever part came first. This is
    the number that disagrees with it.
    """
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    try:
        num, _, den = r.stdout.strip().partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def compile_reel(parts: list[Path], out: Path,
                 hwaccel: str | None = None,
                 progress: Callable[[float], None] | None = None,
                 stop: Callable[[], bool] | None = None) -> Path:
    """Join rendered clips, by copy where a copy is honest and by re-encode where not.

    Parts that share an encode need only be copied, which is the fast path and
    the usual one — everything `clip` writes is 1080x1920 H.264/AAC. Parts that
    disagree about frame rate cannot be, and the reason this checks up front
    rather than afterwards is that the bad copy passes every check made after
    the fact: right length, right size, right codec, wrong pacing. See
    `plan_fps` for the measurement.

    The length is still verified, because a copy can go wrong for reasons that
    have nothing to do with frame rate, and a reel that is silently short is
    worse than one that failed.
    """
    rates = [source_fps(str(p)) for p in parts]
    if plan_fps(rates) is not None:
        print(f"    parts disagree on frame rate "
              f"({', '.join(f'{r:.2f}' for r in sorted(set(rates)))}) "
              f"— joining by re-encode")
        return _join_reencode(parts, out, hwaccel, sum(_duration(p) for p in parts),
                              progress, stop)

    want = sum(_duration(p) for p in parts)
    listing = out.parent / f"{out.stem}_parts.txt"
    # Absolute paths (which -safe 0 already permits) so the parts need not be
    # siblings of the output; a hand-made cut keeps its clips in its own folder.
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts))
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)],
        capture_output=True, text=True)
    listing.unlink(missing_ok=True)

    got = _duration(out) if r.returncode == 0 else 0.0
    if r.returncode == 0 and (want <= 0 or abs(got - want) <= max(0.5, 0.02 * want)):
        return out
    if r.returncode == 0:
        print(f"    join by copy came out {got:.1f}s against {want:.1f}s of parts "
              f"— re-encoding the join")
    return _join_reencode(parts, out, hwaccel, want, progress, stop)


def _join_reencode(parts: list[Path], out: Path, hwaccel: str | None,
                   want: float, progress: Callable[[float], None] | None = None,
                   stop: Callable[[], bool] | None = None) -> Path:
    """Join by decoding, for parts a copy cannot honestly splice.

    The concat *filter* rather than the demuxer: it resamples each input onto
    one timeline instead of trusting them to already share one, which is the
    whole reason for being here.
    """
    hwaccel = hwaccel or config.HWACCEL
    fps = plan_fps([source_fps(str(p)) for p in parts]) or source_fps(str(parts[0]))
    ins: list[str] = []
    pre, legs = [], []
    for n, part in enumerate(parts):
        ins += ["-i", str(part)]
        pre.append(f"[{n}:v]fps={fps:.6f},setsar=1[v{n}];"
                   f"[{n}:a]aresample=48000[a{n}]")
        legs.append(f"[v{n}][a{n}]")
    graph = ";".join(pre) + ";" + "".join(legs) + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    cmd = ["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", graph,
           "-map", "[v]", "-map", "[a]", *_encoder(hwaccel),
           "-c:a", "aac", "-b:a", AUDIO_KBPS, "-ar", "48000",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    r = _run(cmd, progress, stop)
    if r.returncode != 0:
        raise RuntimeError(f"concat failed: {r.stderr.strip()[:300]}")
    got = _duration(out)
    if want > 0 and abs(got - want) > max(1.0, 0.05 * want):
        raise RuntimeError(f"{out.name} is {got:.1f}s but its parts total {want:.1f}s")
    return out


def split_by_budget(clips: list[dict], budget: float = MAX_REEL_S) -> list[list[dict]]:
    """Group clips into reels that each fit the length limit.

    Truncating to the limit would cut a clip mid-action, so a clip that does not
    fit starts the next reel instead. A single clip longer than the budget still
    gets its own reel rather than being dropped — that is a selection problem,
    and silently discarding approved footage would be worse than a long file.
    """
    reels: list[list[dict]] = []
    cur: list[dict] = []
    used = 0.0
    for c in clips:
        d = c["t_out"] - c["t_in"]
        if cur and used + d > budget:
            reels.append(cur)
            cur, used = [], 0.0
        cur.append(c)
        used += d
    if cur:
        reels.append(cur)
    return reels


def _check(path: Path, want: float) -> None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=width,height",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        data = json.loads(r.stdout or "{}")
        got = float(data["format"]["duration"])
        v = next(s for s in data["streams"] if s.get("width"))
    except (KeyError, ValueError, StopIteration, json.JSONDecodeError):
        return
    if abs(got - want) > max(1.0, 0.15 * want):
        raise RuntimeError(f"{path.name} is {got:.1f}s but should be {want:.1f}s")
    if (int(v["width"]), int(v["height"])) != (TARGET_W, TARGET_H):
        raise RuntimeError(f"{path.name} is {v['width']}x{v['height']}, "
                           f"expected {TARGET_W}x{TARGET_H}")
