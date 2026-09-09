"""Planted checks on assembling a cut: the queue, and joining across rides.

The hard part of `orbitcut cut` is not the picking, it is that a join can be
wrong without anything failing. Stream-copying a 30 fps clip onto a 60 fps one
succeeds, gives the right duration, the right size and the right codec — and a
variable frame rate. Two synthetic two-second parts come out as 180 frames over
4.02 s in a file that declares 60 fps and averages 45: sixty frames 33 ms apart,
then 119 at 16.7 ms. This library is 52 rides at 29.97 and 37 at 59.94, so a cut
that mixes them is the normal case, not the exotic one.

That is why the length check this file was first written with is not enough, and
why the checks below test the pacing instead:

  * `plan_fps` must notice a mismatch, and must round *up* — dropping a 60 fps
    ride to 30 throws away the reason it was shot that way.
  * `compile_reel` must refuse to copy parts that disagree, and what it returns
    must be constant-rate, not merely the right length.
  * the queue must hand out jobs oldest-first and must not hand out a cancelled
    one at all.

The two clips are synthesised by ffmpeg rather than taken from the library, so
this runs in a few seconds and needs no footage.

    python tools/cut_selftest.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TMP = tempfile.mkdtemp(prefix="orbitcut_cut_selftest_")
os.environ["ORBITCUT_ROOT"] = TMP          # before config is imported
os.environ["ORBITCUT_DB"] = str(Path(TMP) / "test.db")

from orbitcut import cut, db, render as rn   # noqa: E402


def check(name: str, ok: bool, detail: str) -> int:
    print(f"  {name:<46}{detail:<34}{'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def synth(path: Path, seconds: float, fps: float) -> Path:
    """A 1080x1920 H.264/AAC clip of the shape `render.clip` writes."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=1080x1920:rate={fps}:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-ar", "48000", "-shortest", str(path)],
        check=True, capture_output=True)
    return path


def main() -> int:
    fails = 0
    tmp = Path(TMP)

    # 1. Frame rates. Agreement means "leave it alone" — normalising a set that
    #    already agrees would re-time every clip for nothing.
    fails += check("matching rates need no normalising",
                   rn.plan_fps([29.97, 29.97, 29.97]) is None, "None")
    fails += check("a mismatch resolves upward by default",
                   rn.plan_fps([29.97, 59.94]) == 59.94, "59.94")
    fails += check("...and downward when asked to conform",
                   rn.plan_fps([29.97, 59.94], "slowest") == 29.97, "29.97")
    fails += check("nothing exceeds Instagram's ceiling",
                   rn.plan_fps([59.94, 120.0]) == rn.MAX_FPS, f"{rn.MAX_FPS:g}")

    # 2. The join, given exactly the mismatch the fps plan exists to prevent.
    #    Length is the check that does NOT catch this: a copy of these is 4.02s,
    #    which is right, while pacing 45 fps of frames through a file that says
    #    60. So the planted answer is the rate, and it has to be the fast one.
    a, b = synth(tmp / "a.mp4", 2.0, 30), synth(tmp / "b.mp4", 2.0, 60)
    out = rn.compile_reel([a, b], tmp / "joined.mp4")
    got, avg = rn._duration(out), rn.avg_fps(out)
    fails += check("a mismatched join comes out full length",
                   abs(got - 4.0) < 0.35, f"{got:.2f}s of 4.00s")
    fails += check("...and at one steady rate, the faster one",
                   abs(avg - 60.0) < 1.0, f"{avg:.1f} fps, not 45")

    # 3. And the ordinary case still takes the fast path: same rate, copied.
    c = synth(tmp / "c.mp4", 2.0, 30)
    same = rn.compile_reel([a, c], tmp / "same.mp4")
    fails += check("a matched join is still just a copy",
                   abs(rn._duration(same) - 4.0) < 0.35
                   and abs(rn.avg_fps(same) - 30.0) < 1.0,
                   f"{rn._duration(same):.2f}s at {rn.avg_fps(same):.1f} fps")

    # 4. Names. A render you have already posted must never be overwritten by
    #    the next cut that happens to be called the same thing.
    fails += check("a name is made safe for the filesystem",
                   cut.slug("0603 / best bit!") == "0603_best_bit", "0603_best_bit")
    first = cut._free(tmp / "taken.mp4")
    first.write_bytes(b"")
    second = cut._free(tmp / "taken.mp4")
    fails += check("an existing render is not overwritten",
                   second.name == "taken_2.mp4", second.name)

    # 5. The queue. Order is its only promise, and a cancelled job must not be
    #    handed out — the row is the queue, not a hint about it.
    conn = db.connect()
    one = db.enqueue_job(conn, "one", [1], None, None, 10.0)
    two = db.enqueue_job(conn, "two", [2], None, 59.94, 20.0)
    fails += check("the queue runs oldest first",
                   db.next_job(conn)["id"] == one, f"job {one}")
    db.update_job(conn, one, status="cancelled")
    fails += check("a cancelled job is skipped",
                   db.next_job(conn)["id"] == two, f"job {two}")

    # 6. A worker killed mid-render leaves a row claiming to be running. On the
    #    next start that has to read as a failure, because the file it was
    #    writing is half a clip.
    db.update_job(conn, two, status="running")
    n = db.orphaned_jobs(conn)
    row = conn.execute("SELECT status FROM render_job WHERE id = ?", (two,)).fetchone()
    fails += check("an interrupted job does not stay 'running'",
                   n == 1 and row["status"] == "error", f"{n} recovered")
    conn.close()

    print(f"\n  {'all checks passed' if not fails else str(fails) + ' FAILURE(S)'}")
    print(f"  scratch: {TMP}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
