# CLAUDE.md

OrbitCut turns GoPro mountain-bike footage into Instagram Reels, choosing the clips from
**telemetry rather than pixels**. Every GoPro from HERO5 writes a GPMF track — accelerometer at
~200 Hz, gyro, gravity, GPS — and that track already measured the action. No models, no queue,
no server: one machine, SQLite, and a CLI.

The library is currently 97 files of bikejoring (riding behind a dog, Orbit) on Michigan
singletrack, shot on a HERO11.

## Documentation of record

Read these before changing behaviour; they carry reasoning this file only summarises.

| File                                  | Holds                                                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `README.md`                           | How to run it, what each check means, and a **fully worked example** tracing one second (0603 @ 544 s) from raw accelerometer samples to the final composite |
| `docs/architecture.md`                | Every design decision, each marked assumption or **measured**. The measured ones are findings that cost real work — do not silently reverse one              |
| `docs/telemetry-review-2026-08-24.md` | An audit of the pipeline, with the remediation plan that followed                                                                                            |

## Stack

- **Python 3.14.7**, pinned by `.python-version` (pyenv). `pyproject.toml` deliberately stays at
  `requires-python = ">=3.11"` so the package still installs on the Linux desktop — do not
  tighten it to match the pin.
- numpy, pandas, pyarrow, scipy, matplotlib, pillow; `telemetrik` for GPMF; `astral` optional
  (`[sun]` extra) for sun elevation.
- **ffmpeg / ffprobe** must be on PATH, with videotoolbox on macOS.
- SQLite (WAL), parquet for per-second and raw-rate data.
- No test framework. The tests are `tools/*_selftest.py` — see below.

```bash
python -m venv .venv && source .venv/bin/activate   # `python`, not `python3` — pyenv shim
pip install -e ".[sun]"
orbitcut doctor            # prints __version__ and the import path it actually loaded
```

`doctor` printing an unexpected path is the fastest way to catch an edit that landed in a
different copy of the package than the one being run.

## Environment

Everything has a default; nothing is required. See `.env.example`.

| Variable                | Default             | Notes                                                       |
| ----------------------- | ------------------- | ----------------------------------------------------------- |
| `ORBITCUT_ROOT`         | `~/orbitcut`        | derived data, database, renders                             |
| `ORBITCUT_DB`           | `$ROOT/orbitcut.db` |                                                             |
| `ORBITCUT_HWACCEL`      | `videotoolbox`      | `videotoolbox` \| `cuda` \| `none` — never hardcode         |
| `ORBITCUT_PROXY_HEIGHT` | `540`               |                                                             |
| `ORBITCUT_ARCHIVE`      | unset               | the `archive` stage refuses to run without it, deliberately |

## Two filesystems, and they are not the same one

```
<repo>/                     code only — media is .gitignore'd, aggressively
$ORBITCUT_ROOT/
  orbitcut.db               asset, segment, stage_run
  calibration.json          the corpus percentile tables + weights IN FORCE
  inbox/                    card offload (currently still holds all 97 originals)
  derived/<content_hash>/
      proxy.mp4             540p — what scoring and review read
      contact.jpg           5x3 sheet
      telemetry_10hz.parquet
      imu_raw.parquet       ACCL/GYRO at native ~201 Hz
      scores.parquet        one row per second, raw physical units
  renders/<ride>/           finished clips and per-ride reels
```

Derived data is keyed by **content hash** (`s3b2_…` — size plus three 8 MiB windows), so
originals can be renamed or reorganised freely. If they move, `orbitcut relink <dir>` finds them
again by hash; nothing else in the catalog cares.

## Commands

```bash
orbitcut doctor                       # toolchain check
orbitcut verify FILE                  # one file in detail — run before ingesting a library
orbitcut ingest PATH [--force] [--jobs N]
orbitcut inventory [--csv F] [--files]

orbitcut retime [--dry-run]           # true UTC start from the GPS clock
orbitcut label [ASSET] --style bikejoring --mount chest
orbitcut orient [ASSET] [--all]       # where the container disagrees with the accelerometer
orbitcut relink DIR [--all] [--dry-run]

orbitcut track [ASSET] [--hz N] [--size nano|small|medium] [--windows]
orbitcut score [ASSET]                # per-second physics
orbitcut calibrate [--weights speed=0.1,rough=0.6] [--sharpness N]
orbitcut rank [--by clip|top30|floor] [--top N]
orbitcut overlay ASSET                # proxy with the score curve drawn on
orbitcut overlay ASSET --track        # ...or Orbit's boxes and the crop window
orbitcut level ASSET                  # what the horizon is doing, and what levelling costs

orbitcut clips [ASSET] [--top 6]      # propose candidates
orbitcut review [ASSET] [--port N]    # approve/reject in a browser
orbitcut cut [ASSET] [--port N]       # pick approved clips, queue the renders
orbitcut reel ASSET                   # a ride's candidates back to back, for a fast look
orbitcut log [--export F] [--restart F]
orbitcut fit [--seed N]               # can a fitted model beat the hand-set weights?

orbitcut render [ASSET] [--level none|constant|dynamic]
                        [--frame centre|subject] [--no-compile]
orbitcut shelf [--port N] [--local-only]   # finished renders + QR to the phone

orbitcut timing                       # where ingest spends its time
orbitcut bench ASSET                  # disk, decode or encode bound?
```

`ASSET` accepts a hash prefix, a filename, or a 4-digit ride number (`0603`).

`--weights` / `--sharpness` on `rank`, `clips` and `overlay` are **previews** — they re-apply at
read time without touching `calibration.json`. Only `orbitcut calibrate` bakes them in.

### Tests

```bash
python tools/level_selftest.py     # horizon fit (slow — decodes frames)
python tools/select_selftest.py    # clip growing, peak placement, freefall protection
python tools/orient_selftest.py    # plants a white bar at the top, checks where it lands
python tools/cut_selftest.py       # the queue, and joining clips of different frame rates
python tools/qr_selftest.py        # QR matrices against externally-checked answers
python tools/reframe_selftest.py   # the dead zone, dropouts, priors, one-dog tracking
```

Planted-answer tests, not assertions on remembered numbers. Every check in them corresponds to a
bug that shipped. Run the relevant one after touching `level.py`, `select.py`, `render.py`,
`cut.py`, `reframe.py` or `track.py`.

`reframe_selftest.py` needs neither footage nor the vision extra: the solver
checks are numpy against planted tracks, and the tracker checks run through
`detect.StubDetector`. Two of its checks exist for bugs that shipped in the
first draft and could not have been caught by watching output — a solver that
returned smooth, plausible paths without converging, and an association aged
from the sampled frame rather than the last sighting.

`orient_selftest.py` also checks the decoder itself, not only the arithmetic — see the
autorotation invariant below. Its two "white bar ends up at the top" failures were real: every
chest-mounted ride was rendering upside down.

Also in `tools/`: `gps_probe.py`, `pull_probe.py`, `verify_grade.py` — investigation scripts,
not tests.

## The pipeline, and how re-running works

```
probe → telemetry → proxy → thumbs     (ingest)
      → score → calibrate → clips → review → render
                                             ↖ cut (same stage, by hand)
```

`render` takes everything approved, grouped by ride in the order it happened.
`cut` is the same renderer driven from a browser: pick clips across rides, order
them, and queue one file or several. It writes to `renders/cuts/` and has its own
queue table; it does **not** touch `segment`.

Every stage is a plain function taking a path and returning a dict; the CLI is a thin wrapper.
Completion is recorded per `(content_hash, stage)` in `stage_run` against
`config.STAGE_VERSIONS`, so **re-running is free and safe**. To force a redo of one stage,
bump its number in `config.py` — that is the mechanism, and the comments there record why each
version exists.

`config.STAGE_VERSIONS` is the **only** source of stage versions. Module-level `STAGE_VERSION`
constants used to exist in `score.py` and `select.py`, were read by nothing, and drifted. Do not
reintroduce them.

## Modules

| Module                                | Does                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------- |
| `config.py`                           | Paths, hwaccel, stage versions, `MIN_RIDE_S`. All env-overridable                           |
| `hashing.py`                          | Sampled BLAKE2b content hash                                                                |
| `naming.py`                           | `GX010674` → ride `0674`, chapter `1`. Chapters are one ride split by the camera            |
| `probe.py`                            | ffprobe wrapper; detects the `gpmd` track                                                   |
| `gpmf_compat.py`                      | 64-bit box / `co64` patches for files over 4 GB                                             |
| `gps.py`                              | GPS5 and GPS9 parsed **in-tree** — telemetrik mis-scales GPS5 and returns GPS9 as raw bytes |
| `telemetry.py`                        | GPMF → parquet, sanity diagnostics, GPS clock, sun elevation                                |
| `proxy.py` / `thumbs.py`              | 540p proxy with a hw→sw fallback ladder; contact sheet                                      |
| `ingest.py`                           | Orchestration and idempotency                                                               |
| `score.py`                            | **Physics only.** Per-second features in raw units                                          |
| `calibrate.py`                        | **Opinion.** Corpus percentiles, weights, the composite                                     |
| `select.py`                           | Per-second curve → candidate clips                                                          |
| `review.py`                           | Local HTTP server + single-page UI for approve/reject                                       |
| `cut.py`                              | Picking approved clips in a browser, and the render queue behind it                         |
| `shelf.py`                            | Finished renders in a browser; the **only** thing here that binds a network interface       |
| `qr.py`                               | QR encoder in-tree, byte mode level M, versions 1-10. Frozen spec; do not "improve" it      |
| `webui.py`                            | What `review` and `cut` share: loopback server, 206 range serving                           |
| `fit.py`                              | Fits weights on the decision log; within-ride AUC                                           |
| `level.py`                            | Horizon: measures the tilt, decides whether levelling is worth the crop                     |
| `render.py`                           | Approved clips → 1080×1920 Reels, from the originals                                        |
| `overlay.py` / `reel.py` / `bench.py` | Score curve burn-in, contact reel, throughput probe                                         |
| `db.py`                               | Schema and helpers                                                                          |
| `cli.py`                              | Every subcommand                                                                            |

## Data model

**`asset`** — one row per file, keyed on `content_hash`. Notable columns: `ride_id`/`chapter`
(from the filename), `recorded_at` (the camera's clock, which ran 53–95 days slow) alongside
`recorded_at_gps` and `clock_drift_s` (the truth, from GPS9), `rotation`, `style`/`mount`,
`lighting`/`lighting_source`, and the `*_path` columns pointing into `derived/`.

**`segment`** — one row per candidate clip, `UNIQUE (content_hash, t_in)`. `status` is
`candidate|approved|rejected|rendered`; `t_in_user`/`t_out_user` hold hand-adjusted edges;
`features` is the JSON sub-score vector. Re-running selection replaces candidates but **must
never clobber a decision** — see `replace_candidates()`.

**`stage_run`** — `(content_hash, stage) → version, status`. The idempotency ledger.

**`render_job`** — what `cut` was asked to render: the segment ids in play order, the frame rate
they were normalised to, progress, and where the file landed. Disposable — every row can be made
again by picking the same clips. A rendered clip stays `approved` and is never moved to
`rendered`; `fit` trains on `approved`, so promoting them would quietly shrink the training set
every time you posted something.

The decision log lives in `segment`, and it is the one thing here that cannot be regenerated.
`orbitcut log --restart FILE` exports and verifies the read-back _before_ clearing.

## Invariants worth not breaking

Each of these was learned the expensive way.

**Axis order is positional, not semantic.** `accl_0/1/2`, `gyro_*`, `grav_*` vary by camera
generation and nothing normalises them. Every feature is built to be invariant: roughness and
airtime use `‖accel‖`; turning is the component _about the gravity axis_, obtained with a dot
product. Never index one of these by assumed meaning.

**Physics and opinion stay in separate files.** `score.py` stores measurements in physical units;
`calibrate.py` converts them to 0–1 at read time. This is why changing a weight costs nothing to
re-apply. Do not push weighting into `score.py`.

**The weights in force live in `calibration.json`, not in `calibrate.py`.** `apply()` defaults to
the table's weights, because a level computed with different weights is a different quantity from
the level distribution stored in the table. Editing `WEIGHTS` without re-running
`orbitcut calibrate` changes nothing that `rank` sees.

**Sharpness cancels the weights.** The composite is a power mean at exponent `SHARPNESS`. At high
p it is nearly a maximum, and a maximum has no use for weights — at p = 12 a feature weighted 0.3
contributed 0.00027% for a specialised second. To de-emphasise a feature at high sharpness you
must set it to **zero**; small-but-nonzero does almost nothing. To keep it in play and
de-emphasise it, lower p. Full arithmetic in the README.

**Absence must be loud.** NaN, never a substituted default. A GPS `fix` of 0 is a _measurement of
failure_, not a missing field — five rides passed 3,800 fabricated zero speeds into the corpus
percentile before that distinction existed. Gaps NaN'd by `telemetry._gap_aware_interp` must not
be re-interpolated by a later stage; `score.compute`'s `on_grid` uses the gap-aware version for
exactly that reason.

**Filter before rectifying.** Rectifying a noise-dominated signal turns zero-mean chatter into a
positive floor that scales with roughness. The turn feature was a vibration meter for months
because of this — it read 8.6 °/s while standing still. Worth a factor of 14 on a real second.

**`shelf` is the one thing that leaves loopback, and the narrowing is the feature.** A QR aimed at
127.0.0.1 aims the phone at itself, so serving a file to a phone means being reachable from it. It
binds the LAN address specifically — never `0.0.0.0`, so loopback is not even listening — puts
every path behind a per-run token that is never written to disk, resolves every path and refuses
anything outside `$ORBITCUT_ROOT/renders`, and exposes no endpoint that writes. `--local-only`
refuses the LAN entirely. Do not widen any of those four without a reason written down here.

**The QR encoder was wrong three times and looked perfect every time.** A bad QR is a picture that
looks exactly like a QR code and scans as nothing, so it cannot be reviewed by eye. It was built
against `segno` (bit-exact, all ten versions at exact capacity) and `zxing-cpp` (every payload
length 1-213 decoded); neither is a dependency, so `tools/qr_selftest.py` keeps the answers those
runs produced. Note that **OpenCV's decoder fails on some valid symbols** — including segno's own
output at certain masks — so it is not evidence of a bug. Mask choice is a heuristic and encoders
legitimately differ; only the matrix has to be right.

**Never trust the container for orientation.** GX010600 reports 90 (display matrix), 270 (legacy
tag) and 180 (recorded at ingest) about itself. `render` derives the rotation from gravity, which
is measured rather than declared.

**And do not trust the flag that is supposed to suppress autorotation, either.** On ffmpeg 9.0.1
`-noautorotate` and `-autorotate 0` both do nothing — measured, on both `-vf` and
`-filter_complex`; only `-display_rotation 0` (ffmpeg 6.0+) delivers the coded frame. With
`-noautorotate` believed, `clip` applied its own 180 on top of ffmpeg's and **every file carrying
a 180 matrix rendered upside down — 77 of 97, every chest-mounted ride**. The 20 helmet files
declare 0 and were unaffected, which is why it looked like it worked. `render.decode_flags()` now
plants a known frame under a known matrix and measures what comes back, once per process; a build
that cannot be stopped refuses any file declaring a rotation rather than compensating, because
compensating means subtracting the container's number and 0600 is what happens then.

**ffmpeg: trim inside the filter graph, not with `-ss`/`-t`.** With more than one input those bind
to whichever input follows them — that once produced a reel twelve times too long that reported
success.

**Never `git add .`** — this is a video project. The `.gitignore` is deliberately aggressive.

## House style

Comments explain **why**, and specifically record what was measured, what was tried, and what
failed. A rationale that outlives its numbers is worse than no rationale — when the numbers move,
rewrite or delete the comment. Several module docstrings exist purely to stop a future
"simplification" from reintroducing a bug; treat them as load-bearing.

Prose in comments and docs is plain and direct, with no bullet-point padding, and it names
specific numbers from real files rather than gesturing at "improved accuracy".

## Current state (2026-09-07)

- 97 assets, 49,247 scored seconds. 437 segments: 49 approved, 56 rejected, 332 unreviewed.
- Weights in force: `speed 0.0 / turn 0.15 / rough 0.85`, sharpness **2**.
- All originals are still in `inbox/`. The `archive` stage is **not built** — `archived_path` is
  null everywhere. Write it before you need it, and never let it delete on a transfer exit code.
- Mount is no longer single-valued: 20 helmet files against 77 chest, which corresponds exactly
  to container rotation 0 vs 180. That split is also what made the upside-down render look
  intermittent — only the 77 were affected.

Open items, in rough order of value: six probable helmet files mislabelled `chest` (0614, 0616,
0618, 0619); 24 assets whose `lighting` is not `day` despite the library being all daylight;
`CLIP_MAX_S = 30` is binding on the top decile of approved clips and was meant to be 15–20; the
`MIN_RIDE_S` gate filters `rank` and `clips` but not ingest.

## Commits

Subject line says what changed and, where there is one, the finding behind it. Body explains the
reasoning and cites the numbers. End with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
