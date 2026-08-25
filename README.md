<p align="center">
  <img src="assets/icon/orbitcut-dark-512.png" alt="OrbitCut" width="140">
</p>

<h1 align="center">OrbitCut</h1>

<p align="center">
  Finds the good bits in GoPro bikejoring footage and cuts them for Instagram.<br>
  Named for Orbit, who does most of the work.
</p>

---

Point it at a card of GoPro files and it will hash them, pull the telemetry,
build proxies, score every second of every ride from the sensors, propose the
clips worth watching, let you approve them in a browser, and render what
survives as 1080×1920 Reels from the original footage.

**The bet is that the camera already measured the action.** Every GoPro from
HERO5 writes a GPMF telemetry track — accelerometer at 200 Hz, gyro, GPS, and on
HERO8+ a per-frame gravity vector. Speed, cornering, chatter and airtime are
recorded, not inferred, so "how exciting is this moment" collapses into signal
processing that runs faster than real time on a laptop CPU. Vision is reserved
for the questions sensors cannot answer, and it is not built yet.

No models, no queue, no server. One machine, SQLite, and a CLI.

---

## Install

Python is pinned to **3.14.7** by `.python-version` at the repo root. `pyenv`
reads that file automatically on `cd`, so the version is a property of the
project rather than of your shell.

```bash
brew install ffmpeg pyenv          # ffmpeg must include videotoolbox

cd orbit_cut
pyenv install 3.14.7               # once — skip if `pyenv versions` lists it
python --version                   # must print 3.14.7 before continuing

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sun]"
orbitcut doctor
```

If `python --version` shows anything else, pyenv's shims aren't on your `PATH`.
Add this to `~/.zshrc` and open a new shell:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Create the venv with `python`, not `python3` — `python` is pyenv's shim and
respects `.python-version`, while `python3` may resolve to Homebrew's or the
system interpreter and silently build the venv on the wrong version. Check with
`.venv/bin/python --version` if you're unsure what you got.

`doctor` also prints **which copy of the package is running**. If it says
site-packages rather than your repo, edits will not take effect — reinstall with
`pip install -e .`

Set where things live (add to your shell profile, or copy `.env.example`):

```bash
export ORBITCUT_ROOT=~/orbitcut          # derived data, database, renders
export ORBITCUT_HWACCEL=videotoolbox     # videotoolbox | cuda | none
```

### On Python versions

`.python-version` pins the laptop; `pyproject.toml` deliberately stays at
`requires-python = ">=3.11"`. The two are doing different jobs — the pin makes
_this_ machine reproducible, while the looser floor keeps the package
installable on the Linux desktop, which will be on whatever its distro ships.
Don't tighten `requires-python` to match the pin.

---

## The whole pipeline

```bash
orbitcut verify GX010123.MP4     # one file, in detail — run this first
orbitcut ingest ~/footage/       # hash, probe, telemetry, proxy, thumbs
orbitcut retime                  # true recording times from the GPS clock
orbitcut label --style bikejoring --mount chest

orbitcut score                   # per-second features from the sensors
orbitcut calibrate               # fit the 0-1 scale to your own library
orbitcut rank                    # which rides are worth cutting from
orbitcut overlay 0603            # watch a ride with its score curve drawn on

orbitcut clips                   # propose the clips
orbitcut review                  # approve or reject them in a browser
orbitcut log                     # what the decision log holds
orbitcut fit                     # can a fitted model beat the hand-set weights?

orbitcut level 0603              # what the horizon is doing, and what it costs
orbitcut render                  # approved clips out as 9:16 Reels
```

`doctor`, `inventory`, `timing`, `bench` and `reel` fill in the corners.

---

## Run this first — one file, in detail

Before ingesting a library, verify a single file. This is the step that tells
you whether the whole telemetry-first design applies to _your_ footage.

```bash
orbitcut verify /path/to/GX010123.MP4
```

| Check              | Expected                                        | If it's wrong                                                                        |
| ------------------ | ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| `streams`          | ACCL, GYRO, GPS9, GRAV, CORI, IORI, SHUT, ISOE  | Missing GRAV/CORI means an older camera; missing GPS9/GPS5 means GPS was off          |
| `mean \|accel\|`   | ~9.8 m/s² or a little above                     | Wildly different means the scale factor or axis parsing is off — stop and investigate |
| `mean \|gravity\|` | ~1.0                                            | Same                                                                                  |
| `GPS fix`          | >80%                                            | Lower is normal under canopy; **0% means the receiver never locked** — see below       |

It also reports **roll suppression**, which is how the camera tells you whether
it applied Horizon Lock. Never read that from a capture setting, which varies by
mode and lens.

`GRAV` is gravity in camera coordinates; applying `IORI` expresses it in image
coordinates. If the camera is levelling, gravity stops moving in the frame while
it still swings in your body, and the ratio of those two spreads is the answer:
`> 0.5` levelled in camera, `0.15–0.5` partial, `< 0.15` not levelled.

Two earlier versions of that check were wrong, which is worth knowing before
anyone "simplifies" it. Testing whether `IORI` is non-identity cannot separate
real levelling from HyperSmooth's ordinary rotational correction. Comparing the
spread of `CORI` against `CORI × IORI` fixes that but is swamped by yaw — over a
ten-minute ride your heading changes by more than a hundred degrees, dwarfing the
~25° of roll levelling removes. Gravity is the way out: rotating about the
gravity axis cannot change the gravity direction, so yaw cannot contaminate it.

---

## What the sensors give you

Six features per second, all from telemetry, no pixels:

| Feature     | From             | Notes                                                     |
| ----------- | ---------------- | --------------------------------------------------------- |
| `rough`     | ACCL @ 200 Hz    | RMS of the 5–40 Hz band of \|accel\|                       |
| `yaw_rate`  | GYRO · GRAV      | Signed, low-passed below 1 Hz, **then** rectified          |
| `air_s`     | ACCL @ 200 Hz    | Freefall window plus a landing spike. A measurement, not a guess |
| `speed_ms`  | GPS              | Gated on fix and DOP                                       |
| `lat_accel` | yaw × speed      | Cornering force — what actually looks fast on screen        |
| `grade`     | GPS altitude     | **Disabled.** GoPro altitude cannot support it; see below   |

`calibrate` turns those into 0–1 percentiles against your whole library, so a
score means "better than this share of every second you have ever shot".

**Axis order never comes into it.** GoPro's axis order varies by generation and
this parser does not normalise it, so every feature is built to be invariant:
roughness and airtime use `|accel|`, and yaw is the component about the *gravity*
axis, obtained with a dot product that does not care which axis is which.

---

## Things that were measured, and cost something to learn

The full record is in [`docs/architecture.md`](docs/architecture.md). These are
the ones most likely to be re-broken by someone tidying up.

**The composite is a power mean, not an average.** Averaging punishes
specialisation, and every exciting second is specialised — fast means straight,
twisty means slow. Under an arithmetic mean a briskly-consistent second beat a
sprint, a switchback *and* a rock garden. `SHARPNESS` sets the exponent.

**Calibration buckets on which features exist.** A mean of two terms is more
variable than a mean of four, so rides without GPS reached high values more
often for no real reason: the first live ranking put all six GPS-less rides in
the top six, and a simulation with no real difference reproduced exactly that.

**The turn feature was a vibration meter.** It rectified a 10 Hz-resampled gyro
before averaging, which turns zero-mean noise into a floor that scales with how
rough the trail is: correlation +0.62 to +0.81 with `rough`, and 8.6°/s of
"turning" while standing still. Filter the *signed* rate below 1 Hz at native
rate, then rectify.

**A fix field reading zero is not a missing fix field.** Five rides here have
`fix 0`, `DOP 100`, position `0.0/0.0` and speed exactly `0.00` for the whole
file. Failing open passed that through as a finite zero and fed ~3,800 fabricated
zeros into the corpus percentile for speed.

**The camera clock was 53–95 days slow**, drifting between sessions, and
`recorded_at` came from the container. Solar elevation computed from it labelled
eighteen daylight rides as night. GPS9 carries true UTC; `orbitcut retime`
repairs the record from the parquets without re-ingesting anything.

**Exposure cannot tell canopy shade from dusk**, so it no longer tries — it
reports `day` when the frame is bright and `unknown` otherwise.

**Descent is switched off deliberately.** Checked against a Trailforks profile of
the actual trail, GoPro altitude produced descents peaking at 5.45 m/s where the
trail's entire relief is 23 m. The feature is still computed; the *source* is
what is wrong. Re-enable it when map-matching can supply trail elevation.

**Levelling is calibrated against the pixels.** Which telemetry axis is fore–aft,
which way is positive, and how much of your body roll survives HyperSmooth are
all fitted per ride by regressing the frames' own tilt against the telemetry.
An elegant SVD approach that passed its synthetic test at correlation 1.000
returned −1536° on the first real ride.

---

## Reviewing

`orbitcut review` serves the proxies over loopback and is keyboard-driven:

```
j / k        move            a  approve      x  reject      u  undo
[ ]          nudge in-point  -  =  nudge out-point          r  replay
1-5          reject with a reason chip
d / t / n    day, twilight, night      c / h    chest, helmet
```

Time of day and mount belong to the ride, so setting either applies to every
chapter of it, and a value set here outranks anything computed — `retime` will
not overwrite it.

Every decision is written straight through to SQLite, so closing the laptop
costs nothing. **The decision log is the only artefact here that cannot be
regenerated from the footage**, which is why clearing it has its own ceremony:

```bash
orbitcut log --export decisions.json    # write and stop
orbitcut log --restart decisions.json   # write, read back, verify, then clear
```

---

## Layout

```
$ORBITCUT_ROOT/
  orbitcut.db                       SQLite: asset, segment, stage_run
  calibration.json                  the 0-1 scale, fitted to your library
  derived/<content_hash>/
      proxy.mp4                     ~50 MB — what review and levelling read
      contact.jpg                   15 frames across the ride, one image
      telemetry_10hz.parquet        all streams on one common grid
      imu_raw.parquet               ACCL and GYRO at native ~200 Hz
      scores.parquet                per-second features
      air_events.parquet            freefall windows
  renders/<ride>/                   finished Reels
  inbox/                            transient card offload
```

Derived data is keyed by content hash, so you can rename and reorganise ride
folders forever without invalidating a single artifact.

---

## Structure

Every stage is a plain function taking a path and returning a dict; the CLI is a
thin wrapper. When you eventually want them on a Celery worker, that is a
decorator, not a rewrite.

| Module           | Does                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `config.py`      | Paths, hwaccel, stage versions. Everything env-overridable            |
| `hashing.py`     | Sampled BLAKE2b — size plus three 8 MiB windows                       |
| `probe.py`       | ffprobe wrapper; detects the `gpmd` telemetry track                   |
| `telemetry.py`   | GPMF to parquet, GPS clock, sanity diagnostics                        |
| `gps.py`         | GPS5 and GPS9 parsed in-tree — telemetrik mishandles both             |
| `gpmf_compat.py` | 64-bit MP4 box and `co64` patches for files over 4 GB                 |
| `proxy.py`       | ffmpeg with a hardware-decode path and a software fallback            |
| `thumbs.py`      | 5×3 contact sheet from the proxy                                      |
| `ingest.py`      | Orchestration and idempotency                                         |
| `score.py`       | Per-second features in raw physical units                             |
| `calibrate.py`   | Corpus percentiles, availability buckets, the power-mean composite    |
| `select.py`      | Grow clips from peaks, suppress neighbours, then diversify            |
| `review.py`      | Loopback review UI with real HTTP Range support                       |
| `fit.py`         | Fits weights on the decision log and checks whether they beat the hand-set ones |
| `level.py`       | Horizon measurement, calibrated against the frames                    |
| `render.py`      | 9:16 Reels from the originals                                         |
| `overlay.py`     | A ride with its score curve burned in                                 |
| `reel.py`        | A ride's candidates back to back, for fast triage                     |
| `bench.py`       | What limits proxy speed: disk, decode or encode                       |
| `db.py`          | SQLite schema and helpers                                             |
| `naming.py`      | Ride and chapter numbers out of GoPro's filenames                     |
| `cli.py`         | All nineteen subcommands                                              |

### Self-tests

Two things here are checked by planting a known answer and seeing whether the
code recovers it, because both have failure modes that look like success:

```bash
python tools/level_selftest.py     # tilt planted in synthetic footage
python tools/select_selftest.py    # clip boundaries on planted curves
```

`tools/` also holds `gps_probe.py`, `verify_grade.py` (grade against a real GPX)
and `pull_probe.py` (a documented negative result — the accelerometer cannot see
the dog pulling).

---

## Not built yet

**Phase 4 — vision.** Is-it-mountain-biking, dog detection, and subject-aware
reframing so the crop follows Orbit instead of sitting centred. The `Pull`
sub-score waits on the same detector; `tools/pull_probe.py` established that the
accelerometer cannot substitute for it.

**Night.** The night calibration bucket, illumination gate and retroreflective
detection are designed and have never run, because there is no night footage in
the library to run them against. The MEDIUM risk on retroreflective return at
bikejoring distances is fully open.

**`archive`** — copy the original to the desktop, **re-hash at the destination**,
record `archived_path`, then delete the local copy. Never let it delete on a
transfer's exit code alone.
