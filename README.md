# OrbitCut — Phase 0

Ingest and inventory for GoPro mountain-bike footage. Hash, probe, extract
telemetry, generate proxies, and tell what you actually have.

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

`doctor` should print `ok` for ffmpeg, ffprobe, telemetrik, pandas, pyarrow,
numpy and astral. Fix anything it flags before going further.

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

All three compiled dependencies publish cp314 wheels for Apple Silicon — numpy
≥2.5, pandas ≥3.0, pyarrow ≥25 — so nothing builds from source. If you ever see
pip start compiling pyarrow, you are on a Python version newer than its wheels
and should drop back one minor release rather than wait it out.

---

## Run this first — one file, in detail

Before ingesting a library, verify a single file. This is the step that tells
you whether the whole telemetry-first design actually applies to _your_ footage.

```bash
orbitcut verify /path/to/GX010123.MP4
```

You are looking for four things:

| Check              | Expected                                       | If it's wrong                                                                                   |
| ------------------ | ---------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `streams`          | ACCL, GYRO, GPS5, GRAV, CORI, IORI, SHUT, ISOE | Missing GRAV/CORI means an older camera; missing GPS5 means GPS was off                         |
| `mean \|accel\|`   | ~9.8 m/s²                                      | A wildly different number means the scale factor or axis parsing is off — stop and investigate  |
| `mean \|gravity\|` | ~1.0                                           | Same                                                                                            |
| `GPS fix`          | >80%                                           | Lower is normal under canopy, but it means speed and trail matching will be patchy on that ride |

It also reports **roll suppression**, which is how the camera tells you whether
it applied Horizon Lock — never read that from a capture setting, which varies
by mode and lens.

`GRAV` is gravity in camera coordinates; applying `IORI` expresses it in image
coordinates. If the camera is levelling, gravity stops moving in the frame while
it still swings in your body, and the ratio of those two spreads is the answer:

| Suppression | Means                                             |
| ----------- | ------------------------------------------------- |
| `> 0.5`     | leveled in camera — do **not** level again in post |
| `0.15–0.5`  | partially leveled — inspect before trusting        |
| `< 0.15`    | not leveled — post-leveling applies                 |

Two earlier versions of this check were wrong, which is worth knowing before
anyone "simplifies" it. Testing whether `IORI` is non-identity cannot separate
real levelling from HyperSmooth's ordinary rotational correction. Comparing the
spread of `CORI` against `CORI × IORI` fixes that but is swamped by yaw — over a
ten-minute ride your heading changes by more than a hundred degrees, dwarfing
the ~25° of roll that levelling removes, and full Horizon Lock scores +0.025 on
that metric. Gravity is the way out: rotating about the gravity axis cannot
change the gravity direction, so yaw cannot contaminate it. It is also immune to
an inverted mount, because it measures the variance of a direction and never its
sign.

In 8:7 this should always read "not leveled", because Horizon Lock needs the
Linear + Horizon Lock digital lens that 8:7 does not offer.

### Axis order

`accl_0/1/2`, `gyro_0/1/2` and `grav_0/1/2` are **positional**, not semantic —
GoPro's axis order varies by camera generation and the parser does not normalise
it. This costs nothing for the features that matter most, because roughness and
airtime both use `|accel|`, which is rotation-invariant. It matters for anything
gravity-pitch based.

**Measured on HERO11, chest mount, camera inverted** (GX010598, 628 s):

```
mean gravity   (+0.02, -0.93, +0.22)
```

- **`grav_1` is the camera's vertical axis.** It reads **negative** here because
  the camera is mounted upside down; a right-way-up mount should read positive.
  That sign is therefore a usable inversion check, and it is what a mount
  classifier must not confuse with a genuinely different mount position.
- **`grav_2` is fore–aft.** The steady `+0.22` alongside `-0.93` is a constant
  `atan(0.22 / 0.93) ≈ 13°` tilt — the deliberate downward angle of the chest
  mount, not noise. A helmet mount should sit closer to level.
- **`grav_0` is lateral**, and near zero on average as expected: you spend equal
  time leaning each way.

Confirm the signs against one of your own clips before relying on them; this is
one file from one mount.

### Camera frame vs image frame

This file carries a **180° display matrix**. ffmpeg honours it by default, so
proxies come out the right way up with no special handling — but **telemetry is
in camera coordinates, which stay inverted**. Anything that maps a sensor
direction onto a screen direction has to apply `asset.rotation` first, or a
"pan right" derived from `CORI` becomes a pan left on screen. `probe` records
the value per asset for exactly this reason; nothing in phase 0 needs it yet.

---

## Ingest

```bash
orbitcut ingest /path/to/a/ride/          # directory, recursive
orbitcut ingest /path/to/GX010123.MP4     # or one file
```

Four stages per file:

- **probe** — container metadata, no decoding. Instant.
- **telemetry** — GPMF to parquet. Seconds. Decodes no video.
- **proxy** — 540p H.264. The slow half.
- **thumbs** — a 5x3 contact sheet from the proxy. Nearly free, and the fastest
  way to see whether a file is worth anything before you scrub it.

Telemetry is deliberately separate from proxy so you can score a whole ride
moments after offload while proxies are still churning in the background.

Everything is keyed on a content hash and recorded in `stage_run`, so
re-running is free and safe. Bump a number in `STAGE_VERSIONS` (config.py) and
only that stage redoes on the next run — that is how you reprocess selectively
when you change something later.

```bash
orbitcut ingest ~/footage --force        # redo regardless of cache
```

---

## Inventory

```bash
orbitcut inventory
orbitcut inventory --csv ~/footage-inventory.csv
```

The output is phase 0's actual deliverable: what you own, how long it is, what
aspect ratios you shot, and — most importantly — **which files have telemetry**.
Anything listed `tel: NO` cannot be scored from sensors.

---

## Layout

```
$ORBITCUT_ROOT/
  orbitcut.db                       SQLite: asset, stage_run
  derived/<content_hash>/
      proxy.mp4                     ~50 MB — what every later stage reads
      contact.jpg                   15 frames across the ride, one image
      telemetry_10hz.parquet        all streams on one common grid
      imu_raw.parquet               ACCL at native ~200 Hz, for the freefall detector
  renders/                          phase 3
  inbox/                            transient card offload
```

Originals are organised however you like — by date, by ride. Derived data is
keyed by content hash, so you can rename and reorganise ride folders forever
without invalidating a single artifact.

---

## Structure

Every stage is a plain function taking a path and returning a dict:

```python
from orbitcut import probe, telemetry, proxy, hashing

h    = hashing.content_hash(path)
meta = probe.probe(path)
tel  = telemetry.extract(path, h)
prx  = proxy.make_proxy(path, h)
```

The CLI is a thin wrapper over those. When you eventually want them on a Celery
worker, that is a decorator — not a rewrite.

| Module         | Does                                                       |
| -------------- | ---------------------------------------------------------- |
| `config.py`    | Paths, hwaccel, stage versions. Everything env-overridable |
| `hashing.py`   | Sampled BLAKE2b — size plus three 8 MiB windows            |
| `probe.py`     | ffprobe wrapper; detects the `gpmd` telemetry track        |
| `telemetry.py` | GPMF to parquet, sanity diagnostics, sun elevation         |
| `proxy.py`     | ffmpeg with a hardware-decode path and a software fallback |
| `thumbs.py`    | 5x3 contact sheet from the proxy                           |
| `gpmf_compat.py` | 64-bit MP4 box and `co64` patches for files over 4 GB    |
| `ingest.py`    | Orchestration and idempotency                              |
| `db.py`        | SQLite schema and helpers                                  |
| `cli.py`       | `doctor`, `verify`, `ingest`, `inventory`                  |

---

## Not built yet

`archive` — copy the original to the desktop, **re-hash at the destination**,
record `archived_path`, then delete the local copy. Deliberately left for when
you have somewhere to archive to. Write it before you need it, and never let it
delete on transfer exit code alone.
