# OrbitCut telemetry review — errors and gaps for bikejoring clip detection

Reviewed 2026-08-24 against `docs/architecture.md`, the full `orbitcut/` source, `tools/`,
and the live data: `orbitcut.db` (97 assets, 199 decisions), `calibration.json`, and
telemetry/scores parquets for all 18 GPS rides plus 0603. Every finding marked **verified**
was confirmed against that data, not just read off the code. Scope was the telemetry →
scoring → calibration → selection → decision-log path; `level.py`/`render.py` got only a
light pass, since the horizon work is already heavily self-documented and isn't on the
clip-detection path.

What checked out first: the GPS5 SCAL fix, the arity-keyed column naming, the availability
buckets, the power-mean composite, the air OR-fold, and `fit.py`'s within-ride discipline all
match the architecture doc and look correct in code and in the calibration table.

## Error 1 — every "night" label in the library is wrong: lighting is computed from a stale camera clock (verified)

`ingest` computes sun elevation from `asset.recorded_at`, which `probe` takes from the MP4
container's `creation_time` — the camera's internal clock. The plan said to use "a UTC
timestamp in the GPS stream"; the code never does, even though `gps.py` parses GPS9's
`gps_days`/`gps_secs` (true GPS UTC) into every telemetry parquet.

Decoding those GPS timestamps for all 18 GPS rides and recomputing solar elevation at the
recorded lat/lon:

| ride | container clock (db) | GPS time (true) | clock drift | true elevation | true lighting | stored |
|---|---|---|---|---|---|---|
| 0654 | 2026-02-03 01:55Z | 2026-03-28 16:37Z | 53.6 d | +48.0° | day | night |
| 0657 | 2026-02-04 02:12Z | 2026-03-29 16:55Z | 53.6 d | +49.6° | day | night |
| 0660 | 2026-02-03 02:09Z | 2026-04-12 20:44Z | 68.8 d | +36.7° | day | night |
| 0663 | 2026-02-10 00:01Z | 2026-04-19 20:12Z | 68.8 d | +43.5° | day | night |
| 0667–0673 | Feb 2026 | 2026-04-23…26 | 79.5 d | +36…57° | day | night |
| 0674–0684 | Feb 2026 | 2026-05-07…10 | 93.6–94.9 d | +6…59° | day | night |

Eighteen of eighteen. The camera clock is 53–95 days behind reality and the drift grows
between sessions, so it isn't a one-off reset that a constant offset could repair. All the
"night" rides the database reports are late-March-to-May daytime rides; the library currently
contains **no verified night footage at all**. (The clock *was* right in June 2025 —
0598's 23:51:49Z is exactly the 19:51 EDT ride the doc describes — which is presumably why
this never surfaced.)

A twist worth savouring: the GPS-less siblings of these rides were labeled **day** by the
exposure fallback, and the sun-path files in the same session were labeled **night**. The
exposure fallback — the humble one — was right every time; the "exact, closed-form" solar
path was wrong, because its time input was fiction. The doc's rule that sensor disagreement
is the interesting output would have caught this at ingest: same session, two chapters, two
different lighting labels is exactly such a disagreement, and nothing currently checks it.

Blast radius: `lighting` and `lighting_source` on 21 assets; `recorded_at` itself (dates are
one to three months off, which also poisons `fit.py`'s "month" attribute, inventory ordering,
and the future date-based archive layout); every conclusion drawn from the lighting column in
`orbitcut log` / `fit` attribute breakdowns — the "night rides get approved less" pattern in
the decision log is actually a statement about mislabeled day rides. Composite scores are
untouched today only because lighting isn't yet in the calibration key.

Fixes, in order: (1) at telemetry extract, when GPS9 is present, derive the authoritative
UTC start time as `gps_time − t` and store it (or at least use it for sun elevation);
(2) compare it against `creation_time` and warn loudly past a few minutes of drift — make
disagreement loud; (3) backfill: everything needed is already in the parquets
(`gps_days`, `gps_secs`, lat, lon), so a repair pass over the DB is cheap and needs no
re-ingest — same philosophy as `db._backfill_rides`; (4) sync the camera clock (GPS on will
also keep it honest going forward).

## Error 2 — the turn sub-score is substantially a vibration meter (verified)

`score.yaw_rate()` projects GYRO onto gravity on the **10 Hz grid**, takes `abs()`, then
averages per second. Two problems compound. The 200 Hz gyro was resampled to 10 Hz by
`np.interp` — point sampling with no anti-alias low-pass — so trail chatter folds into the
"yaw" series. Then the absolute value before averaging rectifies zero-mean noise into a
positive floor that scales with vibration.

Measured on ride 0679: **corr(yaw_rate, rough) = 0.81**, and the yaw_rate median on
near-stationary seconds (speed < 0.8 m/s) is **0.15 rad/s ≈ 9°/s of "turning" while standing
still**, versus 0.41 rad/s while moving. A third or more of typical moving yaw is noise
floor, so the turn sub-score (weight 0.30) double-counts roughness (weight 0.20), and
`lat_accel = yaw × speed` inherits the contamination on GPS rides.

This hurts bikejoring rides worst: they are slower and less twisty (0679's speed p95 is
4.2 m/s), so genuine yaw is small and the floor is a large share of what gets percentile-
ranked. It also matters for the *planned* chest/helmet classifier, which correlates CORI yaw
against GPS heading through the same resampling path.

Fix: work at native rate from `imu_raw.parquet` (GYRO is already on the ACCL timeline
there), low-pass the **signed** projected yaw rate below ~1–2 Hz — real cornering lives well
under 1 Hz — and only then take per-second magnitude; or define the per-second turn feature
as |net heading change| (integral of the signed rate), which lets rectified noise integrate
out instead of accumulating. Expect the turn percentiles and every composite to move;
recalibrate and re-rank after, and this is also a real reason to bump the score stage
version (see Error 3).

## Error 3 — stage-version bookkeeping drifted

`score.py` declares `STAGE_VERSION = 3` and `select.py` declares `1`, but nothing reads
either constant: `db.record_stage` writes `config.STAGE_VERSIONS["score"] = 2`. All 97
assets show score v2 in `stage_run`, so if any were scored before later changes to
`compute()`, `stale_stage` cannot flag them — the exact "corpus quietly built from two
extractions" hazard `cmd_score` warns about for telemetry. Make one module the source of
truth, and add a selection entry if clip-selection changes are meant to be tracked at all.

## Gap 1 — no style, mount, or lighting anywhere in the scoring key

The doc's design is a calibration key of availability × lighting × (style, mount), four
weight profiles, and a bikejoring-only Pull sub-score. What exists is availability only —
`calibration.json` shows exactly two buckets, turn+rough (28,157 s) and speed+turn+rough
(21,087 s). There is no classification stage, no `classification` table, and no style or
mount column anywhere in the schema.

Three consequences, in order of bite for bikejoring:

**Speed carries 0.50 of the weight, and on a bikejoring ride speed is the dog's pace.**
0679's speed p95 is 4.2 m/s; a solo sprint is 10+. In a single shared percentile table, a
genuinely hard pull can never rank high on the feature that carries half the composite, so
bikejoring seconds systematically lose to solo seconds at selection time. A per-style
calibration bucket fixes this with zero new sensors — it is the stated design, just not yet
in `calibrate.fit()`'s key.

**The decision log being accumulated now cannot be split by style later** except by ride-id
heuristics. 199 hand-made decisions is real value; a `style` column on the asset row — even
hand-assigned, since you know which rides had Orbit — would let every future fit condition
on it. Cheap now, annoying to reconstruct later; the same argument the doc makes for
building the archive seam early. The same column is where a corrected `lighting` becomes a
calibration key the day there is real night footage.

**The night machinery is entirely unexercised.** Given Error 1, the pipeline has never seen
a true night ride: the night calibration bucket, the illumination gate, and the
retroreflective detection path all remain designs with zero data behind them. The doc's
MEDIUM risk on retroreflective return is still fully open — worth remembering the first time
a real night ride goes through, because nothing will have tested that path before it.

(Pull itself is correctly parked behind vision — the pull_probe negative result is solid
work, and 0603 genuinely cannot be ranked from telemetry. No disagreement there.)

## Gap 2 — GPS5-era rides get no fix or DOP gating

`gps.py` parses only the five spec fields for GPS5 — GPSF/GPSP are never read — so GPS5-only
files have no `gps_fix` column and `score.compute` fails open: pre-lock crawl and canopy
junk would enter `speed_ms` ungated. The comment in `score.py` documents the fail-open
choice, but its premise ("GPS5's fix can be absent altogether") undersells that it is
*never* present from the in-tree parser. Currently moot because the HERO11 writes GPS9 and
the older library has GPS off — but the doc's signal-inventory table says GPS9 is "HERO13+
only" while the code correctly depends on the HERO11 producing it; fix the doc so a future
decision doesn't assume GPS9 is unavailable. Also asymmetric: DOP gates altitude but not
speed.

## Gap 3 — interpolation silently manufactures GPS where none was measured

`telemetry.extract` puts GPS columns onto the 10 Hz grid with `np.interp`, which bridges any
mid-ride dropout linearly and holds first/last values constant beyond coverage — and the
parquet keeps no mask of which grid seconds were actually measured. On the rides checked,
fix = 3.0 throughout and DOP ≈ 3.5, so it hasn't bitten yet; it will on canopy rides the
day GPS dropout becomes real (the doc's remaining HIGH risk). A max-gap guard — NaN out any
interpolated span longer than a few seconds — is one line per column and matches the "make
absence loud" principle the project already paid for twice.

## Smaller items

`select._grow`'s CLIP_MAX clamp only guarantees the peak sits ≥ 1.5 s before the out-point,
so a long plateau can yield a 30 s clip whose peak lands at second 28.5 — cutting right
after the best moment; and the 10–30 s range disagrees with the plan's 7–20 s (biased 8–15)
Reels target. `_peaks` never considers a ride's final second. `_protect_air`'s single pass
can snap an in-point into an adjacent freefall window. `roughness()` runs `sosfiltfilt` over
raw magnitude, so a single NaN row in `imu_raw` (which `_columns_for` emits for a malformed
sample) silently NaNs a whole ride's roughness. Chapters share the recording-start
timestamp, so a late chapter of a long ride can carry a stale twilight/day boundary. And the
1–7 s stub files (0605 ch2, 0607, 0616, 0619, 0655, 0678, 0681) get full lighting labels and
ride rows that pollute per-ride groupings — a minimum-duration quality gate at ingest would
tidy several downstream views at once.

## Suggested order

Fix the clock/lighting error first — it is a data-corruption bug with a cheap backfill, and
every era/lighting analysis of the decision log is misleading until it lands. Fix the turn
sub-score second (it changes every ranking and every future fit; bump the score stage
version properly when you do). Then add the hand-assigned `style` column and the
per-style calibration bucket, which is the single highest-leverage change for bikejoring
clip selection specifically. The GPS5 gating, interpolation guard, and stage-version cleanup
are small and worth doing the same day they're remembered.
