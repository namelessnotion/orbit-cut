# OrbitCut — Architecture Plan

Named for Orbit. Living document; phase 0 is implemented in this repo.

Published artifact: https://claude.ai/code/artifact/70de8ee5-0d26-4fc2-85ae-12aac08fe66b

**Camera: GoPro HERO11 Black. Mounts: chest and helmet only (no bar mount).**
**Compute: M4 Max runs the pipeline; Linux desktop (3700X · 16 GB · RTX 2060 6 GB · big disk) is
the cold archive; ThinkPad is spare.** Wired gigabit between them. Scale: a handful per ride.

Changes from v0.1: camera and mount questions resolved — removes two of three HIGH risks,
simplifies mount classification to a binary, adds an 8:7 capture recommendation, restructures
stage 5 reframing into a four-cell policy.
Changes from v0.2: adds LangGraph topology (control plane vs data plane) and per-node model
selection.
Changes from v0.3: adds full capture settings with pipeline-specific rationale.
Changes from v0.4: adds the night-ride variant (detection, what breaks, night preset).
Changes from v0.5: night = bikejoring with Orbit; retroreflective harness replaces the LED
suggestion; adds the Pull sub-score.
Changes from v0.6: adds the phase-to-machine map — phases 0–4 are single-machine.
Changes from v0.7: originals live on the desktop (Linux, NVIDIA, wired gigabit).
Changes from v0.8: desktop specs are modest (3700X / 16 GB / RTX 2060 6 GB) and originals only
need to be present at ingest and render — so the desktop becomes a **cold archive** and the M4 Max
runs the pipeline. Archive is now an explicit final stage.

---

---

## Phase 0 — measured, not assumed

Validated end to end on GX010598 (628 s, 9.4 GB, HERO11). Everything below is a
measurement from real footage, replacing the corresponding assumption above.

### Streams actually present

| Stream | Rate | Note |
|---|---|---|
| ACCL | 201.3 Hz | full rate — the freefall detector gets ~20 samples per 100 ms window |
| GYRO | 201.3 Hz | full rate |
| GRAV, CORI, IORI | 30.0 Hz | locked to frame rate, as documented for HERO8+ |
| ISOE, SHUT | 30.0 Hz | exposure response — the no-GPS day/night fallback |
| WBAL | 10.0 Hz | |
| TMPC | 1.0 Hz | |
| **GPS5** | **absent** | **GPS is switched off in the camera** |

`mean |accel|` 10.72 m/s², `mean |gravity|` 1.000 — both healthy. Mean |accel|
sits above 9.81 on any real ride because vibration adds to gravity.

### Axis mapping (chest mount, camera inverted)

```
mean gravity   (+0.02, -0.93, +0.22)
```

`grav_1` vertical (negative = inverted mount), `grav_2` fore–aft — the steady
+0.22 is a constant ~13° downward tilt, the deliberate chest-mount angle —
`grav_0` lateral, near zero because lean averages out. This closes the axis
question the plan left open; magnitude-based features never needed it, but the
mount classifier does.

### Horizon Lock: not enabled

Roll suppression **+0.11** — gravity tilts 13.2° in the body and 11.7° on
screen, so the camera removes about a tenth. That is HyperSmooth's incidental
rotation, not levelling. Post-levelling applies to this footage.

Two earlier metrics were wrong and the reasons are recorded in the README: an
`IORI`-magnitude test cannot separate levelling from HyperSmooth, and a
quaternion-spread test is swamped by yaw. Gravity is immune to both problems.

Note also that 13.2° of body tilt is *modest* for mountain biking, and that is
the chest mount behaving as predicted: the torso stays upright while the bike
leans underneath it.

### Two things the plan did not anticipate

- **telemetrik cannot read files over 4 GB.** It assumes 32-bit MP4 box sizes
  and `stco` chunk offsets; large files use 64-bit boxes and `co64`. Patched in
  `orbitcut/gpmf_compat.py`. This affects essentially every long ride, so it is
  not an edge case — worth sending upstream.
- **A 180° display matrix.** The camera tags the inverted mount, ffmpeg honours
  it, and proxies come out the right way up — confirmed by eye on the contact
  sheet. But telemetry stays in camera coordinates, so phase 5 must apply
  `asset.rotation` before mapping a sensor direction onto a screen direction.

### Throughput

Proxy generation runs at roughly **5x realtime** on the M4 Max with
VideoToolbox: 628 s of 10-bit HEVC in 120 s. A twenty-minute ride is about four
minutes. Batch ingest is an over-coffee job, not an overnight one — which
further weakens the case for ever distributing this across three machines.

### Action items

1. **Turn GPS on in the camera.** It costs the speed sub-score, trail
   identification, and sun-elevation day/night. Roughness, airtime and turn rate
   are unaffected — all IMU.
2. Existing footage is **16:9 5.3K30**, predating the 8:7 switch. Telemetry is
   unaffected; only the vertical crop is worse on the back catalogue.
3. The exposure-based day/night fallback read "twilight" for a 19:51 EDT ride in
   late June — full daylight under dense canopy. It cannot separate shade from
   dusk. Another reason GPS matters.


## The core bet: the camera already measured the action

Every GoPro from HERO5 on writes a GPMF telemetry track: accelerometer ~200 Hz, gyro
200–3200 Hz, GPS, and on HERO8+ per-frame camera-orientation quaternions (CORI/IORI) and a
gravity vector (GRAV). Speed, cornering, chatter and airtime are **recorded, not inferred**.

Consequence: the usually-expensive "how exciting is this moment" problem collapses into
signal processing that runs faster than real time on CPU. Vision is reserved for the four
questions sensors cannot answer: is this mountain biking, is the dog there, where is the
trail in frame, does the clip look good.

**Fused-signal rule.** Telemetry is primary for anything physical. Vision is primary for
anything semantic. Where both have an opinion they cross-check; disagreement is a review
flag, not something to average away.

## Signal inventory

The HERO11 is close to the best case for this design — the last generation with `GPS5` before
the HERO12 dropped GPS entirely, and new enough to carry the HERO8+ orientation streams.
Every signal the pipeline wants is present.

| Stream | Rate | Carries | HERO11 |
|---|---|---|---|
| ACCL | ~200 Hz | roughness, impacts, **freefall/airtime**, braking | yes |
| GYRO | 200–3200 Hz | yaw rate → turn count, cornering intensity | yes |
| GPS5 | 18 Hz | speed, gradient, route → trail ID | yes |
| GRAV | frame rate | **mount inference**, horizon leveling | yes |
| CORI | frame rate | camera orientation in world → **head-turn compensation** | yes |
| IORI | frame rate | image orientation vs camera body → **what the camera already corrected** | yes |
| SHUT/ISOE | frame rate | low-light / motion-blur quality gates | yes |
| Frames | 30–120 fps | subject, scene, dog, trail region, aesthetics | yes |
| GPS9 | 10 Hz | higher-precision GPS w/ per-sample timing | HERO13+ only |

### Shoot 8:7 — the single biggest quality decision in this pipeline

The HERO11's 1/1.9" sensor is natively **8:7**, 13% taller than the previous generation, built
specifically so you can crop the sides off for vertical without losing height.

| Capture mode | Source | 9:16 crop | Pan range | What you see |
|---|---|---|---|---|
| **5.3K30 8:7** | 5312×4648 | 2614×4648 | 2698 px | Full sensor height — trail ahead, sky, and the bike. 2.4× oversampled into 1080×1920 |
| **4K60 8:7** | 3956×3460 | 1946×3460 | 2010 px | Same framing at 60 fps — the better everyday mode for MTB |
| 5.3K60 16:9 | 5312×2988 | 1681×2988 | 3631 px | A narrow vertical slice of a wide view. More pan freedom, worse composition |

Shooting 16:9 and cropping to vertical throws away the part of the frame that makes MTB POV
read well — the trail disappearing ahead and the terrain immediately in front of the wheel.
**4K60 8:7 is the recommended default**: 60 fps gives slow-motion headroom on jump landings,
and the crop is still comfortably above 1080 wide.

### Detect Horizon Lock from metadata, never from the setting

HyperSmooth 5.0 offers 360° Horizon Lock, and whether it was on changes stage 5 — applying
your own leveling on top of already-leveled footage double-corrects and looks worse than doing
nothing. Don't read it from capture settings, which vary by FOV and mode. **`IORI` is the
answer**: it records the image's orientation relative to the camera body, which is exactly the
correction the camera applied. If `IORI` tracks the inverse of `CORI`'s roll, the camera
leveled it; if `IORI` is near-identity, it didn't. One check at ingest, robust to any mode.

Related good news: for HERO8–HERO13, gyro data stays valid with HyperSmooth enabled — the same
assumption Gyroflow relies on. You do not need to shoot with stabilization off to keep the
telemetry trustworthy. (This downgrades a MEDIUM risk from v0.1.)

## Capture settings

Most GoPro settings guides optimize for how footage looks straight out of camera. Two of these
deliberately diverge, because footage here is also *input to algorithms* — and a setting that
makes a clip look marginally better while making frame geometry unpredictable is a bad trade.

```
VIDEO
  Resolution      4K              8:7 caps at 5.3K30 / 4K60
  Aspect ratio    8:7             already set
  Frame rate      60              slow-mo headroom on landings
  Digital lens    Wide            only option in 8:7 — and the right one
  HyperSmooth     On          <-- standard, NOT AutoBoost
  Bit rate        High            ~120 Mbps; matters more here than usual
  10-bit          On              supported at 8:7 60p and below

PROTUNE
  Color           Natural     <-- NOT Flat
  White balance   5500K       <-- 4000K under heavy canopy. Never Auto
  ISO min         100
  ISO max         1600            then measure it and tighten
  Shutter         Auto            no ND filter, so no fixed shutter
  EV comp         -0.5            protect highlights through the canopy
  Sharpness       Low             sharpen last, after the crop
  Wind filter     Auto

NOT AVAILABLE IN 8:7
  Horizon Lock    --              needs Linear + Horizon Lock lens. Level in post
```

### Why these, specifically for this pipeline

| Setting | Value | Pipeline-specific reason |
|---|---|---|
| **Bit rate** | High | Your delivered pixels come from about **half the frame width**. A 9:16 crop out of 8:7 is effectively viewing the source magnified, so compression artifacts invisible in a full-frame export become visible in the vertical crop. The one setting where "High" is not a nicety |
| **10-bit** | On | Forest riding is a shadow-gradient nightmare — dappled light under canopy is exactly where 8-bit bands. M4 Max decodes 10-bit HEVC in hardware, so cost is ~zero. Pin color primaries/transfer explicitly in the render command |
| **ISO max** | 1600 | Capping lower pushes the camera to lengthen shutter instead, and **motion blur hurts the dog detector more than noise does** — noise partly denoises out, blur is gone forever. Set 1600, then read the `ISOE` stream across your library: if you never exceed 400, tighten it. Let telemetry answer this |
| **Shutter** | Auto | The 180° rule wants 1/120 at 60 fps, but with no ND a fixed shutter clips every time you exit tree cover into sun. Auto + a blur gate derived from the `SHUT` stream you already record |
| **EV comp** | −0.5 | GoPro meters for the average and blows the sky gaps. Shadows come back, clipped highlights don't. −1.0 on hard-sun days with open canopy |
| **Sharpness** | Low | GoPro's sharpening halos compress badly, wasting the bitrate you just paid for, and the crop magnifies them. Sharpen at the end of the render chain — after crop and scale, the correct order anyway |
| **Digital lens** | Wide | 8:7 gives no choice, which is lucky: HyperView/SuperView apply a nonlinear horizontal stretch that would wreck the optical-flow focus-of-expansion and make crop geometry non-uniform |

### HyperSmooth On, not AutoBoost — a fixed crop is worth more than a smooth one

Every stabilization level buys smoothness by cropping in: standard ≈10% of the frame, Boost
15–20%. **AutoBoost varies that crop dynamically**, expanding and contracting with detected
motion. Clever for handheld. For this pipeline it breaks three things at once:

- The **counter-steer** for helmet head-turn maps a yaw *angle* to a pixel offset. That mapping
  depends on FOV — if FOV drifts mid-clip, the correction is wrong by a time-varying factor.
- The **optical-flow speed cross-check** reads flow magnitude, which scales with FOV. A varying
  crop injects a fake speed modulation into a signal used to sanity-check GPS.
- The **crop-path optimizer** needs to know how far it can pan before hitting the frame edge.
  With AutoBoost that limit is a moving target.

Standard HyperSmooth is a known fixed constant you calibrate once. If a clip is too shaky, fix
it in post — where you have the full frame *and* the gyro track, strictly more information than
the camera had in the moment.

### Natural, not Flat — the vision models were not trained on log footage

Standard advice is to shoot Flat for grading latitude. Two reasons to skip it. First, Flat
requires a grade on every clip — manual work that doesn't automate, and this system exists to
avoid manual passes. Second and more interesting: **flat, desaturated, low-contrast footage sits
outside the distribution SigLIP, CLIP, and detection models were trained on**, all built on
ordinary web imagery. Your dog detector will measurably underperform on log footage, worst
exactly where you need it most — small, distant, motion-blurred subject.

If you do want to grade, the clean fix is to **apply a normalizing LUT when generating the
proxy**, so vision models always see conventional contrast regardless of capture profile, while
the render step works from the original. Decouples the aesthetic choice from the algorithmic
one — but a phase-4 refinement, not something to build now.

### Lock white balance

Auto WB drifts as you pass in and out of tree cover, sometimes *within a single clip*. Three
costs: clips from one ride won't match when posted together; the drift is a nuisance variable
for every vision model; and any "conditions" classification becomes unreliable because the
camera is compensating for the thing you're trying to detect. 5500K for open daylight, 4000K
for heavy canopy, switched at the trailhead.

### Two consequences for the design

- **The `IORI` horizon-lock check becomes a guard, not a branch.** Horizon Lock needs the
  Linear + Horizon Lock digital lens, which 8:7 does not offer — so in normal shooting the
  answer is always "not leveled," and post leveling always applies. Keep the check anyway: it
  costs nothing and protects you the day you shoot a 16:9 clip and forget.
- **High bit rate needs the card to keep up.** 120 Mbps wants V30 minimum, V60 if you also
  shoot 5.3K. A card that can't sustain it fails as dropped frames mid-ride.

## Stage 1 — Catalog

| Question | How |
|---|---|
| Is it mountain biking? | Telemetry signature (5–40 km/h sustained, high-freq vertical accel, GPS off road network) + VLM over 8–12 sampled frames. Two agreeing signals pass; disagreement flags. |
| Solo or bikejoring? | Dog detector over sampled frames; >~30% of frames containing a dog → bikejoring. Telemetry corroborates (steadier speed, less pedaling cadence, flatter). |
| Long enough / enough action? | Duration from container. "Enough action" is a threshold on stage 2's output — don't hard-code a guess here. |
| **Chest or helmet?** | Now a **binary**, and one feature nails it. Compute camera yaw rate from `CORI` and heading rate from GPS, then correlate over the file. **A chest mount is rigidly coupled to direction of travel** (torso points where the bike points) → high correlation. **A helmet is not** — you look into corners, check your line, glance at the dog → correlation drops, and there is yaw energy at frequencies the bike never produces. Secondary confirmations, both free: mean `GRAV` pitch (chest sits pitched down and sees bars/front wheel; helmet looks level and ahead) and a single VLM frame check for handlebars in the lower third. |
| Which trail? | Map-match GPS polyline against a local trail cache (OSM `highway=path` + `mtb:scale` via Overpass, optionally merged with Trailforks GPX). Score on Fréchet distance + direction agreement. Degrade to `trail: unknown` with manual assign in the review UI. |

Also record cheap quality gates: mean luma, lens-obscured detection (frame-difference floor),
audio wind level. These down-weight rather than reject, and surface as badges in review.

## Stage 2 — Score

Five sub-scores on a 3 s sliding window, 1 s hop → a 1 Hz time series. Normalize each to 0–1
by percentile rank against a **corpus-wide calibration table**, not against the file itself.

| Sub-score | Source | Method | w₀ |
|---|---|---|---|
| Speed | GPS | 2D ground speed, gated on GPS9 fix and DOP | **0.50** |
| Turns | GYRO/CORI | Yaw rate in world frame after gravity alignment. Turn events = threshold-crossing zero-crossings. Weight *lateral acceleration* (v × yaw rate) | 0.25 |
| Roughness | ACCL | RMS of the 5–40 Hz band after removing gravity. Chest and helmet are both body-damped so the correction between them is small — but still calibrate: a helmet reads slightly higher on impacts and slightly lower on sustained chatter | 0.18 |
| Airtime | ACCL | Freefall detector: \|accel\| within ~0.15 g of zero for >100 ms, then a landing spike. Duration of the null window *is* the airtime, measured | 0.27 |
| Descent | GPS+GRAV | **Disabled — 0.00.** GoPro altitude cannot support it; see the finding below. Revisit when map-matching can supply trail elevation | 0.00 |
| **Pull** (bikejoring only) | detector box | Orbit's apparent size gives lead distance. A taut, stretched-out line reads as effort the accelerometer never sees | 0.22* |
| Flow (cross-check) | proxy frames | Optical-flow magnitude on downscaled frames. Fallback when GPS is dead; also catches "fast through tight trees" | cross |

The **airtime detector is the cheapest high-value feature in the system** — an unmistakable
accel signature yielding a measurement (0.61 s of air), not a guess.

\* The five base weights sum to 1.00; `Pull` is style-conditional, entering only the two
bikejoring profiles, where the vector renormalizes. Solo profiles never see it.

**Four weight profiles**, one per `(style, mount)` pair: solo/chest, solo/helmet,
bikejoring/chest, bikejoring/helmet. Bikejoring weights speed and steadiness up and roughness
down; a helmet descent weights turns and airtime up. Four is small enough that each profile
will accumulate enough labeled decisions to actually fit.

### Measured — a stream list is not a feature

`inventory` reported GPS present on 26 of 36 files and it was right: the camera wrote GPS5 and
the parser returned it. Every one of those rides still scored on roughness and turn rate alone,
because the extractor named the GPS columns positionally. GoPro's GPS5 is five fields; the
parser in use appends the sticky GPSF and GPSP values, so what arrives is seven. The
named-column table listed five, the arity check failed, and the documented positional fallback —
correct for ACCL and GYRO, whose axis order genuinely varies — renamed `gps_speed2d` to
`gps5_3`. Nothing raised. Speed, descent and cornering force were `NaN` for the entire library
for as long as it existed.

Three separate defences existed and none of them fired, which is the part worth keeping:

- `inventory` checked the *stream list*, one layer above the columns.
- `calibrate` skipped features with too few samples, so a feature that was empty everywhere
  vanished from its own report instead of appearing as a zero.
- `score` renormalized the composite over the features present, so the arithmetic stayed
  well-formed all the way to a plausible-looking ranking.

Each is reasonable alone. Together they turn a naming mismatch into a silent capability loss.
The corrections are cheap and all of the same kind — **make absence loud**: the extractor warns
when a named stream's arity is unrecognised instead of falling back quietly, `calibrate` lists
what it could not use and why, and `verify` says outright when a GPS stream is present but its
columns are not. The general form: a fallback that is right for one stream must not be the
default for every stream, and a pipeline that degrades gracefully has to say that it degraded.

### Measured — calibration must also bucket on *feature availability*

Renormalizing the weighted mean over "whatever sub-scores are present" is the obvious way to
handle a ride shot without GPS, and it is quietly wrong. A mean of two terms is more variable
than a mean of four, so it reaches high values more often even when every underlying feature is
distributed identically. This is not a small effect and it is not theoretical: the first real
run of `orbitcut rank` over 28 rides put **all six** GPS-less rides in the top six places, and a
simulation with no real difference between the rides at all reproduced exactly that — six for
six, from features with identical distributions.

The fix is a second ranking. After the level is computed it is percentile-ranked against other
seconds *carrying the same set of features*, which turns the question from "how high is this
average" into "how good is this for what we can measure here" — a question that is comparable
across rides. In the same simulation the gap between the two groups falls from +0.058 to +0.003
and the top six goes to 3/6, which is what chance looks like. Within a ride the change is
invisible (Spearman ≈0.998 against the old curve), because it only ever mattered *between*
rides.

The general rule this is an instance of: **any calibration key that varies in how much
information it carries needs its own distribution, not a renormalized share of a common one.**
Availability is the first such axis; `lighting` (below) is the second; `(style, mount)` is the
third. They compose into one key, and a bucket too thin to have a distribution falls back to the
global one rather than pretending.

### Measured — the composite must not average across features

The weighted arithmetic mean was wrong, and wrong in a way that produced a
plausible-looking ranking rather than an obvious failure. Averaging punishes
specialisation, and **every exciting second is specialised**: fast means
straight, so a sprint scores near zero on turns; twisty means slow; a rock
garden is neither fast nor flowing. Scored under the mean:

| second | speed | turn | rough | composite |
|---|---|---|---|---|
| sprint | 0.99 | 0.20 | 0.55 | 0.546 |
| tight switchbacks | 0.30 | 0.97 | 0.50 | 0.626 |
| rock garden | 0.25 | 0.45 | 0.98 | 0.542 |
| **briskly consistent, never special** | 0.85 | 0.85 | 0.85 | **0.650** |

All three memorable seconds lose to the forgettable one. On the real library
this showed up exactly as "the top rides are the most consistently paced ones
but may not contain a real sprint" — which is how it was caught, by watching
footage against the ranking rather than by any test.

The composite is now a **power mean**, `(Σ w·s^p / Σ w)^(1/p)`: p = 1 is the old
arithmetic mean, p → ∞ is the maximum, and `SHARPNESS` sets it. This is the same
principle `AIR_GAIN` already encodes for airtime — a rare event and a sustained
level are different kinds of evidence and must not be averaged — applied one
level up, across features instead of across time.

Two further consequences worth keeping:

- **Sharpness cannot repair a weighting that disagrees with the taste.** A
  sprint peaks on speed alone, so while speed carried the smallest weight and
  turn the largest, no value of p promoted sprint rides. Weight and sharpness
  are separate knobs and both are taste. Settled at `speed 0.50 / turn 0.30 /
  rough 0.20`, `SHARPNESS = 8`.
- **"Best clip" and "most good footage" are different questions.** `rank` used
  the mean of a ride's 30 best seconds taken from *anywhere* in it, which
  measures total good footage and favours long even rides. It now also reports
  the best *contiguous* 12 s — the question stage 3 will actually ask — and
  sorts on that. Both columns are shown, because where they disagree is
  informative.

## Stage 3 — Cut

Naïve top-N gives five near-identical clips from one rowdy section. Three constraints beyond
"high score":

- **Anticipation lead-in** — start ~1.5 s before the action; the run-in is what makes a jump land.
- **Non-maximum suppression** — minimum 8–10 s gap between selected peaks.
- **Diversity bonus** — tag each candidate by dominant sub-score (turn / jump / rough / speed)
  and penalize repeats of a type.

Target 7–20 s, biased to 8–15 s. Hard constraints: never cut inside a detected freefall window;
prefer in-points where the composite is rising and out-points where it's falling.

Each candidate carries its full feature vector forward — cheap now, expensive to reconstruct later.

## Stage 4 — Confirm (the only place the system learns)

Local web app on the coordinator. Grid of candidates, each looping its proxy segment with score
breakdown and telemetry sparkline on the scrub bar. Approve / reject / adjust in-out, plus
trail-assign for failed GPS matches.

**Log every decision as training data.** Every approve, reject, and especially every in/out
adjustment is a labeled example: feature vector → your taste. After ~100–150 decisions a
logistic regression or small GBM on that log beats any hand-tuned weights and keeps improving.
Adjustments teach the boundary policy, not just the ranking. Building the logging costs an
afternoon; skipping it means the system never improves past day one.

Worth doing early: keyboard-only review (J/K move, A/X decide) and a reject-reason chip set
(too shaky / bad light / boring / already have one like it).

## Stage 5 — Render

With the mount narrowed to chest or helmet, daytime reframing is a four-cell policy — and the cells
genuinely differ, because a chest mount points where the bike points while a helmet points
where you look. **Night is a fifth cell handled separately**, since illumination replaces both
gaze and subject detection as the framing signal.

| | Chest | Helmet |
|---|---|---|
| **Solo** | **Near-static crop.** Torso already tracks the bike, so the trail sits near frame center. Hold centered with a slow drift toward the optical-flow focus of expansion. Least work, best result. | **Counter-steer to heading.** Head turns swing the frame off direction of travel. Use `CORI` yaw relative to GPS heading and pan the crop *against* the head turn, so the frame stays on where the bike is going. |
| **Bikejoring** | **Track the dog.** The dog is ahead and low and drifts laterally; a chest mount won't follow it. Detect per sampled frame, track between detections, drive crop x-center from the box. A box tracker is enough — you need a center, not a mask. | **Follow the gaze — do NOT counter-steer.** You are already looking at the dog, so head yaw *is* the framing signal. The solo/helmet correction would actively fight you and push the dog out of frame. Track the dog as a cross-check, override only when the gaze loses it. |

Helmet + bikejoring is the cell that breaks a naive "always counter-steer head turn"
implementation. Encode the policy as an explicit lookup on `(style, mount)` rather than a chain
of conditionals — the four cells are stable, and a table makes the inversion visible.

**Crop-path smoothing is where auto-reframe lives or dies.** Because rendering is offline, solve
the whole clip at once rather than filtering causally: minimize (distance from target) + (crop
velocity) + (crop acceleration), with a **dead zone** where small target movement produces no
camera movement. Reads like a human operator — hold, smooth pan, hold. Small QP per clip,
milliseconds. Shooting 8:7 gives ~2000–2700 px of pan range with the crop still oversampled,
which is exactly the condition this needs to produce natural motion.

**Horizon leveling is nearly free — but only apply it once.** Gate it on the `IORI` check: if
the camera's Horizon Lock already leveled the frame, skip it. What remains is a per-clip
toggle — on some rowdy descents the tilt *is* the point.

| Output setting | Value | Why |
|---|---|---|
| Source mode | 4K60 8:7 | Recommended capture default |
| Resolution | 1080×1920 | 9:16, Reels native |
| Container/codec | MP4 · H.264 High | Widest upload compatibility |
| Frame rate | 30 fps | Conform 60 fps source down; keep 60 only for slow-motion sections |
| Bitrate | 12–15 Mbps | IG re-encodes toward ~3.5 — give the transcoder headroom |
| Audio | AAC 128k | GoPro audio is wind. Plan on music, keep an ambient bed low |
| Duration | 7–20 s | Within the 3 s–3 min window; short end is where retention holds |
| Safe area | ~250 px top / ~420 px bottom | Clear of caption and action-button overlays |
| Decode path | VideoToolbox | Hardware HEVC decode on the M4 Max |

## Variant — night rides

Setup: two lights, one bar-mounted (fixed, points where the bike points) and one helmet-mounted
(moves with your head). **Camera is always chest-mounted at night**, since the helmet carries a
light. **Night rides are usually bikejoring** — Orbit comes along. His harness is orange with
retroreflective material, which turns out to matter a great deal (below).

Night is not a label to hang on a clip — it's a different operating regime, and about a third of
the pipeline behaves differently under it. It's also where the telemetry-first bet pays its
largest dividend: **five of six sub-scores don't care that it's dark.** An accelerometer measures
a rock garden identically at noon and at ten at night, so night footage ranks as reliably as day
footage — not true of any vision-first design.

### Detecting it — three signals, no model

| Signal | Source | Method |
|---|---|---|
| **Solar elevation** | GPS + clock | You have lat, lon, and a UTC timestamp in the GPS stream, and solar position is a closed-form function of those three. `astral.sun.elevation()` gives the sun's angle above the horizon: >0° day, 0 to −6° civil twilight, <−6° night. Exact, free, correct at every latitude and season — which "was it after 8pm" is not |
| **Exposure response** | ISOE + SHUT | The camera's own reaction to the scene. On a night ride ISO sits pinned near its ceiling and shutter at its longest. The camera telling you how dark it was, for free |
| **Mean luma** | proxy frames | Sampled frame brightness, plus its *variance across the frame* — a lit trail at night is a bright pool inside a black surround, a histogram nothing in daylight resembles |

Fuse to `lighting ∈ {day, twilight, night}` on the classification record. Disagreement is the
interesting output: solar elevation saying *day* while ISO sits pinned means dense canopy, a
muddy lens, or a camera that spent the ride in a pack.

### What survives the dark, and what breaks

| Component | At night | Handling |
|---|---|---|
| Speed / turns / roughness / airtime / descent | **unaffected** | Sensors don't care about photons. No change |
| Optical-flow cross-check | **DISABLE** | Not down-weight — switch off. A helmet light sweeping across terrain produces apparent motion the flow estimator reads as real. That's a systematic error correlated with head movement, not noise, and averaging it in corrupts the signal you use to sanity-check GPS |
| Is-it-MTB classification | **discount** | SigLIP embeddings on a near-black frame are unreliable. Telemetry already answers this; skip the vision vote rather than trusting a low-confidence one |
| Dog detection | **pool only** | Works inside the illuminated region and nowhere else. See harness light below |
| Mount classification | **prior** | Always chest at night. Use as a strong prior — but keep running the classifier so the day you change setup the system notices instead of silently mislabeling |
| Style classification | **prior** | Night usually means Orbit is along, so bikejoring is the prior. Prior, not hard-code — the retroreflective test confirms it directly and cheaply |

### The helmet light is a gaze signal you can see in frame

Your chest camera is fixed to direction of travel, and the bar light illuminates that same
direction. But the **helmet light moves with your head**, so the chest camera records a bright
pool tracking exactly where you're looking. That's the same gaze information the daytime
helmet-mount case gets from `CORI` — except here it's *visible in the image*, on a camera with
no head-turn problem of its own.

So the night reframing target isn't a model output. It's a luma threshold and a centroid: find
the bright region, weight by brightness, crop toward it. A few lines of numpy per frame. And
it's doubly correct — the illuminated region is also the only part of the frame with usable
detail, so cropping toward the light is simultaneously cropping toward the subject and away
from the noise.

Separating the two pools is optional but cheap: the bar-light pool is **temporally stable**
relative to the frame, the helmet pool isn't. A rolling median over a couple of seconds isolates
the fixed pool; what's left is your gaze.

### Three changes to the pipeline proper

- **Night gets its own calibration bucket.** Add `lighting` to the calibration key alongside
  style and mount. Night riding is slower and more cautious by nature; percentile-normalizing
  night footage against a day-dominated corpus means *no night clip ever clears the selection
  threshold*. Silent failure — the pipeline would appear to work while quietly deciding you
  never ride well after dark.
- **Clip selection gains an illumination gate — the first time vision overrules telemetry.** A
  clip whose peak action happens outside the light pool is unpublishable no matter what the
  accelerometer says. Measure mean luma in the central region across the candidate window and
  veto below a threshold. Everywhere else telemetry wins ties; here it doesn't, because the
  constraint isn't "was this exciting" but "can anyone see it."
- **Render adds a denoise pass before encode.** Night footage is noisy, noise is incompressible,
  and the bitrate it consumes is stolen from detail. Light temporal denoise ahead of the H.264
  encode. Consider a fine dither too — Instagram's re-encode bands badly in the smooth falloff
  around a light pool, and a touch of grain breaks it up.

### The night capture preset

HERO11 stores custom presets — build one, switch at the trailhead. **Change only the
exposure-related Protune values; keep the capture mode identical.** Resolution, aspect ratio,
frame rate, lens, and stabilization all feed frame geometry the pipeline calibrates against.

```
NIGHT PRESET — deltas from day
  Shutter         1/120       <-- FIXED. This is the inversion, see below
  ISO max         3200        <-- up from 1600; bright lights mean it rarely binds
  EV comp         -1.0        <-- the meter sees black and overexposes the pool
  White balance   ~5000K      <-- match your lights. Verify once, then lock

UNCHANGED — deliberately
  Mode            4K60 8:7        same geometry, same calibration
  HyperSmooth     On              still not AutoBoost
  Bit rate        High
  10-bit          On              earns more here than by day
  Color           Natural
  Sharpness       Low             sharpening amplifies noise
```

**Shutter inverts: Auto by day, fixed at night.** The daytime argument for Auto was highlight
clipping — with no ND, a fixed shutter blows out every time you exit tree cover into sun. At
night there is no sun, so that argument disappears entirely, and the opposite concern takes
over: the meter, seeing mostly black, drags exposure to its longest and smears the one part of
the frame that matters. Fixing at 1/120 caps motion blur, gives HyperSmooth clean frames, and
lets ISO absorb the difference — affordable because you aren't shooting ambient darkness, you're
shooting a well-lit trail surrounded by darkness.

Worth knowing either way: at 60 fps exposure cannot exceed 1/60 regardless of settings, so blur
is bounded by frame rate. A quiet argument for staying at 60 rather than dropping to 30 for more
light — 30 fps would double worst-case blur to buy one stop, and here **blur costs more than
noise**, because noise partly denoises out and blur never comes back.

### Orbit's harness already solves night detection — retract the LED

v0.5 recommended adding a coloured LED to the harness. Unnecessary: **orange plus retroreflective
material, lit by two lights mounted within centimetres of the camera, is a better beacon than any
LED you could add.** Retroreflection returns light along the axis it arrived on, so a bar light
and a helmet light near the chest camera's line of sight get that light thrown straight back at
the sensor. Against a near-black frame, the harness should be the brightest saturated object by
a wide margin.

Night detection is therefore a threshold, not a model — specifically a **two-signal threshold**:
a high-luma blob *with* orange chroma at its edges. Match on both rather than hue alone, because
near-coaxial retroreflection is intense enough that the blob's core will often clip to white and
lose its colour entirely. The orange survives in the fringe. Bright core + orange fringe is far
more specific than either test alone, and it's a handful of numpy ops per frame.

**The geometry has a lucky failure mode.** Observation angle is set by light-to-camera separation
divided by distance to target, so it *widens as Orbit closes and narrows as he pulls ahead*:
roughly 5° at a 5 m lead, roughly 15° at 1.5 m. The return is strongest exactly when he is
stretched out in front — which is the shot worth keeping.

**Orange is a night asset and a daytime liability.** The same colour that makes Orbit trivially
findable at night is a poor daytime cue in Michigan — autumn foliage, clay, and trail markers all
live in the orange band. So the detection node runs **two mechanisms behind one interface,
switched on `lighting`**: fine-tuned RF-DETR by day, retroreflective threshold at night. Cleaner
than one model straining across both regimes, and the night path needs no training data at all.

### Night collapses the reframe matrix — and adds a sub-score

If night rides are usually bikejoring, night is effectively **one cell**: chest mount,
bikejoring, dog trackable. The policy is *crop toward Orbit's harness, fall back to the
light-pool centroid when he's out of the beam or occluded* — and those two targets agree most of
the time, since he's generally in the light you're pointing at the trail.

More interesting: the harness box gives a **lead-distance signal**. Apparent height in pixels
scales inversely with distance, so tracking it tells you whether Orbit is stretched out on a taut
line or loping along close in. That's a bikejoring-specific axis of "action" none of the existing
sub-scores capture — a hard pull with the dog well out front looks completely different from the
same speed with a slack line. Worth adding as a **style-conditional sixth sub-score** (`Pull`),
active only when `style = bikejoring`, and it works by day too once the detector produces boxes.
Use apparent *size* not brightness for the distance estimate: brightness confounds with the
observation-angle effect above and would double-count.

## Where agents actually belong

| Kind | Which work | Why |
|---|---|---|
| **Deterministic** | Telemetry parse, all sub-scores, clip selection, crop-path optimization, ffmpeg render, map matching, **chest/helmet classification** | Testable, reproducible, free, fast |
| **Model call** | Is-it-MTB, dog detection, handlebar frame check, trail-region segmentation | Bounded, single-purpose, cached by content hash. Function calls that use a model — no tool loop, no autonomy |
| **True agent** | **Triage supervisor** (reconcile contradicting signals, escalate vs proceed), **caption writer**, **librarian** (natural-language queries over the catalog) | Judgment over ambiguous open-ended input; benefits from tool use and multi-step reasoning |

The failure mode of this kind of project is wrapping deterministic signal processing in an LLM
and inheriting non-determinism, cost and latency for nothing.

The librarian is the sleeper: "find every clip with over half a second of air on Blueberry Hill
where the dog is in frame and it wasn't raining" becomes answerable.

## LangGraph topology

LangGraph is the right tool for a specific slice of this system and the wrong tool for most of
it. That boundary matters more than any model choice, because the failure mode is expensive:
**LangGraph re-executes nodes on resume**, so a node that spends forty minutes running ffmpeg
is both a poor fit for the runtime and a duplicate-work hazard the moment anything interrupts it.

Split into two planes:

- **LangGraph owns the control plane** — per-asset state machine, routing, retries, the human
  gate, and the three genuinely agentic nodes.
- **Celery owns the data plane** — every heavy, long-running, deterministic job. A LangGraph
  node dispatches a Celery task and awaits the result, keeping the node cheap and re-runnable.

Five of the graph's nodes call no model at all — they dispatch deterministic work and wait.
That's the shape you want: **the LLM appears three times in a run, not three thousand.**

Mechanics:

- **The human gate is LangGraph's best feature here.** `interrupt()` + a Postgres checkpointer
  *is* stage 4: the thread pauses mid-graph, persists, and survives a reboot or a week of not
  getting to it. The review UI resumes with `Command(resume=…)`. You already have Postgres —
  use `AsyncPostgresSaver`, don't stand up anything new.
- **Set `durability="sync"` and stop thinking about it.** The three modes (`exit` / `async` /
  `sync`) trade safety for speed; at a handful of files per ride the speed side is worth
  nothing. A node that dispatches a render has a real-world side effect.
- **Nodes re-execute on resume, including LLM calls.** Every dispatching node must be
  idempotent. The data model already does this: key each dispatch on
  `(asset_id, stage, stage_version)` and have the Celery task return the existing result rather
  than recomputing. The idempotency designed for reprocessing is exactly what durable execution
  needs.
- **One graph per asset, `thread_id = content_hash`.** Independent threads, trivial replay, and
  a re-ingested file resumes rather than duplicating.
- **Structured output on every model node.** Pydantic schemas, not prose parsing — triage
  returns a typed verdict with confidence and reason, not a paragraph you regex.
- **12–15 nodes total. Never one node per frame.** A node is "run detection on this clip" (one
  Celery job), not "detect on frame 1", "frame 2", …

## Model per node

Organizing principle: **vision is high-volume and low-judgment; reasoning is low-volume and
high-judgment.** Vision runs locally on models you own and can fine-tune. Reasoning goes to an
API where quality matters and call count is small enough that cost never becomes a factor.

| Node | Model | Runs on | Why this one |
|---|---|---|---|
| **ingest** / parse / proxy | none | M4 Max | ffmpeg + a GPMF parser. No model belongs here |
| **mount** chest/helmet | none | M4 Max | Telemetry correlation. A model here would be slower *and* worse |
| **scene classify** (is-it-MTB, conditions, terrain) | **SigLIP 2 + logistic head**; bootstrap with **Qwen3-VL-8B** (MLX, 4-bit, ~6 GB) | M4 Max | Embed once, classify with a head trained on your own frames — ~10 ms/frame, deterministic, free. Use the VLM as the *labeler* for the first ~200 frames, then retire it from the hot path; keep it for open-ended questions |
| **dog detect** | **RF-DETR Nano or Small, fine-tuned**; bootstrap with **Grounding DINO** or **Moondream 3** | M4 Max (train: MPS or the 2060) | You have *one* dog. A fine-tune on 200 frames of that dog will crush a generic COCO "dog" class at distance, in motion blur, half-occluded by brush — i.e. every frame that matters. RF-DETR is Apache 2.0 and transfers well from few labels; YOLO26 is faster on CPU but AGPL-3.0 |
| **track** | **ByteTrack** (not SAM 2) | M4 Max | You need a crop center, not a mask. Segmentation is a large bill for information you throw away |
| **score** / **cut** | none | M4 Max | numpy + scipy. Later a gradient-boosted tree on the decision log — still not an LLM |
| **triage supervisor** | **Claude Sonnet 5** (`claude-sonnet-5`); Haiku 4.5 if volume grows | API | Judgment over contradicting structured evidence plus a few frames. Once per file |
| **caption writer** | **Claude Sonnet 5** | API | Once per approved clip. Few-shot with your own past captions — the job is your voice, not general fluency |
| **librarian** | **Claude Sonnet 5**; Opus 5 only if queries get hard | API | Multi-step tool use over SQL. Ad hoc, interactive, the one place latency is felt directly |

**The API bill is not a factor.** A ride of four files → twelve candidates → four approvals is
roughly four triage calls and four caption calls. At Sonnet 5 rates ($2/M in, $10/M out) that's
well under a quarter, and the local vision work is free. There is no version of this project
where routing reasoning to a weaker model to save money is correct — spend the effort on the
dog fine-tune, which is where output quality actually moves.

**The distillation loop is worth building deliberately.** Every vision node follows the same
arc: a big general model bootstraps labels → a small specialized model takes over the hot path
→ the big model stays available for the long tail. Qwen3-VL labels frames until SigLIP + head is
good enough; Grounding DINO labels dogs until RF-DETR is fine-tuned. Same shape as stage 4's
decision log feeding the scoring weights, and for the same reason: you are the only source of
ground truth for what your footage should look like, so build to harvest it.

**Two things to check before committing.** Quantization hurts VLMs more than text models — Q4
costs a few points of visual precision, fine for labeling, but run Q8 if a VLM ever lands in a
precision-sensitive hot path. And pin your model IDs deliberately so a triage prompt tuned in
October still behaves in March.

## Machines & storage

### Archive-after-render is the right shape — the desktop stores, the laptop works

This reverses v0.8, which made the desktop the workhorse assuming originals had to stay put. They
don't, and the reason is the proxy branch: **once ingest has produced a proxy and a telemetry
file, the working set for a ride is about 55 MB.** Everything from scoring through review runs on
that. The original is needed twice — at ingest and at render — and after that it's cold.

So the machines sort by what they're actually good at. A Ryzen 3700X with 16 GB and a 6 GB RTX
2060 is the weakest hardware in the fleet for every stage of this pipeline; its genuine virtue is
disk space, which is exactly what an archive wants. **Process on the laptop, archive to the
desktop as the pipeline's final stage.**

| Machine | Role | Holds | Runs |
|---|---|---|---|
| **M4 Max** | The pipeline | All derived data, permanently | Card offload, ingest, scoring, review UI, vision, render. Everything |
| **Desktop** (Linux · 3700X · 2060 · big disk) | Cold archive | Originals, after render | A Samba share and a verification script. Optionally the RF-DETR fine-tune — a batch job, not a pipeline stage |
| **ThinkPad** | Still nothing | — | The portability test, once |

**Durable execution means the laptop is allowed to sleep.** v0.8 argued the review UI belonged on
an always-on machine so a paused LangGraph thread wouldn't sit inside a closed laptop. Weaker than
it sounded: a checkpointed thread lives in a SQLite file on disk, so closing the lid loses nothing
and reopening resumes from the checkpoint. Surviving interruption is the entire purpose of the
checkpointer.

### Archive as a pipeline stage, not a chore

Runs after render succeeds; copies, verifies, records, reclaims.

```
stage: archive
  1 copy      rsync original -> /srv/orbitcut/originals/YYYY/YYYY-MM-DD-slug/
  2 verify    re-hash at the destination         <-- not optional
  3 record    asset.host = desktop, archived_path, archived_at
  4 reclaim   delete local copy — only after step 2 passes
```

Step 2 isn't paranoia. A transfer that reports success and a file that is byte-identical are
different claims, and the whole catalog is keyed on content hash — a silently corrupted archive
copy wouldn't surface until the day you tried to re-render from it, which is exactly the day
you'd least want to find out.

### The working set, and why the round trip is fine

| What | Size per file | Lives | Moves |
|---|---|---|---|
| Original | 10–15 GB | laptop briefly, then desktop forever | once, after render — 3–4 min over gigabit |
| Proxy | ~50 MB | laptop, permanently | never |
| Telemetry | ~5 MB | laptop, permanently | never |
| Renders | ~30 MB | laptop, permanently | to Instagram |

A hundred rides of proxies and telemetry is roughly five gigabytes — it sits on the laptop
indefinitely without you noticing. That's what makes this work: **every reprocessing job except
re-rendering reads only the working set**, so improving scoring weights, re-running clip
selection, or retraining the classifier never touches the archive. Re-rendering an old clip
fetches one original back for a few minutes, occasionally. Keep `host` and `archived_path` on the
asset record so a render job can pull what it needs, then stop thinking about it.

**The one thing to check before committing:** card offload now lands on the laptop, so it needs
room to hold a ride's raw footage until archive runs — a full 64 GB card is 64 GB of transient
space. On a roomy SSD that's nothing; on a half-full 512 GB machine it's a real squeeze and the
answer flips back to offloading on the desktop.

### Hardware notes worth having

- **Turing NVDEC does support HEVC Main 10**, which is what the HERO11 records. The desktop *could*
  ingest if you wanted — the fallback exists and isn't a compromise. Keep hardware decode a config
  value (`videotoolbox` / `cuda` / software) and both machines stay capable.
- **6 GB of VRAM picks your detector size.** Roboflow recommends 8 GB for RF-DETR fine-tuning and
  notes smaller variants fit in 6 GB with a reduced batch. Use **Nano (30.5M, 384px) or Small
  (32.1M, 512px)** — both Apache 2.0, and you're detecting one large subject rather than eighty
  small ones, so it isn't a compromise. Config for 6 GB: `batch_size=1`, `grad_accum_steps=16`,
  `gradient_checkpointing=True`, `fp16_eval=True`.
- **Try the fine-tune on the M4 Max first.** PyTorch's MPS backend handles this class of model, and
  if it works the desktop never does compute at all. If MPS fights you, ship a few hundred labelled
  JPEGs to the 2060 — the training set is tiny even when the footage isn't. Renting an hour of a
  larger GPU is the third option and costs about as much as a coffee.
- **Split telemetry parsing from proxy generation.** Reading the GPMF track decodes no video and
  finishes in seconds; the proxy is the slow half. As two stages, scoring runs on a whole ride
  moments after offload while proxies churn in the background — "plug in the card and see the score
  curve" rather than "plug in the card and wait."
- **Encode split, unchanged:** hardware decode always, hardware encode for proxies, **x264 on CPU
  for finals** — a twenty-second vertical clip at a slow preset costs under a minute and the quality
  difference is what survives Instagram's re-encode.

### Storage layout

```
M4 Max — the working set
  derived/<content_hash>/proxy.mp4 · telemetry.parquet · thumbs/
  renders/<segment_id>.mp4
  inbox/                       transient card offload, cleared by archive
  orbitcut.db                  SQLite + LangGraph checkpoints

Desktop — the archive
  originals/2026/2026-08-19-highland-night/GX010123.MP4
```

**An archive on one disk is not an archive.** Derived data regenerates; originals don't. Once every
ride lives on a single desktop drive, that drive is a single point of failure for the only
irreplaceable thing in the system. The archive stage above is the natural place to eventually fan
out to a second disk — worth building that seam before backfilling hurts.

### If you ever do distribute it

Celery on Redis with **named queues as capability tags**, routing by what a machine can do. Under
this layout the desktop takes `q.cuda` and re-render jobs against archived originals; the laptop
everything else. The trigger is a backlog, not the existence of three computers.

## Data model

| Table | Key fields | Notes |
|---|---|---|
| `asset` | content_hash, path, host, camera_model, duration, fps, resolution, aspect, codec, recorded_at, stabilization, horizon_locked, fov | horizon_locked derived from IORI at ingest |
| `telemetry` | asset_id, parquet_path, streams_present, sample_rates | Pointer not blob. 10 Hz resampled for scoring + raw high-rate ACCL/GYRO for airtime |
| `classification` | asset_id, stage_version, is_mtb+conf, style, mount, **lighting**, trail_id+conf, quality_flags | mount ∈ {chest, helmet}; lighting ∈ {day, twilight, night} |
| `score_series` | asset_id, t, speed, turn, rough, air, descent, flow, composite | 1 Hz; what the review UI draws |
| `segment` | asset_id, t_in, t_out, features (jsonb), dominant_type, **subject**, rank, status | candidate → approved/rejected → rendered. `subject` names who's in it — `orbit`, not `dog_detected`, so the librarian answers questions the way you'd ask them |
| `decision` | segment_id, action, adjusted_in, adjusted_out, reason_chips, decided_at | **The training log.** Append-only |
| `render` | segment_id, preset, crop_path (jsonb), out_path, status, rendered_at | Crop path stored so a preset tweak doesn't redo tracking |
| `trail` | id, name, geometry, region, source, difficulty | Local cache from OSM/Trailforks |

## Build order

Ordered so each phase produces something usable on its own, and so the phase most likely to be
wrong — whether the scoring matches your taste — gets tested before anything expensive sits on it.

### Phases 0 through 4 run on the M4 Max alone

One machine, not three. Distributing any of this earlier would cost a broker, a shared filesystem,
a real database, and a deployment story — all before you know whether the scoring matches your
taste, the thing most likely to send you back to redesign. The desktop's only job in these phases
is to be where originals land once the archive stage runs, and that's a Samba share, not a worker.

The exception is the RF-DETR fine-tune in phase 4, the single job that might want CUDA. It's a
batch job you kick off and walk away from, not a pipeline stage — so it doesn't make the desktop
part of the pipeline.

Phase 5 itself is optional. At a handful of files per ride one machine keeps up comfortably;
distribution is a throughput fix for a problem you may never have, worth building only if you
take on a backlog.

| Phase | Machines | Store | Orchestration | Models |
|---|---|---|---|---|
| 0 · 1 | M4 Max | SQLite | a script and a directory watcher | none |
| 2 · 3 | M4 Max | SQLite | LangGraph + `SqliteSaver` | none |
| 4 | M4 Max (+ desktop for training) | SQLite | LangGraph + `SqliteSaver` | RF-DETR · SigLIP · Sonnet 5 |
| 5 | both | Postgres | + Redis · Celery · `AsyncPostgresSaver` | same, routed by queue |

**SQLite carries you to phase 5**, not just through phase 0 — single machine, single writer, and
LangGraph ships a SQLite checkpointer. Postgres arrives when a second machine does, and the
migration is a schema dump. Running a database server for a script you invoke by hand is
infrastructure maintained for nothing.

Two phase-0 decisions *are* worth making with phase 5 in mind, both nearly free now and annoying
later:

- **Write each step as a function that takes a path and returns a dict**, with a thin CLI wrapper.
  Turning it into a Celery task is then a decorator, not a rewrite.
- **Make hardware decode a config flag** — `videotoolbox`, `cuda`, or software — rather than
  hardcoding one. The laptop ingests normally, the desktop is a working fallback, the ThinkPad is
  proof you haven't bound yourself to one vendor.

**Settled: originals archive to the desktop after render.** The one phase-0 decision expensive to
reverse, now made. Build the `archive` stage in phase 0 even though nothing renders yet — it's
twenty lines, and having `host` and `archived_path` on the asset record from the first file means
never reconciling paths later. Set the retention policy then too: archive after render, or after
ingest if laptop space is tight.

### The phases

- **Phase 0 (a weekend) — Ingest and inventory.** Hash, probe, telemetry → parquet, 540p proxy,
  IORI horizon-lock check. One machine, no queue.
  *Ships: a table of every file you own.*
- **Phase 1 (the critical one) — Telemetry scoring, visualized.** All sub-scores plus composite
  rendered as an overlay on the proxy. Also where chest/helmet classification gets built and
  validated. Tune on real footage. No ML yet.
  *Ships: the answer to "does this score match what I think is exciting."*
- **Phase 2 — Clip selection + review UI.** Peak-finding, NMS, diversity, local web app with
  decision logging.
  *Ships: candidate clips, and the decision log starts filling.*
- **Phase 3 — Render with a static crop.** Center crop to 9:16 from 8:7 source, IORI-gated
  horizon leveling, correct encode settings. This alone covers the solo/chest cell properly.
  *Ships: actual Reels, end to end.*
- **Phase 4 — Vision layer.** Is-it-MTB, dog detection, and the full four-cell reframe policy
  with the smoothed crop path. Note chest/helmet detection does *not* need vision — it comes out
  of phase 1 — so this phase is really just the dog and the tracking.
  *Ships: subject-aware vertical framing.*
- **Phase 5 — Distribute and learn.** Celery queues across three machines; fit scoring weights
  on the accumulated decision log.
  *Ships: a system that gets better while you use it.*

## Risks & open questions

- **HIGH — GPS dropout under canopy breaks trail matching.** With the camera question settled,
  this is the biggest remaining unknown. HERO11 gives GPS5 at 18 Hz, but the terrain you ride is
  where reception is worst, and GPS also feeds the speed sub-score — not just trail naming. Fail
  trail assignment to `unknown`, correctable in review; and hold optical flow as a live speed
  fallback rather than an afterthought.
- **MEDIUM — Retroreflective return at bikejoring distances is unverified.** Reflective garment
  material is characterised at narrow observation angles, and your light-to-camera separation puts
  Orbit around 5–15° depending on lead. The return should still dwarf anything diffuse in a
  near-black frame, but "should" is doing work there. Shoot one night ride, pull ten frames at
  varying lead, measure the actual luma separation between the harness and the next-brightest
  object before building the threshold path around it. If the margin is thin, *then* add the LED.
- **MEDIUM — 8:7 modes cap at 5.3K30 and 4K60.** If you currently shoot 5.3K60 16:9, moving to
  8:7 means choosing between 5.3K30 and 4K60. For MTB, 4K60 8:7 is the better trade. Confirm on
  your own footage before committing the library to one mode.
- **MEDIUM — Helmet vs chest may blur on smooth, straight trail.** The yaw-correlation
  discriminator works because you turn your head; on a fast sight-lined descent you mostly don't.
  Compute correlation only over windows where GPS heading is actually changing, and fall back to
  the `GRAV` pitch feature when there isn't enough turning to measure. Label twenty of your own
  files as a validation set.
- **MEDIUM — Roughness across chest and helmet still needs calibrating.** Much smaller than the
  bar-vs-body problem, but not zero: a helmet sits at the end of a longer, springier lever — higher
  on sharp impacts, lower on sustained chatter. A single scale factor won't capture that;
  calibrate per frequency band if it matters.
- **LOW — 10-bit HEVC color handling.** HERO11's high modes record 10-bit HEVC. Pin color
  primaries, transfer, and matrix explicitly in the ffmpeg command rather than letting them be
  inferred.
- **LOW — Corpus calibration needs a corpus.** Bootstrap with absolute thresholds, switch to
  percentile ranking at ~20 rides.
- **LOW — Storage growth.** 8:7 files are larger than 16:9 equivalents. Decide early whether
  originals stay on one host or get archived after render.

**Resolved since v0.1:** camera body (HERO11 → full telemetry incl. GPS5, GRAV, CORI/IORI);
mount ambiguity (chest/helmet binary, and the bar-mount roughness normalization problem is
mostly gone); HyperSmooth/gyro validity (fine on HERO8+, and Horizon Lock is detectable from IORI).

## Sources

- GoPro GPMF metadata specification — https://gopro.github.io/gpmf-parser/
- HERO11 Black sensor guide (8:7) — https://gopro.com/en/us/news/hero11-black-guide-to-new-sensor
- HERO11 video settings guide — https://abekislevitz.com/hero11-video-settings-guide/
- HERO11 Protune options reference — https://havecamerawilltravel.com/action/gopro-hero11-black-protune/
- GoPro stabilization guide (HyperSmooth crop factors) — https://goproapp.com/blog/gopro-stabilization-guide
- HERO11 night settings — https://suspension-traveler.com/gopro-11-best-night-settings/
- astral (solar elevation / twilight) — https://sffjunkie.github.io/astral/package.html
- Entrance and observation angles for retroreflective sheeting — https://reflectivetape.info/definition-of-entrance-and-observation-angles-for-retro-reflective-tape-sheeting/
- Gyroflow GoPro support notes — https://docs.gyroflow.xyz/app/getting-started/supported-cameras/gopro
- telemetrik (pure-Python GPMF) — https://github.com/kmatzen/telemetrik
- py-gpmf-parser — https://github.com/urbste/py-gpmf-parser
- gopro-telemetry (Node reference) — https://github.com/JuanIrache/gopro-telemetry
- Ultralytics tracking — https://docs.ultralytics.com/modes/track
- SAM 2 — https://docs.ultralytics.com/models/sam-2
- Instagram video specifications 2026 — https://www.socialpilot.co/instagram-marketing/instagram-video-size-specifications
- Celery vs Dramatiq vs Huey — https://www.index.dev/skill-vs-skill/celery-vs-dramatiq-vs-huey
- RF-DETR model sizes and training — https://rfdetr.roboflow.com/latest/
- RF-DETR low-VRAM training parameters — https://rfdetr.roboflow.com/latest/learn/train/training-parameters/
- NVDEC application note (Turing HEVC Main10) — https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvdec-application-note/index.html
- LangGraph durable execution — https://docs.langchain.com/oss/python/langgraph/durable-execution
- LangGraph durability modes explained — https://vadim.blog/durable-execution-agents-that-survive-failure-and-resume-where-they-left-off
- Best object detection models 2026 (RF-DETR, YOLO26) — https://blog.roboflow.com/best-object-detection-models/
- Best local VLMs 2026 — https://tinyweights.dev/posts/best-local-vision-language-models-2026/
- Qwen3-VL 4B vs 8B VRAM/benchmarks — https://codersera.com/blog/qwen3-vl-4b-vs-qwen3-vl-8b-benchmarks-vram-guide/
- Claude API pricing, August 2026 — https://benchlm.ai/anthropic/api-pricing
