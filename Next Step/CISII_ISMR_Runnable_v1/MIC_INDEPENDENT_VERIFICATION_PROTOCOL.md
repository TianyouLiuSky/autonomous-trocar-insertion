# Microscope-based independent entry-point verification — protocol

Written 2026-09-24. Companion to `mic_extrinsic_calib.py` (now with
`--loo`) and the new `mic_independent_verify.py`. Positioned against
`/home/eyebrp/Matthew/ISMR2027_SOTA_and_gap.md` — this closes a piece of
**G1** ("validation of the registration chain is self-referential") and
**G3** ("where accuracy IS established independently, it costs a laser
tracker or an iOCT") for the **entry-point registration** term
specifically, using the microscope instead. It is a candidate contribution
(tentatively "C6") **only after** the campaign below actually produces
n≥10 gated, cross-session numbers — not before.

## What this does and does NOT validate

- **C3** (`measure_shaft_direction_3d`, the D405 vis3d check) validates
  **needle-shaft orientation** (degrees) against the kinematic chain,
  using RGB-D + RANSAC.
- **This protocol** validates **entry-point/registration position** (mm)
  — a different error source — using the microscope + a fitted affine
  pixel→base map. It shares no information with the kinematic chain
  *at measurement time* (the map is fit once, offline; each new
  measurement is a fresh photo run through that fixed map), but the map
  itself WAS fit using robot-touched points, so this is not a totally
  independent channel end-to-end the way an external laser tracker would
  be — be precise about that distinction in any writeup.
- It does **not** by itself resolve the ISMR gap doc's reviewer-attack
  item #3 (oblique-stage 14.6°, n=1) — that is an orientation error, and
  this protocol measures position. Extending it to orientation (e.g. from
  two visible marks) is future work, not assumed here.

## Why the current calibration state cannot be trusted yet

As of 2026-09-24, `mic_calib_touches.jsonl` / `mic_extrinsic_calib.npz`
**do not exist on this system** — the microscope RCM path has been coded
and plumbed (live video, atomic-write fix, PiP UI) but never actually
fit with real data. The original design also only collected 4 points for
a 6-parameter affine model (only 2 residual DOF) and validated on the
SAME 4 points — self-referential by construction, the exact G1 trap.
Two structural fixes are already in place before you run this:

1. `home_to_rcm_insert_v2_precision.py` now has keys **'1'-'8'** (was
   '1'-'4') to log up to 8 calibration touches, and `mic_extrinsic_calib.py
   --fit` now runs **leave-one-out cross-validation** automatically at
   n≥5 and saves `loo_rms_mm` into the `.npz`. `mic_independent_verify.py`
   refuses to gate anything against a calibration with no LOO
   (`loo_rms_mm < 0`).
2. A **separate** key, **'9'**, logs campaign (held-out) ground-truth
   points to `mic_campaign_touches.jsonl` — a different file from
   `mic_calib_touches.jsonl` — so a verification point can never
   accidentally leak into the calibration fit (structural, not just
   disciplinary, train/val separation — same idea as
   `split_pairs_holdout.py`).

## Step 1 — (re-)fit the calibration honestly

1. Physically touch the robot tip to **8 points** spread across the
   *entire* microscope field of view — not clustered near the two ink
   marks. Any sharp, unambiguous, clickable feature counts. Press
   `'1'`...`'8'` at each.
2. Take ONE clean, in-focus, unoccluded microscope still of the same
   scene. Click all 8 points in the same order:
   ```
   python3 click_calib_points.py /path/to/still.jpg --n-points 8
   ```
3. Fit + cross-validate:
   ```
   python3 mic_extrinsic_calib.py --fit --loo \
       --touches mic_calib_touches.jsonl \
       --clicks /path/to/still.jpg.calib_px.json \
       --out mic_extrinsic_calib.npz
   ```
   Read the **LOO RMS**, not the in-sample residual. If `loo_rms_mm` is
   worse than ~0.3-0.5mm, or any single held-out error is much larger
   than the rest, re-touch/re-click before proceeding — do not smooth
   this over (same discipline the ISMR doc applies to the oblique-stage
   gap).

## Step 2 — run the campaign

Target: **n≥10 gate-passing points, across ≥2 sessions** (re-mount /
re-touch the phantom-scope setup between sessions — a fresh session
number, e.g. `--session session2_20260925`), mirroring the n=10/2-session
structure already used for C3.

Per point:
1. Physically touch the robot tip to a **new** point — not one of the 8
   calibration points, ideally spread across the field of view including
   near its edges (where extrapolation risk is highest) and, if the
   phantom allows, at a couple of different heights (to stress-test the
   flat-`z_plane_mm`-assumption). Press **`'9'`** — note the auto-generated
   label printed (`campaign_<timestamp>`).
2. Take a fresh microscope still (or, better, ≥5 stills of the same static
   scene in a row, now that live video is running — this is what makes
   the repeatability gate meaningful; it did not exist before this
   protocol).
3. Click the point on (any) one still:
   ```
   python3 mic_independent_verify.py --click /path/to/campaign_still1.jpg
   ```
   (this only records where to sample the pixel location for the
   verification math — the ≥5 stills are used for the repeatability gate
   in step 4, they do not each need to be clicked individually since the
   detector is what has to reproduce, not your click)
4. Run the gate + log the comparison:
   ```
   python3 mic_independent_verify.py --campaign \
       --label campaign_20260924_153000_123456 \
       --gate-stills campaign_still1.jpg campaign_still2.jpg campaign_still3.jpg campaign_still4.jpg campaign_still5.jpg \
       --calib mic_extrinsic_calib.npz \
       --touches mic_campaign_touches.jsonl \
       --session session1_20260924 \
       --mark 0
   ```
   This appends one row to `mic_campaign_log.csv` whether the gate passes
   or fails — a gate failure is itself data (it tells you the gate's pass
   rate, which matters as much as the accuracy number: a gate that never
   fails on real data is suspicious in exactly the way C3's original
   `inlier_ratio >= 0.5` check was before pt.15 tightened it).

## Step 3 — report

```
python3 mic_independent_verify.py --report --log mic_campaign_log.csv
```

Reports (mirroring the statistical discipline already used for C1/C3):
gate pass rate + fail-reason breakdown, mean/std/RMS of `err_3d_mm`,
`err_xy_mm` (registration-plane accuracy) and `err_z_mm` (flat-plane-
assumption error, reported separately on purpose — a good XY number with
a bad Z number means the plane assumption, not the 2D map, is the
problem), bootstrap 95% CI at n≥5, and a per-session breakdown once ≥2
sessions exist.

## Step 4 (new, 2026-09-28) — learn + validate the D405 ring→entry-point correction

Steps 1-3 above only VERIFY the depth-camera/kinematic chain against the
microscope; they do not use the microscope to make anything the robot
actually uses better. There is a separate, already-partially-built
mechanism that DOES that: the `'x'` key adds a fixed base-frame offset on
top of the live D405 ring lock (`'z'`) so the D405-only APPROACH/HOVER
target lands closer to the true ink-marked entry point than the raw
ring-center (which locks onto the limbus, not the entry mark — see
`mic_extrinsic_calib.py`'s `do_ring_offset()` docstring). As of
2026-09-28 that offset had **never been fit from more than one anecdotal
snapshot, and had zero held-out validation** — the same G1
self-referential trap this whole document exists to avoid, just not yet
fixed for this specific correction. `mic_ring_offset_verify.py` fixes
that:

1. Prerequisite: Step 1 above already done (`mic_extrinsic_calib.npz`
   fit with `loo_rms_mm >= 0`) — this correction is only as trustworthy
   as the map it is measured against.
2. With the D405 ring **locked** (`'z'`), press **`'0'`**
   (`log_ring_offset_sample()`). No physical motion is needed — it just
   pairs the current live ring-lock position with whatever the freshest
   microscope still's auto-detected ink marks say the true entry points
   are (same detector `'y'`/`'e'` already trust; this key does not ask
   you to click anything). Repeat, ideally across ≥2 sessions
   (re-approach, re-lock `'z'`, press `'0'` again) — this builds
   `ring_offset_samples.jsonl`, an append-only file distinct from both
   `mic_calib_touches.jsonl` and `mic_campaign_touches.jsonl`.
3. Fit + cross-validate (mirrors Step 1's `--fit --loo` exactly, just for
   a constant 3D offset instead of a 2x3 affine):
   ```
   python3 mic_ring_offset_verify.py --fit \
       --samples ring_offset_samples.jsonl \
       --out ring_to_entry_offset.npz
   ```
   Read `loo_rms_mm_mark0`/`loo_rms_mm_mark1`, not the in-sample scatter.
   Needs ≥5 paired samples per mark before either number means anything;
   below that the `.npz` stores `-1.0` (matching Step 1's sentinel
   convention) and `mic_ring_offset_verify.py --report` will say so
   explicitly.
4. Target for calling this validated: LOO RMS for the `'x'`-key
   correction should be compared against (a) the raw, un-corrected
   ring-lock error (i.e. `|ring_base - entry_base|` before adding any
   offset — report this baseline too, not just the corrected number) and
   (b) Step 1's `loo_rms_mm` for the full microscope path, since `'x'` is
   explicitly a cheaper D405-only approximation to that, not a
   replacement for it. If the corrected LOO RMS is not meaningfully
   better than the uncorrected baseline, that is itself the honest
   result to report — do not keep an unhelpful correction just because
   code exists for it.

## What "done" looks like before this goes in a paper/slide

- `loo_rms_mm` from Step 1 is stable and reasonable (target: better than
  the D405 ring-lock's own jitter, since the whole point of the
  microscope path is that it resolves the entry marks better than the
  D405 — if it doesn't beat the D405 on a real held-out test, that is an
  important negative result to report, not to hide).
- ≥10 gate-passing campaign points across ≥2 sessions, with the gate's
  fail-reason distribution reported alongside the accuracy number (not
  cherry-picked to only the passes).
- `err_z_mm` explicitly checked and reported — if it is not small, say so
  and propose the tilted-plane extension `mic_extrinsic_calib.py` already
  warns about (z = f(u,v)) rather than continuing to assume a constant Z.
- Session-to-session consistency reported explicitly, the same way C3's
  writeup reports "moved only 1.5° from the original single-session
  estimate" as evidence the measurement is real and not a one-off.

## Known gaps in this first version (say so, don't hide them)

- `MIC_MIN_FOCUS`, `MIC_MAX_EXTRAPOLATION_MM`, `MIC_MAX_REPEAT_JITTER_MM`
  in `mic_independent_verify.py` are **provisional defaults** picked
  without any real data (none existed as of 2026-09-24) — re-tune them
  against the Step 1/2 data, do not trust them blindly.
- The extrapolation-distance-to-mm conversion (`hull_extrapolation_mm`)
  is an isotropic approximation of an anisotropic affine map — good
  enough for a go/no-go gate, not for reporting a directional error bar.
- There is no independent Z cross-check against D405 depth at the mark
  location yet (unlike the shiny needle shaft, the sclera surface near
  the entry marks is diffuse, so D405 depth there is plausibly usable —
  this is a natural next extension, not yet built).
- This protocol does not touch needle-axis/orientation accuracy at all —
  do not let a good result here be read as resolving the ISMR gap doc's
  oblique-stage (#3) attack surface.
