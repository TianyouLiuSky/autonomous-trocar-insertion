# Microscope-guided entry-point registration -- lab protocol

Goal: use the Leica overhead view to REGISTER the RCM point at the true
anatomical entry point (one of the two ink-marked candidate sclerotomy
sites), then hand off to the existing kinematic control (approach / orient /
insert / bi-planar) exactly as it already runs today. Insertion **depth**
does NOT need vision -- it already comes from the robot's own encoders
(`robotTask.cpp::controlModeVRCM`, `insertiondepth = TipInEE.z()` vs VRCM).
Vision's only job here is making sure the RCM point is in the right place
to begin with.

Everything below is already built and unit-tested with synthetic data. The
one thing that has NOT been tested yet is a real calibration fit from real
touches + a real click -- that needs you, physically, in the lab. This
doc is the exact sequence to do that.

## -1. Upstream calibration pre-flight (2026-09-21) -- do this BEFORE step 1 if ANYTHING physical changed

Everything below assumes `cur_tip()` (hand-eye + pivot + kinematic origin)
is currently trustworthy. If the robot's own position was moved and/or a
different Trocar is mounted, it is NOT trustworthy until you redo the
affected calibration(s) -- and doing microscope touch-calibration on top of
a stale one silently bakes that error into `mic_extrinsic_calib.npz` in a
way that is very hard to notice later (both stay internally self-consistent,
just both wrong relative to reality). Check which apply:

- **Robot repositioned relative to the D405** (moved on the table, remounted,
  bracket touched, etc.) -> **redo Hand-eye calibration**
  (`handeye_calibrate.py`, ArUco-on-shaft procedure -- see its own docstring).
  Camera-to-robot-base extrinsic; independent of which Trocar is mounted.
- **Trocar swapped** -> **redo Pivot calibration** (`pivot_calibration.py` +
  `pivot_capture_one.py`, pin the tip in a fixed divot/V-groove and sweep
  orientation -- see that script's docstring). EE-to-tip offset; independent
  of where the robot sits in the room.
- **Either of the above happened** -> after saving the new `handeye_T.npz`
  and/or `tip_offset_pivot.npz` and confirming `home_to_rcm_insert_v2_precision.py`
  has loaded them, **re-run the `c` key** (click the real tip in the D405
  view) to refresh `origin_correction.npz` -- it was computed against the
  OLD calibration and is now stale too.

**Then, and only then, do steps 1-4 below in that SAME continuous robot
session (no power-cycle / re-home in between)** -- the whole point of the
touch-point method is that it trusts `cur_tip()` as the common ruler
against which the microscope gets calibrated; if the robot's kinematic zero
drifts (re-homing on this platform is a known un-implemented no-op, see
chat write-up 2026-08+) between finishing hand-eye/pivot and touching the
4 calibration points, `mic_extrinsic_calib.npz` will be calibrated against
a ruler that has since moved, and every downstream `e`/`m`/`x` lock will be
confidently, silently wrong by however much the origin drifted.

## 0. One-time setup check

`home_to_rcm_insert_v2_precision.py` must be the currently-running instance
(restart it if it's been running since before 2026-09-17 ~16:00, otherwise
keys `1`-`4` and `e` below won't exist). If you're not sure, press `h`
first to check status text, or just look for `1-4=log-MIC-calib-touch` and
`e=auto-RCM(microscope,offline)` in the on-screen help line at the bottom
of the video window.

## 1. Touch 4 points with the robot, log them (keys `1`-`4`)

With the robot running and the phantom stationary:

- Jog the tip (wasd/rf/kmjl/uo) to physically rest ON a landmark you can
  ALSO unambiguously click in a microscope photo afterward. Best choices,
  in priority order:
  1. **calib_pt_1** = ink mark A (one of your two real entry points)
  2. **calib_pt_2** = ink mark B (the other real entry point)
  3. **calib_pt_3** and **calib_pt_4** = two more sharp, non-collinear
     features -- e.g. two more points along the limbus ring edge at
     roughly 90 deg apart from each other and from the marks. Avoid
     3 points that are nearly on a line; the affine fit degenerates.
- After resting the tip on each point, press the matching key (`1`,`2`,`3`,
  or `4`). It logs `cur_tip()` (the SAME position pipeline everything else
  trusts) to `mic_calib_touches.jsonl` -- does not move anything.
- Optional but recommended: touch each point 2-3 times (lift off, come back
  down, log again) -- the fit script averages repeats per label and warns
  if a point's repeat spread is > 0.15mm (bad contact / phantom moved).

## 2. Take ONE clean microscope still of the same scene

Right after step 1, without moving the phantom or the scope, capture one
still where all 4 touched points are clearly visible, in focus, and NOT
occluded by the instrument. Prefer this over any earlier archived stills --
it must be the SAME physical setup you just touched.

## 3. Click the same 4 points on that still

On the display machine (`DISPLAY=:0`):

```bash
python3 /home/eyebrp/Matthew/microscope_autonomy/click_calib_points.py \
    /path/to/that_still.jpg
```

Click `calib_pt_1..4` in THE SAME ORDER you touched them. Press `s` to
save -> writes `<image>.calib_px.json` next to the image.

## 4. Fit the calibration

```bash
cd /home/eyebrp/Matthew/microscope_autonomy
python3 mic_extrinsic_calib.py --fit \
    --touches mic_calib_touches.jsonl \
    --clicks /path/to/that_still.jpg.calib_px.json \
    --out mic_extrinsic_calib.npz
```

Read the printed report:
- **RMS residual** -- this is the calibration's accuracy budget. Good:
  < ~0.3mm. If worse, re-touch more carefully / add a 5th point / check the
  still wasn't taken after the scope moved even slightly.
- **touch-point Z spread** -- if > 0.4mm, the touched points aren't
  co-planar enough for a single calibration-plane Z (curved phantom
  surface); tell me and I'll extend the fit to a tilted plane instead of
  flagging this as "good enough."

## 5. Turn a fresh still into a robot-base entry point

Once calibrated, ANY new clean still (doesn't have to be the calibration
still) can be converted directly:

```bash
python3 mic_extrinsic_calib.py --still /path/to/fresh_still.jpg \
    --calib mic_extrinsic_calib.npz --mark 0
```

(`--mark 0` or `--mark 1` selects which of the two ink-marked entry points
to target for this trial.) This prints the classical-CV landmark fit,
warns if the frame is flagged `occluded`, and -- if `--mark` is given --
writes `mic_entry_point.json` for the robot script to consume.

## 6. Lock it on the robot (`e` key)

In `home_to_rcm_insert_v2_precision.py`, press `e`. This reads
`mic_entry_point.json` and locks it into `_vision_rcm` / `S.rcm_source =
"vision"` -- **the exact same slot the live D405 ring-lock (`z`) uses** --
so `g` (approach), `t` (orient+insert), etc. all work completely
unmodified regardless of which vision source produced the point. The
status line will print the mark index, the calibration RMS, and how many
seconds old the source photo is -- **read that before pressing `g`/`t`**;
a stale or occluded-at-capture entry point is worse than none.

## 7. (2026-09-21) One-key version for use DURING a trial: `m`

Steps 5+6 above (`mic_extrinsic_calib.py --still` then press `e`) are the
right mechanism for correcting a run's RCM registration -- step 7 just
collapses them into ONE key so it's fast enough to use immediately before
the final insertion move on every trial, not just once at session start.

Why this (and not visual tip-tracking): the robot's own kinematics/
encoders are already trusted -- `g`/`t` already rely on them to drive the
tip to whatever target they're given, and `cur_tip()` is built on that
same trusted forward kinematics. The tip-arrival error you're seeing with
the pure-D405 pipeline is therefore almost certainly in the **estimate of
where the target is** (D405 depth + hand-eye noise), not in the robot's
ability to get there. So this re-registers the target using the
microscope (which resolves the entry marks far more precisely) and lets
the trusted kinematics do the rest -- it does NOT try to see/track the
needle tip itself in the microscope image (still unreliable, see below).

Setup (one-time, in addition to steps 1-4 above which must already be done):
- Calibration file `mic_extrinsic_calib.npz` must exist (from step 4).
- Decide how a fresh Leica photo gets from the microscope machine onto
  this one -- same manual transfer path already used for every capture in
  this project (USB stick / network copy / however `Lecia Microscope/`
  got populated before). Point it at
  `/home/eyebrp/Matthew/microscope_autonomy/mic_live_stills/` (or set
  `MIC_LIVE_STILLS_DIR` to wherever is more convenient).

Per-trial use, right before the final insertion move:
1. Pull the instrument out of frame if it's occluding the entry marks,
   take ONE clean photo with the Leica, transfer it into
   `mic_live_stills/`.
2. Press **`m`** in the control script. It automatically picks the
   *newest* file in that folder, runs the same classical-CV landmark fit
   + calibrated pixel->base-frame mapping as `e`, and locks the result as
   the RCM point (same `_vision_rcm` slot as `z`/`e` -- `g`/`t` work
   unmodified). The status line reports the mark index, calibration RMS,
   how many seconds old the photo is, and whether the frame was flagged
   occluded -- **read it before pressing `g`/`t`**. If the newest photo in
   the folder is older than `MIC_LIVE_MAX_AGE_S` (default 120s), `m`
   refuses and tells you, so a stale leftover photo can't get silently
   reused.
3. Which of the two entry points (`--mark 0` vs `1`) it targets is fixed
   per-run by the `MIC_TARGET_MARK` environment variable (default `0`) --
   set it before launching the control script if you want the other one.

## 8. (2026-09-21) Why `z` alone is NOT close to the entry point on this phantom

`z` (D405 live ring lock) finds a red/maroon annulus (`detect_ring()`'s
LAB a*-channel threshold) and targets **its center**. On THIS phantom that
red/maroon ring is the **limbus** -- a natural landmark, but a DIFFERENT
one than the true entry points, which are two small **dark ink dots**
~3-4mm posterior to the limbus (confirmed via the caliper/clinical-offset
discussion). So `z`'s output sits at the phantom's optical center, roughly
11-13mm away from either real entry point -- not a jitter/calibration
problem, a wrong-landmark problem. Screenshot evidence: the orange RCM dot
lands near mid-dome, nowhere near either ink mark.

## 9. (2026-09-21, superseded 2026-09-28) `x` key: teach `z` a fixed correction toward the real entry point

`x` = `z`'s live ring lock **+ a fixed base-frame offset vector** that
shifts "limbus center" toward "the actual ink-marked entry point". This
lets APPROACH/HOVER use D405 alone (no microscope photo needed at the
moment of use) and land much closer than raw `z` -- but it is only a
learned constant, good only as long as the phantom hasn't moved since it
was learned. **`m`/`e`/`y` remain the authoritative correction right
before the actual insertion move.**

**2026-09-28 update: use `mic_ring_offset_verify.py`, not
`mic_extrinsic_calib.py --ring-offset`, to learn this offset.** The
original `--ring-offset` CLI (still present, not removed) fits from
exactly ONE (ring, microscope) snapshot with zero held-out validation --
the same self-referential-accuracy trap the rest of this document exists
to avoid, just never fixed for this one correction. The new workflow:

1. Needs the one-time affine calibration from steps 1-4 above to already
   exist (`mic_extrinsic_calib.npz` with `loo_rms_mm >= 0`).
2. On the robot: press `'z'` (wait for "RING LOCKED"), then press `'0'`
   (new key, `log_ring_offset_sample()`) -- this appends one paired
   sample to `ring_offset_samples.jsonl`, auto-detecting the entry
   marks from the freshest microscope still (same detector `y`/`e`
   already trust; nothing to click by hand). Repeat across >=2 sessions,
   >=5 samples per mark, before trusting any accuracy number.
3. Fit + cross-validate:
   ```bash
   cd /home/eyebrp/Matthew/microscope_autonomy
   python3 mic_ring_offset_verify.py --fit \
       --samples ring_offset_samples.jsonl \
       --out ring_to_entry_offset.npz
   python3 mic_ring_offset_verify.py --report --npz ring_to_entry_offset.npz
   ```
   This prints the SAME sanity-check magnitude as before, PLUS a
   leave-one-out RMS, a bootstrap 95% CI, and a direct
   corrected-vs-uncorrected comparison ("CORRECTION HELPS" /
   "CORRECTION DOES NOT HELP") -- see
   `MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md`'s Step 4 for the full
   rationale. The saved `.npz` is in the same format the `x` key already
   reads (`offset_mark0`/`offset_mark1`), so nothing else changes on the
   robot side -- only how honestly the offset was learned.
4. Re-run whenever the phantom, camera, or scope gets bumped -- old
   samples from before a bump should be discarded (rename with a
   `.bak_*` suffix, don't just delete), not mixed in with post-bump ones.

## 10. (2026-09-22) Getting an actual LIVE feed into this machine

Everything above (steps 1-9) works off manually-transferred still files.
As of today there is **no capture device plugged into this Linux machine
at all** -- confirmed via `lsusb` / `v4l2-ctl --list-devices`: only the
D405 RealSense shows up (`/dev/video0-2`). To go from "manual file copy"
to "live", you need a physical capture path FIRST, then a one-line script
to feed it into the existing pipeline (which needs zero other changes,
see below).

### 10a. Physical setup (do this in the lab, hardware only)

1. Find the video **output** on the Leica camera control unit (CCU) --
   this is whatever currently feeds the external monitor via "DVI 2".
   Look at the back of the CCU box for the connector type: **DVI / HDMI /
   3G-SDI**. Some CCUs have a second dedicated output (often labeled REC,
   AUX, or a 2nd DVI/SDI) meant exactly for recording -- prefer that one if
   present, so you don't have to touch the monitor's existing cable.
2. If there is only ONE output currently going to the monitor, get a
   passive **1-in-2-out HDMI/DVI splitter** so the monitor keeps working
   and the capture card taps the same signal.
3. Get a USB capture card matching the connector:
   - **HDMI output** -> any generic USB3 "HDMI to USB video capture"
     dongle (cheap, ubiquitous, e.g. the ones sold for game-console
     streaming). These enumerate as a standard UVC webcam on Linux --
     zero driver install needed.
   - **DVI-D output** -> same dongle as above + a **passive DVI-to-HDMI
     adapter** cable (DVI-D and HDMI carry the same TMDS signal, a passive
     adapter is enough, no active converter needed).
   - **3G-SDI output** -> needs an SDI-capable USB capture card (e.g.
     Magewell USB Capture SDI). Pricier and occasionally needs a vendor
     driver, but Magewell publishes Linux UVC-compatible firmware for most
     of their line -- check before buying if this is the port you have.
4. Plug the capture card's USB cable into this Linux machine (any port).
5. Verify it's recognized:
   ```
   lsusb                          # should show a new device appear
   v4l2-ctl --list-devices        # should show a new entry besides "Intel RealSense..."
   python3 mic_live_capture.py --list   # convenience wrapper around the above two
   ```
   If nothing new shows up, try a different USB port (prefer a USB3/blue
   port for HDMI-capture-card bandwidth), and double check the capture
   card itself has power/link (most show a small LED when receiving signal).

### 10b. Software: `mic_live_capture.py`

Once step 10a shows a new `/dev/videoN`, run:
```
cd /home/eyebrp/Matthew/microscope_autonomy
python3 mic_live_capture.py --device N --show      # --show pops a local preview window to sanity-check framing/focus
```
(Omit `--device N` to let it auto-probe every non-RealSense `/dev/video*`
and pick the first one that yields a real, non-black frame -- convenient,
but pass the explicit index once you know it, so it never races the D405.)

This writes a rolling window of recent frames (default: keep last 5,
one new file every 0.5s -- both tunable with `--keep`/`--interval`) into
`MIC_LIVE_STILLS_DIR` (= `mic_live_stills/` by default, same folder steps
1-9 already use). **No other code needs to change**: `get_mic_preview()`
(the PiP panel) and the `y` key in `home_to_rcm_insert_v2_precision.py`
already just pick the newest file in that folder -- so the moment this
script is running, the PiP panel starts auto-refreshing every ~0.5s and
`y` starts locking onto genuinely fresh frames instead of a manually
copied still. Leave it running in its own terminal for the whole session.

Note this is still "refresh a still every N ms", not a continuous video
pipe with per-frame tracking -- see the honest gap notes right below.
This is the right amount of "live" for what steps 6/7/9 actually need
(a fresh confirmation photo right before a lock/insertion decision, not
frame-by-frame instrument tracking), but flag it explicitly if the plan
changes to needing true continuous tracking.

## What this does NOT cover yet (be honest about the gaps)

- **No continuous video / per-frame tracking loop.** As of 2026-09-22
  there IS a way to get a rolling, auto-refreshing still feed once a
  capture card is connected (see step 10) -- that's enough for "confirm
  the RCM point right before the trial starts", refreshed every ~0.5s.
  It is still NOT frame-by-frame instrument tracking; "watch the needle
  approach the sclera in real time under the scope" would need a proper
  video pipe + a tracker, which is a separate, bigger piece of work.
- **No needle-tip tracking under the microscope.** Both datasets show the
  instrument as an out-of-focus, mostly-static blob -- there's no reliable
  signal to track yet (see `mic_perception.py` docstring). If you want
  real-time visual confirmation DURING insertion/bi-planar (not just before
  it), that needs either a sharper/closer focus setting during that phase,
  or accepting kinematic depth alone (which is probably fine, since that's
  where the depth number already comes from).
- **Single calibration per session.** If the scope or phantom moves after
  step 2/3, this whole calibration is invalid and must be redone from
  step 1.

If step 4's RMS residual comes back bad, or the phantom surface turns out
to be non-planar (step 4's Z-spread warning), tell me the numbers and I'll
adjust the fit -- don't try to push a bad calibration into `e`.
