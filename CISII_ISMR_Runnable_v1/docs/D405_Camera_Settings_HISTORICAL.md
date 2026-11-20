# Intel RealSense D405 Camera Settings Integration & AI Coding Handoff Guide

**Project:** CIS II / SHER autonomous insertion pipeline  
**Target camera:** Intel RealSense D405  
**Primary objective:** reduce overexposure on the white eye phantom while preserving the existing perception and robot pipeline  
**Validated nominal camera setting:** `Auto Exposure OFF`, `Exposure = 12000`, `Gain = 16`

---

## 0. Read This First — Scope of the Change

This task is a **camera-configuration integration task**.

The intended change is small:

```text
Auto Exposure = OFF
Exposure      = 12000
Gain          = 16
```

The purpose is to make D405 imaging repeatable and to reduce overexposure on the white eye phantom.

### Do not broaden this task automatically

Do **not** treat this as a reason to redesign the ring detector, redo calibration, rewrite the robot controller, or alter phantom-normal logic.

Unless a later test explicitly shows a separate failure, leave the following systems unchanged:

```text
hand-eye calibration
pivot calibration
origin correction
robot kinematics
RCM control
perpendicular construction
phantom-normal estimator
ring threshold logic
depth-to-base transform
robot motion state machine
microscope calibration
```

The camera-setting patch should be isolated so that, if anything behaves unexpectedly, it can be reverted without affecting other subsystems.

---

## 1. Why This Change Is Needed

The D405 image previously showed strong overexposure on the white eye phantom, particularly around the upper scleral/ring region.

The previous operating state observed in RealSense Viewer used an exposure around:

```text
Exposure ≈ 20000
Gain     = 16
```

At that exposure, the white phantom showed broad clipping and severe loss of visible surface detail.

A controlled exposure sweep was therefore performed. Exposure values around `11000–14000` were tested, followed by more focused testing around the final candidate range.

The final selected nominal point was:

```text
Exposure = 12000
Gain     = 16
```

This substantially reduced broad overexposure while preserving ring segmentation, depth validity, 3D RCM locking, and local phantom-normal estimation.

---

## 2. Validated Camera Configuration

Use this as the default controlled imaging configuration:

```text
Auto Exposure: OFF
Exposure:      12000
Gain:          16
```

Recommended manual operating range under the current lab lighting:

```text
Preferred exposure range: 11500–12500
Validated nominal point:  12000
Observed usable range:     approximately 11500–13000
Gain:                      keep at 16
```

### Operational rule

For normal experiments, do **not** continuously tune exposure.

Use:

```text
Exposure = 12000
Gain     = 16
```

and keep the lab lighting controlled.

If outside sunlight becomes unusually strong, reduce the environmental light physically, for example by closing the curtains/blinds, before changing camera settings.

---

## 3. What Has Already Been Validated

The final camera configuration was tested separately from robot motion.

### Ring/depth/RCM validation

With:

```text
Exposure = 12000
Gain     = 16
```

the validation produced:

```text
Ring frames tested:                         750
Ring acceptance rate:                      100%
Valid depth given accepted ring:           100%
Saturation fallback use:                   0 frames

RCM lock:
  achieved:                                yes
  first lock frame:                        15
  jitter at lock:                          0.267 mm
  production jitter threshold:             3.0 mm
```

### Phantom-normal validation

The same camera setting was also tested with the current local-plane normal estimator:

```text
Normal attempts:                           50
Successful normal estimates:              50
Success rate:                              100%

Mean plane RMS:                            ~0.640 mm
Maximum plane RMS:                         ~0.771 mm
Current reference/advisory level:          1.0 mm

Mean centroid disagreement:                ~4.06 mm
Maximum centroid disagreement:             ~4.53 mm
Current gate:                              5.0 mm
```

### Important conclusion

The camera parameter change did **not** break:

```text
ring detection
depth availability
3D target reconstruction
RCM lock
local depth-plane fitting
phantom-normal estimation
```

The remaining baseline-normal mismatch observed during testing is a separate scene/phantom-pose issue and should not be mixed into the camera exposure change.

---

## 4. Current Uploaded `auto_target_vision(2).py`

The uploaded standalone file currently initializes the RealSense camera approximately as follows:

```python
self.pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
self.profile = self.pipe.start(cfg)
self.align = rs.align(rs.stream.color)
self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
cvs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
self.intr = cvs.get_intrinsics()

for _ in range(20):
    self.pipe.wait_for_frames()
```

This starts the streams and performs warm-up, but it does not explicitly:

```text
disable auto exposure
set manual exposure
set gain
read back and report the actual camera settings
```

That is the specific gap this integration should fix.

---

## 5. Important Version Warning

The uploaded standalone `auto_target_vision(2).py` contains:

```python
MAX_RING_RESIDUAL_RATIO = float(
    os.environ.get("MAX_RING_RESIDUAL_RATIO", "0.22")
)
```

The newer integrated production pipeline used during the later validation used:

```text
MAX_RING_RESIDUAL_RATIO = 0.32
```

together with newer ring-processing logic.

These are two different detector versions.

### AI coding assistant instruction

If modifying `auto_target_vision(2).py` only to add the camera settings:

> **Do not modify the ring residual threshold as part of this patch.**

Do not automatically change:

```text
0.22 -> 0.32
```

unless the human developer explicitly asks to synchronize the standalone detector with the newer integrated detector.

The camera patch and detector-version synchronization should be separate changes.

---

## 6. What Files Should Be Modified

### Primary target

The AI coding assistant should modify whichever source file actually creates the active D405 `rs.pipeline()` during normal operation.

For the uploaded standalone script, that is:

```text
auto_target_vision(2).py
```

specifically:

```text
class RSCam
RSCam.__init__()
```

### Integrated pipeline

The project also contains a newer integrated pipeline under approximately:

```text
/home/eyebrp/Matthew/calibration_work_2026-09-21/01_code/
```

If the active system uses `home_to_rcm_insert_v2_precision.py` or another integrated script to open the D405 directly, the same camera-setting logic should be added to that camera initialization path as well.

### Do not edit calibration artifacts

Do not modify:

```text
handeye_T.npz
phantom_normal.npz
auto_target.npz
origin_correction.npz
tip_offset*.npz
mic_extrinsic_calib.npz
ring_to_entry_offset.npz
```

as part of this camera-settings task.

---

## 7. Recommended Code Change

Add configurable camera constants near the existing environment-variable configuration:

```python
D405_AUTO_EXPOSURE = int(
    os.environ.get("D405_AUTO_EXPOSURE", "0")
)

D405_EXPOSURE = float(
    os.environ.get("D405_EXPOSURE", "12000")
)

D405_GAIN = float(
    os.environ.get("D405_GAIN", "16")
)
```

Then apply them immediately after:

```python
self.profile = self.pipe.start(cfg)
```

and before the warm-up frames.

A robust implementation is:

```python
class RSCam(object):
    def __init__(self, width=640, height=480, fps=30):
        self.pipe = rs.pipeline()

        cfg = rs.config()

        cfg.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps,
        )

        cfg.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps,
        )

        self.profile = self.pipe.start(cfg)

        device = self.profile.get_device()

        exposure_sensor = None

        for sensor in device.query_sensors():
            try:
                sensor_name = sensor.get_info(rs.camera_info.name)
            except Exception:
                sensor_name = ""

            if sensor.supports(rs.option.exposure):
                if exposure_sensor is None:
                    exposure_sensor = sensor

                if "Stereo" in sensor_name:
                    exposure_sensor = sensor
                    break

        if exposure_sensor is None:
            raise RuntimeError(
                "No D405 sensor supporting manual exposure was found."
            )

        if exposure_sensor.supports(rs.option.enable_auto_exposure):
            exposure_sensor.set_option(
                rs.option.enable_auto_exposure,
                float(D405_AUTO_EXPOSURE),
            )

        if exposure_sensor.supports(rs.option.exposure):
            exposure_sensor.set_option(
                rs.option.exposure,
                float(D405_EXPOSURE),
            )

        if exposure_sensor.supports(rs.option.gain):
            exposure_sensor.set_option(
                rs.option.gain,
                float(D405_GAIN),
            )

        actual_exposure = (
            exposure_sensor.get_option(rs.option.exposure)
            if exposure_sensor.supports(rs.option.exposure)
            else float("nan")
        )

        actual_gain = (
            exposure_sensor.get_option(rs.option.gain)
            if exposure_sensor.supports(rs.option.gain)
            else float("nan")
        )

        print(
            "D405 imaging settings: "
            "auto_exposure=%d exposure=%.0f gain=%.0f"
            % (
                D405_AUTO_EXPOSURE,
                actual_exposure,
                actual_gain,
            )
        )

        self.align = rs.align(rs.stream.color)

        self.depth_scale = (
            device
            .first_depth_sensor()
            .get_depth_scale()
        )

        cvs = (
            self.profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
        )

        self.intr = cvs.get_intrinsics()

        # Warm up only AFTER applying the intended manual settings.
        for _ in range(30):
            self.pipe.wait_for_frames()
```

---

## 8. Why Warm-Up Must Come After Setting Exposure

The current standalone code warms up immediately after starting the pipeline.

The revised order should be:

```text
start RealSense pipeline
↓
find exposure-capable D405 sensor
↓
disable auto exposure
↓
set exposure = 12000
↓
set gain = 16
↓
read back settings
↓
warm up camera
↓
begin normal perception pipeline
```

This allows the frames used after initialization to come from the intended fixed imaging state.

Do not warm up under one camera state and then change the exposure afterward.

---

## 9. Why Read-Back Is Required

Do not assume that a `set_option()` call means the requested value was accepted exactly.

After setting the camera, read back:

```python
sensor.get_option(rs.option.exposure)
sensor.get_option(rs.option.gain)
```

and print the actual values.

Expected startup output:

```text
D405 imaging settings: auto_exposure=0 exposure=12000 gain=16
```

If the output differs unexpectedly, stop and inspect the RealSense configuration rather than silently continuing.

---

## 10. AI Coding Assistant — Required Behavior

If an AI coding assistant is given this document, the human should ask it to follow the rules in this section.

### Required interpretation

The AI should treat the task as:

```text
Integrate the validated fixed D405 camera settings into the existing
camera-initialization code with the smallest possible code change.
```

### The AI should

```text
locate the active D405 initialization path
add explicit manual exposure/gain configuration
disable auto exposure
apply settings before warm-up
read back and print actual exposure/gain
keep the change local to camera initialization
preserve all existing detection and robot logic
preserve calibration-file formats and paths
keep environment-variable overrides available
```

### The AI should not

```text
rewrite the ring detector
change LAB thresholds
change residual thresholds
change ROI values
change connected-component logic
change Kasa fitting
change depth sampling mode
change RCM smoothing
change MAX_JITTER_MM
change hand-eye math
change robot motion logic
change perpendicular construction
change phantom-normal fitting
rewrite calibration files
rename project files unnecessarily
reorganize the project
```

unless the human explicitly asks for a separate change.

### AI safety instruction

If the AI sees unrelated old/new version differences in the code, it should report them to the human rather than silently "fixing" them in the same patch.

---

## 11. Suggested AI Coding Prompt

A teammate can give an AI coding assistant the following instruction:

```text
Use this D405 camera-settings integration guide as the specification.

Modify only the active RealSense D405 initialization path.

Required defaults:
- Auto Exposure OFF
- Exposure = 12000
- Gain = 16

Apply these settings immediately after pipeline.start() and before camera
warm-up. Read back the actual exposure/gain values and print them at startup.

Keep environment-variable overrides for the three settings.

Do not change detector thresholds, LAB segmentation, ROI, circle fitting,
depth sampling, hand-eye transform, RCM smoothing, phantom-normal logic,
robot-control logic, or any calibration files.

Show me the exact diff before proposing any additional refactor.

After the code change, provide a short test checklist that verifies:
1. camera settings are actually applied;
2. ring tracking still works;
3. depth remains valid;
4. the existing RCM lock criterion still works;
5. no calibration artifact is overwritten merely by camera initialization.

If you find version differences such as residual threshold 0.22 vs 0.32,
report them separately and do not change them without explicit approval.
```

---

## 12. Minimal Test Plan After Integration

The first integration test should be performed with the robot stationary.

### Test A — startup parameter verification

Start the active pipeline.

Confirm terminal output reports:

```text
auto_exposure=0
exposure=12000
gain=16
```

Failure condition:

```text
requested and actual values do not match
camera rejects the setting
camera opens under automatic exposure
```

If this fails, stop before robot testing.

---

## 13. Test B — visual overexposure check

With the phantom in the normal experiment position:

Check that:

```text
broad white clipping is substantially reduced
the red ring remains clearly visible
the scene is not globally too dark
```

A small local specular highlight is acceptable.

Do not lower exposure solely to force every highlight pixel below saturation.

---

## 14. Test C — ring tracking check

Run the existing detector without robot movement.

Expected behavior:

```text
ring remains continuously detectable
ring center does not jump to background objects
ring radius remains stable
normal production detector logic is unchanged
```

If the ring suddenly fails after the camera patch, first verify:

```text
actual exposure/gain read-back
lighting
correct sensor selection
camera stream format/resolution
```

Do not immediately retune the ring detector.

---

## 15. Test D — depth and 3D stability

Run the existing depth path with the robot stationary.

Expected behavior based on prior validation:

```text
depth validity should remain very high
3D target jitter should remain comfortably below the existing 3 mm limit
15-sample lock should be achievable
```

Prior validation at `12000/16` achieved:

```text
100% valid depth on accepted rings
RCM lock at frame 15
~0.267 mm jitter at lock
```

If depth becomes unstable, investigate the RealSense settings and physical scene before changing hand-eye calibration.

---

## 16. Test E — phantom-normal check

If the integrated pipeline uses the current phantom-normal estimator, run the normal estimation with the robot stationary.

Expected prior result at `12000/16`:

```text
50/50 successful estimates
mean plane RMS ~0.640 mm
maximum plane RMS ~0.771 mm
centroid disagreement remained below the 5 mm gate
```

The old stored normal may differ from the current local normal if:

```text
the phantom was repositioned
the sampled surface location changed
the old normal belongs to another local surface point
```

Do not interpret a normal mismatch alone as evidence that the camera exposure patch broke hand-eye calibration.

---

## 17. Test F — robot motion is a later stage

Do not make the first post-patch test a robot-motion test.

Recommended order:

```text
camera startup
↓
visual image check
↓
ring tracking
↓
depth / 3D stability
↓
RCM lock
↓
phantom-normal check
↓
only then consider robot-motion testing
```

This makes failures easier to isolate.

---

## 18. Rollback Plan

The camera change should be easy to undo.

Because settings are exposed through environment variables, temporary rollback can be performed without source edits.

For example:

```bash
D405_EXPOSURE=20000 D405_GAIN=16 python3 auto_target_vision.py
```

This is useful for A/B diagnosis.

However, `12000/16` should remain the production default unless a new controlled test provides evidence for a better setting.

---

## 19. What Would Justify Changing the Camera Settings Again

Consider revisiting exposure only if one of these conditions changes materially:

```text
lab illumination changes
camera position changes enough to alter reflected light
phantom material changes
illumination hardware changes
polarizer/diffuser is added
camera firmware changes in a way that alters sensor behavior
ring signal becomes consistently too dark
large-scale clipping returns
```

If a polarizer is added, run a short exposure sweep again because the polarizer will reduce transmitted light.

---

## 20. What Would Justify Recalibration

The camera exposure change by itself is not a geometric recalibration event.

Hand-eye calibration should be reconsidered if:

```text
D405 is physically moved
D405 is rotated
D405 mount is loosened/reinstalled
camera-to-robot geometry changes
```

Pivot/tool calibration should be reconsidered if:

```text
tool/trocar geometry changes
tip offset changes
tool mounting changes
```

A camera exposure adjustment alone should not trigger those procedures.

---

## 21. Recommended Git / Change-Management Strategy

Keep this modification isolated.

Suggested commit concept:

```text
Fix D405 manual exposure for controlled phantom imaging
```

The commit should contain only:

```text
camera-setting constants
manual RealSense exposure/gain initialization
startup read-back/logging
possibly a small camera-setting sanity check
```

Avoid combining this commit with:

```text
ring detector upgrade
residual threshold synchronization
phantom-normal changes
robot-control changes
calibration regeneration
large refactors
```

This makes rollback and debugging straightforward.

---

## 22. Integration Acceptance Criteria

The camera-settings patch can be considered successfully integrated when all of the following are true:

```text
[ ] Auto Exposure is explicitly OFF.
[ ] Exposure reads back as approximately 12000.
[ ] Gain reads back as approximately 16.
[ ] Settings are applied before warm-up.
[ ] Severe broad overexposure on the white phantom is reduced.
[ ] Ring detection remains stable.
[ ] Depth remains valid.
[ ] Existing 15-frame / <3 mm RCM lock remains achievable.
[ ] No calibration artifact is modified merely by camera startup.
[ ] Detector thresholds were not silently changed.
[ ] Robot motion code was not changed as part of this patch.
```

---

## 23. Final Recommended Team Decision

Adopt the following as the standard D405 imaging configuration for the current controlled laboratory setup:

```text
D405_AUTO_EXPOSURE = 0
D405_EXPOSURE      = 12000
D405_GAIN          = 16
```

Treat:

```text
11500–12500
```

as the preferred manual adjustment window if a controlled lighting change requires retuning.

Keep:

```text
12000
```

as the normal default.

The current evidence supports integrating this camera configuration into the existing pipeline without redoing the geometric calibration chain.

Any later detector synchronization, phantom-normal update, or robot-motion modification should be handled as a separate task with its own test and review.
