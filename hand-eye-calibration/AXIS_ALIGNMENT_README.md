# XYZ Axis-Alignment Test

## Purpose

This test checks whether the robot's reported translation axes and reported
orientation frame use the same physical base coordinate system.

It is designed for the observed failure where:

- translation-only data and orientation-only data are individually consistent,
- their inferred camera-to-base rotations differ by about 15.6 degrees,
- the disagreement is primarily around robot Y,
- ChArUco reprojection error remains low.

The test moves only one reported translation axis at a time while keeping robot
orientation fixed. Each excursion is bracketed by center captures:

```text
CENTER -> X+ -> CENTER -> X- -> CENTER -> Y+ -> ...
```

The analyzer averages the center samples before and after each excursion. This
reduces sensitivity to slow camera drift, robot drift, and center-return error.

## Scripts

- `run_axis_alignment_poses.py`: generates the sequence and moves the robot.
- `collect_axis_alignment_data.py`: captures synchronized FrameEE and ChArUco
  observations.
- `analyze_axis_alignment.py`: compares the measured movements with a
  camera-to-base rotation derived only from orientation validation.

The default run uses 10 mm excursions and two repeats, producing 25 captures.

## Prerequisites

1. The D405 and board must remain rigidly mounted.
2. The full board should remain visible at all six axis extremes.
3. ROS must provide `/SHER20/eye_robot/FrameEE`.
4. Use the same fitted intrinsics NPZ used for the latest orientation dataset.
5. Keep the previously collected orientation validation NPZ. The axis test
   does not estimate its orientation reference from translation data.

Set the fitted-intrinsics path in both terminals:

```bash
cd ~/autonomous-trocar-insertion/hand-eye-calibration

export INTRINSICS="output/d405_charuco_intrinsics_TIMESTAMP.npz"
test -f "$INTRINSICS" && echo "Using $INTRINSICS"
```

## 1. Start the Motion Script

In terminal 1:

```bash
python3 run_axis_alignment_poses.py \
  --step-mm 10 \
  --repeats 2 \
  --intrinsics "$INTRINSICS" \
  --require-fitted-intrinsics
```

The script first moves to the saved calibration home position. If the home
position lacks 10 mm of clearance on both sides of an axis, it shifts the test
center while remaining inside the configured workspace.

It then saves a sequence JSON under `output/` and prints the exact collector
command for terminal 2. Do not continue the motion script until the collector
is showing a live image.

Use `--current-center` only when deliberately testing around the current pose:

```bash
python3 run_axis_alignment_poses.py \
  --current-center \
  --step-mm 10 \
  --repeats 2 \
  --intrinsics "$INTRINSICS" \
  --require-fitted-intrinsics
```

## 2. Start the Collector

In terminal 2, run the exact command printed by the motion script. It will look
like:

```bash
python3 collect_axis_alignment_data.py \
  --sequence "output/axis_alignment_sequence_20260612_123255.json" \
  --intrinsics "$INTRINSICS" \
  --require-fitted-intrinsics
```

`TIMESTAMP` is a placeholder in this README. Do not type it literally. Use the
actual JSON path printed by `run_axis_alignment_poses.py`, for example:

```text
output/axis_alignment_sequence_20260612_153045.json
```

With the current collector, `--sequence` may also be omitted to select the
newest generated sequence automatically:

```bash
python3 collect_axis_alignment_data.py \
  --intrinsics "$INTRINSICS" \
  --require-fitted-intrinsics
```

Confirm that the collector displays `FITTED` and that the board is detected.
The top status line shows the exact next label, such as:

```text
NEXT: X+ repeat 1
```

For every motion-script prompt:

1. Let the robot finish moving and settling.
2. Confirm that the collector's `NEXT` label matches the motion script.
3. Press Space exactly once in the collector.
4. Confirm the collector printed `Recorded Pose`.
5. Press Enter in the motion terminal to continue.

If the test must stop early, press `Ctrl+S` in the collector to save a partial
dataset. Its sample CSV is useful for inspection, but the analyzer deliberately
rejects partial datasets because the final result requires both directions of
all three axes.

The complete collector output is:

```text
output/axis_alignment_dataset_TIMESTAMP.npz
```

It also updates:

```text
axis_alignment_dataset.npz
```

It does not overwrite `validation_dataset.npz`.

## 3. Analyze the Test

Use the timestamped orientation dataset collected with the same intrinsics:

```bash
python3 analyze_axis_alignment.py \
  --axis-data output/axis_alignment_dataset_TIMESTAMP.npz \
  --orientation output/validation_dataset_orientation_.npz
```

When paths are omitted, the analyzer chooses the most recently modified axis
and orientation datasets:

```bash
python3 analyze_axis_alignment.py
```

Explicit timestamped paths are recommended when documenting an experiment.

## Reading the Results

The analyzer prints each physical translation direction in the
orientation-derived robot base frame:

```text
X: delta=[..., ..., ...] mm, direction=[...], angle=... deg
Y: delta=[..., ..., ...] mm, direction=[...], angle=... deg
Z: delta=[..., ..., ...] mm, direction=[...], angle=... deg
```

For ideal alignment with a 10 mm step:

```text
X delta ~= [10, 0, 0] mm
Y delta ~= [0, 10, 0] mm
Z delta ~= [0, 0, 10] mm
```

Good results should approximately satisfy:

- per-axis direction error below 1 degree,
- cross-axis motion below 0.2-0.3 mm for a 10 mm move,
- scale close to 1.0,
- translation-only fit residual below 0.3 mm,
- robot orientation drift below 0.2 degrees,
- axis and orientation camera-matrix difference equal to 0 px.

If the suspected Y-axis rotation is present, expect:

- Y to remain nearly correct,
- X to contain a substantial Z component,
- Z to contain a substantial X component,
- X and Z direction errors near 15-16 degrees,
- about 2.7 mm cross-axis motion for a 10 mm excursion,
- translation-only versus orientation mapping near 15-16 degrees around Y.

The sign of the X/Z components identifies the direction of the mismatch. Do
not apply the measured angle to robot control until both positive and negative
directions repeat consistently.

## Saved Reports

The analyzer writes:

```text
output/axis_alignment_summary_TIMESTAMP.json
output/axis_alignment_residuals_TIMESTAMP.csv
```

The CSV contains every excursion's:

- actual FrameEE displacement,
- camera-observed displacement transformed using orientation validation,
- X/Y/Z error components,
- direction error,
- cross-axis magnitude,
- robot and board orientation drift.

## Important Interpretation

A low translation-only fit residual by itself does not prove frame agreement.
It only proves that the three translation axes form a repeatable rigid basis.

The decisive result is the comparison between:

1. the camera-to-base rotation required to explain isolated XYZ movement, and
2. the camera-to-base rotation independently obtained from rotational motion.

If these differ by about 15.6 degrees again, the inconsistency exists upstream
of hand-eye solver weighting. If they agree within 1 degree, the previous
spatial dataset or its motion design should be investigated instead.

## Confirmed Robot Finding

The June 12, 2026 axis test measured:

- reported X direction error: `15.975 deg`
- reported Y direction error: `1.038 deg`
- reported Z direction error: `15.424 deg`
- translation/orientation basis disagreement: `15.525 deg` about Y
- translation-only residual: `0.149 mm` mean, `0.299 mm` max
- robot orientation drift: at most `0.001 deg`
- ChArUco reprojection error: `0.175 px` mean

This is a coordinate-basis error, not a constant 15 mm position offset.
FrameEE translation and FrameEE rotation do not behave as components of the
same base coordinate frame.

For a valid rigid transform, the translation vector and rotation matrix must
both be expressed relative to the same parent frame. In this robot's output:

- the quaternion is internally consistent with observed rotational motion,
- XYZ is internally consistent with observed translational motion,
- but the XYZ basis is rotated by approximately `15.5 deg` around Y relative
  to the base frame implied by the quaternion.

The measured direction matrix was approximately:

```text
[[ 0.9614,  0.0169, -0.2652],
 [-0.0035,  0.9998,  0.0201],
 [ 0.2752, -0.0065,  0.9640]]
```

Therefore, a reported `+10 mm` X move physically appears as approximately
`[+9.61, -0.03, +2.75] mm` in the orientation-defined base. A reported
`+10 mm` Z move appears as approximately `[-2.65, +0.20, +9.64] mm`. Y is
nearly aligned.

This explains the earlier X/Z residual pattern. Over a 10 mm move, a
15.5-degree axis error creates about 2.7 mm of cross-axis displacement. It also
explains why solver weights traded translation quality against rotation
quality: no single pair of rigid transforms can exactly satisfy `A Y = X B`
when the rotation and translation inside `A` use different base frames.

The experiments rule out the following as the primary cause:

- ChArUco image noise or D405 repeatability
- fitted versus factory intrinsics
- quaternion `x,y,z,w` ordering
- quaternion inversion
- robot/camera capture timing mismatch
- insufficient solver weighting
- failure to reach requested poses, because recorded actual poses were used

The exact robot-source defect is not yet proven. Plausible locations are the
EyeRobot 2.0 forward-kinematics axis definition, a missing fixed base-frame
rotation on translation, or a machine-specific kinematic calibration. The
robot UI matching FrameEE is not an independent validation because both may
display the same computed pose.

## Experimental Processing Correction

`analyze_axis_alignment.py` now saves an orthonormal correction artifact:

```text
output/translation_axis_correction_TIMESTAMP.npz
```

The matrix maps:

```text
reported FrameEE XYZ -> orientation-defined robot base XYZ
```

It is derived from the axis and orientation datasets. It is not a hard-coded
15.5-degree rotation.

The axis experiment measures directions and therefore determines the basis
rotation, but it cannot determine a constant offset between the two coordinate
origins. This does not affect calibration residuals: a constant base-origin
offset is absorbed into the solved camera translation. Consequently, the
corrected `T_cam2base` is expressed in a coordinate convention with
orientation-aligned axes but an origin whose physical offset has not been
independently measured.

First rerun the analyzer on the confirmed datasets:

```bash
python3 analyze_axis_alignment.py \
  --axis-data output/axis_alignment_dataset_20260612_124957.npz \
  --orientation output/validation_dataset_orientation_20260610_152128.npz
```

Select the newly written correction:

```bash
CORRECTION=$(ls -1t output/translation_axis_correction_*.npz | head -n 1)
test -f "$CORRECTION" && echo "Using $CORRECTION"
```

Then start a new calibration GUI with both fitted intrinsics and the correction:

```bash
python3 handeye_calibration.py \
  --intrinsics "$INTRINSICS" \
  --translation-axis-correction "$CORRECTION"
```

The GUI log must display:

```text
Translation-axis correction ACTIVE
```

Run `run_calibration_poses.py` normally in the other terminal. The correction
does not alter robot targets or commands. It changes only the FrameEE XYZ
vectors supplied to the hand-eye solver. Quaternions are unchanged.

After computing and saving, the calibration NPZ records:

- raw FrameEE poses
- corrected solver poses
- board poses
- the exact correction matrix and its source

Existing calibration NPZ files created before this update cannot be repaired by
rotating `T_cam2base` afterward. Their transforms were already optimized using
incompatible pose components. Collect and solve a new 20-pose calibration with
the correction active. Existing validation datasets may be reused because they
contain raw FrameEE poses.

Evaluate a corrected calibration against the existing decoupled datasets:

```bash
CALIBRATION=$(ls -1t output/hand_eye_cal_*.npz | head -n 1)

python3 evaluate_calibration.py \
  --calib "$CALIBRATION" \
  --validation output/validation_dataset_spatial_20260610_151041.npz \
  --solution weighted \
  --no-show

python3 evaluate_calibration.py \
  --calib "$CALIBRATION" \
  --validation output/validation_dataset_orientation_20260610_152128.npz \
  --solution weighted \
  --no-show
```

`evaluate_calibration.py` detects the correction metadata in the calibration
file and applies the same matrix to validation FrameEE translations
automatically. Its console output must say:

```text
Translation-axis correction: ACTIVE
```

Any later program using the corrected `T_cam2base` must also convert incoming
raw FrameEE XYZ with the saved correction matrix. Mixing a corrected camera
transform with raw FrameEE translations recreates the same inconsistency.
Quaternions must not be rotated by this correction.

Treat this as an experimental software correction until it passes validation.
Do not apply the matrix to robot motion commands. Re-run the axis test after
robot firmware, homing, linkage calibration, or kinematic configuration
changes, because those may change the measured basis relationship.

Suggested acceptance criteria:

- spatial validation mean translation error below `1 mm`
- orientation validation mean rotation error below `0.25 deg`
- no coherent X/Z swirl in the spatial residual map
- repeated corrected calibrations give comparable transforms and residuals
