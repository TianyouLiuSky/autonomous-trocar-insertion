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
  --sequence "output/axis_alignment_sequence_TIMESTAMP.json" \
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
  --orientation output/validation_dataset_orientation_TIMESTAMP.npz
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
