# Decoupled Hand-Eye Validation Tests

Use these tests after a normal hand-eye calibration when the residual map is
hard to interpret. The goal is to separate position error from orientation
error instead of moving XYZ and RPY together.

There are two validation modes:

- Spatial validation: move through a `3 x 3 x 3` XYZ grid while keeping one
  constant orientation.
- Orientation validation: keep one fixed XYZ position while sweeping a `5 x 5`
  roll/pitch grid.

Both tests still use two windows:

- `collect_validation_data.py` captures D405/ChArUco and FrameEE samples.
- `run_validation_tests.py` moves the robot, then evaluates the saved dataset.

## Before Starting

Run a normal calibration first and save the weighted result from
`handeye_calibration.py`. The calibration file should look like:

```text
output/hand_eye_cal_07JUN2026_173734.npz
```

The `run_validation_tests.py` script uses the newest saved calibration by
default. To use a specific calibration weight timeframe, pass the timestamp with
`-t`, for example:

```bash
-t 07JUN2026_173734
```

The script also accepts a full `.npz` path.

## Spatial Validation

This test asks: if orientation is held constant, does the camera/robot
translation agreement change across XYZ?

Terminal 1, collector:

```bash
cd ~/autonomous-trocar-insertion/hand-eye-calibration
python3 collect_validation_data.py -s
```

Terminal 2, robot motion and evaluation:

```bash
cd ~/autonomous-trocar-insertion/hand-eye-calibration
python3 run_validation_tests.py -s
```

To evaluate with a specific weighted calibration timeframe:

```bash
python3 run_validation_tests.py -s -t 07JUN2026_173734
```

Expected samples: `27`.

All targets use the same RPY. If this test is bad, suspect spatial effects:

- camera intrinsics or distortion,
- robot translation frame convention,
- base-to-camera translation/rotation,
- depth/scale error in ChArUco pose estimation,
- workspace-dependent robot position reporting.

## Orientation Validation

This test asks: if XYZ is held constant, does rotating the robot change the
agreement between the camera chain and robot chain?

Terminal 1, collector:

```bash
cd ~/autonomous-trocar-insertion/hand-eye-calibration
python3 collect_validation_data.py -o
```

Terminal 2, robot motion and evaluation:

```bash
cd ~/autonomous-trocar-insertion/hand-eye-calibration
python3 run_validation_tests.py -o
```

To evaluate with a specific weighted calibration timeframe:

```bash
python3 run_validation_tests.py -o -t 07JUN2026_173734
```

Expected samples: `25`.

All targets use the same XYZ. If this test is bad, suspect orientation or tool
offset effects:

- board-to-gripper transform,
- FrameEE quaternion convention,
- TCP/tool-center definition,
- subtle board mount flex under rotation,
- Euler command convention mismatch.

## Output Files

The collector saves:

```text
validation_dataset.npz
validation_dataset_spatial.npz
validation_dataset_orientation.npz
output/validation_dataset_spatial_*.npz
output/validation_dataset_orientation_*.npz
output/validation_samples_spatial_*.csv
output/validation_samples_orientation_*.csv
```

The motion script saves:

```text
output/validation_motion_spatial_*.csv
output/validation_motion_orientation_*.csv
```

The evaluator saves mode-specific residuals and plots:

```text
output/validation_residuals_weighted_spatial_*.csv
output/validation_residuals_weighted_orientation_*.csv
output/spatial_error_map_weighted_spatial_*.png
output/orientation_error_map_weighted_orientation_*.png
```

## Manual Evaluation

If automatic evaluation is skipped or the collector saves after the motion
script exits, run evaluation manually:

```bash
python3 evaluate_calibration.py \
  --calib output/hand_eye_cal_07JUN2026_173734.npz \
  --validation output/validation_dataset_spatial_YYYYMMDD_HHMMSS.npz \
  --solution weighted \
  --no-show
```

For orientation:

```bash
python3 evaluate_calibration.py \
  --calib output/hand_eye_cal_07JUN2026_173734.npz \
  --validation output/validation_dataset_orientation_YYYYMMDD_HHMMSS.npz \
  --solution weighted \
  --no-show
```

To compare against the unweighted baseline saved in the same calibration file,
replace `--solution weighted` with:

```bash
--solution legacy
```

## How To Interpret The Results

The evaluator compares two predicted board poses in the robot base frame:

```text
robot chain:  A @ Y
camera chain: X @ B
```

Where:

- `A` is the FrameEE robot pose.
- `Y` is board-to-gripper.
- `X` is camera-to-base.
- `B` is board-to-camera from ChArUco.

Translation residual:

```text
camera-chain board position - robot-chain board position
```

So a `+Z` arrow means the camera chain thinks the board is higher than the
robot chain. A `-X` arrow means the camera chain thinks the board is farther in
the robot negative-X direction than the robot chain.

Rotation residual is an axis-angle vector. Its magnitude is the rotation error
in degrees. Its direction is the axis around which the robot-chain board
orientation would rotate to match the camera-chain board orientation.

Red star: first validation sample.
Yellow star: largest residual sample.
Red circles: samples above `mean + std`; these are visual outliers, not
automatically bad captures.

## Recommended Debug Order

1. Run spatial validation.
2. Run orientation validation using the same calibration file.
3. Compare weighted and legacy on both datasets.
4. If spatial is bad and orientation is good, focus on camera/robot translation
   geometry.
5. If orientation is bad and spatial is good, focus on board-to-gripper, TCP,
   and quaternion/Euler conventions.
6. If both are good but the old combined validation is bad, the previous
   combined pose set was confounding XYZ with roll/pitch.
