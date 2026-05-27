# Hand-Eye Calibration Runbook

This folder calibrates the fixed D405/world camera to the SHER robot base using a
ChArUco board mounted on the robot end effector.

The workflow uses two scripts at the same time:

- `handeye_calibration.py` opens the GUI, captures samples, solves `A Y = X B`,
  and saves the calibration result.
- `run_calibration_poses.py` moves the robot through the 20 calibration poses.

The GUI does not move the robot. The motion script does not capture images.
Run them in separate terminals and advance them together.

## Before Running

Start the normal robot stack first. The calibration GUI listens to the robot
pose over ROS, but it opens the D405 directly through `pyrealsense2`, matching
the working `d405_handeye.py` capture path.

The calibration GUI expects:

- Robot pose topic: `/SHER20/eye_robot/FrameEE`
- Direct D405 color stream: `1280 x 720 @ 15 fps`

Quick checks:

```bash
rostopic echo -n 1 /SHER20/eye_robot/FrameEE
rs-enumerate-devices -c
```

Make sure the ChArUco board matches the code:

- Squares: `8 x 6`
- Square length: `10 mm`
- Marker length: `7 mm`
- Dictionary: `DICT_6X6_250`

## Terminal 1: Launch Capture and Calculation GUI

From this folder:

```bash
cd ~/Autonomous-Trocar-Insertion/hand-eye-calibration
python3 handeye_calibration.py
```

Wait until the GUI shows:

- D405 is ready.
- Robot pose is updating.
- The board is detected in the image.

Press `Set Anchor` if you want to visually record the current reference pose in
the GUI log. The actual home position used by the motion scripts is saved by
`run_calibration_poses.py`.

## Terminal 2: Run Calibration Motion

From this folder:

```bash
cd ~/Autonomous-Trocar-Insertion/hand-eye-calibration
python3 run_calibration_poses.py
```

When this script starts, it records the current robot pose as the calibration
home position and overwrites:

```text
home_position/home_position.json
```

This file stores `[x, y, z, roll, pitch, yaw]` in millimeters and degrees. It is
used later by validation so both calibration and validation can start from the
same physical home position.

The script then generates 20 poses:

- XYZ span: `24 x 24 x 24 mm`
- Roll offsets: `[-12, -6, 0, 6, 12] deg`
- Pitch offsets: `[-9, -3, 3, 9] deg`
- Neighboring orientation difference: at least `6 deg`
- Move timeout: `90 s`
- Max angular velocity: `0.05 rad/s`

At each pose, wait for the motion terminal to say the pose is ready. Then:

1. In the GUI, press `SPACE` or click `Record`.
2. Confirm the sample count increased.
3. Return to the motion terminal and press `Enter`.

If the motion script reports that a pose did not fully reach the target, skip it
unless you have a clear reason to keep it. Bad pose/image pairs are worse than
having fewer samples.

After the last pose, the motion script returns the robot to the saved home
position.

The motion script also writes diagnostic CSV logs under:

```text
motion_logs/
```

Use `calibration_motion_samples_*.csv` to inspect the live position and
orientation residual during each attempted move. This is useful if the robot
appears to stop rotating before reaching a target. Use
`calibration_motion_summary_*.csv` for one final residual row per attempt.

## Compute and Save

In `handeye_calibration.py`, press `Compute Calibration` after collecting enough
accepted samples. The current motion script is designed for 20 samples.

The solver estimates:

- `T_cam2base`
- `T_board2gripper`

Then press `Save (.npz)`. Results are written under:

```text
output/
```

The pose log can also be saved as CSV with `Save Poses (.csv)`.

## Validation

To collect validation data, run the validation collector GUI first:

```bash
python3 collect_validation_data.py
```

Then run the validation motion script in another terminal:

```bash
python3 run_validation_24mm.py
```

`run_validation_24mm.py` loads:

```text
home_position/home_position.json
```

and asks before moving the robot back to that saved home. Use this path whenever
you want validation to be centered on the same physical home used for
calibration.

The validation set has 27 poses over a `24 x 24 x 24 mm` grid with smaller
orientation variation. It is a holdout check, not the calibration solve itself.

After validation data is saved, evaluate:

```bash
python3 evaluate_calibration.py
```

## Generated Files

Generated files are ignored by `.gitignore`, including:

- `output/*.npz`
- `output/*.csv`
- `validation_dataset.npz`
- `hand_eye_calibration.npz`
- `spatial_error_map_*.png`
- `home_position/home_position.json`
- `motion_logs/*.csv`
- `__pycache__/`

## Troubleshooting

If the GUI cannot see the camera, make sure no other process owns the D405 color
stream. The GUI now opens the camera directly, so do not run the ROS D405 camera
publisher or another direct RealSense script at the same time.

If direct RealSense scripts fail with `Couldn't resolve requests`, the requested
stream profile is not available. Check:

```bash
rs-enumerate-devices -c
```

If the board is detected only partially, move the board or camera until the axes
draw reliably. Avoid recording samples with unstable board detection.

If calibration error is bad, first suspect bad sample pairing:

- Robot had not settled before pressing `SPACE`.
- The wrong pose was recorded in the GUI.
- The board pose was noisy or partially detected.
- Validation started from a different home position.

For reliable runs, always let the motion script tell you when to record, and use
the saved home position for validation.

If the robot repeatedly fails on rotation while translation behaves normally,
run the rotation-only limit test:

```bash
cd ~/Autonomous-Trocar-Insertion/motion_script
python3 test_rotation_limits.py --robot-name SHER20 --max-offset-deg 35
```

This publishes angular velocity only, records when each roll/pitch sweep stops
reaching the target, and writes logs under `motion_script/rotation_test_logs/`.
