# Frame Consistency and Intrinsics Diagnostics

Use this workflow after collecting both decoupled validation datasets. It tests
whether robot translation, robot rotation, and camera pose measurements can be
explained by one rigid camera-to-base transform.

The scripts are diagnostic. They do not apply a fitted angular correction to
the saved hand-eye calibration.

## 1. Collect Both Validation Sets

Collect spatial validation:

```bash
python3 collect_validation_data.py -s
python3 run_validation_tests.py -s
```

Then collect orientation validation:

```bash
python3 collect_validation_data.py -o
python3 run_validation_tests.py -o
```

This produces:

```text
validation_dataset_spatial.npz
validation_dataset_orientation.npz
```

## 2. Run the Frame-Consistency Diagnostic

```bash
python3 diagnose_frame_consistency.py
```

The script estimates `R_cam2base` in two independent ways:

- Spatial estimate: rigidly aligns changes in ChArUco translation with changes
  in FrameEE translation while orientation is fixed.
- Orientation estimate: solves the relative-motion rotation equation while XYZ
  is fixed.

It also evaluates these FrameEE interpretations:

- `xyzw`: quaternion is `[x, y, z, w]` and maps end effector to base.
- `xyzw_inverse`: the same quaternion interpreted in the inverse direction.
- `wxyz_source`: the raw message was actually `[w, x, y, z]`.
- `wxyz_source_inverse`: reordered and inverted.

The standard SHER interpretation should be `xyzw`. Treat an alternative with a
smaller disagreement as evidence to inspect the FrameEE publisher, not as
permission to silently change conventions.

Outputs:

```text
output/frame_consistency_summary_*.json
output/frame_consistency_spatial_residuals_*.csv
output/frame_consistency_pairwise_rotations_*.csv
```

Interpretation:

- Below `0.5 deg`: strong agreement.
- `0.5-1.0 deg`: usable but worth monitoring.
- Above `1.0 deg`: investigate frames before tuning solver weights.
- A disagreement axis near `[0, 1, 0]` means the conflict is primarily around
  robot base Y.

The pairwise CSV compares the magnitude of each robot relative rotation with
the corresponding board relative rotation. Similar magnitudes but a bad final
frame disagreement suggests an axis/frame convention issue rather than angular
scale.

## 3. Audit FrameEE

Inspect the code that publishes:

```text
/SHER20/eye_robot/FrameEE
```

Confirm all of the following:

1. Translation and quaternion refer to the same end-effector origin.
2. Translation and quaternion use the same robot base axes.
3. Quaternion fields are published as `x, y, z, w`.
4. Rotation maps end-effector coordinates into robot base coordinates.
5. The translation is not for a tool tip while rotation is for an upstream
   wrist frame.

If the spatial and orientation estimates differ by a nearly fixed angle, check
for a static wrist-to-tool rotation that was applied to only one part of the
published transform.

## 4. Calibrate D405 Intrinsics

Capture at least 40 views:

```bash
python3 calibrate_d405_intrinsics.py --capture --samples 40
```

Move the board throughout the entire color image. Include:

- center, every edge, and all four corners,
- several distances,
- positive and negative roll/pitch tilts,
- views where the board occupies both large and small image areas.

Do not collect 40 nearly identical frontal images.

After closing the capture window, calibrate:

```bash
python3 calibrate_d405_intrinsics.py --calibrate --min-views 20
```

The output reports fitted `fx`, `fy`, `cx`, `cy`, distortion, per-view
reprojection error, and differences from the D405 factory intrinsics. It saves:

```text
output/d405_charuco_intrinsics_*.npz
output/d405_charuco_intrinsics_*.json
output/d405_charuco_intrinsics_views_*.csv
```

## 5. Repeat Calibration With Fitted Intrinsics

Use the same fitted file for calibration and both validation collectors:

```bash
python3 handeye_calibration.py \
  --intrinsics output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz
```

```bash
python3 collect_validation_data.py -s \
  --intrinsics output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz
```

```bash
python3 collect_validation_data.py -o \
  --intrinsics output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz
```

The evaluator prints a warning if calibration and validation used different
camera matrices. The fitted file must match the `1280 x 720` color stream.

## 6. Solver Diagnostics

`handeye_calibration.py` now:

1. Initializes rotations from relative motion.
2. Solves translation linearly with those rotations fixed.
3. Runs robust joint least squares from deterministic multiple starts.
4. Saves the legacy solution for comparison.
5. Saves conditioning and multi-start transform spread.

Useful saved fields include:

```text
solver_jacobian_condition
rotation_nullspace_gap
translation_condition_number
multistart_camera_rotation_spread_deg
multistart_camera_translation_spread_mm
```

Near-zero multi-start spread indicates the optimizer is consistently finding
the same solution. It does not prove the upstream measurements are mutually
consistent; run `diagnose_frame_consistency.py` for that check.
