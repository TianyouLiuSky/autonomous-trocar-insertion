# Next Experiments for the 15.8 Degree Frame Inconsistency

This runbook investigates the current hand-eye result:

- Spatial data fit one rigid mapping with `0.121 mm` mean residual.
- Orientation data fit one rigid mapping with `0.099 deg` mean residual.
- The two mappings disagree by `15.792 deg`, almost entirely around robot
  base Y.
- The standard FrameEE interpretation, quaternion `[x, y, z, w]` mapping the
  end effector into the robot base, remains the best tested convention.

The immediate goal is to determine whether the disagreement comes from D405
intrinsics or from the robot's `FrameEE` definition. Do not apply the measured
`15.792 deg` disagreement as a calibration correction.

## Experimental Rules

Keep these conditions unchanged throughout the experiments:

1. Do not remount the camera or ChArUco board.
2. Use the same `1280 x 720` D405 color stream.
3. Use the same physical ChArUco board:
   - `8 x 6` squares
   - `10 mm` square length
   - `7 mm` marker length
   - `DICT_6X6_250`
4. Let the robot settle before every capture.
5. Capture exactly once at each accepted motion pose.
6. Do not manually reposition the robot between scripted poses.
7. Skip a failed pose unless there is a clear reason to record it.
8. Record the timestamped NPZ filename printed by each collector.

Run all commands from:

```bash
cd ~/autonomous-trocar-insertion/hand-eye-calibration
```

The GUI needs a working display. If `QT_QPA_PLATFORM` was previously forced to
`offscreen`, restore normal display behavior before launching it:

```bash
unset QT_QPA_PLATFORM
echo "$DISPLAY"
```

Only one process may own the D405 at a time.

## Experiment 1: Fresh Factory-Intrinsics Baseline

This experiment repeats the raw frame-consistency test using newly captured
data. It does not require a new hand-eye calibration because the diagnostic
uses raw robot and board poses.

### 1A. Capture Spatial Data

In Terminal 1, launch the collector without `--intrinsics`. This selects the
D405 factory intrinsics:

```bash
python3 collect_validation_data.py -s
```

Confirm that it prints:

```text
Camera intrinsics: D405 factory
```

In Terminal 2, launch the spatial motion:

```bash
python3 run_validation_tests.py -s --no-evaluate
```

For every pose:

1. Let the motion script reach and settle at the pose.
2. Capture once in the collector UI.
3. Confirm that the collector count increased.
4. Return to the motion terminal and continue.

At the end, copy the timestamped path printed by the collector:

```text
output/validation_dataset_spatial_YYYYMMDD_HHMMSS.npz
```

Do not use `validation_dataset_spatial.npz` for the final comparison. That file
is a mutable latest-dataset alias.

### 1B. Capture Orientation Data

Close the spatial collector before starting the orientation collector.

Terminal 1:

```bash
python3 collect_validation_data.py -o
```

Again confirm:

```text
Camera intrinsics: D405 factory
```

Terminal 2:

```bash
python3 run_validation_tests.py -o --no-evaluate
```

Record the timestamped orientation path:

```text
output/validation_dataset_orientation_YYYYMMDD_HHMMSS.npz
```

### 1C. Diagnose the Fresh Factory Data

Replace both placeholders with the exact paths printed by the collectors:

```bash
python3 diagnose_frame_consistency.py \
  --spatial output/validation_dataset_spatial_FACTORY_TIME.npz \
  --orientation output/validation_dataset_orientation_FACTORY_TIME.npz
```

Save the reported:

- spatial translation residual mean and maximum,
- `xyzw` disagreement angle and axis,
- orientation residual mean and maximum,
- spatial and orientation SHA-256 prefixes,
- summary JSON filename.

Expected healthy targets are:

```text
spatial residual mean:       below 0.5 mm
orientation residual mean:   below 0.25 deg
xyzw disagreement:           below 1.0 deg
ideal xyzw disagreement:     below 0.5 deg
```

The first two values measure internal repeatability. The disagreement measures
whether translation and rotation imply the same camera-to-base frame.

## Experiment 2: Fit D405 Color Intrinsics

Use a fresh capture directory so old intrinsic images cannot be mixed into this
fit:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
CAPTURE_DIR="output/intrinsics_capture_${RUN_ID}"
python3 calibrate_d405_intrinsics.py \
  --capture \
  --samples 40 \
  --capture-dir "$CAPTURE_DIR"
```

Collect views that cover:

- image center, all four edges, and all four corners,
- several camera-to-board distances,
- positive and negative board roll and pitch,
- both large and small board footprints in the image.

Do not collect 40 nearly identical frontal views. Keep the complete board
visible when practical, and reject strongly blurred images.

After capture, fit the intrinsics from that same directory:

```bash
python3 calibrate_d405_intrinsics.py \
  --calibrate \
  --min-views 20 \
  --capture-dir "$CAPTURE_DIR"
```

The script saves:

```text
output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz
output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.json
output/d405_charuco_intrinsics_views_YYYYMMDD_HHMMSS.csv
```

Do not proceed to Experiment 3 if the calibration prints either coverage
warning. The intended minimum coverage is:

```text
board-center horizontal span: at least 0.50 of image width
board-center vertical span:   at least 0.40 of image height
largest/smallest board area:  at least 1.5x
```

If coverage fails, create another fresh capture directory and deliberately
place the board near the left edge, right edge, top, bottom, and all four
corners. The board does not need to remain centered. Include near and far
views, but keep enough ChArUco corners visible for a reliable detection.

You may instead append edge and corner views to the existing capture directory.
For example, increase a 40-view set to 60 views:

```bash
CAPTURE_DIR="$(ls -dt output/intrinsics_capture_* | head -n 1)"
echo "Adding views to $CAPTURE_DIR"
python3 calibrate_d405_intrinsics.py \
  --capture \
  --samples 60 \
  --capture-dir "$CAPTURE_DIR"
python3 calibrate_d405_intrinsics.py \
  --calibrate \
  --min-views 20 \
  --capture-dir "$CAPTURE_DIR"
```

Use the new NPZ printed by this second calibration, not the earlier
coverage-warning NPZ.

Set the exact fitted file for the remaining commands:

```bash
INTRINSICS="output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz"
test -f "$INTRINSICS" && echo "Using $INTRINSICS"
```

Shell variables are local to one terminal. Run this assignment in every
terminal that uses `$INTRINSICS`, or replace `$INTRINSICS` with the literal NPZ
path in each command.

Check the calibration report for:

- image size `1280 x 720`,
- fitted `fx`, `fy`, `cx`, and `cy`,
- distortion coefficients,
- mean and maximum per-view reprojection errors,
- differences from the factory intrinsics,
- individual high-error views.

A low reprojection error is necessary, but the decisive test is whether these
intrinsics make the spatial and orientation frame estimates agree.

## Experiment 3: Repeat Raw Diagnostics With Fitted Intrinsics

Use the same fitted NPZ for both collectors.

This experiment must create two new datasets after the intrinsic NPZ was
created. Do not reuse the factory datasets from Experiment 1. Their saved
intrinsics source will remain `D405 factory`, even if a fitted NPZ is created
later.

### 3A. Fitted-Intrinsics Spatial Data

Terminal 1:

```bash
INTRINSICS="output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz"
python3 collect_validation_data.py \
  -s \
  --intrinsics "$INTRINSICS" \
  --require-fitted-intrinsics
```

Confirm that the printed camera-intrinsics source is the absolute path to the
fitted NPZ, not `D405 factory`. The collector window also displays
`Intrinsics: FITTED`. Stop immediately if it displays `FACTORY`.

Terminal 2:

```bash
python3 run_validation_tests.py -s --no-evaluate
```

Record the new timestamped spatial NPZ.

### 3B. Fitted-Intrinsics Orientation Data

Terminal 1:

```bash
INTRINSICS="output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz"
test -f "$INTRINSICS" || {
  echo "Intrinsics file is missing: $INTRINSICS"
  exit 1
}
python3 collect_validation_data.py \
  -o \
  --intrinsics "$INTRINSICS" \
  --require-fitted-intrinsics
```

Replace the timestamp before running this block. Shell variables do not carry
into a newly opened terminal. To inspect the value before launching:

```bash
printf 'INTRINSICS=<%s>\n' "$INTRINSICS"
```

An output of `INTRINSICS=<>` means the variable is unset. Passing that empty
value causes the current working directory to be interpreted as the path.
Using the literal path is also valid:

```bash
python3 collect_validation_data.py -o \
  --intrinsics output/d405_charuco_intrinsics_YYYYMMDD_HHMMSS.npz \
  --require-fitted-intrinsics
```

Terminal 2:

```bash
python3 run_validation_tests.py -o --no-evaluate
```

Record the new timestamped orientation NPZ.

### 3C. Diagnose the Fitted-Intrinsics Data

```bash
python3 diagnose_frame_consistency.py \
  --spatial output/validation_dataset_spatial_FITTED_TIME.npz \
  --orientation output/validation_dataset_orientation_FITTED_TIME.npz
```

Verify that the diagnostic reports the same fitted intrinsics path for both
datasets. It should not print `not recorded`.

## Experiment 4: Compare Factory and Fitted Results

Use the following interpretation:

| Factory result | Fitted result | Likely conclusion |
| --- | --- | --- |
| About `15.8 deg` | Below `1 deg` | Factory intrinsics were the main cause |
| About `15.8 deg` | About `15.8 deg`, same Y axis | Audit the FrameEE publisher |
| Above `1 deg` | Improved but still above `1 deg` | Intrinsics contribute, but another frame issue remains |
| Below `1 deg` | Below `1 deg` | Earlier datasets were stale or otherwise unrepresentative |
| Angle or axis changes greatly between repeats | Unstable | Check capture synchronization, board rigidity, and intrinsic view quality |

Do not select an intrinsic model merely because it gives a lower ChArUco
reprojection error. Prefer the model that also reduces the independent
translation-versus-orientation disagreement on fresh data.

## Experiment 5: Full Calibration With Fitted Intrinsics

Only run this phase if the fitted-intrinsics raw diagnostic is promising, or if
you need a complete fitted-intrinsics validation result for comparison.

### 5A. Collect a New Hand-Eye Calibration

Terminal 1:

```bash
python3 handeye_calibration.py --intrinsics "$INTRINSICS"
```

Confirm the GUI reports the fitted intrinsic path.

Terminal 2:

```bash
python3 run_calibration_poses.py
```

The motion script overwrites:

```text
home_position/home_position.json
```

This is intentional: the new validation runs should return to the home used by
this calibration.

At each accepted pose:

1. Wait for the motion script to finish settling.
2. Press `SPACE` or `Record` once in the calibration GUI.
3. Confirm that the GUI sample count increased.
4. Continue from the motion terminal.

After 20 accepted samples:

1. Click `Compute Calibration`.
2. Review the translation and rotation training residuals.
3. Click `Save (.npz)`.
4. Record the exact `output/hand_eye_cal_*.npz` path.

For the commands below:

```bash
CALIBRATION="output/hand_eye_cal_YYYYMMDD_HHMMSS.npz"
test -f "$CALIBRATION" && echo "Using $CALIBRATION"
```

Run this assignment in the validation motion terminal, or replace
`$CALIBRATION` with the literal calibration NPZ path.

### 5B. Validate the New Calibration Spatially

Terminal 1:

```bash
python3 collect_validation_data.py -s --intrinsics "$INTRINSICS"
```

Terminal 2:

```bash
python3 run_validation_tests.py \
  -s \
  -t "$CALIBRATION" \
  --solution weighted
```

Record the validation-dataset, residual-CSV, and spatial-map filenames.

### 5C. Validate the New Calibration Orientationally

Terminal 1:

```bash
python3 collect_validation_data.py -o --intrinsics "$INTRINSICS"
```

Terminal 2:

```bash
python3 run_validation_tests.py \
  -o \
  -t "$CALIBRATION" \
  --solution weighted
```

Record the validation-dataset, residual-CSV, and orientation-map filenames.

Run the frame-consistency diagnostic once more with the exact timestamped
validation datasets:

```bash
python3 diagnose_frame_consistency.py \
  --spatial output/validation_dataset_spatial_FINAL_TIME.npz \
  --orientation output/validation_dataset_orientation_FINAL_TIME.npz
```

Target final performance:

```text
translation/orientation frame disagreement:  below 1.0 deg
spatial validation mean translation error:   below about 1.0 mm
orientation validation mean rotation error:  below about 0.25 deg
repeat calibrations without remounting:       comparable transforms/results
```

## Experiment 6: Audit FrameEE if the 15.8 Degree Error Remains

If both factory and fitted-intrinsics datasets show approximately the same
`15.8 deg` disagreement around base Y, stop changing solver weights. Identify
the node that publishes FrameEE:

```bash
rostopic info /SHER20/eye_robot/FrameEE
```

Use the publishing node name from that output:

```bash
rosnode info /NAME_OF_PUBLISHING_NODE
```

Inspect one message:

```bash
rostopic echo -n 1 /SHER20/eye_robot/FrameEE
```

In the publisher source, trace the assignments to:

```text
translation.x
translation.y
translation.z
rotation.x
rotation.y
rotation.z
rotation.w
```

Confirm:

1. Translation and quaternion describe one rigid end-effector frame.
2. Both are expressed relative to the same robot base axes.
3. The quaternion order is `[x, y, z, w]`.
4. The quaternion maps end-effector coordinates into base coordinates.
5. No static wrist-to-tool or tool-to-tip rotation is applied to only one part
   of the transform.
6. Translation is not converted through a second coordinate convention before
   publication.

The orientation validation already keeps reported XYZ nearly constant while
changing roll and pitch. Therefore, the useful source-code question is not
whether the robot can execute rotation. It is whether the published translation
axes and published orientation axes define the same base coordinate system.

Do not change quaternion order or invert the quaternion based on the current
diagnostic: those hypotheses were substantially worse than standard `xyzw`.

## Experiment 7: No-Contact Board-Center Arc Test

If a physical pivot test is incompatible with force control, rerun the updated
frame-consistency diagnostic on the existing fitted-intrinsics datasets:

```bash
python3 diagnose_frame_consistency.py \
  --spatial output/validation_dataset_spatial_FITTED_TIME.npz \
  --orientation output/validation_dataset_orientation_FITTED_TIME.npz
```

No new captures are required. During the orientation sweep, FrameEE XYZ stays
nearly fixed, but the board center moves because it has a physical offset from
the end-effector origin. The diagnostic fits this board-center arc using:

- FrameEE rotations,
- FrameEE translations,
- ChArUco translations,
- an unknown board-center offset solved from the data.

It does not use ChArUco board rotations to estimate the arc rotation. Record:

```text
Orientation board-center arc residual mean/max:
Fitted board-center offset in EE:
Arc rotation vs spatial translation:
Arc rotation vs orientation rotation:
Optimization spread/condition:
```

Interpretation:

| Arc result | Most likely area to investigate |
| --- | --- |
| Low residual; close to orientation rotation; far from spatial translation | FrameEE XYZ coordinate basis or spatial translation publication |
| Low residual; close to spatial translation; far from orientation rotation | FrameEE orientation or ChArUco board-rotation pathway |
| Close to both | Original disagreement should also be small; verify selected datasets |
| Far from both or high residual | Board rigidity/model, synchronization, or insufficient arc conditioning |

Compare the fitted board-center offset norm with a rough physical measurement
from the reported end-effector/force-sensor origin to the board center. It does
not need ruler-level precision, but a physically impossible value makes the arc
fit suspect.

## Results to Return for Analysis

For each factory and fitted run, keep:

```text
Spatial dataset path:
Spatial dataset SHA-256 prefix:
Orientation dataset path:
Orientation dataset SHA-256 prefix:
Intrinsics source:
Spatial fit mean/max (mm):
Orientation fit mean/max (deg):
xyzw disagreement angle (deg):
Disagreement axis:
Summary JSON path:
```

Also retain:

- fitted-intrinsics JSON and per-view CSV,
- calibration NPZ used for validation,
- spatial and orientation residual CSVs,
- motion logs,
- spatial and orientation error maps.

The dataset hashes printed by `diagnose_frame_consistency.py` are important. If
two reported experiments have the same hashes, they used the same captured
data and are not independent repeats.
