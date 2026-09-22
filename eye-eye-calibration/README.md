# Leica–D405 eye-eye calibration

This workflow estimates the rigid transform from the **Leica RGB optical frame**
to the **Intel D405 color optical frame**, using the composite board already
printed for the project. It does not require the robot to move. It does not
calibrate the older FLIR stereo pair or recover 3D from a single Leica pixel.

The existing board PDF is
`composite_board_output/composite_board_8tags_02APR2026.pdf`. Do not print a new
pattern or change its dictionaries to use this implementation.

## What was corrected

- Removed guessed focal lengths and zero-distortion placeholders. Both cameras
  require real intrinsics, with an exact image-resolution check.
- Corrected the central ChArUco origin: **(30, 30) mm** from the composite board's
  top-left, rather than its center at (42, 42) mm.
- Replaced independent `IPPE_SQUARE` tag poses and rotation-vector averaging
  with a **joint planar pose fit to all visible tag corners in the board frame**.
  The generic IPPE solver accepts the board-coordinate corner layout. Both
  planar solutions are checked; nearly indistinguishable, materially different
  rotations are rejected.
- Added timestamp pairing, stale/duplicate-frame rejection, reprojection checks,
  raw-image recording, explicit transform directions, and an offline solver.
- Kept fitting captures separate from held-out validation captures. Validation
  observations never enter the fitted transform. Calibration uses raw poses.
- Both historical entry points, `composite_board_tracking.py` and
  `board_tracking_filtered.py`, now forward to this workflow. The old PyQt
  display, smoothing slider, and screenshot-only S key are replaced by the
  OpenCV capture windows and **C/V/Q** controls below.

## 1. Prepare the physical setup

1. Secure the D405 and Leica in the intended operating positions. Check that
   neither mount can shift. Both cameras must see the **same rigid board face**.
2. Select the actual Leica zoom/magnification, objective/working distance, focus,
   video resolution, and any capture crop. Record those settings. Keep them
   unchanged from intrinsic calibration through registration and use.
3. Use the existing print on a rigid, flat backing. Avoid bending, glossy glare,
   covered corners, and transparent covers that reflect illumination.
4. Measure the target with a ruler/calipers. Nominal dimensions are:

   | Feature | Dimension / layout |
   |---|---|
   | Entire square | 84 × 84 mm |
   | Center ChArUco | 24 × 24 mm; 6 × 6 squares, each 4 mm |
   | ChArUco markers | 2.8 mm; `DICT_6X6_250` |
   | AprilTag faces, including black border | 24 × 24 mm; `DICT_APRILTAG_36h11` |
   | Cell pitch | 28 mm |
   | AprilTag ID layout | top row: 0, 4, 1; middle: 5, ChArUco, 6; bottom: 2, 7, 3 |

   The archived PDF has small raster/DPI rounding (about 83.95 mm overall).
   The model uses the original nominal dimensions. This is appropriate for
   initial millimetre-scale testing, but is **not evidence of 0.05 mm accuracy**.
   A visibly rescaled, stretched, or warped print needs measured geometry or
   replacement; software cannot infer the true print scale from these images.
5. Position the central pattern in the Leica field of view and at least two
   surrounding AprilTags in the D405 color view. More well-separated tags are
   preferable. The cameras do not need to see the same individual markers.
6. Adjust illumination and exposure so the black/white boundaries are sharp.
   Move only the target during collection; keep it motionless for a second or
   two before each capture. There is no automatic board-motion detector.

Changing zoom/focus/cropping can invalidate intrinsics. Moving either camera
relative to the other invalidates registration. After a microscope move,
re-register; a different transform is then expected.

## 2. Prepare the software

Use the Linux lab computer with **ROS1 and working Python3 `cv_bridge`**. The
offline commands also work without ROS. These scripts require the modern
`cv2.aruco.CharucoDetector` API and were tested with OpenCV 4.10.0.

Source the lab's ROS environment and camera workspace in every terminal. Use
the workspace actually installed on that machine; this repository does not
contain the launch packages. Melodic's default Python2 `cv_bridge` is not enough:
use the lab's Python3 build, or an appropriate ROS1 Python3 installation.

From this directory (replace the checkout path for the lab computer):

```bash
cd /path/to/Autonomous-Trocar-Insertion/eye-eye-calibration
python3 -c "import rospy, cv_bridge, message_filters, cv2; print(cv2.__version__); print(hasattr(cv2.aruco, 'CharucoDetector'))"
python3 eye_eye_calibration.py --help
```

If a separate environment is needed, create one using the same Python version
as the working ROS Python3 installation:

```bash
python3 -m venv --system-site-packages .venv-eye-eye
source .venv-eye-eye/bin/activate
python -m pip install -r requirements.txt
```

Repeat the import check using `python`. Use that interpreter for all commands
below. Avoid multiple pip OpenCV distributions in one environment; `cv2` must
resolve to the version with the ChArUco API. NumPy is kept below version 2 for
compatibility with older ROS extension modules. Do not replace the lab's global
Python or ROS packages just to run this script.

## 3. Start and check the two cameras

Use the camera launch commands documented in `../unified_camera_display.py`,
each in a separate sourced terminal:

```bash
roslaunch gscam gscam_decklink.launch
```

```bash
roslaunch realsense2_camera d405_eyerobot_tianle.launch
```

Use the lab's existing ROS master; start `roscore` only if needed. The FLIR driver
and robot controller are not needed for this calibration. Do not run another
RealSense Python process against the same D405 at the same time.

Check the topic names and streaming rates:

```bash
rostopic list
rostopic hz /decklink/camera/image_raw
rostopic hz /d405/color/image_raw
rostopic echo -n 1 /d405/color/camera_info
```

Stop each `rostopic hz` with Ctrl+C before entering another command. Header
timestamps must be nonzero and use a common ROS clock. If cameras publish from
different machines, synchronize those machines' clocks. The collector allows
50 ms between image timestamps by default and rejects images over 1 second old.
This is approximate pairing, **not hardware synchronization**; the stationary
target is necessary. No silent fallback to arrival timestamps is used.

The custom publisher in `EyeRobot/Publisher/intel_d405_publisher.py` uses
`/realsense/color/...` instead. If that is your active driver, override both the
image and CameraInfo topic names in the commands below. Prefer the established
lab driver and verify that CameraInfo describes the actual raw color stream.

## 4. Obtain camera intrinsics

### D405 color

Export the **color** CameraInfo, not depth/infrared CameraInfo:

```bash
python3 eye_eye_calibration.py export-camera-info \
  --topic /d405/color/camera_info \
  --image-topic /d405/color/image_raw \
  --output runs/d405_color_intrinsics.json
```

This saves K, distortion, resolution, source topic, and reported ROS frame ID.
It rejects unsupported distortion models and cropped/binned CameraInfo. The
image subscription also supports drivers that publish CameraInfo only while
an image subscriber is connected. The intrinsics are for the raw stream;
do not substitute a rectified, resized, rotated, mirrored, or cropped stream.

### Leica

The existing `../leica-camera-calibration/output/calibration_leica_08APR2026.json`
can be used **only if its 1920×1080 resolution and all optical settings match**
the current setup. Its recorded fitting RMS is about 0.905 px; that number alone
does not establish current validity. The old intrinsic calibration script is
configured for a different 21 mm / 3.5 mm / `DICT_4X4_250` target, so do not run
that script unchanged on this composite print.

If settings are uncertain, recalibrate using the new commands, which explicitly
match the existing 24 mm composite target:

```bash
python3 eye_eye_calibration.py record-intrinsics \
  --topic /decklink/camera/image_raw \
  --output runs/leica_intrinsic_images_v1 \
  --notes "Leica zoom=RECORD_VALUE; focus=RECORD_VALUE; working distance=RECORD_VALUE; resolution=1920x1080"
```

Press **C** for each raw image; **Q** exits. Collect about **30–40 distinct views**:
vary position within the image and tilt around both axes, with modest distance
variation that stays in focus. Cover the usable image area; repeated images of
one centered, front-facing target are insufficient. The collector needs at least
6 detected corners; the fitter uses only views with at least 12 non-collinear
corners and requires at least 20 usable images.

```bash
python3 eye_eye_calibration.py calibrate-intrinsics \
  --images runs/leica_intrinsic_images_v1 \
  --camera leica \
  --notes "Same optical settings as capture; add camera/adapter details" \
  --output runs/leica_intrinsics_v1.json
```

Inspect `rms_px`, `per_image`, `orientation_span_deg`,
`image_coverage_fraction_xy`, and `warnings` in the JSON. Large errors or strong variation
between repeat fits need investigation: blurred corners, insufficient tilt,
wrong board, focus changes, or a poor pinhole approximation for the optics.
The script saves the fit for inspection; it does not certify it from RMS alone.
Use new target views for the registration/validation steps below. If factory
D405 intrinsics are suspect, the same two commands can collect/fit D405 color
intrinsics, provided the central pattern is resolved clearly enough.

All output filenames and capture directories must be new. Increment `v1` to
`v2`, etc. for repeat runs; existing results are not overwritten.

## 5. Capture registration and validation pairs

With both cameras fixed and both image streams live:

```bash
python3 eye_eye_calibration.py capture \
  --leica-intrinsics runs/leica_intrinsics_v1.json \
  --d405-intrinsics runs/d405_color_intrinsics.json \
  --session runs/eye_eye_session_v1 \
  --notes "Leica zoom=RECORD_VALUE; focus=RECORD_VALUE; working distance=RECORD_VALUE; D405 serial=RECORD_VALUE; mounts fixed; board dimensions checked"
```

To use the existing Leica JSON, replace only `--leica-intrinsics` with its path.
The historical tracker filenames accept these same arguments and subcommands.
By default the Leica uses **ChArUco** and D405 uses **AprilTags**. There is no
silent switching between sources. `--leica-source` / `--d405-source` may be
set to `charuco` or `april` if the actual camera views require it.

1. Confirm green detected points and a board axis overlay in both views. The
   board-origin axes may lie outside the narrow Leica image; that is expected.
   Check the displayed reprojection RMS and any rejection message.
2. Hold the board still, then press **C** in an OpenCV window. This saves one
   fitting pair. Watch the terminal for `Saved`; rejected captures do not count.
3. Move the board to another useful position/tilt, wait for it to settle, and
   press C again. Aim for **15–25 diverse fitting poses**; the solver requires 10.
   Include tilt around both axes and positions/depths spanning the intended
   workspace. At least 10° total orientation span is a diagnostic recommendation.
4. Collect **5–10 additional held-out poses with V**, preferably after completing
   the C captures. These are observations at different positions/tilts that the
   fit never sees; include the actual working region. The solver requires 5.
5. Press **Q**. Capture is complete, but the final transform is not yet fitted.

Every accepted pair saves full-resolution **unannotated** PNGs, timestamps,
frame IDs, corner correspondences, reprojection error, poses, and image hashes.
The session saves exact intrinsic data, board definition, capture settings,
topics and notes. A frame cannot be reused across fit and validation captures.
No temporal smoothing enters the calibration, and no observations are silently
removed as outliers. Only complete `sample.json` captures are processed.

The initial 1.5 px per-camera RMS limit and planar ambiguity gap are quality
guards, not measured accuracy guarantees. Do not increase these thresholds
merely to get captures accepted; first improve focus, lighting, target tilt,
coverage, or intrinsics. Frames must also pass resolution and timing checks.

## 6. Solve and inspect the held-out check

This command can run after the cameras and ROS are stopped:

```bash
python3 eye_eye_calibration.py solve \
  --session runs/eye_eye_session_v1 \
  --output runs/eye_eye_registration_v1.json
```

The solver verifies image hashes and re-detects/re-estimates poses from saved
pixels. It fails on a bad saved capture rather than silently skipping it. Keep
the full session together with the result for reproduction.

The JSON contains:

| Field | Meaning |
|---|---|
| `T_d405_color_from_leica` | 4×4 rigid transform taking Leica 3D coordinates into D405 color optical coordinates |
| `T_leica_from_d405_color` | Its inverse |
| `units` | `metres`; translation is not millimetres |
| `metrics.fit` | Residuals on fitting captures |
| `metrics.validation` | Residuals on held-out captures, which were never fitted |
| `board_origin_error_mm` | Distance between predicted and separately observed board origins |
| `board_corner_rms_mm` | RMS discrepancy at the four outer composite corners; includes rotational effects |
| `rotation_error_deg` | Relative rotation discrepancy |
| `*_cross_projection_rms_px` | Reprojection error when predicting one camera's observations through the other camera and the fitted registration |
| `fit_*_span_*` | Target motion diversity diagnostics |
| `physical_accuracy_validated` | Always false: an independent physical test is still needed |

Rotation is averaged on SO(3), with translations averaged in the destination
frame. The fitted equation for each pair is:

```text
T_d405_color_from_leica = T_d405_color_from_board @ inverse(T_leica_from_board)
```

Review mean, RMS **and maximum** held-out errors. A small fitting residual with
a large held-out error is a failed consistency check. There is a warning if
held-out composite-corner RMS exceeds 1 mm, or fitting orientation span is below
10°. A JSON file is still written for diagnosis; absence of warnings is not a
physical validation pass. Low pixel error alone can coexist with poor depth
accuracy, especially with a narrow-angle/high-magnification planar target.

For a strong first result, repeat the complete run with new board poses while
leaving both camera mounts and optical settings unchanged. Compare the predicted
locations of points in the actual workspace, not just matrix entries.

## 7. Physical validation and use

The project's historical test plan calls for <1 mm fiducial error at about
200 mm working distance, <0.5 mm cross-session translation drift, and <0.3 mm
filtered stationary-position standard deviation over 30 frames. These are
**planned criteria**, not certified outcomes of this software. The new held-out
check is a board-based consistency test, and does not implement that filtered
30-frame stability test. Cross-session transform comparisons are meaningful
only when the camera geometry and optical settings are unchanged.

Before using this result to guide the robot:

1. Check a separate, measured fiducial/target in the intended workspace, with
   positions established independently of the calibration images. Measure the
   physical miss, or compare to an independently calibrated reference.
2. If validating the full robot chain, use the already validated hand-eye and
   tool-tip calibrations and a non-contact bench targeting test. A chain error
   cannot automatically be assigned to eye-eye calibration alone.
3. Record the setup, reference method, sample count, mean/RMS/maximum miss and
   repeatability; use the project's acceptance criteria for the intended task.

Example of transforming an already known 3D Leica point:

```python
import json
import numpy as np

with open("runs/eye_eye_registration_v1.json") as f:
    result = json.load(f)
T = np.asarray(result["T_d405_color_from_leica"])
point_leica_m = np.array([0.001, 0.002, 0.200, 1.0])
point_d405_m = T @ point_leica_m
```

A selected Leica **pixel is a ray**, not a 3D point. It still needs an independent
depth, a known plane/surface, or another valid reconstruction method. This
calibration does not solve that depth problem.

The hand-eye script in this repository uses the D405 **color** stream. Before
composing its camera-to-base transform with this result, confirm its frame and
translation units. Some publishers label optical images `realsense_link`; the
label alone is not proof that their coordinates are a ROS body frame. This
workflow always means x right, y down, z forward in the image's optical frame.
If another calibration uses a depth/infrared or rectified optical frame, include
the correct inter-stream transform. Do not assume they are identical.

## Troubleshooting

| Message / symptom | Action |
|---|---|
| Missing `rospy` / `cv_bridge` | Source the correct ROS workspace and use its compatible Python3 interpreter |
| Missing `CharucoDetector` | Check which `cv2` is imported; use the tested modern OpenCV package |
| Waiting for timestamp-paired images | Check both image topics, publisher rates, nonzero stamps, common clock and ROS master |
| CameraInfo export times out | Check topic names and driver; this command subscribes to the image topic as well |
| Intrinsics/image size mismatch | Restore calibration resolution or recalibrate; no automatic scaling occurs |
| Too few ChArUco corners | Improve focus/lighting, center the existing 6×6 pattern; check the dictionary/print |
| Too few AprilTags | Expose at least two distinct known tags, preferably spread across the target |
| Ambiguous planar pose | Add visible target tilt while keeping it in focus; collect informative views |
| High reprojection RMS | Check print flatness, glare, blur, measured scale, intrinsics, crop and optical settings |
| Stale/duplicate capture | Wait for fresh images; check camera processing load and timestamps |
| Good fit but bad validation | Check pose diversity, planar ambiguity, board movement, camera movement and intrinsics |
| One capture cannot be replayed | Check the OpenCV version and raw files; do not discard held-out failures to claim accuracy |
| Output already exists | Choose a new session directory or result filename |

## Software verification

```bash
python3 -m pip install pytest
python3 -m pytest -q test_calibration.py
```

Tests cover common-origin geometry, both pose sources, transform direction,
pixel-noise/error rejection, degenerate views, invalid intrinsics, stale/duplicate
timestamps, rotation averaging, held-out isolation, image detection, and an
offline CLI run from rendered raw images, plus intrinsic fitting from rendered
board views. On 2026-09-22, **14 tests passed** with Python 3.11 / OpenCV 4.10.0
in an isolated environment; syntax and CLI entry-point checks also passed.
They do not test physical accuracy or
live ROS camera drivers. The archived PDF's actual bitmap was also checked with
OpenCV 4.10.0: all 25 ChArUco corners and all eight AprilTags were detected.

OpenCV references:
[PnP coordinate convention and solver requirements](https://docs.opencv.org/4.5.5/d5/d1f/calib3d_solvePnP.html),
[ChArUco detection and calibrated interpolation](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html).
