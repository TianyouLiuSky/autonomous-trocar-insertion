# Robot Workspace Limits Used for Hand-Eye Motion

**Date:** 2026-05-31

## Purpose

This document records the robot working-space limits that were added to the hand-eye calibration movement scripts after workspace-limit experiments. These limits are defined in the robot-reported `FrameEE` coordinate system, using millimeters for translation and degrees for rotation.

## Calibration Workspace

The hand-eye calibration pose generator defines the conservative calibration workspace as:

| Axis | Minimum | Maximum | Center |
|---|---:|---:|---:|
| X | -40.0 mm | 10.0 mm | -15.0 mm |
| Y | -130.0 mm | -85.0 mm | -107.5 mm |
| Z | -13.0 mm | 26.0 mm | 6.5 mm |

The ideal starting position for calibration is the center of this workspace:

```text
FrameEE position = (-15.0, -107.5, 6.5) mm
```

This center is not necessarily the only valid starting pose, but it is the best default because it gives the robot margin in both positive and negative directions before hitting a workspace boundary.

## Orientation Limit

The calibration scripts also enforce an absolute roll limit:

```text
|roll| <= 28.0 deg
```

This prevents the pose generator from requesting orientations that are likely to become unreachable or unstable near the translation limits.

## Validation and Diagnostic Workspace

Later validation and axis-alignment scripts use a closely related, slightly expanded workspace:

| Axis | Minimum | Maximum | Center |
|---|---:|---:|---:|
| X | -42.0 mm | 10.0 mm | -16.0 mm |
| Y | -133.0 mm | -85.0 mm | -109.0 mm |
| Z | -13.0 mm | 26.0 mm | 6.5 mm |

This corresponds to:

```text
FrameEE position = (-16.0, -109.0, 6.5) mm
```

The force data collection scripts later use the same X/Y limits and extend the upper Z bound to `30.0 mm`, producing a force-collection midpoint of:

```text
FrameEE position = (-16.0, -109.0, 8.5) mm
```

## Why the Center Matters

The center of the safe workspace is the ideal starting position because calibration and validation require motion in multiple directions. Starting from the center gives the pose generator room to create positive and negative X/Y/Z offsets, while still preserving margin for orientation changes.

Starting near an edge of the workspace can make a mathematically valid pose sequence physically impossible. For example, a 24 mm validation cube requires approximately `+/-12 mm` around the center on each translation axis. If the robot starts too close to a limit, the script must either compress the motion range or reject targets.

## Scripts Using These Limits

The workspace limits are encoded in the movement scripts rather than kept only in notes:

- `hand-eye-calibration/run_calibration_poses.py`
- `hand-eye-calibration/run_validation_24mm.py`
- `hand-eye-calibration/run_validation_tests.py`
- `hand-eye-calibration/run_axis_alignment_poses.py`
- later force collection scripts under `force_data_collection/code/`

The important rule is that any future calibration or validation motion should either use these limits directly or intentionally document why a different workspace is being used.
