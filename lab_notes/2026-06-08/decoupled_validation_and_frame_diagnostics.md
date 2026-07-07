# Lab Note: Decoupled Validation and Frame Diagnostics

**Date:** 2026-06-08

## Objective

Develop a more detailed diagnostic workflow for the hand-eye calibration error by separating spatial error from orientation error, checking frame consistency, and adding support for fitted D405 intrinsics.

## Background

The combined hand-eye validation dataset changed XYZ position and tool orientation at the same time. This made the residuals hard to interpret, because a large error could be caused by translation geometry, orientation conventions, camera intrinsics, or a coupling between these factors. The goal for this work was to isolate these sources of error.

## Work Completed

- Added decoupled validation tests for spatial and orientation error.
- Updated the validation collector to support spatial and orientation validation modes.
- Added a robot motion script for running decoupled validation and automatically evaluating the collected dataset.
- Added a frame-consistency diagnostic script.
- Added shared hand-eye math helpers for relative-motion and translation-based analysis.
- Added synthetic unit tests for the hand-eye math helpers.
- Added optional fitted D405 intrinsics support.
- Added a D405 ChArUco intrinsic calibration script.
- Updated the hand-eye solver to use relative-motion initialization, robust weighted least squares, and deterministic multi-start optimization.
- Updated documentation for the decoupled validation and frame-consistency workflow.
- Updated the validation dataset after additional validation testing.

## Decoupled Validation

The new validation workflow separates two cases:

- Spatial validation: move through a `3 x 3 x 3` XYZ grid while keeping orientation constant.
- Orientation validation: keep XYZ fixed while sweeping a `5 x 5` roll/pitch grid.

This separation is important because the interpretation of a bad result is different in each case. If spatial validation is poor while orientation validation is acceptable, the problem is more likely related to translation geometry, camera intrinsics, scale, or robot position reporting. If orientation validation is poor while spatial validation is acceptable, the problem is more likely related to board-to-gripper orientation, FrameEE quaternion convention, TCP definition, or board mounting.

The new `run_validation_tests.py` script generates the appropriate target set, moves the robot, writes motion logs, and can automatically evaluate the resulting validation dataset against a selected calibration file.

## Frame-Consistency Diagnostics

I added `diagnose_frame_consistency.py` to compare camera-to-base rotation estimates from independent sources:

- a translation-only estimate from the spatial validation set,
- a relative-motion rotation estimate from the orientation validation set,
- an orientation-arc estimate based on the motion of the ChArUco board origin during fixed-XYZ rotation.

The diagnostic also checks multiple FrameEE quaternion interpretations:

- `xyzw`,
- `xyzw_inverse`,
- `wxyz_source`,
- `wxyz_source_inverse`.

This is intended to reveal whether the standard FrameEE convention is consistent with the observed camera/board motion, or whether a frame convention mismatch may be contributing to the validation error.

## Intrinsics Support

I added support for loading optional fitted D405 intrinsics through `camera_intrinsics.py`. The calibration and validation scripts can now use either the D405 factory intrinsics or a fitted ChArUco intrinsics file, with resolution checks to prevent using intrinsics from the wrong image size.

I also added `calibrate_d405_intrinsics.py`, which can capture ChArUco views, fit D405 intrinsics, reject outlier views, compare against factory intrinsics, and save the fitted camera matrix, distortion coefficients, summary JSON, and per-view CSV.

## Solver Improvements

The hand-eye solver was further revised to improve robustness and interpretability:

- Initialize rotation from relative motion.
- Solve translation linearly using the initialized rotations.
- Run robust weighted least squares from multiple deterministic starting points.
- Save the legacy solution for comparison.
- Save solver diagnostics, including conditioning and multi-start solution spread.

These diagnostics help distinguish an optimizer issue from inconsistent upstream measurements.

## Result

The hand-eye workflow now has a more complete debugging path. Instead of treating validation as one combined error number, the project can separately test spatial behavior, orientation behavior, camera intrinsics, FrameEE convention, and solver stability.

## Next Steps

- Collect both spatial and orientation validation datasets using the same calibration file.
- Run `diagnose_frame_consistency.py` to compare translation-derived and rotation-derived camera-to-base estimates.
- If frame consistency is poor, inspect the FrameEE publisher and quaternion convention.
- Capture D405 intrinsic calibration views and test whether fitted intrinsics improve validation residuals.
- Compare weighted and legacy calibration results on both decoupled validation datasets.
