# Lab Note: Camera Repeatability and Translation Error Work

**Date:** 2026-06-07

## Objective

Investigate the sources of hand-eye calibration error by separating camera/ChArUco repeatability from solver and robot-frame issues, and by modifying the calibration solver to account more directly for translation error.

## Work Completed

- Added an experiment script for stationary D405/ChArUco repeatability testing.
- Updated the calibration pose design to improve orientation diversity while remaining inside the robot workspace.
- Updated the calibration solver to prioritize translation residuals more explicitly.
- Added both weighted and legacy hand-eye calibration solutions to the saved calibration result.
- Updated the evaluator so weighted and legacy solutions can be validated separately.
- Updated the validation dataset file after running additional validation data collection.

## Camera Repeatability Test

I added `test_charuco_repeatability.py` to measure D405/ChArUco pose stability while the robot, board, and camera remain stationary. The goal is to determine whether observed calibration error could be caused by camera pose-estimation noise rather than by the hand-eye transform.

The test records repeated ChArUco board poses and logs the current `/SHER20/eye_robot/FrameEE` pose at the same time. It reports:

- board position repeatability,
- board angular repeatability,
- ChArUco reprojection error,
- detected corner count,
- FrameEE position repeatability,
- FrameEE angular repeatability.

This gives a way to compare camera/PnP jitter against robot pose drift. If the robot pose is stable but board-pose jitter changes across the image, the problem is likely on the vision side. If both change, robot settling, mount rigidity, or vibration may be contributing.

## Calibration Pose Update

The calibration pose sequence was revised so that 20 samples are generated from 10 spatial anchors, with each anchor visited using two different orientations. The roll and pitch targets were expanded to improve orientation diversity:

- Roll targets: `[-16, -8, 0, 8, 16] deg`
- Pitch offsets: `[-12, -4, 4, 12] deg`
- Minimum intended pairwise orientation difference: about `8 deg`

This change was meant to give the GUI's diversity check more margin above the 5 degree threshold, while still keeping targets inside the robot workspace.

## Solver Update

The hand-eye calibration solver was updated to compare a weighted solution against the previous legacy solution. The weighted residual normalizes rotation and translation errors so that translation error can be prioritized more directly. The default profile treated a 0.5 degree rotation error and a 1.0 mm translation error as comparable scales.

The saved calibration file now includes both:

- weighted `T_cam2base` and `T_board2gripper`,
- legacy `T_cam2base` and `T_board2gripper`.

The evaluator was updated with a `--solution` option so I can validate either the weighted or legacy solution on the same validation dataset.

## Result

The project now has tools to test whether the validation error is coming from camera repeatability, pose diversity, or solver weighting. The calibration code can now preserve the old solution for comparison while testing a new translation-prioritized solution.

## Next Steps

- Run the stationary ChArUco repeatability test at the center and near the edges of the image.
- Validate both weighted and legacy calibration results using the same dataset.
- Use the repeatability statistics to decide whether camera pose estimation is a significant contributor to the hand-eye error.
