# Lab Note: Robot Workspace Limit Exploration

**Date:** 2026-05-30

## Objective

Experimentally determine the usable robot workspace for hand-eye calibration and validation motion, with the goal of finding a safe bounded region where the robot can move repeatedly without hitting physical, controller, or orientation-related limits.

## Background

The hand-eye calibration workflow requires the robot to visit many poses while the camera observes the ChArUco board. Earlier calibration attempts showed that simply requesting a large calibration grid was not reliable. Some target poses were too close to the robot's reachable boundary, and the robot could fail to reach them or approach them with large residual motion error.

Because hand-eye calibration depends on accurate robot pose records, unreachable or marginally reachable poses are not just inconvenient; they directly reduce calibration quality. The calibration motion sequence therefore needed to be redesigned around the robot's measured reachable workspace rather than an idealized workspace.

## Work Completed

- Ran exploratory robot motions to identify practical translation limits in the robot's reported FrameEE coordinate system.
- Checked how far the robot could move in X, Y, and Z while remaining controllable and recoverable.
- Observed that the available workspace depends on both position and orientation, especially when the calibration pose sequence requires roll and pitch variation.
- Identified that a centered, conservative workspace would be more useful than using the absolute maximum range of the robot.
- Used these observations to prepare for the May 31 calibration motion updates.

## Observations

The robot should not be commanded directly at the extreme workspace boundaries during calibration. Near the limits, the robot may still report a possible command target, but the actual motion can become slow, inaccurate, or unable to satisfy the desired orientation. This is especially problematic for calibration because the final recorded pose must match the intended pose closely.

The practical workspace therefore needs margin. A smaller but reliable box is better than a larger box that causes intermittent failed movements.

## Experiment Interpretation

The working-space experiment showed that the calibration and validation scripts should explicitly encode:

- absolute X, Y, and Z safety limits,
- an orientation/roll limit,
- automatic checks before commanding a pose,
- pose-generation logic that compresses or shifts requested offsets to remain inside the measured box,
- and a central starting/home position so calibration and validation use a comparable region.

## Result

The May 30 exploration motivated the formal workspace limits and pose-generation changes added on May 31. The main conclusion was that hand-eye calibration should be run from the center of a measured safe workspace, not from arbitrary robot poses or targets near the robot's motion boundary.

## Next Steps

- Encode the measured workspace limits directly in the calibration motion scripts.
- Generate calibration poses around the center of the safe workspace.
- Add validation checks so invalid target poses are rejected before motion.
- Record the home/center position for use by validation and later diagnostic experiments.
