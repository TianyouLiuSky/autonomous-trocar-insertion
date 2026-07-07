# Lab Note: Hand-Eye Calibration Motion Refinement

**Date:** 2026-05-31

## Objective

Improve the hand-eye calibration motion workflow so the robot can collect a complete and usable calibration dataset within its physical workspace and motion limits.

## Background

Previous hand-eye calibration motion attempts were limited by robot reachability, workspace boundaries, and orientation constraints. The calibration pose sequence needed to provide enough spatial and rotational diversity for the ChArUco-based hand-eye solver while still remaining safe and reachable for the SHER robot.

Experiments from May 30 and May 31 were therefore used to determine a practical robot working space. The purpose was not to find the absolute maximum possible robot motion, but to identify a region where the robot could reliably reach targets, preserve orientation, and return usable pose data for calibration.

## Work Completed

- Updated the calibration pose generation to better account for robot workspace limitations.
- Added explicit robot workspace limits for the calibration and validation scripts.
- Added checks to reject generated poses that fall outside the configured workspace.
- Ran and recorded workspace-limit experiments to determine practical X/Y/Z motion boundaries.
- Updated the motion scripts to use measured workspace limits rather than arbitrary calibration offsets.
- Defined the center of the workspace as the ideal starting/home position for calibration motion.
- Tightened the rotation tolerance used to decide whether the robot reached the commanded pose.
- Limited angular motion speed to reduce unstable or overly aggressive rotation commands.
- Fixed an approach/orientation issue in the robot motion command path.
- Updated the hand-eye calibration GUI code so orientation differences are handled more consistently.
- Added motion logging for calibration runs, including live residuals and final per-attempt summaries.
- Updated the hand-eye calibration README with the revised workflow and operating constraints.

## Implementation Notes

The calibration motion script now generates 20 poses from 10 spatial anchors, with each anchor visited using two different orientations. This design gives repeated XYZ positions under different orientations while still maintaining enough orientation difference between accepted samples.

The workspace is constrained using absolute limits in the robot FrameEE coordinate system. The script can shift or compress requested offsets so that the generated calibration targets remain inside the allowed region. It also validates each generated pose before commanding robot motion.

The conservative calibration workspace added to `run_calibration_poses.py` is:

```text
X: [-40.0, 10.0] mm
Y: [-130.0, -85.0] mm
Z: [-13.0, 26.0] mm
```

The center of this workspace is:

```text
(-15.0, -107.5, 6.5) mm
```

This center is the ideal starting position because the calibration sequence needs room to move in both positive and negative directions along each axis. Starting at the center also reduces the chance that a rotation or translation target becomes unreachable near a boundary.

Related validation and diagnostic scripts later use a slightly expanded workspace:

```text
X: [-42.0, 10.0] mm
Y: [-133.0, -85.0] mm
Z: [-13.0, 26.0] mm
```

This expanded validation box has center `(-16.0, -109.0, 6.5) mm`. Both sets of limits describe the same experimentally determined safe region, with calibration using the more conservative box.

The calibration script now uses tighter pose acceptance criteria:

- Position tolerance: 0.5 mm
- Orientation tolerance: 0.2 deg
- Maximum angular velocity: 0.05 rad/s
- Move timeout: 10 s

These settings were chosen to make the recorded samples better match the intended robot poses while avoiding overly aggressive rotation behavior.

## Experiment Result

After the updates, I ran the revised calibration workflow and collected a complete 20-sample dataset. One sample attempt was rejected because it was too similar to an already accepted pose, but the final run still reached 20 accepted samples.

The recorded run reported a calibration error of approximately 3.171 mm with 20 samples.

The workspace-limit changes were successful in the sense that the robot could complete the calibration sequence inside the measured safe region. The result also showed why this workspace work was necessary: without explicit limits and a centered home position, the robot could request calibration poses that were theoretically useful but practically unreliable.

## Observations

The revised pose sequence appears to be more compatible with the robot's physical limits than earlier versions. The workspace limits and automatic offset adjustment helped keep the generated targets reachable. The stricter orientation tolerance and angular velocity limit also made the motion criteria more explicit.

The remaining calibration error indicates that the workflow is improved but still needs additional validation. The 3.171 mm error may come from robot motion residuals, ChArUco pose noise, camera/board geometry, or remaining frame-definition issues.

The main workspace conclusion is that calibration and validation should be centered within the measured safe box. The center position is not just a convenience; it controls the available margin for the full pose set. If future scripts use a different starting position, that change should be documented and checked against the same workspace limits.

## Next Steps

- Review the saved calibration pose CSV and motion logs to identify any poses with large residuals.
- Run validation using the updated validation script and the same saved home position.
- Compare calibration quality across multiple runs to see whether the result is repeatable.
- Continue debugging hand-eye frame consistency if validation error remains high.
