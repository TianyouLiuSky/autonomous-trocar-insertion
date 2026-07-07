# Lab Note: Two-Step Oblique Tip-Centered RCM Sequence

**Date:** 2026-06-26

## Objective

Create an experimental motion script for the actual two-step oblique insertion sequence, using the physical needle tip as the remote center of motion during orientation changes.

## Background

The earlier force data collection scripts supported direct, perpendicular, and 30-degree insertion conditions, but they were primarily one-step insertion workflows. Those scripts were useful for collecting force profiles under controlled single-axis motion, but they did not fully represent the intended trocar insertion procedure.

The intended motion is a two-step oblique sequence: begin from a perpendicular approach, make a small initial insertion, rotate into the oblique condition about the physical tip, advance obliquely, rotate back to perpendicular about the tip, then continue insertion. This requires controlling the tool tip rather than only the robot end-effector origin.

## Work Completed

- Added `motion_script/needle_tip_rcm_sequence.py`.
- Implemented a tip-centered RCM motion sequence using the calibrated end-effector-to-tip offset.
- Defined the physical tip position as:

```text
p_tip_base = p_gripper_base + R_base_gripper * t_gripper_tip
```

- Added compensated end-effector motion during rotation so the physical tip can remain fixed while the handle moves around it.
- Added staged operator prompts and per-stage abort behavior.
- Added workspace checks for the robot-reported FrameEE position.
- Added logging of stage summaries and time-series samples.
- Used the workspace midpoint as the default safe start position.
- Kept the force-collection angle convention where the 30-degree oblique condition is represented as a 60-degree tilt from the straight/down reference pose.
- Made a small cleanup in the force teleoperation argument parsing area.

## Sequence Implemented

The new script implements the following staged sequence:

1. Move FrameEE to a safe start pose, defaulting to the workspace midpoint.
2. Rotate/settle to the perpendicular approach pose, default `RPY = (0, 20, 0)`.
3. Move the physical tip `0.25 mm` along the perpendicular needle direction.
4. Rotate to the 30-degree oblique force-collection condition about the physical tip.
5. Move the physical tip `2.0 mm` along the oblique needle direction.
6. Rotate back to the perpendicular pose about the physical tip.
7. Move the physical tip `8.0 mm` along the perpendicular needle direction.
8. Retract the physical tip `20.0 mm` along the same perpendicular axis.

The key change is that the script controls the physical tip path. During rotation stages, it computes the end-effector linear velocity needed to compensate for handle motion:

```text
v_gripper = v_tip - omega x (R_base_gripper * t_gripper_tip)
```

This is necessary because rotating the handle without compensation would move the physical needle tip, breaking the RCM assumption.

## Safety and Logging

The script enforces workspace limits on the robot-reported FrameEE position:

```text
X: [-42.0, 10.0] mm
Y: [-133.0, -85.0] mm
Z: [-13.0, 30.0] mm
```

The physical tool tip is allowed to differ from FrameEE by the configured rigid offset, but the robot handle/FrameEE must remain inside the safe workspace.

The script also logs stage status, target and final tip positions, target and final FrameEE positions, tip error, orientation error, handle drift, and workspace violations. This makes the sequence suitable for debugging even before it is used for full force data collection.

## Result

The project now has a first implementation of the actual two-step oblique insertion motion. This is a more realistic representation of the intended surgical insertion sequence than the earlier one-step force collection scripts.

## Next Steps

- Test the sequence slowly on the robot with the physical workspace clear.
- Verify that the measured tip offset is correct before trusting tip-centered rotation.
- Compare the logged tip error and handle drift across all stages.
- Connect the sequence to force recording once the motion behavior is verified.
