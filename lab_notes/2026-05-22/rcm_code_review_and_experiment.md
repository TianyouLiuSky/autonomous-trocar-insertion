# Lab Note: RCM Code Review and Experiment Attempt

**Date:** 2026-05-22

## Objective

Review the remote center of motion (RCM) control code and attempt to improve the RCM behavior for the autonomous trocar insertion workflow.

## Code Reviewed

I reviewed the RCM-related motion code in `motion_script/SHER_Controller_rcm_fixed.py` and `motion_script/test_rcm_motion.py`.

The fixed RCM controller attempts to address two issues in the original RCM motion logic:

- The desired tool orientation could be defined in the wrong direction, which may command an unnecessary 180 degree flip.
- The RCM line was checked at the start of motion, but the commanded linear velocity was not fully constrained during the move, so the tool shaft could still drift away from the fixed RCM point.

The revised controller tries to preserve the RCM constraint by splitting tip motion into axial and lateral components. Axial motion is allowed along the tool shaft for insertion or retraction. Lateral motion is paired with an angular velocity so that the tool pivots around the RCM point.

I also reviewed the interactive RCM test script. It computes an RCM point from the current tool axis and a configured distance, then tests no-motion hold, axial slide, and small pivot motions. The script records position error, RCM-line error, tool-axis error, and operator observations in CSV logs.

## Work Completed

- Attempted to fix the RCM behavior by using the revised RCM controller.
- Attempted physical/interactive experiments using the RCM testing workflow.
- Checked whether the tool shaft appeared to maintain a fixed RCM point during small commanded motions.

## Results

The experimental results were not clear enough to draw a confident conclusion. The RCM code now has a more explicit strategy for preserving the remote center of motion, but the physical behavior still needs more controlled testing before I can determine whether the fix is correct and reliable.

## Next Steps

- Repeat the RCM experiments with more controlled starting conditions.
- Record both numerical errors and physical observations for each motion.
- Compare the RCM test logs across multiple runs to determine whether the revised controller consistently maintains the pivot point.
- Continue testing RCM after lab access resumes.
