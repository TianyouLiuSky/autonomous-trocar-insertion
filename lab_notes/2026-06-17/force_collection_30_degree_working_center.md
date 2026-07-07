# Lab Note: 30-Degree Force Collection and Working Center Definition

**Date:** 2026-06-17

## Objective

Make the 30-degree force collection workflow usable on the robot by defining a working center, improving the insertion-axis behavior, and verifying that the 30-degree condition can run successfully.

## Background

The force collection code had the basic direct and oblique modes, but the 30-degree insertion condition needed more careful handling. For oblique insertion, insertion and retraction should follow the needle/tool direction rather than simply moving along robot base Z. The robot also needs a reasonable working center so the operator starts inside the usable fixed-orientation workspace.

## Work Completed

- Updated the force collection recorder and shared helper functions.
- Slightly increased the default linear translation speed to improve usability.
- Defined a working center for the force collection teleoperation workspace.
- Added automatic/pre-teleop centering behavior around the working center.
- Updated the 30-degree insertion behavior so insertion follows the needle/tool direction.
- Added insertion-axis metadata to the recorder and analysis output.
- Made the UI more explicit about the active angle/condition.
- Verified that the 30-degree insertion workflow worked.

## Working Center

A working center was added to make the starting position for force collection more repeatable. Instead of manually beginning from arbitrary robot poses, the teleoperation workflow can guide the robot toward a defined center before the operator begins the actual insertion motion.

This matters because the available robot workspace is limited when orientation is fixed. Starting too close to a workspace boundary can prevent insertion from reaching the intended depth or can cause the robot to hit a translation limit partway through a trial.

## 30-Degree Insertion Behavior

The 30-degree condition was updated so `C/V` insertion and retraction move along the locked tool/needle insertion axis. This is different from direct-down insertion, where base-Z motion is sufficient. For oblique insertion, moving only in base Z would not represent the physical needle direction and would produce force data from the wrong motion path.

The recorder and analyzer were also updated to preserve the insertion axis and condition angle in saved sessions. This allows later analysis to compute force behavior relative to insertion depth rather than only raw robot coordinates.

## Result

The 30-degree force collection workflow became operational. The code now better reflects the physical insertion direction, includes a defined working center, and records the condition and insertion axis more explicitly. The 30-degree mode was verified to work on the robot.

## Next Steps

- Add stronger safety guards for oblique insertion.
- Limit insertion and retraction distance so the robot cannot drift too far from the starting point.
- Continue improving the recorder UI so live force plots are easier to read during experiments.
- Use the working center and insertion-axis metadata for future force trials.
