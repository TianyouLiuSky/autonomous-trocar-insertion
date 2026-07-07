# Lab Note: Force Collection Safety, Direct Mode, and UI Scaling

**Date:** 2026-06-21

## Objective

Improve the safety and usability of force data collection by adding stronger limits for oblique insertion, defining direct insertion as force Z-axis/base-Z motion, increasing usable motion speed, and improving live force plot scaling.

## Background

After verifying the 30-degree insertion workflow, the next issue was making the software safer and easier to use during repeated experiments. Oblique insertion introduces horizontal motion components, which can cause the robot to approach workspace limits or stall against constraints if not guarded. Direct insertion also needed to be distinguished from tool-axis insertion: for the direct condition, the desired force collection motion is pure robot base-Z insertion/retraction.

## Work Completed

- Improved 30-degree insertion safety behavior.
- Added stronger command gating and motion limits in the fixed-angle teleoperation script.
- Updated direct insertion so `C/V` motion is constrained to robot base Z.
- Added insertion-axis mode logic to distinguish direct, tool-axis, and automatic behavior.
- Doubled the maximum linear speed used by the teleoperation script.
- Updated the force recorder UI so live force plots rescale more effectively.
- Added tests for direct/base-Z insertion-axis behavior and plot scaling helpers.
- Updated documentation for the revised safety and control behavior.

## Safety Improvements

The 30-degree workflow was updated with additional checks intended to prevent unsafe or misleading trials. These include workspace limits, travel limits, insertion/retraction distance limits, pose freshness checks, and command gating based on orientation error.

These checks are important because force collection occurs during contact. If the robot is moving obliquely and the actual motion becomes blocked or deviates from the desired axis, the measured force may no longer correspond to the intended insertion condition. Safety logic also helps prevent the operator from accidentally driving the robot too far while focused on the live force plot.

## Direct Insertion Mode

Direct insertion was updated so the insertion/retraction command follows robot base Z only. This separates the direct condition from the oblique and perpendicular conditions:

- Direct insertion: `C/V` moves along robot base Z.
- Oblique/perpendicular insertion: `C/V` can follow the locked tool insertion axis.

This distinction makes the experiment definitions clearer and prevents the direct condition from being affected by small differences in tool-axis orientation.

## UI Scaling

The force recorder UI was updated to rescale the live force plot. This makes it easier to see both small baseline variations and larger force changes during insertion without manually adjusting plotting limits. The scaling helper was added to the common module and covered by tests.

## Result

The force collection workflow became safer and more usable for repeated experiments. Direct insertion now has a clear base-Z definition, oblique insertion has stronger safety checks, and the live force display is easier to interpret during operation.

## Next Steps

- Refine the starting point for 30-degree insertion to increase available travel before reaching robot limits.
- Add a perpendicular insertion condition for comparison.
- Run controlled trials after confirming the force topic and recorder UI behavior.
