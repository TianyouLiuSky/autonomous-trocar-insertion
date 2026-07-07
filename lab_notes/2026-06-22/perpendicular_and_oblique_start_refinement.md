# Lab Note: Perpendicular Insertion and Oblique Start Refinement

**Date:** 2026-06-22

## Objective

Refine the 30-degree insertion starting position and add an experimental perpendicular insertion mode for force data collection.

## Background

The 30-degree workflow was functional, but the default geometric center did not leave enough useful travel before the robot approached fixed-orientation translation limits. For force data collection, the robot should begin from a position that gives enough insertion distance while staying inside the safe workspace. In parallel, a perpendicular-to-eye insertion condition was needed as another experimental mode.

## Work Completed

- Redefined the default starting point for 30-degree insertion.
- Updated the 30-degree wrapper script with the new center position.
- Updated fixed-angle teleoperation to support explicit absolute target RPY and insertion-axis mode options.
- Added `perpendicular_insertion.py` as an experimental perpendicular insertion condition.
- Updated shared helper functions and tests for the new condition behavior.
- Updated the force collection README to document direct, perpendicular, and 30-degree workflows.

## 30-Degree Start Position

The 30-degree insertion start point was moved to:

```text
(-5.5, -109.0, 17.0) mm
```

This position is biased toward the upper/right side of the usable fixed-orientation corridor. The reason is that starting from the geometric workspace center left only a few millimeters of travel before the robot encountered a translation limit during the oblique insertion path.

The updated start point should allow more useful travel for 30-degree insertion while remaining within the defined workspace constraints.

## Perpendicular Insertion Mode

A new perpendicular insertion wrapper was added in an experimental state. This mode locks the robot to an absolute orientation of:

```text
RPY = (0, +20, 0)
```

Unlike direct insertion, the perpendicular mode uses tool-axis insertion/retraction for `C/V`. This keeps the motion aligned with the locked insertion direction rather than forcing base-Z motion.

## Experiment Definitions

At this point, the force collection workflow supports three main conditions:

- Direct-down insertion with straight orientation and base-Z insertion motion.
- Perpendicular-to-eye insertion with absolute `(0, +20, 0)` orientation and tool-axis insertion motion.
- 30-degree oblique insertion labeled as the experimental 30-degree condition, using a 60-degree tilt from the straight reference and a shifted start point.

## Result

The force collection code now supports the main intended insertion conditions with clearer start positions and control definitions. The 30-degree mode should have more usable insertion travel, and the perpendicular mode provides a new experimental condition for comparison.

## Next Steps

- Test the perpendicular insertion mode on the robot.
- Verify that the 30-degree start point provides enough safe travel in real operation.
- Collect pilot force datasets for direct, perpendicular, and 30-degree insertion.
- Compare force profiles using the saved insertion-axis metadata and analyzer outputs.
