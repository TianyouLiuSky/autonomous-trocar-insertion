# Lab Note: Force Collection Angle Convention Update

**Date:** 2026-06-16

## Objective

Clarify and update the insertion angle conventions used by the force data collection scripts, recorder UI, saved metadata, and analysis workflow.

## Background

The force collection workflow needs to compare different insertion conditions. To do this correctly, the condition labels, target orientations, insertion axes, and saved metadata must all use the same convention. If the robot command uses one angle definition while the UI or saved file uses another, the collected force data would be hard to interpret later.

## Work Completed

- Updated the angle definitions used by direct-down and oblique insertion scripts.
- Revised the fixed-angle teleoperation script to handle angle labels more explicitly.
- Updated shared helper functions for insertion condition labeling and insertion-axis handling.
- Updated the recorder UI so the displayed and saved condition labels match the experiment mode.
- Updated tests for the revised angle behavior.
- Updated documentation to describe the current force collection angle convention.

## Angle Convention

The straight/direct-down pose is treated as the reference condition. This is the robot orientation used for direct-down insertion and is documented as approximately:

```text
RPY = (0, -13, 0)
```

The oblique insertion condition is labeled according to the experimental condition being tested, while the internal robot orientation may be represented relative to the straight pose. This distinction matters because the physical question is about insertion angle, while the robot control code needs the orientation needed to realize that insertion axis.

The recorder UI and saved metadata were updated so that the human-readable experiment condition and the robot control angle remain traceable.

## Implementation Details

The shared force collection helper module was expanded so angle handling is not duplicated differently across scripts. This includes helper behavior for condition labels and insertion-axis metadata. The recorder UI was updated to display the target angle and condition more explicitly, and the tests were updated to confirm the expected labels and angle behavior.

## Result

The force collection workflow became more consistent and easier to interpret. This reduces the risk of collecting a dataset where the robot ran one physical insertion condition but the saved session metadata described a different condition.

## Next Steps

- Verify the 30-degree oblique workflow on the robot.
- Confirm whether insertion/retraction should follow robot base Z or the locked tool axis for each condition.
- Add safety checks around workspace limits and allowed insertion travel before longer trials.
