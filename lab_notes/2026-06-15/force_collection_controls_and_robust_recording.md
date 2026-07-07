# Lab Note: Force Collection Controls and Robust Recording

**Date:** 2026-06-15

## Objective

Improve the force data collection workflow by making the teleoperation controls clearer and making the recorder more robust for real experiment sessions.

## Background

The initial force collection code provided the basic structure for recording insertion trials, but the control interface and data recording behavior still needed refinement. For physical experiments, the controls must be predictable, easy to remember, and safe to operate while watching the robot and the force signal. The recorder also needs to save enough metadata to understand each session afterward.

## Work Completed

- Updated the keyboard control mapping for force collection teleoperation.
- Revised the control scripts so the insertion and retraction commands are easier to use consistently.
- Improved shared helper functions for interpreting teleoperation commands.
- Made data collection more robust in the force recorder UI.
- Updated direct-down and 30-degree insertion wrappers to match the improved control behavior.
- Added and updated tests for the control mapping and helper functions.
- Updated the force collection README to document the revised operator workflow.

## Control Updates

The force collection teleoperation workflow was reorganized around hold-to-move keyboard commands:

- `W/S` for motion in robot X.
- `A/D` for motion in robot Y.
- `C/V` for insertion and retraction.
- `Space` for stop.
- `Q` for quit.

This control layout separates planar positioning from insertion depth motion. That is important because force collection requires the operator to make small adjustments near the tissue or phantom surface before starting a controlled insertion.

## Recording Improvements

The force recorder was improved to support more reliable experimental records. The recorder stores raw and processed force data, force baseline information, pose snapshots, insertion axis information, operator action state, and command information. This makes the saved data useful not only for force magnitude analysis, but also for checking whether the robot was moving in the intended direction during each part of the trial.

The workflow also moved toward a clearer session structure, where each run has its own saved folder containing sample data, metadata, summary information, and plots. This will make it easier to compare force trials across different insertion conditions.

## Result

The force collection interface became more operator-friendly and more suitable for repeated physical experiments. The code now better records the context of each force sample, which is necessary for interpreting force curves after the experiment instead of relying only on notes taken during operation.

## Next Steps

- Continue refining angle definitions so direct-down and oblique insertion are labeled consistently.
- Verify that the recorder correctly identifies the force topic during live experiments.
- Add more explicit condition labels in the UI and saved metadata.
- Continue testing the keyboard controls before collecting a larger dataset.
