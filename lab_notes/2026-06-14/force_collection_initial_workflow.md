# Lab Note: Initial Force Data Collection Workflow

**Date:** 2026-06-14

## Objective

Begin the force data collection phase by creating a dedicated workflow for recording insertion force data during controlled trocar insertion experiments.

## Background

After completing the main hand-eye calibration debugging work, the next stage of the project required collecting force data during insertion. The key need was a repeatable experiment workflow that could control robot motion, lock the insertion orientation, record synchronized force and robot pose data, and save each run in a structured format for later analysis.

## Work Completed

- Created the `force_data_collection` directory as a dedicated workspace for force experiments.
- Added documentation describing the force collection workflow and file organization.
- Implemented a fixed-angle teleoperation script for controlled insertion motion.
- Added wrapper scripts for direct-down insertion and 30-degree oblique insertion.
- Implemented shared force collection helper functions.
- Built a real-time force recorder UI.
- Added an analysis script for completed force sessions.
- Added tests for shared force collection math and helper behavior.
- Added `.gitignore` files so collected datasets can be stored locally without accidentally committing large experiment outputs.

## Implementation Details

The initial force collection workflow was separated into three main parts:

1. Robot control through `fixed_angle_teleop.py`
2. Data recording through `force_recorder_ui.py`
3. Offline inspection through `analyze_force_session.py`

The teleoperation script locks the robot to a fixed insertion orientation and allows the operator to command small translational motions from the keyboard. This is intended to make the insertion motion slow, repeatable, and easy to stop.

The recorder UI subscribes to force, pose, wavelength, command, and insertion-axis information. It saves timestamped force samples together with robot pose and insertion metadata. It also supports baseline/tare behavior so later analysis can use force changes relative to the starting state rather than only raw force values.

The analyzer summarizes a saved force session and produces basic plots and statistics. This gives a first way to check whether a session was recorded correctly and whether force changes are visible during insertion.

## Initial Experiment Modes

Two insertion modes were created at this stage:

- Direct-down insertion using the straight tool orientation.
- 30-degree oblique insertion as a separate wrapper around the same fixed-angle teleoperation logic.

The purpose was not yet to finalize the experiment protocol, but to establish the software structure needed for collecting comparable force datasets across insertion conditions.

## Result

The project now had an initial force data collection subsystem with robot teleoperation, synchronized force recording, saved session metadata, offline analysis, and tests. This provided the foundation for later refinement of controls, angle conventions, safety limits, and experiment-specific insertion modes.

## Next Steps

- Refine the keyboard control mapping so the robot motion is clear and operator-friendly.
- Improve robustness of data recording and topic detection.
- Clarify angle conventions for direct, oblique, and future perpendicular insertion conditions.
- Add safety limits before running more extensive insertion experiments.
