# Lab Note: Hand-Eye Camera Troubleshooting Documentation

**Date:** 2026-06-05

## Objective

Clarify the hand-eye calibration troubleshooting instructions for cases where the D405 camera cannot be opened by the calibration GUI.

## Work Completed

- Updated the hand-eye calibration README troubleshooting section.
- Added an explicit reminder to first confirm that the D405 camera is physically connected to the computer.
- Clarified that the calibration GUI opens the D405 color stream directly, so no other process should be using the camera at the same time.

## Notes

This was a documentation-focused update, but it addressed a practical source of experiment failure. The hand-eye calibration GUI depends on direct access to the D405 color stream. If the camera is disconnected, or if another RealSense/ROS process already owns the stream, the GUI may fail before any calibration or validation data can be collected.

## Result

The README now gives a clearer first troubleshooting step for camera-access problems: verify the physical connection, then check for competing D405 processes.

## Next Steps

- Use this troubleshooting flow before future calibration and validation runs.
- If camera access continues to fail, check whether any ROS D405 publisher, RealSense viewer, or other direct camera script is still running.
