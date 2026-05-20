# Lab Note: Pivot Calibration Reproduction

**Date:** 2026-05-19

## Objective

Reproduce the pivot calibration experiment from Luiza's portion of the CIS-II project and investigate the behavior of the pivot point while the robot tip is positioned in the divot.

## Work Completed

- Successfully reproduced the pivot calibration workflow from Luiza's part of the project.
- Observed the same issue reported previously: the estimated pivot point trembles while the robot tip is in the divot.
- Collected three sets of experimental results.
- Reproduced the large calibration error observed in the previous experiments.

## Observations

The pivot point estimate was unstable during the divot-based experiment. The instability appeared as trembling in the pivot point estimate, and the collected results showed a large error across the reproduced trials.

## Hypothesis

One possible explanation is that, in experiment mode, the robot tip moves downward due to gravity, as explained by Tianle. When the tip is inside the divot, the divot applies an upward force on the tip. This upward force may oppose the downward motion and contribute to the observed trembling in the pivot point estimate.

This hypothesis has not yet been verified.

## Next Steps

- Inspect the robot control code to better understand the behavior in experiment mode.
- Run additional experiments to test whether the trembling can be reduced or explained.
- If the issue cannot be resolved through the current setup, try a different pivot calibration approach, such as collecting calibration data only when the robot tip is stationary.
