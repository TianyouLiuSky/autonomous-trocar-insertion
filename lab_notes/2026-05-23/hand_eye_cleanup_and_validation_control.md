# Lab Note: Hand-Eye Calibration Cleanup and Validation Control

**Date:** 2026-05-23

## Objective

Clean up the hand-eye calibration workflow and improve the consistency between calibration and validation so future experiments produce more reliable results.

## Work Completed

- Merged duplicated hand-eye calibration folders into a single working calibration directory.
- Edited the hand-eye calibration motion code so the calibration and validation workflows use similar workspace regions and scopes, while still using different robot poses.
- Added a mechanism to save a home position during calibration.
- Updated the validation workflow so it can return to the saved calibration home position before collecting validation data.

## Hand-Eye Calibration Changes

The calibration pose sequence now saves the starting robot pose as a home position before generating calibration targets. This home position is saved so that later validation runs can begin from the same reference pose.

The calibration and validation pose sets are intentionally similar in spatial region and scope, but they are not identical. This should make validation more meaningful: the validation data tests the same workspace region as calibration while still using separate poses.

The validation script now checks for the saved home position and moves the robot back to it before generating validation poses. This should help control experimental variables, since calibration and validation can start from the same physical setup instead of depending on whatever pose the robot happens to be in at the beginning of validation.

## Rationale

Hand-eye calibration results can be hard to interpret if calibration and validation are collected in different regions or from uncontrolled starting conditions. By keeping the calibration and validation regions similar and reusing the saved home position, the experiment should better isolate the quality of the calibration itself.

## Next Steps

- Run the updated hand-eye calibration experiment when the lab reopens on Tuesday, May 26, 2026.
- Collect validation data from the controlled home position and compare the results against the previous workflow.
- Continue RCM experiments after lab access resumes, using more controlled trials and recorded observations.
