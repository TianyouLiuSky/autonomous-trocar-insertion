# Lab Note: Hand-Eye Validation Process Update

**Date:** 2026-06-02

## Objective

Improve the hand-eye calibration validation workflow so that validation results include enough diagnostic information to identify where calibration error is coming from.

## Background

After collecting hand-eye calibration data, the next step is to validate the result using a separate dataset. The earlier validation workflow only saved the basic robot and board poses, which made it difficult to determine whether large validation errors were caused by robot pose error, ChArUco detection quality, camera noise, or the calibration transform itself.

## Work Completed

- Updated the validation data collector to save more information for each validation sample.
- Increased the D405 validation capture rate to 15 fps.
- Required a stronger ChArUco detection before accepting a validation sample.
- Added ChArUco corner count and reprojection error to the saved validation data.
- Added timestamped validation dataset outputs in addition to the latest `validation_dataset.npz` alias.
- Added CSV export for validation samples so individual captures can be inspected outside the GUI.
- Updated the calibration evaluator to automatically locate the latest calibration and validation files.
- Added per-sample residual CSV output for validation.
- Updated the README to document the revised validation workflow.

## Implementation Notes

The validation collector now saves:

- `validation_dataset.npz`
- timestamped validation datasets under `output/`
- validation sample CSV files under `output/`

Each validation sample now includes the robot pose, board pose, detected ChArUco corner count, and ChArUco reprojection error. This makes it possible to sort validation results by sample quality and check whether outliers correspond to poor board detection.

The evaluator was also updated to write a residual CSV. For each validation sample, it computes the difference between the board pose predicted by the robot chain and the board pose predicted by the camera chain:

- robot chain: `A @ Y`
- camera chain: `X @ B`

The residual output includes translation error, rotation error, per-axis translation error, robot pose, board pose, corner count, and reprojection error.

## Result

The validation process became more structured and easier to debug. Instead of only reporting aggregate error, the workflow now produces per-sample records that can be used to identify outliers and compare validation error against image quality.

## Next Steps

- Use the residual CSV to identify the worst validation samples.
- Check whether high-error samples also have low corner counts or high reprojection error.
- Compare validation results across repeated datasets to determine whether the error is repeatable.
