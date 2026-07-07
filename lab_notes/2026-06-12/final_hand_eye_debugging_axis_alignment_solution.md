# Lab Note: Final Hand-Eye Debugging and Axis-Alignment Solution

**Date:** 2026-06-12

## Objective

Complete the hand-eye calibration debugging process by identifying the root cause of the persistent validation error, implementing the correction in the calibration pipeline, validating the corrected result, and documenting/packaging the final hand-eye calibration workflow.

## Background

Before this point, hand-eye calibration had improved but still showed a persistent validation error of approximately 2-3 mm. The error was not explained by the most obvious causes: ChArUco reprojection error was low, board detection was repeatable, camera intrinsics experiments did not remove the issue, and changing solver weights only traded translation error against rotation error.

The validation residuals also had a structured X/Z pattern rather than random noise. This suggested a frame or coordinate-convention problem rather than a simple measurement noise problem.

## Work Completed

- Extended the frame-consistency diagnostics to better compare translation-derived and rotation-derived camera-to-base estimates.
- Improved board-center and board-origin tracing during orientation validation.
- Designed and implemented a dedicated XYZ axis-alignment experiment.
- Added tools to collect and analyze isolated reported X, Y, and Z robot motions.
- Identified that the robot's reported FrameEE translation axes were not aligned with the base frame implied by the FrameEE quaternion.
- Implemented a translation-axis correction for FrameEE XYZ values.
- Updated hand-eye calibration so the correction is applied before solving the hand-eye equations.
- Updated validation so the same correction is applied before computing residuals.
- Added tests for the axis-alignment and translation-correction math.
- Added a packaging script for the final validated hand-eye result.
- Wrote a full report documenting the diagnosis, correction, validation results, and future recalibration rules.

## Root Cause

The root cause was a coordinate-basis mismatch in the robot pose reported by `FrameEE`.

The FrameEE quaternion was internally consistent with observed robot rotation, and the FrameEE XYZ translation values were internally repeatable. However, the translation vector and quaternion were not expressed in the same base coordinate frame.

The measured disagreement was approximately:

```text
15.5 deg around the robot Y axis
```

This means that the reported XYZ translation basis was rotated relative to the base frame represented by the quaternion. As a result, the robot pose used in the hand-eye equation was not a physically valid rigid transform.

The earlier calibration was therefore trying to solve:

```text
A_reported Y = X B
```

where `A_reported` combined rotation and translation components from two slightly different base frames. No solver weighting or camera-intrinsics change could fully fix this contradiction.

## Axis-Alignment Experiment

To confirm the problem, I implemented an axis-alignment test that moves one reported translation axis at a time while holding orientation fixed:

```text
CENTER -> X+ -> CENTER -> X- -> CENTER -> Y+ -> ...
```

The collector records synchronized FrameEE and ChArUco observations, while the analyzer compares camera-observed board displacement against reported robot displacement using an orientation-derived camera-to-base reference.

The confirmed June 12 axis-alignment result showed:

- reported X direction error: 15.975 deg
- reported Y direction error: 1.038 deg
- reported Z direction error: 15.424 deg
- translation/orientation basis disagreement: 15.525 deg about Y
- translation-only residual: 0.149 mm mean, 0.299 mm max
- robot orientation drift: at most 0.001 deg
- ChArUco reprojection error: 0.175 px mean

This confirmed that the issue was a coordinate-basis error, not a constant position offset.

## Correction

The correction maps reported FrameEE XYZ translation into the orientation-defined robot base:

```text
t_corrected = C t_raw
R_corrected = R_raw
```

Only the translation is corrected. The quaternion is left unchanged.

The correction matrix is computed from the relationship between:

- the orientation-derived camera-to-base rotation, and
- the translation-derived camera-to-base rotation from isolated XYZ motion.

The calibration pipeline now solves the hand-eye equation using corrected robot poses:

```text
A_corrected Y = X B
```

This correction is applied before calibration is solved, not as a post-processing adjustment to a completed `T_cam2base`. The same correction is also applied during validation so calibration and validation use the same coordinate convention.

## Validation Result

After applying the translation-axis correction during calibration and validation, the hand-eye result passed both spatial and orientation validation.

| Validation Mode | Mean Translation | Max Translation | Mean Rotation | Max Rotation |
|---|---:|---:|---:|---:|
| Spatial | 0.362 mm | 0.595 mm | 0.149 deg | 0.301 deg |
| Orientation | 0.319 mm | 0.469 mm | 0.102 deg | 0.202 deg |

Mean ChArUco reprojection error remained low:

- Spatial validation: approximately 0.168 px
- Orientation validation: approximately 0.163 px

The previous coherent X/Z validation-error pattern disappeared. The remaining error is sub-millimeter and appears to be a small residual bias rather than a large frame-consistency failure.

## Packaging and Documentation

I added `package_handeye_calibration.py` to bundle the validated calibration artifacts together. The package includes:

- hand-eye calibration transforms,
- translation-axis correction,
- fitted D405 intrinsics,
- spatial and orientation validation residuals,
- validation plots,
- solver metadata,
- checksums and provenance,
- runtime convention documentation.

The packaging script refuses to promote an uncorrected calibration result. This prevents future use of a corrected camera transform without the matching FrameEE translation correction.

I also wrote `HAND_EYE_CALIBRATION_REPORT.md`, which documents the full diagnosis, the correction, validation results, and the rules for when recalibration is required.

## Conclusion

The hand-eye calibration problem was solved by identifying that FrameEE did not report a single consistent rigid pose. Its XYZ translation basis and quaternion-defined orientation basis differed by approximately 15.5 deg around Y.

After correcting FrameEE translation before solving and validating the hand-eye equations, validation error dropped from approximately 2-3 mm to approximately 0.3-0.4 mm.

This completes the main hand-eye debugging phase. Future work should use the packaged calibration bundle and preserve the rule that live FrameEE translations must be corrected before being combined with the validated `T_cam2base`.

## Next Steps

- Use the packaged hand-eye calibration bundle for downstream visual servoing and insertion experiments.
- Apply the same FrameEE translation correction during runtime.
- Re-run axis alignment if the robot firmware, kinematic model, homing definition, encoder calibration, or linkage calibration changes.
- Re-run hand-eye calibration and validation if the camera, robot base, or ChArUco board mounting changes.
