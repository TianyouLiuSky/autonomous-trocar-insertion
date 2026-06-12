# SHER20 Fixed-Camera Hand-Eye Calibration

## Diagnosis, Correction, and Validated Calibration Pipeline

### Executive Summary

The original hand-eye calibration produced low camera reprojection error and
reasonable rotation agreement, but translation validation remained around
2-3 mm. Repeating calibration, changing solver weights, improving robot motion,
and fitting camera intrinsics did not remove this error.

The main problem was ultimately found in the robot pose supplied by
`FrameEE`. Its reported XYZ translation axes were rotated by approximately
15.5 deg around Y relative to the base frame represented by its quaternion.
Therefore, the translation and rotation components of the reported robot pose
did not behave as parts of one consistent rigid transform.

An axis-alignment correction is now applied to every FrameEE translation before
the hand-eye equations are solved. The quaternion is left unchanged. With this
correction, spatial validation improved to 0.362 mm mean translation error and
orientation validation improved to 0.319 mm mean translation error with
0.102 deg mean rotation error.

The correction is now part of the calibration pipeline rather than a
post-processing adjustment to a completed calibration.

---

## 1. Initial Setup

### Physical Arrangement

- An Intel RealSense D405 is mounted beside the table and faces the robot.
- The camera is fixed in the environment. It is not mounted on the robot.
- A ChArUco calibration board is rigidly attached to the robot end effector.
- The center and coordinate origin of the board have a physical offset from
  the end-effector origin.
- The SHER20 robot reports its end-effector pose through the ROS `FrameEE`
  topic.

The objective is to determine the fixed transformation from the D405 camera
frame to the robot base frame.

### Calibration Model

For each recorded pose, the system obtains:

- `A`: end-effector pose in the robot base frame, reported by FrameEE.
- `B`: ChArUco board pose in the camera frame, estimated from the image.
- `X`: camera-to-robot-base transform to be calibrated.
- `Y`: board-to-end-effector transform to be calibrated.

The fixed-camera hand-eye equation is:

```text
A Y = X B
```

Both sides describe the pose of the calibration board in the robot base frame:

```text
robot chain:   base <- end effector <- board
camera chain:  base <- camera <- board
```

### Data Collection Workflow

Calibration uses two programs running in parallel:

1. A capture and calibration interface records synchronized FrameEE and
   ChArUco observations, checks rotational diversity, solves the hand-eye
   equations, and saves the result.
2. A robot motion program moves through a controlled set of 20 poses while
   respecting the measured translation and rotation limits of the robot.

The calibration poses include translation changes and sufficiently diverse
orientations. Robot motion uses reach tolerances and settling periods so that
small controller residuals do not prevent otherwise valid samples from being
recorded.

Separate spatial and orientation validation datasets are used:

- Spatial validation changes XYZ while keeping orientation nearly fixed.
- Orientation validation changes roll and pitch while keeping XYZ nearly
  fixed.

This separation is important because it allows translation and rotation
problems to be diagnosed independently.

---

## 2. Error Initially Discovered

### Validation Symptoms

The original calibration appeared plausible during image processing:

- ChArUco reprojection error was approximately 0.16-0.17 pixels.
- Board detection was repeatable.
- Rotation error was often relatively small.

However, the transformation did not generalize well to validation poses:

- Mean translation error was repeatedly around 2-3 mm.
- Some validation samples exceeded 3-5 mm.
- Changing solver weights improved translation at the cost of rotation, or
  improved rotation at the cost of translation.
- Repeated calibrations without remounting produced similar levels of error.

### Spatial Error Pattern

The spatial validation map showed a coherent X/Z cross-axis pattern rather than
random measurement noise. As the robot moved through the workspace, the error
vectors changed direction systematically.

This was an important clue:

- Random camera noise would produce scattered residual directions.
- A constant translation offset would produce nearly parallel residuals of
  similar size.
- A coordinate-axis rotation produces residuals that change with position and
  form a structured pattern.

The observed error was therefore more consistent with a frame-definition
problem than with an inaccurate constant translation.

### Why the Initial Calibration Residual Was Misleading

A calibration residual measures how closely the fitted transforms satisfy
`A Y = X B` for the samples used during optimization. It does not by itself
prove that the fitted transformation will generalize.

The solver can compromise between incompatible measurements and produce a
moderate training residual. A low or acceptable training residual may therefore
hide:

- insufficient motion diversity,
- overfitting to a small region,
- correlated pose errors,
- or inconsistent coordinate-frame definitions.

Validation exposed that the original solution was not a physically consistent
mapping across translation and orientation changes.

---

## 3. Debugging Overview

The investigation proceeded from common experimental causes toward coordinate
and robot-model causes.

### Motion and Data Collection

- Robot workspace and angular limits were measured and added to the motion
  scripts.
- Position and orientation reach tolerances were separated.
- Motion timeouts, speed, settling time, early exit, and diagnostic logging
  were improved.
- Calibration poses were revised to preserve rotational diversity.
- Data collection was repeated without manual robot intervention.
- Robot pose and camera capture timing were checked.

### Camera and Board

- D405 connectivity, stream configuration, and image dimensions were checked.
- ChArUco repeatability was measured at multiple positions.
- Factory and fitted color-camera intrinsics were compared.
- A dedicated intrinsic calibration was collected with broad image coverage.
- Board geometry, square size, and board planarity were reviewed.

### Calibration Mathematics

- The fixed-camera equation and validation equation were compared against a
  known working implementation.
- Quaternion order, direction, and inversion hypotheses were evaluated.
- Legacy and weighted solvers were compared.
- Rotation and translation were estimated independently.
- Spatial and orientation validation were decoupled.
- The board-origin arc during orientation motion was modeled without requiring
  the robot to rotate around a physical pivot.

### Robot Frame Investigation

- The FrameEE publisher and EyeRobot 2.0 forward-kinematics source were traced.
- The reported quaternion was confirmed to match the robot orientation
  convention.
- Isolated positive and negative X, Y, and Z movements were measured visually
  using the fixed camera.
- These axis movements provided the decisive evidence of a translation-frame
  alignment error.

---

## 4. Main Cause of the High Validation Error

### FrameEE Was Not Acting as One Consistent Transform

A rigid pose is normally written as:

```text
        [ R  t ]
T   =   [ 0  1 ]
```

For this matrix to represent a physical pose, `R` and `t` must use the same
parent coordinate frame.

The experiments showed that:

- FrameEE quaternion rotation was internally consistent with observed robot
  rotation.
- FrameEE XYZ translation was internally repeatable.
- The coordinate basis used by XYZ was rotated relative to the base frame
  implied by the quaternion.

The disagreement was approximately:

```text
15.5 deg around the robot Y axis
```

Y was nearly aligned, while X and Z were mixed.

For an approximately 10 mm commanded translation:

- Reported X motion contained about 2.5-2.8 mm of physical Z motion.
- Reported Z motion contained about 2.5-2.7 mm of physical X motion.
- Reported Y motion remained close to the expected physical Y direction.

This behavior directly explains why the earlier validation error was commonly
around 2-3 mm.

### Mathematical Consequence

The original robot pose was effectively assembled from two different base
frames:

```text
A_reported = [ R_orientation_base   t_translation_base ]
```

This is not a valid rigid transform when the two bases differ.

As a result, the hand-eye equation requested one pair of transforms `X` and `Y`
to satisfy contradictory translation and rotation measurements:

```text
A_reported Y = X B
```

No adjustment of solver weights can fully solve this contradiction.

- Rotation-heavy weighting follows the quaternion-derived geometry but leaves
  large spatial translation error.
- Translation-heavy weighting follows the XYZ-derived geometry but degrades
  orientation agreement.
- Joint weighting produces a compromise between two incompatible frame
  definitions.

The solver was not fundamentally broken. Its inputs did not describe one
physically consistent pose.

### Causes That Were Not Primary

The investigation showed that the following were not the main source of the
2-3 mm validation error:

- D405 image instability
- ChArUco detection noise
- camera and robot sample mismatch
- quaternion component ordering
- quaternion inversion
- robot pose reach residuals
- solver weighting
- factory versus fitted camera intrinsics

Camera intrinsics still affect metric accuracy, but they did not explain the
approximately 15.5 deg translation-orientation frame disagreement.

---

## 5. Solution and New Calibration Pipeline

### 5.1 Estimate the Translation-Basis Correction

The correction is measured using two independent sources of camera-to-base
rotation.

#### Orientation-Derived Mapping

The orientation validation dataset varies robot orientation while holding
position approximately fixed. Relative rotational motion is used to estimate:

```text
R_orientation
```

This is the camera-to-base rotation consistent with the FrameEE quaternion and
the observed ChArUco rotation.

#### Translation-Derived Mapping

The axis-alignment dataset moves one reported robot translation axis at a time
while holding orientation fixed. Camera-observed board displacements are
compared with FrameEE XYZ displacements to estimate:

```text
R_translation
```

This is the camera-to-base rotation required to explain the reported XYZ
coordinate basis.

#### Correction Matrix

The basis conversion from reported FrameEE XYZ into the orientation-defined
robot base is:

```text
C = R_orientation R_translation^T
```

For the current robot configuration, `C` is approximately a 15.5 deg rotation
around Y.

The exact fitted matrix is saved rather than replacing it with an idealized
single-axis rotation. This preserves the small measured X/Y/Z coupling present
in the data.

### 5.2 Correct FrameEE Before Calibration

For every calibration sample, the raw FrameEE pose is separated into:

```text
R_raw = FrameEE quaternion rotation
t_raw = FrameEE XYZ translation
```

Only the translation is corrected:

```text
t_corrected = C t_raw
R_corrected = R_raw
```

The corrected robot pose is:

```text
              [ R_raw   C t_raw ]
A_corrected = [   0         1   ]
```

The hand-eye solver then operates on:

```text
A_corrected Y = X B
```

This is the key architectural change. The correction is applied before the
hand-eye solution is calculated. It is not applied afterward to an already
fitted `T_cam2base`.

An old calibration cannot be repaired reliably by rotating only its final
camera transform because `X` and `Y` were jointly optimized using inconsistent
robot poses. A new calibration must be solved from corrected sample poses.

### 5.3 Solve the Corrected Hand-Eye Problem

The corrected calibration pipeline:

1. Records the raw FrameEE pose and ChArUco pose.
2. Converts FrameEE XYZ using the measured correction matrix.
3. Preserves the original FrameEE quaternion.
4. Checks that accepted board orientations satisfy the diversity threshold.
5. Initializes rotation from relative motion.
6. Solves translation linearly with rotation held fixed.
7. Refines `X` and `Y` jointly using robust weighted least squares.
8. Uses multiple deterministic starting points to check optimizer stability.
9. Saves both the corrected solution and the original raw measurements.

The saved calibration records:

- `T_cam2base`
- `T_board2gripper`
- raw FrameEE poses
- corrected robot poses
- ChArUco poses
- correction matrix and its provenance
- camera intrinsics and distortion
- solver weights and conditioning diagnostics
- calibration residual statistics

### 5.4 Apply the Same Convention During Validation

Validation must use the same coordinate convention as calibration.

For each validation sample:

```text
t_validation_corrected = C t_validation_raw
R_validation_corrected = R_validation_raw
```

The evaluator then compares:

```text
robot prediction:   A_corrected Y
camera prediction:  X B
```

Applying the correction only during calibration, but not during validation,
would recreate the original inconsistency.

### 5.5 Apply the Same Convention at Runtime

The validated camera transform and translation-axis correction form one
calibration package. Any downstream application using the camera-to-base
transform must process live FrameEE poses in the same way:

```text
t_live_corrected = C t_live_raw
R_live_corrected = R_live_raw
```

The correction must not be applied to the quaternion.

Using corrected `T_cam2base` with uncorrected FrameEE XYZ would mix coordinate
conventions and reintroduce the spatial error.

### 5.6 Coordinate-Origin Limitation

The axis-alignment experiment measures coordinate directions. It determines
the rotation between the reported translation basis and the
orientation-defined base, but it does not independently determine whether
their coordinate origins have a constant offset.

This does not prevent hand-eye calibration. A constant base-origin offset is
absorbed into the solved camera translation. However, the resulting
`T_cam2base` belongs to the corrected coordinate convention and must remain
paired with the correction matrix.

### 5.7 Camera Intrinsics Policy

Fitted D405 intrinsics were not the primary fix. Both factory and fitted
intrinsics showed the same large frame disagreement before the translation
correction.

The fitted intrinsics should nevertheless remain in the validated calibration
package because:

- the final hand-eye calibration was solved with them,
- validation was performed with the same model,
- they preserve the exact tested camera configuration,
- and changing intrinsics changes the ChArUco metric pose estimates.

Returning to factory intrinsics would create a different calibration
configuration. It would require a new hand-eye calibration and new validation,
even if the expected change is small.

### 5.8 Validated Result

After applying the correction during calibration and validation:

| Validation mode | Mean translation | Maximum translation | Mean rotation | Maximum rotation |
|---|---:|---:|---:|---:|
| Spatial | 0.362 mm | 0.595 mm | 0.149 deg | 0.301 deg |
| Orientation | 0.319 mm | 0.469 mm | 0.102 deg | 0.202 deg |

Mean ChArUco reprojection error remained approximately:

| Validation mode | Mean reprojection |
|---|---:|
| Spatial | 0.168 px |
| Orientation | 0.163 px |

The previous coherent X/Z error pattern disappeared. The remaining translation
residual is primarily a small sub-millimeter bias rather than a large
position-dependent frame rotation.

The remaining bias should not be fitted away using the validation set.
Validation must remain independent from calibration so that it continues to
measure generalization honestly.

### 5.9 Calibration Packaging

Once a calibration passes spatial and orientation validation, the following
items are preserved together as one versioned bundle:

- hand-eye calibration transforms
- translation-axis correction
- fitted camera intrinsics
- solver configuration
- spatial and orientation validation residuals
- validation figures
- source-file checksums
- provenance and coordinate-convention documentation

This prevents a future user from accidentally combining:

- a corrected camera transform with raw FrameEE translations,
- a correction from a different robot configuration,
- or intrinsics from a different camera configuration.

### 5.10 Future Recalibration Rules

The required recalibration depends on what changed.

| System change | Required action |
|---|---|
| Camera moved or remounted | Re-run hand-eye calibration and validation using the existing correction |
| Robot base moved relative to camera | Re-run hand-eye calibration and validation |
| ChArUco board remounted | Re-run hand-eye calibration and validation |
| Same camera, lens, focus, and resolution | Reuse the fitted intrinsics |
| Camera, resolution, lens, or focus changed | Refit intrinsics, then recalibrate and validate |
| Robot firmware, kinematics, linkage calibration, encoder calibration, or homing definition changed | Repeat axis alignment, generate a new correction, then recalibrate and validate |
| System restarted without physical or configuration changes | Reuse the validated calibration bundle |

Moving the camera does not normally change the robot's internal translation
basis error. Changes to the robot kinematic model or coordinate definitions
may change it.

---

## Final Takeaway

The high validation error was not primarily caused by camera noise, solver
weights, or insufficient pose accuracy. The robot supplied a pose whose XYZ
translation basis and quaternion rotation basis differed by approximately
15.5 deg around Y.

The corrected pipeline now:

1. measures this basis disagreement independently,
2. converts FrameEE XYZ into the quaternion-defined base frame,
3. solves hand-eye calibration using the corrected robot poses,
4. applies the same correction during validation and runtime,
5. and packages all dependent calibration artifacts together.

This changes the calibration problem from an internally contradictory rigid
transform fit into a consistent hand-eye calibration problem, reducing
translation validation error from approximately 2-3 mm to approximately
0.3-0.4 mm.
