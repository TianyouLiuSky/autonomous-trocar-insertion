# Lab Note: Tip-Measured RCM and Two-Step Oblique Insertion Update

**Date:** 2026-06-30

## Objective

Update the two-step oblique insertion workflow so that RCM behavior is based on the physically measured needle/trocar tip position rather than the unstable divot-based pivot calibration result. Also refine the insertion sequence, workspace safety checks, and tip-offset validation workflow.

## Background

Earlier pivot calibration experiments showed that placing the robot tip in a divot and estimating the pivot point was unreliable. The pivot point trembled because the robot's force-control behavior interacted with the divot contact force. This made strict pivot calibration a poor basis for the final insertion workflow.

For the insertion task, the key requirement is not solving an abstract pivot calibration problem. The key requirement is knowing where the physical trocar tip is relative to the robot-reported `FrameEE` origin. Once that relationship is known, the robot can rotate around the physical tip and maintain the remote center of motion.

Therefore, the RCM strategy was changed to use direct physical measurement of the tip offset.

## Work Completed

- Updated `needle_tip_rcm_sequence.py` to use the manually measured tip offset file by default.
- Added `pivot-calibration/output/manual_measured_tip_offset_29JUN2026_1430.json`.
- Added `motion_script/test_manual_tip_offset.py` to validate the measured tip offset with a small tip-centered orientation sweep.
- Updated the oblique angle logic so the 30-degree experimental condition is computed consistently with the force-collection convention.
- Added FrameEE workspace checks to the tip-centered RCM sequence.
- Redefined and clarified the working space used for the insertion sequence.
- Added explicit language that workspace limits apply to the robot-reported FrameEE position, not directly to the physical tool tip.
- Added `.gitignore` rules for generated RCM and tip-offset test logs.
- Ran a test of the RCM sequence; the test did not complete the full intended insertion protocol.
- Added a safe-start move so the robot first moves FrameEE to the workspace midpoint before beginning tip-centered motion.

## Manual Tip Measurement

The manually measured tip offset is stored as:

```text
pivot-calibration/output/manual_measured_tip_offset_29JUN2026_1430.json
```

Although the file is stored under `pivot-calibration`, the method is explicitly:

```text
manual_physical_measurement
```

The provisional measured offset is:

```text
t_gripper_tip_mm = [0.0, -36.0, -42.74]
```

This means the physical tip is modeled as 36.0 mm in the negative Y direction and 42.74 mm in the negative Z direction from the robot-reported FrameEE origin. The measurement assumes that the FrameEE origin is approximately the marker center, the needle/trocar direction is approximately `-Z`, and the lateral shaft offset is toward robot right.

This is a rough manual estimate and still needs validation, but it is more appropriate for the RCM insertion controller than the unstable divot-based pivot calibration.

## RCM Logic

The physical tool tip is computed from the robot-reported FrameEE pose:

```text
p_tip_base = p_FrameEE_base + R_base_FrameEE * t_FrameEE_tip
```

The goal during an RCM rotation is to change the tool orientation while keeping `p_tip_base` fixed. If the robot only rotates its handle, the physical tip would move in an arc. To prevent this, the script commands a compensating FrameEE linear velocity while applying angular velocity.

For tip-centered motion, the script uses:

```text
v_FrameEE = v_tip - omega x (R_base_FrameEE * t_FrameEE_tip)
```

During pure rotation about the tip, `v_tip` should be approximately zero, so the FrameEE/handle moves in the opposite direction needed to keep the physical tip stationary.

This is the core RCM idea for the insertion script: control the physical tip path, and let the robot handle move as needed to preserve that tip path.

## Tip Offset Validation

The new `test_manual_tip_offset.py` script was added to test whether the physical measurement is plausible before using it for insertion.

The validation idea is:

1. Place the physical tip near a visible mark with no tissue contact.
2. Compute the fixed tip position using the measured offset.
3. Apply a small orientation sweep, default `+/-3 deg`.
4. Watch whether the real physical tip stays near the visible mark.
5. If the tip visibly sweeps around the mark, revise the manual offset before using the insertion sequence.

This is a practical check for the measured `t_FrameEE_tip` value. It does not require divot contact and avoids the force-control instability that made pivot calibration unreliable.

## Insertion Sequence

The two-step insertion sequence was refined as follows:

### Stage 0: Safe Start Move

Move the robot-reported FrameEE to a safe start pose before tip-centered motion begins. The safe start defaults to the workspace midpoint. This setup move is not tip-centered, so the physical tip is allowed to move during this stage.

Purpose: start the experiment from a known safe region with enough workspace margin for the later handle motion.

### Stage 1: Rotate to Perpendicular Pose

Rotate/settle to the perpendicular approach pose while holding the physical tip fixed.

Default perpendicular orientation:

```text
RPY = (0, 20, 0)
```

Purpose: establish the initial perpendicular approach direction without sliding the physical tip.

### Stage 2: First Perpendicular Insertion

Move the physical tip along the perpendicular needle direction.

Default distance:

```text
0.25 mm
```

Purpose: create a small initial perpendicular insertion before changing angle.

### Stage 3: Rotate to 30-Degree Oblique Condition

Rotate into the 30-degree oblique force-collection condition about the current physical tip.

The code computes this condition using the force-collection convention:

```text
straight/down RPY = (0, -13, 0)
oblique tilt = 60 deg from straight
experimental label = 30 deg
```

Purpose: change the insertion angle while preserving the remote center of motion at the tip.

### Stage 4: Oblique Insertion

Move the physical tip along the oblique needle direction.

Default distance on June 30:

```text
0.5 mm
```

Purpose: perform the oblique component of the two-step insertion.

### Stage 5: Rotate Back to Perpendicular

Rotate back to the perpendicular pose about the current physical tip.

Purpose: return to the perpendicular insertion direction while preserving the tip as the motion center.

### Stage 6: Final Perpendicular Insertion

Move the physical tip along the perpendicular needle direction.

Default distance on June 30:

```text
10 mm
```

Purpose: complete the insertion after the oblique step.

## Workspace and Safety

The insertion script uses FrameEE workspace limits:

```text
X: [-42.0, 10.0] mm
Y: [-133.0, -85.0] mm
Z: [-13.0, 30.0] mm
```

These limits are enforced on the robot-reported FrameEE position. The physical tool tip is allowed to differ from FrameEE by the measured rigid offset. This distinction matters because RCM motion intentionally moves the handle/FrameEE around the physical tip.

The script also tracks:

- target and final physical tip position,
- target and final FrameEE position,
- tip error,
- orientation error,
- FrameEE/handle drift,
- workspace violations,
- and per-stage success/failure.

## Test Run Result

A test run was committed on June 30 and marked as failed. The run reached:

- rotation to the perpendicular pose,
- the `0.25 mm` perpendicular insertion,
- and rotation to the oblique condition.

The log showed that the oblique rotation reached with a final orientation error of approximately `0.8045 deg`, but the handle drift during that stage was approximately `19.36 mm`. The sequence did not yet represent a complete successful two-step insertion experiment.

This failure was useful because it showed that the RCM sequence needed clearer workspace handling, safer start positioning, and better validation of the physical tip offset before full insertion experiments.

## Result

The June 30 work changed the RCM insertion strategy from unstable pivot calibration toward a more practical physical-measurement-based model. The code now treats the measured FrameEE-to-tip vector as the key geometric input and uses it to command tip-centered rotations and insertion motions.

The full insertion sequence was not yet validated successfully, but the software now has the correct structure for testing it: manual tip-offset validation, FrameEE-only workspace safety checks, safe-start motion, detailed stage logging, and a staged two-step oblique insertion plan.

## Next Steps

- Validate the manual tip offset with the small physical sweep before running the full insertion.
- Revise the measured offset if the visible tip does not remain fixed during rotation.
- Re-run the two-step sequence from the safe workspace midpoint.
- Check handle drift and workspace violations during the oblique rotation stage.
- Only pair the sequence with force recording after the RCM/tip behavior is physically verified.
