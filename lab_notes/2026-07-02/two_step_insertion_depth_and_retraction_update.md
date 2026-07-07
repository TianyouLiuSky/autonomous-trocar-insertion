# Lab Note: Two-Step Insertion Depth and Retraction Update

**Date:** 2026-07-02

## Objective

Refine the two-step tip-centered RCM insertion protocol after running several physical experiments, with the goal of making the insertion depths better match the intended sclera-entry sequence and adding an explicit retraction step.

## Background

By this point, the project had moved away from one-step insertion trials and toward the staged two-step oblique insertion sequence. The June 30 work established the physical-tip-based RCM logic, safe FrameEE start position, workspace checks, and staged motion script.

The next task was to tune the actual motion distances. The two-step sequence needs to represent the intended insertion behavior rather than only demonstrate that the robot can move through the stages. In particular, the intro sclera/oblique portion needed a more meaningful insertion depth, and the full experiment should include controlled retraction after insertion.

## Work Completed

- Ran several physical experiments with the staged RCM insertion workflow.
- Updated the oblique insertion depth for the intro sclera portion.
- Changed the oblique insertion stage from `0.5 mm` to `2.0 mm`.
- Changed the final perpendicular insertion stage from `10.0 mm` to `8.0 mm`.
- Added an explicit retraction step after the final perpendicular insertion.
- Added `--retract-step-mm` as a configurable argument.
- Set the default retraction distance to `20.0 mm`.
- Updated the script documentation and stage labels to match the revised sequence.
- Added validation that the retraction distance cannot be negative.

## Experiment Notes

Several experiments were run to evaluate the staged insertion behavior. These experiments were used to determine that the previous `0.5 mm` oblique step was too small for the intended intro sclera insertion portion. The protocol was therefore adjusted so the oblique stage performs a `2.0 mm` insertion.

At the same time, the later perpendicular insertion was reduced from `10.0 mm` to `8.0 mm`. This preserves the staged structure while keeping the total insertion motion more controlled.

The experiments also showed that the sequence should not end at the deepest insertion position. A controlled retraction stage is needed so the robot can back the physical tip out along the same perpendicular axis and leave the system in a more recoverable state after a trial.

## Updated Insertion Sequence

The current two-step sequence is:

1. Move the robot-reported FrameEE to a safe start pose, defaulting to the workspace center.
2. Rotate/settle to the perpendicular approach pose, default `RPY = (0, 20, 0)`.
3. Move the physical tool tip `0.25 mm` along the perpendicular needle direction.
4. Rotate to the 30-degree oblique force-collection condition about the physical tip.
5. Move the physical tool tip `2.0 mm` along the oblique needle direction.
6. Rotate back to the perpendicular pose about the physical tip.
7. Move the physical tool tip `8.0 mm` along the perpendicular needle direction.
8. Retract the physical tool tip `20.0 mm` back along the same perpendicular axis.

## Retraction Logic

The retraction stage uses the same perpendicular needle axis as the final insertion stage, but with the sign reversed:

```text
target_tip = current_tip - retract_step_mm * perpendicular_needle_axis
```

This is important because retraction should follow the path of the inserted tool rather than move in an arbitrary robot-base direction. Keeping retraction aligned with the perpendicular needle direction should reduce unnecessary lateral motion during withdrawal.

## Result

The July 2 updates made the staged insertion protocol more complete and more experimentally meaningful. The oblique stage now uses a `2.0 mm` intro sclera insertion depth, the final perpendicular stage uses `8.0 mm`, and the full sequence now includes a `20.0 mm` retraction step.

This makes the script closer to a full trial workflow: approach, small perpendicular insertion, oblique insertion, return to perpendicular, finish insertion, and retract.

## Next Steps

- Continue running physical trials with the updated depth and retraction settings.
- Verify that retraction follows the intended physical path without excessive lateral tip motion.
- Check whether the `2.0 mm` oblique stage produces the expected force profile.
- Pair the completed sequence with force recording once the motion is repeatable.
