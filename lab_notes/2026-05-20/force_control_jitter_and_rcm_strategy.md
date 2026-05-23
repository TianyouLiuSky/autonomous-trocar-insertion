# Lab Note: Force Control Jitter and RCM Strategy

**Date:** 2026-05-20

## Objective

Investigate the source of the pivot point jitter observed during the divot-based pivot calibration experiment and determine whether strict pivot calibration is necessary for the autonomous trocar insertion workflow.

## Work Completed

- Ran additional experiments to isolate the cause of the jitter observed in the pivot point estimate.
- Verified that the jitter is caused by the force control behavior.
- Reconsidered the calibration strategy for resolving the relationship between the robot tip, which is represented by the force sensor, and the physical trocar tip.

## Findings

The experiments showed that the pivot point jitter is a result of the robot's force control behavior. This supports the previous hypothesis that the interaction between the robot tip and the divot introduces forces that make the pivot point estimate unstable.

Because of this, strict pivot calibration using the divot may not be the most reliable approach for this system.

## Revised Approach

I will switch to physical measurements to resolve the relationship between the robot tip, represented by the force sensor, and the actual trocar tip. This should provide a more direct and stable way to define the tool geometry without relying on unstable pivot calibration data.

The key requirement for the project is the remote center of motion (RCM). Instead of performing strict pivot calibration, the system needs to define the point of motion for the RCM as the pivot tip. As long as the RCM point is correctly defined relative to the trocar tip, the system can maintain the desired insertion motion without requiring a strict pivot calibration procedure.

## Next Steps

- Measure the physical offset between the force sensor tip and the actual trocar tip.
- Use this measured relationship to define the RCM point as the pivot tip.
- Update the robot setup or control assumptions so the RCM motion is based on the measured tip relationship rather than divot-based pivot calibration.
