# Lab Note: Two-Step Oblique Protocol Correction

**Date:** 2026-06-27

## Objective

Record the correction in experimental interpretation: the previous one-step insertion experiments are off target for the final intended insertion protocol, because the actual procedure requires a two-step oblique insertion.

## Background

The force collection work from June 14 through June 22 produced useful infrastructure and pilot motion modes. It helped establish force recording, direct insertion, perpendicular insertion, 30-degree insertion, insertion-axis metadata, workspace safety limits, and live plotting.

However, those previous experiments mainly treated each condition as a separate one-step insertion. That is not the final target motion for the project.

## Correction

The previous one-step insertion experiments should be interpreted as pilot/debugging experiments rather than final protocol experiments.

They are "off target" for the final procedure because they do not reproduce the planned two-step oblique insertion sequence:

1. Approach and insert slightly in the perpendicular direction.
2. Rotate around the physical needle tip into the oblique condition.
3. Insert along the oblique direction.
4. Rotate back around the physical needle tip.
5. Continue the insertion in the perpendicular direction.

The important missing part in the previous experiments is the transition between perpendicular and oblique motion while preserving the remote center of motion at the physical tip.

## Interpretation of Previous Work

The previous force collection scripts are still useful, but their role should be understood correctly:

- They validated the force recorder UI and saved session format.
- They helped define keyboard control behavior and insertion-axis conventions.
- They tested direct, perpendicular, and oblique insertion directions separately.
- They helped identify safe workspace limits and usable start positions.
- They did not fully reproduce the actual two-step oblique insertion strategy.

Therefore, data collected from one-step direct/perpendicular/30-degree trials should not be treated as final evidence for the complete insertion method. It should be treated as pilot data for components of the procedure.

## Revised Plan

Future force collection should be built around the two-step oblique sequence added on June 26. The motion script should be validated first without relying on force results, then paired with the force recorder once the tip-centered RCM behavior is confirmed.

The most important validation criteria are:

- the physical tip remains approximately fixed during rotation stages,
- FrameEE remains inside the safe workspace,
- the oblique stage follows the intended needle direction,
- the sequence completes all stages without hitting workspace or handle-drift limits,
- and the force data can be segmented by stage during analysis.

## Next Steps

- Run slow dry tests of the June 26 tip-centered sequence.
- Confirm the manual/measured tip offset used by the script.
- Add or verify force recording synchronization for the staged two-step sequence.
- Use the one-step insertion data only as pilot/reference data, not as the final protocol dataset.
