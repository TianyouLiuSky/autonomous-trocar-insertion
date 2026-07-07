# Lab Note: Scleral Structure and Insertion Force Review

**Date:** 2026-07-06

## Objective

Review scleral anatomy and scleral layer structure, compare published/previous insertion-force experiments with my own eye phantom experiments, and reconsider how the initial insertion stage should be defined in the two-step oblique insertion protocol.

## Background

The current insertion sequence includes a small initial perpendicular insertion before rotating to the oblique condition. In the motion script this has been represented as a fixed depth, currently `0.25 mm`. However, it is not yet clear whether the first stage should be defined by geometric depth alone or by a mechanical/force event associated with initial scleral engagement.

The purpose of today's review was to understand whether scleral structure or previous force experiments provide evidence for a clear force change that could define this initial insertion stage.

## Work Completed

- Read about scleral layer structure and the general organization of scleral tissue.
- Reviewed how the sclera's layered/collagen structure may affect needle insertion behavior.
- Compared other reported insertion-force experiments with my own force data from eye phantom experiments.
- Reconsidered whether the `0.25 mm` initial insertion step should be a fixed-depth command or a force-triggered event.
- Reviewed the limitations of the current force sensor data from the eye phantom experiments.

## Scleral Structure Notes

The sclera is not a uniform simple membrane. It has a layered fibrous structure, with collagen organization and tissue stiffness contributing to nonlinear insertion behavior. This means that the mechanical response during needle insertion may not be a smooth function of depth.

For the project, this matters because a small insertion depth such as `0.25 mm` may not correspond to the same physical tissue state across different samples, phantom conditions, or insertion angles. Depending on the tissue/phantom material and surface condition, the same commanded depth could represent different levels of contact, indentation, partial penetration, or initial puncture.

## Comparison With Other Insertion-Force Experiments

I compared other insertion-force experiments with my own eye phantom experiments. The comparison suggests that insertion-force interpretation is difficult because the measured force depends on:

- material/tissue structure,
- needle geometry,
- insertion angle,
- insertion speed,
- whether the tool is indenting or cutting,
- and sensor noise/baseline stability.

The published or external experiments were useful for understanding the general idea that insertion can include force buildup, puncture, and post-puncture phases. However, I did not find a clear result that directly defines the small initial insertion stage for this project by a specific force change.

## Meeting Note

In the Friday meeting referenced as June 3, we discussed that the initial `0.25 mm` insertion may not necessarily need to be defined by depth. It may instead be better defined by force: for example, the robot could insert until a detectable force event indicates initial scleral engagement.

Date note: June 3, 2026 was a Wednesday, so the exact meeting date should be verified. This likely refers to a Friday project meeting, possibly Friday, July 3, 2026. The technical point recorded here is the meeting conclusion: the initial insertion stage may be force-defined rather than depth-defined.

## Force-Trigger Feasibility

I did not find research that clearly identifies the specific force change or force difference needed to define this initial insertion stage. More importantly, the current force sensor appears too noisy to reliably capture such a small force change in the eye phantom experiments.

Based on the phantom experiments, the expected force signal for the initial insertion event is likely smaller than, or comparable to, the sensor noise and baseline variation. This means that a force-triggered first stage is conceptually attractive, but not currently reliable with the present force sensor setup.

## Interpretation

For now, the `0.25 mm` initial insertion depth should be treated as a practical control approximation rather than a biologically exact definition of scleral entry. It gives the robot a repeatable motion command, but it should not be interpreted as a validated tissue-state boundary.

The longer-term goal could still be to define the initial stage by force, but that would require a cleaner force signal, better filtering, a more reliable sensor, or a larger and more repeatable force feature in the phantom/tissue.

## Next Steps

- Continue using a fixed `0.25 mm` initial insertion depth for controlled experiments unless a reliable force trigger is found.
- Improve force sensing or filtering before attempting force-triggered stage transitions.
- Compare phantom force curves across repeated trials to estimate sensor noise and baseline drift.
- Look for more specific literature on scleral puncture force, not just general insertion force.
- Record whether visible/physical insertion behavior matches the assumed initial insertion stage during future experiments.
