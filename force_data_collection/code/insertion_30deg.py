#!/usr/bin/env python3
"""Launch the 30-degree-from-horizontal oblique insertion condition.

The straight pose is vertical/direct-down, so this condition is 60 degrees from
straight. Data is still labeled as the experimental 30-degree condition.

The default start position is biased toward the upper/right side of the
observed fixed-orientation corridor. Starting from the geometric workspace
center only leaves a few millimeters before the robot-side translation lock
near x=-19.7, z=3.9.
"""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(
        default_angle_deg=60.0,
        default_label_angle_deg=30.0,
        default_center_position_mm=(-5.5, -109.0, 17.0),
    )
