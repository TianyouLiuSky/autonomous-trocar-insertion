#!/usr/bin/env python3
"""Launch the 30-degree-from-horizontal oblique insertion condition.

The straight pose is vertical/direct-down, so this condition is 60 degrees from
straight. Data is still labeled as the experimental 30-degree condition.
"""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(default_angle_deg=60.0, default_label_angle_deg=30.0)
