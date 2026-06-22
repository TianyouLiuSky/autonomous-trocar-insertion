#!/usr/bin/env python3
"""Launch perpendicular-to-eye insertion at absolute RPY (0, +20, 0).

This mode keeps the same keyboard teleoperation behavior as the other force
collection scripts, but the locked orientation is the experimentally observed
perpendicular-to-eye pose. C/V insertion follows the locked tool axis.
"""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(
        default_angle_deg=20.0,
        default_label_angle_deg=20.0,
        default_target_rpy_deg=(0.0, 20.0, 0.0),
        default_insertion_axis_mode="tool-axis",
    )
