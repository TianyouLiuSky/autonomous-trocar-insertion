#!/usr/bin/env python3
"""Launch fixed-angle teleoperation using the startup pose as 0 degrees."""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(default_angle_deg=0.0)

