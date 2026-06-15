#!/usr/bin/env python3
"""Launch fixed-angle teleoperation at 30 degrees from the startup pose."""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(default_angle_deg=30.0)

