#!/usr/bin/env python3
"""Launch teleoperation at the absolute straight orientation (0, -13, 0)."""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(default_angle_deg=0.0)
