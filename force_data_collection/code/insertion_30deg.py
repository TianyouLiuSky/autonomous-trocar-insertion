#!/usr/bin/env python3
"""Launch teleoperation 30 degrees from the straight (0, +13, 0) orientation."""

from fixed_angle_teleop import run


if __name__ == "__main__":
    run(default_angle_deg=30.0)
