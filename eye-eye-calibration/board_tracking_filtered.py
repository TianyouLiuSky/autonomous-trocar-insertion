#!/usr/bin/env python3
"""Compatibility entry point for the calibrated eye-eye workflow.

Run with --help. Both historical tracker names now share the same implementation.
Capture uses raw measurements; no temporal filtering is applied to calibration.
See README.md for the changed capture controls and required intrinsics.
"""
import sys
from eye_eye_calibration import main

if __name__ == "__main__":
    sys.exit(main())
