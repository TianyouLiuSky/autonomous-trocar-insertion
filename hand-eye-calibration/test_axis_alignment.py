#!/usr/bin/env python3
"""Synthetic regression tests for the axis-alignment analyzer."""

import unittest

import numpy as np

from analyze_axis_alignment import analyze_displacements


def rotation_y(angle_deg):
    angle = np.deg2rad(angle_deg)
    return np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])


class AxisAlignmentTest(unittest.TestCase):
    def test_detects_translation_basis_tilt(self):
        orientation_camera_rotation = rotation_y(-31.0)
        basis_tilt = rotation_y(15.5)
        movements = []
        for repeat in (1, 2):
            for axis_index, axis in enumerate(("X", "Y", "Z")):
                for sign in (1, -1):
                    robot_delta = np.eye(3)[axis_index] * sign * 0.010
                    physical_delta = basis_tilt @ robot_delta
                    camera_delta = (
                        orientation_camera_rotation.T @ physical_delta)
                    movements.append({
                        "sample": len(movements) + 1,
                        "label": "{}{}".format(
                            axis, "+" if sign > 0 else "-"),
                        "axis": axis,
                        "sign": sign,
                        "repeat": repeat,
                        "offset_mm": sign * 10.0,
                        "camera_delta_m": camera_delta,
                        "robot_delta_m": robot_delta,
                        "robot_orientation_drift_deg": 0.0,
                        "board_orientation_drift_deg": 0.0,
                    })

        result = analyze_displacements(
            movements, orientation_camera_rotation)
        self.assertAlmostEqual(
            result["translation_vs_orientation_deg"], 15.5, places=6)
        self.assertAlmostEqual(
            result["axis_aggregates"]["Y"]["direction_error_deg"],
            0.0, places=6)
        self.assertAlmostEqual(
            result["axis_aggregates"]["X"]["direction_error_deg"],
            15.5, places=6)
        self.assertLess(
            result["translation_fit_residual_max_mm"], 1e-9)


if __name__ == "__main__":
    unittest.main()
