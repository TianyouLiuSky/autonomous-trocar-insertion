#!/usr/bin/env python3
"""Synthetic regression tests for the hand-eye initialization helpers."""

import unittest

import numpy as np

from handeye_math import (
    estimate_camera_rotation_from_translations,
    estimate_rotations_from_relative_motion,
    rotation_angle_deg,
    solve_translations,
)
from translation_axis_correction import correct_translation


def euler_xyz_matrix(angles_deg):
    x, y, z = np.deg2rad(angles_deg)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(x), -np.sin(x)],
        [0.0, np.sin(x), np.cos(x)],
    ])
    ry = np.array([
        [np.cos(y), 0.0, np.sin(y)],
        [0.0, 1.0, 0.0],
        [-np.sin(y), 0.0, np.cos(y)],
    ])
    rz = np.array([
        [np.cos(z), -np.sin(z), 0.0],
        [np.sin(z), np.cos(z), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rz @ ry @ rx


class HandEyeMathTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.camera_rotation = euler_xyz_matrix(
            [18.0, -31.0, 72.0])
        self.camera_translation = np.array([-0.16, -0.18, -0.01])
        self.board_rotation = euler_xyz_matrix(
            [90.0, -21.0, -60.0])
        self.board_translation = np.array([0.012, -0.006, 0.027])

    def make_samples(self, count=30):
        robot_rotations = np.array([
            euler_xyz_matrix(angles)
            for angles in self.rng.uniform(
                low=[-20.0, -18.0, -5.0],
                high=[20.0, 18.0, 5.0],
                size=(count, 3),
            )
        ])
        robot_translations = self.rng.uniform(
            low=[-0.035, -0.125, -0.010],
            high=[0.005, -0.090, 0.025],
            size=(count, 3),
        )

        board_rotations = []
        board_translations = []
        for robot_rotation, robot_translation in zip(
                robot_rotations, robot_translations):
            board_rotations.append(
                self.camera_rotation.T
                @ robot_rotation
                @ self.board_rotation
            )
            board_translations.append(
                self.camera_rotation.T @ (
                    robot_rotation @ self.board_translation
                    + robot_translation
                    - self.camera_translation
                )
            )
        return (
            robot_rotations,
            robot_translations,
            np.asarray(board_rotations),
            np.asarray(board_translations),
        )

    def test_relative_motion_recovers_rotations(self):
        robot_r, _, board_r, _ = self.make_samples()
        result = estimate_rotations_from_relative_motion(
            robot_r, board_r)
        self.assertLess(
            rotation_angle_deg(
                result["camera_rotation"]
                @ self.camera_rotation.T),
            1e-5,
        )
        self.assertLess(
            rotation_angle_deg(
                result["board_rotation"]
                @ self.board_rotation.T),
            1e-5,
        )

    def test_linear_translation_solve_recovers_offsets(self):
        robot_r, robot_t, _, board_t = self.make_samples()
        result = solve_translations(
            robot_r, robot_t, board_t, self.camera_rotation)
        np.testing.assert_allclose(
            result["board_translation"],
            self.board_translation,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result["camera_translation"],
            self.camera_translation,
            atol=1e-10,
        )

    def test_translation_sweep_recovers_camera_rotation(self):
        robot_rotation = euler_xyz_matrix([2.0, -3.0, 1.0])
        robot_translations = self.rng.uniform(
            low=[-0.035, -0.125, -0.010],
            high=[0.005, -0.090, 0.025],
            size=(27, 3),
        )
        board_translations = np.array([
            self.camera_rotation.T @ (
                robot_rotation @ self.board_translation
                + robot_translation
                - self.camera_translation
            )
            for robot_translation in robot_translations
        ])
        result = estimate_camera_rotation_from_translations(
            robot_translations, board_translations)
        self.assertLess(
            rotation_angle_deg(
                result["rotation"] @ self.camera_rotation.T),
            1e-5,
        )
        self.assertLess(
            np.max(np.linalg.norm(result["residuals"], axis=1)),
            1e-10,
        )

    def test_axis_correction_restores_translation_solve(self):
        robot_r, physical_robot_t, _, board_t = self.make_samples()
        reported_to_orientation_base = euler_xyz_matrix(
            [0.0, -15.5, 0.0])
        reported_robot_t = np.array([
            reported_to_orientation_base.T @ translation
            for translation in physical_robot_t
        ])
        corrected_robot_t = np.array([
            correct_translation(
                translation, reported_to_orientation_base)
            for translation in reported_robot_t
        ])

        result = solve_translations(
            robot_r,
            corrected_robot_t,
            board_t,
            self.camera_rotation,
        )
        np.testing.assert_allclose(
            result["board_translation"],
            self.board_translation,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result["camera_translation"],
            self.camera_translation,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
