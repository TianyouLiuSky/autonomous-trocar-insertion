#!/usr/bin/env python3

import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from force_collection_common import (  # noqa: E402
    clip_norm,
    ema_update,
    force_topic_candidates,
    insertion_metrics,
    locked_target_rotation,
    pad_force,
    select_force_topic,
    select_wavelength_topic,
    session_directory,
    teleop_velocity,
    wavelength_topic_candidates,
)


class ForceCollectionCommonTests(unittest.TestCase):
    def test_clip_norm(self):
        clipped = clip_norm([3.0, 4.0, 0.0], 2.0)
        self.assertAlmostEqual(np.linalg.norm(clipped), 2.0)

    def test_pad_force(self):
        result = pad_force([1.0, 2.0])
        np.testing.assert_allclose(result[:2], [1.0, 2.0])
        self.assertTrue(math.isnan(result[2]))
        self.assertTrue(math.isnan(result[3]))

    def test_force_topic_candidates_include_both_eye_robot_namespaces(self):
        self.assertEqual(
            force_topic_candidates("SHER20"),
            [
                "/eye_robot/FBGForcesTip",
                "/SHER20/eye_robot/FBGForcesTip",
                "/eye_robot2/ScleraForces",
                "/SHER20/eye_robot2/ScleraForces",
                "/eye_robot2/HandleForces",
                "/SHER20/eye_robot2/HandleForces",
                "/CNN/HandleForce",
            ],
        )

    def test_wavelength_topic_candidates_include_both_eye_robot_namespaces(self):
        self.assertEqual(
            wavelength_topic_candidates("SHER20"),
            [
                "/eye_robot/WavelengthsRaw",
                "/SHER20/eye_robot/WavelengthsRaw",
            ],
        )

    def test_select_force_topic_prefers_legacy_eye_robot_namespace(self):
        result = select_force_topic(
            [
                (
                    "/SHER20/eye_robot/FBGForcesTip",
                    "std_msgs/Float64MultiArray",
                ),
                ("/eye_robot/FBGForcesTip", "std_msgs/Float64MultiArray"),
            ],
            "SHER20",
        )
        self.assertEqual(result, "/eye_robot/FBGForcesTip")

    def test_select_force_topic_accepts_legacy_namespace(self):
        result = select_force_topic(
            ["/eye_robot/FBGForcesTip"],
            "SHER20",
        )
        self.assertEqual(result, "/eye_robot/FBGForcesTip")

    def test_select_wavelength_topic_prefers_legacy_eye_robot_namespace(self):
        result = select_wavelength_topic(
            [
                (
                    "/SHER20/eye_robot/WavelengthsRaw",
                    "std_msgs/Float64MultiArray",
                ),
                ("/eye_robot/WavelengthsRaw", "std_msgs/Float64MultiArray"),
            ],
            "SHER20",
        )
        self.assertEqual(result, "/eye_robot/WavelengthsRaw")

    def test_default_locked_orientations(self):
        straight = locked_target_rotation(
            straight_rpy_deg=[0.0, -13.0, 0.0],
            entry_angle_deg=0.0,
        )
        thirty_from_horizontal = locked_target_rotation(
            straight_rpy_deg=[0.0, -13.0, 0.0],
            entry_angle_deg=60.0,
        )
        np.testing.assert_allclose(
            straight.as_euler("xyz", degrees=True),
            [0.0, -13.0, 0.0],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            thirty_from_horizontal.as_euler("xyz", degrees=True),
            [0.0, 47.0, 0.0],
            atol=1e-10,
        )
        relative = thirty_from_horizontal * straight.inv()
        self.assertAlmostEqual(
            np.linalg.norm(relative.as_rotvec()) * 180.0 / math.pi,
            60.0,
        )

    def test_ema_update_preserves_missing_channel(self):
        previous = np.array([1.0, 2.0, np.nan, np.nan])
        current = np.array([3.0, np.nan, 4.0, np.nan])
        result = ema_update(previous, current, 0.25)
        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(result[1], 2.0)
        self.assertAlmostEqual(result[2], 4.0)
        self.assertTrue(math.isnan(result[3]))

    def test_insertion_metrics(self):
        depth, lateral = insertion_metrics(
            position_mm=[1.0, 2.0, -3.0],
            start_position_mm=[0.0, 0.0, 0.0],
            insertion_axis=[0.0, 0.0, -1.0],
        )
        self.assertAlmostEqual(depth, 3.0)
        self.assertAlmostEqual(lateral, math.sqrt(5.0))

    def test_teleop_velocity_stops_without_active_key(self):
        result = teleop_velocity(
            set(),
            speed=0.2,
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_teleop_velocity_combines_keys_without_exceeding_speed(self):
        result = teleop_velocity(
            {"w", "a", "c"},
            speed=0.2,
        )
        self.assertAlmostEqual(np.linalg.norm(result), 0.2)
        self.assertGreater(result[0], 0.0)
        self.assertGreater(result[1], 0.0)
        self.assertLess(result[2], 0.0)

    def test_teleop_velocity_opposite_keys_cancel(self):
        result = teleop_velocity(
            {"w", "s", "c", "v"},
            speed=0.2,
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_teleop_velocity_exact_base_axis_mapping(self):
        cases = {
            "w": [0.2, 0.0, 0.0],
            "s": [-0.2, 0.0, 0.0],
            "a": [0.0, 0.2, 0.0],
            "d": [0.0, -0.2, 0.0],
            "c": [0.0, 0.0, -0.2],
            "v": [0.0, 0.0, 0.2],
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                result = teleop_velocity({key}, speed=0.2)
                np.testing.assert_allclose(result, expected)

    def test_session_directory(self):
        result = session_directory(
            "/tmp/data",
            30.0,
            now=datetime(2026, 6, 15, 12, 30, 45),
        )
        self.assertEqual(
            result.name, "20260615_123045_angle_p30deg"
        )

    def test_session_directory_with_unknown_angle(self):
        result = session_directory(
            "/tmp/data",
            math.nan,
            now=datetime(2026, 6, 15, 12, 30, 45),
        )
        self.assertEqual(
            result.name, "20260615_123045_angle_unknown"
        )


if __name__ == "__main__":
    unittest.main()
