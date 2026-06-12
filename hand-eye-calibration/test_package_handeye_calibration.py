#!/usr/bin/env python3
"""Regression tests for validated hand-eye calibration packaging."""

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from package_handeye_calibration import package


class PackageHandEyeCalibrationTest(unittest.TestCase):
    def write_residuals(self, path, translation, rotation):
        fieldnames = [
            "translation_error_mm",
            "rotation_error_deg",
            "translation_error_x_mm",
            "translation_error_y_mm",
            "translation_error_z_mm",
            "reprojection_error_px",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(4):
                writer.writerow({
                    "translation_error_mm": translation + index * 0.01,
                    "rotation_error_deg": rotation + index * 0.01,
                    "translation_error_x_mm": 0.1,
                    "translation_error_y_mm": 0.2,
                    "translation_error_z_mm": 0.2,
                    "reprojection_error_px": 0.16,
                })

    def test_packages_corrected_validated_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            correction = root / "correction.npz"
            intrinsics = root / "intrinsics.npz"
            calibration = root / "calibration.npz"
            spatial = root / "spatial.csv"
            orientation = root / "orientation.csv"
            bundles = root / "bundles"
            matrix = np.array([
                [0.963, 0.0, -0.269],
                [0.0, 1.0, 0.0],
                [0.269, 0.0, 0.963],
            ])
            u, _, vt = np.linalg.svd(matrix)
            matrix = u @ vt

            np.savez(
                str(correction),
                correction_version=1,
                reported_translation_to_orientation_base=matrix,
            )
            camera_matrix = np.array([
                [644.0, 0.0, 622.0],
                [0.0, 644.0, 370.0],
                [0.0, 0.0, 1.0],
            ])
            np.savez(
                str(intrinsics),
                camera_matrix=camera_matrix,
                distortion_coefficients=np.zeros(5),
                image_width=1280,
                image_height=720,
            )
            np.savez(
                str(calibration),
                T_cam2base=np.eye(4),
                T_board2gripper=np.eye(4),
                camera_matrix=camera_matrix,
                dist_coeffs=np.zeros(5),
                camera_intrinsics_source=str(intrinsics),
                translation_axis_correction_applied=True,
                reported_translation_to_orientation_base=matrix,
                translation_axis_correction_source=str(correction),
                translation_axis_correction_angle_deg=15.6,
                n_samples=20,
                solver_profile="test",
            )
            self.write_residuals(spatial, 0.35, 0.14)
            self.write_residuals(orientation, 0.31, 0.10)

            args = argparse.Namespace(
                calibration=str(calibration),
                correction=None,
                intrinsics=None,
                spatial_residuals=str(spatial),
                orientation_residuals=str(orientation),
                spatial_plot=None,
                orientation_plot=None,
                bundles_dir=str(bundles),
                name="validated_test",
                activate=True,
                allow_failed_validation=False,
            )
            bundle = package(args)

            self.assertTrue(
                (bundle / "hand_eye_calibration.npz").is_file())
            self.assertTrue(
                (bundle / "translation_axis_correction.npz").is_file())
            self.assertTrue((bundle / "camera_intrinsics.npz").is_file())
            self.assertTrue((bundle / "manifest.json").is_file())
            self.assertTrue((bundles / "current").is_symlink())


if __name__ == "__main__":
    unittest.main()
