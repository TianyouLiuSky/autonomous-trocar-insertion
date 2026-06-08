"""Load optional fitted camera intrinsics with resolution checks."""

from pathlib import Path

import numpy as np


def load_intrinsics(path, expected_width, expected_height):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    with np.load(str(resolved), allow_pickle=True) as data:
        if "camera_matrix" not in data:
            raise ValueError(
                "{} does not contain camera_matrix".format(resolved))
        distortion_key = (
            "distortion_coefficients"
            if "distortion_coefficients" in data else "dist_coeffs"
        )
        if distortion_key not in data:
            raise ValueError(
                "{} does not contain distortion coefficients".format(
                    resolved)
            )
        width = (
            int(data["image_width"].item())
            if "image_width" in data else expected_width
        )
        height = (
            int(data["image_height"].item())
            if "image_height" in data else expected_height
        )
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                "{} was calibrated at {}x{}, but the camera stream is {}x{}"
                .format(
                    resolved, width, height,
                    expected_width, expected_height)
            )
        camera_matrix = np.asarray(
            data["camera_matrix"], dtype=float).reshape(3, 3)
        distortion = np.asarray(
            data[distortion_key], dtype=float).reshape(-1)
    return camera_matrix, distortion, str(resolved)
