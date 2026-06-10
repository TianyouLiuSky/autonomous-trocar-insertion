"""Load optional fitted camera intrinsics with resolution checks."""

from pathlib import Path

import numpy as np


def load_intrinsics(path, expected_width, expected_height):
    if path is None:
        return None
    path_text = str(path).strip()
    if not path_text:
        raise ValueError(
            "The fitted-intrinsics path is empty. If using "
            "--intrinsics \"$INTRINSICS\", set INTRINSICS in this terminal "
            "or pass the .npz path directly."
        )
    resolved = Path(path_text).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            "Fitted-intrinsics file does not exist: {}".format(resolved)
        )
    if not resolved.is_file():
        raise ValueError(
            "Fitted-intrinsics path is not a file: {}. Pass the generated "
            "d405_charuco_intrinsics_*.npz file, not its directory."
            .format(resolved)
        )
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
