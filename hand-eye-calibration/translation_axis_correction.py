"""Load and apply the diagnostic FrameEE translation-basis correction."""

from pathlib import Path

import numpy as np

from handeye_math import project_to_rotation, rotation_angle_deg


CORRECTION_VERSION = 1


def load_translation_axis_correction(path):
    """Load a correction mapping reported FrameEE XYZ into orientation base."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(
            "Translation-axis correction is not a file: {}".format(resolved))

    with np.load(str(resolved), allow_pickle=True) as data:
        key = "reported_translation_to_orientation_base"
        if key not in data:
            raise ValueError(
                "{} does not contain {}".format(resolved, key))
        raw_rotation = np.asarray(data[key], dtype=float).reshape(3, 3)
        version = (
            int(data["correction_version"].item())
            if "correction_version" in data else 0
        )
        axis_dataset = (
            str(data["axis_dataset"].item())
            if "axis_dataset" in data else "not recorded"
        )
        orientation_dataset = (
            str(data["orientation_dataset"].item())
            if "orientation_dataset" in data else "not recorded"
        )
        fit_residual_mean_mm = (
            float(data["translation_fit_residual_mean_mm"].item())
            if "translation_fit_residual_mean_mm" in data else float("nan")
        )
        fit_residual_max_mm = (
            float(data["translation_fit_residual_max_mm"].item())
            if "translation_fit_residual_max_mm" in data else float("nan")
        )

    if not np.all(np.isfinite(raw_rotation)):
        raise ValueError("Translation-axis correction contains non-finite data")
    rotation = project_to_rotation(raw_rotation)
    projection_error = float(np.linalg.norm(raw_rotation - rotation))
    if projection_error > 1e-5:
        raise ValueError(
            "Translation-axis correction is not a valid rotation "
            "(projection error {:.3g})".format(projection_error)
        )
    if version not in (0, CORRECTION_VERSION):
        raise ValueError(
            "Unsupported translation-axis correction version: {}".format(
                version)
        )

    return {
        "matrix": rotation,
        "path": str(resolved),
        "version": version,
        "angle_deg": rotation_angle_deg(rotation),
        "axis_dataset": axis_dataset,
        "orientation_dataset": orientation_dataset,
        "translation_fit_residual_mean_mm": fit_residual_mean_mm,
        "translation_fit_residual_max_mm": fit_residual_max_mm,
    }


def correct_translation(translation, correction_matrix):
    """Convert one reported XYZ vector into the orientation-defined base."""
    return (
        np.asarray(correction_matrix, dtype=float).reshape(3, 3)
        @ np.asarray(translation, dtype=float).reshape(3)
    )


def corrected_robot_pose(robot_pose, correction_matrix):
    """Return a pose copy with corrected translation and unchanged rotation."""
    corrected = dict(robot_pose)
    if "t" in robot_pose:
        corrected["t"] = correct_translation(
            robot_pose["t"], correction_matrix)
    if "t_mm" in robot_pose:
        corrected["t_mm"] = correct_translation(
            robot_pose["t_mm"], correction_matrix)
    return corrected
