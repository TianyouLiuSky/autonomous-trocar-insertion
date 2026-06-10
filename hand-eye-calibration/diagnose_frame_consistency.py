#!/usr/bin/env python3
"""Compare camera-to-base rotation inferred from translation and rotation."""

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from handeye_math import (
    estimate_camera_rotation_from_translations,
    estimate_rotations_from_relative_motion,
    rotation_matrix_from_quaternion_xyzw,
    rotation_axis_angle,
    transforms_from_samples,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
HYPOTHESES = (
    "xyzw",
    "xyzw_inverse",
    "wxyz_source",
    "wxyz_source_inverse",
)


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(path, expected_mode):
    with np.load(str(path), allow_pickle=True) as data:
        mode = (
            str(data["validation_mode"].item())
            if "validation_mode" in data else "unknown"
        )
        if mode != expected_mode:
            raise ValueError(
                "{} contains mode {!r}; expected {!r}".format(
                    path, mode, expected_mode)
            )
        required = ("robot_poses", "board_rvecs", "board_tvecs")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "{} is missing {}".format(path, ", ".join(missing))
            )
        robot_poses = list(data["robot_poses"])
        board_rvecs = np.asarray(data["board_rvecs"])
        board_tvecs = np.asarray(data["board_tvecs"])
        metadata = {
            "camera_matrix": (
                np.asarray(data["camera_matrix"])
                if "camera_matrix" in data else None
            ),
            "camera_intrinsics_source": (
                str(data["camera_intrinsics_source"].item())
                if "camera_intrinsics_source" in data
                else "not recorded"
            ),
        }
    return robot_poses, board_rvecs, board_tvecs, metadata


def rotations_for_hypothesis(robot_poses, hypothesis):
    quaternions = np.array([
        np.asarray(pose["q"], dtype=float).reshape(4)
        for pose in robot_poses
    ])
    if hypothesis.startswith("wxyz_source"):
        quaternions = quaternions[:, [1, 2, 3, 0]]
    rotations = np.array([
        rotation_matrix_from_quaternion_xyzw(quaternion)
        for quaternion in quaternions
    ])
    if hypothesis.endswith("_inverse"):
        rotations = np.transpose(rotations, (0, 2, 1))
    return rotations


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def matrix_list(matrix):
    return np.asarray(matrix, dtype=float).round(12).tolist()


def dataset_identity(path, sample_count):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    modified = datetime.datetime.fromtimestamp(
        stat.st_mtime).astimezone().isoformat(timespec="seconds")
    return {
        "path": str(path),
        "sample_count": int(sample_count),
        "size_bytes": int(stat.st_size),
        "modified": modified,
        "sha256": digest.hexdigest(),
        "is_latest_alias": path.name in (
            "validation_dataset_spatial.npz",
            "validation_dataset_orientation.npz",
        ),
    }


def analyze(spatial_path, orientation_path, output_dir):
    (
        spatial_robot,
        spatial_rvecs,
        spatial_tvecs,
        spatial_metadata,
    ) = load_dataset(spatial_path, "spatial")
    (
        orientation_robot,
        orientation_rvecs,
        orientation_tvecs,
        orientation_metadata,
    ) = load_dataset(orientation_path, "orientation")

    (
        _,
        spatial_robot_translations,
        _,
        spatial_board_translations,
    ) = transforms_from_samples(
        spatial_robot, spatial_rvecs, spatial_tvecs)
    spatial_fit = estimate_camera_rotation_from_translations(
        spatial_robot_translations, spatial_board_translations)
    spatial_residual_mm = (
        np.linalg.norm(spatial_fit["residuals"], axis=1) * 1000.0
    )

    _, _, orientation_board_rotations, _ = transforms_from_samples(
        orientation_robot, orientation_rvecs, orientation_tvecs)

    hypothesis_results = []
    orientation_fits = {}
    for hypothesis in HYPOTHESES:
        robot_rotations = rotations_for_hypothesis(
            orientation_robot, hypothesis)
        fit = estimate_rotations_from_relative_motion(
            robot_rotations, orientation_board_rotations)
        disagreement = (
            spatial_fit["rotation"] @ fit["camera_rotation"].T
        )
        axis, angle_deg = rotation_axis_angle(disagreement)
        sample_errors = fit["sample_errors_deg"]
        result = {
            "hypothesis": hypothesis,
            "translation_vs_rotation_disagreement_deg": angle_deg,
            "disagreement_axis_base": axis.round(9).tolist(),
            "orientation_fit_mean_deg": float(np.mean(sample_errors)),
            "orientation_fit_max_deg": float(np.max(sample_errors)),
            "relative_equation_error": fit["equation_error"],
        }
        hypothesis_results.append(result)
        orientation_fits[hypothesis] = fit

    standard = next(
        result for result in hypothesis_results
        if result["hypothesis"] == "xyzw"
    )
    best = min(
        hypothesis_results,
        key=lambda result: (
            result["translation_vs_rotation_disagreement_deg"],
            result["orientation_fit_mean_deg"],
        ),
    )
    standard_fit = orientation_fits["xyzw"]
    spatial_identity = dataset_identity(
        spatial_path, len(spatial_robot))
    orientation_identity = dataset_identity(
        orientation_path, len(orientation_robot))

    stamp = timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_csv = output_dir / (
        "frame_consistency_spatial_residuals_{}.csv".format(stamp)
    )
    pairwise_csv = output_dir / (
        "frame_consistency_pairwise_rotations_{}.csv".format(stamp)
    )
    summary_path = output_dir / (
        "frame_consistency_summary_{}.json".format(stamp)
    )

    spatial_rows = []
    for index, (
            actual, predicted, residual) in enumerate(zip(
                spatial_robot_translations,
                spatial_fit["predicted"],
                spatial_fit["residuals"]), start=1):
        spatial_rows.append({
            "sample": index,
            "robot_x_mm": actual[0] * 1000.0,
            "robot_y_mm": actual[1] * 1000.0,
            "robot_z_mm": actual[2] * 1000.0,
            "predicted_x_mm": predicted[0] * 1000.0,
            "predicted_y_mm": predicted[1] * 1000.0,
            "predicted_z_mm": predicted[2] * 1000.0,
            "residual_x_mm": residual[0] * 1000.0,
            "residual_y_mm": residual[1] * 1000.0,
            "residual_z_mm": residual[2] * 1000.0,
            "residual_mm": np.linalg.norm(residual) * 1000.0,
        })
    write_csv(spatial_csv, spatial_rows)
    write_csv(pairwise_csv, standard_fit["pair_rows"])

    summary = {
        "spatial_dataset": str(spatial_path),
        "orientation_dataset": str(orientation_path),
        "datasets": {
            "spatial": spatial_identity,
            "orientation": orientation_identity,
        },
        "spatial_translation_fit": {
            "mean_residual_mm": float(np.mean(spatial_residual_mm)),
            "max_residual_mm": float(np.max(spatial_residual_mm)),
            "camera_rotation": matrix_list(spatial_fit["rotation"]),
            "singular_values": spatial_fit["singular_values"].tolist(),
        },
        "intrinsics": {
            "spatial_source":
                spatial_metadata["camera_intrinsics_source"],
            "orientation_source":
                orientation_metadata["camera_intrinsics_source"],
            "maximum_camera_matrix_difference_px": (
                float(np.max(np.abs(
                    spatial_metadata["camera_matrix"]
                    - orientation_metadata["camera_matrix"])))
                if (
                    spatial_metadata["camera_matrix"] is not None
                    and orientation_metadata["camera_matrix"] is not None)
                else None
            ),
        },
        "quaternion_hypotheses": hypothesis_results,
        "standard_xyzw": standard,
        "best_diagnostic_hypothesis": best["hypothesis"],
        "acceptance": {
            "target_disagreement_deg": 1.0,
            "ideal_disagreement_deg": 0.5,
            "standard_xyzw_passes": (
                standard[
                    "translation_vs_rotation_disagreement_deg"
                ] < 1.0
            ),
        },
        "warning": (
            "Diagnostic only. Do not apply the fitted disagreement rotation "
            "as a calibration correction."
        ),
        "artifacts": {
            "spatial_residual_csv": str(spatial_csv),
            "pairwise_rotation_csv": str(pairwise_csv),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nFRAME-CONSISTENCY DIAGNOSTIC")
    print("Input datasets:")
    for mode, identity in (
            ("spatial", spatial_identity),
            ("orientation", orientation_identity)):
        print("  {}: {}".format(mode, identity["path"]))
        print(
            "    samples={} modified={} sha256={}...".format(
                identity["sample_count"],
                identity["modified"],
                identity["sha256"][:12],
            )
        )
    if (
            spatial_identity["is_latest_alias"]
            or orientation_identity["is_latest_alias"]):
        print(
            "  NOTE: a mutable latest-dataset alias is in use. Pass the "
            "timestamped NPZ paths when comparing separate captures."
        )
    print(
        "Intrinsics: spatial={} | orientation={}".format(
            spatial_metadata["camera_intrinsics_source"],
            orientation_metadata["camera_intrinsics_source"],
        )
    )
    if (
            spatial_metadata["camera_intrinsics_source"] == "not recorded"
            or orientation_metadata["camera_intrinsics_source"]
            == "not recorded"):
        print(
            "  WARNING: intrinsics provenance is missing. These datasets "
            "cannot distinguish factory from fitted intrinsics."
        )
    intrinsic_delta = summary["intrinsics"][
        "maximum_camera_matrix_difference_px"]
    if intrinsic_delta is not None:
        print("  camera-matrix difference: {:.6f} px".format(
            intrinsic_delta))
        if intrinsic_delta > 1e-6:
            print(
                "  WARNING: the two validation sets used different "
                "camera matrices."
            )
    print("Spatial translation-only fit:")
    print(
        "  residual mean/max: {:.3f} / {:.3f} mm".format(
            np.mean(spatial_residual_mm), np.max(spatial_residual_mm))
    )
    print("\nQuaternion/frame hypotheses:")
    for result in hypothesis_results:
        print(
            "  {:20s} disagreement={:7.3f} deg axis={} "
            "orientation residual={:.3f}/{:.3f} deg".format(
                result["hypothesis"],
                result["translation_vs_rotation_disagreement_deg"],
                np.round(result["disagreement_axis_base"], 4).tolist(),
                result["orientation_fit_mean_deg"],
                result["orientation_fit_max_deg"],
            )
        )
    print("\nStandard FrameEE interpretation: x,y,z,w; EE -> base")
    print(
        "  disagreement: {:.3f} deg".format(
            standard["translation_vs_rotation_disagreement_deg"])
    )
    print(
        "  status: {}".format(
            "PASS (<1 deg)"
            if summary["acceptance"]["standard_xyzw_passes"]
            else "FAIL (>=1 deg)"
        )
    )
    print("  Lowest-disagreement hypothesis: {}".format(
        best["hypothesis"]))
    print("\nSaved:")
    print("  {}".format(summary_path))
    print("  {}".format(spatial_csv))
    print("  {}".format(pairwise_csv))
    print("\nNo correction was applied or saved.")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate camera-to-base rotation independently from spatial and "
            "orientation validation datasets."
        )
    )
    parser.add_argument(
        "--spatial",
        default=str(SCRIPT_DIR / "validation_dataset_spatial.npz"),
        help="Spatial validation .npz.",
    )
    parser.add_argument(
        "--orientation",
        default=str(SCRIPT_DIR / "validation_dataset_orientation.npz"),
        help="Orientation validation .npz.",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Directory for JSON and CSV diagnostics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        Path(args.spatial).expanduser().resolve(),
        Path(args.orientation).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
    )
