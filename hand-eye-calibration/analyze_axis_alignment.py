#!/usr/bin/env python3
"""Analyze isolated XYZ movements against an orientation-derived base frame."""

import argparse
import csv
import datetime
import glob
import json
from pathlib import Path

import numpy as np

from handeye_math import (
    estimate_rotations_from_relative_motion,
    project_to_rotation,
    rotation_angle_deg,
    rotation_axis_angle,
    rotation_matrix_from_quaternion_xyzw,
    rotation_matrix_from_rotvec,
    transforms_from_samples,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
AXES = ("X", "Y", "Z")
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_path(patterns):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(str(pattern)))
    if not matches:
        return None
    return Path(max(matches, key=lambda value: Path(value).stat().st_mtime))


def fit_rotation(source_vectors, target_vectors):
    """Return R minimizing ||R * source - target|| for displacement vectors."""
    source = np.asarray(source_vectors, dtype=float)
    target = np.asarray(target_vectors, dtype=float)
    covariance = source.T @ target
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    return rotation


def vector_angle_deg(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-15:
        return float("nan")
    cosine = np.dot(first, second) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def analyze_displacements(movements, orientation_camera_rotation):
    """Compute per-movement and per-axis alignment metrics."""
    orientation_camera_rotation = np.asarray(
        orientation_camera_rotation, dtype=float).reshape(3, 3)
    rows = []
    camera_vectors = []
    robot_vectors = []

    for movement in movements:
        camera_delta = np.asarray(
            movement["camera_delta_m"], dtype=float).reshape(3)
        robot_delta = np.asarray(
            movement["robot_delta_m"], dtype=float).reshape(3)
        visual_base = orientation_camera_rotation @ camera_delta
        error = visual_base - robot_delta
        sign = int(movement["sign"])
        axis = movement["axis"]
        axis_index = AXIS_INDEX[axis]
        primary = visual_base[axis_index]
        cross = np.delete(visual_base, axis_index)

        row = dict(movement)
        row.update({
            "camera_delta_m": camera_delta,
            "robot_delta_m": robot_delta,
            "visual_base_delta_m": visual_base,
            "error_m": error,
            "direction_error_deg": vector_angle_deg(
                visual_base, robot_delta),
            "length_ratio": (
                float(np.linalg.norm(visual_base)
                      / np.linalg.norm(robot_delta))
                if np.linalg.norm(robot_delta) > 1e-12
                else float("nan")
            ),
            "cross_axis_mm": float(np.linalg.norm(cross) * 1000.0),
            "primary_mm": float(primary * 1000.0),
            "signed_visual_base_m": visual_base * sign,
            "signed_robot_delta_m": robot_delta * sign,
        })
        rows.append(row)
        camera_vectors.append(camera_delta)
        robot_vectors.append(robot_delta)

    translation_rotation = fit_rotation(
        camera_vectors, robot_vectors)
    disagreement = (
        translation_rotation @ orientation_camera_rotation.T
    )
    disagreement_axis, disagreement_deg = rotation_axis_angle(disagreement)
    predicted_robot = (
        translation_rotation @ np.asarray(camera_vectors).T
    ).T
    translation_fit_residual_mm = np.linalg.norm(
        predicted_robot - np.asarray(robot_vectors), axis=1) * 1000.0

    aggregates = {}
    direction_columns = []
    for axis in AXES:
        axis_rows = [row for row in rows if row["axis"] == axis]
        signs = {int(row["sign"]) for row in axis_rows}
        if signs != {-1, 1}:
            raise ValueError(
                "Axis {} needs both positive and negative excursions; "
                "found signs {}".format(axis, sorted(signs))
            )
        signed_visual = np.array([
            row["signed_visual_base_m"] for row in axis_rows])
        signed_robot = np.array([
            row["signed_robot_delta_m"] for row in axis_rows])
        mean_visual = np.mean(signed_visual, axis=0)
        mean_robot = np.mean(signed_robot, axis=0)
        expected_axis = np.eye(3)[AXIS_INDEX[axis]]
        direction = mean_visual / np.linalg.norm(mean_visual)
        direction_columns.append(direction)
        aggregates[axis] = {
            "sample_count": len(axis_rows),
            "mean_visual_delta_mm": mean_visual * 1000.0,
            "std_visual_delta_mm": np.std(
                signed_visual * 1000.0, axis=0),
            "mean_robot_delta_mm": mean_robot * 1000.0,
            "direction": direction,
            "direction_error_deg": vector_angle_deg(
                direction, expected_axis),
            "cross_axis_mm": float(
                np.linalg.norm(np.delete(
                    mean_visual, AXIS_INDEX[axis])) * 1000.0),
            "length_ratio": float(
                np.linalg.norm(mean_visual) / np.linalg.norm(mean_robot)),
        }

    direction_matrix = np.column_stack(direction_columns)
    orthogonality_error = direction_matrix.T @ direction_matrix - np.eye(3)
    return {
        "rows": rows,
        "axis_aggregates": aggregates,
        "direction_matrix": direction_matrix,
        "direction_matrix_determinant": float(
            np.linalg.det(direction_matrix)),
        "orthogonality_error_frobenius": float(
            np.linalg.norm(orthogonality_error)),
        "translation_camera_rotation": translation_rotation,
        "translation_vs_orientation_axis": disagreement_axis,
        "translation_vs_orientation_deg": disagreement_deg,
        "translation_fit_residual_mean_mm": float(
            np.mean(translation_fit_residual_mm)),
        "translation_fit_residual_max_mm": float(
            np.max(translation_fit_residual_mm)),
    }


def load_axis_dataset(path):
    with np.load(str(path), allow_pickle=True) as data:
        required = (
            "axis_alignment_plan", "robot_poses",
            "board_rvecs", "board_tvecs",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "{} is missing {}".format(path, ", ".join(missing)))
        plan = list(data["axis_alignment_plan"])
        robot_poses = list(data["robot_poses"])
        board_rvecs = np.asarray(data["board_rvecs"])
        board_tvecs = np.asarray(data["board_tvecs"]).reshape(-1, 3)
        camera_matrix = (
            np.asarray(data["camera_matrix"])
            if "camera_matrix" in data else None)
        intrinsics_source = (
            str(data["camera_intrinsics_source"].item())
            if "camera_intrinsics_source" in data else "not recorded")
        reprojection_errors = (
            np.asarray(data["reprojection_errors_px"], dtype=float)
            if "reprojection_errors_px" in data else None)
        expected_samples = (
            int(data["expected_samples"].item())
            if "expected_samples" in data else len(plan))
        captured_samples = (
            int(data["captured_samples"].item())
            if "captured_samples" in data else len(robot_poses))
    count = len(robot_poses)
    if not (len(plan) == len(board_rvecs) == len(board_tvecs) == count):
        raise ValueError("Axis-alignment dataset arrays have unequal lengths")
    if count < 3:
        raise ValueError("Axis-alignment dataset is too short to analyze")
    if captured_samples != expected_samples or count != expected_samples:
        raise ValueError(
            "Axis-alignment dataset is partial: captured {}/{} samples. "
            "Run a complete sequence for the final alignment analysis."
            .format(count, expected_samples)
        )
    return {
        "plan": plan,
        "robot_poses": robot_poses,
        "board_rvecs": board_rvecs,
        "board_tvecs": board_tvecs,
        "camera_matrix": camera_matrix,
        "intrinsics_source": intrinsics_source,
        "reprojection_errors": reprojection_errors,
        "expected_samples": expected_samples,
    }


def load_orientation_reference(path):
    with np.load(str(path), allow_pickle=True) as data:
        required = ("robot_poses", "board_rvecs", "board_tvecs")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "{} is missing {}".format(path, ", ".join(missing)))
        robot_poses = list(data["robot_poses"])
        board_rvecs = np.asarray(data["board_rvecs"])
        board_tvecs = np.asarray(data["board_tvecs"])
        camera_matrix = (
            np.asarray(data["camera_matrix"])
            if "camera_matrix" in data else None)
        intrinsics_source = (
            str(data["camera_intrinsics_source"].item())
            if "camera_intrinsics_source" in data else "not recorded")
    robot_rotations, _, board_rotations, _ = transforms_from_samples(
        robot_poses, board_rvecs, board_tvecs)
    fit = estimate_rotations_from_relative_motion(
        robot_rotations, board_rotations)
    return {
        "fit": fit,
        "camera_matrix": camera_matrix,
        "intrinsics_source": intrinsics_source,
        "sample_count": len(robot_poses),
    }


def average_neighbor_centers(values, plan, index):
    center_indices = []
    if index > 0 and plan[index - 1]["kind"] == "center":
        center_indices.append(index - 1)
    if index + 1 < len(plan) and plan[index + 1]["kind"] == "center":
        center_indices.append(index + 1)
    if not center_indices:
        raise ValueError(
            "Excursion sample {} is not bracketed by a center sample"
            .format(index + 1)
        )
    return np.mean(np.asarray(values)[center_indices], axis=0)


def make_movements(axis_data):
    plan = axis_data["plan"]
    robot_translations = np.array([
        np.asarray(pose["t"], dtype=float).reshape(3)
        for pose in axis_data["robot_poses"]
    ])
    robot_rotations = np.array([
        rotation_matrix_from_quaternion_xyzw(pose["q"])
        for pose in axis_data["robot_poses"]
    ])
    board_translations = axis_data["board_tvecs"]
    board_rotations = np.array([
        rotation_matrix_from_rotvec(np.asarray(rvec).reshape(3))
        for rvec in axis_data["board_rvecs"]
    ])

    movements = []
    for index, sample in enumerate(plan):
        if sample["kind"] != "excursion":
            continue
        robot_center = average_neighbor_centers(
            robot_translations, plan, index)
        camera_center = average_neighbor_centers(
            board_translations, plan, index)
        robot_rotation_center = average_neighbor_centers(
            robot_rotations, plan, index)
        robot_rotation_center = project_to_rotation(
            robot_rotation_center)
        board_rotation_center = average_neighbor_centers(
            board_rotations, plan, index)
        board_rotation_center = project_to_rotation(
            board_rotation_center)
        movements.append({
            "sample": int(sample["sample"]),
            "label": sample["label"],
            "axis": sample["axis"],
            "sign": int(sample["sign"]),
            "repeat": int(sample["repeat"]),
            "offset_mm": float(sample["offset_mm"]),
            "robot_delta_m":
                robot_translations[index] - robot_center,
            "camera_delta_m":
                board_translations[index] - camera_center,
            "robot_orientation_drift_deg": rotation_angle_deg(
                robot_rotations[index] @ robot_rotation_center.T),
            "board_orientation_drift_deg": rotation_angle_deg(
                board_rotations[index] @ board_rotation_center.T),
        })
    if not movements:
        raise ValueError("No excursion samples were found")
    return movements


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


def write_results(result, axis_path, orientation_path, axis_data,
                  orientation_reference, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    csv_path = output_dir / "axis_alignment_residuals_{}.csv".format(stamp)
    json_path = output_dir / "axis_alignment_summary_{}.json".format(stamp)

    csv_rows = []
    for row in result["rows"]:
        csv_rows.append({
            "sample": row["sample"],
            "label": row["label"],
            "axis": row["axis"],
            "sign": row["sign"],
            "repeat": row["repeat"],
            "robot_dx_mm": row["robot_delta_m"][0] * 1000.0,
            "robot_dy_mm": row["robot_delta_m"][1] * 1000.0,
            "robot_dz_mm": row["robot_delta_m"][2] * 1000.0,
            "visual_dx_mm": row["visual_base_delta_m"][0] * 1000.0,
            "visual_dy_mm": row["visual_base_delta_m"][1] * 1000.0,
            "visual_dz_mm": row["visual_base_delta_m"][2] * 1000.0,
            "error_x_mm": row["error_m"][0] * 1000.0,
            "error_y_mm": row["error_m"][1] * 1000.0,
            "error_z_mm": row["error_m"][2] * 1000.0,
            "direction_error_deg": row["direction_error_deg"],
            "cross_axis_mm": row["cross_axis_mm"],
            "length_ratio": row["length_ratio"],
            "robot_orientation_drift_deg":
                row["robot_orientation_drift_deg"],
            "board_orientation_drift_deg":
                row["board_orientation_drift_deg"],
        })
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "axis_dataset": str(axis_path),
        "orientation_dataset": str(orientation_path),
        "axis_intrinsics_source": axis_data["intrinsics_source"],
        "orientation_intrinsics_source":
            orientation_reference["intrinsics_source"],
        "orientation_reference_sample_count":
            orientation_reference["sample_count"],
        "orientation_reference_residual_mean_deg": float(np.mean(
            orientation_reference["fit"]["sample_errors_deg"])),
        "orientation_reference_residual_max_deg": float(np.max(
            orientation_reference["fit"]["sample_errors_deg"])),
        "axis_reprojection_mean_px": (
            None if axis_data["reprojection_errors"] is None
            else float(np.mean(axis_data["reprojection_errors"]))),
        "axis_reprojection_max_px": (
            None if axis_data["reprojection_errors"] is None
            else float(np.max(axis_data["reprojection_errors"]))),
        "analysis": serialize({
            key: value for key, value in result.items() if key != "rows"
        }),
        "csv": str(csv_path),
    }
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare isolated XYZ movements with a camera-to-base rotation "
            "derived only from the orientation validation set."
        )
    )
    parser.add_argument(
        "--axis-data", default=None,
        help="Axis-alignment dataset. Defaults to the latest dataset.",
    )
    parser.add_argument(
        "--orientation", default=None,
        help=(
            "Orientation validation dataset. Defaults to the newest "
            "timestamped orientation dataset."
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Directory for analysis JSON and CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    axis_path = (
        Path(args.axis_data).expanduser().resolve()
        if args.axis_data else latest_path([
            OUTPUT_DIR / "axis_alignment_dataset_*.npz",
            SCRIPT_DIR / "axis_alignment_dataset.npz",
        ])
    )
    orientation_path = (
        Path(args.orientation).expanduser().resolve()
        if args.orientation else latest_path([
            OUTPUT_DIR / "validation_dataset_orientation_*.npz",
            SCRIPT_DIR / "validation_dataset_orientation.npz",
        ])
    )
    if axis_path is None or not axis_path.is_file():
        raise SystemExit(
            "No axis-alignment dataset found. Pass --axis-data explicitly.")
    if orientation_path is None or not orientation_path.is_file():
        raise SystemExit(
            "No orientation validation dataset found. Pass --orientation.")

    axis_data = load_axis_dataset(axis_path)
    orientation_reference = load_orientation_reference(orientation_path)
    movements = make_movements(axis_data)
    result = analyze_displacements(
        movements,
        orientation_reference["fit"]["camera_rotation"],
    )

    print("\nAXIS-ALIGNMENT ANALYSIS")
    print("Axis data: {}".format(axis_path))
    print("Orientation reference: {}".format(orientation_path))
    print(
        "Intrinsics: axis={} | orientation={}".format(
            axis_data["intrinsics_source"],
            orientation_reference["intrinsics_source"],
        )
    )
    if (
            axis_data["camera_matrix"] is not None
            and orientation_reference["camera_matrix"] is not None):
        intrinsic_delta = float(np.max(np.abs(
            axis_data["camera_matrix"]
            - orientation_reference["camera_matrix"])))
        print("Camera-matrix difference: {:.6f} px".format(
            intrinsic_delta))
        if intrinsic_delta > 1e-6:
            print(
                "WARNING: axis and orientation datasets used different "
                "intrinsics."
            )
    orientation_errors = orientation_reference["fit"]["sample_errors_deg"]
    print(
        "Orientation reference residual mean/max: "
        "{:.3f}/{:.3f} deg".format(
            np.mean(orientation_errors), np.max(orientation_errors))
    )
    if axis_data["reprojection_errors"] is not None:
        print(
            "Axis reprojection mean/max: {:.3f}/{:.3f} px".format(
                np.mean(axis_data["reprojection_errors"]),
                np.max(axis_data["reprojection_errors"]),
            )
        )

    print("\nAxis directions expressed in the orientation-derived base frame:")
    for axis in AXES:
        aggregate = result["axis_aggregates"][axis]
        print(
            "  {}: visual={} mm, robot={} mm, direction={}, "
            "angle={:.3f} deg, cross-axis={:.3f} mm, scale={:.5f}, "
            "repeat-std={} mm".format(
                axis,
                np.round(
                    aggregate["mean_visual_delta_mm"], 3).tolist(),
                np.round(
                    aggregate["mean_robot_delta_mm"], 3).tolist(),
                np.round(aggregate["direction"], 5).tolist(),
                aggregate["direction_error_deg"],
                aggregate["cross_axis_mm"],
                aggregate["length_ratio"],
                np.round(
                    aggregate["std_visual_delta_mm"], 3).tolist(),
            )
        )
    print("\nDirection matrix [physical X Y Z as columns]:")
    print(np.array2string(
        result["direction_matrix"], precision=6, suppress_small=True))
    print(
        "det={:.6f}, orthogonality error={:.6f}".format(
            result["direction_matrix_determinant"],
            result["orthogonality_error_frobenius"],
        )
    )
    print(
        "\nTranslation-only mapping vs orientation mapping: "
        "{:.3f} deg around axis {}".format(
            result["translation_vs_orientation_deg"],
            np.round(
                result["translation_vs_orientation_axis"], 4).tolist(),
        )
    )
    print(
        "Translation-only fit residual mean/max: {:.3f}/{:.3f} mm"
        .format(
            result["translation_fit_residual_mean_mm"],
            result["translation_fit_residual_max_mm"],
        )
    )
    robot_drift = [
        row["robot_orientation_drift_deg"] for row in result["rows"]]
    board_drift = [
        row["board_orientation_drift_deg"] for row in result["rows"]]
    print(
        "Orientation drift robot mean/max: {:.3f}/{:.3f} deg".format(
            np.mean(robot_drift), np.max(robot_drift)))
    print(
        "Orientation drift board mean/max: {:.3f}/{:.3f} deg".format(
            np.mean(board_drift), np.max(board_drift)))

    csv_path, json_path = write_results(
        result,
        axis_path,
        orientation_path,
        axis_data,
        orientation_reference,
        Path(args.output_dir).expanduser().resolve(),
    )
    print("\nSaved:")
    print("  {}".format(json_path))
    print("  {}".format(csv_path))


if __name__ == "__main__":
    main()
