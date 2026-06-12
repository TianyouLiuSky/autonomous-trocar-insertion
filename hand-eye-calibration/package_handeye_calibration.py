#!/usr/bin/env python3
"""Package a validated hand-eye result into a reproducible deployment bundle."""

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_BUNDLES_DIR = SCRIPT_DIR / "calibration_bundles"
REQUIRED_CALIBRATION_KEYS = ("T_cam2base", "T_board2gripper")


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_path(pattern):
    matches = glob.glob(str(pattern))
    if not matches:
        return None
    return Path(max(matches, key=lambda value: Path(value).stat().st_mtime))


def resolve_file(value, default_pattern, label, required=True):
    path = (
        Path(value).expanduser().resolve()
        if value else latest_path(default_pattern)
    )
    if path is None:
        if required:
            raise ValueError(
                "No {} found. Pass its path explicitly.".format(label))
        return None
    if not path.is_file():
        raise ValueError("{} is not a file: {}".format(label, path))
    return path


def scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_residual_csv(path):
    required = (
        "translation_error_mm",
        "rotation_error_deg",
        "translation_error_x_mm",
        "translation_error_y_mm",
        "translation_error_z_mm",
    )
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in required if key not in reader.fieldnames]
        if missing:
            raise ValueError(
                "{} is missing columns: {}".format(
                    path, ", ".join(missing)))
        rows.extend(reader)
    if not rows:
        raise ValueError("Residual CSV is empty: {}".format(path))

    def values(key):
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    translation = values("translation_error_mm")
    rotation = values("rotation_error_deg")
    vectors = np.column_stack([
        values("translation_error_x_mm"),
        values("translation_error_y_mm"),
        values("translation_error_z_mm"),
    ])
    bias = np.mean(vectors, axis=0)
    centered = vectors - bias
    summary = {
        "samples": len(rows),
        "translation_mean_mm": float(np.mean(translation)),
        "translation_max_mm": float(np.max(translation)),
        "translation_rms_mm": float(np.sqrt(np.mean(translation ** 2))),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_max_deg": float(np.max(rotation)),
        "mean_translation_vector_mm": bias,
        "mean_translation_vector_norm_mm": float(np.linalg.norm(bias)),
        "centered_translation_rms_mm": float(np.sqrt(np.mean(
            np.sum(centered ** 2, axis=1)))),
    }
    if rows[0].get("reprojection_error_px", "") != "":
        reprojection = values("reprojection_error_px")
        summary.update({
            "reprojection_mean_px": float(np.mean(reprojection)),
            "reprojection_max_px": float(np.max(reprojection)),
        })
    return summary


def load_calibration(path):
    with np.load(str(path), allow_pickle=True) as data:
        missing = [key for key in REQUIRED_CALIBRATION_KEYS if key not in data]
        if missing:
            raise ValueError(
                "{} is missing calibration matrices: {}".format(
                    path, ", ".join(missing)))
        correction_applied = bool(scalar(
            data, "translation_axis_correction_applied", False))
        if not correction_applied:
            raise ValueError(
                "Calibration does not record an active translation-axis "
                "correction. Refusing to promote an uncorrected result."
            )
        correction_key = "reported_translation_to_orientation_base"
        if correction_key not in data:
            raise ValueError(
                "Corrected calibration is missing {}".format(correction_key))

        metadata = {
            "T_cam2base": np.asarray(data["T_cam2base"], dtype=float),
            "T_board2gripper":
                np.asarray(data["T_board2gripper"], dtype=float),
            "camera_matrix": (
                np.asarray(data["camera_matrix"], dtype=float)
                if "camera_matrix" in data else None),
            "dist_coeffs": (
                np.asarray(data["dist_coeffs"], dtype=float)
                if "dist_coeffs" in data else None),
            "camera_intrinsics_source":
                str(scalar(data, "camera_intrinsics_source", "not recorded")),
            "correction_matrix": np.asarray(
                data[correction_key], dtype=float).reshape(3, 3),
            "correction_source": str(scalar(
                data, "translation_axis_correction_source", "")),
            "correction_angle_deg": float(scalar(
                data, "translation_axis_correction_angle_deg", float("nan"))),
            "correction_axis_dataset": str(scalar(
                data, "translation_axis_correction_axis_dataset", "")),
            "correction_orientation_dataset": str(scalar(
                data,
                "translation_axis_correction_orientation_dataset",
                "",
            )),
            "robot_translation_convention": str(scalar(
                data, "robot_translation_convention", "not recorded")),
            "n_samples": int(scalar(data, "n_samples", 0)),
        }
        solver_keys = (
            "solver_profile",
            "solver_robust_loss",
            "solver_selected_start",
            "solver_successful_starts",
            "solver_requested_starts",
            "solver_rotation_scale_deg",
            "solver_translation_scale_mm",
            "solver_jacobian_condition",
            "rotation_nullspace_gap",
            "translation_condition_number",
            "translation_rank",
            "relative_rotation_mean_deg",
            "relative_rotation_max_deg",
            "translation_err_mean_mm",
            "translation_err_max_mm",
            "rotation_err_mean_deg",
            "rotation_err_max_deg",
        )
        metadata["solver"] = {
            key: scalar(data, key)
            for key in solver_keys if key in data
        }
    return metadata


def matrices_match(first, second, tolerance=1e-8):
    return np.allclose(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
        atol=tolerance,
        rtol=0.0,
    )


def resolve_correction(explicit_path, calibration):
    source = explicit_path or calibration["correction_source"]
    path = Path(source).expanduser().resolve() if source else None
    if path is not None and path.is_file():
        with np.load(str(path), allow_pickle=True) as data:
            key = "reported_translation_to_orientation_base"
            if key not in data:
                raise ValueError(
                    "Correction file does not contain {}: {}".format(
                        key, path))
            if not matrices_match(
                    data[key], calibration["correction_matrix"]):
                raise ValueError(
                    "Correction matrix does not match the matrix embedded "
                    "in the calibration: {}".format(path)
                )
        return path
    if explicit_path:
        raise ValueError("Correction is not a file: {}".format(path))
    return None


def resolve_intrinsics(explicit_path, calibration):
    source = explicit_path or calibration["camera_intrinsics_source"]
    if not source or source in ("D405 factory", "not recorded"):
        raise ValueError(
            "This promotion workflow expects the fitted intrinsics used by "
            "the validated calibration. Pass --intrinsics explicitly."
        )
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Intrinsics is not a file: {}".format(path))
    with np.load(str(path), allow_pickle=True) as data:
        if "camera_matrix" not in data:
            raise ValueError(
                "Intrinsics file lacks camera_matrix: {}".format(path))
        if (
                calibration["camera_matrix"] is not None
                and not matrices_match(
                    data["camera_matrix"], calibration["camera_matrix"])):
            raise ValueError(
                "Intrinsics camera matrix does not match the calibration: "
                "{}".format(path)
            )
    return path


def write_embedded_correction(path, calibration):
    np.savez(
        str(path),
        correction_version=1,
        reported_translation_to_orientation_base=
            calibration["correction_matrix"],
        correction_angle_deg=calibration["correction_angle_deg"],
        axis_dataset=calibration["correction_axis_dataset"],
        orientation_dataset=
            calibration["correction_orientation_dataset"],
        reconstructed_from_calibration=True,
        diagnostic_only=True,
    )


def copy_artifact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))


def file_record(role, destination, source=None):
    return {
        "role": role,
        "filename": destination.name,
        "source": "" if source is None else str(source),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def validation_acceptance(spatial, orientation):
    checks = {
        "spatial_mean_translation_below_1mm":
            spatial["translation_mean_mm"] < 1.0,
        "spatial_max_translation_below_1mm":
            spatial["translation_max_mm"] < 1.0,
        "orientation_mean_rotation_below_0_25deg":
            orientation["rotation_mean_deg"] < 0.25,
        "orientation_max_rotation_below_0_5deg":
            orientation["rotation_max_deg"] < 0.5,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def bundle_readme(bundle_name, manifest):
    spatial = manifest["validation"]["spatial"]
    orientation = manifest["validation"]["orientation"]
    return """# Validated Hand-Eye Calibration Bundle

Bundle: `{bundle_name}`

This bundle contains one mutually consistent calibration set:

- `hand_eye_calibration.npz`: weighted hand-eye solution and solver metadata
- `translation_axis_correction.npz`: FrameEE XYZ basis correction
- `camera_intrinsics.npz`: fitted D405 color-camera intrinsics
- `validation_spatial_residuals.csv`: spatial holdout residuals
- `validation_orientation_residuals.csv`: orientation holdout residuals
- `manifest.json`: checksums, provenance, transforms, and validation summary

Validation summary:

- Spatial: `{spatial_t:.3f} mm` mean, `{spatial_t_max:.3f} mm` max;
  `{spatial_r:.3f} deg` mean rotation.
- Orientation: `{orientation_t:.3f} mm` mean,
  `{orientation_t_max:.3f} mm` max; `{orientation_r:.3f} deg` mean rotation.

## Required Runtime Convention

The calibration and correction are a pair. Before combining a live FrameEE
pose with `T_cam2base`, convert only the FrameEE translation:

```text
t_corrected = reported_translation_to_orientation_base @ t_frameee
R_corrected = R_frameee
```

Do not rotate the FrameEE quaternion with this correction.

For a new calibration using the same hardware state:

```bash
python3 handeye_calibration.py \
  --intrinsics camera_intrinsics.npz \
  --translation-axis-correction translation_axis_correction.npz
```

Re-run axis alignment and calibration after camera remounting, robot kinematic
changes, firmware changes, homing changes, or linkage calibration.
""".format(
        bundle_name=bundle_name,
        spatial_t=spatial["translation_mean_mm"],
        spatial_t_max=spatial["translation_max_mm"],
        spatial_r=spatial["rotation_mean_deg"],
        orientation_t=orientation["translation_mean_mm"],
        orientation_t_max=orientation["translation_max_mm"],
        orientation_r=orientation["rotation_mean_deg"],
    )


def update_current_link(bundles_dir, bundle_dir):
    current = bundles_dir / "current"
    temporary = bundles_dir / ".current_{}".format(os.getpid())
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(bundle_dir.name, target_is_directory=True)
    os.replace(str(temporary), str(current))
    return current


def package(args):
    calibration_path = resolve_file(
        args.calibration,
        OUTPUT_DIR / "hand_eye_cal_*.npz",
        "calibration",
    )
    spatial_csv = resolve_file(
        args.spatial_residuals,
        OUTPUT_DIR / "validation_residuals_weighted_spatial_*.csv",
        "spatial residual CSV",
    )
    orientation_csv = resolve_file(
        args.orientation_residuals,
        OUTPUT_DIR / "validation_residuals_weighted_orientation_*.csv",
        "orientation residual CSV",
    )
    spatial_plot = resolve_file(
        args.spatial_plot,
        OUTPUT_DIR / "spatial_error_map_weighted_spatial_*.png",
        "spatial plot",
        required=False,
    )
    orientation_plot = resolve_file(
        args.orientation_plot,
        OUTPUT_DIR / "orientation_error_map_weighted_orientation_*.png",
        "orientation plot",
        required=False,
    )

    calibration = load_calibration(calibration_path)
    correction_path = resolve_correction(args.correction, calibration)
    intrinsics_path = resolve_intrinsics(args.intrinsics, calibration)
    spatial_summary = summarize_residual_csv(spatial_csv)
    orientation_summary = summarize_residual_csv(orientation_csv)
    acceptance = validation_acceptance(
        spatial_summary, orientation_summary)
    if not acceptance["all_passed"] and not args.allow_failed_validation:
        failed = [
            name for name, passed in acceptance["checks"].items()
            if not passed
        ]
        raise ValueError(
            "Validation acceptance failed: {}. Use "
            "--allow-failed-validation only for an intentionally "
            "non-production bundle.".format(", ".join(failed))
        )

    bundles_dir = Path(args.bundles_dir).expanduser().resolve()
    bundle_name = args.name or "handeye_{}".format(timestamp())
    bundle_dir = bundles_dir / bundle_name
    if bundle_dir.exists():
        raise ValueError("Bundle already exists: {}".format(bundle_dir))
    bundle_dir.mkdir(parents=True)

    files = []
    calibration_destination = bundle_dir / "hand_eye_calibration.npz"
    copy_artifact(calibration_path, calibration_destination)
    files.append(file_record(
        "hand_eye_calibration", calibration_destination, calibration_path))

    correction_destination = (
        bundle_dir / "translation_axis_correction.npz")
    if correction_path is None:
        write_embedded_correction(
            correction_destination, calibration)
        files.append(file_record(
            "translation_axis_correction",
            correction_destination,
            "embedded in {}".format(calibration_path),
        ))
    else:
        copy_artifact(correction_path, correction_destination)
        files.append(file_record(
            "translation_axis_correction",
            correction_destination,
            correction_path,
        ))

    intrinsics_destination = bundle_dir / "camera_intrinsics.npz"
    copy_artifact(intrinsics_path, intrinsics_destination)
    files.append(file_record(
        "camera_intrinsics", intrinsics_destination, intrinsics_path))

    artifact_specs = [
        (
            "validation_spatial_residuals",
            spatial_csv,
            "validation_spatial_residuals.csv",
        ),
        (
            "validation_orientation_residuals",
            orientation_csv,
            "validation_orientation_residuals.csv",
        ),
        (
            "validation_spatial_plot",
            spatial_plot,
            "validation_spatial_error_map.png",
        ),
        (
            "validation_orientation_plot",
            orientation_plot,
            "validation_orientation_error_map.png",
        ),
    ]
    for role, source, filename in artifact_specs:
        if source is None:
            continue
        destination = bundle_dir / filename
        copy_artifact(source, destination)
        files.append(file_record(role, destination, source))

    manifest = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": (
            "validated" if acceptance["all_passed"]
            else "validation_failed_override"
        ),
        "robot": "SHER20",
        "equation": "A Y = X B",
        "calibration": {
            "n_samples": calibration["n_samples"],
            "T_cam2base": calibration["T_cam2base"],
            "T_board2gripper": calibration["T_board2gripper"],
            "solver": calibration["solver"],
            "robot_translation_convention":
                calibration["robot_translation_convention"],
        },
        "translation_axis_correction": {
            "matrix_reported_xyz_to_orientation_base":
                calibration["correction_matrix"],
            "angle_deg": calibration["correction_angle_deg"],
            "origin_offset_known": False,
            "axis_dataset": calibration["correction_axis_dataset"],
            "orientation_dataset":
                calibration["correction_orientation_dataset"],
        },
        "intrinsics": {
            "source": calibration["camera_intrinsics_source"],
            "camera_matrix": calibration["camera_matrix"],
            "dist_coeffs": calibration["dist_coeffs"],
        },
        "validation": {
            "spatial": spatial_summary,
            "orientation": orientation_summary,
            "acceptance": acceptance,
        },
        "files": files,
    }
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(bundle_readme(bundle_name, manifest))
    files.append(file_record("bundle_readme", readme_path))

    # The manifest lists checksums for every payload except itself, avoiding a
    # recursive self-checksum while keeping the bundle independently auditable.
    manifest_path = bundle_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(serializable(manifest), handle, indent=2)
        handle.write("\n")

    current = None
    if args.activate:
        current = update_current_link(bundles_dir, bundle_dir)

    print("\nVALIDATED HAND-EYE BUNDLE")
    print("  bundle: {}".format(bundle_dir))
    print(
        "  spatial:    {:.3f} mm mean / {:.3f} mm max, "
        "{:.3f} deg mean".format(
            spatial_summary["translation_mean_mm"],
            spatial_summary["translation_max_mm"],
            spatial_summary["rotation_mean_deg"],
        )
    )
    print(
        "  orientation:{:.3f} mm mean / {:.3f} mm max, "
        "{:.3f} deg mean".format(
            orientation_summary["translation_mean_mm"],
            orientation_summary["translation_max_mm"],
            orientation_summary["rotation_mean_deg"],
        )
    )
    print("  acceptance: PASS")
    if current is not None:
        print("  active: {} -> {}".format(current, bundle_dir.name))
    print("\nOriginal files were copied, not moved.")
    return bundle_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a corrected, validated hand-eye calibration and its "
            "dependencies into a stable deployment bundle."
        )
    )
    parser.add_argument(
        "--calibration",
        help="Corrected hand_eye_cal_*.npz. Defaults to the latest output.",
    )
    parser.add_argument(
        "--correction",
        help=(
            "translation_axis_correction_*.npz. Defaults to the source "
            "recorded inside the calibration."
        ),
    )
    parser.add_argument(
        "--intrinsics",
        help=(
            "Fitted D405 intrinsics NPZ. Defaults to the source recorded "
            "inside the calibration."
        ),
    )
    parser.add_argument(
        "--spatial-residuals",
        help="Spatial residual CSV. Defaults to the latest weighted result.",
    )
    parser.add_argument(
        "--orientation-residuals",
        help=(
            "Orientation residual CSV. Defaults to the latest weighted "
            "result."
        ),
    )
    parser.add_argument("--spatial-plot", help="Optional spatial plot PNG.")
    parser.add_argument(
        "--orientation-plot", help="Optional orientation plot PNG.")
    parser.add_argument(
        "--bundles-dir",
        default=str(DEFAULT_BUNDLES_DIR),
        help="Destination directory for timestamped bundles.",
    )
    parser.add_argument(
        "--name",
        help="Optional bundle folder name. Defaults to handeye_TIMESTAMP.",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Atomically point calibration_bundles/current at the new bundle.",
    )
    parser.add_argument(
        "--allow-failed-validation",
        action="store_true",
        help=(
            "Package despite failing acceptance checks. Intended only for "
            "diagnostic archives."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        package(parse_args())
    except (OSError, ValueError) as exc:
        raise SystemExit("ERROR: {}".format(exc))
