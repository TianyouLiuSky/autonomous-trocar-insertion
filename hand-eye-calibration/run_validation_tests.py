#!/usr/bin/env python3
"""
Run decoupled hand-eye validation motion and evaluate the captured dataset.

Use exactly one validation mode:
  -s / --spatial:     vary XYZ while holding RPY constant
  -o / --orientation: vary roll/pitch while holding XYZ constant

The camera collector remains in a separate window because it owns the D405.
After the collector saves, this script evaluates the new dataset using either
the latest weighted calibration or a calibration timestamp selected with -t.
"""

import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
MOTION_SCRIPT_DIR = REPO_DIR / "motion_script"
OUTPUT_DIR = SCRIPT_DIR / "output"
HOME_PATH = SCRIPT_DIR / "home_position" / "home_position.json"

MOVE_TIMEOUT_SEC = 10.0
POSITION_TOL_MM = 0.5
ORIENTATION_TOL_DEG = 0.2
MAX_MOVE_ATTEMPTS = 2
MAX_LINEAR_VEL_MM_S = 5.0
MAX_ANGULAR_VEL_RAD_S = 0.05
SETTLE_SEC = 2.0

X_LIMITS_MM = (-42.0, 10.0)
Y_LIMITS_MM = (-133.0, -85.0)
Z_LIMITS_MM = (-13.0, 26.0)
ROLL_ABS_LIMIT_DEG = 28.0

SPATIAL_OFFSETS_MM = (-12.0, 0.0, 12.0)
ORIENTATION_ROLL_OFFSETS_DEG = (-12.0, -6.0, 0.0, 6.0, 12.0)
ORIENTATION_PITCH_OFFSETS_DEG = (-12.0, -6.0, 0.0, 6.0, 12.0)

MODE_SAMPLE_COUNTS = {
    "spatial": 27,
    "orientation": 25,
}
QUIT_COMMANDS = {"Q", "QUIT"}
RECORD_ANYWAY_COMMANDS = {"R"}


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_quit_command(answer):
    return answer.strip().upper() in QUIT_COMMANDS


def is_record_anyway_command(answer):
    return answer.strip().upper() in RECORD_ANYWAY_COMMANDS


def normalize_command_pose(pose):
    pose = np.asarray(pose, dtype=float).copy()
    pose[5] = 0.0
    return pose


def orientation_error_deg(current_pose, target_pose):
    current_rotation = Rotation.from_euler(
        "xyz", np.asarray(current_pose, dtype=float)[3:], degrees=True
    )
    target_rotation = Rotation.from_euler(
        "xyz", np.asarray(target_pose, dtype=float)[3:], degrees=True
    )
    relative = target_rotation * current_rotation.inv()
    return float(np.linalg.norm(relative.as_rotvec()) * 180.0 / np.pi)


def pose_error(current_pose, target_pose):
    current_pose = np.asarray(current_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    position_error = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
    return position_error, orientation_error_deg(current_pose, target_pose)


def limits_text(limits):
    return "[{:.3f}, {:.3f}]".format(limits[0], limits[1])


def validate_target_pose(target, label="target"):
    target = np.asarray(target, dtype=float)
    for axis_name, value, limits in (
            ("x", target[0], X_LIMITS_MM),
            ("y", target[1], Y_LIMITS_MM),
            ("z", target[2], Z_LIMITS_MM)):
        if value < limits[0] - 1e-6 or value > limits[1] + 1e-6:
            raise ValueError(
                "{} {}={:.3f} mm is outside {} limits {} mm".format(
                    label, axis_name, value, axis_name.upper(),
                    limits_text(limits))
            )
    if abs(target[3]) > ROLL_ABS_LIMIT_DEG + 1e-6:
        raise ValueError(
            "{} roll={:.3f} deg exceeds the absolute roll limit +/-{:.1f} deg"
            .format(label, target[3], ROLL_ABS_LIMIT_DEG)
        )
    if abs(target[5]) > 1e-5:
        raise ValueError(
            "{} yaw={:.3f} deg is not supported; yaw must remain 0"
            .format(label, target[5])
        )


def fit_offsets_to_limits(base_value, desired_offsets, limits, axis_name):
    offsets = np.asarray(desired_offsets, dtype=float).copy()
    low, high = limits
    desired_span = float(np.ptp(offsets))
    available_span = high - low

    if desired_span > available_span:
        center = (np.max(offsets) + np.min(offsets)) / 2.0
        scale = available_span / desired_span
        offsets = (offsets - center) * scale + center
        print(
            "Compressed {} span from {:.3f} to {:.3f} mm to fit limits."
            .format(axis_name.upper(), desired_span, desired_span * scale)
        )

    target_low = base_value + np.min(offsets)
    target_high = base_value + np.max(offsets)
    shift = 0.0
    if target_low < low:
        shift = low - target_low
    elif target_high > high:
        shift = high - target_high
    offsets += shift

    if shift:
        print(
            "Shifted {} offsets by {:+.3f} mm to fit limits {}."
            .format(axis_name.upper(), shift, limits_text(limits))
        )
    return offsets


def load_home_position():
    if not HOME_PATH.exists():
        return None
    with HOME_PATH.open("r") as handle:
        data = json.load(handle)
    return normalize_command_pose(data["pose_mm_deg"])


def center_first(entries):
    center_entries = [
        entry for entry in entries
        if all(abs(value) < 1e-9 for value in entry["offset"])
    ]
    other_entries = [
        entry for entry in entries
        if not all(abs(value) < 1e-9 for value in entry["offset"])
    ]
    return center_entries + other_entries


def generate_spatial_targets(base_pose):
    """Generate a 3x3x3 XYZ grid with exactly one fixed orientation."""
    base_pose = normalize_command_pose(base_pose)
    x_offsets = fit_offsets_to_limits(
        base_pose[0], SPATIAL_OFFSETS_MM, X_LIMITS_MM, "x")
    y_offsets = fit_offsets_to_limits(
        base_pose[1], SPATIAL_OFFSETS_MM, Y_LIMITS_MM, "y")
    z_offsets = fit_offsets_to_limits(
        base_pose[2], SPATIAL_OFFSETS_MM, Z_LIMITS_MM, "z")

    entries = []
    for z_index, dz in enumerate(z_offsets):
        y_order = list(y_offsets)
        if z_index % 2:
            y_order.reverse()
        for y_index, dy in enumerate(y_order):
            x_order = list(x_offsets)
            if (z_index + y_index) % 2:
                x_order.reverse()
            for dx in x_order:
                target = base_pose.copy()
                target[:3] += np.array([dx, dy, dz], dtype=float)
                label = "spatial_dx{:+.1f}_dy{:+.1f}_dz{:+.1f}".format(
                    dx, dy, dz)
                validate_target_pose(target, label)
                entries.append({
                    "label": label,
                    "target": target,
                    "offset": (float(dx), float(dy), float(dz)),
                })
    return center_first(entries)


def generate_orientation_targets(base_pose):
    """Generate a full roll/pitch grid at exactly one fixed XYZ position."""
    base_pose = normalize_command_pose(base_pose)
    roll_offsets = fit_offsets_to_limits(
        base_pose[3], ORIENTATION_ROLL_OFFSETS_DEG,
        (-ROLL_ABS_LIMIT_DEG, ROLL_ABS_LIMIT_DEG), "roll")
    pitch_offsets = np.asarray(
        ORIENTATION_PITCH_OFFSETS_DEG, dtype=float)

    entries = []
    for pitch_index, dpitch in enumerate(pitch_offsets):
        roll_order = list(roll_offsets)
        if pitch_index % 2:
            roll_order.reverse()
        for droll in roll_order:
            target = base_pose.copy()
            target[3] += droll
            target[4] += dpitch
            label = "orientation_droll{:+.1f}_dpitch{:+.1f}".format(
                droll, dpitch)
            validate_target_pose(target, label)
            entries.append({
                "label": label,
                "target": target,
                "offset": (float(droll), float(dpitch)),
            })
    return center_first(entries)


def summarize_targets(mode, targets):
    xyz = np.array([entry["target"][:3] for entry in targets])
    rpy = np.array([entry["target"][3:] for entry in targets])
    print("")
    print("{} validation sequence".format(mode.capitalize()))
    print("  Poses: {}".format(len(targets)))
    print("  XYZ span (mm): {}".format(np.ptp(xyz, axis=0).round(3).tolist()))
    print("  RPY span (deg): {}".format(np.ptp(rpy, axis=0).round(3).tolist()))
    print(
        "  Reach tolerance: {:.1f} mm, {:.1f} deg".format(
            POSITION_TOL_MM, ORIENTATION_TOL_DEG)
    )
    if mode == "spatial":
        print("  Diagnostic invariant: RPY must be constant for all poses.")
        if not np.allclose(rpy, rpy[0], atol=1e-9):
            raise RuntimeError("Spatial validation generated varying orientation")
    else:
        print("  Diagnostic invariant: XYZ must be constant for all poses.")
        if not np.allclose(xyz, xyz[0], atol=1e-9):
            raise RuntimeError("Orientation validation generated varying XYZ")


def resolve_calibration(timeframe, solution):
    def sort_key(path):
        stamp = path.stem.replace("hand_eye_cal_", "", 1)
        try:
            parsed = datetime.datetime.strptime(stamp, "%d%b%Y_%H%M%S")
            return (1, parsed.timestamp())
        except ValueError:
            return (0, path.stat().st_mtime)

    candidates = sorted(
        OUTPUT_DIR.glob("hand_eye_cal_*.npz"),
        key=sort_key,
    )
    if timeframe:
        supplied = Path(timeframe).expanduser()
        if supplied.is_file():
            selected = supplied.resolve()
        else:
            token = supplied.name.lower().replace(".npz", "")
            matches = [
                path for path in candidates
                if token in path.stem.lower()
            ]
            if not matches:
                available = "\n".join(
                    "  {}".format(path.name) for path in candidates[-10:]
                )
                raise FileNotFoundError(
                    "No calibration matched timeframe {!r}. Available:\n{}"
                    .format(timeframe, available or "  none")
                )
            selected = matches[-1]
    else:
        if not candidates:
            raise FileNotFoundError(
                "No output/hand_eye_cal_*.npz calibration files were found"
            )
        selected = candidates[-1]

    with np.load(str(selected), allow_pickle=True) as data:
        required = (
            ("T_cam2base", "T_board2gripper")
            if solution == "weighted"
            else ("legacy_T_cam2base", "legacy_T_board2gripper")
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "{} does not contain the {} solution keys: {}"
                .format(selected.name, solution, ", ".join(missing))
            )
        profile = (
            str(data["solver_profile"].item())
            if "solver_profile" in data else "not recorded"
        )
        rotation_scale = (
            float(data["solver_rotation_scale_deg"].item())
            if "solver_rotation_scale_deg" in data else None
        )
        translation_scale = (
            float(data["solver_translation_scale_mm"].item())
            if "solver_translation_scale_mm" in data else None
        )

    print("")
    print("Selected calibration: {}".format(selected))
    print("  Solution: {}".format(solution))
    print("  Solver profile: {}".format(profile))
    if rotation_scale is not None and translation_scale is not None:
        print(
            "  Weight scales: {:.3f} deg rotation = {:.3f} mm translation"
            .format(rotation_scale, translation_scale)
        )
    return selected


def move_with_retries(robot, target):
    current_pose = robot.get_current_pose()
    position_error, angular_error = pose_error(current_pose, target)

    for attempt in range(1, MAX_MOVE_ATTEMPTS + 1):
        print("Attempt {}/{}".format(attempt, MAX_MOVE_ATTEMPTS))
        success = robot.no_rcm_move_to(
            target,
            position_tol=POSITION_TOL_MM,
            orientation_tol=ORIENTATION_TOL_DEG,
            timeout=MOVE_TIMEOUT_SEC,
            max_linear_vel=MAX_LINEAR_VEL_MM_S,
            max_angular_vel=MAX_ANGULAR_VEL_RAD_S,
            warn_on_angular_limit=False,
        )
        current_pose = normalize_command_pose(robot.get_current_pose())
        position_error, angular_error = pose_error(current_pose, target)
        print("Actual pose: {}".format(
            [round(float(value), 3) for value in current_pose]))
        print(
            "Residual: position={:.3f} mm, orientation={:.3f} deg"
            .format(position_error, angular_error)
        )

        close_enough = (
            position_error <= POSITION_TOL_MM
            and angular_error <= ORIENTATION_TOL_DEG
        )
        if success or close_enough:
            return True, current_pose, position_error, angular_error
        if attempt < MAX_MOVE_ATTEMPTS:
            print("Target was not reached. Retrying from current pose...")

    return False, current_pose, position_error, angular_error


def wait_for_capture(index, total, mode, target, actual_pose):
    print("")
    print("{} pose {}/{} is ready to capture.".format(
        mode.capitalize(), index, total))
    print("Target: {}".format(
        [round(float(value), 3) for value in target]))
    print("Actual: {}".format(
        [round(float(value), 3) for value in actual_pose]))
    print("Settling for {:.1f}s before capture...".format(SETTLE_SEC))
    time.sleep(SETTLE_SEC)
    answer = input(
        "Press SPACE in the matching collector, then press Enter here; "
        "type Q to stop after this sample: "
    )
    return is_quit_command(answer)


def ask_failed_pose(index, total, target, actual_pose,
                    position_error, angular_error):
    print("")
    print("Pose {}/{} did not fully reach the target.".format(index, total))
    print("Target: {}".format(
        [round(float(value), 3) for value in target]))
    print("Actual: {}".format(
        [round(float(value), 3) for value in actual_pose]))
    print(
        "Residual: position={:.3f} mm, orientation={:.3f} deg"
        .format(position_error, angular_error)
    )
    answer = input(
        "Press Enter to skip, type R to capture anyway, or type Q to stop: "
    )
    if is_quit_command(answer):
        return "quit"
    if is_record_anyway_command(answer):
        return "record"
    return "skip"


def log_row(mode, index, entry, actual_pose,
            position_error, angular_error, status):
    target = entry["target"]
    row = {
        "mode": mode,
        "sample": index,
        "label": entry["label"],
        "status": status,
        "position_error_mm": round(float(position_error), 6),
        "orientation_error_deg": round(float(angular_error), 6),
    }
    names = ("x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")
    for prefix, pose in (("target", target), ("actual", actual_pose)):
        for name, value in zip(names, pose):
            row["{}_{}".format(prefix, name)] = round(float(value), 6)
    return row


def write_motion_log(mode, rows):
    if not rows:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "validation_motion_{}_{}.csv".format(
        mode, timestamp())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def find_new_dataset(mode, run_started):
    candidates = sorted(
        OUTPUT_DIR.glob("validation_dataset_{}_*.npz".format(mode)),
        key=lambda path: path.stat().st_mtime,
    )
    new_candidates = [
        path for path in candidates
        if path.stat().st_mtime >= run_started - 2.0
    ]
    return new_candidates[-1] if new_candidates else None


def evaluate_dataset(calibration_path, dataset_path, solution, show_plot):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_calibration.py"),
        "--calib", str(calibration_path),
        "--validation", str(dataset_path),
        "--solution", solution,
    ]
    if not show_plot:
        command.append("--no-show")
    print("")
    print("Running evaluation:")
    print("  {}".format(" ".join(command)))
    return subprocess.call(command)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled spatial or orientation hand-eye validation and "
            "evaluate with a selected calibration timeframe."
        )
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-s", "--spatial", action="store_const", dest="mode",
        const="spatial",
        help="Vary XYZ over 27 poses while holding orientation constant."
    )
    mode_group.add_argument(
        "-o", "--orientation", action="store_const", dest="mode",
        const="orientation",
        help="Vary roll/pitch over 25 poses while holding XYZ constant."
    )
    parser.add_argument(
        "-t", "--calibration-time", "--weight-timeframe",
        dest="calibration_time", default=None,
        help=(
            "Calibration timestamp or .npz path, for example "
            "07JUN2026_173734. Defaults to the newest hand_eye_cal_*.npz."
        )
    )
    parser.add_argument(
        "--solution", choices=("weighted", "legacy"), default="weighted",
        help="Evaluate the weighted solution by default, or its legacy baseline."
    )
    parser.add_argument(
        "--no-evaluate", action="store_true",
        help="Run motion and capture only; do not launch evaluation afterward."
    )
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Open the Matplotlib result window after evaluation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    calibration_path = resolve_calibration(
        args.calibration_time, args.solution)
    expected_samples = MODE_SAMPLE_COUNTS[args.mode]
    run_started = time.time()

    if str(MOTION_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(MOTION_SCRIPT_DIR))
    from SHER_Controller import SHERController

    robot = SHERController(robot_name="SHER20")
    home_pose = load_home_position()
    if home_pose is None:
        print("No saved home position found at: {}".format(HOME_PATH))
        answer = input(
            "Press Enter to use the current pose as the validation center, "
            "or type Q to stop: "
        )
        if is_quit_command(answer):
            return 1
    else:
        print("")
        print("Saved calibration home position:")
        print("  {}".format(
            [round(float(value), 3) for value in home_pose]))
        answer = input(
            "Press Enter to move to this home position, or type Q to stop: "
        )
        if is_quit_command(answer):
            return 1
        reached, actual_home, position_error, angular_error = (
            move_with_retries(robot, home_pose)
        )
        if not reached:
            answer = input(
                "Home was not fully reached. Type CONTINUE to use the actual "
                "pose as center, or press Enter to stop: "
            )
            if answer.strip().upper() != "CONTINUE":
                return 1

    center_pose = normalize_command_pose(robot.get_current_pose())
    targets = (
        generate_spatial_targets(center_pose)
        if args.mode == "spatial"
        else generate_orientation_targets(center_pose)
    )
    if len(targets) != expected_samples:
        raise RuntimeError(
            "Generated {} poses, expected {}".format(
                len(targets), expected_samples)
        )
    summarize_targets(args.mode, targets)

    print("")
    print(
        "In the other terminal run: python3 collect_validation_data.py -{}"
        .format("s" if args.mode == "spatial" else "o")
    )
    answer = input(
        "When that collector is ready, press Enter to start motion, "
        "or type Q to stop: "
    )
    if is_quit_command(answer):
        return 1

    recorded = 0
    skipped = 0
    stopped = False
    motion_rows = []

    try:
        for index, entry in enumerate(targets, start=1):
            target = entry["target"]
            print("")
            print("=" * 80)
            print(
                "Moving to {} pose {}/{} ({}): {}".format(
                    args.mode, index, len(targets), entry["label"],
                    [round(float(value), 3) for value in target])
            )
            print("=" * 80)
            answer = input(
                "Press Enter to move, or type Q to stop the run: "
            )
            if is_quit_command(answer):
                stopped = True
                break

            success, actual_pose, position_error, angular_error = (
                move_with_retries(robot, target)
            )
            status = "reached" if success else "failed"

            if success:
                stop_after_capture = wait_for_capture(
                    index, len(targets), args.mode, target, actual_pose)
                recorded += 1
                status = "recorded"
                if stop_after_capture:
                    stopped = True
            else:
                action = ask_failed_pose(
                    index, len(targets), target, actual_pose,
                    position_error, angular_error)
                if action == "record":
                    stop_after_capture = wait_for_capture(
                        index, len(targets), args.mode, target, actual_pose)
                    recorded += 1
                    status = "recorded_anyway"
                    if stop_after_capture:
                        stopped = True
                elif action == "quit":
                    status = "failed_then_stopped"
                    stopped = True
                else:
                    skipped += 1
                    status = "skipped"

            motion_rows.append(log_row(
                args.mode, index, entry, actual_pose,
                position_error, angular_error, status))
            if stopped:
                break
    finally:
        motion_log_path = write_motion_log(args.mode, motion_rows)
        if motion_log_path:
            print("Motion log -> {}".format(motion_log_path))

    print("")
    print(
        "Sequence {}. Recorded={}, skipped={}."
        .format("stopped" if stopped else "complete", recorded, skipped)
    )
    return_answer = input(
        "Press Enter to return to the validation center, "
        "or type Q to leave the robot at its current pose: "
    )
    if not is_quit_command(return_answer):
        move_with_retries(robot, center_pose)

    if args.no_evaluate:
        return 0
    if recorded != expected_samples or skipped:
        print(
            "Automatic evaluation skipped because this run did not record "
            "all {} expected samples.".format(expected_samples)
        )
        return 0

    answer = input(
        "Wait for the collector to show SAVED, then press Enter to evaluate; "
        "type Q to skip evaluation: "
    )
    if is_quit_command(answer):
        return 0

    dataset_path = find_new_dataset(args.mode, run_started)
    if dataset_path is None:
        print(
            "No new {} timestamped dataset was found under {}."
            .format(args.mode, OUTPUT_DIR)
        )
        print(
            "Run evaluate_calibration.py manually with the collector's saved "
            "dataset and calibration:\n  {}"
            .format(calibration_path)
        )
        return 1

    return evaluate_dataset(
        calibration_path, dataset_path, args.solution, args.show_plot)


if __name__ == "__main__":
    raise SystemExit(main())
