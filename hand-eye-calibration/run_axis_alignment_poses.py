#!/usr/bin/env python3
"""Move the robot through isolated XYZ excursions for axis validation."""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MOTION_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "motion_script"
if str(MOTION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_SCRIPT_DIR))

from SHER_Controller import SHERController


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
HOME_PATH = SCRIPT_DIR / "home_position" / "home_position.json"

X_LIMITS_MM = (-42.0, 10.0)
Y_LIMITS_MM = (-133.0, -85.0)
Z_LIMITS_MM = (-13.0, 26.0)
ROLL_ABS_LIMIT_DEG = 28.0
POSITION_TOL_MM = 0.5
ORIENTATION_TOL_DEG = 0.2
MOVE_TIMEOUT_SEC = 10.0
MAX_LINEAR_VEL_MM_S = 4.0
MAX_ANGULAR_VEL_RAD_S = 0.05
SETTLE_SEC = 2.0
MAX_MOVE_ATTEMPTS = 2


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_home_position():
    if not HOME_PATH.is_file():
        return None
    with HOME_PATH.open("r") as handle:
        data = json.load(handle)
    return np.asarray(data["pose_mm_deg"], dtype=float)


def pose_error(current, target):
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    current_rotation = Rotation.from_euler(
        "xyz", current[3:], degrees=True)
    target_rotation = Rotation.from_euler(
        "xyz", target[3:], degrees=True)
    rotation_error = target_rotation * current_rotation.inv()
    return (
        float(np.linalg.norm(target[:3] - current[:3])),
        float(np.linalg.norm(rotation_error.as_rotvec())
              * 180.0 / np.pi),
    )


def move_with_retries(robot, target):
    actual = np.asarray(robot.get_current_pose(), dtype=float)
    pos_error, orientation_error = pose_error(actual, target)
    for attempt in range(1, MAX_MOVE_ATTEMPTS + 1):
        print("Attempt {}/{}".format(attempt, MAX_MOVE_ATTEMPTS))
        reached = robot.no_rcm_move_to(
            target,
            position_tol=POSITION_TOL_MM,
            orientation_tol=ORIENTATION_TOL_DEG,
            timeout=MOVE_TIMEOUT_SEC,
            max_linear_vel=MAX_LINEAR_VEL_MM_S,
            max_angular_vel=MAX_ANGULAR_VEL_RAD_S,
            warn_on_angular_limit=False,
        )
        actual = np.asarray(robot.get_current_pose(), dtype=float)
        pos_error, orientation_error = pose_error(actual, target)
        print("Actual: {}".format(np.round(actual, 3).tolist()))
        print(
            "Residual: position={:.3f} mm, orientation={:.3f} deg"
            .format(pos_error, orientation_error)
        )
        if (
                reached
                or (
                    pos_error <= POSITION_TOL_MM
                    and orientation_error <= ORIENTATION_TOL_DEG)):
            return True, actual
    return False, actual


def center_with_margin(pose, step_mm):
    centered = np.asarray(pose, dtype=float).copy()
    centered[5] = 0.0
    for index, (axis, limits) in enumerate((
            ("X", X_LIMITS_MM),
            ("Y", Y_LIMITS_MM),
            ("Z", Z_LIMITS_MM))):
        low, high = limits
        motion_margin = step_mm + POSITION_TOL_MM
        allowed_low = low + motion_margin
        allowed_high = high - motion_margin
        if allowed_low > allowed_high:
            raise ValueError(
                "{} step {:.3f} mm cannot fit within limits {}"
                .format(axis, step_mm, limits)
            )
        original = centered[index]
        centered[index] = np.clip(
            centered[index], allowed_low, allowed_high)
        if abs(centered[index] - original) > 1e-6:
            print(
                "Shifted test center {} from {:.3f} to {:.3f} mm so both "
                "directions remain inside workspace limits."
                .format(axis, original, centered[index])
            )
    return centered


def validate_target(target):
    target = np.asarray(target, dtype=float)
    for index, (axis, limits) in enumerate((
            ("X", X_LIMITS_MM),
            ("Y", Y_LIMITS_MM),
            ("Z", Z_LIMITS_MM))):
        if not limits[0] - 1e-6 <= target[index] <= limits[1] + 1e-6:
            raise ValueError(
                "{} target {:.3f} mm is outside limits {}"
                .format(axis, target[index], limits)
            )
    if abs(target[3]) > ROLL_ABS_LIMIT_DEG + 1e-6:
        raise ValueError(
            "Roll {:.3f} deg exceeds +/-{:.1f} deg"
            .format(target[3], ROLL_ABS_LIMIT_DEG)
        )


def make_sequence(center_pose, step_mm, repeats):
    samples = []

    def append_sample(label, kind, target, axis="", sign=0, repeat=0):
        samples.append({
            "sample": len(samples) + 1,
            "label": label,
            "kind": kind,
            "axis": axis,
            "sign": int(sign),
            "repeat": int(repeat),
            "offset_mm": float(step_mm * sign if kind == "excursion" else 0),
            "target_pose_mm_deg": [
                float(value) for value in target
            ],
        })

    append_sample("CENTER start", "center", center_pose)
    for repeat in range(1, repeats + 1):
        for axis_index, axis_name in enumerate(("X", "Y", "Z")):
            for sign, sign_name in ((1, "+"), (-1, "-")):
                target = center_pose.copy()
                target[axis_index] += sign * step_mm
                validate_target(target)
                append_sample(
                    "{}{} repeat {}".format(
                        axis_name, sign_name, repeat),
                    "excursion",
                    target,
                    axis=axis_name,
                    sign=sign,
                    repeat=repeat,
                )
                append_sample(
                    "CENTER after {}{} repeat {}".format(
                        axis_name, sign_name, repeat),
                    "center",
                    center_pose,
                    axis=axis_name,
                    sign=sign,
                    repeat=repeat,
                )
    return samples


def save_sequence(center_pose, step_mm, repeats, samples):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "axis_alignment_sequence_{}.json".format(stamp())
    payload = {
        "test": "axis_alignment",
        "created": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "step_mm": float(step_mm),
        "repeats": int(repeats),
        "center_pose_mm_deg": [
            float(value) for value in center_pose
        ],
        "samples": samples,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def collector_command(sequence_path, intrinsics, require_fitted):
    parts = [
        "python3 collect_axis_alignment_data.py",
        "--sequence \"{}\"".format(sequence_path),
    ]
    if intrinsics:
        parts.append("--intrinsics \"{}\"".format(
            Path(intrinsics).expanduser().resolve()))
    if require_fitted:
        parts.append("--require-fitted-intrinsics")
    return " \\\n  ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run isolated positive and negative XYZ movements."
    )
    parser.add_argument(
        "--step-mm", type=float, default=10.0,
        help="Magnitude of each isolated axis excursion (default: 10 mm).",
    )
    parser.add_argument(
        "--repeats", type=int, default=2,
        help="Positive/negative passes per axis (default: 2).",
    )
    parser.add_argument(
        "--intrinsics", default=os.environ.get("HE_CAMERA_INTRINSICS"),
        help="Fitted intrinsics path to include in the collector command.",
    )
    parser.add_argument(
        "--require-fitted-intrinsics", action="store_true",
        help="Add the collector's fitted-intrinsics safety check.",
    )
    parser.add_argument(
        "--current-center", action="store_true",
        help="Use the current robot pose instead of the saved calibration home.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.step_mm <= 0:
        raise SystemExit("--step-mm must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    robot = SHERController(robot_name="SHER20")
    requested_center = None
    if not args.current_center:
        requested_center = load_home_position()
    if requested_center is None:
        if not args.current_center:
            print(
                "No saved home was found at {}. Using the current pose."
                .format(HOME_PATH)
            )
        requested_center = np.asarray(
            robot.get_current_pose(), dtype=float)

    center = center_with_margin(requested_center, args.step_mm)
    print("\nAxis-alignment center target:")
    print("  {}".format(np.round(center, 3).tolist()))
    input("Press Enter to move to the test center, or Ctrl+C to stop...")
    reached, actual_center = move_with_retries(robot, center)
    if not reached:
        answer = input(
            "Center was not fully reached. Type USE to use the actual pose, "
            "or press Enter to stop: "
        )
        if answer.strip().upper() != "USE":
            raise SystemExit("Axis-alignment test stopped.")
        center = center_with_margin(actual_center, args.step_mm)
        if np.linalg.norm(center[:3] - actual_center[:3]) > 1e-6:
            raise SystemExit(
                "Actual pose lacks workspace margin for this step size. "
                "Reduce --step-mm or reposition the robot."
            )
    else:
        actual_center = np.asarray(actual_center, dtype=float)
        adjusted_center = center_with_margin(
            actual_center, args.step_mm)
        if np.linalg.norm(
                adjusted_center[:3] - actual_center[:3]) > 1e-6:
            print(
                "The reached center is too close to a workspace boundary. "
                "Moving to the adjusted center."
            )
            reached, actual_center = move_with_retries(
                robot, adjusted_center)
            if not reached:
                raise SystemExit(
                    "Adjusted center was not reached. Reduce --step-mm.")
        center = np.asarray(actual_center, dtype=float)
        center[5] = 0.0

    samples = make_sequence(center, args.step_mm, args.repeats)
    sequence_path = save_sequence(
        center, args.step_mm, args.repeats, samples)

    print("\nSequence saved:")
    print("  {}".format(sequence_path))
    print("Samples: {}".format(len(samples)))
    print("\nIn a second terminal, run:")
    print(collector_command(
        sequence_path, args.intrinsics,
        args.require_fitted_intrinsics))
    input(
        "\nWait until the collector shows the camera image and the correct "
        "NEXT label, then press Enter here..."
    )

    for sample in samples:
        print("\n[{}/{}] {}".format(
            sample["sample"], len(samples), sample["label"]))
        answer = input(
            "Press Enter to move, or type Q to stop the test: "
        )
        if answer.strip().upper() in ("Q", "QUIT"):
            print(
                "Stopped. Press Ctrl+S in the collector to save a partial "
                "dataset."
            )
            return

        target = np.asarray(
            sample["target_pose_mm_deg"], dtype=float)
        reached, actual = move_with_retries(robot, target)
        if not reached:
            print("The target was not fully reached.")
            answer = input(
                "Type R to capture the actual pose anyway, or Q to stop: "
            )
            if answer.strip().upper() != "R":
                print(
                    "Stopped to preserve collector/sequence ordering. "
                    "Press Ctrl+S in the collector to save partial data."
                )
                return

        print("Settling for {:.1f} seconds...".format(SETTLE_SEC))
        time.sleep(SETTLE_SEC)
        print(
            "Press SPACE once in the collector for '{}'."
            .format(sample["label"])
        )
        input(
            "After the collector confirms the sample, press Enter here..."
        )

    print("\nAxis-alignment motion sequence complete.")
    print(
        "The collector should report a complete saved dataset. Leave the "
        "robot at the final center pose."
    )


if __name__ == "__main__":
    main()
