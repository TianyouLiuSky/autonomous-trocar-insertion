#!/usr/bin/env python3
"""
Tip-centered RCM needle sequence for SHER.

Sequence:
1. Rotate to the perpendicular approach pose, default RPY (0, 20, 0).
2. Move the physical tool tip along the needle direction for 0.25 mm.
3. Rotate to the oblique pose, default RPY (0, 60, 0), about the tool tip.
4. Move the physical tool tip along the needle direction for 0.25 mm.
5. Rotate back to the perpendicular pose about the tool tip.
6. Move the physical tool tip along the needle direction for 10 mm.

The important distinction is that rotations are centered on the physical tool
tip, not the robot end-effector origin. The script uses the calibrated
end-effector-to-tip translation t_gripper_tip_mm:

    p_tip_base = p_gripper_base + R_base_gripper * t_gripper_tip_mm

During rotation-only stages, the commanded end-effector linear velocity is:

    v_gripper = - omega x (R_base_gripper * t_gripper_tip_mm)

so the physical tip stays fixed while the handle moves around it.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from SHER_Controller import SHERController


DEFAULT_PIVOT_JSON = (
    Path(__file__).resolve().parents[1]
    / "pivot-calibration"
    / "output"
    / "old"
    / "manual_pivot_calibration_19MAY2026_155450.json"
)
LOG_DIR = Path(__file__).resolve().parent / "tip_rcm_sequence_logs"
EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tip-centered RCM needle sequence.")
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument(
        "--pivot-calibration-json",
        default=str(DEFAULT_PIVOT_JSON),
        help="JSON containing t_gripper_tip_mm. Ignored if --tip-offset-mm is set.",
    )
    parser.add_argument(
        "--tip-offset-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Manual end-effector-to-physical-tip offset in gripper coordinates.",
    )
    parser.add_argument(
        "--perpendicular-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 20.0, 0.0),
        help="Absolute RPY used for perpendicular approach.",
    )
    parser.add_argument(
        "--oblique-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 60.0, 0.0),
        help="Absolute RPY used for the 60 degree oblique pose.",
    )
    parser.add_argument("--first-step-mm", type=float, default=0.25)
    parser.add_argument("--second-step-mm", type=float, default=0.25)
    parser.add_argument("--final-step-mm", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--position-gain", type=float, default=1.0)
    parser.add_argument("--orientation-gain", type=float, default=0.8)
    parser.add_argument("--max-linear-vel", type=float, default=0.5)
    parser.add_argument("--max-angular-vel", type=float, default=0.05)
    parser.add_argument("--position-tol-mm", type=float, default=0.05)
    parser.add_argument("--orientation-tol-deg", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--stop-on-handle-drift-mm",
        type=float,
        default=40.0,
        help="Stop if the end-effector origin drifts this far during one stage. Use 0 to disable.",
    )
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip per-stage confirmation prompts.",
    )
    return parser.parse_args()


def load_tip_offset(args):
    if args.tip_offset_mm is not None:
        return np.asarray(args.tip_offset_mm, dtype=float), "manual --tip-offset-mm"

    path = Path(args.pivot_calibration_json).expanduser()
    with path.open() as f:
        data = json.load(f)
    if "t_gripper_tip_mm" not in data:
        raise KeyError(f"{path} does not contain t_gripper_tip_mm")
    return np.asarray(data["t_gripper_tip_mm"], dtype=float), str(path)


def clip_norm(vector, max_norm):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < EPS:
        return vector
    if max_norm <= 0.0:
        return np.zeros_like(vector)
    return vector * (float(max_norm) / norm)


def fit_angular_to_linear_limit(tip_velocity, angular_velocity, tip_world, max_linear):
    """Scale angular velocity so compensated gripper velocity fits max_linear."""
    tip_velocity = np.asarray(tip_velocity, dtype=float)
    angular_velocity = np.asarray(angular_velocity, dtype=float)
    tip_world = np.asarray(tip_world, dtype=float)

    def gripper_velocity(scale):
        return tip_velocity - np.cross(scale * angular_velocity, tip_world)

    if np.linalg.norm(gripper_velocity(1.0)) <= max_linear:
        return angular_velocity
    if np.linalg.norm(tip_velocity) >= max_linear:
        return np.zeros_like(angular_velocity)

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(gripper_velocity(mid)) <= max_linear:
            lo = mid
        else:
            hi = mid
    return lo * angular_velocity


def current_pose(robot):
    return np.asarray(robot.get_current_pose(), dtype=float)


def current_quat(robot):
    return np.array([robot.qx, robot.qy, robot.qz, robot.qw], dtype=float)


def current_rotation(robot):
    return R.from_quat(current_quat(robot))


def physical_tip_position(gripper_pos, rotation, tip_offset_gripper):
    return gripper_pos + rotation.apply(tip_offset_gripper)


def needle_axis(rotation):
    # Force-collection scripts define insertion as -tool local z.
    return -rotation.as_matrix()[:, 2]


def stop_robot(robot):
    robot.pub_linear.publish(0.0, 0.0, 0.0)
    robot.pub_angular.publish(0.0, 0.0, 0.0)


def prompt(args, message):
    if args.yes:
        return
    response = input(
        "\n{}\nKeep hand on e-stop. Press Enter to run, or type q then Enter to stop: ".format(
            message
        )
    ).strip()
    if response.lower() == "q":
        raise KeyboardInterrupt


def move_tip_pose(
    robot,
    label,
    target_tip_pos,
    target_rotation,
    tip_offset_gripper,
    args,
    sample_rows,
):
    start_pose = current_pose(robot)
    start_gripper = start_pose[:3].copy()
    start_tip = physical_tip_position(start_gripper, current_rotation(robot), tip_offset_gripper)
    start_time = time.time()
    deadline = start_time + args.timeout_s
    rate = rospy.Rate(args.rate_hz)
    settled_since = None
    status = "timeout"
    final_tip_error = float("nan")
    final_angle_error = float("nan")
    final_gripper_drift = float("nan")
    final_pose = start_pose.copy()

    print("\nStage: {}".format(label))
    print("  start tip:  {}".format(np.round(start_tip, 4)))
    print("  target tip: {}".format(np.round(target_tip_pos, 4)))
    print(
        "  target rpy: {}".format(
            np.round(target_rotation.as_euler("xyz", degrees=True), 4)
        )
    )

    try:
        while not rospy.is_shutdown():
            now = time.time()
            final_pose = current_pose(robot)
            gripper_pos = final_pose[:3]
            rotation = current_rotation(robot)
            tip_world = rotation.apply(tip_offset_gripper)
            tip_pos = gripper_pos + tip_world

            tip_error_vec = target_tip_pos - tip_pos
            tip_error = float(np.linalg.norm(tip_error_vec))

            rot_error = target_rotation * rotation.inv()
            rotvec = rot_error.as_rotvec()
            angle_error = float(np.linalg.norm(rotvec) * 180.0 / np.pi)
            gripper_drift = float(np.linalg.norm(gripper_pos - start_gripper))

            final_tip_error = tip_error
            final_angle_error = angle_error
            final_gripper_drift = gripper_drift
            sample_rows.append({
                "unix_time": round(now, 6),
                "elapsed_sec": round(now - start_time, 6),
                "stage": label,
                "tip_error_mm": round(tip_error, 6),
                "orientation_error_deg": round(angle_error, 6),
                "gripper_drift_mm": round(gripper_drift, 6),
                "tip_x_mm": round(float(tip_pos[0]), 6),
                "tip_y_mm": round(float(tip_pos[1]), 6),
                "tip_z_mm": round(float(tip_pos[2]), 6),
                "gripper_x_mm": round(float(gripper_pos[0]), 6),
                "gripper_y_mm": round(float(gripper_pos[1]), 6),
                "gripper_z_mm": round(float(gripper_pos[2]), 6),
            })

            if tip_error <= args.position_tol_mm and angle_error <= args.orientation_tol_deg:
                if settled_since is None:
                    settled_since = now
                elif now - settled_since >= args.settle_s:
                    status = "reached"
                    break
            else:
                settled_since = None

            if now >= deadline:
                status = "timeout"
                break
            if args.stop_on_handle_drift_mm > 0.0 and gripper_drift > args.stop_on_handle_drift_mm:
                status = "handle_drift"
                break

            desired_tip_vel = clip_norm(
                tip_error_vec * args.position_gain,
                args.max_linear_vel,
            )
            angular_vel = clip_norm(
                rotvec * args.orientation_gain,
                args.max_angular_vel,
            )
            angular_vel = fit_angular_to_linear_limit(
                desired_tip_vel,
                angular_vel,
                tip_world,
                args.max_linear_vel,
            )

            # Keep the physical tool tip, not the handle, on the planned path.
            gripper_linear_vel = desired_tip_vel - np.cross(angular_vel, tip_world)

            robot.pub_linear.publish(
                float(gripper_linear_vel[0]),
                float(gripper_linear_vel[1]),
                float(gripper_linear_vel[2]),
            )
            robot.pub_angular.publish(
                float(angular_vel[0]),
                float(angular_vel[1]),
                float(angular_vel[2]),
            )
            rate.sleep()
    finally:
        stop_robot(robot)

    ok = status == "reached"
    print(
        "  result: {}  tip_error={:.4f} mm  orientation_error={:.3f} deg  handle_drift={:.3f} mm".format(
            status,
            final_tip_error,
            final_angle_error,
            final_gripper_drift,
        )
    )

    return {
        "stage": label,
        "status": status,
        "reached": ok,
        "elapsed_sec": round(time.time() - start_time, 6),
        "target_tip_x_mm": round(float(target_tip_pos[0]), 6),
        "target_tip_y_mm": round(float(target_tip_pos[1]), 6),
        "target_tip_z_mm": round(float(target_tip_pos[2]), 6),
        "final_gripper_x_mm": round(float(final_pose[0]), 6),
        "final_gripper_y_mm": round(float(final_pose[1]), 6),
        "final_gripper_z_mm": round(float(final_pose[2]), 6),
        "final_tip_error_mm": round(float(final_tip_error), 6),
        "final_orientation_error_deg": round(float(final_angle_error), 6),
        "final_handle_drift_mm": round(float(final_gripper_drift), 6),
    }


def write_csv(path, rows):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_stage_or_abort(rows, row):
    rows.append(row)
    if not row["reached"]:
        raise RuntimeError("Aborting sequence because stage failed: {}".format(row["stage"]))


def validate_args(args):
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive")
    if args.max_linear_vel <= 0.0:
        raise ValueError("--max-linear-vel must be positive")
    if args.max_angular_vel <= 0.0:
        raise ValueError("--max-angular-vel must be positive")
    if args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be positive")
    if args.position_tol_mm <= 0.0:
        raise ValueError("--position-tol-mm must be positive")
    if args.orientation_tol_deg <= 0.0:
        raise ValueError("--orientation-tol-deg must be positive")


def main():
    args = parse_args()
    validate_args(args)
    tip_offset, tip_offset_source = load_tip_offset(args)

    if np.linalg.norm(tip_offset) < EPS:
        raise ValueError("Tip offset is zero; this would rotate about the handle, not the tool tip.")
    if args.max_linear_vel > 0.5 or args.max_angular_vel > 0.1:
        print("WARNING: velocity limits are above the conservative first-test range.")

    perpendicular_rotation = R.from_euler("xyz", args.perpendicular_rpy_deg, degrees=True)
    oblique_rotation = R.from_euler("xyz", args.oblique_rpy_deg, degrees=True)

    robot = SHERController(robot_name=args.robot_name)
    initial_pose = current_pose(robot)
    initial_rotation = current_rotation(robot)
    initial_tip = physical_tip_position(initial_pose[:3], initial_rotation, tip_offset)

    print("\nTip-centered RCM needle sequence")
    print("  robot: {}".format(args.robot_name))
    print("  initial gripper pose: {}".format(np.round(initial_pose, 4)))
    print("  initial physical tip: {}".format(np.round(initial_tip, 4)))
    print("  tip offset source: {}".format(tip_offset_source))
    print("  tip offset gripper mm: {}".format(np.round(tip_offset, 4)))
    print("  perpendicular RPY deg: {}".format(np.round(args.perpendicular_rpy_deg, 4)))
    print("  oblique RPY deg: {}".format(np.round(args.oblique_rpy_deg, 4)))
    print("  steps mm: {}, {}, {}".format(
        args.first_step_mm,
        args.second_step_mm,
        args.final_step_mm,
    ))
    print("  max linear vel: {:.4f} mm/s".format(args.max_linear_vel))
    print("  max angular vel: {:.4f} rad/s".format(args.max_angular_vel))

    if not args.yes:
        response = input(
            "\nThis script moves the handle to keep the physical tool tip as the rotation center.\n"
            "Confirm the tip offset is correct and the workspace is clear, then type YES: "
        ).strip()
        if response != "YES":
            raise RuntimeError("Operator did not confirm the sequence")

    summary_rows = []
    sample_rows = []

    try:
        current_tip = initial_tip.copy()

        prompt(args, "Stage 1: rotate to perpendicular pose while holding the tip fixed.")
        append_stage_or_abort(summary_rows, move_tip_pose(
            robot,
            "rotate_to_perpendicular_20deg",
            current_tip,
            perpendicular_rotation,
            tip_offset,
            args,
            sample_rows,
        ))

        current_tip = physical_tip_position(
            current_pose(robot)[:3],
            current_rotation(robot),
            tip_offset,
        )
        direction = needle_axis(perpendicular_rotation)
        target_tip = current_tip + args.first_step_mm * direction
        prompt(args, "Stage 2: move 0.25 mm along perpendicular needle direction.")
        append_stage_or_abort(summary_rows, move_tip_pose(
            robot,
            "insert_0p25mm_perpendicular",
            target_tip,
            perpendicular_rotation,
            tip_offset,
            args,
            sample_rows,
        ))

        current_tip = physical_tip_position(
            current_pose(robot)[:3],
            current_rotation(robot),
            tip_offset,
        )
        prompt(args, "Stage 3: rotate to 60 degree pose about the current tool tip.")
        append_stage_or_abort(summary_rows, move_tip_pose(
            robot,
            "rotate_to_60deg_about_tip",
            current_tip,
            oblique_rotation,
            tip_offset,
            args,
            sample_rows,
        ))

        current_tip = physical_tip_position(
            current_pose(robot)[:3],
            current_rotation(robot),
            tip_offset,
        )
        direction = needle_axis(oblique_rotation)
        target_tip = current_tip + args.second_step_mm * direction
        prompt(args, "Stage 4: move 0.25 mm along 60 degree needle direction.")
        append_stage_or_abort(summary_rows, move_tip_pose(
            robot,
            "insert_0p25mm_60deg",
            target_tip,
            oblique_rotation,
            tip_offset,
            args,
            sample_rows,
        ))

        current_tip = physical_tip_position(
            current_pose(robot)[:3],
            current_rotation(robot),
            tip_offset,
        )
        prompt(args, "Stage 5: rotate back to perpendicular about the current tool tip.")
        append_stage_or_abort(summary_rows, move_tip_pose(
            robot,
            "rotate_back_to_perpendicular",
            current_tip,
            perpendicular_rotation,
            tip_offset,
            args,
            sample_rows,
        ))

        current_tip = physical_tip_position(
            current_pose(robot)[:3],
            current_rotation(robot),
            tip_offset,
        )
        direction = needle_axis(perpendicular_rotation)
        target_tip = current_tip + args.final_step_mm * direction
        prompt(args, "Stage 6: move 10 mm along perpendicular needle direction.")
        append_stage_or_abort(summary_rows, move_tip_pose(
            robot,
            "insert_10mm_perpendicular",
            target_tip,
            perpendicular_rotation,
            tip_offset,
            args,
            sample_rows,
        ))
    except KeyboardInterrupt:
        print("\nStopped by operator.")
    except RuntimeError as exc:
        print("\n{}".format(exc))
    finally:
        stop_robot(robot)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(args.log_dir).expanduser()
        summary_path = write_csv(log_dir / f"needle_tip_rcm_sequence_summary_{stamp}.csv", summary_rows)
        sample_path = write_csv(log_dir / f"needle_tip_rcm_sequence_samples_{stamp}.csv", sample_rows)
        if summary_path:
            print(f"Summary log -> {summary_path}")
        if sample_path:
            print(f"Samples log -> {sample_path}")


if __name__ == "__main__":
    main()
