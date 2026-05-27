#!/usr/bin/env python3
"""
Rotation-only sweep for SHER.

This script keeps the current XYZ position as diagnostic state only and publishes
zero linear velocity. It increments roll/pitch targets from the start orientation
until a commanded rotation cannot be reached, then writes CSV logs.

Example:
    python3 test_rotation_limits.py --robot-name SHER20 --max-offset-deg 35
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from SHER_Controller import SHERController


LOG_DIR = Path(__file__).resolve().parent / "rotation_test_logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive SHER rotation limit sweep.")
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument("--step-deg", type=float, default=5.0)
    parser.add_argument("--max-offset-deg", type=float, default=35.0)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--orientation-tol-deg", type=float, default=0.75)
    parser.add_argument("--max-angular-vel", type=float, default=0.05)
    parser.add_argument("--angular-gain", type=float, default=0.8)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--stop-on-drift-mm",
        type=float,
        default=5.0,
        help="Stop a rotation if FrameEE translation drifts this far from sweep start. Use 0 to disable.",
    )
    parser.add_argument(
        "--include-diagonals",
        action="store_true",
        help="Also test combined roll/pitch sweeps.",
    )
    parser.add_argument(
        "--no-return-home",
        action="store_true",
        help="Do not rotate back to the start orientation between sweeps.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run without pausing before each target.",
    )
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    return parser.parse_args()


def normalize_pose(pose):
    pose = np.asarray(pose, dtype=float).copy()
    pose[5] = 0.0
    return pose


def current_pose(robot):
    return np.asarray(robot.get_current_pose(), dtype=float)


def current_quat(robot):
    return np.array([robot.qx, robot.qy, robot.qz, robot.qw], dtype=float)


def clip_norm(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm < 1e-12:
        return vec
    if max_norm <= 0.0:
        return np.zeros_like(vec)
    return vec * (max_norm / norm)


def prompt(args, message):
    if args.yes:
        return
    answer = input(f"\n{message}\nPress Enter to run, or type q then Enter to stop: ").strip()
    if answer.lower() == "q":
        raise KeyboardInterrupt


def stop_robot(robot):
    robot.pub_linear.publish(0.0, 0.0, 0.0)
    robot.pub_angular.publish(0.0, 0.0, 0.0)


def rotation_only_move(robot, target_rpy, label, args, sample_rows):
    target_rpy = np.asarray(target_rpy, dtype=float)
    target_rot = R.from_euler("xyz", target_rpy, degrees=True)
    start_pose = current_pose(robot)
    start_pos = start_pose[:3].copy()
    start_time = time.time()
    rate = rospy.Rate(args.rate_hz)
    status = "timeout"
    final_angle_error = None
    final_drift = None
    final_pose = None

    print(f"\nTarget {label}: rpy={[round(v, 3) for v in target_rpy]}")
    print("Commanding angular velocity only; linear velocity is held at zero.")

    try:
        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            final_pose = current_pose(robot)
            current_rot = R.from_quat(current_quat(robot))
            rot_error = target_rot * current_rot.inv()
            rotvec = rot_error.as_rotvec()
            angle_error = np.linalg.norm(rotvec) * 180.0 / np.pi
            drift = np.linalg.norm(final_pose[:3] - start_pos)

            final_angle_error = angle_error
            final_drift = drift
            sample_rows.append({
                "unix_time": round(time.time(), 6),
                "elapsed_sec": round(elapsed, 6),
                "label": label,
                "target_roll_deg": round(float(target_rpy[0]), 6),
                "target_pitch_deg": round(float(target_rpy[1]), 6),
                "target_yaw_deg": round(float(target_rpy[2]), 6),
                "current_x_mm": round(float(final_pose[0]), 6),
                "current_y_mm": round(float(final_pose[1]), 6),
                "current_z_mm": round(float(final_pose[2]), 6),
                "current_roll_deg": round(float(final_pose[3]), 6),
                "current_pitch_deg": round(float(final_pose[4]), 6),
                "current_yaw_deg": round(float(final_pose[5]), 6),
                "orientation_error_deg": round(float(angle_error), 6),
                "position_drift_mm": round(float(drift), 6),
                "rotvec_error_x_deg": round(float(rotvec[0] * 180.0 / np.pi), 6),
                "rotvec_error_y_deg": round(float(rotvec[1] * 180.0 / np.pi), 6),
                "rotvec_error_z_deg": round(float(rotvec[2] * 180.0 / np.pi), 6),
            })

            if angle_error <= args.orientation_tol_deg:
                status = "reached"
                break
            if args.stop_on_drift_mm > 0.0 and drift > args.stop_on_drift_mm:
                status = "position_drift"
                break
            if elapsed >= args.timeout_s:
                status = "timeout"
                break

            angular_vel = clip_norm(rotvec * args.angular_gain, args.max_angular_vel)
            robot.pub_linear.publish(0.0, 0.0, 0.0)
            robot.pub_angular.publish(
                float(angular_vel[0]), float(angular_vel[1]), float(angular_vel[2])
            )
            rate.sleep()
    finally:
        stop_robot(robot)

    if final_pose is None:
        final_pose = current_pose(robot)
    if final_angle_error is None:
        final_angle_error = float("nan")
    if final_drift is None:
        final_drift = float("nan")

    ok = status == "reached"
    print(
        f"Result {label}: {status}, "
        f"orientation_error={final_angle_error:.3f} deg, "
        f"position_drift={final_drift:.3f} mm"
    )

    return {
        "label": label,
        "status": status,
        "reached": ok,
        "elapsed_sec": round(time.time() - start_time, 6),
        "target_roll_deg": round(float(target_rpy[0]), 6),
        "target_pitch_deg": round(float(target_rpy[1]), 6),
        "target_yaw_deg": round(float(target_rpy[2]), 6),
        "final_x_mm": round(float(final_pose[0]), 6),
        "final_y_mm": round(float(final_pose[1]), 6),
        "final_z_mm": round(float(final_pose[2]), 6),
        "final_roll_deg": round(float(final_pose[3]), 6),
        "final_pitch_deg": round(float(final_pose[4]), 6),
        "final_yaw_deg": round(float(final_pose[5]), 6),
        "final_orientation_error_deg": round(float(final_angle_error), 6),
        "final_position_drift_mm": round(float(final_drift), 6),
    }


def sweep_definitions(include_diagonals):
    sweeps = [
        ("roll_pos", 1.0, 0.0),
        ("roll_neg", -1.0, 0.0),
        ("pitch_pos", 0.0, 1.0),
        ("pitch_neg", 0.0, -1.0),
    ]
    if include_diagonals:
        sweeps.extend([
            ("diag_roll_pos_pitch_pos", 1.0, 1.0),
            ("diag_roll_pos_pitch_neg", 1.0, -1.0),
            ("diag_roll_neg_pitch_pos", -1.0, 1.0),
            ("diag_roll_neg_pitch_neg", -1.0, -1.0),
        ])
    return sweeps


def write_csv(path, rows):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    if args.step_deg <= 0.0:
        raise ValueError("--step-deg must be positive")
    if args.max_offset_deg <= 0.0:
        raise ValueError("--max-offset-deg must be positive")
    if args.max_angular_vel > 0.1:
        print("WARNING: max angular velocity is above the conservative test range.")

    robot = SHERController(robot_name=args.robot_name)
    home_pose = normalize_pose(current_pose(robot))
    home_rpy = home_pose[3:].copy()

    print("\nRotation-only limit sweep")
    print(f"  robot: {args.robot_name}")
    print(f"  home pose: {[round(v, 4) for v in home_pose]}")
    print(f"  step: {args.step_deg:.3f} deg")
    print(f"  max offset: {args.max_offset_deg:.3f} deg")
    print(f"  max angular velocity: {args.max_angular_vel:.4f} rad/s")
    print(f"  timeout per target: {args.timeout_s:.1f} s")
    print("Keep hand on the e-stop. This publishes no linear motion.")

    summary_rows = []
    sample_rows = []
    offsets = np.arange(args.step_deg, args.max_offset_deg + 1e-9, args.step_deg)

    try:
        for sweep_index, (sweep_name, roll_sign, pitch_sign) in enumerate(
            sweep_definitions(args.include_diagonals),
            start=1,
        ):
            if sweep_index > 1 and not args.no_return_home:
                prompt(args, f"Return to start orientation before sweep {sweep_name}?")
                summary_rows.append(rotation_only_move(
                    robot, home_rpy, f"{sweep_name}_return_home", args, sample_rows
                ))

            print(f"\n=== Sweep {sweep_name} ===")
            for offset in offsets:
                target_rpy = home_rpy.copy()
                target_rpy[0] += roll_sign * offset
                target_rpy[1] += pitch_sign * offset
                target_rpy[2] = 0.0
                label = f"{sweep_name}_{offset:.1f}deg".replace(".", "p")
                prompt(args, f"Run {label}?")
                row = rotation_only_move(robot, target_rpy, label, args, sample_rows)
                summary_rows.append(row)
                if row["status"] != "reached":
                    print(f"Stopping sweep {sweep_name} at {offset:.1f} deg: {row['status']}")
                    break
    except KeyboardInterrupt:
        print("\nStopped by operator.")
    finally:
        stop_robot(robot)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(args.log_dir).expanduser()
        summary_path = write_csv(log_dir / f"rotation_limit_summary_{stamp}.csv", summary_rows)
        samples_path = write_csv(log_dir / f"rotation_limit_samples_{stamp}.csv", sample_rows)
        if summary_path is not None:
            print(f"Summary log -> {summary_path}")
        if samples_path is not None:
            print(f"Samples log -> {samples_path}")


if __name__ == "__main__":
    main()
