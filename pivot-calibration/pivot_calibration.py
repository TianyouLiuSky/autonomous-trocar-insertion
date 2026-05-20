#!/usr/bin/env python3
"""
Pivot calibration for the ATI trocar tool.

This estimates the fixed translational offset from the robot end-effector
frame to the physical trocar tip. During data collection, keep the trocar tip
seated in a fixed dimple and record many robot end-effector poses while the
tool orientation changes.

For each sample:

    p_tip_base = R_base_gripper * t_gripper_tip + p_gripper_base

Since the physical tip is fixed in the dimple, all p_tip_base values should be
the same unknown pivot point c_base. The linear least-squares system is:

    [R_i  -I] [t_gripper_tip] = -p_i
              [c_base       ]

Outputs are in millimeters, matching the SHER FrameEE convention used by the
rest of this project.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import threading
import time

import numpy as np

try:
    import rospy
    from geometry_msgs.msg import Transform
except ImportError:
    rospy = None
    Transform = None


ROBOT_NAME = "SHER20"
ROBOT_TOPIC = "/SHER20/eye_robot/FrameEE"
DEFAULT_MIN_SAMPLES = 12


def _timestamp():
    return datetime.datetime.now().strftime("%d%b%Y_%H%M%S").upper()


def _default_output_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "output")


def quat_to_rotmat(q):
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion norm is too small")
    x, y, z, w = q / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def rotmat_to_euler_xyz_deg(R):
    """Return XYZ fixed-angle Euler values in degrees for display/logging."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def rotation_angle_deg(R_a, R_b):
    R_delta = np.asarray(R_a).T.dot(np.asarray(R_b))
    c = (np.trace(R_delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def orientation_spread_deg(rotations):
    if len(rotations) < 2:
        return 0.0, 0.0
    angles = []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            angles.append(rotation_angle_deg(rotations[i], rotations[j]))
    return float(np.min(angles)), float(np.max(angles))


def solve_pivot_calibration(positions_mm, quaternions):
    """
    Solve for t_gripper_tip and pivot_base.

    Args:
        positions_mm: Nx3 end-effector positions in robot base frame, mm.
        quaternions: Nx4 end-effector quaternions [x, y, z, w].

    Returns:
        dict containing solution arrays and residual statistics.
    """
    positions = np.asarray(positions_mm, dtype=float)
    quats = np.asarray(quaternions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_mm must have shape Nx3")
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quaternions must have shape Nx4")
    if len(positions) != len(quats):
        raise ValueError("positions and quaternions must have the same length")
    if len(positions) < 3:
        raise ValueError("Need at least 3 samples; 12 or more is recommended")

    rotations = np.array([quat_to_rotmat(q) for q in quats])

    A_rows = []
    b_rows = []
    neg_eye = -np.eye(3)
    for R_i, p_i in zip(rotations, positions):
        A_rows.append(np.hstack([R_i, neg_eye]))
        b_rows.append(-p_i.reshape(3, 1))

    A = np.vstack(A_rows)
    b = np.vstack(b_rows).reshape(-1)
    x, residual_sum, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    t_gripper_tip = x[:3]
    pivot_base = x[3:6]
    tip_positions = np.einsum("nij,j->ni", rotations, t_gripper_tip) + positions
    residual_vectors = tip_positions - pivot_base.reshape(1, 3)
    residual_norms = np.linalg.norm(residual_vectors, axis=1)

    min_angle, max_angle = orientation_spread_deg(rotations)
    condition_number = float(np.linalg.cond(A))

    return {
        "t_gripper_tip_mm": t_gripper_tip,
        "pivot_base_mm": pivot_base,
        "tip_positions_base_mm": tip_positions,
        "residual_vectors_mm": residual_vectors,
        "residual_norms_mm": residual_norms,
        "rms_residual_mm": float(np.sqrt(np.mean(residual_norms ** 2))),
        "mean_residual_mm": float(np.mean(residual_norms)),
        "max_residual_mm": float(np.max(residual_norms)),
        "rank": int(rank),
        "singular_values": singular_values,
        "condition_number": condition_number,
        "min_pairwise_rotation_deg": min_angle,
        "max_pairwise_rotation_deg": max_angle,
        "rotations": rotations,
    }


class RobotPoseListener(object):
    """Thread-safe listener for the SHER FrameEE Transform topic."""

    def __init__(self, topic, translation_scale_to_mm=1.0):
        if rospy is None or Transform is None:
            raise RuntimeError("rospy is not available; use --from-csv for offline solving")
        self.topic = topic
        self.translation_scale_to_mm = float(translation_scale_to_mm)
        self._lock = threading.Lock()
        self._pose = None
        rospy.Subscriber(topic, Transform, self._callback, queue_size=10)

    def _callback(self, msg):
        t_mm = np.array([
            msg.translation.x,
            msg.translation.y,
            msg.translation.z,
        ], dtype=float) * self.translation_scale_to_mm
        q = np.array([
            msg.rotation.x,
            msg.rotation.y,
            msg.rotation.z,
            msg.rotation.w,
        ], dtype=float)
        stamp = time.time()
        with self._lock:
            self._pose = {"t_mm": t_mm, "q": q, "stamp": stamp}

    @property
    def ready(self):
        with self._lock:
            return self._pose is not None

    def get_pose(self):
        with self._lock:
            if self._pose is None:
                return None
            return {
                "t_mm": self._pose["t_mm"].copy(),
                "q": self._pose["q"].copy(),
                "stamp": self._pose["stamp"],
            }


def _sample_to_row(index, pose):
    R = quat_to_rotmat(pose["q"])
    rpy = rotmat_to_euler_xyz_deg(R)
    return {
        "sample": index,
        "stamp_sec": "{:.6f}".format(pose.get("stamp", time.time())),
        "tx_mm": "{:.6f}".format(pose["t_mm"][0]),
        "ty_mm": "{:.6f}".format(pose["t_mm"][1]),
        "tz_mm": "{:.6f}".format(pose["t_mm"][2]),
        "qx": "{:.9f}".format(pose["q"][0]),
        "qy": "{:.9f}".format(pose["q"][1]),
        "qz": "{:.9f}".format(pose["q"][2]),
        "qw": "{:.9f}".format(pose["q"][3]),
        "roll_deg": "{:.6f}".format(rpy[0]),
        "pitch_deg": "{:.6f}".format(rpy[1]),
        "yaw_deg": "{:.6f}".format(rpy[2]),
    }


def _rows_from_samples(samples):
    return [_sample_to_row(i + 1, pose) for i, pose in enumerate(samples)]


def save_sample_csv(samples, path):
    rows = _rows_from_samples(samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "sample", "stamp_sec", "tx_mm", "ty_mm", "tz_mm",
        "qx", "qy", "qz", "qw", "roll_deg", "pitch_deg", "yaw_deg",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float_from_row(row, names):
    for name in names:
        if name in row and row[name] not in ("", None):
            return float(row[name])
    raise KeyError("Missing one of columns: {}".format(", ".join(names)))


def load_sample_csv(path):
    samples = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = np.array([
                    _float_from_row(row, ["tx_mm", "rob_tx_mm", "x_mm"]),
                    _float_from_row(row, ["ty_mm", "rob_ty_mm", "y_mm"]),
                    _float_from_row(row, ["tz_mm", "rob_tz_mm", "z_mm"]),
                ], dtype=float)
                q = np.array([
                    _float_from_row(row, ["qx", "rob_qx"]),
                    _float_from_row(row, ["qy", "rob_qy"]),
                    _float_from_row(row, ["qz", "rob_qz"]),
                    _float_from_row(row, ["qw", "rob_qw"]),
                ], dtype=float)
            except KeyError as exc:
                raise ValueError("CSV does not contain required pose columns: {}".format(exc))
            samples.append({"t_mm": t, "q": q, "stamp": float(row.get("stamp_sec", 0.0) or 0.0)})
    if not samples:
        raise ValueError("No samples loaded from {}".format(path))
    return samples


def save_solution(samples, result, output_dir, prefix="pivot_calibration"):
    ts = _timestamp()
    base = "{}_{}".format(prefix, ts)
    run_dir = os.path.join(output_dir, base)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, base + "_samples.csv")
    json_path = os.path.join(run_dir, base + ".json")
    npz_path = os.path.join(run_dir, base + ".npz")

    positions = np.array([s["t_mm"] for s in samples], dtype=float)
    quaternions = np.array([s["q"] for s in samples], dtype=float)
    T_gripper_tip = np.eye(4)
    T_gripper_tip[:3, 3] = result["t_gripper_tip_mm"]

    save_sample_csv(samples, csv_path)

    np.savez(
        npz_path,
        t_gripper_tip_mm=result["t_gripper_tip_mm"],
        pivot_base_mm=result["pivot_base_mm"],
        T_gripper_tip_mm=T_gripper_tip,
        positions_gripper_base_mm=positions,
        quaternions_xyzw=quaternions,
        tip_positions_base_mm=result["tip_positions_base_mm"],
        residual_vectors_mm=result["residual_vectors_mm"],
        residual_norms_mm=result["residual_norms_mm"],
        rms_residual_mm=result["rms_residual_mm"],
        mean_residual_mm=result["mean_residual_mm"],
        max_residual_mm=result["max_residual_mm"],
        condition_number=result["condition_number"],
        rank=result["rank"],
        singular_values=result["singular_values"],
        min_pairwise_rotation_deg=result["min_pairwise_rotation_deg"],
        max_pairwise_rotation_deg=result["max_pairwise_rotation_deg"],
    )

    payload = {
        "created": ts,
        "units": "millimeters",
        "method": "robot_fk_pivot_calibration",
        "sample_count": len(samples),
        "t_gripper_tip_mm": result["t_gripper_tip_mm"].tolist(),
        "pivot_base_mm": result["pivot_base_mm"].tolist(),
        "rms_residual_mm": result["rms_residual_mm"],
        "mean_residual_mm": result["mean_residual_mm"],
        "max_residual_mm": result["max_residual_mm"],
        "rank": result["rank"],
        "condition_number": result["condition_number"],
        "min_pairwise_rotation_deg": result["min_pairwise_rotation_deg"],
        "max_pairwise_rotation_deg": result["max_pairwise_rotation_deg"],
        "run_dir": run_dir,
        "csv_samples": csv_path,
        "npz": npz_path,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return {"run_dir": run_dir, "csv": csv_path, "json": json_path, "npz": npz_path}


def print_result(result, sample_count, residual_warn_mm):
    tcp = result["t_gripper_tip_mm"]
    pivot = result["pivot_base_mm"]
    print("\n" + "=" * 72)
    print("PIVOT CALIBRATION RESULT")
    print("=" * 72)
    print("Samples: {}".format(sample_count))
    print("t_gripper_tip_mm: [{:9.4f}, {:9.4f}, {:9.4f}]".format(tcp[0], tcp[1], tcp[2]))
    print("pivot_base_mm:    [{:9.4f}, {:9.4f}, {:9.4f}]".format(pivot[0], pivot[1], pivot[2]))
    print("Residual RMS:     {:9.4f} mm".format(result["rms_residual_mm"]))
    print("Residual mean:    {:9.4f} mm".format(result["mean_residual_mm"]))
    print("Residual max:     {:9.4f} mm".format(result["max_residual_mm"]))
    print("Rotation spread:  min pair {:7.3f} deg | max pair {:7.3f} deg".format(
        result["min_pairwise_rotation_deg"],
        result["max_pairwise_rotation_deg"],
    ))
    print("Linear rank:      {} | condition number: {:.3e}".format(
        result["rank"],
        result["condition_number"],
    ))
    if result["max_residual_mm"] > residual_warn_mm:
        print("WARNING: max residual is above {:.3f} mm. Check dimple slip, ".format(residual_warn_mm) +
              "tool looseness, and orientation diversity.")
    print("=" * 72 + "\n")


def current_pose_line(pose):
    R = quat_to_rotmat(pose["q"])
    rpy = rotmat_to_euler_xyz_deg(R)
    t = pose["t_mm"]
    return "pos mm [{:8.3f}, {:8.3f}, {:8.3f}]  rpy deg [{:7.2f}, {:7.2f}, {:7.2f}]".format(
        t[0], t[1], t[2], rpy[0], rpy[1], rpy[2]
    )


def maybe_solve(samples, min_samples, residual_warn_mm):
    if len(samples) < 3:
        print("Need at least 3 samples to solve; collect more.")
        return None
    if len(samples) < min_samples:
        print("Only {} samples collected; {} or more is recommended.".format(len(samples), min_samples))
    positions = np.array([s["t_mm"] for s in samples], dtype=float)
    quats = np.array([s["q"] for s in samples], dtype=float)
    result = solve_pivot_calibration(positions, quats)
    print_result(result, len(samples), residual_warn_mm)
    return result


def live_collect(args):
    if rospy is None:
        print("ERROR: rospy is not installed in this environment. Use --from-csv offline.", file=sys.stderr)
        return 2

    rospy.init_node("ati_pivot_calibration", anonymous=True)
    listener = RobotPoseListener(ROBOT_TOPIC, translation_scale_to_mm=args.translation_scale_to_mm)

    print("\nATI pivot calibration")
    print("Robot: {} (SHER 2.0)".format(ROBOT_NAME))
    print("Robot topic: {}".format(ROBOT_TOPIC))
    print("Translation scale to mm: {}".format(args.translation_scale_to_mm))
    print("\nSetup:")
    print("  1. Attach the trocar/tool exactly as it will be used.")
    print("  2. Seat the physical tip in a fixed dimple.")
    print("  3. Change tool orientation while keeping the tip seated.")
    print("  4. Press Enter to record each settled pose.")
    print("\nCommands: Enter=record, s=solve, w=save, d=delete last, p=print pose, q=quit")
    print("Waiting for first pose...")

    while not rospy.is_shutdown() and not listener.ready:
        rospy.sleep(0.05)

    if rospy.is_shutdown():
        return 1

    print("FrameEE received: {}".format(current_pose_line(listener.get_pose())))
    samples = []
    last_result = None

    while not rospy.is_shutdown():
        cmd = input("[{} samples] command> ".format(len(samples))).strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd in ("p", "pose"):
            pose = listener.get_pose()
            print(current_pose_line(pose))
            continue
        if cmd in ("d", "delete", "back"):
            if samples:
                samples.pop()
                last_result = None
                print("Deleted last sample. {} samples remain.".format(len(samples)))
            else:
                print("No samples to delete.")
            continue
        if cmd in ("s", "solve"):
            last_result = maybe_solve(samples, args.min_samples, args.residual_warn_mm)
            continue
        if cmd in ("w", "write", "save"):
            if last_result is None:
                last_result = maybe_solve(samples, args.min_samples, args.residual_warn_mm)
            if last_result is not None:
                paths = save_solution(samples, last_result, args.output_dir)
                print("Saved:")
                print("  Run: {}".format(paths["run_dir"]))
                print("  NPZ : {}".format(paths["npz"]))
                print("  JSON: {}".format(paths["json"]))
                print("  CSV : {}".format(paths["csv"]))
            continue
        if cmd not in ("", "r", "record"):
            print("Unknown command: {}".format(cmd))
            continue

        pose = listener.get_pose()
        if pose is None:
            print("No pose available yet.")
            continue
        samples.append(pose)
        last_result = None
        print("Recorded sample {}: {}".format(len(samples), current_pose_line(pose)))
        if len(samples) >= 3:
            maybe_solve(samples, args.min_samples, args.residual_warn_mm)

    if samples and args.save_on_exit:
        result = maybe_solve(samples, args.min_samples, args.residual_warn_mm)
        if result is not None:
            paths = save_solution(samples, result, args.output_dir)
            print("Saved on exit:")
            print("  Run: {}".format(paths["run_dir"]))
            print("  NPZ : {}".format(paths["npz"]))
            print("  JSON: {}".format(paths["json"]))
            print("  CSV : {}".format(paths["csv"]))
    return 0


def offline_solve(args):
    samples = load_sample_csv(args.from_csv)
    result = maybe_solve(samples, args.min_samples, args.residual_warn_mm)
    if result is None:
        return 1
    if args.no_save:
        return 0
    paths = save_solution(samples, result, args.output_dir)
    print("Saved:")
    print("  Run: {}".format(paths["run_dir"]))
    print("  NPZ : {}".format(paths["npz"]))
    print("  JSON: {}".format(paths["json"]))
    print("  CSV : {}".format(paths["csv"]))
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect and solve ATI trocar pivot calibration.")
    parser.add_argument("--translation-scale-to-mm", type=float, default=1.0,
                        help="Scale Transform translations into mm. Use 1000 if the topic is in meters.")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help="Recommended minimum sample count before accepting a result.")
    parser.add_argument("--residual-warn-mm", type=float, default=0.5,
                        help="Warn if max residual exceeds this value.")
    parser.add_argument("--output-dir", default=_default_output_dir(),
                        help="Directory for npz/json/csv outputs.")
    parser.add_argument("--from-csv", default=None,
                        help="Solve offline from a previously saved sample CSV.")
    parser.add_argument("--no-save", action="store_true",
                        help="Offline mode only: print result without writing output files.")
    parser.add_argument("--save-on-exit", action="store_true",
                        help="Live mode: solve and save when quitting if samples exist.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.from_csv:
        return offline_solve(args)
    return live_collect(args)


if __name__ == "__main__":
    sys.exit(main())
