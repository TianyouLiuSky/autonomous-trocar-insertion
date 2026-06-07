#!/usr/bin/env python3
"""
Measure stationary D405/ChArUco pose repeatability.

Keep the robot and board completely still while this script records repeated
board-pose estimates. FrameEE is logged at the same time so camera/PnP jitter
can be distinguished from robot pose drift.
"""

import argparse
import csv
import datetime
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation


CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 15
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN = 0.010
MARKER_LEN = 0.007
DICT_ID = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 8
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            CAMERA_W,
            CAMERA_H,
            rs.format.bgr8,
            CAMERA_FPS,
        )
        profile = self.pipeline.start(config)
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
        )
        self.dist = np.array(intr.coeffs[:5])

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self):
        self.pipeline.stop()


class CharucoDetector:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
        self.board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LEN,
            MARKER_LEN,
            dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, image, K, dist):
        corners, ids, _, _ = self.detector.detectBoard(image)
        count = 0 if ids is None else len(ids)
        if count < MIN_CORNERS:
            return None

        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < MIN_CORNERS:
            return None

        try:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        except cv2.error:
            return None
        if not success:
            return None

        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        observed = np.asarray(img_pts, dtype=float).reshape(-1, 2)
        projected = np.asarray(projected, dtype=float).reshape(-1, 2)
        reprojection_error = np.sqrt(
            np.mean(np.sum((observed - projected) ** 2, axis=1))
        )
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return {
            "rvec": rvec,
            "tvec": tvec,
            "rotation_matrix": rotation_matrix,
            "corners": corners,
            "corner_count": count,
            "reprojection_error_px": float(reprojection_error),
        }


class RobotTracker:
    def __init__(self, topic):
        self._pose = None
        self._lock = threading.Lock()
        rospy.Subscriber(topic, Transform, self._callback, queue_size=10)

    def _callback(self, msg):
        pose = {
            "position_mm": np.array(
                [msg.translation.x, msg.translation.y, msg.translation.z],
                dtype=float,
            ),
            "quaternion": np.array(
                [msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w],
                dtype=float,
            ),
            "received_unix_time": time.time(),
        }
        with self._lock:
            self._pose = pose

    def get_pose(self):
        with self._lock:
            if self._pose is None:
                return None
            return {
                "position_mm": self._pose["position_mm"].copy(),
                "quaternion": self._pose["quaternion"].copy(),
                "received_unix_time": self._pose["received_unix_time"],
            }


def mean_rotation(rotation_matrices):
    quaternions = Rotation.from_matrix(rotation_matrices).as_quat()
    reference = quaternions[0]
    aligned = quaternions.copy()
    for i in range(len(aligned)):
        if np.dot(aligned[i], reference) < 0.0:
            aligned[i] *= -1.0
    accumulator = np.zeros((4, 4), dtype=float)
    for quaternion in aligned:
        accumulator += np.outer(quaternion, quaternion)
    _, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, -1]
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return Rotation.from_quat(quaternion)


def rotation_deviations_deg(rotation_matrices):
    average = mean_rotation(rotation_matrices)
    rotations = Rotation.from_matrix(rotation_matrices)
    deviations = []
    for rotation in rotations:
        relative = rotation * average.inv()
        deviations.append(np.linalg.norm(relative.as_rotvec()) * 180.0 / np.pi)
    return average, np.asarray(deviations)


def vector_summary(samples):
    samples = np.asarray(samples, dtype=float)
    center = np.mean(samples, axis=0)
    radial = np.linalg.norm(samples - center, axis=1)
    return {
        "mean": center.tolist(),
        "std": np.std(samples, axis=0, ddof=1).tolist(),
        "peak_to_peak": np.ptp(samples, axis=0).tolist(),
        "radial_mean": float(np.mean(radial)),
        "radial_p95": float(np.percentile(radial, 95)),
        "radial_max": float(np.max(radial)),
    }


def scalar_summary(samples):
    samples = np.asarray(samples, dtype=float)
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def print_vector_summary(title, summary, units):
    print(title)
    print("  mean XYZ:       {} {}".format(np.round(summary["mean"], 6), units))
    print("  std XYZ:        {} {}".format(np.round(summary["std"], 6), units))
    print(
        "  peak-to-peak:   {} {}".format(
            np.round(summary["peak_to_peak"], 6),
            units,
        )
    )
    print(
        "  radial mean/p95/max: {:.6f} / {:.6f} / {:.6f} {}".format(
            summary["radial_mean"],
            summary["radial_p95"],
            summary["radial_max"],
            units,
        )
    )


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Measure stationary D405/ChArUco pose repeatability."
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--interval-sec", type=float, default=0.20)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--max-wait-sec", type=float, default=120.0)
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument("--label", default="stationary")
    args = parser.parse_args()

    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.interval_sec < 0.0:
        raise ValueError("--interval-sec cannot be negative")

    rospy.init_node("charuco_repeatability_test", anonymous=True)
    robot_topic = "/{}/eye_robot/FrameEE".format(args.robot_name)
    robot = RobotTracker(robot_topic)
    camera = RealSenseCamera()
    detector = CharucoDetector()

    print("D405/ChArUco stationary repeatability test")
    print("  Keep the robot, camera, and board completely still.")
    print("  Robot topic: {}".format(robot_topic))
    print("  Samples: {}".format(args.samples))
    print("  Capture interval: {:.3f} s".format(args.interval_sec))
    print("  Starting after {:.1f} s warmup...".format(args.warmup_sec))

    rows = []
    board_positions_mm = []
    board_rotations = []
    robot_positions_mm = []
    robot_rotations = []
    reprojection_errors = []
    corner_counts = []
    rejected_frames = 0
    last_annotated = None

    try:
        warmup_end = time.time() + args.warmup_sec
        while time.time() < warmup_end:
            camera.get_frame()

        start_time = time.time()
        next_capture_time = start_time
        while len(rows) < args.samples:
            if time.time() - start_time > args.max_wait_sec:
                break

            frame = camera.get_frame()
            if frame is None:
                rejected_frames += 1
                continue

            now = time.time()
            if now < next_capture_time:
                continue

            detection = detector.detect(frame, camera.K, camera.dist)
            if detection is None:
                rejected_frames += 1
                continue

            board_position_mm = detection["tvec"].reshape(3) * 1000.0
            board_rotation = detection["rotation_matrix"]
            board_rpy = Rotation.from_matrix(board_rotation).as_euler(
                "xyz",
                degrees=True,
            )
            robot_pose = robot.get_pose()

            row = {
                "sample": len(rows) + 1,
                "elapsed_sec": round(now - start_time, 6),
                "corner_count": detection["corner_count"],
                "reprojection_error_px": round(
                    detection["reprojection_error_px"],
                    6,
                ),
                "board_x_mm": round(float(board_position_mm[0]), 6),
                "board_y_mm": round(float(board_position_mm[1]), 6),
                "board_z_mm": round(float(board_position_mm[2]), 6),
                "board_roll_deg": round(float(board_rpy[0]), 6),
                "board_pitch_deg": round(float(board_rpy[1]), 6),
                "board_yaw_deg": round(float(board_rpy[2]), 6),
                "robot_x_mm": "",
                "robot_y_mm": "",
                "robot_z_mm": "",
                "robot_roll_deg": "",
                "robot_pitch_deg": "",
                "robot_yaw_deg": "",
            }

            if robot_pose is not None:
                robot_rpy = Rotation.from_quat(
                    robot_pose["quaternion"]
                ).as_euler("xyz", degrees=True)
                row.update(
                    {
                        "robot_x_mm": round(
                            float(robot_pose["position_mm"][0]),
                            6,
                        ),
                        "robot_y_mm": round(
                            float(robot_pose["position_mm"][1]),
                            6,
                        ),
                        "robot_z_mm": round(
                            float(robot_pose["position_mm"][2]),
                            6,
                        ),
                        "robot_roll_deg": round(float(robot_rpy[0]), 6),
                        "robot_pitch_deg": round(float(robot_rpy[1]), 6),
                        "robot_yaw_deg": round(float(robot_rpy[2]), 6),
                    }
                )
                robot_positions_mm.append(robot_pose["position_mm"])
                robot_rotations.append(
                    Rotation.from_quat(
                        robot_pose["quaternion"]
                    ).as_matrix()
                )

            rows.append(row)
            board_positions_mm.append(board_position_mm)
            board_rotations.append(board_rotation)
            reprojection_errors.append(detection["reprojection_error_px"])
            corner_counts.append(detection["corner_count"])

            annotated = frame.copy()
            cv2.aruco.drawDetectedCornersCharuco(
                annotated,
                detection["corners"],
            )
            cv2.drawFrameAxes(
                annotated,
                camera.K,
                camera.dist,
                detection["rvec"],
                detection["tvec"],
                0.02,
            )
            last_annotated = annotated

            if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == args.samples:
                print(
                    "  captured {}/{}: corners={}, reproj={:.3f}px".format(
                        len(rows),
                        args.samples,
                        detection["corner_count"],
                        detection["reprojection_error_px"],
                    )
                )
            next_capture_time = now + args.interval_sec
    finally:
        camera.stop()

    if len(rows) < 2:
        raise RuntimeError(
            "Collected only {} valid samples. Check board visibility.".format(
                len(rows)
            )
        )

    board_position_summary = vector_summary(board_positions_mm)
    board_mean_rotation, board_angle_deviations = rotation_deviations_deg(
        board_rotations
    )
    summary = {
        "label": args.label,
        "samples_requested": args.samples,
        "samples_collected": len(rows),
        "rejected_frames": rejected_frames,
        "camera_resolution": [CAMERA_W, CAMERA_H],
        "camera_fps": CAMERA_FPS,
        "board_position_mm": board_position_summary,
        "board_mean_rpy_deg": board_mean_rotation.as_euler(
            "xyz",
            degrees=True,
        ).tolist(),
        "board_angle_deviation_deg": scalar_summary(
            board_angle_deviations
        ),
        "reprojection_error_px": scalar_summary(reprojection_errors),
        "corner_count": scalar_summary(corner_counts),
        "robot_topic": robot_topic,
        "robot_samples_collected": len(robot_positions_mm),
    }

    if len(robot_positions_mm) >= 2:
        robot_position_summary = vector_summary(robot_positions_mm)
        robot_mean_rotation, robot_angle_deviations = rotation_deviations_deg(
            robot_rotations
        )
        summary.update(
            {
                "robot_position_mm": robot_position_summary,
                "robot_mean_rpy_deg": robot_mean_rotation.as_euler(
                    "xyz",
                    degrees=True,
                ).tolist(),
                "robot_angle_deviation_deg": scalar_summary(
                    robot_angle_deviations
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.label
    )
    prefix = "charuco_repeatability_{}_{}".format(safe_label, stamp)
    csv_path = OUTPUT_DIR / "{}.csv".format(prefix)
    json_path = OUTPUT_DIR / "{}_summary.json".format(prefix)
    image_path = OUTPUT_DIR / "{}_last.png".format(prefix)

    write_csv(csv_path, rows)
    with json_path.open("w") as file:
        json.dump(summary, file, indent=2)
    if last_annotated is not None:
        cv2.imwrite(str(image_path), last_annotated)

    print("")
    print_vector_summary(
        "Board position repeatability:",
        board_position_summary,
        "mm",
    )
    print(
        "Board angular deviation mean/std/max: "
        "{:.6f} / {:.6f} / {:.6f} deg".format(
            summary["board_angle_deviation_deg"]["mean"],
            summary["board_angle_deviation_deg"]["std"],
            summary["board_angle_deviation_deg"]["max"],
        )
    )
    print(
        "ChArUco reprojection mean/std/max: "
        "{:.6f} / {:.6f} / {:.6f} px".format(
            summary["reprojection_error_px"]["mean"],
            summary["reprojection_error_px"]["std"],
            summary["reprojection_error_px"]["max"],
        )
    )

    if "robot_position_mm" in summary:
        print("")
        print_vector_summary(
            "FrameEE position repeatability:",
            summary["robot_position_mm"],
            "mm",
        )
        print(
            "FrameEE angular deviation mean/std/max: "
            "{:.6f} / {:.6f} / {:.6f} deg".format(
                summary["robot_angle_deviation_deg"]["mean"],
                summary["robot_angle_deviation_deg"]["std"],
                summary["robot_angle_deviation_deg"]["max"],
            )
        )
    else:
        print("")
        print("FrameEE was not received; camera statistics are still valid.")

    print("")
    print("CSV -> {}".format(csv_path))
    print("Summary -> {}".format(json_path))
    if last_annotated is not None:
        print("Last annotated frame -> {}".format(image_path))


if __name__ == "__main__":
    main()
