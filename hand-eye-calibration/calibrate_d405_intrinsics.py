#!/usr/bin/env python3
"""Capture and calibrate D405 color-camera intrinsics with a ChArUco board."""

import argparse
import csv
import datetime
import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURE_DIR = SCRIPT_DIR / "intrinsics_capture"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 15
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN_M = 0.010
MARKER_LEN_M = 0.007
DICT_ID = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 12


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LEN_M,
        MARKER_LEN_M,
        dictionary,
    )
    return board, cv2.aruco.CharucoDetector(board)


def detect_points(image, board, detector):
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    object_points, image_points = board.matchImagePoints(corners, ids)
    if object_points is None or len(object_points) < MIN_CORNERS:
        return None
    return (
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        corners,
        ids,
    )


def reprojection_rms(
        object_points, image_points, rvec, tvec, camera_matrix, distortion):
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion)
    delta = (
        np.asarray(projected).reshape(-1, 2)
        - np.asarray(image_points).reshape(-1, 2)
    )
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def calibrate_views(
        object_points, image_points, image_size,
        initial_k=None, initial_distortion=None):
    flags = 0
    camera_matrix = None
    distortion = None
    if initial_k is not None:
        camera_matrix = np.asarray(initial_k, dtype=np.float64).copy()
        distortion = (
            np.asarray(initial_distortion, dtype=np.float64).reshape(-1, 1)
            if initial_distortion is not None
            else np.zeros((5, 1), dtype=np.float64)
        )
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        distortion,
        flags=flags,
    )


def load_factory_metadata(capture_dir):
    metadata_path = capture_dir / "factory_intrinsics.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open() as handle:
        return json.load(handle)


def calibrate_capture(capture_dir, output_dir, min_views):
    board, detector = make_detector()
    image_paths = sorted([
        path for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in capture_dir.glob(pattern)
    ])
    if not image_paths:
        raise RuntimeError("No calibration images found in {}".format(
            capture_dir))

    object_points = []
    image_points = []
    accepted_paths = []
    image_size = None
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print("Skipping unreadable image: {}".format(path))
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print("Skipping mismatched image size: {}".format(path))
            continue
        detection = detect_points(image, board, detector)
        if detection is None:
            print("Skipping image with too few corners: {}".format(path.name))
            continue
        obj, img, _, _ = detection
        object_points.append(obj)
        image_points.append(img)
        accepted_paths.append(path)

    if len(object_points) < min_views:
        raise RuntimeError(
            "Only {} usable views; at least {} are required".format(
                len(object_points), min_views)
        )

    factory = load_factory_metadata(capture_dir)
    initial_k = (
        np.asarray(factory["camera_matrix"], dtype=float)
        if factory is not None else None
    )
    initial_distortion = (
        np.asarray(factory["distortion"], dtype=float)
        if factory is not None else None
    )

    active = list(range(len(object_points)))
    rejected = []
    result = None
    per_view = None
    for _ in range(3):
        selected_obj = [object_points[index] for index in active]
        selected_img = [image_points[index] for index in active]
        result = calibrate_views(
            selected_obj, selected_img, image_size,
            initial_k, initial_distortion)
        rms, camera_matrix, distortion, rvecs, tvecs = result
        per_view = np.array([
            reprojection_rms(
                obj, img, rvec, tvec, camera_matrix, distortion)
            for obj, img, rvec, tvec in zip(
                selected_obj, selected_img, rvecs, tvecs)
        ])
        median = float(np.median(per_view))
        mad = float(np.median(np.abs(per_view - median)))
        threshold = median + max(3.0 * 1.4826 * mad, 0.10)
        bad_local = [
            local_index for local_index, error in enumerate(per_view)
            if error > threshold
        ]
        if not bad_local or len(active) - len(bad_local) < min_views:
            break
        bad_global = [active[index] for index in bad_local]
        rejected.extend(bad_global)
        active = [
            index for local_index, index in enumerate(active)
            if local_index not in bad_local
        ]

    selected_obj = [object_points[index] for index in active]
    selected_img = [image_points[index] for index in active]
    result = calibrate_views(
        selected_obj, selected_img, image_size,
        initial_k, initial_distortion)
    rms, camera_matrix, distortion, rvecs, tvecs = result
    per_view = np.array([
        reprojection_rms(
            obj, img, rvec, tvec, camera_matrix, distortion)
        for obj, img, rvec, tvec in zip(
            selected_obj, selected_img, rvecs, tvecs)
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    npz_path = output_dir / (
        "d405_charuco_intrinsics_{}.npz".format(stamp)
    )
    json_path = output_dir / (
        "d405_charuco_intrinsics_{}.json".format(stamp)
    )
    views_csv = output_dir / (
        "d405_charuco_intrinsics_views_{}.csv".format(stamp)
    )

    factory_k = (
        np.asarray(factory["camera_matrix"], dtype=float)
        if factory is not None else np.full((3, 3), np.nan)
    )
    factory_dist = (
        np.asarray(factory["distortion"], dtype=float)
        if factory is not None else np.full(5, np.nan)
    )
    np.savez(
        str(npz_path),
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion.reshape(-1),
        image_width=np.array(image_size[0], dtype=int),
        image_height=np.array(image_size[1], dtype=int),
        calibration_rms_px=np.array(rms, dtype=float),
        per_view_rms_px=per_view,
        used_images=np.array(
            [str(accepted_paths[index]) for index in active]),
        rejected_images=np.array(
            [str(accepted_paths[index]) for index in sorted(set(rejected))]),
        factory_camera_matrix=factory_k,
        factory_distortion_coefficients=factory_dist,
        squares_x=np.array(SQUARES_X, dtype=int),
        squares_y=np.array(SQUARES_Y, dtype=int),
        square_length_m=np.array(SQUARE_LEN_M),
        marker_length_m=np.array(MARKER_LEN_M),
    )

    comparison = None
    if factory is not None:
        comparison = {
            "fx_delta_px": float(camera_matrix[0, 0] - factory_k[0, 0]),
            "fy_delta_px": float(camera_matrix[1, 1] - factory_k[1, 1]),
            "cx_delta_px": float(camera_matrix[0, 2] - factory_k[0, 2]),
            "cy_delta_px": float(camera_matrix[1, 2] - factory_k[1, 2]),
            "fx_delta_percent": float(
                100.0 * (camera_matrix[0, 0] / factory_k[0, 0] - 1.0)),
            "fy_delta_percent": float(
                100.0 * (camera_matrix[1, 1] / factory_k[1, 1] - 1.0)),
        }

    used_centers = []
    used_areas = []
    used_distances = []
    for points, tvec in zip(selected_img, tvecs):
        flat = np.asarray(points).reshape(-1, 2)
        center = np.mean(flat, axis=0)
        used_centers.append([
            center[0] / image_size[0],
            center[1] / image_size[1],
        ])
        hull = cv2.convexHull(flat.astype(np.float32))
        used_areas.append(
            cv2.contourArea(hull) / float(image_size[0] * image_size[1])
        )
        used_distances.append(float(np.linalg.norm(tvec)))
    used_centers = np.asarray(used_centers)
    used_areas = np.asarray(used_areas)
    used_distances = np.asarray(used_distances)
    coverage = {
        "center_x_span_fraction": float(np.ptp(used_centers[:, 0])),
        "center_y_span_fraction": float(np.ptp(used_centers[:, 1])),
        "board_area_fraction_min": float(np.min(used_areas)),
        "board_area_fraction_max": float(np.max(used_areas)),
        "distance_m_min": float(np.min(used_distances)),
        "distance_m_max": float(np.max(used_distances)),
    }

    summary = {
        "capture_directory": str(capture_dir),
        "image_size": list(image_size),
        "usable_views": len(object_points),
        "used_views": len(active),
        "rejected_views": len(set(rejected)),
        "calibration_rms_px": float(rms),
        "mean_view_rms_px": float(np.mean(per_view)),
        "max_view_rms_px": float(np.max(per_view)),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "factory_comparison": comparison,
        "capture_coverage": coverage,
        "npz": str(npz_path),
    }
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    with views_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image", "used", "reprojection_rms_px",
                "center_x_fraction", "center_y_fraction",
                "board_area_fraction", "distance_m",
            ),
        )
        writer.writeheader()
        active_errors = {
            global_index: float(error)
            for global_index, error in zip(active, per_view)
        }
        active_metrics = {
            global_index: (
                used_centers[local_index, 0],
                used_centers[local_index, 1],
                used_areas[local_index],
                used_distances[local_index],
            )
            for local_index, global_index in enumerate(active)
        }
        for index, path in enumerate(accepted_paths):
            metrics = active_metrics.get(index, ("", "", "", ""))
            writer.writerow({
                "image": str(path),
                "used": index in active,
                "reprojection_rms_px": active_errors.get(index, ""),
                "center_x_fraction": metrics[0],
                "center_y_fraction": metrics[1],
                "board_area_fraction": metrics[2],
                "distance_m": metrics[3],
            })

    print("\nD405 INTRINSIC CALIBRATION")
    print("  usable/used views: {}/{}".format(
        len(object_points), len(active)))
    print("  calibration RMS: {:.4f} px".format(rms))
    print("  per-view mean/max: {:.4f} / {:.4f} px".format(
        np.mean(per_view), np.max(per_view)))
    print("  fitted K:\n{}".format(np.array2string(
        camera_matrix, precision=6, suppress_small=True)))
    print("  fitted distortion: {}".format(
        np.round(distortion.reshape(-1), 8).tolist()))
    print(
        "  capture coverage: center span x={:.2f}, y={:.2f}; "
        "area={:.3f}-{:.3f}; distance={:.3f}-{:.3f} m".format(
            coverage["center_x_span_fraction"],
            coverage["center_y_span_fraction"],
            coverage["board_area_fraction_min"],
            coverage["board_area_fraction_max"],
            coverage["distance_m_min"],
            coverage["distance_m_max"],
        )
    )
    if (
            coverage["center_x_span_fraction"] < 0.5
            or coverage["center_y_span_fraction"] < 0.4):
        print(
            "  WARNING: board centers do not span enough of the image. "
            "Capture more edge and corner views."
        )
    if (
            coverage["board_area_fraction_max"]
            < 1.5 * coverage["board_area_fraction_min"]):
        print(
            "  WARNING: board image size has little variation. "
            "Capture additional near and far views."
        )
    if comparison is not None:
        print("  factory delta:")
        print(
            "    fx={:+.3f}px ({:+.3f}%), fy={:+.3f}px ({:+.3f}%)".format(
                comparison["fx_delta_px"],
                comparison["fx_delta_percent"],
                comparison["fy_delta_px"],
                comparison["fy_delta_percent"],
            )
        )
        print(
            "    cx={:+.3f}px, cy={:+.3f}px".format(
                comparison["cx_delta_px"], comparison["cy_delta_px"])
        )
    print("  saved: {}".format(npz_path))
    print("  summary: {}".format(json_path))
    print("  views: {}".format(views_csv))
    return npz_path


def run_capture(capture_dir, sample_count):
    try:
        import pyrealsense2 as rs
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QLabel,
            QPushButton,
            QShortcut,
            QVBoxLayout,
            QWidget,
        )
        from PyQt5.QtGui import QKeySequence
    except ImportError as exc:
        raise RuntimeError(
            "Capture mode requires pyrealsense2 and PyQt5"
        ) from exc

    board, detector = make_detector()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
    profile = pipeline.start(config)
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    factory = {
        "image_width": intr.width,
        "image_height": intr.height,
        "camera_matrix": [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        "distortion": list(intr.coeffs[:5]),
        "distortion_model": str(intr.model),
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    with (capture_dir / "factory_intrinsics.json").open("w") as handle:
        json.dump(factory, handle, indent=2)

    class CaptureWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.frame = None
            self.detection = None
            self.count = len(list(capture_dir.glob("view_*.png")))
            self.status = QLabel()
            self.status.setAlignment(Qt.AlignCenter)
            self.image = QLabel()
            self.image.setAlignment(Qt.AlignCenter)
            self.capture_button = QPushButton("Capture View [Space]")
            self.capture_button.clicked.connect(self.capture)
            layout = QVBoxLayout(self)
            layout.addWidget(self.status)
            layout.addWidget(self.image, stretch=1)
            layout.addWidget(self.capture_button)
            QShortcut(QKeySequence("Space"), self, activated=self.capture)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(33)
            self.resize(1100, 700)
            self.update_status(0)

        def update_status(self, corners):
            self.status.setText(
                "Captured: {} / {} | detected corners: {} | "
                "move board across image, distance, and tilt".format(
                    self.count, sample_count, corners)
            )

        def refresh(self):
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                return
            self.frame = np.asanyarray(color.get_data())
            display = self.frame.copy()
            self.detection = detect_points(display, board, detector)
            corner_count = 0
            if self.detection is not None:
                _, _, corners, ids = self.detection
                corner_count = len(ids)
                cv2.aruco.drawDetectedCornersCharuco(
                    display, corners, ids)
            self.update_status(corner_count)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0],
                rgb.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                1024, 576, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image.setPixmap(pixmap)

        def capture(self):
            if self.frame is None or self.detection is None:
                print(
                    "Cannot capture: need at least {} detected corners."
                    .format(MIN_CORNERS)
                )
                return
            self.count += 1
            path = capture_dir / "view_{:03d}.png".format(self.count)
            cv2.imwrite(str(path), self.frame)
            print("Saved {}".format(path))
            if self.count >= sample_count:
                self.status.setText(
                    "Capture target reached. Close this window and calibrate."
                )

        def closeEvent(self, event):
            self.timer.stop()
            pipeline.stop()
            event.accept()

    app = QApplication.instance() or QApplication([])
    window = CaptureWindow()
    window.show()
    app.exec_()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture or calibrate D405 ChArUco intrinsics.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture", action="store_true",
        help="Open the D405 capture UI.")
    mode.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate from images already in --capture-dir.")
    parser.add_argument(
        "--capture-dir", default=str(DEFAULT_CAPTURE_DIR),
        help="Directory for captured calibration images.")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for fitted intrinsics.")
    parser.add_argument(
        "--samples", type=int, default=40,
        help="Capture target shown in the UI.")
    parser.add_argument(
        "--min-views", type=int, default=20,
        help="Minimum usable views required for calibration.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.capture:
        run_capture(capture_dir, args.samples)
    else:
        calibrate_capture(capture_dir, output_dir, args.min_views)
