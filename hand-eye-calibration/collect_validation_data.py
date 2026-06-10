import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
import os
import csv
import argparse
from datetime import datetime
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from camera_intrinsics import load_intrinsics

CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 15
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN = 0.010
MARKER_LEN = 0.007
DICT_ID = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 8
MODE_SAMPLE_COUNTS = {
    "spatial": 27,
    "orientation": 25,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def reprojection_error_px(obj_pts, img_pts, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    observed = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1))))

class RealSenseCamera:
    def __init__(
            self, intrinsics_path=None, require_fitted_intrinsics=False):
        fitted = load_intrinsics(
            intrinsics_path, CAMERA_W, CAMERA_H)
        if require_fitted_intrinsics and fitted is None:
            raise ValueError(
                "--require-fitted-intrinsics was set, but no fitted "
                "--intrinsics NPZ was provided."
            )
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5])
        self.intrinsics_source = "D405 factory"
        if fitted is not None:
            self.K, self.dist, self.intrinsics_source = fitted
        print("Camera intrinsics: {}".format(self.intrinsics_source))

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self):
        self.pipeline.stop()

class CharucoDetector:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
        self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect_pose(self, image, K, dist):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < MIN_CORNERS:
            return None, None, corners, ids, None
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < MIN_CORNERS:
            return None, None, corners, ids, None
        try:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        except cv2.error:
            return None, None, corners, ids, None
        if not success:
            return None, None, corners, ids, None
        reproj = reprojection_error_px(obj_pts, img_pts, rvec, tvec, K, dist)
        return rvec, tvec, corners, ids, reproj

class RobotTracker:
    def __init__(self, topic="/SHER20/eye_robot/FrameEE"):
        self.pose = None
        rospy.Subscriber(topic, Transform, self._callback, queue_size=10)

    def _callback(self, msg):
        self.pose = {
            't_mm': np.array([msg.translation.x, msg.translation.y, msg.translation.z]),
            't': np.array([msg.translation.x, msg.translation.y, msg.translation.z]) * 0.001,
            'q': np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        }
    
    def is_ready(self):
        return self.pose is not None

    def get_pose(self):
        if self.pose is None:
            return None
        return {
            't_mm': self.pose['t_mm'].copy(),
            't': self.pose['t'].copy(),
            'q': self.pose['q'].copy(),
        }

class DataCollectorGUI(QtWidgets.QWidget):
    def __init__(
            self, mode, expected_samples, intrinsics_path=None,
            require_fitted_intrinsics=False):
        super().__init__()
        self.mode = mode
        self.expected_samples = expected_samples
        self.camera = RealSenseCamera(
            intrinsics_path, require_fitted_intrinsics)
        self.detector = CharucoDetector()
        self.robot = RobotTracker()

        self.robot_poses = []
        self.board_rvecs = []
        self.board_tvecs = []
        self.corner_counts = []
        self.reprojection_errors_px = []
        self.diagnostic_rows = []

        self.setup_ui()
        
        # Start update timer (30fps)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def setup_ui(self):
        self.setWindowTitle(
            "Validation Data Collector - {}".format(self.mode.capitalize())
        )
        self.status = QtWidgets.QLabel(
            f"Captured: 0 / {self.expected_samples} ({self.mode})"
        )
        self.status.setStyleSheet("font-size: 16pt; font-weight: bold; color: blue;")
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        fitted = self.camera.intrinsics_source != "D405 factory"
        intrinsics_label = QtWidgets.QLabel(
            "Intrinsics: {} | {}".format(
                "FITTED" if fitted else "FACTORY",
                self.camera.intrinsics_source,
            )
        )
        intrinsics_label.setStyleSheet(
            "font-size: 11pt; font-weight: bold; color: {};".format(
                "green" if fitted else "darkorange")
        )
        intrinsics_label.setAlignment(QtCore.Qt.AlignCenter)

        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.view.invertY(True)
        self.view.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(intrinsics_label)
        layout.addWidget(self.win, stretch=1)
        self.resize(1280, 800)
        self.show()

        # Bind SPACE bar to capture
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.record_data)

    def update_frame(self):
        image = self.camera.get_frame()
        if image is None: return

        rvec, tvec, corners, ids, reproj = self.detector.detect_pose(image, self.camera.K, self.camera.dist)
        corner_count = 0 if ids is None else len(ids)

        if rvec is not None:
            cv2.drawFrameAxes(image, self.camera.K, self.camera.dist, rvec, tvec, 0.02)
            cv2.aruco.drawDetectedCornersCharuco(image, corners)
            cv2.putText(
                image,
                f"Board: OK corners={corner_count} reproj={reproj:.2f}px",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                image,
                f"Board: NOT FOUND corners={corner_count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Convert BGR to RGB for PyQtGraph
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_item.setImage(np.transpose(img_rgb, (1, 0, 2)), autoLevels=False)

    def record_data(self):
        image = self.camera.get_frame()
        if image is None: return
        
        rvec, tvec, _, ids, reproj = self.detector.detect_pose(image, self.camera.K, self.camera.dist)
        corner_count = 0 if ids is None else len(ids)
        
        if rvec is None:
            print(f"✗ Cannot record: Board not detected well enough ({corner_count} corners)!")
            return
        if not self.robot.is_ready():
            print("✗ Cannot record: Waiting for robot ROS data!")
            return

        # Save synced data
        robot_pose = self.robot.get_pose()
        self.robot_poses.append(robot_pose)
        self.board_rvecs.append(rvec)
        self.board_tvecs.append(tvec)
        self.corner_counts.append(corner_count)
        self.reprojection_errors_px.append(reproj)

        count = len(self.robot_poses)
        robot_rpy = Rotation.from_quat(robot_pose['q']).as_euler('xyz', degrees=True)
        board_rot, _ = cv2.Rodrigues(rvec)
        board_rpy = Rotation.from_matrix(board_rot).as_euler('xyz', degrees=True)
        row = {
            "sample": count,
            "ros_time_sec": round(rospy.Time.now().to_sec(), 6),
            "corner_count": corner_count,
            "reprojection_error_px": round(float(reproj), 6),
            "rob_tx_mm": round(float(robot_pose['t_mm'][0]), 6),
            "rob_ty_mm": round(float(robot_pose['t_mm'][1]), 6),
            "rob_tz_mm": round(float(robot_pose['t_mm'][2]), 6),
            "rob_roll_deg": round(float(robot_rpy[0]), 6),
            "rob_pitch_deg": round(float(robot_rpy[1]), 6),
            "rob_yaw_deg": round(float(robot_rpy[2]), 6),
            "brd_tx_m": round(float(tvec[0, 0]), 9),
            "brd_ty_m": round(float(tvec[1, 0]), 9),
            "brd_tz_m": round(float(tvec[2, 0]), 9),
            "brd_roll_deg": round(float(board_rpy[0]), 6),
            "brd_pitch_deg": round(float(board_rpy[1]), 6),
            "brd_yaw_deg": round(float(board_rpy[2]), 6),
        }
        self.diagnostic_rows.append(row)
        
        self.status.setText(
            f"Captured: {count} / {self.expected_samples} ({self.mode})"
        )
        print(
            f"✓ Recorded Pose {count}/{self.expected_samples} "
            f"(corners={corner_count}, reproj={reproj:.3f}px)"
        )

        if count >= self.expected_samples:
            self.save_and_exit()

    def save_and_exit(self):
        print("\nAll samples collected! Saving dataset...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = timestamp()
        mode_tag = self.mode.lower()
        latest_npz = os.path.join(SCRIPT_DIR, "validation_dataset.npz")
        mode_latest_npz = os.path.join(
            SCRIPT_DIR, f"validation_dataset_{mode_tag}.npz"
        )
        timestamped_npz = os.path.join(
            OUTPUT_DIR, f"validation_dataset_{mode_tag}_{stamp}.npz"
        )
        csv_path = os.path.join(
            OUTPUT_DIR, f"validation_samples_{mode_tag}_{stamp}.csv"
        )

        payload = {
            "validation_mode": np.array(self.mode),
            "expected_samples": np.array(self.expected_samples, dtype=int),
            "robot_poses": np.array(self.robot_poses, dtype=object),
            "board_rvecs": np.array(self.board_rvecs),
            "board_tvecs": np.array(self.board_tvecs),
            "corner_counts": np.array(self.corner_counts, dtype=int),
            "reprojection_errors_px": np.array(self.reprojection_errors_px, dtype=float),
            "diagnostic_rows": np.array(self.diagnostic_rows, dtype=object),
            "camera_matrix": self.camera.K,
            "dist_coeffs": self.camera.dist,
            "camera_intrinsics_source": np.array(
                self.camera.intrinsics_source),
        }
        np.savez(latest_npz, **payload)
        np.savez(mode_latest_npz, **payload)
        np.savez(timestamped_npz, **payload)

        if self.diagnostic_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.diagnostic_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.diagnostic_rows)
        
        self.status.setText(
            f"✓ SAVED {self.mode} validation dataset. You can close this window."
        )
        self.status.setStyleSheet("font-size: 16pt; font-weight: bold; color: green;")
        print(f"✓ Saved latest dataset -> {latest_npz}")
        print(f"✓ Saved latest {self.mode} dataset -> {mode_latest_npz}")
        print(f"✓ Saved timestamped dataset -> {timestamped_npz}")
        print(f"✓ Saved validation sample CSV -> {csv_path}")
        print("✓ You can now run evaluate_calibration.py")

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture spatial or orientation hand-eye validation data."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-s", "--spatial", action="store_const", dest="mode",
        const="spatial", help="Capture the 27-pose spatial validation set."
    )
    mode_group.add_argument(
        "-o", "--orientation", action="store_const", dest="mode",
        const="orientation", help="Capture the 25-pose orientation validation set."
    )
    parser.set_defaults(mode="spatial")
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Override the expected sample count for an interrupted/custom run."
    )
    parser.add_argument(
        "--intrinsics", default=os.environ.get("HE_CAMERA_INTRINSICS"),
        help=(
            "Optional fitted intrinsics .npz. Defaults to D405 factory "
            "intrinsics; may also be set with HE_CAMERA_INTRINSICS."
        ),
    )
    parser.add_argument(
        "--require-fitted-intrinsics", action="store_true",
        help=(
            "Refuse to start unless --intrinsics resolves to a fitted NPZ. "
            "Use this for fitted-intrinsics experiments."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    expected_samples = (
        args.samples
        if args.samples is not None
        else MODE_SAMPLE_COUNTS[args.mode]
    )
    if expected_samples <= 0:
        raise SystemExit("--samples must be positive")

    rospy.init_node('validation_data_collector', anonymous=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui = DataCollectorGUI(
        args.mode, expected_samples, args.intrinsics,
        args.require_fitted_intrinsics)
    app.exec_()
