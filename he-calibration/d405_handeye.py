"""
Hand-Eye Calibration: Fixed Camera + Moving Target on Robot
Goal: Find T_cam2base transformation matrix
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


# ============================================================================
# CONFIGURATION
# ============================================================================
CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 30

# ChArUco board (verify these match your physical board!)
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN = 0.010  # 10mm in meters
MARKER_LEN = 0.007  # 7mm in meters
DICT_ID = cv2.aruco.DICT_6X6_250

N_SAMPLES = 20  # Number of pose pairs needed
MIN_ROTATION_DEG = 5.0  # Minimum rotation between samples


# ============================================================================
# CAMERA
# ============================================================================
class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)

        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5])

        print(f"Camera initialized: {CAMERA_W}x{CAMERA_H} @ {CAMERA_FPS}fps")
        print(f"Intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self):
        self.pipeline.stop()


# ============================================================================
# CHARUCO DETECTOR
# ============================================================================
class CharucoDetector:
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
        self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        print(f"ChArUco board: {SQUARES_X}x{SQUARES_Y}, square={SQUARE_LEN*1000:.1f}mm, marker={MARKER_LEN*1000:.1f}mm")

    def detect_pose(self, image, K, dist):
        """Returns (rvec, tvec) or (None, None) if detection fails"""
        corners, ids, _, _ = self.detector.detectBoard(image)

        if ids is None or len(ids) < 4:
            return None, None, None

        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)

        return (rvec, tvec, corners) if success else (None, None, None)


# ============================================================================
# ROBOT POSE TRACKER
# ============================================================================
class RobotTracker:
    def __init__(self, topic="/SHER21/eye_robot/FrameEE"):
        self.pose = None
        self.count = 0
        rospy.Subscriber(topic, Transform, self._callback, queue_size=10)
        print(f"Subscribed to: {topic}")

    def _callback(self, msg):
        """Robot publishes in mm, convert to meters"""
        self.pose = {
            't': np.array([msg.translation.x, msg.translation.y, msg.translation.z]) * 0.001,
            'q': np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        }
        self.count += 1

    def is_ready(self):
        return self.pose is not None

    def get_pose(self):
        return self.pose


# ============================================================================
# CALIBRATION ENGINE
# ============================================================================
class HandEyeCalibrator:
    def __init__(self):
        self.R_gripper2base = []  # Robot rotations
        self.t_gripper2base = []  # Robot translations
        self.R_target2cam = []    # Board rotations (camera view)
        self.t_target2cam = []    # Board translations (camera view)

    def add_sample(self, robot_pose, board_rvec, board_tvec):
        """Add a calibration sample"""
        # Robot pose: gripper to base (no inversion for eye-to-hand!)
        R_robot = Rotation.from_quat(robot_pose['q']).as_matrix()
        t_robot = robot_pose['t'].reshape(3, 1)

        # Board pose: target to camera
        R_board, _ = cv2.Rodrigues(board_rvec)
        t_board = board_tvec.reshape(3, 1)

        # Check diversity
        is_diverse, min_angle = self._check_diversity(R_board)

        if not is_diverse and len(self.R_target2cam) > 0:
            return False, min_angle

        # Store
        self.R_gripper2base.append(R_robot)
        self.t_gripper2base.append(t_robot)
        self.R_target2cam.append(R_board)
        self.t_target2cam.append(t_board)

        # Print 6D poses
        self._print_pose_pair(len(self.R_gripper2base), robot_pose, R_robot, t_robot, R_board, t_board)

        return True, min_angle

    def _check_diversity(self, R_new):
        """Check if rotation is sufficiently different from existing samples"""
        if len(self.R_target2cam) == 0:
            return True, 0.0

        min_angle = 180.0
        for R_existing in self.R_target2cam:
            angle = np.degrees(np.arccos(np.clip((np.trace(R_existing.T @ R_new) - 1) / 2, -1, 1)))
            min_angle = min(min_angle, angle)

        return min_angle >= MIN_ROTATION_DEG, min_angle

    def _print_pose_pair(self, idx, robot_pose, R_robot, t_robot, R_board, t_board):
        """Print collected 6D poses clearly"""
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx}/{N_SAMPLES}")
        print(f"{'='*80}")

        # Robot pose (6D)
        euler_robot = Rotation.from_matrix(R_robot).as_euler('xyz', degrees=True)
        print(f"Robot End-Effector (base frame):")
        print(f"  Position (m):  [{t_robot[0,0]:7.4f}, {t_robot[1,0]:7.4f}, {t_robot[2,0]:7.4f}]")
        print(f"  Position (mm): [{t_robot[0,0]*1000:7.2f}, {t_robot[1,0]*1000:7.2f}, {t_robot[2,0]*1000:7.2f}]")
        print(f"  Quaternion:    [{robot_pose['q'][0]:7.4f}, {robot_pose['q'][1]:7.4f}, {robot_pose['q'][2]:7.4f}, {robot_pose['q'][3]:7.4f}]")
        print(f"  Euler (deg):   [{euler_robot[0]:7.2f}, {euler_robot[1]:7.2f}, {euler_robot[2]:7.2f}] (XYZ)")

        # Board pose (6D)
        euler_board = Rotation.from_matrix(R_board).as_euler('xyz', degrees=True)
        print(f"\nChArUco Board (camera frame):")
        print(f"  Position (m):  [{t_board[0,0]:7.4f}, {t_board[1,0]:7.4f}, {t_board[2,0]:7.4f}]")
        print(f"  Euler (deg):   [{euler_board[0]:7.2f}, {euler_board[1]:7.2f}, {euler_board[2]:7.2f}] (XYZ)")
        print(f"{'='*80}\n")

    def can_calibrate(self):
        return len(self.R_gripper2base) >= N_SAMPLES

    def calibrate(self):
        """Solve for T_cam2base using OpenCV's hand-eye calibration"""
        if not self.can_calibrate():
            return None, None

        print(f"\n{'='*80}")
        print(f"COMPUTING CALIBRATION ({len(self.R_gripper2base)} samples)")
        print(f"{'='*80}")

        # Try multiple methods
        methods = [
            (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
            (cv2.CALIB_HAND_EYE_PARK, "Park"),
            (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
            (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
            (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis")
        ]

        results = []
        for method, name in methods:
            try:
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    self.R_gripper2base, self.t_gripper2base,
                    self.R_target2cam, self.t_target2cam,
                    method=method
                )

                # Validate
                det = np.linalg.det(R_cam2base)
                has_nan = np.isnan(R_cam2base).any() or np.isnan(t_cam2base).any()

                if not has_nan and 0.99 < det < 1.01:
                    error = self._compute_reprojection_error(R_cam2base, t_cam2base)
                    results.append((name, R_cam2base, t_cam2base, error))
                    print(f"  {name:12s}: det={det:.6f}, error={error:.6f}m")
                else:
                    print(f"  {name:12s}: INVALID (det={det:.4f}, nan={has_nan})")
            except Exception as e:
                print(f"  {name:12s}: FAILED - {e}")

        if not results:
            print("\n✗ All methods failed!")
            return None, None

        # Pick best result (lowest reprojection error)
        results.sort(key=lambda x: x[3])
        best_name, R_best, t_best, best_error = results[0]

        print(f"\n✓ Best method: {best_name} (error: {best_error:.6f}m = {best_error*1000:.3f}mm)")
        self._print_calibration_result(R_best, t_best, best_name)

        return R_best, t_best

    def _compute_reprojection_error(self, R_cam2base, t_cam2base):
        """Compute average reprojection error to validate calibration"""
        errors = []
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.flatten()

        for R_grip, t_grip, R_targ, t_targ in zip(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam, self.t_target2cam
        ):
            # Forward: T_cam2base * T_target2cam should equal T_gripper2base * T_target2gripper
            # We compute T_target2gripper implicitly and check consistency
            T_grip = np.eye(4)
            T_grip[:3, :3] = R_grip
            T_grip[:3, 3] = t_grip.flatten()

            T_targ = np.eye(4)
            T_targ[:3, :3] = R_targ
            T_targ[:3, 3] = t_targ.flatten()

            # Predicted gripper pose from camera observation
            T_grip_predicted = T_cam2base @ T_targ @ np.linalg.inv(T_cam2base @ T_targ @ np.linalg.inv(T_grip))

            # Translation error
            error = np.linalg.norm(T_grip[:3, 3] - T_grip_predicted[:3, 3])
            errors.append(error)

        return np.mean(errors)

    def _print_calibration_result(self, R, t, method):
        """Print calibration result clearly"""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

        print(f"\n{'='*80}")
        print(f"CALIBRATION RESULT: T_cam2base (method: {method})")
        print(f"{'='*80}")
        print(f"\nTransformation Matrix (4x4):")
        print(T)
        print(f"\nTranslation (camera to robot base):")
        print(f"  Meters:      [{t[0,0]:8.5f}, {t[1,0]:8.5f}, {t[2,0]:8.5f}]")
        print(f"  Millimeters: [{t[0,0]*1000:8.2f}, {t[1,0]*1000:8.2f}, {t[2,0]*1000:8.2f}]")
        print(f"\nRotation:")
        print(f"  Euler (deg): [{euler[0]:7.2f}, {euler[1]:7.2f}, {euler[2]:7.2f}] (XYZ)")
        print(f"  Det(R):      {np.linalg.det(R):.8f} (should be 1.0)")
        print(f"{'='*80}\n")


# ============================================================================
# GUI APPLICATION
# ============================================================================
class CalibrationGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Initialize components
        self.camera = RealSenseCamera()
        self.detector = CharucoDetector()
        self.robot = RobotTracker()
        self.calibrator = HandEyeCalibrator()

        self.R_result = None
        self.t_result = None

        # Setup UI
        self.setup_ui()
        self.wait_for_robot()

        # Start update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(33)  # ~30Hz

        print(f"\n{'='*80}")
        print(f"READY TO CALIBRATE")
        print(f"{'='*80}")
        print(f"Goal: Collect {N_SAMPLES} pose pairs with diverse orientations")
        print(f"Press SPACE to record current pose")
        print(f"{'='*80}\n")

    def setup_ui(self):
        self.setWindowTitle("Hand-Eye Calibration")

        # Status
        self.status = QtWidgets.QLabel("Initializing...")
        self.status.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.status.setAlignment(QtCore.Qt.AlignCenter)

        # Buttons
        self.btn_record = QtWidgets.QPushButton("Record Pose [SPACE]")
        self.btn_record.clicked.connect(self.record_pose)
        self.btn_record.setEnabled(False)

        self.btn_calibrate = QtWidgets.QPushButton("Compute Calibration")
        self.btn_calibrate.clicked.connect(self.compute_calibration)
        self.btn_calibrate.setEnabled(False)

        self.btn_save = QtWidgets.QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)

        # Image display
        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.view.invertY(True)
        self.view.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)

        # Layout
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_save)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addLayout(btn_layout)
        layout.addWidget(self.win, stretch=1)

        self.resize(1280, 800)
        self.show()

        # Keyboard shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.record_pose)

    def wait_for_robot(self):
        """Wait for robot data"""
        print("Waiting for robot data...")
        timeout = 10.0
        start = rospy.Time.now().to_sec()

        while not self.robot.is_ready():
            QtWidgets.QApplication.processEvents()
            rospy.sleep(0.05)
            if rospy.Time.now().to_sec() - start > timeout:
                self.status.setText("ERROR: No robot data!")
                print("✗ Timeout waiting for robot!")
                return

        pose = self.robot.get_pose()
        print(f"✓ Robot connected: pos={pose['t']*1000} mm")
        self.status.setText(f"Ready: 0/{N_SAMPLES}")
        self.btn_record.setEnabled(True)

    def update(self):
        """Update display"""
        image = self.camera.get_frame()
        if image is None:
            return

        # Detect board
        rvec, tvec, corners = self.detector.detect_pose(image, self.camera.K, self.camera.dist)

        # Draw
        if rvec is not None:
            cv2.drawFrameAxes(image, self.camera.K, self.camera.dist, rvec, tvec, 0.02)
            if corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(image, corners)

        # Status overlay
        n = len(self.calibrator.R_gripper2base)
        cv2.putText(image, f"Samples: {n}/{N_SAMPLES}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        status_color = (0, 255, 0) if rvec is not None else (0, 0, 255)
        status_text = "Board: DETECTED" if rvec is not None else "Board: NOT FOUND"
        cv2.putText(image, status_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        # Display
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_item.setImage(np.transpose(img_rgb, (1, 0, 2)), autoLevels=False)

        # Update status
        self.status.setText(f"Samples: {n}/{N_SAMPLES}")

        # Enable calibration button
        if self.calibrator.can_calibrate():
            self.btn_calibrate.setEnabled(True)

    def record_pose(self):
        """Record current pose pair"""
        image = self.camera.get_frame()
        if image is None:
            print("✗ No camera frame")
            return

        rvec, tvec, _ = self.detector.detect_pose(image, self.camera.K, self.camera.dist)
        if rvec is None:
            print("✗ Board not detected")
            return

        if not self.robot.is_ready():
            print("✗ Robot not ready")
            return

        robot_pose = self.robot.get_pose()
        success, angle = self.calibrator.add_sample(robot_pose, rvec, tvec)

        if not success:
            print(f"✗ Pose too similar! Rotation diff: {angle:.1f}° (need >{MIN_ROTATION_DEG}°)")

    def compute_calibration(self):
        """Compute calibration"""
        self.status.setText("Computing...")
        QtWidgets.QApplication.processEvents()

        R, t = self.calibrator.calibrate()

        if R is not None:
            self.R_result = R
            self.t_result = t
            self.status.setText("✓ Calibration Complete!")
            self.btn_save.setEnabled(True)

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Calibration Successful!")
            msg.setInformativeText(f"Translation: {t.flatten()*1000} mm\n\nCheck console for details.")
            msg.setWindowTitle("Success")
            msg.exec_()
        else:
            self.status.setText("✗ Calibration Failed")

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Calibration Failed!")
            msg.setInformativeText("Check console for details.")
            msg.setWindowTitle("Error")
            msg.exec_()

    def save_result(self):
        """Save calibration result"""
        if self.R_result is None:
            return

        T = np.eye(4)
        T[:3, :3] = self.R_result
        T[:3, 3] = self.t_result.flatten()

        filename = 'hand_eye_calibration.npz'
        np.savez(filename,
                 R_cam2base=self.R_result,
                 t_cam2base=self.t_result,
                 T_cam2base=T,
                 camera_matrix=self.camera.K,
                 dist_coeffs=self.camera.dist)

        print(f"\n✓ Saved to: {filename}")
        self.status.setText(f"✓ Saved: {filename}")

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    rospy.init_node('hand_eye_calibration', anonymous=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui = CalibrationGUI()
    app.exec_()