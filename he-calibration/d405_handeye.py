"""
Hand-Eye Calibration: AY = XB (correct formulation)
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 5
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN = 0.010
MARKER_LEN = 0.007
DICT_ID = cv2.aruco.DICT_6X6_250
N_SAMPLES = 20
MIN_ROTATION_DEG = 5.0


class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5])
        print(f"Camera: {CAMERA_W}x{CAMERA_H} @ {CAMERA_FPS}fps")

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
        print(f"ChArUco: {SQUARES_X}x{SQUARES_Y}, square={SQUARE_LEN*1000:.1f}mm")

    def detect_pose(self, image, K, dist):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < 4:
            return None, None, None
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        return (rvec, tvec, corners) if success else (None, None, None)


class RobotTracker:
    def __init__(self, topic="/SHER20/eye_robot/FrameEE"):
        self.pose = None
        rospy.Subscriber(topic, Transform, self._callback, queue_size=10)
        print(f"Subscribed: {topic}")

    def _callback(self, msg):
        self.pose = {
            't': np.array([msg.translation.x, msg.translation.y, msg.translation.z]) * 0.001,
            'q': np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        }

    def is_ready(self):
        return self.pose is not None

    def get_pose(self):
        return self.pose


class SimultaneousCalibrator:
    def __init__(self):
        self.robot_poses = []
        self.board_rvecs = []
        self.board_tvecs = []

    def add_sample(self, robot_pose, board_rvec, board_tvec):
        R_board, _ = cv2.Rodrigues(board_rvec)
        is_diverse, min_angle = self._check_diversity(R_board)

        if not is_diverse and len(self.robot_poses) > 0:
            return False, min_angle

        self.robot_poses.append(robot_pose)
        self.board_rvecs.append(board_rvec)
        self.board_tvecs.append(board_tvec)

        # Print sample info
        n = len(self.robot_poses)
        R_robot = Rotation.from_quat(robot_pose['q']).as_matrix()
        euler_robot = Rotation.from_matrix(R_robot).as_euler('xyz', degrees=True)
        euler_board = Rotation.from_matrix(R_board).as_euler('xyz', degrees=True)

        print(f"\n{'='*80}")
        print(f"SAMPLE {n}/{N_SAMPLES}")
        print(f"{'='*80}")
        print(f"Robot Gripper (base frame):")
        print(f"  Position (mm): [{robot_pose['t'][0]*1000:7.2f}, {robot_pose['t'][1]*1000:7.2f}, {robot_pose['t'][2]*1000:7.2f}]")
        print(f"  Euler (deg):   [{euler_robot[0]:7.2f}, {euler_robot[1]:7.2f}, {euler_robot[2]:7.2f}]")
        print(f"Board (camera frame):")
        print(f"  Position (mm): [{board_tvec[0,0]*1000:7.2f}, {board_tvec[1,0]*1000:7.2f}, {board_tvec[2,0]*1000:7.2f}]")
        print(f"  Euler (deg):   [{euler_board[0]:7.2f}, {euler_board[1]:7.2f}, {euler_board[2]:7.2f}]")
        print(f"{'='*80}\n")

        return True, min_angle

    def _check_diversity(self, R_new):
        if len(self.board_rvecs) == 0:
            return True, 0.0
        min_angle = 180.0
        for rvec in self.board_rvecs:
            R_existing, _ = cv2.Rodrigues(rvec)
            angle = np.degrees(np.arccos(np.clip((np.trace(R_existing.T @ R_new) - 1) / 2, -1, 1)))
            min_angle = min(min_angle, angle)
        return min_angle >= MIN_ROTATION_DEG, min_angle

    def can_calibrate(self):
        return len(self.robot_poses) >= N_SAMPLES

    def calibrate(self):
        if not self.can_calibrate():
            return None, None, None, None

        print(f"\n{'='*80}")
        print(f"SOLVING AY = XB ({len(self.robot_poses)} samples)")
        print(f"{'='*80}")

        x0 = np.zeros(12)

        def residual_function(x):
            rvec_Y = x[0:3]
            tvec_Y = x[3:6]
            R_Y = Rotation.from_rotvec(rvec_Y).as_matrix()
            t_Y = tvec_Y.reshape(3, 1)

            rvec_X = x[6:9]
            tvec_X = x[9:12]
            R_X = Rotation.from_rotvec(rvec_X).as_matrix()
            t_X = tvec_X.reshape(3, 1)

            residuals = []

            for robot_pose, board_rvec, board_tvec in zip(
                self.robot_poses, self.board_rvecs, self.board_tvecs
            ):
                R_A = Rotation.from_quat(robot_pose['q']).as_matrix()
                t_A = robot_pose['t'].reshape(3, 1)
                R_B, _ = cv2.Rodrigues(board_rvec)
                t_B = board_tvec.reshape(3, 1)

                # AY = XB
                R_left = R_A @ R_Y
                t_left = R_A @ t_Y + t_A
                R_right = R_X @ R_B
                t_right = R_X @ t_B + t_X

                R_diff = R_left - R_right
                residuals.extend(R_diff.flatten())
                t_diff = t_left - t_right
                residuals.extend(t_diff.flatten())

            return np.array(residuals)

        result = least_squares(residual_function, x0, verbose=1, ftol=1e-8, xtol=1e-8, max_nfev=1000)

        if not result.success:
            print(f"✗ Failed: {result.message}")
            return None, None, None, None

        R_board2gripper = Rotation.from_rotvec(result.x[0:3]).as_matrix()
        t_board2gripper = result.x[3:6].reshape(3, 1)
        R_cam2base = Rotation.from_rotvec(result.x[6:9]).as_matrix()
        t_cam2base = result.x[9:12].reshape(3, 1)

        error = self._compute_error(R_board2gripper, t_board2gripper, R_cam2base, t_cam2base)

        # Print full results
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.flatten()

        T_board2gripper = np.eye(4)
        T_board2gripper[:3, :3] = R_board2gripper
        T_board2gripper[:3, 3] = t_board2gripper.flatten()

        euler_X = Rotation.from_matrix(R_cam2base).as_euler('xyz', degrees=True)
        euler_Y = Rotation.from_matrix(R_board2gripper).as_euler('xyz', degrees=True)

        print(f"\n✓ Optimization converged!")
        print(f"  Average error: {error*1000:.3f}mm")

        print(f"\n{'='*80}")
        print(f"CALIBRATION RESULTS")
        print(f"{'='*80}")

        print(f"\nX = T_cam2base (Camera to Robot Base):")
        print(T_cam2base)
        print(f"  Translation (mm): [{t_cam2base[0,0]*1000:8.2f}, {t_cam2base[1,0]*1000:8.2f}, {t_cam2base[2,0]*1000:8.2f}]")
        print(f"  Euler (deg):      [{euler_X[0]:7.2f}, {euler_X[1]:7.2f}, {euler_X[2]:7.2f}]")

        print(f"\nY = T_board2gripper (Board to Gripper):")
        print(T_board2gripper)
        print(f"  Translation (mm): [{t_board2gripper[0,0]*1000:8.2f}, {t_board2gripper[1,0]*1000:8.2f}, {t_board2gripper[2,0]*1000:8.2f}]")
        print(f"  Euler (deg):      [{euler_Y[0]:7.2f}, {euler_Y[1]:7.2f}, {euler_Y[2]:7.2f}]")
        print(f"{'='*80}\n")

        # Verification
        self._verify_calibration(R_board2gripper, t_board2gripper, R_cam2base, t_cam2base)

        return R_cam2base, t_cam2base, R_board2gripper, t_board2gripper

    def _compute_error(self, R_Y, t_Y, R_X, t_X):
        errors = []
        for robot_pose, board_rvec, board_tvec in zip(
            self.robot_poses, self.board_rvecs, self.board_tvecs
        ):
            R_A = Rotation.from_quat(robot_pose['q']).as_matrix()
            t_A = robot_pose['t'].reshape(3, 1)
            R_B, _ = cv2.Rodrigues(board_rvec)
            t_B = board_tvec.reshape(3, 1)

            t_left = R_A @ t_Y + t_A
            t_right = R_X @ t_B + t_X

            error = np.linalg.norm(t_left - t_right)
            errors.append(error)

        return np.mean(errors)

    def _verify_calibration(self, R_Y, t_Y, R_X, t_X):
        """Verify calibration on first 3 samples"""
        print(f"{'='*80}")
        print(f"VERIFICATION (first 3 samples)")
        print(f"{'='*80}")

        for i in range(min(3, len(self.robot_poses))):
            robot_pose = self.robot_poses[i]
            board_rvec = self.board_rvecs[i]
            board_tvec = self.board_tvecs[i]

            R_A = Rotation.from_quat(robot_pose['q']).as_matrix()
            t_A = robot_pose['t'].reshape(3, 1)
            R_B, _ = cv2.Rodrigues(board_rvec)
            t_B = board_tvec.reshape(3, 1)

            # Gripper actual
            gripper_actual = t_A

            # Gripper predicted from camera: T_cam2base @ T_board2cam @ T_board2gripper
            # But we have T_board2cam, need inverse to get T_cam2board
            # Actually: gripper = T_cam2base @ inv(T_board2cam) @ inv(T_board2gripper)
            # Wait, let's use the equation: A @ Y = X @ B
            # So: T_gripper2base = (X @ B) @ inv(Y)
            # Actually simpler: from AY = XB, we have A = XB inv(Y)
            # Let me just compute from equation: AY = XB
            # t_A + R_A @ t_Y = t_X + R_X @ t_B
            # So t_A = t_X + R_X @ t_B - R_A @ t_Y

            # Predicted gripper from camera observation
            t_gripper_pred = t_X + R_X @ t_B - R_A @ t_Y

            # Actually wait, let me reconsider. We have:
            # A @ Y = X @ B
            # Where A is gripper in base, so t_A is gripper position
            # Y is board in gripper
            # X is camera in base
            # B is board in camera

            # From equation: t_A + R_A @ t_Y = t_X + R_X @ t_B
            # This means gripper position + board offset = camera position + board in camera
            # So gripper_predicted = t_X + R_X @ t_B - R_A @ t_Y

            # Actually, I think the issue is I need to think about this more carefully
            # Let me just compute both sides of the equation
            t_left = t_A + R_A @ t_Y  # This is where board is in base frame (via robot)
            t_right = t_X + R_X @ t_B  # This is where board is in base frame (via camera)

            error = np.linalg.norm(t_left - t_right)

            print(f"\nSample {i+1}:")
            print(f"  Gripper actual (mm):       [{gripper_actual[0,0]*1000:7.2f}, {gripper_actual[1,0]*1000:7.2f}, {gripper_actual[2,0]*1000:7.2f}]")
            print(f"  Board via robot (mm):      [{t_left[0,0]*1000:7.2f}, {t_left[1,0]*1000:7.2f}, {t_left[2,0]*1000:7.2f}]")
            print(f"  Board via camera (mm):     [{t_right[0,0]*1000:7.2f}, {t_right[1,0]*1000:7.2f}, {t_right[2,0]*1000:7.2f}]")
            print(f"  Error (mm):                {error*1000:.3f}")

        print(f"{'='*80}\n")


class CalibrationGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.camera = RealSenseCamera()
        self.detector = CharucoDetector()
        self.robot = RobotTracker()
        self.calibrator = SimultaneousCalibrator()
        self.R_cam2base = None
        self.t_cam2base = None
        self.R_board2gripper = None
        self.t_board2gripper = None

        self.setup_ui()
        self.wait_for_robot()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(33)

        print(f"\n{'='*80}")
        print(f"READY: Collect {N_SAMPLES} samples (Press SPACE)")
        print(f"{'='*80}\n")

    def setup_ui(self):
        self.setWindowTitle("Hand-Eye Calibration")
        self.status = QtWidgets.QLabel("Initializing...")
        self.status.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.status.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_record = QtWidgets.QPushButton("Record [SPACE]")
        self.btn_record.clicked.connect(self.record_pose)
        self.btn_record.setEnabled(False)

        self.btn_calibrate = QtWidgets.QPushButton("Calibrate")
        self.btn_calibrate.clicked.connect(self.compute_calibration)
        self.btn_calibrate.setEnabled(False)

        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)

        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.view.invertY(True)
        self.view.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)

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
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.record_pose)

    def wait_for_robot(self):
        print("Waiting for robot...")
        timeout = 10.0
        start = rospy.Time.now().to_sec()
        while not self.robot.is_ready():
            QtWidgets.QApplication.processEvents()
            rospy.sleep(0.05)
            if rospy.Time.now().to_sec() - start > timeout:
                self.status.setText("ERROR: No robot!")
                return
        print("✓ Robot ready")
        self.status.setText(f"0/{N_SAMPLES}")
        self.btn_record.setEnabled(True)

    def update(self):
        image = self.camera.get_frame()
        if image is None:
            return

        rvec, tvec, corners = self.detector.detect_pose(image, self.camera.K, self.camera.dist)

        if rvec is not None:
            cv2.drawFrameAxes(image, self.camera.K, self.camera.dist, rvec, tvec, 0.02)
            if corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(image, corners)

        n = len(self.calibrator.robot_poses)
        cv2.putText(image, f"{n}/{N_SAMPLES}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        status_color = (0, 255, 0) if rvec is not None else (0, 0, 255)
        cv2.putText(image, "Board: OK" if rvec is not None else "Board: NO", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_item.setImage(np.transpose(img_rgb, (1, 0, 2)), autoLevels=False)
        self.status.setText(f"{n}/{N_SAMPLES}")

        if self.calibrator.can_calibrate():
            self.btn_calibrate.setEnabled(True)

    def record_pose(self):
        image = self.camera.get_frame()
        if image is None:
            return
        rvec, tvec, _ = self.detector.detect_pose(image, self.camera.K, self.camera.dist)
        if rvec is None or not self.robot.is_ready():
            print("✗ Board not detected or robot not ready")
            return
        robot_pose = self.robot.get_pose()
        success, angle = self.calibrator.add_sample(robot_pose, rvec, tvec)
        if not success:
            print(f"✗ Too similar ({angle:.1f}° < {MIN_ROTATION_DEG}°)")

    def compute_calibration(self):
        self.status.setText("Computing...")
        QtWidgets.QApplication.processEvents()

        R_X, t_X, R_Y, t_Y = self.calibrator.calibrate()

        if R_X is not None:
            self.R_cam2base = R_X
            self.t_cam2base = t_X
            self.R_board2gripper = R_Y
            self.t_board2gripper = t_Y
            self.status.setText("✓ Done!")
            self.btn_save.setEnabled(True)
        else:
            self.status.setText("✗ Failed")

    def save_result(self):
        if self.R_cam2base is None:
            return

        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = self.R_cam2base
        T_cam2base[:3, 3] = self.t_cam2base.flatten()

        T_board2gripper = np.eye(4)
        T_board2gripper[:3, :3] = self.R_board2gripper
        T_board2gripper[:3, 3] = self.t_board2gripper.flatten()

        np.savez('hand_eye_calibration.npz',
                 R_cam2base=self.R_cam2base, t_cam2base=self.t_cam2base, T_cam2base=T_cam2base,
                 R_board2gripper=self.R_board2gripper, t_board2gripper=self.t_board2gripper,
                 T_board2gripper=T_board2gripper,
                 camera_matrix=self.camera.K, dist_coeffs=self.camera.dist)
        print("\n✓ Saved to: hand_eye_calibration.npz")
        self.status.setText("✓ Saved")

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()


if __name__ == "__main__":
    rospy.init_node('hand_eye_calibration', anonymous=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui = CalibrationGUI()
    app.exec_()