import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 5
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN = 0.010
MARKER_LEN = 0.007
DICT_ID = cv2.aruco.DICT_6X6_250
N_SAMPLES = 27  # 24mm^3 grid

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
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
        self.board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN, MARKER_LEN, dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect_pose(self, image, K, dist):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < 4: return None, None, None
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        return (rvec, tvec, corners) if success else (None, None, None)

class RobotTracker:
    def __init__(self, topic="/SHER20/eye_robot/FrameEE"):
        self.pose = None
        rospy.Subscriber(topic, Transform, self._callback, queue_size=10)

    def _callback(self, msg):
        self.pose = {
            't': np.array([msg.translation.x, msg.translation.y, msg.translation.z]) * 0.001,
            'q': np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        }
    
    def is_ready(self):
        return self.pose is not None

class DataCollectorGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.camera = RealSenseCamera()
        self.detector = CharucoDetector()
        self.robot = RobotTracker()

        self.robot_poses = []
        self.board_rvecs = []
        self.board_tvecs = []

        self.setup_ui()
        
        # Start update timer (30fps)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def setup_ui(self):
        self.setWindowTitle("Validation Data Collector")
        self.status = QtWidgets.QLabel(f"Captured: 0 / {N_SAMPLES}")
        self.status.setStyleSheet("font-size: 16pt; font-weight: bold; color: blue;")
        self.status.setAlignment(QtCore.Qt.AlignCenter)

        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.view.invertY(True)
        self.view.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.view.addItem(self.img_item)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.win, stretch=1)
        self.resize(1280, 800)
        self.show()

        # Bind SPACE bar to capture
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.record_data)

    def update_frame(self):
        image = self.camera.get_frame()
        if image is None: return

        rvec, tvec, corners = self.detector.detect_pose(image, self.camera.K, self.camera.dist)

        if rvec is not None:
            cv2.drawFrameAxes(image, self.camera.K, self.camera.dist, rvec, tvec, 0.02)
            cv2.putText(image, "Board: OK", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Board: NOT FOUND", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Convert BGR to RGB for PyQtGraph
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_item.setImage(np.transpose(img_rgb, (1, 0, 2)), autoLevels=False)

    def record_data(self):
        image = self.camera.get_frame()
        if image is None: return
        
        rvec, tvec, _ = self.detector.detect_pose(image, self.camera.K, self.camera.dist)
        
        if rvec is None:
            print("✗ Cannot record: Board not detected!")
            return
        if not self.robot.is_ready():
            print("✗ Cannot record: Waiting for robot ROS data!")
            return

        # Save synced data
        self.robot_poses.append(self.robot.pose)
        self.board_rvecs.append(rvec)
        self.board_tvecs.append(tvec)
        
        count = len(self.robot_poses)
        self.status.setText(f"Captured: {count} / {N_SAMPLES}")
        print(f"✓ Recorded Pose {count}/{N_SAMPLES}")

        if count >= N_SAMPLES:
            self.save_and_exit()

    def save_and_exit(self):
        print("\nAll samples collected! Saving dataset...")
        np.savez('validation_dataset.npz', 
                 robot_poses=self.robot_poses, 
                 board_rvecs=self.board_rvecs, 
                 board_tvecs=self.board_tvecs)
        
        self.status.setText("✓ SAVED to validation_dataset.npz. You can close this window.")
        self.status.setStyleSheet("font-size: 16pt; font-weight: bold; color: green;")
        print("✓ Saved successfully! You can now run evaluate_calibration.py")

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

if __name__ == "__main__":
    rospy.init_node('validation_data_collector', anonymous=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui = DataCollectorGUI()
    app.exec_()