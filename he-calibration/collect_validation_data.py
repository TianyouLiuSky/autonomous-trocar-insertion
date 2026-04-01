import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform

CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 5
SQUARES_X, SQUARES_Y = 8, 6
SQUARE_LEN = 0.010
MARKER_LEN = 0.007
DICT_ID = cv2.aruco.DICT_6X6_250
N_SAMPLES = 27 # 24mm^3 grid size

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

if __name__ == "__main__":
    rospy.init_node('validation_data_collector', anonymous=True)
    cam = RealSenseCamera()
    detector = CharucoDetector()
    robot = RobotTracker()

    robot_poses = []
    board_rvecs = []
    board_tvecs = []

    print(f"\n--- VALIDATION DATA COLLECTOR ---")
    print(f"Waiting for you to drive the robot to the {N_SAMPLES} targets.")
    print("Press SPACE to record a synced Robot + Camera frame.")
    print("Press ESC to exit early.")

    while True:
        frame = cam.get_frame()
        if frame is None: continue

        rvec, tvec, corners = detector.detect_pose(frame, cam.K, cam.dist)
        display_frame = frame.copy()

        if rvec is not None:
            cv2.drawFrameAxes(display_frame, cam.K, cam.dist, rvec, tvec, 0.02)
            cv2.putText(display_frame, "BOARD DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "NO BOARD", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Captured: {len(robot_poses)}/{N_SAMPLES}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Data Collector", display_frame)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        elif key == 32: # SPACE
            if rvec is None or robot.pose is None:
                print("Cannot record: Board not visible or Robot not ready.")
            else:
                robot_poses.append(robot.pose)
                board_rvecs.append(rvec)
                board_tvecs.append(tvec)
                print(f"Recorded Pose {len(robot_poses)}/{N_SAMPLES}")

                if len(robot_poses) >= N_SAMPLES:
                    print("\nAll samples collected! Saving dataset...")
                    np.savez('validation_dataset.npz', 
                             robot_poses=robot_poses, 
                             board_rvecs=board_rvecs, 
                             board_tvecs=board_tvecs)
                    print("✓ Saved to 'validation_dataset.npz'. You can now close the robot.")
                    break

    cam.pipeline.stop()
    cv2.destroyAllWindows()