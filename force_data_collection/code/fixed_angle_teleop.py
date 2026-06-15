#!/usr/bin/env python3
"""
Fixed-angle keyboard teleoperation for force insertion experiments.

The pose at startup is the direct-down reference pose. A non-zero entry angle is
applied relative to that pose, then the target orientation remains locked.
"""

import argparse
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import Transform, Vector3
from PyQt5 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, String

from force_collection_common import clip_norm, teleop_velocity


HELP = """
Fixed-angle force insertion teleoperation

  Hold W / S : robot-base X+ / X-
  Hold A / D : robot-base Y+ / Y-
  Hold C / V : robot-base Z- (down) / Z+ (up)
  Space      : stop and hold the current position
  H          : print this help
  Q          : stop and quit

Movement stops when the key is released or this window loses focus.
The keyboard cannot change tool orientation.
Keep a hand on the physical emergency stop.
"""


def parse_args(default_angle_deg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument(
        "--entry-angle-deg",
        type=float,
        default=0.0 if default_angle_deg is None else default_angle_deg,
        help="Angle relative to the startup direct-down reference pose.",
    )
    parser.add_argument(
        "--tilt-axis",
        choices=("local-x", "local-y"),
        default="local-y",
        help="Tool-local axis used to create the entry angle.",
    )
    parser.add_argument(
        "--tilt-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Select the side of the direct-down reference used for the tilt.",
    )
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--position-gain", type=float, default=2.0)
    parser.add_argument("--orientation-gain", type=float, default=1.0)
    parser.add_argument("--max-linear-vel", type=float, default=0.20)
    parser.add_argument("--max-angular-vel", type=float, default=0.05)
    parser.add_argument("--orientation-tol-deg", type=float, default=0.75)
    parser.add_argument("--orientation-settle-s", type=float, default=0.5)
    parser.add_argument("--orientation-timeout-s", type=float, default=30.0)
    parser.add_argument("--pose-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--max-travel-mm",
        type=float,
        default=5.0,
        help="Maximum target displacement from the pose where teleoperation starts.",
    )
    parser.add_argument(
        "--max-insertion-mm",
        type=float,
        default=3.0,
        help="Maximum insertion along the locked tool axis from teleoperation start.",
    )
    parser.add_argument(
        "--max-retraction-mm",
        type=float,
        default=3.0,
        help="Maximum retraction from teleoperation start.",
    )
    parser.add_argument(
        "--linear-hold-error-deg",
        type=float,
        default=2.0,
        help="Suppress linear motion while orientation error exceeds this value.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation before rotating to the requested angle.",
    )
    return parser.parse_args()


class FixedAngleTeleop:
    def __init__(self, args):
        self.args = args
        self.position = None
        self.quaternion = None
        self.pose_receipt_time = None
        self.origin_position = None
        self.target_rotation = None
        self.insertion_axis = None
        self.local_x = None
        self.local_y = None
        self.down_axis = np.array([0.0, 0.0, -1.0])
        self.last_linear_command = np.zeros(3)
        self.last_action = "initializing"

        prefix = "/{}".format(args.robot_name)
        self.linear_topic = prefix + "/eyerobot2/desiredTipVelocities"
        self.angular_topic = prefix + "/eyerobot2/desiredTipVelocitiesAngular"
        self.pose_topic = prefix + "/eye_robot/FrameEE"
        status_prefix = "/ati/force_collection"

        self.linear_pub = rospy.Publisher(
            self.linear_topic, Vector3, queue_size=10
        )
        self.angular_pub = rospy.Publisher(
            self.angular_topic, Vector3, queue_size=10
        )
        self.angle_pub = rospy.Publisher(
            status_prefix + "/target_angle_deg", Float64, queue_size=1, latch=True
        )
        self.mode_pub = rospy.Publisher(
            status_prefix + "/mode", String, queue_size=1, latch=True
        )
        self.action_pub = rospy.Publisher(
            status_prefix + "/action", String, queue_size=10, latch=True
        )
        rospy.Subscriber(
            self.pose_topic, Transform, self._pose_callback, queue_size=10
        )
        rospy.on_shutdown(self.stop)

    def _pose_callback(self, message):
        self.position = np.array(
            [
                message.translation.x,
                message.translation.y,
                message.translation.z,
            ],
            dtype=float,
        )
        self.quaternion = np.array(
            [
                message.rotation.x,
                message.rotation.y,
                message.rotation.z,
                message.rotation.w,
            ],
            dtype=float,
        )
        self.pose_receipt_time = time.monotonic()

    def wait_for_pose(self, timeout_s=10.0):
        deadline = time.monotonic() + timeout_s
        while self.quaternion is None and not rospy.is_shutdown():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for pose topic {}".format(self.pose_topic)
                )
            rospy.sleep(0.05)

    def pose_is_fresh(self):
        return (
            self.pose_receipt_time is not None
            and time.monotonic() - self.pose_receipt_time
            <= self.args.pose_timeout_s
        )

    def configure_locked_orientation(self):
        reference_rotation = R.from_quat(self.quaternion.copy())
        axis_index = 0 if self.args.tilt_axis == "local-x" else 1
        local_axis = np.zeros(3)
        local_axis[axis_index] = self.args.tilt_sign
        delta = R.from_rotvec(
            np.deg2rad(self.args.entry_angle_deg) * local_axis
        )
        self.target_rotation = reference_rotation * delta
        matrix = self.target_rotation.as_matrix()
        self.local_x = matrix[:, 0]
        self.local_y = matrix[:, 1]
        self.insertion_axis = -matrix[:, 2]

        self.angle_pub.publish(float(self.args.entry_angle_deg))
        self.mode_pub.publish(
            "{}deg_{}".format(
                self.args.entry_angle_deg, self.args.tilt_axis
            )
        )

    def orientation_error(self):
        current_rotation = R.from_quat(self.quaternion.copy())
        rotation_error = self.target_rotation * current_rotation.inv()
        rotation_vector = rotation_error.as_rotvec()
        error_deg = float(np.linalg.norm(rotation_vector) * 180.0 / np.pi)
        return rotation_vector, error_deg

    def rotate_to_locked_orientation(self):
        if abs(self.args.entry_angle_deg) < 1e-9:
            print("Startup orientation captured as the direct-down reference.")
            return

        if not self.args.yes:
            response = input(
                "\nThe robot will rotate in place to {:.1f} degrees about {}.\n"
                "Confirm clearance, keep hand on e-stop, then type YES: ".format(
                    self.args.entry_angle_deg, self.args.tilt_axis
                )
            ).strip()
            if response != "YES":
                raise RuntimeError("Operator did not confirm orientation motion")

        hold_position = self.position.copy()
        rate = rospy.Rate(self.args.rate_hz)
        deadline = time.monotonic() + self.args.orientation_timeout_s
        settled_since = None

        print("Rotating to the locked entry orientation...")
        while not rospy.is_shutdown():
            if not self.pose_is_fresh():
                raise RuntimeError("Pose stream became stale during orientation setup")
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out while setting the entry orientation")

            rotation_vector, error_deg = self.orientation_error()
            position_error = hold_position - self.position
            linear = clip_norm(
                position_error * self.args.position_gain,
                min(self.args.max_linear_vel, 0.10),
            )
            angular = clip_norm(
                rotation_vector * self.args.orientation_gain,
                self.args.max_angular_vel,
            )

            self.linear_pub.publish(*[float(v) for v in linear])
            self.angular_pub.publish(*[float(v) for v in angular])

            if error_deg <= self.args.orientation_tol_deg:
                if settled_since is None:
                    settled_since = time.monotonic()
                elif (
                    time.monotonic() - settled_since
                    >= self.args.orientation_settle_s
                ):
                    self.stop()
                    print(
                        "Locked orientation reached; error {:.3f} deg.".format(
                            error_deg
                        )
                    )
                    return
            else:
                settled_since = None
            rate.sleep()

    def start_teleoperation(self):
        self.origin_position = self.position.copy()
        self.last_action = "hold"
        self.action_pub.publish(self.last_action)

    @staticmethod
    def action_label(active_keys):
        labels = []
        for key, label in (
            ("w", "x_plus"),
            ("s", "x_minus"),
            ("a", "y_plus"),
            ("d", "y_minus"),
            ("c", "z_minus_down"),
            ("v", "z_plus_up"),
        ):
            if key in active_keys:
                labels.append(label)
        return "+".join(labels) if labels else "hold"

    def _apply_travel_limits(self, linear_velocity):
        linear_velocity = np.asarray(linear_velocity, dtype=float).copy()
        offset = self.position - self.origin_position
        insertion = float(np.dot(offset, self.down_axis))
        axial_speed = float(np.dot(linear_velocity, self.down_axis))

        if insertion >= self.args.max_insertion_mm and axial_speed > 0.0:
            linear_velocity -= axial_speed * self.down_axis
        elif insertion <= -self.args.max_retraction_mm and axial_speed < 0.0:
            linear_velocity -= axial_speed * self.down_axis

        travel = float(np.linalg.norm(offset))
        if travel >= self.args.max_travel_mm and travel > 0.0:
            outward_axis = offset / travel
            outward_speed = float(np.dot(linear_velocity, outward_axis))
            if outward_speed > 0.0:
                linear_velocity -= outward_speed * outward_axis
        return linear_velocity

    def publish_control(self, active_keys):
        if not self.pose_is_fresh():
            self.stop()
            raise RuntimeError("Pose stream is stale; motion stopped")

        rotation_vector, orientation_error_deg = self.orientation_error()
        angular = clip_norm(
            rotation_vector * self.args.orientation_gain,
            self.args.max_angular_vel,
        )
        linear = teleop_velocity(
            active_keys,
            self.args.max_linear_vel,
        )
        linear = self._apply_travel_limits(linear)
        if orientation_error_deg > self.args.linear_hold_error_deg:
            linear[:] = 0.0
        self.last_linear_command = linear.copy()

        action = self.action_label(active_keys if np.linalg.norm(linear) else set())
        if action != self.last_action:
            self.last_action = action
            self.action_pub.publish(action)

        self.linear_pub.publish(*[float(v) for v in linear])
        self.angular_pub.publish(*[float(v) for v in angular])

        offset = self.position - self.origin_position
        insertion = float(np.dot(offset, self.down_axis))
        lateral = float(
            np.linalg.norm(offset - insertion * self.down_axis)
        )
        return orientation_error_deg, insertion, lateral

    def stop_translation(self):
        self.last_linear_command[:] = 0.0
        self.linear_pub.publish(0.0, 0.0, 0.0)
        if self.last_action != "hold":
            self.last_action = "hold"
            self.action_pub.publish(self.last_action)

    def stop(self):
        if hasattr(self, "linear_pub"):
            for _ in range(3):
                self.linear_pub.publish(0.0, 0.0, 0.0)
                self.angular_pub.publish(0.0, 0.0, 0.0)


class TeleopWindow(QtWidgets.QWidget):
    MOVEMENT_KEYS = {
        QtCore.Qt.Key_W: "w",
        QtCore.Qt.Key_S: "s",
        QtCore.Qt.Key_A: "a",
        QtCore.Qt.Key_D: "d",
        QtCore.Qt.Key_C: "c",
        QtCore.Qt.Key_V: "v",
    }

    def __init__(self, teleop):
        super().__init__()
        self.teleop = teleop
        self.active_keys = set()
        self._closing = False

        self.setWindowTitle(
            "ATI Fixed-Angle Teleop - {:.1f} deg".format(
                teleop.args.entry_angle_deg
            )
        )
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(620, 380)

        title = QtWidgets.QLabel(
            "Fixed-angle insertion: {:.1f} deg".format(
                teleop.args.entry_angle_deg
            )
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        instructions = QtWidgets.QLabel(
            "Hold W/S: robot X+ / X-       Hold A/D: robot Y+ / Y-\n"
            "Hold C: robot Z- (down)       Hold V: robot Z+ (up)\n"
            "Space: stop    Q: quit\n\n"
            "Click this window before controlling the robot.\n"
            "Releasing a key or changing window focus stops translation."
        )
        instructions.setAlignment(QtCore.Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 15px;")

        self.key_status = QtWidgets.QLabel("ACTIVE: HOLD")
        self.key_status.setAlignment(QtCore.Qt.AlignCenter)
        self.key_status.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2e7d32;"
        )
        self.motion_status = QtWidgets.QLabel("")
        self.motion_status.setAlignment(QtCore.Qt.AlignCenter)

        stop_button = QtWidgets.QPushButton("STOP TRANSLATION [Space]")
        stop_button.setMinimumHeight(52)
        stop_button.setFocusPolicy(QtCore.Qt.NoFocus)
        stop_button.clicked.connect(self.clear_motion)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addStretch(1)
        layout.addWidget(self.key_status)
        layout.addWidget(self.motion_status)
        layout.addWidget(stop_button)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.control_tick)
        self.timer.start(max(1, int(round(1000.0 / teleop.args.rate_hz))))

    def clear_motion(self):
        self.active_keys.clear()
        self.teleop.stop_translation()
        self.key_status.setText("ACTIVE: HOLD")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = self.MOVEMENT_KEYS.get(event.key())
        if key is not None:
            self.active_keys.add(key)
            self.key_status.setText(
                "ACTIVE: {}".format(
                    self.teleop.action_label(self.active_keys).upper()
                )
            )
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_Space:
            self.clear_motion()
            event.accept()
            return
        if event.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_H:
            QtWidgets.QMessageBox.information(self, "Controls", HELP.strip())
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key = self.MOVEMENT_KEYS.get(event.key())
        if key is not None:
            self.active_keys.discard(key)
            if not self.active_keys:
                self.teleop.stop_translation()
            self.key_status.setText(
                "ACTIVE: {}".format(
                    self.teleop.action_label(self.active_keys).upper()
                )
            )
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.clear_motion()
        super().focusOutEvent(event)

    def control_tick(self):
        if rospy.is_shutdown():
            self.close()
            return
        try:
            angle_error, insertion, lateral = self.teleop.publish_control(
                self.active_keys
            )
            self.motion_status.setText(
                "Command [x,y,z]: [{:+.3f}, {:+.3f}, {:+.3f}] mm/s\n"
                "Angle error: {:.2f} deg    Down travel: {:+.3f} mm    "
                "XY travel: {:.3f} mm".format(
                    self.teleop.last_linear_command[0],
                    self.teleop.last_linear_command[1],
                    self.teleop.last_linear_command[2],
                    angle_error,
                    insertion,
                    lateral,
                )
            )
        except Exception as error:
            self.clear_motion()
            QtWidgets.QMessageBox.critical(self, "Teleoperation stopped", str(error))
            self.close()

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.clear_motion()
            self.teleop.stop()
        event.accept()


def run(default_angle_deg=None):
    args = parse_args(default_angle_deg=default_angle_deg)
    if args.max_linear_vel > 0.5 or args.max_angular_vel > 0.1:
        print("WARNING: configured velocity exceeds the conservative test range.")

    rospy.init_node(
        "fixed_angle_force_teleop_{}deg".format(
            str(args.entry_angle_deg).replace(".", "p")
        ),
        anonymous=True,
    )
    teleop = FixedAngleTeleop(args)

    try:
        print("Waiting for {}...".format(teleop.pose_topic))
        teleop.wait_for_pose()
        print(
            "Startup pose: position={} quaternion={}".format(
                np.round(teleop.position, 4),
                np.round(teleop.quaternion, 6),
            )
        )
        print(
            "Treat the startup tool orientation as the direct-down reference "
            "for the phantom surface."
        )
        teleop.configure_locked_orientation()
        teleop.rotate_to_locked_orientation()
        teleop.start_teleoperation()
        print(HELP)
        print(
            "Angle {:.1f} deg | hold-to-move speed {:.3f} mm/s".format(
                args.entry_angle_deg, args.max_linear_vel
            )
        )
        application = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication(sys.argv)
        )
        window = TeleopWindow(teleop)
        window.show()
        window.raise_()
        window.activateWindow()
        window.setFocus(QtCore.Qt.OtherFocusReason)
        application.exec_()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        teleop.stop()
        print("\nRobot command stopped.")


if __name__ == "__main__":
    run()
