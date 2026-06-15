#!/usr/bin/env python3
"""
Fixed-angle keyboard teleoperation for force insertion experiments.

The pose at startup is the direct-down reference pose. A non-zero entry angle is
applied relative to that pose, then the target orientation remains locked.
"""

import argparse
import select
import sys
import termios
import time
import tty

import numpy as np
import rospy
from geometry_msgs.msg import Transform, Vector3
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, String

from force_collection_common import clip_norm


HELP = """
Fixed-angle force insertion teleoperation

  W / S      : forward / backward along locked tool Y
  A / D      : left / right along locked tool X
  Down Arrow : insert/down along the locked tool axis
  Up Arrow   : retract/up along the locked tool axis
  Space      : stop and hold the current position
  H          : print this help
  Q          : stop and quit

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
    parser.add_argument("--step-mm", type=float, default=0.05)
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


class KeyReader:
    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard teleoperation requires an interactive terminal")
        self._settings = None

    def __enter__(self):
        self._settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)

    def read_key(self):
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None
        first = sys.stdin.read(1)
        if first != "\x1b":
            return first.lower()

        sequence = first
        deadline = time.time() + 0.01
        while time.time() < deadline and len(sequence) < 3:
            if select.select([sys.stdin], [], [], 0.001)[0]:
                sequence += sys.stdin.read(1)
        return {
            "\x1b[A": "up",
            "\x1b[B": "down",
            "\x1b[C": "right",
            "\x1b[D": "left",
        }.get(sequence, "escape")


class FixedAngleTeleop:
    def __init__(self, args):
        self.args = args
        self.position = None
        self.quaternion = None
        self.pose_receipt_time = None
        self.target_position = None
        self.origin_position = None
        self.target_rotation = None
        self.insertion_axis = None
        self.local_x = None
        self.local_y = None
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
        self.target_position = self.position.copy()
        self.last_action = "hold"
        self.action_pub.publish(self.last_action)

    def _bounded_target(self, candidate):
        offset = candidate - self.origin_position
        insertion = float(np.dot(offset, self.insertion_axis))
        if insertion > self.args.max_insertion_mm:
            print("\rInsertion limit reached.                 ", end="", flush=True)
            return None
        if insertion < -self.args.max_retraction_mm:
            print("\rRetraction limit reached.                ", end="", flush=True)
            return None
        if np.linalg.norm(offset) > self.args.max_travel_mm:
            print("\rTravel limit reached.                    ", end="", flush=True)
            return None
        return candidate

    def apply_key(self, key):
        step = self.args.step_mm
        direction = None
        action = None

        if key == "down":
            direction, action = self.insertion_axis, "insert"
        elif key == "up":
            direction, action = -self.insertion_axis, "retract"
        elif key == "a":
            direction, action = self.local_x, "left"
        elif key == "d":
            direction, action = -self.local_x, "right"
        elif key == "w":
            direction, action = self.local_y, "forward"
        elif key == "s":
            direction, action = -self.local_y, "backward"
        elif key == " ":
            self.target_position = self.position.copy()
            self.last_action = "hold"
            self.action_pub.publish(self.last_action)
            return True
        elif key == "h":
            print(HELP)
            return True
        elif key in ("q", "\x03", "escape"):
            return False
        else:
            return True

        candidate = self._bounded_target(self.target_position + step * direction)
        if candidate is not None:
            self.target_position = candidate
            self.last_action = action
            self.action_pub.publish(action)
        return True

    def publish_control(self):
        if not self.pose_is_fresh():
            self.stop()
            raise RuntimeError("Pose stream is stale; motion stopped")

        rotation_vector, orientation_error_deg = self.orientation_error()
        angular = clip_norm(
            rotation_vector * self.args.orientation_gain,
            self.args.max_angular_vel,
        )
        position_error = self.target_position - self.position
        linear = clip_norm(
            position_error * self.args.position_gain,
            self.args.max_linear_vel,
        )
        if orientation_error_deg > self.args.linear_hold_error_deg:
            linear[:] = 0.0

        self.linear_pub.publish(*[float(v) for v in linear])
        self.angular_pub.publish(*[float(v) for v in angular])

        offset = self.position - self.origin_position
        insertion = float(np.dot(offset, self.insertion_axis))
        lateral = float(
            np.linalg.norm(offset - insertion * self.insertion_axis)
        )
        return orientation_error_deg, insertion, lateral

    def stop(self):
        if hasattr(self, "linear_pub"):
            for _ in range(3):
                self.linear_pub.publish(0.0, 0.0, 0.0)
                self.angular_pub.publish(0.0, 0.0, 0.0)


def run(default_angle_deg=None):
    args = parse_args(default_angle_deg=default_angle_deg)
    if args.step_mm <= 0.0:
        raise ValueError("--step-mm must be positive")
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
            "Step {:.3f} mm | angle {:.1f} deg | linear limit {:.3f} mm/s".format(
                args.step_mm, args.entry_angle_deg, args.max_linear_vel
            )
        )

        rate = rospy.Rate(args.rate_hz)
        last_display = 0.0
        with KeyReader() as keyboard:
            running = True
            while running and not rospy.is_shutdown():
                key = keyboard.read_key()
                if key is not None:
                    running = teleop.apply_key(key)
                    if not running:
                        teleop.stop()
                        break
                angle_error, insertion, lateral = teleop.publish_control()
                now = time.monotonic()
                if now - last_display >= 0.2:
                    print(
                        "\rangle error={:5.2f} deg | insertion={:+6.3f} mm | "
                        "lateral={:6.3f} mm | {:10s}".format(
                            angle_error,
                            insertion,
                            lateral,
                            teleop.last_action,
                        ),
                        end="",
                        flush=True,
                    )
                    last_display = now
                rate.sleep()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        teleop.stop()
        print("\nRobot command stopped.")


if __name__ == "__main__":
    run()
