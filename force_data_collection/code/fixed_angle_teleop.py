#!/usr/bin/env python3
"""
Fixed-angle keyboard teleoperation for force insertion experiments.

The direct-down reference is an absolute robot orientation. A non-zero entry
angle is applied relative to that reference, then the target remains locked.
"""

import argparse
import sys
import time
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import Transform, Vector3
from PyQt5 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, String

from force_collection_common import (
    apply_workspace_limits,
    clip_norm,
    commanded_insertion_axis,
    locked_target_rotation,
    teleop_velocity,
    workspace_center,
)


HELP = """
Fixed-angle force insertion teleoperation

  Hold W / S : robot-base X+ / X-
  Hold A / D : robot-base Y+ / Y-
  Hold C / V : direct-down base Z, oblique locked-axis insertion/retraction
  Space      : stop and hold the current position
  H          : print this help
  Q          : stop and quit

Movement stops when the key is released or this window loses focus.
The keyboard cannot change tool orientation.
Keep a hand on the physical emergency stop.
"""


def parse_args(default_angle_deg=None, default_label_angle_deg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument(
        "--entry-angle-deg",
        type=float,
        default=0.0 if default_angle_deg is None else default_angle_deg,
        help="Tool tilt angle relative to the configured straight orientation.",
    )
    parser.add_argument(
        "--label-angle-deg",
        type=float,
        default=default_label_angle_deg,
        help=(
            "Experimental angle label published to the recorder. If omitted, "
            "uses --entry-angle-deg."
        ),
    )
    parser.add_argument(
        "--straight-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, -13.0, 0.0),
        help="Absolute straight orientation in XYZ Euler degrees.",
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
    parser.add_argument("--max-linear-vel", type=float, default=0.50)
    parser.add_argument("--max-angular-vel", type=float, default=0.05)
    parser.add_argument("--orientation-tol-deg", type=float, default=0.75)
    parser.add_argument("--orientation-settle-s", type=float, default=0.5)
    parser.add_argument("--orientation-timeout-s", type=float, default=30.0)
    parser.add_argument("--pose-timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--workspace-min-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-42.0, -133.0, -13.0),
        help="Minimum allowed robot-base tip position in millimeters.",
    )
    parser.add_argument(
        "--workspace-max-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(10.0, -85.0, 30.0),
        help="Maximum allowed robot-base tip position in millimeters.",
    )
    parser.add_argument(
        "--workspace-tol-mm",
        type=float,
        default=0.5,
        help="Boundary and centering tolerance used for workspace checks.",
    )
    parser.add_argument(
        "--center-position-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "Tip position to move to before teleoperation. Defaults to the "
            "workspace midpoint; use NaN for any axis that should not be moved."
        ),
    )
    parser.add_argument(
        "--center-max-linear-vel",
        type=float,
        default=0.50,
        help="Maximum speed for automatic workspace-centering motion.",
    )
    parser.add_argument("--center-settle-s", type=float, default=0.3)
    parser.add_argument("--center-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--skip-centering",
        action="store_true",
        help="Skip automatic motion to the workspace center before teleoperation.",
    )
    parser.add_argument(
        "--max-travel-mm",
        type=float,
        default=25.0,
        help="Maximum target displacement from the pose where teleoperation starts.",
    )
    parser.add_argument(
        "--max-insertion-mm",
        type=float,
        default=20.0,
        help="Maximum insertion travel along the locked tool axis.",
    )
    parser.add_argument(
        "--max-retraction-mm",
        type=float,
        default=20.0,
        help="Maximum retraction travel opposite the locked tool axis.",
    )
    parser.add_argument(
        "--linear-hold-error-deg",
        type=float,
        default=2.0,
        help="Suppress linear motion while orientation error exceeds this value.",
    )
    parser.add_argument(
        "--disable-horizontal-stall-guard",
        action="store_true",
        help=(
            "Disable the oblique-insertion guard that stops insertion when "
            "horizontal pose progress stalls."
        ),
    )
    parser.add_argument(
        "--horizontal-stall-window-s",
        type=float,
        default=0.6,
        help="Pose-history window used to detect stalled horizontal progress.",
    )
    parser.add_argument(
        "--horizontal-stall-axis-min",
        type=float,
        default=0.35,
        help=(
            "Minimum horizontal fraction of the tool axis required before the "
            "stall guard is active."
        ),
    )
    parser.add_argument(
        "--horizontal-stall-min-axial-vel",
        type=float,
        default=0.03,
        help="Minimum insertion-axis speed considered an active insertion command.",
    )
    parser.add_argument(
        "--horizontal-stall-min-progress-mm-s",
        type=float,
        default=0.02,
        help="Minimum acceptable horizontal progress speed during oblique insertion.",
    )
    parser.add_argument(
        "--horizontal-stall-progress-ratio",
        type=float,
        default=0.10,
        help=(
            "Minimum actual/expected horizontal progress ratio during oblique "
            "insertion."
        ),
    )
    parser.add_argument(
        "--direct-down-base-z-threshold-deg",
        type=float,
        default=1.0,
        help=(
            "Use pure robot-base Z for C/V when --entry-angle-deg is within "
            "this tolerance of direct-down."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation before rotating to the requested angle.",
    )
    args = parser.parse_args()
    if args.label_angle_deg is None:
        args.label_angle_deg = args.entry_angle_deg
    workspace_min = np.asarray(args.workspace_min_mm, dtype=float)
    workspace_max = np.asarray(args.workspace_max_mm, dtype=float)
    if np.any(workspace_min >= workspace_max):
        parser.error("each --workspace-min-mm value must be below --workspace-max-mm")
    if args.workspace_tol_mm < 0.0:
        parser.error("--workspace-tol-mm must be non-negative")
    if np.any(
        workspace_min + args.workspace_tol_mm
        >= workspace_max - args.workspace_tol_mm
    ):
        parser.error("--workspace-tol-mm is too large for the configured workspace")
    if args.center_max_linear_vel <= 0.0:
        parser.error("--center-max-linear-vel must be positive")
    if args.center_settle_s < 0.0:
        parser.error("--center-settle-s must be non-negative")
    if args.center_timeout_s <= 0.0:
        parser.error("--center-timeout-s must be positive")
    if args.horizontal_stall_window_s <= 0.0:
        parser.error("--horizontal-stall-window-s must be positive")
    if args.horizontal_stall_axis_min < 0.0:
        parser.error("--horizontal-stall-axis-min must be non-negative")
    if args.horizontal_stall_min_axial_vel < 0.0:
        parser.error("--horizontal-stall-min-axial-vel must be non-negative")
    if args.horizontal_stall_min_progress_mm_s < 0.0:
        parser.error("--horizontal-stall-min-progress-mm-s must be non-negative")
    if args.horizontal_stall_progress_ratio < 0.0:
        parser.error("--horizontal-stall-progress-ratio must be non-negative")
    if args.direct_down_base_z_threshold_deg < 0.0:
        parser.error("--direct-down-base-z-threshold-deg must be non-negative")
    return args


class FixedAngleTeleop:
    def __init__(self, args):
        self.args = args
        self.position = None
        self.quaternion = None
        self.pose_receipt_time = None
        self.pose_history = deque(maxlen=1000)
        self.origin_position = None
        self.target_rotation = None
        self.tool_insertion_axis = None
        self.insertion_axis = None
        self.local_x = None
        self.local_y = None
        self.down_axis = np.array([0.0, 0.0, -1.0])
        self.last_linear_command = np.zeros(3)
        self.last_action = "initializing"
        self.last_block_reason = "none"
        self.horizontal_stall_latched = False
        self.insertion_command_start_time = None
        self.workspace_min = np.asarray(args.workspace_min_mm, dtype=float)
        self.workspace_max = np.asarray(args.workspace_max_mm, dtype=float)

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
        self.insertion_axis_pub = rospy.Publisher(
            status_prefix + "/insertion_axis", Vector3, queue_size=1, latch=True
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
        self.pose_history.append((self.pose_receipt_time, self.position.copy()))

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
        self.target_rotation = locked_target_rotation(
            straight_rpy_deg=self.args.straight_rpy_deg,
            entry_angle_deg=self.args.entry_angle_deg,
            tilt_axis=self.args.tilt_axis,
            tilt_sign=self.args.tilt_sign,
        )
        matrix = self.target_rotation.as_matrix()
        self.local_x = matrix[:, 0]
        self.local_y = matrix[:, 1]
        self.tool_insertion_axis = -matrix[:, 2]
        self.insertion_axis = commanded_insertion_axis(
            self.tool_insertion_axis,
            self.args.entry_angle_deg,
            self.args.direct_down_base_z_threshold_deg,
        )
        self.target_rpy_deg = self.target_rotation.as_euler(
            "xyz", degrees=True
        )

        self.angle_pub.publish(float(self.args.label_angle_deg))
        self.insertion_axis_pub.publish(*[float(v) for v in self.insertion_axis])
        self.mode_pub.publish(
            "label_{}deg_tilt_{}deg_{}_target_rpy_{:+.3f}_{:+.3f}_{:+.3f}".format(
                self.args.label_angle_deg,
                self.args.entry_angle_deg,
                self.args.tilt_axis,
                *self.target_rpy_deg,
            )
        )

    def orientation_error(self):
        current_rotation = R.from_quat(self.quaternion.copy())
        rotation_error = self.target_rotation * current_rotation.inv()
        rotation_vector = rotation_error.as_rotvec()
        error_deg = float(np.linalg.norm(rotation_vector) * 180.0 / np.pi)
        return rotation_vector, error_deg

    def rotate_to_locked_orientation(self):
        _, initial_error_deg = self.orientation_error()
        if initial_error_deg <= self.args.orientation_tol_deg:
            print(
                "Already at locked target orientation; error {:.3f} deg.".format(
                    initial_error_deg
                )
            )
            return

        if not self.args.yes:
            response = input(
                "\nThe robot will rotate in place to target RPY {} deg.\n"
                "Tilt from straight is {:.1f} degrees about {}.\n"
                "Experimental label is {:.1f} degrees.\n"
                "Confirm clearance, keep hand on e-stop, then type YES: ".format(
                    np.round(self.target_rpy_deg, 3),
                    self.args.entry_angle_deg,
                    self.args.tilt_axis,
                    self.args.label_angle_deg,
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
            linear = apply_workspace_limits(
                self.position,
                linear,
                self.workspace_min,
                self.workspace_max,
                self.args.workspace_tol_mm,
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

    def workspace_center_target(self):
        if self.args.center_position_mm is None:
            target = workspace_center(self.workspace_min, self.workspace_max)
        else:
            target = np.asarray(self.args.center_position_mm, dtype=float)
            if target.shape != (3,):
                raise ValueError("--center-position-mm must contain 3 values")

        target = target.copy()
        active_axes = np.isfinite(target)
        if np.any(active_axes):
            lower = self.workspace_min + self.args.workspace_tol_mm
            upper = self.workspace_max - self.args.workspace_tol_mm
            target[active_axes] = np.clip(
                target[active_axes],
                lower[active_axes],
                upper[active_axes],
            )
        return target, active_axes

    def center_error(self, target, active_axes):
        error = target - self.position
        error[~active_axes] = 0.0
        if not np.any(active_axes):
            return error, 0.0
        max_axis_error = float(np.max(np.abs(error[active_axes])))
        return error, max_axis_error

    def move_to_workspace_center(self):
        if self.args.skip_centering:
            print("Skipping automatic workspace-centering motion.")
            return

        target, active_axes = self.workspace_center_target()
        if not np.any(active_axes):
            print("No finite workspace-centering axes were requested.")
            return

        error, max_axis_error = self.center_error(target, active_axes)
        if max_axis_error <= self.args.workspace_tol_mm:
            print(
                "Already within {:.3f} mm of workspace center target {}.".format(
                    self.args.workspace_tol_mm,
                    np.round(target, 3),
                )
            )
            return

        if not self.args.yes:
            response = input(
                "\nThe robot will translate the tip to workspace center {} mm "
                "with {:.3f} mm tolerance before manual teleoperation.\n"
                "Current position is {} mm.\n"
                "Confirm clearance, keep hand on e-stop, then type YES: ".format(
                    np.round(target, 3),
                    self.args.workspace_tol_mm,
                    np.round(self.position, 3),
                )
            ).strip()
            if response != "YES":
                raise RuntimeError("Operator did not confirm workspace centering")

        print(
            "Moving tip to workspace center target {} mm "
            "(tolerance {:.3f} mm)...".format(
                np.round(target, 3),
                self.args.workspace_tol_mm,
            )
        )
        self.last_action = "moving_to_workspace_center"
        self.action_pub.publish(self.last_action)
        rate = rospy.Rate(self.args.rate_hz)
        deadline = time.monotonic() + self.args.center_timeout_s
        settled_since = None

        while not rospy.is_shutdown():
            if not self.pose_is_fresh():
                raise RuntimeError("Pose stream became stale during centering")
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out while moving to workspace center")

            rotation_vector, orientation_error_deg = self.orientation_error()
            error, max_axis_error = self.center_error(target, active_axes)
            linear = clip_norm(
                error * self.args.position_gain,
                min(self.args.center_max_linear_vel, self.args.max_linear_vel),
            )
            linear = apply_workspace_limits(
                self.position,
                linear,
                self.workspace_min,
                self.workspace_max,
                self.args.workspace_tol_mm,
            )
            angular = clip_norm(
                rotation_vector * self.args.orientation_gain,
                self.args.max_angular_vel,
            )

            if orientation_error_deg > self.args.linear_hold_error_deg:
                linear[:] = 0.0

            self.linear_pub.publish(*[float(v) for v in linear])
            self.angular_pub.publish(*[float(v) for v in angular])

            if max_axis_error <= self.args.workspace_tol_mm:
                if settled_since is None:
                    settled_since = time.monotonic()
                elif time.monotonic() - settled_since >= self.args.center_settle_s:
                    self.stop()
                    print(
                        "Workspace center reached: position={} mm.".format(
                            np.round(self.position, 3)
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
            ("c", "insert_along_tool"),
            ("v", "retract_along_tool"),
        ):
            if key in active_keys:
                labels.append(label)
        return "+".join(labels) if labels else "hold"

    def axial_motion_axis(self):
        if self.insertion_axis is None:
            return self.down_axis
        axis = np.asarray(self.insertion_axis, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm == 0.0:
            return self.down_axis
        return axis / axis_norm

    def horizontal_axis_for_guard(self):
        axis = self.axial_motion_axis()
        horizontal = axis.copy()
        horizontal[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < self.args.horizontal_stall_axis_min:
            return None, horizontal_norm
        return horizontal / horizontal_norm, horizontal_norm

    def horizontal_progress_rate(self, horizontal_axis, start_time=None):
        if len(self.pose_history) < 2:
            return None
        latest_time, latest_position = self.pose_history[-1]
        cutoff = latest_time - self.args.horizontal_stall_window_s
        if start_time is not None:
            cutoff = max(cutoff, start_time)
        oldest_time = None
        oldest_position = None
        for sample_time, sample_position in self.pose_history:
            if sample_time >= cutoff:
                oldest_time = sample_time
                oldest_position = sample_position
                break
        if oldest_time is None or oldest_position is None:
            return None
        elapsed = latest_time - oldest_time
        minimum_elapsed = 0.5 * self.args.horizontal_stall_window_s
        if elapsed < minimum_elapsed:
            return None
        displacement = latest_position - oldest_position
        return float(np.dot(displacement, horizontal_axis) / elapsed)

    def horizontal_stall_guard_reason(
        self,
        linear_velocity,
        active_keys,
        limit_reason,
    ):
        if self.args.disable_horizontal_stall_guard:
            self.horizontal_stall_latched = False
            self.insertion_command_start_time = None
            return None

        axial_axis = self.axial_motion_axis()
        axial_speed = float(np.dot(linear_velocity, axial_axis))
        inserting = (
            "c" in active_keys
            and axial_speed > self.args.horizontal_stall_min_axial_vel
        )
        if not inserting:
            self.horizontal_stall_latched = False
            self.insertion_command_start_time = None
            return None
        if self.insertion_command_start_time is None:
            self.insertion_command_start_time = time.monotonic()

        horizontal_axis, horizontal_fraction = self.horizontal_axis_for_guard()
        if horizontal_axis is None:
            self.horizontal_stall_latched = False
            self.insertion_command_start_time = None
            return None

        if self.horizontal_stall_latched:
            return "horizontal stall guard"

        if limit_reason == "workspace limit":
            self.horizontal_stall_latched = True
            return "horizontal stall guard"

        progress_rate = self.horizontal_progress_rate(
            horizontal_axis,
            start_time=self.insertion_command_start_time,
        )
        if progress_rate is None:
            return None
        expected_rate = axial_speed * horizontal_fraction
        minimum_rate = max(
            self.args.horizontal_stall_min_progress_mm_s,
            expected_rate * self.args.horizontal_stall_progress_ratio,
        )
        if progress_rate < minimum_rate:
            self.horizontal_stall_latched = True
            return "horizontal stall guard"
        return None

    @staticmethod
    def suppress_z_down(linear_velocity):
        linear_velocity = np.asarray(linear_velocity, dtype=float).copy()
        if linear_velocity[2] < 0.0:
            linear_velocity[2] = 0.0
        return linear_velocity

    def _apply_travel_limits(self, linear_velocity):
        linear_velocity = np.asarray(linear_velocity, dtype=float).copy()
        limit_reason = None
        axial_axis = self.axial_motion_axis()
        offset = self.position - self.origin_position
        insertion = float(np.dot(offset, axial_axis))
        axial_speed = float(np.dot(linear_velocity, axial_axis))

        if insertion >= self.args.max_insertion_mm and axial_speed > 0.0:
            linear_velocity -= axial_speed * axial_axis
            limit_reason = "travel limit"
        elif insertion <= -self.args.max_retraction_mm and axial_speed < 0.0:
            linear_velocity -= axial_speed * axial_axis
            limit_reason = "travel limit"

        travel = float(np.linalg.norm(offset))
        if travel >= self.args.max_travel_mm and travel > 0.0:
            outward_axis = offset / travel
            outward_speed = float(np.dot(linear_velocity, outward_axis))
            if outward_speed > 0.0:
                linear_velocity -= outward_speed * outward_axis
                limit_reason = "travel limit"

        workspace_limited = apply_workspace_limits(
            self.position,
            linear_velocity,
            self.workspace_min,
            self.workspace_max,
            self.args.workspace_tol_mm,
        )
        if not np.allclose(workspace_limited, linear_velocity):
            limit_reason = "workspace limit"
        return workspace_limited, limit_reason

    def publish_control(self, active_keys):
        if not self.pose_is_fresh():
            self.stop()
            raise RuntimeError("Pose stream is stale; motion stopped")

        rotation_vector, orientation_error_deg = self.orientation_error()
        angular = clip_norm(
            rotation_vector * self.args.orientation_gain,
            self.args.max_angular_vel,
        )
        requested_linear = teleop_velocity(
            active_keys,
            self.args.max_linear_vel,
            down_axis=self.axial_motion_axis(),
        )
        linear, limit_reason = self._apply_travel_limits(requested_linear)
        self.last_block_reason = "none"
        if (
            np.linalg.norm(requested_linear) > 0.0
            and limit_reason is not None
        ):
            self.last_block_reason = limit_reason
        if (
            np.linalg.norm(requested_linear) > 0.0
            and orientation_error_deg > self.args.linear_hold_error_deg
        ):
            linear[:] = 0.0
            self.last_block_reason = (
                "orientation error {:.2f} > {:.2f} deg".format(
                    orientation_error_deg,
                    self.args.linear_hold_error_deg,
                )
            )
        stall_reason = self.horizontal_stall_guard_reason(
            linear,
            active_keys,
            limit_reason,
        )
        if stall_reason is not None:
            linear = self.suppress_z_down(linear)
            self.last_block_reason = stall_reason
        self.last_linear_command = linear.copy()

        if self.last_block_reason != "none":
            action = "blocked_" + self.last_block_reason.replace(" ", "_")
        else:
            action = self.action_label(
                active_keys if np.linalg.norm(linear) else set()
            )
        if action != self.last_action:
            self.last_action = action
            self.action_pub.publish(action)

        self.linear_pub.publish(*[float(v) for v in linear])
        self.angular_pub.publish(*[float(v) for v in angular])

        axial_axis = self.axial_motion_axis()
        offset = self.position - self.origin_position
        insertion = float(np.dot(offset, axial_axis))
        lateral = float(
            np.linalg.norm(offset - insertion * axial_axis)
        )
        return orientation_error_deg, insertion, lateral

    def stop_translation(self):
        self.horizontal_stall_latched = False
        self.insertion_command_start_time = None
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
            "ATI Fixed-Angle Teleop - label {:.1f} deg".format(
                teleop.args.label_angle_deg
            )
        )
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(620, 380)

        title = QtWidgets.QLabel(
            "Fixed-angle insertion: {:.1f} deg label, {:.1f} deg from straight".format(
                teleop.args.label_angle_deg,
                teleop.args.entry_angle_deg,
            )
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        instructions = QtWidgets.QLabel(
            "Hold W/S: robot X+ / X-       Hold A/D: robot Y+ / Y-\n"
            "Hold C: insert               Hold V: retract\n"
            "Space: stop    Q: quit\n\n"
            "Click this window before controlling the robot.\n"
            "Releasing a key or changing window focus stops translation.\n"
            "Direct-down C/V uses base Z. Oblique C/V follows the locked axis.\n"
            "For oblique insertion, stalled horizontal progress blocks Z-down motion."
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
                "Angle error: {:.2f} deg    Tool-axis travel: {:+.3f} mm    "
                "Lateral travel: {:.3f} mm\n"
                "Command gate: {}".format(
                    self.teleop.last_linear_command[0],
                    self.teleop.last_linear_command[1],
                    self.teleop.last_linear_command[2],
                    angle_error,
                    insertion,
                    lateral,
                    self.teleop.last_block_reason,
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


def run(default_angle_deg=None, default_label_angle_deg=None):
    args = parse_args(
        default_angle_deg=default_angle_deg,
        default_label_angle_deg=default_label_angle_deg,
    )
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
            "Straight reference RPY: {} deg".format(
                np.asarray(args.straight_rpy_deg, dtype=float)
            )
        )
        teleop.configure_locked_orientation()
        print(
            "Locked target RPY: {} deg".format(
                np.round(teleop.target_rpy_deg, 3)
            )
        )
        print(
            "Tool axis in robot base frame: {}".format(
                np.round(teleop.tool_insertion_axis, 4)
            )
        )
        print(
            "Commanded insertion axis in robot base frame: {}".format(
                np.round(teleop.insertion_axis, 4)
            )
        )
        center_target, active_axes = teleop.workspace_center_target()
        print(
            "Workspace bounds: min={} mm max={} mm tol={:.3f} mm".format(
                np.round(teleop.workspace_min, 3),
                np.round(teleop.workspace_max, 3),
                args.workspace_tol_mm,
            )
        )
        if not args.skip_centering and np.any(active_axes):
            print(
                "Pre-teleop center target: {} mm".format(
                    np.round(center_target, 3)
                )
            )
        teleop.rotate_to_locked_orientation()
        teleop.move_to_workspace_center()
        teleop.start_teleoperation()
        print(HELP)
        print(
            "Label angle {:.1f} deg | tilt from straight {:.1f} deg | "
            "hold-to-move speed {:.3f} mm/s".format(
                args.label_angle_deg,
                args.entry_angle_deg,
                args.max_linear_vel,
            )
        )
        if args.disable_horizontal_stall_guard:
            print("Horizontal stall guard: disabled")
        else:
            print(
                "Horizontal stall guard: window {:.2f} s, axis min {:.2f}, "
                "progress min {:.3f} mm/s".format(
                    args.horizontal_stall_window_s,
                    args.horizontal_stall_axis_min,
                    args.horizontal_stall_min_progress_mm_s,
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
