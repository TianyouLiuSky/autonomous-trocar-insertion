#!/usr/bin/env python3
"""Real-time ROS force recorder with synchronized pose snapshots and charts."""

import argparse
import math
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import rospy
from geometry_msgs.msg import Transform, Vector3
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Float64MultiArray, String

from force_collection_common import (
    FORCE_CHANNEL_COUNT,
    ema_update,
    finite_stats,
    force_topic_candidates,
    insertion_metrics,
    pad_force,
    safe_json_number,
    select_force_topic,
    select_wavelength_topic,
    session_directory,
    wavelength_topic_candidates,
    write_csv,
    write_json,
)


SAMPLE_FIELDS = [
    "ros_time_s",
    "elapsed_s",
    "force_message_length",
    "force_raw_0",
    "force_raw_1",
    "force_raw_2",
    "force_raw_3",
    "force_filtered_0",
    "force_filtered_1",
    "force_filtered_2",
    "force_filtered_3",
    "baseline_0",
    "baseline_1",
    "baseline_2",
    "baseline_3",
    "force_delta_0",
    "force_delta_1",
    "force_delta_2",
    "force_delta_3",
    "wavelength_message_length",
    "wavelength_ros_time_s",
    "wavelength_age_s",
    "wavelength_raw_0",
    "wavelength_raw_1",
    "wavelength_raw_2",
    "wavelength_raw_3",
    "pose_ros_time_s",
    "pose_age_s",
    "position_x_mm",
    "position_y_mm",
    "position_z_mm",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "insertion_depth_mm",
    "lateral_displacement_mm",
    "insertion_axis_x",
    "insertion_axis_y",
    "insertion_axis_z",
    "target_entry_angle_deg",
    "operator_action",
    "command_linear_x_mm_s",
    "command_linear_y_mm_s",
    "command_linear_z_mm_s",
    "command_angular_x_rad_s",
    "command_angular_y_rad_s",
    "command_angular_z_rad_s",
]


def parse_args():
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument("--force-topic", default=None)
    parser.add_argument("--wavelength-topic", default=None)
    parser.add_argument("--pose-topic", default=None)
    parser.add_argument("--linear-command-topic", default=None)
    parser.add_argument("--angular-command-topic", default=None)
    parser.add_argument("--insertion-axis-topic", default=None)
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--plot-seconds", type=float, default=20.0)
    parser.add_argument("--force-unit", default="N")
    parser.add_argument(
        "--default-angle-deg",
        type=float,
        default=math.nan,
        help="Used until a teleoperation script publishes its target angle.",
    )
    return parser.parse_args()


def vector3_to_array(message):
    return np.array([message.x, message.y, message.z], dtype=float)


class ForceRecorder:
    def __init__(self, args):
        self.args = args
        prefix = "/{}".format(args.robot_name)
        published_topics = rospy.get_published_topics()
        discovered_force_topic = select_force_topic(
            published_topics, args.robot_name
        )
        if args.force_topic:
            self.force_topic_candidates = [args.force_topic]
        else:
            self.force_topic_candidates = force_topic_candidates(args.robot_name)
            if (
                discovered_force_topic
                and discovered_force_topic not in self.force_topic_candidates
            ):
                self.force_topic_candidates.append(discovered_force_topic)
        self.force_topic = None

        discovered_wavelength_topic = select_wavelength_topic(
            published_topics, args.robot_name
        )
        if args.wavelength_topic:
            self.wavelength_topic_candidates = [args.wavelength_topic]
        else:
            self.wavelength_topic_candidates = wavelength_topic_candidates(
                args.robot_name
            )
            if (
                discovered_wavelength_topic
                and discovered_wavelength_topic
                not in self.wavelength_topic_candidates
            ):
                self.wavelength_topic_candidates.append(discovered_wavelength_topic)
        self.wavelength_topic = None
        self.pose_topic = args.pose_topic or prefix + "/eye_robot/FrameEE"
        self.linear_command_topic = (
            args.linear_command_topic
            or prefix + "/eyerobot2/desiredTipVelocities"
        )
        self.angular_command_topic = (
            args.angular_command_topic
            or prefix + "/eyerobot2/desiredTipVelocitiesAngular"
        )
        self.insertion_axis_topic = (
            args.insertion_axis_topic
            or "/ati/force_collection/insertion_axis"
        )

        self.lock = threading.RLock()
        self.latest_pose = None
        self.latest_pose_time = math.nan
        self.latest_linear_command = np.zeros(3)
        self.latest_angular_command = np.zeros(3)
        self.latest_insertion_axis = np.array([0.0, 0.0, -1.0])
        self.target_angle_deg = float(args.default_angle_deg)
        self.operator_action = "unknown"
        self.filtered_force = None
        self.latest_force_raw = None
        self.last_force_monotonic = None
        self.latest_wavelength = None
        self.latest_wavelength_time = math.nan
        self.latest_wavelength_length = 0
        self.last_wavelength_monotonic = None
        self.baseline = np.zeros(FORCE_CHANNEL_COUNT)
        self.baseline_ready = False
        self.baseline_source = "not_set"

        self.plot_buffer = deque(maxlen=100000)
        self.wavelength_buffer = deque(maxlen=100000)
        self.recording = False
        self.samples = []
        self.record_start_ros_time = None
        self.record_start_wall_time = None
        self.record_start_position = None
        self.record_insertion_axis = None
        self.session_dir = None
        self.notes = ""
        self.last_saved_session = None

        self.force_subscribers = []
        for force_topic in self.force_topic_candidates:
            self.force_subscribers.append(
                rospy.Subscriber(
                    force_topic,
                    Float64MultiArray,
                    self._force_callback,
                    callback_args=force_topic,
                    queue_size=1000,
                    tcp_nodelay=True,
                )
            )
        self.wavelength_subscribers = []
        for wavelength_topic in self.wavelength_topic_candidates:
            self.wavelength_subscribers.append(
                rospy.Subscriber(
                    wavelength_topic,
                    Float64MultiArray,
                    self._wavelength_callback,
                    callback_args=wavelength_topic,
                    queue_size=1000,
                    tcp_nodelay=True,
                )
            )
        rospy.Subscriber(
            self.pose_topic,
            Transform,
            self._pose_callback,
            queue_size=100,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.linear_command_topic,
            Vector3,
            self._linear_command_callback,
            queue_size=100,
        )
        rospy.Subscriber(
            self.angular_command_topic,
            Vector3,
            self._angular_command_callback,
            queue_size=100,
        )
        rospy.Subscriber(
            "/ati/force_collection/target_angle_deg",
            Float64,
            self._angle_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.insertion_axis_topic,
            Vector3,
            self._insertion_axis_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            "/ati/force_collection/action",
            String,
            self._action_callback,
            queue_size=100,
        )

    def _pose_callback(self, message):
        now = rospy.Time.now().to_sec()
        position = np.array(
            [
                message.translation.x,
                message.translation.y,
                message.translation.z,
            ],
            dtype=float,
        )
        quaternion = np.array(
            [
                message.rotation.x,
                message.rotation.y,
                message.rotation.z,
                message.rotation.w,
            ],
            dtype=float,
        )
        with self.lock:
            self.latest_pose = (position, quaternion)
            self.latest_pose_time = now

    def _linear_command_callback(self, message):
        with self.lock:
            self.latest_linear_command = vector3_to_array(message)

    def _angular_command_callback(self, message):
        with self.lock:
            self.latest_angular_command = vector3_to_array(message)

    def _angle_callback(self, message):
        with self.lock:
            self.target_angle_deg = float(message.data)

    def _insertion_axis_callback(self, message):
        axis = vector3_to_array(message)
        axis_norm = float(np.linalg.norm(axis))
        if not np.isfinite(axis_norm) or axis_norm == 0.0:
            rospy.logwarn("Ignoring invalid insertion axis: %s", axis)
            return
        with self.lock:
            self.latest_insertion_axis = axis / axis_norm

    def _action_callback(self, message):
        with self.lock:
            self.operator_action = str(message.data)

    @staticmethod
    def _topic_priority(topic, candidates):
        try:
            return candidates.index(topic)
        except ValueError:
            return len(candidates)

    def _should_use_topic(self, source_topic, current_topic, candidates, last_time):
        if current_topic is None or source_topic == current_topic:
            return True
        if self._topic_priority(source_topic, candidates) < self._topic_priority(
            current_topic, candidates
        ):
            return True
        if last_time is not None and time.monotonic() - last_time > 1.0:
            return True
        return False

    @staticmethod
    def _signal_score(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        return float(np.max(np.abs(values)))

    def _force_callback(self, message, source_topic):
        now = rospy.Time.now().to_sec()
        raw = pad_force(message.data)

        with self.lock:
            source_score = self._signal_score(raw)
            current_score = self._signal_score(self.latest_force_raw)
            if (
                source_topic != self.force_topic
                and current_score > 1e-12
                and source_score <= 1e-12
            ):
                return
            force_signal_detected = (
                source_score > max(current_score, 1e-12)
                and current_score <= 1e-12
            )
            if force_signal_detected or self._should_use_topic(
                source_topic,
                self.force_topic,
                self.force_topic_candidates,
                self.last_force_monotonic,
            ):
                if self.force_topic != source_topic:
                    self.filtered_force = None
                    self.plot_buffer.clear()
                    self.force_topic = source_topic
                    rospy.loginfo("Using force topic: %s", source_topic)
            else:
                return

            if self.force_topic is None:
                self.force_topic = source_topic
                rospy.loginfo("Using force topic: %s", source_topic)
            self.last_force_monotonic = time.monotonic()
            self.latest_force_raw = raw.copy()
            self.filtered_force = ema_update(
                self.filtered_force, raw, self.args.ema_alpha
            )
            filtered = self.filtered_force.copy()
            baseline = self.baseline.copy()
            self.plot_buffer.append((now, raw.copy(), filtered.copy()))

            if not self.recording:
                return

            row = self._make_sample(
                now=now,
                raw=raw,
                filtered=filtered,
                baseline=baseline,
                force_message_length=len(message.data),
            )
            self.samples.append(row)

    def _wavelength_callback(self, message, source_topic):
        now = rospy.Time.now().to_sec()
        raw = pad_force(message.data)
        with self.lock:
            if self._should_use_topic(
                source_topic,
                self.wavelength_topic,
                self.wavelength_topic_candidates,
                self.last_wavelength_monotonic,
            ):
                if self.wavelength_topic != source_topic:
                    self.wavelength_buffer.clear()
                    self.wavelength_topic = source_topic
                    rospy.loginfo("Using wavelength topic: %s", source_topic)
            else:
                return

            self.latest_wavelength = raw.copy()
            self.latest_wavelength_time = now
            self.latest_wavelength_length = len(message.data)
            self.last_wavelength_monotonic = time.monotonic()
            self.wavelength_buffer.append((now, raw.copy()))

    def _make_sample(
        self, now, raw, filtered, baseline, force_message_length
    ):
        if self.latest_pose is None:
            position = np.full(3, np.nan)
            quaternion = np.full(4, np.nan)
            euler = np.full(3, np.nan)
            depth = math.nan
            lateral = math.nan
        else:
            position = self.latest_pose[0].copy()
            quaternion = self.latest_pose[1].copy()
            try:
                euler = R.from_quat(quaternion).as_euler("xyz", degrees=True)
            except ValueError:
                euler = np.full(3, np.nan)
            if (
                self.record_start_position is not None
                and self.record_insertion_axis is not None
            ):
                depth, lateral = insertion_metrics(
                    position,
                    self.record_start_position,
                    self.record_insertion_axis,
                )
            else:
                depth = math.nan
                lateral = math.nan

        force_delta = filtered - baseline
        pose_age = (
            now - self.latest_pose_time
            if math.isfinite(self.latest_pose_time)
            else math.nan
        )
        linear = self.latest_linear_command.copy()
        angular = self.latest_angular_command.copy()
        if self.record_insertion_axis is None:
            insertion_axis = self.latest_insertion_axis.copy()
        else:
            insertion_axis = self.record_insertion_axis.copy()
        if self.latest_wavelength is None:
            wavelength = np.full(FORCE_CHANNEL_COUNT, np.nan)
            wavelength_time = math.nan
            wavelength_age = math.nan
            wavelength_length = 0
        else:
            wavelength = self.latest_wavelength.copy()
            wavelength_time = self.latest_wavelength_time
            wavelength_age = now - wavelength_time
            wavelength_length = self.latest_wavelength_length

        row = {
            "ros_time_s": now,
            "elapsed_s": now - self.record_start_ros_time,
            "force_message_length": int(force_message_length),
            "wavelength_message_length": int(wavelength_length),
            "wavelength_ros_time_s": wavelength_time,
            "wavelength_age_s": wavelength_age,
            "pose_ros_time_s": self.latest_pose_time,
            "pose_age_s": pose_age,
            "position_x_mm": position[0],
            "position_y_mm": position[1],
            "position_z_mm": position[2],
            "quaternion_x": quaternion[0],
            "quaternion_y": quaternion[1],
            "quaternion_z": quaternion[2],
            "quaternion_w": quaternion[3],
            "roll_deg": euler[0],
            "pitch_deg": euler[1],
            "yaw_deg": euler[2],
            "insertion_depth_mm": depth,
            "lateral_displacement_mm": lateral,
            "insertion_axis_x": insertion_axis[0],
            "insertion_axis_y": insertion_axis[1],
            "insertion_axis_z": insertion_axis[2],
            "target_entry_angle_deg": self.target_angle_deg,
            "operator_action": self.operator_action,
            "command_linear_x_mm_s": linear[0],
            "command_linear_y_mm_s": linear[1],
            "command_linear_z_mm_s": linear[2],
            "command_angular_x_rad_s": angular[0],
            "command_angular_y_rad_s": angular[1],
            "command_angular_z_rad_s": angular[2],
        }
        for channel in range(FORCE_CHANNEL_COUNT):
            row["force_raw_{}".format(channel)] = raw[channel]
            row["force_filtered_{}".format(channel)] = filtered[channel]
            row["baseline_{}".format(channel)] = baseline[channel]
            row["force_delta_{}".format(channel)] = force_delta[channel]
            row["wavelength_raw_{}".format(channel)] = wavelength[channel]
        return row

    def set_baseline_from_recent(self, window_s=1.0, source="manual_tare"):
        with self.lock:
            if not self.plot_buffer:
                raise RuntimeError("No force messages have been received")
            newest = self.plot_buffer[-1][0]
            recent = [
                item[1]
                for item in self.plot_buffer
                if newest - item[0] <= window_s
            ]
            if not recent:
                raise RuntimeError("No recent force samples are available")
            stacked = np.vstack(recent)
            self.baseline = np.nanmedian(stacked, axis=0)
            self.baseline_ready = True
            self.baseline_source = source
            return self.baseline.copy(), len(recent)

    def start(self, notes=""):
        with self.lock:
            if self.recording:
                raise RuntimeError("A recording is already active")
            if self.latest_pose is None:
                raise RuntimeError(
                    "No robot pose received from {}".format(self.pose_topic)
                )
            if not self.plot_buffer:
                raise RuntimeError(
                    "No force data received from {}".format(
                        ", ".join(self.force_topic_candidates)
                    )
                )

            if not self.baseline_ready:
                self.set_baseline_from_recent(
                    window_s=1.0, source="automatic_at_start"
                )

            now_wall = datetime.now().astimezone()
            candidate = session_directory(
                self.args.data_dir, self.target_angle_deg, now=now_wall
            )
            suffix = 1
            while candidate.exists():
                candidate = Path(str(candidate) + "_{}".format(suffix))
                suffix += 1
            candidate.mkdir(parents=True)

            position, _ = self.latest_pose
            self.record_start_position = position.copy()
            self.record_insertion_axis = self.latest_insertion_axis.copy()
            self.record_start_ros_time = rospy.Time.now().to_sec()
            self.record_start_wall_time = now_wall
            self.session_dir = candidate
            self.samples = []
            self.notes = notes
            self.recording = True
            self.operator_action = "recording_started"
            return candidate

    def finish(self):
        with self.lock:
            if not self.recording:
                raise RuntimeError("No recording is active")
            self.recording = False
            self.operator_action = "recording_finished"
            samples = list(self.samples)
            session_dir = self.session_dir
            metadata = self._metadata(samples)

        write_csv(
            session_dir / "force_samples.csv",
            samples,
            fieldnames=SAMPLE_FIELDS,
        )
        summary_rows = self._summary_rows(samples)
        write_csv(
            session_dir / "summary.csv",
            summary_rows,
            fieldnames=list(summary_rows[0].keys()),
        )
        write_json(session_dir / "metadata.json", metadata)
        chart_error = self._save_charts(session_dir, samples)

        with self.lock:
            self.last_saved_session = session_dir
            self.samples = []
            self.session_dir = None
            self.baseline_ready = False
        return session_dir, len(samples), chart_error

    def _metadata(self, samples):
        end_wall = datetime.now().astimezone()
        duration = (
            samples[-1]["elapsed_s"] if samples else 0.0
        )
        return {
            "schema_version": 1,
            "created_at": self.record_start_wall_time.isoformat(),
            "finished_at": end_wall.isoformat(),
            "host": socket.gethostname(),
            "robot_name": self.args.robot_name,
            "notes": self.notes,
            "sample_count": len(samples),
            "duration_s": safe_json_number(duration),
            "force_unit": self.args.force_unit,
            "force_topic": self.force_topic,
            "force_topic_candidates": self.force_topic_candidates,
            "wavelength_topic": self.wavelength_topic,
            "wavelength_topic_candidates": self.wavelength_topic_candidates,
            "pose_topic": self.pose_topic,
            "linear_command_topic": self.linear_command_topic,
            "angular_command_topic": self.angular_command_topic,
            "insertion_axis_topic": self.insertion_axis_topic,
            "force_timestamp_semantics": (
                "ROS receipt time; Float64MultiArray has no message header"
            ),
            "pose_synchronization_semantics": (
                "latest pose snapshot at each force callback; inspect pose_age_s"
            ),
            "ema_alpha": self.args.ema_alpha,
            "baseline_method": (
                "per-channel median of approximately 1 second of recent raw force"
            ),
            "baseline_source": self.baseline_source,
            "baseline": [
                safe_json_number(value) for value in self.baseline
            ],
            "target_entry_angle_deg": safe_json_number(
                self.target_angle_deg
            ),
            "record_start_position_mm": [
                safe_json_number(value)
                for value in self.record_start_position
            ],
            "record_insertion_axis_base_frame": [
                safe_json_number(value)
                for value in self.record_insertion_axis
            ],
            "columns": SAMPLE_FIELDS,
        }

    def _summary_rows(self, samples):
        rows = []
        duration = samples[-1]["elapsed_s"] if samples else 0.0
        for channel in range(FORCE_CHANNEL_COUNT):
            raw_stats = finite_stats(
                [row["force_raw_{}".format(channel)] for row in samples]
            )
            filtered_stats = finite_stats(
                [
                    row["force_filtered_{}".format(channel)]
                    for row in samples
                ]
            )
            delta_stats = finite_stats(
                [row["force_delta_{}".format(channel)] for row in samples]
            )
            rows.append(
                {
                    "channel": channel,
                    "force_unit": self.args.force_unit,
                    "sample_count": len(samples),
                    "duration_s": duration,
                    "baseline": self.baseline[channel],
                    "raw_min": raw_stats["minimum"],
                    "raw_max": raw_stats["maximum"],
                    "raw_mean": raw_stats["mean"],
                    "raw_std": raw_stats["std"],
                    "filtered_min": filtered_stats["minimum"],
                    "filtered_max": filtered_stats["maximum"],
                    "filtered_mean": filtered_stats["mean"],
                    "filtered_std": filtered_stats["std"],
                    "delta_min": delta_stats["minimum"],
                    "delta_max": delta_stats["maximum"],
                    "delta_peak_to_peak": delta_stats["peak_to_peak"],
                }
            )
        return rows

    def _save_charts(self, session_dir, samples):
        if not samples:
            return "No samples were recorded, so charts were not created."
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as error:
            return "Could not import matplotlib: {}".format(error)

        try:
            elapsed = np.array([row["elapsed_s"] for row in samples])
            depth = np.array(
                [row["insertion_depth_mm"] for row in samples]
            )
            filtered = np.column_stack(
                [
                    [row["force_filtered_{}".format(c)] for row in samples]
                    for c in range(FORCE_CHANNEL_COUNT)
                ]
            )
            delta = np.column_stack(
                [
                    [row["force_delta_{}".format(c)] for row in samples]
                    for c in range(FORCE_CHANNEL_COUNT)
                ]
            )

            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            for channel in range(FORCE_CHANNEL_COUNT):
                axes[0].plot(
                    elapsed,
                    filtered[:, channel],
                    linewidth=1.0,
                    label="channel {}".format(channel),
                )
            axes[0].set_ylabel("Filtered force ({})".format(self.args.force_unit))
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="best", ncol=2)
            axes[1].plot(elapsed, depth, color="black", linewidth=1.2)
            axes[1].set_xlabel("Elapsed time (s)")
            axes[1].set_ylabel("Insertion depth (mm)")
            axes[1].grid(True, alpha=0.25)
            fig.suptitle(
                "Force collection, target angle {} deg".format(
                    self.target_angle_deg
                )
            )
            fig.tight_layout()
            fig.savefig(session_dir / "force_and_depth_vs_time.png", dpi=180)
            plt.close(fig)

            fig, axis = plt.subplots(figsize=(9, 6))
            for channel in range(FORCE_CHANNEL_COUNT):
                axis.plot(
                    depth,
                    delta[:, channel],
                    linewidth=1.0,
                    label="channel {}".format(channel),
                )
            axis.set_xlabel("Insertion depth (mm)")
            axis.set_ylabel(
                "Baseline-subtracted filtered force ({})".format(
                    self.args.force_unit
                )
            )
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", ncol=2)
            fig.tight_layout()
            fig.savefig(session_dir / "force_vs_insertion_depth.png", dpi=180)
            plt.close(fig)
            return None
        except Exception as error:
            return "Chart generation failed: {}".format(error)

    def plot_snapshot(self):
        with self.lock:
            items = list(self.plot_buffer)
            wavelength_items = list(self.wavelength_buffer)
            baseline = self.baseline.copy()
            recording = self.recording
            sample_count = len(self.samples)
            angle = self.target_angle_deg
            pose_ready = self.latest_pose is not None
            force_topic = self.force_topic
            wavelength_topic = self.wavelength_topic
            force_age_s = (
                time.monotonic() - self.last_force_monotonic
                if self.last_force_monotonic is not None
                else math.nan
            )
            wavelength_age_s = (
                time.monotonic() - self.last_wavelength_monotonic
                if self.last_wavelength_monotonic is not None
                else math.nan
            )
        return (
            items,
            wavelength_items,
            baseline,
            recording,
            sample_count,
            angle,
            pose_ready,
            force_topic,
            force_age_s,
            wavelength_topic,
            wavelength_age_s,
        )


class RecorderWindow(QtWidgets.QWidget):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.setWindowTitle("ATI Force Data Collector")
        self.resize(1100, 760)

        self.channel_box = QtWidgets.QComboBox()
        for channel in range(FORCE_CHANNEL_COUNT):
            self.channel_box.addItem("Force channel {}".format(channel), channel)

        self.subtract_baseline = QtWidgets.QCheckBox(
            "Display baseline-subtracted force"
        )
        self.subtract_baseline.setChecked(True)
        self.tare_button = QtWidgets.QPushButton("Tare / Set Baseline [T]")
        self.start_button = QtWidgets.QPushButton("Start Collection [S]")
        self.finish_button = QtWidgets.QPushButton("Finish and Save [F]")
        self.finish_button.setEnabled(False)
        self.notes = QtWidgets.QLineEdit()
        self.notes.setPlaceholderText(
            "Optional notes: phantom, trial number, tissue condition..."
        )
        self.status = QtWidgets.QLabel("Waiting for force and pose topics...")
        self.topic_label = QtWidgets.QLabel(
            "Force candidates: {}\nWavelength candidates: {}\nPose: {}".format(
                ", ".join(recorder.force_topic_candidates),
                ", ".join(recorder.wavelength_topic_candidates),
                recorder.pose_topic,
            )
        )
        self.topic_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Recent time", units="s")
        self.plot.setLabel(
            "left", "Force", units=recorder.args.force_unit
        )
        self.plot.addLegend()
        self.raw_curve = self.plot.plot(
            pen=pg.mkPen("#6fa8dc", width=1), name="Raw"
        )
        self.filtered_curve = self.plot.plot(
            pen=pg.mkPen("#f6b26b", width=2), name="EMA filtered"
        )
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.channel_box)
        controls.addWidget(self.subtract_baseline)
        controls.addWidget(self.tare_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.finish_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.topic_label)
        layout.addLayout(controls)
        layout.addWidget(self.notes)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.status)

        self.tare_button.clicked.connect(self.tare)
        self.start_button.clicked.connect(self.start)
        self.finish_button.clicked.connect(self.finish)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(50)

    def show_error(self, title, error):
        QtWidgets.QMessageBox.critical(self, title, str(error))

    def tare(self):
        try:
            baseline, count = self.recorder.set_baseline_from_recent()
            self.status.setText(
                "Baseline from {} samples: {}".format(
                    count, np.array2string(baseline, precision=6)
                )
            )
        except Exception as error:
            self.show_error("Could not set baseline", error)

    def start(self):
        try:
            session_dir = self.recorder.start(notes=self.notes.text())
            self.start_button.setEnabled(False)
            self.finish_button.setEnabled(True)
            self.tare_button.setEnabled(False)
            self.status.setText("RECORDING: {}".format(session_dir))
        except Exception as error:
            self.show_error("Could not start recording", error)

    def finish(self):
        try:
            session_dir, count, chart_error = self.recorder.finish()
            self.start_button.setEnabled(True)
            self.finish_button.setEnabled(False)
            self.tare_button.setEnabled(True)
            message = "Saved {} force samples to {}".format(count, session_dir)
            if chart_error:
                message += "\n\n{}".format(chart_error)
            self.status.setText(message)
            QtWidgets.QMessageBox.information(self, "Force data saved", message)
        except Exception as error:
            self.show_error("Could not finish recording", error)

    def refresh(self):
        (
            items,
            wavelength_items,
            baseline,
            recording,
            sample_count,
            angle,
            pose_ready,
            force_topic,
            force_age_s,
            wavelength_topic,
            wavelength_age_s,
        ) = self.recorder.plot_snapshot()
        message_rate_hz = math.nan
        latest_raw = np.full(FORCE_CHANNEL_COUNT, np.nan)
        latest_filtered = np.full(FORCE_CHANNEL_COUNT, np.nan)
        recent_peak_to_peak = np.full(FORCE_CHANNEL_COUNT, np.nan)
        wavelength_rate_hz = math.nan
        latest_wavelength = np.full(FORCE_CHANNEL_COUNT, np.nan)
        wavelength_peak_to_peak = np.full(FORCE_CHANNEL_COUNT, np.nan)
        if items:
            channel = int(self.channel_box.currentData())
            newest = items[-1][0]
            cutoff = newest - self.recorder.args.plot_seconds
            visible = [item for item in items if item[0] >= cutoff]
            times = np.array([item[0] - newest for item in visible])
            raw = np.array([item[1][channel] for item in visible])
            filtered = np.array([item[2][channel] for item in visible])
            latest_raw = items[-1][1]
            latest_filtered = items[-1][2]
            recent_raw = np.vstack([item[1] for item in visible])
            for force_channel in range(FORCE_CHANNEL_COUNT):
                values = recent_raw[:, force_channel]
                values = values[np.isfinite(values)]
                if values.size:
                    recent_peak_to_peak[force_channel] = np.ptp(values)
            if len(visible) >= 2:
                duration = visible[-1][0] - visible[0][0]
                if duration > 0.0:
                    message_rate_hz = (len(visible) - 1) / duration
            if self.subtract_baseline.isChecked():
                raw = raw - baseline[channel]
                filtered = filtered - baseline[channel]
            self.raw_curve.setData(times, raw)
            self.filtered_curve.setData(times, filtered)

        if wavelength_items:
            newest_wavelength = wavelength_items[-1][0]
            wavelength_cutoff = newest_wavelength - self.recorder.args.plot_seconds
            visible_wavelengths = [
                item for item in wavelength_items if item[0] >= wavelength_cutoff
            ]
            latest_wavelength = wavelength_items[-1][1]
            wavelength_raw = np.vstack([item[1] for item in visible_wavelengths])
            for wavelength_channel in range(FORCE_CHANNEL_COUNT):
                values = wavelength_raw[:, wavelength_channel]
                values = values[np.isfinite(values)]
                if values.size:
                    wavelength_peak_to_peak[wavelength_channel] = np.ptp(values)
            if len(visible_wavelengths) >= 2:
                duration = visible_wavelengths[-1][0] - visible_wavelengths[0][0]
                if duration > 0.0:
                    wavelength_rate_hz = (len(visible_wavelengths) - 1) / duration

        delta = latest_filtered - baseline
        state = (
            "RECORDING | samples={}".format(sample_count)
            if recording
            else "READY"
        )
        force_state = "ready"
        if not items:
            force_state = "missing"
        elif not math.isfinite(force_age_s) or force_age_s > 1.0:
            force_state = "stale"
        if force_state != "ready" or not pose_ready:
            state = "WAITING"
        wavelength_state = "ready"
        if not wavelength_items:
            wavelength_state = "missing"
        elif not math.isfinite(wavelength_age_s) or wavelength_age_s > 1.0:
            wavelength_state = "stale"
        self.status.setText(
            "{} | angle={:.1f} deg | force={} | pose={} | rate={:.1f} Hz | "
            "age={:.3f} s\n"
            "source={}\n"
            "raw [0..3] = {}\n"
            "filtered-baseline [0..3] = {}\n"
            "recent raw peak-to-peak [0..3] = {}\n"
            "wavelength={} | rate={:.1f} Hz | age={:.3f} s | source={}\n"
            "wavelength raw [0..3] = {}\n"
            "wavelength peak-to-peak [0..3] = {}".format(
                state,
                angle,
                force_state,
                "ready" if pose_ready else "missing",
                message_rate_hz,
                force_age_s,
                force_topic or "waiting for first force message",
                np.array2string(latest_raw, precision=7),
                np.array2string(delta, precision=7),
                np.array2string(recent_peak_to_peak, precision=7),
                wavelength_state,
                wavelength_rate_hz,
                wavelength_age_s,
                wavelength_topic or "waiting for first wavelength message",
                np.array2string(latest_wavelength, precision=7),
                np.array2string(wavelength_peak_to_peak, precision=7),
            )
        )

    def keyPressEvent(self, event):
        if self.notes.hasFocus():
            super().keyPressEvent(event)
            return
        if event.key() == QtCore.Qt.Key_T:
            self.tare()
            return
        if event.key() == QtCore.Qt.Key_S and not self.recorder.recording:
            self.start()
            return
        if event.key() == QtCore.Qt.Key_F and self.recorder.recording:
            self.finish()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.recorder.recording:
            response = QtWidgets.QMessageBox.question(
                self,
                "Recording active",
                "Finish and save the active recording before closing?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            )
            if response == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            try:
                self.recorder.finish()
            except Exception as error:
                self.show_error("Could not save recording", error)
                event.ignore()
                return
        event.accept()


def main():
    args = parse_args()
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1]")

    rospy.init_node(
        "ati_force_data_recorder", anonymous=True, disable_signals=True
    )
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        sys.argv
    )
    recorder = ForceRecorder(args)
    window = RecorderWindow(recorder)
    window.show()

    exit_code = application.exec_()
    rospy.signal_shutdown("Force recorder UI closed")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
