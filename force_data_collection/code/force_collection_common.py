#!/usr/bin/env python3
"""Shared math and file helpers for force-data collection."""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


FORCE_CHANNEL_COUNT = 4


def force_topic_candidates(robot_name):
    """Return force-topic aliases used by the current EyeRobot codebase."""
    return [
        "/eye_robot/FBGForcesTip",
        "/{}/eye_robot/FBGForcesTip".format(robot_name),
        "/eye_robot2/ScleraForces",
        "/{}/eye_robot2/ScleraForces".format(robot_name),
        "/eye_robot2/HandleForces",
        "/{}/eye_robot2/HandleForces".format(robot_name),
        "/CNN/HandleForce",
    ]


def wavelength_topic_candidates(robot_name):
    """Return raw-wavelength aliases used by the current EyeRobot codebase."""
    return [
        "/eye_robot/WavelengthsRaw",
        "/{}/eye_robot/WavelengthsRaw".format(robot_name),
    ]


def _published_topic_names(published_topics):
    published = set()
    for item in published_topics or []:
        if isinstance(item, (tuple, list)):
            published.add(str(item[0]))
        else:
            published.add(str(item))
    return published


def select_force_topic(published_topics, robot_name):
    """Select the best currently published FBG force topic, if one exists."""
    published = _published_topic_names(published_topics)
    candidates = force_topic_candidates(robot_name)
    for candidate in candidates:
        if candidate in published:
            return candidate

    suffixes = (
        "/eye_robot/FBGForcesTip",
        "/eye_robot2/ScleraForces",
        "/eye_robot2/HandleForces",
        "/CNN/HandleForce",
    )
    suffix_matches = sorted(
        topic for topic in published if any(topic.endswith(suffix) for suffix in suffixes)
    )
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def select_wavelength_topic(published_topics, robot_name):
    """Select the best currently published raw-wavelength topic, if one exists."""
    published = _published_topic_names(published_topics)
    candidates = wavelength_topic_candidates(robot_name)
    for candidate in candidates:
        if candidate in published:
            return candidate

    suffix_matches = sorted(
        topic for topic in published if topic.endswith("/eye_robot/WavelengthsRaw")
    )
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def locked_target_rotation(
    straight_rpy_deg,
    entry_angle_deg,
    tilt_axis="local-y",
    tilt_sign=1.0,
):
    """Build the fixed target orientation from the absolute straight pose."""
    if tilt_axis not in ("local-x", "local-y"):
        raise ValueError("tilt_axis must be local-x or local-y")
    reference = R.from_euler("xyz", straight_rpy_deg, degrees=True)
    local_axis = np.zeros(3)
    local_axis[0 if tilt_axis == "local-x" else 1] = float(tilt_sign)
    delta = R.from_rotvec(np.deg2rad(float(entry_angle_deg)) * local_axis)
    return reference * delta


def clip_norm(vector, max_norm):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or norm <= max_norm:
        return vector
    if max_norm <= 0.0:
        return np.zeros_like(vector)
    return vector * (float(max_norm) / norm)


def teleop_velocity(
    active_keys,
    speed,
):
    """Return a normalized robot-base velocity for game-style teleoperation."""
    velocity = np.zeros(3, dtype=float)
    active_keys = set(active_keys)

    if "w" in active_keys:
        velocity[0] += 1.0
    if "s" in active_keys:
        velocity[0] -= 1.0
    if "a" in active_keys:
        velocity[1] += 1.0
    if "d" in active_keys:
        velocity[1] -= 1.0
    if "c" in active_keys:
        velocity[2] -= 1.0
    if "v" in active_keys:
        velocity[2] += 1.0

    norm = float(np.linalg.norm(velocity))
    if norm == 0.0:
        return velocity
    return velocity * (float(speed) / norm)


def pad_force(values, count=FORCE_CHANNEL_COUNT):
    result = np.full(count, np.nan, dtype=float)
    values = np.asarray(list(values), dtype=float).reshape(-1)
    copied = min(count, values.size)
    if copied:
        result[:copied] = values[:copied]
    return result


def ema_update(previous, current, alpha):
    current = np.asarray(current, dtype=float)
    if previous is None:
        return current.copy()
    previous = np.asarray(previous, dtype=float)
    valid_current = np.isfinite(current)
    valid_previous = np.isfinite(previous)
    result = previous.copy()
    result[valid_current & valid_previous] = (
        alpha * current[valid_current & valid_previous]
        + (1.0 - alpha) * previous[valid_current & valid_previous]
    )
    result[valid_current & ~valid_previous] = current[valid_current & ~valid_previous]
    return result


def insertion_metrics(position_mm, start_position_mm, insertion_axis):
    displacement = np.asarray(position_mm, dtype=float) - np.asarray(
        start_position_mm, dtype=float
    )
    axis = np.asarray(insertion_axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("insertion_axis must be non-zero")
    axis = axis / axis_norm
    depth = float(np.dot(displacement, axis))
    lateral = displacement - depth * axis
    return depth, float(np.linalg.norm(lateral))


def session_directory(base_dir, angle_deg, now=None):
    now = now or datetime.now()
    if angle_deg is None or not math.isfinite(float(angle_deg)):
        suffix = "angle_unknown"
    else:
        angle_label = (
            "{:+g}".format(float(angle_deg))
            .replace("+", "p")
            .replace("-", "m")
        )
        suffix = "angle_{}deg".format(angle_label)
    return Path(base_dir).expanduser() / "{}_{}".format(
        now.strftime("%Y%m%d_%H%M%S"), suffix
    )


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required when rows is empty")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    return path


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "minimum": math.nan,
            "maximum": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "peak_to_peak": math.nan,
        }
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "peak_to_peak": float(np.ptp(values)),
    }


def safe_json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None
