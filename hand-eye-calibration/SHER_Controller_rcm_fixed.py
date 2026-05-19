"""
Drop-in RCM-safe variant of SHER_Controller.py.

Use this file by importing:

    from SHER_Controller_rcm_fixed import SHERController

The original rcm_move_to has two coupled hazards:
1. Its desired orientation uses rcm_point - target_pos, while the example in the
   same file defines the tool z-axis from RCM to tip. That can command a 180 deg
   flip even when the robot is already at the target.
2. It validates the RCM line only once, then publishes unconstrained XYZ linear
   velocity. During the move, that can drag the tool shaft away from the fixed
   trocar/RCM point.

This version commands a velocity pair that preserves the instantaneous RCM line:
the lateral tip velocity is paired with the angular velocity needed to pivot
around rcm_point, while axial motion is allowed for insertion/retraction.
"""

import time

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from SHER_Controller import SHERController as _BaseSHERController


_EPS = 1e-9


def _vec3(name, value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return arr


def _clip_norm(vec, max_norm):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if max_norm is None or norm <= max_norm or norm < _EPS:
        return vec
    if max_norm <= 0.0:
        return np.zeros_like(vec)
    return vec * (max_norm / norm)


def _angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < _EPS:
        return 180.0
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _distance_point_to_axis(point, axis_point, axis_dir):
    axis_dir = np.asarray(axis_dir, dtype=float)
    axis_norm = np.linalg.norm(axis_dir)
    if axis_norm < _EPS:
        return np.inf
    axis_unit = axis_dir / axis_norm
    offset = np.asarray(point, dtype=float) - np.asarray(axis_point, dtype=float)
    return float(np.linalg.norm(offset - np.dot(offset, axis_unit) * axis_unit))


def _fit_secondary_vector(primary, secondary, max_norm):
    """Keep primary intact and add as much of secondary as fits in max_norm."""
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)

    if max_norm is None:
        return secondary
    if max_norm <= 0.0:
        return np.zeros_like(secondary)
    if np.linalg.norm(primary + secondary) <= max_norm:
        return secondary
    if np.linalg.norm(primary) >= max_norm:
        return np.zeros_like(secondary)

    lo, hi = 0.0, 1.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(primary + mid * secondary) <= max_norm:
            lo = mid
        else:
            hi = mid
    return lo * secondary


class SHERController(_BaseSHERController):
    """SHERController with an RCM-constrained rcm_move_to override."""

    def rcm_move_to(
        self,
        target_pos,
        rcm_point,
        rcm_axis_tol=1.0,
        position_tol=0.005,
        orientation_tol=0.1,
        timeout=30.0,
        max_linear_vel=5.0,
        max_angular_vel=5.0,
        min_rcm_depth=1e-3,
    ):
        """
        Move the tip to target_pos while keeping the tool shaft through rcm_point.

        Args:
            target_pos: Target tip position [x, y, z] in mm.
            rcm_point: Fixed trocar/RCM point [x, y, z] in mm.
            rcm_axis_tol: Allowed distance from rcm_point to current tool axis in mm.
            position_tol: Target position tolerance in mm.
            orientation_tol: Final tool-axis tolerance in degrees.
            timeout: Maximum motion time in seconds.
            max_linear_vel: Tip velocity limit in mm/s.
            max_angular_vel: Angular velocity limit in rad/s.
            min_rcm_depth: Reject target/current positions too close to the RCM.

        Returns:
            bool: True if reached target, False if timeout or invalid geometry.
        """
        target_pos = _vec3("target_pos", target_pos)
        rcm_point = _vec3("rcm_point", rcm_point)

        current_pos = np.array([self.x, self.y, self.z], dtype=float)
        current_quat = np.array([self.qx, self.qy, self.qz, self.qw], dtype=float)
        current_rot = R.from_quat(current_quat)
        current_tool_z = current_rot.as_matrix()[:, 2]

        current_shaft = current_pos - rcm_point
        current_depth = np.linalg.norm(current_shaft)
        target_shaft = target_pos - rcm_point
        target_depth = np.linalg.norm(target_shaft)

        if current_depth < min_rcm_depth:
            print("ERROR: Current tip is too close to the RCM point.")
            return False
        if target_depth < min_rcm_depth:
            print("ERROR: Target position is too close to the RCM point.")
            return False

        initial_line_error = _distance_point_to_axis(
            rcm_point, current_pos, current_tool_z
        )
        if initial_line_error > rcm_axis_tol:
            print("ERROR: RCM point is not on current tool axis!")
            print(
                f"  Distance to axis: {initial_line_error:.4f} mm "
                f"(tolerance: {rcm_axis_tol} mm)"
            )
            print(f"  Current position: {current_pos}")
            print(f"  Current z-axis: {current_tool_z}")
            print(f"  RCM point: {rcm_point}")
            return False

        # Preserve the robot's observed z-axis convention. In the checked-in
        # example, current_pos - rcm_point is aligned with +z, but this keeps the
        # method valid if a setup uses the opposite sign.
        axis_sign = 1.0 if np.dot(current_tool_z, current_shaft) >= 0.0 else -1.0
        target_axis_from_rcm = target_shaft / target_depth
        target_tool_z = axis_sign * target_axis_from_rcm

        print("RCM point validated on tool axis")
        print(f"  Initial line error: {initial_line_error:.4f} mm")
        print(f"  Current depth from RCM: {current_depth:.4f} mm")
        print(f"  Target depth from RCM:  {target_depth:.4f} mm")
        print(f"Moving to RCM target: {target_pos}")
        print(f"RCM point: {rcm_point}")

        start_time = time.time()
        rate = rospy.Rate(500)

        while not rospy.is_shutdown():
            if time.time() - start_time > timeout:
                print("Timeout reached!")
                self._stop()
                return False

            current_pos = np.array([self.x, self.y, self.z], dtype=float)
            current_quat = np.array([self.qx, self.qy, self.qz, self.qw], dtype=float)
            current_rot = R.from_quat(current_quat)
            current_tool_z = current_rot.as_matrix()[:, 2]

            shaft = current_pos - rcm_point
            depth = np.linalg.norm(shaft)
            if depth < min_rcm_depth:
                print("ERROR: Tip moved too close to the RCM point.")
                self._stop()
                return False

            shaft_axis = shaft / depth
            desired_path_tool_z = axis_sign * shaft_axis
            line_error = _distance_point_to_axis(rcm_point, current_pos, current_tool_z)

            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            final_axis_error = _angle_deg(current_tool_z, target_tool_z)

            if (
                pos_error_norm < position_tol
                and final_axis_error < orientation_tol
                and line_error <= max(rcm_axis_tol, position_tol)
            ):
                print("RCM target reached!")
                print(f"Final position error: {pos_error_norm:.3f} mm")
                print(f"Final tool-axis error: {final_axis_error:.3f} deg")
                print(f"Final RCM line error: {line_error:.3f} mm")
                self._stop()
                return True

            desired_tip_vel = _clip_norm(
                pos_error * self.linear_gain, max_linear_vel
            )

            # Axial velocity slides through the trocar. Lateral velocity must be
            # paired with omega such that v_lateral = omega x (tip - RCM).
            axial_vel = np.dot(desired_tip_vel, shaft_axis) * shaft_axis
            lateral_vel = desired_tip_vel - axial_vel
            omega_orbit = np.cross(shaft, lateral_vel) / (depth * depth)

            orbit_norm = np.linalg.norm(omega_orbit)
            if orbit_norm > max_angular_vel:
                scale = 0.0 if max_angular_vel <= 0.0 else max_angular_vel / orbit_norm
                omega_orbit *= scale
                lateral_vel *= scale

            linear_vel = axial_vel + lateral_vel

            # Small correction that keeps the reported tool z-axis on the RCM
            # line. It is secondary to omega_orbit, which preserves the pivot.
            omega_align_raw = (
                np.cross(current_tool_z, desired_path_tool_z) * self.angular_gain
            )
            omega_align = _fit_secondary_vector(
                omega_orbit, omega_align_raw, max_angular_vel
            )
            angular_vel = omega_orbit + omega_align

            self.pub_linear.publish(
                float(linear_vel[0]), float(linear_vel[1]), float(linear_vel[2])
            )
            self.pub_angular.publish(
                float(angular_vel[0]), float(angular_vel[1]), float(angular_vel[2])
            )

            rate.sleep()

        return False
