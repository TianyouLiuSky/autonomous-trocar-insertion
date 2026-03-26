import numpy as np
import rospy
from geometry_msgs.msg import Vector3, Transform
from scipy.spatial.transform import Rotation as R
import time

# use this to follow the numeric precision on eyerobot GUI
np.set_printoptions(suppress=True, precision=4)

class SHERController:
    """Simple controller for SHER robot with and without RCM constraint"""

    def __init__(self, robot_name):
        # Initialize ROS node
        rospy.init_node(f"{robot_name}_Controller", anonymous=True)
        linear_vel_pub = f'/{robot_name}/eyerobot2/desiredTipVelocities'
        angular_vel_pub = f'/{robot_name}/eyerobot2/desiredTipVelocitiesAngular'
        frame_ee_sub = f'/{robot_name}/eye_robot/FrameEE'


        rospy.on_shutdown(self._stop)

        # Publishers
        self.pub_linear = rospy.Publisher(linear_vel_pub,
                                          Vector3, queue_size=10)
        self.pub_angular = rospy.Publisher(angular_vel_pub,
                                           Vector3, queue_size=10)

        # Current pose (mm and quaternion)
        self.x = None
        self.y = None
        self.z = None
        self.qx = None
        self.qy = None
        self.qz = None
        self.qw = None

        # Subscriber
        rospy.Subscriber(frame_ee_sub, Transform, self._pose_callback)

        # Wait for first pose message
        time.sleep(0.5)

        # Control gains
        self.linear_gain = 1
        self.angular_gain = 0.8

    def _pose_callback(self, data):
        """Update current pose from ROS message"""
        self.x = data.translation.x
        self.y = data.translation.y
        self.z = data.translation.z
        self.qx = data.rotation.x
        self.qy = data.rotation.y
        self.qz = data.rotation.z
        self.qw = data.rotation.w

    def get_current_pose(self):
        """Returns current pose as [x, y, z, roll, pitch, yaw] in mm and degrees"""
        pos = np.array([self.x, self.y, self.z])
        quat = np.array([self.qx, self.qy, self.qz, self.qw])
        r = R.from_quat(quat)
        # XYZ order: returns [roll, pitch, yaw]
        euler = r.as_euler('xyz', degrees=True)
        return np.concatenate([pos, [euler[0], euler[1], euler[2]]])

    def no_rcm_move_to(self, target_pose, position_tol=0.005, orientation_tol=0.1,
                       timeout=30.0, max_linear_vel=5.0, max_angular_vel=5.0):
        """
        Move to target pose without RCM constraint using proportional control with velocity saturation

        Args:
            target_pose: [x, y, z, roll, pitch, yaw]
                        - position in mm
                        - orientation in DEGREES (XYZ euler angles)
                        - yaw typically 0 for 5-DOF robot
            position_tol: position tolerance in mm (default: 0.005)
            orientation_tol: orientation tolerance in DEGREES (default: 0.1)
            timeout: maximum time in seconds (default: 30.0)
            max_linear_vel: maximum linear velocity in mm/s (default: 1.0 for surgery)
            max_angular_vel: maximum angular velocity in rad/s (default: 0.05 for surgery)

        Returns:
            bool: True if reached target, False if timeout
        """
        # Extract target
        target_pos = np.array(target_pose[:3])  # [x, y, z] in mm
        target_rpy = np.array(target_pose[3:])  # [roll, pitch, yaw] in DEGREES

        if abs(target_rpy[-1] - 0) > 1e-5:
            print("Error, yaw should always be 0!")
            return False

        # Convert to quaternion (XYZ order to match get_current_pose)
        target_rot = R.from_euler('xyz', target_rpy, degrees=True)
        target_quat = target_rot.as_quat()

        start_time = time.time()
        rate = rospy.Rate(500)

        print(f"Moving to target: pos={target_pos}, rpy={target_rpy}")

        while not rospy.is_shutdown():
            if time.time() - start_time > timeout:
                print("Timeout reached!")
                self._stop()
                return False

            current_pos = np.array([self.x, self.y, self.z])
            current_quat = np.array([self.qx, self.qy, self.qz, self.qw])

            # Position error
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Orientation error
            current_rot = R.from_quat(current_quat)
            target_rot_obj = R.from_quat(target_quat)
            rot_error = target_rot_obj * current_rot.inv()
            axis_angle = rot_error.as_rotvec()  # This is in RADIANS
            angle_error = np.linalg.norm(axis_angle) * 180.0 / np.pi  # Convert to degrees for checking

            # Check if reached target
            if pos_error_norm < position_tol and angle_error < orientation_tol:
                print("Target reached!")
                print(f"Final position error: {pos_error_norm:.3f} mm")
                print(f"Final orientation error: {angle_error:.3f} deg")
                self._stop()
                return True

            # Proportional control with saturation
            # Linear velocity: proportional to position error
            linear_vel = pos_error * self.linear_gain
            linear_vel_norm = np.linalg.norm(linear_vel)
            if linear_vel_norm > max_linear_vel:
                print("Warning, reach linear limit")
                linear_vel = (linear_vel / linear_vel_norm) * max_linear_vel

            # Angular velocity: proportional to rotation error
            angular_vel = axis_angle * self.angular_gain  # rad/s
            angular_vel_norm = np.linalg.norm(angular_vel)
            if angular_vel_norm > max_angular_vel:
                print("Warning, reach angular limit")
                angular_vel = (angular_vel / angular_vel_norm) * max_angular_vel

            # Publish commands (linear in mm/s, angular in RAD/s)
            self.pub_linear.publish(linear_vel[0], linear_vel[1], linear_vel[2])
            self.pub_angular.publish(angular_vel[0], angular_vel[1], angular_vel[2])

            rate.sleep()

        return False

    def rcm_move_to(self, target_pos, rcm_point, rcm_axis_tol=1.0, position_tol=0.005,
                    orientation_tol=0.1, timeout=30.0, max_linear_vel=5.0, max_angular_vel=5.0):

        #TODO: Huge bug in this, do not run
        """
        Move to target position with RCM constraint. RCM point must be on current tool axis.
        In surgical robotics, the RCM point (incision/trocar) is fixed along the current
        insertion axis. The robot can only pivot around this point, not arbitrarily move in XYZ.

        Args:
            target_pos: Target position [x, y, z] in mm
            rcm_point: RCM pivot point [x, y, z] in mm - must be on current z-axis
            rcm_axis_tol: Tolerance for RCM point alignment with z-axis in mm (default: 1.0)
            position_tol: Position tolerance in mm (default: 0.005)
            orientation_tol: Orientation tolerance in DEGREES (default: 0.1)
            timeout: Maximum time in seconds (default: 30.0)
            max_linear_vel: Maximum linear velocity in mm/s (default: 5.0)
            max_angular_vel: Maximum angular velocity in rad/s (default: 5.0)

        Returns:
            bool: True if reached target, False if timeout or invalid RCM point
        """
        target_pos = np.array(target_pos)
        rcm_point = np.array(rcm_point)

        # Get current pose
        current_pos = np.array([self.x, self.y, self.z])
        current_quat = np.array([self.qx, self.qy, self.qz, self.qw])

        # Get current z-axis direction (tool insertion axis)
        r_current = R.from_quat(current_quat)
        rot_matrix = r_current.as_matrix()
        z_axis = rot_matrix[:, 2]  # Third column is z-axis in world frame

        # Verify RCM point is on current z-axis
        rcm_to_current = rcm_point - current_pos
        projection_length = np.dot(rcm_to_current, z_axis)
        projected_point = current_pos + projection_length * z_axis
        distance_to_axis = np.linalg.norm(rcm_point - projected_point)

        if distance_to_axis > rcm_axis_tol:
            print(f"ERROR: RCM point is not on current tool axis!")
            print(f"  Distance to axis: {distance_to_axis:.4f} mm (tolerance: {rcm_axis_tol} mm)")
            print(f"  Current position: {current_pos}")
            print(f"  Current z-axis: {z_axis}")
            print(f"  RCM point: {rcm_point}")
            print(f"  Projected RCM on axis: {projected_point}")
            return False

        print(f"RCM point validated on tool axis (distance: {distance_to_axis:.4f} mm)")

        # Calculate desired orientation from RCM constraint
        rcm_to_target = rcm_point - target_pos
        dx, dy, dz = rcm_to_target
        magnitude = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if magnitude < 1e-6:
            print("Error: Target position too close to RCM point!")
            return False

        # Calculate desired Euler angles (tool z-axis points toward RCM)
        theta_y = np.arctan2(dx, dz)
        theta_x = np.arcsin(-dy / magnitude)
        desired_euler = np.array([theta_x, theta_y, 0])

        r_desired = R.from_euler('xyz', desired_euler)
        target_quat = r_desired.as_quat()

        start_time = time.time()
        rate = rospy.Rate(500)

        print(f"Moving to target: {target_pos}")
        print(f"RCM point: {rcm_point}")

        while not rospy.is_shutdown():
            if time.time() - start_time > timeout:
                print("Timeout reached!")
                self._stop()
                return False

            current_pos = np.array([self.x, self.y, self.z])
            current_quat = np.array([self.qx, self.qy, self.qz, self.qw])

            # Position error
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Orientation error (same approach as no_rcm_move_to)
            current_rot = R.from_quat(current_quat)
            target_rot_obj = R.from_quat(target_quat)
            rot_error = target_rot_obj * current_rot.inv()
            axis_angle = rot_error.as_rotvec()  # This is in RADIANS
            angle_error = np.linalg.norm(axis_angle) * 180.0 / np.pi  # Convert to degrees for checking

            # Check if reached target
            if pos_error_norm < position_tol and angle_error < orientation_tol:
                print("Target reached!")
                print(f"Final position error: {pos_error_norm:.3f} mm")
                print(f"Final orientation error: {angle_error:.3f} deg")
                self._stop()
                return True

            # Proportional control with saturation (same as no_rcm_move_to)
            # Linear velocity: proportional to position error
            linear_vel = pos_error * self.linear_gain
            linear_vel_norm = np.linalg.norm(linear_vel)
            if linear_vel_norm > max_linear_vel:
                print("Warning, reach linear limit")
                linear_vel = (linear_vel / linear_vel_norm) * max_linear_vel

            # Angular velocity: proportional to rotation error
            angular_vel = axis_angle * self.angular_gain  # rad/s
            angular_vel_norm = np.linalg.norm(angular_vel)
            if angular_vel_norm > max_angular_vel:
                print("Warning, reach angular limit")
                angular_vel = (angular_vel / angular_vel_norm) * max_angular_vel

            # Publish commands
            self.pub_linear.publish(linear_vel[0], linear_vel[1], linear_vel[2])
            self.pub_angular.publish(angular_vel[0], angular_vel[1], angular_vel[2])

            rate.sleep()

        return False

    def _stop(self):
        """Stop all motion"""
        self.pub_linear.publish(0, 0, 0)
        self.pub_angular.publish(0, 0, 0)


# Example usage
if __name__ == "__main__":
    controller = SHERController(robot_name='SHER20')

    # Example 1: No RCM motion
    print("=" * 50)
    print("Example 1: No RCM motion")
    print("=" * 50)
    target = controller.get_current_pose()
    target[0] = -2.635
    target[1] = -129.700
    target[2] = 31.560
    target[3] = -9.8928
    target[4] = 1.6030
    target[5] = 0
    success = controller.no_rcm_move_to(target)
    if success:
        print("No-RCM motion completed successfully\n")
    else:
        print("No-RCM motion failed\n")

    # Example 2: RCM-constrained motion
    # print("=" * 50)
    # print("Example 3: Sweeping motion around RCM (Eye Surgery)")
    # print("=" * 50)
    #
    # # Get current position and orientation
    # current_pose = controller.get_current_pose()
    # current_pos = current_pose[:3]
    #
    # # RCM point for eye surgery (5-10mm is typical trocar depth)
    # current_quat = np.array([controller.qx, controller.qy, controller.qz, controller.qw])
    # r = R.from_quat(current_quat)
    # z_axis = r.as_matrix()[:, 2]
    # rcm_distance = 19  # mm - trocar entry point
    # rcm_point = current_pos - rcm_distance * z_axis
    #
    # print(f"Current position: {current_pos}")
    # print(f"RCM point: {rcm_point}")
    # print(f"Insertion depth: {rcm_distance:.3f} mm")
    #
    # # Target: Small 2mm movement respecting RCM constraint
    # # Move in a direction perpendicular to insertion axis
    # x_axis = r.as_matrix()[:, 0]  # Get x-axis from current orientation
    # target_offset = 2.0  # mm - small movement for eye surgery
    #
    # # Calculate target on the sphere around RCM
    # # Option 1: Simple lateral offset (perpendicular to insertion)
    # target_pos = current_pos + target_offset * x_axis
    #
    # # Option 2: Or rotate around RCM by small angle (more geometrically correct)
    # # pivot_angle_rad = np.arctan(target_offset / rcm_distance)  # Small angle
    # # rotation = R.from_rotvec(pivot_angle_rad * np.array([1, 0, 0]))
    # # rcm_to_current = current_pos - rcm_point
    # # rcm_to_target = rotation.apply(rcm_to_current)
    # # target_pos = rcm_point + rcm_to_target
    #
    # print(f"Target position: {target_pos}")
    # print(f"Movement distance: {np.linalg.norm(target_pos - current_pos):.3f} mm")
    #
    # # Use surgical-appropriate velocity limits
    # success = controller.rcm_move_to(
    #     target_pos,
    #     rcm_point
    # )
    #
    # if success:
    #     print("RCM motion completed successfully!")
    # else:
    #     print("RCM motion failed!")