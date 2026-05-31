import numpy as np
import sys
import time
import json
from pathlib import Path

MOTION_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "motion_script"
if str(MOTION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_SCRIPT_DIR))

from SHER_Controller import SHERController

HOME_PATH = Path(__file__).resolve().parent / "home_position" / "home_position.json"
MOVE_TIMEOUT_SEC = 90.0
# Validation records the measured robot pose, so the commanded target does not
# need sub-millimeter agreement.
POSITION_TOL_MM = 4.0
ORIENTATION_TOL_DEG = 1.0
MAX_MOVE_ATTEMPTS = 2
MAX_LINEAR_VEL_MM_S = 5.0
MAX_ANGULAR_VEL_RAD_S = 0.05
SETTLE_SEC = 2.0
ROLL_ABS_LIMIT_DEG = 28.0
QUIT_COMMANDS = {"Q", "QUIT", "q"}
RECORD_ANYWAY_COMMAND = {"R", "r"}

# Absolute robot workspace limits in FrameEE coordinates.
# x positive: inward/forward, y positive: left, z positive: up.
X_LIMITS_MM = (-42.0, 10.0)
Y_LIMITS_MM = (-133.0, -85.0)
Z_LIMITS_MM = (-13.0, 30.0)


def is_quit_command(answer):
    return answer.strip().upper() in QUIT_COMMANDS


def pose_error(current_pose, target_pose):
    current_pose = np.asarray(current_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
    ori_err = np.linalg.norm(target_pose[3:] - current_pose[3:])
    return pos_err, ori_err


def normalize_command_pose(pose):
    pose = np.asarray(pose, dtype=float).copy()
    pose[5] = 0.0
    return pose


def limits_text(limits):
    return "[{:.3f}, {:.3f}]".format(limits[0], limits[1])


def axis_within_limits(value, limits):
    low, high = limits
    return low - 1e-6 <= float(value) <= high + 1e-6


def roll_within_limit(roll_deg):
    return abs(float(roll_deg)) <= ROLL_ABS_LIMIT_DEG + 1e-6


def validate_target_pose(target, label="target"):
    target = np.asarray(target, dtype=float)
    axis_limits = [
        ("x", target[0], X_LIMITS_MM),
        ("y", target[1], Y_LIMITS_MM),
        ("z", target[2], Z_LIMITS_MM),
    ]
    for axis_name, value, limits in axis_limits:
        if not axis_within_limits(value, limits):
            raise ValueError(
                f"{label} {axis_name}={value:.3f} mm is outside the configured "
                f"{axis_name.upper()} limits {limits_text(limits)} mm"
            )
    if not roll_within_limit(target[3]):
        raise ValueError(
            f"{label} roll={target[3]:.3f} deg exceeds the absolute roll limit "
            f"+/-{ROLL_ABS_LIMIT_DEG:.1f} deg"
        )


def fit_offsets_to_limits(base_value, desired_offsets, limits, axis_name):
    offsets = np.asarray(desired_offsets, dtype=float).copy()
    low, high = limits
    available_span = high - low
    desired_span = float(np.max(offsets) - np.min(offsets))
    scale = 1.0

    if desired_span > available_span:
        offset_center = (np.max(offsets) + np.min(offsets)) / 2.0
        scale = available_span / desired_span
        offsets = (offsets - offset_center) * scale + offset_center

    target_low = base_value + np.min(offsets)
    target_high = base_value + np.max(offsets)
    shift = 0.0
    if target_low < low:
        shift = low - target_low
    elif target_high > high:
        shift = high - target_high
    offsets = offsets + shift

    target_low = base_value + np.min(offsets)
    target_high = base_value + np.max(offsets)
    if target_low < low - 1e-6 or target_high > high + 1e-6:
        raise ValueError(
            f"Cannot fit {axis_name.upper()} offsets inside limits "
            f"{limits_text(limits)} mm from base {base_value:.3f} mm"
        )

    if scale < 1.0:
        print(
            f"Compressed {axis_name.upper()} offset span from {desired_span:.3f} "
            f"to {desired_span * scale:.3f} mm to fit limits {limits_text(limits)} mm."
        )
    if abs(shift) > 1e-6:
        print(
            f"Shifted {axis_name.upper()} offsets by {shift:+.3f} mm so targets "
            f"fit limits {limits_text(limits)} mm."
        )

    return offsets


def load_home_position():
    if not HOME_PATH.exists():
        return None
    with open(HOME_PATH, "r") as f:
        data = json.load(f)
    return np.array(data["pose_mm_deg"], dtype=float)

def generate_validation_poses(base_pose):
    """
    Generates 27 targets mapping a 24mm^3 workspace (±12mm from center).
    Pairs translations with ±5 degree rotations to ensure diverse viewing angles.
    """
    poses = []
    x, y, z, r, p, y_yaw = base_pose
    x_values = fit_offsets_to_limits(x, np.array([-12.0, 0.0, 12.0]), X_LIMITS_MM, "x")
    y_values = fit_offsets_to_limits(y, np.array([-12.0, 0.0, 12.0]), Y_LIMITS_MM, "y")
    z_values = fit_offsets_to_limits(z, np.array([-12.0, 0.0, 12.0]), Z_LIMITS_MM, "z")
    x_map = {-12: x_values[0], 0: x_values[1], 12: x_values[2]}
    y_map = {-12: y_values[0], 0: y_values[1], 12: y_values[2]}
    z_map = {-12: z_values[0], 0: z_values[1], 12: z_values[2]}
    
    # Format: [X, Y, Z, Roll, Pitch, Yaw]
    offsets = [
        # Z = -12mm (Bottom Layer)
        [-12, -12, -12, -5, -5, 0], [ 0, -12, -12,  0, -5, 0], [ 12, -12, -12,  5, -5, 0],
        [-12,   0, -12, -5,  0, 0], [ 0,   0, -12,  0,  0, 0], [ 12,   0, -12,  5,  0, 0],
        [-12,  12, -12, -5,  5, 0], [ 0,  12, -12,  0,  5, 0], [ 12,  12, -12,  5,  5, 0],

        # Z = 0mm (Middle Layer)
        [-12, -12,   0, -5, -5, 0], [ 0, -12,   0,  0, -5, 0], [ 12, -12,   0,  5, -5, 0],
        [-12,   0,   0, -5,  0, 0], [ 0,   0,   0,  0,  0, 0], [ 12,   0,   0,  5,  0, 0],
        [-12,  12,   0, -5,  5, 0], [ 0,  12,   0,  0,  5, 0], [ 12,  12,   0,  5,  5, 0],

        # Z = +12mm (Top Layer)
        [-12, -12,  12, -5, -5, 0], [ 0, -12,  12,  0, -5, 0], [ 12, -12,  12,  5, -5, 0],
        [-12,   0,  12, -5,  0, 0], [ 0,   0,  12,  0,  0, 0], [ 12,   0,  12,  5,  0, 0],
        [-12,  12,  12, -5,  5, 0], [ 0,  12,  12,  0,  5, 0], [ 12,  12,  12,  5,  5, 0],
    ]
    
    for offset in offsets:
        new_pose = [
            x + x_map[offset[0]], y + y_map[offset[1]], z + z_map[offset[2]],
            r + offset[3], p + offset[4], y_yaw + offset[5]
        ]
        validate_target_pose(new_pose, label=f"generated validation pose {len(poses) + 1}")
        poses.append(new_pose)
    return poses


def move_with_retries(robot, target):
    validate_target_pose(target)
    current_pose = robot.get_current_pose()
    pos_err, ori_err = pose_error(current_pose, target)

    for attempt in range(1, MAX_MOVE_ATTEMPTS + 1):
        print(f"Attempt {attempt}/{MAX_MOVE_ATTEMPTS}")
        success = robot.no_rcm_move_to(
            target,
            position_tol=POSITION_TOL_MM,
            orientation_tol=ORIENTATION_TOL_DEG,
            timeout=MOVE_TIMEOUT_SEC,
            max_linear_vel=MAX_LINEAR_VEL_MM_S,
            max_angular_vel=MAX_ANGULAR_VEL_RAD_S,
            warn_on_angular_limit=False,
        )
        current_pose = robot.get_current_pose()
        pos_err, ori_err = pose_error(current_pose, target)
        print(f"Actual pose: {[round(v, 3) for v in current_pose]}")
        print(f"Residual: position={pos_err:.3f} mm, orientation={ori_err:.3f} deg")

        close_enough = pos_err <= POSITION_TOL_MM and ori_err <= ORIENTATION_TOL_DEG
        if success or close_enough:
            return True, current_pose, pos_err, ori_err

        if attempt < MAX_MOVE_ATTEMPTS:
            print("Target was not reached. Retrying from current pose...")

    return False, current_pose, pos_err, ori_err


def wait_for_gui_capture(index, total, target, actual_pose):
    print("")
    print(f"Validation pose {index}/{total} is ready to capture.")
    print(f"Target: {[round(v, 3) for v in target]}")
    print(f"Actual: {[round(v, 3) for v in actual_pose]}")
    print(f"Settling for {SETTLE_SEC:.1f}s before capture...")
    time.sleep(SETTLE_SEC)
    answer = input(
        "Record this sample in the validation collector, then press Enter here to continue, "
        "or type Q to stop after this sample: "
    )
    return is_quit_command(answer)


def ask_accept_failed_pose(index, total, target, actual_pose, pos_err, ori_err):
    print("")
    print(f"Validation pose {index}/{total} did not fully reach the target.")
    print(f"Target: {[round(v, 3) for v in target]}")
    print(f"Actual: {[round(v, 3) for v in actual_pose]}")
    print(f"Residual: position={pos_err:.3f} mm, orientation={ori_err:.3f} deg")
    answer = input("Press Enter to skip, type R to capture anyway, or type Q to stop: ")
    answer = answer.strip().upper()
    if answer in QUIT_COMMANDS:
        return "quit"
    if answer == RECORD_ANYWAY_COMMAND:
        return "record"
    return "skip"


def summarize_sequence(targets):
    xyz = np.array([pose[:3] for pose in targets])
    rpy = np.array([pose[3:] for pose in targets])
    print("")
    print("Validation pose sequence")
    print(f"  Poses: {len(targets)}")
    print(f"  XYZ span (mm): {np.ptp(xyz, axis=0).round(3).tolist()}")
    print(f"  XYZ min (mm): {np.min(xyz, axis=0).round(3).tolist()}")
    print(f"  XYZ max (mm): {np.max(xyz, axis=0).round(3).tolist()}")
    print(
        "  Workspace limits (mm): "
        f"x={limits_text(X_LIMITS_MM)}, "
        f"y={limits_text(Y_LIMITS_MM)}, "
        f"z={limits_text(Z_LIMITS_MM)}"
    )
    print(f"  RPY span (deg): {np.ptp(rpy, axis=0).round(3).tolist()}")
    print(f"  Absolute roll limit (deg): +/-{ROLL_ABS_LIMIT_DEG:.1f}")
    print(f"  Reach tolerance: {POSITION_TOL_MM:.1f} mm, {ORIENTATION_TOL_DEG:.1f} deg")
    print("  Capture flow: move -> settle -> record in collector -> press Enter here")

if __name__ == "__main__":
    robot = SHERController(robot_name='SHER20')
    home_pose = load_home_position()
    if home_pose is not None:
        if abs(home_pose[5]) > 1e-5:
            print(f"Saved home yaw was {home_pose[5]:.6f} deg; command yaw will be normalized to 0.0 deg.")
        home_pose = normalize_command_pose(home_pose)
        print("Saved calibration home position found:")
        print(f"  {[round(v, 3) for v in home_pose]}")
        input("Press Enter to move to this home position before validation...")
        home_reached, actual_home, pos_err, ori_err = move_with_retries(robot, home_pose)
        if not home_reached:
            print("Home position was not fully reached.")
            answer = input("Press Enter to stop, or type CONTINUE to generate validation poses from the actual pose: ")
            if answer.strip().upper() != "CONTINUE":
                raise SystemExit("Validation stopped before capture.")
    else:
        print(f"No saved home position found at: {HOME_PATH}")
        print("Run run_calibration_poses.py first, or manually start from the intended home position.")
        input("Press Enter to use the current robot pose as the validation center...")

    start_pose = normalize_command_pose(robot.get_current_pose())
    targets = generate_validation_poses(start_pose)
    summarize_sequence(targets)
    
    print(f"\n{'='*50}")
    print(f"Starting validation sequence for 27 poses...")
    print("MAKE SURE THE DATA COLLECTOR SCRIPT IS RUNNING!")
    print(f"{'='*50}\n")

    recorded = 0
    skipped = 0
    stopped = False
    for i, target in enumerate(targets, start=1):
        print(f"\nMoving to Pose {i}/27: {target}")
        answer = input("Press Enter to move to this pose, or type Q to stop the run: ")
        if is_quit_command(answer):
            print("Stopped before moving to this pose.")
            stopped = True
            break

        success, actual_pose, pos_err, ori_err = move_with_retries(robot, target)

        if success:
            stop_after_capture = wait_for_gui_capture(i, len(targets), target, actual_pose)
            recorded += 1
            if stop_after_capture:
                stopped = True
                break
        else:
            action = ask_accept_failed_pose(i, len(targets), target, actual_pose, pos_err, ori_err)
            if action == "record":
                stop_after_capture = wait_for_gui_capture(i, len(targets), target, actual_pose)
                recorded += 1
                if stop_after_capture:
                    stopped = True
                    break
            elif action == "quit":
                print("Stopped after failed pose.")
                stopped = True
                break
            else:
                print("Skipped. Do not record this validation pose.")
                skipped += 1

    status_text = "stopped" if stopped else "complete"
    print(f"\nSequence {status_text}. Recorded={recorded}, skipped={skipped}.")
    print("The data collector should save automatically once it has enough samples.")
