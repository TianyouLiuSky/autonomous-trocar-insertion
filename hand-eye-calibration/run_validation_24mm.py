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
MOVE_TIMEOUT_SEC = 45.0
POSITION_TOL_MM = 0.5
ORIENTATION_TOL_DEG = 0.5
MAX_MOVE_ATTEMPTS = 2
SETTLE_SEC = 2.0


def pose_error(current_pose, target_pose):
    current_pose = np.asarray(current_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
    ori_err = np.linalg.norm(target_pose[3:] - current_pose[3:])
    return pos_err, ori_err


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
            x + offset[0], y + offset[1], z + offset[2],
            r + offset[3], p + offset[4], y_yaw + offset[5]
        ]
        poses.append(new_pose)
    return poses


def move_with_retries(robot, target):
    current_pose = robot.get_current_pose()
    pos_err, ori_err = pose_error(current_pose, target)

    for attempt in range(1, MAX_MOVE_ATTEMPTS + 1):
        print(f"Attempt {attempt}/{MAX_MOVE_ATTEMPTS}")
        success = robot.no_rcm_move_to(
            target,
            position_tol=POSITION_TOL_MM,
            orientation_tol=ORIENTATION_TOL_DEG,
            timeout=MOVE_TIMEOUT_SEC,
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
    input("Record this sample in the validation collector, then press Enter here to continue...")


def ask_accept_failed_pose(index, total, target, actual_pose, pos_err, ori_err):
    print("")
    print(f"Validation pose {index}/{total} did not fully reach the target.")
    print(f"Target: {[round(v, 3) for v in target]}")
    print(f"Actual: {[round(v, 3) for v in actual_pose]}")
    print(f"Residual: position={pos_err:.3f} mm, orientation={ori_err:.3f} deg")
    answer = input("Skip this pose? Press Enter to skip, or type RECORD to capture anyway: ")
    return answer.strip().upper() == "RECORD"


def summarize_sequence(targets):
    xyz = np.array([pose[:3] for pose in targets])
    rpy = np.array([pose[3:] for pose in targets])
    print("")
    print("Validation pose sequence")
    print(f"  Poses: {len(targets)}")
    print(f"  XYZ span (mm): {np.ptp(xyz, axis=0).round(3).tolist()}")
    print(f"  RPY span (deg): {np.ptp(rpy, axis=0).round(3).tolist()}")
    print("  Capture flow: move -> settle -> record in collector -> press Enter here")

if __name__ == "__main__":
    robot = SHERController(robot_name='SHER20')
    home_pose = load_home_position()
    if home_pose is not None:
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

    start_pose = robot.get_current_pose()
    targets = generate_validation_poses(start_pose)
    summarize_sequence(targets)
    
    print(f"\n{'='*50}")
    print(f"Starting validation sequence for 27 poses...")
    print("MAKE SURE THE DATA COLLECTOR SCRIPT IS RUNNING!")
    print(f"{'='*50}\n")

    recorded = 0
    skipped = 0
    for i, target in enumerate(targets, start=1):
        print(f"\nMoving to Pose {i}/27: {target}")
        success, actual_pose, pos_err, ori_err = move_with_retries(robot, target)

        if success:
            wait_for_gui_capture(i, len(targets), target, actual_pose)
            recorded += 1
        elif ask_accept_failed_pose(i, len(targets), target, actual_pose, pos_err, ori_err):
            wait_for_gui_capture(i, len(targets), target, actual_pose)
            recorded += 1
        else:
            print("Skipped. Do not record this validation pose.")
            skipped += 1

    print(f"\nSequence complete. Recorded={recorded}, skipped={skipped}.")
    print("The data collector should save automatically once it has enough samples.")
