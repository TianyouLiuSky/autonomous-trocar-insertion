import sys
import time
import json
from pathlib import Path

import numpy as np

MOTION_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "motion_script"
if str(MOTION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_SCRIPT_DIR))

from SHER_Controller import SHERController


HOME_DIR = Path(__file__).resolve().parent / "home_position"
HOME_PATH = HOME_DIR / "home_position.json"
MOVE_TIMEOUT_SEC = 45.0
POSITION_TOL_MM = 0.5
ORIENTATION_TOL_DEG = 0.5
MAX_MOVE_ATTEMPTS = 2
SETTLE_SEC = 2.0

TRANSLATION_RADIUS_MM = 12.0
Z_RADIUS_MM = 12.0
ROLL_OFFSETS_DEG = [-12.0, -6.0, 0.0, 6.0, 12.0]
PITCH_OFFSETS_DEG = [-9.0, -3.0, 3.0, 9.0]


def pose_error(current_pose, target_pose):
    """Return approximate position/orientation error in mm and deg."""
    current_pose = np.asarray(current_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
    ori_err = np.linalg.norm(target_pose[3:] - current_pose[3:])
    return pos_err, ori_err


def save_home_position(home_pose):
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "pose_mm_deg": [float(v) for v in home_pose],
        "description": "[x, y, z, roll, pitch, yaw] in mm and degrees",
        "created_by": Path(__file__).name,
        "created_unix_time": time.time(),
    }
    with open(HOME_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved home position -> {HOME_PATH}")


def generate_diverse_poses(base_pose):
    """
    Generate exactly 20 reachable hand-eye poses around the current pose.

    The grid uses 5 roll offsets x 4 pitch offsets. Neighboring orientations are
    spaced by 6 degrees, so every adjacent case clears the 5 degree diversity
    requirement while also adding moderate XYZ translation diversity.
    """
    x, y, z, r, p, _ = base_pose

    tr = TRANSLATION_RADIUS_MM
    zr = Z_RADIUS_MM
    x_offsets = np.linspace(-tr, tr, len(ROLL_OFFSETS_DEG))
    y_offsets = np.linspace(-tr, tr, len(PITCH_OFFSETS_DEG))

    poses = []
    for pitch_index, (dp, dy) in enumerate(zip(PITCH_OFFSETS_DEG, y_offsets)):
        roll_order = list(zip(ROLL_OFFSETS_DEG, x_offsets))
        if pitch_index % 2 == 1:
            roll_order.reverse()

        for roll_index, (dr, dx) in enumerate(roll_order):
            dz = -zr if (pitch_index + roll_index) % 2 == 0 else zr
            poses.append({
                "label": f"r{dr:+.0f}_p{dp:+.0f}",
                "target": [x + dx, y + dy, z + dz, r + dr, p + dp, 0.0],
            })

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
    print(f"Pose {index}/{total} is ready to capture.")
    print(f"Target: {[round(v, 3) for v in target]}")
    print(f"Actual: {[round(v, 3) for v in actual_pose]}")
    print(f"Settling for {SETTLE_SEC:.1f}s before capture...")
    time.sleep(SETTLE_SEC)
    input("Record this sample in the calibration GUI, then press Enter here to continue...")


def ask_accept_failed_pose(index, total, target, actual_pose, pos_err, ori_err):
    print("")
    print(f"Pose {index}/{total} did not fully reach the target.")
    print(f"Target: {[round(v, 3) for v in target]}")
    print(f"Actual: {[round(v, 3) for v in actual_pose]}")
    print(f"Residual: position={pos_err:.3f} mm, orientation={ori_err:.3f} deg")
    answer = input("Skip this pose? Press Enter to skip, or type RECORD to capture anyway: ")
    return answer.strip().upper() == "RECORD"


def summarize_sequence(targets):
    xyz = np.array([p["target"][:3] for p in targets])
    rpy = np.array([p["target"][3:] for p in targets])
    print("")
    print("Calibration pose sequence")
    print(f"  Poses: {len(targets)}")
    print(f"  XYZ span (mm): {np.ptp(xyz, axis=0).round(3).tolist()}")
    print(f"  RPY span (deg): {np.ptp(rpy, axis=0).round(3).tolist()}")
    print("  Capture flow: move -> settle -> record in GUI -> press Enter here")


def run_sequence(robot, targets):
    recorded = 0
    skipped = 0
    total = len(targets)

    for i, entry in enumerate(targets, start=1):
        target = entry["target"]
        print("")
        print("=" * 80)
        print(f"Moving to Pose {i}/{total} ({entry['label']}): {[round(v, 3) for v in target]}")
        print("=" * 80)

        success, actual_pose, pos_err, ori_err = move_with_retries(robot, target)

        if success:
            wait_for_gui_capture(i, total, target, actual_pose)
            recorded += 1
        elif ask_accept_failed_pose(i, total, target, actual_pose, pos_err, ori_err):
            wait_for_gui_capture(i, total, target, actual_pose)
            recorded += 1
        else:
            print("Skipped. Do not record this pose in the GUI.")
            skipped += 1

    return recorded, skipped


if __name__ == "__main__":
    robot = SHERController(robot_name="SHER20")

    start_pose = robot.get_current_pose()
    save_home_position(start_pose)
    targets = generate_diverse_poses(start_pose)
    summarize_sequence(targets)

    print("")
    print("MAKE SURE handeye_calibration.py IS RUNNING AND THE BOARD IS VISIBLE.")
    input("Press Enter to start moving through calibration poses...")

    recorded, skipped = run_sequence(robot, targets)

    print("")
    print(f"Sequence complete. Recorded={recorded}, skipped={skipped}.")
    print("Returning to the saved home position...")
    move_with_retries(robot, start_pose)
    print("Click 'Compute Calibration' in the GUI when you have enough accepted samples.")
