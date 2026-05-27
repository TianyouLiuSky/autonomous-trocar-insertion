import csv
import datetime
import sys
import threading
import time
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

MOTION_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "motion_script"
if str(MOTION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_SCRIPT_DIR))

from SHER_Controller import SHERController


HOME_DIR = Path(__file__).resolve().parent / "home_position"
HOME_PATH = HOME_DIR / "home_position.json"
MOTION_LOG_DIR = Path(__file__).resolve().parent / "motion_logs"
MOTION_LOG_HZ = 10.0
MOVE_TIMEOUT_SEC = 90.0
POSITION_TOL_MM = 0.5
ORIENTATION_TOL_DEG = 0.5
MAX_MOVE_ATTEMPTS = 2
MAX_LINEAR_VEL_MM_S = 5.0
MAX_ANGULAR_VEL_RAD_S = 0.05
SETTLE_SEC = 2.0
ROTATION_SOFT_LIMIT_DEG = 55.0

TRANSLATION_RADIUS_MM = 12.0
Z_RADIUS_MM = 12.0
ROLL_OFFSETS_DEG = [-12.0, -6.0, 0.0, 6.0, 12.0]
PITCH_OFFSETS_DEG = [-9.0, -3.0, 3.0, 9.0]


def orientation_error_rotvec_deg(current_pose, target_pose):
    """Return target-to-current rotation error as an axis-angle vector in deg."""
    current_pose = np.asarray(current_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    current_rot = R.from_euler("xyz", current_pose[3:], degrees=True)
    target_rot = R.from_euler("xyz", target_pose[3:], degrees=True)
    return (target_rot * current_rot.inv()).as_rotvec() * 180.0 / np.pi


def pose_error(current_pose, target_pose):
    """Return position/orientation error in mm and deg."""
    current_pose = np.asarray(current_pose, dtype=float)
    target_pose = np.asarray(target_pose, dtype=float)
    pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
    ori_err = np.linalg.norm(orientation_error_rotvec_deg(current_pose, target_pose))
    return pos_err, ori_err


def pose_fields(prefix, pose):
    names = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
    if pose is None:
        return {f"{prefix}_{name}": "" for name in names}
    return {f"{prefix}_{name}": round(float(pose[i]), 6) for i, name in enumerate(names)}


def normalize_command_pose(pose):
    pose = np.asarray(pose, dtype=float).copy()
    pose[5] = 0.0
    return pose


class MotionLogger:
    def __init__(self, robot, log_dir=MOTION_LOG_DIR, sample_hz=MOTION_LOG_HZ):
        self.robot = robot
        self.log_dir = log_dir
        self.sample_period = 1.0 / sample_hz
        self.samples = []
        self.summary = []
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._active_meta = None
        self._stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_attempt(self, pose_index, total, label, attempt, target):
        self._stop_active_sampling()
        self._stop_event = threading.Event()
        self._active_meta = {
            "pose_index": pose_index,
            "total": total,
            "label": label,
            "attempt": attempt,
            "target": np.asarray(target, dtype=float),
            "start_time": time.time(),
        }
        self._thread = threading.Thread(target=self._sample_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop_attempt(self, status, actual_pose, pos_err, ori_err):
        meta = self._stop_active_sampling()
        if meta is None:
            return

        elapsed = time.time() - meta["start_time"]
        row = {
            "pose_index": meta["pose_index"],
            "total": meta["total"],
            "label": meta["label"],
            "attempt": meta["attempt"],
            "status": status,
            "elapsed_sec": round(elapsed, 4),
            "final_position_error_mm": round(float(pos_err), 6) if pos_err is not None else "",
            "final_orientation_error_deg": round(float(ori_err), 6) if ori_err is not None else "",
        }
        row.update(pose_fields("target", meta["target"]))
        row.update(pose_fields("actual", actual_pose))

        with self._lock:
            self.summary.append(row)

    def write(self):
        self._stop_active_sampling()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        sample_path = None
        summary_path = None
        if self.samples:
            sample_path = self.log_dir / f"calibration_motion_samples_{self._stamp}.csv"
            with open(sample_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.samples[0].keys()))
                writer.writeheader()
                writer.writerows(self.samples)

        if self.summary:
            summary_path = self.log_dir / f"calibration_motion_summary_{self._stamp}.csv"
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.summary[0].keys()))
                writer.writeheader()
                writer.writerows(self.summary)

        return sample_path, summary_path

    def _stop_active_sampling(self):
        if self._stop_event is None:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        meta = self._active_meta
        self._thread = None
        self._stop_event = None
        self._active_meta = None
        return meta

    def _sample_loop(self):
        meta = self._active_meta
        stop_event = self._stop_event
        target = meta["target"]

        while not stop_event.is_set():
            row = {
                "unix_time": round(time.time(), 6),
                "elapsed_sec": round(time.time() - meta["start_time"], 6),
                "pose_index": meta["pose_index"],
                "total": meta["total"],
                "label": meta["label"],
                "attempt": meta["attempt"],
                "position_error_mm": "",
                "orientation_error_deg": "",
                "rotvec_error_x_deg": "",
                "rotvec_error_y_deg": "",
                "rotvec_error_z_deg": "",
                "error": "",
            }
            row.update(pose_fields("target", target))

            try:
                current_pose = self.robot.get_current_pose()
                pos_err, ori_err = pose_error(current_pose, target)
                rotvec_err = orientation_error_rotvec_deg(current_pose, target)
                row.update(pose_fields("current", current_pose))
                row["position_error_mm"] = round(float(pos_err), 6)
                row["orientation_error_deg"] = round(float(ori_err), 6)
                row["rotvec_error_x_deg"] = round(float(rotvec_err[0]), 6)
                row["rotvec_error_y_deg"] = round(float(rotvec_err[1]), 6)
                row["rotvec_error_z_deg"] = round(float(rotvec_err[2]), 6)
            except Exception as exc:
                row.update(pose_fields("current", None))
                row["error"] = str(exc)

            with self._lock:
                self.samples.append(row)

            stop_event.wait(self.sample_period)


def save_home_position(home_pose, measured_pose=None):
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "pose_mm_deg": [float(v) for v in home_pose],
        "description": "[x, y, z, roll, pitch, yaw] in mm and degrees",
        "created_by": Path(__file__).name,
        "created_unix_time": time.time(),
    }
    if measured_pose is not None:
        data["measured_start_pose_mm_deg"] = [float(v) for v in measured_pose]
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


def move_with_retries(robot, target, logger=None, pose_index=None, total=None, label=""):
    current_pose = robot.get_current_pose()
    pos_err, ori_err = pose_error(current_pose, target)

    for attempt in range(1, MAX_MOVE_ATTEMPTS + 1):
        print(f"Attempt {attempt}/{MAX_MOVE_ATTEMPTS}")
        if logger is not None:
            logger.start_attempt(pose_index, total, label, attempt, target)

        success = False
        status = "error"
        try:
            success = robot.no_rcm_move_to(
                target,
                position_tol=POSITION_TOL_MM,
                orientation_tol=ORIENTATION_TOL_DEG,
                timeout=MOVE_TIMEOUT_SEC,
                max_linear_vel=MAX_LINEAR_VEL_MM_S,
                max_angular_vel=MAX_ANGULAR_VEL_RAD_S,
            )
            status = "controller_success" if success else "controller_timeout"
        finally:
            if logger is not None and status == "error":
                logger.stop_attempt(status, None, None, None)

        current_pose = None
        pos_err = None
        ori_err = None
        try:
            current_pose = robot.get_current_pose()
            pos_err, ori_err = pose_error(current_pose, target)
        except Exception:
            if logger is not None and status != "error":
                logger.stop_attempt("pose_read_error", None, None, None)
            raise
        print(f"Actual pose: {[round(v, 3) for v in current_pose]}")
        print(f"Residual: position={pos_err:.3f} mm, orientation={ori_err:.3f} deg")

        close_enough = pos_err <= POSITION_TOL_MM and ori_err <= ORIENTATION_TOL_DEG
        if logger is not None and status != "error":
            status = "accepted" if success or close_enough else status
            logger.stop_attempt(status, current_pose, pos_err, ori_err)

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
    print(f"  RPY min (deg): {np.min(rpy, axis=0).round(3).tolist()}")
    print(f"  RPY max (deg): {np.max(rpy, axis=0).round(3).tolist()}")
    max_abs_roll_pitch = np.max(np.abs(rpy[:, :2]), axis=0)
    if np.any(max_abs_roll_pitch > ROTATION_SOFT_LIMIT_DEG):
        print(
            "  WARNING: roll/pitch targets approach the paper's +/-60 deg "
            f"range. Max abs roll/pitch: {max_abs_roll_pitch.round(3).tolist()}"
        )
    print(f"  Timeout: {MOVE_TIMEOUT_SEC:.1f}s")
    print(f"  Max angular velocity: {MAX_ANGULAR_VEL_RAD_S:.3f} rad/s")
    print("  Capture flow: move -> settle -> record in GUI -> press Enter here")


def run_sequence(robot, targets, logger=None):
    recorded = 0
    skipped = 0
    total = len(targets)

    for i, entry in enumerate(targets, start=1):
        target = entry["target"]
        print("")
        print("=" * 80)
        print(f"Moving to Pose {i}/{total} ({entry['label']}): {[round(v, 3) for v in target]}")
        print("=" * 80)

        success, actual_pose, pos_err, ori_err = move_with_retries(
            robot,
            target,
            logger=logger,
            pose_index=i,
            total=total,
            label=entry["label"],
        )

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
    logger = MotionLogger(robot)

    measured_start_pose = robot.get_current_pose()
    start_pose = normalize_command_pose(measured_start_pose)
    if abs(measured_start_pose[5]) > 1e-5:
        print(f"Measured home yaw was {measured_start_pose[5]:.6f} deg; command home yaw saved as 0.0 deg.")
    save_home_position(start_pose, measured_pose=measured_start_pose)
    targets = generate_diverse_poses(start_pose)
    summarize_sequence(targets)

    print("")
    print("MAKE SURE handeye_calibration.py IS RUNNING AND THE BOARD IS VISIBLE.")
    print(f"Motion diagnostics will be written under: {MOTION_LOG_DIR}")
    input("Press Enter to start moving through calibration poses...")

    recorded = 0
    skipped = 0
    try:
        recorded, skipped = run_sequence(robot, targets, logger=logger)

        print("")
        print(f"Sequence complete. Recorded={recorded}, skipped={skipped}.")
        print("Returning to the saved home position...")
        move_with_retries(
            robot,
            start_pose,
            logger=logger,
            pose_index=len(targets) + 1,
            total=len(targets) + 1,
            label="return_home",
        )
        print("Click 'Compute Calibration' in the GUI when you have enough accepted samples.")
    finally:
        sample_path, summary_path = logger.write()
        if sample_path is not None:
            print(f"Motion samples log -> {sample_path}")
        if summary_path is not None:
            print(f"Motion summary log -> {summary_path}")
