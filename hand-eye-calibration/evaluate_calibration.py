import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2
import argparse
import csv
import glob
import os
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def latest_file(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def resolve_path(path, fallback_patterns):
    if path:
        return path
    for pattern in fallback_patterns:
        resolved = latest_file(pattern)
        if resolved:
            return resolved
    return None


def as_float_or_blank(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return round(float(value), 6)


def evaluate_and_plot(
        calib_npz_path=None, validation_npz_path=None,
        output_dir=DEFAULT_OUTPUT_DIR, show_plot=True, solution="weighted"):
    os.makedirs(output_dir, exist_ok=True)
    calib_npz_path = resolve_path(
        calib_npz_path,
        [
            os.path.join(DEFAULT_OUTPUT_DIR, "hand_eye_cal_*.npz"),
            os.path.join(SCRIPT_DIR, "hand_eye_calibration.npz"),
        ],
    )
    validation_npz_path = resolve_path(
        validation_npz_path,
        [
            os.path.join(SCRIPT_DIR, "validation_dataset.npz"),
            os.path.join(DEFAULT_OUTPUT_DIR, "validation_dataset_*.npz"),
        ],
    )
    if calib_npz_path is None:
        print("Failed to find a calibration .npz. Pass --calib explicitly.")
        return None
    if validation_npz_path is None:
        print("Failed to find a validation .npz. Pass --validation explicitly.")
        return None

    # 1. Load the Calibration Matrices you want to test
    try:
        calib_data = np.load(calib_npz_path)
        if solution == "legacy":
            camera_key = "legacy_T_cam2base"
            board_key = "legacy_T_board2gripper"
        else:
            camera_key = "T_cam2base"
            board_key = "T_board2gripper"

        if camera_key not in calib_data or board_key not in calib_data:
            print(
                "Calibration file does not contain the {} solution. "
                "Use a new calibration saved by the weighted "
                "handeye_calibration.py.".format(solution)
            )
            return None
        T_cam2base = calib_data[camera_key]           # Matrix X
        T_board2gripper = calib_data[board_key]       # Matrix Y
        print(f"Loaded {solution} calibration: {calib_npz_path}")
    except Exception as e:
        print(f"Failed to load calibration file: {e}")
        return

    # 2. Load the Static Validation Ground Truth Data
    try:
        val_data = np.load(validation_npz_path, allow_pickle=True)
        robot_poses = val_data['robot_poses'] # Matrix A components
        board_rvecs = val_data['board_rvecs'] # Matrix B components
        board_tvecs = val_data['board_tvecs']
        corner_counts = val_data['corner_counts'] if 'corner_counts' in val_data else None
        reproj_errors = val_data['reprojection_errors_px'] if 'reprojection_errors_px' in val_data else None
        validation_mode = (
            str(val_data['validation_mode'].item())
            if 'validation_mode' in val_data
            else 'validation'
        )
        print(f"Loaded {len(robot_poses)} Ground Truth samples from: {validation_npz_path}")
        print(f"Validation mode: {validation_mode}")
    except Exception as e:
        print(f"Failed to load validation file: {e}")
        return

    coords, t_err_vecs, r_err_vecs = [], [], []
    robot_rpys = []
    t_mags, r_mags = [], []
    residual_rows = []

    # 3. Compute Residuals for AY = XB
    for i, (r_pose, b_rvec, b_tvec) in enumerate(zip(robot_poses, board_rvecs, board_tvecs), start=1):
        # Construct Matrix A (Robot Kinematics)
        T_A = np.eye(4)
        T_A[:3, :3] = Rotation.from_quat(r_pose['q']).as_matrix()
        T_A[:3, 3] = r_pose['t']

        # Construct Matrix B (Vision)
        T_B = np.eye(4)
        T_B[:3, :3], _ = cv2.Rodrigues(b_rvec)
        T_B[:3, 3] = b_tvec.flatten()

        # Where is the board in the base frame? 
        T_left = T_A @ T_board2gripper   # According to Kinematics (A * Y)
        T_right = T_cam2base @ T_B       # According to Vision (X * B)

        board_base_robot_mm = T_left[:3, 3] * 1000
        board_base_camera_mm = T_right[:3, 3] * 1000
        coords.append(board_base_robot_mm) # Save coordinates for plotting (in mm)

        # Translation Error
        t_err = board_base_camera_mm - board_base_robot_mm
        t_err_vecs.append(t_err)
        t_mags.append(np.linalg.norm(t_err))

        # Rotation Error
        R_err_mat = T_right[:3, :3] @ T_left[:3, :3].T
        rot_err_vec = Rotation.from_matrix(R_err_mat).as_rotvec() 
        r_mag_deg = np.linalg.norm(rot_err_vec) * (180.0 / np.pi) 
        
        r_err_vecs.append((rot_err_vec / (np.linalg.norm(rot_err_vec) + 1e-8)) * r_mag_deg) 
        r_mags.append(r_mag_deg)

        robot_rpy = Rotation.from_quat(r_pose['q']).as_euler('xyz', degrees=True)
        robot_rpys.append(robot_rpy)
        board_rot, _ = cv2.Rodrigues(b_rvec)
        board_rpy = Rotation.from_matrix(board_rot).as_euler('xyz', degrees=True)
        residual_rows.append({
            "sample": i,
            "translation_error_mm": round(float(np.linalg.norm(t_err)), 6),
            "rotation_error_deg": round(float(r_mag_deg), 6),
            "translation_error_x_mm": round(float(t_err[0]), 6),
            "translation_error_y_mm": round(float(t_err[1]), 6),
            "translation_error_z_mm": round(float(t_err[2]), 6),
            "board_robot_x_mm": round(float(board_base_robot_mm[0]), 6),
            "board_robot_y_mm": round(float(board_base_robot_mm[1]), 6),
            "board_robot_z_mm": round(float(board_base_robot_mm[2]), 6),
            "board_camera_x_mm": round(float(board_base_camera_mm[0]), 6),
            "board_camera_y_mm": round(float(board_base_camera_mm[1]), 6),
            "board_camera_z_mm": round(float(board_base_camera_mm[2]), 6),
            "robot_x_mm": round(float(r_pose['t'][0] * 1000), 6),
            "robot_y_mm": round(float(r_pose['t'][1] * 1000), 6),
            "robot_z_mm": round(float(r_pose['t'][2] * 1000), 6),
            "robot_roll_deg": round(float(robot_rpy[0]), 6),
            "robot_pitch_deg": round(float(robot_rpy[1]), 6),
            "robot_yaw_deg": round(float(robot_rpy[2]), 6),
            "board_cam_x_m": round(float(b_tvec.flatten()[0]), 9),
            "board_cam_y_m": round(float(b_tvec.flatten()[1]), 9),
            "board_cam_z_m": round(float(b_tvec.flatten()[2]), 9),
            "board_roll_deg": round(float(board_rpy[0]), 6),
            "board_pitch_deg": round(float(board_rpy[1]), 6),
            "board_yaw_deg": round(float(board_rpy[2]), 6),
            "corner_count": "" if corner_counts is None else int(corner_counts[i - 1]),
            "reprojection_error_px": "" if reproj_errors is None else as_float_or_blank(reproj_errors[i - 1]),
        })

    coords = np.array(coords)
    t_err_vecs = np.array(t_err_vecs)
    r_err_vecs = np.array(r_err_vecs)
    robot_rpys = np.array(robot_rpys)
    t_mags = np.array(t_mags)
    r_mags = np.array(r_mags)

    print(f"\n--- {solution.upper()} EVALUATION RESULTS ---")
    print(f"Mean Translation Error: {np.mean(t_mags):.3f} mm (Max: {np.max(t_mags):.3f} mm)")
    print(f"Mean Rotation Error:    {np.mean(r_mags):.3f} deg (Max: {np.max(r_mags):.3f} deg)")
    if reproj_errors is not None:
        print(
            f"Mean ChArUco Reprojection Error: {np.mean(reproj_errors):.3f} px "
            f"(Max: {np.max(reproj_errors):.3f} px)"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_tag = (
        solution
        if validation_mode == "validation"
        else f"{solution}_{validation_mode}"
    )
    csv_path = os.path.join(
        output_dir, f"validation_residuals_{artifact_tag}_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(residual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(residual_rows)
    print(f"\n✓ Saved per-sample residual CSV to: {csv_path}")

    print("\nWorst translation residuals:")
    for idx in np.argsort(t_mags)[-5:][::-1]:
        print(
            "  sample {:02d}: {:6.3f} mm, {:6.3f} deg, reproj={} px".format(
                idx + 1,
                t_mags[idx],
                r_mags[idx],
                "n/a" if reproj_errors is None else f"{float(reproj_errors[idx]):.3f}",
            )
        )

    # 4. Outlier Detection
    t_thresh, r_thresh = np.mean(t_mags) + np.std(t_mags), np.mean(r_mags) + np.std(r_mags)
    t_outliers, r_outliers = np.where(t_mags > t_thresh)[0], np.where(r_mags > r_thresh)[0]
    max_t_idx, max_r_idx = np.argmax(t_mags), np.argmax(r_mags)

    # 5. Create Plots
    fig = plt.figure(figsize=(18, 8))
    if validation_mode == "orientation":
        roll = robot_rpys[:, 0]
        pitch = robot_rpys[:, 1]

        ax1 = fig.add_subplot(121)
        sc1 = ax1.scatter(roll, pitch, c=r_mags, cmap='viridis', s=45)
        ax1.scatter(roll[0], pitch[0], c='red', marker='*', s=120, label='Start Point')
        ax1.scatter(roll[max_r_idx], pitch[max_r_idx], c='yellow', marker='*', s=160, edgecolors='black', label='Max Error Point')
        ax1.scatter(roll[r_outliers], pitch[r_outliers], facecolors='none', edgecolors='red', s=90, linewidths=1.5, label='>1std outlier')
        ax1.set_title('Orientation Sweep Rotation Error (deg)')
        ax1.set_xlabel('Robot roll (deg)')
        ax1.set_ylabel('Robot pitch (deg)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig.colorbar(sc1, ax=ax1, label='Error (deg)', shrink=0.7)

        ax2 = fig.add_subplot(122)
        sc2 = ax2.scatter(roll, pitch, c=t_mags, cmap='viridis', s=45)
        ax2.scatter(roll[0], pitch[0], c='red', marker='*', s=120, label='Start Point')
        ax2.scatter(roll[max_t_idx], pitch[max_t_idx], c='yellow', marker='*', s=160, edgecolors='black', label='Max Error Point')
        ax2.scatter(roll[t_outliers], pitch[t_outliers], facecolors='none', edgecolors='red', s=90, linewidths=1.5, label='>1std outlier')
        ax2.set_title('Orientation Sweep Translation Error (mm)')
        ax2.set_xlabel('Robot roll (deg)')
        ax2.set_ylabel('Robot pitch (deg)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig.colorbar(sc2, ax=ax2, label='Error (mm)', shrink=0.7)
    else:
        # --- PLOT 1: Rotation Error ---
        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(coords[:,0], coords[:,1], coords[:,2], c=r_mags, cmap='viridis', s=10)
        ax1.quiver(coords[:,0], coords[:,1], coords[:,2], r_err_vecs[:,0], r_err_vecs[:,1], r_err_vecs[:,2],
                   length=1.0, color='midnightblue', alpha=0.6)

        ax1.scatter(coords[0,0], coords[0,1], coords[0,2], c='red', marker='*', s=100, label='Start Point')
        ax1.scatter(coords[max_r_idx,0], coords[max_r_idx,1], coords[max_r_idx,2], c='yellow', marker='*', s=150, edgecolors='black', label='Max Error Point')
        ax1.scatter(coords[r_outliers,0], coords[r_outliers,1], coords[r_outliers,2], facecolors='none', edgecolors='red', s=80, linewidths=1.5, label='>1std outlier')

        ax1.set_title('Spatial Rotation Error Map (deg)')
        ax1.set_xlabel('X (mm)'), ax1.set_ylabel('Y (mm)'), ax1.set_zlabel('Z (mm)')
        ax1.legend()
        fig.colorbar(sc1, ax=ax1, label='Error (deg)', shrink=0.7)

        # --- PLOT 2: Translation Error ---
        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(coords[:,0], coords[:,1], coords[:,2], c=t_mags, cmap='viridis', s=10)

        ax2.quiver(coords[:,0], coords[:,1], coords[:,2], t_err_vecs[:,0], t_err_vecs[:,1], t_err_vecs[:,2],
                   length=1.0, color='midnightblue', alpha=0.6)

        ax2.scatter(coords[0,0], coords[0,1], coords[0,2], c='red', marker='*', s=100, label='Start Point')
        ax2.scatter(coords[max_t_idx,0], coords[max_t_idx,1], coords[max_t_idx,2], c='yellow', marker='*', s=150, edgecolors='black', label='Max Error Point')
        ax2.scatter(coords[t_outliers,0], coords[t_outliers,1], coords[t_outliers,2], facecolors='none', edgecolors='red', s=80, linewidths=1.5, label='>1std outlier')

        ax2.set_title('Spatial Translation Error Map (mm)')
        ax2.set_xlabel('X (mm)'), ax2.set_ylabel('Y (mm)'), ax2.set_zlabel('Z (mm)')
        ax2.legend()
        fig.colorbar(sc2, ax=ax2, label='Error (mm)', shrink=0.7)

    plt.tight_layout()

    plot_prefix = (
        "orientation_error_map"
        if validation_mode == "orientation"
        else "spatial_error_map"
    )
    save_filename = os.path.join(
        output_dir, f'{plot_prefix}_{artifact_tag}_{timestamp}.png')
    
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved high-resolution plot to: {save_filename}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "mean_translation_error_mm": float(np.mean(t_mags)),
        "max_translation_error_mm": float(np.max(t_mags)),
        "mean_rotation_error_deg": float(np.mean(r_mags)),
        "max_rotation_error_deg": float(np.max(r_mags)),
        "solution": solution,
        "validation_mode": validation_mode,
        "residual_csv": csv_path,
        "plot": save_filename,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hand-eye calibration against a validation dataset.")
    parser.add_argument("--calib", default=None, help="Calibration .npz. Defaults to latest output/hand_eye_cal_*.npz.")
    parser.add_argument("--validation", default=None, help="Validation .npz. Defaults to validation_dataset.npz.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for residual CSV and plot.")
    parser.add_argument(
        "--solution", choices=("weighted", "legacy"), default="weighted",
        help="Which solution stored in the calibration .npz to validate.")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a Matplotlib window.")
    args = parser.parse_args()
    evaluate_and_plot(
        calib_npz_path=args.calib,
        validation_npz_path=args.validation,
        output_dir=args.output_dir,
        show_plot=not args.no_show,
        solution=args.solution,
    )
