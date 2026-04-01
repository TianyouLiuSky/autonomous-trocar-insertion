import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2
import argparse

def evaluate_and_plot(calib_npz_path, validation_npz_path):
    # 1. Load the Calibration Matrices you want to test
    try:
        calib_data = np.load(calib_npz_path)
        T_cam2base = calib_data['T_cam2base']           # Matrix X
        T_board2gripper = calib_data['T_board2gripper'] # Matrix Y
        print(f"Loaded Calibration: {calib_npz_path}")
    except Exception as e:
        print(f"Failed to load calibration file: {e}")
        return

    # 2. Load the Static Validation Ground Truth Data
    try:
        val_data = np.load(validation_npz_path, allow_pickle=True)
        robot_poses = val_data['robot_poses'] # Matrix A components
        board_rvecs = val_data['board_rvecs'] # Matrix B components
        board_tvecs = val_data['board_tvecs']
        print(f"Loaded {len(robot_poses)} Ground Truth samples from: {validation_npz_path}")
    except Exception as e:
        print(f"Failed to load validation file: {e}")
        return

    coords, t_err_vecs, r_err_vecs = [], [], []
    t_mags, r_mags = [], []

    # 3. Compute Residuals for AY = XB
    for r_pose, b_rvec, b_tvec in zip(robot_poses, board_rvecs, board_tvecs):
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

        coords.append(T_left[:3, 3] * 1000) # Save coordinates for plotting (in mm)

        # Translation Error
        t_err = (T_right[:3, 3] - T_left[:3, 3]) * 1000 
        t_err_vecs.append(t_err)
        t_mags.append(np.linalg.norm(t_err))

        # Rotation Error
        R_err_mat = T_right[:3, :3] @ T_left[:3, :3].T
        rot_err_vec = Rotation.from_matrix(R_err_mat).as_rotvec() 
        r_mag_deg = np.linalg.norm(rot_err_vec) * (180.0 / np.pi) 
        
        r_err_vecs.append((rot_err_vec / (np.linalg.norm(rot_err_vec) + 1e-8)) * r_mag_deg) 
        r_mags.append(r_mag_deg)

    coords = np.array(coords)
    t_err_vecs = np.array(t_err_vecs)
    r_err_vecs = np.array(r_err_vecs)
    t_mags = np.array(t_mags)
    r_mags = np.array(r_mags)

    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Mean Translation Error: {np.mean(t_mags):.3f} mm (Max: {np.max(t_mags):.3f} mm)")
    print(f"Mean Rotation Error:    {np.mean(r_mags):.3f} deg (Max: {np.max(r_mags):.3f} deg)")

    # 4. Outlier Detection
    t_thresh, r_thresh = np.mean(t_mags) + np.std(t_mags), np.mean(r_mags) + np.std(r_mags)
    t_outliers, r_outliers = np.where(t_mags > t_thresh)[0], np.where(r_mags > r_thresh)[0]
    max_t_idx, max_r_idx = np.argmax(t_mags), np.argmax(r_mags)

    # 5. Create Plots
    fig = plt.figure(figsize=(18, 8))

    # --- PLOT 1: Rotation Error ---
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(coords[:,0], coords[:,1], coords[:,2], c=r_mags, cmap='viridis', s=10)
    ax1.quiver(coords[:,0], coords[:,1], coords[:,2], r_err_vecs[:,0], r_err_vecs[:,1], r_err_vecs[:,2], 
               length=1.5, color='midnightblue', alpha=0.6)
    
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
               length=10.0, color='midnightblue', alpha=0.6) 
    
    ax2.scatter(coords[0,0], coords[0,1], coords[0,2], c='red', marker='*', s=100, label='Start Point')
    ax2.scatter(coords[max_t_idx,0], coords[max_t_idx,1], coords[max_t_idx,2], c='yellow', marker='*', s=150, edgecolors='black', label='Max Error Point')
    ax2.scatter(coords[t_outliers,0], coords[t_outliers,1], coords[t_outliers,2], facecolors='none', edgecolors='red', s=80, linewidths=1.5, label='>1std outlier')

    ax2.set_title('Spatial Translation Error Map (mm)')
    ax2.set_xlabel('X (mm)'), ax2.set_ylabel('Y (mm)'), ax2.set_zlabel('Z (mm)')
    ax2.legend()
    fig.colorbar(sc2, ax=ax2, label='Error (mm)', shrink=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # You can change these filenames to test different calibrations against the same dataset
    plot_spatial_error_maps(calib_npz_path='hand_eye_calibration.npz', 
                            validation_npz_path='validation_dataset.npz')