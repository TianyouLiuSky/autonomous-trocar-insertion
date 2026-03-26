import numpy as np
from SHER_Controller import SHERController
import time

def generate_diverse_poses(base_pose):
    """
    Generates 20 targets by adding offsets to a starting position.
    We need high rotation diversity for good hand-eye calibration.
    """
    poses = []
    # Current pose is [x, y, z, roll, pitch, yaw]
    x, y, z, r, p, y_yaw = base_pose
    
    # 20 offsets to create a 'cloud' of points with different tilts
    # Format: [X, Y, Z, Roll, Pitch, Yaw]
    offsets = [
        # Sweeping Right (Pitch = -16)
        [-8, -8,  3, -16, -16, 0],  # 1: Start bottom-left
        [-4, -8,  2,  -8, -16, 0],  # 2
        [ 0, -8,  2,   0, -16, 0],  # 3
        [ 4, -8,  2,   8, -16, 0],  # 4
        [ 8, -8,  3,  16, -16, 0],  # 5: Reach bottom-right
        
        # Move up a row, Sweeping Left (Pitch = -8)
        [ 8, -4,  2,  16,  -8, 0],  # 6: Shift up slightly
        [ 4, -4,  2,   8,  -8, 0],  # 7
        [ 0, -4,  1,   0,  -8, 0],  # 8
        [-4, -4,  2,  -8,  -8, 0],  # 9
        [-8, -4,  2, -16,  -8, 0],  # 10: Reach mid-left
        
        # Move up a row, Sweeping Right (Pitch = 0)
        [-8,  0,  2, -16,   0, 0],  # 11: Shift up slightly
        [-4,  0,  1,  -8,   0, 0],  # 12
        [ 0,  0,  0,   0,   0, 0],  # 13: Dead Center
        [ 4,  0,  1,   8,   0, 0],  # 14
        [ 8,  0,  2,  16,   0, 0],  # 15: Reach mid-right
        
        # Move up a row, Sweeping Left (Pitch = 8)
        [ 8,  4,  2,  16,   8, 0],  # 16: Shift up slightly
        [ 4,  4,  2,   8,   8, 0],  # 17
        [ 0,  4,  1,   0,   8, 0],  # 18
        [-4,  4,  2,  -8,   8, 0],  # 19
        [-8,  4,  2, -16,   8, 0],  # 20: Reach top-left
        
        # Move up to final row, Sweeping Right (Pitch = 16)
        [-8,  8,  3, -16,  16, 0],  # 21: Shift up slightly
        [-4,  8,  2,  -8,  16, 0],  # 22
        [ 0,  8,  2,   0,  16, 0],  # 23
        [ 4,  8,  2,   8,  16, 0],  # 24
        [ 8,  8,  3,  16,  16, 0],  # 25: End top-right
    ]
    
    for offset in offsets:
        new_pose = [
            x + offset[0], y + offset[1], z + offset[2],
            r + offset[3], p + offset[4], y_yaw + offset[5]
        ]
        poses.append(new_pose)
    return poses

if __name__ == "__main__":
    # 1. Initialize Controller
    robot = SHERController(robot_name='SHER20')
    
    # 2. Get safe starting position
    start_pose = robot.get_current_pose()
    targets = generate_diverse_poses(start_pose)
    
    print(f"Starting auto-calibration sequence for 20 poses...")
    print("MAKE SURE THE CALIBRATION GUI IS RUNNING AND VISIBLE!")

    for i, target in enumerate(targets):
        print(f"\nMoving to Pose {i+1}/20: {target}")
        
        # Move robot (No RCM for calibration)
        success = robot.no_rcm_move_to(target, timeout=10.0)
        
        if success:
            print(f"Reached Target {i+1}. Switch to GUI and press SPACE now.")
            # We give you 15 seconds to press Space in the other window
            # Alternatively, you can increase this sleep time
            time.sleep(15) 
        else:
            print(f"Failed to reach Target {i+1}, skipping...")

    print("\nSequence complete. Click 'Compute Calibration' in the GUI.")