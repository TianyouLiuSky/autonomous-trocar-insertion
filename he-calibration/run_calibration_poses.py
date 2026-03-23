import numpy as np
from SHERController import SHERController
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
    offsets = [
        [0, 0, 0, 0, 0, 0],       # 1: Home
        [10, 5, 5, 10, 0, 0],     # 2
        [-10, -5, 10, -10, 5, 0], # 3
        [5, 15, -5, 0, 15, 0],    # 4
        [-5, -15, 0, 5, -15, 0],  # 5
        [15, 0, 10, 15, 10, 0],   # 6
        [-15, 10, -10, -15, -10, 0],# 7
        [0, -10, 15, 10, 15, 0],  # 8
        [10, -15, -5, -10, -5, 0],# 9
        [-10, 5, 10, 5, 10, 0],   # 10
        [20, 0, 0, 20, 0, 0],     # 11
        [0, 20, 0, 0, 20, 0],     # 12
        [-20, -20, 5, -20, -20, 0],# 13
        [5, 5, 20, 15, -15, 0],   # 14
        [-5, -5, -10, -15, 15, 0],# 15
        [10, 20, 15, 5, 5, 0],    # 16
        [-15, -5, -5, -5, -5, 0], # 17
        [10, -10, 10, 20, 10, 0], # 18
        [-10, 15, -15, -20, -10, 0],# 19
        [0, 0, 10, 0, 0, 0],      # 20: Back near start
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
            # We give you 3 seconds to press Space in the other window
            # Alternatively, you can increase this sleep time
            time.sleep(3) 
        else:
            print(f"Failed to reach Target {i+1}, skipping...")

    print("\nSequence complete. Click 'Compute Calibration' in the GUI.")