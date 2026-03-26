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
    
# Format: [X, Y, Z, Roll, Pitch, Yaw]
    # Z-axis restricted to +/- 4mm. Rotations spaced by >= 12 degrees.
    offsets = [
        [0, 0, 0, 0, 0, 0],             # 1: Center
        [5, 5, 2, 12, 0, 0],            # 2
        [-5, -5, -2, -12, 0, 0],        # 3
        [10, -5, 2, 0, 12, 0],          # 4
        [-10, 5, -2, 0, -12, 0],        # 5
        [10, 10, 3, 12, 12, 0],         # 6
        [-10, 10, -3, -12, 12, 0],      # 7
        [10, -10, 3, 12, -12, 0],       # 8
        [-10, -10, -3, -12, -12, 0],    # 9
        [15, 0, 4, 20, 0, 0],           # 10
        [-15, 0, -4, -20, 0, 0],        # 11
        [0, 15, 4, 0, 20, 0],           # 12
        [0, -15, -4, 0, -20, 0],        # 13
        [15, 10, 1, 20, 12, 0],         # 14
        [-15, 10, -1, -20, 12, 0],      # 15
        [15, -10, 2, 20, -12, 0],       # 16
        [-15, -10, -2, -20, -12, 0],    # 17
        [10, 15, 3, 12, 20, 0],         # 18
        [-10, 15, -3, -12, 20, 0],      # 19
        [10, -15, 4, 12, -20, 0],       # 20
        [-10, -15, -4, -12, -20, 0],    # 21
        [15, 15, 0, 20, 20, 0],         # 22
        [-15, 15, 0, -20, 20, 0],       # 23
        [15, -15, 0, 20, -20, 0],       # 24
        [-15, -15, 0, -20, -20, 0],     # 25
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