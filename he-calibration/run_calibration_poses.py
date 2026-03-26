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
    
    offsets = [
        # Center & Pure Roll row (Pitch = 0)
        [0, 0, 0, 0, 0, 0],             # 1: Center
        [2, 0, 1, 8, 0, 0],             # 2
        [4, 0, 2, 16, 0, 0],            # 3: Max Roll +
        [-2, 0, -1, -8, 0, 0],          # 4
        [-4, 0, -2, -16, 0, 0],         # 5: Max Roll -

        # Pitch +8 row
        [0, 3, 2, 0, 8, 0],             # 6
        [2, 3, 3, 8, 8, 0],             # 7
        [4, 3, 4, 16, 8, 0],            # 8
        [-2, 3, 1, -8, 8, 0],           # 9
        [-4, 3, 0, -16, 8, 0],          # 10

        # Pitch +16 row (Max Pitch +)
        [0, 6, 4, 0, 16, 0],            # 11
        [2, 6, 5, 8, 16, 0],            # 12
        [4, 6, 6, 16, 16, 0],           # 13: Extreme +/+
        [-2, 6, 3, -8, 16, 0],          # 14
        [-4, 6, 2, -16, 16, 0],         # 15: Extreme -/+

        # Pitch -8 row
        [0, -3, -2, 0, -8, 0],          # 16
        [2, -3, -1, 8, -8, 0],          # 17
        [4, -3, 0, 16, -8, 0],          # 18
        [-2, -3, -3, -8, -8, 0],        # 19
        [-4, -3, -4, -16, -8, 0],       # 20

        # Pitch -16 row (Max Pitch -)
        [0, -6, -4, 0, -16, 0],         # 21
        [2, -6, -3, 8, -16, 0],         # 22
        [4, -6, -2, 16, -16, 0],        # 23: Extreme +/-
        [-2, -6, -5, -8, -16, 0],       # 24
        [-4, -6, -6, -16, -16, 0],      # 25: Extreme -/-
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