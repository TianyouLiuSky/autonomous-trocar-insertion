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
    offsets = [
        [0, 0, 0, 0, 0, 0],           # 1: Center
        [10, 5, 5, 10, 0, 0],         # 2: Roll +10
        [-10, -5, 10, -10, 0, 0],     # 3: Roll -10
        [5, 15, -5, 0, 10, 0],        # 4: Pitch +10
        [-5, -15, 0, 0, -10, 0],      # 5: Pitch -10
        [15, 0, 10, 15, 15, 0],       # 6: Quadrant 1
        [-15, 10, -10, -15, -15, 0],  # 7: Quadrant 3
        [0, -10, 15, 15, -15, 0],     # 8: Quadrant 4
        [10, -15, -5, -15, 15, 0],    # 9: Quadrant 2
        [-10, 5, 10, 20, 0, 0],       # 10: Extreme Roll +
        [20, 0, 0, -20, 0, 0],        # 11: Extreme Roll -
        [0, 20, 0, 0, 20, 0],         # 12: Extreme Pitch +
        [-20, -20, 5, 0, -20, 0],     # 13: Extreme Pitch -
        [5, 5, 20, 20, 10, 0],        # 14: Mid-grid
        [-5, -5, -10, -20, 10, 0],    # 15: Mid-grid
        [10, 20, 15, -20, -10, 0],    # 16: Mid-grid
        [-15, -5, -5, 20, -10, 0],    # 17: Mid-grid
        [10, -10, 10, 10, 20, 0],     # 18: Mid-grid
        [-10, 15, -15, -10, 20, 0],   # 19: Mid-grid
        [0, 0, 15, -10, -20, 0],      # 20: Mid-grid
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
            time.sleep(10) 
        else:
            print(f"Failed to reach Target {i+1}, skipping...")

    print("\nSequence complete. Click 'Compute Calibration' in the GUI.")