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
# Format: [X, Y, Z, Roll, Pitch, Yaw]
    # Translations are restricted to +/- 2mm to stay centered in the D405 frame.
    # Rotations use a 5x5 grid (-24, -12, 0, 12, 24) to guarantee >12 degree differences.
    
    offsets = [
        # Center Row (Pitch = 0)
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 12, 0, 0],
        [2, 0, 0, 24, 0, 0],
        [-1, 0, 1, -12, 0, 0],
        [-2, 0, 0, -24, 0, 0],

        # Row 2 (Pitch = 12)
        [0, 1, -1, 0, 12, 0],
        [1, 1, 0, 12, 12, 0],
        [2, 1, -1, 24, 12, 0],
        [-1, 1, 0, -12, 12, 0],
        [-2, 1, -1, -24, 12, 0],

        # Row 3 (Pitch = 24)
        [0, 2, 0, 0, 24, 0],
        [1, 2, 1, 12, 24, 0],
        [2, 2, 0, 24, 24, 0],
        [-1, 2, 1, -12, 24, 0],
        [-2, 2, 0, -24, 24, 0],

        # Row 4 (Pitch = -12)
        [0, -1, 1, 0, -12, 0],
        [1, -1, 0, 12, -12, 0],
        [2, -1, 1, 24, -12, 0],
        [-1, -1, 0, -12, -12, 0],
        [-2, -1, 1, -24, -12, 0],

        # Row 5 (Pitch = -24)
        [0, -2, 0, 0, -24, 0],
        [1, -2, -1, 12, -24, 0],
        [2, -2, 0, 24, -24, 0],
        [-1, -2, -1, -12, -24, 0],
        [-2, -2, 0, -24, -24, 0],
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