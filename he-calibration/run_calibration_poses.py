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
    
    # 25 offsets to create a 'cloud' of points with different tilts
    # Format: [X, Y, Z, Roll, Pitch, Yaw]
    offsets = [
        # Row 1: Pitch -16
        [-8, -8, 4, -16, -16, 0],
        [-4, -8, 2,  -8, -16, 0],
        [ 0, -8, 0,   0, -16, 0],
        [ 4, -8, 2,   8, -16, 0],
        [ 8, -8, 4,  16, -16, 0],

        # Row 2: Pitch -8 (Snake backwards)
        [ 8, -4, 2,  16,  -8, 0],
        [ 4, -4, 1,   8,  -8, 0],
        [ 0, -4, 0,   0,  -8, 0],
        [-4, -4, 1,  -8,  -8, 0],
        [-8, -4, 2, -16,  -8, 0],

        # Row 3: Pitch 0 (Center sweep)
        [-8,  0, 0, -16,   0, 0],
        [-4,  0, 0,  -8,   0, 0],
        [ 0,  0, 0,   0,   0, 0], # Home Center
        [ 4,  0, 0,   8,   0, 0],
        [ 8,  0, 0,  16,   0, 0],

        # Row 4: Pitch 8 (Snake backwards)
        [ 8,  4, 2,  16,   8, 0],
        [ 4,  4, 1,   8,   8, 0],
        [ 0,  4, 0,   0,   8, 0],
        [-4,  4, 1,  -8,   8, 0],
        [-8,  4, 2, -16,   8, 0],

        # Row 5: Pitch 16
        [-8,  8, 4, -16,  16, 0],
        [-4,  8, 2,  -8,  16, 0],
        [ 0,  8, 0,   0,  16, 0],
        [ 4,  8, 2,   8,  16, 0],
        [ 8,  8, 4,  16,  16, 0],
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
    
    print(f"Starting auto-calibration sequence for 25 poses...")
    print("MAKE SURE THE CALIBRATION GUI IS RUNNING AND VISIBLE!")

    for i, target in enumerate(targets):
        print(f"\nMoving to Pose {i+1}/25: {target}")
        
        # Move robot (No RCM for calibration)
        success = robot.no_rcm_move_to(target, position_tol=0.5, orientation_tol=0.5, timeout=30.0)
        
        if success:
            print(f"Reached Target {i+1}. Switch to GUI and press SPACE now.")
            # We give you 15 seconds to press Space in the other window
            # Alternatively, you can increase this sleep time
            time.sleep(15) 
        else:
            print(f"Failed to reach Target {i+1}, skipping...")

    print("\nSequence complete. Click 'Compute Calibration' in the GUI.")