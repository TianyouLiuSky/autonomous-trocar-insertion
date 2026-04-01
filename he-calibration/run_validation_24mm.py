import numpy as np
from SHER_Controller import SHERController
import time

def generate_validation_poses(base_pose):
    """
    Generates 27 targets mapping a 24mm^3 workspace (±12mm from center).
    Pairs translations with ±5 degree rotations to ensure diverse viewing angles.
    """
    poses = []
    x, y, z, r, p, y_yaw = base_pose
    
    # Format: [X, Y, Z, Roll, Pitch, Yaw]
    offsets = [
        # Z = -12mm (Bottom Layer)
        [-12, -12, -12, -5, -5, 0], [ 0, -12, -12,  0, -5, 0], [ 12, -12, -12,  5, -5, 0],
        [-12,   0, -12, -5,  0, 0], [ 0,   0, -12,  0,  0, 0], [ 12,   0, -12,  5,  0, 0],
        [-12,  12, -12, -5,  5, 0], [ 0,  12, -12,  0,  5, 0], [ 12,  12, -12,  5,  5, 0],

        # Z = 0mm (Middle Layer)
        [-12, -12,   0, -5, -5, 0], [ 0, -12,   0,  0, -5, 0], [ 12, -12,   0,  5, -5, 0],
        [-12,   0,   0, -5,  0, 0], [ 0,   0,   0,  0,  0, 0], [ 12,   0,   0,  5,  0, 0],
        [-12,  12,   0, -5,  5, 0], [ 0,  12,   0,  0,  5, 0], [ 12,  12,   0,  5,  5, 0],

        # Z = +12mm (Top Layer)
        [-12, -12,  12, -5, -5, 0], [ 0, -12,  12,  0, -5, 0], [ 12, -12,  12,  5, -5, 0],
        [-12,   0,  12, -5,  0, 0], [ 0,   0,  12,  0,  0, 0], [ 12,   0,  12,  5,  0, 0],
        [-12,  12,  12, -5,  5, 0], [ 0,  12,  12,  0,  5, 0], [ 12,  12,  12,  5,  5, 0],
    ]
    
    for offset in offsets:
        new_pose = [
            x + offset[0], y + offset[1], z + offset[2],
            r + offset[3], p + offset[4], y_yaw + offset[5]
        ]
        poses.append(new_pose)
    return poses

if __name__ == "__main__":
    robot = SHERController(robot_name='SHER20')
    start_pose = robot.get_current_pose()
    targets = generate_validation_poses(start_pose)
    
    print(f"Starting validation sequence for 27 poses...")
    print("MAKE SURE THE GUI IS RUNNING (CHANGE N_SAMPLES TO 27 IN GUI!)")

    for i, target in enumerate(targets):
        print(f"\nMoving to Pose {i+1}/27: {target}")
        
        # Relaxed tolerances to prevent timeout failures
        success = robot.no_rcm_move_to(target, position_tol=0.5, orientation_tol=0.5, timeout=15.0)
        
        if success:
            print(f"Reached Target {i+1}. Switch to GUI and press SPACE now.")
            time.sleep(10) 
        else:
            print(f"Failed to reach Target {i+1}, skipping...")

    print("\nSequence complete. Save the data in the GUI.")