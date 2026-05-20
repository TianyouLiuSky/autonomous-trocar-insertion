import numpy as np
import sys
from pathlib import Path

MOTION_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "motion_script"
if str(MOTION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_SCRIPT_DIR))

from SHER_Controller import SHERController

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
    
    print(f"\n{'='*50}")
    print(f"Starting validation sequence for 27 poses...")
    print("MAKE SURE THE DATA COLLECTOR SCRIPT IS RUNNING!")
    print(f"{'='*50}\n")

    for i, target in enumerate(targets):
        print(f"\nMoving to Pose {i+1}/27: {target}")
        
        # Let the robot try to reach the target for up to 15 seconds.
        # It doesn't matter if it returns True or False, it will be physically in a new position.
        robot.no_rcm_move_to(target, position_tol=0.5, orientation_tol=0.5, timeout=15.0)
        
        # Robot has settled (either by reaching tolerance or timing out during micro-adjustments)
        print(f"\n>>> Robot has settled at Pose {i+1}.")
        print(">>> Switch to the Camera window and press SPACE to record.")
        
        # Block execution until the user explicitly says they captured it
        input(">>> Press ENTER in this terminal once you have captured the data to move to the next point...")

    print("\nSequence complete. The data collector should automatically save the dataset!")
