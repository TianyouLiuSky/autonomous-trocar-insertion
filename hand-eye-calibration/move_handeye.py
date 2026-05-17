#!/usr/bin/env python
"""
run_box_motion.py  —  ATI Hand-Eye: 8-corner translation sweep + 9-orientation center sweep
5-DOF robot: yaw fixed at 0 throughout.

Phase 1 — Translation diversity:  8 box corners at nominal orientation
Phase 2 — Rotation diversity:     9 orientation variants at anchor (center)

Total: 17 poses. Waits for Enter at each pose.
"""

import numpy as np
import csv
import os
from datetime import datetime
from SHER_Controller import SHERController

# ── config ──────────────────────────────────────────────────────────────────
BOX_MM    = 24.0
ROLL_DEG  = 10.0
PITCH_DEG = 10.0
TIMEOUT   = 30.0

OUTPUT_DIR = os.path.expanduser("~/Autonomous-Trocar-Insertion/he-calibration/data")

CORNERS = np.array([
    [0,      0,      0     ],
    [BOX_MM, 0,      0     ],
    [0,      BOX_MM, 0     ],
    [BOX_MM, BOX_MM, 0     ],
    [0,      0,      BOX_MM],
    [BOX_MM, 0,      BOX_MM],
    [0,      BOX_MM, BOX_MM],
    [BOX_MM, BOX_MM, BOX_MM],
])

CORNER_LABELS = [
    "C0_x0y0z0", "C1_x1y0z0", "C2_x0y1z0", "C3_x1y1z0",
    "C4_x0y0z1", "C5_x1y0z1", "C6_x0y1z1", "C7_x1y1z1",
]

# (d_roll, d_pitch, suffix) — yaw always 0 (5-DOF)
ORIENT_VARIANTS = [
    ( 0.0,         0.0,        "nom"  ),
    (+ROLL_DEG,    0.0,        "rP"   ),
    (-ROLL_DEG,    0.0,        "rN"   ),
    ( 0.0,        +PITCH_DEG,  "pP"   ),
    ( 0.0,        -PITCH_DEG,  "pN"   ),
    (+ROLL_DEG,   +PITCH_DEG,  "rPpP" ),
    (+ROLL_DEG,   -PITCH_DEG,  "rPpN" ),
    (-ROLL_DEG,   +PITCH_DEG,  "rNpP" ),
    (-ROLL_DEG,   -PITCH_DEG,  "rNpN" ),
]
# ────────────────────────────────────────────────────────────────────────────


def build_phase1_translation(anchor):
    """8 corners at nominal orientation."""
    ax, ay, az, ar, ap, _ = anchor
    pts, labels = [], []
    for corner, clabel in zip(CORNERS, CORNER_LABELS):
        pts.append([ax + corner[0], ay + corner[1], az + corner[2], ar, ap, 0.0])
        labels.append(f"P1_{clabel}_nom")
    return pts, labels


def build_phase2_rotation(anchor):
    """9 orientation variants at anchor position."""
    ax, ay, az, ar, ap, _ = anchor
    pts, labels = [], []
    for d_roll, d_pitch, osuffix in ORIENT_VARIANTS:
        pts.append([ax, ay, az, ar + d_roll, ap + d_pitch, 0.0])
        labels.append(f"P2_center_{osuffix}")
    return pts, labels


def move_and_log(robot, targets, labels, results, phase_name):
    n_total = len(targets)
    for i, (target, label) in enumerate(zip(targets, labels)):
        print(f"\n  [{i+1}/{n_total}]  {label}")
        print(f"           target: {[round(v, 2) for v in target]}")

        ok = robot.no_rcm_move_to(
            target,
            position_tol=0.5,
            orientation_tol=0.5,
            timeout=TIMEOUT,
        )
        results.append((label, target, ok))

        status = "REACHED" if ok else "TIMEOUT"
        print(f"  [{i+1}/{n_total}]  {status} — watch velocity, then press Enter to log...")
        input()


def save_csv(anchor, results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "cmd_x", "cmd_y", "cmd_z",
                    "cmd_roll", "cmd_pitch", "cmd_yaw", "reached"])
        w.writerow(["ANCHOR",
                    round(anchor[0], 4), round(anchor[1], 4), round(anchor[2], 4),
                    round(anchor[3], 4), round(anchor[4], 4), round(anchor[5], 4),
                    "N/A"])
        for label, t, ok in results:
            w.writerow([label,
                        round(t[0], 4), round(t[1], 4), round(t[2], 4),
                        round(t[3], 4), round(t[4], 4), round(t[5], 4),
                        ok])
    print(f"Setpoints saved -> {path}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%d%b%Y").upper()
    csv_path  = os.path.join(OUTPUT_DIR, f"box_setpoints_{timestamp}.csv")

    robot  = SHERController(robot_name="SHER20")
    anchor = robot.get_current_pose()

    p1_pts, p1_labels = build_phase1_translation(anchor)
    p2_pts, p2_labels = build_phase2_rotation(anchor)
    n_total = len(p1_pts) + len(p2_pts)

    print(f"Anchor:  {np.round(anchor, 2)}")
    print(f"Phase 1: {len(p1_pts)} corners  (translation diversity, nominal orientation)")
    print(f"Phase 2: {len(p2_pts)} variants (rotation diversity, anchor position)")
    print(f"Total:   {n_total} poses")
    input("\nPress Enter to begin Phase 1 (translation)...")

    all_results = []

    print("\n── Phase 1: Translation ──────────────────────────────────────────")
    move_and_log(robot, p1_pts, p1_labels, all_results, "Phase 1")

    print("\n── Phase 2: Rotation (returning to anchor first) ────────────────")
    input("Press Enter to begin Phase 2 (rotation)...")
    move_and_log(robot, p2_pts, p2_labels, all_results, "Phase 2")

    save_csv(anchor, all_results, csv_path)
    n_reached = sum(ok for _, _, ok in all_results)
    print(f"\nDone.  {n_reached}/{n_total} poses reached.")
    print(f"CSV: {csv_path}")
    print("Click 'Compute Calibration' in the GUI.")