#!/usr/bin/env python3
"""
Interactive RCM motion test for SHER.

This script is intentionally cautious:
- computes the RCM point from the current tool axis and a configured distance
- runs a no-motion check, an axial slide, and two tiny pivot checks
- prints numeric RCM-line, tool-axis, and position errors
- writes a CSV log so the test has data you can compare across runs
- pauses before every commanded motion so the operator can watch the robot

Run from this directory or anywhere else:

    python3 test_rcm_motion.py --robot-name SHER20 --rcm-distance-mm 19
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path


EPS = 1e-9
np = None
R = None


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive RCM motion test.")
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument(
        "--rcm-distance-mm",
        type=float,
        default=19.0,
        help="Distance from current tip pose back to the trocar/RCM point.",
    )
    parser.add_argument(
        "--rcm-side",
        choices=("minus-z", "plus-z"),
        default="minus-z",
        help="Which side of the current tool z-axis contains the RCM point.",
    )
    parser.add_argument("--axial-step-mm", type=float, default=0.5)
    parser.add_argument("--pivot-step-mm", type=float, default=0.25)
    parser.add_argument("--position-tol-mm", type=float, default=0.2)
    parser.add_argument("--orientation-tol-deg", type=float, default=1.0)
    parser.add_argument("--rcm-axis-tol-mm", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--max-linear-vel", type=float, default=0.2)
    parser.add_argument("--max-angular-vel", type=float, default=0.05)
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent / "rcm_test_logs"),
        help="Directory for CSV logs.",
    )
    parser.add_argument(
        "--skip-y-pivot",
        action="store_true",
        help="Only run the x-axis pivot test.",
    )
    return parser.parse_args()


def get_state(controller):
    pose = controller.get_current_pose()
    pos = np.asarray(pose[:3], dtype=float)
    quat = np.array([controller.qx, controller.qy, controller.qz, controller.qw], dtype=float)
    rot = R.from_quat(quat)
    mat = rot.as_matrix()
    return {
        "pose": pose,
        "pos": pos,
        "quat": quat,
        "rot": rot,
        "x_axis": mat[:, 0],
        "y_axis": mat[:, 1],
        "z_axis": mat[:, 2],
    }


def unit(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < EPS:
        raise ValueError("Cannot normalize a zero vector")
    return vec / norm


def project_perp(vec, axis):
    axis = unit(axis)
    vec = np.asarray(vec, dtype=float)
    perp = vec - np.dot(vec, axis) * axis
    return unit(perp)


def point_to_axis_distance(point, axis_point, axis_dir):
    axis_dir = unit(axis_dir)
    offset = np.asarray(point, dtype=float) - np.asarray(axis_point, dtype=float)
    return float(np.linalg.norm(offset - np.dot(offset, axis_dir) * axis_dir))


def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < EPS:
        return 180.0
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def make_rcm_point(state, args):
    sign = -1.0 if args.rcm_side == "minus-z" else 1.0
    return state["pos"] + sign * args.rcm_distance_mm * state["z_axis"]


def desired_tool_axis(state, rcm_point, pos):
    shaft = np.asarray(pos, dtype=float) - rcm_point
    shaft_axis = unit(shaft)
    initial_shaft = state["pos"] - rcm_point
    axis_sign = 1.0 if np.dot(state["z_axis"], initial_shaft) >= 0.0 else -1.0
    return axis_sign * shaft_axis


def metrics(label, target_pos, rcm_point, initial_state, before_state, after_state, ok, elapsed_s):
    after_pos = after_state["pos"]
    before_pos = before_state["pos"]
    shaft_axis_before = unit(before_pos - rcm_point)
    delta = after_pos - before_pos
    axial_motion = float(np.dot(delta, shaft_axis_before))
    lateral_motion = float(np.linalg.norm(delta - axial_motion * shaft_axis_before))
    desired_axis = desired_tool_axis(initial_state, rcm_point, after_pos)

    return {
        "label": label,
        "ok": bool(ok),
        "elapsed_s": elapsed_s,
        "target_x": target_pos[0],
        "target_y": target_pos[1],
        "target_z": target_pos[2],
        "final_x": after_pos[0],
        "final_y": after_pos[1],
        "final_z": after_pos[2],
        "position_error_mm": float(np.linalg.norm(target_pos - after_pos)),
        "tip_motion_mm": float(np.linalg.norm(delta)),
        "axial_motion_mm": axial_motion,
        "lateral_motion_mm": lateral_motion,
        "rcm_line_error_mm": point_to_axis_distance(rcm_point, after_pos, after_state["z_axis"]),
        "tool_axis_error_deg": angle_deg(after_state["z_axis"], desired_axis),
    }


def print_metric_row(row):
    print(
        "  ok={ok}  pos_err={position_error_mm:.4f} mm  "
        "rcm_line={rcm_line_error_mm:.4f} mm  "
        "axis_err={tool_axis_error_deg:.3f} deg  "
        "tip_motion={tip_motion_mm:.4f} mm  "
        "axial={axial_motion_mm:.4f} mm  lateral={lateral_motion_mm:.4f} mm".format(**row)
    )


def prompt_continue(message):
    response = input(f"\n{message}\nPress Enter to run, or type q then Enter to quit: ").strip()
    if response.lower() == "q":
        print("Stopping before commanded motion.")
        sys.exit(0)


def prompt_observation():
    response = input(
        "Did the physical motion look RCM-correct? "
        "Type y, n, or notes, then Enter: "
    ).strip()
    return response


def run_motion(controller, label, target_pos, rcm_point, initial_state, args):
    print(f"\n=== {label} ===")
    before = get_state(controller)
    print(f"Target position: {np.round(target_pos, 4)}")
    print(f"Current position: {np.round(before['pos'], 4)}")
    print(
        "Initial line error now: "
        f"{point_to_axis_distance(rcm_point, before['pos'], before['z_axis']):.4f} mm"
    )
    prompt_continue(f"Watch the tool shaft around the RCM point for: {label}")

    start = time.time()
    ok = controller.rcm_move_to(
        target_pos=target_pos,
        rcm_point=rcm_point,
        rcm_axis_tol=args.rcm_axis_tol_mm,
        position_tol=args.position_tol_mm,
        orientation_tol=args.orientation_tol_deg,
        timeout=args.timeout_s,
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
    )
    elapsed = time.time() - start
    after = get_state(controller)
    row = metrics(label, target_pos, rcm_point, initial_state, before, after, ok, elapsed)
    print_metric_row(row)
    row["operator_observation"] = prompt_observation()
    return row


def write_csv(rows, args):
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"rcm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    global np, R
    import numpy as _np
    from scipy.spatial.transform import Rotation as _Rotation

    np = _np
    R = _Rotation

    from SHER_Controller_rcm_fixed import SHERController

    if args.rcm_distance_mm <= 0:
        raise ValueError("--rcm-distance-mm must be positive")
    if args.max_linear_vel > 0.5 or args.max_angular_vel > 0.1:
        print("WARNING: Velocity limits are above the conservative first-test range.")

    controller = SHERController(robot_name=args.robot_name)
    initial = get_state(controller)
    rcm_point = make_rcm_point(initial, args)
    initial_line_error = point_to_axis_distance(rcm_point, initial["pos"], initial["z_axis"])

    print("\nRCM test setup")
    print(f"  robot: {args.robot_name}")
    print(f"  current position: {np.round(initial['pos'], 4)}")
    print(f"  current z-axis:   {np.round(initial['z_axis'], 6)}")
    print(f"  rcm point:        {np.round(rcm_point, 4)}")
    print(f"  rcm distance:     {args.rcm_distance_mm:.4f} mm ({args.rcm_side})")
    print(f"  initial line err: {initial_line_error:.4f} mm")
    print(f"  max linear vel:   {args.max_linear_vel:.4f} mm/s")
    print(f"  max angular vel:  {args.max_angular_vel:.4f} rad/s")
    print("\nKeep hand on the e-stop. Use tiny steps first.")

    if initial_line_error > args.rcm_axis_tol_mm:
        print(
            "\nWARNING: The computed RCM point is not close to the current tool axis. "
            "Check --rcm-distance-mm and --rcm-side before moving."
        )

    rows = []

    rows.append(
        run_motion(
            controller,
            "no_motion_hold",
            initial["pos"].copy(),
            rcm_point,
            initial,
            args,
        )
    )

    state = get_state(controller)
    shaft_axis = unit(state["pos"] - rcm_point)
    axial_target = state["pos"] + args.axial_step_mm * shaft_axis
    rows.append(
        run_motion(
            controller,
            "axial_slide",
            axial_target,
            rcm_point,
            initial,
            args,
        )
    )

    state = get_state(controller)
    shaft = state["pos"] - rcm_point
    depth = np.linalg.norm(shaft)
    lateral_x = project_perp(state["x_axis"], shaft)
    pivot_x_target = rcm_point + depth * unit(shaft + args.pivot_step_mm * lateral_x)
    rows.append(
        run_motion(
            controller,
            "pivot_x_small",
            pivot_x_target,
            rcm_point,
            initial,
            args,
        )
    )

    if not args.skip_y_pivot:
        state = get_state(controller)
        shaft = state["pos"] - rcm_point
        depth = np.linalg.norm(shaft)
        lateral_y = project_perp(state["y_axis"], shaft)
        pivot_y_target = rcm_point + depth * unit(shaft + args.pivot_step_mm * lateral_y)
        rows.append(
            run_motion(
                controller,
                "pivot_y_small",
                pivot_y_target,
                rcm_point,
                initial,
                args,
            )
        )

    controller._stop()
    print("\nSummary")
    for row in rows:
        print(f"{row['label']}:")
        print_metric_row(row)

    csv_path = write_csv(rows, args)
    print(f"\nCSV log written to: {csv_path}")
    print("Pass criteria for early testing: low RCM-line error and visually fixed trocar point.")


if __name__ == "__main__":
    main()
