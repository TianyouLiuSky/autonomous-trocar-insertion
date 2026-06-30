#!/usr/bin/env python3
"""
Small physical check for the manually measured tool-tip offset.

Place the needle/trocar tip near a visible mark with no tissue contact. The
script keeps the computed physical tip fixed while applying a tiny orientation
sweep. If the real tip visibly sweeps around the mark, stop and revise the
temporary t_gripper_tip_mm value before running insertion sequences.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R

from SHER_Controller import SHERController
from needle_tip_rcm_sequence import (
    DEFAULT_PIVOT_JSON,
    DEFAULT_WORKSPACE_MAX_MM,
    DEFAULT_WORKSPACE_MIN_MM,
    append_stage_or_abort,
    current_pose,
    current_rotation,
    load_tip_offset,
    move_tip_pose,
    physical_tip_position,
    prompt,
    stop_robot,
    validate_args,
    workspace_enabled,
    workspace_summary,
)


LOG_DIR = Path(__file__).resolve().parent / "tip_offset_test_logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a tiny tip-centered sweep to validate the temporary TCP offset."
    )
    parser.add_argument("--robot-name", default="SHER20")
    parser.add_argument(
        "--pivot-calibration-json",
        default=str(DEFAULT_PIVOT_JSON),
        help="JSON containing t_gripper_tip_mm. Ignored if --tip-offset-mm is set.",
    )
    parser.add_argument(
        "--tip-offset-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Manual end-effector-to-physical-tip offset in gripper coordinates.",
    )
    parser.add_argument(
        "--target-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=None,
        help="Optional first orientation to reach about the current tip.",
    )
    parser.add_argument(
        "--sweep-axis",
        choices=("local-x", "local-y", "local-z"),
        default="local-y",
        help="Tool-local axis for the small validation rotation.",
    )
    parser.add_argument(
        "--sweep-deg",
        type=float,
        default=3.0,
        help="Magnitude of the plus/minus validation sweep.",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--position-gain", type=float, default=1.0)
    parser.add_argument("--orientation-gain", type=float, default=0.8)
    parser.add_argument("--max-linear-vel", type=float, default=0.3)
    parser.add_argument("--max-angular-vel", type=float, default=0.03)
    parser.add_argument("--position-tol-mm", type=float, default=0.05)
    parser.add_argument("--orientation-tol-deg", type=float, default=0.75)
    parser.add_argument("--settle-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--stop-on-handle-drift-mm",
        type=float,
        default=20.0,
        help="Stop if the FrameEE/gripper origin drifts this far during one stage.",
    )
    parser.add_argument(
        "--workspace-min-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_WORKSPACE_MIN_MM,
        help="Minimum allowed FrameEE/gripper position in robot-base millimeters.",
    )
    parser.add_argument(
        "--workspace-max-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_WORKSPACE_MAX_MM,
        help="Maximum allowed FrameEE/gripper position in robot-base millimeters.",
    )
    parser.add_argument("--workspace-tol-mm", type=float, default=0.5)
    parser.add_argument("--disable-workspace-check", action="store_true")
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def local_axis_vector(axis_name):
    axes = {
        "local-x": np.array([1.0, 0.0, 0.0]),
        "local-y": np.array([0.0, 1.0, 0.0]),
        "local-z": np.array([0.0, 0.0, 1.0]),
    }
    return axes[axis_name]


def write_csv(path, rows):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    validate_args(args)
    if args.sweep_deg <= 0.0:
        raise ValueError("--sweep-deg must be positive")
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive")

    tip_offset, tip_offset_source = load_tip_offset(args)
    if np.linalg.norm(tip_offset) == 0.0:
        raise ValueError("Tip offset is zero; there is no pivot geometry to test.")

    robot = SHERController(robot_name=args.robot_name)
    initial_pose = current_pose(robot)
    initial_rotation = current_rotation(robot)
    fixed_tip = physical_tip_position(initial_pose[:3], initial_rotation, tip_offset)

    print("\nManual tip-offset validation sweep")
    print("  robot: {}".format(args.robot_name))
    print("  initial gripper pose: {}".format(np.round(initial_pose, 4)))
    print("  computed fixed tip: {}".format(np.round(fixed_tip, 4)))
    print("  tip offset source: {}".format(tip_offset_source))
    print("  tip offset gripper mm: {}".format(np.round(tip_offset, 4)))
    print("  sweep: +/- {:.3f} deg about {}".format(args.sweep_deg, args.sweep_axis))
    print("  max linear vel: {:.4f} mm/s".format(args.max_linear_vel))
    print("  max angular vel: {:.4f} rad/s".format(args.max_angular_vel))
    if workspace_enabled(args):
        print("  FrameEE workspace: {}".format(workspace_summary(args)))
        print("  robot axes: +X in/forward, +Y left, +Z up")
    else:
        print("  FrameEE workspace check: disabled")

    if not args.yes:
        response = input(
            "\nPlace the physical tip just above a visible mark, with no tissue contact.\n"
            "The tip should stay on that mark while the handle moves. Type YES to run: "
        ).strip()
        if response != "YES":
            raise RuntimeError("Operator did not confirm the validation sweep")

    summary_rows = []
    sample_rows = []

    try:
        if args.target_rpy_deg is not None:
            target_rotation = R.from_euler("xyz", args.target_rpy_deg, degrees=True)
            prompt(
                args,
                "Setup: rotate to target RPY {} while holding the computed tip fixed.".format(
                    np.round(args.target_rpy_deg, 4)
                ),
            )
            append_stage_or_abort(
                summary_rows,
                move_tip_pose(
                    robot,
                    "setup_target_rpy_about_tip",
                    fixed_tip,
                    target_rotation,
                    tip_offset,
                    args,
                    sample_rows,
                ),
            )

        home_rotation = current_rotation(robot)
        axis = local_axis_vector(args.sweep_axis)
        plus_rotation = home_rotation * R.from_rotvec(np.deg2rad(args.sweep_deg) * axis)
        minus_rotation = home_rotation * R.from_rotvec(-np.deg2rad(args.sweep_deg) * axis)

        for cycle in range(1, args.cycles + 1):
            prompt(
                args,
                "Cycle {}: rotate +{:.3f} deg about {}.".format(
                    cycle,
                    args.sweep_deg,
                    args.sweep_axis,
                ),
            )
            append_stage_or_abort(
                summary_rows,
                move_tip_pose(
                    robot,
                    "cycle{}_plus_{}".format(cycle, args.sweep_axis),
                    fixed_tip,
                    plus_rotation,
                    tip_offset,
                    args,
                    sample_rows,
                ),
            )

            prompt(args, "Cycle {}: return to home orientation.".format(cycle))
            append_stage_or_abort(
                summary_rows,
                move_tip_pose(
                    robot,
                    "cycle{}_home_after_plus".format(cycle),
                    fixed_tip,
                    home_rotation,
                    tip_offset,
                    args,
                    sample_rows,
                ),
            )

            prompt(
                args,
                "Cycle {}: rotate -{:.3f} deg about {}.".format(
                    cycle,
                    args.sweep_deg,
                    args.sweep_axis,
                ),
            )
            append_stage_or_abort(
                summary_rows,
                move_tip_pose(
                    robot,
                    "cycle{}_minus_{}".format(cycle, args.sweep_axis),
                    fixed_tip,
                    minus_rotation,
                    tip_offset,
                    args,
                    sample_rows,
                ),
            )

            prompt(args, "Cycle {}: return to home orientation.".format(cycle))
            append_stage_or_abort(
                summary_rows,
                move_tip_pose(
                    robot,
                    "cycle{}_home_after_minus".format(cycle),
                    fixed_tip,
                    home_rotation,
                    tip_offset,
                    args,
                    sample_rows,
                ),
            )
    except KeyboardInterrupt:
        print("\nStopped by operator.")
    except RuntimeError as exc:
        print("\n{}".format(exc))
    finally:
        stop_robot(robot)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(args.log_dir).expanduser()
        summary_path = write_csv(log_dir / f"manual_tip_offset_test_summary_{stamp}.csv", summary_rows)
        sample_path = write_csv(log_dir / f"manual_tip_offset_test_samples_{stamp}.csv", sample_rows)
        if summary_path:
            print(f"Summary log -> {summary_path}")
        if sample_path:
            print(f"Samples log -> {sample_path}")


if __name__ == "__main__":
    main()
