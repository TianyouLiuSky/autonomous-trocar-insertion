#!/usr/bin/env python3
"""
Manual guided pivot calibration for ATI trocar TCP/F_tip.

This script does not command robot motion. It is intended for the workflow where
an operator physically holds or cooperatively moves the tool through a set of
orientation changes while the trocar tip remains seated in a fixed dimple.

At each prompt, settle the tool, keep the tip in the dimple, then press Enter to
record the current /<robot>/eye_robot/FrameEE pose. The script solves and saves
the same t_gripper_tip_mm output as pivot_calibration.py.
"""

import argparse
import os
import sys

import numpy as np

import pivot_calibration as pc


MANUAL_STEPS = [
    ("neutral", "Start with a comfortable neutral tool orientation."),
    ("roll_pos_large", "Roll the tool gently positive, about +10 deg if possible."),
    ("roll_neg_large", "Roll the tool gently negative, about -10 deg if possible."),
    ("pitch_pos_large", "Pitch the tool gently positive, about +10 deg if possible."),
    ("pitch_neg_large", "Pitch the tool gently negative, about -10 deg if possible."),
    ("roll_pos_pitch_pos", "Combine positive roll and positive pitch."),
    ("roll_pos_pitch_neg", "Combine positive roll and negative pitch."),
    ("roll_neg_pitch_pos", "Combine negative roll and positive pitch."),
    ("roll_neg_pitch_neg", "Combine negative roll and negative pitch."),
    ("roll_pos_small", "Return near center, then use a smaller positive roll."),
    ("roll_neg_small", "Return near center, then use a smaller negative roll."),
    ("pitch_pos_small", "Return near center, then use a smaller positive pitch."),
    ("pitch_neg_small", "Return near center, then use a smaller negative pitch."),
    ("neutral_repeat", "Return to neutral. This checks repeatability at the end."),
]


def _default_output_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "output")


def _topic_for_robot(robot_name):
    return "/{}/eye_robot/FrameEE".format(robot_name)


def _print_header(args, topic):
    print("\nManual ATI pivot calibration")
    print("=" * 72)
    print("Robot topic: {}".format(topic))
    print("Translation scale to mm: {}".format(args.translation_scale_to_mm))
    print("\nThis script will NOT move the robot.")
    print("You manually hold/cooperatively move the tool while the tip stays in")
    print("one fixed dimple. Press Enter only after each pose is settled.")
    print("\nCommands at each step:")
    print("  Enter  record current pose")
    print("  p      print current pose again")
    print("  s      skip this prompt")
    print("  b      delete previous sample and repeat this prompt")
    print("  q      stop early and solve with collected samples")
    print("=" * 72 + "\n")


def _quality_hint(result, residual_warn_mm):
    if result["max_residual_mm"] <= residual_warn_mm:
        print("Quality hint: residuals are within the {:.3f} mm warning threshold.".format(
            residual_warn_mm
        ))
    else:
        print("Quality hint: residuals are high. Recheck that the tip did not slide in")
        print("the dimple, the trocar mount is rigid, and the samples include enough")
        print("roll/pitch diversity.")


def run_manual(args):
    if pc.rospy is None:
        print("ERROR: rospy is not available. Run this on the ROS/SHER workstation.", file=sys.stderr)
        return 2

    topic = args.topic or _topic_for_robot(args.robot_name)
    pc.rospy.init_node("ati_manual_pivot_calibration", anonymous=True)
    listener = pc.RobotPoseListener(
        topic,
        translation_scale_to_mm=args.translation_scale_to_mm,
    )

    _print_header(args, topic)
    print("Waiting for first FrameEE pose...")
    while not pc.rospy.is_shutdown() and not listener.ready:
        pc.rospy.sleep(0.05)

    if pc.rospy.is_shutdown():
        return 1

    print("FrameEE received:")
    print("  {}".format(pc.current_pose_line(listener.get_pose())))

    samples = []
    labels = []
    step_i = 0

    while step_i < len(MANUAL_STEPS) and not pc.rospy.is_shutdown():
        label, instruction = MANUAL_STEPS[step_i]
        print("\nStep {}/{}: {}".format(step_i + 1, len(MANUAL_STEPS), label))
        print("  {}".format(instruction))
        print("  Keep the physical tip seated in the dimple.")

        cmd = input("  command> ").strip().lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd in ("p", "pose"):
            pose = listener.get_pose()
            if pose is None:
                print("  No pose available.")
            else:
                print("  {}".format(pc.current_pose_line(pose)))
            continue
        if cmd in ("s", "skip"):
            print("  Skipped.")
            step_i += 1
            continue
        if cmd in ("b", "back", "delete"):
            if samples:
                removed = labels.pop()
                samples.pop()
                print("  Deleted previous sample: {}".format(removed))
                step_i = max(0, step_i - 1)
            else:
                print("  No sample to delete.")
            continue
        if cmd not in ("", "r", "record"):
            print("  Unknown command: {}".format(cmd))
            continue

        pose = listener.get_pose()
        if pose is None:
            print("  No pose available; try again.")
            continue

        samples.append(pose)
        labels.append(label)
        print("  Recorded {}: {}".format(len(samples), pc.current_pose_line(pose)))

        if len(samples) >= 3 and args.preview:
            try:
                result = _solve(samples)
                print("  Preview RMS: {:.4f} mm | max: {:.4f} mm | rotation spread max: {:.2f} deg".format(
                    result["rms_residual_mm"],
                    result["max_residual_mm"],
                    result["max_pairwise_rotation_deg"],
                ))
            except Exception as exc:
                print("  Preview solve failed: {}".format(exc))

        step_i += 1

    if len(samples) < 3:
        print("\nNot enough samples to solve. Need at least 3; 12+ recommended.")
        return 1

    print("\nSolving manual pivot calibration from {} samples...".format(len(samples)))
    result = _solve(samples)
    pc.print_result(result, len(samples), args.residual_warn_mm)
    _quality_hint(result, args.residual_warn_mm)

    if len(samples) < args.min_samples:
        print("WARNING: only {} samples collected; {} or more is recommended.".format(
            len(samples), args.min_samples
        ))

    if result["max_pairwise_rotation_deg"] < args.min_rotation_spread_deg:
        print("WARNING: max orientation spread is only {:.2f} deg; target at least {:.2f} deg.".format(
            result["max_pairwise_rotation_deg"],
            args.min_rotation_spread_deg,
        ))

    if args.no_save:
        print("Not saving because --no-save was passed.")
        return 0

    paths = pc.save_solution(samples, result, args.output_dir, prefix="manual_pivot_calibration")
    _save_label_file(labels, paths["csv"] + ".labels.txt")

    print("Saved:")
    print("  NPZ : {}".format(paths["npz"]))
    print("  JSON: {}".format(paths["json"]))
    print("  CSV : {}".format(paths["csv"]))
    print("  Labels: {}".format(paths["csv"] + ".labels.txt"))
    return 0


def _solve(samples):
    positions = np.array([s["t_mm"] for s in samples], dtype=float)
    quats = np.array([s["q"] for s in samples], dtype=float)
    return pc.solve_pivot_calibration(positions, quats)


def _save_label_file(labels, path):
    with open(path, "w") as f:
        for i, label in enumerate(labels, 1):
            f.write("{} {}\n".format(i, label))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Manual guided ATI pivot calibration. No robot motion is commanded."
    )
    parser.add_argument("--robot-name", default=pc.DEFAULT_ROBOT_NAME,
                        help="Robot name used to build the default FrameEE topic.")
    parser.add_argument("--topic", default=None,
                        help="Override robot end-effector Transform topic.")
    parser.add_argument("--translation-scale-to-mm", type=float, default=1.0,
                        help="Scale Transform translations into mm. Use 1000 if the topic is in meters.")
    parser.add_argument("--min-samples", type=int, default=pc.DEFAULT_MIN_SAMPLES,
                        help="Recommended minimum sample count.")
    parser.add_argument("--min-rotation-spread-deg", type=float, default=15.0,
                        help="Warn if max pairwise orientation spread is below this value.")
    parser.add_argument("--residual-warn-mm", type=float, default=0.5,
                        help="Warn if max residual exceeds this value.")
    parser.add_argument("--output-dir", default=_default_output_dir(),
                        help="Directory for output files.")
    parser.add_argument("--no-save", action="store_true",
                        help="Solve and print results without saving files.")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="Disable quick residual preview after each sample.")
    parser.set_defaults(preview=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_manual(args)


if __name__ == "__main__":
    sys.exit(main())
