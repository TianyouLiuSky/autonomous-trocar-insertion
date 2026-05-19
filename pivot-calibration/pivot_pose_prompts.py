#!/usr/bin/env python3
"""
Print a conservative manual pose plan for pivot calibration.

This script does not move the robot. Before the tool-tip offset is known, a
commanded end-effector rotation can sweep the physical trocar tip. Use these
as operator prompts while keeping the trocar tip seated in the dimple.
"""

import argparse
import csv
import os


DEFAULT_SEQUENCE = [
    ("center", 0.0, 0.0),
    ("roll_pos", 10.0, 0.0),
    ("roll_neg", -10.0, 0.0),
    ("pitch_pos", 0.0, 10.0),
    ("pitch_neg", 0.0, -10.0),
    ("roll_pos_pitch_pos", 10.0, 10.0),
    ("roll_pos_pitch_neg", 10.0, -10.0),
    ("roll_neg_pitch_pos", -10.0, 10.0),
    ("roll_neg_pitch_neg", -10.0, -10.0),
    ("roll_small_pos", 5.0, 0.0),
    ("roll_small_neg", -5.0, 0.0),
    ("pitch_small_pos", 0.0, 5.0),
    ("pitch_small_neg", 0.0, -5.0),
]


def scaled_sequence(roll_deg, pitch_deg):
    sequence = []
    for name, r, p in DEFAULT_SEQUENCE:
        r_scale = 0.0 if abs(r) < 1e-9 else abs(r) / 10.0
        p_scale = 0.0 if abs(p) < 1e-9 else abs(p) / 10.0
        sequence.append((name, (1 if r >= 0 else -1) * roll_deg * r_scale,
                         (1 if p >= 0 else -1) * pitch_deg * p_scale))
    return sequence


def write_csv(sequence, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "delta_roll_deg", "delta_pitch_deg", "delta_yaw_deg"])
        for i, (label, d_roll, d_pitch) in enumerate(sequence, 1):
            writer.writerow([i, label, "{:.3f}".format(d_roll), "{:.3f}".format(d_pitch), "0.000"])


def main():
    parser = argparse.ArgumentParser(description="Print manual pivot-calibration pose prompts.")
    parser.add_argument("--roll-deg", type=float, default=10.0,
                        help="Large roll offset for prompt sequence.")
    parser.add_argument("--pitch-deg", type=float, default=10.0,
                        help="Large pitch offset for prompt sequence.")
    parser.add_argument("--csv", default=None,
                        help="Optional CSV path to save the prompt sequence.")
    args = parser.parse_args()

    sequence = scaled_sequence(args.roll_deg, args.pitch_deg)

    print("\nManual pivot calibration pose prompts")
    print("Keep the trocar tip seated in the dimple for every sample.")
    print("Do not execute these as Cartesian pose commands unless TCP is already known.\n")
    print("{:>3s}  {:24s}  {:>11s}  {:>12s}".format("#", "label", "d_roll_deg", "d_pitch_deg"))
    print("-" * 60)
    for i, (label, d_roll, d_pitch) in enumerate(sequence, 1):
        print("{:3d}  {:24s}  {:11.3f}  {:12.3f}".format(i, label, d_roll, d_pitch))

    if args.csv:
        write_csv(sequence, args.csv)
        print("\nSaved prompts to {}".format(args.csv))


if __name__ == "__main__":
    main()
