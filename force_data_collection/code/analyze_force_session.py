#!/usr/bin/env python3
"""Print a concise integrity report for a saved force collection session."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from force_collection_common import FORCE_CHANNEL_COUNT, finite_stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    return parser.parse_args()


def load_rows(path):
    with Path(path).open(newline="") as source:
        return list(csv.DictReader(source))


def numeric(rows, column):
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            values.append(math.nan)
    return np.asarray(values, dtype=float)


def main():
    args = parse_args()
    session = Path(args.session_dir).expanduser()
    metadata_path = session / "metadata.json"
    samples_path = session / "force_samples.csv"
    if not metadata_path.exists() or not samples_path.exists():
        raise SystemExit(
            "Expected metadata.json and force_samples.csv in {}".format(session)
        )

    with metadata_path.open() as source:
        metadata = json.load(source)
    rows = load_rows(samples_path)

    print("Session: {}".format(session))
    print("Robot: {}".format(metadata.get("robot_name")))
    print("Target angle: {} deg".format(metadata.get("target_entry_angle_deg")))
    print("Samples: {}".format(len(rows)))
    if not rows:
        return

    elapsed = numeric(rows, "elapsed_s")
    pose_age = numeric(rows, "pose_age_s")
    depth = numeric(rows, "insertion_depth_mm")
    dt = np.diff(elapsed[np.isfinite(elapsed)])
    positive_dt = dt[dt > 0]
    print("Duration: {:.3f} s".format(np.nanmax(elapsed)))
    if positive_dt.size:
        print(
            "Force callback rate: median {:.1f} Hz".format(
                1.0 / np.median(positive_dt)
            )
        )
    print(
        "Pose age: median {:.4f} s, max {:.4f} s".format(
            np.nanmedian(pose_age), np.nanmax(pose_age)
        )
    )
    print(
        "Insertion depth: min {:.4f} mm, max {:.4f} mm".format(
            np.nanmin(depth), np.nanmax(depth)
        )
    )

    for channel in range(FORCE_CHANNEL_COUNT):
        stats = finite_stats(numeric(rows, "force_delta_{}".format(channel)))
        print(
            "Channel {} delta: min={:.6g}, max={:.6g}, p2p={:.6g}".format(
                channel,
                stats["minimum"],
                stats["maximum"],
                stats["peak_to_peak"],
            )
        )


if __name__ == "__main__":
    main()
