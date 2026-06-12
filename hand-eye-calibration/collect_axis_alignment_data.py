#!/usr/bin/env python3
"""Capture ChArUco and FrameEE samples for the axis-alignment test."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import rospy
from pyqtgraph.Qt import QtWidgets, QtGui

from collect_validation_data import DataCollectorGUI, timestamp


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"


def latest_sequence_path():
    matches = list(OUTPUT_DIR.glob("axis_alignment_sequence_*.json"))
    if not matches:
        raise ValueError(
            "No axis-alignment sequence was found under {}. Run "
            "run_axis_alignment_poses.py first.".format(OUTPUT_DIR)
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_sequence(path):
    resolved = (
        latest_sequence_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not resolved.is_file():
        raise ValueError(
            "Axis-alignment sequence is not a file: {}".format(resolved)
        )
    with resolved.open("r") as handle:
        sequence = json.load(handle)
    samples = sequence.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(
            "{} does not contain a non-empty samples list".format(resolved)
        )
    for expected_index, sample in enumerate(samples, start=1):
        if int(sample.get("sample", -1)) != expected_index:
            raise ValueError(
                "Sequence sample numbering is not contiguous at sample {}"
                .format(expected_index)
            )
    return resolved, sequence


class AxisAlignmentCollectorGUI(DataCollectorGUI):
    def __init__(
            self, sequence_path, intrinsics_path=None,
            require_fitted_intrinsics=False):
        self.sequence_path, self.sequence = load_sequence(sequence_path)
        self.plan_samples = self.sequence["samples"]
        self.saved = False
        super().__init__(
            "axis_alignment",
            len(self.plan_samples),
            intrinsics_path,
            require_fitted_intrinsics,
        )
        self.setWindowTitle("Axis Alignment Data Collector")
        self._show_next_sample()
        QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+S"),
            self,
            activated=self.save_and_exit,
        )

    def _show_next_sample(self):
        count = len(self.robot_poses)
        if count >= len(self.plan_samples):
            return
        sample = self.plan_samples[count]
        self.status.setText(
            "Captured: {} / {} | NEXT: {}".format(
                count, len(self.plan_samples), sample["label"])
        )

    def sample_metadata(self, count):
        sample = self.plan_samples[count - 1]
        return {
            "plan_label": sample["label"],
            "plan_kind": sample["kind"],
            "plan_axis": sample.get("axis", ""),
            "plan_sign": sample.get("sign", 0),
            "plan_repeat": sample.get("repeat", 0),
            "plan_offset_mm": sample.get("offset_mm", 0.0),
        }

    def record_data(self):
        previous_count = len(self.robot_poses)
        super().record_data()
        if len(self.robot_poses) > previous_count and not self.saved:
            self._show_next_sample()

    def save_and_exit(self):
        if self.saved:
            return
        count = len(self.robot_poses)
        if count == 0:
            print("Cannot save: no axis-alignment samples have been captured.")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = timestamp()
        latest_npz = SCRIPT_DIR / "axis_alignment_dataset.npz"
        timestamped_npz = (
            OUTPUT_DIR / "axis_alignment_dataset_{}.npz".format(stamp)
        )
        csv_path = (
            OUTPUT_DIR / "axis_alignment_samples_{}.csv".format(stamp)
        )

        payload = {
            "dataset_mode": np.array("axis_alignment"),
            "expected_samples": np.array(
                len(self.plan_samples), dtype=int),
            "captured_samples": np.array(count, dtype=int),
            "sequence_path": np.array(str(self.sequence_path)),
            "axis_alignment_plan": np.array(
                self.plan_samples[:count], dtype=object),
            "robot_poses": np.array(self.robot_poses, dtype=object),
            "board_rvecs": np.array(self.board_rvecs),
            "board_tvecs": np.array(self.board_tvecs),
            "corner_counts": np.array(self.corner_counts, dtype=int),
            "reprojection_errors_px": np.array(
                self.reprojection_errors_px, dtype=float),
            "diagnostic_rows": np.array(
                self.diagnostic_rows, dtype=object),
            "camera_matrix": self.camera.K,
            "dist_coeffs": self.camera.dist,
            "camera_intrinsics_source": np.array(
                self.camera.intrinsics_source),
        }
        np.savez(str(latest_npz), **payload)
        np.savez(str(timestamped_npz), **payload)

        if self.diagnostic_rows:
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(self.diagnostic_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(self.diagnostic_rows)

        self.saved = True
        complete = count == len(self.plan_samples)
        self.status.setText(
            "SAVED {} axis-alignment samples{}".format(
                count, "" if complete else " (PARTIAL)")
        )
        self.status.setStyleSheet(
            "font-size: 16pt; font-weight: bold; color: green;")
        print("\nSaved axis-alignment dataset:")
        print("  {}".format(timestamped_npz))
        print("  latest alias: {}".format(latest_npz))
        print("  samples: {}/{}".format(count, len(self.plan_samples)))
        print("  CSV: {}".format(csv_path))
        print(
            "Run analyze_axis_alignment.py after the complete sequence."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture the sequence generated by run_axis_alignment_poses.py."
        )
    )
    parser.add_argument(
        "--sequence", default=None,
        help=(
            "Axis-alignment sequence JSON generated by the motion script. "
            "Defaults to the newest output/axis_alignment_sequence_*.json."
        ),
    )
    parser.add_argument(
        "--intrinsics", default=os.environ.get("HE_CAMERA_INTRINSICS"),
        help=(
            "Optional fitted D405 intrinsics NPZ. This should match the "
            "orientation validation dataset used during analysis."
        ),
    )
    parser.add_argument(
        "--require-fitted-intrinsics", action="store_true",
        help="Refuse to start unless a fitted intrinsics NPZ is supplied.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rospy.init_node("axis_alignment_data_collector", anonymous=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui = AxisAlignmentCollectorGUI(
        args.sequence,
        args.intrinsics,
        args.require_fitted_intrinsics,
    )
    app.exec_()
