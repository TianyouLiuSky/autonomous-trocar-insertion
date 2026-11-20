#!/usr/bin/env python3
"""Headless, single-shot pivot-calibration sample capture: burst-average
FrameEE (p, q) over ~0.3s, reject if the robot wasn't actually still, and
APPEND to a CSV in the same schema pivot_calibration.py/build_and_solve()
expects. Exists so a human can physically pin the needle tip against a fixed
point, sweep to a NEW orientation, and capture that sample non-interactively
from another terminal -- no shared keyboard/GUI focus needed, camera-free
(doesn't conflict with anything using the RealSense).

Usage:
  python3 pivot_capture_one.py [out_csv]   # default: pivot_samples_30deg.csv

Prints n_valid, jitter, and a running orientation-spread number (max pairwise
rotation angle vs all prior samples in the file) so you know when you've
covered enough of the target range.
"""
from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import csv
import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import Transform

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else data_path("pivot_samples_30deg.csv")

BURST_N = int(os.environ.get("PIVOT_BURST_N", "30"))
BURST_HZ = float(os.environ.get("PIVOT_BURST_HZ", "100.0"))
MAX_POS_JITTER_MM = float(os.environ.get("PIVOT_MAX_POS_JITTER_MM", "0.15"))
MAX_ANG_JITTER_DEG = float(os.environ.get("PIVOT_MAX_ANG_JITTER_DEG", "0.5"))

_ee = {"p": None, "q": None}


def _cb(m):
    _ee["p"] = np.array([m.translation.x, m.translation.y, m.translation.z], float)
    _ee["q"] = np.array([m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w], float)


def quat_angle_deg(q1, q2):
    d = min(1.0, abs(float(np.dot(q1, q2))))
    return float(np.degrees(2.0 * np.arccos(d)))


def main():
    rospy.init_node("pivot_capture_one", anonymous=True, disable_signals=True)
    from tip_inee_guard import assert_tip_inee
    assert_tip_inee(context="pivot_capture_one")
    rospy.Subscriber("/SHER21/eye_robot/FrameEE", Transform, _cb, queue_size=1)
    rospy.sleep(1.0)
    if _ee["p"] is None:
        raise SystemExit("no FrameEE received -- is ROS bringup running?")

    ps, qs = [], []
    rate = rospy.Rate(BURST_HZ)
    for _ in range(BURST_N):
        ps.append(_ee["p"].copy())
        qs.append(_ee["q"].copy())
        rate.sleep()
    ps = np.array(ps)
    qs = np.array(qs)
    pos_jitter = float(np.linalg.norm(ps.std(axis=0)))
    ang_jitter = 0.0
    for i in range(0, len(qs), max(1, len(qs) // 8)):
        ang_jitter = max(ang_jitter, quat_angle_deg(qs[0], qs[i]))

    if pos_jitter > MAX_POS_JITTER_MM:
        raise SystemExit("REJECTED: position jitter %.3fmm > %.3fmm -- tip wasn't pinned "
                          "still (or robot moved)" % (pos_jitter, MAX_POS_JITTER_MM))
    if ang_jitter > MAX_ANG_JITTER_DEG:
        raise SystemExit("REJECTED: orientation jitter %.3fdeg > %.3fdeg -- robot wasn't "
                          "still" % (ang_jitter, MAX_ANG_JITTER_DEG))

    p_mean = ps.mean(axis=0)
    M = np.einsum("ni,nj->ij", qs, qs)
    w, v = np.linalg.eigh(M)
    q_mean = v[:, np.argmax(w)]
    if np.dot(q_mean, qs[0]) < 0:
        q_mean = -q_mean

    prior = []
    if os.path.isfile(OUT_CSV):
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                prior.append(np.array([float(row["qx"]), float(row["qy"]),
                                        float(row["qz"]), float(row["qw"])]))
    spread = 0.0
    for q_old in prior:
        spread = max(spread, quat_angle_deg(q_old, q_mean))

    new_file = not os.path.isfile(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        w2 = csv.writer(f)
        if new_file:
            w2.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw",
                         "pos_jitter_mm", "ang_jitter_deg"])
        w2.writerow([p_mean[0], p_mean[1], p_mean[2], q_mean[0], q_mean[1], q_mean[2],
                     q_mean[3], pos_jitter, ang_jitter])

    print("Captured OK: pos_jitter=%.3fmm ang_jitter=%.3fdeg" % (pos_jitter, ang_jitter))
    print("Max angle vs %d prior sample(s) in %s: %.1fdeg" % (len(prior), OUT_CSV, spread))
    print("Appended -> %s (n=%d now)" % (OUT_CSV, len(prior) + 1))


if __name__ == "__main__":
    main()
