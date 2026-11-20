#!/usr/bin/env python3
"""tip_pin_check_v2.py + full raw-pose logging, for feeding pivot_residual_fit.py.

WHY THIS EXISTS (2026-08-25): tip_pin_check_v2.py only logs the DERIVED tip
position (p_ee + R_ee @ d) and its drift from the first sample -- it never
saves the raw FrameEE translation/quaternion. That's enough to get a
GOOD/MARGINAL/POOR verdict on the CURRENT rigid `d`, but it throws away
exactly the information (R_ee per sample) needed to ask "does the error
depend on orientation, and if so how" -- which is the question that matters
once the rigid model has been independently shown to fail (see
pivot_residual_fit.py's docstring for the 2026-08-25 evidence: 3-4 stable,
non-noisy, non-backlash-jump drift PLATEAUS at different held orientations).

This script is IDENTICAL to tip_pin_check_v2.py in behavior (same live
printout, same GOOD/MARGINAL/POOR verdict, same TIP_D/tip_offset_pivot.npz
loading) except the CSV log additionally carries the raw ee_px/py/pz and
ee_qx/qy/qz/qw for every sample, so this file can be fed straight into
pivot_residual_fit.py as one more "group" (one pinned point, orientation
swept) alongside pivot_samples.csv.

PROTOCOL FOR A USEFUL RUN: pin the tip at ONE fixed point (does not need to
be the same point pivot_calibration.py used -- this script solves its own
p_pivot per file) and sweep orientation SLOWLY and CONTINUOUSLY (not just a
few held poses) to maximize how much of the orientation space gets covered
in one file -- dwell at each interesting pose for 2-3s so the 5Hz samples
average out any tremor, then keep moving. More continuous coverage here is
worth more to pivot_residual_fit.py than repeated samples at similar poses.

USAGE: identical to tip_pin_check_v2.py, but ALWAYS set TIP_PIN_LOG_CSV (the
whole point of this script is the log):
  TIP_PIN_LOG_CSV=/tmp/tip_pin_fullpose_$(date +%Y%m%d_%H%M).csv \\
      python3 tip_pin_check_v3_fullpose.py

Ctrl-C to quit.
"""
from __future__ import print_function
import csv
import os
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import Transform

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEEE_TOPIC = os.environ.get("FRAMEEE_TOPIC", "/SHER21/eye_robot/FrameEE")
TIP_OFFSET_PIVOT_NPZ = os.environ.get(
    "TIP_OFFSET_PIVOT_NPZ", os.path.join(HERE, "tip_offset_pivot.npz"))
LOG_CSV = os.environ.get("TIP_PIN_LOG_CSV", "").strip()

_ee = {"p": None, "q": None}


def _cb(m):
    _ee["p"] = np.array([m.translation.x, m.translation.y, m.translation.z], float)
    _ee["q"] = np.array([m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w], float)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def verdict_for(max_drift_mm):
    if max_drift_mm is None:
        return "UNKNOWN"
    if max_drift_mm < 0.3:
        return "GOOD"
    if max_drift_mm < 0.8:
        return "MARGINAL"
    return "POOR"


def main():
    if not LOG_CSV:
        print("!! TIP_PIN_LOG_CSV not set -- this script exists specifically to log full pose, "
              "set it (see docstring) or just use tip_pin_check_v2.py for a live-only check.")

    z = np.load(TIP_OFFSET_PIVOT_NPZ)
    d = np.asarray(z["d"], float)
    fit_rms = float(z["fit_rms"]) if "fit_rms" in z else None
    loo_rms = float(z["loo_rms"]) if "loo_rms" in z else None
    print("Loaded TIP_D = [%.3f %.3f %.3f] mm (EE-frame rigid) from %s" %
          (d[0], d[1], d[2], TIP_OFFSET_PIVOT_NPZ))
    print("fit_rms=%.3fmm  loo_rms=%.3fmm  n=%d  orientation_spread=%.1fdeg"
          % (fit_rms if fit_rms is not None else float("nan"),
             loo_rms if loo_rms is not None else float("nan"),
             int(z["n"]), float(z["orientation_spread_deg"])))
    print()
    print("Pin the tip against a fixed point now. SWEEP ORIENTATION CONTINUOUSLY, dwelling "
          "2-3s per interesting pose, to maximize coverage in this one log. Reading FrameEE...")

    csv_writer = None
    csv_file = None
    if LOG_CSV:
        is_new = not os.path.isfile(LOG_CSV)
        csv_file = open(LOG_CSV, "a")
        csv_writer = csv.writer(csv_file)
        if is_new:
            csv_writer.writerow(["timestamp", "t_s",
                                  "ee_px", "ee_py", "ee_pz",
                                  "ee_qx", "ee_qy", "ee_qz", "ee_qw",
                                  "tip_x_mm", "tip_y_mm", "tip_z_mm", "drift_mm"])
        print("Logging every sample (full pose + derived tip) to %s" % LOG_CSV)

    rospy.init_node("tip_pin_check_v3_fullpose", anonymous=True, disable_signals=True)
    rospy.Subscriber(FRAMEEE_TOPIC, Transform, _cb, queue_size=1)
    rospy.sleep(1.0)
    if _ee["p"] is None:
        raise SystemExit("no FrameEE received -- is ROS bringup running?")

    ref = None
    n_samples = 0
    max_drift = None
    max_drift_t = None
    rate = rospy.Rate(5.0)
    t0 = time.time()
    try:
        while not rospy.is_shutdown():
            p_now, q_now = _ee["p"].copy(), _ee["q"].copy()
            R = quat_to_R(q_now)
            tip = p_now + R.dot(d)
            if ref is None:
                ref = tip.copy()
                print("Reference tip (t=0s) = [%.2f %.2f %.2f] mm" % tuple(ref))
                print()
                print("  t(s)   tip_x    tip_y    tip_z   |drift from ref| (mm)")
            drift = float(np.linalg.norm(tip - ref))
            t_now = time.time() - t0
            n_samples += 1
            if max_drift is None or drift > max_drift:
                max_drift = drift
                max_drift_t = t_now
            print("  %5.1f  %7.2f  %7.2f  %7.2f    %6.3f mm" %
                  (t_now, tip[0], tip[1], tip[2], drift))
            if csv_writer is not None:
                csv_writer.writerow(["%.6f" % time.time(), "%.2f" % t_now,
                                      "%.4f" % p_now[0], "%.4f" % p_now[1], "%.4f" % p_now[2],
                                      "%.6f" % q_now[0], "%.6f" % q_now[1],
                                      "%.6f" % q_now[2], "%.6f" % q_now[3],
                                      "%.3f" % tip[0], "%.3f" % tip[1], "%.3f" % tip[2],
                                      "%.4f" % drift])
                csv_file.flush()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file is not None:
            csv_file.close()

    verdict = verdict_for(max_drift)
    print()
    print("=== tip_pin_check SUMMARY ===")
    print("  samples=%d  duration=%.1fs" % (n_samples, (time.time() - t0)))
    if max_drift is not None:
        print("  max drift from reference = %.3fmm (at t=%.1fs)" % (max_drift, max_drift_t))
    print("  verdict: %s  (thresholds: <0.3mm GOOD, <0.8mm MARGINAL, else POOR)" % verdict)
    if LOG_CSV:
        print("  full-pose log saved -> %s" % LOG_CSV)
        print("  next step: python3 pivot_residual_fit.py pivot_samples.csv %s" % LOG_CSV)


if __name__ == "__main__":
    main()
