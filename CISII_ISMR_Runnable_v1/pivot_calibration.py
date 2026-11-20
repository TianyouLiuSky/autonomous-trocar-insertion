#!/usr/bin/env python3
"""Classical PIVOT CALIBRATION for the FrameEE -> physical needle-tip offset.

Why this exists (see chat/design notes 2026-08-11): the vision-based tools
(tip_click.py / tip_click_3p.py) solve a CONSTANT BASE-FRAME offset
  tip_base = FrameEE + w
which is only valid AT THE SPECIFIC ORIENTATION used during calibration
(every historical tip_click_samples.csv in this project confirms the
operator was told to "keep tip angle fixed, only translate" -- so the
EE-frame-rigid hypothesis was never actually tested with real rotational
diversity). It also couples the tip-offset error with hand-eye calibration
error and camera/click noise, since every observation goes through
R,t (handeye_T.npz) and a human-clicked pixel.

Pivot calibration is the standard, camera-free way to recover the TRUE
rigid EE-frame offset: physically pin the tip at ONE fixed point in space
(a divot/V-groove fixture, NOT attached to the robot) and SWEEP ORIENTATION
(pitch/yaw/roll) through as wide a range as safely possible while the tip
stays pinned. For sample i, let p_i = FrameEE translation (base frame, mm)
and R_i = FrameEE rotation (base frame). Since FrameEE already reports the
robot's OWN assumed tip position (firmware's ComputeFwdKinsTip bakes in a
nominal TipInEE, e.g. (0,0,-22) for the dummy tool -- see robotTask.cpp),
the "d" solved here is a RESIDUAL correction on top of whatever TipInEE the
firmware currently uses, expressed in the EE/flange frame (i.e. rigid,
valid at ANY orientation, unlike the old BASE-frame model):

    true_tip_base = p_i + R_i @ d   =   p_pivot   (constant, unknown)

for every sample, where d (3) and p_pivot (3) are the only unknowns. This
is LINEAR in [d; p_pivot] -- stack all samples and solve by ordinary least
squares (closed form, globally optimal, no nonlinear solver needed):

    [R_i | -I3] @ [d; p_pivot] = -p_i      for i = 1..N

NOTE on OffsetFinder (robotTask.cpp CM_OFFSET_FINDER): that GUI mode's
`tip_wander_mm` readout is NOT a valid correctness check for TipInEE -- its
hold controller actively servos the SAME assumed-tip quantity it reports as
"wander" toward its own latched setpoint, so wander trends toward zero from
closed-loop tracking regardless of whether TipInEE is actually correct.
This script instead does OPEN-LOOP, PASSIVE data collection: run your
regular jog control (jog_teleop.py or the k/m/j/l/u/o keys in
home_to_rcm_insert.py) in ANOTHER terminal, NOT OffsetFinder mode, so
nothing is closed-loop-holding a possibly-wrong assumed point. The physical
divot is what keeps the tip fixed, not the controller.

WORKFLOW:
  1. Physically rest the silver needle tip in a fixed divot/V-groove (fixed
     to the table/fixture, NOT to the robot).
  2. In ANOTHER terminal, run your usual jog control and use ONLY the
     rotation keys (k/m j/l u/o) to sweep pitch/yaw/roll. Do NOT translate
     (that would drag the tip out of the divot / snap it). Vary each axis
     by as much as is mechanically safe (aim for >=20-30 deg spread; more
     is better -- this script tracks and reports your current spread).
  3. Back in THIS terminal: after each new distinct orientation, tap SPACE
     to capture a burst (grabs ~0.3s of FrameEE @ ~100Hz and rejects the
     sample if the robot wasn't actually still -- same quality-gate
     philosophy as handeye_calibrate.py's burst check). Get >=8-10 GOOD
     samples spanning a wide orientation range.
  4. Press 's' to solve + save -> tip_offset_pivot.npz. The script also
     cross-checks against the existing tip_offset.npz / tip_click_samples.csv
     (the old vision-based calibration) by evaluating the NEW model at the
     OLD calibration's orientation and comparing to the OLD w -- agreement
     between two independent methods (kinematic pivot vs vision reprojection)
     is strong evidence the number is real, not a coincidence of one method.

Keys:
  SPACE / c : capture a burst sample at the CURRENT live pose
  u         : undo last sample
  s         : solve + save -> tip_offset_pivot.npz
  p         : print current fit (same as after a capture, without adding one)
  q / ESC   : quit

Run in YOUR OWN terminal (foreground), with jog control running in ANOTHER
terminal (NOT OffsetFinder mode):
  source /opt/ros/melodic/setup.bash
  source /home/eyebrp/SHER_I2RIS_teleoperate/devel/setup.bash
  python3 pivot_calibration.py
"""
from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import os
import sys
import csv as _csv
import select
import termios
import time
import tty

import numpy as np
import rospy
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMEEE_TOPIC = os.environ.get("FRAMEEE_TOPIC", "/SHER21/eye_robot/FrameEE")
OUT_NPZ = data_path("tip_offset_pivot.npz")
CSV_PATH = data_path("pivot_samples.csv")
OLD_TIP_OFFSET_NPZ = data_path("tip_offset.npz")
OLD_TIP_CLICK_CSV = data_path("tip_click_samples.csv")

BURST_N = int(os.environ.get("PIVOT_BURST_N", "30"))          # samples per capture (~0.3s @100Hz)
BURST_HZ = float(os.environ.get("PIVOT_BURST_HZ", "100.0"))
MAX_POS_JITTER_MM = float(os.environ.get("PIVOT_MAX_POS_JITTER_MM", "0.15"))   # "robot truly still" gate
MAX_ANG_JITTER_DEG = float(os.environ.get("PIVOT_MAX_ANG_JITTER_DEG", "0.5"))  # same, in orientation
MIN_DISTINCT_DEG = float(os.environ.get("PIVOT_MIN_DISTINCT_DEG", "3.0"))     # min angle vs prior samples
RESUME = os.environ.get("RESUME", "1") != "0"

_ee = {"p": None, "q": None}


def _cb(m):
    _ee["p"] = np.array([m.translation.x, m.translation.y, m.translation.z], float)
    _ee["q"] = np.array([m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w], float)


def _getch(timeout):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        return sys.stdin.read(1)
    return None


def quat_angle_deg(q1, q2):
    """Angle (deg) of the relative rotation between two unit quaternions."""
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, d)
    return float(np.degrees(2.0 * np.arccos(d)))


def orientation_spread_deg(quats):
    """Max pairwise relative-rotation angle across all samples (deg)."""
    n = len(quats)
    if n < 2:
        return 0.0
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            best = max(best, quat_angle_deg(quats[i], quats[j]))
    return best


class Ctx:
    samples = []   # list of dict(p=[3], q=[4])


def capture_burst():
    """Grab BURST_N FrameEE readings over ~BURST_N/BURST_HZ seconds and
    return the averaged (p, q), or None if the robot/tip wasn't actually
    still (position or orientation jitter above threshold) -- same
    "reject unstable bursts" philosophy as handeye_calibrate.py, adapted
    to pure kinematic (no vision) data."""
    ps, qs = [], []
    rate = rospy.Rate(BURST_HZ)
    for _ in range(BURST_N):
        if _ee["p"] is None:
            return None, "no FrameEE yet (bringup running?)"
        ps.append(_ee["p"].copy())
        qs.append(_ee["q"].copy())
        rate.sleep()
    ps = np.array(ps)
    qs = np.array(qs)
    pos_jitter = float(np.linalg.norm(ps.std(axis=0)))
    # orientation jitter: max pairwise angle within the burst
    ang_jitter = 0.0
    for i in range(0, len(qs), max(1, len(qs) // 8)):
        ang_jitter = max(ang_jitter, quat_angle_deg(qs[0], qs[i]))
    if pos_jitter > MAX_POS_JITTER_MM:
        return None, "REJECTED: position jitter %.3fmm > %.3fmm -- robot wasn't still" % (
            pos_jitter, MAX_POS_JITTER_MM)
    if ang_jitter > MAX_ANG_JITTER_DEG:
        return None, "REJECTED: orientation jitter %.3fdeg > %.3fdeg -- robot wasn't still" % (
            ang_jitter, MAX_ANG_JITTER_DEG)
    p_mean = ps.mean(axis=0)
    # average quaternion: eigenvector of largest eigenvalue of sum(q q^T)
    # (proper way to average rotations close together; a plain component
    # mean is only a good approximation for small spreads within one burst)
    M = np.einsum("ni,nj->ij", qs, qs)
    w, v = np.linalg.eigh(M)
    q_mean = v[:, np.argmax(w)]
    if np.dot(q_mean, qs[0]) < 0:
        q_mean = -q_mean
    return {"p": p_mean, "q": q_mean, "pos_jitter": pos_jitter, "ang_jitter": ang_jitter}, None


def build_and_solve(samples, with_cov=False):
    """Linear pivot-calibration solve: p_i + R_i @ d = p_pivot for all i.
    Returns (d, p_pivot, fit_rms_mm, per_sample_residual_mm[, cov6])."""
    n = len(samples)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    Rs = []
    for i, s in enumerate(samples):
        R = Rot.from_quat(s["q"]).as_matrix()
        Rs.append(R)
        A[3 * i:3 * i + 3, 0:3] = R
        A[3 * i:3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i:3 * i + 3] = -s["p"]
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    d = x[0:3]
    p_pivot = x[3:6]
    resid = np.zeros(n)
    for i, s in enumerate(samples):
        pred = s["p"] + Rs[i].dot(d)
        resid[i] = np.linalg.norm(pred - p_pivot)
    fit_rms = float(np.sqrt(np.mean(resid ** 2)))
    if not with_cov:
        return d, p_pivot, fit_rms, resid
    # Standard linear-regression covariance: cov(x) = sigma^2 * (A^T A)^-1,
    # sigma^2 = RSS / dof. RSS uses the FULL 3n-vector of per-axis residuals
    # (A@x - b), not the per-sample 3D-norm `resid` above (norm-of-residual
    # loses the sign info needed for an unbiased sigma^2).
    dof = max(1, 3 * n - 6)
    rss = float(np.sum((A.dot(x) - b) ** 2))
    sigma2 = rss / dof
    try:
        cov6 = sigma2 * np.linalg.pinv(A.T.dot(A))
    except np.linalg.LinAlgError:
        cov6 = np.full((6, 6), np.nan)
    return d, p_pivot, fit_rms, resid, cov6


def loo_rms(samples):
    n = len(samples)
    if n < 5:
        return None
    errs = np.zeros(n)
    for i in range(n):
        train = [s for j, s in enumerate(samples) if j != i]
        d_i, p_piv_i, _, _ = build_and_solve(train)
        R_i = Rot.from_quat(samples[i]["q"]).as_matrix()
        pred = samples[i]["p"] + R_i.dot(d_i)
        errs[i] = np.linalg.norm(pred - p_piv_i)
    return float(np.sqrt(np.mean(errs ** 2)))


def cross_check_vs_old(d):
    """Evaluate the NEW EE-frame model at the OLD calibration's orientation
    and compare to the OLD saved BASE-frame constant w. Independent-method
    agreement (kinematic pivot vs vision reprojection) is the strongest
    evidence either number is trustworthy."""
    if not os.path.isfile(OLD_TIP_OFFSET_NPZ) or not os.path.isfile(OLD_TIP_CLICK_CSV):
        print("  (no existing tip_offset.npz / tip_click_samples.csv found -- skipping cross-check)")
        return
    old = np.load(OLD_TIP_OFFSET_NPZ)
    w_old = np.asarray(old["w"], float)
    quats = []
    with open(OLD_TIP_CLICK_CSV) as f:
        for row in _csv.DictReader(f):
            quats.append([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])])
    if not quats:
        print("  (old tip_click_samples.csv has no rows -- skipping cross-check)")
        return
    quats = np.array(quats)
    q_ref = quats[0]
    R_ref = Rot.from_quat(q_ref).as_matrix()
    w_pred = R_ref.dot(d)
    diff = float(np.linalg.norm(w_pred - w_old))
    print("\n  === cross-check vs OLD vision-based tip_offset.npz ===")
    print("    OLD w (base frame, @ old calib orientation) = [%.2f %.2f %.2f] mm" % tuple(w_old))
    print("    NEW model evaluated at that SAME orientation = [%.2f %.2f %.2f] mm" % tuple(w_pred))
    print("    |difference| = %.2f mm  %s" % (
        diff, "-- AGREE (independent methods confirm each other)" if diff < 2.0
        else "-- DISAGREE, investigate (hand-eye chain error? real non-rigidity? old data had zero "
             "orientation diversity so its own validity is untested)"))


def print_fit(samples):
    n = len(samples)
    if n < 2:
        print("  need >=2 samples to solve (have %d)" % n)
        return
    spread = orientation_spread_deg([s["q"] for s in samples])
    d, p_pivot, fit_rms, resid = build_and_solve(samples)
    print("  n=%d  orientation spread=%.1fdeg%s" % (
        n, spread, "  (recommend >=20-30deg, more is better)" if spread < 20 else ""))
    print("  d (EE-frame correction, mm) = [%.3f %.3f %.3f]  |d|=%.2fmm" % (
        d[0], d[1], d[2], np.linalg.norm(d)))
    print("  p_pivot (base frame, mm)    = [%.3f %.3f %.3f]" % tuple(p_pivot))
    print("  fit RMS = %.3f mm   per-sample residuals(mm) = %s" %
          (fit_rms, np.array2string(resid, precision=2)))
    if n >= 5:
        lr = loo_rms(samples)
        print("  LOO RMS = %.3f mm  (this, not fit RMS, is the trustworthy number)" % lr)
    else:
        print("  (need >=5 samples for LOO cross-validation)")
    # diagnostic: does residual grow with angular distance from the mean
    # orientation? (would suggest real non-rigidity / backlash rather than
    # plain measurement noise)
    quats = np.array([s["q"] for s in samples])
    M = np.einsum("ni,nj->ij", quats, quats)
    w, v = np.linalg.eigh(M)
    q_mean = v[:, np.argmax(w)]
    ang_from_mean = np.array([quat_angle_deg(q_mean, q) for q in quats])
    order = np.argsort(ang_from_mean)
    print("  residual vs angle-from-mean-orientation (sorted, for non-rigidity check):")
    for i in order:
        print("    angle=%5.1fdeg  residual=%.3fmm" % (ang_from_mean[i], resid[i]))
    return d, p_pivot, fit_rms


def main():
    rospy.init_node("pivot_calibration", anonymous=True, disable_signals=True)
    from tip_inee_guard import assert_tip_inee
    tip_inee_now = assert_tip_inee(context="pivot_calibration.py start")
    rospy.Subscriber(FRAMEEE_TOPIC, Transform, _cb)
    print(__doc__)
    time.sleep(0.5)
    if _ee["p"] is None:
        print("  !! no FrameEE yet on %s -- is bringup running?" % FRAMEEE_TOPIC)

    if RESUME and os.path.isfile(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in _csv.DictReader(f):
                Ctx.samples.append({
                    "p": np.array([float(row["px"]), float(row["py"]), float(row["pz"])]),
                    "q": np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]),
                })
        if Ctx.samples:
            print("  resumed %d existing samples from %s" % (len(Ctx.samples), CSV_PATH))
            print_fit(Ctx.samples)
    else:
        with open(CSV_PATH, "w") as f:
            f.write("px,py,pz,qx,qy,qz,qw,pos_jitter_mm,ang_jitter_deg\n")

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("\nready. SPACE/c=capture  u=undo  s=solve+save  p=print  q=quit\n")
        while not rospy.is_shutdown():
            c = _getch(0.15)
            if c is None:
                continue
            if c in ("q", "\x1b"):
                break
            elif c in (" ", "c"):
                s, err = capture_burst()
                if err:
                    print("  " + err)
                    continue
                if Ctx.samples:
                    d_ang = min(quat_angle_deg(s["q"], o["q"]) for o in Ctx.samples)
                    if d_ang < MIN_DISTINCT_DEG:
                        print("  !! only %.1fdeg from an existing sample (<%.1f) -- "
                              "rotate more before capturing, or this sample adds little info"
                              % (d_ang, MIN_DISTINCT_DEG))
                Ctx.samples.append(s)
                with open(CSV_PATH, "a") as f:
                    f.write("%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (
                        s["p"][0], s["p"][1], s["p"][2],
                        s["q"][0], s["q"][1], s["q"][2], s["q"][3],
                        s["pos_jitter"], s["ang_jitter"]))
                print("\n  captured sample #%d (pos_jitter=%.3fmm ang_jitter=%.3fdeg)" %
                      (len(Ctx.samples), s["pos_jitter"], s["ang_jitter"]))
                print_fit(Ctx.samples)
            elif c == "u" and Ctx.samples:
                Ctx.samples.pop()
                print("  undo -> %d samples" % len(Ctx.samples))
                if Ctx.samples:
                    print_fit(Ctx.samples)
            elif c == "p":
                print_fit(Ctx.samples)
            elif c == "s":
                if len(Ctx.samples) < 2:
                    print("  need >=2 samples")
                    continue
                out = print_fit(Ctx.samples)
                if out is None:
                    continue
                d, p_pivot, fit_rms = out
                lr = loo_rms(Ctx.samples) if len(Ctx.samples) >= 5 else float("nan")
                spread = orientation_spread_deg([s["q"] for s in Ctx.samples])
                _, _, _, _, cov6 = build_and_solve(Ctx.samples, with_cov=True)
                # Re-check TipInEE right before saving too, in case it was
                # changed mid-session after main() started (this is exactly
                # how the 2026-08-11 -> 2026-08-23 drift went undetected).
                tip_inee_at_save = assert_tip_inee(context="pivot_calibration.py solve/save")
                np.savez(OUT_NPZ, d=d, p_pivot=p_pivot, fit_rms=fit_rms, loo_rms=lr,
                         n=len(Ctx.samples), orientation_spread_deg=spread, param_cov6=cov6,
                         tip_inee_at_calib=tip_inee_at_save)
                print("\n  SAVED -> %s  (tip_inee_at_calib=%s)" % (OUT_NPZ, tip_inee_at_save))
                cross_check_vs_old(d)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print("\nbye.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
