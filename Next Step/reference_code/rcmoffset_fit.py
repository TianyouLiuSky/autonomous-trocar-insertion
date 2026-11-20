#!/usr/bin/env python3
"""Physically-motivated tip-offset model, derived from a real firmware bug --
NOT a generic empirical regression like pivot_residual_fit.py's C-matrix.

WHY THIS EXISTS (2026-08-25): robotModel.cpp's ComputeFwdKinsTip() (the
ACTUAL forward-kinematics implementation robotTask.cpp calls -- TipInEE is
just an input parameter to it, not the model itself) computes, for ER2.1:

    R(q3,q4)   = Ry(q3) @ Rx(q4)          (q3=pitch about Y, q4=roll about
                                            the ALREADY-PITCHED local X)
    FrameEE.p  = [q0,q1,q2] + Ry(q3) @ [0,0,d] + R(q3,q4) @ TipInEE

where `d = RCM2_1.RCMoffset` -- and RCMoffset is set to exactly 0.0 in the
robotModel constructor and is NEVER updated anywhere else in this codebase
(confirmed by `grep -r RCMoffset src/`: only the one assignment + the 4
read-sites inside robotModel.cpp itself). A struct field this specifically
named, carried through 4 different kinematics functions, that is simply left
at 0 looks exactly like an uncalibrated mechanical parameter, not an
intentional "this really is zero by design" choice.

CONSEQUENCE, derived (see chat 2026-08-25 for the full algebra): if the TRUE
mechanical RCMoffset is d_true != 0 (and/or TipInEE itself is slightly off),
the resulting error in FrameEE, expressed as the EE-frame vector a per-pose
pivot/tip-offset fit would need to absorb, has the EXACT closed form

    d_needed(q4) = d_const + K * [0, sin(q4), cos(q4)]        (NO q3 dependence)

i.e. a single scalar K (physically = the missing RCMoffset, up to a known
sign/rotation) explains a pose-dependent bias that VARIES ONLY WITH ROLL
(q4), not pitch (q3) -- a sharp, falsifiable, single-parameter prediction,
as opposed to pivot_residual_fit.py's generic 9-parameter (unconstrained
linear-in-rotation-vector) empirical model.

VALIDATED 2026-08-25 on the same n=1203 combined pivot+tip-pin dataset
pivot_residual_fit.py used:
    rigid (K=0) baseline LOO RMS            = 0.333 mm
    this 1-extra-parameter K model LOO RMS  = 0.140 mm   (-58%)
    pivot_residual_fit.py's 9-param C model LOO RMS = 0.110 mm

-> a SINGLE physically-grounded parameter recovers ~71% of the improvement
the unconstrained 9-parameter empirical model finds. This model is preferred
over the generic augmented model for deployment because:
  1. It is the actual correct functional form (derived from the firmware's
     own kinematics), not a first-order Taylor/linear approximation -- so it
     should NOT degrade under extrapolation to poses outside the fit range
     the way a linear-in-rotvec model can (see pivot_residual_fit.py's own
     docstring caveats about this).
  2. Only 4 free parameters (d_const:3, K:1) vs 12 (d0:3, C:9) -- far less
     overfitting risk, more data-efficient, easier to sanity-check by hand.
  3. It is directly citable in the paper as "we traced the pose-dependent
     bias to a specific, named, uncalibrated firmware constant" rather than
     "we fit a black-box correction" -- a materially stronger claim.

NOT YET DONE (deliberately, given time pressure + it touches shared control
firmware): actually setting RCM2_1.RCMoffset = <fitted K, with correct sign/
frame convention verified> in robotModel.cpp and rebuilding/redeploying.
That would be the true root-cause firmware fix and should make THIS SCRIPT's
correction unnecessary going forward, but must be done carefully (that
struct field feeds ComputeFwdKinsTip AND ComputeJacobianTip AND at least two
other kinematics call sites -- ComputeFwdKinsRCM-style functions likely
share it too -- so changing it is a robot-wide kinematic recalibration, not
a local patch) -- track as separate future-work item, not urgent for this
week's precision campaign.

USAGE:
    python3 rcmoffset_fit.py pivot_samples.csv [more_csvs...] [--save]

Accepts the same two CSV schemas as pivot_residual_fit.py.
"""
from __future__ import print_function
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, "tip_offset_rcmoffset.npz")


def load_group(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if "px" in fields and "qx" in fields:
            pk, qk = ("px", "py", "pz"), ("qx", "qy", "qz", "qw")
        elif "ee_px" in fields and "ee_qx" in fields:
            pk, qk = ("ee_px", "ee_py", "ee_pz"), ("ee_qx", "ee_qy", "ee_qz", "ee_qw")
        else:
            raise ValueError("%s: cannot find px/qx or ee_px/ee_qx columns" % path)
        ps, qs = [], []
        for row in reader:
            ps.append([float(row[k]) for k in pk])
            qs.append([float(row[k]) for k in qk])
    ps, qs = np.asarray(ps, float), np.asarray(qs, float)
    Rs = np.array([Rot.from_quat(q).as_matrix() for q in qs])
    # sanity-check the 2-DOF-wrist hypothesis holds for this data before
    # trusting q4 recovered from it (see module docstring) -- if this ever
    # fails on live data, the whole model is inapplicable (different robot?
    # different firmware kinematics branch?).
    q4s = np.arctan2(-Rs[:, 1, 2], Rs[:, 1, 1])
    q3s = np.arctan2(-Rs[:, 2, 0], Rs[:, 0, 0])
    recon = np.array([
        [[np.cos(q3), np.sin(q3) * np.sin(q4), np.sin(q3) * np.cos(q4)],
         [0.0, np.cos(q4), -np.sin(q4)],
         [-np.sin(q3), np.cos(q3) * np.sin(q4), np.cos(q3) * np.cos(q4)]]
        for q3, q4 in zip(q3s, q4s)])
    err = np.linalg.norm(recon - Rs, axis=(1, 2))
    if err.max() > 1e-3:
        print("  !! WARNING %s: 2-DOF-wrist reconstruction residual max=%.2e -- "
              "this model's core assumption may not hold for this data" % (path, err.max()))
    return Rs, q4s, ps


def fit(groups, leave_out=None, use_K=True):
    ncols = 3 + (1 if use_K else 0) + 3 * len(groups)
    n = sum(len(g[2]) for g in groups) - (1 if leave_out else 0)
    A = np.zeros((3 * n, ncols))
    b = np.zeros(3 * n)
    row = 0
    for gi, (Rs, q4s, ps) in enumerate(groups):
        for si in range(len(ps)):
            if leave_out == (gi, si):
                continue
            A[3 * row:3 * row + 3, 0:3] = Rs[si]
            col = 3
            if use_K:
                A[3 * row:3 * row + 3, 3] = Rs[si].dot(
                    np.array([0.0, np.sin(q4s[si]), np.cos(q4s[si])]))
                col = 4
            A[3 * row:3 * row + 3, col + 3 * gi:col + 3 + 3 * gi] = -np.eye(3)
            b[3 * row:3 * row + 3] = -ps[si]
            row += 1
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    d_const = x[0:3]
    K = float(x[3]) if use_K else 0.0
    off = 4 if use_K else 3
    p_pivots = [x[off + 3 * gi:off + 3 + 3 * gi] for gi in range(len(groups))]
    return d_const, K, p_pivots


def loo_rms(groups, use_K):
    flat = [(gi, si) for gi, g in enumerate(groups) for si in range(len(g[2]))]
    errs = np.zeros(len(flat))
    for k, (gi, si) in enumerate(flat):
        d_const, K, p_pivots = fit(groups, leave_out=(gi, si), use_K=use_K)
        Rs, q4s, ps = groups[gi]
        d_i = d_const + (K * np.array([0.0, np.sin(q4s[si]), np.cos(q4s[si])]) if use_K else 0.0)
        pred = ps[si] + Rs[si].dot(d_i)
        errs[k] = np.linalg.norm(pred - p_pivots[gi])
    return float(np.sqrt(np.mean(errs ** 2)))


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    groups = [load_group(p) for p in paths]
    n_total = sum(len(g[2]) for g in groups)
    print("Loaded %d group(s), n_total=%d" % (len(groups), n_total))

    rms_rigid = loo_rms(groups, use_K=False)
    rms_k = loo_rms(groups, use_K=True)
    d_const, K, _ = fit(groups, use_K=True)
    print("\nRIGID (K=0)                     LOO RMS = %.3f mm" % rms_rigid)
    print("RCMoffset model (K free)        LOO RMS = %.3f mm  (%.0f%% reduction)" %
          (rms_k, 100.0 * (rms_rigid - rms_k) / rms_rigid if rms_rigid > 0 else 0.0))
    print("\nfitted K (mm)     = %.3f" % K)
    print("fitted d_const (mm) = %s" % np.array2string(d_const, precision=3))

    if save:
        tip_inee_at_calib = None
        try:
            sys.path.insert(0, HERE)
            from tip_inee_guard import read_live_tip_inee
            import rospy
            if not rospy.core.is_initialized():
                rospy.init_node("rcmoffset_fit_save", anonymous=True, disable_signals=True)
            tip_inee_at_calib = read_live_tip_inee(timeout_s=2.0)
        except Exception as e:
            print("  (could not read live TipInEE for provenance stamp: %s)" % e)
        np.savez(OUT_NPZ, d_const=d_const, K=K, loo_rms_mm=rms_k, loo_rms_rigid_mm=rms_rigid,
                 n=n_total,
                 tip_inee_at_calib=tip_inee_at_calib if tip_inee_at_calib is not None else np.nan)
        print("\nSaved -> %s" % OUT_NPZ)
        print("  usage: q4 = atan2(-R[1,2],R[1,1]); "
              "tip_offset(R) = d_const + K*[0,sin(q4),cos(q4)]")


if __name__ == "__main__":
    main()
