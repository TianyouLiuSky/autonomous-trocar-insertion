#!/usr/bin/env python3
"""Orientation-DEPENDENT pivot calibration -- upgrade of pivot_calibration.py's
rigid model, for when the rigid model has hit its own ceiling.

WHY THIS EXISTS (2026-08-25): pivot_calibration.py assumes

    true_tip_base = p_i + R_i @ d          (d CONSTANT, independent of R_i)

which is the correct model IF the tool + FrameEE forward kinematics were a
perfect rigid body. In practice tip_pin_check_v2.py's independent, camera-free
check (pin the tip, sweep orientation, watch drift) came back POOR on the
best rigid fit we have (n=43, spread=79.8deg, own LOO=0.73mm) -- max drift
1.43mm, and critically the 2026-08-25 log shows this as a small number of
DISCRETE, STABLE PLATEAUS (0.04mm at one held pose, ~0.87-1.0mm at others,
each rock-steady for 80-190s, with smooth <0.15mm/sample transitions between
them, not backlash-style jumps). That is the signature of a DETERMINISTIC,
ORIENTATION-DEPENDENT bias that a single constant `d` cannot represent --
not sensor noise (noise wouldn't hold a stable plateau for 3 minutes) and not
backlash/hysteresis (that would show a step-like jump correlated with a
direction reversal, not a smooth transition). More rigid-model pivot samples
cannot fix this -- the model CLASS is the bottleneck, not the sample count.

MODEL: allow d to vary linearly with orientation, expressed as a rotation-
vector offset from the dataset's own mean orientation (keeps the feature
small/centered/well-conditioned instead of using raw absolute angles):

    r_i      = rotvec(R_mean.T @ R_i)                       (3,)
    d(R_i)   = d0 + C @ r_i                                  C is (3,3)
    pred_i   = p_i + R_i @ d(R_i)   ==   p_pivot_g            (per group g)

This is still LINEAR in the unknowns [d0(3), vec(C)(9), p_pivot_g(3) per
group] -- closed-form ridge least squares, no nonlinear solver, same spirit
as pivot_calibration.py's own closed-form solve. Ridge regularization is
applied ONLY to the 9 entries of C (never to d0 or p_pivot_g) since C is the
part of the model that can overfit with limited data; d0/p_pivot are exactly
as identifiable as in the original rigid model.

MULTI-GROUP: each input CSV is treated as one "session" with its own unknown
constant target point p_pivot_g (a pivot_calibration.py capture session, or a
tip_pin_check_v3_fullpose.py continuous log where the physical tip was pinned
at one spot for the whole file) -- pass as many files as you have. d0/C are
shared (global) across all groups; only p_pivot_g is per-group. This lets you
combine pivot_samples.csv AND one or more tip-pin sweeps into one fit.

MODEL SELECTION HONESTY: the ridge strength lambda is chosen by minimizing
leave-one-SAMPLE-out RMS across the combined dataset -- this is the same
standard this codebase already uses elsewhere, but note it is a *soft* form
of using held-out error to pick model complexity (not fully nested CV). Do
not treat the LOO number at the selected lambda as equivalent to a genuinely
fresh, independent test -- confirm the win on a NEW tip_pin_check_v3_fullpose.py
sweep (poses not used to fit anything here) before trusting this for the paper.

USAGE:
    python3 pivot_residual_fit.py pivot_samples_archive/pivot_samples_GOOD_20260825_1330_n43_loo0.73mm_baseline-22.csv [more_csvs...]

Prints baseline (rigid) vs augmented LOO RMS across a lambda grid, and on
'--save' writes tip_offset_pivot_augmented.npz (d0, C, R_mean, plus the
baseline d/p_pivot for comparison and tip_inee_at_calib for the same
drift-guard purpose as pivot_calibration.py).

Accepts two CSV schemas automatically (detected by header):
  - pivot_calibration.py's pivot_samples.csv:  px,py,pz,qx,qy,qz,qw,...
  - tip_pin_check_v3_fullpose.py's log:        ...,ee_px,ee_py,ee_pz,ee_qx,ee_qy,ee_qz,ee_qw,...
"""
from __future__ import print_function
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, "tip_offset_pivot_augmented.npz")
LAMBDA_GRID = [0.0, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]


def load_group(path):
    """Returns (ps (n,3), qs (n,4)) for one file, auto-detecting schema."""
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if "px" in fields and "qx" in fields:
            pk, qk = ("px", "py", "pz"), ("qx", "qy", "qz", "qw")
        elif "ee_px" in fields and "ee_qx" in fields:
            pk, qk = ("ee_px", "ee_py", "ee_pz"), ("ee_qx", "ee_qy", "ee_qz", "ee_qw")
        else:
            raise ValueError("%s: cannot find px/qx or ee_px/ee_qx columns "
                              "(header=%s)" % (path, fields))
        ps, qs = [], []
        for row in reader:
            ps.append([float(row[k]) for k in pk])
            qs.append([float(row[k]) for k in qk])
    return np.asarray(ps, float), np.asarray(qs, float)


def build_rows(ps, qs, R_mean):
    """Per-sample R_i (n,3,3) and mean-centered rotation-vector feature
    r_i = rotvec(R_mean.T @ R_i) (n,3)."""
    Rs = np.array([Rot.from_quat(q).as_matrix() for q in qs])
    rvecs = np.array([Rot.from_matrix(R_mean.T.dot(R)).as_rotvec() for R in Rs])
    return Rs, rvecs


def fit_baseline(groups, leave_out=None):
    """groups: list of (Rs, rvecs, ps) tuples, one per session. leave_out:
    optional (group_idx, sample_idx) to exclude (for LOO). Returns d, list of
    p_pivot per group."""
    n_total = sum(len(g[2]) for g in groups) - (1 if leave_out else 0)
    ncols = 3 + 3 * len(groups)
    A = np.zeros((3 * n_total, ncols))
    b = np.zeros(3 * n_total)
    row = 0
    for gi, (Rs, rvecs, ps) in enumerate(groups):
        for si in range(len(ps)):
            if leave_out == (gi, si):
                continue
            A[3 * row:3 * row + 3, 0:3] = Rs[si]
            A[3 * row:3 * row + 3, 3 + 3 * gi:6 + 3 * gi] = -np.eye(3)
            b[3 * row:3 * row + 3] = -ps[si]
            row += 1
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    d = x[0:3]
    p_pivots = [x[3 + 3 * gi:6 + 3 * gi] for gi in range(len(groups))]
    return d, p_pivots


def fit_augmented(groups, lam, leave_out=None):
    """Same but with d(R) = d0 + C @ r. Ridge penalty on the 9 C entries
    only. Returns d0 (3,), C (3,3), list of p_pivot per group."""
    n_total = sum(len(g[2]) for g in groups) - (1 if leave_out else 0)
    ncols = 3 + 9 + 3 * len(groups)
    A = np.zeros((3 * n_total + 9, ncols))
    b = np.zeros(3 * n_total + 9)
    row = 0
    for gi, (Rs, rvecs, ps) in enumerate(groups):
        for si in range(len(ps)):
            if leave_out == (gi, si):
                continue
            R_i, r_i = Rs[si], rvecs[si]
            A[3 * row:3 * row + 3, 0:3] = R_i
            Cblock = R_i.dot(np.kron(np.eye(3), r_i.reshape(1, 3)))
            A[3 * row:3 * row + 3, 3:12] = Cblock
            A[3 * row:3 * row + 3, 12 + 3 * gi:15 + 3 * gi] = -np.eye(3)
            b[3 * row:3 * row + 3] = -ps[si]
            row += 1
    # ridge rows: sqrt(lam) * I on the 9 C columns only
    if lam > 0:
        A[3 * n_total:3 * n_total + 9, 3:12] = np.sqrt(lam) * np.eye(9)
    else:
        A = A[:3 * n_total]
        b = b[:3 * n_total]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    d0 = x[0:3]
    C = x[3:12].reshape(3, 3)
    p_pivots = [x[12 + 3 * gi:15 + 3 * gi] for gi in range(len(groups))]
    return d0, C, p_pivots


def loo_rms_baseline(groups):
    flat = [(gi, si) for gi, g in enumerate(groups) for si in range(len(g[2]))]
    errs = np.zeros(len(flat))
    for k, (gi, si) in enumerate(flat):
        d, p_pivots = fit_baseline(groups, leave_out=(gi, si))
        Rs, rvecs, ps = groups[gi]
        pred = ps[si] + Rs[si].dot(d)
        errs[k] = np.linalg.norm(pred - p_pivots[gi])
    return float(np.sqrt(np.mean(errs ** 2)))


def loo_rms_augmented(groups, lam):
    flat = [(gi, si) for gi, g in enumerate(groups) for si in range(len(g[2]))]
    errs = np.zeros(len(flat))
    for k, (gi, si) in enumerate(flat):
        d0, C, p_pivots = fit_augmented(groups, lam, leave_out=(gi, si))
        Rs, rvecs, ps = groups[gi]
        pred = ps[si] + Rs[si].dot(d0 + C.dot(rvecs[si]))
        errs[k] = np.linalg.norm(pred - p_pivots[gi])
    return float(np.sqrt(np.mean(errs ** 2)))


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)

    raw_groups = [load_group(p) for p in paths]
    n_total = sum(len(g[0]) for g in raw_groups)
    print("Loaded %d group(s), %d samples total:" % (len(raw_groups), n_total))
    for p, (ps, qs) in zip(paths, raw_groups):
        print("  %-70s n=%d" % (os.path.basename(p), len(ps)))
    if n_total < 15:
        print("\n!! only %d samples total -- the augmented model (15 params) needs "
              "substantially more data than the rigid model (6 params) to be "
              "trustworthy. Proceeding, but treat results as exploratory." % n_total)

    all_qs = np.vstack([qs for _, qs in raw_groups])
    M = np.einsum("ni,nj->ij", all_qs, all_qs)
    w, v = np.linalg.eigh(M)
    q_mean = v[:, np.argmax(w)]
    R_mean = Rot.from_quat(q_mean).as_matrix()

    groups = []
    for ps, qs in raw_groups:
        Rs, rvecs = build_rows(ps, qs, R_mean)
        groups.append((Rs, rvecs, ps))
    all_rvecs = np.vstack([g[1] for g in groups])
    print("\nOrientation feature (rotvec from dataset mean) range per axis (deg): %s"
          % np.array2string(np.degrees(np.ptp(all_rvecs, axis=0)), precision=1))

    print("\n=== BASELINE (rigid constant d, same model as pivot_calibration.py) ===")
    d_base, p_piv_base = fit_baseline(groups)
    base_loo = loo_rms_baseline(groups)
    print("  d = [%.3f %.3f %.3f] mm   |d|=%.2fmm" % (d_base[0], d_base[1], d_base[2], np.linalg.norm(d_base)))
    print("  LOO RMS = %.3f mm" % base_loo)

    print("\n=== AUGMENTED (d depends linearly on orientation) -- lambda sweep ===")
    results = []
    for lam in LAMBDA_GRID:
        lr = loo_rms_augmented(groups, lam)
        d0, C, _ = fit_augmented(groups, lam)
        results.append((lam, lr, d0, C))
        flag = "  <-- best so far" if lr == min(r[1] for r in results) else ""
        print("  lambda=%-7g  LOO RMS = %.3f mm   ||C||_F=%.2f%s" % (lam, lr, np.linalg.norm(C), flag))

    best = min(results, key=lambda r: r[1])
    lam_best, loo_best, d0_best, C_best = best
    improvement = 100.0 * (base_loo - loo_best) / base_loo if base_loo > 0 else 0.0

    print("\n=== VERDICT ===")
    print("  baseline (rigid) LOO      = %.3f mm" % base_loo)
    print("  best augmented LOO        = %.3f mm  (lambda=%g)" % (loo_best, lam_best))
    if loo_best < base_loo * 0.95:
        print("  -> augmented model reduces LOO error by %.0f%% -- orientation-dependence "
              "is real signal, not just noise, on THIS dataset." % improvement)
        print("  -> IMPORTANT: this LOO number was used to pick lambda from a grid, which is "
              "a soft form of model selection on held-out error. Before trusting this for the "
              "paper, run a NEW tip_pin_check_v3_fullpose.py sweep (poses not used here) and "
              "check this model's prediction error on that fully independent data.")
    else:
        print("  -> augmented model does NOT meaningfully beat the rigid model on this data "
              "-- not enough evidence yet for orientation-dependence; more/better-covering "
              "data needed before this is worth adopting.")

    if save:
        _, p_piv_aug = fit_augmented(groups, lam_best)[2], None
        d0f, Cf, p_piv_aug = fit_augmented(groups, lam_best)
        tip_inee_at_calib = None
        try:
            sys.path.insert(0, HERE)
            from tip_inee_guard import read_live_tip_inee
            import rospy
            if not rospy.core.is_initialized():
                rospy.init_node("pivot_residual_fit_save", anonymous=True, disable_signals=True)
            tip_inee_at_calib = read_live_tip_inee(timeout_s=2.0)
        except Exception as e:
            print("  (could not read live TipInEE for provenance stamp: %s)" % e)
        np.savez(OUT_NPZ, d0=d0f, C=Cf, R_mean=R_mean, lam=lam_best,
                 loo_rms_augmented=loo_best, loo_rms_baseline=base_loo,
                 d_baseline=d_base, n=n_total,
                 tip_inee_at_calib=tip_inee_at_calib if tip_inee_at_calib is not None else np.nan)
        print("\nSaved -> %s" % OUT_NPZ)
        print("  usage: tip_offset(R) = d0 + C @ rotvec(R_mean.T @ R)")


if __name__ == "__main__":
    main()
