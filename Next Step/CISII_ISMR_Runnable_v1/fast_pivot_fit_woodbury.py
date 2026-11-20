#!/usr/bin/env python3
"""fast_pivot_fit_woodbury.py -- drop-in FAST replacement for the
leave-one-out (LOO) cross-validation math inside pivot_residual_fit.py and
rcmoffset_fit.py.

WHY THIS EXISTS (2026-09-21): pivot_residual_fit.py's own loo_rms_fn()
(and rcmoffset_fit.py's loo_rms()) compute LOO by literally refitting the
WHOLE least-squares problem once per left-out sample, rebuilding the design
matrix via a pure-Python for-loop each time. That is O(N) refits x O(N)
python-loop matrix-build each = O(N^2) Python-level work. On today's
combined dataset (N=2273: 40 discrete pivot_samples.csv + 2233 continuous
tip_pin_check_v3_fullpose.py samples) this made the ORIGINAL script's
10-point lambda sweep take 20+ minutes and never finished in one sitting.

THE FIX: leave-one-BLOCK-out (each sample contributes a rigid 3-row block
to the OLS/ridge normal equations) has a closed-form shortcut, the
multivariate generalization of the classic scalar LOO/"hat matrix" trick:

    Fit ONCE on the FULL data:      x_full = pinv(A^T A) @ A^T @ b
    Per-sample leverage (3x3):      H_i = A_i @ pinv(A^T A) @ A_i^T
    Full-fit residual (3-vector):   e_i  = A_i @ x_full - b_i
    LOO residual for that sample:   e_i_loo = solve(I3 - H_i, e_i)

This replaces N full refits with ONE fit + N cheap 3x3 matrix ops. Verified
EXACTLY (max abs diff 0.000000mm) against the slow brute-force refit method
on the n=40 pivot_samples.csv set, and against a slow brute-force augmented-
model refit on a small subsample -- see chat transcript 2026-09-21 for the
validation runs. Runs the full N=2273, 10-lambda augmented-model sweep in
~3.5s instead of 20+ minutes.

Ridge rows (used by the augmented model to regularize the 9-param C matrix)
are treated as FIXED pseudo-observations, exactly matching the original
scripts' semantics (they are appended once to A and never removed by any
per-sample leave-out, in either the slow or fast version).

USAGE: import build_full_A_b / fast_loo from this file, or run standalone:
    python3 fast_pivot_fit_woodbury.py pivot_samples.csv fullpose_log.csv

Produces the RIGID-multi-group and AUGMENTED (lambda sweep) LOO comparison,
matching pivot_residual_fit.py's own printed report format, and (with
--save) writes tip_offset_pivot.npz (rigid) and tip_offset_pivot_augmented.npz
(augmented, best lambda) in the exact schema
home_to_rcm_insert_v2_precision.py's load_tip_offset_pivot() /
load_tip_offset_pivot_augmented() expect.
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
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
LAMBDA_GRID = [0.0, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]


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
    return np.asarray(ps, float), np.asarray(qs, float)


def build_rows(ps, qs, R_mean):
    Rs = np.array([Rot.from_quat(q).as_matrix() for q in qs])
    rvecs = np.array([Rot.from_matrix(R_mean.T.dot(R)).as_rotvec() for R in Rs])
    return (Rs, rvecs, ps)


def build_full_A_b(groups, lam, augmented=True):
    """groups: list of (Rs, rvecs, ps) tuples, one per input CSV/session.
    augmented=True -> 3+9+3*ngroups columns (d0, C, per-group p_pivot)
    augmented=False -> 3+3*ngroups columns (rigid d, per-group p_pivot)"""
    n_total = sum(len(g[2]) for g in groups)
    ncols = (3 + 9 + 3 * len(groups)) if augmented else (3 + 3 * len(groups))
    nridge = 9 if (augmented and lam > 0) else 0
    A = np.zeros((3 * n_total + nridge, ncols))
    b = np.zeros(3 * n_total + nridge)
    row = 0
    sample_rows = []
    for gi, (Rs, rvecs, ps) in enumerate(groups):
        for si in range(len(ps)):
            R_i, r_i = Rs[si], rvecs[si]
            A[3 * row:3 * row + 3, 0:3] = R_i
            if augmented:
                A[3 * row:3 * row + 3, 3:12] = R_i.dot(np.kron(np.eye(3), r_i.reshape(1, 3)))
                A[3 * row:3 * row + 3, 12 + 3 * gi:15 + 3 * gi] = -np.eye(3)
            else:
                A[3 * row:3 * row + 3, 3 + 3 * gi:6 + 3 * gi] = -np.eye(3)
            b[3 * row:3 * row + 3] = -ps[si]
            sample_rows.append(3 * row)
            row += 1
    if nridge:
        A[3 * n_total:3 * n_total + 9, 3:12] = np.sqrt(lam) * np.eye(9)
    return A, b, sample_rows, n_total


def fast_loo(groups, lam, augmented=True):
    """Returns (loo_rms_mm, x_full). See module docstring for the math."""
    A, b, sample_rows, n_total = build_full_A_b(groups, lam, augmented)
    x_full, *_ = np.linalg.lstsq(A, b, rcond=None)
    M = A.T.dot(A)
    Minv = np.linalg.pinv(M)
    errs = np.zeros(n_total)
    for k, r0 in enumerate(sample_rows):
        Ai = A[r0:r0 + 3]
        bi = b[r0:r0 + 3]
        Hi = Ai.dot(Minv).dot(Ai.T)
        ei = Ai.dot(x_full) - bi
        try:
            e_loo = np.linalg.solve(np.eye(3) - Hi, ei)
        except np.linalg.LinAlgError:
            e_loo = np.linalg.lstsq(np.eye(3) - Hi, ei, rcond=None)[0]
        errs[k] = np.linalg.norm(e_loo)
    return float(np.sqrt(np.mean(errs ** 2))), x_full


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    raw_groups = [load_group(p) for p in paths]
    n_total = sum(len(g[0]) for g in raw_groups)
    print("Loaded %d group(s), %d samples total:" % (len(raw_groups), n_total))
    for p, (ps, qs) in zip(paths, raw_groups):
        print("  %-50s n=%d" % (os.path.basename(p), len(ps)))

    all_qs = np.vstack([qs for _, qs in raw_groups])
    Mq = np.einsum("ni,nj->ij", all_qs, all_qs)
    w, v = np.linalg.eigh(Mq)
    q_mean = v[:, np.argmax(w)]
    R_mean = Rot.from_quat(q_mean).as_matrix()

    groups = [build_rows(ps, qs, R_mean) for ps, qs in raw_groups]

    t0 = time.time()
    print("\n=== BASELINE (rigid constant d) ===")
    base_loo, x_base = fast_loo(groups, 0.0, augmented=False)
    d_base = x_base[0:3]
    print("  d = %s  |d|=%.2fmm" % (np.round(d_base, 3), np.linalg.norm(d_base)))
    print("  LOO RMS = %.4fmm   (took %.2fs)" % (base_loo, time.time() - t0))

    print("\n=== AUGMENTED (d depends on orientation) -- lambda sweep ===")
    results = []
    t0 = time.time()
    for lam in LAMBDA_GRID:
        lr, x_full = fast_loo(groups, lam, augmented=True)
        d0 = x_full[0:3]
        C = x_full[3:12].reshape(3, 3)
        results.append((lam, lr, d0, C))
        print("  lambda=%-7g  LOO RMS = %.4f mm   ||C||_F=%.2f" % (lam, lr, np.linalg.norm(C)))
    print("  (lambda sweep took %.2fs total)" % (time.time() - t0))

    best = min(results, key=lambda r: r[1])
    lam_best, loo_best, d0_best, C_best = best
    print("\n=== VERDICT ===")
    print("  baseline (rigid) LOO = %.4f mm" % base_loo)
    print("  best augmented LOO   = %.4f mm  (lambda=%s)" % (loo_best, lam_best))

    if save:
        np.savez(data_path("tip_offset_pivot.npz"),
                 d=d_base, fit_rms=base_loo, loo_rms=base_loo, n=n_total)
        np.savez(data_path("tip_offset_pivot_augmented.npz"),
                 d0=d0_best, C=C_best, R_mean=R_mean, lam=lam_best,
                 loo_rms_augmented=loo_best, loo_rms_baseline=base_loo,
                 d_baseline=d_base, n=n_total)
        print("\nSaved -> tip_offset_pivot.npz, tip_offset_pivot_augmented.npz")


if __name__ == "__main__":
    main()
