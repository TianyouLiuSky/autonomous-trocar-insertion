#!/usr/bin/env python3
"""Depth-FREE refinement of the hand-eye calibration, using ONLY the
already-collected pairs.csv -- no new data collection, no printer needed.

WHY: the live calibration (handeye_calibrate.py / handeye_T.npz) solves R,t,m
by 3D-3D alignment (Kabsch) between the DEPTH-based deprojected marker point
and the FK-based base-frame point. Every mm of depth-sensor noise/bias goes
straight into R,t,m. This script instead minimizes REPROJECTION error in
PIXEL space -- it only trusts the marker's 2D pixel location (robust,
sub-pixel, no depth) and each pose's FrameEE (accurate, mechanical), and
solves for the camera extrinsic (R,t) + marker offset (m) that best explains
ALL observed pixels via projection, the same way a checkerboard camera
calibration works except the "checkerboard points" are the N teach poses.

v2 (2026-08-11) additions, all in the name of "correct AND better LOO",
given the hardware constraint of a single ArUco marker (no ChArUco):
  1. Robust loss (Huber, default) instead of plain least-squares, so that a
     handful of bad detections among 60-100+ manually-collected poses can't
     silently dominate the fit the way a single-outlier-sensitive L2 loss
     would (method switches lm -> trf, since scipy's LM does not support
     robust losses).
  2. Multi-start check: re-solve from a few randomly perturbed initial
     guesses and verify they all converge to essentially the same R,t,m --
     guards against silently reporting a local minimum.
  3. Optional TRUE held-out validation set (HANDEYE_VAL_CSV env var) --
     poses collected but deliberately withheld from the fit entirely, not
     just left-one-out within the same pool. LOO can still be optimistic if
     the fit pool is spatially clustered; a separate validation pool is the
     harder, more honest check.
  4. Parameter uncertainty (std dev on t, m in mm) from the LM/TRF Jacobian
     at the optimum, instead of only a point estimate.
  5. Promotes its own result to the OFFICIALLY-USED handeye_T.npz (backing
     up whatever was there before with a timestamp) -- previously this
     script only ever wrote handeye_T_reproj.npz, which nothing downstream
     actually loads, so "method B" never became the calibration actually in
     effect. Set PROMOTE=0 to disable and keep the old write-only-reproj
     behavior.

v3 (2026-08-24) addition -- automatic, pre-registered outlier rejection:
  A single bad ArUco detection (motion blur, partial occlusion, glare) can
  sit at 5-10x the residual of every other pose and drag down BOTH fit and
  LOO RMS, even under the Huber robust loss (Huber down-weights an outlier's
  influence on OTHER poses' fit, but its own held-out LOO residual is still
  exactly as bad as the underlying detection). Manually eyeballing which row
  to delete is not defensible for a paper. Instead:
    1. Fit once on the FULL dataset (robust loss, as always).
    2. Compute each pose's own reprojection residual under that fit.
    3. Flag pose i as an outlier iff resid_i > median(resid) + OUTLIER_MAD_K
       * 1.4826 * MAD(resid)  (the 1.4826 factor makes MAD a consistent
       estimator of sigma for a Gaussian core), AND resid_i > OUTLIER_MIN_PX
       (a hard floor so a already-excellent, low-spread dataset doesn't get
       poses rejected just for being 3px instead of 1px).
    4. Refit from scratch on the surviving poses; report n_before/n_after and
       exactly which CSV rows were dropped, in both the console output and
       the saved .npz (outlier_rows_excluded).
  This is a fixed, dataset-agnostic rule decided BEFORE looking at which rows
  it flags on any particular session's data -- it is not curve-fit to make
  one specific session's numbers look better. Set OUTLIER_REJECT=0 to disable
  and fit on 100% of the rows as before.

Recovering the pixel from pairs.csv: pairs.csv only stores the ALREADY
deprojected cam_mm (3D point in camera frame from depth+intrinsics). Since
deprojection is an exactly invertible pinhole operation, forward-projecting
cam_mm back through the SAME intrinsics recovers the EXACT original detected
pixel (u,v) with zero information loss -- so we get genuine 2D observations
without touching a single new frame or re-running detection on saved images.

Usage:
  python3 handeye_reproj_refine.py [pairs.csv] [old_handeye_T.npz] [out.npz]
Needs:
  handeye_data/pairs.csv   (existing)
  cam_intrinsics.npz       (ideally from camera_intrinsics_calibrate.py's
                             checkerboard self-cal, not just RealSense
                             factory values -- see that script's docstring)
Env:
  HANDEYE_VAL_CSV   optional path to a held-out validation pairs.csv (same
                    schema, poses NOT used for fitting)
  ROBUST_LOSS       'huber' (default) | 'linear' (old plain-L2 behavior)
  ROBUST_F_SCALE    Huber delta in pixels (default 3.0)
  PROMOTE           1 (default) also writes handeye_T.npz (backing up the
                    old one with a timestamp) | 0 = only write OUT_NPZ
Outputs:
  handeye_T_reproj.npz  (R,t,m + fit/loo/val stats + parameter std devs)
  handeye_T.npz         (same content, if PROMOTE=1 -- this is what
                         home_to_rcm_insert.py / auto_target_detect.py /
                         etc. actually load)
  prints OLD (depth/Kabsch) vs NEW (reprojection) side by side.
"""
from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import datetime
import os
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV = sys.argv[1] if len(sys.argv) > 1 else data_path("handeye_data", "pairs.csv")
OLD_NPZ = sys.argv[2] if len(sys.argv) > 2 else data_path("handeye_T.npz")
INTR_NPZ = data_path("cam_intrinsics.npz")
OUT_NPZ = sys.argv[3] if len(sys.argv) > 3 else data_path("handeye_T_reproj.npz")
OFFICIAL_NPZ = data_path("handeye_T.npz")

VAL_CSV = os.environ.get("HANDEYE_VAL_CSV", "")
ROBUST_LOSS = os.environ.get("ROBUST_LOSS", "huber")
ROBUST_F_SCALE = float(os.environ.get("ROBUST_F_SCALE", "3.0"))
PROMOTE = os.environ.get("PROMOTE", "1") != "0"
N_MULTISTART = int(os.environ.get("N_MULTISTART", "5"))
OUTLIER_REJECT = os.environ.get("OUTLIER_REJECT", "1") != "0"
OUTLIER_MAD_K = float(os.environ.get("OUTLIER_MAD_K", "6.0"))
OUTLIER_MIN_PX = float(os.environ.get("OUTLIER_MIN_PX", "10.0"))


def load_intrinsics():
    if os.path.isfile(INTR_NPZ):
        z = np.load(INTR_NPZ)
        intr = rs.intrinsics()
        intr.fx, intr.fy = float(z["fx"]), float(z["fy"])
        intr.ppx, intr.ppy = float(z["ppx"]), float(z["ppy"])
        intr.width, intr.height = int(z["width"]), int(z["height"])
        intr.model = rs.distortion.brown_conrady
        intr.coeffs = [float(c) for c in z["coeffs"]]
        if "loo_rms_px" in z.files:
            print("  (cam_intrinsics.npz is self-calibrated: fit=%.2fpx LOO=%.2fpx, n=%d)" %
                  (float(z["fit_rms_px"]), float(z["loo_rms_px"]), int(z["n"])))
        else:
            print("  !! cam_intrinsics.npz has no fit/LOO stats -- likely still RAW FACTORY "
                  "values, not self-calibrated. Consider running camera_intrinsics_calibrate.py "
                  "first (see its docstring); a biased K/dist here silently biases every R,t,m "
                  "solved below without showing up in fit/LOO RMS.")
        return intr
    raise ValueError("Fresh camera intrinsics required")


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def K_from_intr(intr):
    return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)


def dist_from_intr(intr):
    return np.array(list(intr.coeffs[:5]), float)


def load_pairs_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    Pcam = data[:, 0:3]
    Pbe = data[:, 3:6]
    Rbe = np.array([quat_to_R(q) for q in data[:, 6:10]])
    return Pcam, Pbe, Rbe


def recover_pixels(cam_mm, intr):
    """Exact inverse of rs2_deproject_pixel_to_point: cam_mm (mm) -> (u,v)."""
    uv = []
    for p in cam_mm:
        px = rs.rs2_project_point_to_pixel(intr, (p / 1000.0).tolist())
        uv.append(px)
    return np.asarray(uv, float)


def project_cam_pts(cam_pts_mm, K, dist):
    """cam_pts already in camera frame (mm) -> pixel, with distortion."""
    rvec = tvec = np.zeros(3)
    obj = cam_pts_mm.reshape(-1, 1, 3).astype(np.float64)
    img, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return img.reshape(-1, 2)


def residuals(x, Pbe_ee, Rbe, uv_obs, K, dist):
    rvec_cb = x[0:3]
    t_cb = x[3:6]
    m = x[6:9]
    Rcb, _ = cv2.Rodrigues(rvec_cb)
    base_pt = np.einsum("nij,j->ni", Rbe, m) + Pbe_ee
    cam_pt = (base_pt - t_cb).dot(Rcb)  # R^T @ (base-t), row-vector form
    pred_uv = project_cam_pts(cam_pt, K, dist)
    return (pred_uv - uv_obs).ravel()


def solve_reproj(Pbe_ee, Rbe, uv_obs, K, dist, x0, robust=True):
    """LM (plain L2) if robust=False; TRF+Huber if robust=True. scipy's 'lm'
    method does not support robust loss functions, hence the method switch."""
    if robust:
        sol = least_squares(residuals, x0, args=(Pbe_ee, Rbe, uv_obs, K, dist),
                             method="trf", loss=ROBUST_LOSS, f_scale=ROBUST_F_SCALE,
                             max_nfev=20000)
    else:
        sol = least_squares(residuals, x0, args=(Pbe_ee, Rbe, uv_obs, K, dist),
                             method="lm", max_nfev=20000)
    rvec_cb, t_cb, m = sol.x[0:3], sol.x[3:6], sol.x[6:9]
    Rcb, _ = cv2.Rodrigues(rvec_cb)
    # sol.fun is always the RAW (unweighted) residual vector regardless of
    # `loss` -- the loss function only reshapes the internal cost surface
    # during optimization, it does not transform the reported residuals.
    res = sol.fun.reshape(-1, 2)
    px_rms = float(np.sqrt((res ** 2).sum(axis=1).mean()))
    return Rcb, t_cb, m, px_rms, sol


def param_cov(sol, n_obs):
    """Rough parameter covariance from the Gauss-Newton/TRF Jacobian at the
    optimum: cov ~= sigma^2 * pinv(J^T J). Treat with the usual caveats
    (assumes locally-linear, homoscedastic Gaussian-ish residuals; the Huber
    loss's IRLS weighting is not fully accounted for, so this is a useful
    order-of-magnitude uncertainty estimate, not a rigorous confidence
    interval). Returns the FULL 9x9 matrix (not just the diagonal) so that
    downstream Monte Carlo error-budget propagation can sample [rvec,t,m]
    jointly with their true correlations, instead of wrongly treating them
    as independent."""
    J = sol.jac
    dof = max(1, n_obs - len(sol.x))
    sigma2 = float(np.sum(sol.fun ** 2)) / dof
    try:
        return sigma2 * np.linalg.pinv(J.T.dot(J))
    except np.linalg.LinAlgError:
        return np.full((len(sol.x), len(sol.x)), np.nan)


def param_std_devs(sol, n_obs):
    return np.sqrt(np.clip(np.diag(param_cov(sol, n_obs)), 0, None))


def multistart_check(Pbe_ee, Rbe, uv_obs, K, dist, x0, n=5, seed=0):
    """Re-solve from several randomly perturbed initial guesses; report how
    consistent the converged solutions are. Large spread => the problem is
    poorly conditioned for this dataset (not enough pose diversity) and the
    "best" solution found should not be trusted as a unique global optimum."""
    rng = np.random.RandomState(seed)
    sols = []
    for i in range(n):
        pert = x0.copy()
        pert[0:3] += rng.normal(0, 0.05, 3)     # ~a few deg equivalent on rvec
        pert[3:6] += rng.normal(0, 5.0, 3)      # +-5mm on t
        pert[6:9] += rng.normal(0, 2.0, 3)      # +-2mm on m
        _, _, _, px_rms, sol = solve_reproj(Pbe_ee, Rbe, uv_obs, K, dist, pert, robust=True)
        sols.append((px_rms, sol.x))
    best = min(sols, key=lambda s: s[0])
    spreads_t = np.array([s[1][3:6] for s in sols])
    spreads_m = np.array([s[1][6:9] for s in sols])
    t_spread_mm = float(np.max(np.ptp(spreads_t, axis=0)))
    m_spread_mm = float(np.max(np.ptp(spreads_m, axis=0)))
    return best, t_spread_mm, m_spread_mm


def base_pts(Pbe_ee, Rbe, m):
    return np.einsum("nij,j->ni", Rbe, m) + Pbe_ee


def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T.dot(Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T.dot(U.T)))
    R = Vt.T.dot(np.diag([1.0, 1.0, d])).dot(U.T)
    return R, cQ - R.dot(cP)


def solve_point_marker_kabsch(Pcam, Pbe, Rbe, iters=50):
    """Same iterative depth-based 3D-3D alignment as handeye_calibrate.py,
    reproduced here so we can compute its LOO error in PIXEL space for a
    fair apples-to-apples comparison against the reprojection method."""
    m = np.zeros(3)
    R, t = kabsch(Pcam, Pbe)
    for _ in range(iters):
        b = np.einsum("nij,j->ni", Rbe, m) + Pbe
        R, t = kabsch(Pcam, b)
        r = (Pcam.dot(R.T) + t) - Pbe
        A = Rbe.reshape(-1, 3)
        m = np.linalg.lstsq(A, r.reshape(-1), rcond=None)[0]
    return R, t, m


def mm_rms_from_RTM(Pcam, Pbe_ee, Rbe, R, t, m):
    pred = base_pts(Pbe_ee, Rbe, m)
    res = np.linalg.norm(Pcam.dot(R.T) + t - pred, axis=1)
    return float(np.sqrt((res ** 2).mean())), res


def detect_outliers(Pbe, Rbe, uv_obs, K, dist, x0, mad_k=OUTLIER_MAD_K, min_px=OUTLIER_MIN_PX):
    """Fit once on everything, flag poses whose own residual is both (a) far
    from the pack in robust-statistics terms and (b) large in absolute terms.
    Returns (keep_mask, resid_norms_from_this_fit)."""
    _, _, _, _, sol = solve_reproj(Pbe, Rbe, uv_obs, K, dist, x0, robust=True)
    resid = sol.fun.reshape(-1, 2)
    resid_norm = np.linalg.norm(resid, axis=1)
    med = np.median(resid_norm)
    mad = np.median(np.abs(resid_norm - med))
    thresh = med + mad_k * 1.4826 * mad
    is_outlier = (resid_norm > thresh) & (resid_norm > min_px)
    return ~is_outlier, resid_norm, med, mad, thresh


def eval_on_set(Pbe_ee, Rbe, uv_obs, K, dist, R, t, m):
    base = base_pts(Pbe_ee, Rbe, m)
    cam = (base - t).dot(R)
    uv_pred = project_cam_pts(cam, K, dist)
    err = np.linalg.norm(uv_pred - uv_obs, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), err


def main():
    if not os.path.isfile(PAIRS_CSV):
        print("No pairs.csv at", PAIRS_CSV); return
    Pcam, Pbe, Rbe = load_pairs_csv(PAIRS_CSV)
    n = len(Pcam)
    if n < 4:
        print("Need >=4 poses, have", n); return

    intr = load_intrinsics()
    K = K_from_intr(intr)
    dist = dist_from_intr(intr)
    print("Intrinsics: fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f dist=%s" %
          (intr.fx, intr.fy, intr.ppx, intr.ppy, dist))

    uv_obs = recover_pixels(Pcam, intr)
    print("\n%d poses loaded from %s" % (n, PAIRS_CSV))

    n_total = n
    outlier_rows = np.array([], dtype=int)
    if OUTLIER_REJECT and n >= 8:
        R_seed, t_seed, m_seed = solve_point_marker_kabsch(Pcam, Pbe, Rbe)
        rvec_seed, _ = cv2.Rodrigues(R_seed)
        x0_seed = np.concatenate([rvec_seed.ravel(), t_seed, m_seed])
        keep_mask, resid_norm, med, mad, thresh = detect_outliers(
            Pbe, Rbe, uv_obs, K, dist, x0_seed)
        n_out = int((~keep_mask).sum())
        print("\n--- Outlier screen (median=%.2fpx MAD=%.2fpx -> threshold=%.2fpx, "
              "MAD_K=%.1f, min_px=%.1f) ---" % (med, mad, thresh, OUTLIER_MAD_K, OUTLIER_MIN_PX))
        if n_out > 0:
            outlier_rows = np.where(~keep_mask)[0] + 2  # +2: 1-index + header row
            print("  Flagged %d/%d poses as outliers -> CSV row(s): %s  (residuals: %s)" %
                  (n_out, n, list(outlier_rows),
                   np.array2string(resid_norm[~keep_mask], precision=1)))
            print("  Excluding and refitting on the remaining %d poses. Set OUTLIER_REJECT=0"
                  " to disable this and use all %d poses as-is." % (n - n_out, n))
            Pcam, Pbe, Rbe, uv_obs = Pcam[keep_mask], Pbe[keep_mask], Rbe[keep_mask], uv_obs[keep_mask]
            n = len(Pcam)
        else:
            print("  No outliers flagged -- using all %d poses." % n)

    # --- OLD (depth/Kabsch) model, for comparison -- solved fresh on THIS
    # exact dataset (not loaded from a possibly-mismatched npz on disk), so
    # the comparison is always apples-to-apples for whatever CSV is passed in.
    R_old, t_old, m_old = solve_point_marker_kabsch(Pcam, Pbe, Rbe)
    mm_rms_old, res_old = mm_rms_from_RTM(Pcam, Pbe, Rbe, R_old, t_old, m_old)
    px_rms_old, _ = eval_on_set(Pbe, Rbe, uv_obs, K, dist, R_old, t_old, m_old)

    # --- NEW (reprojection, robust) model ---
    rvec0, _ = cv2.Rodrigues(R_old)
    x0 = np.concatenate([rvec0.ravel(), t_old, m_old])
    R_new, t_new, m_new, px_rms_new, sol = solve_reproj(
        Pbe, Rbe, uv_obs, K, dist, x0, robust=(ROBUST_LOSS != "linear"))
    mm_rms_new, res_new = mm_rms_from_RTM(Pcam, Pbe, Rbe, R_new, t_new, m_new)
    cov9 = param_cov(sol, 2 * n)  # full [rvec(3),t(3),m(3)] 9x9 covariance
    std = np.sqrt(np.clip(np.diag(cov9), 0, None))
    t_std, m_std = std[3:6], std[6:9]

    print("\nMulti-start check (%d restarts, verifying no local-minimum trap) ..." % N_MULTISTART)
    (best_rms, best_x), t_spread, m_spread = multistart_check(
        Pbe, Rbe, uv_obs, K, dist, x0, n=N_MULTISTART)
    consistent = t_spread < 1.0 and m_spread < 1.0
    print("  best of %d restarts: px_rms=%.2f  |  spread across restarts: t=%.2fmm m=%.2fmm  -> %s" %
          (N_MULTISTART, best_rms, t_spread, m_spread,
           "CONSISTENT (looks like a real optimum)" if consistent else
           "!! INCONSISTENT -- dataset may be poorly conditioned (not enough pose diversity); "
           "treat the solution below with caution"))

    # LOO for the new (reprojection) model
    loo_px = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Ri, ti, mi, _, _ = solve_reproj(Pbe[idx], Rbe[idx], uv_obs[idx], K, dist, x0,
                                         robust=(ROBUST_LOSS != "linear"))
        base_i = Rbe[i].dot(mi) + Pbe[i]
        cam_i = (base_i - ti).dot(Ri)
        uv_i = project_cam_pts(cam_i[None, :], K, dist)[0]
        loo_px[i] = np.linalg.norm(uv_i - uv_obs[i])
    loo_px_rms = float(np.sqrt((loo_px ** 2).mean()))

    # LOO for the OLD (depth-based Kabsch) model, in pixel space, for a fair
    # apples-to-apples comparison against the new method's LOO-px above.
    loo_px_old = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Ri, ti, mi = solve_point_marker_kabsch(Pcam[idx], Pbe[idx], Rbe[idx])
        base_i = Rbe[i].dot(mi) + Pbe[i]
        cam_i = (base_i - ti).dot(Ri)
        uv_i = project_cam_pts(cam_i[None, :], K, dist)[0]
        loo_px_old[i] = np.linalg.norm(uv_i - uv_obs[i])
    loo_px_rms_old = float(np.sqrt((loo_px_old ** 2).mean()))

    print("\n=== OLD (depth-based Kabsch, current handeye_T.npz) ===")
    print("  3D fit RMS  = %.3f mm  (per-pose: %s)" % (mm_rms_old, np.array2string(res_old, precision=2)))
    print("  reprojection RMS (of that solution) = %.2f px" % px_rms_old)
    print("  reprojection LOO  RMS (of Kabsch method) = %.2f px" % loo_px_rms_old)

    print("\n=== NEW (depth-free reprojection refinement, loss=%s) ===" % ROBUST_LOSS)
    print("  reprojection fit RMS = %.2f px" % px_rms_new)
    print("  reprojection LOO  RMS = %.2f px" % loo_px_rms)
    print("  3D-equivalent fit RMS = %.3f mm (for comparison only; not what's optimized)" % mm_rms_new)
    print("  |m| = %.2f mm  m=%s  (1-sigma std: %s)" %
          (np.linalg.norm(m_new), np.array2string(m_new, precision=2), np.array2string(m_std, precision=2)))
    print("  t   = %s  (1-sigma std: %s)" %
          (np.array2string(t_new, precision=2), np.array2string(t_std, precision=2)))

    val_rms = None
    if VAL_CSV and os.path.isfile(VAL_CSV):
        Pcam_v, Pbe_v, Rbe_v = load_pairs_csv(VAL_CSV)
        uv_v = recover_pixels(Pcam_v, intr)
        val_rms, val_err = eval_on_set(Pbe_v, Rbe_v, uv_v, K, dist, R_new, t_new, m_new)
        print("\n=== TRUE held-out validation (%s, %d poses NEVER used in fitting) ===" %
              (VAL_CSV, len(Pcam_v)))
        print("  validation RMS = %.2f px  (per-pose: %s)" %
              (val_rms, np.array2string(val_err, precision=2)))
        print("  (this is the most honest generalization estimate -- more trustworthy than LOO"
              " if the fit poses are spatially/orientation clustered)")
    elif VAL_CSV:
        print("\n  !! HANDEYE_VAL_CSV=%s set but file not found -- skipping validation" % VAL_CSV)

    print("\n=== VERDICT (LOO/validation is what actually matters -- fit RMS can overfit) ===")
    if loo_px_rms < loo_px_rms_old * 0.7:
        print("  Reprojection refinement is CLEARLY better: LOO %.1fpx -> %.1fpx"
              " (fit: %.1fpx -> %.1fpx). Depth noise was indeed the dominant"
              " error source." % (loo_px_rms_old, loo_px_rms, px_rms_old, px_rms_new))
    elif loo_px_rms < loo_px_rms_old:
        print("  Reprojection refinement is somewhat better: LOO %.1fpx -> %.1fpx."
              % (loo_px_rms_old, loo_px_rms))
    else:
        print("  Reprojection LOO did NOT improve (%.1fpx vs %.1fpx) -- with only"
              " %d poses this may be underdetermined either way; collect more"
              " poses spanning more of the workspace (position AND orientation)."
              % (loo_px_rms_old, loo_px_rms, n))
    if not consistent:
        print("  !! Also flagged as INCONSISTENT under multi-start above -- do not treat this"
              " as final until more/better-spread poses are collected.")

    np.savez(OUT_NPZ, R=R_new, t=t_new, m=m_new,
             fit_rms_px=px_rms_new, loo_rms_px=loo_px_rms,
             val_rms_px=(val_rms if val_rms is not None else np.nan),
             fit_rms_mm=mm_rms_new, t_std_mm=t_std, m_std_mm=m_std,
             param_cov9=cov9,
             multistart_consistent=consistent, n=n, n_total=n_total,
             outlier_rows_excluded=outlier_rows)
    print("\nSaved ->", OUT_NPZ)
    if len(outlier_rows):
        print("  (n=%d used for this fit; %d of %d originally-collected poses were "
              "excluded as outliers -- see outlier_rows_excluded in this .npz)" %
              (n, len(outlier_rows), n_total))

    if PROMOTE:
        if os.path.isfile(OFFICIAL_NPZ):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = data_path("handeye_T_prev_%s.npz" % stamp)
            old = np.load(OFFICIAL_NPZ)
            np.savez(backup, **{k: old[k] for k in old.files})
            print("Backed up previous official calibration -> %s" % backup)
        np.savez(OFFICIAL_NPZ, R=R_new, t=t_new, m=m_new,
                 fit_rms_px=px_rms_new, loo_rms_px=loo_px_rms,
                 val_rms_px=(val_rms if val_rms is not None else np.nan),
                 fit_rms_mm=mm_rms_new, t_std_mm=t_std, m_std_mm=m_std,
                 param_cov9=cov9,
                 multistart_consistent=consistent, n=n, n_total=n_total,
                 outlier_rows_excluded=outlier_rows)
        print("PROMOTED -> %s (this is now what home_to_rcm_insert.py / auto_target_detect.py"
              " / etc. will load)" % OFFICIAL_NPZ)
    else:
        print("PROMOTE=0 -- official handeye_T.npz left untouched.")


if __name__ == "__main__":
    main()
