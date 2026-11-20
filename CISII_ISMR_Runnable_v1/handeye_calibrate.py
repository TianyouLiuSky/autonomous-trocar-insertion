#!/usr/bin/env python3
"""Hand-eye calibration using ArUco on the tool shaft (above tip).

Auto-detects an ArUco / square fiducial on the shaft (no QR), records pairs
with FrameEE, and solves eye-to-hand:

    R @ p_cam + t  =  R_ee @ m + p_tip

where m is the marker position in the EE/tip frame (shaft offset above tip).
Saves handeye_T.npz in this folder for show_tip_vrcm_cam.py.

Workflow:
  1. eye_robot running (/SHER21/eye_robot/FrameEE)
  2. ArUco on shaft facing the RealSense; tip free to move
  3. Run this script — green box tracks the marker
  4. SPACE = record. STOP moving fully first: each press grabs a short
     (6-frame) burst and auto-rejects it (prints why) if the marker/robot
     wasn't actually still, or depth was noisy -- this is what makes the fit
     trustworthy (an earlier 6-pose set had fit RMS 1.8mm but LOO RMS 23.9mm,
     i.e. it silently included inconsistent samples). Spread poses in X/Y/Z
     and rotate a bit so m is observable (>= 8-12 GOOD poses; more is better).
  5. s = solve + save   |  q = quit

Keys:
  SPACE / r : record
  u         : undo
  s         : solve + save npz
  m         : toggle auto ArUco / manual click
  left-click: manual mark (if auto off or as override)
  c         : clear manual click
  q / Esc   : quit

Output:
  third_part/handeye_T.npz
  third_part/handeye_data/pairs.csv
"""

from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path

import json
import os
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform

# #region agent log
_DBG_LOG = data_path("handeye_debug.jsonl")
_DBG_SESSION = "17d00f"


def _dbg(hypothesisId, location, message, data=None, runId="pre"):
    try:
        payload = {
            "sessionId": _DBG_SESSION,
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_DBG_LOG, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
# #endregion

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "HANDEYE_DATA_DIR", data_path("handeye_data")
)
OUT_NPZ = os.environ.get(
    "HANDEYE_NPZ", data_path("handeye_T.npz")
)
FRAMEEE_TOPIC = os.environ.get(
    "FRAMEEE_TOPIC", "/SHER21/eye_robot/FrameEE"
)
PAIRS_CSV = os.path.join(DATA_DIR, "pairs.csv")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")

_lock = threading.Lock()
_ee = {"p": None, "q": None}
_click = {"uv": None}
_auto = {"on": True}


def _cb_ee(msg):
    with _lock:
        _ee["p"] = np.array(
            [msg.translation.x, msg.translation.y, msg.translation.z], float
        )
        _ee["q"] = np.array(
            [msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w],
            float,
        )


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        with _lock:
            _click["uv"] = (int(x), int(y))
            _auto["on"] = False
        print("  manual mark @ (%d, %d)  (auto ArUco off — press m to re-enable)"
              % (x, y))


# ---------------------------------------------------------------------------
# ArUco detection only (no QR)
# ---------------------------------------------------------------------------
def _fiducial_score(gray, corners):
    dst = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warp = cv2.warpPerspective(gray, M, (80, 80))
    _, bw = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate(
        [bw[0:10].ravel(), bw[-10:].ravel(),
         bw[:, 0:10].ravel(), bw[:, -10:].ravel()]
    )
    inner = bw[20:60, 20:60]
    frac_dark = (inner == 0).mean()
    frac_bright = (inner == 255).mean()
    border_dark = (border == 0).mean()
    return min(frac_dark, frac_bright) * 2.0 * min(border_dark, 1.0)


def _order_corners(pts):
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    diff = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)],
         pts[np.argmax(s)], pts[np.argmax(diff)]],
        np.float32,
    )


def _set_best_corner_refinement(aruco, params, fast=False):
    """Default cornerRefinementMethod is CORNER_REFINE_NONE (pixel-quantized
    corners) -- this is a free, zero-cost precision win for hand-eye/pivot
    accuracy since every downstream R,t,m fit depends on corner precision.
    APRILTAG refinement (line-fitting) is generally more accurate than plain
    SUBPIX; fall back gracefully if this OpenCV build lacks it.

    2026-09-28: fast=True picks SUBPIX (or NONE) instead. Measured on this
    OpenCV build (4.5.2, legacy aruco.detectMarkers() API): APRILTAG is
    ~9-10x slower than SUBPIX per detectMarkers() call (~0.1s vs ~0.012s per
    dictionary on a 640x480 frame, x4 dictionaries tried = ~0.4s vs ~0.05s
    per detect_aruco() call when no marker is in frame). detect_aruco() is
    called on EVERY grab_show() frame for the live SHIFT+H status readout
    (home_to_rcm_insert_v2_precision.py) -- that ~0.4s/frame cost stretched
    the effective cv2.waitKey(20) polling interval to ~0.4-2s+ per loop
    iteration in practice, which is exactly what made keyboard teleop look
    "frozen" (confirmed live via py-spy: MainThread was sitting inside this
    exact detectMarkers() call). See detect_aruco()'s fast= param."""
    if fast:
        method = getattr(aruco, "CORNER_REFINE_SUBPIX", None)
        if method is None:
            method = getattr(aruco, "CORNER_REFINE_NONE", None)
    else:
        method = getattr(aruco, "CORNER_REFINE_APRILTAG", None)
        if method is None:
            method = getattr(aruco, "CORNER_REFINE_SUBPIX", None)
    if method is not None:
        params.cornerRefinementMethod = method


def detect_aruco(color_bgr, min_score=0.25, fast=False):
    """Return (center, corners, label) for ArUco / square fiducial only.

    fast=True (2026-09-28): use for any PER-FRAME / live-polling call site
    -- trades corner precision for speed (SUBPIX instead of APRILTAG, see
    _set_best_corner_refinement's docstring for the measured ~9-10x cost).
    Leave fast=False (default) for deliberate one-shot calibration sample
    captures (e.g. record_handeye_sample()'s SHIFT+H burst average), where
    the extra ~0.35s once, on purpose, is a good trade for accuracy.
    """
    # Prefer OpenCV contrib ArUco if present; otherwise square-fiducial fallback.
    if hasattr(cv2, "aruco"):
        try:
            aruco = cv2.aruco
            dicts = [
                aruco.DICT_4X4_50,
                aruco.DICT_5X5_50,
                aruco.DICT_6X6_50,
                aruco.DICT_ARUCO_ORIGINAL,
            ]
            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            for dname in dicts:
                dictionary = aruco.getPredefinedDictionary(dname)
                if hasattr(aruco, "ArucoDetector"):
                    params = aruco.DetectorParameters()
                    _set_best_corner_refinement(aruco, params, fast=fast)
                    det = aruco.ArucoDetector(dictionary, params)
                    corners, ids, _ = det.detectMarkers(gray)
                else:
                    params = aruco.DetectorParameters_create()
                    _set_best_corner_refinement(aruco, params, fast=fast)
                    corners, ids, _ = aruco.detectMarkers(
                        gray, dictionary, parameters=params
                    )
                if ids is not None and len(ids) > 0:
                    c0 = corners[0].reshape(-1, 2).astype(np.float32)
                    return c0.mean(axis=0), _order_corners(c0), "ArUco:%d" % int(ids[0])
        except Exception:
            pass

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    cnts = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    H, W = gray.shape[:2]
    area_img = float(H * W)
    cand = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_img * 0.0008 or a > area_img * 0.5:
            continue
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(ap) != 4 or not cv2.isContourConvex(ap):
            continue
        (w, h) = cv2.minAreaRect(ap)[1]
        if min(w, h) == 0 or not (0.6 < max(w, h) / min(w, h) < 1.7):
            continue
        corners = ap.reshape(-1, 2).astype(np.float32)
        score = _fiducial_score(gray, corners)
        if score >= min_score:
            cand.append((score, a, corners))
    if not cand:
        return None, None, ""
    cand.sort(key=lambda x: -x[0])
    top_s = cand[0][0]
    near = [c for c in cand if top_s - c[0] <= 0.10]
    best = max(near, key=lambda x: x[1])[2]
    return best.mean(axis=0), _order_corners(best), "ArUco"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class RSCam(object):
    def __init__(self, width=640, height=480, fps=30):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = (
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        cvs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = __import__("intr_util").load_best_intrinsics(self.profile)
        for _ in range(20):
            self.pipe.wait_for_frames()

    def get_frames(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None, None
        c = np.asanyarray(color.get_data())
        d = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return c, d

    def depth_at(self, depth_m, px, py, win=5, mode="median"):
        h, w = depth_m.shape
        x0, x1 = max(0, px - win), min(w, px + win + 1)
        y0, y1 = max(0, py - win), min(h, py + win + 1)
        valid = depth_m[y0:y1, x0:x1]
        valid = valid[valid > 0]
        if valid.size == 0:
            return float("nan")
        if mode == "nearest":
            return float(np.percentile(valid, 10))
        return float(np.median(valid))

    def deproject_mm(self, px, py, z_m):
        pt = rs.rs2_deproject_pixel_to_point(
            self.intr, [float(px), float(py)], float(z_m)
        )
        return np.array(pt, dtype=np.float64) * 1000.0

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T.dot(Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T.dot(U.T)))
    R = Vt.T.dot(np.diag([1.0, 1.0, d])).dot(U.T)
    return R, cQ - R.dot(cP)


def quat_to_R(q):
    """geometry_msgs xyzw -> 3x3."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def solve_point_marker(Pcam, Pbe, Rbe, iters=50):
    """R @ p_cam + t = R_ee @ m + p_tip."""
    m = np.zeros(3)
    R, t = kabsch(Pcam, Pbe)
    for _ in range(iters):
        b = np.einsum("nij,j->ni", Rbe, m) + Pbe
        R, t = kabsch(Pcam, b)
        r = (Pcam.dot(R.T) + t) - Pbe
        A = Rbe.reshape(-1, 3)
        m = np.linalg.lstsq(A, r.reshape(-1), rcond=None)[0]
    return R, t, m


def solve_and_save(pairs, out_npz):
    """
    pairs rows: cam_xyz(3), tip_xyz(3), quat_xyzw(4).
    Eye-to-hand: camera fixed, robot tip moves.
    """
    pairs, n_dup = dedupe_pairs_by_tip(pairs)
    if n_dup:
        print("  dropped %d near-duplicate tip pose(s) before solve" % n_dup)
        # #region agent log
        _dbg("D", "handeye_calibrate.py:solve_and_save", "deduped before solve",
             {"dropped": n_dup, "n_kept": len(pairs)}, runId="post-fix")
        # #endregion
    if len(pairs) < 4:
        print("Need >= 4 pairs (have %d)." % len(pairs))
        return False
    rows = [_as_row10(p) for p in pairs]
    data = np.asarray(rows, dtype=np.float64)
    Pcam = data[:, 0:3]
    Pbe = data[:, 3:6]
    n = len(Pcam)
    Rbe = np.array([quat_to_R(q) for q in data[:, 6:10]])
    # If all quats are identity (legacy 6-col resume), fall back to plain Kabsch.
    have_orient = np.any(np.abs(data[:, 6:9]) > 1e-6)

    print("\n=== Hand-eye solve (eye-to-hand, fixed camera) ===")
    print("  N=%d  orientation_variation=%s" % (n, have_orient))

    if have_orient:
        R, t, m = solve_point_marker(Pcam, Pbe, Rbe)
        pred = np.einsum("nij,j->ni", Rbe, m) + Pbe
        res = np.linalg.norm(Pcam.dot(R.T) + t - pred, axis=1)
        fit_rms = float(np.sqrt((res ** 2).mean()))
        print("  [ArUco + tip offset m]")
        print("  fit RMS=%.3f max=%.3f mm  |m|=%.2f mm  m=%s"
              % (fit_rms, float(res.max()), float(np.linalg.norm(m)),
                 np.array2string(m, precision=2)))
        loo = np.zeros(n)
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            Ri, ti, mi = solve_point_marker(Pcam[idx], Pbe[idx], Rbe[idx])
            bi = Rbe[i].dot(mi) + Pbe[i]
            loo[i] = np.linalg.norm(Ri.dot(Pcam[i]) + ti - bi)
        loo_rms = float(np.sqrt((loo ** 2).mean()))
    else:
        R, t = kabsch(Pcam, Pbe)
        m = np.zeros(3)
        res = np.linalg.norm(Pcam.dot(R.T) + t - Pbe, axis=1)
        fit_rms = float(np.sqrt((res ** 2).mean()))
        print("  [plain Kabsch]")
        print("  fit RMS=%.3f max=%.3f mm" % (fit_rms, float(res.max())))
        loo = np.zeros(n)
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            Ri, ti = kabsch(Pcam[idx], Pbe[idx])
            loo[i] = np.linalg.norm(Ri.dot(Pcam[i]) + ti - Pbe[i])
        loo_rms = float(np.sqrt((loo ** 2).mean()))

    print("  LOO RMS=%.3f max=%.3f mm" % (loo_rms, float(loo.max())))
    print("  t (mm) =", np.array2string(t, precision=2))
    # if loo_rms > 3.0:
    #     print("  WARN: LOO > 3 mm — more pose spread / keep ArUco facing camera.")
    # if loo_rms > 8.0:
    #     print("  !! LOO > 8 mm — NOT saving (would overwrite a usable handeye_T.npz).")
    #     print("     Move tip more between SPACE presses; need >= 12 well-spread poses.")
    #     # #region agent log
    #     _dbg("D", "handeye_calibrate.py:solve_and_save", "refuse save high LOO",
    #          {"loo_rms": loo_rms, "fit_rms": fit_rms, "n": n,
    #           "m": m.tolist()}, runId="post-fix")
    #     # #endregion
    #     return False

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    if os.path.isfile(out_npz):
        bak = out_npz.replace(".npz", "_prev.npz")
        shutil.copy2(out_npz, bak)
        print("  backed up ->", bak)
    np.savez(
        out_npz, R=R, t=t, m=m, loo_rms=loo_rms, fit_rms=fit_rms
    )
    print("  saved", out_npz)
    print("  model: p_base = R @ p_cam + t  (mm);  m = ArUco in tip/EE frame")
    return True


def _as_row10(row):
    """Normalize a pair to length 10: cam(3)+tip(3)+quat(4)."""
    r = [float(x) for x in list(row)]
    if len(r) >= 10:
        return r[:10]
    if len(r) >= 6:
        # No orientation stored — use identity quat (m not observable for these).
        return r[:6] + [0.0, 0.0, 0.0, 1.0]
    raise ValueError("pair row too short: %s" % (r,))


def save_pairs_csv(pairs):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not pairs:
        return
    rows = [_as_row10(p) for p in pairs]
    arr = np.asarray(rows, dtype=np.float64)
    hdr = "cam_x,cam_y,cam_z,base_x,base_y,base_z,qx,qy,qz,qw"
    np.savetxt(PAIRS_CSV, arr, delimiter=",", header=hdr, comments="")


def load_pairs_csv(path):
    if not os.path.isfile(path):
        return []
    try:
        old = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception:
        return []
    if old.size == 0:
        return []
    if old.ndim == 1:
        old = old[None, :]
    out = []
    for row in old:
        try:
            out.append(_as_row10(row))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Quality-gated burst capture (avoids poisoning the fit with a shaky/blurred
# single-frame sample -- this is what made the first 6-pose set unreliable:
# LOO RMS 23.9mm on a fit_rms of only 1.8mm indicates noisy/inconsistent
# inputs, not a bad model).
# ---------------------------------------------------------------------------
BURST_N = 6
MAX_PIX_JITTER = 3.0     # px std of marker center across the burst
MAX_CAM_JITTER_MM = 2.0  # mm std of deprojected 3D point across the burst
MAX_EE_JITTER_MM = 0.15  # mm std of FrameEE across the burst (robot truly still)
# Reject SPACE if tip has not moved enough vs an existing sample (key-repeat
# previously accepted 6 identical tips in ~5s and poisoned the solve).
MIN_TIP_DELTA_MM = 5.0
SPACE_DEBOUNCE_S = 1.0


def _min_tip_delta_mm(p_ee, pairs):
    """Distance from p_ee to nearest already-recorded tip (mm)."""
    if not pairs:
        return float("inf")
    p = np.asarray(p_ee, dtype=np.float64).reshape(3)
    best = float("inf")
    for row in pairs:
        d = float(np.linalg.norm(p - np.asarray(row[3:6], dtype=np.float64)))
        if d < best:
            best = d
    return best


def dedupe_pairs_by_tip(pairs, min_delta_mm=MIN_TIP_DELTA_MM):
    """Keep first of each near-duplicate tip cluster."""
    kept = []
    dropped = 0
    for row in pairs:
        if _min_tip_delta_mm(row[3:6], kept) < min_delta_mm:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def capture_stable_sample(cam, auto_getter, burst_n=BURST_N):
    """Grab a short burst, require marker+depth+FrameEE present & STABLE in
    every frame. Returns (cam_mm, p_ee, q_ee, view, info) on success, or
    (None, None, None, view_or_None, reason) on rejection."""
    centers, cams, ees, view = [], [], [], None
    for i in range(burst_n):
        color, depth = cam.get_frames()
        if color is None:
            return None, None, None, None, "camera frame grab failed (%d/%d)" % (i + 1, burst_n)
        auto, uv_manual, p_ee, q_ee = auto_getter()
        center, corners, label = (None, None, "")
        if auto:
            center, corners, label = detect_aruco(color)
        if center is None and uv_manual is not None:
            center = np.array(uv_manual, float); label = "click"
        if center is None:
            # #region agent log
            _dbg("E", "handeye_calibrate.py:capture_stable_sample", "REJECT no marker in burst",
                 {"frame": i + 1, "burst_n": burst_n, "auto": auto, "uv_manual": uv_manual})
            # #endregion
            return None, None, None, color, "marker not detected (frame %d/%d)" % (i + 1, burst_n)
        if p_ee is None or q_ee is None:
            # #region agent log
            _dbg("D", "handeye_calibrate.py:capture_stable_sample", "REJECT no FrameEE in burst",
                 {"frame": i + 1})
            # #endregion
            return None, None, None, color, "no FrameEE yet -- bringup running?"
        cx, cy = int(round(center[0])), int(round(center[1]))
        mode = "nearest" if label == "click" else "median"
        z = cam.depth_at(depth, cx, cy, win=5, mode=mode)
        if not (np.isfinite(z) and z > 0.05):
            # #region agent log
            _dbg("C", "handeye_calibrate.py:capture_stable_sample", "REJECT depth invalid",
                 {"frame": i + 1, "z": z, "uv": [cx, cy], "label": label})
            # #endregion
            return None, None, None, color, "depth invalid at marker (frame %d/%d)" % (i + 1, burst_n)
        cam_mm = cam.deproject_mm(cx, cy, z)
        centers.append(center); cams.append(cam_mm)
        ees.append(np.hstack([p_ee, q_ee]))
        view = annotate_view(color, center, corners, label)
        time.sleep(0.04)

    centers = np.asarray(centers); cams = np.asarray(cams); ees = np.asarray(ees)
    pix_jitter = centers.std(axis=0).max()
    cam_jitter = np.linalg.norm(cams.std(axis=0))
    ee_jitter = np.linalg.norm(ees[:, 0:3].std(axis=0))
    if pix_jitter > MAX_PIX_JITTER:
        # #region agent log
        _dbg("A", "handeye_calibrate.py:capture_stable_sample", "REJECT pix_jitter",
             {"pix_jitter": float(pix_jitter), "cam_jitter": float(cam_jitter),
              "ee_jitter": float(ee_jitter), "limit": MAX_PIX_JITTER})
        # #endregion
        return None, None, None, view, "marker pixel jitter %.2fpx too high -- hold camera/marker still" % pix_jitter
    if cam_jitter > MAX_CAM_JITTER_MM:
        # #region agent log
        _dbg("A", "handeye_calibrate.py:capture_stable_sample", "REJECT cam_jitter",
             {"pix_jitter": float(pix_jitter), "cam_jitter": float(cam_jitter),
              "ee_jitter": float(ee_jitter), "limit": MAX_CAM_JITTER_MM,
              "cams": cams.tolist()})
        # #endregion
        return None, None, None, view, "depth/3D jitter %.2fmm too high -- reorient / steadier depth read" % cam_jitter
    if ee_jitter > MAX_EE_JITTER_MM:
        # #region agent log
        _dbg("A", "handeye_calibrate.py:capture_stable_sample", "REJECT ee_jitter",
             {"pix_jitter": float(pix_jitter), "cam_jitter": float(cam_jitter),
              "ee_jitter": float(ee_jitter), "limit": MAX_EE_JITTER_MM})
        # #endregion
        return None, None, None, view, "FrameEE jitter %.3fmm -- robot still settling, wait a beat" % ee_jitter

    cam_mm_avg = cams.mean(axis=0)
    ee_avg = ees.mean(axis=0)
    info = {"pix_jitter": pix_jitter, "cam_jitter": cam_jitter, "ee_jitter": ee_jitter}
    # #region agent log
    _dbg("A", "handeye_calibrate.py:capture_stable_sample", "ACCEPT burst",
         {"pix_jitter": float(pix_jitter), "cam_jitter": float(cam_jitter),
          "ee_jitter": float(ee_jitter), "cam_mm": cam_mm_avg.tolist(),
          "p_ee": ee_avg[0:3].tolist(), "centers": centers.tolist()})
    # #endregion
    return cam_mm_avg, ee_avg[0:3], ee_avg[3:7], view, info


def annotate_view(color, center, corners, label):
    out = color.copy()
    if corners is not None:
        cv2.polylines(out, [corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(out, label or "marker", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def put(img, txt, y, color=(255, 255, 255)):
    cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def estimate_handeye_from_pairs(pairs):
    """Quick R,t,m from recorded pairs (no LOO / no save). None if < 4."""
    if len(pairs) < 4:
        return None
    kept, _ = dedupe_pairs_by_tip(pairs)
    if len(kept) < 4:
        return None
    data = np.asarray([_as_row10(p) for p in kept], dtype=np.float64)
    Pcam, Pbe = data[:, 0:3], data[:, 3:6]
    Rbe = np.array([quat_to_R(q) for q in data[:, 6:10]])
    have_orient = np.any(np.abs(data[:, 6:9]) > 1e-6)
    if have_orient:
        R, t, m = solve_point_marker(Pcam, Pbe, Rbe)
    else:
        R, t = kabsch(Pcam, Pbe)
        m = np.zeros(3)
    return R, t, m


def base_to_pixel(p_base_mm, R_he, t_he, intr):
    """p_base = R_he @ p_cam + t_he  =>  p_cam = R_he.T @ (p_base - t_he)."""
    p_cam = R_he.T.dot(np.asarray(p_base_mm, float) - t_he)
    if p_cam[2] <= 1e-3:
        return None
    u = intr.fx * (p_cam[0] / p_cam[2]) + intr.ppx
    v = intr.fy * (p_cam[1] / p_cam[2]) + intr.ppy
    return float(u), float(v)


def draw_ee_frame(img, p_ee, q_ee, R_he, t_he, intr, axis_len_mm=20.0):
    """Project live FrameEE origin + XYZ axes into the camera image.

    Uses hand-eye from recorded pairs: p_base = R @ p_cam + t.
    Colors: X red, Y green, Z blue (OpenCV BGR).
    """
    if p_ee is None or q_ee is None or R_he is None or t_he is None:
        return False
    uv0 = base_to_pixel(p_ee, R_he, t_he, intr)
    if uv0 is None:
        return False
    R_ee = quat_to_R(q_ee)
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # X,Y,Z in BGR
    labels = ("X", "Y", "Z")
    o = (int(round(uv0[0])), int(round(uv0[1])))
    h, w = img.shape[:2]
    drawn = False
    for i in range(3):
        p_axis = np.asarray(p_ee, float) + R_ee[:, i] * axis_len_mm
        uv1 = base_to_pixel(p_axis, R_he, t_he, intr)
        if uv1 is None:
            continue
        a = (int(round(uv1[0])), int(round(uv1[1])))
        if (o[0] < -50 or o[1] < -50 or o[0] >= w + 50 or o[1] >= h + 50) and (
            a[0] < -50 or a[1] < -50 or a[0] >= w + 50 or a[1] >= h + 50
        ):
            continue
        cv2.arrowedLine(img, o, a, colors[i], 2, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.putText(
            img, labels[i], (a[0] + 4, a[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2, cv2.LINE_AA,
        )
        drawn = True
    if drawn or (0 <= o[0] < w and 0 <= o[1] < h):
        cv2.circle(img, o, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, o, 5, (0, 200, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            img, "EE", (o[0] + 8, o[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA,
        )
        return True
    return False


def draw_recorded_pair_marker(img, pair_row, R_he, t_he, intr):
    """Show last recorded tip (base) projected with current hand-eye."""
    if pair_row is None or R_he is None:
        return
    p_tip = np.asarray(pair_row[3:6], float)
    uv = base_to_pixel(p_tip, R_he, t_he, intr)
    if uv is None:
        return
    pt = (int(round(uv[0])), int(round(uv[1])))
    cv2.drawMarker(img, pt, (255, 128, 0), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.putText(
        img, "last tip", (pt[0] + 10, pt[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1, cv2.LINE_AA,
    )


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    rospy.init_node(
        "handeye_calibrate_aruco", anonymous=True, disable_signals=True
    )
    rospy.Subscriber(FRAMEEE_TOPIC, Transform, _cb_ee, queue_size=1)

    cam = RSCam()
    time.sleep(0.3)

    pairs = load_pairs_csv(PAIRS_CSV)
    if pairs:
        before = len(pairs)
        pairs, n_dup = dedupe_pairs_by_tip(pairs)
        print("Resumed %d pairs from %s" % (len(pairs), PAIRS_CSV))
        if n_dup:
            print("  purged %d near-duplicate tip pose(s) (were %d)" % (n_dup, before))
            save_pairs_csv(pairs)
        else:
            save_pairs_csv(pairs)  # rewrite as uniform 10-col
        # #region agent log
        _dbg("D", "handeye_calibrate.py:main", "resume pairs",
             {"before": before, "after": len(pairs), "dropped_dup": n_dup},
             runId="post-fix")
        # #endregion

    win = "Hand-eye eye-to-hand (fixed cam)  SPACE=record  s=solve  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    cv2.setMouseCallback(win, on_mouse)

    print("=== Hand-eye (eye-to-hand: camera FIXED on table) ===")
    print("  FrameEE :", FRAMEEE_TOPIC)
    print("  Output  :", OUT_NPZ)
    print("  Robot tip moves; camera stays fixed. Track ArUco on shaft (no QR).")
    print("  SPACE=record (tip must move >= %.1f mm)  s=solve  x=clear pairs  q=quit"
          % MIN_TIP_DELTA_MM)
    # #region agent log
    _dbg("B", "handeye_calibrate.py:main", "startup",
         {"pairs_resumed": len(pairs), "pairs_csv": PAIRS_CSV, "out_npz": OUT_NPZ,
          "MAX_PIX_JITTER": MAX_PIX_JITTER, "MAX_CAM_JITTER_MM": MAX_CAM_JITTER_MM,
          "MAX_EE_JITTER_MM": MAX_EE_JITTER_MM, "BURST_N": BURST_N,
          "MIN_TIP_DELTA_MM": MIN_TIP_DELTA_MM},
         runId="post-fix")
    # #endregion

    try:
        _loop_i = 0
        _prev_center = None
        _last_space_t = 0.0
        _he_cache = {"n": -1, "R": None, "t": None, "m": None}
        while not rospy.is_shutdown():
            color, depth = cam.get_frames()
            if color is None:
                continue
            view = color.copy()

            with _lock:
                auto = _auto["on"]
                uv_manual = _click["uv"]
                p_ee = None if _ee["p"] is None else _ee["p"].copy()
                q_ee = None if _ee["q"] is None else _ee["q"].copy()

            # Refresh hand-eye from recorded pairs whenever count changes.
            if len(pairs) != _he_cache["n"]:
                est = estimate_handeye_from_pairs(pairs)
                _he_cache["n"] = len(pairs)
                if est is None:
                    _he_cache["R"] = _he_cache["t"] = _he_cache["m"] = None
                else:
                    _he_cache["R"], _he_cache["t"], _he_cache["m"] = est

            center = None
            corners = None
            label = ""
            if auto:
                center, corners, label = detect_aruco(color)
            if center is None and uv_manual is not None:
                center = np.array(uv_manual, float)
                label = "click"

            cam_mm = None
            z = float("nan")
            if center is not None:
                cx, cy = int(round(center[0])), int(round(center[1]))
                if corners is not None:
                    cv2.polylines(
                        view,
                        [corners.astype(int).reshape(-1, 1, 2)],
                        True, (0, 255, 0), 2,
                    )
                cv2.circle(view, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    view, label or "marker", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                # Flat ArUco: median depth. Manual tip click: nearest.
                mode = "nearest" if label == "click" else "median"
                z = cam.depth_at(depth, cx, cy, win=5, mode=mode)
                if np.isfinite(z) and z > 0.05:
                    cam_mm = cam.deproject_mm(cx, cy, z)
                    put(
                        view,
                        "cam(mm)=[%.1f %.1f %.1f]  z=%.3fm  %s"
                        % (cam_mm[0], cam_mm[1], cam_mm[2], z, label),
                        72, (0, 255, 0),
                    )
                else:
                    put(view, "depth invalid at marker — reorient / closer", 72,
                        (0, 0, 255))

            # Live EE frame from pairs-based hand-eye (needs >= 4 pairs).
            ee_drawn = False
            if _he_cache["R"] is not None and p_ee is not None and q_ee is not None:
                ee_drawn = draw_ee_frame(
                    view, p_ee, q_ee, _he_cache["R"], _he_cache["t"], cam.intr
                )
                if pairs:
                    draw_recorded_pair_marker(
                        view, pairs[-1], _he_cache["R"], _he_cache["t"], cam.intr
                    )
                # Predicted ArUco (tip + R_ee @ m) for visual check vs green box.
                if _he_cache["m"] is not None:
                    R_ee = quat_to_R(q_ee)
                    p_m = p_ee + R_ee.dot(_he_cache["m"])
                    uv_m = base_to_pixel(p_m, _he_cache["R"], _he_cache["t"], cam.intr)
                    if uv_m is not None:
                        pm = (int(round(uv_m[0])), int(round(uv_m[1])))
                        cv2.circle(view, pm, 8, (255, 0, 255), 2, lineType=cv2.LINE_AA)
                        cv2.putText(
                            view, "pred ArUco", (pm[0] + 8, pm[1] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1,
                            cv2.LINE_AA,
                        )

            ready = cam_mm is not None and p_ee is not None and q_ee is not None
            # #region agent log
            _loop_i += 1
            if (_loop_i % 30) == 1:
                jump = None
                if center is not None and _prev_center is not None:
                    jump = float(np.linalg.norm(center - _prev_center))
                _dbg("B", "handeye_calibrate.py:main_loop", "preview tick",
                     {"aruco": center is not None, "label": label,
                      "center": None if center is None else [float(center[0]), float(center[1])],
                      "center_jump_px": jump, "z_m": float(z) if np.isfinite(z) else None,
                      "cam_mm": None if cam_mm is None else [float(x) for x in cam_mm],
                      "frame_ee": None if p_ee is None else [float(x) for x in p_ee],
                      "ready": ready, "auto": auto, "n_pairs": len(pairs),
                      "ee_overlay": ee_drawn})
                if center is not None:
                    _prev_center = np.array(center, float)
            # #endregion
            put(
                view,
                "pairs=%d  ArUco=%s  FrameEE=%s  auto=%s  ready=%s  EE-frame=%s"
                % (
                    len(pairs),
                    "YES" if center is not None else "no",
                    "YES" if p_ee is not None else "no",
                    "ON" if auto else "OFF",
                    "YES" if ready else "no",
                    "ON" if ee_drawn else ("need>=4 pairs" if len(pairs) < 4 else "off"),
                ),
                24,
                (0, 255, 0) if ready else (0, 200, 255),
            )
            if p_ee is not None:
                put(
                    view,
                    "tip FrameEE(mm)=[%.1f %.1f %.1f]" % tuple(p_ee),
                    48,
                )
            put(
                view,
                "SPACE=record | u=undo | x=clear | s=solve | m=auto | click=manual | q=quit",
                460, (200, 200, 200),
            )

            cv2.imshow(win, view)
            k = cv2.waitKey(30) & 0xFF

            if k in (ord("q"), 27):
                break
            elif k == ord("m"):
                with _lock:
                    _auto["on"] = not _auto["on"]
                    if _auto["on"]:
                        _click["uv"] = None
                print("  auto ArUco:", "ON" if _auto["on"] else "OFF")
            elif k == ord("c"):
                with _lock:
                    _click["uv"] = None
            elif k == ord("x"):
                n = len(pairs)
                pairs = []
                save_pairs_csv(pairs)
                # wipe CSV content
                open(PAIRS_CSV, "w").close()
                _he_cache = {"n": 0, "R": None, "t": None, "m": None}
                print("  cleared %d pairs (start fresh)" % n)
                # #region agent log
                _dbg("D", "handeye_calibrate.py:main", "cleared pairs",
                     {"cleared": n}, runId="post-fix")
                # #endregion
            elif k == ord("u") and pairs:
                pairs.pop()
                save_pairs_csv(pairs)
                print("  undo -> %d pairs" % len(pairs))
            elif k == ord("s"):
                # #region agent log
                _dbg("D", "handeye_calibrate.py:main", "solve requested",
                     {"n_pairs": len(pairs)}, runId="post-fix")
                # #endregion
                ok = solve_and_save(pairs, OUT_NPZ)
                # #region agent log
                try:
                    if ok and os.path.isfile(OUT_NPZ):
                        zz = np.load(OUT_NPZ)
                        _dbg("D", "handeye_calibrate.py:main", "solve result",
                             {"fit_rms": float(zz["fit_rms"]), "loo_rms": float(zz["loo_rms"]),
                              "m": zz["m"].tolist(), "t": zz["t"].tolist()},
                             runId="post-fix")
                    else:
                        _dbg("D", "handeye_calibrate.py:main", "solve not saved",
                             {"ok": ok}, runId="post-fix")
                except Exception as e:
                    _dbg("D", "handeye_calibrate.py:main", "solve result read fail",
                         {"err": str(e)}, runId="post-fix")
                # #endregion
            elif k in (ord(" "), ord("r")):
                now = time.time()
                if now - _last_space_t < SPACE_DEBOUNCE_S:
                    # #region agent log
                    _dbg("D", "handeye_calibrate.py:main", "SPACE debounce",
                         {"dt": now - _last_space_t}, runId="post-fix")
                    # #endregion
                    continue
                if p_ee is None or q_ee is None:
                    # #region agent log
                    _dbg("D", "handeye_calibrate.py:main", "SPACE blocked no FrameEE", {},
                         runId="post-fix")
                    # #endregion
                    print("  !! no FrameEE — is eye_robot running?")
                    continue
                if cam_mm is None:
                    # #region agent log
                    _dbg("B", "handeye_calibrate.py:main", "SPACE blocked no ArUco/depth",
                         {"center": None if center is None else [float(center[0]), float(center[1])],
                          "z": float(z) if np.isfinite(z) else None},
                         runId="post-fix")
                    # #endregion
                    print("  !! no ArUco/depth — face marker to camera")
                    continue

                tip_d = _min_tip_delta_mm(p_ee, pairs)
                if tip_d < MIN_TIP_DELTA_MM:
                    # #region agent log
                    _dbg("D", "handeye_calibrate.py:main", "SPACE rejected near-duplicate tip",
                         {"tip_delta_mm": tip_d, "min": MIN_TIP_DELTA_MM,
                          "p_ee": [float(x) for x in p_ee], "n_pairs": len(pairs)},
                         runId="post-fix")
                    # #endregion
                    print("  !! REJECTED: tip only %.2f mm from an existing pose "
                          "(need >= %.1f mm) — move the robot, then SPACE"
                          % (tip_d, MIN_TIP_DELTA_MM))
                    continue

                def _getter():
                    with _lock:
                        return (_auto["on"], _click["uv"],
                                None if _ee["p"] is None else _ee["p"].copy(),
                                None if _ee["q"] is None else _ee["q"].copy())

                put(view, "capturing burst -- hold still...", 96, (0, 255, 255))
                cv2.imshow(win, view)
                cv2.waitKey(1)
                cam_mm_b, p_ee_b, q_ee_b, view_b, info = capture_stable_sample(cam, _getter)
                if cam_mm_b is None:
                    # #region agent log
                    _dbg("A", "handeye_calibrate.py:main", "SPACE rejected",
                         {"reason": info}, runId="post-fix")
                    # #endregion
                    print("  !! REJECTED: %s" % info)
                    continue

                tip_d2 = _min_tip_delta_mm(p_ee_b, pairs)
                if tip_d2 < MIN_TIP_DELTA_MM:
                    # #region agent log
                    _dbg("D", "handeye_calibrate.py:main", "SPACE rejected dup after burst",
                         {"tip_delta_mm": tip_d2}, runId="post-fix")
                    # #endregion
                    print("  !! REJECTED: tip only %.2f mm from an existing pose after burst"
                          % tip_d2)
                    continue

                row = _as_row10(list(cam_mm_b) + list(p_ee_b) + list(q_ee_b))
                # #region agent log
                _dbg("D", "handeye_calibrate.py:main", "SPACE accepted",
                     {"n_after": len(pairs) + 1, "cam_mm": list(map(float, cam_mm_b)),
                      "p_ee": list(map(float, p_ee_b)), "tip_delta_from_prev_mm": tip_d2,
                      "jitters": info},
                     runId="post-fix")
                # #endregion
                pairs.append(row)
                _last_space_t = time.time()
                try:
                    save_pairs_csv(pairs)
                except Exception as e:
                    print("  !! save failed:", e)
                    pairs.pop()
                    continue
                idx = len(pairs) - 1
                cv2.imwrite(
                    os.path.join(FRAMES_DIR, "frame%03d.png" % idx),
                    view_b if view_b is not None else view,
                )
                msg = "  #%d cam=[%.1f %.1f %.1f] tip=[%.1f %.1f %.1f] (%s)  " \
                      "px_jit=%.2f cam_jit=%.2fmm ee_jit=%.3fmm  d_tip=%.1fmm" % (
                    idx,
                    cam_mm_b[0], cam_mm_b[1], cam_mm_b[2],
                    p_ee_b[0], p_ee_b[1], p_ee_b[2],
                    label or "?",
                    info["pix_jitter"], info["cam_jitter"], info["ee_jitter"],
                    tip_d2 if np.isfinite(tip_d2) else -1.0,
                )
                if len(pairs) >= 2:
                    prev = np.asarray(pairs[-2], dtype=np.float64)
                    dc = np.linalg.norm(cam_mm_b - prev[0:3])
                    db = np.linalg.norm(p_ee_b - prev[3:6])
                    msg += "  |dcam-dbase|=%.1f mm" % abs(dc - db)
                print(msg)
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        save_pairs_csv(pairs)
        print("Saved %d pairs -> %s" % (len(pairs), PAIRS_CSV))
        if len(pairs) >= 4:
            ans = input("Solve and save handeye_T.npz now? [y/N] ").strip().lower()
            if ans == "y":
                solve_and_save(pairs, OUT_NPZ)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print("Fatal:", type(e).__name__, e, file=sys.stderr)
        sys.exit(1)
