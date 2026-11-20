#!/usr/bin/env python3
"""RGB camera intrinsics self-calibration (RealSense color stream) using the
physical ChArUco board this rig actually has -- companion to
camera_intrinsics_calibrate.py (which needs an ORDINARY plain checkerboard;
the user has no printer available on 2026-08-23, so this script targets the
ChArUco board instead, since cv2.findChessboardCorners() cannot detect a
ChArUco board's pattern -- the embedded per-square ArUco markers break the
plain corner-detector's assumptions about uniform black/white squares).

BOARD PARAMETERS -- measured empirically on 2026-08-23, NOT assumed:
  A raw 1280x720 RealSense color frame of the board (no display/window
  resizing in the loop) was analyzed offline with cv2.aruco.detectMarkers()
  using RELAXED DetectorParameters (default params found ZERO markers at
  both 640x480 and 1280x720 -- the embedded markers are small in-frame at
  this rig's working distance, so minMarkerPerimeterRate had to be lowered
  from the 0.03 default to 0.01, and the adaptive-threshold window range
  widened, before any marker was found at all):
    - dictionary: DICT_4X4_50 was the ONLY one (of 21 tried, all standard
      ArUco + AprilTag dictionaries) that decoded anything; it found 17 of
      18 markers with ids exactly spanning 0..17 (id 14 missed once, almost
      certainly a single glare/crease artifact on that one square -- 17/18
      is already well above the >=4 valid ids interpolateCornersCharuco
      needs per frame).
    - layout: marker centers analyzed on a uniform grid recovered EXACTLY
      6 marker-columns x 3 marker-rows-per-phase x 2 phases = 6 total rows,
      each row alternating which 3 of 6 columns are black (standard
      checkerboard coloring) -- i.e. SQUARES_X=6, SQUARES_Y=6, 18 black
      squares (ids 0-17), 18 white squares, matching DICT_4X4_50's own
      50-id capacity comfortably.
    - square size: printed "10 mm" scale bar visible on the board itself
      -> SQUARE_LENGTH_MM=10.0 (matches the measured ~27.5-28.75px marker
      center-to-center pitch in the raw frame, consistent across rows/cols).
    - marker size: measured marker side (~19.0px mean over 17 detections)
      vs measured square pitch (~28.1px) -> ratio ~0.676 -> MARKER_LENGTH_MM
      set to 6.8 (do not need extreme precision here: markerLength mainly
      aids initial marker-to-board association and pose sanity-checks;
      the checkerboard-corner subpixel positions that actually drive
      calibrateCameraCharuco come from interpolateCornersCharuco's own
      corner refinement, not from markerLength directly).
  These are all overridable via env vars below in case this board is ever
  swapped for a different physical one -- but do NOT change the defaults
  without re-measuring a real frame the same way (see
  Matthew/ISMR_Package_2026-08-20 session notes for the analysis commands).

Uses the OpenCV 4.5.2 LEGACY aruco API (CharucoBoard_create /
interpolateCornersCharuco / calibrateCameraCharuco) -- this cv2 build does
NOT have the newer CharucoDetector class.

CHANGED (2026-08-23, pt.2): the first version of this script kept all
captures ONLY in an in-memory list -- a live 37-capture session was lost in
full when the process exited (a stray keypress, most likely 'q'/ESC hitting
the OpenCV window while trying to press 's' twice) with no way to recover
even the raw corner data, let alone selectively prune the handful of bad
captures the user had identified from the per-view LOO error breakdown.
Fix: every accepted capture is now immediately pickled to a per-run session
directory (captures_session/) as its own file, AND on startup this script
auto-resumes by loading any files already there -- a crash/accidental-quit
no longer loses progress. Also added 'l' (list all captures with index +
corner count, so you can see exactly which ones are candidates for removal)
and 'x' (type a 1-based index on the keyboard, ENTER to confirm, to delete
that SPECIFIC capture -- not just the most recent one like 'u' does).

Keys:
  c        : try to interpolate ChArUco corners in the CURRENT frame and, if
             enough are found, add as a capture (shows overlay first)
  u        : undo LAST capture
  l        : list all captures (index + corner count) in the terminal
  x        : delete a SPECIFIC capture by index (type digits, ENTER to
             confirm, ESC to cancel) -- index is 1-based, as printed by 'l'
  s        : solve (cv2.aruco.calibrateCameraCharuco) + held-out check + save
  q / ESC  : quit

Run:
  python3 camera_intrinsics_calibrate_charuco.py
Env vars:
  CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y : total squares per side (default 6x6)
  CHARUCO_SQUARE_MM   : printed square size (default 10.0mm, see board label)
  CHARUCO_MARKER_MM   : printed marker size (default 6.8mm, measured ratio)
  CHARUCO_DICT        : cv2.aruco dictionary name (default DICT_4X4_50)
  MIN_CHARUCO_CORNERS : min interpolated corners to accept a capture (default 6)
  SESSION_DIR         : where per-capture files are persisted (default
                         captures_session/ next to this script) -- delete
                         this directory to start a fully fresh session.
Output:
  cam_intrinsics.npz  (backs up any existing file to cam_intrinsics_prev.npz)
  -- SAME schema as camera_intrinsics_calibrate.py's output, so downstream
     code (handeye scripts, home_to_rcm_insert*.py) does not need to change.
"""
from __future__ import print_function
import glob
import os
import pickle

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, "cam_intrinsics.npz")
PREV_NPZ = os.path.join(HERE, "cam_intrinsics_prev.npz")
SESSION_DIR = os.environ.get("SESSION_DIR", os.path.join(HERE, "captures_session"))

SQUARES_X = int(os.environ.get("CHARUCO_SQUARES_X", "6"))
SQUARES_Y = int(os.environ.get("CHARUCO_SQUARES_Y", "6"))
SQUARE_MM = float(os.environ.get("CHARUCO_SQUARE_MM", "10.0"))
MARKER_MM = float(os.environ.get("CHARUCO_MARKER_MM", "6.8"))
DICT_NAME = os.environ.get("CHARUCO_DICT", "DICT_4X4_50")
MIN_CORNERS = int(os.environ.get("MIN_CHARUCO_CORNERS", "6"))

ARUCO_DICT = cv2.aruco.Dictionary_get(getattr(cv2.aruco, DICT_NAME))
BOARD = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM, ARUCO_DICT)

# 2026-08-23: default DetectorParameters found ZERO markers on this board at
# either 640x480 or 1280x720 -- these small-marker-friendly overrides are
# what actually made detection work (see docstring above).
DET_PARAMS = cv2.aruco.DetectorParameters_create()
DET_PARAMS.adaptiveThreshWinSizeMin = 3
DET_PARAMS.adaptiveThreshWinSizeMax = 30
DET_PARAMS.adaptiveThreshWinSizeStep = 2
DET_PARAMS.minMarkerPerimeterRate = 0.01
DET_PARAMS.maxMarkerPerimeterRate = 4.0
DET_PARAMS.polygonalApproxAccuracyRate = 0.05


def find_charuco_corners(gray):
    """Returns (charuco_corners, charuco_ids) or (None, None)."""
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        gray, ARUCO_DICT, parameters=DET_PARAMS)
    if marker_ids is None or len(marker_ids) < 2:
        return None, None
    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, BOARD)
    if n < MIN_CORNERS:
        return None, None
    return ch_corners, ch_ids


def coverage_report(all_corners, w, h):
    hits = np.zeros((3, 3), dtype=int)
    for c in all_corners:
        pts = c.reshape(-1, 2)
        cx = np.clip((pts[:, 0] / w * 3).astype(int), 0, 2)
        cy = np.clip((pts[:, 1] / h * 3).astype(int), 0, 2)
        for a, b in zip(cx, cy):
            hits[b, a] += 1
    return hits


# ---- session persistence (2026-08-23 pt.2) ---------------------------------
def session_path(seq):
    return os.path.join(SESSION_DIR, "capture_%04d.pkl" % seq)


def load_session():
    """Resume any captures already on disk from a previous (possibly
    crashed/accidentally-quit) run of this script."""
    if not os.path.isdir(SESSION_DIR):
        os.makedirs(SESSION_DIR)
        return [], 0
    files = sorted(glob.glob(os.path.join(SESSION_DIR, "capture_*.pkl")))
    captures = []
    max_seq = 0
    for fp in files:
        try:
            with open(fp, "rb") as f:
                corners, ids = pickle.load(f)
            captures.append((corners, ids))
            seq = int(os.path.basename(fp).split("_")[1].split(".")[0])
            max_seq = max(max_seq, seq)
        except Exception as e:
            print("  !! could not load %s (%s), skipping" % (fp, e))
    if captures:
        print("  resumed %d capture(s) from previous session (%s)" %
              (len(captures), SESSION_DIR))
    return captures, max_seq


def persist_capture(seq, corners, ids):
    with open(session_path(seq), "wb") as f:
        pickle.dump((corners, ids), f)


def remove_all_session_files():
    for fp in glob.glob(os.path.join(SESSION_DIR, "capture_*.pkl")):
        os.remove(fp)


def rewrite_session_files(captures):
    """After a delete-by-index, renumber everything on disk 1..N so the
    files always mirror the in-memory list 1:1 (simplest invariant to keep
    load_session()/persist_capture() correct after arbitrary removals)."""
    remove_all_session_files()
    for i, (corners, ids) in enumerate(captures, start=1):
        persist_capture(i, corners, ids)


def main():
    print(__doc__)
    print("  board = %dx%d squares, square=%.2fmm marker=%.2fmm dict=%s" %
          (SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM, DICT_NAME))
    print("  session dir = %s (delete this folder for a fully fresh start)" % SESSION_DIR)

    captures, next_seq = load_session()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipe.start(cfg)
    intr0 = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    w, h = intr0.width, intr0.height
    for _ in range(15):
        pipe.wait_for_frames()

    win = "charuco intrinsics calib (c=capture u=undo l=list x=delete# s=solve q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    delete_mode = False
    delete_buf = ""

    try:
        while True:
            frames = pipe.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            view = img.copy()
            ch_corners, ch_ids = find_charuco_corners(gray)
            if ch_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(view, ch_corners, ch_ids, (0, 255, 0))
            cv2.putText(view, "captures=%d  %s" % (
                len(captures),
                ("BOARD FOUND (%d corners)" % len(ch_corners)) if ch_corners is not None else "no board"),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if delete_mode:
                cv2.putText(view, "DELETE which #? %s_  (ENTER=confirm, ESC=cancel)" % delete_buf,
                            (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(20) & 0xFF

            if delete_mode:
                if k in (ord('\r'), ord('\n'), 13, 10):
                    if delete_buf:
                        idx = int(delete_buf)
                        if 1 <= idx <= len(captures):
                            captures.pop(idx - 1)
                            rewrite_session_files(captures)
                            print("  deleted capture #%d -> %d captures remain" % (idx, len(captures)))
                        else:
                            print("  index %d out of range (1..%d), nothing deleted" % (idx, len(captures)))
                    delete_mode = False
                    delete_buf = ""
                elif k == 27:  # ESC cancels delete-entry only
                    print("  delete cancelled")
                    delete_mode = False
                    delete_buf = ""
                elif ord('0') <= k <= ord('9'):
                    delete_buf += chr(k)
                elif k in (8, 127):  # backspace
                    delete_buf = delete_buf[:-1]
                continue

            if k in (ord('q'), 27):
                break
            elif k == ord('c'):
                if ch_corners is None:
                    print("  no board detected (or too few corners), not captured")
                else:
                    captures.append((ch_corners, ch_ids))
                    next_seq += 1
                    persist_capture(next_seq, ch_corners, ch_ids)
                    print("  captured #%d (%d corners)" % (len(captures), len(ch_corners)))
            elif k == ord('u') and captures:
                captures.pop()
                rewrite_session_files(captures)
                print("  undo -> %d captures" % len(captures))
            elif k == ord('l'):
                print("  --- %d captures ---" % len(captures))
                for i, (c, _) in enumerate(captures, start=1):
                    print("   #%d: %d corners" % (i, len(c)))
            elif k == ord('x'):
                if not captures:
                    print("  no captures to delete")
                else:
                    delete_mode = True
                    delete_buf = ""
                    print("  type the capture # to delete (1..%d), ENTER to confirm, ESC to cancel" % len(captures))
            elif k == ord('s'):
                if len(captures) < 10:
                    print("  need >=10 captures (have %d), keep going" % len(captures))
                    continue
                solve_and_save(captures, w, h)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def solve_and_save(captures, w, h):
    n = len(captures)
    print("\n  solving with %d captures ..." % n)

    all_corners = [c for c, _ in captures]
    all_ids = [i for _, i in captures]

    hits = coverage_report(all_corners, w, h)
    print("  image coverage (3x3 grid, corner-detection hit counts):")
    for row in hits:
        print("   ", row)
    if hits[0, 0] == 0 or hits[0, 2] == 0 or hits[2, 0] == 0 or hits[2, 2] == 0:
        print("  !! WARNING: at least one image CORNER region has zero coverage --"
              " distortion coefficients will be poorly constrained there."
              " Add more captures with the board near the frame corners.")

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, BOARD, (w, h), None, None)
    print("  fit reprojection RMS = %.3f px" % rms)
    print("  K =\n", K)
    print("  dist =", dist.ravel())

    # held-out check (leave-one-out over captures), same discipline as the
    # plain-checkerboard script: don't trust a single in-sample RMS number.
    loo_errs = []
    for i in range(n):
        train_corners = [c for j, c in enumerate(all_corners) if j != i]
        train_ids = [idc for j, idc in enumerate(all_ids) if j != i]
        try:
            _, K_i, dist_i, _, _ = cv2.aruco.calibrateCameraCharuco(
                train_corners, train_ids, BOARD, (w, h), None, None)
        except cv2.error as e:
            print("  LOO fold %d failed to solve (%s), skipping" % (i, e))
            continue
        obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(
            BOARD, all_corners[i], all_ids[i])
        ok, rvec_i, tvec_i = cv2.solvePnP(obj_pts, img_pts, K_i, dist_i)
        proj, _ = cv2.projectPoints(obj_pts, rvec_i, tvec_i, K_i, dist_i)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
        loo_errs.append(float(np.sqrt(np.mean(err ** 2))))
    if loo_errs:
        loo_rms = float(np.sqrt(np.mean(np.array(loo_errs) ** 2)))
        print("  LOO RMS (held-out views) = %.3f px  (per-view, index=1-based capture#: %s)" %
              (loo_rms, np.array2string(np.array(loo_errs), precision=2)))
        worst = sorted(range(len(loo_errs)), key=lambda i: -loo_errs[i])[:5]
        print("  worst 5 captures by LOO error: %s" %
              ", ".join("#%d(%.1fpx)" % (i + 1, loo_errs[i]) for i in worst))
        if loo_rms > 2.0 * rms:
            print("  !! LOO RMS much worse than fit RMS -- likely overfitting / poor"
                  " coverage or distance-tilt diversity. Add more varied captures"
                  " before trusting this result, or press 'x' to delete the worst"
                  " offenders listed just above.")
    else:
        loo_rms = float("nan")
        print("  !! all LOO folds failed to solve -- cannot report held-out RMS")

    fx, fy = K[0, 0], K[1, 1]
    ppx, ppy = K[0, 2], K[1, 2]

    if os.path.isfile(OUT_NPZ):
        old = np.load(OUT_NPZ)
        np.savez(PREV_NPZ, **{k: old[k] for k in old.files})
        print("  backed up existing cam_intrinsics.npz -> %s" % PREV_NPZ)

    np.savez(OUT_NPZ, fx=fx, fy=fy, ppx=ppx, ppy=ppy,
             coeffs=dist.ravel()[:5], width=w, height=h,
             fit_rms_px=rms, loo_rms_px=loo_rms, n=n)
    print("  SAVED -> %s" % OUT_NPZ)
    print("  fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f" % (fx, fy, ppx, ppy))


if __name__ == "__main__":
    main()
