#!/usr/bin/env python3
"""Standalone RGB camera intrinsics self-calibration (RealSense color stream)
using an ORDINARY PRINTED CHECKERBOARD -- no ArUco/ChArUco needed at all.

WHY THIS MATTERS (see 2026-08-11 pipeline design discussion): cam_intrinsics.npz
currently just stores the RealSense FACTORY-reported intrinsics
(profile.get_stream(...).get_intrinsics()), never independently verified.
Factory RGB intrinsics on RealSense modules commonly carry a few pixels of
systematic error in the principal point and distortion coefficients. Since
handeye_reproj_refine.py / home_to_rcm_insert.py etc. all treat K,dist as
KNOWN CONSTANTS when solving for R,t,m (hand-eye) or when back-projecting
targets, any bias in K,dist becomes an unmeasured systematic bias baked into
every downstream number -- it will NOT show up in fit RMS or LOO RMS, because
those only measure how well the OTHER free parameters explain the data GIVEN
this (possibly wrong) K,dist. Standard camera calibration (Zhang's method via
cv2.calibrateCamera) is the correct, standard way to independently measure
K,dist with a physically known planar pattern before trusting them elsewhere.

HOW TO MAKE THE CHECKERBOARD: print any standard chessboard pattern (e.g.
search "9x6 chessboard calibration pattern pdf", print at 100% scale --
NO fit-to-page). Tape it to something rigid and FLAT (a clipboard / thick
cardboard / acrylic sheet -- warping ruins accuracy). Measure the actual
printed square size with calipers (not a ruler if possible) and set
SQUARE_SIZE_MM to the measured value.

PROTOCOL for a good calibration (matters more than magic parameter values):
  - >=20 captures, covering the ENTIRE image area including all 4 CORNERS of
    the frame (distortion is worst at the edges -- if you only ever show the
    board near the image center, the distortion coefficients are barely
    constrained and will be unreliable even with a low reported RMS).
  - Vary DISTANCE (near/far) and TILT (significant angle, not just
    fronto-parallel) across captures -- this is what makes focal length and
    principal point jointly observable (a stack of only fronto-parallel,
    same-distance shots is a classically ill-conditioned dataset for
    calibrateCamera, even with many captures).
  - Keep the board RIGID and STILL during each capture (the script grabs a
    live frame on keypress, no burst-averaging of the board itself, so a
    blurred capture silently hurts corner accuracy without any error message).

Keys:
  c        : try to detect the checkerboard in the CURRENT frame and, if
             found, add it as a capture (shows overlay so you can confirm
             corners look right before trusting it)
  u        : undo last capture
  s        : solve (cv2.calibrateCamera) + held-out check + save
  q / ESC  : quit

Run:
  python3 camera_intrinsics_calibrate.py
Env vars:
  BOARD_COLS, BOARD_ROWS   : internal corner counts (default 9x6)
  SQUARE_SIZE_MM           : measured square size (default 25.0 -- SET THIS)
Output:
  cam_intrinsics.npz  (backs up any existing file to cam_intrinsics_prev.npz)
"""
from __future__ import print_function
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, "cam_intrinsics.npz")
PREV_NPZ = os.path.join(HERE, "cam_intrinsics_prev.npz")

BOARD_COLS = int(os.environ.get("BOARD_COLS", "9"))   # internal corners, not squares
BOARD_ROWS = int(os.environ.get("BOARD_ROWS", "6"))
SQUARE_SIZE_MM = float(os.environ.get("SQUARE_SIZE_MM", "25.0"))

CRITERIA_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)


def make_objp():
    objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    grid = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp[:, :2] = grid * SQUARE_SIZE_MM
    return objp


def find_corners(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags=flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA_SUBPIX)
    return corners


def coverage_report(all_corners, w, h):
    """Rough diagnostic: what fraction of the image (split into a 3x3 grid)
    has ever had a corner land in it. Warns if edges/corners are unused,
    which is the single most common reason a checkerboard self-cal ends up
    LOOKING fine (low fit RMS) but has unreliable distortion coefficients."""
    hits = np.zeros((3, 3), dtype=int)
    for c in all_corners:
        pts = c.reshape(-1, 2)
        cx = np.clip((pts[:, 0] / w * 3).astype(int), 0, 2)
        cy = np.clip((pts[:, 1] / h * 3).astype(int), 0, 2)
        for a, b in zip(cx, cy):
            hits[b, a] += 1
    return hits


def main():
    print(__doc__)
    print("  board = %dx%d internal corners, square = %.2fmm" %
          (BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_MM))
    objp = make_objp()

    # 2026-08-23: bumped 640x480 -> 1280x720 (confirmed supported by this
    # RealSense unit) -- empirically, findChessboardCorners/detectMarkers
    # were both failing to find the board at all at 640x480 from this rig's
    # working distance; 4x the pixels gives small board features (esp. a
    # ChArUco board's embedded per-square markers) a much better chance of
    # being resolved. Also lowered FPS to 15 (720p/30 may not be supported
    # on every RealSense color sensor mode) -- fine for a capture-on-keypress
    # workflow, no live tracking speed requirement here.
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipe.start(cfg)
    intr0 = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    w, h = intr0.width, intr0.height
    for _ in range(15):
        pipe.wait_for_frames()

    img_points = []
    win = "checkerboard intrinsics calib (c=capture u=undo s=solve q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    try:
        while True:
            frames = pipe.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            view = img.copy()
            corners_live = find_corners(gray)
            if corners_live is not None:
                cv2.drawChessboardCorners(view, (BOARD_COLS, BOARD_ROWS), corners_live, True)
            cv2.putText(view, "captures=%d  %s" % (
                len(img_points), "BOARD FOUND" if corners_live is not None else "no board"),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('c'):
                if corners_live is None:
                    print("  no board detected, not captured")
                else:
                    img_points.append(corners_live)
                    print("  captured #%d" % len(img_points))
            elif k == ord('u') and img_points:
                img_points.pop()
                print("  undo -> %d captures" % len(img_points))
            elif k == ord('s'):
                if len(img_points) < 10:
                    print("  need >=10 captures (have %d), keep going" % len(img_points))
                    continue
                solve_and_save(img_points, objp, w, h)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def solve_and_save(img_points, objp, w, h):
    n = len(img_points)
    obj_points = [objp] * n
    print("\n  solving with %d captures ..." % n)

    hits = coverage_report(img_points, w, h)
    print("  image coverage (3x3 grid, corner-detection hit counts):")
    for row in hits:
        print("   ", row)
    if hits[0, 0] == 0 or hits[0, 2] == 0 or hits[2, 0] == 0 or hits[2, 2] == 0:
        print("  !! WARNING: at least one image CORNER region has zero coverage --"
              " distortion coefficients will be poorly constrained there."
              " Add more captures with the board near the frame corners.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)
    print("  fit reprojection RMS = %.3f px" % rms)
    print("  K =\n", K)
    print("  dist =", dist.ravel())

    # held-out check (leave-one-out over captures) -- same discipline as the
    # rest of this codebase: don't trust a single in-sample RMS number.
    loo_errs = []
    for i in range(n):
        train_obj = [objp] * (n - 1)
        train_img = [c for j, c in enumerate(img_points) if j != i]
        _, K_i, dist_i, rvecs_i, tvecs_i = cv2.calibrateCamera(
            train_obj, train_img, (w, h), None, None)
        # re-estimate this view's extrinsics with the LOO intrinsics, then
        # check reprojection error on the held-out view
        ok, rvec_i, tvec_i = cv2.solvePnP(objp, img_points[i], K_i, dist_i)
        proj, _ = cv2.projectPoints(objp, rvec_i, tvec_i, K_i, dist_i)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_points[i].reshape(-1, 2), axis=1)
        loo_errs.append(float(np.sqrt(np.mean(err ** 2))))
    loo_rms = float(np.sqrt(np.mean(np.array(loo_errs) ** 2)))
    print("  LOO RMS (held-out views) = %.3f px  (per-view: %s)" %
          (loo_rms, np.array2string(np.array(loo_errs), precision=2)))
    if loo_rms > 2.0 * rms:
        print("  !! LOO RMS much worse than fit RMS -- likely overfitting / poor"
              " coverage or distance-tilt diversity. Add more varied captures"
              " before trusting this result.")

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
