#!/usr/bin/env python3
"""Same-day, independent visual cross-check between TODAY's hand-eye
calibration (handeye_T.npz) and TODAY's pivot calibration (tip_offset_pivot.npz).

WHY: comparing today's pivot d against the OLD tip_offset.npz (from a
different day, different camera position, unknown TipInEE at the time, and
possibly different needle mounting) has too many uncontrolled confounds to
be a valid check -- see chat 2026-08-11. This script instead projects
p_pivot (a pure-kinematics point from TODAY's pivot fit, independent of the
camera) through TODAY's hand-eye (R,t) into the live camera image. If both
of today's calibrations are self-consistent, this projected point should
land ON the visible physical needle tip, and should STAY on it as the tool
is rotated to different orientations (since p_pivot is supposed to be the
one fixed point the whole pivot calibration is built around).

Usage:
  python3 pivot_visual_crosscheck.py
Draws a magenta circle at the predicted pivot point on the live RealSense
color stream. Press 's' to save a snapshot, 'q' to quit.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import os
import numpy as np
import cv2
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))

he = np.load(data_path("handeye_T.npz"))
R_cb, t_cb = he["R"], he["t"]
piv = np.load(data_path("tip_offset_pivot.npz"))
p_pivot = piv["p_pivot"]
print("Loaded handeye_T.npz: t =", t_cb)
print("Loaded tip_offset_pivot.npz: p_pivot =", p_pivot, " n=", int(piv["n"]),
      " orientation_spread_deg=", float(piv["orientation_spread_deg"]))

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
dist = np.array(list(intr.coeffs[:5]), float)

for _ in range(15):
    pipe.wait_for_frames()

p_cam = R_cb.T.dot(p_pivot - t_cb)
uv, _ = cv2.projectPoints(p_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
u, v = float(uv[0, 0, 0]), float(uv[0, 0, 1])
print("p_pivot projects to pixel (u,v) = (%.1f, %.1f)  cam_depth=%.1fmm" % (u, v, p_cam[2]))

win = "pivot vs hand-eye visual cross-check (magenta = predicted p_pivot)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 960, 720)
snap_i = 0
try:
    while True:
        frames = pipe.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data()).copy()
        pt = (int(round(u)), int(round(v)))
        cv2.drawMarker(img, pt, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(img, pt, 14, (255, 0, 255), 2)
        cv2.putText(img, "predicted p_pivot", (pt[0] + 18, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "does this land ON the real needle tip?", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('s'):
            path = data_path("pivot_crosscheck_snap_%d.png" % snap_i)
            cv2.imwrite(path, img)
            print("saved", path)
            snap_i += 1
finally:
    pipe.stop()
    cv2.destroyAllWindows()
