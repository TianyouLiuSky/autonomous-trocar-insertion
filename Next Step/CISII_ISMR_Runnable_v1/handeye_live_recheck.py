#!/usr/bin/env python3
"""Same-day apples-to-apples re-check: has hand-eye (R,t,m) drifted since the
16:31 calibration finalized? Detects the SAME ArUco marker used during
calibration in the CURRENT live frame, predicts its pixel location from
current FrameEE + handeye_T.npz's R,t,m, and reports the reprojection error
in the exact same units (px) that handeye_reproj_refine.py reported for the
100-pose fit/LOO (4.34px / 3.02px val). If this live number is similar,
hand-eye is still valid now; if it's much larger, something (camera or
mount) moved after 16:31.

Usage: python3 handeye_live_recheck.py
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handeye_calibrate import detect_aruco  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
he = np.load(data_path("handeye_T.npz"))
R_cb, t_cb, m = he["R"], he["t"], he["m"]
print("Loaded handeye_T.npz (fit=%.2fpx loo=%.2fpx val=%.2fpx, n=%d)" %
      (float(he["fit_rms_px"]), float(he["loo_rms_px"]), float(he["val_rms_px"]), int(he["n"])))

ee = {"p": None, "q": None}


def _cb(msg):
    ee["p"] = np.array([msg.translation.x, msg.translation.y, msg.translation.z], float)
    ee["q"] = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w], float)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


rospy.init_node("handeye_live_recheck", anonymous=True, disable_signals=True)
rospy.Subscriber("/SHER21/eye_robot/FrameEE", Transform, _cb, queue_size=1)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
intr = __import__("intr_util").load_best_intrinsics(profile)
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
dist = np.array(list(intr.coeffs[:5]), float)
for _ in range(15):
    pipe.wait_for_frames()

win = "hand-eye live re-check (green=predicted marker, red=detected marker)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 960, 720)

try:
    t_end = time.time() + 60
    while time.time() < t_end:
        frames = pipe.wait_for_frames(timeout_ms=2000)
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data()).copy()

        center, corners, label = detect_aruco(img)

        if ee["p"] is not None:
            R_be = quat_to_R(ee["q"])
            base_pt = R_be.dot(m) + ee["p"]
            cam_pt = R_cb.T.dot(base_pt - t_cb)
            uv_pred, _ = cv2.projectPoints(cam_pt.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
            up, vp = float(uv_pred[0, 0, 0]), float(uv_pred[0, 0, 1])
            cv2.drawMarker(img, (int(up), int(vp)), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(img, "predicted", (int(up) + 14, int(vp) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if center is not None:
                cv2.drawMarker(img, (int(center[0]), int(center[1])), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                cv2.putText(img, "detected %s" % label, (int(center[0]) + 14, int(center[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                err_px = float(np.hypot(up - center[0], vp - center[1]))
                cv2.putText(img, "LIVE reproj error = %.1f px  (calib reported fit=%.1f loo=%.1f val=%.1f)" %
                            (err_px, float(he["fit_rms_px"]), float(he["loo_rms_px"]), float(he["val_rms_px"])),
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                print("live reproj error = %.2f px" % err_px)
            else:
                cv2.putText(img, "marker not detected this frame", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('s'):
            cv2.imwrite(data_path("handeye_live_recheck_snap.png"), img)
            print("saved snapshot")
finally:
    pipe.stop()
    cv2.destroyAllWindows()
