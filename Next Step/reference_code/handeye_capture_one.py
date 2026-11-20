#!/usr/bin/env python3
"""Headless, single-shot hand-eye sample capture: grab one burst-averaged
(camera-frame marker point, FrameEE pose) pair and APPEND it to a CSV in the
same schema as handeye_data/pairs.csv, so it can be combined with the
existing 12-pose set and refit by handeye_reproj_refine.py.

Exists because handeye_calibrate.py's own capture loop needs an interactive
cv2 window + keyboard (SPACE to record) which requires a human at the
keyboard; this script instead runs non-interactively right after the
operator has jogged the tool to a new, stationary orientation with
home_to_rcm_insert.py's k/m/j/l/u/o keys (in ANOTHER terminal, camera
released via 'q' before running this).

Usage:
    python3 handeye_capture_one.py [out_csv]   # default: handeye_data/pairs_extra_rot.csv
"""
import csv
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
OUT_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "handeye_data", "pairs_extra_rot.csv")

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


rospy.init_node("handeye_capture_one", anonymous=True, disable_signals=True)
rospy.Subscriber("/SHER21/eye_robot/FrameEE", Transform, _cb, queue_size=1)
rospy.sleep(1.0)
if ee["p"] is None:
    raise SystemExit("no FrameEE received -- is ROS bringup running?")

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

N = 12
centers = []
depths = []
p_samples = []
q_samples = []
frame_last = None
for i in range(N + 8):
    frames = pipe.wait_for_frames(timeout_ms=2000)
    frames = align.process(frames)
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth_frame = frames.get_depth_frame()
    frame_last = color
    center, corners, label = detect_aruco(color)
    if center is None:
        continue
    u, v = int(round(center[0])), int(round(center[1]))
    vals = []
    for du in range(-2, 3):
        for dv in range(-2, 3):
            d = depth_frame.get_distance(u + du, v + dv) * 1000.0
            if d > 0:
                vals.append(d)
    if not vals:
        continue
    z = float(np.median(vals))
    x = (center[0] - intr.ppx) / intr.fx * z
    y = (center[1] - intr.ppy) / intr.fy * z
    centers.append((x, y, z))
    p_samples.append(ee["p"].copy())
    q_samples.append(ee["q"].copy())
    if len(centers) >= N:
        break
pipe.stop()

if len(centers) < 6:
    cv2.imwrite(os.path.join(HERE, "handeye_capture_one_fail.png"), frame_last)
    raise SystemExit("only got %d/%d valid marker+depth frames -- marker not "
                      "reliably visible at this pose (saved last frame to "
                      "handeye_capture_one_fail.png for inspection)" % (len(centers), N))

centers = np.array(centers)
p_arr = np.array(p_samples)
q_arr = np.array(q_samples)
cam_pt = centers.mean(axis=0)
cam_jitter = centers.std(axis=0)
p_mean = p_arr.mean(axis=0)
p_jitter = p_arr.std(axis=0)
q_mean = q_arr.mean(axis=0)
q_mean /= np.linalg.norm(q_mean)

print("n_valid=%d/%d  cam_pt=%s (jitter mm std=%s)" % (len(centers), N, cam_pt, cam_jitter))
print("FrameEE p=%s (jitter mm std=%s)" % (p_mean, p_jitter))
if np.any(p_jitter > 0.5) or np.any(cam_jitter > 3.0):
    print("  !! WARNING: robot or marker doesn't look fully still -- consider "
          "waiting and retrying for a cleaner sample.")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
new_file = not os.path.isfile(OUT_CSV)
with open(OUT_CSV, "a") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["cam_x", "cam_y", "cam_z", "base_x", "base_y", "base_z", "qx", "qy", "qz", "qw"])
    w.writerow([cam_pt[0], cam_pt[1], cam_pt[2], p_mean[0], p_mean[1], p_mean[2],
                q_mean[0], q_mean[1], q_mean[2], q_mean[3]])
print("Appended 1 row -> %s" % OUT_CSV)
