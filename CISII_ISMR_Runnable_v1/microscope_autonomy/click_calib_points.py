#!/usr/bin/env python3
"""click_calib_points.py -- manually click the calibration landmarks
(calib_pt_1..calib_pt_N) on ONE microscope still, to pair with the
base-frame touch log recorded on the robot via the '1'-'8' keys in
home_to_rcm_insert_v2_precision.py.

Usage (run on the machine with the physical display, DISPLAY=:0):
    python3 click_calib_points.py /path/to/a/clean/microscope/still.jpg [--n-points N]

2026-09-24: default N raised from 4 to 8. Reasoning: the affine fit in
mic_extrinsic_calib.py has 6 free parameters (2x3 matrix); with exactly
4 points (8 equations) there are only 2 residual degrees of freedom left,
so the in-sample fit residual is nearly guaranteed to look small whether
or not the map actually generalizes across the field of view -- the same
"low residual does not imply accuracy" trap the ISMR gap analysis
documents for pivot calibration. 8 points, spread across the full field
of view (not clustered near the two ink marks), gives
mic_extrinsic_calib.py --loo a real held-out set to report an honest
accuracy number on. If you only have 4-6 physical landmarks you trust,
that's fine -- just pass --n-points to match how many '1'-'8' touches you
actually logged; --loo degrades gracefully (and warns) down to n=4.

Controls:
    click       -- place the next point (1, then 2, ... then N)
    u           -- undo last point
    s           -- save and quit (writes <image>.calib_px.json next to the image)
    q / ESC     -- quit without saving

IMPORTANT: pick a still where all N physical points you touched with the
robot are clearly visible and NOT occluded by the instrument/speculum
(use mic_perception.py's `occluded` flag, or just eyeball it -- the
2026-09-15/09-17 datasets show occlusion in ~15-20% of frames). Click them
in EXACTLY the same order/labels you touched them on the robot with the
'1'-'8' keys.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from runtime_session import ROOT, data_path
import sys
import os
import json
import cv2
import argparse

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("still")
ap.add_argument("--n-points", type=int, default=8,
                 help="how many calib_pt_1..N landmarks to click (default 8; match "
                      "however many '1'-'8' touches you logged on the robot)")
if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    raise SystemExit(1)
args = ap.parse_args()

img_path = args.still
img = cv2.imread(img_path)
if img is None:
    print("could not read", img_path)
    raise SystemExit(1)

if not (3 <= args.n_points <= 8):
    raise SystemExit("--n-points must be 3-8 (only 8 robot-touch keys '1'-'8' exist)")
labels = ["calib_pt_%d" % i for i in range(1, args.n_points + 1)]
points = []

win = "click_calib_points"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 720)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
        points.append((x, y))
        print("  %s = (%d, %d)" % (labels[len(points) - 1], x, y))


cv2.setMouseCallback(win, on_mouse)

print("Click the 4 points IN THE SAME ORDER you touched them on the robot:")
for lb in labels:
    print("  -", lb)
print("u=undo  s=save+quit  q/ESC=quit without saving")

while True:
    view = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.drawMarker(view, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(view, labels[i], (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
    nxt = labels[len(points)] if len(points) < len(labels) else "DONE -- press s to save"
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(view, "next: %s   (u=undo s=save q=quit)" % nxt, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(win, view)
    k = cv2.waitKey(20) & 0xFF
    if k in (27, ord('q')):
        print("quit without saving")
        break
    if k == ord('u') and points:
        removed = points.pop()
        print("  undo ->", removed)
    if k == ord('s'):
        if len(points) < len(labels):
            print("  need all %d points before saving (%d so far)" % (len(labels), len(points)))
            continue
        out = {labels[i]: {"u": points[i][0], "v": points[i][1]} for i in range(len(labels))}
        out["_source_image"] = os.path.abspath(img_path)
        out_path = img_path + ".calib_px.json"
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print("saved ->", out_path)
        break

cv2.destroyAllWindows()
