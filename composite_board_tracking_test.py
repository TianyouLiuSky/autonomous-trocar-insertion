"""
composite_board_tracker.py
Live webcam tracker for the ATI composite calibration board.
Detects: AprilTag36h11 corners (IDs 0-7) + center 6x6 ChArUco board.

Usage:
    python composite_board_tracker.py
    python composite_board_tracker.py --camera 1      # if laptop cam is index 1
    python composite_board_tracker.py --no-pose       # skip PnP if no intrinsics

Keys:
    Q / ESC  — quit
    S        — save current frame to disk
    P        — toggle pose axes overlay
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import time
from datetime import datetime

# ── Board config (must match generate_composite_board.py) ─────────────────────
SQUARE_MM    = 4.0
MARKER_RATIO = 0.70
TAG_MM       = 24.0       # AprilTag face size in mm
CHARUCO_SQ   = 6          # 6x6 ChArUco
CELL_MM      = 28.0       # cell size = board + 2*margin

# If you have intrinsics from your laptop cam, drop them in here.
# Otherwise leave None → pose estimation skipped, detection still works.
LAPTOP_K    = None   # np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
LAPTOP_DIST = None   # np.array([k1,k2,p1,p2,k3])

# ── Colors ────────────────────────────────────────────────────────────────────
COL_APRIL  = (0,   200, 255)   # orange-ish for AprilTags
COL_CHARU  = (0,   255, 120)   # green for ChArUco corners
COL_AXIS_X = (0,   0,   255)
COL_AXIS_Y = (0,   255, 0)
COL_AXIS_Z = (255, 0,   0)
COL_TEXT   = (255, 255, 255)
COL_BOX    = (30,  30,  30)

# ── Setup dicts ───────────────────────────────────────────────────────────────
april_dict   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

charuco_board = aruco.CharucoBoard(
    (CHARUCO_SQ, CHARUCO_SQ),
    SQUARE_MM / 1000.0,
    (SQUARE_MM * MARKER_RATIO) / 1000.0,
    charuco_dict
)

# Detector params — tuned for printed board under variable lighting
det_params = aruco.DetectorParameters()
det_params.adaptiveThreshWinSizeMin  = 5
det_params.adaptiveThreshWinSizeMax  = 23
det_params.adaptiveThreshWinSizeStep = 4
det_params.minMarkerPerimeterRate    = 0.02
det_params.maxMarkerPerimeterRate    = 0.5
det_params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX

april_detector   = aruco.ArucoDetector(april_dict,   det_params)
charuco_detector = aruco.CharucoDetector(charuco_board)


def draw_axes(img, K, dist, rvec, tvec, length_mm=15.0):
    """Draw XYZ axes at the marker origin."""
    pts, _ = cv2.projectPoints(
        np.float32([[0,0,0],[length_mm,0,0],[0,length_mm,0],[0,0,length_mm]]) / 1000.0,
        rvec, tvec, K, dist
    )
    o  = tuple(pts[0].ravel().astype(int))
    px = tuple(pts[1].ravel().astype(int))
    py = tuple(pts[2].ravel().astype(int))
    pz = tuple(pts[3].ravel().astype(int))
    cv2.arrowedLine(img, o, px, COL_AXIS_X, 2, tipLength=0.2)
    cv2.arrowedLine(img, o, py, COL_AXIS_Y, 2, tipLength=0.2)
    cv2.arrowedLine(img, o, pz, COL_AXIS_Z, 2, tipLength=0.2)
    cv2.putText(img, "X", px, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_AXIS_X, 1)
    cv2.putText(img, "Y", py, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_AXIS_Y, 1)
    cv2.putText(img, "Z", pz, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_AXIS_Z, 1)


def overlay_text(img, lines, x=10, y=20, scale=0.55, thickness=1):
    """Draw a block of text lines with a dark background."""
    lh = int(scale * 30)
    pad = 4
    max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
                for l in lines)
    cv2.rectangle(img, (x - pad, y - lh),
                  (x + max_w + pad, y + lh * len(lines) + pad),
                  COL_BOX, -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, COL_TEXT, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",   type=int,  default=0,    help="Camera index")
    parser.add_argument("--no-pose",  action="store_true",     help="Skip pose estimation")
    parser.add_argument("--width",    type=int,  default=1280, help="Capture width")
    parser.add_argument("--height",   type=int,  default=720,  help="Capture height")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        return

    K    = LAPTOP_K
    dist = LAPTOP_DIST
    show_pose = (K is not None) and (not args.no_pose)

    # Rough single-camera intrinsics from frame size if none provided
    # (good enough to eyeball axes, not for metric use)
    def make_fallback_K(w, h):
        f = max(w, h) * 0.85
        return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)

    frame_count = 0
    fps_t = time.time()
    fps = 0.0
    save_count = 0

    print("[INFO] Running. Q/ESC=quit  S=save  P=toggle pose")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if K is None:
            K_use = make_fallback_K(w, h)
            d_use = np.zeros(5)
        else:
            K_use = K
            d_use = dist

        display = frame.copy()

        # ── AprilTag detection ────────────────────────────────────────────
        april_corners, april_ids, april_rejected = april_detector.detectMarkers(gray)

        n_april = 0
        if april_ids is not None:
            n_april = len(april_ids)
            aruco.drawDetectedMarkers(display, april_corners, april_ids,
                                      borderColor=COL_APRIL)

            if show_pose or K is None:
                tag_half = (TAG_MM / 1000.0) / 2.0
                tag_obj = np.float32([
                    [-tag_half, -tag_half, 0],
                    [ tag_half, -tag_half, 0],
                    [ tag_half,  tag_half, 0],
                    [-tag_half,  tag_half, 0]
                ])
                for i, corners in enumerate(april_corners):
                    ok, rvec, tvec = cv2.solvePnP(
                        tag_obj, corners[0], K_use, d_use,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if ok and show_pose:
                        draw_axes(display, K_use, d_use, rvec, tvec)
                    # Distance label
                    if ok:
                        z_mm = tvec[2, 0] * 1000
                        cx_  = int(corners[0][:, 0].mean())
                        cy_  = int(corners[0][:, 1].mean())
                        tag_id = april_ids[i][0]
                        cv2.putText(display, f"AT{tag_id} {z_mm:.0f}mm",
                                    (cx_ - 20, cy_),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    COL_APRIL, 1, cv2.LINE_AA)

        # ── ChArUco detection ─────────────────────────────────────────────
        charu_corners, charu_ids, _, _ = charuco_detector.detectBoard(gray)

        n_charu = 0
        charu_pose_ok = False
        if charu_ids is not None and len(charu_ids) >= 4:
            n_charu = len(charu_ids)
            aruco.drawDetectedCornersCharuco(display, charu_corners, charu_ids,
                                             cornerColor=COL_CHARU)

            ok, rvec, tvec = cv2.solvePnP(
                charuco_board.getChessboardCorners()[charu_ids.flatten()],
                charu_corners, K_use, d_use
            )
            if ok:
                charu_pose_ok = True
                if show_pose:
                    draw_axes(display, K_use, d_use, rvec, tvec, length_mm=20.0)
                z_mm = tvec[2, 0] * 1000
                # Board center label
                cx_ = int(charu_corners[:, 0, 0].mean())
                cy_ = int(charu_corners[:, 0, 1].mean())
                cv2.putText(display, f"ChArUco {z_mm:.0f}mm",
                            (cx_ - 30, cy_ - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COL_CHARU, 1, cv2.LINE_AA)

        # ── FPS ───────────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            fps = 15 / (time.time() - fps_t)
            fps_t = time.time()

        # ── HUD ───────────────────────────────────────────────────────────
        pose_str = "ON (fallback K)" if (show_pose and K is None) else \
                   "ON (calibrated)" if show_pose else "OFF"
        hud = [
            f"FPS: {fps:.1f}",
            f"AprilTags: {n_april}",
            f"ChArUco corners: {n_charu}  pose={'ok' if charu_pose_ok else '--'}",
            f"Pose axes: {pose_str}",
            f"[S]ave  [P]ose  [Q]uit",
        ]
        overlay_text(display, hud)

        cv2.imshow("ATI Composite Board Tracker", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            fname = f"tracker_frame_{datetime.now().strftime('%H%M%S')}_{save_count:03d}.png"
            cv2.imwrite(fname, display)
            save_count += 1
            print(f"[SAVE] {fname}")
        elif key == ord('p'):
            show_pose = not show_pose
            print(f"[INFO] Pose axes {'ON' if show_pose else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()