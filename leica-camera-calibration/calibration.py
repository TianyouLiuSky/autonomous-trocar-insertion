# Generate intrinsic matrix and distortion coefficients for a single camera using a CharUco board
# Steps:
# 1. Capture multiple images of the CharUco board from different angles and distances
# 2. Detect the CharUco corners and their corresponding IDs in each image
# 3. Use OpenCV's calibrateCameraCharuco function to compute the camera matrix (K) and distortion coefficients
import os
import glob
import json
import numpy as np
from collections import defaultdict
import cv2
from datetime import datetime
print(cv2.__version__)
print("Has CharucoDetector:", hasattr(cv2.aruco, "CharucoDetector"))
print("Has calibrateCameraCharuco:", hasattr(cv2.aruco, "calibrateCameraCharuco"))

# -----------------------------
# USER CONFIG
# -----------------------------
# IMAGE_DIR = "./data/21Feb2026/" # First cal
IMAGE_DIR = "./data/08Apr2026_1/" # Second cal 
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

# ChArUco board definition (MUST match what you printed)
# SQUARES_X = 13
# SQUARES_Y = 13
# SQUARE_MM = 3.0
# MARKER_RATIO = 0.70
# MARKER_MM = SQUARE_MM * MARKER_RATIO

# SQUARES_X = 10
# SQUARES_Y = 10
# SQUARE_MM = 4.0
# MARKER_RATIO = 0.70
# MARKER_MM = SQUARE_MM * MARKER_RATIO

# BOARD_MM = 24.0
# SQUARE_MM = 4.0
# MARKER_RATIO = 0.70
# SQUARES_X = 6
# SQUARES_Y = 6
# MARKER_MM = SQUARE_MM * MARKER_RATIO

BOARD_MM = 21.0
SQUARE_MM = 3.5
SQUARES_X = 6
SQUARES_Y = 6
MARKER_RATIO = 0.70
MARKER_MM = SQUARE_MM * MARKER_RATIO

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250

# Detection tuning
MIN_MARKERS = 4
MIN_CHARUCO_CORNERS = 12
SHOW_REJECTS_PREVIEW = False

OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def parse_camera_id(filename: str) -> str:
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    return parts[0] if parts else "unknown"

def imshow_fit(winname, img, max_w=1200, max_h=800):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)

def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Build board + detector
# -----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

board = cv2.aruco.CharucoBoard(
    size=(SQUARES_X, SQUARES_Y),
    squareLength=float(SQUARE_MM),
    markerLength=float(MARKER_MM),
    dictionary=aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# -----------------------------
# Load images grouped by camera id
# -----------------------------
all_paths = []
for ext in IMAGE_EXTS:
    all_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

if not all_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR} with extensions {IMAGE_EXTS}")

by_cam = defaultdict(list)
for p in sorted(all_paths):
    by_cam[parse_camera_id(p)].append(p)

print(f"Found {len(all_paths)} images across {len(by_cam)} camera ids: {list(by_cam.keys())}")

# -----------------------------
# Calibrate each camera independently
# -----------------------------
for cam_id, paths in by_cam.items():
    print("\n" + "=" * 80)
    print(f"Calibrating camera: {cam_id}  (images: {len(paths)})")

    # Build once per camera (safe, cheap, avoids accidental state issues)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        float(SQUARE_MM),
        float(MARKER_MM),
        aruco_dict
    )
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    accepted = 0
    rejected = 0

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] Could not read: {p}")
            rejected += 1
            continue

        gray = ensure_gray(img)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        # 1) Detect ArUco markers
        corners, ids, _ = aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) < MIN_MARKERS:
            rejected += 1
            if SHOW_REJECTS_PREVIEW:
                vis = img.copy()
                imshow_fit("Rejected (few markers)", vis)
                cv2.waitKey(30)
            continue

        # 2) Detect ChArUco corners/ids (OpenCV 4.13 path)
        out = charuco_detector.detectBoard(gray)
        charuco_corners, charuco_ids = out[0], out[1]

        if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
            rejected += 1
            if SHOW_REJECTS_PREVIEW:
                vis = img.copy()
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                if charuco_corners is not None and charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                imshow_fit("Rejected (few charuco)", vis)
                cv2.waitKey(30)
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        accepted += 1

        if SHOW_REJECTS_PREVIEW:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            imshow_fit("Accepted", vis)
            cv2.waitKey(30)

    print(f"Accepted frames: {accepted} | Rejected frames: {rejected}")

    if accepted < 10:
        print("  [ERROR] Not enough valid frames to calibrate. Get more images or improve detection.")
        continue

    # Build matched object/image point lists for cv2.calibrateCamera
    obj_points = []
    img_points = []
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and len(obj_pts) >= 4:
            obj_points.append(obj_pts)
            img_points.append(img_pts)

    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags
    )

    print(f"RMS reprojection error: {rms:.6f} px")
    print("K =\n", K)
    print("dist =", dist.reshape(-1))

    npz_path = os.path.join(OUT_DIR, f"calibration_{cam_id}_{datetime.now().strftime('%d%b%Y').upper()}.npz")
    np.savez(
        npz_path,
        cam_id=cam_id,
        image_size=np.array(image_size),
        rms=rms,
        K=K,
        dist=dist,
        squares_x=SQUARES_X,
        squares_y=SQUARES_Y,
        square_mm=SQUARE_MM,
        marker_mm=MARKER_MM,
        aruco_dict_id=ARUCO_DICT_ID,
        accepted=accepted,
        rejected=rejected
    )
    print(f"Saved: {npz_path}")

    json_path = os.path.join(OUT_DIR, f"calibration_{cam_id}_{datetime.now().strftime('%d%b%Y').upper()}.json") 
    payload = {
        "cam_id": cam_id,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "rms_px": float(rms),
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "board": {
            "squares_x": int(SQUARES_X),
            "squares_y": int(SQUARES_Y),
            "square_mm": float(SQUARE_MM),
            "marker_mm": float(MARKER_MM),
            "aruco_dict_id": int(ARUCO_DICT_ID),
        },
        "frames": {"accepted": int(accepted), "rejected": int(rejected)},
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {json_path}")

print("\nDone.")
if SHOW_REJECTS_PREVIEW:
    cv2.destroyAllWindows()