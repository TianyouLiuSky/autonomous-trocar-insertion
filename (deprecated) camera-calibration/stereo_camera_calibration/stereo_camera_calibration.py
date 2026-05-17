# Stereo extrinsic calibration for FLIR Blackfly S stereo pair
# Computes R, T, E, F between cam_left and cam_right
# Intrinsics are fixed (CALIB_FIX_INTRINSIC) — load from JSON files
#
# Requirements:
#   - Synced image pairs captured with collect_cal_images.py (dual mode)
#   - Individual intrinsics JSON files from single_camera_calibration
#   - Same ChArUco board used for intrinsic calibration
#
# Run:
#   python stereo_camera_calibration.py

import os
import re
import glob
import json
import numpy as np
import cv2
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG — fill these in
# ─────────────────────────────────────────────────────────────────────────────

# Intrinsics JSON files (from single_camera_calibration output)
INTRINSICS_RIGHT  = "../single_camera_calibration/output/calibration_25332589_31MAR2026.json"   # e.g. "../single_camera_calibration/output/calibration_24213548_31MAR2026.json"
INTRINSICS_LEFT = "../single_camera_calibration/output/calibration_24213548_31MAR2026.json"   # e.g. "../single_camera_calibration/output/calibration_25332589_31MAR2026.json"

# Directory containing synced image pairs
# Expected naming: <cam_id>_<timestamp>.bmp  (same timestamp = paired)
IMAGE_DIR = "./data/31Mar2026/"          # e.g. "./data/31Mar2026_stereo/"

# Output directory
OUT_DIR = "./output/"

# ChArUco board config — MUST match what was printed
# Should match what's stored in your intrinsics JSON but set explicitly here
# as a sanity check
SQUARES_X    = 6
SQUARES_Y    = 6
SQUARE_MM    = 3.5
MARKER_RATIO = 0.70
MARKER_MM    = SQUARE_MM * MARKER_RATIO
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250

# Cam IDs — must match prefix in image filenames and in intrinsics JSON
CAM_ID_LEFT  = "24213548"
CAM_ID_RIGHT = "25332589"

# Detection tuning
MIN_CHARUCO_CORNERS = 6   # minimum corners needed in BOTH images to use a pair

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    K    = np.array(data["K"],    dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    image_size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    return K, dist, image_size


def build_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        float(SQUARE_MM),
        float(MARKER_MM),
        aruco_dict
    )
    return board, cv2.aruco.CharucoDetector(board)


def detect_charuco(img_gray, board, detector):
    """
    Returns (charuco_corners, charuco_ids) or (None, None) if detection fails.
    """
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(img_gray)
    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None, None
    return charuco_corners, charuco_ids


def collect_pairs(image_dir, cam_id_left, cam_id_right):
    """
    Scans image_dir for files matching <cam_id>_<timestamp>.<ext>.
    Returns list of (path_left, path_right) matched by timestamp.
    """
    exts = ["*.bmp", "*.png", "*.jpg", "*.tif", "*.tiff"]
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    left_by_ts  = {}
    right_by_ts = {}

    for p in sorted(all_paths):
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        cam_id, ts = parts
        if cam_id == cam_id_left:
            left_by_ts[ts] = p
        elif cam_id == cam_id_right:
            right_by_ts[ts] = p

    # match by timestamp
    shared_ts = sorted(set(left_by_ts.keys()) & set(right_by_ts.keys()))
    pairs = [(left_by_ts[ts], right_by_ts[ts]) for ts in shared_ts]

    print(f"[pairs] Found {len(left_by_ts)} left, {len(right_by_ts)} right → {len(pairs)} matched pairs")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load intrinsics ───────────────────────────────────────────────────────
    print("[stereo_cal] Loading intrinsics...")
    K_left,  dist_left,  image_size_left  = load_intrinsics(INTRINSICS_LEFT)
    K_right, dist_right, image_size_right = load_intrinsics(INTRINSICS_RIGHT)

    assert image_size_left == image_size_right, \
        f"Image size mismatch: {image_size_left} vs {image_size_right}"
    image_size = image_size_left

    print(f"  Left  K:\n{K_left}")
    print(f"  Right K:\n{K_right}")

    # ── build board ───────────────────────────────────────────────────────────
    board, charuco_detector = build_board()

    # ── collect image pairs ───────────────────────────────────────────────────
    pairs = collect_pairs(IMAGE_DIR, CAM_ID_LEFT, CAM_ID_RIGHT)
    if not pairs:
        raise RuntimeError(f"No matched image pairs found in {IMAGE_DIR}")

    # ── detect corners in all pairs ───────────────────────────────────────────
    obj_points  = []   # 3D board points
    img_pts_left  = []
    img_pts_right = []

    accepted = 0
    rejected = 0

    for path_l, path_r in pairs:
        img_l = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)

        if img_l is None or img_r is None:
            print(f"  [WARN] Could not read pair: {os.path.basename(path_l)}")
            rejected += 1
            continue

        cc_l, ids_l = detect_charuco(img_l, board, charuco_detector)
        cc_r, ids_r = detect_charuco(img_r, board, charuco_detector)

        if cc_l is None or cc_r is None:
            rejected += 1
            continue

        # find common IDs visible in both images
        ids_l_flat = ids_l.flatten()
        ids_r_flat = ids_r.flatten()
        common_ids = np.intersect1d(ids_l_flat, ids_r_flat)

        if len(common_ids) < MIN_CHARUCO_CORNERS:
            rejected += 1
            continue

        # filter to common corners only
        mask_l = np.isin(ids_l_flat, common_ids)
        mask_r = np.isin(ids_r_flat, common_ids)

        cc_l_common  = cc_l[mask_l]
        cc_r_common  = cc_r[mask_r]
        ids_common   = ids_l[mask_l]

        # get 3D object points for common corners
        obj_pts, _ = board.matchImagePoints(cc_l_common, ids_common)
        if obj_pts is None or len(obj_pts) < MIN_CHARUCO_CORNERS:
            rejected += 1
            continue

        # recompute img_pts from matched subset to ensure alignment
        _, img_pts_l = board.matchImagePoints(cc_l_common, ids_common)
        _, img_pts_r = board.matchImagePoints(cc_r_common, ids_common)

        obj_points.append(obj_pts)
        img_pts_left.append(img_pts_l)
        img_pts_right.append(img_pts_r)
        print(f"  pair {accepted}: {len(common_ids)} common corners — {os.path.basename(path_l)}")

        accepted += 1

    print(f"[stereo_cal] Pairs accepted: {accepted} | rejected: {rejected}")

    if accepted < 10:
        raise RuntimeError("Not enough valid pairs — need at least 10. Check board config and image quality.")

    # ── stereo calibration ────────────────────────────────────────────────────
    print("[stereo_cal] Running stereoCalibrate...")
    

    flags = cv2.CALIB_FIX_INTRINSIC   # K and dist are fixed, only solve R, T, E, F

    rms, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_pts_left,
        img_pts_right,
        K_left,
        dist_left,
        K_right,
        dist_right,
        image_size,
        flags=flags
    )

    print(f"[stereo_cal] Stereo RMS reprojection error: {rms:.6f} px")
    print(f"[stereo_cal] R:\n{R}")
    print(f"[stereo_cal] T (mm):\n{T.flatten()}")
    baseline_mm = np.linalg.norm(T)
    print(f"[stereo_cal] Baseline: {baseline_mm:.3f} mm")

    # ── stereo rectify ────────────────────────────────────────────────────────
    print("[stereo_cal] Running stereoRectify...")

    R_left, R_right, P_left, P_right, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, dist_left,
        K_right, dist_right,
        image_size, R, T,
        alpha=0   # 0 = crop to valid pixels, 1 = keep all pixels
    )

    # rectification maps (use for remap in stereo depth pipeline)
    map_left_x,  map_left_y  = cv2.initUndistortRectifyMap(
        K_left,  dist_left,  R_left,  P_left,  image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, dist_right, R_right, P_right, image_size, cv2.CV_32FC1
    )

    print(f"[stereo_cal] ROI left:  {roi_left}")
    print(f"[stereo_cal] ROI right: {roi_right}")

    # ── save results ──────────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%d%b%Y").upper()

    # JSON — human readable extrinsics + rectification params
    json_path = os.path.join(OUT_DIR, f"stereo_calibration_{date_str}.json")
    payload = {
        "date": date_str,
        "stereo_rms_px": float(rms),
        "baseline_mm": float(baseline_mm),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "cam_left": {
            "cam_id": CAM_ID_LEFT,
            "K": K_left.tolist(),
            "dist": dist_left.reshape(-1).tolist(),
            "R_rect": R_left.tolist(),
            "P_rect": P_left.tolist(),
        },
        "cam_right": {
            "cam_id": CAM_ID_RIGHT,
            "K": K_right.tolist(),
            "dist": dist_right.reshape(-1).tolist(),
            "R_rect": R_right.tolist(),
            "P_rect": P_right.tolist(),
        },
        "R": R.tolist(),
        "T": T.flatten().tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "Q": Q.tolist(),
        "roi_left":  list(roi_left),
        "roi_right": list(roi_right),
        "pairs_used": accepted,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[stereo_cal] Saved JSON: {json_path}")

    # NPZ — includes rectification maps for direct use in depth pipeline
    npz_path = os.path.join(OUT_DIR, f"stereo_calibration_{date_str}.npz")
    np.savez(
        npz_path,
        R=R, T=T, E=E, F=F, Q=Q,
        K_left=K_left,   dist_left=dist_left,
        K_right=K_right, dist_right=dist_right,
        R_left=R_left,   R_right=R_right,
        P_left=P_left,   P_right=P_right,
        map_left_x=map_left_x,   map_left_y=map_left_y,
        map_right_x=map_right_x, map_right_y=map_right_y,
        roi_left=np.array(roi_left),
        roi_right=np.array(roi_right),
        image_size=np.array(image_size),
        baseline_mm=baseline_mm,
        rms=rms,
    )
    print(f"[stereo_cal] Saved NPZ:  {npz_path}")
    print(f"\n[stereo_cal] Done. Stereo RMS: {rms:.4f} px  |  Baseline: {baseline_mm:.3f} mm")


if __name__ == "__main__":
    main()