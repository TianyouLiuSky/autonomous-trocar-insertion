import os
import glob
import json
import numpy as np
from collections import defaultdict
import cv2

VALIDATION_DIR = "./validation/21Feb2026/"
INTRINSICS_DIR = "./output/"
REPORT_PATH = "./output/validation_report_21Feb2026.txt"  # Change later

# -----------------------------
# Parsing intrinsics (your JSON format)
# -----------------------------
def parse_intrinsics(json_path):
    """
    Parses intrinsics JSON in the format:
    {
      "cam_id": "...",
      "image_size": {"width":..., "height":...},
      "rms_px": ...,
      "K": [[...],[...],[...]],
      "dist": [k1,k2,p1,p2,k3],
      "board": {...},
      "frames": {...}
    }
    Returns: (cam_id, cam_data_dict)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    cam_id = str(data["cam_id"])
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)

    image_size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    board_cfg = data.get("board", {})
    rms_px = data.get("rms_px", None)

    return cam_id, {
        "K": K,
        "dist": dist,
        "image_size": image_size,
        "board": board_cfg,
        "rms_px": rms_px,
        "source_path": json_path,
    }


def load_all_intrinsics(intrinsics_dir):
    """
    Loads all *.json intrinsics files from INTRINSICS_DIR and returns:
      intrinsics[cam_id] = cam_data
    """
    json_files = sorted(glob.glob(os.path.join(intrinsics_dir, "*.json")))
    intrinsics = {}
    for jf in json_files:
        try:
            cam_id, cam_data = parse_intrinsics(jf)
            intrinsics[cam_id] = cam_data
            print(f"[intrinsics] Loaded cam_id={cam_id} from {jf}")
        except Exception as e:
            print(f"[intrinsics] Skipping {jf}: {e}")
    return intrinsics


# -----------------------------
# Board + dict helpers
# -----------------------------
def aruco_dict_from_id(aruco_dict_id):
    """
    Your JSON has 'aruco_dict_id': 2.
    That is NOT guaranteed to match OpenCV's DICT_* enums.

    Common mapping used by many projects:
      0 -> DICT_4X4_50
      1 -> DICT_4X4_100
      2 -> DICT_4X4_250
      3 -> DICT_4X4_1000
      4 -> DICT_5X5_50
      ...
    If your detection fails, change this mapping to match how you created the board.
    """
    mapping = {
        0: cv2.aruco.DICT_4X4_50,
        1: cv2.aruco.DICT_4X4_100,
        2: cv2.aruco.DICT_4X4_250,
        3: cv2.aruco.DICT_4X4_1000,
        4: cv2.aruco.DICT_5X5_50,
        5: cv2.aruco.DICT_5X5_100,
        6: cv2.aruco.DICT_5X5_250,
        7: cv2.aruco.DICT_5X5_1000,
        8: cv2.aruco.DICT_6X6_50,
        9: cv2.aruco.DICT_6X6_100,
        10: cv2.aruco.DICT_6X6_250,
        11: cv2.aruco.DICT_6X6_1000,
        12: cv2.aruco.DICT_7X7_50,
        13: cv2.aruco.DICT_7X7_100,
        14: cv2.aruco.DICT_7X7_250,
        15: cv2.aruco.DICT_7X7_1000,
        16: cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    if aruco_dict_id not in mapping:
        raise ValueError(
            f"Unknown aruco_dict_id={aruco_dict_id}. Update mapping in aruco_dict_from_id()."
        )
    return mapping[aruco_dict_id]


def build_charuco_board(board_cfg):
    """
    board_cfg:
      squares_x, squares_y, square_mm, marker_mm, aruco_dict_id
    Returns: (board, detector)
    """
    squares_x = int(board_cfg["squares_x"])
    squares_y = int(board_cfg["squares_y"])
    square_len = float(board_cfg["square_mm"])
    marker_len = float(board_cfg["marker_mm"])
    aruco_id = int(board_cfg["aruco_dict_id"])

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_from_id(aruco_id))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    return board, detector


# -----------------------------
# Validation image discovery (YOUR NAMING CONVENTION)
# Example: 24213548_20260221201628.bmp
# cam_id is the prefix before the first underscore.
# -----------------------------
def collect_validation_images(validation_dir):
    """
    Collects all images in VALIDATION_DIR (single folder), grouped by cam_id,
    where cam_id is the filename prefix before the first underscore:
      <cam_id>_<timestamp>.<ext>

    Returns: images_by_cam[cam_id] = [list of image paths]
    """
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(validation_dir, ext)))
    all_paths = sorted(all_paths)

    images_by_cam = defaultdict(list)

    for p in all_paths:
        base = os.path.basename(p)
        if "_" not in base:
            # Not following convention; ignore or log
            continue
        cam_id = base.split("_", 1)[0]
        images_by_cam[cam_id].append(p)

    return images_by_cam


# -----------------------------
# Core validation: reprojection stats
# -----------------------------
def validate_camera_reprojection(cam_id, cam_data, image_paths):
    """
    Runs a hold-out reprojection validation for a single camera.
    Returns dict with summary + per-frame stats.

    Uses K/dist fixed from calibration and estimates pose per image using estimatePoseCharucoBoard.
    """
    K = cam_data["K"]
    dist = cam_data["dist"]
    fx = float(K[0, 0])
    fy = float(K[1, 1])

    board_cfg = cam_data["board"]
    board, detector = build_charuco_board(board_cfg)

    all_err = []
    all_ex = []
    all_ey = []
    per_frame = []

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            per_frame.append({"image": os.path.basename(p), "status": "read_fail"})
            continue

        # Detect markers
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None or len(ids) < 4:
            per_frame.append({"image": os.path.basename(p), "status": "no_markers", "n_charuco": 0})
            continue

        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, img, board
        )

        if retval is None or retval < 6:
            per_frame.append(
                {"image": os.path.basename(p), "status": "too_few_charuco", "n_charuco": int(retval) if retval else 0}
            )
            continue

        # Pose estimate
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist, None, None
        )
        if not ok:
            per_frame.append({"image": os.path.basename(p), "status": "pose_fail", "n_charuco": int(retval)})
            continue

        # 3D object points for detected ChArUco corners
        obj_all = board.getChessboardCorners()  # (N,3)
        obj_pts = obj_all[charuco_ids.flatten(), :]  # (n,3)
        img_pts = charuco_corners.reshape(-1, 2)     # (n,2)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        err_vec = img_pts - proj
        err = np.linalg.norm(err_vec, axis=1)
        ex = np.abs(err_vec[:, 0])
        ey = np.abs(err_vec[:, 1])

        all_err.append(err)
        all_ex.append(ex)
        all_ey.append(ey)

        per_frame.append(
            {
                "image": os.path.basename(p),
                "status": "ok",
                "n_charuco": int(err.shape[0]),
                "mean_px": float(np.mean(err)),
                "p95_px": float(np.percentile(err, 95)),
                "max_px": float(np.max(err)),
            }
        )

    # Concatenate global arrays
    if all_err:
        all_err = np.concatenate(all_err)
        all_ex = np.concatenate(all_ex)
        all_ey = np.concatenate(all_ey)
    else:
        all_err = np.array([])
        all_ex = np.array([])
        all_ey = np.array([])

    def stats(arr):
        if arr.size == 0:
            return None
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    # Angular small-angle approx: theta ~= e / f
    theta_x = (all_ex / fx) if all_ex.size else np.array([])
    theta_y = (all_ey / fy) if all_ey.size else np.array([])

    result = {
        "cam_id": cam_id,
        "n_images": len(image_paths),
        "n_ok_images": sum(1 for r in per_frame if r.get("status") == "ok"),
        "pixel_error": {
            "mag": stats(all_err),
            "ex": stats(all_ex),
            "ey": stats(all_ey),
        },
        "angular_error": {
            "theta_x_rad": stats(theta_x),
            "theta_y_rad": stats(theta_y),
            "theta_x_mrad_p95": float(np.percentile(theta_x, 95) * 1e3) if theta_x.size else None,
            "theta_y_mrad_p95": float(np.percentile(theta_y, 95) * 1e3) if theta_y.size else None,
        },
        "per_frame": per_frame,
    }

    return result


# -----------------------------
# Reporting
# -----------------------------
def format_report(all_results, intrinsics):
    lines = []
    lines.append("Camera Calibration Hold-out Validation Report")
    lines.append("================================================\n")

    for res in all_results:
        cam_id = res["cam_id"]
        lines.append(f"Camera: {cam_id}")
        lines.append(f"  Intrinsics file: {intrinsics[cam_id]['source_path']}")
        lines.append(f"  Training RMS (px): {intrinsics[cam_id].get('rms_px', None)}")
        lines.append(f"  Images validated: {res['n_images']} (OK: {res['n_ok_images']})")

        pe = res["pixel_error"]["mag"]
        if pe is None:
            lines.append("  Pixel error: NO VALID DATA (no successful frames)\n")
            continue

        lines.append("  Pixel reprojection error |e| (px):")
        lines.append(f"    mean   = {pe['mean']:.4f}")
        lines.append(f"    median = {pe['median']:.4f}")
        lines.append(f"    p95    = {pe['p95']:.4f}")
        lines.append(f"    max    = {pe['max']:.4f}")

        ax = res["angular_error"]["theta_x_rad"]
        ay = res["angular_error"]["theta_y_rad"]
        lines.append("  Angular error (small-angle) (rad):")
        lines.append(
            f"    theta_x: mean={ax['mean']:.6e}, p95={ax['p95']:.6e}  (~{res['angular_error']['theta_x_mrad_p95']:.3f} mrad p95)"
        )
        lines.append(
            f"    theta_y: mean={ay['mean']:.6e}, p95={ay['p95']:.6e}  (~{res['angular_error']['theta_y_mrad_p95']:.3f} mrad p95)"
        )

        lines.append("  Per-frame summary (only OK frames shown):")
        for fr in res["per_frame"]:
            if fr.get("status") != "ok":
                continue
            lines.append(
                f"    {fr['image']}: N={fr['n_charuco']}, mean={fr['mean_px']:.3f}px, p95={fr['p95_px']:.3f}px, max={fr['max_px']:.3f}px"
            )

        failures = [fr for fr in res["per_frame"] if fr.get("status") != "ok"]
        if failures:
            lines.append("  Non-OK frames:")
            for fr in failures:
                lines.append(f"    {fr['image']}: {fr.get('status')} (n_charuco={fr.get('n_charuco', 'NA')})")

        lines.append("")  # blank line

    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    # Load intrinsics for all cameras
    intrinsics = load_all_intrinsics(INTRINSICS_DIR)
    if not intrinsics:
        raise RuntimeError(f"No intrinsics JSON files found in {INTRINSICS_DIR}")

    # Collect validation images per camera (from filename prefix)
    images_by_cam = collect_validation_images(VALIDATION_DIR)
    if not images_by_cam:
        raise RuntimeError(
            f"No validation images found in {VALIDATION_DIR}. Expected files like <cam_id>_<timestamp>.bmp"
        )

    # Validate cameras that have both intrinsics + images
    results = []
    for cam_id, cam_data in intrinsics.items():
        if cam_id not in images_by_cam:
            print(f"[validate] No validation images found for cam_id={cam_id} in {VALIDATION_DIR}")
            continue

        img_paths = images_by_cam[cam_id]
        print(f"[validate] cam_id={cam_id}: validating {len(img_paths)} images...")
        res = validate_camera_reprojection(cam_id, cam_data, img_paths)
        results.append(res)

    if not results:
        raise RuntimeError("No cameras validated (no matching cam_id between intrinsics and validation image filenames).")

    report = format_report(results, intrinsics)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nWrote report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()