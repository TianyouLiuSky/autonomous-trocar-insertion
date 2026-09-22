"""ROS-independent geometry for the existing 84 mm composite target.

T_destination_from_source maps source coordinates into destination coordinates.
All distances in this module and exported transforms are in metres.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BOARD_SPEC = {
    "width_mm": 84.0, "height_mm": 84.0, "cell_mm": 28.0,
    "charuco_squares": [6, 6], "square_mm": 4.0, "marker_ratio": 0.70,
    "charuco_dictionary": "DICT_6X6_250", "charuco_origin_mm": [30.0, 30.0, 0.0],
    "april_dictionary": "DICT_APRILTAG_36h11", "tag_mm": 24.0,
    "axes": "origin at composite top-left; x right, y down, z into printed face",
}
TAG_CELLS = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2),
             4: (0, 1), 5: (1, 0), 6: (1, 2), 7: (2, 1)}


def write_json(path, value):
    """Atomic write, refusing to silently replace an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Output already exists: {} (choose a new filename)".format(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    metadata: dict

    @classmethod
    def from_dict(cls, data):
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
        size = data["image_size"]
        width, height = int(size["width"]), int(size["height"])
        model = data.get("distortion_model", "plumb_bob")
        if model not in ("plumb_bob", "rational_polynomial"):
            raise ValueError("Unsupported distortion model: {}".format(model))
        if (not np.isfinite(K).all() or not np.isfinite(dist).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0 or width <= 0 or height <= 0
                or not np.allclose(K[2], [0, 0, 1])
                or not np.allclose([K[0, 1], K[1, 0]], 0)
                or len(dist) not in (4, 5, 8, 12, 14)):
            raise ValueError("Invalid camera matrix, distortion coefficients, or image size")
        return cls(K, dist, width, height, dict(data))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).expanduser().read_text())
        data["loaded_from"] = str(Path(path).expanduser().resolve())
        return cls.from_dict(data)

    def check_image(self, image):
        if image.shape[:2] != (self.height, self.width):
            raise ValueError("Intrinsics are {}x{}, image is {}x{}; no automatic scaling"
                             .format(self.width, self.height, image.shape[1], image.shape[0]))


def make_charuco_board():
    return cv2.aruco.CharucoBoard(
        (6, 6), 0.004, 0.0028,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))


def charuco_points(ids):
    # getChessboardCorners is local to the TOP LEFT of the 24 mm pattern.
    return (make_charuco_board().getChessboardCorners()[np.asarray(ids).reshape(-1)]
            .astype(np.float64) + [0.030, 0.030, 0.0])


def tag_points(tag_id):
    row, col = TAG_CELLS[int(tag_id)]
    # Marker detector order is TL, TR, BR, BL. Generic IPPE accepts this
    # board-coordinate layout; IPPE_SQUARE requires a different local frame.
    center = np.array([(col + 0.5) * 0.028, (row + 0.5) * 0.028, 0.0])
    return center + np.array([[-.012, -.012, 0], [.012, -.012, 0],
                              [.012, .012, 0], [-.012, .012, 0]])


def transform(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def project(points, T, intrinsics):
    return cv2.projectPoints(np.asarray(points, dtype=np.float64),
                             cv2.Rodrigues(T[:3, :3])[0], T[:3, 3],
                             intrinsics.K, intrinsics.dist)[0].reshape(-1, 2)


def rotation_degrees(R):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))


def pose_difference(reference, observed):
    return (float(np.linalg.norm(reference[:3, 3] - observed[:3, 3]) * 1000),
            rotation_degrees(reference[:3, :3].T @ observed[:3, :3]))


def solve_board_pose(object_points, image_points, intrinsics,
                     max_rms_px=1.5, min_ambiguity_gap_px=0.05):
    obj = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
    if (len(obj) < 4 or len(obj) != len(img) or not np.isfinite(obj).all()
            or not np.isfinite(img).all()
            or np.linalg.matrix_rank(obj - obj.mean(axis=0), tol=1e-8) != 2):
        raise ValueError("Need non-collinear, finite planar correspondences")
    result = cv2.solvePnPGeneric(obj, img, intrinsics.K, intrinsics.dist,
                                flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rv, tv in zip(result[1], result[2]):
        T = transform(rv, tv)
        if not np.isfinite(T).all() or np.min((obj @ T[:3, :3].T + T[:3, 3])[:, 2]) <= 0:
            continue
        rms = float(np.sqrt(np.mean(np.sum((project(obj, T, intrinsics) - img)**2, axis=1))))
        candidates.append((rms, T))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        raise ValueError("No positive-depth planar pose")
    gap = None
    if len(candidates) > 1:
        gap = candidates[1][0] - candidates[0][0]
        angle = pose_difference(candidates[0][1], candidates[1][1])[1]
        if gap < min_ambiguity_gap_px and angle > 2.0:
            raise ValueError("Ambiguous planar pose: tilt target more (gap {:.3f}px)".format(gap))
    T = candidates[0][1]
    rv, tv = cv2.solvePnPRefineLM(obj, img, intrinsics.K, intrinsics.dist,
                                cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].copy())
    T = transform(rv, tv)
    rms = float(np.sqrt(np.mean(np.sum((project(obj, T, intrinsics) - img)**2, axis=1))))
    if (not np.isfinite(T).all() or not np.isfinite(rms)
            or np.min((obj @ T[:3, :3].T + T[:3, 3])[:, 2]) <= 0 or rms > max_rms_px):
        raise ValueError("Pose rejected: reprojection RMS {:.3f}px (limit {:.3f})"
                         .format(rms, max_rms_px))
    return {"T_camera_from_board": T.tolist(), "reprojection_rms_px": rms,
            "ambiguity_gap_px": gap, "object_points_m": obj.tolist(), "image_points_px": img.tolist()}


class BoardDetector:
    def __init__(self, source, intrinsics=None):
        self.source = source
        self.intrinsics = intrinsics
        if source == "charuco":
            params = cv2.aruco.CharucoParameters()
            if intrinsics is not None:
                params.cameraMatrix = intrinsics.K
                params.distCoeffs = intrinsics.dist
            self.detector = cv2.aruco.CharucoDetector(make_charuco_board(), params)
        elif source == "april":
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
        else:
            raise ValueError("Source must be charuco or april")

    def correspondences(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.source == "charuco":
            corners, ids, _, _ = self.detector.detectBoard(gray)
            if ids is None or len(ids) < 6:
                raise ValueError("Need at least 6 ChArUco corners")
            return charuco_points(ids), corners.reshape(-1, 2), ids.reshape(-1).tolist()
        corners, ids, _ = self.detector.detectMarkers(gray)
        selected = [] if ids is None else [(int(i), c) for i, c in zip(ids.reshape(-1), corners)
                                           if int(i) in TAG_CELLS]
        if len(selected) < 2 or len({i for i, _ in selected}) != len(selected):
            raise ValueError("Need at least 2 distinct AprilTags from IDs 0-7")
        return (np.concatenate([tag_points(i) for i, _ in selected]),
                np.concatenate([c.reshape(4, 2) for _, c in selected]), [i for i, _ in selected])

    def detect(self, image, max_rms_px=1.5, min_ambiguity_gap_px=0.05):
        self.intrinsics.check_image(image)
        obj, img, ids = self.correspondences(image)
        result = solve_board_pose(obj, img, self.intrinsics, max_rms_px, min_ambiguity_gap_px)
        result.update(source=self.source, ids=ids)
        return result


def mean_transform(transforms):
    Ts = np.asarray(transforms)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.mean(Ts[:, :3, 3], axis=0)
    return T


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values**2))),
            "max": float(values.max())}


def validate_pair_timing(stamps, ages, slop, max_age, used_stamps=()):
    if any(not np.isfinite(x) or x <= 0 for x in stamps):
        raise ValueError("Image timestamps must be positive and finite")
    if abs(stamps[0] - stamps[1]) > slop:
        raise ValueError("Image timestamps exceed pairing tolerance")
    if any(not np.isfinite(x) or x < -0.05 or x > max_age for x in ages):
        raise ValueError("Image is stale or its clock is inconsistent")
    if any((camera, stamp) in used_stamps for camera, stamp in enumerate(stamps)):
        raise ValueError("This frame has already been captured; wait for new frames")


def fit_registration(samples, intrinsics, min_fit=10, min_validation=5):
    fit = [s for s in samples if s["role"] == "fit"]
    validation = [s for s in samples if s["role"] == "validation"]
    if len(fit) < min_fit or len(validation) < min_validation:
        raise ValueError("Need at least {} fit and {} held-out validation pairs".format(min_fit, min_validation))
    def pair_transform(s):
        L = np.asarray(s["leica"]["T_camera_from_board"])
        D = np.asarray(s["d405"]["T_camera_from_board"])
        return D @ np.linalg.inv(L)
    T = mean_transform([pair_transform(s) for s in fit])
    results = {}
    for name, group in [("fit", fit), ("validation", validation)]:
        rows = []
        for s in group:
            L = np.asarray(s["leica"]["T_camera_from_board"])
            D = np.asarray(s["d405"]["T_camera_from_board"])
            predicted_D = T @ L
            predicted_L = np.linalg.inv(T) @ D
            translation, rotation = pose_difference(D, predicted_D)
            # Compare all composite corners, not only the board origin.
            board = np.array([[0, 0, 0], [.084, 0, 0], [.084, .084, 0], [0, .084, 0]])
            p = board @ predicted_D[:3, :3].T + predicted_D[:3, 3]
            q = board @ D[:3, :3].T + D[:3, 3]
            row = {"capture": s["capture"], "board_origin_error_mm": translation,
                   "rotation_error_deg": rotation,
                   "board_corner_rms_mm": float(np.sqrt(np.mean(np.sum((p-q)**2, axis=1))) * 1000)}
            for cam, predicted in [("leica", predicted_L), ("d405", predicted_D)]:
                record = s[cam]
                delta = project(record["object_points_m"], predicted, intrinsics[cam]) - record["image_points_px"]
                row[cam + "_cross_projection_rms_px"] = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))
            rows.append(row)
        results[name] = {"count": len(group), "samples": rows,
                         "summary": {k: summarize([r[k] for r in rows])
                                     for k in rows[0] if k != "capture"}}
    translations = [np.asarray(s["d405"]["T_camera_from_board"])[:3, 3] for s in fit]
    poses = [np.asarray(s["d405"]["T_camera_from_board"]) for s in fit]
    span = max(pose_difference(a, b)[1] for a in poses for b in poses)
    # These are diagnostic defaults, not the project's physical acceptance test.
    warnings = []
    if span < 10:
        warnings.append("Fit target orientation span <10 degrees; collect more diverse poses.")
    if results["validation"]["summary"]["board_corner_rms_mm"]["rms"] > 1.0:
        warnings.append("Held-out board corner consistency exceeds 1 mm RMS.")
    return {"schema_version": 1, "units": "metres", "board": BOARD_SPEC,
            "T_d405_color_from_leica": T.tolist(),
            "T_leica_from_d405_color": np.linalg.inv(T).tolist(),
            "equation": "p_d405_color = T_d405_color_from_leica @ p_leica (homogeneous metres)",
            "fit_translation_span_mm": (np.ptp(translations, axis=0) * 1000).tolist(),
            "fit_orientation_span_deg": span, "metrics": results, "warnings": warnings,
            "physical_accuracy_validated": False,
            "validation_note": "Held-out board consistency, not independent physical ground truth. "
                               "No smoothing, trimming or validation samples used in fitting."}
