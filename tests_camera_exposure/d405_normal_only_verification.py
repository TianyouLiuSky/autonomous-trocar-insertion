#!/usr/bin/env python3
"""
D405 NORMAL-ONLY VERIFICATION
=============================

Purpose
-------
Validate the CURRENT phantom-normal estimator with the proposed D405 image
settings while keeping the robot completely uninvolved.

Default camera settings:
    Exposure = 12000
    Gain     = 16

This script:
  1) opens only the Intel RealSense D405;
  2) runs the CURRENT integrated ring detector logic:
       LAB a* primary path
       close kernel = 11 px
       nearby-fragment merge fallback
       HSV Saturation/Otsu sticky-lock fallback
       production residual threshold = 0.32
  3) builds the SAME 15-sample / <3 mm 3D ring lock;
  4) FREEZES the first stable RCM point, like accepting a 'z' lock;
  5) repeatedly runs the CURRENT local-plane phantom-normal estimator:
       15 px circular patch
       red-rim exclusion with LAB a* > 145
       SVD plane fit
       centroid disagreement gate <= 5 mm
       plane RMS diagnostic
  6) compares every new normal with the existing phantom_normal.npz;
  7) reports internal repeatability of the new normal measurements.

Safety
------
- NO rospy import.
- NO robot publishers/subscribers.
- NO SHER commands.
- NO production auto_target.npz write.
- NO production phantom_normal.npz write.
- NO calibration-file modification.
- Only reads calibration files and writes into its own
  normal_verification_runs/<timestamp>/ folder.

The script is designed to live in ~/Desktop/tests and automatically searches
/home/eyebrp/Matthew for:
    handeye_T.npz
    phantom_normal.npz
    intr_util.py

Controls
--------
q / ESC : quit early
s       : save one extra annotated snapshot
r       : reset the temporary ring lock + normal samples IN MEMORY ONLY

Examples
--------
python3 d405_normal_only_verification.py

Collect 50 successful normal estimates:
python3 d405_normal_only_verification.py --samples 50

Run indefinitely until q/ESC:
python3 d405_normal_only_verification.py --seconds 0
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ---------------------------------------------------------------------------
# Known project location
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_ROOT = Path("/home/eyebrp/Matthew")
PREFERRED_CODE_DIR = DEFAULT_PROJECT_ROOT / "calibration_work_2026-09-21" / "01_code"

# ---------------------------------------------------------------------------
# Proposed camera settings
# ---------------------------------------------------------------------------
DEFAULT_EXPOSURE = 12000.0
DEFAULT_GAIN = 16.0

# ---------------------------------------------------------------------------
# Current 2026-09-28 production ring configuration
# ---------------------------------------------------------------------------
RING_A_STAR_THRESH = 145
RING_MIN_AREA_PX = 1200
RING_MAX_AREA_PX = 20000
RING_MAX_RESIDUAL_RATIO = 0.32
RING_MIN_RADIUS_PX = 20.0

RING_ROI_X0 = 100
RING_ROI_X1 = 540
RING_ROI_Y0 = 200
RING_ROI_Y1 = 480

RING_SMOOTH_N = 15
RING_MAX_JITTER_MM = 3.0
RING_STICKY_MAX_JUMP_PX = 80.0
RING_DEPTH_WIN_PX = 5

RING_CLOSE_KERNEL_PX = 11
RING_MERGE_MAX_DIST_PX = 150.0
RING_MERGE_MIN_PIECE_PX = 150.0

RING_USE_SAT_FALLBACK = True
RING_SAT_OTSU_MIN = 30.0
RING_SAT_OTSU_MAX = 200.0

# ---------------------------------------------------------------------------
# Current 2026-09-28 phantom-normal configuration
# ---------------------------------------------------------------------------
NORMAL_A_STAR_THRESH = 145
NORMAL_PATCH_RADIUS_PX = 15
NORMAL_MIN_POINTS = 80
NORMAL_MAX_CENTROID_DISAGREE_MM = 5.0
NORMAL_MAX_PLANE_RMS_MM = 1.0

# ---------------------------------------------------------------------------
# Display / logging
# ---------------------------------------------------------------------------
DEFAULT_SECONDS = 45.0
DEFAULT_SAMPLE_TARGET = 50
DEFAULT_NORMAL_EVERY = 15
SAVE_EVERY_N = 120

# Eye-only exposure diagnostic ROI from prior exposure tests.
EYE_ROI = (140, 285, 285, 430)
SATURATION_GRAY_THRESHOLD = 250


# ===========================================================================
# Project-file discovery
# ===========================================================================

def discover_project_file(filename, explicit_path=None, project_root=DEFAULT_PROJECT_ROOT):
    root = Path(project_root).expanduser()

    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.exists():
            return p, [p], "explicit"
        return None, [], f"explicit path missing: {p}"

    preferred = root / "calibration_work_2026-09-21" / "01_code" / filename
    if preferred.exists():
        return preferred.resolve(), [preferred.resolve()], "preferred 01_code"

    # handeye_T.npz is known to live under 03_results_npz on the current rig.
    if filename == "handeye_T.npz":
        handeye_preferred = root / "calibration_work_2026-09-21" / "03_results_npz" / filename
        if handeye_preferred.exists():
            return handeye_preferred.resolve(), [handeye_preferred.resolve()], "preferred 03_results_npz"

    direct = root / filename
    if direct.exists():
        return direct.resolve(), [direct.resolve()], "project-root direct child"

    if not root.exists():
        return None, [], f"project root missing: {root}"

    try:
        matches = [p.resolve() for p in root.rglob(filename) if p.is_file()]
    except Exception as e:
        return None, [], f"recursive search failed: {e}"

    if not matches:
        return None, [], f"not found under {root}"

    def rank(p):
        s = str(p)
        if filename == "handeye_T.npz" and "03_results_npz" in s:
            preferred_rank = 0
        elif "calibration_work_2026-09-21/01_code" in s:
            preferred_rank = 1
        else:
            preferred_rank = 2

        try:
            depth = len(p.relative_to(root).parts)
        except Exception:
            depth = 999

        try:
            newest = -p.stat().st_mtime
        except Exception:
            newest = 0.0

        return (preferred_rank, depth, newest, s)

    matches.sort(key=rank)
    return matches[0], matches, "recursive project search"


def print_discovery(label, chosen, matches, reason):
    if chosen is None:
        print(f"{label}: MISSING ({reason})")
        return

    print(f"{label}: {chosen}  [{reason}]")

    if len(matches) > 1:
        print(f"  {len(matches)} copies found; using the path above.")
        for p in matches[1:8]:
            print(f"    other: {p}")
        if len(matches) > 8:
            print(f"    ... plus {len(matches) - 8} more")


def prepare_intrinsics_import(project_root, explicit_intr_util=None):
    chosen, matches, reason = discover_project_file(
        "intr_util.py",
        explicit_path=explicit_intr_util,
        project_root=project_root,
    )
    print_discovery("intr_util.py", chosen, matches, reason)

    if chosen is None:
        return None

    parent = str(chosen.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    return chosen


def load_handeye(path):
    if path is None:
        return None, None, "missing"

    p = Path(path)

    try:
        d = np.load(str(p))
        R = np.asarray(d["R"], dtype=float)
        t = np.asarray(d["t"], dtype=float).reshape(3)

        if R.shape != (3, 3):
            return None, None, f"invalid R shape {R.shape}"

        return R, t, str(p)
    except Exception as e:
        return None, None, f"load failed: {e}"


def load_baseline_normal(path):
    if path is None:
        return None, "missing"

    p = Path(path)

    try:
        d = np.load(str(p))
        normal = np.asarray(d["normal_base"], dtype=float).reshape(3)
        normal = normal / np.linalg.norm(normal)

        rec = {
            "normal_base": normal,
            "point_base": np.asarray(d["point_base"], dtype=float).reshape(3),
            "n_points": int(d["n_points"]) if "n_points" in d.files else None,
            "plane_rms_mm": float(d["plane_rms_mm"]) if "plane_rms_mm" in d.files else None,
            "timestamp": float(d["t"]) if "t" in d.files else None,
        }

        return rec, str(p)
    except Exception as e:
        return None, f"load failed: {e}"


# ===========================================================================
# RealSense helpers
# ===========================================================================

def clamp_option(sensor, option, value):
    rng = sensor.get_option_range(option)
    value = float(np.clip(value, rng.min, rng.max))

    if rng.step > 0:
        value = rng.min + round((value - rng.min) / rng.step) * rng.step
        value = float(np.clip(value, rng.min, rng.max))

    return value


def find_exposure_sensor(device):
    candidates = []

    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except Exception:
            name = "Unknown Sensor"

        if sensor.supports(rs.option.exposure):
            candidates.append((sensor, name))

    if not candidates:
        raise RuntimeError("No RealSense sensor supports manual exposure.")

    for sensor, name in candidates:
        if "Stereo" in name:
            return sensor, name

    return candidates[0]


def deproject_mm(intr, px, py, z_m):
    pt = rs.rs2_deproject_pixel_to_point(
        intr,
        [float(px), float(py)],
        float(z_m),
    )
    return np.asarray(pt, dtype=np.float64) * 1000.0


def depth_min_at(depth_mm, px, py, win):
    h, w = depth_mm.shape
    x0 = max(0, px - win)
    x1 = min(w, px + win + 1)
    y0 = max(0, py - win)
    y1 = min(h, py + win + 1)

    vals = depth_mm[y0:y1, x0:x1]
    vals = vals[vals > 0]

    if vals.size == 0:
        return float("nan")

    return float(np.min(vals))


# ===========================================================================
# Current ring detector logic
# ===========================================================================

def kasa_circle_fit(xs, ys):
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    A = np.c_[2.0 * xs, 2.0 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol

    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None

    return float(cx), float(cy), float(np.sqrt(r2))


def ring_residual_ratio(xs, ys, cx, cy, r):
    d = np.hypot(xs - cx, ys - cy)
    resid = np.sqrt(np.mean((d - r) ** 2))
    return float(resid / max(r, 1e-6))


def ring_try_candidates(candidate_list, last_center):
    worst_reason = None

    for area, xs, ys, source in candidate_list:
        fit = kasa_circle_fit(xs, ys)

        if fit is None:
            worst_reason = "circle fit degenerate"
            continue

        cx, cy, r = fit

        if last_center is not None:
            jump = float(
                np.hypot(
                    cx - last_center[0],
                    cy - last_center[1],
                )
            )

            if jump > RING_STICKY_MAX_JUMP_PX:
                worst_reason = (
                    f"center jumped {jump:.1f}px > "
                    f"{RING_STICKY_MAX_JUMP_PX:.0f}px"
                )
                continue
        else:
            jump = float("nan")

        if r < RING_MIN_RADIUS_PX:
            worst_reason = (
                f"radius {r:.1f}px < {RING_MIN_RADIUS_PX:.1f}px"
            )
            continue

        residual = ring_residual_ratio(xs, ys, cx, cy, r)

        if residual > RING_MAX_RESIDUAL_RATIO:
            worst_reason = (
                f"residual {residual:.3f} > "
                f"{RING_MAX_RESIDUAL_RATIO:.2f}"
            )
            continue

        return {
            "ok": True,
            "cx": cx,
            "cy": cy,
            "radius_px": r,
            "n_px": int(len(xs)),
            "residual_ratio": residual,
            "jump_px": jump,
            "source": source,
            "fail_reason": None,
        }

    return {
        "ok": False,
        "cx": np.nan,
        "cy": np.nan,
        "radius_px": np.nan,
        "n_px": 0,
        "residual_ratio": np.nan,
        "jump_px": np.nan,
        "source": "",
        "fail_reason": worst_reason or "no candidate passed",
    }


def segment_ring_from_mask(mask, last_center, channel_label):
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (RING_CLOSE_KERNEL_PX, RING_CLOSE_KERNEL_PX),
            np.uint8,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if n_labels <= 1:
        return {
            "ok": False,
            "fail_reason": f"no {channel_label} pixels in ROI",
        }, mask

    candidates = []
    best_area = 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        best_area = max(best_area, area)

        if (
            area < RING_MIN_AREA_PX
            or area > RING_MAX_AREA_PX
        ):
            continue

        ys, xs = np.nonzero(labels == i)
        candidates.append(
            (float(area), xs, ys, f"{channel_label}:single")
        )

    candidates.sort(key=lambda x: -x[0])

    result = ring_try_candidates(
        candidates,
        last_center,
    )

    if result["ok"]:
        return result, mask

    # Current production fallback: merge nearby non-trivial fragments.
    small_pieces = [
        i
        for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= RING_MERGE_MIN_PIECE_PX
    ]

    merged_candidates = []

    for ai in range(len(small_pieces)):
        for bi in range(ai + 1, len(small_pieces)):
            i = small_pieces[ai]
            j = small_pieces[bi]

            dist = float(
                np.hypot(
                    centroids[i][0] - centroids[j][0],
                    centroids[i][1] - centroids[j][1],
                )
            )

            if dist > RING_MERGE_MAX_DIST_PX:
                continue

            merged_area = (
                stats[i, cv2.CC_STAT_AREA]
                + stats[j, cv2.CC_STAT_AREA]
            )

            if (
                merged_area < RING_MIN_AREA_PX
                or merged_area > RING_MAX_AREA_PX
            ):
                continue

            ys_i, xs_i = np.nonzero(labels == i)
            ys_j, xs_j = np.nonzero(labels == j)

            xs = np.concatenate([xs_i, xs_j])
            ys = np.concatenate([ys_i, ys_j])

            merged_candidates.append(
                (
                    float(merged_area),
                    xs,
                    ys,
                    f"{channel_label}:merged",
                )
            )

    merged_candidates.sort(key=lambda x: -x[0])

    merged_result = ring_try_candidates(
        merged_candidates,
        last_center,
    )

    if merged_result["ok"]:
        return merged_result, mask

    return {
        "ok": False,
        "cx": np.nan,
        "cy": np.nan,
        "radius_px": np.nan,
        "n_px": 0,
        "residual_ratio": np.nan,
        "jump_px": np.nan,
        "source": channel_label,
        "fail_reason": (
            merged_result.get("fail_reason")
            or result.get("fail_reason")
            or (
                f"{channel_label}: largest component={best_area}px; "
                "no valid component/merge"
            )
        ),
    }, mask


def detect_ring(color_bgr, last_center):
    lab = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2LAB,
    )

    a_full = lab[:, :, 1]
    h, w = a_full.shape

    x0 = max(0, RING_ROI_X0)
    x1 = min(w, RING_ROI_X1)
    y0 = max(0, RING_ROI_Y0)
    y1 = min(h, RING_ROI_Y1)

    a_mask = np.zeros((h, w), dtype=np.uint8)
    a_mask[y0:y1, x0:x1] = (
        a_full[y0:y1, x0:x1] > RING_A_STAR_THRESH
    ).astype(np.uint8) * 255

    result_a, mask_a = segment_ring_from_mask(
        a_mask,
        last_center,
        f"a*>{RING_A_STAR_THRESH}",
    )

    if result_a["ok"]:
        return result_a, mask_a

    if not RING_USE_SAT_FALLBACK:
        return result_a, mask_a

    # Production safety gate: Saturation may maintain/reacquire an existing
    # sticky lock, but cannot originate the first-ever lock.
    if last_center is None:
        result_a["fail_reason"] = (
            (result_a.get("fail_reason") or "a* failed")
            + " | saturation fallback skipped: no sticky center yet"
        )
        return result_a, mask_a

    s_full = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2HSV,
    )[:, :, 1]

    s_roi = s_full[y0:y1, x0:x1]

    otsu_th, _ = cv2.threshold(
        s_roi,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    otsu_th = float(
        np.clip(
            otsu_th,
            RING_SAT_OTSU_MIN,
            RING_SAT_OTSU_MAX,
        )
    )

    s_mask = np.zeros((h, w), dtype=np.uint8)
    s_mask[y0:y1, x0:x1] = (
        s_roi > otsu_th
    ).astype(np.uint8) * 255

    result_s, mask_s = segment_ring_from_mask(
        s_mask,
        last_center,
        f"Sat>{otsu_th:.0f}(Otsu)",
    )

    if result_s["ok"]:
        return result_s, mask_s

    result_a["fail_reason"] = (
        (result_a.get("fail_reason") or "a* failed")
        + " | "
        + (result_s.get("fail_reason") or "Sat failed")
    )

    return result_a, mask_a


# ===========================================================================
# Current phantom-normal estimator, kept intentionally close to production
# ===========================================================================

def estimate_phantom_normal_from_frame(
    color_bgr,
    depth_frame,
    intr,
    R_he,
    t_he,
    rcm_base,
):
    if R_he is None:
        return {
            "ok": False,
            "fail_reason": "hand-eye missing",
        }

    rcm_cam = R_he.T.dot(
        rcm_base - t_he
    )

    if rcm_cam[2] <= 1e-3:
        return {
            "ok": False,
            "fail_reason": "RCM projects behind camera",
        }

    u0 = int(
        round(
            intr.fx * rcm_cam[0] / rcm_cam[2]
            + intr.ppx
        )
    )

    v0 = int(
        round(
            intr.fy * rcm_cam[1] / rcm_cam[2]
            + intr.ppy
        )
    )

    lab = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2LAB,
    )

    a_star = lab[:, :, 1]
    h = depth_frame.get_height()
    w = depth_frame.get_width()

    pts_cam = []
    n_red_dropped = 0

    for dv in range(
        -NORMAL_PATCH_RADIUS_PX,
        NORMAL_PATCH_RADIUS_PX + 1,
    ):
        for du in range(
            -NORMAL_PATCH_RADIUS_PX,
            NORMAL_PATCH_RADIUS_PX + 1,
        ):
            if (
                du * du + dv * dv
                > NORMAL_PATCH_RADIUS_PX
                * NORMAL_PATCH_RADIUS_PX
            ):
                continue

            uu = u0 + du
            vv = v0 + dv

            if (
                uu < 0
                or vv < 0
                or uu >= w
                or vv >= h
            ):
                continue

            if a_star[vv, uu] > NORMAL_A_STAR_THRESH:
                n_red_dropped += 1
                continue

            z = depth_frame.get_distance(
                uu,
                vv,
            ) * 1000.0

            if z <= 0:
                continue

            x = (
                (uu - intr.ppx)
                / intr.fx
                * z
            )

            y = (
                (vv - intr.ppy)
                / intr.fy
                * z
            )

            pts_cam.append(
                (x, y, z)
            )

    pts_cam = np.asarray(
        pts_cam,
        dtype=float,
    )

    if len(pts_cam) < NORMAL_MIN_POINTS:
        return {
            "ok": False,
            "n_points": int(len(pts_cam)),
            "n_red_dropped": int(n_red_dropped),
            "u0": u0,
            "v0": v0,
            "fail_reason": (
                f"only {len(pts_cam)} non-red depth points "
                f"(<{NORMAL_MIN_POINTS})"
            ),
        }

    centroid = pts_cam.mean(axis=0)
    X = pts_cam - centroid

    _, _, Vt = np.linalg.svd(
        X,
        full_matrices=False,
    )

    normal_cam = Vt[-1]
    normal_cam /= np.linalg.norm(
        normal_cam
    )

    resid = X.dot(
        normal_cam
    )

    plane_rms_mm = float(
        np.sqrt(
            np.mean(
                resid ** 2
            )
        )
    )

    # Match production sign convention: point toward camera.
    view_dir = centroid / np.linalg.norm(
        centroid
    )

    if np.dot(
        normal_cam,
        view_dir,
    ) > 0:
        normal_cam = -normal_cam

    centroid_base = (
        R_he.dot(centroid)
        + t_he
    )

    centroid_disagree_mm = float(
        np.linalg.norm(
            centroid_base - rcm_base
        )
    )

    if (
        centroid_disagree_mm
        > NORMAL_MAX_CENTROID_DISAGREE_MM
    ):
        return {
            "ok": False,
            "n_points": int(len(pts_cam)),
            "n_red_dropped": int(n_red_dropped),
            "u0": u0,
            "v0": v0,
            "plane_rms_mm": plane_rms_mm,
            "centroid_disagree_mm": centroid_disagree_mm,
            "fail_reason": (
                f"sampled bowl centroid disagrees with RCM by "
                f"{centroid_disagree_mm:.2f} mm "
                f"(limit {NORMAL_MAX_CENTROID_DISAGREE_MM:.1f} mm)"
            ),
        }

    normal_base = R_he.dot(
        normal_cam
    )

    normal_base /= np.linalg.norm(
        normal_base
    )

    advisory = None

    if plane_rms_mm > NORMAL_MAX_PLANE_RMS_MM:
        advisory = (
            f"plane RMS {plane_rms_mm:.2f} mm > "
            f"{NORMAL_MAX_PLANE_RMS_MM:.1f} mm"
        )

    return {
        "ok": True,
        "normal_base": normal_base,
        "normal_cam": normal_cam,
        "point_base": rcm_base.copy(),
        "n_points": int(len(pts_cam)),
        "n_red_dropped": int(n_red_dropped),
        "u0": u0,
        "v0": v0,
        "plane_rms_mm": plane_rms_mm,
        "centroid_disagree_mm": centroid_disagree_mm,
        "fail_reason": advisory,
    }


# ===========================================================================
# Diagnostics
# ===========================================================================

def angle_deg(a, b, sign_invariant=False):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = float(np.dot(a, b))

    if sign_invariant:
        dot = abs(dot)

    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    dot,
                    -1.0,
                    1.0,
                )
            )
        )
    )


def align_normal_sign(normal, reference):
    normal = np.asarray(
        normal,
        dtype=float,
    ).copy()

    reference = np.asarray(
        reference,
        dtype=float,
    )

    if np.dot(
        normal,
        reference,
    ) < 0:
        normal = -normal

    return normal


def normalized_mean(vectors, reference=None):
    if not vectors:
        return None

    arr = np.asarray(
        vectors,
        dtype=float,
    ).copy()

    if reference is None:
        reference = arr[0]

    for i in range(len(arr)):
        if np.dot(
            arr[i],
            reference,
        ) < 0:
            arr[i] *= -1.0

    mean = arr.mean(axis=0)
    n = np.linalg.norm(mean)

    if n <= 1e-12:
        return None

    return mean / n


def scalar_stats(values):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def eye_exposure_stats(color_bgr):
    x0, x1, y0, y1 = EYE_ROI
    roi = color_bgr[
        y0:y1,
        x0:x1,
    ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    return {
        "sat_pct": float(
            np.mean(
                gray >= SATURATION_GRAY_THRESHOLD
            ) * 100.0
        ),
        "mean_gray": float(
            np.mean(gray)
        ),
        "p99_gray": float(
            np.percentile(gray, 99)
        ),
    }


def save_snapshot(
    raw_dir,
    ann_dir,
    mask_dir,
    index,
    color,
    annotated,
    mask,
):
    cv2.imwrite(
        str(
            raw_dir
            / f"frame_{index:05d}.png"
        ),
        color,
    )

    cv2.imwrite(
        str(
            ann_dir
            / f"frame_{index:05d}.png"
        ),
        annotated,
    )

    cv2.imwrite(
        str(
            mask_dir
            / f"frame_{index:05d}.png"
        ),
        mask,
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Camera-only phantom-normal verification. "
            "No ROS and no robot motion."
        )
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=DEFAULT_EXPOSURE,
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_GAIN,
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="0 = run until q/ESC or sample target is reached",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_TARGET,
        help="Stop after this many successful normal estimates; 0 disables",
    )

    parser.add_argument(
        "--normal-every",
        type=int,
        default=DEFAULT_NORMAL_EVERY,
        help="Estimate normal every N frames after RCM lock",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--project-root",
        default=str(
            DEFAULT_PROJECT_ROOT
        ),
    )

    parser.add_argument(
        "--handeye",
        default=None,
        help="Optional explicit handeye_T.npz path",
    )

    parser.add_argument(
        "--baseline-normal",
        default=None,
        help="Optional explicit phantom_normal.npz path",
    )

    parser.add_argument(
        "--intr-util",
        default=None,
        help="Optional explicit intr_util.py path",
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    ).expanduser()

    handeye_path, handeye_matches, handeye_reason = discover_project_file(
        "handeye_T.npz",
        explicit_path=args.handeye,
        project_root=project_root,
    )

    normal_path, normal_matches, normal_reason = discover_project_file(
        "phantom_normal.npz",
        explicit_path=args.baseline_normal,
        project_root=project_root,
    )

    print_discovery(
        "handeye_T.npz",
        handeye_path,
        handeye_matches,
        handeye_reason,
    )

    print_discovery(
        "phantom_normal.npz",
        normal_path,
        normal_matches,
        normal_reason,
    )

    R_he, t_he, handeye_status = load_handeye(
        handeye_path
    )

    baseline, baseline_status = load_baseline_normal(
        normal_path
    )

    if R_he is None:
        raise RuntimeError(
            "Cannot run the normal verification without handeye_T.npz."
        )

    if baseline is None:
        print(
            "WARNING: baseline phantom_normal.npz could not be loaded. "
            "The test can still measure repeatability, but baseline comparison "
            "will be unavailable."
        )

    script_dir = Path(
        __file__
    ).resolve().parent

    run_dir = (
        script_dir
        / "normal_verification_runs"
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    raw_dir = run_dir / "raw_samples"
    ann_dir = run_dir / "annotated_samples"
    mask_dir = run_dir / "ring_masks"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    ann_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print("D405 NORMAL-ONLY VERIFICATION")
    print("=" * 78)
    print("NO ROS / NO ROBOT COMMANDS / NO production NPZ writes")
    print(f"Output only: {run_dir}")
    print(
        f"Requested camera: exposure={args.exposure:.0f}, "
        f"gain={args.gain:.0f}"
    )
    print(
        f"Production ring threshold: residual <= "
        f"{RING_MAX_RESIDUAL_RATIO:.2f}"
    )
    print(
        f"Normal patch: radius={NORMAL_PATCH_RADIUS_PX}px, "
        f"min_points={NORMAL_MIN_POINTS}, "
        f"centroid gate={NORMAL_MAX_CENTROID_DISAGREE_MM:.1f}mm"
    )
    print(f"Hand-eye: {handeye_status}")
    print(f"Baseline normal: {baseline_status}")
    print()

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        args.fps,
    )

    cfg.enable_stream(
        rs.stream.depth,
        640,
        480,
        rs.format.z16,
        args.fps,
    )

    profile = pipe.start(
        cfg
    )

    align = rs.align(
        rs.stream.color
    )

    device = profile.get_device()

    depth_scale = (
        device
        .first_depth_sensor()
        .get_depth_scale()
    )

    intr_util_path = prepare_intrinsics_import(
        project_root,
        explicit_intr_util=args.intr_util,
    )

    try:
        from intr_util import load_best_intrinsics

        intr = load_best_intrinsics(
            profile
        )

        intr_source = (
            f"load_best_intrinsics(profile) from {intr_util_path}"
        )
    except Exception as e:
        intr = (
            profile
            .get_stream(
                rs.stream.color
            )
            .as_video_stream_profile()
            .get_intrinsics()
        )

        intr_source = (
            "factory/profile intrinsics fallback "
            f"({e})"
        )

    sensor, sensor_name = find_exposure_sensor(
        device
    )

    if sensor.supports(
        rs.option.enable_auto_exposure
    ):
        sensor.set_option(
            rs.option.enable_auto_exposure,
            0,
        )

    actual_exposure = clamp_option(
        sensor,
        rs.option.exposure,
        args.exposure,
    )

    sensor.set_option(
        rs.option.exposure,
        actual_exposure,
    )

    actual_gain = float(
        "nan"
    )

    if sensor.supports(
        rs.option.gain
    ):
        actual_gain = clamp_option(
            sensor,
            rs.option.gain,
            args.gain,
        )

        sensor.set_option(
            rs.option.gain,
            actual_gain,
        )

        actual_gain = float(
            sensor.get_option(
                rs.option.gain
            )
        )

    actual_exposure = float(
        sensor.get_option(
            rs.option.exposure
        )
    )

    print(
        f"Exposure sensor: {sensor_name}; "
        f"actual exposure={actual_exposure:.0f}; "
        f"actual gain={actual_gain:.0f}"
    )

    print(
        f"Intrinsics source: {intr_source}"
    )

    if baseline is not None:
        print(
            "Baseline n = "
            f"{np.round(baseline['normal_base'], 6)}"
        )

        print(
            "Baseline point = "
            f"{np.round(baseline['point_base'], 3)} mm"
        )

        if baseline["plane_rms_mm"] is not None:
            print(
                "Baseline stored plane RMS = "
                f"{baseline['plane_rms_mm']:.3f} mm"
            )

    # Warm up after manual exposure/gain are applied.
    for _ in range(30):
        pipe.wait_for_frames()

    ring_history = deque(
        maxlen=RING_SMOOTH_N
    )

    last_center = None
    frozen_rcm = None
    frozen_rcm_jitter = None
    frozen_rcm_frame = None

    normal_records = []
    frame_rows = []

    total_frames = 0
    ring_accept_count = 0
    depth_valid_count = 0
    normal_attempt_count = 0
    normal_success_count = 0
    sat_fallback_count = 0

    start = time.time()
    manual_snapshot_index = 0
    last_normal_result = None

    cv2.namedWindow(
        "D405 NORMAL ONLY - NO ROBOT MOTION",
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        "D405 NORMAL ONLY - NO ROBOT MOTION",
        1100,
        820,
    )

    try:
        while True:
            elapsed = (
                time.time()
                - start
            )

            if (
                args.seconds > 0
                and elapsed >= args.seconds
            ):
                break

            if (
                args.samples > 0
                and normal_success_count >= args.samples
            ):
                break

            frames = align.process(
                pipe.wait_for_frames()
            )

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if (
                not color_frame
                or not depth_frame
            ):
                continue

            color = np.asanyarray(
                color_frame.get_data()
            )

            depth_mm = (
                np.asanyarray(
                    depth_frame.get_data()
                ).astype(np.float64)
                * depth_scale
                * 1000.0
            )

            total_frames += 1

            ring, ring_mask = detect_ring(
                color,
                last_center,
            )

            ring_jitter_mm = float(
                "nan"
            )

            ring_smoothed = None

            if ring["ok"]:
                ring_accept_count += 1

                last_center = (
                    ring["cx"],
                    ring["cy"],
                )

                if "Sat>" in ring.get(
                    "source",
                    "",
                ):
                    sat_fallback_count += 1

                z_mm = depth_min_at(
                    depth_mm,
                    int(
                        round(
                            ring["cx"]
                        )
                    ),
                    int(
                        round(
                            ring["cy"]
                        )
                    ),
                    RING_DEPTH_WIN_PX,
                )

                if (
                    np.isfinite(z_mm)
                    and z_mm > 0
                ):
                    depth_valid_count += 1

                    cam_mm = deproject_mm(
                        intr,
                        ring["cx"],
                        ring["cy"],
                        z_mm / 1000.0,
                    )

                    base_pt = (
                        R_he.dot(cam_mm)
                        + t_he
                    )

                    ring_history.append(
                        base_pt
                    )

            if len(ring_history) >= 3:
                ring_arr = np.asarray(
                    ring_history,
                    dtype=float,
                )

                ring_smoothed = (
                    ring_arr.mean(
                        axis=0
                    )
                )

                ring_jitter_mm = float(
                    ring_arr.std(
                        axis=0
                    ).max()
                )

            # Freeze the FIRST production-like stable RCM.
            if (
                frozen_rcm is None
                and len(ring_history)
                >= RING_SMOOTH_N
                and np.isfinite(
                    ring_jitter_mm
                )
                and ring_jitter_mm
                < RING_MAX_JITTER_MM
            ):
                frozen_rcm = (
                    ring_smoothed.copy()
                )

                frozen_rcm_jitter = (
                    ring_jitter_mm
                )

                frozen_rcm_frame = (
                    total_frames
                )

                print()
                print(
                    "RCM FROZEN for this verification: "
                    f"{np.round(frozen_rcm, 4)} mm"
                )

                print(
                    f"  lock jitter={frozen_rcm_jitter:.4f} mm "
                    f"at frame {frozen_rcm_frame}"
                )

                if baseline is not None:
                    point_shift = float(
                        np.linalg.norm(
                            frozen_rcm
                            - baseline["point_base"]
                        )
                    )

                    print(
                        "  distance from baseline phantom_normal point = "
                        f"{point_shift:.3f} mm"
                    )

            # Normal-only repeated samples: always use the same frozen RCM.
            if (
                frozen_rcm is not None
                and args.normal_every > 0
                and total_frames
                % args.normal_every
                == 0
            ):
                normal_attempt_count += 1

                res = estimate_phantom_normal_from_frame(
                    color,
                    depth_frame,
                    intr,
                    R_he,
                    t_he,
                    frozen_rcm,
                )

                last_normal_result = res

                if res["ok"]:
                    normal_success_count += 1

                    current_normal = (
                        res["normal_base"].copy()
                    )

                    if baseline is not None:
                        angle_to_baseline = angle_deg(
                            current_normal,
                            baseline["normal_base"],
                            sign_invariant=False,
                        )

                        angle_to_baseline_axis = angle_deg(
                            current_normal,
                            baseline["normal_base"],
                            sign_invariant=True,
                        )
                    else:
                        angle_to_baseline = float(
                            "nan"
                        )

                        angle_to_baseline_axis = float(
                            "nan"
                        )

                    # Compare to current aggregate mean using a consistent sign.
                    reference = (
                        baseline["normal_base"]
                        if baseline is not None
                        else (
                            normal_records[0]["normal_base"]
                            if normal_records
                            else current_normal
                        )
                    )

                    all_vectors = [
                        r["normal_base"]
                        for r in normal_records
                    ] + [current_normal]

                    mean_normal = normalized_mean(
                        all_vectors,
                        reference=reference,
                    )

                    angle_to_mean = (
                        angle_deg(
                            current_normal,
                            mean_normal,
                            sign_invariant=False,
                        )
                        if mean_normal is not None
                        else float("nan")
                    )

                    record = {
                        "sample_index": normal_success_count,
                        "frame": total_frames,
                        "elapsed_s": elapsed,
                        "normal_base": current_normal,
                        "plane_rms_mm": res["plane_rms_mm"],
                        "centroid_disagree_mm": res["centroid_disagree_mm"],
                        "n_points": res["n_points"],
                        "n_red_dropped": res["n_red_dropped"],
                        "angle_to_baseline_deg": angle_to_baseline,
                        "axis_angle_to_baseline_deg": angle_to_baseline_axis,
                        "angle_to_running_mean_deg": angle_to_mean,
                        "advisory": res.get("fail_reason"),
                    }

                    normal_records.append(
                        record
                    )

                    print(
                        f"NORMAL {normal_success_count:02d}: "
                        f"n={np.round(current_normal, 5)} "
                        f"RMS={res['plane_rms_mm']:.3f}mm "
                        f"centroidDiff={res['centroid_disagree_mm']:.3f}mm "
                        f"baselineAngle="
                        + (
                            f"{angle_to_baseline:.2f}deg"
                            if np.isfinite(
                                angle_to_baseline
                            )
                            else "NA"
                        )
                        + f" runningMeanDiff={angle_to_mean:.2f}deg"
                    )
                else:
                    print(
                        "NORMAL attempt failed: "
                        f"{res.get('fail_reason', '?')}"
                    )

            exposure_stats = eye_exposure_stats(
                color
            )

            # Current aggregate normal statistics.
            mean_normal = normalized_mean(
                [
                    r["normal_base"]
                    for r in normal_records
                ],
                reference=(
                    baseline["normal_base"]
                    if baseline is not None
                    else None
                ),
            )

            if (
                mean_normal is not None
                and normal_records
            ):
                spread_angles = [
                    angle_deg(
                        r["normal_base"],
                        mean_normal,
                    )
                    for r in normal_records
                ]

                mean_spread = float(
                    np.mean(
                        spread_angles
                    )
                )

                max_spread = float(
                    np.max(
                        spread_angles
                    )
                )

                if baseline is not None:
                    mean_to_baseline = angle_deg(
                        mean_normal,
                        baseline["normal_base"],
                    )
                else:
                    mean_to_baseline = float(
                        "nan"
                    )
            else:
                mean_spread = float(
                    "nan"
                )

                max_spread = float(
                    "nan"
                )

                mean_to_baseline = float(
                    "nan"
                )

            frame_rows.append({
                "frame": total_frames,
                "elapsed_s": elapsed,
                "ring_ok": int(
                    ring["ok"]
                ),
                "ring_source": ring.get(
                    "source",
                    "",
                ),
                "ring_cx_px": (
                    ring["cx"]
                    if ring["ok"]
                    else ""
                ),
                "ring_cy_px": (
                    ring["cy"]
                    if ring["ok"]
                    else ""
                ),
                "ring_radius_px": (
                    ring["radius_px"]
                    if ring["ok"]
                    else ""
                ),
                "ring_residual": (
                    ring["residual_ratio"]
                    if ring["ok"]
                    else ""
                ),
                "ring_jitter_mm": (
                    ring_jitter_mm
                    if np.isfinite(
                        ring_jitter_mm
                    )
                    else ""
                ),
                "rcm_frozen": int(
                    frozen_rcm is not None
                ),
                "eye_saturation_pct": exposure_stats["sat_pct"],
                "eye_mean_gray": exposure_stats["mean_gray"],
                "eye_p99_gray": exposure_stats["p99_gray"],
                "normal_success_count": normal_success_count,
                "normal_mean_to_baseline_deg": (
                    mean_to_baseline
                    if np.isfinite(
                        mean_to_baseline
                    )
                    else ""
                ),
                "normal_mean_spread_deg": (
                    mean_spread
                    if np.isfinite(
                        mean_spread
                    )
                    else ""
                ),
                "normal_max_spread_deg": (
                    max_spread
                    if np.isfinite(
                        max_spread
                    )
                    else ""
                ),
            })

            # ---------------------------------------------------------------
            # Live display
            # ---------------------------------------------------------------
            view = color.copy()

            cv2.rectangle(
                view,
                (
                    RING_ROI_X0,
                    RING_ROI_Y0,
                ),
                (
                    RING_ROI_X1,
                    RING_ROI_Y1,
                ),
                (110, 110, 110),
                1,
            )

            ex0, ex1, ey0, ey1 = EYE_ROI

            cv2.rectangle(
                view,
                (ex0, ey0),
                (ex1, ey1),
                (255, 255, 255),
                1,
            )

            if ring["ok"]:
                cv2.circle(
                    view,
                    (
                        int(
                            round(
                                ring["cx"]
                            )
                        ),
                        int(
                            round(
                                ring["cy"]
                            )
                        ),
                    ),
                    int(
                        round(
                            ring["radius_px"]
                        )
                    ),
                    (0, 255, 80),
                    2,
                )

                cv2.circle(
                    view,
                    (
                        int(
                            round(
                                ring["cx"]
                            )
                        ),
                        int(
                            round(
                                ring["cy"]
                            )
                        ),
                    ),
                    4,
                    (0, 255, 255),
                    -1,
                )

            if (
                last_normal_result is not None
                and "u0"
                in last_normal_result
            ):
                patch_color = (
                    (255, 255, 0)
                    if last_normal_result.get(
                        "ok"
                    )
                    else (0, 0, 255)
                )

                cv2.circle(
                    view,
                    (
                        int(
                            last_normal_result["u0"]
                        ),
                        int(
                            last_normal_result["v0"]
                        ),
                    ),
                    NORMAL_PATCH_RADIUS_PX,
                    patch_color,
                    2,
                )

            panel_h = 194

            cv2.rectangle(
                view,
                (0, 0),
                (640, panel_h),
                (0, 0, 0),
                -1,
            )

            ring_rate = (
                100.0
                * ring_accept_count
                / total_frames
            )

            line1 = (
                f"SAFE NORMAL-ONLY | EXP={actual_exposure:.0f} "
                f"GAIN={actual_gain:.0f} | no ROS / no motion"
            )

            line2 = (
                f"RING prod<=0.32: "
                f"{int(ring['ok'])} "
                f"({ring_rate:.1f}%) "
                + (
                    f"resid={ring['residual_ratio']:.3f} "
                    f"r={ring['radius_px']:.1f}px"
                    if ring["ok"]
                    else (
                        f"reject="
                        f"{(ring.get('fail_reason') or '?')[:45]}"
                    )
                )
            )

            if frozen_rcm is None:
                line3 = (
                    f"RCM collecting "
                    f"{len(ring_history)}/{RING_SMOOTH_N}; "
                    + (
                        f"jitter={ring_jitter_mm:.3f}mm"
                        if np.isfinite(
                            ring_jitter_mm
                        )
                        else "jitter=NA"
                    )
                )
            else:
                line3 = (
                    f"RCM FROZEN frame={frozen_rcm_frame} "
                    f"jitter={frozen_rcm_jitter:.3f}mm "
                    f"[{frozen_rcm[0]:.2f}, "
                    f"{frozen_rcm[1]:.2f}, "
                    f"{frozen_rcm[2]:.2f}]"
                )

            line4 = (
                f"NORMAL success={normal_success_count}/"
                f"{args.samples if args.samples > 0 else 'inf'} "
                f"attempts={normal_attempt_count}"
            )

            if mean_normal is not None:
                line5 = (
                    f"NEW mean n="
                    f"[{mean_normal[0]:+.3f} "
                    f"{mean_normal[1]:+.3f} "
                    f"{mean_normal[2]:+.3f}] "
                    f"internal mean/max="
                    f"{mean_spread:.2f}/{max_spread:.2f}deg"
                )
            else:
                line5 = (
                    "NEW mean n=waiting..."
                )

            if (
                baseline is not None
                and np.isfinite(
                    mean_to_baseline
                )
            ):
                line6 = (
                    f"mean vs STORED baseline="
                    f"{mean_to_baseline:.2f}deg "
                    f"| eye sat={exposure_stats['sat_pct']:.2f}%"
                )
            else:
                line6 = (
                    f"baseline compare=NA "
                    f"| eye sat={exposure_stats['sat_pct']:.2f}%"
                )

            lines = [
                line1,
                line2,
                line3,
                line4,
                line5,
                line6,
            ]

            y = 23

            for line in lines:
                cv2.putText(
                    view,
                    line,
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                y += 29

            cv2.imshow(
                "D405 NORMAL ONLY - NO ROBOT MOTION",
                view,
            )

            if (
                SAVE_EVERY_N > 0
                and total_frames
                % SAVE_EVERY_N
                == 0
            ):
                save_snapshot(
                    raw_dir,
                    ann_dir,
                    mask_dir,
                    total_frames,
                    color,
                    view,
                    ring_mask,
                )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

            if key == ord("s"):
                manual_snapshot_index += 1

                save_snapshot(
                    raw_dir,
                    ann_dir,
                    mask_dir,
                    900000
                    + manual_snapshot_index,
                    color,
                    view,
                    ring_mask,
                )

                print(
                    "Saved manual snapshot."
                )

            if key == ord("r"):
                ring_history.clear()
                last_center = None
                frozen_rcm = None
                frozen_rcm_jitter = None
                frozen_rcm_frame = None
                normal_records.clear()
                last_normal_result = None
                normal_attempt_count = 0
                normal_success_count = 0

                print()
                print(
                    "Temporary RCM + normal history reset IN MEMORY ONLY."
                )

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    mean_normal = normalized_mean(
        [
            r["normal_base"]
            for r in normal_records
        ],
        reference=(
            baseline["normal_base"]
            if baseline is not None
            else None
        ),
    )

    internal_angles = []
    baseline_angles = []
    baseline_axis_angles = []

    if mean_normal is not None:
        internal_angles = [
            angle_deg(
                r["normal_base"],
                mean_normal,
            )
            for r in normal_records
        ]

    if baseline is not None:
        baseline_angles = [
            r["angle_to_baseline_deg"]
            for r in normal_records
            if np.isfinite(
                r["angle_to_baseline_deg"]
            )
        ]

        baseline_axis_angles = [
            r["axis_angle_to_baseline_deg"]
            for r in normal_records
            if np.isfinite(
                r["axis_angle_to_baseline_deg"]
            )
        ]

    plane_rms_values = [
        r["plane_rms_mm"]
        for r in normal_records
    ]

    centroid_values = [
        r["centroid_disagree_mm"]
        for r in normal_records
    ]

    point_shift_mm = None

    if (
        frozen_rcm is not None
        and baseline is not None
    ):
        point_shift_mm = float(
            np.linalg.norm(
                frozen_rcm
                - baseline["point_base"]
            )
        )

    mean_to_baseline = None
    mean_to_baseline_axis = None

    if (
        mean_normal is not None
        and baseline is not None
    ):
        mean_to_baseline = angle_deg(
            mean_normal,
            baseline["normal_base"],
        )

        mean_to_baseline_axis = angle_deg(
            mean_normal,
            baseline["normal_base"],
            sign_invariant=True,
        )

    summary = {
        "safe_scope": {
            "ros_imported": False,
            "robot_motion_commanded": False,
            "production_auto_target_written": False,
            "production_phantom_normal_written": False,
            "calibration_files_modified": False,
        },

        "camera": {
            "sensor_name": sensor_name,
            "exposure_requested": args.exposure,
            "exposure_actual": actual_exposure,
            "gain_requested": args.gain,
            "gain_actual": actual_gain,
            "intrinsics_source": intr_source,
        },

        "files_read_only": {
            "handeye": handeye_status,
            "baseline_normal": baseline_status,
        },

        "production_parameters_reproduced": {
            "ring_residual_threshold": RING_MAX_RESIDUAL_RATIO,
            "ring_close_kernel_px": RING_CLOSE_KERNEL_PX,
            "ring_smooth_n": RING_SMOOTH_N,
            "ring_max_jitter_mm": RING_MAX_JITTER_MM,
            "normal_patch_radius_px": NORMAL_PATCH_RADIUS_PX,
            "normal_a_star_threshold": NORMAL_A_STAR_THRESH,
            "normal_min_points": NORMAL_MIN_POINTS,
            "normal_max_centroid_disagree_mm": NORMAL_MAX_CENTROID_DISAGREE_MM,
            "normal_max_plane_rms_mm": NORMAL_MAX_PLANE_RMS_MM,
        },

        "rcm_lock": {
            "frozen": frozen_rcm is not None,
            "frozen_frame": frozen_rcm_frame,
            "frozen_rcm_base_mm": (
                frozen_rcm.tolist()
                if frozen_rcm is not None
                else None
            ),
            "jitter_mm_at_freeze": frozen_rcm_jitter,
            "distance_to_baseline_normal_point_mm": point_shift_mm,
        },

        "normal_sampling": {
            "attempts": normal_attempt_count,
            "successes": normal_success_count,
            "success_rate_pct": (
                100.0
                * normal_success_count
                / normal_attempt_count
                if normal_attempt_count
                else 0.0
            ),
            "mean_normal_base": (
                mean_normal.tolist()
                if mean_normal is not None
                else None
            ),
            "baseline_normal_base": (
                baseline["normal_base"].tolist()
                if baseline is not None
                else None
            ),
            "mean_to_baseline_deg": mean_to_baseline,
            "mean_to_baseline_sign_invariant_deg": mean_to_baseline_axis,
            "individual_to_baseline_deg": scalar_stats(
                baseline_angles
            ),
            "individual_to_baseline_sign_invariant_deg": scalar_stats(
                baseline_axis_angles
            ),
            "internal_angle_to_new_mean_deg": scalar_stats(
                internal_angles
            ),
            "plane_rms_mm": scalar_stats(
                plane_rms_values
            ),
            "centroid_disagree_mm": scalar_stats(
                centroid_values
            ),
        },

        "ring_tracking": {
            "frames": total_frames,
            "ring_accept_rate_pct": (
                100.0
                * ring_accept_count
                / total_frames
                if total_frames
                else 0.0
            ),
            "depth_valid_given_ring_accept_pct": (
                100.0
                * depth_valid_count
                / ring_accept_count
                if ring_accept_count
                else 0.0
            ),
            "sat_fallback_count": sat_fallback_count,
        },
    }

    summary_path = (
        run_dir
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    frame_csv = (
        run_dir
        / "frame_metrics.csv"
    )

    if frame_rows:
        with open(
            frame_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    frame_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                frame_rows
            )

    normals_csv = (
        run_dir
        / "normal_samples.csv"
    )

    if normal_records:
        flattened = []

        for r in normal_records:
            n = r["normal_base"]

            flattened.append({
                "sample_index": r["sample_index"],
                "frame": r["frame"],
                "elapsed_s": r["elapsed_s"],
                "normal_x": n[0],
                "normal_y": n[1],
                "normal_z": n[2],
                "plane_rms_mm": r["plane_rms_mm"],
                "centroid_disagree_mm": r["centroid_disagree_mm"],
                "n_points": r["n_points"],
                "n_red_dropped": r["n_red_dropped"],
                "angle_to_baseline_deg": r["angle_to_baseline_deg"],
                "axis_angle_to_baseline_deg": r["axis_angle_to_baseline_deg"],
                "angle_to_running_mean_deg": r["angle_to_running_mean_deg"],
                "advisory": r["advisory"] or "",
            })

        with open(
            normals_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    flattened[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                flattened
            )

    print()
    print("=" * 78)
    print("NORMAL-ONLY RESULT")
    print("=" * 78)

    print(
        f"Frames: {total_frames}"
    )

    print(
        f"Ring accepted: "
        f"{summary['ring_tracking']['ring_accept_rate_pct']:.1f}%"
    )

    print(
        f"RCM frozen: "
        f"{summary['rcm_lock']['frozen']}"
    )

    if frozen_rcm is not None:
        print(
            "Frozen RCM: "
            f"{np.round(frozen_rcm, 4)} mm"
        )

        print(
            "RCM jitter at freeze: "
            f"{frozen_rcm_jitter:.4f} mm"
        )

    if point_shift_mm is not None:
        print(
            "Distance from stored normal's point_base: "
            f"{point_shift_mm:.3f} mm"
        )

    print(
        f"Normal attempts/successes: "
        f"{normal_attempt_count}/{normal_success_count}"
    )

    if mean_normal is not None:
        print(
            "New mean normal: "
            f"{np.round(mean_normal, 6)}"
        )

        internal = scalar_stats(
            internal_angles
        )

        print(
            "Internal angular spread around new mean: "
            f"mean={internal['mean']:.3f}deg, "
            f"std={internal['std']:.3f}deg, "
            f"p95={internal['p95']:.3f}deg, "
            f"max={internal['max']:.3f}deg"
        )

    if (
        mean_to_baseline is not None
    ):
        print(
            "New mean vs stored baseline normal: "
            f"{mean_to_baseline:.3f}deg"
        )

        print(
            "Sign-invariant axis difference: "
            f"{mean_to_baseline_axis:.3f}deg"
        )

    rms_stats = scalar_stats(
        plane_rms_values
    )

    centroid_stats = scalar_stats(
        centroid_values
    )

    if rms_stats["n"]:
        print(
            "Plane RMS: "
            f"mean={rms_stats['mean']:.3f}mm, "
            f"max={rms_stats['max']:.3f}mm"
        )

    if centroid_stats["n"]:
        print(
            "Centroid disagreement: "
            f"mean={centroid_stats['mean']:.3f}mm, "
            f"max={centroid_stats['max']:.3f}mm"
        )

    print()
    print(
        "No production calibration/artifact files were modified."
    )

    print(
        f"Results saved only under:\n  {run_dir}"
    )

    print(
        f"Summary:\n  {summary_path}"
    )

    print(
        f"Normal samples:\n  {normals_csv}"
    )

    print(
        f"Frame metrics:\n  {frame_csv}"
    )


if __name__ == "__main__":
    main()
