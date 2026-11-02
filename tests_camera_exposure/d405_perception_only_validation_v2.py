#!/usr/bin/env python3
"""
D405 perception-only live validator for the current 2026-09-28 pipeline.

SAFE SCOPE
----------
This script opens ONLY the Intel RealSense D405.
It does NOT import rospy.
It does NOT create SHER publishers/subscribers.
It does NOT command robot motion.
It does NOT overwrite auto_target.npz.
It does NOT overwrite phantom_normal.npz.
It does NOT modify hand-eye, pivot, origin, or microscope calibration files.

It tests the proposed image settings:
    Exposure = 12000
    Gain     = 16

Against the CURRENT integrated ring detector behavior:
    production residual threshold = 0.32
    close kernel = 11
    fragment merge fallback
    LAB a* primary segmentation
    HSV Saturation/Otsu sticky-lock fallback

It simultaneously reports a stricter diagnostic gate:
    residual threshold = 0.24
without changing production behavior.

It also reproduces the current in-process phantom-normal estimator, read-only:
    local circular depth patch
    red-rim exclusion
    SVD plane fit
    centroid disagreement gate
    plane RMS diagnostic

Default output:
    perception_validation_runs/YYYYMMDD_HHMMSS/

Controls:
    q / ESC : quit early
    s       : save one extra annotated/raw/mask snapshot
    r       : reset live tracking history (does not touch any production file)

Typical run from ~/Desktop/tests:
    python3 d405_perception_only_validation_v2.py

The script automatically searches /home/eyebrp/Matthew for:
    handeye_T.npz
    phantom_normal.npz
    intr_util.py

Longer run:
    python3 d405_perception_only_validation_v2.py --seconds 60

Indefinite live display:
    python3 d405_perception_only_validation_v2.py --seconds 0
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ---------------------------------------------------------------------
# Proposed D405 camera settings
# ---------------------------------------------------------------------
DEFAULT_EXPOSURE = 12000.0
DEFAULT_GAIN = 16.0

# ---------------------------------------------------------------------
# CURRENT 2026-09-28 production ring parameters
# ---------------------------------------------------------------------
RING_A_STAR_THRESH = 145
RING_MIN_AREA_PX = 1200
RING_MAX_AREA_PX = 20000
RING_PRODUCTION_RESIDUAL = 0.32
RING_DIAGNOSTIC_RESIDUAL = 0.24
RING_MIN_RADIUS_PX = 20

RING_ROI_X0 = 100
RING_ROI_X1 = 540
RING_ROI_Y0 = 200
RING_ROI_Y1 = 480

RING_SMOOTH_N = 15
RING_MAX_JITTER_MM = 3.0
RING_STICKY_MAX_JUMP_PX = 80
RING_DEPTH_WIN_PX = 5

RING_CLOSE_KERNEL_PX = 11
RING_MERGE_MAX_DIST_PX = 150
RING_MERGE_MIN_PIECE_PX = 150

RING_USE_SAT_FALLBACK = 1
RING_SAT_OTSU_MIN = 30
RING_SAT_OTSU_MAX = 200

# ---------------------------------------------------------------------
# CURRENT 2026-09-28 phantom-normal parameters
# ---------------------------------------------------------------------
NORMAL_A_STAR_THRESH = 145
NORMAL_PATCH_RADIUS_PX = 15
NORMAL_MIN_POINTS = 80
NORMAL_MAX_CENTROID_DISAGREE_MM = 5.0
NORMAL_MAX_PLANE_RMS_MM = 1.0

# Eye-only ROI used only for exposure/highlight statistics.
DEFAULT_EYE_ROI = (140, 285, 285, 430)  # x0, x1, y0, y1
SATURATION_GRAY_THRESHOLD = 250

# Display / saving
SAVE_EVERY_N = 60


def kasa_circle_fit(xs, ys):
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
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


def _try_candidates(cand_list, last_center, residual_threshold):
    worst_reason = None
    best_diag = None

    for area, xs, ys, source_kind in cand_list:
        fit = kasa_circle_fit(xs, ys)
        if fit is None:
            worst_reason = "circle fit degenerate"
            continue

        cx, cy, r = fit

        if last_center is not None:
            jump = float(
                np.hypot(cx - last_center[0], cy - last_center[1])
            )
            if jump > RING_STICKY_MAX_JUMP_PX:
                worst_reason = (
                    f"center jump {jump:.1f}px > "
                    f"{RING_STICKY_MAX_JUMP_PX}px"
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

        diag = {
            "candidate_found": True,
            "accepted": residual <= residual_threshold,
            "cx": cx,
            "cy": cy,
            "radius_px": r,
            "n_px": int(len(xs)),
            "area_px": float(area),
            "residual_ratio": residual,
            "jump_px": jump,
            "source_kind": source_kind,
            "reason": "",
        }

        # Preserve the best actual annulus-like candidate for diagnostics
        # even when a stricter threshold rejects it.
        if best_diag is None or residual < best_diag["residual_ratio"]:
            best_diag = diag.copy()

        if residual > residual_threshold:
            worst_reason = (
                f"residual={residual:.3f} > {residual_threshold:.2f}"
            )
            continue

        return diag

    if best_diag is not None:
        best_diag["accepted"] = False
        best_diag["reason"] = worst_reason or "candidate rejected"
        return best_diag

    return {
        "candidate_found": False,
        "accepted": False,
        "cx": np.nan,
        "cy": np.nan,
        "radius_px": np.nan,
        "n_px": 0,
        "area_px": 0.0,
        "residual_ratio": np.nan,
        "jump_px": np.nan,
        "source_kind": "",
        "reason": worst_reason or "no candidate passed",
    }


def _segment_ring_from_mask(
    mask,
    last_center,
    residual_threshold,
    chan_label,
):
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((RING_CLOSE_KERNEL_PX, RING_CLOSE_KERNEL_PX), np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    n_pass_px = int(np.count_nonzero(mask))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if n_labels <= 1:
        return {
            "candidate_found": False,
            "accepted": False,
            "reason": f"no {chan_label} pixels in ROI",
            "n_pass_px": n_pass_px,
        }, mask

    candidates = []
    best_area = 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        best_area = max(best_area, area)

        if area < RING_MIN_AREA_PX or area > RING_MAX_AREA_PX:
            continue

        ys, xs = np.nonzero(labels == i)
        candidates.append(
            (float(area), xs, ys, f"{chan_label}:single")
        )

    candidates.sort(key=lambda x: -x[0])

    result = _try_candidates(
        candidates, last_center, residual_threshold
    )
    result["n_pass_px"] = n_pass_px

    if result["accepted"]:
        return result, mask

    # Current pt.19 behavior: merge nearby fragments if needed.
    small_pieces = [
        i
        for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= RING_MERGE_MIN_PIECE_PX
    ]

    merged = []

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

            merged.append(
                (
                    float(merged_area),
                    xs,
                    ys,
                    f"{chan_label}:merged",
                )
            )

    merged.sort(key=lambda x: -x[0])

    merged_result = _try_candidates(
        merged, last_center, residual_threshold
    )
    merged_result["n_pass_px"] = n_pass_px

    if merged_result["accepted"]:
        return merged_result, mask

    # For diagnostics, return whichever rejected candidate had lower residual.
    choices = [
        x for x in (result, merged_result)
        if x.get("candidate_found")
        and np.isfinite(x.get("residual_ratio", np.nan))
    ]

    if choices:
        best = min(choices, key=lambda x: x["residual_ratio"])
        best["n_pass_px"] = n_pass_px
        return best, mask

    return {
        "candidate_found": False,
        "accepted": False,
        "cx": np.nan,
        "cy": np.nan,
        "radius_px": np.nan,
        "n_px": 0,
        "area_px": float(best_area),
        "residual_ratio": np.nan,
        "jump_px": np.nan,
        "source_kind": chan_label,
        "n_pass_px": n_pass_px,
        "reason": (
            f"{chan_label}: {n_pass_px}px passed, largest area="
            f"{best_area}px; no valid component/merge"
        ),
    }, mask


def detect_ring(
    color_bgr,
    last_center,
    residual_threshold,
    allow_sat_fallback=True,
):
    """
    Current production-style ring detector, parameterized only so this
    validator can compare threshold=0.32 and threshold=0.24 side-by-side.
    """
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    a_full = lab[:, :, 1]

    h, w = a_full.shape

    x0 = max(0, RING_ROI_X0)
    x1 = min(w, RING_ROI_X1)
    y0 = max(0, RING_ROI_Y0)
    y1 = min(h, RING_ROI_Y1)

    a_mask = np.zeros((h, w), dtype=np.uint8)
    a_sub = (
        a_full[y0:y1, x0:x1] > RING_A_STAR_THRESH
    ).astype(np.uint8) * 255
    a_mask[y0:y1, x0:x1] = a_sub

    result_a, mask_a = _segment_ring_from_mask(
        a_mask,
        last_center,
        residual_threshold,
        f"a*>{RING_A_STAR_THRESH}",
    )

    if result_a["accepted"]:
        return result_a, mask_a

    if not allow_sat_fallback or not RING_USE_SAT_FALLBACK:
        return result_a, mask_a

    # Current safety rule: Saturation fallback cannot originate first lock.
    if last_center is None:
        result_a["reason"] = (
            (result_a.get("reason") or "a* failed")
            + " | Sat fallback skipped: no established sticky center"
        )
        return result_a, mask_a

    s_full = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
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

    result_s, mask_s = _segment_ring_from_mask(
        s_mask,
        last_center,
        residual_threshold,
        f"Sat>{otsu_th:.0f}(Otsu)",
    )

    if result_s["accepted"]:
        return result_s, mask_s

    # Return the more ring-like rejected candidate for diagnostics.
    candidates = [
        r for r in (result_a, result_s)
        if r.get("candidate_found")
        and np.isfinite(r.get("residual_ratio", np.nan))
    ]

    if candidates:
        best = min(
            candidates,
            key=lambda r: r["residual_ratio"],
        )
        mask = (
            mask_s
            if "Sat>" in best.get("source_kind", "")
            else mask_a
        )
        return best, mask

    result_a["reason"] = (
        (result_a.get("reason") or "a* failed")
        + " | "
        + (result_s.get("reason") or "Sat failed")
    )
    return result_a, mask_a


def clamp_option(sensor, option, value):
    rng = sensor.get_option_range(option)

    value = float(
        np.clip(value, rng.min, rng.max)
    )

    if rng.step > 0:
        value = (
            rng.min
            + round((value - rng.min) / rng.step) * rng.step
        )
        value = float(
            np.clip(value, rng.min, rng.max)
        )

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
        raise RuntimeError(
            "No RealSense sensor supports manual exposure."
        )

    for sensor, name in candidates:
        if "Stereo" in name:
            return sensor, name

    return candidates[0]


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


def deproject_mm(intr, px, py, z_m):
    pt = rs.rs2_deproject_pixel_to_point(
        intr,
        [float(px), float(py)],
        float(z_m),
    )
    return np.asarray(pt, dtype=np.float64) * 1000.0



DEFAULT_PROJECT_ROOT = Path("/home/eyebrp/Matthew")
PREFERRED_CODE_DIR = DEFAULT_PROJECT_ROOT / "calibration_work_2026-09-21" / "01_code"


def _discover_project_file(filename, explicit_path=None, project_root=DEFAULT_PROJECT_ROOT):
    """
    Resolve a read-only support file when this validator lives outside the
    production code directory (for example ~/Desktop/tests).

    Priority:
      1) explicit CLI path, if supplied;
      2) /home/eyebrp/Matthew/calibration_work_2026-09-21/01_code/<filename>;
      3) direct child /home/eyebrp/Matthew/<filename>;
      4) recursive search below /home/eyebrp/Matthew.

    If several recursive matches exist, prefer paths containing
    calibration_work_2026-09-21/01_code, then shallower paths, then newest.
    The chosen path and all ambiguity are printed; nothing is modified.
    """
    root = Path(project_root).expanduser()

    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.exists():
            return p, [p], "explicit"
        return None, [], f"explicit path missing: {p}"

    preferred = root / "calibration_work_2026-09-21" / "01_code" / filename
    if preferred.exists():
        return preferred.resolve(), [preferred.resolve()], "preferred production-code directory"

    direct = root / filename
    if direct.exists():
        return direct.resolve(), [direct.resolve()], "project-root direct child"

    if not root.exists():
        return None, [], f"project root missing: {root}"

    matches = []
    try:
        matches = [p.resolve() for p in root.rglob(filename) if p.is_file()]
    except Exception as e:
        return None, [], f"recursive search failed: {e}"

    if not matches:
        return None, [], f"not found anywhere under {root}"

    def rank(p):
        s = str(p)
        preferred_flag = 0 if "calibration_work_2026-09-21/01_code" in s else 1
        try:
            depth = len(p.relative_to(root).parts)
        except Exception:
            depth = 999
        try:
            mtime_rank = -p.stat().st_mtime
        except Exception:
            mtime_rank = 0
        return (preferred_flag, depth, mtime_rank, s)

    matches.sort(key=rank)
    return matches[0], matches, "recursive project search"


def _print_discovery(label, chosen, matches, reason):
    if chosen is None:
        print(f"{label}: MISSING ({reason})")
        return

    print(f"{label}: {chosen}  [{reason}]")
    if len(matches) > 1:
        print(f"  {len(matches)} copies found; chosen path above. Other matches:")
        for p in matches[1:8]:
            print(f"    - {p}")
        if len(matches) > 8:
            print(f"    ... plus {len(matches) - 8} more")


def _prepare_project_imports(project_root=DEFAULT_PROJECT_ROOT, explicit_intr_util=None):
    """
    Make the production intr_util.py importable even when this validator is
    launched from ~/Desktop/tests.
    """
    chosen, matches, reason = _discover_project_file(
        "intr_util.py",
        explicit_path=explicit_intr_util,
        project_root=project_root,
    )

    _print_discovery("intr_util.py", chosen, matches, reason)

    if chosen is None:
        return None

    parent = str(chosen.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    return chosen


def load_handeye(path):
    p = Path(path)

    if not p.exists():
        return None, None, f"missing: {p}"

    try:
        d = np.load(str(p))
        R = np.asarray(d["R"], dtype=float)
        t = np.asarray(d["t"], dtype=float).reshape(3)

        if R.shape != (3, 3):
            return None, None, f"invalid R shape: {R.shape}"

        return R, t, str(p)
    except Exception as e:
        return None, None, f"load error: {e}"


def load_baseline_normal(path):
    p = Path(path)

    if not p.exists():
        return None, f"missing: {p}"

    try:
        d = np.load(str(p))

        rec = {
            "normal": np.asarray(
                d["normal_base"], dtype=float
            ).reshape(3),
            "point": np.asarray(
                d["point_base"], dtype=float
            ).reshape(3),
            "plane_rms_mm": float(d["plane_rms_mm"]),
            "n_points": int(d["n_points"]),
        }

        rec["normal"] /= np.linalg.norm(rec["normal"])

        return rec, str(p)
    except Exception as e:
        return None, f"load error: {e}"


def eye_exposure_stats(color_bgr, eye_roi):
    x0, x1, y0, y1 = eye_roi
    h, w = color_bgr.shape[:2]

    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    roi = color_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    return {
        "sat_pct": float(
            np.mean(gray >= SATURATION_GRAY_THRESHOLD) * 100.0
        ),
        "mean_gray": float(np.mean(gray)),
        "p95_gray": float(np.percentile(gray, 95)),
        "p99_gray": float(np.percentile(gray, 99)),
        "max_gray": int(np.max(gray)),
    }


def estimate_phantom_normal(
    color_bgr,
    depth_frame,
    intr,
    R_he,
    t_he,
    rcm_base,
):
    """
    Read-only reproduction of the integrated 2026-09-28
    estimate_phantom_normal_from_frame() logic.
    """
    if R_he is None:
        return {
            "ok": False,
            "fail_reason": "hand-eye missing",
        }

    rcm_cam = R_he.T.dot(rcm_base - t_he)

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

    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    a_star = lab[:, :, 1]

    h = depth_frame.get_height()
    w = depth_frame.get_width()

    pts_cam = []
    n_red_dropped = 0

    patch_pixels = []

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

            z = depth_frame.get_distance(uu, vv) * 1000.0

            if z <= 0:
                continue

            x = (uu - intr.ppx) / intr.fx * z
            y = (vv - intr.ppy) / intr.fy * z

            pts_cam.append((x, y, z))
            patch_pixels.append((uu, vv))

    pts_cam = np.asarray(pts_cam, dtype=float)

    if len(pts_cam) < NORMAL_MIN_POINTS:
        return {
            "ok": False,
            "n_points": int(len(pts_cam)),
            "n_red_dropped": int(n_red_dropped),
            "u0": u0,
            "v0": v0,
            "patch_pixels": patch_pixels,
            "fail_reason": (
                f"only {len(pts_cam)} non-red depth points "
                f"(<{NORMAL_MIN_POINTS})"
            ),
        }

    centroid = pts_cam.mean(axis=0)
    X = pts_cam - centroid

    _, _, Vt = np.linalg.svd(
        X, full_matrices=False
    )

    normal_cam = Vt[-1]
    normal_cam /= np.linalg.norm(normal_cam)

    resid = X.dot(normal_cam)
    plane_rms_mm = float(
        np.sqrt(np.mean(resid ** 2))
    )

    view_dir = centroid / np.linalg.norm(centroid)

    if np.dot(normal_cam, view_dir) > 0:
        normal_cam = -normal_cam

    centroid_base = R_he.dot(centroid) + t_he

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
            "patch_pixels": patch_pixels,
            "plane_rms_mm": plane_rms_mm,
            "centroid_disagree_mm": centroid_disagree_mm,
            "fail_reason": (
                f"centroid disagreement "
                f"{centroid_disagree_mm:.2f}mm > "
                f"{NORMAL_MAX_CENTROID_DISAGREE_MM:.1f}mm"
            ),
        }

    normal_base = R_he.dot(normal_cam)
    normal_base /= np.linalg.norm(normal_base)

    advisory = ""

    if plane_rms_mm > NORMAL_MAX_PLANE_RMS_MM:
        advisory = (
            f"plane RMS {plane_rms_mm:.2f}mm > "
            f"{NORMAL_MAX_PLANE_RMS_MM:.1f}mm"
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
        "patch_pixels": patch_pixels,
        "plane_rms_mm": plane_rms_mm,
        "centroid_disagree_mm": centroid_disagree_mm,
        "fail_reason": advisory,
    }


def angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    c = float(
        np.clip(np.dot(a, b), -1.0, 1.0)
    )

    return float(np.degrees(np.arccos(c)))


def mean_unit_vector(vectors):
    arr = np.asarray(vectors, dtype=float)

    if arr.size == 0:
        return None

    m = arr.mean(axis=0)
    n = np.linalg.norm(m)

    if n <= 1e-9:
        return None

    return m / n


def save_snapshot(
    raw_dir,
    ann_dir,
    mask_dir,
    idx,
    color,
    annotated,
    mask,
):
    cv2.imwrite(
        str(raw_dir / f"frame_{idx:05d}.png"),
        color,
    )
    cv2.imwrite(
        str(ann_dir / f"frame_{idx:05d}.png"),
        annotated,
    )
    cv2.imwrite(
        str(mask_dir / f"frame_{idx:05d}.png"),
        mask,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "D405 perception-only validation: no ROS, no robot motion."
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
        default=30.0,
        help="0 = run until q/ESC",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--eye-roi",
        nargs=4,
        type=int,
        default=list(DEFAULT_EYE_ROI),
        metavar=("X0", "X1", "Y0", "Y1"),
    )
    parser.add_argument(
        "--normal-every",
        type=int,
        default=15,
        help=(
            "Estimate phantom normal every N frames after production-like "
            "3D lock; 0 disables normal testing."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY_N,
    )

    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
        help=(
            "Root containing the real CIS/SHER project files. "
            "Default: /home/eyebrp/Matthew"
        ),
    )
    parser.add_argument(
        "--handeye",
        default=None,
        help=(
            "Optional explicit handeye_T.npz path. If omitted, search "
            "--project-root and prefer calibration_work_2026-09-21/01_code."
        ),
    )
    parser.add_argument(
        "--baseline-normal",
        default=None,
        help=(
            "Optional explicit phantom_normal.npz path. If omitted, search "
            "--project-root and prefer calibration_work_2026-09-21/01_code."
        ),
    )
    parser.add_argument(
        "--intr-util",
        default=None,
        help=(
            "Optional explicit intr_util.py path. If omitted, search "
            "--project-root so this test can live in ~/Desktop/tests."
        ),
    )

    args = parser.parse_args()

    eye_roi = tuple(args.eye_roi)

    run_dir = (
        script_dir
        / "perception_validation_runs"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    raw_dir = run_dir / "raw_samples"
    ann_dir = run_dir / "annotated_samples"
    mask_dir = run_dir / "ring_masks"

    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(args.project_root).expanduser()

    handeye_path, handeye_matches, handeye_reason = _discover_project_file(
        "handeye_T.npz",
        explicit_path=args.handeye,
        project_root=project_root,
    )
    normal_path, normal_matches, normal_reason = _discover_project_file(
        "phantom_normal.npz",
        explicit_path=args.baseline_normal,
        project_root=project_root,
    )

    _print_discovery(
        "handeye_T.npz",
        handeye_path,
        handeye_matches,
        handeye_reason,
    )
    _print_discovery(
        "phantom_normal.npz",
        normal_path,
        normal_matches,
        normal_reason,
    )

    R_he, t_he, handeye_status = load_handeye(
        handeye_path if handeye_path is not None else "__missing_handeye__"
    )
    baseline_normal, baseline_status = load_baseline_normal(
        normal_path if normal_path is not None else "__missing_phantom_normal__"
    )

    print("\n" + "=" * 78)
    print("D405 PERCEPTION-ONLY VALIDATION")
    print("=" * 78)
    print("NO ROS / NO ROBOT COMMANDS / NO production NPZ writes")
    print(f"Output only: {run_dir}")
    print(
        f"Requested: Exposure={args.exposure:.0f}, "
        f"Gain={args.gain:.0f}"
    )
    print(
        f"Production residual threshold: "
        f"{RING_PRODUCTION_RESIDUAL:.2f}"
    )
    print(
        f"Diagnostic comparison threshold: "
        f"{RING_DIAGNOSTIC_RESIDUAL:.2f}"
    )
    print(f"Hand-eye read-only: {handeye_status}")
    print(f"Baseline phantom normal read-only: {baseline_status}")
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

    profile = pipe.start(cfg)

    align = rs.align(rs.stream.color)

    device = profile.get_device()
    depth_scale = (
        device.first_depth_sensor().get_depth_scale()
    )

    # Import the SAME production intr_util even though this validator may
    # live in ~/Desktop/tests.
    intr_util_path = _prepare_project_imports(
        project_root=project_root,
        explicit_intr_util=args.intr_util,
    )

    try:
        from intr_util import load_best_intrinsics
        intr = load_best_intrinsics(profile)
        intr_source = (
            f"load_best_intrinsics(profile) from {intr_util_path}"
            if intr_util_path is not None
            else "load_best_intrinsics(profile)"
        )
    except Exception as e:
        intr = (
            profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        intr_source = (
            "factory/profile color intrinsics fallback "
            f"(project intr_util unavailable/failed: {e})"
        )

    sensor, sensor_name = find_exposure_sensor(device)

    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(
            rs.option.enable_auto_exposure,
            0,
        )

    exposure = clamp_option(
        sensor,
        rs.option.exposure,
        args.exposure,
    )
    sensor.set_option(
        rs.option.exposure,
        exposure,
    )

    actual_gain = float("nan")

    if sensor.supports(rs.option.gain):
        gain = clamp_option(
            sensor,
            rs.option.gain,
            args.gain,
        )
        sensor.set_option(
            rs.option.gain,
            gain,
        )
        actual_gain = float(
            sensor.get_option(rs.option.gain)
        )

    actual_exposure = float(
        sensor.get_option(rs.option.exposure)
    )

    print(
        f"Exposure sensor: {sensor_name}; "
        f"actual exposure={actual_exposure:.0f}; "
        f"actual gain={actual_gain:.0f}"
    )
    print(f"Intrinsics source: {intr_source}")

    # Warmup after manual settings are applied.
    for _ in range(30):
        pipe.wait_for_frames()

    hist_3d = deque(maxlen=RING_SMOOTH_N)
    center_hist = deque(maxlen=RING_SMOOTH_N)
    radius_hist = deque(maxlen=RING_SMOOTH_N)

    last_center = None

    normal_history = deque(maxlen=30)
    last_normal_result = None

    rows = []

    total_frames = 0
    prod_accept_count = 0
    diag_accept_count = 0
    depth_valid_count = 0
    prod_lock_frames = 0
    normal_ok_count = 0
    sat_fallback_count = 0

    start = time.time()
    snapshot_counter = 0

    cv2.namedWindow(
        "D405 PERCEPTION ONLY - NO ROBOT MOTION",
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        "D405 PERCEPTION ONLY - NO ROBOT MOTION",
        1100,
        820,
    )

    try:
        while True:
            if args.seconds > 0:
                if time.time() - start >= args.seconds:
                    break

            frames = align.process(
                pipe.wait_for_frames()
            )

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
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

            # Use same pre-frame sticky center for fair 0.32 vs 0.24 comparison.
            sticky_before = last_center

            prod, prod_mask = detect_ring(
                color,
                sticky_before,
                RING_PRODUCTION_RESIDUAL,
                allow_sat_fallback=True,
            )

            diag, _ = detect_ring(
                color,
                sticky_before,
                RING_DIAGNOSTIC_RESIDUAL,
                allow_sat_fallback=True,
            )

            if prod["accepted"]:
                prod_accept_count += 1
                last_center = (
                    prod["cx"],
                    prod["cy"],
                )

                center_hist.append(
                    [prod["cx"], prod["cy"]]
                )
                radius_hist.append(
                    prod["radius_px"]
                )

                if "Sat>" in prod.get("source_kind", ""):
                    sat_fallback_count += 1

            if diag["accepted"]:
                diag_accept_count += 1

            base_or_cam_pt = None
            z_mm = float("nan")

            if prod["accepted"]:
                z_mm = depth_min_at(
                    depth_mm,
                    int(round(prod["cx"])),
                    int(round(prod["cy"])),
                    RING_DEPTH_WIN_PX,
                )

                if np.isfinite(z_mm) and z_mm > 0:
                    depth_valid_count += 1

                    cam_pt = deproject_mm(
                        intr,
                        prod["cx"],
                        prod["cy"],
                        z_mm / 1000.0,
                    )

                    if R_he is not None:
                        base_or_cam_pt = (
                            R_he.dot(cam_pt) + t_he
                        )
                    else:
                        base_or_cam_pt = cam_pt

                    hist_3d.append(
                        base_or_cam_pt
                    )

            jitter_mm = float("nan")
            smoothed_3d = None
            production_like_locked = False

            if len(hist_3d) >= 3:
                arr3 = np.asarray(
                    hist_3d,
                    dtype=float,
                )
                smoothed_3d = arr3.mean(axis=0)
                jitter_mm = float(
                    arr3.std(axis=0).max()
                )

            if (
                len(hist_3d) >= RING_SMOOTH_N
                and np.isfinite(jitter_mm)
                and jitter_mm < RING_MAX_JITTER_MM
            ):
                production_like_locked = True
                prod_lock_frames += 1

            # Normal estimate only when we have a production-like stable base RCM.
            if (
                args.normal_every > 0
                and production_like_locked
                and R_he is not None
                and total_frames % args.normal_every == 0
            ):
                normal_result = estimate_phantom_normal(
                    color,
                    depth_frame,
                    intr,
                    R_he,
                    t_he,
                    smoothed_3d,
                )

                last_normal_result = normal_result

                if normal_result.get("ok"):
                    normal_ok_count += 1
                    normal_history.append(
                        normal_result["normal_base"]
                    )

                    if baseline_normal is not None:
                        normal_result["angle_to_baseline_deg"] = angle_deg(
                            normal_result["normal_base"],
                            baseline_normal["normal"],
                        )
                    else:
                        normal_result["angle_to_baseline_deg"] = float("nan")

            exposure_stats = eye_exposure_stats(
                color,
                eye_roi,
            )

            # 2D stability
            center_std_x = float("nan")
            center_std_y = float("nan")
            center_radial_rms = float("nan")
            radius_std = float("nan")

            if len(center_hist) >= 3:
                c_arr = np.asarray(
                    center_hist,
                    dtype=float,
                )
                c_mean = c_arr.mean(axis=0)
                c_std = c_arr.std(axis=0)

                center_std_x = float(c_std[0])
                center_std_y = float(c_std[1])

                center_radial_rms = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (c_arr - c_mean) ** 2,
                                axis=1,
                            )
                        )
                    )
                )

                radius_std = float(
                    np.std(
                        np.asarray(
                            radius_hist,
                            dtype=float,
                        )
                    )
                )

            # Normal stability against rolling mean
            normal_mean = mean_unit_vector(
                normal_history
            )
            normal_spread_max_deg = float("nan")

            if (
                normal_mean is not None
                and len(normal_history) >= 2
            ):
                diffs = [
                    angle_deg(n, normal_mean)
                    for n in normal_history
                ]
                normal_spread_max_deg = float(
                    np.max(diffs)
                )

            row = {
                "frame": total_frames,
                "elapsed_s": time.time() - start,
                "exposure": actual_exposure,
                "gain": actual_gain,

                "eye_saturation_pct": exposure_stats["sat_pct"],
                "eye_mean_gray": exposure_stats["mean_gray"],
                "eye_p99_gray": exposure_stats["p99_gray"],

                "prod_candidate_found": int(prod["candidate_found"]),
                "prod_accepted_032": int(prod["accepted"]),
                "prod_source": prod.get("source_kind", ""),
                "prod_cx_px": prod["cx"] if prod["candidate_found"] else "",
                "prod_cy_px": prod["cy"] if prod["candidate_found"] else "",
                "prod_radius_px": prod["radius_px"] if prod["candidate_found"] else "",
                "prod_n_px": prod["n_px"] if prod["candidate_found"] else "",
                "prod_residual": prod["residual_ratio"] if prod["candidate_found"] else "",
                "prod_reason": prod.get("reason", ""),

                "diag_accepted_024": int(diag["accepted"]),
                "diag_residual": diag["residual_ratio"] if diag["candidate_found"] else "",

                "center_std_x_px": center_std_x,
                "center_std_y_px": center_std_y,
                "center_radial_rms_px": center_radial_rms,
                "radius_std_px": radius_std,

                "depth_mm": z_mm if np.isfinite(z_mm) else "",
                "tracking_3d_jitter_mm": jitter_mm if np.isfinite(jitter_mm) else "",
                "production_like_locked": int(production_like_locked),

                "normal_ok": (
                    int(last_normal_result.get("ok", False))
                    if last_normal_result is not None
                    else ""
                ),
                "normal_n_points": (
                    last_normal_result.get("n_points", "")
                    if last_normal_result is not None
                    else ""
                ),
                "normal_red_dropped": (
                    last_normal_result.get("n_red_dropped", "")
                    if last_normal_result is not None
                    else ""
                ),
                "normal_plane_rms_mm": (
                    last_normal_result.get("plane_rms_mm", "")
                    if last_normal_result is not None
                    else ""
                ),
                "normal_centroid_disagree_mm": (
                    last_normal_result.get("centroid_disagree_mm", "")
                    if last_normal_result is not None
                    else ""
                ),
                "normal_angle_to_baseline_deg": (
                    last_normal_result.get(
                        "angle_to_baseline_deg",
                        "",
                    )
                    if last_normal_result is not None
                    else ""
                ),
                "normal_rolling_max_angle_from_mean_deg": normal_spread_max_deg,
                "normal_fail_or_advisory": (
                    last_normal_result.get("fail_reason", "")
                    if last_normal_result is not None
                    else ""
                ),
            }

            rows.append(row)

            # -------------------------------------------------------------
            # Live display
            # -------------------------------------------------------------
            view = color.copy()

            cv2.rectangle(
                view,
                (RING_ROI_X0, RING_ROI_Y0),
                (RING_ROI_X1, RING_ROI_Y1),
                (110, 110, 110),
                1,
            )

            ex0, ex1, ey0, ey1 = eye_roi
            cv2.rectangle(
                view,
                (ex0, ey0),
                (ex1, ey1),
                (255, 255, 255),
                1,
            )

            if prod["candidate_found"]:
                col = (
                    (0, 255, 80)
                    if prod["accepted"]
                    else (0, 165, 255)
                )

                cv2.circle(
                    view,
                    (
                        int(round(prod["cx"])),
                        int(round(prod["cy"])),
                    ),
                    int(round(prod["radius_px"])),
                    col,
                    2,
                )

                cv2.circle(
                    view,
                    (
                        int(round(prod["cx"])),
                        int(round(prod["cy"])),
                    ),
                    4,
                    (0, 255, 255),
                    -1,
                )

            # Show normal patch center when available.
            if (
                last_normal_result is not None
                and "u0" in last_normal_result
            ):
                cv2.circle(
                    view,
                    (
                        int(last_normal_result["u0"]),
                        int(last_normal_result["v0"]),
                    ),
                    NORMAL_PATCH_RADIUS_PX,
                    (255, 255, 0),
                    1,
                )

            # Large status panel
            panel_h = 186
            cv2.rectangle(
                view,
                (0, 0),
                (640, panel_h),
                (0, 0, 0),
                -1,
            )

            prod_rate = (
                100.0 * prod_accept_count / total_frames
            )
            diag_rate = (
                100.0 * diag_accept_count / total_frames
            )
            lock_rate = (
                100.0 * prod_lock_frames / total_frames
            )

            residual_txt = (
                f"{prod['residual_ratio']:.3f}"
                if prod["candidate_found"]
                and np.isfinite(prod["residual_ratio"])
                else "NA"
            )

            line1 = (
                f"SAFE CAMERA-ONLY | EXP={actual_exposure:.0f} "
                f"GAIN={actual_gain:.0f} | no ROS / no motion"
            )

            line2 = (
                f"RING prod<=0.32: {int(prod['accepted'])} "
                f"({prod_rate:.1f}%) | diag<=0.24: "
                f"{int(diag['accepted'])} ({diag_rate:.1f}%) "
                f"| resid={residual_txt}"
            )

            line3 = (
                f"src={prod.get('source_kind','') or 'none'} "
                f"| r={prod['radius_px']:.2f}px "
                f"| center_RMS={center_radial_rms:.3f}px "
                f"| radius_std={radius_std:.3f}px"
                if prod["candidate_found"]
                else (
                    f"ring reject: "
                    f"{(prod.get('reason') or '?')[:80]}"
                )
            )

            line4 = (
                f"EYE sat>={SATURATION_GRAY_THRESHOLD}: "
                f"{exposure_stats['sat_pct']:.2f}% "
                f"| mean={exposure_stats['mean_gray']:.1f} "
                f"| p99={exposure_stats['p99_gray']:.1f}"
            )

            line5 = (
                f"3D samples={len(hist_3d)}/{RING_SMOOTH_N} "
                f"| jitter={jitter_mm:.3f}mm "
                f"| prod-like LOCK={int(production_like_locked)} "
                f"({lock_rate:.1f}% frames)"
                if np.isfinite(jitter_mm)
                else (
                    f"3D samples={len(hist_3d)}/{RING_SMOOTH_N} "
                    f"| jitter=NA | prod-like LOCK=0"
                )
            )

            if last_normal_result is None:
                line6 = (
                    "NORMAL waiting for stable 3D ring lock..."
                )
            elif last_normal_result.get("ok"):
                ang = last_normal_result.get(
                    "angle_to_baseline_deg",
                    float("nan"),
                )
                line6 = (
                    f"NORMAL OK n={last_normal_result['n_points']} "
                    f"RMS={last_normal_result['plane_rms_mm']:.3f}mm "
                    f"centroidDiff="
                    f"{last_normal_result['centroid_disagree_mm']:.3f}mm "
                    f"baselineAngle="
                    f"{ang:.2f}deg"
                )
            else:
                line6 = (
                    "NORMAL FAIL: "
                    + (
                        last_normal_result.get(
                            "fail_reason",
                            "?",
                        )
                    )[:90]
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

            for text_line in lines:
                cv2.putText(
                    view,
                    text_line,
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 29

            cv2.imshow(
                "D405 PERCEPTION ONLY - NO ROBOT MOTION",
                view,
            )

            if (
                args.save_every > 0
                and total_frames % args.save_every == 0
            ):
                save_snapshot(
                    raw_dir,
                    ann_dir,
                    mask_dir,
                    total_frames,
                    color,
                    view,
                    prod_mask,
                )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("s"):
                snapshot_counter += 1
                save_snapshot(
                    raw_dir,
                    ann_dir,
                    mask_dir,
                    900000 + snapshot_counter,
                    color,
                    view,
                    prod_mask,
                )
                print("Saved manual snapshot.")

            if key == ord("r"):
                hist_3d.clear()
                center_hist.clear()
                radius_hist.clear()
                normal_history.clear()
                last_center = None
                last_normal_result = None
                print(
                    "Tracking history reset in memory only."
                )

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    def finite(vals):
        arr = np.asarray(vals, dtype=float)
        return arr[np.isfinite(arr)]

    def stats(vals):
        arr = finite(vals)

        if arr.size == 0:
            return {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

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
        "ring_configuration": {
            "production_residual_threshold": RING_PRODUCTION_RESIDUAL,
            "diagnostic_residual_threshold": RING_DIAGNOSTIC_RESIDUAL,
            "a_star_threshold": RING_A_STAR_THRESH,
            "close_kernel_px": RING_CLOSE_KERNEL_PX,
            "merge_max_dist_px": RING_MERGE_MAX_DIST_PX,
            "sat_fallback_enabled": bool(RING_USE_SAT_FALLBACK),
            "smooth_n": RING_SMOOTH_N,
            "max_jitter_mm": RING_MAX_JITTER_MM,
        },
        "files_read_only": {
            "handeye": handeye_status,
            "baseline_normal": baseline_status,
        },
        "results": {
            "frames": total_frames,
            "production_accept_rate_pct": (
                100.0 * prod_accept_count / total_frames
                if total_frames
                else 0.0
            ),
            "diagnostic_024_accept_rate_pct": (
                100.0 * diag_accept_count / total_frames
                if total_frames
                else 0.0
            ),
            "depth_valid_given_prod_accept_pct": (
                100.0 * depth_valid_count / prod_accept_count
                if prod_accept_count
                else 0.0
            ),
            "production_like_lock_frame_pct": (
                100.0 * prod_lock_frames / total_frames
                if total_frames
                else 0.0
            ),
            "sat_fallback_frame_count": sat_fallback_count,
            "normal_ok_count": normal_ok_count,
        },
        "metrics": {
            "eye_saturation_pct": stats(
                [
                    r["eye_saturation_pct"]
                    for r in rows
                ]
            ),
            "ring_residual_prod_candidates": stats(
                [
                    r["prod_residual"]
                    for r in rows
                    if r["prod_residual"] != ""
                ]
            ),
            "ring_radius_px": stats(
                [
                    r["prod_radius_px"]
                    for r in rows
                    if r["prod_radius_px"] != ""
                ]
            ),
            "center_radial_rms_px": stats(
                [
                    r["center_radial_rms_px"]
                    for r in rows
                    if r["center_radial_rms_px"] != ""
                ]
            ),
            "tracking_3d_jitter_mm": stats(
                [
                    r["tracking_3d_jitter_mm"]
                    for r in rows
                    if r["tracking_3d_jitter_mm"] != ""
                ]
            ),
            "normal_plane_rms_mm": stats(
                [
                    r["normal_plane_rms_mm"]
                    for r in rows
                    if r["normal_plane_rms_mm"] != ""
                ]
            ),
            "normal_centroid_disagree_mm": stats(
                [
                    r["normal_centroid_disagree_mm"]
                    for r in rows
                    if r["normal_centroid_disagree_mm"] != ""
                ]
            ),
            "normal_angle_to_baseline_deg": stats(
                [
                    r["normal_angle_to_baseline_deg"]
                    for r in rows
                    if r["normal_angle_to_baseline_deg"] != ""
                ]
            ),
        },
    }

    csv_path = run_dir / "frame_metrics.csv"
    summary_path = run_dir / "summary.json"

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("RESULT SUMMARY")
    print("=" * 78)

    if total_frames:
        print(
            f"Production <=0.32 accept: "
            f"{100.0 * prod_accept_count / total_frames:.1f}%"
        )
        print(
            f"Diagnostic <=0.24 accept: "
            f"{100.0 * diag_accept_count / total_frames:.1f}%"
        )
        print(
            f"Production-like 15-sample/<3mm lock frames: "
            f"{100.0 * prod_lock_frames / total_frames:.1f}%"
        )

    print(
        f"Saturation fallback used on "
        f"{sat_fallback_count} accepted frames."
    )
    print(
        f"Successful phantom-normal estimates: "
        f"{normal_ok_count}"
    )
    print(f"Results saved only under:\n  {run_dir}")
    print(f"CSV:\n  {csv_path}")
    print(f"Summary:\n  {summary_path}")


if __name__ == "__main__":
    main()
