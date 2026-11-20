#!/usr/bin/env python3
"""mic_perception.py -- importable, low-latency classical-CV landmark
detector for the Leica overhead eye-phantom view.

This is a straight refactor of the core detection functions in
../leica_entry_point/segment_leica.py into a single importable call,
`locate(img_bgr) -> dict`, so the SAME tested geometry logic can be:
  1. run offline over a folder of stills (as segment_leica.py already does), and
  2. called per-frame from a live capture loop or from a one-shot
     "here's today's microscope still, tell me the entry point" script,
without duplicating the algorithm. Runtime is ~10-30ms/frame on CPU
(Hough + a 1D radial profile + one contour pass) -- fine for either use.

Deliberately does NOT do instrument/needle-tip detection: the 2026-09-15
and 2026-09-17 datasets both show the instrument as an out-of-focus blob
parked in roughly the same spot (see chat writeup), so there is no reliable
signal to detect yet. `locate()` sets occluded=True as a cheap proxy any
time the classical fit fails or lands on an implausible radius, which is
the main symptom that occlusion produces.
"""
import numpy as np
import cv2

PLUG_MIN_R, PLUG_MAX_R = 140, 280
PLUG_EXPECTED_R = 236          # empirical stable value across both datasets
RING_MAX_THICK = 140


def focus_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_plug_circle(gray):
    """Hough-detect the plug/limbus circle.

    NOTE (2026-09-28): under some lighting conditions HoughCircles also
    returns a spurious circle centered on one of the two phantom mounting
    screws (their radius can land inside [PLUG_MIN_R, PLUG_MAX_R] and
    sometimes even closer to PLUG_EXPECTED_R than the true limbus edge).
    Picking purely by "closest radius to PLUG_EXPECTED_R" is therefore not
    robust -- it flickered between the true ring and a screw frame-to-frame
    with identical lighting. The camera framing is fixed, so the true plug
    is always near the image center while the screws sit well off to the
    side; combining radius-closeness with center-distance (weighted 2x,
    normalized) reliably separates the two (validated: true ring scores
    ~0.22, screw scores ~0.6-0.8 on real captured frames).
    """
    g = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=500,
                                param1=80, param2=40,
                                minRadius=PLUG_MIN_R, maxRadius=PLUG_MAX_R)
    if circles is None:
        return None
    h, w = gray.shape
    cx0, cy0 = w / 2.0, h / 2.0

    def score(c):
        x, y, r = c
        r_err = abs(r - PLUG_EXPECTED_R) / PLUG_EXPECTED_R
        center_err = float(np.hypot(x - cx0, y - cy0)) / max(w, h)
        return r_err + 2.0 * center_err

    best = min(circles[0], key=score)
    return float(best[0]), float(best[1]), float(best[2])


def find_ring_outer_radius(gray, cx, cy, r_inner):
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rs = np.arange(int(r_inner), int(r_inner) + RING_MAX_THICK, 3)
    profile = []
    for r in rs:
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if valid.sum() < 60:
            profile.append(np.nan)
            continue
        profile.append(float(np.median(gray[ys[valid], xs[valid]])))
    profile = np.array(profile)
    if np.all(np.isnan(profile)):
        return r_inner + 60.0
    dark_level = np.nanmin(profile)
    bright_ref = float(gray[max(0, int(cy - r_inner)):int(cy), int(cx)].mean()) if cy - r_inner > 0 else 150.0
    thresh = 0.5 * (dark_level + bright_ref)
    dark_idx = int(np.nanargmin(profile))
    for i in range(dark_idx, len(profile)):
        if not np.isnan(profile[i]) and profile[i] >= thresh:
            return float(rs[i])
    return float(rs[-1])


MARK_SEARCH_MARGIN = 180   # px beyond r_outer_ring (was 220; tightened 2026-09-28
                           # to keep the search annulus from reaching into the
                           # decorative outer ring's scalloped texture, which
                           # otherwise merges with the real ink mark into one
                           # low-solidity blob and gets rejected)
import os as _os
# dark_thresh = (85th-pct background gray) * this. 2026-10-12: under that day's lighting the
# left (mark0) screw head read median gray 112 vs thresh 104 at 0.72 -> only 1 mark -> occluded;
# 0.80 found both marks in 5/5 live frames (run_main.sh sets it).
MARK_DARK_FRACTION = float(_os.environ.get("MARK_DARK_FRACTION", "0.72"))
# 2026-10-11: morphological CLOSE kernel for the mark-blob mask. The old fixed 9x9
# close bridged a screw-head mark with a nearby dark patch of the pink outer ring
# in ~35% of frames (merged blob, solidity 0.36 < 0.6 -> rejected -> only 1 mark ->
# occluded=True), so the Leica panel flickered Live<->Occluded on a static phantom.
# Offline sweep on 50 static frames: close=1 (off) -> 49/50 frames with both marks,
# centroid std ~1.4px; close=9 -> 33/50. Set MARK_CLOSE_K=9 to restore the old behaviour.
import os as _os
MARK_CLOSE_K = int(_os.environ.get("MARK_CLOSE_K", "1"))
# If set (0 or 1), the circle target uses THAT mark's own radius from the plug centre
# (a point on the circle through that mark) instead of the mean over both marks.
_trm = _os.environ.get("MIC_TARGET_RADIUS_MARK")
TARGET_RADIUS_MARK = int(_trm) if _trm not in (None, "") else None
# If set (0 or 1), target_angle_deg is an offset from THAT mark's measured angle around
# the plug centre (rotation-invariant) instead of an absolute image angle.
_tam = _os.environ.get("MIC_TARGET_ANGLE_REL_MARK")
TARGET_ANGLE_REL_MARK = int(_tam) if _tam not in (None, "") else None


def detect_entry_marks(img_bgr, cx, cy, r_outer_ring):
    """Find the two dark ink entry-marks in the annulus just outside the ring.

    NOTE (2026-09-28): the "dark" cutoff used to be a fixed gray<90, which
    was tuned under one specific lighting setup. When the microscope's
    brightness/exposure drifts (confirmed to happen across sessions), the
    marks' actual gray level drifts too (observed ~90-110 under dimmer
    lighting vs ~<90 under the original tuning), so a fixed absolute
    threshold silently stops finding them (occluded=True with no real
    occlusion). Fixed by thresholding relative to the *current* frame's
    own background brightness (85th percentile of the search annulus,
    which is dominated by sclera-colored background) instead of an
    absolute gray level -- this tracks lighting changes automatically.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    annulus = np.zeros((h, w), np.uint8)
    cv2.circle(annulus, (int(cx), int(cy)), int(r_outer_ring + MARK_SEARCH_MARGIN), 255, -1)
    cv2.circle(annulus, (int(cx), int(cy)), int(r_outer_ring * 1.05), 0, -1)

    ann_vals = gray[annulus > 0]
    if ann_vals.size < 60:
        return []
    bright_ref = float(np.percentile(ann_vals, 85))
    dark_thresh = bright_ref * MARK_DARK_FRACTION

    dark = ((gray < dark_thresh) & (gray > 5)).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, dark, mask=annulus)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if MARK_CLOSE_K > 1:
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((MARK_CLOSE_K, MARK_CLOSE_K), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 3000 or area > 45000:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.6:
            continue
        blobs.append((float(x), float(y), float(r), float(area), float(solidity)))
    blobs.sort(key=lambda b: -b[3])
    return sorted(blobs[:2], key=lambda b: b[0])  # sorted left->right (mark0, mark1)


def polar_from_center(cx, cy, px, py):
    """(radius_px, angle_deg) of point (px,py) relative to (cx,cy).
    angle_deg uses the SAME convention as detect_entry_marks' left/right
    sort (atan2(dy,dx), degrees, range (-180,180])."""
    r = float(np.hypot(px - cx, py - cy))
    theta = float(np.degrees(np.arctan2(py - cy, px - cx)))
    return r, theta


def point_on_plug_circle(plug_cx, plug_cy, radius_px, angle_deg):
    """2026-10-05: THE fix for 'autonomous targeting should not be locked
    onto the two literal ink dots'. detect_entry_marks() only ever finds
    the two marks the phantom happens to have painted on it -- those are
    arbitrary calibration landmarks, not the clinically meaningful target.
    The real target for an autonomous insertion is ANY point on the circle
    concentric with the limbus (plug_cx, plug_cy) at the clinically-correct
    radius from it -- i.e. parameterized by (radius, angle), not by "which
    of the 2 ink blobs". This is the general primitive: given a limbus
    center + a radius (clinical distance from limbus to sclerotomy site,
    estimated once from the known ink marks via entry_radius_from_marks()
    below, or supplied directly) + any desired angle, return that point's
    pixel coordinates. mark0/mark1 are then just the two special cases
    angle=theta(mark0), theta(mark1) -- kept only for backward
    compatibility with the existing calibration/offset pipeline."""
    th = np.radians(angle_deg)
    return (plug_cx + radius_px * np.cos(th), plug_cy + radius_px * np.sin(th))


def entry_radius_from_marks(rec):
    """Average radius-from-plug-center over whatever ink marks locate()
    found in THIS frame (1 or 2). Self-calibrating fallback for
    point_on_plug_circle()'s radius_px when no externally-supplied/learned
    radius is available. Returns None if no marks or no plug."""
    if rec.get("plug_cx") is None:
        return None
    rs = []
    for i in (0, 1):
        if rec.get("mark%d_r_from_plug" % i) is not None:
            rs.append(rec["mark%d_r_from_plug" % i])
    return float(np.mean(rs)) if rs else None


def locate(img_bgr, target_angle_deg=None, target_radius_px=None):
    """One-shot landmark localization on a single BGR frame.

    Returns a dict:
      ok            -- bool, plug/limbus edge found at all
      occluded      -- bool, cheap proxy for "don't trust this frame"
                       (plug not found, OR ring fit degenerate, OR fewer
                       than 2 ink marks -- all symptoms seen when the
                       instrument/blue occluder is in frame)
      plug_cx/cy/r, ring_outer_r  -- px
      mark0_cx/cy/r, mark1_cx/cy/r -- px (None if not found)
      mark0_r_from_plug/theta_deg, mark1_r_from_plug/theta_deg -- polar
                       coords of each mark relative to the plug/limbus
                       center (2026-10-05 addition -- see
                       point_on_plug_circle()'s docstring for WHY: this is
                       the diagnostic that lets you check whether the 2
                       ink marks actually sit at a consistent radius from
                       the limbus, i.e. whether "radius" is a safe,
                       angle-independent clinical constant to target any
                       other angle at)
      target_cx/cy  -- (2026-10-05) if target_angle_deg is given, the
                       pixel coords of the GENERAL point on the
                       limbus-concentric circle at that angle, using
                       target_radius_px if given else the self-calibrated
                       entry_radius_from_marks() of THIS frame. This is
                       the new, angle-general alternative to mark0/mark1.
      focus         -- Laplacian-variance focus score (higher = sharper)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    plug = find_plug_circle(gray)
    rec = {"ok": False, "occluded": True, "focus": focus_score(gray),
           "plug_cx": None, "plug_cy": None, "plug_r": None, "ring_outer_r": None,
           "mark0_cx": None, "mark0_cy": None, "mark0_r": None,
           "mark1_cx": None, "mark1_cy": None, "mark1_r": None,
           "mark0_r_from_plug": None, "mark0_theta_deg": None,
           "mark1_r_from_plug": None, "mark1_theta_deg": None,
           "target_cx": None, "target_cy": None, "target_radius_px": None,
           "n_marks": 0}
    if plug is None:
        return rec
    cx, cy, r_plug = plug
    r_outer = find_ring_outer_radius(gray, cx, cy, r_plug)
    marks = detect_entry_marks(img_bgr, cx, cy, r_outer)
    rec.update({"ok": True, "plug_cx": cx, "plug_cy": cy, "plug_r": r_plug,
                "ring_outer_r": r_outer, "n_marks": len(marks)})
    for i, m in enumerate(marks):
        rec["mark%d_cx" % i], rec["mark%d_cy" % i], rec["mark%d_r" % i] = m[0], m[1], m[2]
        r_fp, th_fp = polar_from_center(cx, cy, m[0], m[1])
        rec["mark%d_r_from_plug" % i], rec["mark%d_theta_deg" % i] = r_fp, th_fp
    ring_degenerate = r_outer > r_plug + RING_MAX_THICK - 5
    rec["occluded"] = ring_degenerate or len(marks) < 2
    if target_angle_deg is not None:
        r_use = target_radius_px
        if r_use is None and TARGET_RADIUS_MARK is not None:
            r_use = rec.get("mark%d_r_from_plug" % TARGET_RADIUS_MARK)
        if r_use is None:
            r_use = entry_radius_from_marks(rec)
        ang = target_angle_deg
        if TARGET_ANGLE_REL_MARK is not None:
            th_m = rec.get("mark%d_theta_deg" % TARGET_ANGLE_REL_MARK)
            ang = None if th_m is None else th_m + target_angle_deg
        if r_use is not None and ang is not None:
            tx, ty = point_on_plug_circle(cx, cy, r_use, ang)
            rec["target_cx"], rec["target_cy"], rec["target_radius_px"] = tx, ty, r_use
    return rec


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: mic_perception.py <image.jpg>")
        raise SystemExit(1)
    img = cv2.imread(path)
    rec = locate(img)
    import json
    print(json.dumps(rec, indent=2))
