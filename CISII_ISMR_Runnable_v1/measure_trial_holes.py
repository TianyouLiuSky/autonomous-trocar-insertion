#!/usr/bin/env python3
"""Ground-truth entry error per trial from the post-RETRACT microscope still.

For each trial*.json in trial_plunge_stills/ (top level only; pilot/ and
excluded/ are ignored) the user clicks the puncture hole in the sharpest
post-retract still (or trialNN_*_manual.jpg if present, e.g. a re-shot after
re-seating a dragged phantom). Two errors are reported:

  err_phantom  hole vs the planned target re-derived IN THIS IMAGE from the
               phantom's own landmarks (plug/limbus centre, mark0 radius,
               mark0 angle + MIC_RCM_ANGLE_DEG). Valid even if the phantom
               moved after insertion -> primary metric.
  err_cmd      hole vs the commanded RCM target (rcm_target_base_mm mapped
               back to pixels through the microscope extrinsic A). Valid only
               if the phantom did not move between lock and this still
               (phantom_shift_mm reports that).

Radial error is along (target - limbus centre): negative = toward limbus.

Keys:  left-click = hole (in 'hole' mode)
       p = click 3 points on the plug/limbus edge (manual limbus)
       m = click mark0 centre (manual mark0)
       h = back to hole mode     r = reset clicks
       x = no hole visible (failed / unmeasurable)
       s / Enter = save & next   k = skip (no row)   q = quit
Usage:  python3 measure_trial_holes.py [--redo] [--summary]
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIC_DIR = str(ROOT / "microscope_autonomy")
sys.path.insert(0, MIC_DIR)
import mic_perception as mp  # noqa: E402

STILLS_DIR = data_path("trial_plunge_stills")
OUT_CSV = os.path.join(STILLS_DIR, "hole_measurements.csv")
EXTRINSIC = data_path('mic_extrinsic_calib.npz')
ANGLE_DEG = float(os.environ.get("MIC_RCM_ANGLE_DEG", "25"))
DISPLAY_SCALE = float(os.environ.get("MEASURE_DISPLAY_SCALE", "0.8"))
FIELDS = ["trial", "json", "image", "status", "hole_u", "hole_v",
          "target_u", "target_v", "target_src", "cmd_u", "cmd_v",
          "err_phantom_mm", "err_radial_mm", "err_tangential_mm", "err_cmd_mm",
          "phantom_shift_mm", "limbus_src", "mark0_src", "mm_per_px",
          "insert_angle_deg", "plunge_depth_mm", "tilt_deg", "yaw_diff_deg", "rcm_z_src",
          "target_angle_deg", "retract_adhesion"]

A = np.load(EXTRINSIC)["A"]
MM_PER_PX = float(np.sqrt(abs(np.linalg.det(A[:2, :2]))))


def base_xy_to_px(xy):
    return np.linalg.solve(A[:2, :2], np.asarray(xy[:2], float) - A[:2, 2])


def circle_from_3(pts):
    (x1, y1), (x2, y2), (x3, y3) = pts
    M = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]], float) * 2
    b = np.array([x2**2 - x1**2 + y2**2 - y1**2, x3**2 - x1**2 + y3**2 - y1**2], float)
    cx, cy = np.linalg.solve(M, b)
    return cx, cy, float(np.hypot(x1 - cx, y1 - cy))


def trial_images(jpath):
    stem = jpath[:-5]
    manual = sorted(glob.glob(stem + "_manual*.jpg"))
    if manual:
        return manual[-1]
    posts = sorted(glob.glob(stem + "_post*.jpg"))
    if not posts:
        return None
    return max(posts, key=lambda p: mp.focus_score(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)))


def load_done():
    if not os.path.isfile(OUT_CSV):
        return {}
    with open(OUT_CSV) as f:
        return {r["json"]: r for r in csv.DictReader(f)}


def write_rows(rows):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: int(r["trial"])):
            w.writerow({k: r.get(k, "") for k in FIELDS})


def measure_one(jpath):
    d = json.load(open(jpath))
    img_path = trial_images(jpath)
    if img_path is None:
        print("  no post-retract still for %s" % os.path.basename(jpath))
        return None
    img = cv2.imread(img_path)
    rec = mp.locate(img)
    freeze = d.get("mic_geo_freeze") or {}
    cmd_px = base_xy_to_px(d["rcm_target_base_mm"]) if d.get("rcm_target_base_mm") else None

    st = {"mode": "hole", "hole": None, "plug_pts": [], "mark0": None, "status": "ok"}
    win = "measure %s" % os.path.basename(jpath)

    def geometry():
        if len(st["plug_pts"]) == 3:
            cx, cy, _ = circle_from_3(st["plug_pts"]); lsrc = "manual"
        elif rec.get("plug_cx") is not None:
            cx, cy = rec["plug_cx"], rec["plug_cy"]; lsrc = "auto"
        else:
            return None
        if st["mark0"] is not None:
            m0 = st["mark0"]; msrc = "manual"
        elif rec.get("mark0_cx") is not None:
            m0 = (rec["mark0_cx"], rec["mark0_cy"]); msrc = "auto"
        else:
            return None
        r, th = mp.polar_from_center(cx, cy, m0[0], m0[1])
        ang = d.get("mic_rcm_angle_deg")
        tx, ty = mp.point_on_plug_circle(cx, cy, r, th + (ANGLE_DEG if ang is None else float(ang)))
        return dict(cx=cx, cy=cy, m0=m0, r=r, tx=tx, ty=ty, lsrc=lsrc, msrc=msrc)

    def on_mouse(ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        p = (x / DISPLAY_SCALE, y / DISPLAY_SCALE)
        if st["mode"] == "hole":
            st["hole"] = p
        elif st["mode"] == "plug":
            st["plug_pts"] = (st["plug_pts"] + [p])[-3:]
        elif st["mode"] == "mark0":
            st["mark0"] = p; st["mode"] = "hole"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = img.copy()
        g = geometry()
        if g:
            ic = lambda a, b: (int(round(a)), int(round(b)))
            cv2.circle(vis, ic(g["cx"], g["cy"]), int(g["r"]), (255, 255, 0), 1)
            cv2.drawMarker(vis, ic(g["cx"], g["cy"]), (255, 255, 0), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(vis, ic(*g["m0"]), 12, (0, 165, 255), 2)
            cv2.drawMarker(vis, ic(g["tx"], g["ty"]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 40, 2)
            cv2.circle(vis, ic(g["tx"], g["ty"]), int(round(1.0 / MM_PER_PX)), (0, 255, 0), 1)
        if cmd_px is not None:
            cv2.drawMarker(vis, (int(cmd_px[0]), int(cmd_px[1])), (255, 0, 255), cv2.MARKER_DIAMOND, 24, 2)
        for p in st["plug_pts"]:
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (255, 255, 0), -1)
        if st["hole"]:
            cv2.drawMarker(vis, (int(st["hole"][0]), int(st["hole"][1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        txt = "trial %s  mode=%s  [green=planned target(1mm circle) magenta=commanded red=hole]" % (d.get("trial"), st["mode"])
        cv2.putText(vis, txt, (20, 40), 0, 0.9, (0, 255, 255), 2)
        if g and st["hole"]:
            e = np.hypot(st["hole"][0] - g["tx"], st["hole"][1] - g["ty"]) * MM_PER_PX
            cv2.putText(vis, "err %.2f mm" % e, (20, 80), 0, 1.0, (0, 0, 255), 2)
        if g is None:
            cv2.putText(vis, "limbus/mark0 not detected: press p (3 edge clicks) and/or m", (20, 80), 0, 0.9, (0, 0, 255), 2)
        cv2.imshow(win, cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("p"): st["mode"] = "plug"; st["plug_pts"] = []
        elif k == ord("m"): st["mode"] = "mark0"
        elif k == ord("h"): st["mode"] = "hole"
        elif k == ord("r"): st.update(hole=None, plug_pts=[], mark0=None, mode="hole")
        elif k == ord("x"): st["status"] = "no_hole"; break
        elif k in (ord("s"), 13):
            if st["hole"] is None or g is None:
                print("  need a hole click and limbus+mark0 (auto or manual) before saving")
                continue
            break
        elif k == ord("k"): cv2.destroyWindow(win); return None
        elif k == ord("q"): cv2.destroyWindow(win); raise KeyboardInterrupt
    cv2.destroyWindow(win)

    pose = (d.get("last_correspondence") or {}).get("_pose") or {}
    row = {"trial": d.get("trial"), "json": os.path.basename(jpath), "image": os.path.basename(img_path),
           "status": st["status"], "mm_per_px": "%.5f" % MM_PER_PX,
           "insert_angle_deg": d.get("insert_angle_deg"), "plunge_depth_mm": d.get("plunge_depth_mm"),
           "tilt_deg": pose.get("tilt_deg", ""), "yaw_diff_deg": pose.get("yaw_diff_deg", ""),
           "rcm_z_src": pose.get("rcm_z_src", ""),
           "target_angle_deg": d.get("mic_rcm_angle_deg", ANGLE_DEG),
           "retract_adhesion": int(bool(d.get("retract_adhesion", False)))}
    if cmd_px is not None:
        row.update(cmd_u="%.1f" % cmd_px[0], cmd_v="%.1f" % cmd_px[1])
    if g is not None and freeze.get("cx") is not None:
        row["phantom_shift_mm"] = "%.2f" % (np.hypot(g["cx"] - freeze["cx"], g["cy"] - freeze["cy"]) * MM_PER_PX)
    if st["status"] == "ok":
        h = np.array(st["hole"]); t = np.array([g["tx"], g["ty"]]); c = np.array([g["cx"], g["cy"]])
        e = (h - t) * MM_PER_PX
        u_rad = (t - c) / np.linalg.norm(t - c)
        u_tan = np.array([-u_rad[1], u_rad[0]])
        row.update(hole_u="%.1f" % h[0], hole_v="%.1f" % h[1], target_u="%.1f" % t[0], target_v="%.1f" % t[1],
                   target_src="phantom_landmarks", limbus_src=g["lsrc"], mark0_src=g["msrc"],
                   err_phantom_mm="%.3f" % np.linalg.norm(e), err_radial_mm="%.3f" % e.dot(u_rad),
                   err_tangential_mm="%.3f" % e.dot(u_tan))
        if cmd_px is not None:
            row["err_cmd_mm"] = "%.3f" % (np.linalg.norm(h - cmd_px) * MM_PER_PX)
        vis = img.copy()
        cv2.drawMarker(vis, (int(t[0]), int(t[1])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 40, 2)
        cv2.circle(vis, (int(t[0]), int(t[1])), int(round(1.0 / MM_PER_PX)), (0, 255, 0), 1)
        cv2.drawMarker(vis, (int(h[0]), int(h[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.putText(vis, "trial %s  err %.2f mm" % (d.get("trial"), np.linalg.norm(e)), (20, 50), 0, 1.4, (0, 0, 255), 3)
        cv2.imwrite(jpath[:-5] + "_measured.jpg", vis)
    print("  trial %s: %s err_phantom=%s mm (radial %s, tangential %s) err_cmd=%s phantom_shift=%s" % (
        row["trial"], row["status"], row.get("err_phantom_mm"), row.get("err_radial_mm"),
        row.get("err_tangential_mm"), row.get("err_cmd_mm"), row.get("phantom_shift_mm")))
    return row


def summary(rows):
    ok = [r for r in rows.values() if r["status"] == "ok"]
    n = len(rows)
    print("\n=== %d trials measured, %d with visible hole, %d no hole ===" % (n, len(ok), n - len(ok)))
    print("retraction adhesion events: %d/%d" % (sum(int(r.get("retract_adhesion") or 0) for r in rows.values()), n))
    if not ok:
        return
    e = np.array([float(r["err_phantom_mm"]) for r in ok])
    rad = np.array([float(r["err_radial_mm"]) for r in ok])
    tan = np.array([float(r["err_tangential_mm"]) for r in ok])
    sd = e.std(ddof=1) if len(e) > 1 else 0.0
    print("entry error (phantom-relative): mean %.2f +- %.2f mm, median %.2f, RMS %.2f, max %.2f" % (
        e.mean(), sd, np.median(e), np.sqrt((e**2).mean()), e.max()))
    if len(e) > 1:
        print("  95%% CI of mean: [%.2f, %.2f] mm" % (e.mean() - 1.96 * sd / np.sqrt(len(e)), e.mean() + 1.96 * sd / np.sqrt(len(e))))
    print("  radial bias %.2f +- %.2f mm (negative = toward limbus), tangential %.2f +- %.2f mm" % (
        rad.mean(), rad.std(ddof=1) if len(rad) > 1 else 0, tan.mean(), tan.std(ddof=1) if len(tan) > 1 else 0))
    for thr in (0.5, 1.0, 1.5, 2.0):
        print("  within %.1f mm: %d/%d" % (thr, int((e <= thr).sum()), len(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true", help="re-measure trials already in the CSV")
    ap.add_argument("--summary", action="store_true", help="only print the summary")
    a = ap.parse_args()
    rows = load_done()
    if not a.summary:
        for jp in sorted(glob.glob(os.path.join(STILLS_DIR, "trial*.json"))):
            if os.path.basename(jp) in rows and not a.redo:
                continue
            print("measuring %s" % os.path.basename(jp))
            try:
                r = measure_one(jp)
            except KeyboardInterrupt:
                break
            if r is not None:
                rows[r["json"]] = r
                write_rows(rows)
    summary(rows)


if __name__ == "__main__":
    main()
