#!/usr/bin/env python3
"""mic_independent_verify.py -- the microscope-registration analogue of
measure_shaft_direction_3d() in home_to_rcm_insert_v2_precision.py (the
"C3" independent needle-axis check in the ISMR gap analysis).

WHY this exists (2026-09-24): the D405 vis3d check validates whether the
needle's ORIENTATION matches what was commanded, by measuring it with a
channel that shares no information with the kinematic chain. The
microscope path added on 2026-09-22 gives live video of the entry marks,
but until now nothing validated whether the ENTRY-POINT REGISTRATION it
produces (mic_extrinsic_calib.py's affine px->base-XY map) is actually
accurate -- the only existing signal was the in-sample fit residual on
the SAME 4 points used to fit the map, which is self-referential (G1 in
the ISMR gap analysis: "the standard quality figure ... is a residual of
the very objective that was minimised"). This module is the fix: a
pre-declared, non-self-referential gating cascade for any single
microscope-derived lock (mirrors the honesty conventions of
measure_shaft_direction_3d -- NEVER raises, cheapest/most-fundamental
checks first, reports the fail_reason, reports the MOST PESSIMISTIC
uncertainty estimate available), PLUS a campaign logger/report so a real
n>=10, >=2-session accuracy claim can be made on truly held-out points
that were structurally never available to the calibration fit (see
MIC_CAMPAIGN_LOG / the '9' key in home_to_rcm_insert_v2_precision.py).

This measures a DIFFERENT error source than C3: entry-point/registration
position (mm, in the base-frame XY plane + the flat-plane Z assumption),
not needle-shaft orientation (degrees). See
MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md for the full campaign protocol
this is meant to be run under.

Subcommands:
  --click STILL.jpg
      Interactively click ONE point on a still (e.g. a campaign touch
      point) and save STILL.jpg.campaign_click.json next to it.

  --gate STILL.jpg [STILL2.jpg ...] --calib mic_extrinsic_calib.npz --mark {0,1}
      Run the gating cascade on 1+ fresh stills of the SAME static scene
      (pass >=MIC_MIN_REPEAT_FRAMES of them to also exercise the
      repeatability gate) and print the verdict. Does not log anything.

  --campaign --label LABEL --click STILL.jpg.campaign_click.json
             --calib mic_extrinsic_calib.npz --touches mic_campaign_touches.jsonl
             --gate-stills STILL.jpg [STILL2.jpg ...] --mark {0,1}
             --session SESSION_ID [--log mic_campaign_log.csv]
      Look up LABEL's ground-truth robot-touched position in
      mic_campaign_touches.jsonl (written by the '9' key -- NEVER the
      calibration-fit touches file), run the gate on --gate-stills, and
      append one row (gate result + error vs ground truth, if the gate
      passed) to --log.

  --report [--log mic_campaign_log.csv]
      Print n, gate pass rate, mean/std/RMS error, bootstrap 95% CI, and a
      per-session breakdown -- the same statistical discipline already
      used for C1/C3 in the ISMR gap analysis.
"""
import argparse
import collections
import datetime
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mic_perception as mp  # noqa: E402
from mic_extrinsic_calib import bootstrap_ci_mean  # noqa: E402

# ---------------------------------------------------------------------------
# Gating thresholds -- env-var configurable, matching the VIS3D_* convention
# in home_to_rcm_insert_v2_precision.py. Defaults below are provisional
# (this system has zero real calibration data collected yet as of
# 2026-09-24 -- no mic_calib_touches.jsonl/.npz exists) -- RE-TUNE these
# against the first real campaign's data rather than trusting them blindly.
# ---------------------------------------------------------------------------
MIC_MIN_FOCUS = float(os.environ.get("MIC_MIN_FOCUS", "15.0"))
MIC_MAX_EXTRAPOLATION_MM = float(os.environ.get("MIC_MAX_EXTRAPOLATION_MM", "3.0"))
MIC_MAX_REPEAT_JITTER_MM = float(os.environ.get("MIC_MAX_REPEAT_JITTER_MM", "0.30"))
MIC_MIN_REPEAT_FRAMES = int(os.environ.get("MIC_MIN_REPEAT_FRAMES", "5"))

CAMPAIGN_LOG_FIELDS = [
    "timestamp_iso", "session", "label", "mark_index",
    "gate_ok", "fail_reason",
    "uncertainty_mm", "loo_rms_mm", "repeat_jitter_mm", "n_repeat_frames",
    "extrapolation_mm", "in_hull", "focus",
    # error vs ground truth -- ONLY populated when gate_ok, since an error
    # number from a measurement the gate itself rejected would misleadingly
    # suggest the gate's job is optional
    "err_3d_mm", "err_xy_mm", "err_z_mm",
    "gt_x_mm", "gt_y_mm", "gt_z_mm",
    "meas_x_mm", "meas_y_mm", "meas_z_mm",
    "calib_npz", "gate_stills",
]


def local_jacobian(A):
    """2x2 linear part of the affine map's X,Y rows -- exact local px->mm
    rate of change for the registration-plane (X, Y) coordinates (the map
    IS affine, so this is exact everywhere, not just locally -- named for
    clarity/parity with a general nonlinear map). Used only for the
    isotropic hull-extrapolation-distance-to-mm conversion below, which is
    explicitly an XY-plane gate, not a Z gate.

    CHANGED 2026-10-01 (pt.1): A can now be either a legacy 2x3 (X,Y only,
    constant Z stored separately) or the current 3x3 (X,Y,Z all fit from
    (u,v)) map produced by mic_extrinsic_calib.py. Either way, the first
    two rows are always X and Y, so slicing [:2, :2] is correct for both
    shapes."""
    return np.asarray(A, np.float64)[:2, :2]


def px_to_mm_distance(A, du, dv):
    d_mm = local_jacobian(A).dot(np.array([du, dv], np.float64))
    return float(np.linalg.norm(d_mm))


def hull_extrapolation_mm(u, v, calib_px_uv, A):
    """True if (u,v) is inside (or on) the convex hull of the calibration
    click points; if outside, also return the extrapolation distance
    converted to mm via the affine map's local Jacobian (isotropic
    approximation of the true anisotropic distance -- fine for a go/no-go
    gate, not fine for reporting a directional error).
    """
    pts = np.array(calib_px_uv, np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    dist_px = cv2.pointPolygonTest(hull, (float(u), float(v)), True)
    if dist_px >= 0:
        return True, 0.0
    extrap_mm = px_to_mm_distance(A, abs(dist_px), 0.0)
    return False, extrap_mm


def repeatability_jitter_mm(recs, mark_index, A):
    """Base-frame RMS spread of the mark centroid across repeat frames of
    the SAME static scene -- the microscope-side analogue of C3's
    half-split stability gate: tests whether the DETECTION reproduces,
    independent of whether the fitted MAP is accurate. Returns
    (n_usable_frames, jitter_mm or None if <2 usable frames)."""
    pts_mm = []
    for r in recs:
        cx, cy = r.get("mark%d_cx" % mark_index), r.get("mark%d_cy" % mark_index)
        if cx is None or r.get("occluded"):
            continue
        xy = np.asarray(A, np.float64).dot(np.array([cx, cy, 1.0]))
        pts_mm.append(xy)
    if len(pts_mm) < 2:
        return len(pts_mm), None
    pts_mm = np.array(pts_mm, np.float64)
    jitter_mm = float(np.linalg.norm(pts_mm.std(axis=0)))
    return len(pts_mm), jitter_mm


def load_calib(calib_npz_path):
    calib = np.load(calib_npz_path, allow_pickle=True)
    loo_rms_mm = float(calib["loo_rms_mm"]) if "loo_rms_mm" in calib.files else -1.0
    return calib, loo_rms_mm


def verify_microscope_lock(still_paths, calib_npz_path, mark_index=0):
    """The gating cascade. `still_paths`: list of >=1 fresh microscope
    stills of the SAME static scene (pass >=MIC_MIN_REPEAT_FRAMES for the
    repeatability gate to run). NEVER raises -- any failure to load/decode
    comes back as ok=False with a fail_reason, exactly like
    measure_shaft_direction_3d(). Order of checks: cheapest / most
    fundamental first (mirrors that function's own comment on this)."""
    try:
        calib, loo_rms_mm = load_calib(calib_npz_path)
    except Exception as e:
        return {"ok": False, "fail_reason": "could not load %s: %s" % (calib_npz_path, e)}

    if loo_rms_mm < 0:
        return {"ok": False, "fail_reason":
                "%s has no leave-one-out validation (loo_rms_mm=-1, saved when the "
                "calibration was fit with <5 points) -- refit with mic_extrinsic_calib.py "
                "--fit --loo using >=5 (ideally 8) '1'-'8' touch points before this "
                "calibration can be gated at all. An unvalidated map is not eligible "
                "regardless of how any single measurement looks." % calib_npz_path}

    if not still_paths:
        return {"ok": False, "fail_reason": "no stills given"}

    recs = []
    for p in still_paths:
        img = cv2.imread(p)
        if img is None:
            return {"ok": False, "fail_reason": "could not read %s" % p}
        recs.append(mp.locate(img))

    rec0 = recs[0]
    if not rec0["ok"] or rec0["occluded"]:
        return {"ok": False, "fail_reason":
                "plug/limbus not found, or frame flagged occluded, in %s (n_marks=%d)" %
                (os.path.basename(still_paths[0]), rec0["n_marks"])}

    if rec0["focus"] < MIC_MIN_FOCUS:
        return {"ok": False, "fail_reason":
                "focus score %.1f < %.1f in %s -- too blurry to trust the centroid "
                "(re-tune MIC_MIN_FOCUS against real in-focus/out-of-focus examples "
                "once you have some -- this default is provisional)" %
                (rec0["focus"], MIC_MIN_FOCUS, os.path.basename(still_paths[0]))}

    cx, cy = rec0.get("mark%d_cx" % mark_index), rec0.get("mark%d_cy" % mark_index)
    if cx is None:
        return {"ok": False, "fail_reason":
                "mark%d not found in %s (n_marks=%d)" %
                (mark_index, os.path.basename(still_paths[0]), rec0["n_marks"])}

    A = calib["A"]
    calib_px_uv = calib["px_uv"]
    in_hull, extrap_mm = hull_extrapolation_mm(cx, cy, calib_px_uv, A)
    if not in_hull and extrap_mm > MIC_MAX_EXTRAPOLATION_MM:
        return {"ok": False, "extrapolation_mm": extrap_mm, "in_hull": in_hull,
                "fail_reason":
                "mark%d pixel (%.0f,%.0f) is ~%.2fmm outside the calibration hull "
                "(>%.2fmm limit) -- an affine map is not trustworthy this far outside "
                "the points it was fit on; touch a calibration point nearer this "
                "location, or don't trust this lock" %
                (mark_index, cx, cy, extrap_mm, MIC_MAX_EXTRAPOLATION_MM)}

    n_repeat, jitter_mm = (1, None) if len(recs) < 2 else repeatability_jitter_mm(recs, mark_index, A)
    repeatability_checked = len(recs) >= MIC_MIN_REPEAT_FRAMES
    if repeatability_checked:
        if jitter_mm is None:
            return {"ok": False, "fail_reason":
                    "repeatability check requested (%d stills given) but mark%d was found "
                    "in <2 of them -- detection is not even consistently present, let alone "
                    "reproducible" % (len(recs), mark_index)}
        if jitter_mm > MIC_MAX_REPEAT_JITTER_MM:
            return {"ok": False, "repeat_jitter_mm": jitter_mm, "n_repeat_frames": n_repeat,
                    "fail_reason":
                    "repeat-frame jitter %.3fmm > %.3fmm across %d frames of the same "
                    "static scene -- centroid detection is not reproducible right now "
                    "(lighting/focus/instrument flicker?), don't trust a single-frame lock" %
                    (jitter_mm, MIC_MAX_REPEAT_JITTER_MM, n_repeat)}

    # 2026-10-01 (pt.1): A is now a 3x3 affine that predicts Z from (u,v)
    # too (mic_extrinsic_calib.py no longer assumes a single constant
    # z_plane_mm across the touched surface) -- fall back to the old
    # 2x3-plus-constant-Z behaviour only for a calibration fit before
    # this change.
    out = np.asarray(A, np.float64).dot(np.array([cx, cy, 1.0]))
    if out.shape[0] == 3:
        position_mm = out
    else:
        position_mm = np.array([out[0], out[1], float(calib["z_plane_mm"])], np.float64)

    # Report the MOST PESSIMISTIC available uncertainty estimate -- same
    # principle as measure_shaft_direction_3d()'s
    # max(analytic_unc_deg, stability_deg). loo_rms_mm is always available
    # (gated above); jitter_mm/extrap_mm may not be (repeatability wasn't
    # requested, or the point was safely inside the hull).
    uncertainty_mm = max(loo_rms_mm, jitter_mm or 0.0, extrap_mm)

    return {
        "ok": True,
        "fail_reason": None,
        "position_base_mm": position_mm.tolist(),
        "mark_index": mark_index,
        "uncertainty_mm": uncertainty_mm,
        "loo_rms_mm": loo_rms_mm,
        "repeat_jitter_mm": jitter_mm,
        "n_repeat_frames": n_repeat,
        "repeatability_checked": repeatability_checked,
        "extrapolation_mm": extrap_mm,
        "in_hull": in_hull,
        "focus": rec0["focus"],
        "n_stills": len(still_paths),
    }


# ---------------------------------------------------------------------------
# --click : single-point manual click tool for a campaign still
# ---------------------------------------------------------------------------
def do_click(args):
    img = cv2.imread(args.click)
    if img is None:
        raise SystemExit("could not read %s" % args.click)
    win = "mic_independent_verify --click (LMB=place  u=undo  s=save  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    pt = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[0] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    print("Click the ONE point this still is meant to verify (e.g. the campaign-touched "
          "landmark), then press 's' to save, or 'q' to quit without saving.")
    while True:
        view = img.copy()
        if pt[0] is not None:
            cv2.drawMarker(view, pt[0], (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, "s=save q=quit  (%s)" % ("point placed" if pt[0] else "click a point"),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):
            print("quit without saving")
            break
        if k == ord('s'):
            if pt[0] is None:
                print("  click a point first")
                continue
            out = {"u": pt[0][0], "v": pt[0][1], "source_image": os.path.abspath(args.click)}
            out_path = args.click + ".campaign_click.json"
            with open(out_path, "w") as fh:
                json.dump(out, fh, indent=2)
            print("saved -> %s" % out_path)
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# --gate : run the cascade on 1+ stills and print the verdict
# ---------------------------------------------------------------------------
def do_gate(args):
    res = verify_microscope_lock(args.gate, args.calib, mark_index=args.mark)
    print(json.dumps(res, indent=2))
    if res["ok"]:
        print("\nVERDICT: PASS -- position=[%.2f %.2f %.2f]mm base  uncertainty=+/-%.3fmm" %
              tuple(res["position_base_mm"]) + (res["uncertainty_mm"],))
    else:
        print("\nVERDICT: FAIL -- %s" % res["fail_reason"])


# ---------------------------------------------------------------------------
# --campaign : compare a gated measurement against a held-out ground-truth
# touch point, append one row to the campaign log
# ---------------------------------------------------------------------------
def load_campaign_touch(touches_path, label):
    with open(touches_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") == label:
                return np.array(rec["cur_tip_base_mm"], np.float64)
    return None


def do_campaign(args):
    gt = load_campaign_touch(args.touches, args.label)
    if gt is None:
        raise SystemExit("label '%s' not found in %s -- press '9' in "
                          "home_to_rcm_insert_v2_precision.py first" % (args.label, args.touches))

    res = verify_microscope_lock(args.gate_stills, args.calib, mark_index=args.mark)

    row = {k: "" for k in CAMPAIGN_LOG_FIELDS}
    row.update({
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "session": args.session,
        "label": args.label,
        "mark_index": args.mark,
        "gate_ok": res["ok"],
        "fail_reason": res.get("fail_reason") or "",
        "uncertainty_mm": res.get("uncertainty_mm", ""),
        "loo_rms_mm": res.get("loo_rms_mm", ""),
        "repeat_jitter_mm": res.get("repeat_jitter_mm", "") if res.get("repeat_jitter_mm") is not None else "",
        "n_repeat_frames": res.get("n_repeat_frames", ""),
        "extrapolation_mm": res.get("extrapolation_mm", ""),
        "in_hull": res.get("in_hull", ""),
        "focus": res.get("focus", ""),
        "gt_x_mm": gt[0], "gt_y_mm": gt[1], "gt_z_mm": gt[2],
        "calib_npz": os.path.abspath(args.calib),
        "gate_stills": ";".join(os.path.basename(s) for s in args.gate_stills),
    })

    if res["ok"]:
        meas = np.array(res["position_base_mm"], np.float64)
        err_xyz = meas - gt
        row.update({
            "meas_x_mm": meas[0], "meas_y_mm": meas[1], "meas_z_mm": meas[2],
            "err_3d_mm": float(np.linalg.norm(err_xyz)),
            "err_xy_mm": float(np.linalg.norm(err_xyz[:2])),
            "err_z_mm": float(err_xyz[2]),
        })
        print("GATE: PASS  measured=[%.2f %.2f %.2f]  gt=[%.2f %.2f %.2f]  "
              "err_3d=%.3fmm (xy=%.3fmm, z=%.3fmm)  uncertainty=+/-%.3fmm" %
              (meas[0], meas[1], meas[2], gt[0], gt[1], gt[2],
               row["err_3d_mm"], row["err_xy_mm"], row["err_z_mm"], res["uncertainty_mm"]))
    else:
        print("GATE: FAIL -- %s\n(logged as a gate failure -- no error number is computed "
              "for a measurement the gate itself rejected)" % res["fail_reason"])

    file_exists = os.path.isfile(args.log)
    import csv
    with open(args.log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPAIGN_LOG_FIELDS, restval="", extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print("-> appended to %s" % args.log)


# ---------------------------------------------------------------------------
# --report : stats over the campaign log
# ---------------------------------------------------------------------------
def do_report(args):
    import csv
    if not os.path.isfile(args.log):
        raise SystemExit("no campaign log at %s yet -- run --campaign first" % args.log)
    rows = list(csv.DictReader(open(args.log)))
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["gate_ok"] in ("True", "true", "1"))
    print("=== mic_independent_verify campaign report: %s ===" % args.log)
    print("total attempts: %d   gate PASS: %d (%.0f%%)   gate FAIL: %d (%.0f%%)" %
          (n_total, n_pass, 100.0 * n_pass / max(n_total, 1),
           n_total - n_pass, 100.0 * (n_total - n_pass) / max(n_total, 1)))
    if n_total - n_pass:
        reasons = collections.Counter(r["fail_reason"].split(" -- ")[0] for r in rows if r["gate_ok"] not in ("True", "true", "1"))
        print("\nfail reasons (first clause):")
        for reason, cnt in reasons.most_common():
            print("  %3d x  %s" % (cnt, reason))

    passed = [r for r in rows if r["gate_ok"] in ("True", "true", "1") and r["err_3d_mm"]]
    if not passed:
        print("\nNo gate-passing rows with a logged error yet -- nothing to report on accuracy.")
        return

    err3d = np.array([float(r["err_3d_mm"]) for r in passed])
    errxy = np.array([float(r["err_xy_mm"]) for r in passed])
    errz = np.array([float(r["err_z_mm"]) for r in passed])
    n = len(passed)
    print("\n=== accuracy on %d GATE-PASSING held-out points ===" % n)
    print("  err_3d_mm: mean=%.3f std=%.3f rms=%.3f max=%.3f" %
          (err3d.mean(), err3d.std(), np.sqrt((err3d ** 2).mean()), err3d.max()))
    print("  err_xy_mm: mean=%.3f std=%.3f rms=%.3f max=%.3f  (registration-plane accuracy)" %
          (errxy.mean(), errxy.std(), np.sqrt((errxy ** 2).mean()), errxy.max()))
    print("  err_z_mm:  mean=%.3f std=%.3f rms=%.3f max=%.3f  (flat-plane-assumption error --"
          " large values here mean z_plane_mm is not safe for this point)" %
          (errz.mean(), errz.std(), np.sqrt((errz ** 2).mean()), errz.max()))
    if n >= 5:
        lo, hi = bootstrap_ci_mean(err3d)
        print("  bootstrap 95%% CI on mean err_3d_mm: [%.3f, %.3f]  (n_resamples=2000)" % (lo, hi))
    else:
        print("  (n<5 -- skipping bootstrap CI, not enough data for it to mean anything)")

    sessions = collections.defaultdict(list)
    for r in passed:
        sessions[r["session"]].append(float(r["err_3d_mm"]))
    if len(sessions) > 1:
        print("\nper-session breakdown (does accuracy survive re-mounting the phantom/scope?):")
        for s, errs in sorted(sessions.items()):
            e = np.array(errs)
            print("  session '%-20s' n=%-3d mean_err_3d=%.3fmm std=%.3fmm" % (s, len(e), e.mean(), e.std()))
    elif n_total > 0:
        print("\nNOTE: all points are from a single session so far -- MIC_INDEPENDENT_"
              "VERIFICATION_PROTOCOL.md requires >=2 sessions (re-mount the phantom/scope "
              "in between) before this accuracy number means anything beyond "
              "'accurate under one specific unperturbed setup'.")

    if n < 10:
        print("\nNOTE: n=%d gate-passing points -- protocol target is n>=10. Keep running "
              "--campaign." % n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--click", metavar="STILL.jpg")
    ap.add_argument("--gate", nargs="+", metavar="STILL.jpg")
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--calib", default=os.path.join(HERE, "mic_extrinsic_calib.npz"))
    ap.add_argument("--mark", type=int, default=0, choices=[0, 1])
    ap.add_argument("--label")
    ap.add_argument("--touches", default=os.path.join(HERE, "mic_campaign_touches.jsonl"))
    ap.add_argument("--gate-stills", nargs="+", metavar="STILL.jpg")
    ap.add_argument("--session", default=time.strftime("session_%Y%m%d"))
    ap.add_argument("--log", default=os.path.join(HERE, "mic_campaign_log.csv"))
    args = ap.parse_args()

    if args.click:
        do_click(args)
    elif args.gate:
        do_gate(args)
    elif args.campaign:
        if not (args.label and args.gate_stills):
            raise SystemExit("--campaign needs --label and --gate-stills")
        do_campaign(args)
    elif args.report:
        do_report(args)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
