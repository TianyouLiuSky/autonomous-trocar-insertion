#!/usr/bin/env python3
"""mic_extrinsic_calib.py -- fit the microscope-pixel -> robot-base-frame
mapping, and (once fit) turn a fresh microscope still into an entry-point
estimate the robot can actually use for RCM registration.

WHY this exists (see chat write-up, 2026-09-17): insertion DEPTH along the
RCM axis already comes from the robot's own joint encoders + forward
kinematics (robotTask.cpp::controlModeVRCM, insertiondepth = TipInEE.z()
corrected against VRCM) -- no vision needed there. What vision (and
specifically the microscope, which resolves the entry marks far better
than the D405) is actually needed for is making sure the RCM point is
REGISTERED at the true anatomical entry point in the first place. That is
a ONE-TIME, offline, per-session calibration problem, not a real-time
tracking problem -- so it does not require a live video feed, a capture
card, or solving stereo/PnP depth for the needle. It only requires:

  1. Touch the tool tip to >=3 non-collinear, visually-unambiguous points
     on the (stationary) phantom, logging cur_tip() each time -- use the
     '1'-'8' keys in home_to_rcm_insert_v2_precision.py ('1'-'4' added
     2026-09-17, widened to '1'-'8' on 2026-09-24, see point 3b below for
     why). This writes MIC_CALIB_LOG (mic_calib_touches.jsonl).
  2. Take ONE clean (unoccluded, in-focus) microscope still of the same
     scene and click those same points, in the same order, with
     click_calib_points.py --n-points 8 (or however many you logged).
  3. Run this script: it fits a 2D affine px->base-XY map (the phantom
     surface is locally planar and the microscope views it ~orthogonally
     from directly overhead, so a full 6-DoF hand-eye/PnP solve is
     unnecessary overkill here -- an affine fit is the right-sized tool).
     3a. It reports the IN-SAMPLE fit residual, same as before -- but with
         only 4 points (6-DOF model, 8 equations) that residual has only 2
         DOF left and is nearly guaranteed to look small by construction,
         NOT because the map is accurate elsewhere in the field of view.
         (This is the exact "low residual != accuracy" trap the ISMR gap
         analysis documents for pivot calibration -- it applies here too.)
     3b. Use >=5 (ideally 8) points and this script ALSO runs leave-one-
         out cross-validation automatically: refit on every point but one,
         predict the held-out point, repeat for each point. `loo_rms_mm`
         (saved into the .npz) is the honest, non-self-referential
         accuracy number -- report THIS one, not the in-sample residual.
  4. From then on, `pixel_to_base_xy()` (or the CLI's `--still` mode, which
     chains mic_perception.locate() -> this mapping) turns any fresh,
     clean microscope still of the entry marks into a base-frame (X, Y)
     for RCM registration. Z is the calibration-plane Z recorded in step 1
     (report the touch-point Z spread; if it's more than ~0.3-0.5mm, the
     phantom surface is curved enough that a single Z is not safe to
     assume, and this needs the plane replaced by a low-order surface fit
     -- the script warns about this automatically).
  5. Before trusting this for a real trial, run mic_independent_verify.py's
     gating cascade on any candidate lock, and (separately, before
     reporting any accuracy claim) run its --campaign mode against genuinely
     new touched points across >=2 sessions -- see
     MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md. loo_rms_mm here is a
     same-session estimate; it is not a substitute for that campaign.

Usage:
    # after collecting >=5 touches (any subset of calib_pt_1..8, 8
    # recommended) and one click_calib_points.py run on a clean still:
    python3 mic_extrinsic_calib.py --fit \\
        --touches mic_calib_touches.jsonl \\
        --clicks /path/to/still.jpg.calib_px.json \\
        --out mic_extrinsic_calib.npz
    # (leave-one-out runs automatically at n>=5; add --loo to force/highlight it)

    # later, turn a fresh still into a base-frame entry point, and pick
    # which of the two ink-marked entry points (0 or 1) this trial targets --
    # add --mark to ALSO write mic_entry_point.json, which
    # home_to_rcm_insert_v2_precision.py's new 'e' key reads directly:
    python3 mic_extrinsic_calib.py --still /path/to/fresh_still.jpg \\
        --calib mic_extrinsic_calib.npz --mark 0

See /home/eyebrp/Matthew/microscope_autonomy/README.md for the full,
step-by-step lab protocol tying all of this (plus the 'e' key) together.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from runtime_session import ROOT, data_path
import argparse
import json
import os
import collections
import numpy as np


def load_touches(path):
    by_label = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_label[rec["label"]].append(np.array(rec["cur_tip_base_mm"], np.float64))
    out = {}
    for label, pts in by_label.items():
        arr = np.array(pts)
        out[label] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "n": len(arr),
        }
    return out


def fit_affine(px_uv, base_xyz):
    """Least-squares fit of [X, Y, Z] = A @ [u, v, 1]. Returns 3x3 matrix A
    and per-point residual norms (mm).

    CHANGED 2026-10-01 (pt.1): this used to fit only [X, Y] and store a
    single constant Z (the mean of the touched points' Z, "z_plane_mm"),
    which assumes the touched points are co-planar at one height -- the
    module docstring and do_fit() below already warned "if Z spread >
    0.4mm ... extend this script to fit a tilted plane (z = f(u,v))
    instead of a constant" (point 4 of the module docstring, and the
    do_fit warning text). The user confirmed in person (2026-10-01) that
    on this phantom's curved surface, touching 8 points that are BOTH
    well-spread across the field of view AND co-planar at one Z is hard
    to do reliably -- exactly the failure mode this comment anticipated.
    Fix: fit Z the same way as X and Y (one more column in the same
    least-squares solve, not a separate model) using the ALREADY-RECORDED
    full (X, Y, Z) touch position -- cur_tip_base_mm in
    mic_calib_touches.jsonl was always 3D, pixel_to_base_xy() always
    returned 3 values; only the fit itself was discarding the real Z and
    substituting a constant. No touch-collection protocol change is
    required: touch points wherever they naturally land on the real
    surface, do not try to hold a constant height.
    """
    n = len(px_uv)
    M = np.hstack([np.array(px_uv, np.float64), np.ones((n, 1))])  # n x 3
    Y = np.array(base_xyz, np.float64)                             # n x 3
    A, *_ = np.linalg.lstsq(M, Y, rcond=None)   # 3x3
    pred = M.dot(A)                              # n x 3
    resid = np.linalg.norm(pred - Y, axis=1)
    return A.T, resid   # A.T is 3x3: [X;Y;Z] = A.T @ [u,v,1]


def leave_one_out_errors(px_uv, base_xyz):
    """The honest accuracy number this calibration is missing without it:
    for each point, refit the affine map on every OTHER point and predict
    this one, so every reported error comes from a point that was never
    used to determine the map that predicts it -- unlike the in-sample
    residual in do_fit(), which is computed on the same points used to fit
    A and is therefore self-referential (see module docstring pt.3 and the
    ISMR gap-analysis writeup: "low residual does not guarantee accuracy").

    CHANGED 2026-10-01 (pt.1): errors are now full 3D (X, Y, AND Z), since
    fit_affine above now predicts Z instead of assuming a constant --
    read as "3D entry-point error", not "2D registration-plane error" as
    before. With a 9-DOF affine model (3x3), n=4 points leaves 0 points to
    fit a valid map for each held-out point once you remove one (3 remain
    < the 3-point minimum this script already enforces) -- so LOO only
    starts being meaningful at n>=5, and is not well-conditioned until
    n>=6-8 (now MORE true than before: 9 free params instead of 6, so
    prefer 8 points, not just >=5, before trusting this). Returns
    errors_mm array (order-matched externally by the caller) or raises if
    n is too small.
    """
    n = len(px_uv)
    if n < 5:
        raise ValueError("need >=5 points for leave-one-out (have %d) -- log more "
                          "'1'-'8' touches on the robot + re-click them" % n)
    px_uv = np.array(px_uv, np.float64)
    base_xyz = np.array(base_xyz, np.float64)
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        A_loo, _ = fit_affine(px_uv[mask], base_xyz[mask])
        pred = A_loo.dot(np.array([px_uv[i, 0], px_uv[i, 1], 1.0]))
        errors[i] = float(np.linalg.norm(pred - base_xyz[i]))
    return errors


def bootstrap_ci_mean(values, n_resamples=2000, seed=20260924):
    """Nonparametric bootstrap 95% CI on the mean of `values` -- same
    discipline as the bootstrap CIs already used for C1/C3 in the ISMR gap
    analysis, applied here to the LOO error distribution."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values, np.float64)
    n = len(values)
    means = np.array([rng.choice(values, n, replace=True).mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def tangent_frame_from_normal(normal, ref_axis=None):
    """2026-10-05: deterministic orthonormal in-plane basis (u, v) for the
    tangent plane perpendicular to `normal` (unit 3-vector, BASE frame,
    e.g. PHANTOM_NORMAL["normal"] from phantom_normal_estimate.py).

    WHY THIS EXISTS: the ring->entry offset learned by
    mic_ring_offset_verify.py's --fit-circle (home_to_rcm_insert_v2_precision.py's
    'x' key in MIC_RCM_MODE=circle) is only phantom-pose-INVARIANT if it is
    expressed as (radius, angle) WITHIN this tangent frame, not as a raw
    base-frame XYZ vector (see that offset's original fixed-vector form's
    documented limitation: "only as good as ... the phantom not having
    moved since that offset was learned"). A radius/angle pair means
    nothing, though, unless "angle=0" is pinned to the SAME physical
    direction every time it's used -- this function IS that pin. It MUST be
    called with the exact same convention at fit-time (mic_ring_offset_verify.py)
    and use-time (home_to_rcm_insert_v2_precision.py) or the learned angle
    silently means something different each time, i.e. this function being
    imported from one shared place (not copy-pasted into both files) is a
    deliberate guard against that exact drift bug -- see this project's
    history of two separate MIC_EXTRINSIC_CALIB_NPZ/RING_TO_ENTRY_OFFSET_NPZ
    default-path copy-paste bugs for why that caution is not theoretical.

    Default ref_axis = global Z ([0,0,1]) -- the robot BASE frame's Z is
    fixed regardless of how the phantom itself happens to be mounted/
    rotated on any given day, so projecting it onto the tangent plane gives
    a repeatable "angle=0" direction without needing any extra calibration
    step. Falls back to global X if normal is too close to parallel with Z
    (projection would be numerically unstable / ill-defined)."""
    normal = np.asarray(normal, np.float64)
    normal = normal / np.linalg.norm(normal)
    if ref_axis is None:
        ref_axis = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref_axis, normal)) > 0.95:
            ref_axis = np.array([1.0, 0.0, 0.0])
    else:
        ref_axis = np.asarray(ref_axis, np.float64)
    u = ref_axis - np.dot(ref_axis, normal) * normal
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-9:
        raise ValueError("ref_axis parallel (or too close) to normal -- cannot build a stable "
                          "tangent frame; pass an explicit ref_axis not aligned with normal")
    u = u / u_norm
    v = np.cross(normal, u)
    return u, v


def do_fit(args):
    touches = load_touches(args.touches)
    clicks = json.load(open(args.clicks))
    labels = [k for k in clicks.keys() if not k.startswith("_")]

    common = [lb for lb in labels if lb in touches]
    missing = [lb for lb in labels if lb not in touches]
    if missing:
        print("WARNING: clicked labels with no matching robot touch (skipped):", missing)
    if len(common) < 3:
        raise SystemExit("need >=3 matched points to fit an affine map, got %d (%s). "
                          "Log more '1'-'8' touches on the robot." % (len(common), common))

    px_uv, base_xyz, labels_used = [], [], []
    for lb in common:
        px_uv.append([clicks[lb]["u"], clicks[lb]["v"]])
        m = touches[lb]["mean"]
        base_xyz.append([m[0], m[1], m[2]])
        labels_used.append(lb)
        if touches[lb]["n"] > 1 and np.linalg.norm(touches[lb]["std"]) > 0.15:
            print("  NOTE: '%s' has %d touches with std=%.3fmm -- consider re-touching more "
                  "carefully / averaging more samples" % (lb, touches[lb]["n"], np.linalg.norm(touches[lb]["std"])))

    A, resid = fit_affine(px_uv, base_xyz)
    base_z = [p[2] for p in base_xyz]
    z_arr = np.array(base_z)
    z_mean, z_spread = float(z_arr.mean()), float(z_arr.max() - z_arr.min())

    print("\n=== fit report (IN-SAMPLE -- see LOO section below before trusting this) ===")
    for lb, r, z in zip(labels_used, resid, base_z):
        print("  %-14s residual=%.3fmm  touch_z=%.2fmm" % (lb, r, z))
    n_pts = len(common)
    dof = 3 * n_pts - 9  # 3x3 affine (X,Y,Z) = 9 free params, 3 residual eqns/point
    print("  RMS residual: %.3fmm  (n=%d points, %d residual DOF)" % (np.sqrt((resid ** 2).mean()), n_pts, dof))
    if dof <= 4:
        print("  ******************************************************************")
        print("  WARNING: only %d residual DOF left -- with this few points the fit" % dof)
        print("  can memorize noise and the in-sample residual above is NOT a")
        print("  trustworthy accuracy estimate (same trap as a low pivot-calibration")
        print("  residual under limited rotation -- see the ISMR gap analysis).")
        print("  This number describes how well A explains THESE %d points, not" % n_pts)
        print("  how accurate the map is anywhere else in the field of view.")
        print("  ******************************************************************")

    loo_rms_mm = None
    loo_errors = None
    if args.loo or n_pts >= 5:
        try:
            loo_errors = leave_one_out_errors(px_uv, base_xyz)
            loo_rms_mm = float(np.sqrt((loo_errors ** 2).mean()))
            lo, hi = bootstrap_ci_mean(loo_errors)
            print("\n=== leave-one-out cross-validation (HONEST accuracy estimate) ===")
            for lb, e in zip(labels_used, loo_errors):
                print("  %-14s held-out error=%.3fmm (full 3D X/Y/Z)" % (lb, e))
            print("  LOO RMS error: %.3fmm  (mean=%.3fmm  std=%.3fmm  max=%.3fmm)" %
                  (loo_rms_mm, loo_errors.mean(), loo_errors.std(), loo_errors.max()))
            print("  bootstrap 95%% CI on mean LOO error: [%.3f, %.3f] mm  (n_resamples=2000)" % (lo, hi))
            print("  --> USE loo_rms_mm AS THE CALIBRATION ACCURACY NUMBER, not the in-sample")
            print("      residual above. This is what mic_independent_verify.py's gating cascade")
            print("      reads from the saved .npz.")
            if n_pts < 8:
                print("  NOTE: n=%d points -- 8 is recommended (use all '1'-'8' keys) for a" % n_pts)
                print("        well-conditioned LOO estimate; treat this one as provisional.")
        except ValueError as e:
            print("\nNOTE: leave-one-out skipped -- %s" % e)
    else:
        print("\nNOTE: only %d points and --loo not requested -- run again with --loo once "
              "you have >=5 points if you want an honest (non-self-referential) accuracy "
              "number instead of the in-sample residual above." % n_pts)

    print("\n  touch-point Z spread: %.3fmm (mean Z=%.2fmm)" % (z_spread, z_mean))
    print("  (2026-10-01: Z spread is now informational only -- fit_affine() predicts Z "
          "per-point from (u,v) instead of assuming one constant, so a curved/sloped "
          "touch surface is no longer a problem. A large Z spread with a LARGE loo_rms_mm "
          "above would still mean the surface isn't well-approximated by one FLAT tilted "
          "plane either -- in that case the honest fix is more points, not a workaround.)")

    # loo_rms_mm=-1 (not NaN) so downstream code can do a cheap `>= 0` check
    # without importing numpy just to test for NaN; -1 unambiguously means
    # "no LOO was run for this calibration, do not trust it for gating".
    np.savez(args.out, A=A, z_plane_mm=z_mean, z_spread_mm=z_spread,
             labels_used=labels_used, resid_mm=resid, px_uv=px_uv, base_xy=np.array(base_xyz)[:, :2],
             base_xyz=base_xyz, model="affine3d_pt1_20261001",
             source_image=clicks.get("_source_image", ""),
             loo_rms_mm=(loo_rms_mm if loo_rms_mm is not None else -1.0),
             loo_errors_mm=(loo_errors if loo_errors is not None else np.array([])),
             n_calib_points=n_pts)
    print("\nSaved calibration -> %s" % args.out)
    if loo_rms_mm is None:
        print("  (loo_rms_mm=-1 in the .npz -- mic_independent_verify.py's gating cascade will "
              "refuse to pass any measurement made against this calibration until you re-fit "
              "with >=5 points)")


def pixel_to_base_xy(u, v, calib_npz_path):
    """Returns (X, Y, Z) mm in the robot base frame for one microscope
    pixel. CHANGED 2026-10-01 (pt.1): Z used to be a single constant
    (z["z_plane_mm"], the mean of the touch points' Z) regardless of
    (u, v) -- now A is a full 3x3 affine ([X;Y;Z] = A @ [u,v,1]) fit the
    same way as X and Y, so Z varies correctly across a curved/sloped
    touched surface instead of being forced flat. Still works on an
    OLDER 2x3-only .npz (if one exists from before this change) by
    falling back to the stored z_plane_mm for Z."""
    z = np.load(calib_npz_path)
    A = np.asarray(z["A"], np.float64)
    if A.shape == (3, 3):
        xyz = A.dot(np.array([u, v, 1.0]))
        return float(xyz[0]), float(xyz[1]), float(xyz[2])
    xy = A.dot(np.array([u, v, 1.0]))  # legacy 2x3 calibration
    return float(xy[0]), float(xy[1]), float(z["z_plane_mm"])


def do_still(args):
    import cv2
    import sys
    import time
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import mic_perception as mp

    img = cv2.imread(args.still)
    rec = mp.locate(img)
    print(json.dumps(rec, indent=2))
    if not rec["ok"]:
        raise SystemExit("plug/limbus not found in this still -- cannot compute entry point")
    if rec["occluded"]:
        print("WARNING: this frame is flagged occluded=True (instrument/blue occluder likely in "
              "frame, or fewer than 2 ink marks found) -- the entry-point estimate below may be "
              "unreliable. Prefer a cleaner still if one is available.")

    calib_rms = None
    try:
        z = np.load(args.calib)
        calib_rms = float(np.sqrt((z["resid_mm"] ** 2).mean()))
    except Exception:
        pass

    results = {}
    for i in range(2):
        cx, cy = rec.get("mark%d_cx" % i), rec.get("mark%d_cy" % i)
        if cx is None:
            continue
        X, Y, Z = pixel_to_base_xy(cx, cy, args.calib)
        results[i] = (X, Y, Z)
        print("mark%d (entry point %d) -> robot base frame: X=%.3f Y=%.3f Z=%.3f mm" % (i, i, X, Y, Z))

    if args.mark is not None:
        if args.mark not in results:
            raise SystemExit("--mark %d was requested but that mark was not found in this still "
                              "(found: %s)" % (args.mark, list(results.keys())))
        X, Y, Z = results[args.mark]
        out = {"X": X, "Y": Y, "Z": Z, "mark_index": args.mark,
               "calib_rms_mm": calib_rms, "occluded": rec["occluded"],
               "source_image": os.path.abspath(args.still), "timestamp": time.time()}
        with open(args.write_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("\nWrote %s -- press 'e' in home_to_rcm_insert_v2_precision.py to lock this as RCM "
              "(after confirming it looks sane!)" % args.write_json)


def do_ring_offset(args):
    """2026-09-21: learn a fixed base-frame vector from "wherever the D405's
    color-ring detector centers itself" ('z' key -- which, on THIS phantom,

    *** SUPERSEDED 2026-09-28 -- see mic_ring_offset_verify.py instead. ***
    This function fits from exactly ONE (ring, microscope) snapshot with
    ZERO held-out validation (self-referential accuracy, the same G1 trap
    the rest of this project's --loo/campaign machinery exists to avoid).
    mic_ring_offset_verify.py --fit does the same job from MULTIPLE paired
    samples (logged via the new '0' key in home_to_rcm_insert_v2_precision.py)
    with leave-one-out cross-validation and a bootstrap CI, and writes the
    identical .npz format the 'x' key reads. Left here, not removed, only
    because deleting working code that some old invocation might still
    reference is worse than a clearly-labeled superseded function.
    locks onto the LIMBUS, not an entry mark, since detect_ring() looks for
    a red/maroon annulus and the entry marks are small dark ink dots offset
    ~3-4mm posterior to the limbus -- see chat write-up 2026-09-21) to "where
    the microscope says the true entry mark actually is". Once learned, the
    control script's 'x' key adds this offset on top of the LIVE 'z' lock,
    so the D405-only APPROACH/HOVER target lands much closer to the real
    entry point than raw ring-center did -- while the microscope 'm'/'e'
    keys remain the authoritative, final correction right before insertion
    (this offset is only as good as "the phantom hasn't moved since it was
    learned", exactly like every other calibration in this project).

    Needs, at the SAME phantom pose:
      - a fresh 'z' lock on the robot (writes auto_target.npz -- this is
        what "ring_base" below reads)
      - one clean microscope still of the same scene (this is what mark0/1's
        base-frame positions get computed from, via the already-fit
        mic_extrinsic_calib.npz)
    """
    import cv2
    import sys
    import time
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import mic_perception as mp

    img = cv2.imread(args.still)
    rec = mp.locate(img)
    if not rec["ok"]:
        raise SystemExit("plug/limbus not found in this still -- cannot compute entry point")
    if rec["occluded"]:
        print("WARNING: this still is flagged occluded=True -- offset learned from it may be "
              "unreliable. Prefer a cleaner still if one is available.")

    ring = np.load(args.ring_npz)
    ring_base = np.array(ring["base_xyz"], np.float64)
    ring_age_s = time.time() - float(ring["t"])
    print("D405 ring lock ('z'): [%.2f %.2f %.2f]mm base-frame, %.0fs old, jitter=%.2fmm n=%d" %
          (ring_base[0], ring_base[1], ring_base[2], ring_age_s, float(ring["jitter_mm"]), int(ring["n"])))
    if ring_age_s > args.max_ring_age_s:
        print("WARNING: that ring lock is %.0fs old (>%.0fs limit) -- re-press 'z' on the robot "
              "for a FRESH lock (phantom/scope may have moved since) before trusting this offset" %
              (ring_age_s, args.max_ring_age_s))

    offsets = {}
    for i in range(2):
        cx, cy = rec.get("mark%d_cx" % i), rec.get("mark%d_cy" % i)
        if cx is None:
            continue
        X, Y, Z = pixel_to_base_xy(cx, cy, args.calib)
        entry_base = np.array([X, Y, Z], np.float64)
        offset = entry_base - ring_base
        offsets[i] = offset
        print("mark%d: entry=[%.2f %.2f %.2f]mm  offset-from-ring=[%.2f %.2f %.2f]mm  |offset|=%.2fmm" %
              (i, X, Y, Z, offset[0], offset[1], offset[2], np.linalg.norm(offset)))

    if not offsets:
        raise SystemExit("no ink marks found in this still -- cannot learn any offset")

    save = {"timestamp": time.time(), "source_image": os.path.abspath(args.still),
            "ring_npz_t": float(ring["t"]), "ring_base_xyz": ring_base}
    for i, off in offsets.items():
        save["offset_mark%d" % i] = off
    np.savez(args.ring_offset_out, **save)
    print("\nSaved -> %s" % args.ring_offset_out)
    print("home_to_rcm_insert_v2_precision.py's 'x' key will now add this offset on top of the "
          "LIVE 'z' ring lock. Still verify with 'm'/'e' before the actual insertion move --this "
          "is an approach/hover aid, not a replacement for the final microscope check.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--loo", action="store_true",
                     help="also run leave-one-out cross-validation (auto-runs anyway if "
                          "n>=5; this flag mainly forces the attempt / makes intent explicit)")
    ap.add_argument("--touches", default=data_path("mic_calib_touches.jsonl"))
    ap.add_argument("--clicks")
    ap.add_argument("--out", default=data_path("mic_extrinsic_calib.npz"))
    ap.add_argument("--still")
    ap.add_argument("--calib", default=data_path("mic_extrinsic_calib.npz"))
    ap.add_argument("--mark", type=int, default=None, choices=[0, 1],
                     help="which entry-point candidate (0 or 1) to write out for the robot to consume")
    ap.add_argument("--write-json", default=data_path("mic_entry_point.json"),
        help="path home_to_rcm_insert_v2_precision.py's 'e' key reads from")
    ap.add_argument("--ring-offset", action="store_true",
                     help="learn a ring-center->entry-point base-frame offset from a live 'z' "
                          "lock (auto_target.npz) + this --still + --calib, for the 'x' key")
    ap.add_argument("--ring-npz", default=data_path("auto_target.npz"),
        help="the D405 'z'-key ring lock artifact to learn the offset from")
    ap.add_argument("--ring-offset-out", default=data_path("ring_to_entry_offset.npz"))
    ap.add_argument("--max-ring-age-s", type=float, default=30.0)
    a = ap.parse_args()
    if a.fit:
        do_fit(a)
    elif a.ring_offset:
        do_ring_offset(a)
    elif a.still:
        do_still(a)
    else:
        print(__doc__)
