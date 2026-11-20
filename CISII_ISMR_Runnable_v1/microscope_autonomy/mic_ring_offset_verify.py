#!/usr/bin/env python3
"""mic_ring_offset_verify.py -- learn AND honestly validate the fixed
base-frame correction from "wherever the D405 ring-detector centers itself"
('z' key in home_to_rcm_insert_v2_precision.py -- on this phantom, the
LIMBUS, not an entry mark) to "where the microscope says the true ink-marked
entry point actually is" (the same authoritative source
mic_extrinsic_calib.py already uses for the RCM point itself).

WHY THIS FILE EXISTS (2026-09-28): mic_extrinsic_calib.py's `--ring-offset`
mode learns this correction from EXACTLY ONE (ring-lock, microscope-still)
snapshot and has ZERO held-out validation -- it is a single anecdotal
sample, not a calibration. That is the same "self-referential accuracy
number" trap the ISMR gap analysis flags for pivot calibration and that
mic_extrinsic_calib.py's own --loo mode was written to fix for the
px->base-XY affine map -- it had just never been fixed for THIS correction.
This script applies the identical discipline (multiple independent samples,
leave-one-out cross-validation once n>=5, bootstrap 95% CI, an explicit
"-1.0 = not yet validated" sentinel) to the ring->entry offset instead.

WORKFLOW
--------
1. One-time per lab session (unchanged): finish the microscope extrinsic
   calibration (mic_extrinsic_calib.py --fit --loo) so mic_extrinsic_calib.npz
   exists and pixel_to_base_xy() is trustworthy. This script refuses to run
   without it.
2. Repeatedly, ideally across >=2 sessions/re-mounts (mirrors the campaign
   discipline in MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md): with the D405
   ring LOCKED ('z' lit up / _ring_live["locked"]) and a fresh, clean
   microscope still available, press '0' in the control script
   (log_ring_offset_sample()). This appends ONE line to
   RING_OFFSET_SAMPLES_LOG (default: ring_offset_samples.jsonl) containing
   the live ring-lock base position AND (via mic_perception.locate() on the
   freshest microscope still, same perception path 'y'/'e' already trust)
   the true entry-mark base position(s), timestamped together. No manual
   pixel-clicking needed -- mic_perception's ink-mark detector is already
   the trusted, mature part of the pipeline; only the TOOL-TIP was ever
   unreliable to auto-detect (see mic_perception.py's own docstring).
3. Run this script:
       python3 mic_ring_offset_verify.py --fit \\
           --samples ring_offset_samples.jsonl \\
           --out ring_to_entry_offset.npz
   It reports, per mark (0 and 1 tracked separately -- they are two
   different physical entry points and there is no reason to assume the
   same offset applies to both):
     - in-sample residual (misleading alone, printed for completeness)
     - leave-one-out RMS error (the honest number, once n>=5)
     - bootstrap 95% CI on the mean LOO error
   and saves ring_to_entry_offset.npz in the SAME format
   home_to_rcm_insert_v2_precision.py's 'x' key already reads
   (offset_mark0/offset_mark1), PLUS loo_rms_mm_mark0/1 and
   n_samples_mark0/1 so a future gating script can refuse to trust an
   under-validated offset the same way mic_independent_verify.py's cascade
   already refuses an under-validated extrinsic fit (loo_rms_mm == -1.0).
4. `--report` on an existing .npz just re-prints its saved validation
   numbers without re-fitting anything -- use this to sanity-check what
   the 'x' key is CURRENTLY trusting.

This does not replace 'y'/'e' (fresh, authoritative microscope check right
before the actual insertion move) -- it only makes the D405-only
APPROACH/HOVER aid ('x') an honestly-validated correction instead of an
anecdotal one.
"""
import argparse
import json
import os
import sys
import time
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mic_extrinsic_calib import bootstrap_ci_mean, tangent_frame_from_normal  # reuse, don't duplicate


def load_paired_samples(path, mark):
    """Returns (ring_xyz Nx3, entry_xyz Nx3, meta list) for every logged
    sample that has BOTH a ring lock and this mark's entry-point base
    position. Skips samples flagged occluded=True for that mark (same
    caution mic_perception's callers already apply elsewhere)."""
    ring_pts, entry_pts, meta = [], [], []
    if not os.path.isfile(path):
        return ring_pts, entry_pts, meta
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = "mark%d_base_mm" % mark
            if rec.get(key) is None:
                continue
            if rec.get("occluded"):
                continue
            ring_pts.append(np.array(rec["ring_base_mm"], np.float64))
            entry_pts.append(np.array(rec[key], np.float64))
            meta.append(rec)
    return ring_pts, entry_pts, meta


def leave_one_out_offset_errors(ring_pts, entry_pts):
    """Same principle as mic_extrinsic_calib.py's leave_one_out_errors(),
    applied to a constant 3D offset instead of a 2x3 affine: for each
    sample, fit the offset (just the mean) on every OTHER sample, predict
    this one, so every reported error comes from a sample that never
    contributed to the offset that predicts it."""
    n = len(ring_pts)
    if n < 5:
        raise ValueError("need >=5 paired samples for leave-one-out (have %d) -- press "
                          "'0' more times (ideally across >=2 sessions)" % n)
    ring_pts = np.array(ring_pts, np.float64)
    entry_pts = np.array(entry_pts, np.float64)
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        offset_loo = (entry_pts[mask] - ring_pts[mask]).mean(axis=0)
        pred = ring_pts[i] + offset_loo
        errors[i] = float(np.linalg.norm(pred - entry_pts[i]))
    return errors


def fit_one_mark(ring_pts, entry_pts, mark, force_loo=False):
    """Returns a dict of everything worth saving/printing for one mark, or
    None if there are zero paired samples for it."""
    n = len(ring_pts)
    if n == 0:
        return None
    ring_pts = np.array(ring_pts, np.float64)
    entry_pts = np.array(entry_pts, np.float64)
    raw_offsets = entry_pts - ring_pts          # Nx3, one per sample
    offset = raw_offsets.mean(axis=0)
    in_sample_resid = np.linalg.norm(raw_offsets - offset, axis=1)  # mm, per sample

    # BASELINE, 2026-09-28: how far off is the raw, uncorrected D405 ring
    # lock from the true entry point, with NO offset applied at all? This
    # needs no fitting (nothing to leave out), so it's just the per-sample
    # |ring - entry| distance. Reported alongside the corrected LOO number
    # so "does this correction actually help" is answered with a number,
    # not assumed -- see protocol Step 4 point 4.
    uncorrected_errors = np.linalg.norm(ring_pts - entry_pts, axis=1)
    uncorrected_rms = float(np.sqrt((uncorrected_errors ** 2).mean()))

    print("\n=== mark%d: %d paired sample(s) ===" % (mark, n))
    print("  BASELINE (no correction): raw |ring - entry| RMS=%.3fmm  "
          "(mean=%.3fmm max=%.3fmm) -- this is what 'z' alone gives you" %
          (uncorrected_rms, uncorrected_errors.mean(), uncorrected_errors.max()))
    print("  offset = mean(entry - ring) = [%.3f %.3f %.3f]mm  (std=[%.3f %.3f %.3f]mm)" %
          (offset[0], offset[1], offset[2], *raw_offsets.std(axis=0)))
    print("  in-sample scatter around that mean: RMS=%.3fmm max=%.3fmm  "
          "(NOT an accuracy number -- see LOO below)" %
          (float(np.sqrt((in_sample_resid ** 2).mean())), float(in_sample_resid.max())))

    loo_rms_mm, loo_errors, ci = None, None, None
    if force_loo or n >= 5:
        try:
            loo_errors = leave_one_out_offset_errors(ring_pts, entry_pts)
            loo_rms_mm = float(np.sqrt((loo_errors ** 2).mean()))
            lo, hi = bootstrap_ci_mean(loo_errors)
            ci = (lo, hi)
            print("  --- leave-one-out cross-validation (HONEST accuracy estimate) ---")
            print("  LOO RMS error: %.3fmm  (mean=%.3fmm std=%.3fmm max=%.3fmm)" %
                  (loo_rms_mm, loo_errors.mean(), loo_errors.std(), loo_errors.max()))
            print("  bootstrap 95%% CI on mean LOO error: [%.3f, %.3f] mm  (n_resamples=2000)" %
                  (lo, hi))
            print("  --> this is what a future 'x'-key gate should check, NOT the in-sample "
                  "scatter above.")
            if loo_rms_mm < uncorrected_rms:
                print("  CORRECTION HELPS: %.3fmm -> %.3fmm (%.0f%% reduction vs. raw ring lock)" %
                      (uncorrected_rms, loo_rms_mm, 100.0 * (1 - loo_rms_mm / uncorrected_rms)))
            else:
                print("  *** CORRECTION DOES NOT HELP on held-out data (%.3fmm -> %.3fmm, WORSE) "
                      "-- report this negative result, don't keep using 'x' as-is ***" %
                      (uncorrected_rms, loo_rms_mm))
            if n < 8:
                print("  NOTE: n=%d -- keep pressing '0' across sessions; treat this as provisional." % n)
        except ValueError as e:
            print("  LOO skipped: %s" % e)
    else:
        print("  NOTE: only %d paired sample(s) -- need >=5 for an honest (non-self-referential) "
              "accuracy number. Every reported offset before that point is UNVALIDATED -- do not "
              "trust the 'x' key's approach/hover accuracy claim until this runs with >=5." % n)

    return {
        "mark": mark, "n": n, "offset": offset,
        "in_sample_rms_mm": float(np.sqrt((in_sample_resid ** 2).mean())),
        "uncorrected_rms_mm": uncorrected_rms,
        "loo_rms_mm": loo_rms_mm, "loo_errors_mm": loo_errors,
        "bootstrap_ci_mm": ci,
    }


def do_fit(args):
    results = {}
    for mark in (0, 1):
        ring_pts, entry_pts, meta = load_paired_samples(args.samples, mark)
        r = fit_one_mark(ring_pts, entry_pts, mark, force_loo=args.loo)
        if r is not None:
            results[mark] = r

    if not results:
        raise SystemExit("no paired (ring, entry) samples found in %s for either mark -- "
                          "press '0' in the control script at least once first" % args.samples)

    save = {"timestamp": time.time(), "source_samples": os.path.abspath(args.samples)}
    for mark, r in results.items():
        save["offset_mark%d" % mark] = r["offset"]
        # -1.0 sentinel (not NaN) so a downstream gate can do a cheap `>= 0`
        # check, same convention as mic_extrinsic_calib.py's loo_rms_mm.
        save["loo_rms_mm_mark%d" % mark] = r["loo_rms_mm"] if r["loo_rms_mm"] is not None else -1.0
        save["n_samples_mark%d" % mark] = r["n"]
        save["uncorrected_rms_mm_mark%d" % mark] = r["uncorrected_rms_mm"]
        if r["loo_errors_mm"] is not None:
            save["loo_errors_mm_mark%d" % mark] = r["loo_errors_mm"]
        if r["bootstrap_ci_mm"] is not None:
            save["bootstrap_ci_mm_mark%d" % mark] = np.array(r["bootstrap_ci_mm"])

    np.savez(args.out, **save)
    print("\nSaved -> %s" % args.out)
    print("home_to_rcm_insert_v2_precision.py's 'x' key reads offset_mark%d from this file "
          "(MIC_TARGET_MARK). It does not yet check loo_rms_mm itself -- read it back with "
          "--report before trusting 'x' for anything beyond a rough approach/hover aid." %
          int(os.environ.get("MIC_TARGET_MARK", "0")))
    for mark, r in results.items():
        if r["loo_rms_mm"] is None:
            print("  mark%d: loo_rms_mm=-1.0 (NOT VALIDATED -- need >=5 paired samples, have %d)" %
                  (mark, r["n"]))


def load_paired_samples_with_normal(path, mark):
    """2026-10-05: like load_paired_samples() above, but ALSO requires a
    logged phantom_normal_base (needed to build the per-sample tangent
    frame -- see tangent_frame_from_normal()). Samples predating the
    2026-10-05 log_ring_offset_sample() update (no normal field, e.g. all
    7 samples from the 2026-10-01 session) are silently skipped here --
    NOT an error, they just can't support --fit-circle, only the original
    --fit. This is why --fit-circle needs a FRESH batch of >=5 '0'-key
    samples before it can validate anything."""
    ring_pts, entry_pts, normals, meta = [], [], [], []
    if not os.path.isfile(path):
        return ring_pts, entry_pts, normals, meta
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = "mark%d_base_mm" % mark
            if rec.get(key) is None:
                continue
            if rec.get("occluded"):
                continue
            if rec.get("phantom_normal_base") is None:
                continue
            ring_pts.append(np.array(rec["ring_base_mm"], np.float64))
            entry_pts.append(np.array(rec[key], np.float64))
            normals.append(np.array(rec["phantom_normal_base"], np.float64))
            meta.append(rec)
    return ring_pts, entry_pts, normals, meta


def _local_components(ring_pts, entry_pts, normals):
    """Per-sample (delta_u, delta_v, delta_n) -- the (entry-ring) vector
    expressed in THAT SAMPLE's own tangent frame (built fresh from that
    sample's own logged normal, not a single shared/averaged normal -- the
    normal estimate itself can drift slightly re-measurement to
    re-measurement, and using each sample's own normal is what makes this
    exactly mirror the real use-time situation: a fresh ring lock + a
    fresh normal fit, with only the LEARNED local (radius/angle) carried
    over from calibration)."""
    n = len(ring_pts)
    local = np.zeros((n, 3))
    for i in range(n):
        u, v = tangent_frame_from_normal(normals[i])
        nrm = normals[i] / np.linalg.norm(normals[i])
        delta = entry_pts[i] - ring_pts[i]
        local[i] = [delta.dot(u), delta.dot(v), delta.dot(nrm)]
    return local


def leave_one_out_circle_errors(ring_pts, entry_pts, normals):
    """Same LOO principle as leave_one_out_offset_errors() above, but for
    the local-tangent-frame representation: for each held-out sample i,
    average the OTHER samples' local (u,v,n) components, then reconstruct
    the prediction for sample i using sample i's OWN tangent frame (its own
    ring lock + its own normal) -- exactly mirroring the real use-time
    situation of 'x' in MIC_RCM_MODE=circle."""
    n = len(ring_pts)
    if n < 5:
        raise ValueError("need >=5 paired (ring,entry,normal) samples for leave-one-out (have %d)"
                          " -- press '0' more times (ideally across >=2 sessions/re-mounts, with "
                          "phantom_normal_base now being logged -- see log_ring_offset_sample())" % n)
    ring_pts = np.array(ring_pts, np.float64)
    entry_pts = np.array(entry_pts, np.float64)
    local = _local_components(ring_pts, entry_pts, normals)
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mean_local = local[mask].mean(axis=0)
        u, v = tangent_frame_from_normal(normals[i])
        nrm = normals[i] / np.linalg.norm(normals[i])
        pred = ring_pts[i] + mean_local[0] * u + mean_local[1] * v + mean_local[2] * nrm
        errors[i] = float(np.linalg.norm(pred - entry_pts[i]))
    return errors, local


def fit_one_mark_circle(ring_pts, entry_pts, normals, mark, force_loo=False):
    """2026-10-05: --fit-circle's per-mark analog of fit_one_mark() -- fits
    the SAME correction, but as (radial_u, radial_v, normal_n) components
    in each sample's own phantom-tangent frame instead of a raw base-frame
    XYZ vector. radius_local_mm/theta_local_deg below are exactly the 3D,
    metric analog of mic_perception.py's mark0/1_r_from_plug/theta_deg
    pixel-space diagnostic -- if mark0 and mark1 report a consistent
    radius_local_mm here too, that is STRONGER evidence than the pixel-only
    check (no camera-projection ambiguity) that "distance from limbus to
    entry mark" is a real, position-invariant anatomical constant of this
    phantom, not a projection artifact."""
    n = len(ring_pts)
    if n == 0:
        return None
    ring_pts = np.array(ring_pts, np.float64)
    entry_pts = np.array(entry_pts, np.float64)
    local = _local_components(ring_pts, entry_pts, normals)
    mean_local = local.mean(axis=0)
    in_sample_resid = np.linalg.norm(local - mean_local, axis=1)

    uncorrected_errors = np.linalg.norm(ring_pts - entry_pts, axis=1)
    uncorrected_rms = float(np.sqrt((uncorrected_errors ** 2).mean()))
    radius_local_mm = float(np.hypot(mean_local[0], mean_local[1]))
    theta_local_deg = float(np.degrees(np.arctan2(mean_local[1], mean_local[0])))

    print("\n=== mark%d (circle/local-frame): %d paired sample(s) ===" % (mark, n))
    print("  BASELINE (no correction): raw |ring - entry| RMS=%.3fmm" % uncorrected_rms)
    print("  local offset = mean(u,v,n) = [%.3f %.3f %.3f]mm  (std=[%.3f %.3f %.3f]mm)" %
          (mean_local[0], mean_local[1], mean_local[2], *local.std(axis=0)))
    print("  => radius_local=%.3fmm  theta_local=%.1fdeg  (tangent-plane polar form, "
          "angle=0 along global-Z projected onto the plane -- see tangent_frame_from_normal())"
          % (radius_local_mm, theta_local_deg))
    print("  normal-direction residual (delta_n)=%.3fmm -- should be ~0 if the entry mark is "
          "roughly coplanar with the ring; large values mean the ring and the true entry point "
          "are NOT at the same depth along the phantom normal" % mean_local[2])
    print("  in-sample scatter: RMS=%.3fmm max=%.3fmm  (NOT an accuracy number -- see LOO below)" %
          (float(np.sqrt((in_sample_resid ** 2).mean())), float(in_sample_resid.max())))

    loo_rms_mm, loo_errors, ci = None, None, None
    if force_loo or n >= 5:
        try:
            loo_errors, _ = leave_one_out_circle_errors(ring_pts, entry_pts, normals)
            loo_rms_mm = float(np.sqrt((loo_errors ** 2).mean()))
            lo, hi = bootstrap_ci_mean(loo_errors)
            ci = (lo, hi)
            print("  --- leave-one-out cross-validation (HONEST accuracy estimate) ---")
            print("  LOO RMS error: %.3fmm  (mean=%.3fmm std=%.3fmm max=%.3fmm)" %
                  (loo_rms_mm, loo_errors.mean(), loo_errors.std(), loo_errors.max()))
            print("  bootstrap 95%% CI on mean LOO error: [%.3f, %.3f] mm  (n_resamples=2000)" %
                  (lo, hi))
            if loo_rms_mm < uncorrected_rms:
                print("  CORRECTION HELPS: %.3fmm -> %.3fmm (%.0f%% reduction vs. raw ring lock)" %
                      (uncorrected_rms, loo_rms_mm, 100.0 * (1 - loo_rms_mm / uncorrected_rms)))
            else:
                print("  *** CORRECTION DOES NOT HELP on held-out data (%.3fmm -> %.3fmm, WORSE) "
                      "-- report this negative result ***" % (uncorrected_rms, loo_rms_mm))
            if n < 8:
                print("  NOTE: n=%d -- keep pressing '0' across sessions; treat this as provisional." % n)
        except ValueError as e:
            print("  LOO skipped: %s" % e)
    else:
        print("  NOTE: only %d paired sample(s) -- need >=5 for an honest accuracy number." % n)

    return {
        "mark": mark, "n": n, "mean_local": mean_local,
        "radius_local_mm": radius_local_mm, "theta_local_deg": theta_local_deg,
        "in_sample_rms_mm": float(np.sqrt((in_sample_resid ** 2).mean())),
        "uncorrected_rms_mm": uncorrected_rms,
        "loo_rms_mm": loo_rms_mm, "loo_errors_mm": loo_errors,
        "bootstrap_ci_mm": ci,
    }


def do_fit_circle(args):
    results = {}
    for mark in (0, 1):
        ring_pts, entry_pts, normals, meta = load_paired_samples_with_normal(args.samples, mark)
        r = fit_one_mark_circle(ring_pts, entry_pts, normals, mark, force_loo=args.loo)
        if r is not None:
            results[mark] = r

    if not results:
        raise SystemExit("no paired (ring, entry, phantom_normal) samples found in %s for either "
                          "mark -- need samples logged AFTER the 2026-10-05 log_ring_offset_sample() "
                          "update (which added phantom_normal_base); re-press '0' with a current "
                          "phantom_normal.npz loaded" % args.samples)

    # 2026-10-05: the key cross-mark diagnostic -- a 3D, metric (mm),
    # camera-projection-free analog of the 2D pixel-radius check already
    # run on mic_perception.py's output. If these two numbers agree, that
    # is strong, largely independent confirmation that "limbus-to-entry
    # distance" really is a position-invariant anatomical constant of this
    # phantom (the premise the whole circle-mode redesign rests on).
    if 0 in results and 1 in results:
        r0, r1 = results[0]["radius_local_mm"], results[1]["radius_local_mm"]
        print("\n=== cross-mark radius check (3D, metric) ===")
        print("  mark0 radius_local=%.3fmm   mark1 radius_local=%.3fmm   diff=%.3fmm" %
              (r0, r1, abs(r0 - r1)))
        print("  (if these agree well, radius_local_mm is a trustworthy phantom-intrinsic "
              "constant -- a single shared value could then be used for a NEW target angle that "
              "is neither mark0's nor mark1's own theta_local_deg)")

    save = {"timestamp": time.time(), "source_samples": os.path.abspath(args.samples),
            "ref_axis_convention": "tangent_frame_from_normal() in mic_extrinsic_calib.py "
                                    "(global-Z projected onto tangent plane, fallback global-X)"}
    for mark, r in results.items():
        save["local_offset_mark%d" % mark] = r["mean_local"]  # (u, v, n) components, mm
        save["radius_local_mm_mark%d" % mark] = r["radius_local_mm"]
        save["theta_local_deg_mark%d" % mark] = r["theta_local_deg"]
        save["loo_rms_mm_mark%d" % mark] = r["loo_rms_mm"] if r["loo_rms_mm"] is not None else -1.0
        save["n_samples_mark%d" % mark] = r["n"]
        save["uncorrected_rms_mm_mark%d" % mark] = r["uncorrected_rms_mm"]
        if r["loo_errors_mm"] is not None:
            save["loo_errors_mm_mark%d" % mark] = r["loo_errors_mm"]
        if r["bootstrap_ci_mm"] is not None:
            save["bootstrap_ci_mm_mark%d" % mark] = np.array(r["bootstrap_ci_mm"])

    np.savez(args.out_circle, **save)
    print("\nSaved -> %s" % args.out_circle)
    print("home_to_rcm_insert_v2_precision.py's 'x' key reads this when MIC_RCM_MODE=circle "
          "(radius self-consistency across marks is diagnostic only -- it does NOT itself pick "
          "which radius_local_mm_mark{0,1} gets used; that is still an explicit choice).")
    for mark, r in results.items():
        if r["loo_rms_mm"] is None:
            print("  mark%d: loo_rms_mm=-1.0 (NOT VALIDATED -- need >=5 paired samples, have %d)" %
                  (mark, r["n"]))


def do_report_circle(args):
    if not os.path.isfile(args.npz_circle):
        raise SystemExit("%s does not exist -- run --fit-circle first" % args.npz_circle)
    z = np.load(args.npz_circle, allow_pickle=True)
    age_s = time.time() - float(z["timestamp"])
    print("=== %s (fit %.0fs ago, from %s) ===" % (args.npz_circle, age_s, z.get("source_samples", "?")))
    radii = {}
    for mark in (0, 1):
        key = "radius_local_mm_mark%d" % mark
        if key not in z:
            print("mark%d: no local offset saved" % mark)
            continue
        radius = float(z[key])
        theta = float(z["theta_local_deg_mark%d" % mark])
        n = int(z["n_samples_mark%d" % mark]) if ("n_samples_mark%d" % mark) in z else -1
        loo = float(z["loo_rms_mm_mark%d" % mark]) if ("loo_rms_mm_mark%d" % mark) in z else -1.0
        radii[mark] = radius
        print("mark%d: radius_local=%.3fmm  theta_local=%.1fdeg  n_samples=%d  loo_rms_mm=%s" %
              (mark, radius, theta, n,
               ("%.3f (VALIDATED)" % loo) if loo >= 0 else "-1.0 (NOT VALIDATED -- need >=5 samples)"))
    if 0 in radii and 1 in radii:
        print("cross-mark radius diff: %.3fmm" % abs(radii[0] - radii[1]))


def do_compare(args):
    """2026-10-05: run BOTH --fit (fixed base-frame vector) and --fit-circle
    (phantom-pose-invariant tangent-plane local frame) on the EXACT SAME
    samples file, side by side, and print an explicit recommendation.

    WHY THIS EXISTS: the user correctly pushed back on just assuming the
    newer, more complex, more failure-mode-prone (depends on a FRESH,
    accurate phantom_normal_estimate.py fit on top of everything the fixed
    model already needed) tangent-frame model is better just because it is
    theoretically more general / pose-invariant. "More general" is not the
    same claim as "more accurate on this real phantom, with this real
    camera, with this real amount of normal-estimation noise" -- that is
    an empirical question, and this is the tool that actually answers it,
    on whatever data you have, rather than asserting it. Only trust
    --fit-circle for the live demo if ITS own LOO RMS here is actually
    competitive with (not meaningfully worse than) --fit's, on the SAME
    samples -- if it's not, that is itself a valid, reportable finding
    (the extra normal-estimation noise outweighs the pose-invariance
    benefit for this particular setup), not something to paper over.

    NOTE: this needs samples collected AFTER 2026-10-05 (with
    phantom_normal_base logged) for the --fit-circle side; see
    load_paired_samples_with_normal(). If a sample predates that, --fit
    can still use it but --fit-circle cannot -- the two sides may end up
    comparing different, overlapping-but-not-identical subsets of n, and
    that is called out explicitly below rather than silently ignored."""
    print("=" * 78)
    print("CORRESPONDENCE MODEL COMPARISON: fixed-vector vs. tangent-frame-circle")
    print("=" * 78)
    summary = {}
    for mark in (0, 1):
        ring_f, entry_f, _ = load_paired_samples(args.samples, mark)
        ring_c, entry_c, normals_c, _ = load_paired_samples_with_normal(args.samples, mark)
        print("\n" + "-" * 78)
        print("MARK %d -- fixed-vector uses n=%d sample(s), circle uses n=%d sample(s) "
              "(circle needs phantom_normal_base, i.e. samples logged after 'n' was pressed)"
              % (mark, len(ring_f), len(ring_c)))
        if len(ring_f) != len(ring_c):
            print("  !! n MISMATCH -- %d sample(s) are missing a phantom_normal_base and are "
                  "silently excluded from the circle-mode comparison below. Press 'n' before '0' "
                  "every time to keep both sides comparable." % (len(ring_f) - len(ring_c)))
        print("-" * 78)
        r_fixed = fit_one_mark(ring_f, entry_f, mark, force_loo=args.loo) if ring_f else None
        r_circle = fit_one_mark_circle(ring_c, entry_c, normals_c, mark, force_loo=args.loo) if ring_c else None
        summary[mark] = (r_fixed, r_circle)

    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    for mark, (r_fixed, r_circle) in summary.items():
        if r_fixed is None and r_circle is None:
            print("mark%d: no data for either model" % mark)
            continue
        lf = r_fixed["loo_rms_mm"] if r_fixed and r_fixed["loo_rms_mm"] is not None else None
        lc = r_circle["loo_rms_mm"] if r_circle and r_circle["loo_rms_mm"] is not None else None
        if lf is None and lc is None:
            print("mark%d: neither model has >=5 samples yet -- NOT VALIDATED, collect more "
                  "('0' key, ideally with 'n' pressed fresh first each time) before trusting "
                  "either for the demo" % mark)
        elif lc is None:
            print("mark%d: only fixed-vector is validated (LOO=%.3fmm) -- circle needs more "
                  "samples WITH phantom_normal_base (press 'n' before '0')" % (mark, lf))
        elif lf is None:
            print("mark%d: only circle is validated (LOO=%.3fmm) -- unusual, check your samples"
                  % (mark, lc))
        elif lc <= lf * 1.1:  # within 10% -- treat as "competitive", prefer circle for its
                               # pose-invariance since it isn't meaningfully worse in accuracy
            verdict = "circle is AS GOOD OR BETTER" if lc <= lf else "circle is within 10% of fixed (close enough)"
            print("mark%d: fixed LOO=%.3fmm  circle LOO=%.3fmm  -> %s -- USE CIRCLE for the demo "
                  "(pose-invariance benefit at no meaningful accuracy cost)" % (mark, lf, lc, verdict))
        else:
            pct_worse = 100.0 * (lc / lf - 1)
            print("mark%d: fixed LOO=%.3fmm  circle LOO=%.3fmm  -> circle is %.0f%% WORSE -- "
                  "USE FIXED-VECTOR for the demo instead (normal-estimation noise is costing more "
                  "accuracy than pose-invariance is worth here; re-check phantom_normal_estimate.py's "
                  "own repeatability with phantom_normal_repeatability.py before trying circle again)"
                  % (mark, lf, lc, pct_worse))


def do_report(args):
    if not os.path.isfile(args.npz):
        raise SystemExit("%s does not exist -- run --fit first" % args.npz)
    z = np.load(args.npz, allow_pickle=True)
    age_s = time.time() - float(z["timestamp"])
    print("=== %s (fit %.0fs ago, from %s) ===" % (args.npz, age_s, z.get("source_samples", "?")))
    for mark in (0, 1):
        key = "offset_mark%d" % mark
        if key not in z:
            print("mark%d: no offset saved" % mark)
            continue
        off = z[key]
        n = int(z["n_samples_mark%d" % mark]) if ("n_samples_mark%d" % mark) in z else -1
        loo = float(z["loo_rms_mm_mark%d" % mark]) if ("loo_rms_mm_mark%d" % mark) in z else -1.0
        base_key = "uncorrected_rms_mm_mark%d" % mark
        base = float(z[base_key]) if base_key in z else None
        print("mark%d: offset=[%.3f %.3f %.3f]mm  n_samples=%d  loo_rms_mm=%s" %
              (mark, off[0], off[1], off[2], n,
               ("%.3f (VALIDATED)" % loo) if loo >= 0 else "-1.0 (NOT VALIDATED -- need >=5 samples)"))
        if base is not None:
            print("        baseline (no correction) RMS=%.3fmm  --  %s" %
                  (base, ("correction helps" if (loo >= 0 and loo < base) else
                          ("correction does NOT help" if loo >= 0 else "not yet compared (unvalidated)"))))
        ci_key = "bootstrap_ci_mm_mark%d" % mark
        if ci_key in z and loo >= 0:
            ci = z[ci_key]
            print("        bootstrap 95%% CI on mean LOO error: [%.3f, %.3f] mm" % (ci[0], ci[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--report", action="store_true")
    # 2026-10-05: --fit-circle / --report-circle -- the phantom-pose-invariant,
    # tangent-plane-local-frame generalization (radius_local_mm/theta_local_deg
    # instead of a raw fixed base-frame vector). See tangent_frame_from_normal()
    # in mic_extrinsic_calib.py and fit_one_mark_circle()'s docstring above for
    # the full rationale. Entirely separate CLI flags/output file from --fit/
    # --report on purpose -- this does NOT replace or alter the original
    # fixed-vector path at all, it is an additive, opt-in alternative.
    ap.add_argument("--fit-circle", action="store_true",
                     help="phantom-pose-invariant local-tangent-frame fit (needs samples logged "
                          "with phantom_normal_base, i.e. collected after 2026-10-05)")
    ap.add_argument("--report-circle", action="store_true")
    ap.add_argument("--compare", action="store_true",
                     help="2026-10-05: run --fit and --fit-circle on the SAME samples side by "
                          "side and print an explicit recommendation -- use this, not a priori "
                          "assumption, to decide which model the demo should actually trust")
    ap.add_argument("--loo", action="store_true",
                     help="force-attempt leave-one-out even below n=5 (mainly for debugging; "
                          "the resulting number is not trustworthy)")
    ap.add_argument("--samples", default="ring_offset_samples.jsonl")
    ap.add_argument("--out", default="ring_to_entry_offset.npz")
    ap.add_argument("--npz", default="ring_to_entry_offset.npz", help="for --report")
    ap.add_argument("--out-circle", default="ring_to_entry_offset_circle.npz")
    ap.add_argument("--npz-circle", default="ring_to_entry_offset_circle.npz", help="for --report-circle")
    a = ap.parse_args()
    if a.fit:
        do_fit(a)
    elif a.report:
        do_report(a)
    elif a.fit_circle:
        do_fit_circle(a)
    elif a.report_circle:
        do_report_circle(a)
    elif a.compare:
        do_compare(a)
    else:
        print(__doc__)
