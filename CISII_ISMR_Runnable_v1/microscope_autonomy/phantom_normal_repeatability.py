#!/usr/bin/env python3
"""phantom_normal_repeatability.py -- quantify how much phantom_normal_estimate.py's
own normal estimate varies between repeated 'n' presses AT THE SAME PHYSICAL
PHANTOM POSE (robot/phantom/microscope untouched between presses).

WHY THIS EXISTS (2026-10-05): home_to_rcm_insert_v2_precision.py's
MIC_RCM_MODE="circle" (the phantom-pose-invariant ring-offset generalization,
see mic_extrinsic_calib.tangent_frame_from_normal() + mic_ring_offset_verify.py
--fit-circle) is the FIRST part of this pipeline that depends on
phantom_normal_estimate.py's live normal fit at ALL -- the original fixed-
vector model (MIC_RCM_MODE="mark") never used a surface normal for anything.
So switching to "circle" introduces exactly one new error source that did not
exist before: how noisy is a single 'n' press, really?

This matters a lot: a radius_local_mm of ~15mm with a normal-direction error
of just 2 degrees produces an in-plane placement error of
    15mm * sin(2deg) ~= 0.52mm
on its own -- comparable to or bigger than the ENTIRE accuracy budget this
project has been fighting for since August. mic_ring_offset_verify.py
--compare answers "is circle worse than fixed, overall, on real paired
samples" after the fact; THIS script answers the narrower, upstream question
"how much of any such gap is just normal-estimation noise", which is useful
BEFORE spending any of the precious 10 campaign trials, and is also the
right place to look if --compare ever reports "circle is worse".

WORKFLOW
--------
1. Get the robot/phantom/scope into position exactly as for a real trial,
   with the D405 ring already locked ('z') -- 'n' requires rcm_ok().
2. Press 'n' AT LEAST 5 times in a row, deliberately NOT moving anything
   (not the phantom, not the robot, not the camera) between presses. Each
   press appends one row to NORMAL_SAMPLES_LOG (default:
   phantom_normal_samples.jsonl, see that env var in the main script).
3. Run this script:
       python3 phantom_normal_repeatability.py [path/to/phantom_normal_samples.jsonl] [radius_mm]
   radius_mm defaults to 15.0 (round-number stand-in for a typical
   limbus-to-pars-plana-entry distance) -- pass your own
   radius_local_mm_mark0/1 (from ring_to_entry_offset_circle.npz, once fit)
   once you have a real one, to get an error estimate in YOUR actual units
   instead of a generic placeholder.

Reports:
  - angular spread (deg) of the n>=5 normals around their own mean direction
  - the SAME number expressed as an equivalent in-plane mm placement error
    at the given radius (small-angle: radius * sin(angle))
  - an explicit verdict: is this noise small enough to trust circle mode,
    relative to the project's established ~0.3-0.5mm precision targets?

No ROS / robot / camera dependency -- pure JSONL analysis, safe to run
anywhere once phantom_normal_samples.jsonl has been copied over.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from runtime_session import ROOT, data_path
import json
import os
import sys
import numpy as np


def load_normals(path):
    normals = []
    if not os.path.isfile(path):
        return normals
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n = np.array(rec["normal_base"], np.float64)
            normals.append(n / np.linalg.norm(n))
    return normals


def angle_between(a, b):
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "phantom_normal_samples.jsonl")
    radius_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    normals = load_normals(path)
    print("=== phantom_normal_repeatability (%s) ===" % path)
    print("n samples: %d" % len(normals))
    if len(normals) < 2:
        raise SystemExit("need >=2 'n'-key samples at a FIXED phantom pose (ideally >=5) -- "
                          "have %d. Press 'n' several more times without moving anything, "
                          "then re-run." % len(normals))
    if len(normals) < 5:
        print("!! only %d samples -- treat everything below as provisional, press 'n' more "
              "times for an honest spread estimate (same caution this project applies "
              "everywhere else below n=5)" % len(normals))

    normals = np.array(normals)
    mean_n = normals.mean(axis=0)
    mean_n = mean_n / np.linalg.norm(mean_n)
    angles = np.array([angle_between(n, mean_n) for n in normals])

    print("\nangular spread around mean normal direction:")
    print("  mean=%.3fdeg  std=%.3fdeg  max=%.3fdeg" % (angles.mean(), angles.std(), angles.max()))

    # pairwise max (worst-case any-two-samples disagreement) -- a more
    # conservative number than "vs the mean" above, useful because in a
    # real campaign you don't get to average away your own noise within a
    # single trial's single 'n' press.
    pairwise = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            pairwise.append(angle_between(normals[i], normals[j]))
    pairwise = np.array(pairwise)
    print("  pairwise (any 2 samples): mean=%.3fdeg max=%.3fdeg" % (pairwise.mean(), pairwise.max()))

    err_mean_mm = radius_mm * np.sin(np.radians(angles.mean()))
    err_max_mm = radius_mm * np.sin(np.radians(angles.max()))
    print("\nequivalent in-plane placement error at radius=%.1fmm (small-angle, radius*sin(angle)):" % radius_mm)
    print("  from mean angular spread: %.3fmm" % err_mean_mm)
    print("  from max angular spread:  %.3fmm" % err_max_mm)

    print("\n=== VERDICT ===")
    # 0.3mm chosen as the comparison bar: it's the ballpark of this
    # project's own established PLUNGE/APPROACH error numbers (e.g. Trial 1
    # 2026-10-01: PLUNGE err=0.289mm) -- if normal noise ALONE is already
    # comparable to or bigger than that, circle mode cannot plausibly beat
    # the fixed-vector model's accuracy no matter how many ring-offset
    # samples you collect, because this noise floor is per-press, not
    # something averaging over more samples removes.
    if err_max_mm < 0.15:
        print("Normal-estimation noise is SMALL (<0.15mm equivalent even in the worst pair) -- "
              "circle mode's accuracy ceiling is not meaningfully limited by this. Safe to trust "
              "--fit-circle's own LOO number from mic_ring_offset_verify.py.")
    elif err_max_mm < 0.3:
        print("Normal-estimation noise is MODERATE (%.3fmm worst-case) -- comparable to this "
              "project's own established precision targets. --fit-circle's LOO RMS should be "
              "interpreted with this noise floor in mind; don't expect circle mode to beat the "
              "fixed-vector model by much even if it IS more pose-invariant." % err_max_mm)
    else:
        print(("Normal-estimation noise is LARGE (%.3fmm worst-case) -- likely to dominate "
               "circle mode's error budget. Re-check phantom_normal_estimate.py's lighting/ROI "
               "assumptions, or just use MIC_RCM_MODE=mark (fixed-vector) for the 10-trial "
               "campaign instead unless the phantom genuinely must move between trials.")
              % err_max_mm)


if __name__ == "__main__":
    main()
