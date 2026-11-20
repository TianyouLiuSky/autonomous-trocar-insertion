#!/usr/bin/env python3
"""Diagnose the ~20-38deg vis3d_err_deg discrepancy seen 2026-08-25 evening
(runs #2/#3 of the 30deg campaign, REORIENT(internal,perp) and ORIENT(30deg)).

WHY THIS EXISTS: home_to_rcm_insert_v2_precision.py's independent depth-
camera 3D shaft-direction check (measure_shaft_direction_3d, module
docstring pt.4) started returning well-conditioned ("ok=True", tight RMS,
good inlier count, low half-split resampling disagreement) measurements
tonight showing the REAL physical needle-shaft direction disagrees with the
KINEMATICALLY-COMMANDED target_dir by ~20-38deg -- while the control loop's
OWN self-check (err_deg, comparing its computed v_cur=R.dot(NEEDLE_AXIS_LOCAL)
to target_dir) reports success at <0.3deg every time. Since vis3d's
measurement is independent of NEEDLE_AXIS_LOCAL/pivot calibration (it only
depends on hand-eye R_he/t_he, shared by everything in this codebase, and
is NOT expected to be biased by this), the most likely explanation is that
NEEDLE_AXIS_LOCAL (the assumed EE-frame axis of the tool, currently derived
from normalizing the rcmoffset model's tip_inee_at_calib=(0.830,-2.844,
-13.027)) is itself off by roughly that amount from the tool's TRUE
EE-frame axis -- i.e. the SAME class of bug as the 159deg NEEDLE_AXIS_LOCAL
sign flip found and fixed earlier the same day, just smaller in magnitude
and not yet caught because tonight is the first time vis3d_ok fired True
with trustworthy conditioning stats (see that module's docstring pt.4 final
paragraph: "confirm vis3d_ok actually starts firing True ... before citing
any vis3d_err_deg number as a validated result" -- tonight is that moment).

METHOD: home_to_rcm_insert_v2_precision.py was just patched (2026-08-25
evening) to additionally log, on every vischeck row: the RAW measured 3D
shaft direction in BASE frame (vis3d_dx/dy/dz, sign-ambiguous -- d and -d
fit the same points equally well) and the FrameEE orientation quaternion AT
THE MOMENT of that measurement (ee_qx/qy/qz/qw). Given N such rows from N
DIFFERENT tool orientations, for a rigid tool there is a SINGLE fixed
EE-frame axis a_ee such that, for every sample i,
    R(q_i) @ a_ee  ==  +-d_i      (sign ambiguous per-sample)
This script:
  1. loads all rows with vis3d_ok==True and non-empty ee_q*/vis3d_d*,
  2. for each, computes candidate a_ee = R(q_i)^T @ d_i AND its negation,
     picks whichever is closer to the CURRENT NEEDLE_AXIS_LOCAL hypothesis
     (this only resolves the sign, it does not bias the answer toward the
     current hypothesis in any other way -- both candidates are unit
     vectors and exactly one of the two is within 90deg of any given
     reference, so this is a deterministic, harmless disambiguation),
  3. reports the pairwise angular spread across samples' a_ee estimates
     (if the tool is rigid and hand-eye/vis3d are unbiased, this spread
     should be small, comparable to each sample's own vis3d_unc_deg) and
     the mean direction,
  4. compares the mean measured a_ee to the NEEDLE_AXIS_LOCAL this run
     actually used (normalize(tip_inee_at_calib) from tip_offset_rcmoffset.
     npz) and reports the angle between them -- if that angle is close to
     the vis3d_err_deg values actually observed, this confirms
     NEEDLE_AXIS_LOCAL is the root cause, and the printed mean a_ee is the
     empirically-corrected axis to use instead.

LIMITATIONS (stated up front, not discovered after the fact):
  - Needs >=2 samples at MEANINGFULLY DIFFERENT orientations to be
    well-conditioned (a single sample has a 1-parameter family of a_ee
    that would satisfy it exactly along with measurement noise in the
    other 2 -- not actually true, R@a_ee=d for a single R fully
    determines a_ee given d, but noise in a SINGLE sample cannot be
    distinguished from a systematic bias; only agreement ACROSS
    differently-oriented samples can do that).
  - Shares whatever bias (if any) hand-eye ROTATION (R_he) has -- this
    script cannot rule that out, it can only show whether the CURRENT
    NEEDLE_AXIS_LOCAL is self-consistent with independent vision
    measurements taken under the assumption that R_he is correct (which
    is the same assumption home_to_rcm_insert_v2_precision.py's ORIENT
    loop and every prior calibration step in this codebase already makes).
  - vis3d's direction has a sign ambiguity resolved per-sample against the
    CURRENT NEEDLE_AXIS_LOCAL; if the current hypothesis is off by close to
    90deg for some sample the disambiguation could pick the wrong branch --
    watch for this as a bimodal-looking spread in the printed per-sample
    list, not a single tight cluster.

USAGE:
  python3 needle_axis_diagnose.py

Reads PRECISION_LOG (same env var as the main script, default
third_part/precision_log.csv). Does not touch the robot or camera --
pure offline analysis of already-collected rows.
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRECISION_LOG = os.environ.get("PRECISION_LOG", os.path.join(HERE, "precision_log.csv"))
TIP_OFFSET_RCMOFFSET_NPZ = os.environ.get(
    "TIP_OFFSET_RCMOFFSET_NPZ", os.path.join(HERE, "tip_offset_rcmoffset.npz"))


def quat_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def main():
    if not os.path.isfile(PRECISION_LOG):
        print("!! %s not found" % PRECISION_LOG)
        return

    z = np.load(TIP_OFFSET_RCMOFFSET_NPZ)
    tip_inee_at_calib = np.asarray(z["tip_inee_at_calib"], float)
    axis_hypothesis = tip_inee_at_calib / np.linalg.norm(tip_inee_at_calib)
    print("Current NEEDLE_AXIS_LOCAL hypothesis (from %s):" % TIP_OFFSET_RCMOFFSET_NPZ)
    print("  tip_inee_at_calib = %s  ->  unit axis = %s" %
          (np.round(tip_inee_at_calib, 3), np.round(axis_hypothesis, 4)))
    print()

    with open(PRECISION_LOG) as f:
        rows = list(csv.DictReader(f))

    samples = []
    for r in rows:
        if r.get("vis3d_ok") != "True":
            continue
        needed = ["vis3d_dx", "vis3d_dy", "vis3d_dz", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
        if any(not r.get(k) for k in needed):
            continue
        d = np.array([float(r["vis3d_dx"]), float(r["vis3d_dy"]), float(r["vis3d_dz"])])
        q = np.array([float(r["ee_qx"]), float(r["ee_qy"]), float(r["ee_qz"]), float(r["ee_qw"])])
        samples.append({
            "timestamp": r["timestamp"], "action": r["action"],
            "d_base": d, "q": q,
            "vis3d_err_deg": r.get("vis3d_err_deg", ""),
            "vis3d_unc_deg": r.get("vis3d_unc_deg", ""),
        })

    print("Found %d vis3d_ok=True rows with direction+quaternion logged." % len(samples))
    if len(samples) == 0:
        print("\n!! No usable samples yet. The direction/quaternion logging was only added just\n"
              "   now (2026-08-25 evening) -- every row from tonight's runs #1-3 predates it.\n"
              "   Run a few QUICK ORIENT-only cycles (z -> g -> t at a couple of different\n"
              "   azimuths/angles; no need for i/p/full insertion -- cheaper and lower-risk\n"
              "   than a full campaign run) with VISION_ANGLE_CHECK=1 (the default) and\n"
              "   re-run this script.")
        return
    if len(samples) < 2:
        print("\n!! Only 1 sample -- cannot check rigidity (need >=2 at DIFFERENT orientations).\n"
              "   Collect at least one more ORIENT/REORIENT at a clearly different azimuth or\n"
              "   angle and re-run.")

    a_ee_list = []
    print()
    for s in samples:
        R = quat_to_R(s["q"])
        cand = R.T.dot(s["d_base"])
        cand = cand / np.linalg.norm(cand)
        if np.dot(cand, axis_hypothesis) < 0:
            cand = -cand
        a_ee_list.append(cand)
        ang_vs_hyp = np.degrees(np.arccos(np.clip(abs(np.dot(cand, axis_hypothesis)), 0, 1)))
        print("  %s  %-28s a_ee=%s  (%.1fdeg from current hypothesis; vis3d_err_deg=%s +-%s)" %
              (s["timestamp"], s["action"], np.round(cand, 4), ang_vs_hyp,
               s["vis3d_err_deg"], s["vis3d_unc_deg"]))

    a_ee_arr = np.array(a_ee_list)
    mean_a = a_ee_arr.mean(axis=0)
    mean_a = mean_a / np.linalg.norm(mean_a)
    spreads = [np.degrees(np.arccos(np.clip(abs(np.dot(a, mean_a)), 0, 1))) for a in a_ee_arr]
    print()
    print("Mean measured EE-local axis a_ee = %s" % np.round(mean_a, 4))
    print("Pairwise spread vs mean: max=%.1fdeg mean=%.1fdeg (small = rigid/consistent; "
          "large or bimodal = sign-disambiguation problem or a genuinely non-rigid/loose "
          "mount -- inspect the per-sample list above before trusting the mean)" %
          (max(spreads), float(np.mean(spreads))))
    ang_final = np.degrees(np.arccos(np.clip(abs(np.dot(mean_a, axis_hypothesis)), 0, 1)))
    print()
    print("Mean measured axis vs current NEEDLE_AXIS_LOCAL hypothesis: %.1fdeg apart" % ang_final)
    if ang_final > 5.0:
        print("  -> CONFIRMS a real NEEDLE_AXIS_LOCAL error of this rough magnitude.")
        print("     Consider using the measured axis above (if spread is small) as an")
        print("     explicit override, decoupled from the rcmoffset POSITION model:")
        print("     NEEDLE_AXIS_LOCAL = %s" % np.round(mean_a, 4))
    else:
        print("  -> Small; NEEDLE_AXIS_LOCAL does not look like the dominant cause here --")
        print("     look elsewhere (hand-eye rotation bias, target_dir computation, or the")
        print("     vis3d measurement itself under this specific viewing geometry).")


if __name__ == "__main__":
    main()
