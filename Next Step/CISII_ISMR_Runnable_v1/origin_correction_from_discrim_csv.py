#!/usr/bin/env python3
"""origin_correction_from_discrim_csv.py -- build an ORIGIN_CORRECTION
(base-frame constant, as used by home_to_rcm_insert_v2_precision.py's 'c'
key / ORIGIN_CORRECTION_NPZ) directly from an ALREADY-COLLECTED
handeye_vs_tip_discriminate.py click session, instead of doing a fresh
single depth-camera click.

BACKGROUND (2026-09-21): after re-doing hand-eye + pivot calibration
following a trocar swap, the green tip overlay still showed a visible
offset. Ran handeye_vs_tip_discriminate.py for n=24 multi-pose clicks; its
own verdict: the residual is explained far better by a constant BASE-frame
(hand-eye) delta (5.21px) than a constant TOOL-frame (tip-offset) delta
(7.40px) -- see that script's printed report. The user did not want to
redo the click session a third time, so this script reuses the ALREADY-
FITTED base-frame vector instead of asking for a fresh 'c'-key click.

CRITICAL CAVEAT (do not skip this when reusing the fitted vector):
handeye_vs_tip_discriminate.py's own report decomposes its fitted vector
into a component ACROSS the camera's line of sight (observable, trustworthy)
and a component ALONG it (unconstrained with a single static monocular
camera -- adding more clicks from more tool orientations does NOT resolve
this, because the blind direction is the SAME fixed base-frame direction
at every sample as long as the camera itself doesn't move). Blindly writing
the raw 3D fitted vector into ORIGIN_CORRECTION would inject an essentially
made-up depth-direction shift. This script instead:
  1. Reloads R_he from handeye_T.npz, computes cax = R_he @ [0,0,1] (the
     camera's viewing axis in base frame).
  2. Projects the already-reported fitted BASE-frame vector onto the plane
     perpendicular to cax (the reliable part) and discards the along-cax
     component (the unreliable part) instead of guessing at it.
  3. Saves ONLY the perpendicular component as `correction` in the exact
     schema load_origin_correction() expects.

This gives a PARTIAL but principled fix: since pixel-plane position is
overwhelmingly driven by the perpendicular-to-ray components (the along-ray
error only shows up at higher order, via depth-dependent foreshortening),
this should remove most of the visible on-screen misalignment, but does
NOT fully pin down the true 3D depth-direction bias. A depth-camera-based
'c'-key click is still the right way to resolve that remaining ambiguity
if/when it matters (e.g. if the true 3D insertion depth accuracy, not just
the 2D overlay, needs independent verification).

USAGE:
    python3 origin_correction_from_discrim_csv.py <fitted_base_vec_x> \\
        <fitted_base_vec_y> <fitted_base_vec_z> [--handeye handeye_T.npz] \\
        [--out origin_correction.npz]

The three positional args are read directly off handeye_vs_tip_discriminate.py's
own printed "BASE-frame delta = [...]" line (its 'f' key report) -- this
script does not re-run the click-fitting itself, it only does the
ray-decomposition + save step.
"""
from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vx", type=float)
    ap.add_argument("vy", type=float)
    ap.add_argument("vz", type=float)
    ap.add_argument("--handeye", default=data_path("handeye_T.npz"))
    ap.add_argument("--out", default=data_path("origin_correction.npz"))
    ap.add_argument("--source-note", default="")
    args = ap.parse_args()

    z = np.load(args.handeye)
    R_he = np.asarray(z["R"], float)

    v = np.array([args.vx, args.vy, args.vz])
    cax = R_he.dot(np.array([0.0, 0.0, 1.0]))
    cax /= np.linalg.norm(cax)

    par = float(v.dot(cax))
    perp_vec = v - par * cax
    perp_mag = np.linalg.norm(perp_vec)

    print("camera axis in base frame (cax) =", np.round(cax, 4))
    print("input fitted BASE-frame delta v =", v, " |v|=%.2f mm" % np.linalg.norm(v))
    print("along-ray component (DROPPED, unreliable) = %.3f mm -> vector %s"
          % (par, np.round(par * cax, 3)))
    print("across-ray component (KEPT, reliable)      = %s  |perp|=%.2f mm"
          % (np.round(perp_vec, 3), perp_mag))

    np.savez(args.out, correction=perp_vec, timestamp=time.time(),
             source="perpendicular-to-camera-ray component of a "
                     "handeye_vs_tip_discriminate.py multi-pose fit "
                     "(%s)" % args.source_note,
             full_fitted_vec=v, dropped_along_ray_mm=par)
    print("\nSaved -> %s  correction=%s mm" % (args.out, np.round(perp_vec, 3)))
    print("NOTE: process holding a live home_to_rcm_insert_v2_precision.py "
          "session must be RESTARTED to pick this up -- ORIGIN_CORRECTION "
          "is only read once, at module import time.")


if __name__ == "__main__":
    main()
