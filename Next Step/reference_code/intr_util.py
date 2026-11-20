#!/usr/bin/env python3
"""Minimal shared helper: pick the BEST available color-stream intrinsics.

WHY THIS EXISTS (2026-08-23, written to unblock pt.16 two-axis testing):
home_to_rcm_insert_v2_precision.py's new copy imports
`load_best_intrinsics` from this module instead of calling
`profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()`
directly. The intent (see camera_intrinsics_calibrate.py's docstring) is to
prefer a self-calibrated cam_intrinsics.npz (Zhang's method via
cv2.calibrateCamera on a printed checkerboard, with a held-out LOO-RMS
check) over RealSense's FACTORY-reported intrinsics, which are known to
carry a few pixels of unverified systematic error in the principal point
and distortion coefficients.

STATUS: this file did not exist anywhere in the project (checked both
SHER_I2RIS_teleoperate/ and the ISMR_Package_2026-08-20/ doc copy) before
2026-08-23 -- home_to_rcm_insert_v2_precision.py's new (pt.15/pt.16) copy
would raise ImportError on the very first line if run without it. Written
now specifically to (a) unblock running that script at all, and (b) match
the exact cam_intrinsics.npz schema camera_intrinsics_calibrate.py already
writes (fx, fy, ppx, ppy, coeffs, width, height, fit_rms_px, loo_rms_px, n)
so that TODAY's planned camera self-calibration (see time-budget plan)
slots in with zero further code changes: run
camera_intrinsics_calibrate.py, it overwrites cam_intrinsics.npz, and the
NEXT run of home_to_rcm_insert_v2_precision.py automatically picks it up.

Before a fresh self-cal exists, or if the existing cam_intrinsics.npz on
this rig (dated 2026-08-03, i.e. FACTORY intrinsics that were merely
copied into that filename, never independently checkerboard-calibrated --
see camera_intrinsics_calibrate.py's own docstring pt. "currently just
stores the RealSense FACTORY-reported intrinsics") predates that script's
existence, this loader still works correctly -- it just returns something
numerically identical to the factory intrinsics every prior script used,
so behavior for anyone not yet re-running the self-cal is UNCHANGED.

NEVER raises: any problem with the self-cal file (missing, wrong
resolution, corrupt, missing keys) falls back to RealSense factory
intrinsics with a printed diagnostic, so this is a safe drop-in
replacement for the one-liner it replaces.
"""
from __future__ import print_function
import os

import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
CAM_INTRINSICS_NPZ = os.environ.get("CAM_INTRINSICS_NPZ", os.path.join(HERE, "cam_intrinsics.npz"))


def load_best_intrinsics(profile):
    """profile: the rs.pipeline_profile returned by pipe.start(cfg).

    Returns an rs.intrinsics object for the COLOR stream, suitable for
    rs.rs2_deproject_pixel_to_point(intr, ...) and the manual
    fx/fy/ppx/ppy projection math used throughout home_to_rcm_insert*.py.
    """
    factory = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    if not os.path.isfile(CAM_INTRINSICS_NPZ):
        print("intr_util: no self-cal file at %s -- using RealSense FACTORY intrinsics "
              "(fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f)" %
              (CAM_INTRINSICS_NPZ, factory.fx, factory.fy, factory.ppx, factory.ppy))
        return factory

    try:
        d = np.load(CAM_INTRINSICS_NPZ)
        needed = ("fx", "fy", "ppx", "ppy", "coeffs", "width", "height")
        missing = [k for k in needed if k not in d.files]
        if missing:
            print("intr_util: %s is missing keys %s (old-format file?) -- using RealSense "
                  "FACTORY intrinsics instead" % (CAM_INTRINSICS_NPZ, missing))
            return factory

        w, h = int(d["width"]), int(d["height"])
        if w != factory.width or h != factory.height:
            print("intr_util: cam_intrinsics.npz resolution %dx%d != live stream %dx%d -- "
                  "using RealSense FACTORY intrinsics instead (self-cal file is stale / for a "
                  "different mode)" % (w, h, factory.width, factory.height))
            return factory

        intr = rs.intrinsics()
        intr.width, intr.height = factory.width, factory.height
        intr.fx, intr.fy = float(d["fx"]), float(d["fy"])
        intr.ppx, intr.ppy = float(d["ppx"]), float(d["ppy"])
        intr.model = factory.model  # distortion MODEL unchanged, only coeffs come from self-cal
        coeffs = [float(c) for c in np.asarray(d["coeffs"]).ravel()]
        coeffs = (coeffs + [0.0] * 5)[:5]
        intr.coeffs = coeffs

        extra = ""
        if "fit_rms_px" in d.files and "loo_rms_px" in d.files and "n" in d.files:
            extra = " (fit_rms=%.3fpx loo_rms=%.3fpx n=%d captures)" % (
                float(d["fit_rms_px"]), float(d["loo_rms_px"]), int(d["n"]))
        print("intr_util: using SELF-CALIBRATED intrinsics from %s%s -- "
              "fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f coeffs=%s" %
              (CAM_INTRINSICS_NPZ, extra, intr.fx, intr.fy, intr.ppx, intr.ppy, coeffs))
        return intr
    except Exception as e:
        print("intr_util: FAILED to load/parse %s (%r) -- falling back to RealSense FACTORY "
              "intrinsics" % (CAM_INTRINSICS_NPZ, e))
        return factory
