#!/usr/bin/env python3
"""Shared TipInEE consistency guard.

WHY: TipInEE (GUI "Tip Offset") has silently drifted multiple times across
sessions (0,0,-22 default -> 0,-2.3,22 -> 0,0,100 test -> 0,-5,22 -> 0,-17,22
-> ...), each time invalidating whatever `d` was solved by pivot_calibration.py
against the PREVIOUS live value, with no error or warning anywhere -- this
was the root cause of the "green tip dot sometimes accurate, sometimes not"
symptom that persisted across multiple sessions. There is no ROS-writable
topic for TipInEE (only a read-only publisher), so it can't be enforced from
outside the GUI; this module is the enforcement layer on the Python side:
every pivot/demo script that depends on a specific TipInEE should call
`assert_tip_inee(expected)` right after ROS init, before doing anything else.

CANONICAL VALUE (2026-08-25, v6): TipInEE = (0.82968, -2.84431,
-13.02703) mm. This is v5's Abar = (0.03777, 0.73839, -21.74584), from
tip_offset_refit_zeromean.py's camera-free pinned-tip fit, PLUS a
camera-validated tool-frame correction of (0.792, -3.583, 8.719) mm found
by handeye_vs_tip_discriminate.py's click-based test (n=28 clicks,
orientation spread 94.7 deg, bootstrap sigma (0.29, 0.20, 0.17) mm).

Why the correction exists: v5's pinned-tip (CM_OFFSET_FINDER) calibration
was done under a STIFF FORCE-CONTROLLED hold while the operator rotated
the tool (robotTask.cpp CM_OFFSET_FINDER, from line 1906). A long thin
needle held that way can deflect elastically under the reaction load, so
the point the jig converges on is a loaded mechanical equilibrium, not
the true unloaded geometric tip. That is a bias in the tool-frame
CONSTANT part of the offset; the orientation-DEPENDENT part (K=-2.947,
the RCM2_1.RCMoffset kinematic term) is a differential measurement and is
not expected to share this bias, so d_const/K are unchanged in this
version -- only tip_inee_at_calib (Abar) shifted.

How this was found and cross-checked, so it is not "yet another eyeball
tune": handeye_vs_tip_discriminate.py fits TWO 3-parameter models to
(predicted-pixel, clicked-true-tip-pixel) residuals over many DIFFERENT
orientations -- a constant vector in the BASE frame (hand-eye error) vs a
constant vector in the TOOL frame (tip offset error) -- and lets the data
pick the winner. At n=28, spread=94.7 deg: as-is RMS 32.6 px, base-frame
model 10.0 px, tool-frame model 3.6 px. Tool-frame wins by 2.7x, so the
error is geometrically a tool-frame constant, not a hand-eye bias (ruling
out the hand-eye hypothesis that was originally suspected from
handeye_T.npz's fit_rms_mm=13.9mm / val_rms_px=nan alone). The result was
verified with a bootstrap (500 resamples, sigma <= 0.3 mm on each
component) and cross-checked via two independently-written code paths
before being saved (see tip_offset_apply_visual_correction.py). It also
agrees with the operator's own independent manual GUI tuning
((0.214, -2.0, -14.0), found by eye across a large rotation) to 1.43 mm --
two independent methods converging on the same answer.

Why the mean and not the pure geometry. The offset genuinely depends on
orientation, because robotModel.cpp hardcodes `RCM2_1.RCMoffset = 0` in
the parallel-linkage FK/Jacobian (see ComputeFwdKinsTip /
ComputeJacobianTip); the fitted K = -2.947 +/- 0.034 mm (87 sigma) is
estimating exactly that missing structural parameter, since
`RCMoffset*[s3,0,c3]` in base frame == `R @ (RCMoffset*[0,sin q4,cos q4])`
in EE frame. The pure tool geometry would be Abar - K*ubar =
(0.038, 0.043, -18.908). But firmware applies NO Python correction to its
own Jacobian pivot point (controlModeToolPoseVelocity) or to sclerainEE,
so feeding it the pure geometry would leave those internals biased by
|K*ubar| ~ 2.9 mm on average. Abar is the best single constant for them.
Once robotModel.cpp's RCMoffset is actually set to K, switch the GUI to
the pure geometry value above.

Earlier mistakes worth not repeating. (1) Fitting TipInEE by eye so the
green overlay dot looks right at ONE orientation: a hand-eye bias (base
frame) and a tip-offset bias (tool frame) are indistinguishable at a
single pose, so a single-pose tune "looks right" by coincidence and
drifts elsewhere. Always test across a wide orientation sweep, or better,
use handeye_vs_tip_discriminate.py's two-model fit to let the geometry
decide which frame the error is actually in. (2) Selecting an offset
model on scatter/LOO alone: the pinned-tip location is itself a free
parameter, so extra model parameters can absorb a constant bias while
scatter still improves. The n=1203 "augmented" model scored the best
scatter (0.171 mm) while placing the tip 4.54 mm away -- do not use it.
(3) Trusting a camera-free pinned-tip fit as ground truth without also
checking it against real imagery: v5 assumed CM_OFFSET_FINDER's
force-controlled hold measures the true unloaded tip, which turned out
to be wrong by ~9.4 mm (mechanical flex under load) -- camera-free
precision (tight scatter) is not the same as camera-free ACCURACY.

The Python side applies K*(u(q4) - ubar) on top of tip_inee_at_calib, so
it cannot translate the tip as a side effect of fitting the orientation
shape. If you change the GUI Tip Offset again you MUST update this
constant and tip_offset_rcmoffset.npz's tip_inee_at_calib to match.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import numpy as np
import rospy
from geometry_msgs.msg import Vector3

CANONICAL_TIP_INEE = None  # Historical values are documentation only.
DEFAULT_TOL_MM = 0.5


def read_live_tip_inee(timeout_s=3.0):
    """Blocks until one message is received on /SHER21/eye_robot/TipInEE,
    or raises TimeoutError. Safe to call even if rospy.init_node() was
    already called elsewhere (does not re-init)."""
    result = {"v": None}

    def _cb(msg):
        result["v"] = np.array([msg.x, msg.y, msg.z])

    sub = rospy.Subscriber("/SHER21/eye_robot/TipInEE", Vector3, _cb, queue_size=1)
    t0 = rospy.get_time() if rospy.core.is_initialized() else __import__("time").time()
    r = rospy.Rate(20) if rospy.core.is_initialized() else None
    import time
    start = time.time()
    while result["v"] is None and (time.time() - start) < timeout_s:
        if r is not None:
            try:
                r.sleep()
            except Exception:
                time.sleep(0.05)
        else:
            time.sleep(0.05)
    sub.unregister()
    if result["v"] is None:
        raise TimeoutError("No /SHER21/eye_robot/TipInEE message received in %.1fs -- "
                            "is ROS bring-up running?" % timeout_s)
    return result["v"]


def assert_tip_inee(expected=None, tol_mm=DEFAULT_TOL_MM, context=""):
    """Read the LIVE TipInEE and hard-abort (raise SystemExit) if it doesn't
    match `expected` within `tol_mm`. Call this at the top of any script
    whose numeric results (pivot d, precision_log entries, recorded demos)
    are only valid for a specific TipInEE. This is the single choke point
    that stops a silent GUI TipInEE change from silently corrupting data."""
    from runtime_session import expected_tip
    expected = expected_tip(expected)
    if not np.isfinite(tol_mm) or tol_mm <= 0:
        raise ValueError("positive finite TipInEE tolerance required")
    live = read_live_tip_inee()
    if np.asarray(live).shape != (3,) or not np.isfinite(live).all():
        raise SystemExit("Invalid live TipInEE")
    err = np.abs(live - np.asarray(expected))
    if np.any(err > tol_mm):
        raise SystemExit(
            "\n!!! TipInEE MISMATCH -- ABORTING%s !!!\n"
            "  expected (current-session model) = %s mm\n"
            "  live GUI value       = %s mm\n"
            "  |diff|               = %s mm  (tol=%.2fmm)\n"
            "  Go set the GUI Tip Offset back to %s and do NOT touch it again,\n"
            "  or if you deliberately changed tools/offset, you must redo\n"
            "  pivot_calibration.py against the new value before trusting any\n"
            "  tip position from this codebase.\n"
            % (" (%s)" % context if context else "", expected, live, err, tol_mm, expected))
    print("[tip_inee_guard] OK: live TipInEE = %s mm matches current-session model %s mm (tol=%.2fmm)"
          % (live, expected, tol_mm))
    return live


if __name__ == "__main__":
    import sys
    if not rospy.core.is_initialized():
        rospy.init_node("tip_inee_guard_check", anonymous=True, disable_signals=True)
    try:
        assert_tip_inee()
        print("PASS")
    except SystemExit as e:
        print(e)
        sys.exit(1)
