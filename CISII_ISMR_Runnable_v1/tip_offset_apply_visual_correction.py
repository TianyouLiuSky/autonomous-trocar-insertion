#!/usr/bin/env python3
"""
Apply the camera-validated tool-frame correction found by
handeye_vs_tip_discriminate.py on top of the existing zero-mean rcmoffset
tip model, WITHOUT re-deriving the orientation-dependent shape (d_const, K).

Why only the constant shifts. The 2026-08-25 pinned-tip (CM_OFFSET_FINDER)
calibration was done under a stiff FORCE-CONTROLLED hold while the operator
rotated the tool -- see robotTask.cpp's CM_OFFSET_FINDER (from line 1906).
A long thin needle held that way can deflect elastically under the reaction
load, so the point the jig converges on is a loaded mechanical equilibrium,
offset from the true unloaded geometric tip by a roughly constant vector in
the TOOL frame. That is a bias in the MEAN offset, not in how it varies with
orientation (q4) -- the K*[0,sin q4,cos q4] term is a differential measurement
of the robot's own missing RCM2_1.RCMoffset kinematic parameter and is not
expected to share this bias. So: keep d_const/K, shift only tip_inee_at_calib.

Evidence this is real and not a fluke, from handeye_vs_tip_discriminate.py's
click-based measurement (camera + human-judged true tip location, NOT
vision-only, NOT camera-free pivot):
    n=28 clicks, orientation spread=94.7 deg
    as-is RMS=32.6 px  |  base-frame-constant fit RMS=10.0 px  |
    tool-frame-constant fit RMS=3.6 px   (tool wins by 2.7x -> not hand-eye)
    bootstrap (300 resamples) tool-frame delta std = (0.29, 0.20, 0.17) mm
    -> resulting new baseline agrees with the operator's INDEPENDENT manual
       GUI tuning (0.214, -2.0, -14.0) to 1.43 mm -- two independent methods
       (least-squares click fit vs. human eyeball-across-rotation tuning)
       converging on the same answer is the strongest evidence in this whole
       debugging session.

Run with no args; edit SRC_CSV / OLD_NPZ below if repeating this later with a
fresh discriminator run.
"""
import os

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_CSV = os.path.join(HERE, "handeye_vs_tip_20260825_180754.csv")
LIVE_NPZ = os.path.join(HERE, "tip_offset_rcmoffset.npz")
HANDEYE_NPZ = os.path.join(HERE, "handeye_T.npz")


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def main():
    import csv
    import pyrealsense2 as rs
    from scipy.spatial.transform import Rotation as Rot

    z_he = np.load(HANDEYE_NPZ)
    R_he, t_he = np.asarray(z_he["R"], float), np.asarray(z_he["t"], float).ravel()

    intr = rs.intrinsics()
    intr.width, intr.height = 640, 480
    intr.fx, intr.fy = 430.64, 430.28
    intr.ppx, intr.ppy = 310.64, 245.56
    intr.model = rs.distortion.inverse_brown_conrady
    intr.coeffs = [0.0] * 5

    z_old = np.load(LIVE_NPZ)
    d_const = np.asarray(z_old["d_const"], float).ravel()
    K = float(z_old["K"])
    ubar = np.asarray(z_old["ubar"], float).ravel() if "ubar" in z_old.files else None
    old_abar = np.asarray(z_old["tip_inee_at_calib"], float).ravel()

    def q4_of(R):
        return np.arctan2(-R[1, 2], R[1, 1])

    def tip_offset_vec(R):
        # NOTE: does NOT add old_abar. p (below) is FrameEE.translation, which
        # firmware already computed via ComputeFwdKinsTip(joints, TipInEE_live)
        # with TipInEE_live == old_abar (enforced by tip_inee_guard before this
        # data was collected) -- old_abar is already baked into p. This must
        # mirror handeye_vs_tip_discriminate.py's TipModel.tip() EXACTLY:
        #     offset_base(R) = R.dot(d_const + K*[0,sin q4,cos q4])
        #     tip(p, R)       = p + offset_base(R)
        # An earlier version of this script wrongly added old_abar a second
        # time here, which silently produced a corrupted tip_inee_at_calib
        # (caught before it reached the GUI/firmware -- see git history for
        # the wrong intermediate value and why it looked deceptively plausible).
        q4 = q4_of(R)
        return R.dot(d_const + K * np.array([0.0, np.sin(q4), np.cos(q4)]))

    rows = []
    with open(SRC_CSV) as f:
        for r in csv.DictReader(f):
            p = np.array([float(r["px"]), float(r["py"]), float(r["pz"])])
            q = np.array([float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])])
            click = np.array([float(r["click_u"]), float(r["click_v"])])
            R = quat_to_R(q)
            rows.append({"R": R, "tip_pred": p + tip_offset_vec(R), "click": click})
    n = len(rows)

    def base_to_pixel(p_base):
        c = R_he.T.dot(p_base - t_he)
        u, v = rs.rs2_project_point_to_pixel(intr, [float(c[0]), float(c[1]), float(c[2])])
        return np.array([u, v], float)

    def resid(x):
        return np.concatenate([base_to_pixel(s["tip_pred"] + s["R"].dot(x)) - s["click"]
                               for s in rows])

    fit = least_squares(resid, np.zeros(3), diff_step=1e-2, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    delta = fit.x
    rms = float(np.sqrt((fit.fun ** 2).reshape(-1, 2).sum(axis=1).mean()))
    print("n=%d  tool-frame delta = %s  |.|=%.3f mm  fit RMS=%.2f px"
          % (n, np.round(delta, 4), np.linalg.norm(delta), rms))

    # Cross-check against handeye_vs_tip_discriminate.py's OWN TipModel/fit_models
    # (independent code path) before trusting this number for anything that
    # touches the GUI/firmware.
    disc_src = open(os.path.join(HERE, "handeye_vs_tip_discriminate.py")).read()
    disc_head = disc_src.split("def main()")[0]
    for a in ("import rospy", "from geometry_msgs.msg import Transform, Vector3",
              "from tip_inee_guard import assert_tip_inee  # noqa: E402"):
        disc_head = disc_head.replace(a, "")
    DM = {"__file__": os.path.join(HERE, "handeye_vs_tip_discriminate.py")}
    exec(compile(disc_head, "discriminator_module", "exec"), DM)
    disc_rows = []
    with open(SRC_CSV) as f:
        for r in csv.DictReader(f):
            p = np.array([float(r["px"]), float(r["py"]), float(r["pz"])])
            q = np.array([float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])])
            click = np.array([float(r["click_u"]), float(r["click_v"])])
            R = DM["quat_to_R"](q)
            disc_rows.append({"R": R, "tip_pred": p + R.dot(d_const + K * np.array(
                [0.0, np.sin(np.arctan2(-R[1, 2], R[1, 1])),
                 np.cos(np.arctan2(-R[1, 2], R[1, 1]))])), "click": click})
    disc_res, _ = DM["fit_models"](disc_rows, R_he, t_he, intr)
    cross_delta = disc_res["tool"]["vec"]
    agree = np.linalg.norm(cross_delta - delta)
    print("cross-check via discriminator's own fit_models(): delta=%s  |diff from above|=%.4f mm"
          % (np.round(cross_delta, 4), agree))
    if agree > 0.05:
        raise SystemExit("ABORT: the two independent computations of the tool-frame "
                         "delta disagree by %.3f mm -- refusing to save. Investigate "
                         "before touching tip_offset_rcmoffset.npz." % agree)
    print("cross-check OK (agree to %.4f mm) -- proceeding to save." % agree)

    rng = np.random.RandomState(0)
    boots = []
    for _ in range(500):
        idx = rng.randint(0, n, n)
        sub = [rows[i] for i in idx]
        rb = least_squares(lambda x: np.concatenate(
            [base_to_pixel(s["tip_pred"] + s["R"].dot(x)) - s["click"] for s in sub]),
            np.zeros(3), diff_step=1e-2, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        boots.append(rb.x)
    boots = np.array(boots)
    sigma = boots.std(axis=0)
    print("bootstrap std (n=500) = %s mm" % np.round(sigma, 3))

    new_abar = old_abar + delta
    print("\nold Abar = %s" % np.round(old_abar, 5))
    print("new Abar = %s   (GUI Tip Offset must be set to this)" % np.round(new_abar, 5))

    # Sanity bound: the correction should be a plausible pivot-rig bias
    # (mm-scale, tip-frame Z should stay negative/shaft-like), not a sign flip
    # or a value that would put the "tip" behind the EE origin.
    if new_abar[2] >= 0 or abs(np.linalg.norm(delta) - 9.4) > 3.0:
        raise SystemExit("ABORT: new Abar=%s / |delta|=%.2f mm looks physically "
                         "implausible -- refusing to save." % (new_abar, np.linalg.norm(delta)))

    import shutil
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(HERE, "tip_offset_rcmoffset_prev_%s.npz" % stamp)
    shutil.copy2(LIVE_NPZ, bak)
    print("backed up previous -> %s" % bak)

    np.savez(
        LIVE_NPZ,
        d_const=d_const,
        K=np.array([K]),
        tip_inee_at_calib=new_abar,
        ubar=ubar if ubar is not None else np.zeros(3),
        loo_rms_mm=z_old["loo_rms_mm"] if "loo_rms_mm" in z_old.files else np.array([np.nan]),
        loo_rms_rigid_mm=z_old["loo_rms_rigid_mm"] if "loo_rms_rigid_mm" in z_old.files else np.array([np.nan]),
        visual_correction_mm=delta,
        visual_correction_sigma_mm=sigma,
        visual_correction_n=np.array([n]),
        visual_correction_spread_deg=np.array([94.7]),
        visual_correction_rms_px=np.array([rms]),
        visual_correction_source_csv=os.path.basename(SRC_CSV),
        visual_correction_note=np.array(
            "pivot (CM_OFFSET_FINDER) baseline biased ~9.4mm vs true optical tip, "
            "likely elastic needle flex under force-controlled hold; corrected via "
            "handeye_vs_tip_discriminate.py click-based fit, n=28, spread=94.7deg; "
            "agrees with operator's independent manual GUI tuning to 1.43mm"),
    )
    print("saved -> %s" % LIVE_NPZ)


if __name__ == "__main__":
    main()
