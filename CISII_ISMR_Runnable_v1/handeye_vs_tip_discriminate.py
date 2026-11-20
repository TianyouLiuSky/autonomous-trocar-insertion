#!/usr/bin/env python3
"""
Decide WHERE the green-dot error lives: hand-eye, or the tip offset.

The overlay dot is base_to_pixel(cur_tip(), R_he, t_he, intr). When it does not
land on the physical tool tip, the error can sit in the camera->base transform
or in the tip offset, and a single view cannot tell them apart. Tuning either
one against the picture "works" at the pose you tuned and drifts everywhere
else -- which is exactly the symptom that has been chased for several sessions.

The two candidates are geometrically distinguishable, and that is the whole
point of this script:

    hand-eye translation error  ->  a constant vector in the BASE frame
                                    (does NOT rotate with the tool)
    tip offset error            ->  a constant vector in the TOOL frame
                                    (DOES rotate with the tool)

So: park the tool at several poses with genuinely different ORIENTATIONS, click
the true tip in the image at each, then fit both 3-parameter models to the pixel
residuals and see which one explains them. Same parameter count, so the residual
comparison is apples-to-apples.

This also produces the number that handeye_T.npz is currently missing: its
val_rms_px is nan, i.e. the live calibration has never been scored on data it
was not fit to. Model 0 below ("as-is") IS that held-out score.

Deliberately identical to the demo's chain -- same handeye_T.npz, same
load_best_intrinsics(), same rs2_project_point_to_pixel (distortion in the
loop, per the 2026-08-23 finding that a pinhole-only projection made the
overlay drift toward the image edges), same zero-mean rcmoffset tip model. If
any of those differed, the measured residual would not be attributable.

Run it with the demo STOPPED (the demo holds the RealSense exclusively), and
with the GUI Tip Offset already at tip_inee_guard.CANONICAL_TIP_INEE -- the
guard enforces that, since the measurement is only valid for one TipInEE.

Keys:  click = record this pose        u = undo last     f = fit/report
       s = save samples to csv         q = quit (stops the robot first)
       i/k = pitch +/-   j/l = yaw +/-   o/p = roll +/-   (hold to rotate)
       [ / ] = jog speed down / up       space = stop

The rotation keys publish on the SAME topics the main demo uses
(/SHER21/eyerobot2/desiredTipVelocities[Angular]), pure angular velocity, no
RCM-hold compensation -- the tip is allowed to drift; only the ORIENTATION
change matters here, not keeping the tip fixed. robotTask.cpp's ControlMode
is whatever it already was (unaffected by killing the demo process), so this
only works if that mode accepts velocity commands, same as the demo did.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path

import datetime
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Transform, Vector3
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intr_util import load_best_intrinsics  # noqa: E402
from tip_inee_guard import assert_tip_inee  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HANDEYE_NPZ = os.environ.get("HANDEYE_NPZ", data_path("handeye_T.npz"))
RCM_NPZ = os.environ.get("TIP_OFFSET_RCMOFFSET_NPZ",
                         data_path("tip_offset_rcmoffset.npz"))
FRAMEEE_TOPIC = os.environ.get("FRAMEEE_TOPIC", "/SHER21/eye_robot/FrameEE")
VEL_TOPIC = os.environ.get("VEL_TOPIC", "/SHER21/eyerobot2/desiredTipVelocities")
ANG_TOPIC = os.environ.get("ANG_TOPIC", "/SHER21/eyerobot2/desiredTipVelocitiesAngular")

# Pure-rotation jog, same pulse mechanism as the main demo (hold a key -> OS
# key-repeat keeps re-arming a short pulse). No RCM-hold compensation: we only
# need orientation diversity, the tip is free to drift while jogging.
PULSE_S = float(os.environ.get("PULSE_S", "0.20"))
ASPEED_DEFAULT = float(os.environ.get("JOG_ASPEED", "0.15"))   # rad/s
ASPEED_MAX = float(os.environ.get("JOG_AMAX", "0.4"))
ANG_KEYS = {ord('i'): (1, 0, 0), ord('k'): (-1, 0, 0),
            ord('j'): (0, 1, 0), ord('l'): (0, -1, 0),
            ord('o'): (0, 0, 1), ord('p'): (0, 0, -1)}

MIN_SAMPLES = 6
GOOD_SAMPLES = 10
# Below this orientation spread the two models are not separable and any verdict
# would be meaningless -- the script says so rather than pretending otherwise.
MIN_SPREAD_DEG = 20.0

ZOOM = 6          # magnification of the click-assist inset
ZOOM_SRC = 40     # source window side, px

_ee = {"p": None, "q": None}


def _cb_ee(m):
    _ee["p"] = np.array([m.translation.x, m.translation.y, m.translation.z], float)
    _ee["q"] = np.array([m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w], float)


def quat_to_R(q):
    """Same expansion the demo uses, so R is bit-identical to its overlay."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def base_to_pixel(p_base, R_he, t_he, intr):
    c = R_he.T.dot(p_base - t_he)
    if c[2] <= 1e-3:
        return None
    u, v = rs.rs2_project_point_to_pixel(intr, [float(c[0]), float(c[1]), float(c[2])])
    return np.array([u, v], float)


def cam_depth_mm(p_base, R_he, t_he):
    return float(R_he.T.dot(p_base - t_he)[2])


class TipModel(object):
    """Zero-mean rcmoffset model, matching the demo's tip_offset_vec()."""

    def __init__(self, path):
        z = np.load(path)
        self.d_const = np.asarray(z["d_const"], float).ravel()
        self.K = float(z["K"])
        self.at_calib = np.asarray(z["tip_inee_at_calib"], float).ravel()

    def offset_base(self, R):
        q4 = np.arctan2(-R[1, 2], R[1, 1])
        return R.dot(self.d_const + self.K * np.array([0.0, np.sin(q4), np.cos(q4)]))

    def tip(self, p, R):
        return p + self.offset_base(R)


def fit_models(samples, R_he, t_he, intr):
    """samples: list of dicts with p, R, tip_pred (base mm), click (px)."""
    clicks = np.array([s["click"] for s in samples])

    def resid(corr_base_fn, x):
        out = []
        for s in samples:
            px = base_to_pixel(s["tip_pred"] + corr_base_fn(s, x), R_he, t_he, intr)
            if px is None:
                out += [1e3, 1e3]
            else:
                out += list(px - s["click"])
        return np.asarray(out)

    def rms(r):
        r = np.asarray(r).reshape(-1, 2)
        return float(np.sqrt((r ** 2).sum(axis=1).mean()))

    out = {}
    out["as_is"] = {"rms_px": rms(resid(lambda s, x: np.zeros(3), np.zeros(3))),
                    "vec": None, "label": "no correction (current pipeline)"}

    # rs2_project_point_to_pixel is float32 internally, so least_squares' default
    # finite-difference step (~1e-8 mm) is swallowed entirely and the numeric
    # Jacobian comes back all zeros -- the solver then "converges" at nfev=1
    # without ever moving. 0.01 mm steps move the projection by ~0.02 px, which
    # float32 resolves fine.
    fd = dict(diff_step=1e-2, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    rA = least_squares(lambda x: resid(lambda s, xx: xx, x), np.zeros(3), **fd)
    out["base"] = {"rms_px": rms(rA.fun), "vec": rA.x,
                   "label": "constant in BASE frame  -> hand-eye translation"}

    rB = least_squares(lambda x: resid(lambda s, xx: s["R"].dot(xx), x),
                       np.zeros(3), **fd)
    out["tool"] = {"rms_px": rms(rB.fun), "vec": rB.x,
                   "label": "constant in TOOL frame  -> tip offset"}
    return out, clicks


def main():
    rospy.init_node("handeye_vs_tip_discriminate", anonymous=True, disable_signals=True)
    rospy.Subscriber(FRAMEEE_TOPIC, Transform, _cb_ee)

    assert_tip_inee(context="handeye_vs_tip_discriminate.py -- the measured "
                            "residual is only attributable for one TipInEE")

    z = np.load(HANDEYE_NPZ)
    R_he, t_he = np.asarray(z["R"], float), np.asarray(z["t"], float).ravel()
    print("hand-eye: %s" % HANDEYE_NPZ)
    for k in ("fit_rms_px", "loo_rms_px", "val_rms_px", "fit_rms_mm"):
        if k in z.files:
            print("   %-12s = %.3f" % (k, float(z[k])))
    tip = TipModel(RCM_NPZ)
    print("tip model: %s  (K=%.4f, calib TipInEE=%s)"
          % (os.path.basename(RCM_NPZ), tip.K, np.round(tip.at_calib, 4)))

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr = load_best_intrinsics(profile)
    for _ in range(15):
        pipe.wait_for_frames()

    pub = rospy.Publisher(VEL_TOPIC, Vector3, queue_size=1)
    pub_a = rospy.Publisher(ANG_TOPIC, Vector3, queue_size=1)

    def zero_vel():
        pub.publish(Vector3(0, 0, 0))
        pub_a.publish(Vector3(0, 0, 0))

    aspeed = ASPEED_DEFAULT
    jog_ang = None
    jog_until = 0.0

    state = {"click": None, "mouse": (0, 0)}

    def on_mouse(ev, x, y, flags, param):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "handeye vs tip -- click the REAL tool tip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 960)
    cv2.setMouseCallback(win, on_mouse)

    samples = []
    print("\nPark the tool, then click the real tip. CHANGE ORIENTATION between "
          "samples -- that is what separates the two models.")
    print("Aim for >= %d samples spanning >= %.0f deg." % (GOOD_SAMPLES, MIN_SPREAD_DEG))

    while True:
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data()).copy()

        p, q = _ee["p"], _ee["q"]
        pred_px = None
        if p is not None and q is not None:
            R = quat_to_R(q)
            tip_base = tip.tip(p, R)
            pred_px = base_to_pixel(tip_base, R_he, t_he, intr)
            if pred_px is not None:
                u, v = int(round(pred_px[0])), int(round(pred_px[1]))
                cv2.circle(img, (u, v), 6, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (u, v), 6, (255, 255, 255), 1, cv2.LINE_AA)

        if (state["click"] is not None and pred_px is not None
                and time.time() - state.get("last_sample_t", 0.0) > 0.4):
            cx, cy = state["click"]
            state["click"] = None
            state["last_sample_t"] = time.time()
            samples.append({"p": p.copy(), "q": q.copy(), "R": quat_to_R(q),
                            "tip_pred": tip_base.copy(),
                            "click": np.array([cx, cy], float),
                            "pred": pred_px.copy(),
                            "depth_mm": cam_depth_mm(tip_base, R_he, t_he)})
            e = np.linalg.norm(pred_px - np.array([cx, cy], float))
            mmpx = samples[-1]["depth_mm"] / float(intr.fx)
            print("  sample %2d: err = %5.1f px  (~%.2f mm at %.0f mm depth)"
                  % (len(samples), e, e * mmpx, samples[-1]["depth_mm"]))

        # click-assist inset: a thin needle is the accuracy floor here
        mx, my = state["mouse"]
        h, w = img.shape[:2]
        x0, y0 = max(0, mx - ZOOM_SRC // 2), max(0, my - ZOOM_SRC // 2)
        x1, y1 = min(w, x0 + ZOOM_SRC), min(h, y0 + ZOOM_SRC)
        if x1 > x0 and y1 > y0:
            patch = cv2.resize(img[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM,
                               interpolation=cv2.INTER_NEAREST)
            ph, pw = patch.shape[:2]
            cv2.line(patch, (pw // 2, 0), (pw // 2, ph), (0, 0, 255), 1)
            cv2.line(patch, (0, ph // 2), (pw, ph // 2), (0, 0, 255), 1)
            img[0:min(ph, h), w - min(pw, w):w] = patch[0:min(ph, h), 0:min(pw, w)]

        spread = 0.0
        if len(samples) >= 2:
            rv = [Rot.from_matrix(s["R"]).as_rotvec() for s in samples]
            spread = float(np.rad2deg(np.linalg.norm(np.ptp(rv, axis=0))))
        jogging = time.time() < jog_until
        cv2.putText(img, "n=%d  spread=%.0fdeg  aspeed=%.2f  %s  [click/u/f/s/q | i/k j/l o/p rotate]"
                    % (len(samples), spread, aspeed, "JOG" if jogging else ""),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)

        k = cv2.waitKey(20) & 0xFF
        now = time.time()

        if k in ANG_KEYS:
            jog_ang = ANG_KEYS[k]
            jog_until = now + PULSE_S
        if k == ord(' '):
            jog_ang = None
            jog_until = 0.0
            zero_vel()
        if k == ord(']'):
            aspeed = min(aspeed * 1.4, ASPEED_MAX)
        if k == ord('['):
            aspeed = max(aspeed / 1.4, 0.01)

        if now < jog_until and jog_ang is not None:
            pub.publish(Vector3(0, 0, 0))
            pub_a.publish(Vector3(*(c * aspeed for c in jog_ang)))
        else:
            jog_ang = None
            zero_vel()

        if k == ord('q'):
            zero_vel()
            break
        if k == ord('u') and samples:
            samples.pop()
            print("  undo -> n=%d" % len(samples))
        if k == ord('s') and samples:
            path = data_path("handeye_vs_tip_%s.csv"
                                % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            with open(path, "w") as fh:
                fh.write("px,py,pz,qx,qy,qz,qw,pred_u,pred_v,click_u,click_v,depth_mm\n")
                for s in samples:
                    fh.write(",".join("%.6f" % v for v in list(s["p"]) + list(s["q"])
                                      + list(s["pred"]) + list(s["click"])
                                      + [s["depth_mm"]]) + "\n")
            print("  saved -> %s" % path)
        if k == ord('f'):
            report(samples, R_he, t_he, intr, spread)

    zero_vel()
    pipe.stop()
    cv2.destroyAllWindows()
    if samples:
        report(samples, R_he, t_he, intr, spread)


def report(samples, R_he, t_he, intr, spread):
    n = len(samples)
    print("\n" + "=" * 68)
    if n < MIN_SAMPLES:
        print("only %d samples -- need >= %d." % (n, MIN_SAMPLES))
        return
    res, _ = fit_models(samples, R_he, t_he, intr)
    mmpx = float(np.mean([s["depth_mm"] for s in samples])) / float(intr.fx)
    print("n=%d   orientation spread=%.0f deg   scale ~%.3f mm/px" % (n, spread, mmpx))
    print("\n%-46s %-11s %-9s" % ("model (3 params each)", "RMS px", "RMS mm"))
    for key in ("as_is", "base", "tool"):
        r = res[key]
        print("%-46s %-11.2f %-9.3f" % (r["label"], r["rms_px"], r["rms_px"] * mmpx))
    # One camera constrains only the component across the line of sight; a shift
    # ALONG the viewing ray barely moves a pixel. For the base-frame model that
    # blind direction is the same at every pose, so its along-ray part is
    # essentially unconstrained and |delta| alone is not trustworthy. The
    # tool-frame model is not hurt the same way, because R_i rotates the vector
    # into the observable plane differently at each pose.
    cax = R_he.dot(np.array([0.0, 0.0, 1.0]))
    cax /= np.linalg.norm(cax)
    print("\nfitted correction vectors (mm):")
    for key, name in (("base", "BASE"), ("tool", "TOOL")):
        v = res[key]["vec"]
        par = float(v.dot(cax))
        perp = np.linalg.norm(v - par * cax)
        print("   %s-frame delta = %s   |.| = %.2f mm"
              % (name, np.round(v, 3), np.linalg.norm(v)))
        print("        across line of sight %.2f mm (observable) | "
              "along it %.2f mm (%s)"
              % (perp, par,
                 "unconstrained -- ignore" if key == "base" else "weakly constrained"))

    print("\n--- verdict")
    print("  as-is RMS %.2f px is the held-out score handeye_T.npz lacks "
          "(its val_rms_px is nan)." % res["as_is"]["rms_px"])
    if spread < MIN_SPREAD_DEG:
        print("  !! orientation spread %.0f deg < %.0f -- the two models are NOT "
              "separable from this data. Collect poses with genuinely different "
              "tool orientations before believing any verdict below."
              % (spread, MIN_SPREAD_DEG))
        return
    rb, rt = res["base"]["rms_px"], res["tool"]["rms_px"]
    ratio = max(rb, rt) / max(min(rb, rt), 1e-9)
    if ratio < 1.3:
        print("  Neither model wins (%.2f vs %.2f px, ratio %.2f). The error is "
              "not a single constant in either frame -- suspect intrinsics/"
              "distortion, or a pose-dependent kinematic term (robotModel.cpp's "
              "RCM2_1.RCMoffset is still hardcoded to 0)." % (rb, rt, ratio))
    elif rb < rt:
        print("  HAND-EYE. A base-frame constant explains the residual (%.2f px) "
              "far better than any tool-frame constant (%.2f px)." % (rb, rt))
        print("  -> Recalibrate hand-eye, with a held-out validation set this time. "
              "Do NOT touch TipInEE: it is fixed by camera-free pinned-tip data to "
              "0.02 mm, and a base-frame error cannot be cancelled by a tool-frame "
              "vector except at one single orientation.")
        print("  Note the base model also gets a free unobservable direction (along "
              "the line of sight), so it is mildly favoured here; treat a narrow "
              "win with more caution than a wide one.")
    else:
        print("  TIP OFFSET. A tool-frame constant of %.2f mm explains the residual "
              "(%.2f px) far better than any base-frame constant (%.2f px)."
              % (np.linalg.norm(res["tool"]["vec"]), rt, rb))
        print("  -> The pinned-tip calibration converged on a point that is not the "
              "needle tip (flexed needle, or contact at a shoulder above the tip). "
              "Redo the pivot with verified tip contact; hand-eye is not the issue.")
    print("=" * 68)


if __name__ == "__main__":
    main()
