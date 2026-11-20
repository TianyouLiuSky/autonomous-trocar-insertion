#!/usr/bin/env python3
"""
rcm_definition.py -- the code that defines the RCM (remote centre of motion)
in the autonomous bi-planar scleral trocar insertion.

Author: Wenchao Feng (JHU LCSR).

Note: E is placed from the microscope calibration only. The code does not
compensate the constant base-frame offset that was estimated afterwards.

Two RCMs:

  1. SURFACE RCM  E   the entry point on the sclera.
       Original: try_lock_microscope_rcm_live()   (lines ~3670-3834)
                 -> writes _vision_rcm["p"], S.rcm_source = "vision"
                 cur_rcm() / rcm_ok()              (lines ~2961-2971)
       Used by the oblique TUNNEL. Initial ORIENT pivots about the current hover tip, not E.

  2. INTERNAL RCM P   the tip position at the end of the tunnel.
       Original: do_tunnel()                      (lines ~4092-4120)
                 -> S.internal_rcm = cur_tip().copy()
       Used by REORIENT/PLUNGE ('p' key, lines ~6408-6425): the needle is
       rotated to -n about P (not about E) and advanced PLUNGE_DEPTH_MM.

Supporting pieces also extracted:
  - the tip position model with the wrist RCM-offset term   (tip_offset_vec, ~2885-2900)
  - the pivot compensation v_lin = omega x (EE - RCM)       (auto_orient_about_rcm, ~5944-5948)
  - the limbus-circle target point in the microscope image  (mic_perception.point_on_plug_circle)

Run `python3 rcm_definition.py` for a numerical self-test with synthetic data.
"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import numpy as np

# ----------------------------------------------------------------------------
# Constants (defaults from the original file)
# ----------------------------------------------------------------------------
TUNNEL_DEPTH_MM = 1.5            # original line 1258 (env TUNNEL_DEPTH_MM)
MIC_FRAME_W, MIC_FRAME_H = 1920, 1080   # original line 899
MIC_TARGET_EDGE_MARGIN_PX = 60.0        # original line 900
MIC_MARK_SMOOTH_N = 6                   # original line 950
MIC_MARK_MAX_JITTER_PX = 4.0            # original line 951


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def quat_to_R(q):
    """(x, y, z, w) -> 3x3 rotation matrix.  Original line 1786."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def point_on_plug_circle(plug_cx, plug_cy, radius_px, angle_deg):
    """Pixel position of the entry point: the point at (radius, angle) on the
    circle concentric with the limbus (plug).
    Original: microscope_autonomy/mic_perception.py, line 179."""
    th = np.radians(angle_deg)
    return (plug_cx + radius_px * np.cos(th), plug_cy + radius_px * np.sin(th))


# ----------------------------------------------------------------------------
# Tip position (needed to turn the end-effector pose into the tip, and so to
# set the internal RCM)
# ----------------------------------------------------------------------------
def tip_offset_vec(ee_quat, d_const, K):
    """Tool-tip offset in the base frame, 'rcmoffset' model (the default,
    TIP_MODEL=rcmoffset).  Original: tip_offset_vec(), lines 2885-2891, with
    d_const and K fitted by rcmoffset_fit.py.

        offset = R (d_const + K [0, sin q4, cos q4]),  q4 = atan2(-R12, R11)

    The K-term is the wrist's RCM offset, which the firmware fixes at 0."""
    R = quat_to_R(ee_quat)
    q4 = np.arctan2(-R[1, 2], R[1, 1])
    return R.dot(np.asarray(d_const, float) + K * np.array([0.0, np.sin(q4), np.cos(q4)]))


def cur_tip(ee_pos, ee_quat, d_const, K):
    """Tool tip in the base frame.  Original: cur_tip(), line 2899."""
    return np.asarray(ee_pos, float) + tip_offset_vec(ee_quat, d_const, K)


# ----------------------------------------------------------------------------
# 1. SURFACE RCM (entry point E) -- from the surgical microscope
# ----------------------------------------------------------------------------
def surface_rcm_from_microscope(target_px, px_history, A, z_plane_mm, calib_resid_mm):
    """Entry point E in the robot base frame.

    Original: try_lock_microscope_rcm_live(), lines 3806-3826 (the numerical
    part; the frame-freshness, lock-stability and in-view checks are kept
    below in `microscope_lock_gate`).

    target_px      smoothed (u, v) of the target in the microscope image.
                   In circle mode this is point_on_plug_circle(plug_cx,
                   plug_cy, radius_px, MIC_RCM_ANGLE_DEG), smoothed over the
                   last MIC_MARK_SMOOTH_N frames.
    px_history     (N, 2) array of those recent pixel positions.
    A              3x3 matrix from mic_extrinsic_calib.py (calib["A"], model
                   'affine3d'); only the first two rows of A·[u, v, 1] are used
                   (base-frame x, y). A 2x3 matrix also works.
    z_plane_mm     base-frame z of the phantom plane (calib["z_plane_mm"]).
                   NOTE: with MIC_Z_SOURCE=d405plane (the default) this z is
                   replaced afterwards by `override_z_from_d405_plane`.
    calib_resid_mm residuals of that calibration fit (calib["resid_mm"]).

    Returns dict(p=E [mm], jitter_mm, calib_rms_mm)."""
    A = np.asarray(A, float)
    xy = A.dot(np.array([target_px[0], target_px[1], 1.0]))
    p = np.array([xy[0], xy[1], float(z_plane_mm)], np.float64)
    calib_rms = float(np.sqrt((np.asarray(calib_resid_mm) ** 2).mean()))
    # temporal jitter: push the same pixel history through the same affine
    hist_px = np.asarray(px_history, float)
    hist_mm = (A[:, :2].dot(hist_px.T) + A[:, 2:3]).T
    mark_jitter_mm = float(np.linalg.norm(hist_mm - hist_mm.mean(axis=0), axis=1).std())
    return {"p": p, "jitter_mm": max(calib_rms, mark_jitter_mm), "calib_rms_mm": calib_rms,
            "mark_jitter_mm": mark_jitter_mm}


def override_z_from_d405_plane(rcm_p, z_plane):
    """Original lines 3970-3975 (MIC_Z_SOURCE=d405plane, the default):
    after the microscope lock, x and y stay as above and z
    is replaced by the height of the phantom face at that (x, y), taken from
    the plane fitted to the D405 depth patch around the limbus
    (d405_plane_z_at, lines 3836-3860; valid only if that plane was fitted
    within MIC_Z_NORMAL_MAX_AGE_MM = 3 mm of the current limbus position).
    If no valid plane exists, the fixed calibration z_plane_mm is kept."""
    if z_plane is None:
        return np.asarray(rcm_p, float)
    return np.array([rcm_p[0], rcm_p[1], float(z_plane)])


def microscope_lock_gate(n_samples, jitter_px, smoothed_px):
    """Conditions the original checks before it accepts a lock (lines 3742-3768):
    the rolling window must be full and its jitter below MIC_MARK_MAX_JITTER_PX
    ('locked'), and the target must lie MIC_TARGET_EDGE_MARGIN_PX inside the
    microscope view, so that the hole can be measured afterwards.
    Returns (ok, reason)."""
    if n_samples < MIC_MARK_SMOOTH_N or jitter_px is None or jitter_px >= MIC_MARK_MAX_JITTER_PX:
        return False, "target not stable yet (%d/%d samples, jitter=%s px)" % (
            n_samples, MIC_MARK_SMOOTH_N, jitter_px)
    m = MIC_TARGET_EDGE_MARGIN_PX
    if not (m <= smoothed_px[0] <= MIC_FRAME_W - m and m <= smoothed_px[1] <= MIC_FRAME_H - m):
        return False, "target outside the microscope view"
    return True, "ok"


# ----------------------------------------------------------------------------
# 2. INTERNAL RCM (tunnel end P)
# ----------------------------------------------------------------------------
def tunnel_goal_and_internal_rcm(tip_now, surface_rcm, tunnel_depth_mm=TUNNEL_DEPTH_MM):
    """Oblique TUNNEL goal and the direction it follows.

    Original: do_tunnel(), lines 4101-4111.  The insertion direction is
    recomputed from the real tip position through the real surface RCM:

        u    = (E - tip) / |E - tip|
        goal = E + L_t u            (L_t = TUNNEL_DEPTH_MM)

    When the move succeeds the original sets
        S.internal_rcm = cur_tip().copy()      (line 4115)
    i.e. the internal RCM P is the tip position measured at the end of the
    tunnel, not the commanded goal.  Pass the measured tip to
    `internal_rcm_after_tunnel`."""
    surface_rcm = np.asarray(surface_rcm, float)
    u = np.asarray(tip_now, float) - surface_rcm   # tip -> RCM, sign flipped below
    un = np.linalg.norm(u)
    if un < 1e-6:
        raise ValueError("cannot compute TUNNEL goal (tip == RCM)")
    u = -u / un
    return surface_rcm + u * tunnel_depth_mm, u


def internal_rcm_after_tunnel(measured_tip):
    """P = measured tip at the end of TUNNEL.  Original line 4115."""
    return np.asarray(measured_tip, float).copy()


def plunge_goal(internal_rcm, phantom_normal, plunge_depth_mm):
    """REORIENT about P, then PLUNGE along -n.  Original: 'p' key handler,
    lines 6410-6422: perp_dir = -n/|n|; the needle axis is rotated to perp_dir
    about `internal_rcm` (auto_orient_about_rcm(..., pivot_point=S.internal_rcm)),
    then goal = internal_rcm + perp_dir * PLUNGE_DEPTH_MM."""
    n = np.asarray(phantom_normal, float)
    n = n / np.linalg.norm(n)
    perp_dir = -n
    return np.asarray(internal_rcm, float) + perp_dir * plunge_depth_mm, perp_dir


# ----------------------------------------------------------------------------
# RCM-constrained motion: the velocity that keeps the shaft through the RCM
# ----------------------------------------------------------------------------
def pivot_linear_velocity(omega, ee_pos, rcm_pt):
    """Feed-forward linear velocity for a rotation `omega` about `rcm_pt`.

    Original: auto_orient_about_rcm(), lines 5944-5948 (compensate=True):
        lever = EE_origin - RCM          (the EE origin, not the tip)
        v_lin = omega x lever
    The original adds a closed-loop trim on top (ORIENT_HOLD_KP * (hold_point
    - tip_now), saturated at ORIENT_HOLD_MAX_MM_S) and clips |v_lin| to
    SPEED_MAX; those gains are environment constants in the original."""
    lever = np.asarray(ee_pos, float) - np.asarray(rcm_pt, float)
    return np.cross(np.asarray(omega, float), lever)


# ----------------------------------------------------------------------------
# Self-test (synthetic numbers, no robot)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # microscope -> E
    A = np.array([[0.10, 0.0, -90.0],       # 0.10 mm/px
                  [0.0, 0.10, -50.0]])
    rng = np.random.default_rng(0)
    plug_c, r_px, ang = (960.0, 540.0), 180.0, 25.0
    tgt = point_on_plug_circle(plug_c[0], plug_c[1], r_px, ang)
    hist = np.array(tgt) + rng.normal(0, 0.8, size=(MIC_MARK_SMOOTH_N, 2))
    ok, why = microscope_lock_gate(len(hist), float(hist.std(axis=0).mean()), hist.mean(axis=0))
    print("lock gate:", ok, why)
    E = surface_rcm_from_microscope(hist.mean(axis=0), hist, A, 40.0, [0.1, -0.1, 0.12, 0.05])
    print("surface RCM E  [mm]:", E["p"], " jitter:", round(E["jitter_mm"], 3), "mm")

    # tunnel -> P
    tip = E["p"] + np.array([0.0, -8.0, 6.0])        # hover position
    goal, u = tunnel_goal_and_internal_rcm(tip, E["p"])
    print("tunnel direction u:", u, " goal:", goal, " depth:", np.linalg.norm(goal - E["p"]), "mm")
    P = internal_rcm_after_tunnel(goal)              # assume the move reached the goal
    print("internal RCM P [mm]:", P)

    # reorient + plunge
    pg, perp = plunge_goal(P, [0.0, 0.0, 1.0], 3.0)
    print("plunge goal    [mm]:", pg, " direction:", perp)

    # pivot compensation: a pure rotation about E should leave the point E fixed
    omega = np.array([0.0, 0.05, 0.0])               # rad/s
    ee = E["p"] + np.array([0.0, 2.0, 12.0])
    v = pivot_linear_velocity(omega, ee, E["p"])
    print("v_lin:", v, " velocity of the RCM point (should be 0):",
          v + np.cross(omega, E["p"] - ee))
