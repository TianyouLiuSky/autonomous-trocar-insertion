#!/usr/bin/env python3
"""
visual_servo.py
---------------
Hand-tunable PID visual servoing for SHER 2.0.

No Jacobian. No calibration step. Pixel error → mm command via raw gains.
Sign and axis mapping are runtime-toggleable so you can fix wrong-direction
behavior without restarting. Tune with hand on the e-stop.

Control law:
    e = target_px - tip_px                    (2D pixel error)
    u = Kp·e + Ki·∫e dt + Kd·de/dt            (per-axis PID, image frame)
    Δq_xy = axis_map(u, sign_x, sign_y, swap)  → robot F_b xy (mm)
    clamp |Δq_xy| ≤ max_step
    no_rcm_move_to(current_pose with x,y replaced)

Subscribes:
  /ati/tool_tip_pixel       geometry_msgs/PointStamped   (z=1.0 valid)
  /ati/target_pixel         geometry_msgs/PointStamped

Uses SHERController (handles its own ROS init + FrameEE subscription).

Trackbars (in the cv2 window):
  Kp /10000        gain on error.       slider/10000 → mm/px
  Ki /100000       gain on integral.    slider/100000 → mm/(px·s)
  Kd /100000       gain on derivative.  slider/100000 → mm·s/px
  max_step *100    per-iter clamp.      slider/100   → mm
  deadband px      stop threshold.

Keys:
  s          start SERVO (must already be IDLE)
  SPACE / e  EMERGENCY STOP — back to IDLE, zero velocities
  1          flip x sign     (image u → robot ±x)
  2          flip y sign     (image v → robot ±y)
  3          swap axes       (toggle: image u→x or u→y)
  r          reset integral
  q / ESC    quit

Status display always shows the *candidate* Δq (what would be commanded if you
pressed s). Verify direction makes sense with the robot still in IDLE before
committing.
"""

import os
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped

# ── Robot import — ADAPT to actual SHERController file location ─────────────
try:
    from sher_controller import SHERController
    _ROBOT_AVAILABLE = True
except ImportError:
    print("[warn] SHERController not importable — running in DRY-RUN mode")
    _ROBOT_AVAILABLE = False

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ROBOT_NAME       = "SHER20"
TIP_TOPIC        = "/ati/tool_tip_pixel"
TARGET_TOPIC     = "/ati/target_pixel"

# loop
SERVO_RATE_HZ    = 5.0
SERVO_LIN_VEL    = 1.5       # mm/s passed to no_rcm_move_to
TIP_FRESHNESS_S  = 0.5       # halt servo if tip data older than this

# default gains (also initial trackbar positions). Start GENTLE.
KP_DEFAULT       = 0.0005    # mm/px
KI_DEFAULT       = 0.0       # mm/(px·s)
KD_DEFAULT       = 0.0       # mm·s/px
MAX_STEP_DEFAULT = 0.30      # mm per iter
DEADBAND_DEFAULT = 8         # px

# operating mode
DRY_RUN          = not _ROBOT_AVAILABLE   # set True manually to bench-test without robot

# UI
WIN_NAME = "ATI - Visual Servo (PID)"
WIN_W, WIN_H = 760, 720
DARK     = (20, 20, 20)
GRAY     = (200, 200, 200)
DIM      = (140, 140, 140)
GREEN    = (153, 204,  76)
BLUE     = (250, 170, 119)
AMBER    = ( 68, 170, 255)
RED      = ( 76,  76, 220)
PINK     = (180, 105, 255)
BORDER   = ( 68,  68,  68)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
# ─────────────────────────────────────────────────────────────────────────────


# ─── Perception ─────────────────────────────────────────────────────────────
class PerceptionInterface:
    def __init__(self):
        self._lock = threading.Lock()
        self._tip = None
        self._tgt = None
        rospy.Subscriber(TIP_TOPIC,    PointStamped, self._tip_cb, queue_size=1)
        rospy.Subscriber(TARGET_TOPIC, PointStamped, self._tgt_cb, queue_size=1)

    def _tip_cb(self, msg):
        with self._lock:
            self._tip = (msg.point.x, msg.point.y, msg.point.z >= 0.5, msg.header.stamp)

    def _tgt_cb(self, msg):
        with self._lock:
            self._tgt = (msg.point.x, msg.point.y, msg.point.z >= 0.5, msg.header.stamp)

    @staticmethod
    def _unpack(v, max_age_s):
        if v is None or not v[2]:
            return None
        if max_age_s is not None and (rospy.Time.now() - v[3]).to_sec() > max_age_s:
            return None
        return float(v[0]), float(v[1])

    def get_tip(self, max_age_s=None):
        with self._lock:
            v = self._tip
        return self._unpack(v, max_age_s)

    def get_target(self, max_age_s=None):
        with self._lock:
            v = self._tgt
        return self._unpack(v, max_age_s)


# ─── Robot ──────────────────────────────────────────────────────────────────
class RobotInterface:
    """SHERController wrapper. xy-only commands, holds z + orientation, yaw=0."""
    def __init__(self):
        if not DRY_RUN:
            self._ctrl = SHERController(robot_name=ROBOT_NAME)   # calls rospy.init_node
            self._dry_pose = None
        else:
            rospy.init_node("visual_servo_dryrun", anonymous=True, disable_signals=True)
            self._ctrl = None
            self._dry_pose = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])

    def get_full_pose(self):
        if DRY_RUN:
            return self._dry_pose.copy()
        return self._ctrl.get_current_pose()

    def get_pose_xy(self):
        p = self.get_full_pose()
        return None if p is None else (float(p[0]), float(p[1]))

    def move_xy_to(self, x_mm, y_mm, max_lin_vel=2.0):
        cur = self.get_full_pose()
        if cur is None:
            return False
        target = cur.copy()
        target[0], target[1], target[5] = x_mm, y_mm, 0.0
        if DRY_RUN:
            print(f"[DRY RUN] move_xy_to ({x_mm:+.3f}, {y_mm:+.3f}) mm")
            rospy.sleep(0.15)
            self._dry_pose = target
            return True
        return self._ctrl.no_rcm_move_to(target, max_linear_vel=max_lin_vel)

    def move_xy_relative(self, dx_mm, dy_mm, max_lin_vel=2.0):
        cur = self.get_pose_xy()
        if cur is None:
            return False
        return self.move_xy_to(cur[0] + dx_mm, cur[1] + dy_mm, max_lin_vel=max_lin_vel)

    def stop(self):
        if DRY_RUN:
            print("[DRY RUN] stop()")
            return
        self._ctrl._stop()


# ─── PID controller ─────────────────────────────────────────────────────────
class PIDServo:
    """
    PID on 2D pixel error. Output is robot xy displacement (mm).

    State carried across calls: integral (px·s), prev_err (px), prev_t (s).
    Anti-windup: integral only accumulates when the commanded step isn't clamped.
    Dead-band reset: when |e| < deadband, state is zeroed so a future re-engage
    starts clean.
    """
    def __init__(self):
        self.integral = np.zeros(2, dtype=float)
        self.prev_err = None
        self.prev_t   = None

    def reset(self):
        self.integral[:] = 0
        self.prev_err = None
        self.prev_t   = None

    def step(self, err_pix, params, commit):
        """
        err_pix : (eu, ev) or None
        params  : dict {kp, ki, kd, sign_x, sign_y, swap_axes, max_step_mm, deadband}
        commit  : True = update internal state (in SERVO); False = preview only (in IDLE)
        Returns (dq_xy_mm or None, status_str).
        """
        if err_pix is None:
            return None, "no error"

        e = np.array(err_pix, dtype=float)
        mag = float(np.linalg.norm(e))
        if mag < params["deadband"]:
            if commit:
                self.reset()
            return None, f"in deadband (|e|={mag:.1f}px)"

        now = rospy.get_time()
        if self.prev_err is not None and self.prev_t is not None:
            dt = max(now - self.prev_t, 1e-3)
            de = (e - self.prev_err) / dt
        else:
            dt = 0.0
            de = np.zeros(2)

        # PID output in image-frame pixels-scaled-to-mm
        u = params["kp"] * e + params["ki"] * self.integral + params["kd"] * de

        # axis mapping (image u,v → robot x,y)
        if params["swap_axes"]:
            dq = np.array([u[1] * params["sign_x"], u[0] * params["sign_y"]])
        else:
            dq = np.array([u[0] * params["sign_x"], u[1] * params["sign_y"]])

        # clamp
        step_mag = float(np.linalg.norm(dq))
        saturated = False
        if step_mag > params["max_step_mm"]:
            dq *= params["max_step_mm"] / step_mag
            step_mag = params["max_step_mm"]
            saturated = True

        # update state only on commit + anti-windup
        if commit:
            if not saturated and dt > 0:
                self.integral += e * dt
            self.prev_err = e
            self.prev_t   = now

        sat_tag = "  SAT" if saturated else ""
        return dq, f"|e|={mag:.1f}px  step={step_mag:.3f}mm{sat_tag}"


# ─── Trackbar helpers ───────────────────────────────────────────────────────
def make_trackbars(win):
    cv2.createTrackbar("Kp /10000",     win, int(KP_DEFAULT * 10000),     200, lambda _: None)
    cv2.createTrackbar("Ki /100000",    win, int(KI_DEFAULT * 100000),    200, lambda _: None)
    cv2.createTrackbar("Kd /100000",    win, int(KD_DEFAULT * 100000),    200, lambda _: None)
    cv2.createTrackbar("max_step *100", win, int(MAX_STEP_DEFAULT * 100), 100, lambda _: None)
    cv2.createTrackbar("deadband px",   win, DEADBAND_DEFAULT,             50, lambda _: None)


def read_trackbars(win, sign_x, sign_y, swap_axes):
    return {
        "kp":          cv2.getTrackbarPos("Kp /10000",     win) / 10000.0,
        "ki":          cv2.getTrackbarPos("Ki /100000",    win) / 100000.0,
        "kd":          cv2.getTrackbarPos("Kd /100000",    win) / 100000.0,
        "max_step_mm": max(0.01, cv2.getTrackbarPos("max_step *100", win) / 100.0),
        "deadband":    max(1, cv2.getTrackbarPos("deadband px", win)),
        "sign_x":      sign_x,
        "sign_y":      sign_y,
        "swap_axes":   swap_axes,
    }


# ─── Status drawing ─────────────────────────────────────────────────────────
def make_status_image(state, lines):
    img = np.full((WIN_H, WIN_W, 3), DARK, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (WIN_W - 1, WIN_H - 1), BORDER, 1)

    state_color = {"IDLE": GRAY, "SERVO": GREEN, "STOPPED": RED}.get(state, GRAY)
    cv2.putText(img, f"[ {state} ]", (16, 36), HUD_FONT, 0.9, state_color, 2, cv2.LINE_AA)
    cv2.line(img, (16, 50), (WIN_W - 16, 50), BORDER, 1)

    for i, (label, value, color) in enumerate(lines):
        y = 80 + i * 26
        cv2.putText(img, f"{label:<14}", (16, y),  HUD_FONT, 0.5, DIM,    1, cv2.LINE_AA)
        cv2.putText(img, value,          (180, y), HUD_FONT, 0.55, color, 1, cv2.LINE_AA)

    keys = "  s:SERVO   SPACE/e:STOP   1:flip-x  2:flip-y  3:swap  r:reset-I   q:QUIT"
    cv2.putText(img, keys, (16, WIN_H - 18), HUD_FONT, 0.42, DIM, 1, cv2.LINE_AA)
    return img


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    robot      = RobotInterface()       # this inits the ROS node
    perception = PerceptionInterface()
    pid        = PIDServo()
    print(f"[init] robot + perception + PID ready  (DRY_RUN={DRY_RUN}, robot={ROBOT_NAME})")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    make_trackbars(WIN_NAME)

    state = "IDLE"
    last_status = "ready"
    last_dq = np.array([0.0, 0.0])
    last_err = None

    sign_x = 1
    sign_y = 1
    swap_axes = False

    rate = rospy.Rate(SERVO_RATE_HZ)

    def status(msg):
        nonlocal last_status
        last_status = msg
        print(f"[{state}] {msg}")

    try:
        while not rospy.is_shutdown():
            tip     = perception.get_tip(max_age_s=TIP_FRESHNESS_S)
            tgt     = perception.get_target(max_age_s=TIP_FRESHNESS_S)
            pose    = robot.get_full_pose()
            pose_xy = (float(pose[0]), float(pose[1])) if pose is not None else None

            err = None
            if tip is not None and tgt is not None:
                err = (tgt[0] - tip[0], tgt[1] - tip[1])
                last_err = np.array(err)

            params = read_trackbars(WIN_NAME, sign_x, sign_y, swap_axes)

            # ── candidate Δq (preview when IDLE, actually applied when SERVO) ───
            candidate_dq, msg = pid.step(err, params, commit=(state == "SERVO"))
            if state == "SERVO" and candidate_dq is not None:
                ok = robot.move_xy_relative(float(candidate_dq[0]), float(candidate_dq[1]),
                                            max_lin_vel=SERVO_LIN_VEL)
                if not ok:
                    robot.stop()
                    state = "STOPPED"
                    status("HALT — move failed")
                else:
                    last_dq = candidate_dq
                    status(msg)
            elif state == "SERVO" and candidate_dq is None:
                if "deadband" in msg:
                    status(f"DONE — {msg}")
                    state = "IDLE"
                else:
                    robot.stop()
                    status(f"HALT — {msg}")
                    state = "STOPPED"

            # ── status display ─────────────────────────────────────────────
            tip_s = f"({tip[0]:7.1f}, {tip[1]:7.1f})"            if tip else "—"
            tgt_s = f"({tgt[0]:7.1f}, {tgt[1]:7.1f})"            if tgt else "—"
            pos_s = f"({pose_xy[0]:+7.3f}, {pose_xy[1]:+7.3f}) mm" if pose_xy else "—"
            err_s = (f"|e|={np.linalg.norm(last_err):5.1f}px  ({last_err[0]:+5.0f}, {last_err[1]:+5.0f})"
                     if last_err is not None else "—")

            # show candidate command always — preview when IDLE
            if candidate_dq is not None:
                tag = "(applied)" if state == "SERVO" else "(preview)"
                dq_show = candidate_dq if state == "SERVO" else candidate_dq
                dq_s = f"({dq_show[0]:+.3f}, {dq_show[1]:+.3f}) mm  {tag}"
                dq_color = GREEN if state == "SERVO" else AMBER
            else:
                dq_s = "—"
                dq_color = DIM

            gains_s = f"Kp={params['kp']:.5f}  Ki={params['ki']:.6f}  Kd={params['kd']:.6f}"
            cfg_s   = f"sx={'+' if sign_x>0 else '-'}1  sy={'+' if sign_y>0 else '-'}1  swap={swap_axes}"
            int_s   = f"({pid.integral[0]:+.1f}, {pid.integral[1]:+.1f}) px·s"

            lines = [
                ("tip px",     tip_s,   GREEN if tip else RED),
                ("target px",  tgt_s,   GREEN if tgt else RED),
                ("err",        err_s,   AMBER),
                ("pose xy",    pos_s,   GREEN if pose_xy else RED),
                ("Δq",         dq_s,    dq_color),
                ("gains",      gains_s, BLUE),
                ("axis cfg",   cfg_s,   PINK),
                ("integral",   int_s,   DIM),
                ("status",     last_status, GRAY),
                ("",           f"DRY_RUN={DRY_RUN}  max_step={params['max_step_mm']:.2f}mm  "
                               f"deadband={params['deadband']}px  servo_v={SERVO_LIN_VEL}mm/s", DIM),
            ]
            cv2.imshow(WIN_NAME, make_status_image(state, lines))
            key = cv2.waitKey(20) & 0xFF

            # ── key dispatch ───────────────────────────────────────────────
            if key in (ord('q'), 27):
                break
            elif key in (ord(' '), ord('e')):
                if state != "IDLE":
                    robot.stop()
                pid.reset()
                state = "IDLE"
                status("EMERGENCY STOP — integral reset")
            elif key == ord('s') and state == "IDLE":
                pid.reset()
                state = "SERVO"
                status("loop started")
            elif key == ord('1'):
                sign_x *= -1
                pid.reset()
                status(f"sign_x → {sign_x:+d}  (integral reset)")
            elif key == ord('2'):
                sign_y *= -1
                pid.reset()
                status(f"sign_y → {sign_y:+d}  (integral reset)")
            elif key == ord('3'):
                swap_axes = not swap_axes
                pid.reset()
                status(f"swap_axes → {swap_axes}  (integral reset)")
            elif key == ord('r'):
                pid.reset()
                status("integral reset")

            if state == "SERVO":
                rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        try:
            robot.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()