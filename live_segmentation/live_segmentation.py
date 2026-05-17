#!/usr/bin/env python3
"""
live_segmentation.py
--------------------
Live YOLO tool segmentation on the Leica DeckLink feed (via gscam ROS topic).
Produces tool tip pixel + servo target pixel and publishes both for downstream
visual servoing.

Target modes (cycle with [m]):
  center          — image center pixel       (default; placeholder until limbus model lands)
  click           — left-click in viewer     (manual override)
  limbus_offset   — 4mm posterior to limbus  (FUTURE — disabled until limbus model exists)

Prereqs (other terminals):
  roslaunch gscam gscam_decklink.launch       # publishes /decklink/camera/image_raw
  # verify:  rostopic hz /decklink/camera/image_raw

Publishes:
  /ati/tool_tip_pixel    geometry_msgs/PointStamped   x,y in image px ; z=1.0 valid, 0.0 invalid
  /ati/target_pixel      geometry_msgs/PointStamped   same convention

Controls:
  left-click   set click target (auto-switches mode → click)
  right-click  clear click + revert mode → center
  m            cycle target mode
  s            save annotated frame
  q / ESC      quit
"""

import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# MODEL_PATH       = "./best.pt"
MODEL_PATH       = "./best_new_1723.pt"
MICROSCOPE_TOPIC = "/decklink/camera/image_raw"

PUB_TIP_TOPIC    = "/ati/tool_tip_pixel"
PUB_TARGET_TOPIC = "/ati/target_pixel"
FRAME_ID         = "leica_image"

DISPLAY_W        = 960
DISPLAY_H        = 540
CONF_THRESH      = 0.25
INFER_EVERY_N    = 1
WAIT_FIRST_FRAME = 10.0

# add after TOOL_TIP_K / TIP_SMOOTH_ALPHA lines
LIMBUS_DIAM_MM    = 11.7   # nominal human limbus diameter (scale prior)
PARS_PLANA_MM     =  3.0   # posterior offset to pars plana insertion zone
PARS_PLANA_COLOR  = (  0,   0, 220)   # red

# colors (BGR)
MASK_COLOR    = (180, 105, 255)    # hot pink
ELLIPSE_COLOR = (255, 170, 68)     # amber
TIP_COLOR     = (  0,   0, 255)    # red
TARGET_COLOR  = (255, 100, 184)    # magenta
CENTER_COLOR  = (200, 200, 200)    # gray   — image-center marker (mode=center)
HUD_COLOR     = (200, 200, 200)
MASK_ALPHA    = 0.35

# ─── CLASS IDs ───────────────────────────────────────────────────────────────
CLS_LIMBUS       = 0
CLS_CANNULA_BODY = 1
CLS_TROCAR       = 2

CLS_COLORS = {          # BGR mask colors
    CLS_LIMBUS:       (100, 220, 100),   # green
    CLS_CANNULA_BODY: (180, 105, 255),   # hot pink  (keep existing MASK_COLOR)
    CLS_TROCAR:       ( 68, 170, 255),   # amber
}
CLS_NAMES = {CLS_LIMBUS: "limbus", CLS_CANNULA_BODY: "cannula", CLS_TROCAR: "trocar"}
# ─────────────────────────────────────────────────────────────────────────────

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN_NAME = "Leica YOLO Live"

TOOL_TIP_SIDE    = "right"        # right | left | top | bottom — tool entry side
TOOL_TIP_K       = 5              # avg top-K extremal pixels
TIP_SMOOTH_ALPHA = 0.35
TIP_RESET_DIST   = 80
# ─────────────────────────────────────────────────────────────────────────────


# ─── ROS image decoding ──────────────────────────────────────────────────────
def _decode_microscope_msg(msg):
    enc = msg.encoding
    try:
        if enc == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        if enc == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return arr[:, :, ::-1].copy()
        if enc in ("yuv422", "yuyv422", "uyvy"):
            raw  = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
            code = cv2.COLOR_YUV2BGR_UYVY if enc == "uyvy" else cv2.COLOR_YUV2BGR_YUYV
            return cv2.cvtColor(raw, code)
        if enc == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if enc == "mono16":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            return cv2.cvtColor((arr >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if enc == "bayer_rggb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
        if enc == "rgba8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        return arr[:, :, :3].copy()
    except Exception as e:
        print(f"[decode] enc={enc} failed: {e}")
        return None


class MicroscopeSubscriber:
    """Latest-frame subscriber with lock + sequence counter."""
    def __init__(self, topic):
        self._lock  = threading.Lock()
        self._frame = None
        self._seq   = 0
        self._sub   = rospy.Subscriber(topic, Image, self._cb, queue_size=1, buff_size=2**24)

    def _cb(self, msg):
        bgr = _decode_microscope_msg(msg)
        if bgr is None:
            return
        with self._lock:
            self._frame = bgr
            self._seq  += 1

    def get(self, last_seq):
        with self._lock:
            if self._seq == last_seq or self._frame is None:
                return None, self._seq
            return self._frame.copy(), self._seq


# ─── Target selection ────────────────────────────────────────────────────────
class TargetSelector:
    """
    Decides where the robot should drive the tool tip in pixel space.
    Pluggable: today supports center + click. limbus_offset slot reserved for
    when the limbus segmentation model exists.
    """
    AVAIL_BASE = ("center", "click")        # always available
    AVAIL_FUTURE = ("limbus_offset",)       # only when limbus_xy/axes are set

    def __init__(self, frame_w, frame_h):
        self.mode = "center"
        self._fw, self._fh = frame_w, frame_h
        self._click = None
        # future hooks — set by a limbus detector callback
        self._limbus_center = None
        self._limbus_axes   = None
        self._limbus_angle  = None

    # ── manual control ───────────────────────────────────────────────────────
    def set_click(self, xy):
        self._click = xy
        self.mode = "click"

    def clear_click(self):
        self._click = None
        if self.mode == "click":
            self.mode = "center"

    def cycle_mode(self):
        avail = list(self.AVAIL_BASE)
        if self._limbus_center is not None:
            avail += list(self.AVAIL_FUTURE)
        # drop modes whose data isn't available right now
        if self._click is None and "click" in avail:
            avail.remove("click")
        if not avail:
            return
        i = avail.index(self.mode) if self.mode in avail else -1
        self.mode = avail[(i + 1) % len(avail)]

    # ── future hook for limbus model ────────────────────────────────────────
    def set_limbus(self, center_xy, axes, angle_deg):
        """Called by limbus detector when available. center_xy in px, axes (major, minor) in px."""
        self._limbus_center = center_xy
        self._limbus_axes   = axes
        self._limbus_angle  = angle_deg

    # ── target query ─────────────────────────────────────────────────────────
    def get_target(self):
        """Returns (x, y) target pixel, or None if mode's data isn't available."""
        if self.mode == "center":
            return self._fw // 2, self._fh // 2
        if self.mode == "click":
            return self._click
        if self.mode == "limbus_offset":
            # PLACEHOLDER — proper math goes here once limbus model is wired in.
            # Plan: scale = 11.7mm / max(axes_px)  → 1px = scale mm
            #       offset_px = 4.0 / scale       → posterior offset in px
            #       posterior direction = unit vector along entry side, in image plane
            # For now, fall back to center so the system still runs.
            if self._limbus_center is not None:
                return self._limbus_center
            return self._fw // 2, self._fh // 2
        return None


class ClickTarget:
    """Mouse callback. Stores last click in FULL-RES coords."""
    def __init__(self, scale_x=1.0, scale_y=1.0, on_click=None, on_clear=None):
        self._sx, self._sy = scale_x, scale_y
        self.on_click = on_click
        self.on_clear = on_clear

    def cb(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = (int(x / self._sx), int(y / self._sy))
            print(f"click target -> {xy}")
            if self.on_click:
                self.on_click(xy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("click target cleared")
            if self.on_clear:
                self.on_clear()


# ─── Tool tip extraction ─────────────────────────────────────────────────────
class TipSmoother:
    def __init__(self, alpha=0.35, reset_dist=80):
        self.alpha, self.reset_dist = alpha, reset_dist
        self._x = self._y = None

    def update(self, tip):
        if tip is None:
            return None
        x, y = tip
        if self._x is None:
            self._x, self._y = float(x), float(y)
        else:
            d = ((x - self._x) ** 2 + (y - self._y) ** 2) ** 0.5
            if d > self.reset_dist:
                self._x, self._y = float(x), float(y)
            else:
                self._x = self.alpha * x + (1 - self.alpha) * self._x
                self._y = self.alpha * y + (1 - self.alpha) * self._y
        return int(round(self._x)), int(round(self._y))

    def reset(self):
        self._x = self._y = None


def find_tool_tip(contour, side="right", k=5):
    if contour is None or len(contour) == 0:
        return None
    pts = contour.reshape(-1, 2)
    if   side == "right":  order = np.argsort(pts[:, 0])[::-1]
    elif side == "left":   order = np.argsort(pts[:, 0])
    elif side == "top":    order = np.argsort(pts[:, 1])
    elif side == "bottom": order = np.argsort(pts[:, 1])[::-1]
    else: raise ValueError(side)
    sel = pts[order[:k]]
    return int(sel[:, 0].mean()), int(sel[:, 1].mean())

def draw_pars_plana(img, limbus_ellipse):
    """
    Expand the fitted limbus ellipse outward by PARS_PLANA_MM using the
    limbus axes as a scale prior (LIMBUS_DIAM_MM → px/mm).
    Draws the result as a dashed red ellipse on img in-place.
    """
    if limbus_ellipse is None:
        return
    (cx, cy), (d_maj, d_min), angle = limbus_ellipse
    semi_maj = d_maj / 2.0
    semi_min = d_min / 2.0
    mean_r   = (semi_maj + semi_min) / 2.0
    if mean_r < 5:
        return

    px_per_mm = mean_r / (LIMBUS_DIAM_MM / 2.0)
    offset_px = PARS_PLANA_MM * px_per_mm

    center_i = (int(cx), int(cy))
    axes_i   = (int((d_maj + 2 * offset_px) / 2), int((d_min + 2 * offset_px) / 2))

    N_SEGS = 36
    for i in range(0, N_SEGS, 2):
        start_deg = i       * (360 / N_SEGS)
        end_deg   = (i + 1) * (360 / N_SEGS)
        cv2.ellipse(img, center_i, axes_i, angle,
                    start_deg, end_deg, PARS_PLANA_COLOR, 2, cv2.LINE_AA)

    label_pt = (center_i[0] + 6, center_i[1] - axes_i[1] - 8)
    cv2.putText(img, f"pars plana ~{PARS_PLANA_MM:.0f}mm", label_pt,
                HUD_FONT, 0.42, PARS_PLANA_COLOR, 1, cv2.LINE_AA)
def overlay_masks(frame, results, smoother=None):
    """
    Returns (out, n_det, tool_ellipse, tip, limbus_ellipse).
    - limbus_ellipse: cv2.fitEllipse result for class 0, or None
    - tool_ellipse / tip: from trocar (preferred) then cannula_body fallback
    """
    out = frame.copy()
    masks_data  = results[0].masks
    classes_raw = results[0].boxes.cls  # tensor of class IDs, one per detection

    if masks_data is None or len(masks_data) == 0:
        if smoother is not None:
            smoother.reset()
        return out, 0, None, None, None

    h, w = frame.shape[:2]

    # per-class accumulators: best by contour area
    best = {cls: {"ellipse": None, "contour": None, "area": 0}
            for cls in (CLS_LIMBUS, CLS_CANNULA_BODY, CLS_TROCAR)}

    for i, mask_tensor in enumerate(masks_data.data):
        cls_id = int(classes_raw[i].item())
        if cls_id not in best:
            continue

        mask_full = cv2.resize(
            (mask_tensor.cpu().numpy() * 255).astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST
        )
        binary = (mask_full > 127).astype(np.uint8)

        color = CLS_COLORS.get(cls_id, MASK_COLOR)
        coloured = np.zeros_like(frame, dtype=np.uint8)
        coloured[binary == 1] = color
        out = cv2.addWeighted(out, 1.0, coloured, MASK_ALPHA, 0)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > best[cls_id]["area"]:
                fit = cv2.fitEllipse(c) if (cls_id != CLS_CANNULA_BODY and len(c) >= 5) else None
                best[cls_id].update(area=area, ellipse=fit, contour=c)

    # ── draw all detected ellipses ────────────────────────────────────────────
    for cls_id, b in best.items():
        if b["ellipse"] is None:
            continue
        color = CLS_COLORS[cls_id]
        cv2.ellipse(out, b["ellipse"], color, 2, cv2.LINE_AA)
        cx, cy = int(b["ellipse"][0][0]), int(b["ellipse"][0][1])
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        # class label near ellipse center
        cv2.putText(out, CLS_NAMES[cls_id], (cx + 8, cy - 8),
                    HUD_FONT, 0.45, color, 1, cv2.LINE_AA)

    # ── tip: cannula only ─────────────────────────────────────────────────────
    tool_ellipse = best[CLS_CANNULA_BODY]["ellipse"]
    raw_tip = find_tool_tip(best[CLS_CANNULA_BODY]["contour"], TOOL_TIP_SIDE, TOOL_TIP_K)
    tip = smoother.update(raw_tip) if smoother is not None else raw_tip

    if tip is not None:
        cv2.circle(out, tip, 10, TIP_COLOR, 2,  cv2.LINE_AA)
        cv2.circle(out, tip, 3,  TIP_COLOR, -1, cv2.LINE_AA)
        if raw_tip is not None and raw_tip != tip:
            cv2.circle(out, raw_tip, 4, (120, 120, 120), 1, cv2.LINE_AA)

    # ── pars plana ring ───────────────────────────────────────────────────────
    limbus_ellipse = best[CLS_LIMBUS]["ellipse"]
    draw_pars_plana(out, limbus_ellipse)

    n_det = sum(1 for b in best.values() if b["ellipse"] is not None)
    return out, n_det, tool_ellipse, tip, limbus_ellipse

# ─── Drawing helpers ─────────────────────────────────────────────────────────
def draw_target_overlay(img, tip, target, mode):
    """Target marker, error arrow tip→target, and a faint center crosshair if mode=center."""
    h, w = img.shape[:2]
    if mode == "center":
        cx, cy = w // 2, h // 2
        cv2.drawMarker(img, (cx, cy), CENTER_COLOR, cv2.MARKER_TILTED_CROSS, 24, 1, cv2.LINE_AA)
    if target is not None:
        cv2.drawMarker(img, target, TARGET_COLOR, cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(img, target, 14, TARGET_COLOR, 1, cv2.LINE_AA)
        if tip is not None:
            cv2.arrowedLine(img, tip, target, TARGET_COLOR, 2, cv2.LINE_AA, tipLength=0.05)


def draw_hud(img, fps, n_det, tip, target, mode):
    lines = [f"{fps:5.1f} fps   conf={CONF_THRESH}   side={TOOL_TIP_SIDE}   ema={TIP_SMOOTH_ALPHA:.2f}"]
    lines.append(f"mode  [{mode}]")
    if n_det == 0:
        lines.append("no detection")
    else:
        lines.append(f"{n_det} det")
        if tip is not None:
            lines.append(f"tip   ({tip[0]:4d}, {tip[1]:4d})")
        if target is not None and tip is not None:
            du, dv = target[0] - tip[0], target[1] - tip[1]
            mag = (du*du + dv*dv) ** 0.5
            lines.append(f"err   du={du:+4d}  dv={dv:+4d}  |e|={mag:5.1f}px")
        elif target is not None:
            lines.append(f"target ({target[0]:4d}, {target[1]:4d}) — no tip")

    for i, t in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(img, t, (10, y), HUD_FONT, 0.55, (0, 0, 0),  3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), HUD_FONT, 0.55, HUD_COLOR, 1, cv2.LINE_AA)


def _publish_pixel(pub, xy, valid):
    msg = PointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = FRAME_ID
    if xy is not None and valid:
        msg.point.x, msg.point.y, msg.point.z = float(xy[0]), float(xy[1]), 1.0
    else:
        msg.point.x, msg.point.y, msg.point.z = 0.0, 0.0, 0.0
    pub.publish(msg)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    rospy.init_node("leica_yolo_live", anonymous=True, disable_signals=True)

    model = YOLO(MODEL_PATH)
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), conf=CONF_THRESH, verbose=False)
    print(f"[init] model loaded: {MODEL_PATH}")

    sub = MicroscopeSubscriber(MICROSCOPE_TOPIC)
    print(f"[init] subscribed: {MICROSCOPE_TOPIC}  — waiting for first frame...")

    t0 = rospy.Time.now()
    first_frame = None
    while not rospy.is_shutdown():
        first_frame, _ = sub.get(0)
        if first_frame is not None:
            print(f"[init] first frame  [{first_frame.shape[1]}x{first_frame.shape[0]}]")
            break
        if (rospy.Time.now() - t0).to_sec() > WAIT_FIRST_FRAME:
            sys.exit(f"No frames on {MICROSCOPE_TOPIC} after {WAIT_FIRST_FRAME:.0f}s — gscam running?")
        rospy.sleep(0.1)

    fh, fw = first_frame.shape[:2]
    sx, sy = DISPLAY_W / fw, DISPLAY_H / fh

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

    # target system
    target_sel = TargetSelector(fw, fh)
    clicker = ClickTarget(scale_x=sx, scale_y=sy,
                          on_click=target_sel.set_click,
                          on_clear=target_sel.clear_click)
    cv2.setMouseCallback(WIN_NAME, clicker.cb)

    smoother = TipSmoother(alpha=TIP_SMOOTH_ALPHA, reset_dist=TIP_RESET_DIST)

    # publishers
    pub_tip    = rospy.Publisher(PUB_TIP_TOPIC,    PointStamped, queue_size=1)
    pub_target = rospy.Publisher(PUB_TARGET_TOPIC, PointStamped, queue_size=1)

    # state
    last_seq = 0
    infer_count = fps_frames = 0
    fps_ts = datetime.now()
    fps = 0.0
    last_annotated = None
    last_n_det = 0
    last_tip = None

    try:
        while not rospy.is_shutdown():
            frame, last_seq = sub.get(last_seq)

            if frame is None:
                if last_annotated is not None:
                    overlay = last_annotated.copy()
                    target = target_sel.get_target()
                    draw_target_overlay(overlay, last_tip, target, target_sel.mode)
                    draw_hud(overlay, fps, last_n_det, last_tip, target, target_sel.mode)
                    cv2.imshow(WIN_NAME, cv2.resize(overlay, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA))
                key = cv2.waitKey(5) & 0xFF
                if key in (ord('q'), 27): break
                if key == ord('m'): target_sel.cycle_mode()
                continue

            infer_count += 1
            fps_frames  += 1

            if infer_count % INFER_EVERY_N == 0:
                results = model.predict(frame, conf=CONF_THRESH, verbose=False)
                annotated, last_n_det, _, last_tip, limbus_ell = overlay_masks(frame, results, smoother)
                if limbus_ell is not None:
                    cxy = (int(limbus_ell[0][0]), int(limbus_ell[0][1]))
                    axes = (limbus_ell[1][0] / 2, limbus_ell[1][1] / 2)   # semi-axes px
                    target_sel.set_limbus(cxy, axes, limbus_ell[2])
                last_annotated = annotated
            else:
                annotated = last_annotated if last_annotated is not None else frame

            # fps window
            now = datetime.now()
            dt  = (now - fps_ts).total_seconds()
            if dt >= 1.0:
                fps = fps_frames / dt
                fps_frames = 0
                fps_ts = now

            target = target_sel.get_target()

            # publish — independent of GUI rate, so servo always gets fresh data
            _publish_pixel(pub_tip,    last_tip, valid=last_tip is not None)
            _publish_pixel(pub_target, target,   valid=target is not None)

            # composite display
            disp_full = annotated.copy()
            draw_target_overlay(disp_full, last_tip, target, target_sel.mode)
            draw_hud(disp_full, fps, last_n_det, last_tip, target, target_sel.mode)
            disp = cv2.resize(disp_full, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
            cv2.imshow(WIN_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('m'):
                target_sel.cycle_mode()
                print(f"[mode] -> {target_sel.mode}")
            elif key == ord('s') and last_annotated is not None:
                ts   = datetime.now().strftime("%d%b%Y_%H%M%S").upper()
                path = f"leica_yolo_{ts}.png"
                cv2.imwrite(path, disp_full)
                print(f"saved -> {path}")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()