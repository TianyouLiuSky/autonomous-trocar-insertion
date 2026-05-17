# Live stereo depth viewer
# Subscribes to both FLIR camera topics, rectifies in real-time, runs SGBM
# ROS1 Melodic | Python 3.6 | PyQt5
#
# Run: python stereo_depth_viewer.py
#
# Requires: stereo calibration NPZ from stereo_camera_calibration.py
# Nodelets: roslaunch spinnaker_sdk_camera_driver acquisition.launch

import os
import sys
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QShortcut,
    QFrame, QGroupBox, QSlider, QSizePolicy
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
STEREO_NPZ   = "../stereo_camera_calibration/output/stereo_calibration_31MAR2026.npz"    # e.g. "../stereo_camera_calibration/output/stereo_calibration_31MAR2026.npz"
TOPIC_LEFT   = "/camera_array/cam_left/image_raw"
TOPIC_RIGHT  = "/camera_array/cam_right/image_raw"
CAM_ID_LEFT  = "24213548"
CAM_ID_RIGHT = "25332589"

DISPLAY_W = 480
DISPLAY_H = 300
TIMER_MS  = 66    # ~15 Hz display refresh

# SGBM defaults
DEFAULT_NUM_DISP  = 128   # must be multiple of 16
DEFAULT_BLOCK     = 11    # odd, 3–21
DEFAULT_P1_SCALE  = 8
DEFAULT_P2_SCALE  = 32
DEFAULT_UNIQUE    = 10
DEFAULT_SPECKLE_W = 100
DEFAULT_SPECKLE_R = 2

# Depth colormap range (mm)
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
DARK   = "#141414"
DARK2  = "#1a1a1a"
BORDER = "#444"
TEXT   = "#ddd"
DIM    = "#888"
GREEN  = "#4c9"
BLUE   = "#7af"
MONO   = "monospace"

GLOBAL_STYLE = f"""
    QWidget {{
        background: {DARK};
        color: {TEXT};
        font-family: {MONO};
        font-size: 11px;
    }}
    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 3px;
        margin-top: 6px;
        padding: 6px;
        color: {DIM};
        font-size: 11px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
    }}
    QLabel  {{ color: {DIM}; border: none; }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: #333;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {BLUE};
        width: 12px;
        height: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }}
    QSlider::sub-page:horizontal {{
        background: {BLUE};
        border-radius: 2px;
    }}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bgr_to_pixmap(bgr, w, h):
    disp = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def _gray_to_pixmap(gray, w, h):
    disp = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    qimg = QImage(disp.data, disp.shape[1], disp.shape[0], disp.strides[0], QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg)

def _placeholder(w, h, text="Waiting..."):
    img = np.full((h, w, 3), 26, dtype=np.uint8)
    cv2.putText(img, text, (12, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (120, 120, 120), 1, cv2.LINE_AA)
    return _bgr_to_pixmap(img, w, h)

def _colormap_depth(depth_mm, vmin, vmax, auto=False):
    valid = depth_mm[depth_mm > 0]
    if auto or valid.size == 0:
        vmin = float(np.percentile(valid, 5))  if valid.size else vmin
        vmax = float(np.percentile(valid, 95)) if valid.size else vmax
    norm  = np.clip((depth_mm.astype(float) - vmin) / (vmax - vmin + 1e-6), 0, 1)
    u8    = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[depth_mm <= 0] = 0
    return color

def _cam_panel(title_text):
    outer = QFrame()
    outer.setStyleSheet(
        f"QFrame {{ border: 1px solid {BORDER}; background: {DARK2}; border-radius: 2px; }}"
    )
    title = QLabel(title_text)
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet(f"color:{DIM}; font-size:10px; padding:3px; border:none;")

    img_lbl = QLabel()
    img_lbl.setAlignment(Qt.AlignCenter)
    img_lbl.setFixedSize(DISPLAY_W, DISPLAY_H)
    img_lbl.setPixmap(_placeholder(DISPLAY_W, DISPLAY_H, title_text))

    stat_lbl = QLabel("● no stream")
    stat_lbl.setAlignment(Qt.AlignCenter)
    stat_lbl.setStyleSheet(f"color:#555; font-size:10px; padding:2px; border:none;")

    layout = QVBoxLayout(outer)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(2)
    layout.addWidget(title)
    layout.addWidget(img_lbl)
    layout.addWidget(stat_lbl)
    return outer, img_lbl, stat_lbl


def _slider_row(label, vmin, vmax, default, step=1):
    """Returns (QWidget row, QSlider, QLabel value)."""
    row = QWidget()
    row.setStyleSheet("background: transparent;")
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)

    lbl = QLabel(label)
    lbl.setFixedWidth(110)
    lbl.setStyleSheet(f"color:{DIM}; font-size:10px; border:none;")

    sl = QSlider(Qt.Horizontal)
    sl.setMinimum(vmin)
    sl.setMaximum(vmax)
    sl.setSingleStep(step)
    sl.setValue(default)

    val_lbl = QLabel(str(default))
    val_lbl.setFixedWidth(40)
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    val_lbl.setStyleSheet(f"color:{BLUE}; font-size:10px; border:none;")

    sl.valueChanged.connect(lambda v: val_lbl.setText(str(v)))

    h.addWidget(lbl)
    h.addWidget(sl, stretch=1)
    h.addWidget(val_lbl)
    return row, sl, val_lbl


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber
# ─────────────────────────────────────────────────────────────────────────────
class CameraSubscriber(object):
    def __init__(self, topic):
        self._lock  = threading.Lock()
        self._frame = None
        rospy.Subscriber(topic, Image, self._cb, queue_size=1, buff_size=2**24)

    def _cb(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        enc   = msg.encoding.lower()
        if enc == "rgb8":
            bgr = frame[..., ::-1].copy()
        elif enc == "bgr8":
            bgr = frame.copy()
        else:
            bgr = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
        with self._lock:
            self._frame = bgr

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def has_frame(self):
        with self._lock:
            return self._frame is not None


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class StereoDepthViewer(QMainWindow):

    def __init__(self):
        super(StereoDepthViewer, self).__init__()
        self.setWindowTitle("Stereo Depth Viewer")
        self.setStyleSheet(GLOBAL_STYLE)

        # load calibration
        cal = np.load(STEREO_NPZ)
        self._map_lx = cal["map_left_x"]
        self._map_ly = cal["map_left_y"]
        self._map_rx = cal["map_right_x"]
        self._map_ry = cal["map_right_y"]
        self._Q      = cal["Q"]
        self._baseline_mm = float(cal["baseline_mm"])

        # rectified focal length from P_left
        if "P_left" in cal:
            self._f_rect = float(cal["P_left"][0, 0])
        else:
            self._f_rect = float(self._Q[2, 3])

        print(f"[viewer] Baseline: {self._baseline_mm:.1f} mm  |  f_rect: {self._f_rect:.1f}")

        # ROS subscribers
        self._sub_left  = CameraSubscriber(TOPIC_LEFT)
        self._sub_right = CameraSubscriber(TOPIC_RIGHT)

        # timing
        self._fps_times = []
        self._fps       = 0.0

        self._build_ui()

        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(TIMER_MS)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── top row: left | right | depth ────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        self._panel_l,     self._img_left,   self._stat_left   = _cam_panel("LEFT  [{}]".format(CAM_ID_LEFT))
        self._panel_r,     self._img_right,  self._stat_right  = _cam_panel("RIGHT [{}]".format(CAM_ID_RIGHT))
        self._panel_depth, self._img_depth,  self._stat_depth  = _cam_panel("DEPTH  (SGBM)")

        top_row.addWidget(self._panel_l)
        top_row.addWidget(self._panel_r)
        top_row.addWidget(self._panel_depth)
        root.addLayout(top_row, stretch=3)

        # ── bottom row: epipolar | disparity ─────────────────────────────────
        bot_row = QHBoxLayout()
        bot_row.setSpacing(4)

        self._panel_epi,  self._img_epi,   self._stat_epi   = _cam_panel("EPIPOLAR CHECK (rectified)")
        self._panel_disp, self._img_disp,  self._stat_disp  = _cam_panel("DISPARITY")

        bot_row.addWidget(self._panel_epi)
        bot_row.addWidget(self._panel_disp)
        root.addLayout(bot_row, stretch=2)

        # ── SGBM controls ─────────────────────────────────────────────────────
        ctrl_box = QGroupBox("SGBM Parameters")
        ctrl_box.setStyleSheet(
            "QGroupBox { color:#aaa; font-family:monospace; font-size:11px;"
            "            border:1px solid #444; border-radius:3px; padding:8px; margin-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        ctrl_grid = QHBoxLayout(ctrl_box)
        ctrl_grid.setSpacing(16)

        # column 1
        col1 = QVBoxLayout()
        col1.setSpacing(4)
        row_nd, self._sl_num_disp, _ = _slider_row(
            "Num disparities", 16, 512, DEFAULT_NUM_DISP, step=16
        )
        row_bs, self._sl_block, _    = _slider_row(
            "Block size",       3,  21, DEFAULT_BLOCK,    step=2
        )
        row_un, self._sl_unique, _   = _slider_row(
            "Uniqueness",       0,  30, DEFAULT_UNIQUE
        )
        col1.addWidget(row_nd)
        col1.addWidget(row_bs)
        col1.addWidget(row_un)

        # column 2
        col2 = QVBoxLayout()
        col2.setSpacing(4)
        row_sw, self._sl_speckle_w, _ = _slider_row(
            "Speckle window",   0, 300, DEFAULT_SPECKLE_W
        )
        row_sr, self._sl_speckle_r, _ = _slider_row(
            "Speckle range",    1,  10, DEFAULT_SPECKLE_R
        )

        # depth range
        row_dmin, self._sl_dmin, _ = _slider_row(
            "Depth min (mm)",   0, 500, DEPTH_MIN_MM, step=10
        )
        row_dmax, self._sl_dmax, _ = _slider_row(
            "Depth max (mm)", 100, 2000, DEPTH_MAX_MM, step=50
        )
        col2.addWidget(row_sw)
        col2.addWidget(row_sr)
        col2.addWidget(row_dmin)
        col2.addWidget(row_dmax)

        ctrl_grid.addLayout(col1, stretch=1)
        ctrl_grid.addLayout(col2, stretch=1)
        root.addWidget(ctrl_box)

        # ── status footer ─────────────────────────────────────────────────────
        footer = QHBoxLayout()
        footer.setSpacing(12)

        self._lbl_fps      = QLabel("fps: --")
        self._lbl_fps.setStyleSheet(f"color:{BLUE}; font-size:11px; border:none;")

        self._lbl_baseline = QLabel("baseline: {:.1f} mm".format(self._baseline_mm))
        self._lbl_baseline.setStyleSheet(f"color:{DIM}; font-size:11px; border:none;")

        self._lbl_status = QLabel("Waiting for streams...")
        self._lbl_status.setStyleSheet(f"color:{DIM}; font-size:11px; border:none;")
        self._lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        footer.addWidget(self._lbl_fps)
        footer.addWidget(self._lbl_baseline)
        footer.addWidget(self._lbl_status, stretch=1)
        root.addLayout(footer)

        self.resize(1920, 1060)
        self.show()

    # ── tick ──────────────────────────────────────────────────────────────────
    def _tick(self):
        frame_l = self._sub_left.get_frame()
        frame_r = self._sub_right.get_frame()

        if frame_l is None or frame_r is None:
            missing = " + ".join(
                (["left"]  if frame_l is None else []) +
                (["right"] if frame_r is None else [])
            )
            self._lbl_status.setText("● waiting for: " + missing)
            return

        t0 = time.time()

        # ── rectify ───────────────────────────────────────────────────────────
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        rect_l = cv2.remap(gray_l, self._map_lx, self._map_ly, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gray_r, self._map_rx, self._map_ry, cv2.INTER_LINEAR)

        # ── SGBM ─────────────────────────────────────────────────────────────
        nd = (self._sl_num_disp.value() // 16) * 16  # ensure multiple of 16
        bs = self._sl_block.value()
        bs = bs if bs % 2 == 1 else bs + 1           # ensure odd

        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max(nd, 16),
            blockSize=bs,
            P1=DEFAULT_P1_SCALE * bs * bs,
            P2=DEFAULT_P2_SCALE * bs * bs,
            disp12MaxDiff=1,
            uniquenessRatio=self._sl_unique.value(),
            speckleWindowSize=self._sl_speckle_w.value(),
            speckleRange=self._sl_speckle_r.value(),
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY
        )

        disp = sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0
        disp[disp <= 0] = 0
        
        valid = disp[disp > 0]
        if valid.size > 0:
            print(f"[disp] unique vals: {np.unique(valid.round()).shape[0]}  "
                f"range: {valid.min():.1f} – {valid.max():.1f} px  "
                f"implied Z: {self._f_rect * self._baseline_mm / valid.max():.0f} – "
                f"{self._f_rect * self._baseline_mm / valid.min():.0f} mm")
        else:
            print("[disp] no valid disparity")

        # ── depth ─────────────────────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_mm = np.where(
                disp > 0,
                self._f_rect * self._baseline_mm / (disp + 1e-6),
                0
            ).astype(np.float32)

        # ── visualize ─────────────────────────────────────────────────────────
        dmin = self._sl_dmin.value()
        dmax = self._sl_dmax.value()

        depth_vis = _colormap_depth(depth_mm, dmin, dmax)
        disp_norm = np.clip(disp / max(nd, 1) * 255, 0, 255).astype(np.uint8)
        disp_vis  = cv2.applyColorMap(disp_norm, cv2.COLORMAP_MAGMA)

        # epipolar check — draw lines on rectified pair side-by-side
        epi_l = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)
        epi_r = cv2.cvtColor(rect_r, cv2.COLOR_GRAY2BGR)
        h = epi_l.shape[0]
        for y in np.linspace(30, h - 30, 16).astype(int):
            c = (0, 180, 100)
            cv2.line(epi_l, (0, y), (epi_l.shape[1], y), c, 1)
            cv2.line(epi_r, (0, y), (epi_r.shape[1], y), c, 1)
        epi_vis = np.hstack([epi_l, epi_r])

        # ── paint panels ──────────────────────────────────────────────────────
        self._img_left.setPixmap( _bgr_to_pixmap(frame_l,    DISPLAY_W, DISPLAY_H))
        self._img_right.setPixmap(_bgr_to_pixmap(frame_r,    DISPLAY_W, DISPLAY_H))
        self._img_depth.setPixmap(_bgr_to_pixmap(depth_vis,  DISPLAY_W, DISPLAY_H))
        self._img_disp.setPixmap( _bgr_to_pixmap(disp_vis,   DISPLAY_W, DISPLAY_H))
        self._img_epi.setFixedSize(DISPLAY_W * 2, DISPLAY_H)  # do this once in _build_ui
        self._img_epi.setPixmap(  _bgr_to_pixmap(epi_vis,    DISPLAY_W * 2, DISPLAY_H))

        # stream indicators
        for stat, frame in [
            (self._stat_left,  frame_l),
            (self._stat_right, frame_r),
        ]:
            stat.setText("● streaming  [{}×{}]".format(frame.shape[1], frame.shape[0]))
            stat.setStyleSheet(f"color:{GREEN}; font-size:10px; padding:2px; border:none;")

        for stat in (self._stat_depth, self._stat_disp, self._stat_epi):
            stat.setText("● live")
            stat.setStyleSheet(f"color:{GREEN}; font-size:10px; padding:2px; border:none;")

        # fps
        now = time.time()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 2.0]
        if len(self._fps_times) > 1:
            self._fps = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])

        elapsed_ms = (time.time() - t0) * 1000
        self._lbl_fps.setText("fps: {:.1f}  |  proc: {:.0f}ms".format(self._fps, elapsed_ms))
        self._lbl_status.setText("● live  —  nd={}  bs={}".format(nd, bs))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("stereo_depth_viewer", anonymous=True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = StereoDepthViewer()

    threading.Thread(target=rospy.spin, daemon=True).start()
    sys.exit(app.exec_())