#!/usr/bin/env python
"""
collect_cal_images.py  —  Calibration image collector for FLIR stereo cameras
ROS1 Melodic | Python 3.6 | PyQt5

Run:   python collect_cal_images.py
       (no args needed — select mode in the GUI)
"""

import os
import sys
import datetime
import threading

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton, QShortcut,
    QStatusBar, QFrame, QButtonGroup, QRadioButton, QGroupBox
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_LEFT  = "/camera_array/cam_left/image_raw"
TOPIC_RIGHT = "/camera_array/cam_right/image_raw"


CAM_ID_LEFT  = "24213548"
CAM_ID_RIGHT = "25332589"

# OUT_DIR = "./single_camera_calibration/data/31Mar2026_2" # For intrinsic cal  
# OUT_DIR = "./single_camera_calibration/validation/31Mar2026" # For intrinsic cal validation
OUT_DIR = "./stereo_camera_calibration/data/31Mar2026"  # For stereo cal (both cams at once)


DISPLAY_W = 720
DISPLAY_H = 540
TIMER_MS  = 66   # ~15 Hz


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber
# ─────────────────────────────────────────────────────────────────────────────
class CameraSubscriber(object):
    def __init__(self, topic, cam_id):
        self.topic  = topic
        self.cam_id = cam_id
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def _bgr_to_pixmap(bgr, w, h):
    disp = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _placeholder(w, h, text="Waiting for stream..."):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(img, text, (16, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (160, 160, 160), 1, cv2.LINE_AA)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _cam_panel(title_text):
    """Returns (QFrame, img_label, stream_status_label)."""
    frame = QFrame()
    frame.setFrameStyle(QFrame.Box | QFrame.Plain)
    frame.setStyleSheet("QFrame { border: 1px solid #444; background: #1a1a1a; }")

    title = QLabel(title_text)
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet(
        "color:#888; font-size:10px; font-family:monospace; padding:3px; border:none;"
    )

    img_lbl = QLabel()
    img_lbl.setAlignment(Qt.AlignCenter)
    img_lbl.setFixedSize(DISPLAY_W, DISPLAY_H)
    img_lbl.setPixmap(_placeholder(DISPLAY_W, DISPLAY_H))

    stream_lbl = QLabel("● no stream")
    stream_lbl.setAlignment(Qt.AlignCenter)
    stream_lbl.setStyleSheet(
        "color:#555; font-size:10px; font-family:monospace; padding:2px; border:none;"
    )

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(2)
    layout.addWidget(title)
    layout.addWidget(img_lbl)
    layout.addWidget(stream_lbl)
    return frame, img_lbl, stream_lbl


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class CalCollectorWindow(QMainWindow):

    # mode IDs
    DUAL  = 0
    LEFT  = 1
    RIGHT = 2

    def __init__(self):
        super(CalCollectorWindow, self).__init__()
        self.count = 0
        os.makedirs(OUT_DIR, exist_ok=True)

        # always subscribe to both — only paint/save the active ones
        self._sub_left  = CameraSubscriber(TOPIC_LEFT,  CAM_ID_LEFT)
        self._sub_right = CameraSubscriber(TOPIC_RIGHT, CAM_ID_RIGHT)

        self._build_ui()
        self._apply_mode()

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._capture)
        QShortcut(QKeySequence("Ctrl+Q"),     self, activated=self.close)

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(TIMER_MS)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("Cal Image Collector")
        self.setStyleSheet("background:#141414; color:#ddd;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── top bar ───────────────────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        # mode radio buttons
        mode_box = QGroupBox("Mode")
        mode_box.setStyleSheet(
            "QGroupBox { color:#aaa; font-family:monospace; font-size:11px;"
            "            border:1px solid #444; border-radius:3px; padding:6px; margin-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        mode_row = QHBoxLayout(mode_box)
        mode_row.setSpacing(16)

        rb_style = "QRadioButton { color:#ccc; font-family:monospace; font-size:11px; }"
        self._rb_dual  = QRadioButton("Dual  (stereo)")
        self._rb_left  = QRadioButton("Single — Left")
        self._rb_right = QRadioButton("Single — Right")
        self._rb_dual.setChecked(True)
        for rb in (self._rb_dual, self._rb_left, self._rb_right):
            rb.setStyleSheet(rb_style)
            mode_row.addWidget(rb)

        self._mode_grp = QButtonGroup()
        self._mode_grp.addButton(self._rb_dual,  self.DUAL)
        self._mode_grp.addButton(self._rb_left,  self.LEFT)
        self._mode_grp.addButton(self._rb_right, self.RIGHT)
        self._mode_grp.buttonClicked.connect(self._apply_mode)

        top_bar.addWidget(mode_box)

        # output path
        path_box = QGroupBox("Output directory")
        path_box.setStyleSheet(
            "QGroupBox { color:#aaa; font-family:monospace; font-size:11px;"
            "            border:1px solid #444; border-radius:3px; padding:6px; margin-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
        )
        path_row = QHBoxLayout(path_box)
        lbl_path = QLabel(os.path.abspath(OUT_DIR))
        lbl_path.setStyleSheet("color:#7af; font-family:monospace; font-size:11px; border:none;")
        path_row.addWidget(lbl_path)
        top_bar.addWidget(path_box, stretch=1)

        # saved counter
        self._lbl_count = QLabel("Saved: 0")
        self._lbl_count.setStyleSheet(
            "color:#4c9; font-family:monospace; font-size:14px; font-weight:bold; padding:0 8px;"
        )
        top_bar.addWidget(self._lbl_count)

        root.addLayout(top_bar)

        # ── camera panels ─────────────────────────────────────────────────
        self._panels_row = QHBoxLayout()
        self._panels_row.setSpacing(6)

        self._frame_left,  self._img_left,  self._stat_left  = _cam_panel(
            "LEFT  [{}]  {}".format(CAM_ID_LEFT, TOPIC_LEFT)
        )
        self._frame_right, self._img_right, self._stat_right = _cam_panel(
            "RIGHT [{}]  {}".format(CAM_ID_RIGHT, TOPIC_RIGHT)
        )
        self._panels_row.addWidget(self._frame_left)
        self._panels_row.addWidget(self._frame_right)
        root.addLayout(self._panels_row)

        # ── capture button ────────────────────────────────────────────────
        self._btn_capture = QPushButton("  CAPTURE  [SPACE]")
        self._btn_capture.setFixedHeight(46)
        self._btn_capture.setStyleSheet(
            "QPushButton {"
            "  background:#2a6099; color:#fff; border:none;"
            "  font-size:14px; font-family:monospace; font-weight:bold; border-radius:4px;"
            "}"
            "QPushButton:hover   { background:#3a80bb; }"
            "QPushButton:pressed { background:#1a4070; }"
        )
        self._btn_capture.clicked.connect(self._capture)
        root.addWidget(self._btn_capture)

        # ── status bar ────────────────────────────────────────────────────
        self._sbar = QStatusBar()
        self._sbar.setStyleSheet(
            "color:#888; font-family:monospace; font-size:10px;"
        )
        self.setStatusBar(self._sbar)
        self._sbar.showMessage("SPACE = capture    Ctrl+Q = quit")

        self.adjustSize()

    # ── mode switching ────────────────────────────────────────────────────────
    def _apply_mode(self, *_):
        mid = self._mode_grp.checkedId()
        self._frame_left.setVisible(mid  in (self.DUAL, self.LEFT))
        self._frame_right.setVisible(mid in (self.DUAL, self.RIGHT))
        self.adjustSize()

    @property
    def _mode(self):
        return self._mode_grp.checkedId()

    # ── display refresh ───────────────────────────────────────────────────────
    def _refresh(self):
        if self._mode in (self.DUAL, self.LEFT):
            self._paint(self._sub_left,  self._img_left,  self._stat_left)
        if self._mode in (self.DUAL, self.RIGHT):
            self._paint(self._sub_right, self._img_right, self._stat_right)

    def _paint(self, sub, img_lbl, stat_lbl):
        frame = sub.get_frame()
        if frame is not None:
            img_lbl.setPixmap(_bgr_to_pixmap(frame, DISPLAY_W, DISPLAY_H))
            stat_lbl.setText("● streaming  [{}x{}]".format(frame.shape[1], frame.shape[0]))
            stat_lbl.setStyleSheet(
                "color:#4c9; font-size:10px; font-family:monospace; padding:2px; border:none;"
            )
        else:
            stat_lbl.setText("● no stream")
            stat_lbl.setStyleSheet(
                "color:#555; font-size:10px; font-family:monospace; padding:2px; border:none;"
            )

    # ── capture ───────────────────────────────────────────────────────────────
    def _capture(self):
        ts = _timestamp()

        if self._mode == self.DUAL:
            fl = self._sub_left.get_frame()
            fr = self._sub_right.get_frame()
            if fl is None or fr is None:
                missing = " ".join(
                    (["left"] if fl is None else []) + (["right"] if fr is None else [])
                )
                self._sbar.showMessage("  no frame from: " + missing, 3000)
                return
            cv2.imwrite(os.path.join(OUT_DIR, "{}_{}.bmp".format(CAM_ID_LEFT,  ts)), fl)
            cv2.imwrite(os.path.join(OUT_DIR, "{}_{}.bmp".format(CAM_ID_RIGHT, ts)), fr)
            msg = "saved pair  {}_{}.bmp  +  {}_{}.bmp".format(
                CAM_ID_LEFT, ts, CAM_ID_RIGHT, ts
            )

        elif self._mode == self.LEFT:
            fl = self._sub_left.get_frame()
            if fl is None:
                self._sbar.showMessage("  no frame from left", 3000)
                return
            cv2.imwrite(os.path.join(OUT_DIR, "{}_{}.bmp".format(CAM_ID_LEFT, ts)), fl)
            msg = "saved  {}_{}.bmp".format(CAM_ID_LEFT, ts)

        else:  # RIGHT
            fr = self._sub_right.get_frame()
            if fr is None:
                self._sbar.showMessage("  no frame from right", 3000)
                return
            cv2.imwrite(os.path.join(OUT_DIR, "{}_{}.bmp".format(CAM_ID_RIGHT, ts)), fr)
            msg = "saved  {}_{}.bmp".format(CAM_ID_RIGHT, ts)

        self.count += 1
        self._lbl_count.setText("Saved: {}".format(self.count))
        self._sbar.showMessage("[{:03d}]  {}".format(self.count, msg), 4000)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("cal_image_collector", anonymous=True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = CalCollectorWindow()
    win.show()

    threading.Thread(target=rospy.spin, daemon=True).start()
    sys.exit(app.exec_())