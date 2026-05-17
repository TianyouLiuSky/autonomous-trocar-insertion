#!/usr/bin/env python3
"""
collect_cal_images_leica.py  —  Calibration image collector for Leica DeckLink
ROS1 Melodic | Python 3 | PyQt5

Run:   python3 collect_cal_images_leica.py
SPACE = capture    Ctrl+Q = quit
"""

import os
import sys
import datetime
import threading

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
import time

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton, QShortcut,
    QStatusBar, QFrame
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TOPIC    = "/decklink/camera/image_raw"
OUT_DIR = "./data/{datetime.datetime.now().strftime('%d%b%Y').upper()}"

DISPLAY_W = 960
DISPLAY_H = 540
TIMER_MS  = 66   # ~15 Hz display


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber
# ─────────────────────────────────────────────────────────────────────────────
class CameraSubscriber:
    def __init__(self, topic):
        self._lock  = threading.Lock()
        self._frame = None
        rospy.Subscriber(topic, Image, self._cb, queue_size=1, buff_size=2**24)

    def _cb(self, msg):
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            enc = msg.encoding.lower()
            h, w = msg.height, msg.width
            if   enc == "rgb8":
                bgr = raw.reshape(h, w, 3)[:, :, ::-1].copy()
            elif enc == "bgr8":
                bgr = raw.reshape(h, w, 3).copy()
            elif enc in ("yuv422", "yuv422"):
                bgr = cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
            elif enc == "uyvy":
                bgr = cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
            elif enc == "mono8":
                bgr = cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2BGR)
            else:
                bgr = raw.reshape(h, w, -1)[:, :, :3].copy()
            with self._lock:
                self._frame = bgr
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[leica_cal_collector] decode ({msg.encoding}): {e}")

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
    return datetime.datetime.now().strftime("%d%b%Y_%H%M%S_%f")[:22].upper()


def _bgr_to_pixmap(bgr, w, h):
    disp = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                  rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _placeholder(w, h):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.putText(img, "Waiting for stream...", (16, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, img.shape[1], img.shape[0],
                  img.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class LeicaCalCollector(QMainWindow):
    def __init__(self):
        super().__init__()
        self._count = 0
        os.makedirs(OUT_DIR, exist_ok=True)

        self._sub = CameraSubscriber(TOPIC)
        self._build_ui()

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._capture)
        QShortcut(QKeySequence("Ctrl+Q"),     self, activated=self.close)

        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(TIMER_MS)

    def _build_ui(self):
        self.setWindowTitle("Leica Intrinsic Cal — Image Collector")
        self.setStyleSheet("background:#141414; color:#ddd;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── header ────────────────────────────────────────────────────────
        header = QHBoxLayout()

        topic_lbl = QLabel(f"● {TOPIC}")
        topic_lbl.setStyleSheet(
            "color:#7af; font-family:monospace; font-size:11px;"
        )

        path_lbl = QLabel(f"→  {os.path.abspath(OUT_DIR)}")
        path_lbl.setStyleSheet(
            "color:#888; font-family:monospace; font-size:11px;"
        )

        self._lbl_count = QLabel("Saved: 0")
        self._lbl_count.setStyleSheet(
            "color:#4c9; font-family:monospace; font-size:14px; font-weight:bold; padding:0 8px;"
        )

        header.addWidget(topic_lbl)
        header.addWidget(path_lbl, stretch=1)
        header.addWidget(self._lbl_count)
        root.addLayout(header)

        # ── camera panel ──────────────────────────────────────────────────
        cam_frame = QFrame()
        cam_frame.setStyleSheet(
            "QFrame { border:1px solid #444; background:#1a1a1a; }"
        )
        cam_lay = QVBoxLayout(cam_frame)
        cam_lay.setContentsMargins(2, 2, 2, 2)
        cam_lay.setSpacing(2)

        title = QLabel("Leica Proveo 8  (DeckLink)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color:#888; font-size:10px; font-family:monospace; padding:3px; border:none;"
        )

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setMinimumSize(DISPLAY_W, DISPLAY_H)
        self._img_lbl.setPixmap(_placeholder(DISPLAY_W, DISPLAY_H))
        self._img_lbl.setStyleSheet("background:#111; border:none;")

        self._stream_lbl = QLabel("● no stream")
        self._stream_lbl.setAlignment(Qt.AlignCenter)
        self._stream_lbl.setStyleSheet(
            "color:#555; font-size:10px; font-family:monospace; padding:2px; border:none;"
        )

        cam_lay.addWidget(title)
        cam_lay.addWidget(self._img_lbl, stretch=1)
        cam_lay.addWidget(self._stream_lbl)
        root.addWidget(cam_frame, stretch=1)

        # ── capture button ────────────────────────────────────────────────
        self._btn = QPushButton("  CAPTURE  [SPACE]")
        self._btn.setFixedHeight(46)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  background:#2a6099; color:#fff; border:none;"
            "  font-size:14px; font-family:monospace; font-weight:bold; border-radius:4px;"
            "}"
            "QPushButton:hover   { background:#3a80bb; }"
            "QPushButton:pressed { background:#1a4070; }"
        )
        self._btn.clicked.connect(self._capture)
        root.addWidget(self._btn)

        # ── status bar ────────────────────────────────────────────────────
        self._sbar = QStatusBar()
        self._sbar.setStyleSheet(
            "color:#888; font-family:monospace; font-size:10px;"
        )
        self.setStatusBar(self._sbar)
        self._sbar.showMessage("SPACE = capture    Ctrl+Q = quit")

        self.resize(DISPLAY_W + 24, DISPLAY_H + 140)

    # ── display refresh ───────────────────────────────────────────────────
    def _refresh(self):
        frame = self._sub.get_frame()
        if frame is not None:
            self._img_lbl.setPixmap(
                _bgr_to_pixmap(frame, self._img_lbl.width(), self._img_lbl.height())
            )
            h, w = frame.shape[:2]
            self._stream_lbl.setText(f"● streaming  [{w}×{h}]")
            self._stream_lbl.setStyleSheet(
                "color:#4c9; font-size:10px; font-family:monospace; padding:2px; border:none;"
            )
        else:
            self._stream_lbl.setText("● no stream")
            self._stream_lbl.setStyleSheet(
                "color:#555; font-size:10px; font-family:monospace; padding:2px; border:none;"
            )

    # ── capture ───────────────────────────────────────────────────────────
    def _capture(self):
        frame = self._sub.get_frame()
        if frame is None:
            self._sbar.showMessage("  no frame — is the DeckLink streaming?", 3000)
            return

        ts   = _timestamp()
        fname = f"leica_{ts}.bmp"
        path  = os.path.join(OUT_DIR, fname)
        cv2.imwrite(path, frame)

        self._count += 1
        self._lbl_count.setText(f"Saved: {self._count}")
        self._sbar.showMessage(f"[{self._count:03d}]  saved  {fname}", 4000)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("leica_cal_image_collector", anonymous=True)

    app = QApplication.instance() or QApplication(sys.argv)
    win = LeicaCalCollector()
    win.show()

    threading.Thread(target=rospy.spin, daemon=True).start()
    sys.exit(app.exec_())