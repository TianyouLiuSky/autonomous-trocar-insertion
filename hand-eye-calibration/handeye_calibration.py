#!/usr/bin/env python
"""
he_calibration_unified.py  —  ATI Hand-Eye Calibration  (GUI only)
ROS1 Melodic | Python 3.6 | PyQt5

No robot motion - run run_calibration_poses.py in a separate terminal for that.

Workflow:
  1. Press "Set Anchor" (records current pose as reference)
  2. Press SPACE any time to record a sample (robot pose + board detection)
  3. Compute + Save when you have 20 diverse samples
"""

import os
import sys
import datetime
import threading
import csv

import cv2
import pyrealsense2 as rs
import numpy as np
import rospy
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QPushButton, QShortcut,
    QFrame, QGroupBox, QTextEdit
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_NAME  = 'SHER20'
ROBOT_TOPIC = '/{}/eye_robot/FrameEE'.format(ROBOT_NAME)

CAMERA_W, CAMERA_H, CAMERA_FPS = 1280, 720, 15

SQUARES_X   = 8
SQUARES_Y   = 6
SQUARE_LEN  = 0.010
MARKER_LEN  = 0.007
DICT_ID     = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 4
N_SAMPLES = 20
MIN_ROTATION_DEG = 5.0


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
DARK  = "#141414"
DARK2 = "#1a1a1a"
DARK3 = "#0e0e0e"
BDR   = "#444"
TEXT  = "#ddd"
DIM   = "#888"
GREEN = "#4c9"
BLUE  = "#7af"
RED   = "#e55"
AMBER = "#fa3"
MONO  = "monospace"

GLOBAL_STYLE = (
    "QWidget{{background:{D};color:{T};font-family:{M};font-size:11px;}}"
    "QGroupBox{{border:1px solid {B};border-radius:3px;margin-top:8px;"
    "padding:8px;color:{DM};}}"
    "QGroupBox::title{{subcontrol-origin:margin;left:8px;}}"
    "QLabel{{color:{DM};border:none;}}"
    "QPushButton{{background:#1e1e1e;color:{T};border:1px solid {B};"
    "border-radius:3px;padding:5px 12px;font-family:{M};font-size:11px;}}"
    "QPushButton:hover{{background:#2a2a2a;border-color:#666;}}"
    "QPushButton:pressed{{background:#111;}}"
    "QPushButton:disabled{{color:#444;border-color:#2a2a2a;background:#161616;}}"
    "QTextEdit{{background:{D3};color:{G};border:1px solid {B};"
    "font-family:{M};font-size:10px;}}"
).format(D=DARK, D2=DARK2, D3=DARK3, B=BDR, T=TEXT,
         DM=DIM, G=GREEN, BL=BLUE, M=MONO)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _bgr_to_pixmap(bgr, w, h):
    disp = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    qi   = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                  rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qi)

def _placeholder_px(w, h, msg="Waiting..."):
    img = np.full((h, w, 3), 26, dtype=np.uint8)
    cv2.putText(img, msg, (12, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (100,100,100), 1, cv2.LINE_AA)
    return _bgr_to_pixmap(img, w, h)

def _btn(text, color=None, h=34):
    b = QPushButton(text)
    b.setMinimumHeight(h)
    if color:
        b.setStyleSheet(
            "QPushButton{{background:{c};color:#fff;border:none;"
            "border-radius:3px;font-family:{m};font-size:11px;padding:5px 12px;}}"
            "QPushButton:hover{{background:{c}bb;}}"
            "QPushButton:disabled{{background:#222;color:#444;border:none;}}"
            .format(c=color, m=MONO))
    return b

def _lbl(text, style=""):
    l = QLabel(text)
    if style:
        l.setStyleSheet(style)
    return l

def _ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

def _timestamp():
    return datetime.datetime.now().strftime('%d%b%Y_%H%M%S').upper()

# ─────────────────────────────────────────────────────────────────────────────
# D405  (direct RealSense)
# ─────────────────────────────────────────────────────────────────────────────
class D405Camera(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5])
        print("D405 direct: {}x{} @ {}fps".format(CAMERA_W, CAMERA_H, CAMERA_FPS))

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self):
        self.pipeline.stop()

    @property
    def ready(self):
        return self.K is not None

# ─────────────────────────────────────────────────────────────────────────────
# ChArUco  (partial board)
# ─────────────────────────────────────────────────────────────────────────────
class ChArUcoDetector(object):
    def __init__(self):
        d = cv2.aruco.getPredefinedDictionary(DICT_ID)
        self.board    = cv2.aruco.CharucoBoard((SQUARES_X,SQUARES_Y), SQUARE_LEN, MARKER_LEN, d)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect(self, image, K, dist):
        corners, ids, _, _ = self.detector.detectBoard(image)
        if ids is None or len(ids) < MIN_CORNERS:
            return None, None, corners, ids
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < MIN_CORNERS:
            return None, None, corners, ids
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        return (rvec, tvec, corners, ids) if ok else (None, None, corners, ids)

# ─────────────────────────────────────────────────────────────────────────────
# Robot tracker  (read-only — just listens to FrameEE)
# ─────────────────────────────────────────────────────────────────────────────
class RobotTracker(object):
    def __init__(self):
        self._pose = None
        self._lock = threading.Lock()
        rospy.Subscriber(ROBOT_TOPIC, Transform, self._cb, queue_size=10)

    def _cb(self, msg):
        t = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        with self._lock:
            self._pose = {'t_mm': t, 't': t*0.001, 'q': q}

    @property
    def ready(self):
        with self._lock:
            return self._pose is not None

    def get_pose(self):
        with self._lock:
            return dict(self._pose) if self._pose else None

    def get_pose_rpy(self):
        p = self.get_pose()
        if p is None:
            return None
        euler = Rotation.from_quat(p['q']).as_euler('xyz', degrees=True)
        return np.concatenate([p['t_mm'], euler])

# ─────────────────────────────────────────────────────────────────────────────
# Calibration engine
# ─────────────────────────────────────────────────────────────────────────────
class HandEyeCalibrator(object):
    def __init__(self):
        self.robot_poses = []
        self.board_rvecs = []
        self.board_tvecs = []
        self._poses_log  = []   # flat rows for CSV

    def reset(self):
        self.__init__()

    def add_sample(self, robot_pose, rvec, tvec, stamp=None):
        R_board, _ = cv2.Rodrigues(rvec)
        is_diverse, min_angle = self._check_diversity(R_board)
        if not is_diverse and len(self.robot_poses) > 0:
            return False, min_angle

        self.robot_poses.append(robot_pose)
        self.board_rvecs.append(rvec.copy())
        self.board_tvecs.append(tvec.copy())
        euler = Rotation.from_quat(robot_pose['q']).as_euler('xyz', degrees=True)
        board_euler = Rotation.from_matrix(R_board).as_euler('xyz', degrees=True)
        ros_secs  = stamp.secs  if stamp is not None else 0
        ros_nsecs = stamp.nsecs if stamp is not None else 0
        self._poses_log.append({
            'sample':        self.n,
            'min_board_delta_deg': round(min_angle, 4),
            'ros_secs':      ros_secs,
            'ros_nsecs':     ros_nsecs,
            'ros_t_sec':     round(ros_secs + ros_nsecs * 1e-9, 6),
            'rob_tx_mm':     round(robot_pose['t_mm'][0], 4),
            'rob_ty_mm':     round(robot_pose['t_mm'][1], 4),
            'rob_tz_mm':     round(robot_pose['t_mm'][2], 4),
            'rob_qx':        round(robot_pose['q'][0], 6),
            'rob_qy':        round(robot_pose['q'][1], 6),
            'rob_qz':        round(robot_pose['q'][2], 6),
            'rob_qw':        round(robot_pose['q'][3], 6),
            'rob_roll_deg':  round(euler[0], 4),
            'rob_pitch_deg': round(euler[1], 4),
            'rob_yaw_deg':   round(euler[2], 4),
            'brd_tx_m':      round(tvec.flatten()[0], 6),
            'brd_ty_m':      round(tvec.flatten()[1], 6),
            'brd_tz_m':      round(tvec.flatten()[2], 6),
            'brd_roll_deg':  round(board_euler[0], 4),
            'brd_pitch_deg': round(board_euler[1], 4),
            'brd_yaw_deg':   round(board_euler[2], 4),
        })
        return True, min_angle

    def _check_diversity(self, R_new):
        if len(self.board_rvecs) == 0:
            return True, 0.0

        min_angle = 180.0
        for rvec in self.board_rvecs:
            R_existing, _ = cv2.Rodrigues(rvec)
            cos_angle = (np.trace(R_existing.T @ R_new) - 1) / 2
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            min_angle = min(min_angle, angle)
        return min_angle >= MIN_ROTATION_DEG, min_angle
        
    @property
    def n(self):
        return len(self.robot_poses)

    def can_calibrate(self):
        return self.n >= N_SAMPLES

    def calibrate(self):
        if not self.can_calibrate():
            return None
        x0 = np.zeros(12)
        def res(x):
            RY=Rotation.from_rotvec(x[0:3]).as_matrix(); tY=x[3:6].reshape(3,1)
            RX=Rotation.from_rotvec(x[6:9]).as_matrix(); tX=x[9:12].reshape(3,1)
            r=[]
            for rp,rv,tv in zip(self.robot_poses,self.board_rvecs,self.board_tvecs):
                RA=Rotation.from_quat(rp['q']).as_matrix(); tA=rp['t'].reshape(3,1)
                RB,_=cv2.Rodrigues(rv); tB=tv.reshape(3,1)
                r.extend((RA@RY-RX@RB).flatten())
                r.extend((RA@tY+tA-RX@tB-tX).flatten())
            return np.array(r)
        sol = least_squares(res, x0, ftol=1e-9, xtol=1e-9, max_nfev=2000)
        if not sol.success:
            return None
        RY=Rotation.from_rotvec(sol.x[0:3]).as_matrix(); tY=sol.x[3:6].reshape(3,1)
        RX=Rotation.from_rotvec(sol.x[6:9]).as_matrix(); tX=sol.x[9:12].reshape(3,1)
        errs=[]
        for rp,rv,tv in zip(self.robot_poses,self.board_rvecs,self.board_tvecs):
            RA=Rotation.from_quat(rp['q']).as_matrix(); tA=rp['t'].reshape(3,1)
            RB,_=cv2.Rodrigues(rv); tB=tv.reshape(3,1)
            errs.append(np.linalg.norm(RA@tY+tA-RX@tB-tX))
        Tc=np.eye(4); Tc[:3,:3]=RX; Tc[:3,3]=tX.flatten()
        Tb=np.eye(4); Tb[:3,:3]=RY; Tb[:3,3]=tY.flatten()
        return {'T_cam2base':Tc,'T_board2gripper':Tb,'err_mm':np.mean(errs)*1000,'n':self.n}

# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ATI  |  Hand-Eye Calibration")
        self.setStyleSheet(GLOBAL_STYLE)
        self.resize(1280, 800)

        self._cam     = D405Camera()
        self._det     = ChArUcoDetector()
        self._tracker = RobotTracker()
        self._cal     = HandEyeCalibrator()

        self._anchor  = None
        self._result  = None
        self._partial_warned = False

        self._build_ui()
        QShortcut(QKeySequence("Space"), self, activated=self._record)

        self._disp_timer = QTimer()
        self._disp_timer.timeout.connect(self._refresh_display)
        self._disp_timer.start(66)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        root = QVBoxLayout(c); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # header
        hdr = QFrame()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet("background:{};border-bottom:1px solid {};".format(DARK3,BDR))
        hl = QHBoxLayout(hdr); hl.setContentsMargins(10,0,10,0)
        hl.addWidget(_lbl("ATI  /  Hand-Eye Calibration  (AY = XB)",
                          "color:{};font-size:12px;font-weight:bold;".format(BLUE)))
        hl.addStretch()
        hl.addWidget(_lbl("mode: GUI only  -  run run_calibration_poses.py for motion",
                          "color:{};font-size:10px;".format(AMBER)))
        hl.addSpacing(20)
        self._cam_hdr   = _lbl("D405: waiting...", "color:{};font-size:10px;".format(DIM))
        self._robot_hdr = _lbl("Robot: connecting...", "color:{};font-size:10px;".format(AMBER))
        hl.addWidget(self._cam_hdr); hl.addSpacing(20); hl.addWidget(self._robot_hdr)
        root.addWidget(hdr)

        body = QHBoxLayout(); body.setContentsMargins(8,8,8,8); body.setSpacing(10)
        root.addLayout(body)

        # Left: camera
        left = QVBoxLayout(); left.setSpacing(6)
        feed = QGroupBox("D405 RGB  —  ChArUco Detection")
        fl   = QVBoxLayout(feed); fl.setContentsMargins(4,10,4,4)
        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setFixedSize(720,480)
        self._img_lbl.setPixmap(_placeholder_px(
            720, 480, "D405 direct: {}x{} @ {}fps".format(CAMERA_W, CAMERA_H, CAMERA_FPS)))
        fl.addWidget(self._img_lbl)
        dr = QHBoxLayout()
        self._board_lbl   = _lbl("● board  NOT DETECTED","color:{};font-size:10px;".format(RED))
        self._corners_lbl = _lbl("corners: 0","color:{};font-size:10px;".format(DIM))
        dr.addWidget(self._board_lbl); dr.addStretch(); dr.addWidget(self._corners_lbl)
        fl.addLayout(dr)
        left.addWidget(feed)
        pb = QGroupBox("Robot Pose  (live)")
        pl = QHBoxLayout(pb)
        self._pose_lbl = _lbl("—  waiting  —","color:{};font-size:10px;".format(GREEN))
        self._pose_lbl.setAlignment(Qt.AlignCenter)
        pl.addWidget(self._pose_lbl)
        left.addWidget(pb)
        body.addLayout(left,5)

        # Right: controls
        right = QVBoxLayout(); right.setSpacing(8)

        # Set Anchor
        ab = QGroupBox("Set Anchor")
        al = QVBoxLayout(ab); al.setSpacing(6)
        al.addWidget(_lbl("Press before starting run_calibration_poses.py.",
                          "color:{};font-size:10px;".format(DIM)))
        self._btn_anchor = _btn("Set Anchor  (current pose)","#2a5a2a",h=40)
        self._btn_anchor.clicked.connect(self._set_anchor)
        al.addWidget(self._btn_anchor)
        self._anchor_lbl = _lbl("No anchor set","color:{};font-size:10px;".format(DIM))
        self._anchor_lbl.setAlignment(Qt.AlignCenter)
        al.addWidget(self._anchor_lbl)
        right.addWidget(ab)

        # Record
        rb = QGroupBox("Record Samples")
        rl = QVBoxLayout(rb); rl.setSpacing(6)
        self._samples_lbl = _lbl("0 / {} samples".format(N_SAMPLES),"color:{};font-size:12px;".format(AMBER))
        self._samples_lbl.setAlignment(Qt.AlignCenter)
        rl.addWidget(self._samples_lbl)
        self._btn_space = _btn("Record  [SPACE]","#2a5a2a",h=48)
        self._btn_space.clicked.connect(self._record)
        rl.addWidget(self._btn_space)
        self._btn_reset = _btn("Reset All Samples",h=30)
        self._btn_reset.clicked.connect(self._reset)
        rl.addWidget(self._btn_reset)
        right.addWidget(rb)

        # Compute & Save
        cb = QGroupBox("Compute & Save")
        cl = QVBoxLayout(cb); cl.setSpacing(6)
        self._btn_compute = _btn("Compute Calibration","#603a00",h=40)
        self._btn_compute.clicked.connect(self._compute)
        self._btn_compute.setEnabled(False)
        cl.addWidget(self._btn_compute)
        self._err_lbl = _lbl("","color:{};font-size:11px;".format(DIM))
        self._err_lbl.setAlignment(Qt.AlignCenter)
        cl.addWidget(self._err_lbl)
        self._btn_save = _btn("Save  (.npz)","#1a4a1a",h=40)
        self._btn_save.clicked.connect(self._save)
        self._btn_save.setEnabled(False)
        cl.addWidget(self._btn_save)
        self._btn_csv = _btn("Save Poses  (.csv)", h=34)
        self._btn_csv.clicked.connect(self._save_csv)
        cl.addWidget(self._btn_csv)
        right.addWidget(cb)
        

        # Log
        lb = QGroupBox("Log")
        ll = QVBoxLayout(lb)
        self._log = QTextEdit(); self._log.setReadOnly(True)
        ll.addWidget(self._log)
        right.addWidget(lb,1)

        body.addLayout(right,2)



    # ── anchor ────────────────────────────────────────────────────────────
    def _set_anchor(self):
        rpy = self._tracker.get_pose_rpy()
        if rpy is None:
            self._log_msg("x Robot not ready"); return
        self._anchor = rpy.copy()
        self._anchor_lbl.setText(
            "x={:.2f}  y={:.2f}  z={:.2f} mm  r={:.1f}  p={:.1f}  yaw={:.1f}".format(*rpy))
        self._anchor_lbl.setStyleSheet("color:{};font-size:10px;".format(GREEN))
        self._log_msg("Anchor set: {}".format(rpy.round(2)))

    # ── record ────────────────────────────────────────────────────────────
    def _record(self):
        frame = self._cam.get_frame()
        if frame is None or self._cam.K is None:
            self._log_msg("x No camera frame"); return

        rvec, tvec, _, _ = self._det.detect(frame, self._cam.K, self._cam.dist)
        if rvec is None:
            self._log_msg("x Board not detected"); return

        rp = self._tracker.get_pose()
        if rp is None:
            self._log_msg("x No robot pose"); return

        stamp = rospy.Time.now()
        accepted, min_angle = self._cal.add_sample(rp, rvec, tvec, stamp)
        if not accepted:
            self._log_msg("x Too similar ({:.1f} deg < {:.1f} deg); move to a more different board pose".format(
                min_angle, MIN_ROTATION_DEG))
            return

        if self._cal.n == 1:
            self._log_msg("v Sample {}/{}".format(self._cal.n, N_SAMPLES))
        else:
            self._log_msg("v Sample {}/{}  min board delta={:.1f} deg".format(
                self._cal.n, N_SAMPLES, min_angle))
        self._samples_lbl.setText("{} / {} samples".format(self._cal.n, N_SAMPLES))
        self._samples_lbl.setStyleSheet("color:{};font-size:12px;".format(GREEN))
        if self._cal.can_calibrate():
            self._btn_compute.setEnabled(True)

    def _reset(self):
        self._cal.reset()
        self._result = None
        self._samples_lbl.setText("0 / {} samples".format(N_SAMPLES))
        self._samples_lbl.setStyleSheet("color:{};font-size:12px;".format(AMBER))
        self._btn_compute.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._err_lbl.setText("")
        self._log_msg("-- Samples reset --")

    # ── calibration ───────────────────────────────────────────────────────
    def _compute(self):
        self._err_lbl.setText("Computing...")
        QApplication.processEvents()
        r = self._cal.calibrate()
        if r is None:
            self._err_lbl.setText("x Failed - need {} diverse samples".format(N_SAMPLES))
            self._err_lbl.setStyleSheet("color:{};font-size:11px;".format(RED)); return
        self._result = r
        col = GREEN if r['err_mm']<1.0 else AMBER if r['err_mm']<3.0 else RED
        self._err_lbl.setText("Mean error: {:.3f} mm  |  {} samples".format(r['err_mm'],r['n']))
        self._err_lbl.setStyleSheet("color:{};font-size:11px;".format(col))
        self._log_msg("Done — err={:.3f}mm  n={}".format(r['err_mm'],r['n']))
        self._btn_save.setEnabled(True)

    def _save(self):
        if not self._result: return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR,"hand_eye_cal_{}.npz".format(_timestamp()))
        np.savez(path,
                 T_cam2base=self._result['T_cam2base'],
                 T_board2gripper=self._result['T_board2gripper'],
                 camera_matrix=self._cam.K, dist_coeffs=self._cam.dist,
                 err_mm=self._result['err_mm'], n_samples=self._result['n'])
        self._log_msg("v Saved -> {}".format(path))
        self._err_lbl.setText("v Saved: {}".format(os.path.basename(path)))

    def _save_csv(self):
        rows = self._cal._poses_log
        if not rows:
            self._log_msg("x No poses to save"); return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "he_poses_{}.csv".format(_timestamp()))
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self._log_msg("v CSV -> {}".format(path))

    def _log_msg(self, msg):
        self._log.append("[{}]  {}".format(_ts(), msg))

    # ── display refresh ──────────────────────────────────────────────────
    def _refresh_display(self):
        if self._tracker.ready:
            rpy = self._tracker.get_pose_rpy()
            self._robot_hdr.setText("Robot v  [{:.1f},{:.1f},{:.1f}] mm".format(*rpy[:3]))
            self._robot_hdr.setStyleSheet("color:{};font-size:10px;".format(GREEN))
            self._pose_lbl.setText(
                "x={:.2f}  y={:.2f}  z={:.2f} mm  |  r={:.1f}  p={:.1f}  yaw={:.1f} deg".format(*rpy))
            self._pose_lbl.setStyleSheet("color:{};font-size:10px;".format(GREEN))
        else:
            self._robot_hdr.setText("Robot: no signal")
            self._robot_hdr.setStyleSheet("color:{};font-size:10px;".format(RED))

        if self._cam.ready:
            self._cam_hdr.setText("D405 direct v")
            self._cam_hdr.setStyleSheet("color:{};font-size:10px;".format(GREEN))
        elif self._cam.K is not None:
            self._cam_hdr.setText("D405: no image yet")
            self._cam_hdr.setStyleSheet("color:{};font-size:10px;".format(AMBER))

        frame = self._cam.get_frame()
        if frame is not None:
            disp = frame.copy()
            nc = 0
            if self._cam.K is not None:
                rvec, tvec, corners, ids = self._det.detect(disp, self._cam.K, self._cam.dist)
                if ids is not None:
                    nc = len(ids)
                    cv2.aruco.drawDetectedCornersCharuco(disp, corners)
                                # AFTER
                if rvec is not None:
                    cv2.drawFrameAxes(disp, self._cam.K, self._cam.dist, rvec, tvec, 0.015)
                    self._board_lbl.setText("● board  DETECTED")
                    self._board_lbl.setStyleSheet("color:{};font-size:10px;".format(GREEN))
                    self._partial_warned = False

                    # Project board origin into image and draw coordinate overlay
                    origin_3d = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
                    pt2d, _ = cv2.projectPoints(origin_3d, rvec, tvec, self._cam.K, self._cam.dist)
                    ox, oy = int(pt2d[0][0][0]), int(pt2d[0][0][1])
                    cv2.drawMarker(disp, (ox, oy), (0, 255, 200), cv2.MARKER_CROSS, 14, 2)
                    tx, ty, tz = tvec.flatten()
                    coord_str = "x={:.1f}  y={:.1f}  z={:.1f} mm".format(tx*1000, ty*1000, tz*1000)
                    # Draw in bottom-left of the image
                    ih, iw = disp.shape[:2]
                    cv2.rectangle(disp, (6, ih-28), (6 + len(coord_str)*7 + 4, ih-6), (20,20,20), -1)
                    cv2.putText(disp, coord_str, (8, ih-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)
                else:
                    if nc > 0:
                        self._board_lbl.setText("● board  PARTIAL ({} corners)".format(nc))
                        self._board_lbl.setStyleSheet("color:{};font-size:10px;".format(AMBER))
                        if not self._partial_warned:
                            self._log_msg("~ Board partially visible ({} corners) — OK for ChArUco".format(nc))
                            self._partial_warned = True
                    else:
                        self._board_lbl.setText("● board  NOT DETECTED")
                        self._board_lbl.setStyleSheet("color:{};font-size:10px;".format(RED))
                        self._partial_warned = False
            self._corners_lbl.setText("corners: {}".format(nc))
            self._img_lbl.setPixmap(_bgr_to_pixmap(disp,720,480))

    def closeEvent(self, event):
        self._disp_timer.stop()
        self._cam.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node('ati_he_calibration', anonymous=True)
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
