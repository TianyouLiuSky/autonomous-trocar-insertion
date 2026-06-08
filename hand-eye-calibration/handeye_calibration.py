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
import argparse

import cv2
import pyrealsense2 as rs
import numpy as np
import rospy
from geometry_msgs.msg import Transform
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from camera_intrinsics import load_intrinsics
from handeye_math import (
    estimate_rotations_from_relative_motion,
    rotation_angle_deg,
    solution_vector,
    solve_translations,
    transforms_from_samples,
)

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
DISPLAY_W = 720
DISPLAY_H = int(DISPLAY_W * float(CAMERA_H) / float(CAMERA_W))

SQUARES_X   = 8
SQUARES_Y   = 6
SQUARE_LEN  = 0.010
MARKER_LEN  = 0.007
DICT_ID     = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 8
N_SAMPLES = 20
MIN_ROTATION_DEG = 5.0

# Residuals are normalized before optimization. By default, a 0.5 degree
# rotation error and a 1.0 mm translation error have equal influence. Increase
# a scale to reduce that component's influence. Environment overrides make
# weight experiments possible without editing this file.
SOLVER_PROFILE = 'relative_init_robust_multistart_v2'
SOLVER_ROTATION_SCALE_DEG = float(
    os.environ.get('HE_SOLVER_ROTATION_SCALE_DEG', '0.5'))
SOLVER_TRANSLATION_SCALE_MM = float(
    os.environ.get('HE_SOLVER_TRANSLATION_SCALE_MM', '1.0'))
SOLVER_ROBUST_LOSS = os.environ.get(
    'HE_SOLVER_ROBUST_LOSS', 'soft_l1')
SOLVER_MULTISTARTS = int(
    os.environ.get('HE_SOLVER_MULTISTARTS', '5'))


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
    def __init__(self, intrinsics_path=None):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist = np.array(intr.coeffs[:5])
        self.intrinsics_source = "D405 factory"
        fitted = load_intrinsics(
            intrinsics_path, CAMERA_W, CAMERA_H)
        if fitted is not None:
            self.K, self.dist, self.intrinsics_source = fitted
        print("D405 direct: {}x{} @ {}fps".format(CAMERA_W, CAMERA_H, CAMERA_FPS))
        print("Camera intrinsics: {}".format(self.intrinsics_source))

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
        try:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        except cv2.error:
            return None, None, corners, ids
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
        is_diverse, min_angle, closest_sample = self._check_diversity(R_board)
        if not is_diverse and len(self.robot_poses) > 0:
            return False, min_angle, closest_sample

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
        return True, min_angle, closest_sample

    def _check_diversity(self, R_new):
        if len(self.board_rvecs) == 0:
            return True, 0.0, None

        min_angle = 180.0
        closest_sample = None
        for i, rvec in enumerate(self.board_rvecs, start=1):
            R_existing, _ = cv2.Rodrigues(rvec)
            cos_angle = (np.trace(R_existing.T @ R_new) - 1) / 2
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            if angle < min_angle:
                min_angle = angle
                closest_sample = i
        return min_angle >= MIN_ROTATION_DEG, min_angle, closest_sample
        
    @property
    def n(self):
        return len(self.robot_poses)

    def can_calibrate(self):
        return self.n >= N_SAMPLES

    @staticmethod
    def _unpack_solution(x):
        RY = Rotation.from_rotvec(x[0:3]).as_matrix()
        tY = x[3:6].reshape(3, 1)
        RX = Rotation.from_rotvec(x[6:9]).as_matrix()
        tX = x[9:12].reshape(3, 1)
        return RY, tY, RX, tX

    def _equation_terms(self, x):
        RY, tY, RX, tX = self._unpack_solution(x)
        for rp, rv, tv in zip(
                self.robot_poses, self.board_rvecs, self.board_tvecs):
            RA = Rotation.from_quat(rp['q']).as_matrix()
            tA = rp['t'].reshape(3, 1)
            RB, _ = cv2.Rodrigues(rv)
            tB = tv.reshape(3, 1)
            yield RA @ RY, RA @ tY + tA, RX @ RB, RX @ tB + tX

    def _legacy_residual(self, x):
        residuals = []
        for R_left, t_left, R_right, t_right in self._equation_terms(x):
            residuals.extend((R_left - R_right).flatten())
            residuals.extend((t_left - t_right).flatten())
        return np.asarray(residuals)

    def _weighted_residual(self, x):
        rotation_scale_rad = np.deg2rad(SOLVER_ROTATION_SCALE_DEG)
        translation_scale_m = SOLVER_TRANSLATION_SCALE_MM * 0.001
        residuals = []
        for R_left, t_left, R_right, t_right in self._equation_terms(x):
            R_error = R_right.T @ R_left
            rotation_error = Rotation.from_matrix(R_error).as_rotvec()
            translation_error = (t_left - t_right).flatten()
            residuals.extend(rotation_error / rotation_scale_rad)
            residuals.extend(translation_error / translation_scale_m)
        return np.asarray(residuals)

    def _solution_diagnostics(self, x):
        translation_errors_mm = []
        rotation_errors_deg = []
        for R_left, t_left, R_right, t_right in self._equation_terms(x):
            translation_errors_mm.append(
                np.linalg.norm(t_left - t_right) * 1000.0)
            R_error = R_right.T @ R_left
            rotation_errors_deg.append(
                np.linalg.norm(Rotation.from_matrix(R_error).as_rotvec())
                * 180.0 / np.pi)
        return {
            'translation_mean_mm': float(np.mean(translation_errors_mm)),
            'translation_max_mm': float(np.max(translation_errors_mm)),
            'rotation_mean_deg': float(np.mean(rotation_errors_deg)),
            'rotation_max_deg': float(np.max(rotation_errors_deg)),
        }

    def _relative_motion_initialization(self):
        (
            robot_rotations,
            robot_translations,
            board_rotations,
            board_translations,
        ) = transforms_from_samples(
            self.robot_poses, self.board_rvecs, self.board_tvecs)
        rotation_fit = estimate_rotations_from_relative_motion(
            robot_rotations, board_rotations)
        translation_fit = solve_translations(
            robot_rotations,
            robot_translations,
            board_translations,
            rotation_fit['camera_rotation'],
        )
        initial = solution_vector(
            rotation_fit['board_rotation'],
            translation_fit['board_translation'],
            rotation_fit['camera_rotation'],
            translation_fit['camera_translation'],
        )
        singular_values = rotation_fit['singular_values']
        nullspace_gap = float(
            singular_values[-2] / max(singular_values[-1], 1e-15)
        )
        return initial, {
            'rotation_nullspace_gap': nullspace_gap,
            'translation_condition_number':
                translation_fit['condition_number'],
            'translation_rank': translation_fit['rank'],
            'relative_rotation_mean_deg': float(np.mean(
                rotation_fit['sample_errors_deg'])),
            'relative_rotation_max_deg': float(np.max(
                rotation_fit['sample_errors_deg'])),
        }

    def _multistart_initializations(self, relative_x, legacy_x):
        count = max(2, SOLVER_MULTISTARTS)
        starts = [
            ('relative_motion', relative_x.copy()),
            ('legacy', legacy_x.copy()),
        ]
        rng = np.random.RandomState(20260608)
        while len(starts) < count:
            base_name, base = starts[(len(starts) - 2) % 2]
            perturbed = base.copy()
            perturbed[0:3] += rng.normal(
                0.0, np.deg2rad(2.0), size=3)
            perturbed[6:9] += rng.normal(
                0.0, np.deg2rad(2.0), size=3)
            perturbed[3:6] += rng.normal(0.0, 0.002, size=3)
            perturbed[9:12] += rng.normal(0.0, 0.002, size=3)
            starts.append(
                ('{}_perturbed_{}'.format(base_name, len(starts)), perturbed)
            )
        return starts

    @staticmethod
    def _jacobian_condition(jacobian):
        singular_values = np.linalg.svd(
            jacobian, compute_uv=False)
        if singular_values[-1] < 1e-15:
            return float('inf')
        return float(singular_values[0] / singular_values[-1])

    @staticmethod
    def _matrices_from_solution(x):
        RY, tY, RX, tX = HandEyeCalibrator._unpack_solution(x)
        Tc = np.eye(4)
        Tc[:3, :3] = RX
        Tc[:3, 3] = tX.flatten()
        Tb = np.eye(4)
        Tb[:3, :3] = RY
        Tb[:3, 3] = tY.flatten()
        return Tc, Tb

    def calibrate(self):
        if not self.can_calibrate():
            return None
        x0 = np.zeros(12)

        # First reproduce the old solution. It provides a stable initialization
        # and a direct same-dataset baseline for the weighting experiment.
        legacy_sol = least_squares(
            self._legacy_residual, x0, ftol=1e-9, xtol=1e-9,
            gtol=1e-9, max_nfev=2000)
        if not legacy_sol.success:
            return None

        if SOLVER_ROBUST_LOSS not in (
                'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'):
            raise ValueError(
                "Unsupported HE_SOLVER_ROBUST_LOSS={!r}".format(
                    SOLVER_ROBUST_LOSS)
            )

        try:
            relative_x, conditioning = (
                self._relative_motion_initialization())
        except Exception as exc:
            print(
                "Relative-motion initialization failed; using legacy "
                "initialization: {}".format(exc)
            )
            relative_x = legacy_sol.x.copy()
            conditioning = {
                'rotation_nullspace_gap': float('nan'),
                'translation_condition_number': float('nan'),
                'translation_rank': 0,
                'relative_rotation_mean_deg': float('nan'),
                'relative_rotation_max_deg': float('nan'),
            }

        candidates = []
        for start_name, start_x in self._multistart_initializations(
                relative_x, legacy_sol.x):
            solution = least_squares(
                self._weighted_residual,
                start_x,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                x_scale='jac',
                loss=SOLVER_ROBUST_LOSS,
                f_scale=1.0,
                max_nfev=6000,
            )
            if solution.success and np.all(np.isfinite(solution.x)):
                candidates.append((solution.cost, start_name, solution))
        if not candidates:
            return None

        _, selected_start, weighted_sol = min(
            candidates, key=lambda item: item[0])

        Tc, Tb = self._matrices_from_solution(weighted_sol.x)
        legacy_Tc, legacy_Tb = self._matrices_from_solution(legacy_sol.x)
        metrics = self._solution_diagnostics(weighted_sol.x)
        legacy_metrics = self._solution_diagnostics(legacy_sol.x)

        solved_camera_rotations = []
        solved_camera_translations = []
        for _, _, candidate in candidates:
            candidate_tc, _ = self._matrices_from_solution(candidate.x)
            solved_camera_rotations.append(candidate_tc[:3, :3])
            solved_camera_translations.append(candidate_tc[:3, 3])
        rotation_spread_deg = 0.0
        translation_spread_mm = 0.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                rotation_spread_deg = max(
                    rotation_spread_deg,
                    rotation_angle_deg(
                        solved_camera_rotations[j]
                        @ solved_camera_rotations[i].T),
                )
                translation_spread_mm = max(
                    translation_spread_mm,
                    np.linalg.norm(
                        solved_camera_translations[j]
                        - solved_camera_translations[i]) * 1000.0,
                )
        return {
            'T_cam2base': Tc,
            'T_board2gripper': Tb,
            'legacy_T_cam2base': legacy_Tc,
            'legacy_T_board2gripper': legacy_Tb,
            'err_mm': metrics['translation_mean_mm'],
            'translation_err_mean_mm': metrics['translation_mean_mm'],
            'translation_err_max_mm': metrics['translation_max_mm'],
            'rotation_err_mean_deg': metrics['rotation_mean_deg'],
            'rotation_err_max_deg': metrics['rotation_max_deg'],
            'legacy_translation_err_mean_mm':
                legacy_metrics['translation_mean_mm'],
            'legacy_translation_err_max_mm':
                legacy_metrics['translation_max_mm'],
            'legacy_rotation_err_mean_deg':
                legacy_metrics['rotation_mean_deg'],
            'legacy_rotation_err_max_deg':
                legacy_metrics['rotation_max_deg'],
            'solver_profile': SOLVER_PROFILE,
            'solver_robust_loss': SOLVER_ROBUST_LOSS,
            'solver_selected_start': selected_start,
            'solver_successful_starts': len(candidates),
            'solver_requested_starts': max(2, SOLVER_MULTISTARTS),
            'solver_rotation_scale_deg': SOLVER_ROTATION_SCALE_DEG,
            'solver_translation_scale_mm': SOLVER_TRANSLATION_SCALE_MM,
            'solver_jacobian_condition':
                self._jacobian_condition(weighted_sol.jac),
            'rotation_nullspace_gap':
                conditioning['rotation_nullspace_gap'],
            'translation_condition_number':
                conditioning['translation_condition_number'],
            'translation_rank': conditioning['translation_rank'],
            'relative_rotation_mean_deg':
                conditioning['relative_rotation_mean_deg'],
            'relative_rotation_max_deg':
                conditioning['relative_rotation_max_deg'],
            'multistart_camera_rotation_spread_deg':
                rotation_spread_deg,
            'multistart_camera_translation_spread_mm':
                translation_spread_mm,
            'n': self.n,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self, intrinsics_path=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ATI  |  Hand-Eye Calibration")
        self.setStyleSheet(GLOBAL_STYLE)
        self.resize(1280, 800)

        self._cam     = D405Camera(intrinsics_path)
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
        self._img_lbl.setFixedSize(DISPLAY_W, DISPLAY_H)
        self._img_lbl.setPixmap(_placeholder_px(
            DISPLAY_W, DISPLAY_H, "D405 direct: {}x{} @ {}fps".format(CAMERA_W, CAMERA_H, CAMERA_FPS)))
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
        accepted, min_angle, closest_sample = self._cal.add_sample(rp, rvec, tvec, stamp)
        if not accepted:
            self._log_msg(
                "x Too similar ({:.1f} deg < {:.1f} deg) to accepted sample {}; "
                "move to a more different board pose".format(
                    min_angle, MIN_ROTATION_DEG, closest_sample
                )
            )
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
        self._err_lbl.setText(
            "Weighted: t={:.3f} mm, r={:.3f} deg  |  {} samples".format(
                r['translation_err_mean_mm'],
                r['rotation_err_mean_deg'],
                r['n']))
        self._err_lbl.setStyleSheet("color:{};font-size:11px;".format(col))
        self._log_msg(
            "Weighted ({}, {:.2f}deg={:.1f}mm): "
            "t mean/max={:.3f}/{:.3f}mm, "
            "r mean/max={:.3f}/{:.3f}deg".format(
                r['solver_profile'],
                r['solver_rotation_scale_deg'],
                r['solver_translation_scale_mm'],
                r['translation_err_mean_mm'],
                r['translation_err_max_mm'],
                r['rotation_err_mean_deg'],
                r['rotation_err_max_deg']))
        self._log_msg(
            "Legacy same samples: t mean/max={:.3f}/{:.3f}mm, "
            "r mean/max={:.3f}/{:.3f}deg".format(
                r['legacy_translation_err_mean_mm'],
                r['legacy_translation_err_max_mm'],
                r['legacy_rotation_err_mean_deg'],
                r['legacy_rotation_err_max_deg']))
        self._log_msg(
            "Robust solver: loss={}, start={}, starts={}/{}, "
            "Jacobian cond={:.3g}".format(
                r['solver_robust_loss'],
                r['solver_selected_start'],
                r['solver_successful_starts'],
                r['solver_requested_starts'],
                r['solver_jacobian_condition']))
        self._log_msg(
            "Initialization conditioning: rotation nullspace gap={:.3g}, "
            "translation cond={:.3g}, rank={}".format(
                r['rotation_nullspace_gap'],
                r['translation_condition_number'],
                r['translation_rank']))
        self._log_msg(
            "Multi-start X spread: rotation={:.6f}deg, "
            "translation={:.6f}mm".format(
                r['multistart_camera_rotation_spread_deg'],
                r['multistart_camera_translation_spread_mm']))
        self._btn_save.setEnabled(True)

    def _save(self):
        if not self._result: return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR,"hand_eye_cal_{}.npz".format(_timestamp()))
        np.savez(path,
                 T_cam2base=self._result['T_cam2base'],
                 T_board2gripper=self._result['T_board2gripper'],
                 legacy_T_cam2base=self._result['legacy_T_cam2base'],
                 legacy_T_board2gripper=self._result['legacy_T_board2gripper'],
                 camera_matrix=self._cam.K, dist_coeffs=self._cam.dist,
                 camera_intrinsics_source=self._cam.intrinsics_source,
                 err_mm=self._result['err_mm'],
                 translation_err_mean_mm=self._result['translation_err_mean_mm'],
                 translation_err_max_mm=self._result['translation_err_max_mm'],
                 rotation_err_mean_deg=self._result['rotation_err_mean_deg'],
                 rotation_err_max_deg=self._result['rotation_err_max_deg'],
                 legacy_translation_err_mean_mm=
                     self._result['legacy_translation_err_mean_mm'],
                 legacy_translation_err_max_mm=
                     self._result['legacy_translation_err_max_mm'],
                 legacy_rotation_err_mean_deg=
                     self._result['legacy_rotation_err_mean_deg'],
                 legacy_rotation_err_max_deg=
                     self._result['legacy_rotation_err_max_deg'],
                 solver_profile=self._result['solver_profile'],
                 solver_robust_loss=self._result['solver_robust_loss'],
                 solver_selected_start=
                     self._result['solver_selected_start'],
                 solver_successful_starts=
                     self._result['solver_successful_starts'],
                 solver_requested_starts=
                     self._result['solver_requested_starts'],
                 solver_rotation_scale_deg=
                     self._result['solver_rotation_scale_deg'],
                 solver_translation_scale_mm=
                     self._result['solver_translation_scale_mm'],
                 solver_jacobian_condition=
                     self._result['solver_jacobian_condition'],
                 rotation_nullspace_gap=
                     self._result['rotation_nullspace_gap'],
                 translation_condition_number=
                     self._result['translation_condition_number'],
                 translation_rank=self._result['translation_rank'],
                 relative_rotation_mean_deg=
                     self._result['relative_rotation_mean_deg'],
                 relative_rotation_max_deg=
                     self._result['relative_rotation_max_deg'],
                 multistart_camera_rotation_spread_deg=
                     self._result[
                         'multistart_camera_rotation_spread_deg'],
                 multistart_camera_translation_spread_mm=
                     self._result[
                         'multistart_camera_translation_spread_mm'],
                 n_samples=self._result['n'])
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
            self._img_lbl.setPixmap(_bgr_to_pixmap(disp, DISPLAY_W, DISPLAY_H))

    def closeEvent(self, event):
        self._disp_timer.stop()
        self._cam.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run D405 fixed-camera hand-eye calibration.")
    parser.add_argument(
        "--intrinsics", default=os.environ.get("HE_CAMERA_INTRINSICS"),
        help=(
            "Optional fitted intrinsics .npz. Defaults to D405 factory "
            "intrinsics; may also be set with HE_CAMERA_INTRINSICS."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rospy.init_node('ati_he_calibration', anonymous=True)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    win = MainWindow(args.intrinsics)
    win.show()
    sys.exit(app.exec_())
