#!/usr/bin/env python3
"""
composite_board_tracker_ros.py
Live composite board tracker: Leica DeckLink + Intel D405 RGB.

Overlay modes (toggle with keys):
  D  — detailed: per-tag axes, ChArUco corners, IDs, individual z-distances
  B  — board:    unified F_board pose axes on both views, off-screen indicator
  (both can be on simultaneously)

Keys:  D=detail  B=board  S=save  Q/ESC=quit
"""

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import cv2.aruco as aruco
import threading
import time
from datetime import datetime
import os
from PyQt5 import QtWidgets, QtCore, QtGui

# ── Board config ──────────────────────────────────────────────────────────────
SQUARE_MM    = 4.0
MARKER_RATIO = 0.70
TAG_MM       = 24.0
CELL_MM      = 28.0   # TAG_MM + 2*MARGIN_MM

# ── Precomputed F_board offsets (top-left of 84x84mm board = origin) ─────────
# T_board_to_X means: "where is X's origin expressed in board frame"
# All pure translations (Z=0, same plane, same orientation convention)
def _t(tx_mm, ty_mm):
    T = np.eye(4)
    T[0, 3] = tx_mm / 1000.0
    T[1, 3] = ty_mm / 1000.0
    return T

T_BOARD_TO_CHARUCO = _t(1.5 * CELL_MM, 1.5 * CELL_MM)   # center cell center
T_BOARD_TO_AT = {
    0: _t(0.5 * CELL_MM, 0.5 * CELL_MM),   # TL corner
    1: _t(2.5 * CELL_MM, 0.5 * CELL_MM),   # TR corner
    2: _t(0.5 * CELL_MM, 2.5 * CELL_MM),   # BL corner
    3: _t(2.5 * CELL_MM, 2.5 * CELL_MM),   # BR corner
    4: _t(1.5 * CELL_MM, 0.5 * CELL_MM),   # top edge
    5: _t(0.5 * CELL_MM, 1.5 * CELL_MM),   # left edge
    6: _t(2.5 * CELL_MM, 1.5 * CELL_MM),   # right edge
    7: _t(1.5 * CELL_MM, 2.5 * CELL_MM),   # bottom edge
}
T_CHARUCO_TO_BOARD = np.linalg.inv(T_BOARD_TO_CHARUCO)
T_AT_TO_BOARD      = {k: np.linalg.inv(v) for k, v in T_BOARD_TO_AT.items()}

# ── Style ─────────────────────────────────────────────────────────────────────
DARK     = "#141414"
DARK2    = "#1a1a1a"
BORDER   = "#444"
TEXT     = "#ddd"
TEXT_DIM = "#888"
GREEN    = "#4c9"
BLUE     = "#7af"
RED      = "#c44"
AMBER    = "#fa4"
PURPLE   = "#b8f"
MONO     = "monospace"
ACCENT   = "#2a6099"
ACCENT_H = "#3a80bb"
ACCENT_P = "#1a4070"

GLOBAL_STYLE = f"""
    QWidget {{ background:{DARK}; color:{TEXT}; font-family:{MONO}; font-size:11px; }}
    QGroupBox {{ border:1px solid {BORDER}; border-radius:3px; margin-top:6px;
                 padding:6px; color:{TEXT_DIM}; font-size:11px; }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
    QLabel {{ color:{TEXT_DIM}; border:none; }}
    QPushButton {{ background:{ACCENT}; color:#fff; border:none; font-size:12px;
                   font-family:{MONO}; font-weight:bold; border-radius:4px; padding:0 12px; }}
    QPushButton:hover   {{ background:{ACCENT_H}; }}
    QPushButton:pressed {{ background:{ACCENT_P}; }}
    QPushButton:checked {{ background:#2a4a22; color:{GREEN}; border:1px solid {GREEN}; }}
"""

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_APRIL  = (0,   200, 255)
C_CHARU  = (0,   255, 120)
C_BOARD  = (180,  80, 255)   # purple — unified board frame
C_AX_X   = (0,     0, 255)
C_AX_Y   = (0,   220,   0)
C_AX_Z   = (255,  60,  60)
C_OOF    = (180,  80, 255)   # off-frame indicator color


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_K(w, h):
    f = max(w, h) * 0.85
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _rvec_tvec_from_T44(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def _T44_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.ravel()
    return T


def _project_pt(pt_3d, K, dist, rvec, tvec):
    """Project single 3D point, return (int, int) pixel or None if behind camera."""
    if tvec[2] < 0.001:
        return None
    pts, _ = cv2.projectPoints(
        np.float32([pt_3d]), rvec, tvec, K, dist
    )
    return tuple(pts[0].ravel().astype(int))


def _draw_axes_clipped(img, K, dist, rvec, tvec, length_m=0.020, color_x=C_AX_X,
                       color_y=C_AX_Y, color_z=C_AX_Z, thickness=2, label=""):
    """
    Draw XYZ axes. If origin projects off-screen, draw an edge arrow indicator
    pointing toward where the origin is, with a label.
    """
    h, w = img.shape[:2]
    margin = 18

    axes_3d = np.float32([
        [0,        0,        0       ],
        [length_m, 0,        0       ],
        [0,        length_m, 0       ],
        [0,        0,        length_m],
    ])
    try:
        pts2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
        pts2d = pts2d.reshape(4, 2)
    except Exception:
        return

    o  = pts2d[0]
    px = pts2d[1]
    py = pts2d[2]
    pz = pts2d[3]

    o_in  = (0 <= o[0] < w) and (0 <= o[1] < h)

    def _clip_to(pt):
        return (int(np.clip(pt[0], margin, w - margin)),
                int(np.clip(pt[1], margin, h - margin)))

    if o_in:
        o_i = (int(o[0]), int(o[1]))
        for pt, col in [(px, color_x), (py, color_y), (pz, color_z)]:
            pt_i = (int(np.clip(pt[0], -9999, 9999)),
                    int(np.clip(pt[1], -9999, 9999)))
            cv2.arrowedLine(img, o_i, pt_i, col, thickness, tipLength=0.2, line_type=cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (o_i[0] + 6, o_i[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_BOARD, 1, cv2.LINE_AA)
    else:
        # Origin is off-screen — draw edge indicator
        # Find intersection of line from frame center → projected origin with frame boundary
        cx, cy = w // 2, h // 2
        dx, dy = o[0] - cx, o[1] - cy
        norm   = max(abs(dx), abs(dy), 1e-6)
        dx_n, dy_n = dx / norm, dy / norm

        # Walk from center toward off-screen origin until we hit an edge
        steps = max(w, h)
        ex, ey = cx, cy
        for _ in range(steps):
            nx, ny = ex + dx_n, ey + dy_n
            if not (margin < nx < w - margin and margin < ny < h - margin):
                break
            ex, ey = nx, ny
        ex, ey = int(ex), int(ey)

        # Draw a small diamond + arrow at edge
        arrow_end = (int(ex + dx_n * 20), int(ey + dy_n * 20))
        cv2.arrowedLine(img, (ex, ey), arrow_end, C_OOF, 2, tipLength=0.4, line_type=cv2.LINE_AA)
        pts_diamond = np.int32([
            [ex,      ey - 8],
            [ex + 8,  ey    ],
            [ex,      ey + 8],
            [ex - 8,  ey    ],
        ])
        cv2.polylines(img, [pts_diamond], True, C_OOF, 1, cv2.LINE_AA)

        lbl = label if label else "F_board"
        cv2.putText(img, f"{lbl} (off-frame)", (ex + 10, ey + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_OOF, 1, cv2.LINE_AA)


def _average_board_pose_from_tags(tag_results, K, dist):
    """
    Given list of (tag_id, T_cam_to_board 4x4), return averaged T_cam_to_board.
    Uses mean of rotation vectors and translation vectors (good enough for
    small angular spread during waving; could use SE3 geodesic mean if needed).
    """
    if not tag_results:
        return None, None
    rvecs, tvecs = [], []
    for _, T_cb in tag_results:
        rv, tv = _rvec_tvec_from_T44(T_cb)
        rvecs.append(rv.ravel())
        tvecs.append(tv.ravel())
    rvec_mean = np.mean(rvecs, axis=0).reshape(3, 1)
    tvec_mean = np.mean(tvecs, axis=0).reshape(3, 1)
    return rvec_mean, tvec_mean


# ─────────────────────────────────────────────────────────────────────────────
# Core detector
# ─────────────────────────────────────────────────────────────────────────────
class CompositeDetector:
    def __init__(self):
        april_dict   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        self.charuco_board = aruco.CharucoBoard(
            (6, 6),
            SQUARE_MM / 1000.0,
            (SQUARE_MM * MARKER_RATIO) / 1000.0,
            charuco_dict
        )

        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 5
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        params.maxMarkerPerimeterRate    = 0.5
        params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX

        self.april_detector   = aruco.ArucoDetector(april_dict, params)
        self.charuco_detector = aruco.CharucoDetector(self.charuco_board)

    def run(self, frame_rgb, K, dist, show_detail, show_board):
        """
        Returns (annotated_bgr, stats_dict).
        stats_dict keys:
          n_april, april_ids, n_charu, charu_pose_ok,
          T_leica_to_board (4x4 or None),   <- from ChArUco
          T_d405_to_board  (4x4 or None),   <- from AprilTags (averaged)
          board_rvec, board_tvec             <- in cam frame, unified
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        out  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = gray.shape

        stats = {
            "n_april": 0, "april_ids": [], "n_charu": 0,
            "charu_pose_ok": False, "board_rvec": None, "board_tvec": None,
            "T_cam_to_board": None,
        }

        tag_half = (TAG_MM / 1000.0) / 2.0
        tag_obj  = np.float32([
            [-tag_half, -tag_half, 0],
            [ tag_half, -tag_half, 0],
            [ tag_half,  tag_half, 0],
            [-tag_half,  tag_half, 0],
        ])

        # ── AprilTag detection ─────────────────────────────────────────────
        corners, ids, _ = self.april_detector.detectMarkers(gray)
        tag_board_results = []   # list of (tag_id, T_cam_to_board)

        if ids is not None:
            stats["n_april"]   = len(ids)
            stats["april_ids"] = ids.flatten().tolist()

            if show_detail:
                aruco.drawDetectedMarkers(out, corners, ids, borderColor=C_APRIL)

            for i, c in enumerate(corners):
                tag_id = int(ids[i][0])
                ok, rvec, tvec = cv2.solvePnP(
                    tag_obj, c[0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not ok:
                    continue

                if show_detail:
                    _draw_axes_clipped(out, K, dist, rvec, tvec,
                                       length_m=0.010, thickness=1)
                    z_mm = tvec[2, 0] * 1000
                    cx_  = int(c[0][:, 0].mean())
                    cy_  = int(c[0][:, 1].mean())
                    cv2.putText(out, f"AT{tag_id} {z_mm:.0f}mm",
                                (cx_ - 22, cy_ - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_APRIL, 1, cv2.LINE_AA)

                if tag_id in T_AT_TO_BOARD:
                    T_cam_to_tag   = _T44_from_rvec_tvec(rvec, tvec)
                    T_cam_to_board = T_cam_to_tag @ T_AT_TO_BOARD[tag_id]
                    tag_board_results.append((tag_id, T_cam_to_board))

        # ── ChArUco detection ──────────────────────────────────────────────
        ch_corners, ch_ids, _, _ = self.charuco_detector.detectBoard(gray)
        T_cam_to_board_charuco = None

        if ch_ids is not None and len(ch_ids) >= 4:
            stats["n_charu"] = len(ch_ids)

            if show_detail:
                aruco.drawDetectedCornersCharuco(out, ch_corners, ch_ids,
                                                 cornerColor=C_CHARU)

            obj_pts = self.charuco_board.getChessboardCorners()[ch_ids.flatten()]
            ok, rvec_ch, tvec_ch = cv2.solvePnP(obj_pts, ch_corners, K, dist)

            if ok:
                stats["charu_pose_ok"] = True

                if show_detail:
                    _draw_axes_clipped(out, K, dist, rvec_ch, tvec_ch,
                                       length_m=0.015, thickness=1,
                                       label="F_charuco")
                    z_mm = tvec_ch[2, 0] * 1000
                    cx_  = int(ch_corners[:, 0, 0].mean())
                    cy_  = int(ch_corners[:, 0, 1].mean())
                    cv2.putText(out, f"ChArUco {z_mm:.0f}mm",
                                (cx_ - 35, cy_ - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CHARU, 1, cv2.LINE_AA)

                # Remap ChArUco pose → board frame
                T_cam_to_charuco   = _T44_from_rvec_tvec(rvec_ch, tvec_ch)
                T_cam_to_board_charuco = T_cam_to_charuco @ T_CHARUCO_TO_BOARD

        # ── Unified board pose ─────────────────────────────────────────────
        # Priority: ChArUco (more accurate corners) > averaged AprilTags
        T_unified = None
        source    = None

        if T_cam_to_board_charuco is not None:
            T_unified = T_cam_to_board_charuco
            source    = "ChArUco"
        elif tag_board_results:
            rvec_avg, tvec_avg = _average_board_pose_from_tags(tag_board_results, K, dist)
            if rvec_avg is not None:
                T_unified = _T44_from_rvec_tvec(rvec_avg, tvec_avg)
                source    = f"AT×{len(tag_board_results)}"

        if T_unified is not None:
            stats["T_cam_to_board"] = T_unified
            rv, tv = _rvec_tvec_from_T44(T_unified)
            stats["board_rvec"] = rv
            stats["board_tvec"] = tv

            if show_board:
                z_mm = tv[2, 0] * 1000
                _draw_axes_clipped(out, K, dist, rv, tv,
                                   length_m=0.030, thickness=3,
                                   color_x=C_AX_X, color_y=C_AX_Y, color_z=C_AX_Z,
                                   label=f"F_board [{source}]")
                # Board pose readout in corner
                euler = _rvec_to_euler_deg(rv)
                _draw_pose_box(out, tv, euler, source, z_mm)

        return out, stats


def _rvec_to_euler_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        rx = np.degrees(np.arctan2( R[2,1], R[2,2]))
        ry = np.degrees(np.arctan2(-R[2,0], sy))
        rz = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        rx = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        ry = np.degrees(np.arctan2(-R[2,0], sy))
        rz = 0.0
    return rx, ry, rz


def _draw_pose_box(img, tvec, euler, source, z_mm):
    """Small pose readout box in bottom-left corner."""
    h, w = img.shape[:2]
    x0, y0 = 8, h - 76
    lines = [
        f"F_board  [{source}]",
        f"t: {tvec[0,0]*1000:6.1f} {tvec[1,0]*1000:6.1f} {z_mm:6.1f} mm",
        f"r: {euler[0]:6.1f} {euler[1]:6.1f} {euler[2]:6.1f} deg",
    ]
    lh = 16
    bw = 280
    cv2.rectangle(img, (x0 - 4, y0 - lh), (x0 + bw, y0 + lh * len(lines) + 2),
                  (20, 20, 20), -1)
    cv2.rectangle(img, (x0 - 4, y0 - lh), (x0 + bw, y0 + lh * len(lines) + 2),
                  (100, 60, 180), 1)
    for i, ln in enumerate(lines):
        col = (200, 140, 255) if i == 0 else (200, 200, 200)
        cv2.putText(img, ln, (x0, y0 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber (unchanged from previous script)
# ─────────────────────────────────────────────────────────────────────────────
class ROSImageSub:
    def __init__(self, topic, name):
        self.name     = name
        self.frame    = None
        self._lock    = threading.Lock()
        self._t_last  = None
        self._fps_buf = []
        rospy.Subscriber(topic, Image, self._cb, queue_size=1, buff_size=2**24)
        print(f"[ROSImageSub] {name} → {topic}")

    def _cb(self, msg):
        arr = self._decode(msg)
        if arr is None:
            return
        now = time.time()
        with self._lock:
            self.frame   = arr
            self._t_last = now
        self._fps_buf.append(now)
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)

    def _decode(self, msg):
        try:
            enc = msg.encoding
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if   enc == "rgb8":        return raw.reshape(msg.height, msg.width, 3)
            elif enc == "bgr8":        return raw.reshape(msg.height, msg.width, 3)[:,:,::-1].copy()
            elif enc == "mono8":       return cv2.cvtColor(raw.reshape(msg.height, msg.width), cv2.COLOR_GRAY2RGB)
            elif enc == "yuv422":      return cv2.cvtColor(raw.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2RGB_YUYV)
            elif enc == "uyvy":        return cv2.cvtColor(raw.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2RGB_UYVY)
            elif enc == "mono16":
                r16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                return cv2.cvtColor((r16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif enc == "bayer_rggb8": return cv2.cvtColor(raw.reshape(msg.height, msg.width), cv2.COLOR_BayerBG2RGB)
            else:                      return raw.reshape(msg.height, msg.width, -1)[:,:,:3].copy()
        except Exception as e:
            print(f"[{self.name}] decode error ({msg.encoding}): {e}")
            return None

    def get(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def fps(self):
        b = self._fps_buf
        if len(b) < 2:
            return 0.0
        return (len(b) - 1) / (b[-1] - b[0])

    @property
    def streaming(self):
        return self._t_last is not None and (time.time() - self._t_last) < 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera panel widget
# ─────────────────────────────────────────────────────────────────────────────
class CameraPanel(QtWidgets.QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._title = QtWidgets.QLabel(label)
        self._title.setAlignment(QtCore.Qt.AlignCenter)
        self._title.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
        )
        self._img = QtWidgets.QLabel()
        self._img.setAlignment(QtCore.Qt.AlignCenter)
        self._img.setMinimumSize(560, 380)
        self._img.setStyleSheet(f"background:{DARK2}; border:none;")

        self._stat = QtWidgets.QLabel("● no stream")
        self._stat.setAlignment(QtCore.Qt.AlignCenter)
        self._stat.setStyleSheet(
            f"color:#555; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
        )
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        lay.addWidget(self._title)
        lay.addWidget(self._img, stretch=1)
        lay.addWidget(self._stat)
        self.setStyleSheet(
            f"QWidget {{ border:1px solid {BORDER}; background:{DARK2}; border-radius:2px; }}"
        )

    def push(self, bgr, streaming, fps, res, stats):
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            self._img.setPixmap(
                QtGui.QPixmap.fromImage(qimg).scaled(
                    self._img.size(), QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
            )
        if streaming:
            n_at  = stats.get("n_april", 0)
            ids_s = str(stats.get("april_ids", []))
            n_ch  = stats.get("n_charu", 0)
            board = "F_board OK" if stats.get("T_cam_to_board") is not None else "--"
            self._stat.setText(
                f"● {fps:.1f}Hz  {res}  |  AT:{n_at}{ids_s}  Ch:{n_ch}  {board}"
            )
            col = GREEN if stats.get("T_cam_to_board") is not None else AMBER
            self._stat.setStyleSheet(
                f"color:{col}; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
            )
        else:
            self._stat.setText("● no stream")
            self._stat.setStyleSheet(
                f"color:#555; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class TrackerWindow(QtWidgets.QWidget):
    def __init__(self, leica_sub, d405_sub, detector):
        super().__init__()
        self.leica_sub  = leica_sub
        self.d405_sub   = d405_sub
        self.detector   = detector
        self._det_mode  = True    # D key
        self._board_mode = True   # B key
        self._save_dir  = os.path.expanduser("./data/test_photos")
        self._save_n    = 0
        self._last_bgr  = {"leica": None, "d405": None}

        self.setWindowTitle("ATI — Composite Board Tracker")
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._timer.stop)

    def _build_ui(self):
        self._panel_leica = CameraPanel("Leica Proveo 8  (DeckLink)")
        self._panel_d405  = CameraPanel("Intel D405  (RGB)")

        panels = QtWidgets.QHBoxLayout()
        panels.setSpacing(6)
        panels.addWidget(self._panel_leica, stretch=1)
        panels.addWidget(self._panel_d405,  stretch=1)

        # ── Toggle buttons ────────────────────────────────────────────────
        self._btn_det = QtWidgets.QPushButton("Detail  [D]")
        self._btn_det.setCheckable(True)
        self._btn_det.setChecked(True)
        self._btn_det.setFixedHeight(36)
        self._btn_det.clicked.connect(self._toggle_det)

        self._btn_board = QtWidgets.QPushButton("F_board  [B]")
        self._btn_board.setCheckable(True)
        self._btn_board.setChecked(True)
        self._btn_board.setFixedHeight(36)
        self._btn_board.clicked.connect(self._toggle_board)

        self._btn_save = QtWidgets.QPushButton("Save  [S]")
        self._btn_save.setFixedHeight(36)
        self._btn_save.clicked.connect(self._save)

        self._lbl_status = QtWidgets.QLabel("Waiting for streams...")
        self._lbl_status.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{MONO}; font-size:11px; border:none;"
        )

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self._btn_det)
        footer.addWidget(self._btn_board)
        footer.addWidget(self._btn_save)
        footer.addWidget(self._lbl_status, stretch=1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addLayout(panels, stretch=1)
        root.addLayout(footer)

        QtWidgets.QShortcut(QtGui.QKeySequence("D"),      self, activated=self._btn_det.click)
        QtWidgets.QShortcut(QtGui.QKeySequence("B"),      self, activated=self._btn_board.click)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"),      self, activated=self._save)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"),      self, activated=self.close)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.close)

        self.resize(1700, 800)
        self.show()

    def _toggle_det(self):
        self._det_mode = self._btn_det.isChecked()

    def _toggle_board(self):
        self._board_mode = self._btn_board.isChecked()

    def _save(self):
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")[:10]
        for name, bgr in self._last_bgr.items():
            if bgr is not None:
                path = os.path.join(self._save_dir, f"{name}_{ts}_{self._save_n:04d}.png")
                cv2.imwrite(path, bgr)
        self._save_n += 1
        self._lbl_status.setText(f"Saved → {self._save_dir}")
        self._lbl_status.setStyleSheet(
            f"color:{GREEN}; font-family:{MONO}; font-size:11px; border:none;"
        )

    def _tick(self):
        for sub, panel, name in [
            (self.leica_sub, self._panel_leica, "leica"),
            (self.d405_sub,  self._panel_d405,  "d405"),
        ]:
            frame = sub.get()
            bgr   = None
            stats = {}

            if frame is not None:
                h, w = frame.shape[:2]
                K    = _fallback_K(w, h)   # swap real intrinsics in here
                dist = np.zeros(5)
                bgr, stats = self.detector.run(
                    frame, K, dist,
                    show_detail=self._det_mode,
                    show_board=self._board_mode,
                )
                self._last_bgr[name] = bgr

            res = f"{frame.shape[1]}×{frame.shape[0]}" if frame is not None else "--"
            panel.push(bgr, sub.streaming, sub.fps, res, stats)

        # Global status
        both = self.leica_sub.streaming and self.d405_sub.streaming
        if both:
            self._lbl_status.setText(
                "Both streams live  |  [D] detail  [B] F_board  [S] save  [Q] quit"
            )
            self._lbl_status.setStyleSheet(
                f"color:{GREEN}; font-family:{MONO}; font-size:11px; border:none;"
            )
        else:
            missing = [n for n, s in [("Leica", self.leica_sub), ("D405", self.d405_sub)]
                       if not s.streaming]
            self._lbl_status.setText(f"Waiting: {', '.join(missing)}")
            self._lbl_status.setStyleSheet(
                f"color:{AMBER}; font-family:{MONO}; font-size:11px; border:none;"
            )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("composite_board_tracker", anonymous=True)

    leica_sub = ROSImageSub("/decklink/camera/image_raw", "Leica")
    d405_sub  = ROSImageSub("/d405/color/image_raw",      "D405")
    detector  = CompositeDetector()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = TrackerWindow(leica_sub, d405_sub, detector)
    app.exec_()