#!/usr/bin/env python3
"""
composite_board_tracker_ros.py
Leica DeckLink + Intel D405 composite board tracker.

Modes (toggleable, both can be on simultaneously):
  D  — detail:  per-tag axes, ChArUco corners, IDs, z-distances
  B  — board:   unified F_board pose axes + off-screen indicator

Pose readouts docked below each image panel, updated at display rate.
SE(3) exponential moving average filter for jitter suppression.

Keys:  D=detail  B=board  S=save  Q/ESC=quit
Alpha slider in footer controls filter aggressiveness.
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
CELL_MM      = 28.0

def _t(tx_mm, ty_mm):
    T = np.eye(4)
    T[0, 3] = tx_mm / 1000.0
    T[1, 3] = ty_mm / 1000.0
    return T

T_BOARD_TO_CHARUCO = _t(1.5 * CELL_MM, 1.5 * CELL_MM)
T_BOARD_TO_AT = {
    0: _t(0.5 * CELL_MM, 0.5 * CELL_MM),
    1: _t(2.5 * CELL_MM, 0.5 * CELL_MM),
    2: _t(0.5 * CELL_MM, 2.5 * CELL_MM),
    3: _t(2.5 * CELL_MM, 2.5 * CELL_MM),
    4: _t(1.5 * CELL_MM, 0.5 * CELL_MM),
    5: _t(0.5 * CELL_MM, 1.5 * CELL_MM),
    6: _t(2.5 * CELL_MM, 1.5 * CELL_MM),
    7: _t(1.5 * CELL_MM, 2.5 * CELL_MM),
}
T_CHARUCO_TO_BOARD = np.linalg.inv(T_BOARD_TO_CHARUCO)
T_AT_TO_BOARD      = {k: np.linalg.inv(v) for k, v in T_BOARD_TO_AT.items()}

# ── Style ─────────────────────────────────────────────────────────────────────
DARK     = "#141414"
DARK2    = "#1a1a1a"
DARK3    = "#111111"
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
    QGroupBox {{
        border:1px solid {BORDER}; border-radius:3px; margin-top:6px;
        padding:6px; color:{TEXT_DIM}; font-size:11px;
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
    QLabel {{ color:{TEXT_DIM}; border:none; }}
    QPushButton {{
        background:{ACCENT}; color:#fff; border:none; font-size:12px;
        font-family:{MONO}; font-weight:bold; border-radius:4px; padding:0 12px;
    }}
    QPushButton:hover   {{ background:{ACCENT_H}; }}
    QPushButton:pressed {{ background:{ACCENT_P}; }}
    QPushButton:checked {{ background:#1e3a1e; color:{GREEN}; border:1px solid {GREEN}; }}
    QSlider::groove:horizontal {{
        height:4px; background:{BORDER}; border-radius:2px;
    }}
    QSlider::handle:horizontal {{
        background:{BLUE}; width:12px; height:12px; margin:-4px 0;
        border-radius:6px;
    }}
    QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
"""

C_APRIL = (0,   200, 255)
C_CHARU = (0,   255, 120)
C_BOARD = (180,  80, 255)
C_AX_X  = (0,     0, 255)
C_AX_Y  = (0,   220,   0)
C_AX_Z  = (255,  60,  60)


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) exponential moving average filter
# ─────────────────────────────────────────────────────────────────────────────
class SE3Filter:
    """
    Causal EMA on SE(3):
      t_filtered  = alpha * t_raw  + (1-alpha) * t_prev
      R_filtered  = R_prev * Exp(alpha * Log(R_prev^T * R_raw))
    alpha=1.0 → no filtering (raw), alpha→0 → very smooth/lagged.
    """
    def __init__(self, alpha=0.3):
        self.alpha  = alpha
        self._t     = None   # (3,) translation
        self._R     = None   # (3,3) rotation

    def reset(self):
        self._t = None
        self._R = None

    def update(self, rvec, tvec):
        """
        rvec: (3,1) or (3,), tvec: (3,1) or (3,).
        Returns filtered (rvec_f, tvec_f) same shapes.
        """
        R_raw, _ = cv2.Rodrigues(rvec.ravel())
        t_raw    = tvec.ravel().copy()

        if self._t is None:
            self._R = R_raw.copy()
            self._t = t_raw.copy()
        else:
            # Translation EMA
            self._t = self.alpha * t_raw + (1.0 - self.alpha) * self._t

            # Rotation EMA via geodesic step on SO(3)
            # dR = R_prev^T @ R_raw  (relative rotation)
            dR     = self._R.T @ R_raw
            drvec, _ = cv2.Rodrigues(dR)
            # Scale the relative rotation vector by alpha (geodesic interpolation)
            drvec_scaled = self.alpha * drvec
            dR_scaled, _ = cv2.Rodrigues(drvec_scaled)
            self._R = self._R @ dR_scaled

        rvec_f, _ = cv2.Rodrigues(self._R)
        return rvec_f.reshape(rvec.shape), self._t.reshape(tvec.shape)

    @property
    def has_data(self):
        return self._t is not None


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_K(w, h):
    f = max(w, h) * 0.85
    return np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)


def _T44_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.ravel()
    return T


def _rvec_tvec_from_T44(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec, T[:3, 3:4]


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


def _draw_axes_clipped(img, K, dist, rvec, tvec, length_m=0.020,
                       color_x=C_AX_X, color_y=C_AX_Y, color_z=C_AX_Z,
                       thickness=2, label=""):
    h, w = img.shape[:2]
    margin = 18
    axes_3d = np.float32([
        [0, 0, 0], [length_m, 0, 0], [0, length_m, 0], [0, 0, length_m]
    ])
    try:
        pts2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
        pts2d = pts2d.reshape(4, 2)
    except Exception:
        return

    o  = pts2d[0]
    o_in = (0 <= o[0] < w) and (0 <= o[1] < h)

    if o_in:
        o_i = (int(o[0]), int(o[1]))
        for pt, col in [(pts2d[1], color_x), (pts2d[2], color_y), (pts2d[3], color_z)]:
            pt_i = (int(np.clip(pt[0], -9999, 9999)), int(np.clip(pt[1], -9999, 9999)))
            cv2.arrowedLine(img, o_i, pt_i, col, thickness, tipLength=0.2, line_type=cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (o_i[0] + 6, o_i[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_BOARD, 1, cv2.LINE_AA)
    else:
        # Off-screen: edge diamond indicator
        cx, cy = w // 2, h // 2
        dx, dy = o[0] - cx, o[1] - cy
        norm = max(abs(dx), abs(dy), 1e-6)
        dx_n, dy_n = dx / norm, dy / norm
        ex, ey = float(cx), float(cy)
        for _ in range(max(w, h)):
            nx, ny = ex + dx_n, ey + dy_n
            if not (margin < nx < w - margin and margin < ny < h - margin):
                break
            ex, ey = nx, ny
        ex, ey = int(ex), int(ey)
        tip = (int(ex + dx_n * 20), int(ey + dy_n * 20))
        cv2.arrowedLine(img, (ex, ey), tip, C_BOARD, 2, tipLength=0.4, line_type=cv2.LINE_AA)
        diamond = np.int32([[ex, ey-8], [ex+8, ey], [ex, ey+8], [ex-8, ey]])
        cv2.polylines(img, [diamond], True, C_BOARD, 1, cv2.LINE_AA)
        lbl = (label if label else "F_board") + " (off-frame)"
        cv2.putText(img, lbl, (ex + 10, ey + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BOARD, 1, cv2.LINE_AA)


def _average_board_pose(tag_results):
    if not tag_results:
        return None, None
    rvecs = [_rvec_tvec_from_T44(T)[0].ravel() for _, T in tag_results]
    tvecs = [_rvec_tvec_from_T44(T)[1].ravel() for _, T in tag_results]
    return np.mean(rvecs, axis=0).reshape(3, 1), np.mean(tvecs, axis=0).reshape(3, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────
class CompositeDetector:
    def __init__(self):
        april_dict   = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        charuco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.charuco_board = aruco.CharucoBoard(
            (6, 6), SQUARE_MM / 1000.0,
            (SQUARE_MM * MARKER_RATIO) / 1000.0, charuco_dict
        )
        params = aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin  = 5
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 4
        params.minMarkerPerimeterRate    = 0.02
        params.maxMarkerPerimeterRate    = 0.5
        params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX
        self.april_det   = aruco.ArucoDetector(april_dict, params)
        self.charuco_det = aruco.CharucoDetector(self.charuco_board)

    def run(self, frame_rgb, K, dist, show_detail, show_board):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        out  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        stats = {
            "n_april": 0, "april_ids": [], "n_charu": 0,
            "charu_pose_ok": False, "board_rvec": None, "board_tvec": None,
            "T_cam_to_board": None, "source": "--",
        }

        tag_half = (TAG_MM / 1000.0) / 2.0
        tag_obj  = np.float32([
            [-tag_half, -tag_half, 0], [ tag_half, -tag_half, 0],
            [ tag_half,  tag_half, 0], [-tag_half,  tag_half, 0],
        ])

        # ── AprilTags ──────────────────────────────────────────────────────
        corners, ids, _ = self.april_det.detectMarkers(gray)
        tag_board_results = []

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
                    z_mm = float(tvec[2]) * 1000
                    cx_  = int(c[0][:, 0].mean())
                    cy_  = int(c[0][:, 1].mean())
                    cv2.putText(out, f"AT{tag_id} {z_mm:.0f}mm",
                                (cx_ - 22, cy_ - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, C_APRIL, 1, cv2.LINE_AA)
                if tag_id in T_AT_TO_BOARD:
                    T_cam_to_board = _T44_from_rvec_tvec(rvec, tvec) @ T_AT_TO_BOARD[tag_id]
                    tag_board_results.append((tag_id, T_cam_to_board))

        # ── ChArUco ────────────────────────────────────────────────────────
        ch_corners, ch_ids, _, _ = self.charuco_det.detectBoard(gray)
        T_cam_to_board_ch = None

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
                                       length_m=0.012, thickness=1, label="F_ch")
                    z_mm = float(tvec_ch[2]) * 1000
                    cx_  = int(ch_corners[:, 0, 0].mean())
                    cy_  = int(ch_corners[:, 0, 1].mean())
                    cv2.putText(out, f"Ch {z_mm:.0f}mm", (cx_ - 20, cy_ - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CHARU, 1, cv2.LINE_AA)
                T_cam_to_board_ch = _T44_from_rvec_tvec(rvec_ch, tvec_ch) @ T_CHARUCO_TO_BOARD

        # ── Unified board pose ─────────────────────────────────────────────
        T_unified = None
        source    = "--"
        if T_cam_to_board_ch is not None:
            T_unified = T_cam_to_board_ch
            source    = f"Ch+{len(ch_ids)}c"
        elif tag_board_results:
            rv, tv = _average_board_pose(tag_board_results)
            if rv is not None:
                T_unified = _T44_from_rvec_tvec(rv, tv)
                source    = f"AT×{len(tag_board_results)}"

        if T_unified is not None:
            rv, tv = _rvec_tvec_from_T44(T_unified)
            stats.update({
                "T_cam_to_board": T_unified,
                "board_rvec": rv,
                "board_tvec": tv,
                "source": source,
            })
            if show_board:
                _draw_axes_clipped(out, K, dist, rv, tv,
                                   length_m=0.030, thickness=3,
                                   label=f"F_board [{source}]")

        return out, stats


# ─────────────────────────────────────────────────────────────────────────────
# Pose readout widget  (docked below each camera panel)
# ─────────────────────────────────────────────────────────────────────────────
class PoseReadout(QtWidgets.QFrame):
    """
    Fixed-height panel showing filtered t (xyz mm) and R (euler deg) for F_board.
    Updates from push(rvec, tvec, source) called each tick.
    """
    _N_HIST = 80   # sparkline history length

    def __init__(self, cam_name, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90)
        self.setStyleSheet(
            f"QFrame {{ background:{DARK3}; border-top:1px solid {BORDER}; border-radius:0px; }}"
        )

        self._cam_name = cam_name

        # History for mini sparklines (tx, ty, tz)
        self._t_hist = [[], [], []]

        # Labels
        self._lbl_src = QtWidgets.QLabel("● --")
        self._lbl_src.setStyleSheet(
            f"color:#555; font-family:{MONO}; font-size:10px; border:none; padding:0;"
        )

        self._lbl_t = QtWidgets.QLabel("t:  --")
        self._lbl_t.setStyleSheet(
            f"color:{BLUE}; font-family:{MONO}; font-size:11px; border:none; padding:0;"
        )

        self._lbl_r = QtWidgets.QLabel("R:  --")
        self._lbl_r.setStyleSheet(
            f"color:{PURPLE}; font-family:{MONO}; font-size:11px; border:none; padding:0;"
        )

        self._lbl_filt = QtWidgets.QLabel("α=0.30")
        self._lbl_filt.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{MONO}; font-size:10px; border:none; padding:0;"
        )

        # Sparkline canvas (drawn via paintEvent override on a QLabel subclass)
        self._spark = SparklineWidget(self._N_HIST)
        self._spark.setFixedSize(120, 54)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(2)
        left_col.setContentsMargins(8, 6, 0, 6)
        left_col.addWidget(self._lbl_src)
        left_col.addWidget(self._lbl_t)
        left_col.addWidget(self._lbl_r)
        left_col.addWidget(self._lbl_filt)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 8, 0)
        root.setSpacing(6)
        root.addLayout(left_col, stretch=1)
        root.addWidget(self._spark)

    def push(self, rvec, tvec, source, alpha):
        """Called each display tick with filtered pose. Pass None,None if no pose."""
        self._lbl_filt.setText(f"α={alpha:.2f}")

        if rvec is None or tvec is None:
            self._lbl_src.setText("● no pose")
            self._lbl_src.setStyleSheet(
                f"color:#555; font-family:{MONO}; font-size:10px; border:none; padding:0;"
            )
            self._lbl_t.setText("t:  --   --   --   mm")
            self._lbl_r.setText("R:  --   --   --   deg")
            for h in self._t_hist:
                h.append(float("nan"))
                if len(h) > self._N_HIST:
                    h.pop(0)
            self._spark.update_data(self._t_hist)
            return

        tv = tvec.ravel() * 1000.0   # → mm
        eu = _rvec_to_euler_deg(rvec)

        self._lbl_src.setText(f"● F_board  [{source}]")
        self._lbl_src.setStyleSheet(
            f"color:{GREEN}; font-family:{MONO}; font-size:10px; border:none; padding:0;"
        )
        self._lbl_t.setText(
            f"t:  x{tv[0]:+7.1f}  y{tv[1]:+7.1f}  z{tv[2]:+7.1f}  mm"
        )
        self._lbl_r.setText(
            f"R:  rx{eu[0]:+6.1f}  ry{eu[1]:+6.1f}  rz{eu[2]:+6.1f}  deg"
        )

        for i, v in enumerate(tv[:3]):
            self._t_hist[i].append(float(v))
            if len(self._t_hist[i]) > self._N_HIST:
                self._t_hist[i].pop(0)
        self._spark.update_data(self._t_hist)


class SparklineWidget(QtWidgets.QWidget):
    """Tiny 3-channel sparkline for tx, ty, tz."""
    _COLORS = [
        QtGui.QColor(80, 80, 255),    # x — blue
        QtGui.QColor(40, 210, 40),    # y — green
        QtGui.QColor(255, 70, 70),    # z — red
    ]

    def __init__(self, n, parent=None):
        super().__init__(parent)
        self._n    = n
        self._data = [[], [], []]

    def update_data(self, data):
        self._data = [list(d) for d in data]
        self.update()

    def paintEvent(self, _event):
        p   = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(17, 17, 17))
        w, h = self.width(), self.height()
        pad  = 4

        all_vals = [v for ch in self._data for v in ch if not (v != v)]  # filter nan
        if len(all_vals) < 2:
            p.end()
            return

        mn, mx = min(all_vals), max(all_vals)
        rng = max(mx - mn, 1.0)

        for ci, channel in enumerate(self._data):
            if len(channel) < 2:
                continue
            pen = QtGui.QPen(self._COLORS[ci], 1, QtCore.Qt.SolidLine)
            p.setPen(pen)
            pts = []
            for i, v in enumerate(channel):
                if v != v:   # nan
                    continue
                x = pad + (i / max(self._n - 1, 1)) * (w - 2 * pad)
                y = h - pad - ((v - mn) / rng) * (h - 2 * pad)
                pts.append(QtCore.QPointF(x, y))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i+1])

        # Axis labels x/y/z
        p.setFont(QtGui.QFont(MONO, 7))
        for ci, lbl in enumerate(["x", "y", "z"]):
            p.setPen(self._COLORS[ci])
            p.drawText(w - 14, 10 + ci * 14, lbl)

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Camera panel  (image + docked pose readout)
# ─────────────────────────────────────────────────────────────────────────────
class CameraPanel(QtWidgets.QWidget):
    def __init__(self, label, cam_name, parent=None):
        super().__init__(parent)

        self._title = QtWidgets.QLabel(label)
        self._title.setAlignment(QtCore.Qt.AlignCenter)
        self._title.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
        )

        self._img = QtWidgets.QLabel()
        self._img.setAlignment(QtCore.Qt.AlignCenter)
        self._img.setMinimumSize(560, 340)
        self._img.setStyleSheet(f"background:{DARK2}; border:none;")

        self._stream_lbl = QtWidgets.QLabel("● no stream")
        self._stream_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._stream_lbl.setStyleSheet(
            f"color:#555; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
        )

        self.pose_readout = PoseReadout(cam_name)   # public — window updates directly

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 0)
        lay.setSpacing(2)
        lay.addWidget(self._title)
        lay.addWidget(self._img, stretch=1)
        lay.addWidget(self._stream_lbl)
        lay.addWidget(self.pose_readout)

        self.setStyleSheet(
            f"QWidget {{ border:1px solid {BORDER}; background:{DARK2}; border-radius:2px; }}"
        )

    def push_image(self, bgr, streaming, fps, res, stats):
        if bgr is not None:
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
            n_ch  = stats.get("n_charu",  0)
            ok    = stats.get("T_cam_to_board") is not None
            col   = GREEN if ok else TEXT_DIM
            self._stream_lbl.setText(
                f"● {fps:.1f}Hz  {res}  |  AT:{n_at}{ids_s}  Ch:{n_ch}"
                f"  {'F_board OK' if ok else 'no board'}"
            )
            self._stream_lbl.setStyleSheet(
                f"color:{col}; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
            )
        else:
            self._stream_lbl.setText("● no stream")
            self._stream_lbl.setStyleSheet(
                f"color:#555; font-size:10px; font-family:{MONO}; padding:2px; border:none;"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber
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
        if len(self._fps_buf) > 60:
            self._fps_buf.pop(0)

    def _decode(self, msg):
        try:
            enc = msg.encoding
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if   enc == "rgb8":         return raw.reshape(msg.height, msg.width, 3)
            elif enc == "bgr8":         return raw.reshape(msg.height, msg.width, 3)[:,:,::-1].copy()
            elif enc == "mono8":        return cv2.cvtColor(raw.reshape(msg.height, msg.width), cv2.COLOR_GRAY2RGB)
            elif enc == "yuv422":       return cv2.cvtColor(raw.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2RGB_YUYV)
            elif enc == "uyvy":         return cv2.cvtColor(raw.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2RGB_UYVY)
            elif enc == "mono16":
                r16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                return cv2.cvtColor((r16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            elif enc == "bayer_rggb8":  return cv2.cvtColor(raw.reshape(msg.height, msg.width), cv2.COLOR_BayerBG2RGB)
            else:                       return raw.reshape(msg.height, msg.width, -1)[:,:,:3].copy()
        except Exception as e:
            print(f"[{self.name}] decode ({msg.encoding}): {e}")
            return None

    def get(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def fps(self):
        b = self._fps_buf
        return 0.0 if len(b) < 2 else (len(b) - 1) / (b[-1] - b[0])

    @property
    def streaming(self):
        return self._t_last is not None and (time.time() - self._t_last) < 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class TrackerWindow(QtWidgets.QWidget):
    def __init__(self, leica_sub, d405_sub, detector):
        super().__init__()
        self.leica_sub  = leica_sub
        self.d405_sub   = d405_sub
        self.detector   = detector
        self._det_mode  = True
        self._board_mode = True
        self._alpha     = 0.30
        self._save_dir  = os.path.expanduser("~/ATI_tracker_frames")
        self._save_n    = 0
        self._last_bgr  = {"leica": None, "d405": None}

        # One SE3Filter per camera — independent histories
        self._filt = {
            "leica": SE3Filter(alpha=self._alpha),
            "d405":  SE3Filter(alpha=self._alpha),
        }

        self.setWindowTitle("ATI — Composite Board Tracker")
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)   # 30 Hz display
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._timer.stop)

    def _build_ui(self):
        self._panel_leica = CameraPanel("Leica Proveo 8  (DeckLink)", "leica")
        self._panel_d405  = CameraPanel("Intel D405  (RGB)",           "d405")

        panels = QtWidgets.QHBoxLayout()
        panels.setSpacing(6)
        panels.addWidget(self._panel_leica, stretch=1)
        panels.addWidget(self._panel_d405,  stretch=1)

        # ── footer controls ───────────────────────────────────────────────
        self._btn_det = QtWidgets.QPushButton("Detail  [D]")
        self._btn_det.setCheckable(True)
        self._btn_det.setChecked(True)
        self._btn_det.setFixedHeight(34)
        self._btn_det.clicked.connect(lambda: setattr(self, "_det_mode",   self._btn_det.isChecked()))

        self._btn_board = QtWidgets.QPushButton("F_board  [B]")
        self._btn_board.setCheckable(True)
        self._btn_board.setChecked(True)
        self._btn_board.setFixedHeight(34)
        self._btn_board.clicked.connect(lambda: setattr(self, "_board_mode", self._btn_board.isChecked()))

        self._btn_save = QtWidgets.QPushButton("Save  [S]")
        self._btn_save.setFixedHeight(34)
        self._btn_save.clicked.connect(self._save)

        # Alpha slider
        alpha_lbl = QtWidgets.QLabel("Filter α:")
        alpha_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-family:{MONO}; font-size:11px; border:none;")
        self._alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._alpha_slider.setRange(1, 100)
        self._alpha_slider.setValue(int(self._alpha * 100))
        self._alpha_slider.setFixedWidth(140)
        self._alpha_slider.setToolTip("Filter alpha: 1=raw/jittery, 0=smooth/lagged")
        self._alpha_val_lbl = QtWidgets.QLabel(f"{self._alpha:.2f}")
        self._alpha_val_lbl.setStyleSheet(f"color:{BLUE}; font-family:{MONO}; font-size:11px; border:none; min-width:32px;")
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)

        self._lbl_status = QtWidgets.QLabel("Waiting for streams...")
        self._lbl_status.setStyleSheet(
            f"color:{TEXT_DIM}; font-family:{MONO}; font-size:11px; border:none;"
        )

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self._btn_det)
        footer.addWidget(self._btn_board)
        footer.addWidget(self._btn_save)
        footer.addSpacing(12)
        footer.addWidget(alpha_lbl)
        footer.addWidget(self._alpha_slider)
        footer.addWidget(self._alpha_val_lbl)
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

        self.resize(1700, 860)
        self.show()

    def _on_alpha_changed(self, val):
        self._alpha = val / 100.0
        self._alpha_val_lbl.setText(f"{self._alpha:.2f}")
        for f in self._filt.values():
            f.alpha = self._alpha

    def _save(self):
        os.makedirs(self._save_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")[:10]
        for name, bgr in self._last_bgr.items():
            if bgr is not None:
                cv2.imwrite(
                    os.path.join(self._save_dir, f"{name}_{ts}_{self._save_n:04d}.png"),
                    bgr
                )
        self._save_n += 1
        self._lbl_status.setText(f"Saved → {self._save_dir}")
        self._lbl_status.setStyleSheet(
            f"color:{GREEN}; font-family:{MONO}; font-size:11px; border:none;"
        )

    def _tick(self):
        configs = [
            (self.leica_sub, self._panel_leica, "leica"),
            (self.d405_sub,  self._panel_d405,  "d405"),
        ]

        for sub, panel, name in configs:
            frame = sub.get()
            bgr   = None
            stats = {}

            if frame is not None:
                h, w  = frame.shape[:2]
                K     = _fallback_K(w, h)   # ← swap in real intrinsics per camera
                dist  = np.zeros(5)

                bgr, stats = self.detector.run(
                    frame, K, dist,
                    show_detail=self._det_mode,
                    show_board=self._board_mode,
                )
                self._last_bgr[name] = bgr

                # Apply SE3 filter to board pose
                rv_raw = stats.get("board_rvec")
                tv_raw = stats.get("board_tvec")
                filt   = self._filt[name]

                if rv_raw is not None and tv_raw is not None:
                    rv_f, tv_f = filt.update(rv_raw, tv_raw)
                else:
                    rv_f, tv_f = None, None
                    if not filt.has_data:
                        pass  # never had a pose yet — that's fine
                    # don't reset — let it hold last filtered value
                    # uncomment below to reset on loss:
                    # filt.reset()

                panel.pose_readout.push(rv_f, tv_f, stats.get("source", "--"), self._alpha)

            res = f"{frame.shape[1]}×{frame.shape[0]}" if frame is not None else "--"
            panel.push_image(bgr, sub.streaming, sub.fps, res, stats)

        # Status bar
        both = self.leica_sub.streaming and self.d405_sub.streaming
        if both:
            self._lbl_status.setText(
                "[D] detail  [B] F_board  [S] save  [Q] quit  —  α=filter"
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