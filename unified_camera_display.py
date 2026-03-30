"""
Unified synchronized recorder for microscope video, Intel D405 depth camera, and robot data.
All data is synchronized at 25Hz (depth camera rate).
"""
# initial version 25Mar2026 imported from Tianle Wu
# Edits: Added stereo leica flir cam_left and cam_right to QT UI
#        Dark industrial UI reskin
# TODO: Add stereo leica depth map

# Needs 3 nodelets to run: decklink (leica), d405, and flir_driver
# decklink: roslaunch gscam gscam_decklink.launch
# d405: roslaunch realsense2_camera d405_eyerobot_tianle.launch
# Flir: roslaunch spinnaker_sdk_camera_driver acquisition.launch
# Optional: ROS1 multi master for unified ros_core across SHER20, camera, SHER21 computers: roslaunch center_multimaster.launch

import os
import cv2
import time
import subprocess
from datetime import datetime
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# TODO: replace stub with real import once GUI_subscriber package is available
# from GUI_subscriber.dumb_demo.robot_subscriber_messy import TestSub
class TestSub:
    """Stub for robot logger — no-op until GUI_subscriber package is integrated."""
    def is_ready_to_log(self): return True
    def log_trigger(self, timestamp): pass
    def clear(self): pass
    def save_csv(self, path): print(f"[TestSub stub] save_csv called (no-op): {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────
DARK        = "#141414"
DARK2       = "#1a1a1a"
BORDER      = "#444"
TEXT        = "#ddd"
TEXT_DIM    = "#888"
GREEN       = "#4c9"
BLUE        = "#7af"
ACCENT      = "#2a6099"
ACCENT_H    = "#3a80bb"
ACCENT_P    = "#1a4070"
RED         = "#c44"
MONO        = "monospace"

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
        color: {TEXT_DIM};
        font-size: 11px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
    }}
    QLabel {{
        color: {TEXT_DIM};
        border: none;
    }}
    QPushButton {{
        background: {ACCENT};
        color: #fff;
        border: none;
        font-size: 13px;
        font-family: {MONO};
        font-weight: bold;
        border-radius: 4px;
        padding: 0 16px;
    }}
    QPushButton:hover   {{ background: {ACCENT_H}; }}
    QPushButton:pressed {{ background: {ACCENT_P}; }}
    QPushButton:checked {{ background: {RED}; }}
    QPushButton:checked:hover {{ background: #d55; }}
    QPushButton:disabled {{ background: #2d2d2d; color: #555; }}
"""

def _panel_label(text):
    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; font-family:{MONO}; padding:2px; border:none;")
    return lbl

def _stream_indicator(text="● no stream"):
    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setStyleSheet(f"color:#555; font-size:10px; font-family:{MONO}; padding:2px; border:none;")
    return lbl

def _set_streaming(lbl, res_str=None):
    lbl.setText("● streaming" + (f"  [{res_str}]" if res_str else ""))
    lbl.setStyleSheet(f"color:{GREEN}; font-size:10px; font-family:{MONO}; padding:2px; border:none;")

def _set_no_stream(lbl):
    lbl.setText("● no stream")
    lbl.setStyleSheet(f"color:#555; font-size:10px; font-family:{MONO}; padding:2px; border:none;")


# ─────────────────────────────────────────────────────────────────────────────
# Video writer helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class _BaseWriter:
    def write(self, frame_bgr: np.ndarray): ...
    def release(self): ...


class _OpenCVWriter(_BaseWriter):
    def __init__(self, path, w, h, fps, fourcc_str="mp4v"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self._vw = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    def write(self, frame_bgr):
        self._vw.write(frame_bgr)

    def release(self):
        self._vw.release()


class _FFmpegWriter(_BaseWriter):
    def __init__(self, path, w, h, fps, crf=18, preset="medium", pix_fmt_out="yuv420p"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", pix_fmt_out, path
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr):
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        self._proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        self._proc.stdin.close()
        self._proc.wait()


def _make_writer(backend, path, w, h, fps, fourcc="mp4v", crf=23, preset="veryfast"):
    if backend == "ffmpeg":
        return _FFmpegWriter(path, w, h, fps, crf=crf, preset=preset)
    return _OpenCVWriter(path, w, h, fps, fourcc_str=fourcc)


# ─────────────────────────────────────────────────────────────────────────────
# Depth recorder  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class DepthRecorder:
    def __init__(self, max_buffer_bytes=1_000_000_000):
        self.frames = []
        self.timestamps = []
        self.camera_info = None
        self.max_buffer_bytes = max_buffer_bytes
        self._buffer_bytes = 0
        self._base_prefix = None
        self._chunk_idx = 0
        self._save_queue = queue.Queue()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def start_session(self, base_prefix):
        self._base_prefix = base_prefix
        self.frames = []
        self.timestamps = []
        self._buffer_bytes = 0
        self._chunk_idx = 0

    def add_frame(self, depth_mm, timestamp):
        depth_001mm = np.clip(depth_mm.astype(np.uint32) * 100, 0, 65535).astype(np.uint16)
        self.frames.append(depth_001mm)
        self.timestamps.append(timestamp)
        self._buffer_bytes += depth_001mm.nbytes
        if self._base_prefix is not None and self._buffer_bytes >= self.max_buffer_bytes:
            self._flush_async()

    def set_camera_info(self, camera_info):
        if self.camera_info is None:
            self.camera_info = {
                'width': camera_info.width, 'height': camera_info.height,
                'fx': camera_info.K[0], 'fy': camera_info.K[4],
                'cx': camera_info.K[2], 'cy': camera_info.K[5],
                'distortion_model': camera_info.distortion_model, 'D': camera_info.D
            }

    def _flush_async(self):
        if not self.frames:
            return
        frames, timestamps = self.frames, self.timestamps
        self.frames, self.timestamps = [], []
        self._buffer_bytes = 0
        idx = self._chunk_idx
        self._chunk_idx += 1
        self._save_queue.put((idx, frames, timestamps))

    def _writer_loop(self):
        while True:
            item = self._save_queue.get()
            if item is None:
                self._save_queue.task_done()
                break
            idx, frames, timestamps = item
            path = f"{self._base_prefix}_chunk{idx:04d}.npz"
            np.savez_compressed(
                path, depth=np.stack(frames, axis=0),
                timestamps=np.array(timestamps), camera_info=self.camera_info,
                depth_unit='0.01mm',
                note='Depth values are in 0.01mm units. Divide by 100 to get millimeters.'
            )
            print(f"[DepthRecorder] Saved chunk {idx} to: {path}")
            self._save_queue.task_done()

    def save(self, filepath):
        if not self.frames and self._chunk_idx == 0:
            print("[DepthRecorder] No frames to save.")
            return
        if self._base_prefix is None:
            self._base_prefix = os.path.splitext(filepath)[0]
        if self._chunk_idx == 0:
            np.savez_compressed(
                filepath, depth=np.stack(self.frames, axis=0),
                timestamps=np.array(self.timestamps), camera_info=self.camera_info,
                depth_unit='0.01mm',
                note='Depth values are in 0.01mm units. Divide by 100 to get millimeters.'
            )
            print(f"[DepthRecorder] Saved {len(self.frames)} frames to: {filepath}")
        else:
            if self.frames:
                self._flush_async()
            self._save_queue.join()
            self._save_queue.put(None)
            self._save_queue.join()
            print(f"[DepthRecorder] Saved {self._chunk_idx} chunks with prefix: {self._base_prefix}_chunkXXXX")

    def clear(self):
        self.frames = []
        self.timestamps = []
        self._buffer_bytes = 0


# ─────────────────────────────────────────────────────────────────────────────
# FLIR decode helper  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _decode_flir_image(msg):
    try:
        dtype = np.uint8
        arr = (
            np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
            if msg.encoding not in ("mono8", "bayer_rggb8")
            else np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        )
        if msg.encoding == "rgb8":
            return arr
        elif msg.encoding == "bgr8":
            return arr[:, :, ::-1].copy()
        elif msg.encoding == "mono8":
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif msg.encoding == "bayer_rggb8":
            return cv2.cvtColor(arr, cv2.COLOR_BayerBG2RGB)
        else:
            return arr[:, :, :3].copy()
    except Exception as e:
        print(f"[FLIR decode] Failed: {e}")
        return None


def _to_display(arr_rgb):
    return np.transpose(np.flipud(arr_rgb), (1, 0, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Main recorder widget
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedRecorder(QtWidgets.QWidget):
    def __init__(
        self,
        microscope_topic="/decklink/camera/image_raw",
        intel_rgb_topic="/d405/color/image_raw",
        intel_depth_topic="/d405/aligned_depth_to_color/image_raw",
        intel_camera_info_topic="/d405/color/camera_info",
        flir_left_topic="/camera_array/cam_left/image_raw",
        flir_right_topic="/camera_array/cam_right/image_raw",
        backend="ffmpeg",
        out_dir="unified_recordings",
        sync_fps=25.0,
        ocv_fourcc="mp4v",
        x264_crf=23,
        x264_preset="veryfast",
        invert_depth_colormap=False
    ):
        super().__init__()

        self.bridge = CvBridge()
        self.invert_depth_colormap = invert_depth_colormap
        self.backend = backend
        self.out_dir = out_dir
        self.sync_fps = sync_fps
        self.ocv_fourcc = ocv_fourcc
        self.x264_crf = x264_crf
        self.x264_preset = x264_preset

        self.setWindowTitle("Unified Recorder")
        self.setStyleSheet(GLOBAL_STYLE)

        # latest frames
        self.latest_microscope  = None
        self.latest_intel_rgb   = None
        self.latest_intel_depth = None
        self.latest_flir_left   = None
        self.latest_flir_right  = None

        # display buffers
        self.display_microscope  = None
        self.display_intel_rgb   = None
        self.display_intel_depth = None
        self.display_flir_left   = None
        self.display_flir_right  = None

        # recording state
        self.recording          = False
        self.microscope_writer  = None
        self.intel_rgb_writer   = None
        self.flir_left_writer   = None
        self.flir_right_writer  = None
        self.depth_recorder     = DepthRecorder(max_buffer_bytes=3_000_000_000)
        self.robot_logger       = TestSub()
        self.session_dir        = None

        # fps measurement
        self.fps_measurement_window    = []
        self.measured_fps              = None
        self.fps_measurement_complete  = False

        # ROS subscribers
        rospy.Subscriber(microscope_topic,       Image,      self.microscope_callback,   queue_size=1, buff_size=2**24)
        rospy.Subscriber(intel_rgb_topic,        Image,      self.intel_rgb_callback,    queue_size=1, buff_size=2**24)
        rospy.Subscriber(intel_depth_topic,      Image,      self.intel_depth_callback,  queue_size=1, buff_size=2**24)
        rospy.Subscriber(intel_camera_info_topic,CameraInfo, self.camera_info_callback,  queue_size=1)
        rospy.Subscriber(flir_left_topic,        Image,      self.flir_left_callback,    queue_size=1, buff_size=2**24)
        rospy.Subscriber(flir_right_topic,       Image,      self.flir_right_callback,   queue_size=1, buff_size=2**24)

        print(f"[UnifiedRecorder] Subscribed to FLIR left:  {flir_left_topic}")
        print(f"[UnifiedRecorder] Subscribed to FLIR right: {flir_right_topic}")

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

        QtWidgets.QApplication.instance().aboutToQuit.connect(self.cleanup)
        print("[UnifiedRecorder] Initialized")

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        """
        Layout:
          Top row:    [FLIR Left] | [Microscope 2×] | [FLIR Right]
          Bottom row: [Intel RGB] | [Intel Depth]
          Footer:     [REC btn] [status] [fps] [out path]
        """

        def _pg_panel(title_text, stream_attr):
            """Returns (outer QFrame, pg.ImageItem, stream_label)."""
            outer = QtWidgets.QFrame()
            outer.setStyleSheet(
                f"QFrame {{ border: 1px solid {BORDER}; background: {DARK2}; border-radius: 2px; }}"
            )

            win = pg.GraphicsLayoutWidget()
            win.setBackground(DARK2)
            view = win.addViewBox()
            view.setAspectLocked(True)
            view.invertY(False)
            img_item = pg.ImageItem()
            view.addItem(img_item)

            title_lbl  = _panel_label(title_text)
            stream_lbl = _stream_indicator()
            setattr(self, stream_attr, stream_lbl)

            layout = QtWidgets.QVBoxLayout(outer)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.setSpacing(2)
            layout.addWidget(title_lbl)
            layout.addWidget(win, stretch=1)
            layout.addWidget(stream_lbl)
            return outer, img_item

        # ── top row ───────────────────────────────────────────────────────
        fl_panel,   self.img_flir_left  = _pg_panel("FLIR Cam Left",          "_stat_flir_left")
        mic_panel,  self.img_microscope = _pg_panel("Microscope (2×)",         "_stat_microscope")
        fr_panel,   self.img_flir_right = _pg_panel("FLIR Cam Right",          "_stat_flir_right")

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(4)
        top_row.addWidget(fl_panel,  stretch=1)
        top_row.addWidget(mic_panel, stretch=2)
        top_row.addWidget(fr_panel,  stretch=1)

        # ── bottom row ────────────────────────────────────────────────────
        rgb_panel,   self.img_intel_rgb   = _pg_panel("Intel D405 RGB (25 Hz)",   "_stat_intel_rgb")
        depth_panel, self.img_intel_depth = _pg_panel("Intel D405 Depth (25 Hz)", "_stat_intel_depth")

        bot_row = QtWidgets.QHBoxLayout()
        bot_row.setSpacing(4)
        bot_row.addWidget(rgb_panel)
        bot_row.addWidget(depth_panel)

        # ── footer ────────────────────────────────────────────────────────
        self._btn_rec = QtWidgets.QPushButton("⬤  START REC  [S]")
        self._btn_rec.setCheckable(True)
        self._btn_rec.setFixedHeight(44)
        self._btn_rec.clicked.connect(self.toggle_record)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setStyleSheet(f"color:{TEXT_DIM}; font-family:{MONO}; font-size:11px; border:none;")

        self._lbl_fps = QtWidgets.QLabel("fps: --")
        self._lbl_fps.setStyleSheet(f"color:{BLUE}; font-family:{MONO}; font-size:11px; border:none;")

        self._lbl_path = QtWidgets.QLabel(os.path.abspath(self.out_dir))
        self._lbl_path.setStyleSheet(f"color:{BLUE}; font-family:{MONO}; font-size:10px; border:none;")
        self._lbl_path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(10)
        footer.addWidget(self._btn_rec)
        footer.addWidget(self._lbl_status, stretch=1)
        footer.addWidget(self._lbl_fps)
        footer.addWidget(self._lbl_path)

        # ── root ──────────────────────────────────────────────────────────
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addLayout(top_row,  stretch=3)
        root.addLayout(bot_row,  stretch=2)
        root.addLayout(footer)

        QtWidgets.QShortcut(QtGui.QKeySequence("S"),      self, activated=self._btn_rec.click)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, activated=self.close)

        self.resize(1800, 1000)
        self.show()

    # ── ROS callbacks  (logic unchanged, just stream indicator updates) ────────
    def microscope_callback(self, msg):
        try:
            if msg.encoding in ("mono8", "bayer_rggb8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif msg.encoding == "mono16":
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                arr = (arr >> 8).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif msg.encoding in ("yuv422", "yuyv422", "uyvy"):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
                code = cv2.COLOR_YUV2RGB_UYVY if msg.encoding == "uyvy" else cv2.COLOR_YUV2RGB_YUYV
                arr = cv2.cvtColor(raw, code)
            elif msg.encoding == "bgr8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                arr = arr[:, :, ::-1].copy()
            elif msg.encoding == "rgba8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                arr = arr[:, :, :3]
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                arr = arr[:, :, :3]
        except Exception as e:
            print(f"[microscope_callback] Decode failed (encoding={msg.encoding}): {e}")
            return

        self.latest_microscope = arr
        self.display_microscope = _to_display(arr) if arr.ndim == 3 else np.transpose(np.flipud(arr), (1, 0))

    def intel_rgb_callback(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_intel_rgb = arr
        self.display_intel_rgb = _to_display(arr)

    def intel_depth_callback(self, msg):
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self.latest_intel_depth = depth_mm

        if not self.fps_measurement_complete:
            self.fps_measurement_window.append(time.time())
            if len(self.fps_measurement_window) >= 50:
                span = self.fps_measurement_window[-1] - self.fps_measurement_window[0]
                self.measured_fps = (len(self.fps_measurement_window) - 1) / span
                self.fps_measurement_complete = True
                print(f"[Recorder] Measured depth camera FPS: {self.measured_fps:.2f} Hz")

        depth_norm = np.clip((depth_mm.astype(float) - 70) / (500 - 70), 0, 1)
        if self.invert_depth_colormap:
            depth_norm = 1.0 - depth_norm
        depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        bar_h, bar_w = 30, depth_colormap.shape[1]
        gradient = np.tile(np.linspace(0, 255, bar_w).astype(np.uint8), (bar_h, 1))
        color_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        close_lbl = "CLOSE" if not self.invert_depth_colormap else "FAR"
        far_lbl   = "FAR"   if not self.invert_depth_colormap else "CLOSE"
        cv2.putText(color_bar, close_lbl, (10, 20),         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(color_bar, far_lbl,   (bar_w-60, 20),   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        self.display_intel_depth = _to_display(np.vstack([color_bar, depth_colormap]))

        if self.recording and self.all_writers_ready():
            if self.robot_logger.is_ready_to_log():
                ts = time.time()
                self.robot_logger.log_trigger(ts)
                if self.latest_microscope  is not None: self.write_frame(self.microscope_writer,  self.latest_microscope)
                if self.latest_intel_rgb   is not None: self.write_frame(self.intel_rgb_writer,   self.latest_intel_rgb)
                if self.latest_flir_left   is not None: self.write_frame(self.flir_left_writer,   self.latest_flir_left)
                if self.latest_flir_right  is not None: self.write_frame(self.flir_right_writer,  self.latest_flir_right)
                self.depth_recorder.add_frame(depth_mm, ts)
            else:
                print("[Debug] Robot logger not ready — frame skipped.")

    def camera_info_callback(self, msg):
        self.depth_recorder.set_camera_info(msg)

    def flir_left_callback(self, msg):
        arr = _decode_flir_image(msg)
        if arr is None:
            return
        self.latest_flir_left = arr
        self.display_flir_left = _to_display(arr)

    def flir_right_callback(self, msg):
        arr = _decode_flir_image(msg)
        if arr is None:
            return
        self.latest_flir_right = arr
        self.display_flir_right = _to_display(arr)

    # ── recording management  (unchanged logic) ────────────────────────────────
    def toggle_record(self):
        if not self.recording:
            if not self.check_frames_available():
                print("[Recorder] Waiting for all camera frames...")
                self._btn_rec.setChecked(False)
                self._lbl_status.setText("Waiting for frames...")
                self._lbl_status.setStyleSheet(f"color:{RED}; font-family:{MONO}; font-size:11px; border:none;")
                return

            if not self.fps_measurement_complete:
                print("[Recorder] Measuring camera FPS, please wait...")
                self._btn_rec.setChecked(False)
                self._lbl_status.setText(f"Measuring FPS ({len(self.fps_measurement_window)}/50)...")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.out_dir, f"session_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)

            self.create_writers()

            if not self.all_writers_ready():
                print("[Recorder] Failed to create writers.")
                self._btn_rec.setChecked(False)
                self._lbl_status.setText("Writer creation failed")
                self._lbl_status.setStyleSheet(f"color:{RED}; font-family:{MONO}; font-size:11px; border:none;")
                return

            self.robot_logger.clear()
            self.depth_recorder.start_session(os.path.join(self.session_dir, "intel_depth"))

            self.recording = True
            self._btn_rec.setText("⬛  STOP REC  [S]")
            self._lbl_status.setText(f"● RECORDING  →  {self.session_dir}")
            self._lbl_status.setStyleSheet(f"color:{RED}; font-family:{MONO}; font-size:11px; font-weight:bold; border:none;")
            print(f"[Recorder] Started recording to: {self.session_dir}")

        else:
            self.recording = False
            self._btn_rec.setText("⬤  START REC  [S]")
            self._btn_rec.setChecked(False)
            self._lbl_status.setText("Saving files...")
            self._lbl_status.setStyleSheet(f"color:{BLUE}; font-family:{MONO}; font-size:11px; border:none;")

            self.save_all_data()
            self.release_writers()

            self._lbl_status.setText(f"Saved  →  {self.session_dir}")
            self._lbl_status.setStyleSheet(f"color:{GREEN}; font-family:{MONO}; font-size:11px; border:none;")
            print(f"[Recorder] Recording complete: {self.session_dir}")

    def check_frames_available(self):
        return (
            self.latest_microscope  is not None
            and self.latest_intel_rgb   is not None
            and self.latest_intel_depth is not None
        )

    def create_writers(self):
        fps = self.measured_fps if self.measured_fps else self.sync_fps

        if self.latest_microscope is not None:
            h, w = self.latest_microscope.shape[:2]
            path = os.path.join(self.session_dir, "microscope.mp4")
            self.microscope_writer = _make_writer(self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset)
            print(f"[Recorder] Created microscope writer: {path}")

        if self.latest_intel_rgb is not None:
            h, w = self.latest_intel_rgb.shape[:2]
            path = os.path.join(self.session_dir, "intel_rgb.mp4")
            self.intel_rgb_writer = _make_writer(self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset)
            print(f"[Recorder] Created Intel RGB writer: {path}")

        if self.latest_flir_left is not None:
            h, w = self.latest_flir_left.shape[:2]
            path = os.path.join(self.session_dir, "flir_left.mp4")
            self.flir_left_writer = _make_writer(self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset)
            print(f"[Recorder] Created FLIR left writer:  {path}")
        else:
            print("[Recorder] FLIR left not available — skipping writer.")

        if self.latest_flir_right is not None:
            h, w = self.latest_flir_right.shape[:2]
            path = os.path.join(self.session_dir, "flir_right.mp4")
            self.flir_right_writer = _make_writer(self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset)
            print(f"[Recorder] Created FLIR right writer: {path}")
        else:
            print("[Recorder] FLIR right not available — skipping writer.")

    def all_writers_ready(self):
        return self.microscope_writer is not None and self.intel_rgb_writer is not None

    def write_frame(self, writer, frame):
        if writer is None:
            return
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    def save_all_data(self):
        if self.session_dir is None:
            return
        self.robot_logger.save_csv(os.path.join(self.session_dir, "robot_data.csv"))
        self.depth_recorder.save(os.path.join(self.session_dir, "intel_depth.npz"))
        print(f"[Recorder] All data saved to: {self.session_dir}")

    def release_writers(self):
        for attr in ("microscope_writer", "intel_rgb_writer", "flir_left_writer", "flir_right_writer"):
            w = getattr(self, attr, None)
            if w is not None:
                w.release()
                setattr(self, attr, None)

    # ── GUI tick ──────────────────────────────────────────────────────────────
    def tick(self):
        # update image panels + stream indicators
        panels = [
            (self.display_flir_left,   self.img_flir_left,   self._stat_flir_left,   self.latest_flir_left),
            (self.display_microscope,  self.img_microscope,  self._stat_microscope,  self.latest_microscope),
            (self.display_flir_right,  self.img_flir_right,  self._stat_flir_right,  self.latest_flir_right),
            (self.display_intel_rgb,   self.img_intel_rgb,   self._stat_intel_rgb,   self.latest_intel_rgb),
            (self.display_intel_depth, self.img_intel_depth, self._stat_intel_depth, self.latest_intel_depth),
        ]
        for disp, img_item, stat_lbl, raw in panels:
            if disp is not None:
                img_item.setImage(disp, autoLevels=False)
                res = f"{raw.shape[1]}×{raw.shape[0]}" if raw is not None and hasattr(raw, 'shape') else None
                _set_streaming(stat_lbl, res)
            else:
                _set_no_stream(stat_lbl)

        # footer status when not recording
        if not self.recording:
            if not self.fps_measurement_complete:
                n = len(self.fps_measurement_window)
                self._lbl_status.setText(f"Measuring FPS...  ({n}/50)")
                self._lbl_status.setStyleSheet(f"color:{TEXT_DIM}; font-family:{MONO}; font-size:11px; border:none;")
            else:
                fl = "✓" if self.latest_flir_left  is not None else "✗"
                fr = "✓" if self.latest_flir_right is not None else "✗"
                self._lbl_status.setText(f"Ready  |  FLIR L:{fl}  R:{fr}")
                self._lbl_status.setStyleSheet(f"color:{GREEN}; font-family:{MONO}; font-size:11px; border:none;")

        if self.measured_fps:
            self._lbl_fps.setText(f"fps: {self.measured_fps:.1f}")

    def cleanup(self):
        if self.recording:
            print("[Recorder] Cleaning up on exit...")
            self.toggle_record()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("unified_recorder", anonymous=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    recorder = UnifiedRecorder(
        microscope_topic="/decklink/camera/image_raw",
        intel_rgb_topic="/d405/color/image_raw",
        intel_depth_topic="/d405/aligned_depth_to_color/image_raw",
        intel_camera_info_topic="/d405/color/camera_info",
        flir_left_topic="/camera_array/cam_left/image_raw",
        flir_right_topic="/camera_array/cam_right/image_raw",
        backend="ffmpeg",
        out_dir=os.path.expanduser("~/Documents/unified_recordings"),
        sync_fps=25.0,
        invert_depth_colormap=True
    )

    app.exec_()