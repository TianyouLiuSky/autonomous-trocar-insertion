"""
Unified synchronized recorder for microscope video, Intel D405 depth camera, and robot data.
All data is synchronized at 25Hz (depth camera rate).
"""
# initial version 25Mar2026 imported from Tianle Wu
# Edits: Added stereo leica flir cam_left and cam_right to QT UI
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


# --- Video Writer Helper Classes ---
class _BaseWriter:
    def write(self, frame_bgr: np.ndarray): ...
    def release(self): ...


class _OpenCVWriter(_BaseWriter):
    def __init__(self, path: str, w: int, h: int, fps: float, fourcc_str: str = "mp4v"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self._vw = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    def write(self, frame_bgr: np.ndarray):
        self._vw.write(frame_bgr)

    def release(self):
        self._vw.release()


class _FFmpegWriter(_BaseWriter):
    def __init__(self, path: str, w: int, h: int, fps: float, crf: int = 18, preset: str = "medium", pix_fmt_out="yuv420p"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", pix_fmt_out, path
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray):
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        self._proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        self._proc.stdin.close()
        self._proc.wait()


def _make_writer(backend: str, path: str, w: int, h: int, fps: float,
                 fourcc: str = "mp4v", crf: int = 23, preset: str = "veryfast") -> _BaseWriter:
    if backend == "ffmpeg":
        return _FFmpegWriter(path, w, h, fps, crf=crf, preset=preset)
    return _OpenCVWriter(path, w, h, fps, fourcc_str=fourcc)


# --- Depth Storage Helper ---
class DepthRecorder:
    """Records depth frames with camera info metadata."""

    def __init__(self, max_buffer_bytes: int = 1_000_000_000):
        self.frames = []
        self.timestamps = []
        self.camera_info = None

        self.max_buffer_bytes = max_buffer_bytes
        self._buffer_bytes = 0
        self._base_prefix = None
        self._chunk_idx = 0

        self._save_queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )
        self._writer_thread.start()

    def start_session(self, base_prefix: str):
        self._base_prefix = base_prefix
        self.frames = []
        self.timestamps = []
        self._buffer_bytes = 0
        self._chunk_idx = 0

    def add_frame(self, depth_mm: np.ndarray, timestamp: float):
        depth_001mm = np.clip(
            depth_mm.astype(np.uint32) * 100, 0, 65535
        ).astype(np.uint16)

        self.frames.append(depth_001mm)
        self.timestamps.append(timestamp)
        self._buffer_bytes += depth_001mm.nbytes

        if (
            self._base_prefix is not None
            and self._buffer_bytes >= self.max_buffer_bytes
        ):
            self._flush_async()

    def set_camera_info(self, camera_info: CameraInfo):
        if self.camera_info is None:
            self.camera_info = {
                'width': camera_info.width,
                'height': camera_info.height,
                'fx': camera_info.K[0],
                'fy': camera_info.K[4],
                'cx': camera_info.K[2],
                'cy': camera_info.K[5],
                'distortion_model': camera_info.distortion_model,
                'D': camera_info.D
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

            depth_array = np.stack(frames, axis=0)
            np.savez_compressed(
                path,
                depth=depth_array,
                timestamps=np.array(timestamps),
                camera_info=self.camera_info,
                depth_unit='0.01mm',
                note='Depth values are in 0.01mm units. Divide by 100 to get millimeters.'
            )
            print(f"[DepthRecorder] Saved chunk {idx} to: {path}")
            self._save_queue.task_done()

    def save(self, filepath: str):
        if not self.frames and self._chunk_idx == 0:
            print("[DepthRecorder] No frames to save.")
            return

        if self._base_prefix is None:
            self._base_prefix = os.path.splitext(filepath)[0]

        if self._chunk_idx == 0:
            depth_array = np.stack(self.frames, axis=0)
            np.savez_compressed(
                filepath,
                depth=depth_array,
                timestamps=np.array(self.timestamps),
                camera_info=self.camera_info,
                depth_unit='0.01mm',
                note='Depth values are in 0.01mm units. Divide by 100 to get millimeters.'
            )
            print(f"[DepthRecorder] Saved {len(self.frames)} frames to: {filepath}")
            print(f"[DepthRecorder] File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
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


# --- FLIR Camera Subscriber ---
def _decode_flir_image(msg: Image) -> np.ndarray:
    """
    Manual decode for FLIR sensor_msgs/Image — avoids cv_bridge on ROS1 Melodic.
    Returns an RGB uint8 array, or None on failure.
    FLIR BFS cameras typically publish mono8 or rgb8; handles both.
    """
    try:
        dtype = np.uint8
        arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1) \
            if msg.encoding not in ("mono8", "bayer_rggb8") \
            else np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)

        if msg.encoding == "rgb8":
            return arr  # already RGB
        elif msg.encoding == "bgr8":
            return arr[:, :, ::-1].copy()  # BGR -> RGB
        elif msg.encoding == "mono8":
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif msg.encoding == "bayer_rggb8":
            return cv2.cvtColor(arr, cv2.COLOR_BayerBG2RGB)
        else:
            # Fallback: assume first 3 channels are usable as RGB
            return arr[:, :, :3].copy()
    except Exception as e:
        print(f"[FLIR decode] Failed: {e}")
        return None


def _to_display(arr_rgb: np.ndarray) -> np.ndarray:
    """Transpose and flip for PyQtGraph display (shared helper)."""
    return np.transpose(np.flipud(arr_rgb), (1, 0, 2))


# --- Main Unified Recorder ---
class UnifiedRecorder(QtWidgets.QWidget):
    def __init__(
        self,
        microscope_topic: str = "/decklink/camera/image_raw",
        intel_rgb_topic: str = "/d405/color/image_raw",
        intel_depth_topic: str = "/d405/aligned_depth_to_color/image_raw",
        intel_camera_info_topic: str = "/d405/color/camera_info",
        flir_left_topic: str = "/camera_array/cam_left/image_raw",
        flir_right_topic: str = "/camera_array/cam_right/image_raw",
        backend: str = "ffmpeg",
        out_dir: str = "unified_recordings",
        sync_fps: float = 25.0,
        ocv_fourcc: str = "mp4v",
        x264_crf: int = 23,
        x264_preset: str = "veryfast",
        invert_depth_colormap: bool = False
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

        self.setWindowTitle("Unified Recorder (Microscope + Stereo FLIR + Intel D405 + Robot)")
        self.setup_ui()

        # --- Latest raw frames (BGR/RGB for video writing) ---
        self.latest_microscope = None
        self.latest_intel_rgb = None
        self.latest_intel_depth = None
        self.latest_flir_left = None   # RGB uint8
        self.latest_flir_right = None  # RGB uint8

        # --- Display buffers (transposed for PyQtGraph) ---
        self.display_microscope = None
        self.display_intel_rgb = None
        self.display_intel_depth = None
        self.display_flir_left = None
        self.display_flir_right = None

        # Recording state
        self.recording = False
        self.microscope_writer = None
        self.intel_rgb_writer = None
        self.flir_left_writer = None
        self.flir_right_writer = None
        self.depth_recorder = DepthRecorder(max_buffer_bytes=3_000_000_000)
        self.robot_logger = TestSub()
        self.session_dir = None

        # FPS measurement
        self.fps_measurement_window = []
        self.measured_fps = None
        self.fps_measurement_complete = False

        # --- ROS Subscribers ---
        rospy.Subscriber(microscope_topic, Image, self.microscope_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(intel_rgb_topic, Image, self.intel_rgb_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(intel_depth_topic, Image, self.intel_depth_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(intel_camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber(flir_left_topic, Image, self.flir_left_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(flir_right_topic, Image, self.flir_right_callback, queue_size=1, buff_size=2 ** 24)

        print(f"[UnifiedRecorder] Subscribed to FLIR left:  {flir_left_topic}")
        print(f"[UnifiedRecorder] Subscribed to FLIR right: {flir_right_topic}")

        # GUI refresh timer ~60Hz
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

        QtWidgets.QApplication.instance().aboutToQuit.connect(self.cleanup)

        print("[UnifiedRecorder] Initialized")
        print(f"[UnifiedRecorder] Syncing at {sync_fps} Hz (depth camera rate)")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def setup_ui(self):
        """
        Layout:
          Top row:    [FLIR Left] | [Microscope] | [FLIR Right]
          Bottom row: [Intel RGB] | [Intel Depth]
          Controls:   [Record btn] [Status]
        """

        def _make_view(label_text: str):
            """Helper: returns (container_widget, pg.ImageItem)."""
            win = pg.GraphicsLayoutWidget()
            view = win.addViewBox()
            view.setAspectLocked(True)
            view.invertY(False)
            img_item = pg.ImageItem()
            view.addItem(img_item)

            lbl = QtWidgets.QLabel(label_text)
            lbl.setAlignment(QtCore.Qt.AlignCenter)

            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.addWidget(lbl)
            layout.addWidget(win)
            return container, img_item

        # --- Top row: stereo + microscope ---
        flir_left_widget,  self.img_flir_left   = _make_view("FLIR Cam Left")
        microscope_widget, self.img_microscope   = _make_view("Microscope (60 Hz)")
        flir_right_widget, self.img_flir_right   = _make_view("FLIR Cam Right")

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(flir_left_widget,  stretch=1)
        top_row.addWidget(microscope_widget, stretch=2)   # microscope gets more width
        top_row.addWidget(flir_right_widget, stretch=1)

        # --- Bottom row: Intel RGB + Depth ---
        intel_rgb_widget,   self.img_intel_rgb   = _make_view("Intel D405 RGB (25 Hz)")
        intel_depth_widget, self.img_intel_depth = _make_view("Intel D405 Depth (25 Hz)")

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addWidget(intel_rgb_widget)
        bottom_row.addWidget(intel_depth_widget)

        # --- Controls ---
        self.btn = QtWidgets.QPushButton("Start REC [S]")
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.toggle_record)

        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(self.btn)
        ctrl_row.addWidget(self.status_label, stretch=1)

        # --- Main layout ---
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(ctrl_row)
        main_layout.addLayout(top_row,    stretch=3)
        main_layout.addLayout(bottom_row, stretch=2)

        self.resize(1800, 1000)
        self.show()

        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=self.btn.click)

    # ------------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------------
    def microscope_callback(self, msg: Image):
        """Manual decode — mirrors _decode_flir_image but kept separate
        since the decklink card may publish uyvy/yuyv422 or mono formats."""
        try:
            if msg.encoding in ("mono8", "bayer_rggb8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif msg.encoding == "mono16":
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                arr = (arr >> 8).astype(np.uint8)          # scale to 8-bit
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif msg.encoding in ("yuv422", "yuyv422", "uyvy"):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
                code = cv2.COLOR_YUV2RGB_UYVY if msg.encoding == "uyvy" else cv2.COLOR_YUV2RGB_YUYV
                arr = cv2.cvtColor(raw, code)
            elif msg.encoding == "bgr8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                arr = arr[:, :, ::-1].copy()               # BGR -> RGB
            elif msg.encoding == "rgba8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                arr = arr[:, :, :3]                        # drop alpha
            else:
                # Default: assume rgb8 / 3-channel uint8
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                arr = arr[:, :, :3]
        except Exception as e:
            print(f"[microscope_callback] Decode failed (encoding={msg.encoding}): {e}")
            return

        self.latest_microscope = arr
        if arr.ndim == 3:
            self.display_microscope = _to_display(arr)
        else:
            self.display_microscope = np.transpose(np.flipud(arr), (1, 0))

    def intel_rgb_callback(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_intel_rgb = arr  # BGR for writing
        self.display_intel_rgb = _to_display(arr)

    def intel_depth_callback(self, msg: Image):
        """SYNC TRIGGER — 25 Hz callback that drives all synchronized logging."""
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self.latest_intel_depth = depth_mm

        # FPS measurement (first 50 depth frames)
        if not self.fps_measurement_complete:
            current_time = time.time()
            self.fps_measurement_window.append(current_time)
            if len(self.fps_measurement_window) >= 50:
                time_span = self.fps_measurement_window[-1] - self.fps_measurement_window[0]
                self.measured_fps = (len(self.fps_measurement_window) - 1) / time_span
                self.fps_measurement_complete = True
                print(f"[Recorder] Measured depth camera FPS: {self.measured_fps:.2f} Hz")

        # Build colorized depth for display
        depth_normalized = np.clip((depth_mm.astype(float) - 70) / (500 - 70), 0, 1)
        if self.invert_depth_colormap:
            depth_normalized = 1.0 - depth_normalized
        depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        bar_h, bar_w = 30, depth_colormap.shape[1]
        gradient = np.tile(np.linspace(0, 255, bar_w).astype(np.uint8), (bar_h, 1))
        color_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        close_lbl = "CLOSE" if not self.invert_depth_colormap else "FAR"
        far_lbl   = "FAR"   if not self.invert_depth_colormap else "CLOSE"
        cv2.putText(color_bar, close_lbl, (10, 20),          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(color_bar, far_lbl,   (bar_w - 60, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.display_intel_depth = _to_display(np.vstack([color_bar, depth_colormap]))

        # --- Synchronized logging ---
        if self.recording and self.all_writers_ready():
            if self.robot_logger.is_ready_to_log():
                sync_timestamp = time.time()
                self.robot_logger.log_trigger(sync_timestamp)

                if self.latest_microscope is not None:
                    self.write_frame(self.microscope_writer, self.latest_microscope)
                if self.latest_intel_rgb is not None:
                    self.write_frame(self.intel_rgb_writer, self.latest_intel_rgb)
                if self.latest_flir_left is not None:
                    self.write_frame(self.flir_left_writer, self.latest_flir_left)
                if self.latest_flir_right is not None:
                    self.write_frame(self.flir_right_writer, self.latest_flir_right)

                self.depth_recorder.add_frame(depth_mm, sync_timestamp)
            else:
                print("[Debug] Robot logger not ready — frame skipped.")

    def camera_info_callback(self, msg: CameraInfo):
        self.depth_recorder.set_camera_info(msg)

    def flir_left_callback(self, msg: Image):
        arr = _decode_flir_image(msg)
        if arr is None:
            return
        self.latest_flir_left = arr
        self.display_flir_left = _to_display(arr)

    def flir_right_callback(self, msg: Image):
        arr = _decode_flir_image(msg)
        if arr is None:
            return
        self.latest_flir_right = arr
        self.display_flir_right = _to_display(arr)

    # ------------------------------------------------------------------
    # Recording Management
    # ------------------------------------------------------------------
    def toggle_record(self):
        if not self.recording:
            if not self.check_frames_available():
                print("[Recorder] Waiting for all camera frames...")
                self.btn.setChecked(False)
                self.status_label.setText("Waiting for frames...")
                return

            if not self.fps_measurement_complete:
                print("[Recorder] Measuring camera FPS, please wait...")
                self.btn.setChecked(False)
                self.status_label.setText(f"Measuring FPS ({len(self.fps_measurement_window)}/50)...")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.out_dir, f"session_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)

            self.create_writers()

            if not self.all_writers_ready():
                print("[Recorder] Failed to create writers.")
                self.btn.setChecked(False)
                self.status_label.setText("Writer creation failed")
                return

            self.robot_logger.clear()
            self.depth_recorder.start_session(
                os.path.join(self.session_dir, "intel_depth")
            )

            self.recording = True
            self.btn.setText("Stop REC [S]")
            self.btn.setChecked(True)
            self.status_label.setText(f"● RECORDING to {self.session_dir}")
            print(f"[Recorder] Started recording to: {self.session_dir}")

        else:
            self.recording = False
            self.btn.setText("Start REC [S]")
            self.btn.setChecked(False)
            self.status_label.setText("Saving files...")

            self.save_all_data()
            self.release_writers()

            self.status_label.setText(f"Saved to {self.session_dir}")
            print(f"[Recorder] Recording complete: {self.session_dir}")

    def check_frames_available(self) -> bool:
        """
        Core sources (microscope + intel) must be present.
        FLIR frames are optional — we won't block recording if cameras
        aren't publishing yet (e.g. during bench testing without stereo rig).
        """
        return (
            self.latest_microscope is not None
            and self.latest_intel_rgb is not None
            and self.latest_intel_depth is not None
        )

    def create_writers(self):
        fps = self.measured_fps if self.measured_fps else self.sync_fps

        if self.latest_microscope is not None:
            h, w = self.latest_microscope.shape[:2]
            path = os.path.join(self.session_dir, "microscope.mp4")
            self.microscope_writer = _make_writer(
                self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset
            )
            print(f"[Recorder] Created microscope writer: {path}")

        if self.latest_intel_rgb is not None:
            h, w = self.latest_intel_rgb.shape[:2]
            path = os.path.join(self.session_dir, "intel_rgb.mp4")
            self.intel_rgb_writer = _make_writer(
                self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset
            )
            print(f"[Recorder] Created Intel RGB writer: {path}")

        # FLIR writers — created only if frames are available; non-blocking
        if self.latest_flir_left is not None:
            h, w = self.latest_flir_left.shape[:2]
            path = os.path.join(self.session_dir, "flir_left.mp4")
            self.flir_left_writer = _make_writer(
                self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset
            )
            print(f"[Recorder] Created FLIR left writer:  {path}")
        else:
            print("[Recorder] FLIR left not available — skipping writer.")

        if self.latest_flir_right is not None:
            h, w = self.latest_flir_right.shape[:2]
            path = os.path.join(self.session_dir, "flir_right.mp4")
            self.flir_right_writer = _make_writer(
                self.backend, path, w, h, fps,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset
            )
            print(f"[Recorder] Created FLIR right writer: {path}")
        else:
            print("[Recorder] FLIR right not available — skipping writer.")

    def all_writers_ready(self) -> bool:
        """Core writers (microscope + intel) must exist; FLIR writers are optional."""
        return (
            self.microscope_writer is not None
            and self.intel_rgb_writer is not None
        )

    def write_frame(self, writer: _BaseWriter, frame: np.ndarray):
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
        robot_csv = os.path.join(self.session_dir, "robot_data.csv")
        self.robot_logger.save_csv(robot_csv)
        depth_npy = os.path.join(self.session_dir, "intel_depth.npz")
        self.depth_recorder.save(depth_npy)
        print(f"[Recorder] All data saved to: {self.session_dir}")

    def release_writers(self):
        for attr in ("microscope_writer", "intel_rgb_writer", "flir_left_writer", "flir_right_writer"):
            writer = getattr(self, attr, None)
            if writer is not None:
                writer.release()
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    # GUI Update
    # ------------------------------------------------------------------
    def tick(self):
        """Refresh all display panels at ~60 Hz."""
        if self.display_flir_left is not None:
            self.img_flir_left.setImage(self.display_flir_left, autoLevels=False)

        if self.display_microscope is not None:
            self.img_microscope.setImage(self.display_microscope, autoLevels=False)

        if self.display_flir_right is not None:
            self.img_flir_right.setImage(self.display_flir_right, autoLevels=False)

        if self.display_intel_rgb is not None:
            self.img_intel_rgb.setImage(self.display_intel_rgb, autoLevels=False)

        if self.display_intel_depth is not None:
            self.img_intel_depth.setImage(self.display_intel_depth, autoLevels=False)

        if not self.recording:
            if not self.fps_measurement_complete:
                self.status_label.setText(
                    f"Measuring camera FPS... ({len(self.fps_measurement_window)}/50)"
                )
            elif self.measured_fps:
                flir_l = "✓" if self.latest_flir_left  is not None else "✗"
                flir_r = "✓" if self.latest_flir_right is not None else "✗"
                self.status_label.setText(
                    f"Ready  |  {self.measured_fps:.1f} fps  |  "
                    f"FLIR L:{flir_l}  R:{flir_r}"
                )

    def cleanup(self):
        if self.recording:
            print("[Recorder] Cleaning up on exit...")
            self.toggle_record()


# --- Main Entry Point ---
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