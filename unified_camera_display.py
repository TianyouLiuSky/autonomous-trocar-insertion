"""
Unified synchronized recorder for microscope video, Intel D405 depth camera, and robot data.
All data is synchronized at 25Hz (depth camera rate).
"""

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
from GUI_subscriber.utils.image_conversion_without_using_ros import image_to_numpy
from GUI_subscriber.dumb_demo.robot_subscriber_messy import TestSub


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
        self.frames = []          # List of depth frames (current chunk)
        self.timestamps = []      # Corresponding timestamps (current chunk)
        self.camera_info = None

        self.max_buffer_bytes = max_buffer_bytes
        self._buffer_bytes = 0
        self._base_prefix = None  # e.g. /path/to/session/intel_depth
        self._chunk_idx = 0

        # Background writer
        self._save_queue = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )
        self._writer_thread.start()

    def start_session(self, base_prefix: str):
        """
        Called at the start of each recording session.
        base_prefix example: '/.../session_xxx/intel_depth'
        """
        self._base_prefix = base_prefix
        self.frames = []
        self.timestamps = []
        self._buffer_bytes = 0
        self._chunk_idx = 0

    def add_frame(self, depth_mm: np.ndarray, timestamp: float):
        """
        Stores depth in 0.01mm precision using uint16.
        Original: uint16 millimeters (0-65535mm)
        Stored: uint16 in 0.01mm units (0-655.35mm with 0.01mm precision)

        Args:
            depth_mm: Depth array in millimeters
            timestamp: Synchronized timestamp (from time.time()) at sync trigger
        """
        depth_001mm = np.clip(
            depth_mm.astype(np.uint32) * 100, 0, 65535
        ).astype(np.uint16)

        self.frames.append(depth_001mm)
        self.timestamps.append(timestamp)
        self._buffer_bytes += depth_001mm.nbytes

        # When buffer grows beyond threshold, dump current chunk to disk
        if (
            self._base_prefix is not None
            and self._buffer_bytes >= self.max_buffer_bytes
        ):
            self._flush_async()

    def set_camera_info(self, camera_info: CameraInfo):
        """Store camera intrinsics."""
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
        """Move current in-memory chunk to background writer."""
        if not self.frames:
            return

        frames, timestamps = self.frames, self.timestamps
        self.frames, self.timestamps = [], []
        self._buffer_bytes = 0

        idx = self._chunk_idx
        self._chunk_idx += 1
        self._save_queue.put((idx, frames, timestamps))

    def _writer_loop(self):
        """Background thread that writes chunks to disk."""
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
        """
        Finalize saving.
        If no chunking happened, behaves like before and writes a single NPZ.
        If chunking happened, writes remaining frames as a final chunk and
        leaves you with multiple files:
            <base>_chunk0000.npz, <base>_chunk0001.npz, ...
        """
        if not self.frames and self._chunk_idx == 0:
            print("[DepthRecorder] No frames to save.")
            return

        if self._base_prefix is None:
            # Derive base prefix from the requested filepath
            self._base_prefix = os.path.splitext(filepath)[0]

        # No chunking: single file, original behavior
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
            # We already wrote one or more chunks.
            # Flush the remaining in-memory frames as a final chunk.
            if self.frames:
                self._flush_async()

            # Wait for all chunks to finish writing
            self._save_queue.join()

            # Stop writer thread cleanly (optional for short-lived process)
            self._save_queue.put(None)
            self._save_queue.join()

            print(f"[DepthRecorder] Saved {self._chunk_idx} chunks with prefix: {self._base_prefix}_chunkXXXX")

    def clear(self):
        """Reset in-memory buffers for a new session (does not touch files)."""
        self.frames = []
        self.timestamps = []
        self._buffer_bytes = 0


# --- Main Unified Recorder ---
class UnifiedRecorder(QtWidgets.QWidget):
    def __init__(
        self,
        microscope_topic: str = "/decklink/camera/image_raw",
        intel_rgb_topic: str = "/d405/color/image_raw",
        intel_depth_topic: str = "/d405/aligned_depth_to_color/image_raw",  # <--- CRITICAL: Aligned topic
        intel_camera_info_topic: str = "/d405/color/camera_info",
        backend: str = "ffmpeg",
        out_dir: str = "unified_recordings",
        sync_fps: float = 25.0,
        ocv_fourcc: str = "mp4v",
        x264_crf: int = 23,
        x264_preset: str = "veryfast",
        invert_depth_colormap: bool = False  # Set to True if colormap looks reversed
    ):
        super().__init__()

        # Configuration
        self.bridge = CvBridge()
        self.invert_depth_colormap = invert_depth_colormap
        self.backend = backend
        self.out_dir = out_dir
        self.sync_fps = sync_fps
        self.ocv_fourcc = ocv_fourcc
        self.x264_crf = x264_crf
        self.x264_preset = x264_preset

        # Setup UI
        self.setWindowTitle("Unified Recorder (Microscope + Intel D405 + Robot)")
        self.setup_ui()

        # Data buffers (latest frame from each source)
        self.latest_microscope = None
        self.latest_intel_rgb = None
        self.latest_intel_depth = None

        # Display buffers (transposed for PyQtGraph)
        self.display_microscope = None
        self.display_intel_rgb = None
        self.display_intel_depth = None

        # Recording state
        self.recording = False
        self.microscope_writer = None
        self.intel_rgb_writer = None
        self.depth_recorder = DepthRecorder(max_buffer_bytes=3_000_000_000)  # ~1 GB buffer
        self.robot_logger = TestSub()
        self.session_dir = None

        # FPS measurement
        self.fps_measurement_window = []
        self.measured_fps = None
        self.fps_measurement_complete = False

        # Subscribe to topics
        rospy.Subscriber(microscope_topic, Image, self.microscope_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(intel_rgb_topic, Image, self.intel_rgb_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(intel_depth_topic, Image, self.intel_depth_callback, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(intel_camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        # GUI refresh timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60Hz GUI refresh

        # Cleanup on exit
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.cleanup)

        print("[UnifiedRecorder] Initialized")
        print(f"[UnifiedRecorder] Syncing at {sync_fps} Hz (depth camera rate)")

    def setup_ui(self):
        """Create the UI with 3 video displays."""
        # Record button
        self.btn = QtWidgets.QPushButton("Start REC [S]")
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.toggle_record)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)

        # Graphics windows for video display
        self.win_microscope = pg.GraphicsLayoutWidget()
        self.view_microscope = self.win_microscope.addViewBox()
        self.img_microscope = pg.ImageItem()
        self.view_microscope.addItem(self.img_microscope)
        self.view_microscope.setAspectLocked(True)

        self.win_intel_rgb = pg.GraphicsLayoutWidget()
        self.view_intel_rgb = self.win_intel_rgb.addViewBox()
        self.img_intel_rgb = pg.ImageItem()
        self.view_intel_rgb.addItem(self.img_intel_rgb)
        self.view_intel_rgb.setAspectLocked(True)

        self.win_intel_depth = pg.GraphicsLayoutWidget()
        self.view_intel_depth = self.win_intel_depth.addViewBox()
        self.img_intel_depth = pg.ImageItem()
        self.view_intel_depth.addItem(self.img_intel_depth)
        self.view_intel_depth.setAspectLocked(True)

        # Labels
        label_microscope = QtWidgets.QLabel("Microscope (60Hz)")
        label_microscope.setAlignment(QtCore.Qt.AlignCenter)
        label_intel_rgb = QtWidgets.QLabel("Intel D405 RGB (25Hz)")
        label_intel_rgb.setAlignment(QtCore.Qt.AlignCenter)
        label_intel_depth = QtWidgets.QLabel("Intel D405 Depth (25Hz)")
        label_intel_depth.setAlignment(QtCore.Qt.AlignCenter)

        # Layout: Top row = microscope, Bottom row = intel RGB + depth
        top_layout = QtWidgets.QVBoxLayout()
        top_layout.addWidget(label_microscope)
        top_layout.addWidget(self.win_microscope)

        bottom_left = QtWidgets.QVBoxLayout()
        bottom_left.addWidget(label_intel_rgb)
        bottom_left.addWidget(self.win_intel_rgb)

        bottom_right = QtWidgets.QVBoxLayout()
        bottom_right.addWidget(label_intel_depth)
        bottom_right.addWidget(self.win_intel_depth)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addLayout(bottom_left)
        bottom_layout.addLayout(bottom_right)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.btn)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(top_layout, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=1)

        self.resize(1600, 1000)
        self.show()

        # Keyboard shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=self.btn.click)

    # --- Callbacks ---
    def microscope_callback(self, msg: Image):
        """Store latest microscope frame."""
        arr = image_to_numpy(msg)
        if arr is None:
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        self.latest_microscope = arr

        # Prepare for display (transpose for PyQtGraph)
        if arr.ndim == 3:
            self.display_microscope = np.transpose(np.flipud(arr), (1, 0, 2))
        else:
            self.display_microscope = np.transpose(np.flipud(arr), (1, 0))

    def intel_rgb_callback(self, msg: Image):
        """Store latest Intel RGB frame."""
        # RealSense publishes in BGR8 format (not RGB8)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_intel_rgb = arr  # Keep as BGR for video writing

        # Convert BGR to RGB for display
        # arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr_rgb = arr
        self.display_intel_rgb = np.transpose(np.flipud(arr_rgb), (1, 0, 2))

    def intel_depth_callback(self, msg: Image):
        """
        SYNC TRIGGER: This is the 25Hz callback that triggers all synchronized logging.
        """
        # Store depth frame (uint16 millimeters)
        depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self.latest_intel_depth = depth_mm

        # Measure actual FPS during startup (first 50 frames)
        if not self.fps_measurement_complete:
            current_time = time.time()
            self.fps_measurement_window.append(current_time)

            if len(self.fps_measurement_window) >= 50:
                time_span = self.fps_measurement_window[-1] - self.fps_measurement_window[0]
                self.measured_fps = (len(self.fps_measurement_window) - 1) / time_span
                self.fps_measurement_complete = True
                print(f"[Recorder] Measured depth camera FPS: {self.measured_fps:.2f} Hz")
                print(f"[Recorder] Videos will be saved at {self.measured_fps:.2f} fps for accurate playback")

        # Create colorized depth for display
        # D405 optimal range: 70-500mm
        depth_normalized = np.clip((depth_mm.astype(float) - 70) / (500 - 70), 0, 1)

        # Apply inversion if configured
        if self.invert_depth_colormap:
            depth_normalized = 1.0 - depth_normalized

        # COLORMAP_JET: 0.0=blue, 1.0=red
        # Default (no invert): close (70mm)→0.0→blue, far (500mm)→1.0→red
        # Inverted: close (70mm)→1.0→red, far (500mm)→0.0→blue
        depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Add color reference bar
        bar_height = 30
        bar_width = depth_colormap.shape[1]
        color_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        # Create gradient bar
        gradient = np.linspace(0, 255, bar_width).astype(np.uint8)
        gradient = np.tile(gradient, (bar_height, 1))
        color_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

        # Add labels
        label_text = "CLOSE" if not self.invert_depth_colormap else "FAR"
        cv2.putText(color_bar, label_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        label_text = "FAR" if not self.invert_depth_colormap else "CLOSE"
        cv2.putText(color_bar, label_text, (bar_width-60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine depth image with color bar
        depth_with_bar = np.vstack([color_bar, depth_colormap])

        self.display_intel_depth = np.transpose(np.flipud(depth_with_bar), (1, 0, 2))

        # --- SYNCHRONIZED LOGGING BLOCK ---
        if self.recording and self.all_writers_ready():
            if self.robot_logger.is_ready_to_log():
                # Create single synchronized timestamp for this frame
                sync_timestamp = time.time()

                # Log robot data with timestamp
                self.robot_logger.log_trigger(sync_timestamp)

                # Write microscope frame
                if self.latest_microscope is not None:
                    self.write_frame(self.microscope_writer, self.latest_microscope)

                # Write Intel RGB frame
                if self.latest_intel_rgb is not None:
                    self.write_frame(self.intel_rgb_writer, self.latest_intel_rgb)

                # Record depth frame with same timestamp
                self.depth_recorder.add_frame(depth_mm, sync_timestamp)
            else:
                # Add this to see if the robot is holding you back
                print(f"[Debug] Robot logger not ready! Recording skipped for this frame.")

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        self.depth_recorder.set_camera_info(msg)

    # --- Recording Management ---
    def toggle_record(self):
        """Start/stop recording."""
        if not self.recording:
            # --- START RECORDING ---
            if not self.check_frames_available():
                print("[Recorder] Waiting for all camera frames...")
                self.btn.setChecked(False)
                self.status_label.setText("Waiting for frames...")
                return

            # Wait for FPS measurement to complete
            if not self.fps_measurement_complete:
                print("[Recorder] Measuring camera FPS, please wait...")
                self.btn.setChecked(False)
                self.status_label.setText(f"Measuring FPS ({len(self.fps_measurement_window)}/50)...")
                return

            # Create session directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.out_dir, f"session_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)

            # Initialize writers
            self.create_writers()

            if not self.all_writers_ready():
                print("[Recorder] Failed to create writers.")
                self.btn.setChecked(False)
                self.status_label.setText("Writer creation failed")
                return

            # Clear buffers / start new depth session
            self.robot_logger.clear()
            self.depth_recorder.start_session(
                os.path.join(self.session_dir, "intel_depth")
            )

            # Start recording
            self.recording = True
            self.btn.setText("Stop REC [S]")
            self.btn.setChecked(True)
            self.status_label.setText(f"● RECORDING to {self.session_dir}")
            print(f"[Recorder] Started recording to: {self.session_dir}")

        else:
            # --- STOP RECORDING ---
            self.recording = False
            self.btn.setText("Start REC [S]")
            self.btn.setChecked(False)
            self.status_label.setText("Saving files...")

            # Save all data
            self.save_all_data()

            # Cleanup
            self.release_writers()

            self.status_label.setText(f"Saved to {self.session_dir}")
            print(f"[Recorder] Recording complete: {self.session_dir}")

    def check_frames_available(self) -> bool:
        """Check if we have at least one frame from all sources."""
        return (self.latest_microscope is not None and
                self.latest_intel_rgb is not None and
                self.latest_intel_depth is not None)

    def create_writers(self):
        """Create video writers for microscope and Intel RGB."""
        # Use measured FPS for accurate playback
        fps_to_use = self.measured_fps if self.measured_fps else self.sync_fps

        if self.latest_microscope is not None:
            h, w = self.latest_microscope.shape[:2]
            path = os.path.join(self.session_dir, "microscope.mp4")
            self.microscope_writer = _make_writer(
                backend=self.backend, path=path, w=w, h=h, fps=fps_to_use,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset
            )
            print(f"[Recorder] Created microscope writer: {path} @ {fps_to_use:.2f} fps")

        if self.latest_intel_rgb is not None:
            h, w = self.latest_intel_rgb.shape[:2]
            path = os.path.join(self.session_dir, "intel_rgb.mp4")
            self.intel_rgb_writer = _make_writer(
                backend=self.backend, path=path, w=w, h=h, fps=fps_to_use,
                fourcc=self.ocv_fourcc, crf=self.x264_crf, preset=self.x264_preset
            )
            print(f"[Recorder] Created Intel RGB writer: {path} @ {fps_to_use:.2f} fps")

    def all_writers_ready(self) -> bool:
        """Check if all writers are initialized."""
        return (self.microscope_writer is not None and
                self.intel_rgb_writer is not None)

    def write_frame(self, writer: _BaseWriter, frame: np.ndarray):
        """Convert and write frame to video file."""
        # Handle grayscale (Depth or IR)
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Handle RGBA (some cameras)
        elif frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame_bgr)

    def save_all_data(self):
        """Save robot data and depth data."""
        if self.session_dir is None:
            return

        # Save robot data as CSV
        robot_csv = os.path.join(self.session_dir, "robot_data.csv")
        self.robot_logger.save_csv(robot_csv)

        # Save depth data as compressed numpy
        depth_npy = os.path.join(self.session_dir, "intel_depth.npz")
        self.depth_recorder.save(depth_npy)

        print(f"[Recorder] All data saved to: {self.session_dir}")

    def release_writers(self):
        """Release all video writers."""
        if self.microscope_writer is not None:
            self.microscope_writer.release()
            self.microscope_writer = None

        if self.intel_rgb_writer is not None:
            self.intel_rgb_writer.release()
            self.intel_rgb_writer = None

    # --- GUI Update ---
    def tick(self):
        """Update display at ~60Hz."""
        if self.display_microscope is not None:
            self.img_microscope.setImage(self.display_microscope, autoLevels=False)

        if self.display_intel_rgb is not None:
            self.img_intel_rgb.setImage(self.display_intel_rgb, autoLevels=False)

        if self.display_intel_depth is not None:
            self.img_intel_depth.setImage(self.display_intel_depth, autoLevels=False)

        # Update status during FPS measurement
        if not self.recording and not self.fps_measurement_complete:
            fps_progress = len(self.fps_measurement_window)
            self.status_label.setText(f"Measuring camera FPS... ({fps_progress}/50)")
        elif not self.recording and self.fps_measurement_complete and self.measured_fps:
            self.status_label.setText(f"Ready - Camera: {self.measured_fps:.1f} fps")

    def cleanup(self):
        """Cleanup on exit."""
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
        intel_depth_topic="/d405/aligned_depth_to_color/image_raw",  # <--- Aligned Depth
        intel_camera_info_topic="/d405/color/camera_info",
        backend="ffmpeg",
        out_dir=os.path.expanduser("~/Documents/unified_recordings"),
        sync_fps=25.0,
        invert_depth_colormap=True  # Set to True if depth colors look reversed
    )

    app.exec_()
