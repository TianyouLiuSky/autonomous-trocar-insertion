#!/usr/bin/env python3
"""mic_live_capture_ros.py -- ROS-topic version of mic_live_capture.py.

Use this INSTEAD of mic_live_capture.py once the DeckLink capture card is
actually publishing the Leica feed to ROS (i.e. once
`roslaunch gscam gscam_decklink.launch connection:=<sdi|hdmi>` is running
and `rostopic hz /decklink/camera/image_raw` shows real data).

See /home/eyebrp/Matthew/microscope_autonomy/DECKLINK_SETUP.md for the full
setup checklist (what's already confirmed working on this machine, and the
two steps still blocked on manual user action: sudo package install +
downloading the Blackmagic Desktop Video SDK from their site).

Same contract as mic_live_capture.py: writes a rolling window of recent
frames into MIC_LIVE_STILLS_DIR (default mic_live_stills/), which is the
exact folder get_mic_preview() / the 'y' key / the PiP panel in
home_to_rcm_insert_v2_precision.py already watch (newest-file-wins). No
other code needs to change once this is running.

USAGE (after sourcing ROS + the gscam launch file is already up):
    python3 mic_live_capture_ros.py
    python3 mic_live_capture_ros.py --show                 # + local preview window
    python3 mic_live_capture_ros.py --topic /decklink/camera/image_raw --interval 0.3
"""
from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from runtime_session import ROOT, data_path
import argparse
import glob
import os
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.environ.get(
    "MIC_LIVE_STILLS_DIR", data_path("mic_live_stills")
)
DEFAULT_TOPIC = "/decklink/camera/image_raw"
DEFAULT_KEEP = 5


# Manual decode (no cv_bridge) -- same pattern already used by
# unified_camera_display.py / live_segmentation.py in the
# Autonomous-Trocar-Insertion package, and avoids any cv_bridge<->cv2
# version coupling issues.
def decode_image_msg(msg):
    enc = msg.encoding
    h, w = msg.height, msg.width
    try:
        if enc == "bgr8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
        if enc == "rgb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            return arr[:, :, ::-1].copy()
        if enc in ("yuv422", "yuyv422", "uyvy"):
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 2)
            code = cv2.COLOR_YUV2BGR_UYVY if enc == "uyvy" else cv2.COLOR_YUV2BGR_YUYV
            return cv2.cvtColor(raw, code)
        if enc == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if enc == "mono16":
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            return cv2.cvtColor((arr >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if enc == "bayer_rggb8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(arr, cv2.COLOR_BayerBG2BGR)
        if enc == "rgba8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
        return arr[:, :, :3].copy()
    except Exception as e:
        print("[decode] enc=%s failed: %s" % (enc, e))
        return None


def prune_old(out_dir, keep):
    files = sorted(glob.glob(os.path.join(out_dir, "live_*.jpg")), key=os.path.getmtime)
    for f in files[:-keep] if keep > 0 else []:
        try:
            os.remove(f)
        except OSError:
            pass


class Bridge(object):
    def __init__(self, out_dir, interval, keep, show):
        self.out_dir = out_dir
        self.interval = interval
        self.keep = keep
        self.show = show
        self.last_save = 0.0
        self.n_saved = 0
        self.n_recv = 0
        self.t_first = None

    def cb(self, msg):
        self.n_recv += 1
        if self.t_first is None:
            self.t_first = time.time()
            print("[mic_live_capture_ros] first frame received: %dx%d enc=%s"
                  % (msg.width, msg.height, msg.encoding))
        now = time.time()
        if now - self.last_save < self.interval:
            return
        frame = decode_image_msg(msg)
        if frame is None:
            return
        # 2026-09-22: write-then-atomic-rename, not a direct imwrite to the
        # final name -- now that this folder is refreshed continuously (was
        # a manual one-off file drop before), get_mic_preview() in
        # home_to_rcm_insert_v2_precision.py can race a direct imwrite and
        # cv2.imread() a half-written JPEG ("Premature end of JPEG file"),
        # since imwrite is NOT atomic (readers can observe partial writes
        # mid-flush). os.rename() on the same filesystem IS atomic on POSIX,
        # so any reader doing os.listdir()+imread() only ever sees either the
        # old file or the fully-written new one, never a partial one.
        fname = "live_%013d.jpg" % int(now * 1000)
        final_path = os.path.join(self.out_dir, fname)
        tmp_path = final_path + ".tmp"
        # 2026-09-24 BUGFIX: this used to be
        #   cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # which FAILED on every single call ("could not find a writer for
        # the specified extension") because cv2.imwrite picks its encoder
        # from the filename's trailing extension, and tmp_path ends in
        # ".tmp", not ".jpg" -- silently, via ROS's "bad callback" handler,
        # which does NOT crash this node or stop it from looking "running".
        # Net effect for ~43 hours (2026-09-22 18:12 through 2026-09-24
        # 13:13, caught via the ROS log filling with this exact traceback at
        # ~60Hz): NO new still was EVER written after the atomic-write fix's
        # own restart, so get_mic_preview() in
        # home_to_rcm_insert_v2_precision.py kept re-reading the same
        # last-good frame from before this bug and (because
        # MIC_LIVE_MAX_AGE_S was ALSO left disabled, see that constant)
        # never flagged it as stale -- the PiP panel silently froze on a
        # 2-day-old photo while still labeled "REAL TIME IMAGE" in green.
        #
        # Fix: encode to JPEG bytes explicitly with cv2.imencode(".jpg", ...)
        # -- which takes the format as an explicit argument, not inferred
        # from a path -- then write those bytes with plain file I/O.
        # tmp_path can then keep the ".tmp" extension it needs to NOT match
        # get_mic_preview()'s os.listdir() filter
        # (f.lower().endswith((".jpg",".jpeg",".png"))) while a write is
        # still in progress -- that filter match is exactly what the
        # temp-then-rename pattern is trying to prevent a reader from ever
        # seeing, so the temp file must NOT look like a candidate image.
        ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok_enc:
            rospy.logwarn_throttle(5, "[mic_live_capture_ros] cv2.imencode failed, skipping frame")
            return
        with open(tmp_path, "wb") as fh:
            fh.write(buf.tobytes())
        os.rename(tmp_path, final_path)
        prune_old(self.out_dir, self.keep)
        self.last_save = now
        self.n_saved += 1
        if self.n_saved % 20 == 1:
            print("  saved %d stills so far (latest: %s, %d msgs received total)"
                  % (self.n_saved, fname, self.n_recv))
        if self.show:
            disp = frame if frame.shape[1] <= 960 else cv2.resize(frame, (960, 540))
            cv2.putText(disp, "mic_live_capture_ros -- Ctrl+C in terminal to quit", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("mic_live_capture_ros preview", disp)
            cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rospy.init_node("mic_live_capture_ros", anonymous=True, disable_signals=True)

    bridge = Bridge(args.out_dir, args.interval, args.keep, args.show)
    rospy.Subscriber(args.topic, Image, bridge.cb, queue_size=1, buff_size=2**24)

    print("Subscribed to %s -> writing to %s every %.2fs (Ctrl+C to stop)"
          % (args.topic, args.out_dir, args.interval))
    print("Waiting for first frame... (if this hangs, gscam_decklink.launch "
          "probably isn't running or isn't receiving a signal -- check "
          "`rostopic hz %s` in another terminal)" % args.topic)

    t0 = time.time()
    try:
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
            if bridge.t_first is None and time.time() - t0 > 10:
                print("  ... still no frames after 10s. Checklist:")
                print("      1) is gscam_decklink.launch actually running?")
                print("      2) `rostopic list | grep decklink` shows the topic?")
                print("      3) is the Leica's video cable actually plugged into "
                      "the DeckLink card's input on the BACK of THIS machine?")
                t0 = time.time()  # only nag once per 10s
    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()
        print("\nStopped. %d stills written (%d ROS messages received total)."
              % (bridge.n_saved, bridge.n_recv))


if __name__ == "__main__":
    main()
