#!/usr/bin/env python3
"""mic_live_capture.py -- turn a physically-connected capture card into a
continuously-refreshed "live" feed for the existing microscope pipeline.

WHY THIS EXISTS (2026-09-22): up to now every Leica still used by this
project (segment_leica.py, mic_perception.py, the PiP panel and the 'y' key
in home_to_rcm_insert_v2_precision.py) was a ONE-OFF file manually copied
off the Leica's separate Windows PC. There has never been a live video path
from the microscope into this Linux machine -- confirmed today: only the
D405 RealSense shows up in `v4l2-ctl --list-devices` / `lsusb`, no capture
card is plugged in yet.

This script does NOT change that hardware fact -- you still need a physical
capture device (HDMI/DVI/SDI-to-USB) plugged into THIS Linux box, tapped
off the same video signal that already drives the Leica's external monitor.
See the README section this script ships with for the physical setup steps.

WHAT THIS SCRIPT ADDS, once that capture device exists: it opens the new
/dev/videoN with OpenCV and writes a rolling window of recent frames into
MIC_LIVE_STILLS_DIR (default: mic_live_stills/, the exact folder
get_mic_preview() / the 'y' key / the PiP panel in
home_to_rcm_insert_v2_precision.py already watch and always show the
NEWEST file from). That means: NO changes are needed anywhere else --
the existing PiP panel starts refreshing automatically, and 'y' starts
locking onto genuinely fresh frames instead of a manually-copied still,
the moment this script is running.

USAGE (once the capture card shows up as e.g. /dev/video3):
    python3 mic_live_capture.py --device 3
    python3 mic_live_capture.py --device 3 --show          # + local preview window
    python3 mic_live_capture.py --device 3 --interval 0.3  # write every 0.3s (default 0.5s)
    python3 mic_live_capture.py --list                     # just list candidate devices and exit

Auto-detect mode (no --device given): probes every /dev/video* NOT already
claimed by the RealSense (D405) and opens the first one that yields a
plausible non-black frame -- convenient, but pass --device explicitly once
you know the right index to avoid ever racing the D405 by accident.
"""
from __future__ import print_function
import argparse
import glob
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.environ.get(
    "MIC_LIVE_STILLS_DIR", os.path.join(HERE, "mic_live_stills")
)
DEFAULT_KEEP = 5  # how many recent stills to retain in the watch folder


def realsense_video_nodes():
    """Return the set of /dev/videoN indices already claimed by the D405,
    so auto-detect never opens (and steals) one of those by accident."""
    claimed = set()
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"],
                                       stderr=subprocess.DEVNULL).decode()
    except Exception:
        return claimed
    blocks = re.split(r"\n(?=\S)", out)
    for b in blocks:
        if "realsense" in b.lower():
            for m in re.finditer(r"/dev/video(\d+)", b):
                claimed.add(int(m.group(1)))
    return claimed


def candidate_devices():
    claimed = realsense_video_nodes()
    idxs = []
    for path in sorted(glob.glob("/dev/video*")):
        m = re.match(r"/dev/video(\d+)$", path)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in claimed:
            idxs.append(idx)
    return idxs


def looks_like_real_frame(frame):
    if frame is None:
        return False
    return float(np.std(frame)) > 3.0  # not a flat black/gray dead frame


def open_capture(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    # Ask for a sane HD size; capture cards will clamp to what they actually
    # output if this doesn't match, so this is a hint, not a hard requirement.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def prune_old(out_dir, keep):
    files = sorted(glob.glob(os.path.join(out_dir, "live_*.jpg")),
                    key=os.path.getmtime)
    for f in files[:-keep] if keep > 0 else []:
        try:
            os.remove(f)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None,
                    help="/dev/videoN index to use. Omit to auto-detect.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--interval", type=float, default=0.5,
                    help="seconds between saved stills (default 0.5s)")
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                    help="how many recent stills to retain (default 5, 0=keep all)")
    ap.add_argument("--show", action="store_true",
                    help="also pop a local OpenCV preview window")
    ap.add_argument("--list", action="store_true",
                    help="list candidate (non-RealSense) /dev/video* devices and exit")
    args = ap.parse_args()

    if args.list:
        idxs = candidate_devices()
        claimed = realsense_video_nodes()
        print("RealSense (D405) already owns:", sorted(claimed) or "(none found)")
        print("Candidate microscope-capture devices:", idxs or "(none -- capture card not connected/detected yet)")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device is not None:
        idxs_to_try = [args.device]
    else:
        idxs_to_try = candidate_devices()
        if not idxs_to_try:
            print("No non-RealSense /dev/video* device found.")
            print("-> Capture card is not plugged in / not recognized yet.")
            print("   Run 'lsusb' and 'v4l2-ctl --list-devices' to check, or")
            print("   see this file's docstring / the README for the physical")
            print("   HDMI/DVI/SDI capture-card setup steps.")
            sys.exit(1)
        print("Auto-detect candidates:", idxs_to_try)

    cap, chosen = None, None
    for idx in idxs_to_try:
        c = open_capture(idx)
        if c is None:
            continue
        ok, frame = c.read()
        if ok and looks_like_real_frame(frame):
            cap, chosen = c, idx
            break
        c.release()

    if cap is None:
        print("Could not open a working, non-black video feed on any candidate device.")
        print("Tried:", idxs_to_try)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Opened /dev/video%d  (%dx%d)  -> writing to %s every %.2fs"
          % (chosen, w, h, args.out_dir, args.interval))
    print("Press Ctrl+C to stop.")

    last_save = 0.0
    n_saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed, retrying...")
                time.sleep(0.2)
                continue
            now = time.time()
            if now - last_save >= args.interval:
                # write-then-atomic-rename -- see mic_live_capture_ros.py's
                # matching comment (2026-09-22): avoids readers seeing a
                # half-written JPEG mid-imwrite ("Premature end of JPEG file").
                fname = "live_%013d.jpg" % int(now * 1000)
                fpath = os.path.join(args.out_dir, fname)
                tmp_path = fpath + ".tmp"
                # 2026-09-24 BUGFIX: cv2.imwrite(tmp_path, ...) fails on
                # every call because it picks its encoder from the
                # filename's trailing extension, and tmp_path ends in
                # ".tmp", not ".jpg" -- see mic_live_capture_ros.py's
                # matching fix/comment (that copy ran for ~43 hours silently
                # failing every single write because of this exact bug).
                # Fix: encode explicitly with cv2.imencode(".jpg", ...) --
                # format is a direct argument there, not inferred from a
                # path -- then write the bytes with plain file I/O, keeping
                # tmp_path's ".tmp" extension so it does NOT match any
                # reader's os.listdir()+endswith(".jpg") filter while the
                # write is still in progress (the whole point of this
                # temp-then-rename pattern).
                ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok_enc:
                    print("cv2.imencode failed, skipping frame")
                else:
                    with open(tmp_path, "wb") as fh:
                        fh.write(buf.tobytes())
                    os.rename(tmp_path, fpath)
                prune_old(args.out_dir, args.keep)
                last_save = now
                n_saved += 1
                if n_saved % 20 == 1:
                    print("  saved %d stills so far (latest: %s)" % (n_saved, fname))
            if args.show:
                disp = cv2.resize(frame, (960, 540)) if w > 960 else frame
                cv2.putText(disp, "mic_live_capture -- q to quit", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("mic_live_capture preview", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print("\nStopped. %d stills written to %s (rolling window of %d kept)."
              % (n_saved, args.out_dir, args.keep))


if __name__ == "__main__":
    main()
