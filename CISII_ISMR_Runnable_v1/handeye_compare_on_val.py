#!/usr/bin/env python3
"""Fair, independent comparison of two candidate hand-eye calibrations
(e.g. the old n=12 vs the new n=100) on a common, TRUE held-out validation
set that was never used to fit either one.

Usage:
    python3 handeye_compare_on_val.py VAL_CSV NPZ_A NPZ_B [NPZ_C ...]

VAL_CSV must be in the same schema as handeye_data/pairs.csv (collect with
handeye_capture_one.py, appending fresh poses to a NEW file -- do not reuse
any CSV that was already used to fit one of the candidates).

Both candidates are evaluated with the SAME intrinsics (live RealSense
factory values, to match how the promoted n=100 fit was actually solved and
avoid introducing a self-cal-intrinsics confound into the comparison).
"""
import os
import sys

import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handeye_reproj_refine import load_pairs_csv, recover_pixels, eval_on_set  # noqa: E402

if len(sys.argv) < 4:
    raise SystemExit(__doc__)

VAL_CSV = sys.argv[1]
NPZ_PATHS = sys.argv[2:]

print("Loading LIVE RealSense factory intrinsics (camera must be free)...")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
for _ in range(10):
    pipe.wait_for_frames()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pipe.stop()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
dist = np.array(list(intr.coeffs[:5]), float)
print("  fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f dist=%s\n" % (intr.fx, intr.fy, intr.ppx, intr.ppy, dist))

Pcam_v, Pbe_v, Rbe_v = load_pairs_csv(VAL_CSV)
n_val = len(Pcam_v)
uv_v = recover_pixels(Pcam_v, intr)
print("Held-out validation set: %d poses from %s\n" % (n_val, VAL_CSV))

results = []
for path in NPZ_PATHS:
    d = np.load(path)
    R, t, m = d["R"], d["t"], d["m"]
    n_fit = int(d["n"]) if "n" in d.files else -1
    val_rms, val_err = eval_on_set(Pbe_v, Rbe_v, uv_v, K, dist, R, t, m)
    results.append((path, n_fit, val_rms, val_err))
    print("=== %s  (fit n=%d) ===" % (os.path.basename(path), n_fit))
    print("  validation RMS on %d NEW poses = %.2f px" % (n_val, val_rms))
    print("  per-pose errors: %s\n" % np.array2string(val_err, precision=2))

print("=== VERDICT ===")
best = min(results, key=lambda r: r[2])
for path, n_fit, val_rms, _ in sorted(results, key=lambda r: r[2]):
    flag = "  <-- LOWER = better generalization" if path == best[0] else ""
    print("  %-55s fit_n=%-4d val_rms=%.2fpx%s" % (os.path.basename(path), n_fit, val_rms, flag))
