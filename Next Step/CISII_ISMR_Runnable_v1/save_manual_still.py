#!/usr/bin/env python3
"""Copy the newest live microscope still to the latest trial as trialNN_<stamp>_manual.jpg
(measure_trial_holes.py prefers it over the automatic post-retract stills) and mark the trial
JSON with retract_adhesion=true. Use after re-seating a phantom that was dragged on RETRACT.
Usage: python3 save_manual_still.py [--no-adhesion]"""

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STILLS_DIR = data_path("trial_plunge_stills")
LIVE_DIR = data_path("mic_live_stills")

jsons = sorted(glob.glob(os.path.join(STILLS_DIR, "trial*.json")), key=os.path.getmtime)
if not jsons:
    sys.exit("no trial JSON in %s" % STILLS_DIR)
jp = jsons[-1]
lives = [os.path.join(LIVE_DIR, f) for f in os.listdir(LIVE_DIR) if f.endswith(".jpg")]
if not lives:
    sys.exit("no live microscope stills in %s" % LIVE_DIR)
src = max(lives, key=os.path.getmtime)
age = time.time() - os.path.getmtime(src)
if age > 5:
    sys.exit("newest live still is %.0fs old -- is mic_live_capture_ros.py running?" % age)
dst = jp[:-5] + "_manual.jpg"
shutil.copy2(src, dst)
d = json.load(open(jp))
d["manual_still"] = os.path.basename(dst)
d["retract_adhesion"] = "--no-adhesion" not in sys.argv
json.dump(d, open(jp, "w"), indent=1)
print("saved %s  (retract_adhesion=%s)" % (os.path.basename(dst), d["retract_adhesion"]))
