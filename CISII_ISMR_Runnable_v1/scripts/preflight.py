#!/usr/bin/env python3
"""Offline current-session preflight. No ROS/hardware imports or motion."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_session import startup_check,registered_problems
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['calibration','insertion'],default='calibration');a=p.parse_args()
try:
 startup_check();problems=registered_problems() if a.phase=='insertion' else []
except (ValueError,OSError,KeyError,SystemExit) as e:problems=[str(e)]
print(json.dumps({'phase':a.phase,'status':'BLOCKED' if problems else 'OFFLINE_PASS_HARDWARE_UNVERIFIED','problems':problems},indent=2))
sys.exit(bool(problems))
