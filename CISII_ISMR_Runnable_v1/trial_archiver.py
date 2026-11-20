#!/usr/bin/env python3
"""Background archiver for the formal trial campaign.

Polls trial_plunge_stills/ (top level) and, for every trial JSON, maintains
campaign_<date>/trialNN_<stamp>/ with:
  - the trial JSON and every still (mic/post/manual/measured; re-synced when new ones appear)
  - precision_log / correspondence_log rows inside the trial's time window
  - every orient_trace_last.csv version written during the trial
  - config snapshot: relevant env of the running controller + md5 of code and calibration files
  - the session video (hard link) and the trial's offset into it
and one summary row per trial in campaign_<date>/campaign_trials.csv.
A trial is finalized once its JSON has post_retract_timestamp; it keeps being re-synced so a
later save_manual_still.py / measure_trial_holes.py result is picked up too.

Usage: nohup python3 trial_archiver.py > campaign_archiver.log 2>&1 &
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
STILLS = os.path.join(HERE, "trial_plunge_stills")
CAMPAIGN = os.path.join(os.path.dirname(HERE), "campaign_" + os.environ.get("CAMPAIGN_DATE", "20261012"))
MIC = "/home/eyebrp/Matthew/microscope_autonomy"
PREC = os.path.join(HERE, "precision_log.csv")
CORR = os.path.join(HERE, "correspondence_log.csv")
ORIENT_TRACE = os.path.join(HERE, "orient_trace_last.csv")
REC_DIR = os.path.join(HERE, "recordings")
HASHED = [os.path.join(HERE, "home_to_rcm_insert_v2_precision.py"), os.path.join(HERE, "run_main.sh"),
          os.path.join(MIC, "mic_perception.py"), os.path.join(MIC, "mic_extrinsic_calib.npz"),
          os.path.join(MIC, "ring_to_entry_offset.npz"), os.path.join(HERE, "origin_correction.npz"),
          os.path.join(HERE, "tip_offset_rcmoffset.npz"), os.path.join(HERE, "phantom_normal.npz"),
          os.path.join(os.path.dirname(HERE), "03_results_npz", "handeye_T.npz")]
ENV_KEYS = ("MIC_", "TIP_", "PIERCE", "RETRACT", "MARK_", "NORMAL_", "HOVER", "PLUNGE", "TUNNEL",
            "INSERT", "ORIENT", "JOG", "BIPLANAR", "REC_", "PHANTOM", "MOVE_", "POS_TOL", "TRIAL_")
STAGES = [("APPROACH", "err_mm"), ("ORIENT", "err_deg"), ("TUNNEL", "err_mm"),
          ("REORIENT(internal,perp)", "err_deg"), ("PLUNGE", "err_mm"), ("RETRACT_AXIAL", "err_mm"),
          ("RETRACT_CLEAR", "err_mm"), ("RETRACT", "err_mm")]
SUMMARY = os.path.join(CAMPAIGN, "campaign_trials.csv")
_env_cache = {}


def iso_ts(s):
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, fmt).timestamp()


def md5(p):
    if not os.path.isfile(p):
        return "missing"
    h = hashlib.md5()
    with open(os.path.realpath(p), "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def controller_env():
    try:
        pids = subprocess.check_output(["pgrep", "-f", "home_to_rcm_insert_v2_precision.py"]).split()
    except subprocess.CalledProcessError:
        return None
    for pid in pids:
        try:
            raw = open("/proc/%s/environ" % pid.decode(), "rb").read().split(b"\0")
        except OSError:
            continue
        env = {}
        for kv in raw:
            k, _, v = kv.decode(errors="replace").partition("=")
            if k.startswith(ENV_KEYS):
                env[k] = v
        st = subprocess.check_output(["ps", "-o", "lstart=", "-p", pid.decode()]).decode().strip()
        return {"pid": int(pid), "started": st, "env": env}
    return None


def log_rows(path, t0, t1):
    if not os.path.isfile(path):
        return None, []
    with open(path) as f:
        lines = f.read().splitlines()
    header, out = lines[0], []
    for ln in lines[1:]:
        ts = ln.split(",", 1)[0]
        try:
            t = iso_ts(ts)
        except ValueError:
            continue
        if t0 <= t <= t1:
            out.append(ln)
    return header, out


def video_for(t_start):
    best = None
    for p in glob.glob(os.path.join(REC_DIR, "home_rcm_v2_*.mp4")):
        stem = os.path.basename(p)[len("home_rcm_v2_"):-4]
        if "_" not in stem or not stem.replace("_", "").isdigit():
            continue
        t = datetime.strptime(stem, "%Y%m%d_%H%M%S").timestamp()
        if t <= t_start and (best is None or t > best[1]):
            best = (p, t)
    return best


def archive(jp, prev_end, next_plunge=None):
    d = json.load(open(jp))
    stem = os.path.basename(jp)[:-5]
    out = os.path.join(CAMPAIGN, stem)
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(STILLS, stem + "*")):
        dst = os.path.join(out, os.path.basename(f))
        if not os.path.exists(dst) or os.path.getmtime(f) > os.path.getmtime(dst):
            shutil.copy2(f, dst)
    t_plunge = d["timestamp"]
    t_end = d.get("post_retract_timestamp")
    if not t_end:
        t_end = min(time.time(), t_plunge + 300, (next_plunge - 120) if next_plunge else float("inf"))
    t0 = max(prev_end, t_plunge - 600) if prev_end else t_plunge - 600
    if d.get("analysis_start_ts"):
        t0 = max(t0, d["analysis_start_ts"])
    for xp in glob.glob(os.path.join(STILLS, "*", "trial*.json")):
        xe = json.load(open(xp)).get("post_retract_timestamp")
        if xe and t0 < xe < t_plunge:
            t0 = xe
    rows = {}
    for path, name in ((PREC, "precision_rows.csv"), (CORR, "correspondence_rows.csv")):
        header, lines = log_rows(path, t0, t_end + 2)
        if header:
            with open(os.path.join(out, name), "w") as f:
                f.write("\n".join([header] + lines) + "\n")
            rows[name] = lines
    for f in glob.glob(os.path.join(CAMPAIGN, "_orient_traces", "orient_trace_*.csv")):
        t = float(os.path.basename(f)[len("orient_trace_"):-4])
        if t0 <= t <= t_end + 2:
            dst = os.path.join(out, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
    if stem not in _env_cache:
        _env_cache[stem] = {"controller": controller_env(), "md5": {os.path.basename(p): md5(p) for p in HASHED},
                            "snapshot_time": time.time()}
        cfg_path = os.path.join(out, "config_snapshot.json")
        if not os.path.exists(cfg_path):
            json.dump(_env_cache[stem], open(cfg_path, "w"), indent=1)
    cfg = json.load(open(os.path.join(out, "config_snapshot.json")))

    stage = {}
    t_first = None
    for ln in rows.get("precision_rows.csv", []):
        f = next(csv.reader([ln]))
        if len(f) < 6 or f[2] != "reached":
            continue
        t = iso_ts(f[0])
        name = "ORIENT" if f[1].startswith("ORIENT(") else f[1]
        if t <= t_plunge + 1:
            if name == "APPROACH":
                t_first = t - float(f[5] or 0)
            stage[name] = (f[3], f[4], f[5])
        elif name.startswith("RETRACT") and name not in stage:
            stage[name] = (f[3], f[4], f[5])
    vid = video_for(t_first or t_plunge - 120)
    if vid:
        vdst = os.path.join(CAMPAIGN, "videos", os.path.basename(vid[0]))
        os.makedirs(os.path.dirname(vdst), exist_ok=True)
        if not os.path.exists(vdst):
            os.link(vid[0], vdst)
    lc = d.get("last_correspondence") or {}
    pose = lc.get("_pose") or {}
    row = {"trial": d.get("trial"), "folder": stem, "plunge_time": d.get("timestamp_iso"),
           "target_angle_deg": d.get("mic_rcm_angle_deg"), "insert_angle_deg": d.get("insert_angle_deg"),
           "finalized": int("post_retract_timestamp" in d),
           "retract_adhesion": int(bool(d.get("retract_adhesion"))), "manual_still": d.get("manual_still", ""),
           "n_mic_stills": len(d.get("stills") or []), "n_post_stills": len(d.get("post_retract_stills") or []),
           "cycle_s": "%.1f" % (t_end - t_first) if (t_first and d.get("post_retract_timestamp")) else "",
           "leica_vs_d405_corr_mm": lc.get("disc_corr_mm", ""), "leica_vs_d405_raw_mm": lc.get("disc_raw_mm", ""),
           "phantom_tilt_deg": pose.get("tilt_deg", ""), "phantom_yaw_diff_deg": pose.get("yaw_diff_deg", ""),
           "rcm_z_src": pose.get("rcm_z_src", ""), "rcm_target_base_mm": d.get("rcm_target_base_mm"),
           "video": os.path.basename(vid[0]) if vid else "",
           "video_offset_plunge_s": "%.1f" % (t_plunge - vid[1]) if vid else "",
           "code_md5": cfg["md5"].get("home_to_rcm_insert_v2_precision.py"),
           "pierce_speed": ((cfg.get("controller") or {}).get("env") or {}).get("PIERCE_SPEED", ""),
           "tip_model": ((cfg.get("controller") or {}).get("env") or {}).get("TIP_MODEL", "")}
    for name, kind in STAGES:
        e_deg, e_mm, el = stage.get(name, ("", "", ""))
        key = name.split("(")[0].lower()
        row["%s_err_%s" % (key, "deg" if kind == "err_deg" else "mm")] = e_deg if kind == "err_deg" else e_mm
        row["%s_s" % key] = el
    return row, t_end


def main():
    os.makedirs(os.path.join(CAMPAIGN, "_orient_traces"), exist_ok=True)
    last_trace_mtime = None
    print("archiving into %s" % CAMPAIGN, flush=True)
    while True:
        try:
            if os.path.isfile(ORIENT_TRACE):
                m = os.path.getmtime(ORIENT_TRACE)
                if m != last_trace_mtime:
                    shutil.copy2(ORIENT_TRACE, os.path.join(CAMPAIGN, "_orient_traces", "orient_trace_%.3f.csv" % m))
                    last_trace_mtime = m
            rows, prev_end = [], None
            jps = sorted(glob.glob(os.path.join(STILLS, "trial*.json")), key=lambda p: json.load(open(p))["timestamp"])
            plunges = [json.load(open(p))["timestamp"] for p in jps]
            for i, jp in enumerate(jps):
                row, prev_end = archive(jp, prev_end, plunges[i + 1] if i + 1 < len(jps) else None)
                rows.append(row)
            tmp = SUMMARY + ".tmp"
            with open(tmp, "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            os.replace(tmp, SUMMARY)
            for f in ("hole_measurements.csv",):
                src = os.path.join(STILLS, f)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(CAMPAIGN, f))
        except Exception as e:
            print("%s archiver error (will retry): %r" % (time.strftime("%H:%M:%S"), e), flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
