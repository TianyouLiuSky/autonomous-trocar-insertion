#!/usr/bin/env python3
"""click_calib_points_per_touch.py -- click ONE point on EACH of the
per-touch microscope stills saved automatically by '1'-'8' in
home_to_rcm_insert_v2_precision.py (as of 2026-10-01, pt.2), instead of
clicking all N points on a single shared still with click_calib_points.py.

WHY THIS EXISTS (2026-10-01): the original protocol (click_calib_points.py)
asks the operator to touch N points, then separately recognize/remember
where each of those N points was on ONE later still and click them in the
same order. In practice this recognition step is unreliable -- confirmed
in person: points touched without a deliberate, rememberable visual
landmark cannot be reliably re-identified on a plain photo. A wrong click
paired with a real touched 3D point looks EXACTLY like a correct point to
every downstream script (same magnitude of silent-corruption risk as a bad
pivot calibration) -- there is no residual check that can catch "clicked
the wrong pixel for the right reason".

Fix: log_mic_calib_touch() now saves, at the exact moment of each touch, a
copy of the then-freshest live microscope frame into
MIC_CALIB_TOUCH_STILLS_DIR as "<label>.jpg". The tool tip is still resting
on the just-touched point in that exact frame, so it visibly marks its own
pixel location -- nothing needs to be remembered. This script just walks
through those per-label stills in order and asks for ONE click per still
(where the tip visibly contacts the surface), instead of N clicks on one
image.

Output format is IDENTICAL to click_calib_points.py's (a dict of
label -> {"u":, "v":}, plus "_"-prefixed metadata keys that
mic_extrinsic_calib.py's do_fit() already ignores) so mic_extrinsic_calib.py
--clicks needs no changes to consume it.

Usage:
    python3 click_calib_points_per_touch.py \\
        --stills-dir mic_calib_touch_stills \\
        --touches mic_calib_touches.jsonl \\
        --out mic_calib_touch_stills/combined.calib_px.json

Controls (per still):
    click       -- place the point (click again to move it before saving)
    s           -- save this still's point and advance to the next label
    u           -- clear the current click (click again to replace it)
    q / ESC     -- quit without saving anything further (keeps labels
                   already saved to --out if this is the 2nd+ run)
"""
import argparse
import json
import os
import sys

import cv2


def label_sort_key(label):
    # "calib_pt_10" must sort after "calib_pt_9", not between "calib_pt_1"
    # and "calib_pt_2" -- extract the trailing integer if present.
    digits = "".join(ch for ch in label if ch.isdigit())
    return (int(digits) if digits else 0, label)


def load_touch_labels(touches_path):
    labels = []
    seen = set()
    with open(touches_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lb = rec["label"]
            if lb not in seen:
                seen.add(lb)
                labels.append(lb)
    return sorted(labels, key=label_sort_key)


def click_one(path, label):
    img = cv2.imread(path)
    if img is None:
        print("  SKIP %s: could not read %s" % (label, path))
        return None
    win = "click_calib_points_per_touch -- %s (LMB=place u=clear s=save q=quit)" % label
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    pt = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[0] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    result = "quit"
    while True:
        view = img.copy()
        if pt[0] is not None:
            cv2.drawMarker(view, pt[0], (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
        cv2.rectangle(view, (0, 0), (view.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(view, "%s -- click the tip-contact point  (u=clear s=save q=quit)" % label,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):
            result = "quit"
            break
        if k == ord('u'):
            pt[0] = None
        if k == ord('s'):
            if pt[0] is None:
                print("    click a point first")
                continue
            result = "saved"
            break
    cv2.destroyWindow(win)
    if result == "saved":
        return {"u": pt[0][0], "v": pt[0][1]}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stills-dir", default="mic_calib_touch_stills")
    ap.add_argument("--touches", default="mic_calib_touches.jsonl")
    ap.add_argument("--out", default=None,
                     help="default: <stills-dir>/combined.calib_px.json")
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.stills_dir, "combined.calib_px.json")

    labels = load_touch_labels(args.touches)
    if not labels:
        raise SystemExit("no touch labels found in %s" % args.touches)

    clicks = {}
    if os.path.isfile(out_path):
        prev = json.load(open(out_path))
        clicks.update({k: v for k, v in prev.items() if not k.startswith("_")})
        print("Resuming -- %d point(s) already clicked and saved in %s" % (len(clicks), out_path))

    source_images = {}
    missing = []
    for label in labels:
        still_path = os.path.join(args.stills_dir, label + ".jpg")
        if label in clicks:
            print("  %-14s already clicked -- skipping (delete its entry in %s to redo)" %
                  (label, out_path))
            continue
        if not os.path.isfile(still_path):
            print("  %-14s NO per-touch still found at %s -- skipping (fall back to "
                  "click_calib_points.py for this one, or re-touch it)" % (label, still_path))
            missing.append(label)
            continue
        print("Click %s (%s) ..." % (label, still_path))
        pt = click_one(still_path, label)
        if pt is None:
            print("  quit requested -- stopping here, %d point(s) saved so far" % len(clicks))
            break
        clicks[label] = pt
        source_images[label] = os.path.abspath(still_path)
        print("  %s = (%d, %d)" % (label, pt["u"], pt["v"]))

    if not clicks:
        print("\nNothing to save.")
        return

    out = dict(clicks)
    out["_source_images"] = source_images
    out["_tool"] = "click_calib_points_per_touch.py"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved %d point(s) -> %s" % (len(clicks), out_path))
    if missing:
        print("Still missing (%d): %s" % (len(missing), ", ".join(missing)))
        print("-- mic_extrinsic_calib.py --fit can still run on the %d points you have "
              "(needs >=3, >=5 for LOO), or re-touch the missing ones first." % len(clicks))
    else:
        print("All touch labels have a click. Ready for:")
        print("  python3 mic_extrinsic_calib.py --fit --loo --touches %s --clicks %s "
              "--out mic_extrinsic_calib.npz" % (args.touches, out_path))


if __name__ == "__main__":
    main()
