#!/usr/bin/env python3
"""
D405 Exposure Sweep Test

Purpose:
- Test several manual exposure values on an Intel RealSense D405.
- Keep gain fixed.
- Save ALL captured images inside this script's own output folder.
- Measure:
    1) ROI saturation ratio (gray >= 250)
    2) ring detection success rate
    3) ring pixel count
    4) fitted ring radius
    5) ring residual ratio
- Save per-frame metrics and per-exposure summaries as CSV files.

Default exposure sweep:
    11000, 12000, 13000, 14000

Default ROI and ring detector parameters match auto_target_vision.py.

Controls:
    q / ESC : stop early

Example:
    python3 d405_exposure_sweep.py

Optional:
    python3 d405_exposure_sweep.py --exposures 10000 11000 12000 13000 \
        --gain 16 --frames 60 --warmup 30
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ---------------------------------------------------------------------
# Defaults copied from the existing auto_target_vision.py
# ---------------------------------------------------------------------
A_STAR_THRESH = 145
MIN_AREA_PX = 1200
MAX_AREA_PX = 20000
MAX_RING_RESIDUAL_RATIO = 0.22
MIN_RING_RADIUS_PX = 20

ROI_X0 = 100
ROI_X1 = 540
ROI_Y0 = 200
ROI_Y1 = 480

SATURATION_THRESHOLD = 250


def kasa_circle_fit(xs, ys):
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol

    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None

    return float(cx), float(cy), float(np.sqrt(r2))


def ring_residual_ratio(xs, ys, cx, cy, r):
    d = np.hypot(xs - cx, ys - cy)
    residual = np.sqrt(np.mean((d - r) ** 2))
    return float(residual / max(r, 1e-6))


def detect_ring(color_bgr):
    """
    Same core logic as auto_target_vision.py.

    Returns a dict with:
        detected
        cx, cy, radius_px
        n_px
        residual_ratio
        fail_reason
    """
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    a_full = lab[:, :, 1]

    h, w = a_full.shape
    x0, x1 = max(0, ROI_X0), min(w, ROI_X1)
    y0, y1 = max(0, ROI_Y0), min(h, ROI_Y1)

    mask = np.zeros((h, w), dtype=np.uint8)
    sub = (a_full[y0:y1, x0:x1] > A_STAR_THRESH).astype(np.uint8) * 255
    mask[y0:y1, x0:x1] = sub

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    n_red_px = int(np.count_nonzero(mask))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if n_labels <= 1:
        return {
            "detected": False,
            "cx": np.nan,
            "cy": np.nan,
            "radius_px": np.nan,
            "n_px": 0,
            "residual_ratio": np.nan,
            "fail_reason": f"no red pixels in ROI (a*>{A_STAR_THRESH})"
        }

    candidates = []
    best_area = 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        best_area = max(best_area, area)

        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        candidates.append((area, i))

    candidates.sort(reverse=True)

    if not candidates:
        return {
            "detected": False,
            "cx": np.nan,
            "cy": np.nan,
            "radius_px": np.nan,
            "n_px": 0,
            "residual_ratio": np.nan,
            "fail_reason": (
                f"{n_red_px} red px, largest component area={best_area} "
                f"(need {MIN_AREA_PX}-{MAX_AREA_PX})"
            )
        }

    worst_reason = "no candidate passed shape check"

    for area, i in candidates:
        ys, xs = np.nonzero(labels == i)

        fit = kasa_circle_fit(xs, ys)
        if fit is None:
            worst_reason = "circle fit degenerate"
            continue

        cx, cy, r = fit

        if r < MIN_RING_RADIUS_PX:
            worst_reason = (
                f"fitted radius={r:.1f}px < MIN_RING_RADIUS_PX={MIN_RING_RADIUS_PX}"
            )
            continue

        residual = ring_residual_ratio(xs, ys, cx, cy, r)

        if residual > MAX_RING_RESIDUAL_RATIO:
            worst_reason = (
                f"residual={residual:.3f} > "
                f"MAX_RING_RESIDUAL_RATIO={MAX_RING_RESIDUAL_RATIO}"
            )
            continue

        return {
            "detected": True,
            "cx": cx,
            "cy": cy,
            "radius_px": r,
            "n_px": int(len(xs)),
            "residual_ratio": residual,
            "fail_reason": ""
        }

    return {
        "detected": False,
        "cx": np.nan,
        "cy": np.nan,
        "radius_px": np.nan,
        "n_px": 0,
        "residual_ratio": np.nan,
        "fail_reason": worst_reason
    }


def saturation_ratio_percent(color_bgr):
    """
    Saturation is measured only inside the same ROI used by the ring detector.
    A pixel is counted as saturated if grayscale >= SATURATION_THRESHOLD.
    """
    h, w = color_bgr.shape[:2]

    x0, x1 = max(0, ROI_X0), min(w, ROI_X1)
    y0, y1 = max(0, ROI_Y0), min(h, ROI_Y1)

    roi = color_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    ratio = np.mean(gray >= SATURATION_THRESHOLD) * 100.0
    return float(ratio)


def find_exposure_sensor(device):
    """
    D405 commonly exposes exposure/gain controls through the Stereo Module.
    Select a sensor that supports manual exposure.
    """
    candidates = []

    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except Exception:
            name = "Unknown Sensor"

        supports_exposure = sensor.supports(rs.option.exposure)
        supports_gain = sensor.supports(rs.option.gain)

        print(
            f"Sensor: {name} | exposure={supports_exposure} | gain={supports_gain}"
        )

        if supports_exposure:
            candidates.append((sensor, name))

    if not candidates:
        raise RuntimeError("No RealSense sensor supporting rs.option.exposure was found.")

    # Prefer the Stereo Module on D405.
    for sensor, name in candidates:
        if "Stereo" in name:
            print(f"Using exposure sensor: {name}")
            return sensor

    sensor, name = candidates[0]
    print(f"Using exposure sensor: {name}")
    return sensor


def clamp_option(sensor, option, value):
    rng = sensor.get_option_range(option)
    value = float(np.clip(value, rng.min, rng.max))

    # Respect option step where practical.
    if rng.step > 0:
        value = rng.min + round((value - rng.min) / rng.step) * rng.step
        value = float(np.clip(value, rng.min, rng.max))

    return value


def set_manual_camera_settings(sensor, exposure, gain):
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 0)

    exposure = clamp_option(sensor, rs.option.exposure, exposure)
    sensor.set_option(rs.option.exposure, exposure)

    actual_gain = np.nan
    if sensor.supports(rs.option.gain):
        gain = clamp_option(sensor, rs.option.gain, gain)
        sensor.set_option(rs.option.gain, gain)
        actual_gain = sensor.get_option(rs.option.gain)

    actual_exposure = sensor.get_option(rs.option.exposure)

    return float(actual_exposure), float(actual_gain)


def annotate_frame(frame, exposure, gain, sat_pct, ring):
    view = frame.copy()

    cv2.rectangle(
        view,
        (ROI_X0, ROI_Y0),
        (ROI_X1, ROI_Y1),
        (100, 100, 100),
        1
    )

    if ring["detected"]:
        cx = int(round(ring["cx"]))
        cy = int(round(ring["cy"]))
        r = int(round(ring["radius_px"]))

        cv2.circle(view, (cx, cy), r, (0, 255, 80), 2)
        cv2.circle(view, (cx, cy), 4, (0, 255, 255), -1)

        ring_text = (
            f"RING OK  r={ring['radius_px']:.1f}px  "
            f"n={ring['n_px']}  resid={ring['residual_ratio']:.3f}"
        )
        ring_color = (0, 255, 80)
    else:
        ring_text = "RING NOT DETECTED"
        ring_color = (0, 0, 255)

    lines = [
        f"Exposure={exposure:.0f}  Gain={gain:.0f}",
        f"ROI saturation >= {SATURATION_THRESHOLD}: {sat_pct:.3f}%",
        ring_text
    ]

    y = 24
    for text in lines:
        cv2.putText(
            view,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ring_color if "RING" in text else (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        y += 24

    return view


def safe_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def safe_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else np.nan


def main():
    parser = argparse.ArgumentParser(description="Intel RealSense D405 exposure sweep.")
    parser.add_argument(
        "--exposures",
        nargs="+",
        type=float,
        default=[11000, 12000, 13000, 14000],
        help="Manual exposure values to test."
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=16,
        help="Fixed gain used for every exposure."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of measured frames per exposure."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Frames discarded after changing exposure."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30
    )
    args = parser.parse_args()

    # Everything created by this script stays under this folder.
    script_dir = Path(__file__).resolve().parent
    root_output = script_dir / "d405_exposure_test"

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_output / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAll output will be saved under:\n  {run_dir}\n")

    pipeline = rs.pipeline()
    config = rs.config()

    # We only need the color stream for this exposure test.
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps
    )

    profile = pipeline.start(config)

    try:
        device = profile.get_device()
        sensor = find_exposure_sensor(device)

        if sensor.supports(rs.option.exposure):
            r = sensor.get_option_range(rs.option.exposure)
            print(
                f"Exposure range: min={r.min}, max={r.max}, "
                f"step={r.step}, default={r.default}"
            )

        if sensor.supports(rs.option.gain):
            r = sensor.get_option_range(rs.option.gain)
            print(
                f"Gain range: min={r.min}, max={r.max}, "
                f"step={r.step}, default={r.default}"
            )

        # Disable auto exposure before all measurements.
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            print("Auto exposure: OFF")

        frame_csv_path = run_dir / "frame_metrics.csv"
        summary_csv_path = run_dir / "summary.csv"

        frame_rows = []
        summary_rows = []

        stop_requested = False

        for requested_exposure in args.exposures:
            actual_exposure, actual_gain = set_manual_camera_settings(
                sensor,
                requested_exposure,
                args.gain
            )

            exp_dir = run_dir / f"exp_{int(round(actual_exposure))}"
            raw_dir = exp_dir / "raw"
            annotated_dir = exp_dir / "annotated"

            raw_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n=== Exposure {actual_exposure:.0f}, "
                f"Gain {actual_gain:.0f} ==="
            )
            print(f"Warm-up: {args.warmup} frames")

            for _ in range(args.warmup):
                pipeline.wait_for_frames()

            saturation_values = []
            detected_values = []
            n_px_values = []
            radius_values = []
            residual_values = []

            for frame_idx in range(args.frames):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    print("Warning: no color frame")
                    continue

                color = np.asanyarray(color_frame.get_data())

                sat_pct = saturation_ratio_percent(color)
                ring = detect_ring(color)

                saturation_values.append(sat_pct)
                detected_values.append(1 if ring["detected"] else 0)

                if ring["detected"]:
                    n_px_values.append(ring["n_px"])
                    radius_values.append(ring["radius_px"])
                    residual_values.append(ring["residual_ratio"])

                # Save raw image.
                raw_path = raw_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(raw_path), color)

                # Save annotated image.
                annotated = annotate_frame(
                    color,
                    actual_exposure,
                    actual_gain,
                    sat_pct,
                    ring
                )
                annotated_path = annotated_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(annotated_path), annotated)

                frame_rows.append({
                    "requested_exposure": requested_exposure,
                    "actual_exposure": actual_exposure,
                    "gain": actual_gain,
                    "frame": frame_idx,
                    "saturation_pct": sat_pct,
                    "ring_detected": int(ring["detected"]),
                    "ring_n_px": ring["n_px"] if ring["detected"] else "",
                    "ring_radius_px": ring["radius_px"] if ring["detected"] else "",
                    "ring_residual_ratio": (
                        ring["residual_ratio"] if ring["detected"] else ""
                    ),
                    "fail_reason": ring["fail_reason"]
                })

                cv2.imshow(
                    "D405 Exposure Sweep - press q or ESC to stop",
                    annotated
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_requested = True
                    break

            detection_rate = (
                100.0 * sum(detected_values) / len(detected_values)
                if detected_values else 0.0
            )

            summary = {
                "requested_exposure": requested_exposure,
                "actual_exposure": actual_exposure,
                "gain": actual_gain,
                "frames_collected": len(saturation_values),
                "mean_saturation_pct": safe_mean(saturation_values),
                "std_saturation_pct": safe_std(saturation_values),
                "ring_detection_success_pct": detection_rate,
                "mean_ring_n_px": safe_mean(n_px_values),
                "std_ring_n_px": safe_std(n_px_values),
                "mean_ring_radius_px": safe_mean(radius_values),
                "std_ring_radius_px": safe_std(radius_values),
                "mean_ring_residual_ratio": safe_mean(residual_values),
                "std_ring_residual_ratio": safe_std(residual_values)
            }

            summary_rows.append(summary)

            print(
                f"Mean saturation: {summary['mean_saturation_pct']:.3f}%"
            )
            print(
                f"Ring detection success: "
                f"{summary['ring_detection_success_pct']:.1f}%"
            )
            print(
                f"Mean radius: {summary['mean_ring_radius_px']:.3f}px "
                f"(std {summary['std_ring_radius_px']:.3f})"
            )
            print(
                f"Mean residual: "
                f"{summary['mean_ring_residual_ratio']:.4f}"
            )

            if stop_requested:
                break

        # Save per-frame CSV.
        if frame_rows:
            with open(frame_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(frame_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(frame_rows)

        # Save summary CSV.
        if summary_rows:
            with open(summary_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(summary_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(summary_rows)

        print("\nFinished.")
        print(f"Images and CSV files are only under:\n  {run_dir}")
        print(f"Per-frame metrics:\n  {frame_csv_path}")
        print(f"Summary:\n  {summary_csv_path}")

        print("\nExposure summary:")
        for row in summary_rows:
            print(
                f"  exp={row['actual_exposure']:.0f}  "
                f"sat={row['mean_saturation_pct']:.3f}%  "
                f"detect={row['ring_detection_success_pct']:.1f}%  "
                f"r={row['mean_ring_radius_px']:.2f}px  "
                f"r_std={row['std_ring_radius_px']:.3f}  "
                f"resid={row['mean_ring_residual_ratio']:.4f}"
            )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
