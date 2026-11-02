#!/usr/bin/env python3
"""
D405 Exposure Sweep Test V2

Purpose:
- Fine-scan D405 manual exposure near the promising 12k-13k range.
- Keep gain fixed.
- Save ALL captured data inside this script's own output folder.
- Use two separate ROIs:
    1) EYE ROI for overexposure statistics
    2) RING ROI for red-ring detection
- Record ring geometry even when the final residual threshold rejects it.

Default exposure sweep:
    11500, 12000, 12500, 13000, 13500

Default gain:
    16

Default per-exposure collection:
    60 measured frames after 30 warm-up frames

Outputs:
    d405_exposure_test_v2/YYYYMMDD_HHMMSS/
        frame_metrics.csv
        summary.csv
        exp_11500/
            raw/
            annotated/
            mask/
        exp_12000/
            ...

Controls:
    q / ESC : stop early

Example:
    python3 d405_exposure_sweep_v2.py

Optional examples:
    python3 d405_exposure_sweep_v2.py --frames 100
    python3 d405_exposure_sweep_v2.py --exposures 12000 12500 13000
    python3 d405_exposure_sweep_v2.py --eye-roi 140 285 285 430
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ---------------------------------------------------------------------
# Defaults based on auto_target_vision.py
# ---------------------------------------------------------------------
A_STAR_THRESH = 145
MIN_AREA_PX = 1200
MAX_AREA_PX = 20000
MAX_RING_RESIDUAL_RATIO = 0.22
MIN_RING_RADIUS_PX = 20

# Original detector ROI from auto_target_vision.py
DEFAULT_RING_ROI = (100, 540, 200, 480)   # x0, x1, y0, y1

# Eye-only ROI estimated from current D405 images.
# This is used ONLY for saturation measurement.
DEFAULT_EYE_ROI = (140, 285, 285, 430)    # x0, x1, y0, y1

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


def clamp_roi(roi, width, height):
    x0, x1, y0, y1 = roi
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI after clipping: {(x0, x1, y0, y1)}")

    return x0, x1, y0, y1


def detect_ring_diagnostic(color_bgr, ring_roi):
    """
    Diagnostic ring detection.

    Important V2 behavior:
    The best geometrically plausible red component is fitted and its metrics
    are ALWAYS returned, even when residual_ratio > MAX_RING_RESIDUAL_RATIO.

    This lets exposure testing study:
        - component pixel count
        - fitted radius
        - residual ratio
    without losing the data simply because the production detector threshold
    (0.22) rejects the frame.

    Returns:
        result dict
        binary mask
    """
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    a_full = lab[:, :, 1]

    h, w = a_full.shape
    x0, x1, y0, y1 = clamp_roi(ring_roi, w, h)

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
            "candidate_found": False,
            "passes_production_threshold": False,
            "cx": np.nan,
            "cy": np.nan,
            "radius_px": np.nan,
            "n_px": 0,
            "component_area_px": 0,
            "residual_ratio": np.nan,
            "fail_reason": f"no red pixels in RING ROI (a*>{A_STAR_THRESH})"
        }, mask

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
            "candidate_found": False,
            "passes_production_threshold": False,
            "cx": np.nan,
            "cy": np.nan,
            "radius_px": np.nan,
            "n_px": 0,
            "component_area_px": best_area,
            "residual_ratio": np.nan,
            "fail_reason": (
                f"{n_red_px} red px, largest component area={best_area}, "
                f"required={MIN_AREA_PX}-{MAX_AREA_PX}"
            )
        }, mask

    # Use the largest valid-area component, as in the current detector.
    # If fitting fails, try the next candidate.
    for area, i in candidates:
        ys, xs = np.nonzero(labels == i)

        fit = kasa_circle_fit(xs, ys)
        if fit is None:
            continue

        cx, cy, r = fit
        residual = ring_residual_ratio(xs, ys, cx, cy, r)

        radius_ok = r >= MIN_RING_RADIUS_PX
        residual_ok = residual <= MAX_RING_RESIDUAL_RATIO
        passes = bool(radius_ok and residual_ok)

        fail_reason = ""
        if not radius_ok:
            fail_reason = (
                f"radius={r:.2f}px < MIN_RING_RADIUS_PX={MIN_RING_RADIUS_PX}"
            )
        elif not residual_ok:
            fail_reason = (
                f"residual={residual:.3f} > "
                f"MAX_RING_RESIDUAL_RATIO={MAX_RING_RESIDUAL_RATIO}"
            )

        return {
            "candidate_found": True,
            "passes_production_threshold": passes,
            "cx": cx,
            "cy": cy,
            "radius_px": r,
            "n_px": int(len(xs)),
            "component_area_px": int(area),
            "residual_ratio": residual,
            "fail_reason": fail_reason
        }, mask

    return {
        "candidate_found": False,
        "passes_production_threshold": False,
        "cx": np.nan,
        "cy": np.nan,
        "radius_px": np.nan,
        "n_px": 0,
        "component_area_px": 0,
        "residual_ratio": np.nan,
        "fail_reason": "all candidate circle fits degenerate"
    }, mask


def saturation_stats(color_bgr, eye_roi):
    """
    Calculate saturation only inside the eye ROI.

    Returns:
        saturation_pct
        mean_gray
        p95_gray
        p99_gray
        max_gray
    """
    h, w = color_bgr.shape[:2]
    x0, x1, y0, y1 = clamp_roi(eye_roi, w, h)

    roi = color_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    return {
        "saturation_pct": float(np.mean(gray >= SATURATION_THRESHOLD) * 100.0),
        "mean_gray": float(np.mean(gray)),
        "p95_gray": float(np.percentile(gray, 95)),
        "p99_gray": float(np.percentile(gray, 99)),
        "max_gray": int(np.max(gray))
    }


def find_exposure_sensor(device):
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
        raise RuntimeError(
            "No RealSense sensor supporting rs.option.exposure was found."
        )

    # Prefer D405 Stereo Module.
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


def annotate_frame(
    frame,
    eye_roi,
    ring_roi,
    exposure,
    gain,
    sat,
    ring
):
    view = frame.copy()

    ex0, ex1, ey0, ey1 = eye_roi
    rx0, rx1, ry0, ry1 = ring_roi

    # Eye saturation ROI: white rectangle
    cv2.rectangle(
        view,
        (ex0, ey0),
        (ex1, ey1),
        (255, 255, 255),
        2
    )

    # Ring detector ROI: gray rectangle
    cv2.rectangle(
        view,
        (rx0, ry0),
        (rx1, ry1),
        (100, 100, 100),
        1
    )

    if ring["candidate_found"]:
        cx = int(round(ring["cx"]))
        cy = int(round(ring["cy"]))
        radius = int(round(ring["radius_px"]))

        circle_color = (
            (0, 255, 80)
            if ring["passes_production_threshold"]
            else (0, 165, 255)
        )

        cv2.circle(view, (cx, cy), radius, circle_color, 2)
        cv2.circle(view, (cx, cy), 4, (0, 255, 255), -1)

        ring_text = (
            f"candidate r={ring['radius_px']:.1f}px "
            f"n={ring['n_px']} resid={ring['residual_ratio']:.3f} "
            f"pass={int(ring['passes_production_threshold'])}"
        )
    else:
        ring_text = "no valid ring candidate"

    overlay_lines = [
        f"Exposure={exposure:.0f}  Gain={gain:.0f}",
        (
            f"EYE sat>={SATURATION_THRESHOLD}: "
            f"{sat['saturation_pct']:.3f}%  "
            f"mean={sat['mean_gray']:.1f}  "
            f"p99={sat['p99_gray']:.1f}"
        ),
        ring_text
    ]

    # Readable black background strip
    cv2.rectangle(view, (0, 0), (639, 82), (0, 0, 0), -1)

    y = 22
    for text_line in overlay_lines:
        cv2.putText(
            view,
            text_line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
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


def safe_min(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else np.nan


def safe_max(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Intel RealSense D405 fine exposure sweep V2."
    )

    parser.add_argument(
        "--exposures",
        nargs="+",
        type=float,
        default=[11500, 12000, 12500, 13000, 13500],
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
        help="Measured frames per exposure."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Frames discarded after each exposure change."
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument(
        "--eye-roi",
        nargs=4,
        type=int,
        metavar=("X0", "X1", "Y0", "Y1"),
        default=list(DEFAULT_EYE_ROI),
        help="Eye-only ROI used for saturation statistics."
    )

    parser.add_argument(
        "--ring-roi",
        nargs=4,
        type=int,
        metavar=("X0", "X1", "Y0", "Y1"),
        default=list(DEFAULT_RING_ROI),
        help="ROI used for ring detection."
    )

    args = parser.parse_args()

    eye_roi = tuple(args.eye_roi)
    ring_roi = tuple(args.ring_roi)

    script_dir = Path(__file__).resolve().parent
    root_output = script_dir / "d405_exposure_test_v2"

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_output / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nD405 Exposure Sweep V2")
    print(f"All output:\n  {run_dir}")
    print(f"Eye ROI:  {eye_roi}")
    print(f"Ring ROI: {ring_roi}")
    print(
        "Important: production residual threshold is still "
        f"{MAX_RING_RESIDUAL_RATIO}, but V2 records candidate metrics even "
        "when that threshold rejects the frame.\n"
    )

    pipeline = rs.pipeline()
    config = rs.config()

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
            mask_dir = exp_dir / "mask"

            raw_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n=== Exposure {actual_exposure:.0f}, "
                f"Gain {actual_gain:.0f} ==="
            )
            print(f"Warm-up: {args.warmup} frames")

            for _ in range(args.warmup):
                pipeline.wait_for_frames()

            saturation_values = []
            mean_gray_values = []
            p95_values = []
            p99_values = []

            candidate_values = []
            production_pass_values = []

            n_px_values = []
            radius_values = []
            residual_values = []
            component_area_values = []

            for frame_idx in range(args.frames):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    print("Warning: no color frame")
                    continue

                color = np.asanyarray(color_frame.get_data())

                sat = saturation_stats(color, eye_roi)
                ring, mask = detect_ring_diagnostic(color, ring_roi)

                saturation_values.append(sat["saturation_pct"])
                mean_gray_values.append(sat["mean_gray"])
                p95_values.append(sat["p95_gray"])
                p99_values.append(sat["p99_gray"])

                candidate_values.append(1 if ring["candidate_found"] else 0)
                production_pass_values.append(
                    1 if ring["passes_production_threshold"] else 0
                )

                if ring["candidate_found"]:
                    n_px_values.append(ring["n_px"])
                    radius_values.append(ring["radius_px"])
                    residual_values.append(ring["residual_ratio"])
                    component_area_values.append(ring["component_area_px"])

                raw_path = raw_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(raw_path), color)

                mask_path = mask_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(mask_path), mask)

                annotated = annotate_frame(
                    color,
                    eye_roi,
                    ring_roi,
                    actual_exposure,
                    actual_gain,
                    sat,
                    ring
                )

                annotated_path = annotated_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(annotated_path), annotated)

                frame_rows.append({
                    "requested_exposure": requested_exposure,
                    "actual_exposure": actual_exposure,
                    "gain": actual_gain,
                    "frame": frame_idx,

                    "eye_saturation_pct": sat["saturation_pct"],
                    "eye_mean_gray": sat["mean_gray"],
                    "eye_p95_gray": sat["p95_gray"],
                    "eye_p99_gray": sat["p99_gray"],
                    "eye_max_gray": sat["max_gray"],

                    "ring_candidate_found": int(ring["candidate_found"]),
                    "passes_production_threshold": int(
                        ring["passes_production_threshold"]
                    ),

                    "ring_n_px": (
                        ring["n_px"] if ring["candidate_found"] else ""
                    ),
                    "ring_component_area_px": (
                        ring["component_area_px"]
                        if ring["candidate_found"]
                        else ""
                    ),
                    "ring_radius_px": (
                        ring["radius_px"]
                        if ring["candidate_found"]
                        else ""
                    ),
                    "ring_residual_ratio": (
                        ring["residual_ratio"]
                        if ring["candidate_found"]
                        else ""
                    ),
                    "fail_reason": ring["fail_reason"]
                })

                cv2.imshow(
                    "D405 Exposure Sweep V2 - q/ESC to stop",
                    annotated
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_requested = True
                    break

            n_frames = len(saturation_values)

            candidate_rate = (
                100.0 * sum(candidate_values) / len(candidate_values)
                if candidate_values
                else 0.0
            )

            production_pass_rate = (
                100.0 * sum(production_pass_values)
                / len(production_pass_values)
                if production_pass_values
                else 0.0
            )

            summary = {
                "requested_exposure": requested_exposure,
                "actual_exposure": actual_exposure,
                "gain": actual_gain,
                "frames_collected": n_frames,

                "mean_eye_saturation_pct": safe_mean(saturation_values),
                "std_eye_saturation_pct": safe_std(saturation_values),
                "mean_eye_gray": safe_mean(mean_gray_values),
                "mean_eye_p95_gray": safe_mean(p95_values),
                "mean_eye_p99_gray": safe_mean(p99_values),

                "ring_candidate_success_pct": candidate_rate,
                "production_threshold_pass_pct": production_pass_rate,

                "mean_ring_n_px": safe_mean(n_px_values),
                "std_ring_n_px": safe_std(n_px_values),

                "mean_ring_component_area_px": safe_mean(
                    component_area_values
                ),
                "std_ring_component_area_px": safe_std(
                    component_area_values
                ),

                "mean_ring_radius_px": safe_mean(radius_values),
                "std_ring_radius_px": safe_std(radius_values),

                "mean_ring_residual_ratio": safe_mean(residual_values),
                "std_ring_residual_ratio": safe_std(residual_values),
                "min_ring_residual_ratio": safe_min(residual_values),
                "max_ring_residual_ratio": safe_max(residual_values)
            }

            summary_rows.append(summary)

            print(
                f"Eye saturation: "
                f"{summary['mean_eye_saturation_pct']:.3f}% "
                f"(std {summary['std_eye_saturation_pct']:.3f})"
            )
            print(
                f"Eye mean gray: {summary['mean_eye_gray']:.2f}, "
                f"mean p99: {summary['mean_eye_p99_gray']:.2f}"
            )
            print(
                f"Ring candidate found: "
                f"{summary['ring_candidate_success_pct']:.1f}%"
            )
            print(
                f"Production threshold pass: "
                f"{summary['production_threshold_pass_pct']:.1f}%"
            )
            print(
                f"Ring radius: "
                f"{summary['mean_ring_radius_px']:.3f}px "
                f"(std {summary['std_ring_radius_px']:.3f})"
            )
            print(
                f"Ring residual: "
                f"{summary['mean_ring_residual_ratio']:.4f} "
                f"(std {summary['std_ring_residual_ratio']:.4f}, "
                f"range "
                f"{summary['min_ring_residual_ratio']:.4f}-"
                f"{summary['max_ring_residual_ratio']:.4f})"
            )

            if stop_requested:
                break

        if frame_rows:
            with open(frame_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(frame_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(frame_rows)

        if summary_rows:
            with open(summary_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(summary_rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(summary_rows)

        print("\nFinished.")
        print(f"All images and CSV files are under:\n  {run_dir}")
        print(f"Per-frame metrics:\n  {frame_csv_path}")
        print(f"Summary:\n  {summary_csv_path}")

        print("\nCompact exposure summary:")
        for row in summary_rows:
            print(
                f"  exp={row['actual_exposure']:.0f} | "
                f"eye_sat={row['mean_eye_saturation_pct']:.3f}% | "
                f"eye_p99={row['mean_eye_p99_gray']:.1f} | "
                f"candidate={row['ring_candidate_success_pct']:.1f}% | "
                f"prod_pass={row['production_threshold_pass_pct']:.1f}% | "
                f"r={row['mean_ring_radius_px']:.2f}px | "
                f"r_std={row['std_ring_radius_px']:.3f} | "
                f"resid={row['mean_ring_residual_ratio']:.4f}"
            )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
