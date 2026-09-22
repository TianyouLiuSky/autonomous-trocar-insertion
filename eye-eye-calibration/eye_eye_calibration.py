#!/usr/bin/env python3
"""Capture, solve and validate Leica/D405 eye-eye registration. See README.md."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from calibration_core import (BOARD_SPEC, BoardDetector, Intrinsics, fit_registration,
                              pose_difference, transform, validate_pair_timing, write_json)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError("Cannot save {}".format(path))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ros_imports():
    try:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, CameraInfo
        import message_filters
    except ImportError as error:
        raise RuntimeError("Live capture requires ROS1 Python3, cv_bridge and message_filters; "
                           "source your ROS workspace. Offline solve needs no ROS.") from error
    return rospy, CvBridge, Image, CameraInfo, message_filters


def preview(image, text, pose=None, intrinsics=None):
    out = image.copy()
    if pose:
        for point in pose["image_points_px"]:
            cv2.circle(out, tuple(np.rint(point).astype(int)), 4, (0, 255, 0), 1)
        T = np.asarray(pose["T_camera_from_board"])
        cv2.drawFrameAxes(out, intrinsics.K, intrinsics.dist,
                         cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], 0.012)
    scale = min(1.0, 900.0 / out.shape[1])
    out = cv2.resize(out, None, fx=scale, fy=scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, text[:105], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
    cv2.putText(out, "C: fit capture | V: validation capture | Q: quit", (8, 47),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return out


def capture(args):
    intr = {c: Intrinsics.load(getattr(args, c + "_intrinsics")) for c in ("leica", "d405")}
    rospy, CvBridge, Image, _, message_filters = ros_imports()
    root = Path(args.session).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    detectors = {c: BoardDetector(getattr(args, c + "_source"), intr[c]) for c in intr}
    manifest = {"schema_version": 1, "created_utc": utc_now(), "opencv_version": cv2.__version__,
                "board": BOARD_SPEC, "notes": args.notes,
                "intrinsics": {c: intr[c].metadata for c in intr},
                "topics": {c: getattr(args, c + "_topic") for c in intr},
                "sources": {c: getattr(args, c + "_source") for c in intr},
                "max_pair_delta_s": args.max_pair_delta, "max_age_s": args.max_age,
                "max_rms_px": args.max_rms, "min_ambiguity_gap_px": args.min_ambiguity_gap,
                "timing": "ROS header timestamps; receipt age also checked; stationary target required"}
    write_json(root / "session.json", manifest)
    rospy.init_node("eye_eye_calibration", anonymous=True)
    bridge, lock = CvBridge(), threading.Lock()
    latest = [None]

    def received(leica, d405):
        try:
            stamps = [m.header.stamp.to_sec() for m in (leica, d405)]
            now = rospy.Time.now().to_sec()
            validate_pair_timing(stamps, [now-s for s in stamps], args.max_pair_delta, args.max_age)
            images = {c: bridge.imgmsg_to_cv2(m, desired_encoding="bgr8").copy()
                      for c, m in zip(("leica", "d405"), (leica, d405))}
            pair = {"images": images, "timestamps": stamps, "received": time.monotonic(),
                    "frame_ids": {c: m.header.frame_id for c, m in zip(("leica", "d405"), (leica, d405))}}
            with lock:
                latest[0] = pair
        except Exception as error:
            # Callback exceptions must not terminate the ROS subscription thread.
            rospy.logwarn_throttle(3, str(error))

    subscribers = [message_filters.Subscriber(getattr(args, c + "_topic"), Image,
                                             queue_size=5, buff_size=2**25) for c in intr]
    synchronizer = message_filters.ApproximateTimeSynchronizer(subscribers, 15, args.max_pair_delta)
    synchronizer.registerCallback(received)
    used = set()
    captured_frame_ids = None
    count = {"fit": 0, "validation": 0}
    last_processed, poses, failures = None, {}, {}
    print("Keep BOTH cameras fixed. Hold the target still before each C or V capture.")
    print("C = fit, V = held-out validation, Q = quit. Saving raw images to", root)
    try:
        while not rospy.is_shutdown():
            with lock:
                pair = latest[0]
            if pair is None:
                canvas = np.zeros((120, 900, 3), np.uint8)
                cv2.putText(canvas, "Waiting for timestamp-paired images; Q quits. Check topics/clocks.",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                cv2.imshow("Eye-eye status", canvas)
            else:
                if pair is not last_processed:
                    poses, failures = {}, {}
                    for c in intr:
                        try:
                            poses[c] = detectors[c].detect(pair["images"][c], args.max_rms, args.min_ambiguity_gap)
                        except (ValueError, cv2.error) as error:
                            failures[c] = str(error)
                    last_processed = pair
                stale = time.monotonic() - pair["received"] > args.max_age
                for c in intr:
                    label = "STALE - no capture" if stale else failures.get(c)
                    if label is None:
                        label = "{} RMS {:.3f}px | fit {} validation {}".format(
                            c, poses[c]["reprojection_rms_px"], count["fit"], count["validation"])
                    cv2.imshow(c, preview(pair["images"][c], label, poses.get(c), intr[c]))
            key = cv2.waitKey(20) & 0xff
            if key in (ord("q"), 27):
                break
            if key not in (ord("c"), ord("v")):
                continue
            try:
                if pair is None or failures or len(poses) != 2:
                    raise ValueError("Both cameras need a valid current pose: {}".format(failures))
                age = time.monotonic() - pair["received"]
                now = rospy.Time.now().to_sec()
                validate_pair_timing(pair["timestamps"], [now-s for s in pair["timestamps"]],
                                     args.max_pair_delta, args.max_age, used)
                if age > args.max_age:
                    raise ValueError("Pair is stale")
                if captured_frame_ids is not None and pair["frame_ids"] != captured_frame_ids:
                    raise ValueError("Image frame IDs changed during this session; restart with a fixed setup")
                role = "fit" if key == ord("c") else "validation"
                number = sum(count.values())
                folder = root / "{:04d}_{}".format(number, role)
                folder.mkdir()
                sample = {"capture": folder.name, "role": role, "created_utc": utc_now(),
                          "timestamps_s": pair["timestamps"], "frame_ids": pair["frame_ids"],
                          "images": {}, "sha256": {}}
                for c in intr:
                    path = folder / (c + ".png")
                    save_png(path, pair["images"][c])
                    sample["images"][c] = c + ".png"
                    sample["sha256"][c] = digest(path)
                    sample[c] = poses[c]
                write_json(folder / "sample.json", sample)
                captured_frame_ids = pair["frame_ids"]
                used.update(enumerate(pair["timestamps"]))
                count[role] += 1
                print("Saved", folder.name, count, flush=True)
            except ValueError as error:
                print("Capture rejected:", error, flush=True)
    finally:
        for subscriber in subscribers:
            subscriber.unregister()
        cv2.destroyAllWindows()
    print("Captured", count, "Run the solve command to produce the registration JSON.")


def solve(args):
    root = Path(args.session).expanduser().resolve()
    manifest = json.loads((root / "session.json").read_text())
    if manifest["board"] != BOARD_SPEC:
        raise ValueError("Session board model differs from this code")
    intr = {c: Intrinsics.from_dict(manifest["intrinsics"][c]) for c in ("leica", "d405")}
    detectors = {c: BoardDetector(manifest["sources"][c], intr[c]) for c in intr}
    samples, used, frame_ids = [], set(), None
    for path in sorted(root.glob("*/sample.json")):
        sample = json.loads(path.read_text())
        if sample["role"] not in ("fit", "validation"):
            raise ValueError("Unknown sample role: {}".format(path))
        if frame_ids is not None and sample["frame_ids"] != frame_ids:
            raise ValueError("Camera frame IDs changed within session")
        frame_ids = sample["frame_ids"]
        validate_pair_timing(sample["timestamps_s"], [0, 0], manifest["max_pair_delta_s"], 1, used)
        used.update(enumerate(sample["timestamps_s"]))
        for c in intr:
            image_path = path.parent / sample["images"][c]
            if digest(image_path) != sample["sha256"][c]:
                raise ValueError("Image changed since capture: {}".format(image_path))
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Cannot read {}".format(image_path))
            # Re-estimate from original pixels, not cached display poses or smoothed values.
            try:
                sample[c] = detectors[c].detect(image, manifest["max_rms_px"], manifest["min_ambiguity_gap_px"])
            except (ValueError, cv2.error) as error:
                raise ValueError("{}: {}: {}".format(path.parent.name, c, error)) from error
        samples.append(sample)
    result = fit_registration(samples, intr)
    result.update(created_utc=utc_now(), session=str(root), opencv_version=cv2.__version__,
                  intrinsics=manifest["intrinsics"], notes=manifest["notes"],
                  ros_image_frame_ids=frame_ids,
                  pose_quality={k: manifest[k] for k in ("max_rms_px", "min_ambiguity_gap_px")})
    write_json(args.output, result)
    print(json.dumps(result["metrics"]["validation"]["summary"], indent=2))
    for warning in result["warnings"]:
        print("CHECK:", warning)
    print("Saved", args.output, "(physical accuracy still requires an independent check)")


def export_camera_info(args):
    rospy, _, Image, CameraInfo, _ = ros_imports()
    rospy.init_node("eye_eye_intrinsics_export", anonymous=True)
    # The repository's custom publisher sends CameraInfo only with image subscribers.
    subscription = rospy.Subscriber(args.image_topic, Image, lambda msg: None, queue_size=1)
    try:
        info = rospy.wait_for_message(args.topic, CameraInfo, timeout=15)
    finally:
        subscription.unregister()
    if (info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset
            or info.roi.width or info.roi.height):
        raise ValueError("Cropped/binned CameraInfo needs explicit adjustment; use full raw stream")
    data = {"K": np.asarray(info.K).reshape(3, 3).tolist(), "dist": list(info.D),
            "image_size": {"width": info.width, "height": info.height},
            "distortion_model": info.distortion_model, "ros_frame_id": info.header.frame_id,
            "source_topic": args.topic, "image_topic": args.image_topic, "created_utc": utc_now()}
    Intrinsics.from_dict(data)
    write_json(args.output, data)
    print("Saved raw-image intrinsics:", args.output)


def record_intrinsics(args):
    rospy, CvBridge, Image, _, _ = ros_imports()
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    rospy.init_node("eye_eye_intrinsic_images", anonymous=True)
    bridge, lock, latest = CvBridge(), threading.Lock(), [None]
    detector = BoardDetector("charuco")
    def receive(msg):
        try:
            image = bridge.imgmsg_to_cv2(msg, "bgr8").copy()
            with lock:
                latest[0] = (image, msg.header.stamp.to_sec(), time.monotonic())
        except Exception as error:
            rospy.logwarn_throttle(3, str(error))
    sub = rospy.Subscriber(args.topic, Image, receive, queue_size=1, buff_size=2**25)
    records, used = [], set()
    try:
        while not rospy.is_shutdown():
            with lock:
                item = latest[0]
            if item is not None:
                out = preview(item[0], "{} images | C saves intrinsic image; Q quits".format(len(records)))
                cv2.rectangle(out, (0, 31), (out.shape[1], 60), (0, 0, 0), -1)
                cv2.imshow("Intrinsic images", out)
            key = cv2.waitKey(30) & 0xff
            if key in (ord("q"), 27):
                break
            if key == ord("c") and item is not None:
                try:
                    image, stamp, received = item
                    validate_pair_timing([stamp, stamp], [rospy.Time.now().to_sec()-stamp]*2, .05, 1)
                    if time.monotonic() - received > 1 or stamp in used:
                        raise ValueError("Stale or duplicate image")
                    _, _, ids = detector.correspondences(image)
                    name = "{:04d}.png".format(len(records))
                    save_png(root / name, image)
                    records.append({"image": name, "timestamp_s": stamp, "corners": len(ids)})
                    used.add(stamp)
                    print("Saved", name, "corners", len(ids), flush=True)
                except (ValueError, cv2.error) as error:
                    print("Not saved:", error, flush=True)
    finally:
        sub.unregister()
        cv2.destroyAllWindows()
        write_json(root / "capture.json", {"topic": args.topic, "notes": args.notes,
                                          "board": BOARD_SPEC, "images": records})


def calibrate_intrinsics(args):
    detector = BoardDetector("charuco")
    objects, images, names, size = [], [], [], None
    for path in sorted(Path(args.images).expanduser().glob("*.png")):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Unreadable image: {}".format(path))
        current = (image.shape[1], image.shape[0])
        if size is not None and size != current:
            raise ValueError("Mixed image resolutions in intrinsic calibration")
        size = current
        try:
            obj, img, _ = detector.correspondences(image)
            if len(obj) < 12 or np.linalg.matrix_rank(obj - obj.mean(axis=0), tol=1e-8) != 2:
                raise ValueError("Need at least 12 non-collinear corners")
        except (ValueError, cv2.error) as error:
            print("Skipping", path.name, error)
            continue
        objects.append(obj.astype(np.float32))
        images.append(img.astype(np.float32))
        names.append(path.name)
    if len(objects) < 20:
        raise ValueError("Need at least 20 usable intrinsic images; found {}".format(len(objects)))
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, images, size, None, None)
    errors = []
    for name, obj, img, rv, tv in zip(names, objects, images, rvecs, tvecs):
        projected = cv2.projectPoints(obj, rv, tv, K, dist)[0].reshape(-1, 2)
        errors.append({"image": name, "rms_px": float(np.sqrt(np.mean(np.sum((projected-img)**2, axis=1))))})
    data = {"cam_id": args.camera, "K": K.tolist(), "dist": dist.reshape(-1).tolist(),
            "image_size": {"width": size[0], "height": size[1]}, "distortion_model": "plumb_bob",
            "rms_px": float(rms), "per_image": errors, "board": BOARD_SPEC,
            "notes": args.notes, "created_utc": utc_now(), "opencv_version": cv2.__version__,
            "physical_accuracy_validated": False}
    poses = [transform(rv, tv) for rv, tv in zip(rvecs, tvecs)]
    span = max(pose_difference(a, b)[1] for a in poses for b in poses)
    coverage = np.ptp(np.concatenate(images), axis=0) / np.asarray(size)
    data.update(orientation_span_deg=span, image_coverage_fraction_xy=coverage.tolist(), warnings=[])
    if span < 10:
        data["warnings"].append("Less than 10 degrees pose diversity; intrinsic fit may be poorly constrained.")
    if rms > 1.5:
        data["warnings"].append("Intrinsic fitting RMS exceeds 1.5 px; inspect images and settings.")
    if min(coverage) < .3:
        data["warnings"].append("Limited image coverage; distortion away from the target may be poorly constrained.")
    Intrinsics.from_dict(data)
    write_json(args.output, data)
    print("Saved", args.output, "RMS {:.3f}px; inspect per-image errors and validate on new images".format(rms))
    for warning in data["warnings"]:
        print("CHECK:", warning)


def positive(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    live = sub.add_parser("capture", help="Capture paired raw images and board poses (ROS1)")
    for camera, topic, source in [("leica", "/decklink/camera/image_raw", "charuco"),
                                  ("d405", "/d405/color/image_raw", "april")]:
        live.add_argument("--" + camera + "-intrinsics", required=True)
        live.add_argument("--" + camera + "-topic", default=topic)
        live.add_argument("--" + camera + "-source", choices=["charuco", "april"], default=source)
    live.add_argument("--session", required=True, help="New output directory")
    live.add_argument("--notes", required=True, help="Optical settings, camera IDs and mounting details")
    live.add_argument("--max-pair-delta", type=positive, default=.05, help="ROS timestamp tolerance, seconds")
    live.add_argument("--max-age", type=positive, default=1., help="Maximum image age, seconds")
    live.add_argument("--max-rms", type=positive, default=1.5, help="Per-camera reprojection RMS limit, pixels")
    live.add_argument("--min-ambiguity-gap", type=positive, default=.05, help="Planar ambiguity gap, pixels")
    live.set_defaults(func=capture)
    fit = sub.add_parser("solve", help="Reprocess raw captures; fit and validate without ROS")
    fit.add_argument("--session", required=True)
    fit.add_argument("--output", required=True)
    fit.set_defaults(func=solve)
    info = sub.add_parser("export-camera-info", help="Save raw-stream CameraInfo as intrinsic JSON")
    info.add_argument("--topic", default="/d405/color/camera_info")
    info.add_argument("--image-topic", default="/d405/color/image_raw")
    info.add_argument("--output", required=True)
    info.set_defaults(func=export_camera_info)
    record = sub.add_parser("record-intrinsics", help="Save raw ChArUco images from one ROS camera")
    record.add_argument("--topic", default="/decklink/camera/image_raw")
    record.add_argument("--output", required=True, help="New directory")
    record.add_argument("--notes", required=True)
    record.set_defaults(func=record_intrinsics)
    intrinsic = sub.add_parser("calibrate-intrinsics", help="Fit intrinsics using the existing printed composite board")
    intrinsic.add_argument("--images", required=True)
    intrinsic.add_argument("--output", required=True)
    intrinsic.add_argument("--camera", default="leica")
    intrinsic.add_argument("--notes", required=True)
    intrinsic.set_defaults(func=calibrate_intrinsics)
    return p


def main():
    args = parser().parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError, cv2.error, KeyError) as error:
        print("ERROR:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
