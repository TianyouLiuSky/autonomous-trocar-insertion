"""Synthetic geometry and image tests. These do not certify physical accuracy."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from calibration_core import (BOARD_SPEC, BoardDetector, Intrinsics, TAG_CELLS,
                              charuco_points, fit_registration, make_charuco_board,
                              mean_transform, pose_difference, project,
                              solve_board_pose, tag_points, transform,
                              validate_pair_timing, write_json)
from eye_eye_calibration import digest


def camera(f=1500, width=1280, height=960):
    return Intrinsics.from_dict({"K": [[f, 0, width/2], [0, f, height/2], [0, 0, 1]],
                                 "dist": [0, 0, 0, 0, 0],
                                 "image_size": {"width": width, "height": height}})


def record(obj, T, intr):
    return solve_board_pose(obj, project(obj, T, intr), intr)


def test_common_origin_and_tag_layout():
    np.testing.assert_allclose(charuco_points([0, 24]), [[.034, .034, 0], [.050, .050, 0]], atol=1e-8)
    np.testing.assert_allclose(tag_points(0), [[.002, .002, 0], [.026, .002, 0],
                                             [.026, .026, 0], [.002, .026, 0]])
    np.testing.assert_allclose(tag_points(7).mean(0), [.042, .070, 0])


@pytest.mark.parametrize("objects", [charuco_points(range(25)),
                                    np.concatenate([tag_points(i) for i in range(8)])])
def test_pose_recovery_for_both_sources(objects):
    intr = camera()
    truth = transform([.3, -.2, .1], [-.04, -.04, .35])
    solved = record(objects, truth, intr)
    np.testing.assert_allclose(solved["T_camera_from_board"], truth, atol=1e-6)
    assert solved["reprojection_rms_px"] < 1e-5


def test_noisy_pose_and_reprojection_rejection():
    intr = camera()
    obj = charuco_points(range(25))
    truth = transform([.35, -.25, .1], [-.04, -.04, .3])
    pixels = project(obj, truth, intr)
    noisy = pixels + np.random.default_rng(8).normal(0, .03, pixels.shape)
    result = solve_board_pose(obj, noisy, intr)
    assert pose_difference(truth, np.asarray(result["T_camera_from_board"]))[0] < .5
    pixels[4] += 25
    with pytest.raises(ValueError, match="reprojection"):
        solve_board_pose(obj, pixels, intr, min_ambiguity_gap_px=0)


def test_distortion_is_used_and_ambiguous_pose_rejected():
    intr = camera()
    intr.dist = np.array([.2, -.05, .002, -.003, .01])
    obj = np.concatenate([tag_points(i) for i in range(8)])
    truth = transform([.35, -.25, .1], [-.02, -.01, .25])
    pixels = project(obj, truth, intr)
    result = solve_board_pose(obj, pixels, intr)
    np.testing.assert_allclose(result["T_camera_from_board"], truth, atol=1e-6)
    with pytest.raises(ValueError, match="Ambiguous"):
        solve_board_pose(obj, pixels, intr, min_ambiguity_gap_px=1e6)


def test_degenerate_correspondences_rejected():
    with pytest.raises(ValueError, match="non-collinear"):
        solve_board_pose(np.array([[i, 0, 0] for i in range(6)]), np.zeros((6, 2)), camera())


def test_intrinsics_fail_closed():
    intr = camera()
    with pytest.raises(ValueError, match="no automatic scaling"):
        intr.check_image(np.zeros((480, 640, 3), np.uint8))
    bad = dict(intr.metadata, distortion_model="equidistant")
    with pytest.raises(ValueError, match="Unsupported"):
        Intrinsics.from_dict(bad)
    with pytest.raises(KeyError):
        Intrinsics.from_dict({"K": intr.K.tolist(), "dist": intr.dist.tolist()})
    bad = dict(intr.metadata, K=[[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        Intrinsics.from_dict(bad)


def test_timestamps_stale_and_duplicate():
    validate_pair_timing([100, 100.03], [.1, .07], .05, 1)
    for stamps, ages, used in [([0, 0], [0, 0], set()),
                               ([100, 100.1], [0, 0], set()),
                               ([100, 100.01], [2, 2], set()),
                               ([100, 100.01], [-5, -5], set()),
                               ([100, 100.01], [0, 0], {(0, 100)})]:
        with pytest.raises(ValueError):
            validate_pair_timing(stamps, ages, .05, 1, used)


def synthetic_samples():
    intr = {"leica": camera(4000), "d405": camera(1500)}
    registration = transform([.1, -.15, .04], [.02, -.015, .02])
    samples = []
    for i in range(15):
        L = transform([.15 + i*.018, -.25 + i*.025, .05], [-.035 + i*.001, -.045, .35+i*.001])
        D = registration @ L
        samples.append({"capture": str(i), "role": "fit" if i < 10 else "validation",
                        "leica": record(charuco_points(range(25)), L, intr["leica"]),
                        "d405": record(np.concatenate([tag_points(j) for j in range(8)]), D, intr["d405"])})
    return samples, intr, registration


def test_transform_direction_and_held_out_validation():
    samples, intr, truth = synthetic_samples()
    result = fit_registration(samples, intr)
    np.testing.assert_allclose(result["T_d405_color_from_leica"], truth, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result["T_leica_from_d405_color"]) @ truth, np.eye(4), atol=1e-6)
    assert result["metrics"]["validation"]["summary"]["board_corner_rms_mm"]["rms"] < .001
    samples[-1]["d405"]["T_camera_from_board"][0][3] += .01
    changed = fit_registration(samples, intr)
    np.testing.assert_allclose(changed["T_d405_color_from_leica"], truth, atol=1e-6)
    assert changed["metrics"]["validation"]["summary"]["board_origin_error_mm"]["max"] > 9.9
    assert changed["warnings"]
    assert not changed["physical_accuracy_validated"]


def test_requires_held_out_captures():
    samples, intr, _ = synthetic_samples()
    with pytest.raises(ValueError, match="held-out"):
        fit_registration(samples[:10], intr)


def test_rotation_mean_handles_wraparound():
    mean = mean_transform([transform([0, 0, np.deg2rad(a)], [0, 0, 0]) for a in (179, -179)])
    assert pose_difference(mean, transform([0, 0, np.pi], [0, 0, 0]))[1] < 1e-5
    assert np.linalg.det(mean[:3, :3]) == pytest.approx(1)


def board_bitmap():
    # Same nominal geometry and IDs as the existing generator, at 10 px/mm.
    bitmap = np.full((840, 840), 255, np.uint8)
    bitmap[300:540, 300:540] = make_charuco_board().generateImage((240, 240))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id, (row, col) in TAG_CELLS.items():
        patch = cv2.aruco.generateImageMarker(dictionary, tag_id, 240)
        y, x = row*280+20, col*280+20
        bitmap[y:y+240, x:x+240] = patch
    return bitmap


def rendered_image(bitmap, T, intr):
    dst = project([[0, 0, 0], [.084, 0, 0], [.084, .084, 0], [0, .084, 0]], T, intr).astype(np.float32)
    size = bitmap.shape[0]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [size, 0], [size, size], [0, size]]), dst)
    image = cv2.warpPerspective(bitmap, H, (intr.width, intr.height), borderValue=255)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_image_detection_uses_common_board_frame():
    intr = camera(2500)
    T = transform([.25, -.2, .06], [-.045, -.042, .30])
    image = rendered_image(board_bitmap(), T, intr)
    for source in ("charuco", "april"):
        result = BoardDetector(source, intr).detect(image)
        mm, deg = pose_difference(T, np.asarray(result["T_camera_from_board"]))
        assert mm < 1.0
        assert deg < 1.0


def test_offline_cli_reprocesses_saved_pixels(tmp_path):
    intr = {"leica": camera(3000), "d405": camera(2200)}
    registration = transform([.05, -.08, .02], [.015, -.01, .03])
    session = tmp_path / "session"
    session.mkdir()
    write_json(session / "session.json", {"board": BOARD_SPEC, "notes": "synthetic",
        "intrinsics": {c: i.metadata for c, i in intr.items()},
        "sources": {"leica": "charuco", "d405": "april"},
        "max_pair_delta_s": .05, "max_rms_px": 1.5, "min_ambiguity_gap_px": .05})
    for i in range(15):
        role = "fit" if i < 10 else "validation"
        folder = session / str(i)
        folder.mkdir()
        L = transform([.2+i*.01, -.3+i*.01, .04], [-.04, -.04, .35+i*.001])
        sample = {"capture": str(i), "role": role, "timestamps_s": [100+i, 100+i+.01],
                  "frame_ids": {"leica": "leica_optical", "d405": "d405_color_optical"},
                  "images": {}, "sha256": {}}
        for c, T in [("leica", L), ("d405", registration @ L)]:
            path = folder / (c + ".png")
            cv2.imwrite(str(path), rendered_image(board_bitmap(), T, intr[c]))
            sample["images"][c], sample["sha256"][c] = path.name, digest(path)
        write_json(folder / "sample.json", sample)
    output = tmp_path / "result.json"
    command = [sys.executable, str(Path(__file__).with_name("eye_eye_calibration.py")),
               "solve", "--session", str(session), "--output", str(output)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    mm, deg = pose_difference(registration, np.asarray(result["T_d405_color_from_leica"]))
    assert mm < 2.0
    assert deg < 1.0
    assert result["metrics"]["validation"]["count"] == 5
    assert subprocess.run(command, capture_output=True).returncode != 0  # cannot overwrite
    (session / "0" / "leica.png").write_bytes(b"changed")
    run = subprocess.run(command[:-1] + [str(tmp_path / "other.json")], capture_output=True, text=True)
    assert run.returncode != 0 and "changed since capture" in run.stderr


def test_intrinsic_calibration_cli_on_rendered_board(tmp_path):
    intr = camera(2500)
    rng = np.random.default_rng(52)
    images = tmp_path / "images"
    images.mkdir()
    for i in range(30):
        rv = [rng.uniform(-.45, .45), rng.uniform(-.45, .45), rng.uniform(-.15, .15)]
        tv = [rng.uniform(-.06, -.025), rng.uniform(-.06, -.025), rng.uniform(.28, .4)]
        cv2.imwrite(str(images / (str(i) + ".png")), rendered_image(board_bitmap(), transform(rv, tv), intr))
    output = tmp_path / "intrinsics.json"
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("eye_eye_calibration.py")),
                          "calibrate-intrinsics", "--images", str(images), "--output", str(output),
                          "--notes", "synthetic"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    estimated = Intrinsics.load(output)
    assert estimated.metadata["rms_px"] < .5
    assert estimated.K[0, 0] == pytest.approx(intr.K[0, 0], rel=.05)
    assert estimated.K[1, 1] == pytest.approx(intr.K[1, 1], rel=.05)
