"""Numerical helpers for fixed-camera hand-eye calibration (A Y = X B)."""

import numpy as np


def rotation_matrix_from_quaternion_xyzw(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-15:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])


def rotation_matrix_from_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def rotation_vector_from_matrix(rotation):
    rotation = project_to_rotation(rotation)
    value = (np.trace(rotation) - 1.0) / 2.0
    angle = float(np.arccos(np.clip(value, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3)
    if abs(np.sin(angle)) > 1e-8:
        axis = np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]) / (2.0 * np.sin(angle))
    else:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        axis /= np.linalg.norm(axis)
    return axis * angle


def project_to_rotation(matrix):
    """Return the nearest proper rotation matrix in Frobenius norm."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_angle_deg(rotation):
    value = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def rotation_axis_angle(rotation):
    rotvec = rotation_vector_from_matrix(rotation)
    angle_rad = float(np.linalg.norm(rotvec))
    if angle_rad < 1e-12:
        return np.zeros(3), 0.0
    return rotvec / angle_rad, float(np.degrees(angle_rad))


def transforms_from_samples(robot_poses, board_rvecs, board_tvecs):
    robot_rotations = np.array([
        rotation_matrix_from_quaternion_xyzw(pose["q"])
        for pose in robot_poses
    ])
    robot_translations = np.array([
        np.asarray(pose["t"], dtype=float).reshape(3)
        for pose in robot_poses
    ])
    board_rotations = np.array([
        rotation_matrix_from_rotvec(np.asarray(rvec).reshape(3))
        for rvec in board_rvecs
    ])
    board_translations = np.array([
        np.asarray(tvec, dtype=float).reshape(3)
        for tvec in board_tvecs
    ])
    return (
        robot_rotations,
        robot_translations,
        board_rotations,
        board_translations,
    )


def estimate_camera_rotation_from_translations(
        robot_translations, board_translations):
    """Estimate R_cam2base from a pure-translation validation sweep."""
    robot_points = np.asarray(robot_translations, dtype=float)
    camera_points = np.asarray(board_translations, dtype=float)
    if len(robot_points) < 3:
        raise ValueError("At least three spatial samples are required")

    robot_centered = robot_points - np.mean(robot_points, axis=0)
    camera_centered = camera_points - np.mean(camera_points, axis=0)
    cross_covariance = camera_centered.T @ robot_centered
    u, singular_values, vt = np.linalg.svd(cross_covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    offset = np.mean(robot_points, axis=0) - (
        rotation @ np.mean(camera_points, axis=0)
    )
    predicted = (rotation @ camera_points.T).T + offset
    residuals = predicted - robot_points
    return {
        "rotation": rotation,
        "offset": offset,
        "predicted": predicted,
        "residuals": residuals,
        "singular_values": singular_values,
    }


def estimate_camera_rotation_from_orientation_arc(
        robot_rotations, robot_translations, board_translations,
        initial_rotations):
    """Fit camera rotation from board-center motion during pure rotation.

    The board center is offset from the end-effector origin, so an orientation
    sweep traces an arc even when the reported robot XYZ remains fixed. This
    fits that arc without requiring a physical pivot constraint.
    """
    from scipy.optimize import least_squares

    robot_rotations = np.asarray(robot_rotations, dtype=float)
    robot_translations = np.asarray(robot_translations, dtype=float)
    board_translations = np.asarray(board_translations, dtype=float)
    if len(robot_rotations) < 4:
        raise ValueError("At least four orientation samples are required")

    def evaluate(rotation_vector):
        camera_rotation = rotation_matrix_from_rotvec(rotation_vector)
        fit = evaluate_orientation_arc_with_fixed_rotation(
            robot_rotations,
            robot_translations,
            board_translations,
            camera_rotation,
        )
        return fit["residuals"], fit["translation_fit"]

    starts = [
        rotation_vector_from_matrix(rotation)
        for rotation in initial_rotations
    ]
    starts.append(np.zeros(3))
    candidates = []
    for start_index, start in enumerate(starts):
        solution = least_squares(
            lambda value: evaluate(value)[0].reshape(-1),
            start,
            loss="linear",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        residuals, translation_fit = evaluate(solution.x)
        candidates.append({
            "start_index": start_index,
            "solution": solution,
            "rotation": rotation_matrix_from_rotvec(solution.x),
            "residuals": residuals,
            "translation_fit": translation_fit,
            "cost": float(np.sum(residuals ** 2)),
        })

    best = min(candidates, key=lambda candidate: candidate["cost"])
    best_mean_residual = float(np.mean(
        np.linalg.norm(best["residuals"], axis=1)))
    candidate_summaries = []
    for candidate in candidates:
        residual_norms = np.linalg.norm(candidate["residuals"], axis=1)
        candidate_summaries.append({
            "start_index": candidate["start_index"],
            "success": bool(candidate["solution"].success),
            "mean_residual_mm": float(np.mean(residual_norms)) * 1000.0,
            "max_residual_mm": float(np.max(residual_norms)) * 1000.0,
            "rotation_from_best_deg": rotation_angle_deg(
                candidate["rotation"] @ best["rotation"].T),
        })
    solved_rotations = [
        candidate["rotation"] for candidate in candidates
        if candidate["solution"].success
    ]
    rotation_spread_deg = 0.0
    for i in range(len(solved_rotations)):
        for j in range(i + 1, len(solved_rotations)):
            rotation_spread_deg = max(
                rotation_spread_deg,
                rotation_angle_deg(
                    solved_rotations[i] @ solved_rotations[j].T),
            )
    competitive = []
    for candidate in candidates:
        mean_residual = float(np.mean(
            np.linalg.norm(candidate["residuals"], axis=1)))
        if (
                candidate["solution"].success
                and mean_residual <= max(
                    best_mean_residual * 1.10,
                    best_mean_residual + 0.01e-3,
                )):
            competitive.append(candidate)
    competitive_rotation_spread_deg = 0.0
    for i in range(len(competitive)):
        for j in range(i + 1, len(competitive)):
            competitive_rotation_spread_deg = max(
                competitive_rotation_spread_deg,
                rotation_angle_deg(
                    competitive[i]["rotation"]
                    @ competitive[j]["rotation"].T),
            )

    jacobian = best["solution"].jac
    jacobian_condition = (
        float(np.linalg.cond(jacobian))
        if jacobian.size else float("inf")
    )
    translation_fit = best["translation_fit"]
    board_offset = translation_fit["board_translation"]
    camera_translation = translation_fit["camera_translation"]
    robot_board_centers = (
        robot_translations
        + np.einsum("nij,j->ni", robot_rotations, board_offset)
    )
    camera_board_centers = (
        np.einsum(
            "ij,nj->ni", best["rotation"], board_translations)
        + camera_translation
    )
    return {
        "rotation": best["rotation"],
        "board_offset": board_offset,
        "camera_translation": camera_translation,
        "robot_board_centers": robot_board_centers,
        "camera_board_centers": camera_board_centers,
        "residuals": best["residuals"],
        "jacobian_condition": jacobian_condition,
        "translation_condition_number":
            translation_fit["condition_number"],
        "translation_rank": translation_fit["rank"],
        "successful_starts": sum(
            candidate["solution"].success for candidate in candidates),
        "requested_starts": len(candidates),
        "rotation_spread_deg": rotation_spread_deg,
        "competitive_starts": len(competitive),
        "competitive_rotation_spread_deg":
            competitive_rotation_spread_deg,
        "candidate_summaries": candidate_summaries,
    }


def evaluate_orientation_arc_with_fixed_rotation(
        robot_rotations, robot_translations, board_translations,
        camera_rotation):
    """Evaluate orientation-sweep translations at a fixed camera rotation."""
    robot_rotations = np.asarray(robot_rotations, dtype=float)
    robot_translations = np.asarray(robot_translations, dtype=float)
    board_translations = np.asarray(board_translations, dtype=float)
    camera_rotation = np.asarray(camera_rotation, dtype=float).reshape(3, 3)
    translation_fit = solve_translations(
        robot_rotations,
        robot_translations,
        board_translations,
        camera_rotation,
    )
    board_offset = translation_fit["board_translation"]
    camera_translation = translation_fit["camera_translation"]
    robot_board_centers = (
        robot_translations
        + np.einsum("nij,j->ni", robot_rotations, board_offset)
    )
    camera_board_centers = (
        np.einsum("ij,nj->ni", camera_rotation, board_translations)
        + camera_translation
    )
    return {
        "camera_rotation": camera_rotation,
        "board_offset": board_offset,
        "camera_translation": camera_translation,
        "robot_board_centers": robot_board_centers,
        "camera_board_centers": camera_board_centers,
        "residuals": camera_board_centers - robot_board_centers,
        "translation_fit": translation_fit,
    }


def estimate_rotations_from_relative_motion(
        robot_rotations, board_rotations):
    """Estimate R_board2gripper and R_cam2base from rotational motion."""
    robot_rotations = np.asarray(robot_rotations, dtype=float)
    board_rotations = np.asarray(board_rotations, dtype=float)
    if len(robot_rotations) < 3:
        raise ValueError("At least three orientation samples are required")

    equations = []
    pair_rows = []
    for i in range(len(robot_rotations)):
        for j in range(i + 1, len(robot_rotations)):
            relative_robot = robot_rotations[i].T @ robot_rotations[j]
            relative_board = board_rotations[i].T @ board_rotations[j]
            equations.append(
                np.kron(np.eye(3), relative_robot)
                - np.kron(relative_board.T, np.eye(3))
            )
            pair_rows.append({
                "sample_i": i + 1,
                "sample_j": j + 1,
                "robot_relative_deg": rotation_angle_deg(relative_robot),
                "camera_relative_deg": rotation_angle_deg(relative_board),
            })

    equation_matrix = np.vstack(equations)
    _, singular_values, vt = np.linalg.svd(equation_matrix)
    raw = vt[-1].reshape((3, 3), order="F")

    candidates = []
    for sign in (1.0, -1.0):
        board_to_gripper = project_to_rotation(sign * raw)
        equation_errors = []
        for row in pair_rows:
            i = row["sample_i"] - 1
            j = row["sample_j"] - 1
            relative_robot = robot_rotations[i].T @ robot_rotations[j]
            relative_board = board_rotations[i].T @ board_rotations[j]
            equation_errors.append(np.linalg.norm(
                relative_robot @ board_to_gripper
                - board_to_gripper @ relative_board
            ))
        candidates.append((
            float(np.mean(equation_errors)),
            board_to_gripper,
        ))

    equation_error, board_to_gripper = min(candidates, key=lambda item: item[0])
    camera_to_base = project_to_rotation(np.sum([
        robot_rotation @ board_to_gripper @ board_rotation.T
        for robot_rotation, board_rotation in zip(
            robot_rotations, board_rotations)
    ], axis=0))

    sample_errors_deg = []
    for robot_rotation, board_rotation in zip(
            robot_rotations, board_rotations):
        left = robot_rotation @ board_to_gripper
        right = camera_to_base @ board_rotation
        sample_errors_deg.append(rotation_angle_deg(right @ left.T))

    for row in pair_rows:
        row["relative_angle_difference_deg"] = (
            row["camera_relative_deg"] - row["robot_relative_deg"]
        )

    return {
        "camera_rotation": camera_to_base,
        "board_rotation": board_to_gripper,
        "sample_errors_deg": np.asarray(sample_errors_deg),
        "pair_rows": pair_rows,
        "equation_error": equation_error,
        "singular_values": singular_values,
    }


def solve_translations(
        robot_rotations, robot_translations, board_translations,
        camera_rotation):
    """Solve t_board2gripper and t_cam2base with rotations held fixed."""
    blocks = []
    values = []
    for robot_rotation, robot_translation, board_translation in zip(
            robot_rotations, robot_translations, board_translations):
        blocks.append(np.hstack([robot_rotation, -np.eye(3)]))
        values.append(
            camera_rotation @ board_translation - robot_translation
        )
    system = np.vstack(blocks)
    target = np.concatenate(values)
    solution, _, rank, singular_values = np.linalg.lstsq(
        system, target, rcond=None)
    return {
        "board_translation": solution[:3],
        "camera_translation": solution[3:],
        "rank": int(rank),
        "singular_values": singular_values,
        "condition_number": float(np.linalg.cond(system)),
    }


def solution_vector(
        board_rotation, board_translation,
        camera_rotation, camera_translation):
    return np.concatenate([
        rotation_vector_from_matrix(board_rotation),
        np.asarray(board_translation).reshape(3),
        rotation_vector_from_matrix(camera_rotation),
        np.asarray(camera_translation).reshape(3),
    ])
