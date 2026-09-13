#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure OpenCV helpers for robust stereo calibration.

This module intentionally has no ROS imports so its numerical and quality-gate
logic can be regression-tested on a development computer.  It remains Python 2
compatible for ROS Melodic on Jetson Nano.
"""
from __future__ import print_function

import math
import errno
import glob
import os
import tempfile

import cv2
import numpy as np


STEREO_SAMPLE_FORMAT_VERSION = 1


def checkerboard_object_points(columns, rows, square_size_m):
    """Return the metric object points for one fixed checkerboard view."""
    columns = int(columns)
    rows = int(rows)
    square_size_m = float(square_size_m)
    if columns < 2 or rows < 2 or square_size_m <= 0.0:
        raise ValueError("invalid checkerboard geometry")
    result = np.zeros((columns * rows, 1, 3), dtype=np.float32)
    grid = np.mgrid[0:columns, 0:rows].T.reshape((-1, 2))
    result[:, 0, :2] = grid.astype(np.float32) * square_size_m
    return result


def _sample_scalar(archive, name):
    value = np.asarray(archive[name])
    if value.size != 1 or value.dtype.kind == "O":
        raise ValueError("invalid scalar %s in persisted sample" % name)
    return value.reshape((-1,))[0]


def _normalise_sample_points(values, dimensions, expected_count, name):
    value = np.asarray(values)
    if value.dtype.kind == "O":
        raise ValueError("object arrays are forbidden in persisted samples")
    value = np.asarray(value, dtype=np.float32).reshape((-1, 1, dimensions))
    if value.shape[0] != int(expected_count):
        raise ValueError("invalid %s point count" % name)
    if not np.isfinite(value).all():
        raise ValueError("non-finite %s points" % name)
    return np.ascontiguousarray(value)


def _normalise_gray_image(values, image_size, name):
    value = np.asarray(values)
    expected_shape = (int(image_size[1]), int(image_size[0]))
    if value.dtype != np.uint8 or value.shape != expected_shape:
        raise ValueError(
            "%s gray image must be uint8 with shape %s" % (
                name, expected_shape))
    return np.ascontiguousarray(value)


def _normalise_sample_params(values):
    value = np.asarray(values, dtype=np.float64).reshape((-1,))
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("sample params must contain four finite values")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("sample params must be normalized to [0, 1]")
    return value


def save_stereo_sample(directory, sample_index, object_points, left_points,
                       right_points, left_gray, right_gray, sample_params,
                       image_size, columns, rows, square_size_m,
                       left_stamp_ns=0, right_stamp_ns=0,
                       left_sequence=0, right_sequence=0):
    """Atomically persist one accepted pair as an offline-recoverable NPZ.

    Temporary files never match ``sample_*.npz``.  A crash can therefore
    leave at most an ignored temporary file, never a half-written sample.
    """
    sample_index = int(sample_index)
    columns = int(columns)
    rows = int(rows)
    square_size_m = float(square_size_m)
    image_size = tuple(int(value) for value in image_size)
    if sample_index < 1:
        raise ValueError("sample_index must start at one")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain two positive values")
    expected_count = columns * rows
    expected_object = checkerboard_object_points(
        columns, rows, square_size_m)
    objects = _normalise_sample_points(
        object_points, 3, expected_count, "object")
    left = _normalise_sample_points(
        left_points, 2, expected_count, "left")
    right = _normalise_sample_points(
        right_points, 2, expected_count, "right")
    left_image = _normalise_gray_image(left_gray, image_size, "left")
    right_image = _normalise_gray_image(right_gray, image_size, "right")
    params = _normalise_sample_params(sample_params)
    if not np.allclose(objects, expected_object, rtol=0.0, atol=1e-6):
        raise ValueError("object points do not match checkerboard geometry")

    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    final_path = os.path.join(directory, "sample_%03d.npz" % sample_index)
    if os.path.exists(final_path):
        raise IOError("refusing to overwrite persisted sample %s" % final_path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".stereo-sample-", suffix=".npz", dir=directory)
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            format_version=np.asarray(
                [STEREO_SAMPLE_FORMAT_VERSION], dtype=np.int32),
            sample_index=np.asarray([sample_index], dtype=np.int32),
            object_points=objects,
            left_points=left,
            right_points=right,
            left_gray=left_image,
            right_gray=right_image,
            sample_params=params,
            image_size=np.asarray(image_size, dtype=np.int32),
            board_columns=np.asarray([columns], dtype=np.int32),
            board_rows=np.asarray([rows], dtype=np.int32),
            square_size_m=np.asarray([square_size_m], dtype=np.float64),
            left_stamp_ns=np.asarray([int(left_stamp_ns)], dtype=np.int64),
            right_stamp_ns=np.asarray([int(right_stamp_ns)], dtype=np.int64),
            left_sequence=np.asarray([int(left_sequence)], dtype=np.int64),
            right_sequence=np.asarray([int(right_sequence)], dtype=np.int64))
        # np.savez_compressed closes its own handle; fsync the completed bytes
        # before the atomic rename so a sudden power loss cannot expose a
        # zero-length final sample.
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.rename(temporary, final_path)
        return final_path
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_stereo_samples(directory, image_size, columns, rows, square_size_m):
    """Load and strictly validate the contiguous persisted sample sequence."""
    image_size = tuple(int(value) for value in image_size)
    columns = int(columns)
    rows = int(rows)
    square_size_m = float(square_size_m)
    expected_count = columns * rows
    expected_object = checkerboard_object_points(
        columns, rows, square_size_m)
    if not os.path.isdir(directory):
        return []
    paths = sorted(glob.glob(os.path.join(directory, "sample_*.npz")))
    samples = []
    for expected_index, path in enumerate(paths, 1):
        archive = np.load(path, allow_pickle=False)
        try:
            version = int(_sample_scalar(archive, "format_version"))
            index = int(_sample_scalar(archive, "sample_index"))
            saved_columns = int(_sample_scalar(archive, "board_columns"))
            saved_rows = int(_sample_scalar(archive, "board_rows"))
            saved_square = float(_sample_scalar(
                archive, "square_size_m"))
            saved_size = tuple(int(value) for value in np.asarray(
                archive["image_size"]).reshape((-1,)).tolist())
            if version != STEREO_SAMPLE_FORMAT_VERSION:
                raise ValueError("unsupported persisted sample version")
            if index != expected_index:
                raise ValueError("persisted sample indices are not contiguous")
            if os.path.basename(path) != "sample_%03d.npz" % index:
                raise ValueError("persisted sample filename/index mismatch")
            if (saved_columns, saved_rows) != (columns, rows):
                raise ValueError("persisted checkerboard dimensions mismatch")
            if abs(saved_square - square_size_m) > 1e-9:
                raise ValueError("persisted checkerboard square size mismatch")
            if saved_size != image_size:
                raise ValueError("persisted image size mismatch")
            objects = _normalise_sample_points(
                archive["object_points"], 3, expected_count, "object")
            left = _normalise_sample_points(
                archive["left_points"], 2, expected_count, "left")
            right = _normalise_sample_points(
                archive["right_points"], 2, expected_count, "right")
            left_gray = _normalise_gray_image(
                archive["left_gray"], image_size, "left")
            right_gray = _normalise_gray_image(
                archive["right_gray"], image_size, "right")
            params = _normalise_sample_params(archive["sample_params"])
            if not np.allclose(
                    objects, expected_object, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "persisted object points do not match checkerboard")
            samples.append({
                "path": path,
                "sample_index": index,
                "object_points": objects,
                "left_points": left,
                "right_points": right,
                "left_gray": left_gray,
                "right_gray": right_gray,
                "sample_params": params.tolist(),
                "left_stamp_ns": int(_sample_scalar(
                    archive, "left_stamp_ns")),
                "right_stamp_ns": int(_sample_scalar(
                    archive, "right_stamp_ns")),
            })
        finally:
            archive.close()
    return samples


def default_calibration_flags():
    """Match cameracalibrator.py's default two radial coefficients."""
    flags = 0
    for name in ("CALIB_FIX_K3", "CALIB_FIX_K4", "CALIB_FIX_K5",
                 "CALIB_FIX_K6"):
        flags |= int(getattr(cv2, name, 0))
    return flags


def _points(values, dimensions):
    result = []
    for value in values:
        array = np.asarray(value, dtype=np.float32)
        result.append(np.ascontiguousarray(array.reshape((-1, 1, dimensions))))
    return result


def _view_reprojection_errors(object_points, image_points, rvecs, tvecs,
                              camera_matrix, distortion):
    errors = []
    for object_point, image_point, rvec, tvec in zip(
            object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            object_point, rvec, tvec, camera_matrix, distortion)
        delta = projected.reshape((-1, 2)) - image_point.reshape((-1, 2))
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def _rectified_y_errors(left_points, right_points, solution):
    per_view = []
    all_errors = []
    for left, right in zip(left_points, right_points):
        left_rect = cv2.undistortPoints(
            left, solution["K1"], solution["D1"],
            R=solution["R1"], P=solution["P1"])
        right_rect = cv2.undistortPoints(
            right, solution["K2"], solution["D2"],
            R=solution["R2"], P=solution["P2"])
        delta_y = (left_rect[:, 0, 1] - right_rect[:, 0, 1]).astype(
            np.float64)
        per_view.append(float(np.sqrt(np.mean(delta_y * delta_y))))
        all_errors.extend(np.abs(delta_y).tolist())
    all_errors = np.asarray(all_errors, dtype=np.float64)
    rms = float(np.sqrt(np.mean(all_errors * all_errors)))
    p95 = float(np.percentile(all_errors, 95.0))
    return per_view, rms, p95


def _rotation_angle_degrees(rotation):
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _board_normal_spread_degrees(rvecs):
    normals = []
    for rvec in rvecs:
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        normal = np.asarray(rotation[:, 2], dtype=np.float64).reshape((3,))
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    maximum = 0.0
    for first in range(len(normals)):
        for second in range(first + 1, len(normals)):
            cosine = float(np.dot(normals[first], normals[second]))
            cosine = max(-1.0, min(1.0, cosine))
            maximum = max(maximum, math.degrees(math.acos(cosine)))
    return float(maximum)


def _solve_once(object_points, left_points, right_points, image_size,
                calibration_flags):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1e-6)
    mono_left = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None,
        flags=calibration_flags, criteria=criteria)
    mono_right = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None,
        flags=calibration_flags, criteria=criteria)
    left_rms, K1, D1, left_rvecs, left_tvecs = mono_left
    right_rms, K2, D2, right_rvecs, right_tvecs = mono_right

    stereo = cv2.stereoCalibrate(
        object_points, left_points, right_points,
        K1, D1, K2, D2, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    (stereo_rms, K1, D1, K2, D2, rotation, translation,
     essential, fundamental) = stereo
    rectify = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = rectify

    solution = {
        "image_size": tuple(int(value) for value in image_size),
        "left_rms": float(left_rms),
        "right_rms": float(right_rms),
        "stereo_rms": float(stereo_rms),
        "K1": np.asarray(K1, dtype=np.float64),
        "D1": np.asarray(D1, dtype=np.float64),
        "K2": np.asarray(K2, dtype=np.float64),
        "D2": np.asarray(D2, dtype=np.float64),
        "R": np.asarray(rotation, dtype=np.float64),
        "T": np.asarray(translation, dtype=np.float64),
        "E": np.asarray(essential, dtype=np.float64),
        "F": np.asarray(fundamental, dtype=np.float64),
        "R1": np.asarray(R1, dtype=np.float64),
        "R2": np.asarray(R2, dtype=np.float64),
        "P1": np.asarray(P1, dtype=np.float64),
        "P2": np.asarray(P2, dtype=np.float64),
        "Q": np.asarray(Q, dtype=np.float64),
        "valid_roi1": tuple(int(value) for value in valid_roi1),
        "valid_roi2": tuple(int(value) for value in valid_roi2),
    }
    solution["left_view_rms"] = _view_reprojection_errors(
        object_points, left_points, left_rvecs, left_tvecs, K1, D1)
    solution["right_view_rms"] = _view_reprojection_errors(
        object_points, right_points, right_rvecs, right_tvecs, K2, D2)
    (solution["epipolar_view_rms"], solution["epipolar_rms"],
     solution["epipolar_p95"]) = _rectified_y_errors(
         left_points, right_points, solution)
    solution["baseline_m"] = float(-P2[0, 3] / P2[0, 0])
    solution["translation_norm_m"] = float(np.linalg.norm(translation))
    solution["rotation_deg"] = float(_rotation_angle_degrees(rotation))
    solution["pose_spread_deg"] = _board_normal_spread_degrees(left_rvecs)
    return solution


def _outlier_index(solution):
    scores = np.maximum.reduce([
        np.asarray(solution["left_view_rms"], dtype=np.float64),
        np.asarray(solution["right_view_rms"], dtype=np.float64),
        np.asarray(solution["epipolar_view_rms"], dtype=np.float64),
    ])
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    # The absolute floor prevents useful edge/tilted views with slightly larger
    # residuals from being discarded merely because a clean data set has a
    # near-zero MAD.
    limit = max(1.0, median + 3.0 * 1.4826 * mad)
    worst = int(np.argmax(scores))
    if float(scores[worst]) > limit:
        return worst
    return None


def solve_stereo(object_points, left_points, right_points, image_size,
                 min_views=20, max_outlier_fraction=0.20,
                 calibration_flags=None):
    """Solve stereo calibration and robustly discard isolated bad views.

    Returns a dictionary containing matrices, metrics, retained original view
    indices and rejected original view indices.
    """
    if calibration_flags is None:
        calibration_flags = default_calibration_flags()
    if len(object_points) != len(left_points) or len(object_points) != len(
            right_points):
        raise ValueError("object/left/right view counts must match")
    if len(object_points) < int(min_views):
        raise ValueError("need at least %d stereo views" % int(min_views))
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain two positive values")

    objects = _points(object_points, 3)
    left = _points(left_points, 2)
    right = _points(right_points, 2)
    kept = list(range(len(objects)))
    rejected = []
    max_remove = int(math.floor(len(objects) * float(max_outlier_fraction)))

    while True:
        selected_objects = [objects[index] for index in kept]
        selected_left = [left[index] for index in kept]
        selected_right = [right[index] for index in kept]
        solution = _solve_once(
            selected_objects, selected_left, selected_right,
            tuple(int(value) for value in image_size), calibration_flags)
        local_worst = _outlier_index(solution)
        can_remove = (local_worst is not None and
                      len(rejected) < max_remove and
                      len(kept) - 1 >= int(min_views))
        if not can_remove:
            break
        rejected.append(kept.pop(local_worst))

    solution["kept_indices"] = kept
    solution["rejected_indices"] = rejected
    solution["views_used"] = len(kept)
    solution["views_rejected"] = len(rejected)
    return solution


def quality_errors(solution, min_views=20, max_mono_rms=1.0,
                   max_stereo_rms=1.2, max_epipolar_rms=0.8,
                   max_epipolar_p95=1.5, min_baseline_m=0.045,
                   max_baseline_m=0.075, max_rotation_deg=10.0,
                   min_pose_spread_deg=10.0, min_focal_width_ratio=0.35,
                   max_focal_width_ratio=1.35,
                   max_focal_relative_difference=0.25):
    """Return human-readable hard-gate failures; empty means acceptable."""
    errors = []
    scalar_names = (
        "left_rms", "right_rms", "stereo_rms", "epipolar_rms",
        "epipolar_p95", "baseline_m", "translation_norm_m", "rotation_deg",
        "pose_spread_deg")
    try:
        if not np.isfinite(np.asarray(
                [solution.get(name, float("nan")) for name in scalar_names],
                dtype=np.float64)).all():
            errors.append("质量指标包含 NaN 或 Inf")
    except (TypeError, ValueError):
        errors.append("质量指标缺失或格式无效")
    if int(solution.get("views_used", 0)) < int(min_views):
        errors.append("有效样本少于 %d 对" % int(min_views))
    for side in ("left", "right"):
        value = float(solution.get(side + "_rms", float("inf")))
        if value > float(max_mono_rms):
            errors.append("%s 单目标定 RMS %.3f px 超过 %.3f px" % (
                side, value, float(max_mono_rms)))
    stereo_rms = float(solution.get("stereo_rms", float("inf")))
    if stereo_rms > float(max_stereo_rms):
        errors.append("双目标定 RMS %.3f px 超过 %.3f px" % (
            stereo_rms, float(max_stereo_rms)))
    epi_rms = float(solution.get("epipolar_rms", float("inf")))
    if epi_rms > float(max_epipolar_rms):
        errors.append("极线 RMS %.3f px 超过 %.3f px" % (
            epi_rms, float(max_epipolar_rms)))
    epi_p95 = float(solution.get("epipolar_p95", float("inf")))
    if epi_p95 > float(max_epipolar_p95):
        errors.append("极线误差 P95 %.3f px 超过 %.3f px" % (
            epi_p95, float(max_epipolar_p95)))

    baseline = float(solution.get("baseline_m", float("nan")))
    translation_norm = float(solution.get(
        "translation_norm_m", float("nan")))
    if not (float(min_baseline_m) <= baseline <= float(max_baseline_m)):
        errors.append("投影基线 %.4f m 不在 [%.3f, %.3f] m" % (
            baseline, float(min_baseline_m), float(max_baseline_m)))
    if not (float(min_baseline_m) <= translation_norm <=
            float(max_baseline_m)):
        errors.append("平移长度 %.4f m 不在 [%.3f, %.3f] m" % (
            translation_norm, float(min_baseline_m), float(max_baseline_m)))

    P1 = np.asarray(solution.get("P1"), dtype=np.float64)
    P2 = np.asarray(solution.get("P2"), dtype=np.float64)
    if P1.shape != (3, 4) or P2.shape != (3, 4):
        errors.append("投影矩阵尺寸无效")
    else:
        if P1[0, 0] <= 0.0 or P1[1, 1] <= 0.0 or P2[0, 0] <= 0.0 or P2[
                1, 1] <= 0.0:
            errors.append("投影矩阵焦距不是正数")
        if abs(float(P1[0, 3])) > 1e-6:
            errors.append("左投影矩阵 P[3] 应为 0")
        if float(P2[0, 3]) >= 0.0:
            errors.append("右投影矩阵 P[3] 必须为负；请检查左右相机映射")

    rotation = float(solution.get("rotation_deg", float("inf")))
    if rotation > float(max_rotation_deg):
        errors.append("相机间旋转 %.2f 度超过 %.2f 度" % (
            rotation, float(max_rotation_deg)))

    pose_spread = float(solution.get("pose_spread_deg", float("nan")))
    if pose_spread < float(min_pose_spread_deg):
        errors.append("棋盘姿态跨度 %.2f 度小于 %.2f 度" % (
            pose_spread, float(min_pose_spread_deg)))

    image_size = tuple(solution.get("image_size", (0, 0)))
    K1 = np.asarray(solution.get("K1"), dtype=np.float64)
    K2 = np.asarray(solution.get("K2"), dtype=np.float64)
    if len(image_size) != 2 or min(image_size) <= 0:
        errors.append("标定图像尺寸无效")
    elif K1.shape == (3, 3) and K2.shape == (3, 3):
        width = float(image_size[0])
        height = float(image_size[1])
        minimum_focal = float(min_focal_width_ratio) * width
        maximum_focal = float(max_focal_width_ratio) * width
        focals = (K1[0, 0], K1[1, 1], K2[0, 0], K2[1, 1])
        if any(float(value) < minimum_focal or float(value) > maximum_focal
               for value in focals):
            errors.append(
                "焦距像素值不在合理范围 [%.1f, %.1f]" % (
                    minimum_focal, maximum_focal))
        for axis, first, second in (
                ("fx", K1[0, 0], K2[0, 0]),
                ("fy", K1[1, 1], K2[1, 1])):
            denominator = max(1e-9, 0.5 * abs(float(first) + float(second)))
            difference = abs(float(first) - float(second)) / denominator
            if difference > float(max_focal_relative_difference):
                errors.append("左右 %s 相差比例 %.3f 过大" % (
                    axis, difference))
        for name, matrix in (("left", K1), ("right", K2)):
            if not (0.15 * width <= matrix[0, 2] <= 0.85 * width and
                    0.15 * height <= matrix[1, 2] <= 0.85 * height):
                errors.append("%s 主点明显偏离图像中心区域" % name)

    matrices = [solution.get(name) for name in (
        "K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2")]
    try:
        if not all(np.isfinite(np.asarray(matrix)).all() for matrix in matrices):
            errors.append("标定矩阵包含 NaN 或 Inf")
    except (TypeError, ValueError):
        errors.append("标定矩阵缺失或格式无效")
    return errors


def scalar_report(solution):
    """Return YAML-safe scalar/list metrics without NumPy objects."""
    return {
        "views_used": int(solution["views_used"]),
        "views_rejected": int(solution["views_rejected"]),
        "rejected_indices": [int(value) for value in
                             solution["rejected_indices"]],
        "left_rms_px": float(solution["left_rms"]),
        "right_rms_px": float(solution["right_rms"]),
        "stereo_rms_px": float(solution["stereo_rms"]),
        "epipolar_rms_px": float(solution["epipolar_rms"]),
        "epipolar_p95_px": float(solution["epipolar_p95"]),
        "baseline_m": float(solution["baseline_m"]),
        "translation_norm_m": float(solution["translation_norm_m"]),
        "rotation_deg": float(solution["rotation_deg"]),
        "pose_spread_deg": float(solution["pose_spread_deg"]),
        "left_focal_px": [float(solution["K1"][0, 0]),
                          float(solution["K1"][1, 1])],
        "right_focal_px": [float(solution["K2"][0, 0]),
                           float(solution["K2"][1, 1])],
        "left_principal_point_px": [float(solution["K1"][0, 2]),
                                    float(solution["K1"][1, 2])],
        "right_principal_point_px": [float(solution["K2"][0, 2]),
                                     float(solution["K2"][1, 2])],
    }
