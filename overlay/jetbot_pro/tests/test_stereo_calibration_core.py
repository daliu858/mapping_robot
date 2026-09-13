#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numerical regression tests for the headless stereo calibration solver."""
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from stereo_calibration_core import (
    load_stereo_samples, quality_errors, save_stereo_sample, solve_stereo)


class StereoCalibrationCoreTest(unittest.TestCase):
    @staticmethod
    def _synthetic_views(count=26, noisy_index=None):
        rng = np.random.RandomState(7)
        image_size = (640, 480)
        K1 = np.array([
            [430.0, 0.0, 320.0],
            [0.0, 428.0, 240.0],
            [0.0, 0.0, 1.0]], dtype=np.float64)
        K2 = np.array([
            [432.0, 0.0, 318.0],
            [0.0, 429.0, 241.0],
            [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = np.zeros((5, 1), dtype=np.float64)
        object_template = np.zeros((8 * 6, 1, 3), dtype=np.float32)
        grid = np.mgrid[0:8, 0:6].T.reshape((-1, 2)).astype(np.float32)
        object_template[:, 0, :2] = grid * 0.024
        baseline = np.array([[-0.060], [0.0], [0.0]], dtype=np.float64)

        objects = []
        left = []
        right = []
        for index in range(count):
            rvec = np.array([[
                rng.uniform(-0.25, 0.25),
                rng.uniform(-0.30, 0.30),
                rng.uniform(-0.20, 0.20)]], dtype=np.float64).reshape((3, 1))
            tvec = np.array([[
                rng.uniform(-0.10, 0.04),
                rng.uniform(-0.07, 0.04),
                rng.uniform(0.38, 0.85)]], dtype=np.float64).reshape((3, 1))
            left_pixels, _ = cv2.projectPoints(
                object_template, rvec, tvec, K1, distortion)
            right_pixels, _ = cv2.projectPoints(
                object_template, rvec, tvec + baseline, K2, distortion)
            left_pixels += rng.normal(0.0, 0.06, left_pixels.shape).astype(
                np.float32)
            right_pixels += rng.normal(0.0, 0.06, right_pixels.shape).astype(
                np.float32)
            if noisy_index is not None and index == noisy_index:
                right_pixels[:, 0, 1] += 4.0
            objects.append(object_template.copy())
            left.append(left_pixels.astype(np.float32))
            right.append(right_pixels.astype(np.float32))
        return objects, left, right, image_size

    def test_clean_solution_recovers_baseline_and_passes_quality_gate(self):
        objects, left, right, size = self._synthetic_views()
        solution = solve_stereo(objects, left, right, size, min_views=20)
        self.assertAlmostEqual(solution["baseline_m"], 0.060, delta=0.003)
        self.assertLess(solution["epipolar_rms"], 0.5)
        self.assertEqual([], solution["rejected_indices"])
        self.assertEqual([], quality_errors(solution))

    def test_exactly_five_views_can_solve_and_pass_hard_gates(self):
        objects, left, right, size = self._synthetic_views(count=5)
        solution = solve_stereo(objects, left, right, size, min_views=5)
        self.assertEqual(5, solution["views_used"])
        self.assertAlmostEqual(solution["baseline_m"], 0.060, delta=0.004)
        self.assertEqual([], quality_errors(solution, min_views=5))

    def test_each_persisted_sample_survives_and_loads_without_pickle(self):
        directory = tempfile.mkdtemp(prefix="stereo-five-shot-")
        try:
            objects, left, right, size = self._synthetic_views(count=5)
            for index in range(5):
                path = save_stereo_sample(
                    directory, index + 1,
                    objects[index], left[index], right[index],
                    np.full((480, 640), index, dtype=np.uint8),
                    np.full((480, 640), index + 10, dtype=np.uint8),
                    [0.1 * index, 0.1, 0.2, 0.05],
                    size, 8, 6, 0.024,
                    left_stamp_ns=1000 + index,
                    right_stamp_ns=1005 + index)
                self.assertTrue(os.path.isfile(path))
                loaded = load_stereo_samples(
                    directory, size, 8, 6, 0.024)
                self.assertEqual(index + 1, len(loaded))
                self.assertTrue(np.allclose(
                    left[index], loaded[-1]["left_points"]))
                self.assertTrue(np.allclose(
                    right[index], loaded[-1]["right_points"]))
                self.assertEqual(
                    index, int(loaded[-1]["left_gray"][0, 0]))
                self.assertEqual(
                    index + 10, int(loaded[-1]["right_gray"][0, 0]))
            solution = solve_stereo(
                [sample["object_points"] for sample in loaded],
                [sample["left_points"] for sample in loaded],
                [sample["right_points"] for sample in loaded],
                size, min_views=5)
            self.assertEqual([], quality_errors(solution, min_views=5))
        finally:
            shutil.rmtree(directory)

    def test_incomplete_npz_temporary_file_is_ignored_on_resume(self):
        directory = tempfile.mkdtemp(prefix="stereo-five-shot-")
        try:
            with open(os.path.join(
                    directory, ".stereo-sample-crash.npz"), "wb") as handle:
                handle.write(b"partial")
            loaded = load_stereo_samples(
                directory, (640, 480), 8, 6, 0.024)
            self.assertEqual([], loaded)
        finally:
            shutil.rmtree(directory)

    def test_persisted_object_array_is_rejected_without_pickle(self):
        directory = tempfile.mkdtemp(prefix="stereo-pickle-reject-")
        try:
            np.savez(
                os.path.join(directory, "sample_001.npz"),
                format_version=np.asarray([{"untrusted": True}],
                                          dtype=object))
            with self.assertRaises(ValueError):
                load_stereo_samples(
                    directory, (640, 480), 8, 6, 0.024)
        finally:
            shutil.rmtree(directory)

    def test_single_bad_pair_is_rejected(self):
        objects, left, right, size = self._synthetic_views(noisy_index=9)
        solution = solve_stereo(objects, left, right, size, min_views=20)
        self.assertIn(9, solution["rejected_indices"])
        self.assertEqual([], quality_errors(solution))

    def test_positive_right_projection_is_rejected(self):
        objects, left, right, size = self._synthetic_views()
        solution = solve_stereo(objects, left, right, size, min_views=20)
        solution["P2"] = solution["P2"].copy()
        solution["P2"][0, 3] = abs(solution["P2"][0, 3])
        solution["baseline_m"] = -abs(solution["baseline_m"])
        errors = quality_errors(solution)
        self.assertTrue(any("P[3]" in error for error in errors))

    def test_mismatched_view_counts_fail_before_opencv(self):
        objects, left, right, size = self._synthetic_views(count=20)
        with self.assertRaises(ValueError):
            solve_stereo(objects, left[:-1], right, size, min_views=20)

    def test_nonfinite_metric_is_rejected(self):
        objects, left, right, size = self._synthetic_views()
        solution = solve_stereo(objects, left, right, size, min_views=20)
        solution["epipolar_rms"] = float("nan")
        errors = quality_errors(solution)
        self.assertTrue(any("NaN" in error for error in errors))

    def test_five_view_pose_and_focal_degeneracy_are_rejected(self):
        objects, left, right, size = self._synthetic_views(count=5)
        solution = solve_stereo(objects, left, right, size, min_views=5)
        solution["pose_spread_deg"] = 5.0
        solution["K1"] = solution["K1"].copy()
        solution["K1"][0, 0] = 1100.0
        errors = quality_errors(solution, min_views=5)
        self.assertTrue(any("姿态跨度" in error for error in errors))
        self.assertTrue(any("焦距像素值" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
