#!/usr/bin/env python
"""ROS-host tests for calibration YAML validation and pair installation."""
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from camera_calibration.calibrator import StereoCalibrator, CAMERA_MODEL
    import stereo_calibrate_headless as headless
    ROS_CALIBRATION_AVAILABLE = True
    ROS_CALIBRATION_ERROR = ""
except ImportError as error:
    ROS_CALIBRATION_AVAILABLE = False
    ROS_CALIBRATION_ERROR = str(error)


@unittest.skipUnless(
    ROS_CALIBRATION_AVAILABLE,
    "ROS camera_calibration environment unavailable: %s" %
    ROS_CALIBRATION_ERROR)
class StereoCalibrationHelpersTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="stereo-calibration-test-")
        size = (640, 480)
        K = np.array([
            [430.0, 0.0, 320.0],
            [0.0, 430.0, 240.0],
            [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.zeros((1, 5), dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        P1 = np.hstack((K, np.zeros((3, 1), dtype=np.float64)))
        P2 = P1.copy()
        P2[0, 3] = -430.0 * 0.060
        self.left_text = StereoCalibrator.lryaml(
            "left", D, K, R, P1, size, CAMERA_MODEL.PINHOLE)
        self.right_text = StereoCalibrator.lryaml(
            "right", D, K, R, P2, size, CAMERA_MODEL.PINHOLE)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_expected_camera_names_and_dimensions_validate(self):
        left = headless._validate_yaml_text(
            self.left_text, "left", (640, 480))
        right = headless._validate_yaml_text(
            self.right_text, "right", (640, 480))
        self.assertEqual("left", left["camera_name"])
        self.assertEqual("right", right["camera_name"])

    def test_wrong_camera_name_is_rejected(self):
        with self.assertRaises(ValueError):
            headless._validate_yaml_text(
                self.left_text, "right", (640, 480))

    def test_nonfinite_matrix_is_rejected(self):
        invalid = self.left_text.replace("430.", ".nan", 1)
        with self.assertRaises(ValueError):
            headless._validate_yaml_text(invalid, "left", (640, 480))

    def test_second_replace_failure_rolls_back_new_left(self):
        left_path = os.path.join(self.tempdir, "left.yaml")
        right_path = os.path.join(self.tempdir, "right.yaml")
        original_rename = headless.os.rename
        calls = [0]

        def fail_second_replace(source, destination):
            if destination in (left_path, right_path):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("simulated second rename failure")
            return original_rename(source, destination)

        headless.os.rename = fail_second_replace
        try:
            with self.assertRaises(OSError):
                headless._install_yaml_pair(
                    left_path, right_path, self.left_text, self.right_text,
                    (640, 480))
        finally:
            headless.os.rename = original_rename
        self.assertFalse(os.path.exists(left_path))
        self.assertFalse(os.path.exists(right_path))

    def test_second_replace_failure_restores_existing_pair(self):
        left_path = os.path.join(self.tempdir, "left.yaml")
        right_path = os.path.join(self.tempdir, "right.yaml")
        with open(left_path, "wb") as stream:
            stream.write(b"old-left\n")
        with open(right_path, "wb") as stream:
            stream.write(b"old-right\n")
        original_rename = headless.os.rename
        calls = [0]

        def fail_second_replace(source, destination):
            if destination in (left_path, right_path):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("simulated second rename failure")
            return original_rename(source, destination)

        headless.os.rename = fail_second_replace
        try:
            with self.assertRaises(OSError):
                headless._install_yaml_pair(
                    left_path, right_path, self.left_text, self.right_text,
                    (640, 480))
        finally:
            headless.os.rename = original_rename
        with open(left_path, "rb") as stream:
            self.assertEqual(b"old-left\n", stream.read())
        with open(right_path, "rb") as stream:
            self.assertEqual(b"old-right\n", stream.read())

    def test_restore_failure_leaves_one_side_missing_not_mixed(self):
        left_path = os.path.join(self.tempdir, "left.yaml")
        right_path = os.path.join(self.tempdir, "right.yaml")
        with open(left_path, "wb") as stream:
            stream.write(b"old-left\n")
        with open(right_path, "wb") as stream:
            stream.write(b"old-right\n")
        original_rename = headless.os.rename
        original_atomic_write = headless._atomic_write
        second_failed = [False]
        calls = [0]

        def fail_second_replace(source, destination):
            if destination in (left_path, right_path):
                calls[0] += 1
                if calls[0] == 2:
                    second_failed[0] = True
                    raise OSError("simulated second rename failure")
            return original_rename(source, destination)

        def fail_left_restore(path, text):
            if second_failed[0] and path == left_path:
                raise OSError("simulated rollback write failure")
            return original_atomic_write(path, text)

        headless.os.rename = fail_second_replace
        headless._atomic_write = fail_left_restore
        try:
            with self.assertRaises(RuntimeError):
                headless._install_yaml_pair(
                    left_path, right_path, self.left_text, self.right_text,
                    (640, 480))
        finally:
            headless.os.rename = original_rename
            headless._atomic_write = original_atomic_write
        self.assertFalse(os.path.exists(left_path))
        self.assertFalse(os.path.exists(right_path))

    def test_timeout_is_deferred_but_not_lost_while_callback_is_busy(self):
        node = headless.HeadlessStereoCalibrator.__new__(
            headless.HeadlessStereoCalibrator)
        node.lock = threading.Lock()
        node.finished = False
        node.busy = True
        node.timeout_requested = False
        node.started_at = time.time() - 10.0
        node.timeout_s = 1.0
        failures = []
        node._finish_failure = lambda message: failures.append(message)

        node.timer_callback(None)
        self.assertTrue(node.timeout_requested)
        self.assertEqual([], failures)

        node.busy = False
        node.timer_callback(None)
        self.assertTrue(node.finished)
        self.assertEqual(1, len(failures))


if __name__ == "__main__":
    unittest.main()
