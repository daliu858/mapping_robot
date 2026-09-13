#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Headless stereo checkerboard collection and calibration for ROS Melodic."""
from __future__ import print_function

import errno
import os
import sys
import tempfile
import threading
import time

import cv2
import message_filters
import numpy as np
import rospy
import rospkg
import yaml
from camera_calibration.calibrator import (
    ChessboardInfo, Patterns, StereoCalibrator)
from sensor_msgs.msg import Image

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from stereo_calibration_core import (
    checkerboard_object_points, load_stereo_samples, quality_errors,
    save_stereo_sample, scalar_report, solve_stereo)


def _stamp_nanoseconds(message):
    stamp = message.header.stamp
    return int(stamp.secs) * 1000000000 + int(stamp.nsecs)


def _boolean_param(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _stage_text(directory, text):
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
    if not isinstance(text, bytes):
        text = text.encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".stereo-calibration-", suffix=".tmp",
        dir=directory, delete=False)
    temporary = handle.name
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        return temporary
    except Exception:
        try:
            handle.close()
        except Exception:
            pass
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_write(path, text):
    temporary = _stage_text(os.path.dirname(path), text)
    try:
        os.rename(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _validate_yaml_text(text, expected_name, expected_size):
    data = yaml.safe_load(text)
    if data.get("camera_name") != expected_name:
        raise ValueError("camera_name mismatch for %s" % expected_name)
    if (int(data.get("image_width", 0)), int(data.get("image_height", 0))) != (
            int(expected_size[0]), int(expected_size[1])):
        raise ValueError("image size mismatch for %s" % expected_name)
    for key, rows, cols in (
            ("camera_matrix", 3, 3),
            ("rectification_matrix", 3, 3),
            ("projection_matrix", 3, 4)):
        matrix = data.get(key, {})
        values = matrix.get("data", [])
        if int(matrix.get("rows", 0)) != rows or int(
                matrix.get("cols", 0)) != cols or len(values) != rows * cols:
            raise ValueError("invalid %s for %s" % (key, expected_name))
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ValueError("non-finite %s for %s" % (key, expected_name))
    distortion = data.get("distortion_coefficients", {})
    distortion_values = distortion.get("data", [])
    if (int(distortion.get("rows", 0)) != 1 or
            int(distortion.get("cols", 0)) != len(distortion_values) or
            len(distortion_values) < 4 or
            not np.isfinite(np.asarray(
                distortion_values, dtype=np.float64)).all()):
        raise ValueError("invalid distortion coefficients for %s" %
                         expected_name)
    if data.get("distortion_model") not in ("plumb_bob", "rational_polynomial"):
        raise ValueError("invalid distortion model for %s" % expected_name)
    return data


def _yaml_unicode(value):
    """Keep Python 2 PyYAML from serialising UTF-8 text as !!binary."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return dict((_yaml_unicode(key), _yaml_unicode(item))
                    for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [_yaml_unicode(item) for item in value]
    return value


def _install_yaml_pair(left_path, right_path, left_text, right_text,
                       image_size):
    _validate_yaml_text(left_text, "left", image_size)
    _validate_yaml_text(right_text, "right", image_size)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    old_contents = {}
    for path in (left_path, right_path):
        if os.path.exists(path):
            with open(path, "rb") as source:
                old_contents[path] = source.read()
            _atomic_write(path + ".bak." + timestamp, old_contents[path])
        else:
            old_contents[path] = None
    left_temporary = _stage_text(os.path.dirname(left_path), left_text)
    try:
        right_temporary = _stage_text(os.path.dirname(right_path), right_text)
    except Exception:
        if os.path.exists(left_temporary):
            os.unlink(left_temporary)
        raise
    paths = (left_path, right_path)
    deactivated = []
    replaced = []
    try:
        # Remove both active names before publishing either new file.  Thus a
        # second-rename failure leaves one side missing, never a persistent
        # new/old pair.  Backup copies above remain available for rollback.
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
            deactivated.append(path)
        os.rename(left_temporary, left_path)
        replaced.append(left_path)
        left_temporary = None
        os.rename(right_temporary, right_path)
        replaced.append(right_path)
        right_temporary = None
    except Exception as install_error:
        rollback_errors = []

        # First quarantine every newly installed side.  If even that fails,
        # do not restore its old partner: the other active name remains
        # absent, so camera_info_manager cannot load a mixed pair.
        new_pair_quarantined = True
        for path in replaced:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as error:
                new_pair_quarantined = False
                rollback_errors.append(
                    "cannot quarantine %s: %s" % (path, error))

        if new_pair_quarantined:
            # Restore only paths that were actually deactivated.  Restoring
            # left then failing on right still leaves right missing, which is
            # fail-closed rather than a valid-looking mixed stereo pair.
            for path in deactivated:
                previous = old_contents[path]
                if previous is None:
                    continue
                try:
                    _atomic_write(path, previous)
                except Exception as error:
                    rollback_errors.append(
                        "cannot restore %s: %s" % (path, error))
                    break

        if rollback_errors:
            raise RuntimeError(
                "stereo YAML pair install failed (%s); fail-closed rollback "
                "was incomplete: %s" %
                (install_error, "; ".join(rollback_errors)))
        raise
    finally:
        for temporary in (left_temporary, right_temporary):
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


class HeadlessStereoCalibrator(object):
    def __init__(self):
        self.columns = int(rospy.get_param("~board_columns", 8))
        self.rows = int(rospy.get_param("~board_rows", 6))
        self.square_size = float(rospy.get_param("~square_size_m", 0.024))
        self.expected_width = int(rospy.get_param("~expected_width", 640))
        self.expected_height = int(rospy.get_param("~expected_height", 480))
        self.min_samples = int(rospy.get_param("~min_samples", 5))
        self.max_samples = int(rospy.get_param("~max_samples", 5))
        self.process_hz = float(rospy.get_param("~process_hz", 4.0))
        self.timeout_s = float(rospy.get_param("~timeout_s", 900.0))
        self.preview_path = rospy.get_param(
            "~preview_path", "/tmp/stereo_calibration_preview.jpg")
        self.auto_calibrate = _boolean_param("~auto_calibrate", True)
        self.exit_code = 1
        self.finished = False
        self.busy = False
        self.timeout_requested = False
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.last_processed_at = 0.0
        self.last_status_at = 0.0
        self.last_preview_at = 0.0

        if self.columns < 2 or self.rows < 2 or self.square_size <= 0.0:
            raise ValueError("invalid checkerboard geometry")
        if self.min_samples < 5 or self.max_samples < self.min_samples:
            raise ValueError("invalid sample limits")
        if self.process_hz <= 0.0:
            raise ValueError("~process_hz must be positive")

        board = ChessboardInfo(
            "chessboard", self.columns, self.rows, self.square_size)
        flags = 0
        for name in ("CALIB_FIX_K3", "CALIB_FIX_K4", "CALIB_FIX_K5",
                     "CALIB_FIX_K6"):
            flags |= int(getattr(cv2, name, 0))
        self.calibrator = StereoCalibrator(
            [board], flags=flags, pattern=Patterns.Chessboard,
            checkerboard_flags=0,
            max_chessboard_speed=float(rospy.get_param(
                "~max_chessboard_speed_px", 3.0)))

        package_path = rospkg.RosPack().get_path("jetbot_pro")
        output_dir = rospy.get_param(
            "~output_dir",
            os.path.join(package_path, "config", "camera_calibration"))
        self.left_path = rospy.get_param(
            "~left_output", os.path.join(
                output_dir, "stereo_left_640x480.yaml"))
        self.right_path = rospy.get_param(
            "~right_output", os.path.join(
                output_dir, "stereo_right_640x480.yaml"))
        self.report_path = rospy.get_param(
            "~report_output", os.path.join(
                output_dir, "stereo_calibration_report.yaml"))
        self.sample_dir = rospy.get_param(
            "~sample_dir", os.path.join(
                os.path.expanduser("~"), ".ros",
                "stereo_five_shot_samples"))
        self.persisted_samples = load_stereo_samples(
            self.sample_dir,
            (self.expected_width, self.expected_height),
            self.columns, self.rows, self.square_size)
        if len(self.persisted_samples) > self.max_samples:
            raise ValueError(
                "persisted sample count %d exceeds max_samples %d" % (
                    len(self.persisted_samples), self.max_samples))
        # Restore the calibration library's diversity database as well as the
        # solver points.  This prevents a restarted session from accepting a
        # pose that is effectively identical to an already persisted view.
        for sample in self.persisted_samples:
            self.calibrator.db.append((
                list(sample["sample_params"]),
                sample["left_gray"], sample["right_gray"]))
            self.calibrator.good_corners.append((
                sample["left_points"], sample["right_points"],
                None, None, board))
        if self.persisted_samples:
            self.calibrator.compute_goodenough()

        resume_ready = len(self.persisted_samples) >= self.max_samples
        if resume_ready:
            # Set this before subscriber callbacks can run; a complete
            # recovered session must solve, never accept a sixth sample.
            self.busy = True

        left_topic = rospy.get_param(
            "~left_topic", "/stereo/left/image_sync")
        right_topic = rospy.get_param(
            "~right_topic", "/stereo/right/image_sync")
        queue_size = int(rospy.get_param("~queue_size", 5))
        self.left_sub = message_filters.Subscriber(
            left_topic, Image, queue_size=1, buff_size=4 * 1024 * 1024)
        self.right_sub = message_filters.Subscriber(
            right_topic, Image, queue_size=1, buff_size=4 * 1024 * 1024)
        self.sync = message_filters.TimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size)
        self.sync.registerCallback(self.callback)
        self.timer = rospy.Timer(rospy.Duration(2.0), self.timer_callback)
        self.resume_timer = None
        if resume_ready:
            # Do not allow a camera callback to append a sixth in-memory view
            # while the recovered five are queued for immediate solving.
            self.resume_timer = rospy.Timer(
                rospy.Duration(0.1), self._resume_finalize, oneshot=True)
        if self.persisted_samples:
            rospy.loginfo(
                "Recovered %d/%d persisted stereo samples from %s",
                len(self.persisted_samples), self.max_samples,
                self.sample_dir)

        rospy.loginfo(
            "无界面双目标定已就绪：%dx%d 内角点，方格 %.1f mm；目标 %d–%d 对",
            self.columns, self.rows, self.square_size * 1000.0,
            self.min_samples, self.max_samples)
        rospy.loginfo(
            "把棋盘完整放进两个镜头画面；每到一个新姿态，停稳约 1 秒")

    @staticmethod
    def _progress_text(params):
        if not params:
            return "X=0% Y=0% 大小=0% 倾斜=0%"
        labels = {"X": "X", "Y": "Y", "Size": "大小", "Skew": "倾斜"}
        values = []
        for name, _minimum, _maximum, progress in params:
            values.append("%s=%d%%" % (
                labels.get(str(name), str(name)),
                int(round(100.0 * float(progress)))))
        return " ".join(values)

    @staticmethod
    def _movement_hint(params):
        if not params:
            return "先让整张棋盘同时出现在左右两个镜头中"
        progress = dict((str(name), float(value))
                        for name, _low, _high, value in params)
        choices = [
            (progress.get("X", 0.0), "把棋盘换到画面左边或右边"),
            (progress.get("Y", 0.0), "把棋盘换到画面上方或下方"),
            (progress.get("Size", 0.0), "把棋盘明显靠近或远离相机"),
            (progress.get("Skew", 0.0), "把棋盘向左或向右倾斜"),
        ]
        return min(choices, key=lambda item: item[0])[1]

    def _write_preview(self, drawable, sample_count):
        if not self.preview_path:
            return
        try:
            preview = np.hstack((drawable.lscrib, drawable.rscrib))
            cv2.putText(
                preview, "samples %d/%d" % (sample_count, self.max_samples),
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(self.preview_path, preview)
        except Exception as error:
            rospy.logwarn_throttle(10.0, "无法写标定预览图：%s", error)

    def _resume_finalize(self, _event):
        if not self.finished and len(
                self.persisted_samples) >= self.max_samples:
            self.finalize()

    def _persist_accepted_sample(self, sample, left_message, right_message):
        index = len(self.persisted_samples) + 1
        # camera_calibration's helper stores the same square grid with the X/Y
        # axes transposed.  Use our canonical OpenCV row-major convention so
        # persistence validation and offline re-solving agree exactly.
        object_points = checkerboard_object_points(
            self.columns, self.rows, self.square_size)
        left_stamp_ns = _stamp_nanoseconds(left_message)
        right_stamp_ns = _stamp_nanoseconds(right_message)
        path = save_stereo_sample(
            self.sample_dir, index,
            object_points, sample[0], sample[1],
            self.calibrator.db[-1][1], self.calibrator.db[-1][2],
            self.calibrator.db[-1][0],
            (self.expected_width, self.expected_height),
            self.columns, self.rows, self.square_size,
            left_stamp_ns=left_stamp_ns,
            right_stamp_ns=right_stamp_ns,
            left_sequence=int(left_message.header.seq),
            right_sequence=int(right_message.header.seq))
        self.persisted_samples.append({
            "path": path,
            "sample_index": index,
            "object_points": np.asarray(
                object_points, dtype=np.float32).reshape((-1, 1, 3)),
            "left_points": np.asarray(
                sample[0], dtype=np.float32).reshape((-1, 1, 2)),
            "right_points": np.asarray(
                sample[1], dtype=np.float32).reshape((-1, 1, 2)),
            "left_gray": np.asarray(
                self.calibrator.db[-1][1], dtype=np.uint8).copy(),
            "right_gray": np.asarray(
                self.calibrator.db[-1][2], dtype=np.uint8).copy(),
            "sample_params": list(self.calibrator.db[-1][0]),
            "left_stamp_ns": left_stamp_ns,
            "right_stamp_ns": right_stamp_ns,
        })
        return path

    def callback(self, left_message, right_message):
        now = time.time()
        if self.finished or now - self.last_processed_at < 1.0 / self.process_hz:
            return
        with self.lock:
            if self.busy or self.finished:
                return
            self.busy = True
        try:
            observed_size = (int(left_message.width), int(left_message.height))
            right_size = (int(right_message.width), int(right_message.height))
            expected_size = (self.expected_width, self.expected_height)
            if observed_size != expected_size or right_size != expected_size:
                self._finish_failure(
                    "图像尺寸错误：left=%s right=%s，当前标定文件只允许 %s" % (
                        observed_size, right_size, expected_size))
                return
            if left_message.encoding != right_message.encoding:
                self._finish_failure(
                    "左右图像编码不一致：left=%s right=%s" % (
                        left_message.encoding, right_message.encoding))
                return
            previous_count = len(self.calibrator.good_corners)
            drawable = self.calibrator.handle_msg((left_message, right_message))
            current_count = len(self.calibrator.good_corners)
            accepted = current_count > previous_count
            if current_count - previous_count > 1:
                self._finish_failure(
                    "calibrator accepted more than one sample per callback")
                return
            if accepted:
                if len(self.persisted_samples) >= self.max_samples:
                    self._finish_failure(
                        "refusing to collect beyond configured sample limit")
                    return
                try:
                    self._persist_accepted_sample(
                        self.calibrator.good_corners[-1],
                        left_message, right_message)
                except Exception as error:
                    self._finish_failure(
                        "failed to persist accepted stereo sample: %s" % error)
                    return
            sample_count = len(self.persisted_samples)
            if (accepted or
                    now - self.last_preview_at >= 2.0):
                self._write_preview(drawable, sample_count)
                self.last_preview_at = now
            if accepted:
                rospy.loginfo(
                    "已收第 %d 对：%s；下一步：%s，然后停稳",
                    sample_count, self._progress_text(drawable.params),
                    self._movement_hint(drawable.params))
            elif now - self.last_status_at >= 3.0:
                self.last_status_at = now
                if self.calibrator.last_frame_corners is None:
                    rospy.logwarn("左图没有完整棋盘：退远一点，露出整张纸并减少反光")
                else:
                    rospy.loginfo(
                        "当前未新增样本：请停稳 1 秒，或换一个更不同的位置/角度")

            if self.timeout_requested:
                self._finish_failure("等待棋盘超时；没有覆盖或替换正式标定文件")
                return

            if self.auto_calibrate and sample_count >= self.max_samples:
                self.finalize()
        except Exception as error:
            rospy.logerr("标定采集异常：%s", error)
        finally:
            # Throttle from the end of an expensive detection, leaving a real
            # idle gap on Nano instead of immediately consuming queued frames.
            self.last_processed_at = time.time()
            with self.lock:
                self.busy = False

    def timer_callback(self, _event):
        expired = False
        with self.lock:
            if self.finished:
                return
            if time.time() - self.started_at > self.timeout_s:
                if self.busy:
                    self.timeout_requested = True
                else:
                    self.finished = True
                    expired = True
        if expired:
            self._finish_failure("等待棋盘超时；没有覆盖或替换正式标定文件")

    def _thresholds(self):
        return {
            "min_views": int(rospy.get_param(
                "~quality_min_views", self.min_samples)),
            "max_mono_rms": float(rospy.get_param(
                "~quality_max_mono_rms", 1.0)),
            "max_stereo_rms": float(rospy.get_param(
                "~quality_max_stereo_rms", 1.2)),
            "max_epipolar_rms": float(rospy.get_param(
                "~quality_max_epipolar_rms", 0.8)),
            "max_epipolar_p95": float(rospy.get_param(
                "~quality_max_epipolar_p95", 1.5)),
            "min_baseline_m": float(rospy.get_param(
                "~quality_min_baseline_m", 0.052)),
            "max_baseline_m": float(rospy.get_param(
                "~quality_max_baseline_m", 0.068)),
            "max_rotation_deg": float(rospy.get_param(
                "~quality_max_rotation_deg", 10.0)),
            "min_pose_spread_deg": float(rospy.get_param(
                "~quality_min_pose_spread_deg", 10.0)),
            "min_focal_width_ratio": float(rospy.get_param(
                "~quality_min_focal_width_ratio", 0.35)),
            "max_focal_width_ratio": float(rospy.get_param(
                "~quality_max_focal_width_ratio", 1.35)),
            "max_focal_relative_difference": float(rospy.get_param(
                "~quality_max_focal_relative_difference", 0.25)),
        }

    def _write_report(self, report, passed):
        report = dict(report)
        report["passed"] = bool(passed)
        report["board"] = {
            "inner_corners": "%dx%d" % (self.columns, self.rows),
            "square_size_m": float(self.square_size),
        }
        report["persisted_sample_dir"] = self.sample_dir
        report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _atomic_write(
            self.report_path, yaml.safe_dump(
                _yaml_unicode(report), default_flow_style=False,
                allow_unicode=True))

    def finalize(self):
        timed_out = False
        with self.lock:
            if self.timeout_requested:
                timed_out = True
            else:
                self.finished = True
        if timed_out:
            self._finish_failure("等待棋盘超时；没有覆盖或替换正式标定文件")
            return
        rospy.loginfo("样本覆盖完成，正在解算；请不要再移动相机或棋盘……")
        try:
            samples = list(self.persisted_samples)
            if len(samples) != self.max_samples:
                raise ValueError(
                    "expected exactly %d persisted views, found %d" % (
                        self.max_samples, len(samples)))
            object_points = [sample["object_points"] for sample in samples]
            left_points = [sample["left_points"] for sample in samples]
            right_points = [sample["right_points"] for sample in samples]
            image_size = (self.expected_width, self.expected_height)
            thresholds = self._thresholds()
            solution = solve_stereo(
                object_points, left_points, right_points, image_size,
                min_views=thresholds["min_views"])
            errors = quality_errors(solution, **thresholds)
            report = scalar_report(solution)
            report["quality_errors"] = list(errors)
            if errors:
                self._write_report(report, False)
                self._finish_failure(
                    "质量检查未通过：" + "；".join(errors), report_written=True)
                return

            left_yaml = self.calibrator.lryaml(
                "left", solution["D1"], solution["K1"], solution["R1"],
                solution["P1"], image_size, self.calibrator.camera_model)
            right_yaml = self.calibrator.lryaml(
                "right", solution["D2"], solution["K2"], solution["R2"],
                solution["P2"], image_size, self.calibrator.camera_model)
            _install_yaml_pair(
                self.left_path, self.right_path, left_yaml, right_yaml,
                image_size)
            try:
                self._write_report(report, True)
            except Exception as report_error:
                # The calibrated YAML pair is already valid and installed; a
                # diagnostic report failure must not be reported as calibration
                # failure or imply that the old pair is still active.
                rospy.logwarn("标定参数已保存，但报告写入失败：%s", report_error)
            self.exit_code = 0
            rospy.loginfo(
                "标定通过：基线 %.2f mm，极线 RMS %.3f px，双目 RMS %.3f px",
                solution["baseline_m"] * 1000.0,
                solution["epipolar_rms"], solution["stereo_rms"])
            rospy.loginfo("已保存：%s 和 %s", self.left_path, self.right_path)
            rospy.signal_shutdown("stereo calibration completed")
        except Exception as error:
            self._finish_failure("解算或保存失败：%s" % error)

    def _finish_failure(self, message, report_written=False):
        self.finished = True
        self.exit_code = 2
        rospy.logerr(message)
        if not report_written:
            try:
                self._write_report({"quality_errors": [message]}, False)
            except Exception as report_error:
                rospy.logerr("写失败报告也未成功：%s", report_error)
        rospy.signal_shutdown("stereo calibration failed")


def main():
    rospy.init_node("stereo_calibrate_headless")
    node = HeadlessStereoCalibrator()
    rospy.spin()
    return node.exit_code


if __name__ == "__main__":
    sys.exit(main())
