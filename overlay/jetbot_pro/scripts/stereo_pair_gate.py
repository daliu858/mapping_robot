#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict timestamp gate for the unsynchronised IMX219-83 stereo pair.

The board has no hardware trigger.  This node accepts left/right Image plus
CameraInfo only when all four header stamps fall within ``~max_pair_dt_s``.
Accepted messages are republished with the left image stamp so ROS1
stereo_image_proc can use ExactTime and the resulting disparity keeps the same
stamp as detections made on the original left image.
"""
from __future__ import print_function

import math
import message_filters
import rospy
import threading
from sensor_msgs.msg import CameraInfo, Image


class StereoPairGate(object):
    def __init__(self):
        self.max_dt = float(rospy.get_param("~max_pair_dt_s", 0.020))
        self.queue_size = int(rospy.get_param("~queue_size", 30))
        require_calibration = rospy.get_param("~require_calibration", False)
        if isinstance(require_calibration, bool):
            self.require_calibration = require_calibration
        else:
            self.require_calibration = str(require_calibration).lower() in (
                "1", "true", "yes", "on")
        if self.max_dt <= 0.0:
            raise ValueError("~max_pair_dt_s must be positive")

        self.left_pub = rospy.Publisher(
            "left/image_sync", Image, queue_size=2)
        self.left_info_pub = rospy.Publisher(
            "left/camera_info_sync", CameraInfo, queue_size=2)
        self.right_pub = rospy.Publisher(
            "right/image_sync", Image, queue_size=2)
        self.right_info_pub = rospy.Publisher(
            "right/camera_info_sync", CameraInfo, queue_size=2)

        self.left_sub = message_filters.Subscriber("left/image_raw", Image)
        self.left_info_sub = message_filters.Subscriber(
            "left/camera_info", CameraInfo)
        self.right_sub = message_filters.Subscriber("right/image_raw", Image)
        self.right_info_sub = message_filters.Subscriber(
            "right/camera_info", CameraInfo)

        self.left_seen = 0
        self.right_seen = 0
        self.accepted = 0
        self.calibration_rejects = 0
        self.dt_sum = 0.0
        self.dt_max = 0.0
        self.window_left = 0
        self.window_right = 0
        self.window_accepted = 0
        self.window_calibration_rejects = 0
        self.window_dt_sum = 0.0
        self.window_dt_max = 0.0
        self.lock = threading.Lock()
        self.left_sub.registerCallback(self.cb_left_seen)
        self.right_sub.registerCallback(self.cb_right_seen)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.left_info_sub,
             self.right_sub, self.right_info_sub],
            self.queue_size, self.max_dt)
        self.sync.registerCallback(self.cb_pair)
        rospy.Timer(rospy.Duration(5.0), self.cb_health)
        rospy.loginfo(
            "stereo_pair_gate: rejecting pairs wider than %.1f ms",
            self.max_dt * 1000.0)

    def cb_left_seen(self, _msg):
        with self.lock:
            self.left_seen += 1
            self.window_left += 1

    def cb_right_seen(self, _msg):
        with self.lock:
            self.right_seen += 1
            self.window_right += 1

    def cb_pair(self, left, left_info, right, right_info):
        stamps = [left.header.stamp, left_info.header.stamp,
                  right.header.stamp, right_info.header.stamp]
        times = [stamp.to_sec() for stamp in stamps]
        span = max(times) - min(times)
        # ApproximateTime already applies this limit; keep an explicit check so
        # a future synchronizer/config change cannot silently weaken the gate.
        if span > self.max_dt + 1e-9:
            return
        if self.require_calibration and not self.calibration_ok(
                left_info, right_info, left, right):
            with self.lock:
                self.calibration_rejects += 1
                self.window_calibration_rejects += 1
            rospy.logwarn_throttle(
                5.0, "stereo_pair_gate: invalid/missing stereo calibration; "
                "need valid focal/projection/rotation values, right P[3] "
                "baseline, and matching frame/image dimensions")
            return

        pair_dt = abs((left.header.stamp - right.header.stamp).to_sec())
        common_stamp = left.header.stamp
        left_info.header.stamp = common_stamp
        right.header.stamp = common_stamp
        right_info.header.stamp = common_stamp

        # Publish metadata first; ExactTime queues it until both images arrive.
        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)
        self.left_pub.publish(left)
        self.right_pub.publish(right)
        with self.lock:
            self.accepted += 1
            self.dt_sum += pair_dt
            self.dt_max = max(self.dt_max, pair_dt)
            self.window_accepted += 1
            self.window_dt_sum += pair_dt
            self.window_dt_max = max(self.window_dt_max, pair_dt)

    @staticmethod
    def calibration_ok(left_info, right_info, left_image=None,
                       right_image=None):
        try:
            left_finite = all(
                value == value and not math.isinf(value)
                for value in list(left_info.D) + list(left_info.K) +
                list(left_info.R) +
                list(left_info.P))
            right_finite = all(
                value == value and not math.isinf(value)
                for value in list(right_info.D) + list(right_info.K) +
                list(right_info.R) +
                list(right_info.P))
            lr = left_info.R
            rr = right_info.R
            left_det = (lr[0] * (lr[4] * lr[8] - lr[5] * lr[7]) -
                        lr[1] * (lr[3] * lr[8] - lr[5] * lr[6]) +
                        lr[2] * (lr[3] * lr[7] - lr[4] * lr[6]))
            right_det = (rr[0] * (rr[4] * rr[8] - rr[5] * rr[7]) -
                         rr[1] * (rr[3] * rr[8] - rr[5] * rr[6]) +
                         rr[2] * (rr[3] * rr[7] - rr[4] * rr[6]))
            image_sizes_ok = True
            if left_image is not None or right_image is not None:
                image_sizes_ok = (
                    left_image is not None and right_image is not None and
                    left_image.width == left_info.width and
                    left_image.height == left_info.height and
                    right_image.width == right_info.width and
                    right_image.height == right_info.height and
                    left_image.width == right_image.width and
                    left_image.height == right_image.height)
            return (left_finite and right_finite and
                    len(left_info.D) >= 4 and len(right_info.D) >= 4 and
                    image_sizes_ok and
                    left_info.width == right_info.width and
                    left_info.height == right_info.height and
                    left_info.width > 0 and left_info.height > 0 and
                    left_info.K[0] > 0.0 and left_info.K[4] > 0.0 and
                    left_info.P[0] > 0.0 and left_info.P[5] > 0.0 and
                    right_info.K[0] > 0.0 and right_info.K[4] > 0.0 and
                    right_info.P[0] > 0.0 and right_info.P[5] > 0.0 and
                    0.5 < left_det < 1.5 and
                    0.5 < right_det < 1.5 and
                    abs(float(left_info.P[3]) / left_info.P[0]) < 1e-6 and
                    float(right_info.P[3]) / right_info.P[0] < -1e-6 and
                    bool(left_info.header.frame_id) and
                    left_info.header.frame_id == right_info.header.frame_id)
        except (AttributeError, IndexError, TypeError, ZeroDivisionError):
            return False

    def cb_health(self, _event):
        with self.lock:
            left = self.window_left
            right = self.window_right
            accepted = self.window_accepted
            calibration_rejects = self.window_calibration_rejects
            dt_sum = self.window_dt_sum
            dt_max = self.window_dt_max
            self.window_left = self.window_right = self.window_accepted = 0
            self.window_calibration_rejects = 0
            self.window_dt_sum = self.window_dt_max = 0.0

        if left == 0 or right == 0:
            rospy.logwarn(
                "stereo_pair_gate last 5s: raw frames left=%d right=%d; check "
                "both CSI sensors and sensor-id mapping", left, right)
            return
        if calibration_rejects:
            rospy.logwarn(
                "stereo_pair_gate last 5s: rejected %d timestamp-valid pairs "
                "because stereo calibration is invalid", calibration_rejects)
            return
        if accepted == 0:
            rospy.logwarn(
                "stereo_pair_gate last 5s: left=%d right=%d but accepted=0 at "
                "%.1f ms; measure timestamp offset, verify frame rates, then "
                "tune ~max_pair_dt_s", left, right,
                self.max_dt * 1000.0)
            return
        rospy.loginfo(
            "stereo_pair_gate last 5s: left=%d right=%d accepted=%d pair_dt "
            "mean=%.1fms max=%.1fms", left, right, accepted,
            1000.0 * dt_sum / accepted, 1000.0 * dt_max)


if __name__ == "__main__":
    rospy.init_node("stereo_pair_gate")
    StereoPairGate()
    rospy.spin()
