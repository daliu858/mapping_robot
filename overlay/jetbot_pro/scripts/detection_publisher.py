#!/usr/bin/env python
# -*- coding: utf-8 -*-
# detection_publisher.py  —  v5 离线方案 STEP C(ROS 节点, py2/3 兼容)
#
# 读 PC 开放词汇检测器产出的 detections.json, 在回放 rosbag 时, 把离线
# 检测结果"伪装成 NVIDIA detectnet 的话题"(Detection2DArray + VisionInfo)发出
# 去。下游 semantic_mapper.py 仍保持检测器无关，只认标准 vision_msgs。
#
# 同步方式: 订阅回放出来的相机话题(带原始时间戳), 每收到一帧就按时间戳查
# detections.json; 命中(被离线模型处理过的帧)就发该帧的框, 时间戳沿用
# 图像帧的 stamp -> semantic_mapper 据此与双目视差同步并查询 TF。
#
# 关键: Detection2D.results[0].id 用的是 POI 类别在 classes 列表里的索引,
# 并把 classes 写进 rosparam + 用 VisionInfo.database_location 指过去, 于是
# semantic_mapper 的 label_of(id) 能查到类别名 —— 与在线 detectnet 行为一致。
from __future__ import print_function
import bisect
import os
import sys
import threading
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import (Detection2DArray, Detection2D,
                             ObjectHypothesisWithPose, VisionInfo)
from offline_replay_runner import validate_recording_bag
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from replay_contract import load_detection_contract


def match_detection_frame(index, sorted_stamps, stamp_ns, tol_ns):
    """Return ``(source_stamp, detections)`` without crossing the gate.

    Formal replay passes ``tol_ns == 0``.  Keeping the matching rule in this
    small pure helper makes the exact-timestamp safety contract testable on a
    host that does not have ROS installed.
    """
    dets = index.get(stamp_ns)
    if dets is not None:
        return stamp_ns, dets
    if not sorted_stamps or tol_ns <= 0:
        return None, None

    pos = bisect.bisect_left(sorted_stamps, stamp_ns)
    candidates = []
    if pos < len(sorted_stamps):
        candidates.append(sorted_stamps[pos])
    if pos > 0:
        candidates.append(sorted_stamps[pos - 1])
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda value: abs(value - stamp_ns))
    if abs(nearest - stamp_ns) <= tol_ns:
        return nearest, index[nearest]
    return None, None


def match_detections(index, sorted_stamps, stamp_ns, tol_ns):
    """Backward-compatible pure helper returning only detections."""
    return match_detection_frame(
        index, sorted_stamps, stamp_ns, tol_ns)[1]


def image_dimensions(message, compressed):
    """Read actual dimensions from the replayed image message."""
    if not compressed:
        width, height = int(message.width), int(message.height)
        if width <= 0 or height <= 0:
            raise ValueError("raw replay image has invalid dimensions")
        return width, height
    encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError("compressed replay image cannot be decoded")
    return int(image.shape[1]), int(image.shape[0])


def publisher_completion_state(expected_stamps, matched_stamps,
                               callbacks_inflight, pipeline_error="",
                               input_connections=0):
    """Return the fail-closed state for the detector input EOS service."""
    if pipeline_error:
        return False, "INVALID: %s" % pipeline_error
    expected = set(int(value) for value in expected_stamps)
    matched = set(int(value) for value in matched_stamps)
    unexpected = matched - expected
    if unexpected:
        return False, "INVALID: unexpected matched stamps: %s" % \
            sorted(unexpected)[:3]
    missing = expected - matched
    if missing:
        return False, "NOT_READY: waiting for %d detection frames" % \
            len(missing)
    inflight = int(callbacks_inflight)
    if inflight != 0:
        return False, "NOT_READY: waiting for %d publisher callbacks" % \
            inflight
    connections = int(input_connections)
    if connections != 0:
        return False, "NOT_READY: waiting for %d image transports to close" % \
            connections
    return True, "published=%d/%d" % (len(matched), len(expected))


class DetectionPublisher(object):
    def __init__(self):
        det_file = rospy.get_param("~det_file")
        data = load_detection_contract(det_file)
        bag = os.path.abspath(os.path.expanduser(rospy.get_param("~bag")))
        self.optical = rospy.get_param(
            "~optical_frame", "stereo_left_optical")
        valid_recording, recording = validate_recording_bag(
            bag, "/survey/recording_status",
            expected_optical_frame=self.optical)
        if not valid_recording:
            raise ValueError("survey bag transaction is invalid: %s" % recording)
        json_token = str(data.get("recording_token", ""))
        json_map_hash = str(data.get("map_image_sha256", "")).lower()
        json_manifest_hash = str(
            data.get("map_manifest_sha256", "")).lower()
        if (not json_token or json_token != str(recording.get("token", "")) or
                json_map_hash != str(
                    recording.get("map_image_sha256", "")).lower() or
                json_manifest_hash != str(
                    recording.get("map_manifest_sha256", "")).lower() or
                list(data["camera_mount"]) !=
                    list(recording.get("camera_mount", []))):
            raise ValueError(
                "detections.json was not produced from this survey bag; "
                "recording token/map manifest/camera_mount mismatch")
        self.classes = data["classes"]
        self.backend = str(data.get("backend", "open-vocabulary-offline"))
        compressed = bool(rospy.get_param("~compressed", True))
        self.compressed = compressed
        img_topic = rospy.get_param("~image_topic",
                                    "/stereo/left/image_raw/compressed")
        source_topic = data.get("image_topic")
        if source_topic != img_topic:
            raise ValueError(
                "detection image_topic mismatch: JSON=%r replay=%r; "
                "refusing bbox/depth fusion from different cameras" %
                (source_topic, img_topic))
        source_rectified = bool(data.get("detections_rectified", False))
        expected_rectified = bool(rospy.get_param(
            "~detections_rectified", source_rectified))
        if source_rectified != expected_rectified:
            raise ValueError(
                "detections_rectified mismatch: JSON=%s launch=%s; refusing "
                "geometrically invalid bbox/depth fusion" %
                (source_rectified, expected_rectified))
        self.tol_ns = int(float(rospy.get_param("~match_tol_ms", 0.0)) * 1e6)
        require_exact = bool(rospy.get_param("~require_exact_timestamps", True))
        if require_exact and self.tol_ns != 0:
            raise ValueError(
                "formal replay requires ~match_tol_ms=0; refusing to reuse "
                "boxes from a nearby frame")

        # stamp(ns) -> dets, 用于按图像帧时间戳精确命中
        self.index = {}
        for fr in data["frames"]:
            self.index[int(fr["stamp_ns"])] = fr["dets"]
        self.frame_sizes = data["frame_sizes"]
        self.sorted_stamps = sorted(self.index.keys())
        self.expected_stamps = set(self.sorted_stamps)
        self.matched_stamps = set()
        self.processing_stamps = set()
        self.callbacks_inflight = 0
        self.pipeline_error = ""
        self.input_closed = False
        self.lock = threading.Lock()
        self.recv = 0
        self.matched = 0

        # 把类别表挂到 rosparam, 供 semantic_mapper 经 VisionInfo 取用
        labels_param = rospy.get_param("~labels_param",
                                       "/detection_publisher/class_labels")
        rospy.set_param(labels_param, self.classes)

        self.pub = rospy.Publisher("detections", Detection2DArray, queue_size=10)
        self.vpub = rospy.Publisher("vision_info", VisionInfo, queue_size=1,
                                    latch=True)
        vi = VisionInfo()
        vi.method = self.backend
        vi.database_location = labels_param
        vi.database_version = 0
        self.vpub.publish(vi)                      # latched, 后到的订阅者也能收到

        if compressed:
            self.image_subscriber = rospy.Subscriber(
                img_topic, CompressedImage, self.cb, queue_size=30)
        else:
            self.image_subscriber = rospy.Subscriber(
                img_topic, Image, self.cb, queue_size=30)
        from std_srvs.srv import Trigger, TriggerResponse
        self._TriggerResponse = TriggerResponse
        self.ready_service = rospy.Service(
            "~ready", Trigger, self.cb_ready)
        self.complete_input_service = rospy.Service(
            "~complete_input", Trigger, self.cb_complete_input)
        if self.tol_ns > 0:
            rospy.logwarn(
                "detection_publisher: nonzero ~match_tol_ms may reuse boxes "
                "from a nearby frame; exact source timestamps are preferred")
        rospy.loginfo("detection_publisher: indexed %d frames, %d classes: %s",
                      len(self.index), len(self.classes), ",".join(self.classes))

    def cb_ready(self, _request):
        with self.lock:
            if self.pipeline_error:
                return self._TriggerResponse(
                    success=False,
                    message="INVALID: %s" % self.pipeline_error)
            if self.input_closed:
                return self._TriggerResponse(
                    success=False, message="input is already closed")
        return self._TriggerResponse(
            success=True,
            message="indexed=%d source=%s" %
                    (len(self.index), self.backend))

    def cb_complete_input(self, _request):
        """Seal detector output after rosbag EOF and all indexed callbacks."""
        input_connections = self.image_subscriber.get_num_connections()
        with self.lock:
            if self.input_closed:
                return self._TriggerResponse(
                    success=True,
                    message="input already closed: published=%d/%d" %
                            (len(self.matched_stamps),
                             len(self.expected_stamps)))
            ready, detail = publisher_completion_state(
                self.expected_stamps, self.matched_stamps,
                self.callbacks_inflight, self.pipeline_error,
                input_connections)
            if not ready:
                return self._TriggerResponse(success=False, message=detail)
            self.input_closed = True
            subscriber = self.image_subscriber
        # unregister() may touch ROS transport state, so never call it while
        # holding the callback/service state lock.
        subscriber.unregister()
        rospy.loginfo("detection_publisher input sealed: %s", detail)
        return self._TriggerResponse(success=True, message=detail)

    def cb(self, msg):
        with self.lock:
            if self.input_closed:
                return
            self.recv += 1
            self.callbacks_inflight += 1
            received = self.recv
        try:
            self._handle_image(msg, received)
        except Exception as error:
            detail = "detection publisher callback failed: %s" % str(error)
            with self.lock:
                if not self.pipeline_error:
                    self.pipeline_error = detail
            rospy.logfatal(detail)
            rospy.signal_shutdown(detail)
        finally:
            with self.lock:
                self.callbacks_inflight = max(
                    0, self.callbacks_inflight - 1)

    def _handle_image(self, msg, received):
        ns = msg.header.stamp.secs * 1000000000 + msg.header.stamp.nsecs
        matched_stamp, dets = match_detection_frame(
            self.index, self.sorted_stamps, ns, self.tol_ns)
        if dets is None:
            # stride 跳过的帧本就无检测; 但大量收帧却 0 命中多半是 stamp 对不上,
            # 显式告警, 避免静默产出空地图。
            with self.lock:
                matched_count = len(self.matched_stamps)
            if received >= 20 and matched_count == 0:
                rospy.logwarn_throttle(
                    5.0,
                    "detection_publisher: received %d frames but matched 0; "
                    "check image stamps vs detections.json stamp_ns "
                    "(or raise ~match_tol_ms)",
                    received)
            return            # 这帧被离线模型跳过了(stride), 不发
        try:
            actual_size = image_dimensions(msg, self.compressed)
        except ValueError as error:
            raise ValueError(
                "detection replay image validation failed: %s" % str(error))
        expected_size = self.frame_sizes[matched_stamp]
        if actual_size != expected_size:
            detail = (
                "detection/replay image dimension mismatch at %d: "
                "JSON=%s replay=%s" %
                (matched_stamp, expected_size, actual_size))
            raise ValueError(detail)
        with self.lock:
            if (matched_stamp in self.matched_stamps or
                    matched_stamp in self.processing_stamps):
                detail = "duplicate replay image stamp: %d" % matched_stamp
                self.pipeline_error = detail
                raise ValueError(detail)
            self.processing_stamps.add(matched_stamp)
        arr = Detection2DArray()
        arr.header.stamp = msg.header.stamp
        # 强制左目光学帧：视差反投影和 map TF 都以它为坐标基准。
        arr.header.frame_id = self.optical
        for d in dets:
            det = Detection2D()
            det.header = arr.header
            det.bbox.center.x = float(d["cx"])
            det.bbox.center.y = float(d["cy"])
            det.bbox.size_x = float(d["sx"])
            det.bbox.size_y = float(d["sy"])
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(d["cls"])
            hyp.score = float(d["score"])
            det.results.append(hyp)
            arr.detections.append(det)
        try:
            self.pub.publish(arr)
        except Exception:
            with self.lock:
                self.processing_stamps.discard(matched_stamp)
            raise
        with self.lock:
            self.processing_stamps.discard(matched_stamp)
            self.matched_stamps.add(matched_stamp)
            self.matched = len(self.matched_stamps)
            matched_count = self.matched
        if matched_count % 50 == 0:
            rospy.loginfo("detection_publisher: published %d detection frames",
                          matched_count)


if __name__ == "__main__":
    rospy.init_node("detection_publisher")
    DetectionPublisher()
    rospy.spin()
