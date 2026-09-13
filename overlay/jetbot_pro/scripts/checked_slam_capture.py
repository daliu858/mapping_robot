#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Atomically capture one gmapping run and bind it to its final map.

The physical robot only traverses the site once.  While gmapping is active we
record stereo images, scans, odometry and TF to a staging bag.  On an explicit
completion request the robot must already be stationary; the recorder waits
for the final map update, saves and validates the map, publishes the final map
identity plus a STARTED -> COMPLETED transaction into the still-open bag,
closes and verifies the bag, then atomically exposes the final pathname.

AMCL, move_base and TEB are intentionally absent from this capture contract.
"""
from __future__ import print_function

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from map_identity import load_map_identity
from replay_contract import (camera_transform_matches,
                             canonical_camera_mount,
                             validate_recording_states)


def normalize_bag_path(path):
    normalized = os.path.abspath(os.path.expanduser(str(path)))
    return normalized if normalized.lower().endswith(".bag") \
        else normalized + ".bag"


def capture_paths(final_bag):
    final_bag = normalize_bag_path(final_bag)
    return {
        "final": final_bag,
        "staging": final_bag[:-4] + ".recording.bag",
        "active": final_bag + ".active",
        "complete": final_bag + ".complete.json",
        "failed": final_bag + ".failed.json",
    }


def canonical_frame(frame):
    return str(frame or "").strip().lstrip("/")


def tf_edge(parent, child):
    return canonical_frame(parent), canonical_frame(child)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def free_space_bytes(path):
    probe = os.path.abspath(path or ".")
    if not os.path.isdir(probe):
        probe = os.path.dirname(probe) or "."
    stat = os.statvfs(probe)
    block_size = getattr(stat, "f_frsize", 0) or stat.f_bsize
    return int(stat.f_bavail) * int(block_size)


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


def parse_space_bytes(value):
    """Parse the strict K/M/G syntax accepted for the recorder disk floor."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("rosbag min-space must not be empty")
    multiplier = 1
    if text[-1] in "KkMmGg":
        suffix = text[-1].lower()
        text = text[:-1]
        multiplier = {"k": 1024, "m": 1024 ** 2,
                      "g": 1024 ** 3}[suffix]
    try:
        amount = int(text)
    except (TypeError, ValueError):
        raise ValueError("rosbag min-space must be an integer with K/M/G")
    if amount <= 0:
        raise ValueError("rosbag min-space must be positive")
    return amount * multiplier


def camera_info_pair_valid(left, right,
                           expected_frame="stereo_left_optical"):
    """Validate the calibrated stereo geometry needed by offline depth."""
    try:
        left_values = list(left.D) + list(left.K) + list(left.R) + list(left.P)
        right_values = (list(right.D) + list(right.K) + list(right.R) +
                        list(right.P))
        if not all(finite_number(item)
                   for item in left_values + right_values):
            return False
        if (len(left.D) < 4 or len(right.D) < 4 or
                len(left.K) != 9 or len(right.K) != 9 or
                len(left.R) != 9 or len(right.R) != 9 or
                len(left.P) != 12 or len(right.P) != 12):
            return False
        if (int(left.width) != int(right.width) or
                int(left.height) != int(right.height) or
                int(left.width) <= 0 or int(left.height) <= 0):
            return False
        if (str(left.header.frame_id).lstrip("/") != expected_frame or
                str(right.header.frame_id).lstrip("/") != expected_frame):
            return False
        if (float(left.P[0]) <= 0.0 or float(left.P[5]) <= 0.0 or
                float(right.P[0]) <= 0.0 or float(right.P[5]) <= 0.0):
            return False
        if abs(float(left.P[3]) / float(left.P[0])) > 1e-6:
            return False
        baseline = -float(right.P[3]) / float(right.P[0])
        return 0.05 <= baseline <= 0.07
    except (AttributeError, IndexError, TypeError, ValueError,
            ZeroDivisionError):
        return False


def wait_for_path(path, timeout_s):
    deadline = time.time() + max(0.0, float(timeout_s))
    while not os.path.isfile(path) and time.time() < deadline:
        time.sleep(0.10)
    return os.path.isfile(path)


def rosbag_record_command(executable, staging_bag, topics,
                          min_space="700M"):
    """Build argv for the C++ ``rosbag/record`` executable.

    Melodic's Python ``rosbag record`` wrapper does not expose ``--min-space``
    even though the underlying recorder does.  Calling the resolved C++ node
    directly preserves the disk floor instead of failing in optparse.
    """
    executable = str(executable or "").strip()
    if not executable:
        raise ValueError("rosbag record executable is missing")
    topics = [str(topic).strip() for topic in topics if str(topic).strip()]
    if not topics:
        raise ValueError("rosbag record topic list is empty")
    return [
        executable,
        "--buffsize", "256",
        "--chunksize", "768",
        "--min-space", str(min_space),
        "-O", str(staging_bag),
    ] + topics


def find_rosbag_record_executable():
    import roslib.packages
    candidates = roslib.packages.find_node("rosbag", "record") or []
    candidates = [path for path in candidates
                  if os.path.isfile(path) and os.access(path, os.X_OK)]
    if not candidates:
        raise RuntimeError("cannot find executable rosbag/record node")
    return candidates[0]


def required_bag_minimums(required_topics):
    """Return minimum counts for health topics that must also be on disk."""
    result = {}
    for topic in required_topics:
        topic = str(topic).strip()
        if not topic or topic == "/map":
            continue
        result[topic] = 10
    result["/tf"] = 10
    for topic in (
            "/stereo/left/camera_info",
            "/stereo/right/camera_info",
            "/stereo/left/image_raw/compressed",
            "/stereo/right/image_raw/compressed"):
        if topic in result:
            result[topic] = 30
    return result


def commit_file_no_replace(source, target):
    """Atomically expose a same-filesystem file without overwriting target."""
    if os.path.exists(target):
        raise RuntimeError("commit target already exists: %s" % target)
    os.link(source, target)
    try:
        os.unlink(source)
    except Exception:
        try:
            os.unlink(target)
        except Exception:
            pass
        raise


def capture_status_record(state, token, camera_mount, started_unix,
                          map_hash="", manifest_hash="", reason="",
                          finished_unix=None):
    record = {
        "schema": 1,
        "state": str(state),
        "token": str(token),
        "pose_source": "gmapping",
        "camera_mount": list(camera_mount),
        "started_unix": float(started_unix),
    }
    if map_hash:
        record["map_image_sha256"] = str(map_hash)
    if manifest_hash:
        record["map_manifest_sha256"] = str(manifest_hash)
    if state in ("completed", "failed"):
        record["finished_unix"] = float(
            time.time() if finished_unix is None else finished_unix)
    if reason:
        record["reason"] = str(reason)
    return record


class CheckedSlamCapture(object):
    def __init__(self):
        self.paths = capture_paths(rospy.get_param("~bag"))
        self.map_file = os.path.abspath(os.path.expanduser(
            str(rospy.get_param("~map_file"))))
        if not self.map_file.endswith(".yaml"):
            raise ValueError("~map_file must end in .yaml")
        self.map_prefix = self.map_file[:-5]
        self.map_image = self.map_prefix + ".pgm"
        self.camera_mount = canonical_camera_mount(
            rospy.get_param("~camera_mount"))
        self.camera_frame = canonical_frame(rospy.get_param(
            "~camera_optical_frame", "stereo_left_optical"))
        self.topics = self._csv("~topics_csv")
        self.required_topics = self._csv("~required_topics_csv")
        self.status_topic = str(rospy.get_param(
            "~recording_status_topic", "/survey/recording_status"))
        self.startup_timeout = float(rospy.get_param(
            "~startup_timeout_s", 60.0))
        self.topic_stale_timeout = float(rospy.get_param(
            "~topic_stale_timeout_s", 3.5))
        self.tf_stale_timeout = float(rospy.get_param(
            "~tf_stale_timeout_s", 2.0))
        self.map_settle_s = float(rospy.get_param("~map_settle_s", 3.0))
        self.completion_wait_s = float(rospy.get_param(
            "~completion_wait_s", 8.0))
        self.map_save_timeout = float(rospy.get_param(
            "~map_save_timeout_s", 30.0))
        self.bag_finalize_timeout = float(rospy.get_param(
            "~bag_finalize_timeout_s", 15.0))
        self.rosbag_stop_timeout = float(rospy.get_param(
            "~rosbag_stop_timeout_s", 45.0))
        self.min_free_gib = float(rospy.get_param(
            "~min_free_space_gib", 2.5))
        self.finalize_free_gib = float(rospy.get_param(
            "~finalize_free_space_gib", 0.8))
        self.rosbag_min_space = str(rospy.get_param(
            "~rosbag_min_space", "700M"))
        self.motion_linear_min = float(rospy.get_param(
            "~motion_linear_min_mps", 0.02))
        self.motion_angular_min = float(rospy.get_param(
            "~motion_angular_min_radps", 0.03))
        self.motion_topic = str(rospy.get_param(
            "~motion_odom_topic", "/odom_raw"))
        self.max_pair_dt_s = float(rospy.get_param(
            "~max_pair_dt_s", 0.020))

        if not self.camera_frame or not self.topics or not self.required_topics:
            raise ValueError("capture topics and camera frame must not be empty")
        for topic in (self.status_topic, "/survey/map_image_sha256",
                      "/survey/map_manifest_sha256", "/tf"):
            if topic not in self.topics:
                raise ValueError("required recording topic missing: %s" % topic)
        positive_values = (
            self.startup_timeout, self.topic_stale_timeout,
            self.tf_stale_timeout, self.map_settle_s,
            self.completion_wait_s, self.map_save_timeout,
            self.bag_finalize_timeout, self.rosbag_stop_timeout,
            self.min_free_gib,
            self.finalize_free_gib, self.max_pair_dt_s)
        if (not all(finite_number(value) for value in positive_values) or
                min(positive_values) <= 0.0):
            raise ValueError("capture timeouts and free-space gate must be positive")
        if (not finite_number(self.motion_linear_min) or
                not finite_number(self.motion_angular_min) or
                self.motion_linear_min < 0.0 or
                self.motion_angular_min < 0.0):
            raise ValueError("motion thresholds must be finite and nonnegative")
        if self.finalize_free_gib >= self.min_free_gib:
            raise ValueError(
                "finalize free-space reserve must be below startup reserve")
        if int(self.finalize_free_gib * (1024 ** 3)) <= parse_space_bytes(
                self.rosbag_min_space):
            raise ValueError(
                "finalize reserve must exceed rosbag min-space")

        conflicts = []
        for path in self.paths.values():
            if os.path.exists(path) or os.path.exists(path + ".active"):
                conflicts.append(path)
        for path in (self.map_file, self.map_image):
            if os.path.exists(path):
                conflicts.append(path)
        if conflicts:
            raise ValueError(
                "capture output already exists; choose a new run: %s" %
                ", ".join(sorted(set(conflicts))))

        self.session_token = uuid.uuid4().hex
        self.started_at = time.time()
        self.map_hash = ""
        self.map_manifest_hash = ""
        self.process = None
        self.stop_forced = False
        self.finished_state = None
        self.completion_requested = False
        self.last = {topic: None for topic in self.required_topics}
        self.tf_last = {
            tf_edge("map", "odom"): None,
            tf_edge("odom", "base_footprint"): None,
            tf_edge("base_footprint", self.camera_frame): None,
            tf_edge("base_footprint", "laser_frame"): None,
        }
        self.camera_tf_error = ""
        self.camera_input_error = ""
        self.motion_error = ""
        self.camera_info = {"left": None, "right": None}
        self.camera_stamps = {
            "left": deque(maxlen=20), "right": deque(maxlen=20)}
        self.stereo_pair_last = None
        self.robot_moving = True
        self.stationary_since = None
        self.lock = threading.Lock()
        self.finish_lock = threading.Lock()
        self.request_lock = threading.Lock()

        self.map_hash_pub = rospy.Publisher(
            "/survey/map_image_sha256", String, queue_size=2, latch=True)
        self.manifest_hash_pub = rospy.Publisher(
            "/survey/map_manifest_sha256", String, queue_size=2, latch=True)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=3, latch=True)
        self.stop_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.inhibit_pub = rospy.Publisher(
            "/safety_stop", Bool, queue_size=1, latch=True)
        from std_srvs.srv import Trigger, TriggerResponse
        self._TriggerResponse = TriggerResponse
        self.complete_service = rospy.Service(
            "~complete", Trigger, self.cb_complete)

        self.subscribers = [
            rospy.Subscriber(topic, rospy.AnyMsg, self.cb_topic,
                             callback_args=topic, queue_size=1)
            for topic in self.required_topics
            if topic not in (
                "/tf", self.motion_topic,
                "/stereo/left/camera_info",
                "/stereo/right/camera_info",
                "/stereo/left/image_raw/compressed",
                "/stereo/right/image_raw/compressed")]
        self.subscribers.extend([
            rospy.Subscriber(
                "/stereo/left/camera_info", CameraInfo,
                self.cb_camera_info,
                callback_args=("left", "/stereo/left/camera_info"),
                queue_size=2),
            rospy.Subscriber(
                "/stereo/right/camera_info", CameraInfo,
                self.cb_camera_info,
                callback_args=("right", "/stereo/right/camera_info"),
                queue_size=2),
            rospy.Subscriber(
                "/stereo/left/image_raw/compressed", CompressedImage,
                self.cb_camera_image,
                callback_args=("left", "/stereo/left/image_raw/compressed"),
                queue_size=3),
            rospy.Subscriber(
                "/stereo/right/image_raw/compressed", CompressedImage,
                self.cb_camera_image,
                callback_args=("right", "/stereo/right/image_raw/compressed"),
                queue_size=3),
        ])
        self.tf_subscriber = rospy.Subscriber(
            "/tf", TFMessage, self.cb_tf, queue_size=20)
        self.motion_subscriber = rospy.Subscriber(
            self.motion_topic, Odometry, self.cb_motion, queue_size=5)
        rospy.on_shutdown(self._on_shutdown)

    @staticmethod
    def _csv(name):
        return [item.strip() for item in
                str(rospy.get_param(name, "")).split(",") if item.strip()]

    @staticmethod
    def _prepare_json(path, record):
        parent = os.path.dirname(path) or "."
        if not os.path.isdir(parent):
            os.makedirs(parent)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", prefix=".slam-capture.", suffix=".tmp",
                    dir=parent, delete=False) as stream:
                temporary = stream.name
                json.dump(record, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            result = temporary
            temporary = None
            return result
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _atomic_json(cls, path, record):
        temporary = cls._prepare_json(path, record)
        try:
            os.rename(temporary, path)
            temporary = None
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _record(self, state, reason="", finished_unix=None):
        return capture_status_record(
            state, self.session_token, self.camera_mount, self.started_at,
            self.map_hash, self.map_manifest_hash, reason, finished_unix)

    def _publish_status(self, record):
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        self.status_pub.publish(String(data=payload))

    def cb_topic(self, _message, topic):
        with self.lock:
            self.last[topic] = time.time()

    def cb_camera_info(self, message, callback_args):
        side, topic = callback_args
        now = time.time()
        with self.lock:
            self.last[topic] = now
            self.camera_info[side] = message
            left = self.camera_info["left"]
            right = self.camera_info["right"]
            if left is not None and right is not None:
                if camera_info_pair_valid(
                        left, right, expected_frame=self.camera_frame):
                    if self.camera_input_error.startswith(
                            "invalid stereo CameraInfo"):
                        self.camera_input_error = ""
                else:
                    self.camera_input_error = (
                        "invalid stereo CameraInfo/calibration or baseline")

    def cb_camera_image(self, message, callback_args):
        side, topic = callback_args
        now = time.time()
        try:
            stamp = float(message.header.stamp.to_sec())
        except (AttributeError, TypeError, ValueError):
            stamp = float("nan")
        with self.lock:
            self.last[topic] = now
            if not finite_number(stamp) or stamp <= 0.0:
                self.camera_input_error = (
                    "stereo compressed image has invalid timestamp")
                return
            self.camera_stamps[side].append(stamp)
            other = "right" if side == "left" else "left"
            if (self.camera_stamps[other] and
                    min(abs(stamp - candidate)
                        for candidate in self.camera_stamps[other]) <=
                    self.max_pair_dt_s):
                self.stereo_pair_last = now

    def cb_motion(self, message):
        twist = message.twist.twist
        velocities = (twist.linear.x, twist.linear.y, twist.angular.z)
        valid = all(finite_number(value) for value in velocities)
        moving = (not valid or
                  abs(float(twist.linear.x)) >= self.motion_linear_min or
                  abs(float(twist.linear.y)) >= self.motion_linear_min or
                  abs(float(twist.angular.z)) >= self.motion_angular_min)
        now = time.time()
        with self.lock:
            if self.motion_topic in self.last:
                self.last[self.motion_topic] = now
            self.motion_error = ("odometry twist is non-finite"
                                 if not valid else "")
            if moving:
                self.stationary_since = None
            elif self.robot_moving or self.stationary_since is None:
                self.stationary_since = now
            self.robot_moving = moving

    def cb_tf(self, message):
        now = time.time()
        camera_edge = tf_edge("base_footprint", self.camera_frame)
        for transform in getattr(message, "transforms", []):
            edge = tf_edge(transform.header.frame_id,
                           transform.child_frame_id)
            if edge not in self.tf_last:
                continue
            if edge == camera_edge:
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                valid = camera_transform_matches(
                    self.camera_mount, transform.header.frame_id,
                    transform.child_frame_id,
                    (translation.x, translation.y, translation.z),
                    (rotation.x, rotation.y, rotation.z, rotation.w),
                    expected_child=self.camera_frame)
                if not valid:
                    with self.lock:
                        self.camera_tf_error = (
                            "base_footprint -> %s TF does not match "
                            "camera_mount" % self.camera_frame)
                    continue
            with self.lock:
                self.tf_last[edge] = now

    def _health(self):
        now = time.time()
        with self.lock:
            missing_topics = sorted(
                topic for topic, stamp in self.last.items()
                if stamp is None)
            stale_topics = sorted(
                topic for topic, stamp in self.last.items()
                if stamp is not None and
                now - stamp > self.topic_stale_timeout)
            missing_tf = sorted(
                "%s->%s" % edge for edge, stamp in self.tf_last.items()
                if stamp is None)
            stale_tf = sorted(
                "%s->%s" % edge for edge, stamp in self.tf_last.items()
                if stamp is not None and now - stamp > self.tf_stale_timeout)
            camera_error = self.camera_tf_error
            if not camera_error:
                camera_error = self.camera_input_error or self.motion_error
            if self.stereo_pair_last is None:
                missing_topics.append("stereo_pair<=%.0fms" %
                                      (self.max_pair_dt_s * 1000.0))
            elif now - self.stereo_pair_last > self.topic_stale_timeout:
                stale_topics.append("stereo_pair<=%.0fms" %
                                    (self.max_pair_dt_s * 1000.0))
            stationary_since = self.stationary_since
            moving = self.robot_moving
        return (missing_topics, stale_topics, missing_tf, stale_tf,
                camera_error, moving, stationary_since)

    def _ensure_disk(self, minimum_gib=None, phase="capture"):
        parent = os.path.dirname(self.paths["final"]) or "."
        required_gib = (self.min_free_gib if minimum_gib is None else
                        float(minimum_gib))
        minimum = int(required_gib * (1024 ** 3))
        available = free_space_bytes(parent)
        if available < minimum:
            raise RuntimeError(
                "insufficient %s disk space: %.2f GiB < %.2f GiB" %
                (phase, available / float(1024 ** 3), required_gib))

    def wait_ready(self):
        deadline = time.time() + self.startup_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            health = self._health()
            missing, stale, missing_tf, stale_tf, camera_error = health[:5]
            if camera_error:
                raise RuntimeError(camera_error)
            if not missing and not stale and not missing_tf and not stale_tf:
                self._ensure_disk(self.min_free_gib, "startup")
                return
            rospy.loginfo_throttle(
                3.0,
                "SLAM capture waiting: missing=%s stale=%s tf_missing=%s "
                "tf_stale=%s",
                ",".join(missing) or "-", ",".join(stale) or "-",
                ",".join(missing_tf) or "-", ",".join(stale_tf) or "-")
            time.sleep(0.20)
        raise RuntimeError("SLAM capture inputs not ready within %.1fs" %
                           self.startup_timeout)

    def _wait_stationary(self):
        deadline = time.time() + self.completion_wait_s
        while not rospy.is_shutdown() and time.time() < deadline:
            self.stop_pub.publish(Twist())
            health = self._health()
            moving, stationary_since = health[5], health[6]
            if (not moving and stationary_since is not None and
                    time.time() - stationary_since >= self.map_settle_s):
                return True
            time.sleep(0.10)
        return False

    def _command_stop(self):
        """Latch the base inhibit and repeatedly publish a zero command."""
        self.inhibit_pub.publish(Bool(data=True))
        for _index in range(12):
            self.stop_pub.publish(Twist())
            time.sleep(0.025)

    def _assert_completion_ready(self):
        if rospy.is_shutdown():
            raise RuntimeError("ROS shutdown started during completion")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("rosbag recorder is not running")
        health = self._health()
        missing, stale, missing_tf, stale_tf, camera_error = health[:5]
        moving, stationary_since = health[5], health[6]
        if camera_error or missing or stale or missing_tf or stale_tf:
            raise RuntimeError(
                camera_error or
                "capture unhealthy during completion: topics=%s/%s "
                "tf=%s/%s" %
                (",".join(missing) or "-", ",".join(stale) or "-",
                 ",".join(missing_tf) or "-",
                 ",".join(stale_tf) or "-"))
        if (moving or stationary_since is None or
                time.time() - stationary_since < self.map_settle_s):
            raise RuntimeError("robot did not remain stationary for map save")

    def _save_map(self):
        parent = os.path.dirname(self.map_file) or "."
        if not os.path.isdir(parent):
            os.makedirs(parent)
        if os.path.exists(self.map_file) or os.path.exists(self.map_image):
            raise RuntimeError("final map target appeared during capture")
        command = ["rosrun", "map_server", "map_saver",
                   "-f", self.map_prefix]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        deadline = time.time() + self.map_save_timeout
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.10)
        if process.poll() is None:
            process.terminate()
            time.sleep(0.50)
            if process.poll() is None:
                process.kill()
        output = process.communicate()[0]
        if not isinstance(output, str):
            output = output.decode("utf-8", "replace")
        if process.returncode != 0:
            raise RuntimeError(
                "map_saver failed with code %s: %s" %
                (str(process.returncode), output[-500:]))
        if (not wait_for_path(self.map_file, 3.0) or
                not wait_for_path(self.map_image, 3.0)):
            raise RuntimeError("map_saver did not produce YAML and PGM")
        identity = load_map_identity(self.map_file, require_zero_yaw=True)
        self.map_hash = identity["map_image_sha256"]
        self.map_manifest_hash = identity["map_manifest_sha256"]

    def _wait_record_subscribers(self):
        deadline = time.time() + 5.0
        publishers = (self.map_hash_pub, self.manifest_hash_pub,
                      self.status_pub)
        while time.time() < deadline:
            if all(publisher.get_num_connections() > 0
                   for publisher in publishers):
                return
            time.sleep(0.10)
        raise RuntimeError("rosbag did not subscribe to transaction topics")

    def _publish_completion_transaction(self):
        self._wait_record_subscribers()
        started = self._record("started")
        self.map_hash_pub.publish(String(data=self.map_hash))
        self.manifest_hash_pub.publish(String(data=self.map_manifest_hash))
        self._publish_status(started)
        time.sleep(0.25)
        # Duplicate STARTED/hash publications are allowed and protect against
        # a late transport connection.  COMPLETED remains exactly once.
        self.map_hash_pub.publish(String(data=self.map_hash))
        self.manifest_hash_pub.publish(String(data=self.map_manifest_hash))
        self._publish_status(started)
        time.sleep(0.25)
        self._assert_completion_ready()
        completed = self._record("completed")
        self._publish_status(completed)
        time.sleep(1.0)
        return completed

    def _stop_process(self):
        process = self.process
        if process is None:
            return None
        forced = False
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            deadline = time.time() + self.rosbag_stop_timeout
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.10)
            if process.poll() is None:
                forced = True
                process.terminate()
                deadline = time.time() + 2.0
                while process.poll() is None and time.time() < deadline:
                    time.sleep(0.10)
            if process.poll() is None:
                forced = True
                process.kill()
                process.wait()
        code = process.poll()
        self.process = None
        self.stop_forced = forced
        return code

    def _verify_staging_bag(self):
        import rosbag
        payloads = []
        image_hashes = set()
        manifest_hashes = set()
        observed_edges = set()
        with rosbag.Bag(self.paths["staging"], "r") as bag:
            minimums = required_bag_minimums(self.required_topics)
            counts = dict(
                (topic, bag.get_message_count(topic_filters=[topic]))
                for topic in minimums)
            insufficient = sorted(
                "%s=%d<%d" % (topic, counts[topic], minimums[topic])
                for topic in minimums
                if counts[topic] < minimums[topic])
            if insufficient:
                raise RuntimeError(
                    "staging bag is missing sensor data: %s" %
                    ", ".join(insufficient))
            left_count = counts.get(
                "/stereo/left/image_raw/compressed", 0)
            right_count = counts.get(
                "/stereo/right/image_raw/compressed", 0)
            if min(left_count, right_count) * 1.25 < max(
                    left_count, right_count):
                raise RuntimeError(
                    "staging bag stereo image counts are imbalanced: "
                    "left=%d right=%d" % (left_count, right_count))
            for topic, message, _stamp in bag.read_messages(topics=[
                    self.status_topic, "/survey/map_image_sha256",
                    "/survey/map_manifest_sha256", "/tf"]):
                if topic == self.status_topic:
                    payloads.append(str(message.data))
                elif topic == "/survey/map_image_sha256":
                    image_hashes.add(str(message.data).strip().lower())
                elif topic == "/survey/map_manifest_sha256":
                    manifest_hashes.add(str(message.data).strip().lower())
                elif topic == "/tf":
                    for transform in getattr(message, "transforms", []):
                        observed_edges.add(tf_edge(
                            transform.header.frame_id,
                            transform.child_frame_id))
        valid, detail = validate_recording_states(
            payloads, self.map_hash, self.map_manifest_hash)
        if not valid:
            raise RuntimeError("recording transaction invalid: %s" % detail)
        if image_hashes != set([self.map_hash]):
            raise RuntimeError("recorded map image identity is ambiguous")
        if manifest_hashes != set([self.map_manifest_hash]):
            raise RuntimeError("recorded map manifest identity is ambiguous")
        missing_edges = sorted(set(self.tf_last) - observed_edges)
        if missing_edges:
            raise RuntimeError("recorded TF chain is incomplete: %s" %
                               ", ".join("%s->%s" % edge
                                         for edge in missing_edges))

    def _complete(self):
        with self.finish_lock:
            if self.finished_state is not None:
                return self.finished_state == "completed"
            try:
                # A successful long recording is expected to consume part of
                # the startup headroom.  At completion only retain enough for
                # map_saver, bag indexing and the sidecar; the C++ recorder's
                # --min-space gate separately enforces its 700 MiB floor.
                self._ensure_disk(
                    self.finalize_free_gib, "finalization")
                self._assert_completion_ready()
                self._save_map()
                self._assert_completion_ready()
                completed = self._publish_completion_transaction()
                code = self._stop_process()
                if self.stop_forced or code not in (0, -signal.SIGINT):
                    raise RuntimeError(
                        "rosbag did not stop cleanly (code=%s forced=%s)" %
                        (str(code), str(self.stop_forced)))
                if not wait_for_path(
                        self.paths["staging"], self.bag_finalize_timeout):
                    raise RuntimeError("staging rosbag was not finalized")
                self._verify_staging_bag()
                identity = load_map_identity(
                    self.map_file, require_zero_yaw=True)
                if (identity["map_image_sha256"] != self.map_hash or
                        identity["map_manifest_sha256"] !=
                        self.map_manifest_hash):
                    raise RuntimeError(
                        "saved map changed before formal commit")
                completed["bag_sha256"] = file_sha256(
                    self.paths["staging"])
                completed["bag_bytes"] = os.path.getsize(
                    self.paths["staging"])
                sidecar_temp = self._prepare_json(
                    self.paths["complete"], completed)
                sidecar_committed = False
                try:
                    # The sidecar may briefly exist without a final bag, which
                    # no consumer can accept.  The reverse ordering would
                    # expose a formally completed bag if sidecar creation ran
                    # out of space.
                    commit_file_no_replace(
                        sidecar_temp, self.paths["complete"])
                    sidecar_temp = None
                    sidecar_committed = True
                    commit_file_no_replace(
                        self.paths["staging"], self.paths["final"])
                except Exception:
                    if sidecar_committed and os.path.exists(
                            self.paths["complete"]):
                        try:
                            os.unlink(self.paths["complete"])
                        except OSError:
                            pass
                    raise
                finally:
                    if sidecar_temp and os.path.exists(sidecar_temp):
                        os.unlink(sidecar_temp)
                self.finished_state = "completed"
                if os.path.exists(self.paths["active"]):
                    try:
                        os.unlink(self.paths["active"])
                    except OSError as error:
                        rospy.logwarn(
                            "could not remove capture active marker: %s",
                            str(error))
                rospy.loginfo(
                    "one-pass SLAM capture completed: %s", self.paths["final"])
                return True
            except Exception as error:
                self._fail_locked("completion failed: %s" % error)
                return False

    def _fail_locked(self, reason):
        if self.finished_state is not None:
            return
        record = self._record("failed", reason)
        try:
            if self.process is not None and self.process.poll() is None:
                self._publish_status(record)
                time.sleep(0.25)
        except Exception:
            pass
        try:
            self._stop_process()
        except Exception:
            pass
        record["staging_bag"] = self.paths["staging"]
        quarantined = []
        for path in (self.map_file, self.map_image):
            if not os.path.exists(path):
                continue
            target = "%s.%s.failed" % (path, self.session_token)
            try:
                os.rename(path, target)
                quarantined.append(target)
            except OSError:
                pass
        if quarantined:
            record["quarantined_map_files"] = quarantined
        try:
            self._atomic_json(self.paths["failed"], record)
        except Exception as error:
            rospy.logerr("could not write capture failure sidecar: %s",
                         str(error))
        finally:
            if os.path.exists(self.paths["active"]):
                try:
                    os.unlink(self.paths["active"])
                except OSError:
                    pass
            self.finished_state = "failed"

    def _fail(self, reason):
        with self.finish_lock:
            self._fail_locked(reason)

    def cb_complete(self, _request):
        if not self.request_lock.acquire(False):
            return self._TriggerResponse(
                success=False, message="completion already in progress")
        try:
            if self.completion_requested:
                return self._TriggerResponse(
                    success=False, message="completion already requested")
            health = self._health()
            missing, stale, missing_tf, stale_tf, camera_error = health[:5]
            if camera_error or missing or stale or missing_tf or stale_tf:
                return self._TriggerResponse(
                    success=False,
                    message=(camera_error or
                             "capture unhealthy: topics=%s/%s tf=%s/%s" %
                             (",".join(missing) or "-",
                              ",".join(stale) or "-",
                              ",".join(missing_tf) or "-",
                              ",".join(stale_tf) or "-")))
            if self.process is None or self.process.poll() is not None:
                return self._TriggerResponse(
                    success=False, message="rosbag recorder is not running")
            self._command_stop()
            if not self._wait_stationary():
                return self._TriggerResponse(
                    success=False,
                    message="stop the robot and wait %.1fs before completion" %
                            self.map_settle_s)
            self.completion_requested = True
            success = self._complete()
            return self._TriggerResponse(
                success=bool(success),
                message=("SLAM map and bag committed" if success else
                         "capture completion failed; inspect .failed.json"))
        finally:
            self.request_lock.release()

    def _on_shutdown(self):
        if self.finished_state is None:
            self._fail("uncontrolled shutdown before explicit completion")

    def run(self):
        try:
            bag_parent = os.path.dirname(self.paths["final"]) or "."
            map_parent = os.path.dirname(self.map_file) or "."
            for parent in (bag_parent, map_parent):
                if not os.path.isdir(parent):
                    os.makedirs(parent)
            self.wait_ready()
            active = self._record("recording")
            active["staging_bag"] = self.paths["staging"]
            active["map_target"] = self.map_file
            self._atomic_json(self.paths["active"], active)
            command = rosbag_record_command(
                find_rosbag_record_executable(), self.paths["staging"],
                self.topics, self.rosbag_min_space)
            self.process = subprocess.Popen(command)
            time.sleep(1.0)
            if self.process.poll() is not None:
                raise RuntimeError(
                    "rosbag record exited during startup: %s" %
                    str(self.process.poll()))
            rospy.loginfo(
                "one-pass SLAM capture recording to staging bag %s",
                self.paths["staging"])
            while not rospy.is_shutdown():
                if self.completion_requested:
                    if self.finished_state in ("completed", "failed"):
                        break
                    time.sleep(0.05)
                    continue
                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError("rosbag recorder exited unexpectedly")
                health = self._health()
                missing, stale, missing_tf, stale_tf, camera_error = health[:5]
                if camera_error:
                    raise RuntimeError(camera_error)
                if stale:
                    raise RuntimeError(
                        "capture topics went stale: %s" % ", ".join(stale))
                if stale_tf:
                    raise RuntimeError(
                        "capture TF went stale: %s" % ", ".join(stale_tf))
                if missing or missing_tf:
                    raise RuntimeError("capture health regressed")
                time.sleep(0.20)
        except Exception as error:
            self._fail(str(error))
            raise


def main():
    rospy.init_node("checked_slam_capture")
    CheckedSlamCapture().run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("one-pass SLAM capture failed: %s", str(error))
        sys.exit(1)
