#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run rosbag once, drain fusion queues, then require a successful map save."""
from __future__ import print_function

import os
import subprocess
import sys
import time

import rospy
from jetbot_pro.srv import FinalizeSemanticMap
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from map_identity import load_map_identity
from replay_contract import (camera_transform_matches,
                             canonical_camera_mount, file_sha256,
                             load_detection_contract, replay_abort_reason,
                             validate_recording_states,
                             validate_stereo_source_contract)


def validate_recording_bag(path, status_topic, expected_map_hash=None,
                           expected_manifest_hash=None,
                           expected_optical_frame="stereo_left_optical"):
    """Read the three small identity topics before allowing formal replay."""
    import rosbag
    payloads = []
    image_hashes = set()
    manifest_hashes = set()
    image_topic = "/survey/map_image_sha256"
    manifest_topic = "/survey/map_manifest_sha256"
    camera_transforms = []
    with rosbag.Bag(path, "r") as bag:
        for topic, message, _stamp in bag.read_messages(
                topics=[status_topic, image_topic, manifest_topic, "/tf"]):
            if topic == "/tf":
                for transform in getattr(message, "transforms", []):
                    if (str(transform.child_frame_id).strip("/") !=
                            str(expected_optical_frame).strip("/")):
                        continue
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation
                    camera_transforms.append((
                        transform.header.frame_id,
                        transform.child_frame_id,
                        (translation.x, translation.y, translation.z),
                        (rotation.x, rotation.y, rotation.z, rotation.w)))
                continue
            value = str(message.data)
            if topic == status_topic:
                payloads.append(value)
            elif topic == image_topic:
                image_hashes.add(value.strip().lower())
            elif topic == manifest_topic:
                manifest_hashes.add(value.strip().lower())
    valid, detail = validate_recording_states(
        payloads, expected_map_hash, expected_manifest_hash)
    if not valid:
        return valid, detail
    completion_image = str(detail.get("map_image_sha256", "")).lower()
    completion_manifest = str(
        detail.get("map_manifest_sha256", "")).lower()
    if image_hashes != set([completion_image]):
        return False, "map image identity topic/status mismatch"
    if manifest_hashes != set([completion_manifest]):
        return False, "map manifest identity topic/status mismatch"
    if not camera_transforms:
        return False, "bag has no recorded base-to-camera TF"
    if any(not camera_transform_matches(
            detail["camera_mount"], parent, child, translation, quaternion,
            expected_child=expected_optical_frame)
            for parent, child, translation, quaternion in camera_transforms):
        return False, "recorded camera TF does not match camera_mount"
    return True, detail


class OfflineReplayRunner(object):
    def __init__(self):
        self.bag = os.path.abspath(os.path.expanduser(
            rospy.get_param("~bag")))
        self.detections_file = os.path.abspath(os.path.expanduser(
            rospy.get_param("~detections_file")))
        self.detection_contract = load_detection_contract(
            self.detections_file)
        self.rate = float(rospy.get_param("~rate", 0.25))
        self.start_delay = float(rospy.get_param("~start_delay_s", 2.0))
        self.finalize_timeout = float(rospy.get_param(
            "~finalize_timeout_s", 120.0))
        self.finalize_retry = float(rospy.get_param(
            "~finalize_retry_s", 0.5))
        self.finalize_service = rospy.get_param(
            "~finalize_service", "/semantic_mapper/finalize")
        self.finalize_token = str(rospy.get_param("~finalize_token"))
        self.require_complete_marker = bool(
            rospy.get_param("~require_complete_marker", True))
        self.recording_status_topic = str(rospy.get_param(
            "~recording_status_topic", "/survey/recording_status"))
        self.detection_ready_service = str(rospy.get_param(
            "~detection_ready_service", "/detection_publisher/ready"))
        self.detection_complete_service = str(rospy.get_param(
            "~detection_complete_service",
            "/detection_publisher/complete_input"))
        self.pipeline_status_service = str(rospy.get_param(
            "~pipeline_status_service", "/semantic_mapper/pipeline_status"))
        self.invalid_poll = float(rospy.get_param("~invalid_poll_s", 2.0))
        if self.invalid_poll <= 0.0:
            raise ValueError("~invalid_poll_s must be positive")
        self.map_file = os.path.abspath(os.path.expanduser(
            str(rospy.get_param("~map_file"))))
        self.map_identity = load_map_identity(
            self.map_file, require_zero_yaw=True)
        self.range_source = str(rospy.get_param("~range_source", "stereo"))
        source_values = {
            name: rospy.get_param("~" + name)
            for name in (
                "img_topic", "left_image_base", "right_image_base",
                "camera_info_topic", "disparity_topic", "optical_frame")}
        source_values["detections_rectified"] = bool(rospy.get_param(
            "~detections_rectified", False))
        source_values["stereo_lidar_fallback"] = bool(rospy.get_param(
            "~stereo_lidar_fallback", False))
        self.optical_frame = source_values["optical_frame"]
        validate_stereo_source_contract(self.range_source, source_values)
        if not self.finalize_token:
            raise ValueError("~finalize_token must not be empty")
        if self.finalize_timeout <= 0.0 or self.finalize_retry <= 0.0:
            raise ValueError("finalize timeout/retry must be positive")
        self.process = None
        rospy.on_shutdown(self.stop)

    def stop(self):
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        deadline = time.time() + 3.0
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.10)
        if process.poll() is None:
            process.kill()

    def wait_for_replay_exit(self, status_proxy):
        """Wait for rosbag play, aborting once the mapper locks INVALID.

        Bag 19 kept playing for ~90 minutes after the transaction had been
        permanently INVALID since second 8.  The mapper's read-only
        pipeline_status probe lets the runner stop paying for a bag that
        can no longer produce a yaml.  A failing/slow probe never kills a
        healthy replay -- the finalize barrier still fail-closes at EOF.
        """
        while True:
            code = self.process.poll()
            if code is not None:
                return code
            reason = None
            try:
                status = status_proxy()
                reason = replay_abort_reason(status.success, status.message)
            except Exception:
                reason = None
            if reason is not None:
                self.stop()
                raise RuntimeError(
                    "stopping replay early; the transaction is already "
                    "terminal and the remaining bag cannot change that: %s"
                    % reason)
            time.sleep(self.invalid_poll)

    def run(self):
        if not os.path.isfile(self.bag):
            raise RuntimeError("bag does not exist: %s" % self.bag)
        expected_bag_sha256 = str(self.detection_contract.get(
            "source_bag_sha256", "")).lower()
        if file_sha256(self.bag) != expected_bag_sha256:
            raise RuntimeError(
                "detections JSON was produced from different bag bytes")
        if self.rate <= 0.0:
            raise ValueError("replay rate must be positive")
        if self.require_complete_marker:
            valid, detail = validate_recording_bag(
                self.bag, self.recording_status_topic,
                self.map_identity["map_image_sha256"],
                self.map_identity["map_manifest_sha256"],
                self.optical_frame)
            if not valid:
                raise RuntimeError("incomplete/failed survey bag: %s" % detail)
            expected_detection_identity = (
                str(detail.get("token", "")),
                str(detail.get("map_image_sha256", "")).lower(),
                str(detail.get("map_manifest_sha256", "")).lower(),
                list(detail.get("camera_mount", [])),
            )
            actual_detection_identity = (
                str(self.detection_contract.get("recording_token", "")),
                str(self.detection_contract.get(
                    "map_image_sha256", "")).lower(),
                str(self.detection_contract.get(
                    "map_manifest_sha256", "")).lower(),
                list(self.detection_contract.get("camera_mount", [])),
            )
            if actual_detection_identity != expected_detection_identity:
                raise RuntimeError(
                    "detections JSON identity/camera_mount does not match bag")
            rospy.loginfo(
                "checked recording completion token=%s",
                str(detail.get("token", ""))[:12])
        rospy.loginfo("waiting for semantic finalize service: %s",
                      self.finalize_service)
        rospy.wait_for_service(self.finalize_service, timeout=30.0)
        rospy.loginfo("waiting for detection publisher readiness: %s",
                      self.detection_ready_service)
        rospy.wait_for_service(self.detection_ready_service, timeout=30.0)
        rospy.loginfo("waiting for detection publisher EOS service: %s",
                      self.detection_complete_service)
        rospy.wait_for_service(self.detection_complete_service, timeout=30.0)
        from std_srvs.srv import Trigger
        ready = rospy.ServiceProxy(
            self.detection_ready_service, Trigger)()
        if not ready.success:
            raise RuntimeError(
                "detection publisher is not ready: %s" % ready.message)
        command = ["rosbag", "play", "--clock", "--delay",
                   str(self.start_delay), "--rate", str(self.rate), self.bag]
        rospy.loginfo("starting one-shot replay: %s", self.bag)
        status_proxy = rospy.ServiceProxy(
            self.pipeline_status_service, Trigger)
        self.process = subprocess.Popen(command)
        code = self.wait_for_replay_exit(status_proxy)
        self.process = None
        if code != 0:
            raise RuntimeError("rosbag play exited with code %d" % code)
        if file_sha256(self.bag) != expected_bag_sha256:
            raise RuntimeError("bag bytes changed while replay was running")
        # First close the detection input explicitly.  The publisher proves
        # that every indexed bag frame has been validated and published before
        # unregistering its image subscriber, so no late detection can race a
        # successful semantic-map commit.
        complete_proxy = rospy.ServiceProxy(
            self.detection_complete_service, Trigger)
        deadline = time.time() + self.finalize_timeout
        complete_response = None
        while time.time() < deadline and not rospy.is_shutdown():
            complete_response = complete_proxy()
            if complete_response.success:
                break
            if not str(complete_response.message).startswith("NOT_READY:"):
                raise RuntimeError(
                    "detection publisher rejected EOS: %s" %
                    complete_response.message)
            rospy.loginfo_throttle(
                5.0, "waiting for detection publisher EOS: %s",
                complete_response.message)
            time.sleep(self.finalize_retry)
        if complete_response is None or not complete_response.success:
            detail = (complete_response.message
                      if complete_response is not None else "no response")
            raise RuntimeError(
                "detection publisher EOS timed out after %.1fs: %s" %
                (self.finalize_timeout, detail))

        # The input is now sealed.  The mapper still proves exact expected
        # stamp sets, zero in-flight callbacks, and its queue quiet period.
        proxy = rospy.ServiceProxy(
            self.finalize_service, FinalizeSemanticMap)
        deadline = time.time() + self.finalize_timeout
        response = None
        while time.time() < deadline and not rospy.is_shutdown():
            response = proxy(token=self.finalize_token)
            if response.success:
                break
            if not str(response.message).startswith("NOT_READY:"):
                raise RuntimeError(
                    "semantic replay rejected: %s" % response.message)
            rospy.loginfo_throttle(
                5.0, "waiting for replay barrier: %s", response.message)
            time.sleep(self.finalize_retry)
        if response is None or not response.success:
            detail = response.message if response is not None else "no response"
            raise RuntimeError(
                "semantic replay barrier timed out after %.1fs: %s" %
                (self.finalize_timeout, detail))
        rospy.loginfo("semantic replay complete: %s", response.message)


def main():
    rospy.init_node("offline_replay_runner")
    OfflineReplayRunner().run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("offline replay failed: %s", str(error))
        sys.exit(1)
