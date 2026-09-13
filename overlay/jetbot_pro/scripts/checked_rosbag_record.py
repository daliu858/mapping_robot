#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Record a survey only while sensors/localization pass explicit checks.

The recorder writes a STARTED and exactly one COMPLETED transaction marker
inside a cleanly closed bag.  Any detected failure leaves ``.failed.json``
and, when the bag is writable, a FAILED marker.  Formal replay can therefore
reject a partial bag instead of mistaking it for a successful survey.
"""
from __future__ import print_function

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

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Empty as EmptySrv
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from map_identity import load_map_identity
from replay_contract import (camera_transform_matches,
                             canonical_camera_mount)


def _finite(value):
    return not math.isnan(value) and not math.isinf(value)


def localization_is_healthy(ready, bad_samples, bad_limit, moving,
                            last_good, now, moving_timeout,
                            motion_started=None):
    """Steady-state AMCL policy, kept pure for host-side regression tests.

    Once startup has seen several fresh, low-covariance samples, a still
    robot may keep recording.  Bag 19 sat ~0.27 m from furniture: AMCL kept
    republishing high-covariance poses, ``bad_samples`` climbed, and the
    old stationary-after-grace gate aborted the required recorder, which
    tore down the whole stack before the tour could call ``~complete``.
    A good sample inside ``moving_timeout`` still covers brief spikes
    *while driving* and the first moments after stopping.  Motion without
    recovery still revokes health: after the grace, too many bad samples
    or a stale last-good / motion-start still fail closed.
    """
    if not ready:
        return False
    if last_good is not None and now - last_good <= moving_timeout:
        return True
    if not moving:
        return True
    if bad_samples >= bad_limit:
        return False
    references = [value for value in (last_good, motion_started)
                  if value is not None]
    return bool(references) and now - max(references) <= moving_timeout


def localization_abort_message(moving, bad_samples, last_good, now):
    """Human-readable abort; the old text always said 'while moving'."""
    if last_good is None:
        age = "none"
    else:
        age = "%.1f" % (now - last_good)
    phase = "while moving" if moving else "while stationary"
    return (
        "AMCL localization became stale/uncertain %s "
        "(bad_samples=%s last_good_age_s=%s)" %
        (phase, bad_samples, age))


def motion_latch(sample_moving, hits, confirm_samples, was_moving, now,
                 started_at):
    """Debounce odom so one MCU tick cannot look like driving.

    Returns ``(hits, robot_moving, motion_started_at)``.  Still samples
    clear the latch immediately; moving needs ``confirm_samples`` hits.
    """
    confirm = max(1, int(confirm_samples))
    if not sample_moving:
        return 0, False, None
    hits = int(hits) + 1
    if hits < confirm:
        return hits, bool(was_moving), started_at
    if not was_moving:
        started_at = now
    return hits, True, started_at


def localization_sample_is_good(frame_id, covariance, stamp_s, now_s,
                                max_age_s, max_xy_std, max_yaw_std):
    """Accept a map-frame AMCL estimate with finite, bounded covariance.

    AMCL keeps the last *filter-update* stamp while the robot is still, even
    when it republishes /amcl_pose or answers request_nomotion_update.  Header
    age is therefore not a localization-quality signal; callback freshness is
    tracked separately.  Zero or future stamps are still rejected.
    """
    del max_age_s
    frame = str(frame_id or "").lstrip("/")
    if frame not in ("", "map"):
        return False
    if len(covariance) < 36:
        return False
    try:
        values = (float(covariance[0]), float(covariance[7]),
                  float(covariance[35]))
        stamp_s, now_s = float(stamp_s), float(now_s)
    except (TypeError, ValueError):
        return False
    if stamp_s <= 0.0 or stamp_s - now_s > 0.10:
        return False
    return (all(_finite(value) and value >= 0.0 for value in values) and
            math.sqrt(max(values[0], values[1])) <= float(max_xy_std) and
            math.sqrt(values[2]) <= float(max_yaw_std))


def normalize_bag_path(path):
    normalized = os.path.abspath(os.path.expanduser(str(path)))
    return normalized if normalized.lower().endswith(".bag") \
        else normalized + ".bag"


def wait_for_finalized_bag(path, timeout_s=10.0):
    """Wait for ``rosbag record`` to rename ``.active`` to the final bag.

    On the Jetson, the recorder process can report its SIGINT exit just before
    the filesystem rename becomes visible to this process.  Treating that
    short visibility race as a failed survey discards an otherwise complete
    recording, so completion waits for the final pathname for a bounded time.
    """
    deadline = time.time() + max(0.0, float(timeout_s))
    while not os.path.isfile(path) and time.time() < deadline:
        time.sleep(0.10)
    return os.path.isfile(path)


def anymsg_required_topics(required_topics, motion_odom_topic,
                           require_localization):
    """Return required topics that do not already have typed subscribers."""
    typed_topics = {str(motion_odom_topic)}
    if require_localization:
        typed_topics.add("/amcl_pose")
    return [topic for topic in required_topics
            if topic != "/tf" and topic not in typed_topics]


class CheckedRosbagRecorder(object):
    def __init__(self):
        self.bag = normalize_bag_path(rospy.get_param("~bag"))
        self.map_file = os.path.abspath(os.path.expanduser(
            rospy.get_param("~map_file")))
        self.topics = self._csv("~topics_csv")
        self.required_topics = self._csv("~required_topics_csv")
        self.require_localization = bool(
            rospy.get_param("~require_localization", True))
        self.startup_timeout = float(rospy.get_param("~startup_timeout_s", 60.0))
        self.stale_timeout = float(rospy.get_param("~stale_timeout_s", 2.0))
        self.max_xy_std = float(rospy.get_param("~max_amcl_xy_std_m", 0.75))
        self.max_yaw_std = float(rospy.get_param("~max_amcl_yaw_std_rad", 0.40))
        self.stable_required = int(rospy.get_param("~amcl_stable_samples", 1))
        self.bad_limit = int(rospy.get_param("~amcl_bad_samples", 3))
        self.motion_timeout = float(rospy.get_param(
            "~amcl_motion_timeout_s", 10.0))
        self.motion_odom_topic = str(rospy.get_param(
            "~motion_odom_topic", "/odom_raw"))
        self.motion_linear_min = float(rospy.get_param(
            "~motion_linear_min_mps", 0.02))
        self.motion_angular_min = float(rospy.get_param(
            "~motion_angular_min_radps", 0.03))
        self.motion_confirm_samples = int(rospy.get_param(
            "~motion_confirm_samples", 3))
        self.status_topic = str(rospy.get_param(
            "~recording_status_topic", "/survey/recording_status"))
        self.camera_mount = canonical_camera_mount(
            rospy.get_param("~camera_mount"))
        self.camera_optical_frame = str(rospy.get_param(
            "~camera_optical_frame"))
        if not self.camera_optical_frame.strip("/"):
            raise ValueError("~camera_optical_frame must not be empty")
        if not self.topics or not self.required_topics:
            raise ValueError("record/required topic lists must not be empty")
        if self.status_topic not in self.topics:
            raise ValueError("recording status topic must be in ~topics_csv")
        if self.stable_required < 1 or self.bad_limit < 1:
            raise ValueError("AMCL sample counts must be positive")
        if self.motion_confirm_samples < 1:
            raise ValueError("~motion_confirm_samples must be positive")

        self.complete_file = self.bag + ".complete.json"
        self.failed_file = self.bag + ".failed.json"
        self.active_file = self.bag + ".active"
        conflicts = [path for path in
                     (self.bag, self.active_file, self.complete_file,
                      self.failed_file) if os.path.exists(path)]
        if conflicts:
            raise ValueError(
                "survey output already exists; choose a new bag: %s" %
                ", ".join(conflicts))

        self.last = {topic: None for topic in self.required_topics}
        self.camera_tf_error = ""
        self.lock = threading.Lock()
        self.finish_lock = threading.Lock()
        self.completion_request_lock = threading.Lock()
        self.localized_samples = 0
        self.bad_localization_samples = 0
        self.localization_ready = False
        self.last_localization = None
        self.last_good_localization = None
        self.robot_moving = False
        self.motion_hits = 0
        self.motion_started_at = None
        self.process = None
        self.last_stop_forced = False
        self.finished_state = None
        self.completion_requested = False
        self.started_at = time.time()
        self.session_token = uuid.uuid4().hex
        map_identity = load_map_identity(
            self.map_file, require_zero_yaw=True)
        self.map_hash = map_identity["map_image_sha256"]
        self.map_manifest_hash = map_identity["map_manifest_sha256"]

        self.map_hash_pub = rospy.Publisher(
            "/survey/map_image_sha256", String, queue_size=1, latch=True)
        self.map_hash_pub.publish(String(data=self.map_hash))
        self.map_manifest_hash_pub = rospy.Publisher(
            "/survey/map_manifest_sha256", String, queue_size=1, latch=True)
        self.map_manifest_hash_pub.publish(
            String(data=self.map_manifest_hash))
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=2, latch=True)
        self._publish_status("started")
        from std_srvs.srv import Trigger, TriggerResponse
        self._TriggerResponse = TriggerResponse
        self.complete_service = rospy.Service(
            "~complete", Trigger, self.cb_complete)
        # Controlled abort for supervisors (the survey tour's crash handler).
        # Without it a crashed tour leaves a detached recorder holding
        # `.active` and filling the disk until someone kills roslaunch.
        self.fail_service = rospy.Service(
            "~fail", Trigger, self.cb_fail)
        generic_required_topics = anymsg_required_topics(
            self.required_topics, self.motion_odom_topic,
            self.require_localization)
        self.subscribers = [
            rospy.Subscriber(topic, rospy.AnyMsg, self.cb_topic,
                             callback_args=topic, queue_size=1)
            for topic in generic_required_topics]
        if "/tf" in self.required_topics:
            from tf2_msgs.msg import TFMessage
            self.camera_tf_subscriber = rospy.Subscriber(
                "/tf", TFMessage, self.cb_camera_tf, queue_size=10)
        self.motion_sub = rospy.Subscriber(
            self.motion_odom_topic, Odometry, self.cb_motion, queue_size=5)
        if self.require_localization:
            self.localization_sub = rospy.Subscriber(
                "/amcl_pose", PoseWithCovarianceStamped,
                self.cb_localization, queue_size=5)
        rospy.on_shutdown(self._on_shutdown)

    @staticmethod
    def _csv(param):
        return [item.strip() for item in
                str(rospy.get_param(param, "")).split(",") if item.strip()]

    def _status_record(self, state, reason=""):
        record = {
            "schema": 1,
            "state": state,
            "token": self.session_token,
            "pose_source": "amcl",
            "map_image_sha256": self.map_hash,
            "map_manifest_sha256": self.map_manifest_hash,
            "camera_mount": self.camera_mount,
            "started_unix": self.started_at,
        }
        if state in ("completed", "failed"):
            record["finished_unix"] = time.time()
        if reason:
            record["reason"] = str(reason)
        return record

    def _publish_status(self, state, reason=""):
        record = self._status_record(state, reason)
        self.status_pub.publish(String(data=json.dumps(
            record, sort_keys=True, separators=(",", ":"))))
        return record

    @staticmethod
    def _atomic_json(path, record):
        parent = os.path.dirname(path) or "."
        if not os.path.isdir(parent):
            os.makedirs(parent)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", prefix=".survey-status.", suffix=".tmp",
                    dir=parent, delete=False) as stream:
                temporary = stream.name
                json.dump(record, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.rename(temporary, path)
            temporary = None
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def _append_status_to_bag(self, record):
        # rospy Bag("a") emits INDEX_DATA with count-first headers;
        # C++ rosbag record uses conn-first. PC `rosbags` caches the
        # first layout, so complete bags need rosbags_index_compat.
        import rosbag
        payload = String(data=json.dumps(
            record, sort_keys=True, separators=(",", ":")))
        with rosbag.Bag(self.bag, "a") as bag:
            bag.write(self.status_topic, payload,
                      rospy.Time.from_sec(float(record["finished_unix"])))

    def cb_topic(self, _message, topic):
        with self.lock:
            self.last[topic] = time.time()

    def cb_camera_tf(self, message):
        target = self.camera_optical_frame.strip("/")
        for transform in getattr(message, "transforms", []):
            child = str(transform.child_frame_id).strip("/")
            if child != target:
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            valid = camera_transform_matches(
                self.camera_mount, transform.header.frame_id,
                transform.child_frame_id,
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w),
                expected_child=self.camera_optical_frame)
            with self.lock:
                if valid:
                    self.last["/tf"] = time.time()
                else:
                    self.camera_tf_error = (
                        "recorded base_footprint -> %s TF does not match "
                        "~camera_mount" % self.camera_optical_frame)

    def cb_motion(self, msg):
        twist = msg.twist.twist
        sample_moving = (
            abs(float(twist.linear.x)) >= self.motion_linear_min or
            abs(float(twist.linear.y)) >= self.motion_linear_min or
            abs(float(twist.angular.z)) >= self.motion_angular_min)
        now = time.time()
        with self.lock:
            if self.motion_odom_topic in self.last:
                self.last[self.motion_odom_topic] = now
            was_moving = self.robot_moving
            self.motion_hits, self.robot_moving, self.motion_started_at = (
                motion_latch(
                    sample_moving, self.motion_hits,
                    self.motion_confirm_samples, was_moving, now,
                    self.motion_started_at))
            # Sitting at a wall fills bad_samples; those must not instantly
            # fail the first second of the next drive.
            if self.robot_moving and not was_moving:
                self.bad_localization_samples = 0

    def cb_localization(self, msg):
        cov = msg.pose.covariance
        now = time.time()
        try:
            ros_now = rospy.Time.now().to_sec()
            stamp_s = msg.header.stamp.to_sec()
        except Exception:
            ros_now = stamp_s = 0.0
        good = localization_sample_is_good(
            msg.header.frame_id, cov, stamp_s, ros_now,
            self.stale_timeout, self.max_xy_std, self.max_yaw_std)
        with self.lock:
            if "/amcl_pose" in self.last:
                self.last["/amcl_pose"] = now
            self.last_localization = now
            if good:
                self.localized_samples += 1
                self.bad_localization_samples = 0
                self.last_good_localization = now
                if self.localized_samples >= self.stable_required:
                    self.localization_ready = True
            else:
                self.localized_samples = 0
                self.bad_localization_samples += 1

    def _stop_process(self):
        process = self.process
        if process is None:
            return None
        forced = False
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
                deadline = time.time() + 5.0
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
            except Exception:
                if process.poll() is None:
                    process.terminate()
        code = process.poll()
        self.process = None
        self.last_stop_forced = forced
        return code

    def _fail(self, reason):
        with self.finish_lock:
            if self.finished_state is not None:
                return
            record = self._status_record("failed", reason)
            try:
                self.status_pub.publish(String(data=json.dumps(
                    record, sort_keys=True, separators=(",", ":"))))
            except Exception:
                pass
            # Give rosbag's subscriber a chance to receive FAILED, then close
            # and append it directly as a second, reliable path.
            time.sleep(0.10)
            self._stop_process()
            try:
                if os.path.isfile(self.bag):
                    self._append_status_to_bag(record)
            except Exception as error:
                record["marker_append_error"] = str(error)
            self._atomic_json(self.failed_file, record)
            try:
                if os.path.exists(self.active_file):
                    os.unlink(self.active_file)
            except OSError:
                pass
            self.finished_state = "failed"

    def _complete(self):
        with self.finish_lock:
            if self.finished_state is not None:
                return self.finished_state == "completed"
            code = self._stop_process()
            record = self._status_record("completed")
            try:
                if self.last_stop_forced or code not in (0, -signal.SIGINT):
                    raise RuntimeError(
                        "rosbag did not stop cleanly (code=%s forced=%s)" %
                        (str(code), str(self.last_stop_forced)))
                if not wait_for_finalized_bag(self.bag):
                    raise RuntimeError("rosbag output was not finalized")
                self._append_status_to_bag(record)
                self._atomic_json(self.complete_file, record)
                try:
                    if os.path.exists(self.active_file):
                        os.unlink(self.active_file)
                except OSError:
                    pass
                self.finished_state = "completed"
                rospy.loginfo("survey recording completed: %s", self.bag)
                return True
            except Exception as error:
                record["state"] = "failed"
                record["reason"] = "completion marker failed: %s" % error
                self._atomic_json(self.failed_file, record)
                self.finished_state = "failed"
                rospy.logerr("survey completion failed: %s", str(error))
                return False

    def _on_shutdown(self):
        # A ROS shutdown can mean a required sensor/TF node died.  It is never
        # authoritative success; only the explicit ~complete service may
        # commit COMPLETED.
        if self.finished_state is None:
            self._fail(
                "uncontrolled ROS shutdown before explicit survey completion")

    def cb_complete(self, _request):
        """Commit only an explicit, healthy operator completion request."""
        if not self.completion_request_lock.acquire(False):
            return self._TriggerResponse(
                success=False, message="survey completion already requested")
        try:
            if self.completion_requested:
                return self._TriggerResponse(
                    success=False,
                    message="survey completion already requested")
            with self.lock:
                camera_tf_error = self.camera_tf_error
            if camera_tf_error:
                return self._TriggerResponse(
                    success=False, message=camera_tf_error)
            missing, stale, localized = self._health(startup=False)
            if missing or stale:
                return self._TriggerResponse(
                    success=False,
                    message="survey is unhealthy: missing=%s stale=%s" %
                            (",".join(missing) or "-",
                             ",".join(stale) or "-"))
            if self.require_localization and not localized:
                return self._TriggerResponse(
                    success=False, message="AMCL localization is not healthy")
            process = self.process
            if process is None or process.poll() is not None:
                return self._TriggerResponse(
                    success=False, message="rosbag recorder is not running")
            self.completion_requested = True
            success = self._complete()
            return self._TriggerResponse(
                success=bool(success),
                message=("survey completed" if success else
                         "survey completion failed; inspect .failed.json"))
        finally:
            self.completion_request_lock.release()

    def cb_fail(self, _request):
        """Mark the recording failed on behalf of a crashed supervisor.

        Reading finished_state outside finish_lock is safe: it is only ever
        written under finish_lock and _fail() rechecks it there, so a race
        with an in-flight completion resolves through the lock either way
        (fail-closed).
        """
        if not self.completion_request_lock.acquire(False):
            return self._TriggerResponse(
                success=False, message="survey completion already in progress")
        try:
            if self.finished_state is not None:
                return self._TriggerResponse(
                    success=False,
                    message="survey already finished: %s" % self.finished_state)
            self._fail("external abort request (survey tour reported failure)")
            return self._TriggerResponse(
                success=True, message="survey recording marked failed")
        finally:
            self.completion_request_lock.release()

    def _health(self, startup=False):
        now = time.time()
        with self.lock:
            missing = [topic for topic, stamp in self.last.items()
                       if stamp is None]
            stale = [topic for topic, stamp in self.last.items()
                     if stamp is not None and now - stamp > self.stale_timeout]
            if startup:
                # AMCL only republishes after motion.  A still robot therefore
                # has a good but aging /amcl_pose stamp; requiring 2 s freshness
                # here prevents the survey from ever starting.
                got_good = (
                    self.localized_samples >= self.stable_required and
                    self.bad_localization_samples == 0 and
                    self.last_good_localization is not None)
                if not got_good:
                    localized = False
                elif self.robot_moving:
                    localized = (
                        now - self.last_good_localization <= self.stale_timeout)
                else:
                    localized = True
            else:
                localized = localization_is_healthy(
                    self.localization_ready,
                    self.bad_localization_samples, self.bad_limit,
                    self.robot_moving, self.last_good_localization,
                    now, self.motion_timeout, self.motion_started_at)
        return missing, stale, localized

    def _request_nomotion_update(self):
        """Force AMCL to publish a fresh pose while the robot is still."""
        if not self.require_localization:
            return
        try:
            rospy.ServiceProxy("/request_nomotion_update", EmptySrv)()
        except Exception:
            pass

    def wait_ready(self):
        deadline = time.time() + self.startup_timeout
        next_nomotion = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                camera_tf_error = self.camera_tf_error
            if camera_tf_error:
                raise RuntimeError(camera_tf_error)
            now = time.time()
            # One nomotion per second is enough to refresh a still pose.
            # A tight loop starves /scan callbacks on the Nano.
            if now >= next_nomotion:
                self._request_nomotion_update()
                next_nomotion = now + 1.0
            missing, stale, localized = self._health(startup=True)
            if not missing and not stale and (
                    localized or not self.require_localization):
                return
            rospy.loginfo_throttle(
                3.0,
                "survey waiting: missing=%s stale=%s amcl_ready=%s "
                "samples=%s bad=%s moving=%s",
                ",".join(missing) or "-", ",".join(stale) or "-",
                str(localized or not self.require_localization),
                str(self.localized_samples),
                str(self.bad_localization_samples),
                str(self.robot_moving))
            time.sleep(0.20)
        if rospy.is_shutdown():
            raise RuntimeError("ROS shutdown before survey inputs were ready")
        raise RuntimeError("survey inputs/AMCL not ready within %.1fs" %
                           self.startup_timeout)

    def run(self):
        try:
            self.wait_ready()
            parent = os.path.dirname(self.bag)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            self._atomic_json(
                self.active_file, self._status_record("started"))
            command = ["rosbag", "record", "-O", self.bag] + self.topics
            rospy.loginfo("survey healthy; recording to %s (map sha256=%s)",
                          self.bag, self.map_hash[:12])
            self.process = subprocess.Popen(command)
            # Re-publish the latched start marker after rosbag has subscribed.
            self._publish_status("started")
            while not rospy.is_shutdown():
                # Ctrl-C / required-node death must fail-closed via
                # _on_shutdown, not a misleading "rosbag exited 0" race.
                if rospy.is_shutdown():
                    break
                if self.completion_requested:
                    if self.finished_state == "completed":
                        break
                    if self.finished_state == "failed":
                        raise RuntimeError("explicit survey completion failed")
                    time.sleep(0.05)
                    continue
                process = self.process
                if process is None:
                    if self.completion_requested:
                        time.sleep(0.05)
                        continue
                    if (rospy.is_shutdown() or
                            self.finished_state is not None):
                        break
                    raise RuntimeError("rosbag process disappeared")
                code = process.poll()
                if code is not None:
                    if self.completion_requested:
                        time.sleep(0.05)
                        continue
                    if self.process is process:
                        self.process = None
                    raise RuntimeError(
                        "rosbag record exited unexpectedly: %d" % code)
                with self.lock:
                    camera_tf_error = self.camera_tf_error
                if camera_tf_error:
                    raise RuntimeError(camera_tf_error)
                _missing, stale, localized = self._health(startup=False)
                if stale:
                    raise RuntimeError("survey topics went stale: %s" %
                                       ", ".join(sorted(stale)))
                if self.require_localization and not localized:
                    with self.lock:
                        message = localization_abort_message(
                            self.robot_moving, self.bad_localization_samples,
                            self.last_good_localization, time.time())
                    raise RuntimeError(message)
                time.sleep(0.20)
        except Exception as error:
            self._fail(str(error))
            raise


def main():
    rospy.init_node("checked_rosbag_record")
    CheckedRosbagRecorder().run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("survey recording failed: %s", str(error))
        sys.exit(1)
