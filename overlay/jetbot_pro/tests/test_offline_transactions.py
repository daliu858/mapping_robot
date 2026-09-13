#!/usr/bin/env python
"""Host-side transaction tests; no ROS installation is required."""
from __future__ import print_function

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET

try:
    from importlib import util as importlib_util
except ImportError:
    importlib_util = None


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = types.ModuleType(parent_name)
            parent.__path__ = []
            sys.modules[parent_name] = parent
        setattr(parent, child_name, module)


def _load(name, filename):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", filename)
    if importlib_util is not None:
        spec = importlib_util.spec_from_file_location(name, path)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import imp
    return imp.load_source(name, path)


dummy = type("Dummy", (), {})


class _Message(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


_stub("rospy", ROSInterruptException=Exception)
_stub("geometry_msgs.msg", Twist=_Message, PoseWithCovarianceStamped=dummy)
_stub("nav_msgs.msg", Odometry=dummy)
_stub("sensor_msgs.msg", LaserScan=dummy, CameraInfo=dummy,
      CompressedImage=dummy)
_stub("std_msgs.msg", String=dummy, Bool=_Message)
_stub("tf2_msgs.msg", TFMessage=dummy)
_stub("std_srvs.srv", Empty=dummy)
_stub("jetbot_pro.srv", FinalizeSemanticMap=dummy)

RECORDER = _load("checked_slam_capture_test", "checked_slam_capture.py")
SURVEY = _load("checked_rosbag_record_test", "checked_rosbag_record.py")
RUNNER = _load("offline_replay_runner_test", "offline_replay_runner.py")
WATCHDOG = _load("scan_watchdog_test", "scan_watchdog.py")


def _payload(state, token="abc", reason=None, pose_source="gmapping"):
    data = {"schema": 1, "state": state, "token": token,
            "pose_source": pose_source,
            "map_image_sha256": "a" * 64,
            "map_manifest_sha256": "c" * 64,
            "camera_mount": [0.1, 0.0, 0.2, -1.5708, 0.0, -1.5708]}
    if reason:
        data["reason"] = reason
    return json.dumps(data)


class OfflineTransactionTest(unittest.TestCase):
    def test_motor_driver_crash_requires_explicit_full_stack_restart(self):
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(
            __file__)))
        launch_root = ET.parse(os.path.join(
            package_root, "launch", "jetbot.launch")).getroot()
        nodes = [node for node in launch_root.findall("node")
                 if node.get("name") == "jetbot"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].get("required"), "true")
        self.assertEqual(nodes[0].get("respawn"), "false")

    def test_bag_path_is_normalized_before_conflict_checks(self):
        self.assertTrue(RECORDER.normalize_bag_path("survey").endswith(
            "survey.bag"))
        self.assertTrue(RECORDER.normalize_bag_path("survey.BAG").endswith(
            "survey.BAG"))

    def test_capture_uses_hidden_staging_path(self):
        paths = RECORDER.capture_paths("survey.bag")
        self.assertTrue(paths["final"].endswith("survey.bag"))
        self.assertTrue(paths["staging"].endswith(
            "survey.recording.bag"))
        self.assertNotEqual(paths["final"], paths["staging"])

    def test_capture_invokes_cpp_recorder_for_min_space(self):
        command = RECORDER.rosbag_record_command(
            "/opt/ros/melodic/lib/rosbag/record",
            "/tmp/run.recording.bag", ["/tf", "/scan"], "700M")
        self.assertEqual(
            command[0], "/opt/ros/melodic/lib/rosbag/record")
        self.assertNotEqual(command[:2], ["rosbag", "record"])
        self.assertIn("--min-space", command)
        self.assertEqual(command[command.index("--min-space") + 1], "700M")
        self.assertEqual(command[-2:], ["/tf", "/scan"])

    def test_capture_rejects_empty_cpp_recorder_inputs(self):
        with self.assertRaises(ValueError):
            RECORDER.rosbag_record_command("", "/tmp/run.bag", ["/tf"])
        with self.assertRaises(ValueError):
            RECORDER.rosbag_record_command(
                "/opt/ros/melodic/lib/rosbag/record", "/tmp/run.bag", [])

    def test_rosbag_space_parser_matches_cpp_suffixes(self):
        self.assertEqual(RECORDER.parse_space_bytes("700M"),
                         700 * 1024 * 1024)
        self.assertEqual(RECORDER.parse_space_bytes("1G"), 1024 ** 3)
        with self.assertRaises(ValueError):
            RECORDER.parse_space_bytes("0M")
        with self.assertRaises(ValueError):
            RECORDER.parse_space_bytes("700MB")

    def test_stereo_camera_info_requires_metric_baseline(self):
        header = _Message(frame_id="stereo_left_optical")
        common = dict(
            D=[0.0] * 5, K=[500.0, 0.0, 320.0,
                             0.0, 500.0, 240.0,
                             0.0, 0.0, 1.0],
            R=[1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0],
            width=640, height=480, header=header)
        left = _Message(P=[500.0, 0.0, 320.0, 0.0,
                           0.0, 500.0, 240.0, 0.0,
                           0.0, 0.0, 1.0, 0.0], **common)
        right = _Message(P=[500.0, 0.0, 320.0, -30.0,
                            0.0, 500.0, 240.0, 0.0,
                            0.0, 0.0, 1.0, 0.0], **common)
        self.assertTrue(RECORDER.camera_info_pair_valid(left, right))
        right.P[3] = 0.0
        self.assertFalse(RECORDER.camera_info_pair_valid(left, right))

    def test_no_replace_commit_never_overwrites_target(self):
        temp_dir = tempfile.mkdtemp()
        try:
            source = os.path.join(temp_dir, "staging")
            target = os.path.join(temp_dir, "final")
            with open(source, "w") as stream:
                stream.write("new")
            RECORDER.commit_file_no_replace(source, target)
            self.assertFalse(os.path.exists(source))
            with open(target, "r") as stream:
                self.assertEqual(stream.read(), "new")
            with open(source, "w") as stream:
                stream.write("other")
            with self.assertRaises(RuntimeError):
                RECORDER.commit_file_no_replace(source, target)
            with open(target, "r") as stream:
                self.assertEqual(stream.read(), "new")
        finally:
            shutil.rmtree(temp_dir)

    def test_required_bag_counts_exclude_live_map_but_include_tf(self):
        minimums = RECORDER.required_bag_minimums([
            "/map", "/scan", "/stereo/left/image_raw/compressed"])
        self.assertNotIn("/map", minimums)
        self.assertEqual(minimums["/scan"], 10)
        self.assertEqual(minimums["/tf"], 10)
        self.assertEqual(
            minimums["/stereo/left/image_raw/compressed"], 30)

    def test_finalized_bag_wait_accepts_existing_file(self):
        handle, path = tempfile.mkstemp(suffix=".bag")
        os.close(handle)
        try:
            self.assertTrue(RECORDER.wait_for_path(path, 0.0))
        finally:
            os.unlink(path)

    def test_finalized_bag_wait_rejects_missing_file(self):
        path = os.path.join(tempfile.gettempdir(),
                            "codex-missing-survey-output.bag")
        if os.path.exists(path):
            os.unlink(path)
        self.assertFalse(RECORDER.wait_for_path(path, 0.0))

    def test_motion_callback_tracks_required_odom_and_stationary_time(self):
        recorder = RECORDER.CheckedSlamCapture.__new__(
            RECORDER.CheckedSlamCapture)
        recorder.lock = RECORDER.threading.Lock()
        recorder.last = {"/odom_raw": None}
        recorder.motion_topic = "/odom_raw"
        recorder.motion_linear_min = 0.02
        recorder.motion_angular_min = 0.03
        recorder.robot_moving = True
        recorder.stationary_since = None
        recorder.motion_error = ""

        motion = _Message(twist=_Message(twist=_Message(
            linear=_Message(x=0.1, y=0.0),
            angular=_Message(z=0.0))))
        recorder.cb_motion(motion)
        self.assertIsNotNone(recorder.last["/odom_raw"])
        self.assertIsNone(recorder.stationary_since)
        motion.twist.twist.linear.x = 0.0
        recorder.cb_motion(motion)
        self.assertIsNotNone(recorder.stationary_since)

        motion.twist.twist.linear.x = float("nan")
        recorder.cb_motion(motion)
        self.assertTrue(recorder.robot_moving)
        self.assertIsNone(recorder.stationary_since)
        self.assertIn("non-finite", recorder.motion_error)

    def test_amcl_survey_health_gates_and_typed_topics(self):
        self.assertTrue(SURVEY.localization_sample_is_good(
            "map", [0.04, 0, 0, 0, 0, 0, 0, 0.04] + [0] * 27 + [0.01],
            10.0, 10.2, 2.0, 0.75, 0.40))
        self.assertTrue(SURVEY.localization_sample_is_good(
            "", [0.04, 0, 0, 0, 0, 0, 0, 0.04] + [0] * 27 + [0.01],
            10.0, 10.2, 2.0, 0.75, 0.40))
        self.assertFalse(SURVEY.localization_sample_is_good(
            "odom", [0.04, 0, 0, 0, 0, 0, 0, 0.04] + [0] * 27 + [0.01],
            10.0, 10.2, 2.0, 0.75, 0.40))
        self.assertTrue(SURVEY.localization_sample_is_good(
            "map", [0.04, 0, 0, 0, 0, 0, 0, 0.04] + [0] * 27 + [0.01],
            10.0, 60.0, 2.0, 0.75, 0.40))
        self.assertFalse(SURVEY.localization_sample_is_good(
            "map", [0.04, 0, 0, 0, 0, 0, 0, 0.04] + [0] * 27 + [0.01],
            0.0, 10.0, 2.0, 0.75, 0.40))
        self.assertFalse(SURVEY.localization_is_healthy(
            True, 0, 3, True, 1.0, 12.0, 10.0, 1.0))
        self.assertTrue(SURVEY.localization_is_healthy(
            True, 0, 3, True, 11.0, 12.0, 10.0, 11.0))
        self.assertTrue(SURVEY.localization_is_healthy(
            True, 20, 3, True, 11.0, 12.0, 10.0, 11.0))
        # Odom can become stationary before AMCL recovers at a waypoint.
        self.assertTrue(SURVEY.localization_is_healthy(
            True, 20, 3, False, 11.0, 12.0, 10.0, None))
        # Bag 19: still, grace expired, AMCL republishing high covariance.
        # Must keep recording so required=true does not tear the stack down.
        self.assertTrue(SURVEY.localization_is_healthy(
            True, 20, 3, False, 1.0, 12.0, 10.0, None))
        self.assertTrue(SURVEY.localization_is_healthy(
            True, 8, 8, False, 0.0, 31.0, 30.0, None))
        self.assertFalse(SURVEY.localization_is_healthy(
            True, 8, 8, True, 0.0, 31.0, 30.0, 0.0))
        self.assertFalse(SURVEY.localization_is_healthy(
            False, 0, 3, False, 11.0, 12.0, 10.0, None))
        self.assertIn("while moving", SURVEY.localization_abort_message(
            True, 8, 0.0, 31.0))
        self.assertIn("while stationary", SURVEY.localization_abort_message(
            False, 8, 0.0, 31.0))
        hits, moving, started = SURVEY.motion_latch(
            True, 0, 3, False, 10.0, None)
        self.assertEqual((hits, moving, started), (1, False, None))
        hits, moving, started = SURVEY.motion_latch(
            True, hits, 3, moving, 10.1, started)
        self.assertEqual((hits, moving), (2, False))
        hits, moving, started = SURVEY.motion_latch(
            True, hits, 3, moving, 10.2, started)
        self.assertEqual((hits, moving, started), (3, True, 10.2))
        hits, moving, started = SURVEY.motion_latch(
            False, hits, 3, moving, 10.3, started)
        self.assertEqual((hits, moving, started), (0, False, None))
        self.assertEqual(
            SURVEY.anymsg_required_topics(
                ["/tf", "/scan", "/odom_raw", "/amcl_pose",
                 "/stereo/left/image_raw/compressed"],
                "/odom_raw", True),
            ["/scan", "/stereo/left/image_raw/compressed"])
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "checked_rosbag_record.py")
        with open(source_path, "r") as stream:
            source = stream.read()
        self.assertIn('"pose_source": "amcl"', source)

    def test_uncontrolled_ros_shutdown_cannot_commit_completion(self):
        recorder = RECORDER.CheckedSlamCapture.__new__(
            RECORDER.CheckedSlamCapture)
        recorder.finished_state = None
        failures = []
        recorder._fail = failures.append
        recorder._on_shutdown()
        self.assertEqual(len(failures), 1)
        self.assertIn("uncontrolled shutdown", failures[0])
        recorder.finished_state = "completed"
        recorder._on_shutdown()
        self.assertEqual(len(failures), 1)

    def test_external_fail_request_fail_closes_exactly_once(self):
        # ~fail lets a crashed survey tour abort the recording transaction
        # without route knowledge leaking into the recorder.  It must be
        # idempotent (a finished transaction stays finished) and must not
        # race an in-flight ~complete.
        recorder = SURVEY.CheckedRosbagRecorder.__new__(
            SURVEY.CheckedRosbagRecorder)
        recorder.completion_request_lock = SURVEY.threading.Lock()
        recorder.finished_state = None
        recorder._TriggerResponse = _Message
        reasons = []
        recorder._fail = reasons.append

        first = recorder.cb_fail(None)
        self.assertTrue(first.success)
        self.assertEqual(len(reasons), 1)
        self.assertIn("external abort request", reasons[0])

        recorder.finished_state = "failed"
        second = recorder.cb_fail(None)
        self.assertFalse(second.success)
        self.assertIn("survey already finished: failed", second.message)
        self.assertEqual(len(reasons), 1)

        recorder.finished_state = None
        recorder.completion_request_lock.acquire()
        try:
            busy = recorder.cb_fail(None)
        finally:
            recorder.completion_request_lock.release()
        self.assertFalse(busy.success)
        self.assertIn("already in progress", busy.message)
        self.assertEqual(len(reasons), 1)

    def test_capture_status_binds_gmapping_only_after_map_exists(self):
        initial = RECORDER.capture_status_record(
            "recording", "abc", [0, 0, 0, 0, 0, 0], 10.0)
        self.assertEqual(initial["pose_source"], "gmapping")
        self.assertNotIn("map_image_sha256", initial)
        completed = RECORDER.capture_status_record(
            "completed", "abc", [0, 0, 0, 0, 0, 0], 10.0,
            "a" * 64, "b" * 64, finished_unix=20.0)
        self.assertEqual(completed["map_image_sha256"], "a" * 64)
        self.assertEqual(completed["finished_unix"], 20.0)

    def test_tf_edges_ignore_legacy_leading_slashes(self):
        self.assertEqual(RECORDER.tf_edge("/map", "/odom"),
                         ("map", "odom"))

    @staticmethod
    def _replay_runner():
        runner = RUNNER.OfflineReplayRunner.__new__(
            RUNNER.OfflineReplayRunner)
        runner.invalid_poll = 0.0
        return runner

    def test_locked_invalid_stops_the_bag_instead_of_playing_to_eof(self):
        # Bag 19 kept playing ~90 minutes after the mapper transaction was
        # already permanently INVALID.  The runner must kill rosbag play
        # the moment the status probe reports a locked INVALID.
        runner = self._replay_runner()
        stopped = []
        runner.stop = lambda: stopped.append(True)
        runner.process = _Message(poll=lambda: None)
        status = _Message(
            success=False,
            message="INVALID: duplicate det callback stamp 5")
        try:
            runner.wait_for_replay_exit(lambda: status)
        except RuntimeError as error:
            self.assertIn("INVALID: duplicate det callback stamp 5",
                          str(error))
        else:
            self.fail("runner kept playing a permanently INVALID replay")
        self.assertEqual(stopped, [True])

    def test_healthy_replay_waits_for_the_rosbag_exit_code(self):
        runner = self._replay_runner()
        codes = [None, None, 0]
        runner.process = _Message(poll=lambda: codes.pop(0))
        status = _Message(success=True, message="OK: det_rx=5 sync_rx=5")
        self.assertEqual(runner.wait_for_replay_exit(lambda: status), 0)

    def test_not_ready_status_does_not_abort_the_replay(self):
        runner = self._replay_runner()
        codes = [None, 3]
        runner.process = _Message(poll=lambda: codes.pop(0))
        status = _Message(success=False, message="NOT_READY: waiting")
        self.assertEqual(runner.wait_for_replay_exit(lambda: status), 3)

    def test_status_probe_failure_never_kills_a_healthy_replay(self):
        runner = self._replay_runner()
        codes = [None, 7]
        runner.process = _Message(poll=lambda: codes.pop(0))

        def broken_probe():
            raise Exception("service unavailable")

        self.assertEqual(runner.wait_for_replay_exit(broken_probe), 7)

    def test_recording_requires_matching_start_and_single_completion(self):
        valid, detail = RUNNER.validate_recording_states([
            _payload("started"), _payload("completed")])
        self.assertTrue(valid)
        self.assertEqual(detail["token"], "abc")
        self.assertFalse(RUNNER.validate_recording_states(
            [_payload("started")])[0])
        self.assertFalse(RUNNER.validate_recording_states(
            [_payload("completed")])[0])
        self.assertFalse(RUNNER.validate_recording_states([
            _payload("started"), _payload("completed"),
            _payload("completed")])[0])
        self.assertFalse(RUNNER.validate_recording_states([
            _payload("started", token="a"),
            _payload("started", token="b"),
            _payload("completed", token="b")])[0])

    def test_recording_marker_order_duplicates_and_expected_hash(self):
        self.assertFalse(RUNNER.validate_recording_states([
            _payload("completed"), _payload("started")])[0])
        valid, _detail = RUNNER.validate_recording_states([
            _payload("started"), _payload("started"),
            _payload("completed")], expected_map_hash="a" * 64)
        self.assertTrue(valid)
        self.assertFalse(RUNNER.validate_recording_states([
            _payload("started"), _payload("completed")],
            expected_map_hash="b" * 64)[0])
        mismatched_start = json.loads(_payload("started"))
        mismatched_start["map_image_sha256"] = "b" * 64
        self.assertFalse(RUNNER.validate_recording_states([
            json.dumps(mismatched_start), _payload("completed")])[0])
        mismatched_mount = json.loads(_payload("started"))
        mismatched_mount["camera_mount"][0] = 0.2
        self.assertFalse(RUNNER.validate_recording_states([
            json.dumps(mismatched_mount), _payload("completed")])[0])
        mismatched_source = json.loads(_payload("started"))
        mismatched_source["pose_source"] = "cartographer"
        self.assertFalse(RUNNER.validate_recording_states([
            json.dumps(mismatched_source), _payload("completed")])[0])
        valid_amcl, detail_amcl = RUNNER.validate_recording_states([
            _payload("started", pose_source="amcl"),
            _payload("completed", pose_source="amcl")])
        self.assertTrue(valid_amcl)
        self.assertEqual(detail_amcl["pose_source"], "amcl")
        missing_source_start = json.loads(_payload("started"))
        missing_source_done = json.loads(_payload("completed"))
        del missing_source_start["pose_source"]
        del missing_source_done["pose_source"]
        self.assertFalse(RUNNER.validate_recording_states([
            json.dumps(missing_source_start),
            json.dumps(missing_source_done)])[0])

    def test_capture_never_appends_to_a_closed_rosbag(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "scripts", "checked_slam_capture.py")
        with open(path, "r") as stream:
            source = stream.read()
        self.assertNotIn('rosbag.Bag(self.paths["staging"], "a")', source)
        complete = source[source.index("    def _complete(self):"):
                          source.index("    def _fail_locked(self, reason):")]
        self.assertLess(complete.index("_verify_staging_bag()"),
                        complete.index("commit_file_no_replace("))

    def test_failure_marker_always_rejects_recording(self):
        valid, detail = RUNNER.validate_recording_states([
            _payload("started"), _payload("completed"),
            _payload("failed", reason="camera stale")])
        self.assertFalse(valid)
        self.assertIn("camera stale", detail)

    def test_frozen_scan_header_cannot_refresh_watchdog(self):
        watchdog = WATCHDOG.ScanWatchdog.__new__(WATCHDOG.ScanWatchdog)
        watchdog.lock = WATCHDOG.threading.Lock()
        watchdog.last_scan = None
        watchdog.last_scan_stamp = None
        watchdog.terminal_failure = None
        watchdog.healthy = False
        watchdog.inhibit_pub = _Message(published=[])
        watchdog.inhibit_pub.publish = watchdog.inhibit_pub.published.append

        stamp = _Message(secs=10, nsecs=20)
        msg = _Message(
            header=_Message(stamp=stamp), angle_increment=0.01,
            ranges=[1.0] * 100, range_min=0.1, range_max=12.0)
        watchdog.cb_scan(msg)
        first_receipt = watchdog.last_scan
        self.assertEqual(watchdog.last_scan_stamp, (10, 20))
        self.assertFalse(watchdog.inhibit_pub.published[-1].data)

        watchdog.cb_scan(msg)
        self.assertEqual(watchdog.last_scan, first_receipt)
        self.assertEqual(watchdog.last_scan_stamp, (10, 20))
        msg.header.stamp = _Message(secs=10, nsecs=21)
        watchdog.cb_scan(msg)
        self.assertEqual(watchdog.last_scan_stamp, (10, 21))
        msg.header.stamp = _Message(secs=0, nsecs=0)
        watchdog.cb_scan(msg)
        self.assertEqual(watchdog.last_scan_stamp, (10, 21))
        msg.header.stamp = _Message(secs=9, nsecs=999999999)
        watchdog.cb_scan(msg)
        self.assertEqual(watchdog.last_scan_stamp, (10, 21))

    def test_terminal_scan_failure_cannot_be_released_by_late_scan(self):
        watchdog = WATCHDOG.ScanWatchdog.__new__(WATCHDOG.ScanWatchdog)
        watchdog.lock = WATCHDOG.threading.Lock()
        watchdog.started = 1.0
        watchdog.startup_grace = 10.0
        watchdog.stale_timeout = 1.0
        watchdog.last_scan = 5.0
        watchdog.last_scan_stamp = (10, 20)
        watchdog.terminal_failure = None
        watchdog.healthy = True
        watchdog.inhibit_pub = _Message(published=[])
        watchdog.inhibit_pub.publish = watchdog.inhibit_pub.published.append

        failure = watchdog.latch_failure_if_due(6.1)
        self.assertIn("/scan stale", failure)
        self.assertFalse(watchdog.healthy)

        msg = _Message(
            header=_Message(stamp=_Message(secs=10, nsecs=21)),
            angle_increment=0.01, ranges=[1.0] * 100,
            range_min=0.1, range_max=12.0)
        watchdog.cb_scan(msg)
        self.assertEqual(watchdog.last_scan, 5.0)
        self.assertEqual(watchdog.last_scan_stamp, (10, 20))
        self.assertEqual(watchdog.inhibit_pub.published, [])


if __name__ == "__main__":
    unittest.main()
