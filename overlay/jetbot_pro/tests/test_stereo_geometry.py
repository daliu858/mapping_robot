#!/usr/bin/env python
"""Host-side tests for stereo depth and 3-D transform math (no ROS needed)."""
import math
import os
import sys
import tempfile
import threading
import types
import unittest

import numpy as np

try:
    from importlib import util as importlib_util
except ImportError:  # Python 2 on ROS Melodic
    importlib_util = None

if not hasattr(types, "SimpleNamespace"):
    class SimpleNamespace(object):
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    types.SimpleNamespace = SimpleNamespace


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    # Python 2 resolves ``from package.child import ...`` through the parent
    # package even when the fully-qualified child is already in sys.modules.
    # Build that parent explicitly so these host-side tests stay independent
    # of whichever ROS message packages happen to be installed.
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = types.ModuleType(parent_name)
            parent.__path__ = []
            sys.modules[parent_name] = parent
        setattr(parent, child_name, module)


def _load_mapper_module():
    dummy = type("Dummy", (), {})
    _stub_module(
        "rospy", Time=lambda *args: 0, Duration=lambda *args: 0,
        logwarn_throttle=lambda *args, **kwargs: None,
        logwarn=lambda *args, **kwargs: None,
        loginfo_throttle=lambda *args, **kwargs: None,
        logerr_throttle=lambda *args, **kwargs: None,
        loginfo=lambda *args, **kwargs: None,
        logerr=lambda *args, **kwargs: None,
        logfatal=lambda *args, **kwargs: None,
        get_param=lambda *args, **kwargs: (_ for _ in ()).throw(KeyError()),
        signal_shutdown=lambda *args, **kwargs: None)
    _stub_module("message_filters")
    _stub_module("tf2_ros")
    trigger_response = type(
        "TriggerResponse", (),
        {"__init__": lambda self, success=False, message="":
         self.__dict__.update(success=success, message=message)})
    _stub_module("std_srvs.srv", Trigger=dummy,
                 TriggerResponse=trigger_response)
    finalize_response = type(
        "FinalizeSemanticMapResponse", (),
        {"__init__": lambda self, success=False, message="":
         self.__dict__.update(success=success, message=message)})
    _stub_module("jetbot_pro.srv", FinalizeSemanticMap=dummy,
                 FinalizeSemanticMapResponse=finalize_response)
    _stub_module("std_msgs.msg", String=dummy)
    for package, names in {
            "nav_msgs.msg": ["OccupancyGrid", "Odometry"],
            "sensor_msgs.msg": ["CameraInfo", "Image", "LaserScan"],
            "vision_msgs.msg": ["Detection2DArray", "VisionInfo"],
            "visualization_msgs.msg": ["Marker", "MarkerArray"],
            "stereo_msgs.msg": ["DisparityImage"],
    }.items():
        _stub_module(package, **{name: dummy for name in names})

    script = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "scripts", "semantic_mapper.py")
    if importlib_util is not None:
        spec = importlib_util.spec_from_file_location("semantic_mapper", script)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import imp
    return imp.load_source("semantic_mapper", script)


MAPPER_MODULE = _load_mapper_module()


def _load_gate_module():
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "scripts", "stereo_pair_gate.py")
    if importlib_util is not None:
        spec = importlib_util.spec_from_file_location("stereo_pair_gate", script)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import imp
    return imp.load_source("stereo_pair_gate", script)


GATE_MODULE = _load_gate_module()


class StereoGeometryTest(unittest.TestCase):
    def setUp(self):
        self.mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        self.mapper.detections_rectified = True
        self.mapper.depth_roi_scale = 0.4
        self.mapper.depth_roi_expand = True
        self.mapper.depth_min = 0.3
        self.mapper.depth_max = 10.0
        self.mapper.depth_iqr_max = 1.5
        self.mapper.depth_min_samples = 4
        self.mapper.depth_near_gap = 0.60
        self.mapper.max_bearing = math.radians(40.0)
        self.mapper.lock = threading.Lock()
        self.mapper.stats = {}
        self.mapper.P = np.array([
            [400.0, 0.0, 5.0, 0.0],
            [0.0, 400.0, 5.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        self.mapper.K = self.mapper.P[:, :3].copy()
        self.mapper.D = np.zeros(5)
        self.mapper.R = np.eye(3)

    def _disparity(self, values, f=400.0, baseline=0.06):
        values = np.asarray(values, dtype="<f4")
        image = types.SimpleNamespace(
            encoding="32FC1", height=values.shape[0], width=values.shape[1],
            step=values.shape[1] * 4, is_bigendian=False,
            data=values.tobytes())
        return types.SimpleNamespace(
            image=image, f=f, T=baseline,
            min_disparity=0.0, max_disparity=64.0)

    @staticmethod
    def _bbox(cx=5.0, cy=5.0, sx=6.0, sy=6.0):
        return types.SimpleNamespace(
            center=types.SimpleNamespace(x=cx, y=cy),
            size_x=sx, size_y=sy)

    @staticmethod
    def _transaction_mapper():
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.lock = threading.Lock()
        mapper.stats = {"sync_rx": 0, "callbacks_inflight": 0,
                        "infrastructure_drop": 0}
        mapper.sync_stamps_seen = set()
        mapper.det_stamps_seen = set()
        mapper.pipeline_contract_error = ""
        mapper.last_pipeline_activity_wall = None
        mapper.transactional_finalize_only = True
        return mapper

    def test_missing_odom_makes_synced_transaction_terminal_invalid(self):
        mapper = self._transaction_mapper()
        mapper.odom_max_age = 0.12
        mapper.nearest_odom = lambda _stamp: None
        stamp = types.SimpleNamespace(secs=1, nsecs=23)
        detections = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=stamp))
        mapper.cb_det_stereo(detections, object())
        self.assertEqual(mapper.sync_stamps_seen, {1000000023})
        self.assertEqual(mapper.stats["callbacks_inflight"], 0)
        self.assertEqual(mapper.stats["infrastructure_drop"], 1)
        self.assertIn("odometry", mapper.pipeline_contract_error)

    def test_frame_mismatch_is_infrastructure_drop(self):
        mapper = self._transaction_mapper()
        mapper.K = np.eye(3)
        mapper.twist_ok = True
        mapper.range_source = "stereo"
        mapper.use_map_check = False
        mapper.require_map_received = False
        mapper.require_detection_camera_stamp = True
        mapper.camera_stamps = []
        disparity = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="left_rect"),
            image=types.SimpleNamespace(
                header=types.SimpleNamespace(frame_id="left_rect")))
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(secs=1, nsecs=0),
                frame_id="other_frame"))
        mapper.cb_det(message, disparity)
        self.assertEqual(mapper.stats["infrastructure_drop"], 1)
        self.assertIn("frame mismatch", mapper.pipeline_contract_error)

    def test_transactional_label_lookup_never_falls_back_to_coco(self):
        mapper = self._transaction_mapper()
        mapper.labels = None
        mapper.contract_labels = ["chair", "table", "door"]
        try:
            mapper.label_of(1)
        except RuntimeError as error:
            self.assertIn("validated detections", str(error))
        else:
            self.fail("label_of did not reject a missing validated label table")
        self.assertIn("label table", mapper.pipeline_contract_error)

    def test_transactional_vision_info_mismatch_is_terminal(self):
        mapper = self._transaction_mapper()
        mapper.contract_labels = ["chair", "table", "door"]
        mapper.labels = list(mapper.contract_labels)
        original_get_param = MAPPER_MODULE.rospy.get_param
        MAPPER_MODULE.rospy.get_param = lambda _name: ["wrong"]
        try:
            mapper.cb_vinfo(types.SimpleNamespace(database_location="/labels"))
        finally:
            MAPPER_MODULE.rospy.get_param = original_get_param
        self.assertEqual(mapper.labels, mapper.contract_labels)
        self.assertIn("disagrees", mapper.pipeline_contract_error)

    def test_transactional_vision_info_match_keeps_contract_copy(self):
        mapper = self._transaction_mapper()
        mapper.contract_labels = ["chair", "table", "door"]
        mapper.labels = list(mapper.contract_labels)
        side_labels = list(mapper.contract_labels)
        original_get_param = MAPPER_MODULE.rospy.get_param
        MAPPER_MODULE.rospy.get_param = lambda _name: side_labels
        try:
            mapper.cb_vinfo(types.SimpleNamespace(database_location="/labels"))
        finally:
            MAPPER_MODULE.rospy.get_param = original_get_param
        side_labels[0] = "mutated_after_callback"
        self.assertEqual(mapper.labels, mapper.contract_labels)
        self.assertEqual(mapper.pipeline_contract_error, "")

    def test_transactional_save_rejects_late_contract_error(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.require_map_received = False
        mapper.require_map_identity = False
        mapper.objects = []
        item = MAPPER_MODULE.TrackedObject("chair", 0)
        item.add(1.0, 2.0, object())
        mapper.objects.append(item)
        mapper.min_hits = 1
        mapper.required_labels = {"chair"}
        mapper.transactional_finalize_only = True
        mapper.lock = threading.Lock()
        mapper.recorded_map_hashes = set()
        mapper.recorded_map_manifest_hashes = set()
        mapper.stats = {}
        mapper.pipeline_contract_error = "late label contract mismatch"
        mapper.last_save_error = ""
        mapper.save_path = os.path.join(
            tempfile.gettempdir(), "must_not_commit_semantic_objects.yaml")
        self.assertFalse(mapper.save())
        self.assertIn("pipeline is invalid", mapper.last_save_error)

    def test_low_confidence_rejection_is_not_infrastructure_drop(self):
        mapper = self._tf_mapper()
        mapper.K = np.eye(3)
        mapper.twist_ok = True
        mapper.range_source = "stereo"
        mapper.use_map_check = False
        mapper.require_map_received = False
        mapper.require_detection_camera_stamp = True
        mapper.camera_stamps = []
        mapper.tf_lookup_timeout = 0.0
        mapper.tfbuf = types.SimpleNamespace(
            lookup_transform=lambda *_args, **_kwargs: object())
        mapper.stereo_lidar_fallback = False
        mapper.score_min = 0.5
        mapper.exclude = set()
        mapper.publish = lambda: None
        mapper.stats.update({"eligible": 0, "stereo_ok": 0,
                             "stereo_bad": 0, "lidar_fallback": 0,
                             "map_reject": 0})
        disparity = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="left_rect"),
            image=types.SimpleNamespace(
                header=types.SimpleNamespace(frame_id="left_rect")))
        hypothesis = types.SimpleNamespace(score=0.1, id=0)
        detection = types.SimpleNamespace(results=[hypothesis])
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(secs=1, nsecs=0),
                frame_id="left_rect"), detections=[detection])
        mapper.cb_det(message, disparity)
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.pipeline_contract_error, "")

    def _bbox_gate_mapper(self, min_h):
        mapper = self._tf_mapper()
        mapper.K = np.eye(3)
        mapper.twist_ok = True
        mapper.range_source = "stereo"
        mapper.use_map_check = False
        mapper.require_map_received = False
        mapper.require_detection_camera_stamp = True
        mapper.camera_stamps = []
        mapper.tfbuf = types.SimpleNamespace(
            lookup_transform=lambda *_args, **_kwargs: object())
        mapper.stereo_lidar_fallback = False
        mapper.score_min = 0.40
        mapper.exclude = set()
        mapper.publish = lambda: None
        mapper.transactional_finalize_only = False
        mapper.labels = ["door", "chair"]
        mapper.structure_labels = {"door"}
        mapper.min_structure_bbox_h = float(min_h)
        mapper.stats.update({"eligible": 0, "stereo_ok": 0, "stereo_bad": 0,
                             "lidar_fallback": 0, "map_reject": 0})
        mapper.sampled = []
        mapper.stereo_point = (
            lambda bbox, disp: mapper.sampled.append(bbox) and None)
        return mapper

    @staticmethod
    def _gate_detection(class_id, sy):
        return types.SimpleNamespace(
            results=[types.SimpleNamespace(score=0.41, id=class_id)],
            bbox=types.SimpleNamespace(
                center=types.SimpleNamespace(x=65.0, y=358.0),
                size_x=86.0, size_y=float(sy)))

    def test_structure_bbox_height_gate_rejects_detector_slivers(self):
        # _19 full bag: 631/834 gated door boxes were a persistent
        # ~86x41 px bottom-left sliver; real doors sat above the
        # 100-119 px histogram valley.  The gate must reject the sliver
        # before geometry, admit the tall door, and never touch
        # non-structure labels.
        mapper = self._bbox_gate_mapper(min_h=120.0)
        disparity = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="left_rect"),
            image=types.SimpleNamespace(
                header=types.SimpleNamespace(frame_id="left_rect")))
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(secs=1, nsecs=0),
                frame_id="left_rect"),
            detections=[self._gate_detection(0, 41.0),
                        self._gate_detection(0, 300.0),
                        self._gate_detection(1, 10.0)])
        mapper.cb_det(message, disparity)
        self.assertEqual(mapper.stats.get("bbox_h_reject"), 1)
        self.assertEqual(mapper.stats["eligible"], 2)
        self.assertEqual([b.size_y for b in mapper.sampled], [300.0, 10.0])
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.pipeline_contract_error, "")

    def test_refused_save_names_the_binding_gate_per_track(self):
        # r3 ended confirmed=0 with 28 accepted observations and no way to
        # tell hits-starvation from spread failure.  The refusal message
        # must carry the strongest tracks' hits/spread so the next zero
        # needs no extra replay to diagnose.
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.require_map_received = False
        mapper.require_map_identity = False
        mapper.transactional_finalize_only = False
        mapper.lock = threading.Lock()
        mapper.recorded_map_hashes = set()
        mapper.recorded_map_manifest_hashes = set()
        mapper.stats = {}
        mapper.min_hits = 5
        mapper.max_spread_m = 0.25
        mapper.required_labels = {"door"}
        mapper.objects = []
        near_miss = MAPPER_MODULE.TrackedObject("door", 70)
        for _ in range(4):
            near_miss.add(2.0, 3.0, object())
        mapper.objects.append(near_miss)
        mapper.last_save_error = ""
        mapper.save_path = os.path.join(
            tempfile.gettempdir(), "must_not_commit_semantic_objects.yaml")
        self.assertFalse(mapper.save())
        self.assertIn("missing required POIs: door", mapper.last_save_error)
        self.assertIn("door hits=4 spread=0.00", mapper.last_save_error)
        self.assertIn("at (2.00,3.00)", mapper.last_save_error)

    def test_structure_bbox_height_gate_defaults_off(self):
        mapper = self._bbox_gate_mapper(min_h=0.0)
        disparity = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="left_rect"),
            image=types.SimpleNamespace(
                header=types.SimpleNamespace(frame_id="left_rect")))
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(secs=1, nsecs=0),
                frame_id="left_rect"),
            detections=[self._gate_detection(0, 41.0)])
        mapper.cb_det(message, disparity)
        self.assertEqual(mapper.stats.get("bbox_h_reject", 0), 0)
        self.assertEqual(mapper.stats["eligible"], 1)

    def _tf_mapper(self, map_tf_seen=False):
        mapper = self._transaction_mapper()
        mapper.tf_lookup_timeout = 0.0
        mapper.tf_future_tolerance = 0.25
        mapper.map_tf_seen = map_tf_seen
        mapper.stats.update({"tf_startup_miss": 0, "tf_latest_fallback": 0,
                             "tf_stale_miss": 0})
        return mapper

    @staticmethod
    def _sim_stamp(seconds):
        return types.SimpleNamespace(to_sec=lambda: seconds)

    def test_bag_open_tf_gap_drops_one_frame_not_the_whole_bag(self):
        # Bag 19: one '"map" ... does not exist' at second 8 locked the
        # whole 90-minute replay INVALID.  Startup misses must now cost
        # only their own frame.
        mapper = self._tf_mapper()

        def raise_missing(*_args):
            raise Exception('"map" passed to lookupTransform argument '
                            'target_frame does not exist.')

        mapper.tfbuf = types.SimpleNamespace(lookup_transform=raise_missing)
        self.assertIsNone(mapper._stereo_map_transform(
            "stereo_left_optical", self._sim_stamp(10.0)))
        self.assertEqual(mapper.stats["tf_startup_miss"], 1)
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.pipeline_contract_error, "")
        self.assertFalse(mapper.map_tf_seen)

    def test_few_ms_future_slip_uses_the_latest_buffered_transform(self):
        mapper = self._tf_mapper()
        latest = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=self._sim_stamp(9.994)))

        def lookup(_target, _source, _when, *timeout):
            if timeout:
                raise Exception(
                    "Lookup would require extrapolation into the future.  "
                    "Requested time 10.000 but the latest data is at time "
                    "9.994, when looking up transform from frame "
                    "[stereo_left_optical] to frame [map]")
            return latest

        mapper.tfbuf = types.SimpleNamespace(lookup_transform=lookup)
        self.assertIs(mapper._stereo_map_transform(
            "stereo_left_optical", self._sim_stamp(10.0)), latest)
        self.assertEqual(mapper.stats["tf_latest_fallback"], 1)
        self.assertEqual(mapper.stats["tf_stale_miss"], 0)
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.pipeline_contract_error, "")
        self.assertTrue(mapper.map_tf_seen)

    def test_stale_map_chain_drops_frames_without_poisoning_the_bag(self):
        # The stale transform must never be substituted (it would move the
        # object); the frame drops and a dedicated counter records it.
        mapper = self._tf_mapper(map_tf_seen=True)
        latest = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=self._sim_stamp(4.0)))

        def lookup(_target, _source, _when, *timeout):
            if timeout:
                raise Exception(
                    "Lookup would require extrapolation into the future.  "
                    "Requested time 10.000 but the latest data is at time "
                    "4.000, when looking up transform from frame "
                    "[stereo_left_optical] to frame [map]")
            return latest

        mapper.tfbuf = types.SimpleNamespace(lookup_transform=lookup)
        self.assertIsNone(mapper._stereo_map_transform(
            "stereo_left_optical", self._sim_stamp(10.0)))
        self.assertEqual(mapper.stats["tf_stale_miss"], 1)
        self.assertEqual(mapper.stats["tf_latest_fallback"], 0)
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.pipeline_contract_error, "")

    def test_mid_replay_chain_break_drops_frames_without_invalid(self):
        mapper = self._tf_mapper(map_tf_seen=True)

        def raise_connectivity(*_args):
            raise Exception(
                "Could not find a connection between 'map' and "
                "'stereo_left_optical' because they are not part of the "
                "same tree.")

        mapper.tfbuf = types.SimpleNamespace(
            lookup_transform=raise_connectivity)
        self.assertIsNone(mapper._stereo_map_transform(
            "stereo_left_optical", self._sim_stamp(10.0)))
        self.assertEqual(mapper.stats["tf_stale_miss"], 1)
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.pipeline_contract_error, "")

    def test_cb_det_tf_miss_is_a_frame_drop_not_an_infrastructure_drop(self):
        mapper = self._tf_mapper()
        mapper.K = np.eye(3)
        mapper.twist_ok = True
        mapper.range_source = "stereo"
        mapper.use_map_check = False
        mapper.require_map_received = False
        mapper.require_detection_camera_stamp = True
        mapper.camera_stamps = []
        mapper.stereo_lidar_fallback = False
        mapper.publish = lambda: None
        mapper.stats.update({"eligible": 0, "stereo_ok": 0,
                             "stereo_bad": 0, "lidar_fallback": 0,
                             "map_reject": 0})

        def raise_missing(*_args):
            raise Exception('"map" passed to lookupTransform argument '
                            'target_frame does not exist.')

        mapper.tfbuf = types.SimpleNamespace(lookup_transform=raise_missing)
        disparity = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="left_rect"),
            image=types.SimpleNamespace(
                header=types.SimpleNamespace(frame_id="left_rect")))
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=self._sim_stamp(10.0), frame_id="left_rect"),
            detections=[])
        mapper.cb_det(message, disparity)
        self.assertEqual(mapper.stats["infrastructure_drop"], 0)
        self.assertEqual(mapper.stats["tf_startup_miss"], 1)
        self.assertEqual(mapper.pipeline_contract_error, "")

    def test_pipeline_status_probe_reports_ok_then_locked_invalid(self):
        mapper = self._transaction_mapper()
        mapper.stats.update({"det_rx": 3})
        healthy = mapper.cb_pipeline_status(None)
        self.assertTrue(healthy.success)
        self.assertTrue(healthy.message.startswith("OK"))
        mapper.pipeline_contract_error = "duplicate det callback stamp 5"
        doomed = mapper.cb_pipeline_status(None)
        self.assertFalse(doomed.success)
        self.assertTrue(doomed.message.startswith("INVALID:"))
        self.assertIn("duplicate det callback stamp 5", doomed.message)

    def test_short_wall_smear_leak_is_locked_until_next_round(self):
        # Round 7 finding: a door trail spanning less than ~0.87 m passes
        # the std()<=0.25 spread gate and confirms.  This round is
        # yield-first (write the yaml); tightening the smear gate is
        # explicitly deferred, so lock today's behavior to make the next
        # round change it deliberately rather than by accident.
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.objects = []
        mapper.lock = threading.Lock()
        mapper.assoc_default = 0.5
        mapper.min_hits = 5
        mapper.max_spread_m = 0.25
        mapper.accepting_observations = True
        for index in range(6):
            mapper.associate("door", 2, 0.1 * index, 0.0, object())
        confirmed = mapper.confirmed()
        self.assertEqual(len(confirmed), 1)
        self.assertLessEqual(confirmed[0].std(), 0.25)

    def test_constant_disparity_backprojects_to_expected_depth(self):
        # Z = f*T/d = 400*0.06/12 = 2 metres.
        disparity = self._disparity(np.full((10, 10), 12.0))
        xyz = self.mapper.stereo_point(self._bbox(), disparity)
        self.assertTrue(np.allclose(xyz, (0.0, 0.0, 2.0)))

    def test_invalid_disparity_is_rejected(self):
        disparity = self._disparity(np.full((10, 10), np.nan))
        self.assertIsNone(self.mapper.stereo_point(self._bbox(), disparity))

    def test_negative_stereo_baseline_is_rejected(self):
        disparity = self._disparity(
            np.full((10, 10), 12.0), baseline=-0.06)
        self.assertIsNone(self.mapper.stereo_point(self._bbox(), disparity))
        self.assertEqual(self.mapper.stats.get("stereo_rej_calib"), 1)

    def test_reject_reasons_split_depth_gate_iqr_and_bearing(self):
        # All disparities back-project past depth_max -> depth_gate.
        disparity = self._disparity(np.full((10, 10), 2.0))  # 12 m
        self.assertIsNone(self.mapper.stereo_point(self._bbox(), disparity))
        self.assertEqual(self.mapper.stats.get("stereo_rej_depth_gate"), 1)
        # Thin near cluster in front of a dominant far wall -> iqr_or_gap.
        values = np.full((10, 10), 4.8, dtype="<f4")  # 5 m
        values[4, 4:6] = 24.0                # two 1 m samples inside the ROI
        self.assertIsNone(
            self.mapper.stereo_point(self._bbox(), self._disparity(values)))
        self.assertEqual(self.mapper.stats.get("stereo_rej_iqr_or_gap"), 1)
        # Valid depth far outside max_bearing_deg -> bearing.
        wide = self._disparity(np.full((10, 500), 12.0))
        self.assertIsNone(self.mapper.stereo_point(
            self._bbox(cx=400.0, cy=5.0), wide))
        self.assertEqual(self.mapper.stats.get("stereo_rej_bearing"), 1)

    def test_padded_disparity_rows_are_decoded_without_row_shift(self):
        rows = np.full((10, 12), -1.0, dtype="<f4")
        rows[:, :10] = 12.0
        image = types.SimpleNamespace(
            encoding="32FC1", height=10, width=10, step=12 * 4,
            is_bigendian=False, data=rows.tobytes())
        disparity = types.SimpleNamespace(
            image=image, f=400.0, T=0.06,
            min_disparity=0.0, max_disparity=64.0)
        xyz = self.mapper.stereo_point(self._bbox(), disparity)
        self.assertTrue(np.allclose(xyz, (0.0, 0.0, 2.0)))

    def test_big_endian_disparity_is_decoded(self):
        values = np.full((10, 10), 12.0, dtype=">f4")
        image = types.SimpleNamespace(
            encoding="32FC1", height=10, width=10, step=10 * 4,
            is_bigendian=True, data=values.tobytes())
        disparity = types.SimpleNamespace(
            image=image, f=400.0, T=0.06,
            min_disparity=0.0, max_disparity=64.0)
        self.assertTrue(np.allclose(
            self.mapper.stereo_point(self._bbox(), disparity),
            (0.0, 0.0, 2.0)))

    def test_mixed_foreground_background_depth_prefers_the_near_surface(self):
        values = np.full((10, 10), 4.8, dtype="<f4")  # 5 m hallway
        values[:, :5] = 24.0                         # 1 m door panel
        disparity = self._disparity(values)
        xyz = self.mapper.stereo_point(self._bbox(), disparity)
        self.assertTrue(np.allclose(xyz, (0.0, 0.0, 1.0)))

    def test_hallway_dominated_roi_is_rejected_instead_of_jumping_rooms(self):
        depths = np.concatenate([
            np.full(10, 1.5),
            np.full(90, 5.0),
        ])
        self.assertIsNone(MAPPER_MODULE.select_stereo_depth(
            depths, min_samples=20, iqr_max=1.5, gap_m=0.60))
        self.assertAlmostEqual(MAPPER_MODULE.select_stereo_depth(
            depths, min_samples=8, iqr_max=1.5, gap_m=0.60), 1.5)

    def test_rectified_lidar_fallback_uses_projection_intrinsics(self):
        self.mapper.detections_rectified = True
        self.assertAlmostEqual(self.mapper.bearing(405.0, 5.0),
                               -math.pi / 4.0)

    def test_nearest_odom_is_selected_and_stale_sample_rejected(self):
        class Delta(object):
            def __init__(self, seconds):
                self.seconds = seconds

            def to_sec(self):
                return self.seconds

        class Stamp(object):
            def __init__(self, seconds):
                self.seconds = seconds

            def __sub__(self, other):
                return Delta(self.seconds - other.seconds)

        before = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=Stamp(9.94)))
        after = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=Stamp(10.04)))
        self.mapper.odom_max_age = 0.12
        self.mapper.odom_cache = types.SimpleNamespace(
            getElemBeforeTime=lambda _stamp: before,
            getElemAfterTime=lambda _stamp: after)
        self.assertIs(self.mapper.nearest_odom(Stamp(10.0)), after)
        self.mapper.odom_max_age = 0.01
        self.assertIsNone(self.mapper.nearest_odom(Stamp(10.0)))

    @unittest.skipIf(MAPPER_MODULE.cv2 is None, "OpenCV is unavailable")
    def test_zero_distortion_rectification_preserves_pixels(self):
        self.mapper.detections_rectified = False
        points = np.array([[2.0, 3.0], [7.0, 8.0]])
        self.assertTrue(np.allclose(self.mapper.rectify_pixels(points), points))

    @unittest.skipIf(MAPPER_MODULE.cv2 is None, "OpenCV is unavailable")
    def test_rectification_uses_distortion_rotation_and_projection(self):
        self.mapper.detections_rectified = False
        self.mapper.K = np.array([
            [400.0, 0.0, 320.0], [0.0, 410.0, 240.0], [0.0, 0.0, 1.0]])
        self.mapper.D = np.array([0.10, -0.02, 0.001, -0.0005, 0.01])
        angle = 0.04
        self.mapper.R = np.array([
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)]])
        self.mapper.P = np.array([
            [430.0, 0.0, 300.0, 0.0],
            [0.0, 425.0, 235.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]])

        xu, yu = 0.20, -0.10
        k1, k2, p1, p2, k3 = self.mapper.D
        r2 = xu * xu + yu * yu
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        yd = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        raw = np.array([[400.0 * xd + 320.0, 410.0 * yd + 240.0]])

        ray = np.dot(self.mapper.R, np.array([xu, yu, 1.0]))
        expected = np.array([[
            430.0 * ray[0] / ray[2] + 300.0,
            425.0 * ray[1] / ray[2] + 235.0]])
        self.assertTrue(np.allclose(
            self.mapper.rectify_pixels(raw), expected, atol=1e-4))

    def test_full_3d_transform_uses_rotation_and_translation(self):
        q = types.SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(math.pi / 4.0),
            w=math.cos(math.pi / 4.0))
        translation = types.SimpleNamespace(x=1.0, y=2.0, z=3.0)
        transform = types.SimpleNamespace(
            transform=types.SimpleNamespace(
                rotation=q, translation=translation))
        result = MAPPER_MODULE.transform_xyz(transform, (1.0, 0.0, 0.0))
        self.assertTrue(np.allclose(result, (1.0, 3.0, 3.0)))

    def test_recorded_optical_mount_sends_camera_forward_along_robot_x(self):
        # complete.json camera_mount for _19.  Optical +Z (pinhole forward)
        # must become base +X; a sign flip here would scatter every door.
        scripts = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from replay_contract import camera_mount_quaternion
        mount = [0.105, 0.03, 0.094, -1.5708, 0.0, -1.5708]
        qx, qy, qz, qw = camera_mount_quaternion(mount)
        q = types.SimpleNamespace(x=qx, y=qy, z=qz, w=qw)
        translation = types.SimpleNamespace(
            x=mount[0], y=mount[1], z=mount[2])
        transform = types.SimpleNamespace(
            transform=types.SimpleNamespace(
                rotation=q, translation=translation))
        mx, my, mz = MAPPER_MODULE.transform_xyz(transform, (0.0, 0.0, 2.0))
        self.assertAlmostEqual(mx, 2.0 + mount[0], places=3)
        self.assertAlmostEqual(my, mount[1], places=3)
        self.assertAlmostEqual(mz, mount[2], places=3)
        right_x, right_y, _right_z = MAPPER_MODULE.transform_xyz(
            transform, (1.0, 0.0, 0.0))
        self.assertLess(right_y, 0.0)

    def test_door_uses_large_object_association_radius(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.assoc_default = 0.5
        self.assertEqual(mapper.assoc_radius("door"), 0.9)
        self.assertEqual(mapper.assoc_radius("table"), 0.9)

    def test_mapping_keeps_the_full_observation_trail(self):
        item = MAPPER_MODULE.TrackedObject("door", 2)
        for index in range(80):
            item.add(float(index), 0.0, object())
        self.assertEqual(len(item.obs), 80)
        self.assertEqual(item.hits, 80)

    def test_hallway_detection_trail_is_not_confirmed_as_doors(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.objects = []
        mapper.lock = threading.Lock()
        mapper.assoc_default = 0.5
        mapper.min_hits = 5
        mapper.max_spread_m = 0.25
        mapper.accepting_observations = True
        for index in range(21):
            mapper.associate("door", 2, 0.1 * index, 0.0, object())
        self.assertEqual(sum(item.hits for item in mapper.objects), 21)
        self.assertGreater(
            max(item.std() for item in mapper.objects), mapper.max_spread_m)
        self.assertEqual(mapper.confirmed(), [])

    def test_axis_averaged_std_cannot_hide_a_one_dimensional_smear(self):
        item = MAPPER_MODULE.TrackedObject("door", 2)
        for index in range(13):
            item.add(0.1 * index, 0.0, object())
        axis_mean = float(np.array(item.obs).std(axis=0).mean())
        self.assertLess(axis_mean, 0.25)
        self.assertGreater(item.std(), 0.25)

    def test_compact_multi_hit_cluster_is_confirmed(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.objects = []
        mapper.lock = threading.Lock()
        mapper.assoc_default = 0.5
        mapper.min_hits = 5
        mapper.max_spread_m = 0.25
        mapper.accepting_observations = True
        jitter = (
            (0.00, 0.00), (0.04, -0.02), (-0.03, 0.03),
            (0.02, 0.04), (-0.04, -0.01), (0.01, -0.03),
            (0.03, 0.02), (-0.02, 0.04))
        for dx, dy in jitter:
            mapper.associate("door", 2, 1.0 + dx, 2.0 + dy, object())
        confirmed = mapper.confirmed()
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].hits, 8)

    def test_door_in_free_space_is_rejected_until_near_a_wall(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.use_map_check = True
        mapper.require_map_received = True
        mapper.structure_labels = {"door"}
        mapper.occupied_radius_m = 0.30
        mapper.occupied_threshold = 50
        width = height = 20
        data = [0] * (width * height)
        for row in range(height):
            data[row * width + 18] = 100
        mapper.grid = types.SimpleNamespace(
            info=types.SimpleNamespace(
                origin=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=0.0, y=0.0)),
                resolution=0.05, width=width, height=height),
            data=data)
        self.assertFalse(mapper.map_ok(0.10, 0.10, "door"))
        self.assertTrue(mapper.map_ok(0.10, 0.10, "chair"))
        self.assertTrue(mapper.map_ok(0.92, 0.10, "door"))
        self.assertTrue(mapper.map_ok(0.80, 0.10, "door"))

    def test_pair_gate_requires_valid_baseline_and_matching_frames(self):
        common = dict(
            width=640, height=480,
            D=[0.0, 0.0, 0.0, 0.0, 0.0],
            K=[400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0],
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            header=types.SimpleNamespace(frame_id="stereo_left_optical"))
        left = types.SimpleNamespace(
            P=[400.0, 0.0, 320.0, 0.0,
               0.0, 400.0, 240.0, 0.0,
               0.0, 0.0, 1.0, 0.0], **common)
        right = types.SimpleNamespace(
            P=[400.0, 0.0, 320.0, -24.0,
               0.0, 400.0, 240.0, 0.0,
               0.0, 0.0, 1.0, 0.0], **common)
        self.assertTrue(GATE_MODULE.StereoPairGate.calibration_ok(left, right))
        left_image = types.SimpleNamespace(width=640, height=480)
        right_image = types.SimpleNamespace(width=640, height=480)
        self.assertTrue(GATE_MODULE.StereoPairGate.calibration_ok(
            left, right, left_image, right_image))
        right_image.width = 320
        self.assertFalse(GATE_MODULE.StereoPairGate.calibration_ok(
            left, right, left_image, right_image))
        right_image.width = 640
        right.D[0] = float("nan")
        self.assertFalse(GATE_MODULE.StereoPairGate.calibration_ok(left, right))
        right.D[0] = 0.0
        right.P[3] = 0.0
        self.assertFalse(GATE_MODULE.StereoPairGate.calibration_ok(left, right))
        right.P[3] = 24.0
        self.assertFalse(GATE_MODULE.StereoPairGate.calibration_ok(left, right))
        right.P[3] = -24.0
        left.P[3] = 1.0
        self.assertFalse(GATE_MODULE.StereoPairGate.calibration_ok(left, right))
        left.P[3] = 0.0
        right.R = [-1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0]
        self.assertFalse(GATE_MODULE.StereoPairGate.calibration_ok(left, right))

    def test_duplicate_detections_in_one_frame_count_once(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.objects = []
        mapper.lock = threading.Lock()
        mapper.assoc_default = 0.5
        stamp = object()
        mapper.associate("door", 2, 1.0, 2.0, stamp)
        mapper.associate("door", 2, 1.01, 2.01, stamp)
        self.assertEqual(mapper.objects[0].hits, 1)
        mapper.associate("door", 2, 1.01, 2.01, object())
        self.assertEqual(mapper.objects[0].hits, 2)

    def test_formal_table_uses_large_object_association_radius(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.assoc_default = 0.5
        self.assertEqual(mapper.assoc_radius("table"), 0.9)

    def test_empty_session_does_not_overwrite_existing_map(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.objects = []
        mapper.min_hits = 2
        handle = tempfile.NamedTemporaryFile(delete=False)
        try:
            handle.write(b"existing semantic map\n")
            handle.close()
            mapper.save_path = handle.name
            mapper.save()
            with open(handle.name, "rb") as saved:
                self.assertEqual(saved.read(), b"existing semantic map\n")
        finally:
            if not handle.closed:
                handle.close()
            if os.path.exists(handle.name):
                os.unlink(handle.name)

    def test_recorded_map_identity_must_be_unique_and_match(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.require_map_identity = True
        mapper.expected_map_hash = "a" * 64
        mapper.expected_map_manifest_hash = "c" * 64
        mapper.recorded_map_hashes = set()
        mapper.recorded_map_manifest_hashes = set()
        mapper.lock = threading.Lock()
        self.assertFalse(mapper.map_identity_status()[0])
        mapper.recorded_map_hashes.add("b" * 64)
        self.assertFalse(mapper.map_identity_status()[0])
        mapper.recorded_map_hashes = {"a" * 64, "b" * 64}
        self.assertFalse(mapper.map_identity_status()[0])
        mapper.recorded_map_hashes = {"a" * 64}
        self.assertFalse(mapper.map_identity_status()[0])
        mapper.recorded_map_manifest_hashes = {"d" * 64}
        self.assertFalse(mapper.map_identity_status()[0])
        mapper.recorded_map_manifest_hashes = {"c" * 64}
        self.assertTrue(mapper.map_identity_status()[0])

    def test_semantic_mapper_rejects_nonzero_occupancy_origin_yaw(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.map_geometry_error = ""
        mapper.grid = object()
        yaw = 0.1
        orientation = types.SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0))
        message = types.SimpleNamespace(info=types.SimpleNamespace(
            origin=types.SimpleNamespace(orientation=orientation)))
        mapper.cb_map(message)
        self.assertIsNone(mapper.grid)
        self.assertIn("yaw must be zero", mapper.map_geometry_error)

    def test_partial_three_poi_result_does_not_overwrite_existing_map(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.objects = []
        item = MAPPER_MODULE.TrackedObject("door", 2)
        item.add(1.0, 2.0, object())
        mapper.objects.append(item)
        mapper.min_hits = 1
        mapper.require_map_identity = False
        mapper.required_labels = {"chair", "table", "door"}
        handle = tempfile.NamedTemporaryFile(delete=False)
        try:
            handle.write(b"existing semantic map\n")
            handle.close()
            mapper.save_path = handle.name
            self.assertFalse(mapper.save())
            self.assertIn("missing required POIs", mapper.last_save_error)
            with open(handle.name, "rb") as saved:
                self.assertEqual(saved.read(), b"existing semantic map\n")
        finally:
            if not handle.closed:
                handle.close()
            if os.path.exists(handle.name):
                os.unlink(handle.name)

    def test_transactional_offline_mode_disables_periodic_save(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.range_source = "lidar"
        calls = []
        mapper.save = lambda: calls.append(True)
        mapper.transactional_finalize_only = True
        mapper.cb_timer(None)
        self.assertEqual(calls, [])
        mapper.transactional_finalize_only = False
        mapper.cb_timer(None)
        self.assertEqual(calls, [True])

    def test_required_map_rejects_observations_before_map_arrives(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.grid = None
        mapper.use_map_check = True
        mapper.require_map_received = True
        self.assertFalse(mapper.map_ok(1.0, 2.0))
        mapper.require_map_received = False
        self.assertTrue(mapper.map_ok(1.0, 2.0))

    def test_map_cell_conversion_rejects_just_below_origin(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.use_map_check = True
        mapper.require_map_received = True
        mapper.grid = types.SimpleNamespace(
            info=types.SimpleNamespace(
                origin=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=0.0, y=0.0)),
                resolution=1.0, width=2, height=2),
            data=[0, 0, 0, 0])
        self.assertFalse(mapper.map_ok(-0.01, 0.5))
        self.assertFalse(mapper.map_ok(0.5, -0.01))
        self.assertTrue(mapper.map_ok(0.0, 0.0))

    def test_finalize_requires_token_freezes_and_commits_once(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.finalize_lock = threading.Lock()
        mapper.lock = threading.Lock()
        mapper.finalize_token = "run-token"
        mapper.finalize_called = False
        mapper.accepting_observations = True
        mapper.objects = []
        mapper.stats = {}
        mapper.save_path = "unused.yaml"
        mapper.last_save_error = ""
        mapper.save = lambda: True
        wrong = mapper.cb_finalize(types.SimpleNamespace(token="wrong"))
        self.assertFalse(wrong.success)
        self.assertFalse(mapper.finalize_called)
        accepted = mapper.cb_finalize(
            types.SimpleNamespace(token="run-token"))
        self.assertTrue(accepted.success)
        self.assertTrue(mapper.finalize_called)
        self.assertFalse(mapper.accepting_observations)
        repeated = mapper.cb_finalize(
            types.SimpleNamespace(token="run-token"))
        self.assertFalse(repeated.success)

    def test_concurrent_finalize_is_rejected_without_blocking(self):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.finalize_lock = threading.Lock()
        mapper.finalize_lock.acquire()
        try:
            response = mapper.cb_finalize(
                types.SimpleNamespace(token="anything"))
        finally:
            mapper.finalize_lock.release()
        self.assertFalse(response.success)
        self.assertIn("in progress", response.message)


class LiveDisparityValid0Test(unittest.TestCase):
    """Regression fixture from the _19 live /stereo/disparity sample.

    Observed on-robot at ~24 percent of the bag: 640x480 32FC1,
    f=884.85675 T=+0.06006, only 2213 of 307200 pixels positive, every
    other finite pixel exactly -1.0 (block-matcher no-match), positives
    clustered in a narrow high-texture band (u median 530) while the
    image center - where a door bbox center ROI samples - is all -1.
    The mapper then reported valid=0 invalid=281 eligible=281.
    """

    F = 884.85675
    T = 0.06006

    def _mapper(self, expand=True):
        mapper = MAPPER_MODULE.SemanticMapper.__new__(
            MAPPER_MODULE.SemanticMapper)
        mapper.detections_rectified = True
        mapper.depth_roi_scale = 0.40
        mapper.depth_roi_expand = expand
        mapper.depth_min = 0.30
        mapper.depth_max = 10.0
        mapper.depth_iqr_max = 1.50
        mapper.depth_min_samples = 12
        mapper.depth_near_gap = 0.60
        mapper.max_bearing = math.radians(40.0)
        mapper.P = np.array([
            [self.F, 0.0, 320.0, 0.0],
            [0.0, self.F, 240.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        mapper.K = mapper.P[:, :3].copy()
        mapper.D = np.zeros(5)
        mapper.R = np.eye(3)
        mapper.lock = threading.Lock()
        mapper.stats = {}
        return mapper

    def _live_disparity(self, values):
        image = types.SimpleNamespace(
            encoding="32FC1", height=480, width=640, step=640 * 4,
            is_bigendian=False, data=values.astype("<f4").tobytes())
        return types.SimpleNamespace(
            image=image, f=self.F, T=self.T,
            min_disparity=0.0, max_disparity=127.0)

    def _frame_band_disparity(self, band_d=19.31):
        # A 7 px wide vertical door-frame band, ~2240 positive pixels,
        # matching the live 2213-positive histogram and u-median 530.
        values = np.full((480, 640), -1.0)
        values[80:400, 527:534] = band_d
        return self._live_disparity(values)

    @staticmethod
    def _door_bbox():
        # Rectified door candidate: the frame edge overlaps the textured
        # band, the 40 percent center ROI (x 340..460) does not.
        return types.SimpleNamespace(
            center=types.SimpleNamespace(x=400.0, y=240.0),
            size_x=300.0, size_y=400.0)

    def test_center_roi_starves_on_live_histogram(self):
        # Root cause reproduced: without expansion the door frame band is
        # outside the center ROI, every sample is -1, the frame is lost.
        mapper = self._mapper(expand=False)
        self.assertIsNone(
            mapper.stereo_point(self._door_bbox(),
                                self._frame_band_disparity()))
        self.assertEqual(mapper.stats.get("stereo_rej_no_pos_d"), 1)
        self.assertFalse(mapper.stats.get("stereo_roi_expanded"))

    def test_full_bbox_expansion_recovers_the_door_frame_depth(self):
        mapper = self._mapper(expand=True)
        xyz = mapper.stereo_point(self._door_bbox(),
                                  self._frame_band_disparity())
        self.assertIsNotNone(xyz)
        # Z = f*T/d = 884.85675*0.06006/19.31 = 2.752 m on the door plane.
        self.assertAlmostEqual(xyz[2], 2.752, places=3)
        self.assertEqual(mapper.stats.get("stereo_roi_expanded"), 1)
        self.assertEqual(mapper.stats.get("stereo_rej_no_pos_d", 0), 0)

    def test_door_bbox_in_dead_zone_still_rejects_with_reason(self):
        # A bbox whose full extent has no positive disparity must still
        # fail (this is what would justify the lidar-fallback debate),
        # and the reason counter must say no_pos_d, not a generic invalid.
        mapper = self._mapper(expand=True)
        bbox = types.SimpleNamespace(
            center=types.SimpleNamespace(x=150.0, y=240.0),
            size_x=200.0, size_y=300.0)
        self.assertIsNone(
            mapper.stereo_point(bbox, self._frame_band_disparity()))
        self.assertEqual(mapper.stats.get("stereo_rej_no_pos_d"), 1)
        self.assertFalse(mapper.stats.get("stereo_roi_expanded"))

    def test_bbox_fully_outside_disparity_frame_counts_roi_empty(self):
        # r2 live finding at 61 s: reject split showed roi_empty=35/35.
        # A rectified bbox with zero overlap against the 640x480 frame
        # (the alpha=0 crop keeps ~2/3 of the raw FOV) must reject with
        # the dedicated counter, off either side, and never be mistaken
        # for a sampling failure.
        mapper = self._mapper(expand=True)
        off_right = types.SimpleNamespace(
            center=types.SimpleNamespace(x=900.0, y=240.0),
            size_x=200.0, size_y=300.0)
        self.assertIsNone(
            mapper.stereo_point(off_right, self._frame_band_disparity()))
        off_left = types.SimpleNamespace(
            center=types.SimpleNamespace(x=-200.0, y=240.0),
            size_x=200.0, size_y=300.0)
        self.assertIsNone(
            mapper.stereo_point(off_left, self._frame_band_disparity()))
        self.assertEqual(mapper.stats.get("stereo_rej_roi_empty"), 2)
        self.assertEqual(mapper.stats.get("stereo_rej_no_pos_d", 0), 0)
        self.assertFalse(mapper.stats.get("stereo_roi_expanded"))

    def test_partially_cropped_bbox_is_rescued_by_expansion(self):
        # Center ROI clipped away (rect center beyond the right edge) but
        # the bbox still overlaps the frame: under the old code this was a
        # silent None; expansion must sample the in-frame remainder.
        mapper = self._mapper(expand=True)
        bbox = types.SimpleNamespace(
            center=types.SimpleNamespace(x=720.0, y=240.0),
            size_x=400.0, size_y=400.0)
        xyz = mapper.stereo_point(bbox, self._frame_band_disparity())
        self.assertIsNotNone(xyz)
        self.assertAlmostEqual(xyz[2], 2.752, places=3)
        self.assertEqual(mapper.stats.get("stereo_roi_expanded"), 1)
        self.assertEqual(mapper.stats.get("stereo_rej_roi_empty", 0), 0)

    def test_expanded_roi_still_prefers_near_door_frame_over_hallway(self):
        # Open-door safety with the widened ROI: a starved center ROI that
        # sees a few far hallway pixels expands to the full bbox, and the
        # near-cluster selector must return the door frame plane, not the
        # room behind the opening.
        mapper = self._mapper(expand=True)
        values = np.full((480, 640), -1.0)
        fT = self.F * self.T
        values[80:400, 527:534] = fT / 2.0    # door frame at 2 m
        values[230:232, 400:404] = fT / 5.0   # hallway at 5 m, center ROI
        xyz = mapper.stereo_point(self._door_bbox(),
                                  self._live_disparity(values))
        self.assertIsNotNone(xyz)
        self.assertAlmostEqual(xyz[2], 2.0, places=3)
        self.assertEqual(mapper.stats.get("stereo_roi_expanded"), 1)


if __name__ == "__main__":
    unittest.main()
