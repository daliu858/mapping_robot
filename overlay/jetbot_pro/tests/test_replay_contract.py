#!/usr/bin/env python
"""Host-side tests for formal offline replay source and completion gates."""
from __future__ import print_function

import os
import sys
import unittest

try:
    from importlib import util as importlib_util
except ImportError:
    importlib_util = None


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "scripts", "replay_contract.py")
if importlib_util is not None:
    SPEC = importlib_util.spec_from_file_location("replay_contract_test", PATH)
    CONTRACT = importlib_util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CONTRACT)
else:
    import imp
    CONTRACT = imp.load_source("replay_contract_test", PATH)


class ReplayContractTest(unittest.TestCase):
    def test_formal_poi_class_order_is_stable(self):
        self.assertEqual(
            ("chair", "table", "door"), CONTRACT.FORMAL_POI_CLASSES)
        self.assertEqual(("door",), CONTRACT.DOOR_TRIAL_POI_CLASSES)
        self.assertEqual(
            ("door", "sofa", "cabinet", "spray"),
            CONTRACT.HALL_V1_POI_CLASSES)
        self.assertEqual(
            (CONTRACT.FORMAL_POI_CLASSES, CONTRACT.DOOR_TRIAL_POI_CLASSES,
             CONTRACT.HALL_V1_POI_CLASSES),
            CONTRACT.ALLOWED_POI_CLASS_TABLES)

    def test_formal_pose_sources_include_gmapping_and_amcl(self):
        self.assertEqual(
            ("gmapping", "amcl"), CONTRACT.FORMAL_POSE_SOURCES)

    def test_camera_mount_requires_exactly_six_finite_numbers(self):
        self.assertEqual(CONTRACT.canonical_camera_mount("1 2 3 4 5 6"),
                         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for bad in ("1 2", "1 2 3 4 5 nan", None):
            with self.assertRaises(ValueError):
                CONTRACT.canonical_camera_mount(bad)

    def test_camera_mount_must_match_recorded_tf_edge(self):
        mount = [0.1, -0.02, 0.2, -1.5708, 0.0, -1.5708]
        quaternion = CONTRACT.camera_mount_quaternion(mount)
        self.assertTrue(CONTRACT.camera_transform_matches(
            mount, "/base_footprint", "/stereo_left_optical",
            mount[:3], quaternion))
        self.assertTrue(CONTRACT.camera_transform_matches(
            mount, "base_footprint", "stereo_left_optical",
            mount[:3], [-value for value in quaternion]))
        self.assertFalse(CONTRACT.camera_transform_matches(
            mount, "base_footprint", "stereo_left_optical",
            [0.2, mount[1], mount[2]], quaternion))

    def test_formal_stereo_rejects_right_image_or_rectified_override(self):
        good = dict(CONTRACT.FORMAL_STEREO_SOURCES)
        good["detections_rectified"] = False
        good["stereo_lidar_fallback"] = False
        self.assertTrue(CONTRACT.validate_stereo_source_contract("stereo", good))
        bad = dict(good)
        bad["img_topic"] = "/stereo/right/image_raw/compressed"
        with self.assertRaises(ValueError):
            CONTRACT.validate_stereo_source_contract("stereo", bad)
        with self.assertRaises(ValueError):
            CONTRACT.validate_stereo_source_contract("lidar", good)
        bad = dict(good)
        bad["stereo_lidar_fallback"] = True
        with self.assertRaises(ValueError):
            CONTRACT.validate_stereo_source_contract("stereo", bad)
        bad = dict(good)
        bad["detections_rectified"] = True
        with self.assertRaises(ValueError):
            CONTRACT.validate_stereo_source_contract("stereo", bad)

    def test_detection_frames_reject_duplicates_nonpositive_and_size_drift(self):
        good = [{"stamp_ns": 1, "w": 640, "h": 480, "dets": []},
                {"stamp_ns": 2, "w": 640, "h": 480, "dets": []}]
        self.assertEqual(len(CONTRACT.validate_detection_frames(good)), 2)
        cases = [
            [dict(good[0]), dict(good[0])],
            [{"stamp_ns": 0, "w": 640, "h": 480, "dets": []}],
            [good[0], {"stamp_ns": 2, "w": 320, "h": 240, "dets": []}],
        ]
        for frames in cases:
            with self.assertRaises(ValueError):
                CONTRACT.validate_detection_frames(frames)

    def test_tf_miss_classes_separate_startup_slip_and_chain_break(self):
        classify = CONTRACT.classify_stereo_tf_miss
        # Verbatim bag-19 failure: map frame not replayed into the buffer
        # yet at bag open.  Before any successful lookup this is warm-up;
        # after one it means the chain is genuinely gone.
        missing = ('"map" passed to lookupTransform argument target_frame '
                   'does not exist.')
        self.assertEqual(classify(missing, False), CONTRACT.TF_MISS_STARTUP)
        self.assertEqual(classify(missing, True), CONTRACT.TF_MISS_BROKEN)
        future = ("Lookup would require extrapolation into the future.  "
                  "Requested time 100.501 but the latest data is at time "
                  "100.495, when looking up transform from frame [odom] "
                  "to frame [map]")
        self.assertEqual(classify(future, False), CONTRACT.TF_MISS_FUTURE)
        self.assertEqual(classify(future, True), CONTRACT.TF_MISS_FUTURE)
        past = ("Lookup would require extrapolation into the past.  "
                "Requested time 5.0 but the earliest data is at time 8.0")
        self.assertEqual(classify(past, False), CONTRACT.TF_MISS_STARTUP)
        self.assertEqual(classify(past, True), CONTRACT.TF_MISS_BROKEN)
        connectivity = ("Could not find a connection between 'map' and "
                        "'stereo_left_optical' because they are not part "
                        "of the same tree.")
        self.assertEqual(classify(connectivity, False),
                         CONTRACT.TF_MISS_BROKEN)
        self.assertEqual(classify(connectivity, True),
                         CONTRACT.TF_MISS_BROKEN)

    def test_latest_transform_fallback_accepts_only_bounded_slip(self):
        ok = CONTRACT.latest_tf_fallback_ok
        self.assertTrue(ok(10.0, 9.995, 0.25))
        self.assertTrue(ok(10.0, 10.2, 0.25))
        self.assertFalse(ok(10.0, 9.0, 0.25))
        self.assertFalse(ok(10.0, 9.995, -0.1))
        self.assertFalse(ok(float("nan"), 9.995, 0.25))
        self.assertFalse(ok(None, 9.995, 0.25))
        self.assertFalse(ok(10.0, None, 0.25))

    def test_replay_aborts_only_on_a_locked_invalid_status(self):
        abort = CONTRACT.replay_abort_reason
        self.assertEqual(abort(False, "INVALID: duplicate det stamp"),
                         "INVALID: duplicate det stamp")
        self.assertIsNone(abort(True, "OK: det_rx=5 sync_rx=5"))
        self.assertIsNone(abort(False, "NOT_READY: waiting"))
        self.assertIsNone(abort(False, None))
        self.assertIsNone(abort(False, ""))
        # A healthy probe must never abort even with odd text.
        self.assertIsNone(abort(True, "INVALID: impossible"))

    def test_pipeline_barrier_requires_all_frames_and_quiet_period(self):
        self.assertFalse(CONTRACT.pipeline_ready(
            5, {"det_rx": 5, "sync_rx": 4}, 9.0, 12.0, 1.0)[0])
        self.assertFalse(CONTRACT.pipeline_ready(
            5, {"det_rx": 5, "sync_rx": 5}, 11.5, 12.0, 1.0)[0])
        self.assertTrue(CONTRACT.pipeline_ready(
            5, {"det_rx": 5, "sync_rx": 5}, 10.0, 12.0, 1.0)[0])
        self.assertFalse(CONTRACT.pipeline_ready(
            5, {"det_rx": 6, "sync_rx": 5}, 10.0, 12.0, 1.0)[0])
        self.assertFalse(CONTRACT.pipeline_ready(
            5, {"det_rx": 5, "sync_rx": 5,
                "callbacks_inflight": 1}, 10.0, 12.0, 1.0)[0])
        valid, detail = CONTRACT.pipeline_ready(
            5, {"det_rx": 5, "sync_rx": 5,
                "infrastructure_drop": 1}, 10.0, 12.0, 1.0)
        self.assertFalse(valid)
        self.assertTrue(detail.startswith("INVALID:"))
        self.assertTrue(CONTRACT.pipeline_ready(
            2, {"det_rx": 2, "sync_rx": 2}, 10.0, 12.0, 1.0,
            expected_stamps={11, 22}, det_stamps={11, 22},
            sync_stamps={11, 22})[0])
        self.assertFalse(CONTRACT.pipeline_ready(
            2, {"det_rx": 2, "sync_rx": 2}, 10.0, 12.0, 1.0,
            expected_stamps={11, 22}, det_stamps={11},
            sync_stamps={11})[0])
        valid, detail = CONTRACT.pipeline_ready(
            2, {"det_rx": 2, "sync_rx": 2}, 10.0, 12.0, 1.0,
            expected_stamps={11, 22}, det_stamps={11},
            sync_stamps={11}, pipeline_error="duplicate det callback stamp 11")
        self.assertFalse(valid)
        self.assertTrue(detail.startswith("INVALID:"))

    def test_bounded_unsynced_budget_commits_instead_of_timing_out(self):
        # A stamp whose exact-timestamp disparity never existed can never
        # sync once the input is sealed; within the budget the barrier
        # passes and reports the shortfall instead of timing out the whole
        # replay into no yaml.
        stats = {"det_rx": 3, "sync_rx": 2}
        ready, detail = CONTRACT.pipeline_ready(
            3, stats, 10.0, 12.0, 1.0,
            expected_stamps={11, 22, 33}, det_stamps={11, 22, 33},
            sync_stamps={11, 22}, max_unsynced_frames=1)
        self.assertTrue(ready)
        self.assertIn("unsynced=1", detail)
        # The same shortfall still waits when the budget is zero.
        self.assertFalse(CONTRACT.pipeline_ready(
            3, stats, 10.0, 12.0, 1.0,
            expected_stamps={11, 22, 33}, det_stamps={11, 22, 33},
            sync_stamps={11, 22})[0])
        # The detection side stays exact regardless of the budget.
        self.assertFalse(CONTRACT.pipeline_ready(
            3, {"det_rx": 2, "sync_rx": 2}, 10.0, 12.0, 1.0,
            expected_stamps={11, 22, 33}, det_stamps={11, 22},
            sync_stamps={11, 22}, max_unsynced_frames=1)[0])
        # Budgets that could excuse every frame are rejected outright.
        ready, detail = CONTRACT.pipeline_ready(
            3, stats, 10.0, 12.0, 1.0,
            expected_stamps={11, 22, 33}, det_stamps={11, 22, 33},
            sync_stamps={11, 22}, max_unsynced_frames=3)
        self.assertFalse(ready)
        self.assertTrue(detail.startswith("INVALID:"))
        self.assertFalse(CONTRACT.pipeline_ready(
            3, stats, 10.0, 12.0, 1.0,
            expected_stamps={11, 22, 33}, det_stamps={11, 22, 33},
            sync_stamps={11, 22}, max_unsynced_frames=-1)[0])


if __name__ == "__main__":
    unittest.main()
