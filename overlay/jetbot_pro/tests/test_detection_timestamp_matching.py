#!/usr/bin/env python
"""Host-side tests for the offline detection timestamp safety contract."""
from __future__ import print_function

import os
import sys
import types
import unittest

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


def _load():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "detection_publisher.py")
    if importlib_util is not None:
        spec = importlib_util.spec_from_file_location(
            "detection_publisher_timestamp_test", path)
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import imp
    return imp.load_source("detection_publisher_timestamp_test", path)


dummy = type("Dummy", (), {})
_stub("rospy")
_stub("sensor_msgs.msg", CompressedImage=dummy, Image=dummy)
_stub(
    "vision_msgs.msg", Detection2DArray=dummy, Detection2D=dummy,
    ObjectHypothesisWithPose=dummy, VisionInfo=dummy)
_stub("offline_replay_runner", validate_recording_bag=lambda *_args: (True, {}))

PUBLISHER = _load()


class DetectionTimestampMatchingTest(unittest.TestCase):
    def setUp(self):
        self.index = {1000: ["left"], 1100: ["right"]}
        self.stamps = sorted(self.index)

    def test_exact_timestamp_matches_at_zero_tolerance(self):
        self.assertEqual(PUBLISHER.match_detections(
            self.index, self.stamps, 1000, 0), ["left"])

    def test_nearby_timestamp_cannot_match_at_zero_tolerance(self):
        self.assertIsNone(PUBLISHER.match_detections(
            self.index, self.stamps, 1001, 0))

    def test_nonzero_legacy_tolerance_is_inclusive(self):
        self.assertEqual(PUBLISHER.match_detections(
            self.index, self.stamps, 1001, 1), ["left"])
        self.assertIsNone(PUBLISHER.match_detections(
            self.index, self.stamps, 1002, 1))

    def test_empty_index_never_matches(self):
        self.assertIsNone(PUBLISHER.match_detections({}, [], 1000, 100))

    def test_publisher_eos_requires_every_expected_stamp(self):
        ready, detail = PUBLISHER.publisher_completion_state(
            [1000, 1100], [1000], 0)
        self.assertFalse(ready)
        self.assertTrue(detail.startswith("NOT_READY:"))

    def test_publisher_eos_requires_zero_inflight_callbacks(self):
        ready, detail = PUBLISHER.publisher_completion_state(
            [1000], [1000], 1)
        self.assertFalse(ready)
        self.assertTrue(detail.startswith("NOT_READY:"))

    def test_publisher_eos_waits_for_rosbag_transport_disconnect(self):
        ready, detail = PUBLISHER.publisher_completion_state(
            [1000], [1000], 0, input_connections=1)
        self.assertFalse(ready)
        self.assertTrue(detail.startswith("NOT_READY:"))

    def test_publisher_eos_rejects_terminal_error_and_extra_stamp(self):
        self.assertTrue(PUBLISHER.publisher_completion_state(
            [1000], [1000], 0, "bad image")[1].startswith("INVALID:"))
        self.assertTrue(PUBLISHER.publisher_completion_state(
            [1000], [1000, 1100], 0)[1].startswith("INVALID:"))

    def test_publisher_eos_succeeds_only_when_exactly_drained(self):
        ready, detail = PUBLISHER.publisher_completion_state(
            [1000, 1100], [1100, 1000], 0)
        self.assertTrue(ready)
        self.assertEqual(detail, "published=2/2")

    def test_runtime_log_format_strings_are_ascii(self):
        # Melodic rospy on Jetson defaults to ascii. A non-ASCII format
        # string in loginfo() crashes the node after indexing succeeds --
        # exactly how the first on-robot fusion run died, so every script
        # in the offline replay stack is checked, not just the publisher.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("detection_publisher.py", "semantic_mapper.py",
                     "offline_replay_runner.py", "replay_contract.py",
                     "stereo_pair_gate.py"):
            path = os.path.join(root, "scripts", name)
            with open(path, "rb") as handle:
                for line_no, raw in enumerate(handle, 1):
                    if b"rospy.log" not in raw:
                        continue
                    try:
                        raw.decode("ascii")
                    except UnicodeDecodeError:
                        self.fail("non-ASCII rospy.log in %s line %d: %r" % (
                            name, line_no, raw.strip()))


if __name__ == "__main__":
    unittest.main()
