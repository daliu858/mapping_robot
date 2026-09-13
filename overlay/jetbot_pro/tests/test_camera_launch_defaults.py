#!/usr/bin/env python
"""Static regression tests for mono/stereo camera launch defaults."""
from __future__ import print_function

import os
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH_DIR = os.path.join(PACKAGE_ROOT, "launch")


def _args(filename):
    root = ET.parse(os.path.join(LAUNCH_DIR, filename)).getroot()
    return {node.get("name"): node.get("default")
            for node in root.findall("arg")}


class CameraLaunchDefaultsTest(unittest.TestCase):
    def test_installed_stereo_calibration_is_finite_and_has_forward_baseline(self):
        calibration_dir = os.path.join(
            PACKAGE_ROOT, "config", "camera_calibration")
        documents = []
        for side in ("left", "right"):
            path = os.path.join(
                calibration_dir, "stereo_%s_640x480.yaml" % side)
            with open(path, "r") as stream:
                document = yaml.safe_load(stream)
            self.assertEqual(document["camera_name"], side)
            self.assertEqual(document["image_width"], 640)
            self.assertEqual(document["image_height"], 480)
            for field in ("camera_matrix", "distortion_coefficients",
                          "rectification_matrix", "projection_matrix"):
                values = np.asarray(document[field]["data"], dtype=float)
                self.assertTrue(np.all(np.isfinite(values)),
                                "%s contains non-finite %s" % (side, field))
            rotation = np.asarray(
                document["rectification_matrix"]["data"],
                dtype=float).reshape(3, 3)
            self.assertGreater(np.linalg.det(rotation), 0.5)
            documents.append(document)

        left_projection = documents[0]["projection_matrix"]["data"]
        right_projection = documents[1]["projection_matrix"]["data"]
        self.assertAlmostEqual(float(left_projection[3]), 0.0, places=9)
        self.assertLess(float(right_projection[3]), 0.0)
        baseline = -float(right_projection[3]) / float(right_projection[0])
        self.assertGreater(baseline, 0.055)
        self.assertLess(baseline, 0.065)

    def test_camera_process_death_tears_down_owning_launch(self):
        filename = os.path.join(LAUNCH_DIR, "csi_camera.launch")
        root = ET.parse(filename).getroot()
        self.assertEqual(_args("csi_camera.launch")["required"], "true")
        self.assertFalse(_args("csi_camera.launch")["frame_id"].startswith("/"))
        node = next(item for item in root.findall("node")
                    if item.get("type") == "gscam")
        self.assertEqual(node.get("required"), "$(arg required)")

    def test_enabled_stereo_processing_nodes_are_required(self):
        filename = os.path.join(LAUNCH_DIR, "stereo_camera.launch")
        root = ET.parse(filename).getroot()
        nodes = {node.get("name"): node for node in root.iter("node")}
        for name in ("stereo_pair_gate", "stereo_image_proc"):
            self.assertEqual(nodes[name].get("required"), "true")

    def test_all_stereo_entrypoints_keep_verified_sensor_mapping(self):
        filenames = (
            "camera_select.launch",
            "nav.launch",
            "semantic_map.launch",
            "slam.launch",
            "slam_nav.launch",
            "stereo_calibrate_headless.launch",
            "stereo_camera.launch",
        )
        for filename in filenames:
            args = _args(filename)
            self.assertEqual(args["left_sensor_id"], "1", filename)
            self.assertEqual(args["right_sensor_id"], "0", filename)

    def test_direct_stereo_capture_defaults_to_robot_verified_rate(self):
        args = _args("stereo_camera.launch")
        self.assertEqual(args["left_sensor_id"], "1")
        self.assertEqual(args["right_sensor_id"], "0")
        self.assertEqual(args["fps"], "30")
        self.assertEqual(args["start_pair_gate"],
                         "$(arg start_stereo_image_proc)")
        self.assertEqual(args["max_pair_dt_s"], "0.020")

    def test_direct_stereo_disparity_keeps_verified_near_field_range(self):
        filename = os.path.join(LAUNCH_DIR, "stereo_camera.launch")
        root = ET.parse(filename).getroot()
        self.assertEqual(_args("stereo_camera.launch")["disparity_range"],
                         "128")
        node = next(item for item in root.iter("node")
                    if item.get("name") == "stereo_image_proc")
        params = {item.get("name"): item.get("value")
                  for item in node.findall("param")}
        self.assertEqual(params["disparity_range"],
                         "$(arg disparity_range)")

    def test_stereo_detection_and_disparity_share_left_optical_frame(self):
        root = ET.parse(os.path.join(
            LAUNCH_DIR, "stereo_camera.launch")).getroot()
        camera_frames = []
        for include in root.findall("./group/include"):
            values = {node.get("name"): node.get("value")
                      for node in include.findall("arg")}
            camera_frames.append(values.get("frame_id"))
        self.assertEqual(camera_frames,
                         ["stereo_left_optical", "stereo_left_optical"])
        self.assertEqual(
            _args("semantic_map_offline.launch")["optical_frame"],
            "stereo_left_optical")

    def test_selector_keeps_distinct_stereo_and_legacy_mono_rates(self):
        args = _args("camera_select.launch")
        self.assertEqual(args["camera_mode"], "stereo")
        self.assertEqual(
            args["fps"],
            "$(eval '30' if arg('camera_mode') == 'stereo' else '20')")

    def test_headless_calibration_is_strict_and_never_starts_disparity(self):
        filename = os.path.join(LAUNCH_DIR, "stereo_calibrate_headless.launch")
        root = ET.parse(filename).getroot()
        args = _args("stereo_calibrate_headless.launch")
        self.assertEqual(args["max_pair_dt_s"], "0.020")
        self.assertEqual(args["board_columns"], "8")
        self.assertEqual(args["board_rows"], "6")
        self.assertEqual(args["square_size_m"], "0.024")
        self.assertEqual(args["min_samples"], "5")
        self.assertEqual(args["max_samples"], "5")
        self.assertEqual(args["left_sensor_id"], "1")
        self.assertEqual(args["right_sensor_id"], "0")
        self.assertIn("stereo_five_shot_samples", args["sample_dir"])
        include = root.find("include")
        include_args = {node.get("name"): node.get("value")
                        for node in include.findall("arg")}
        self.assertEqual(include_args["start_pair_gate"], "true")
        self.assertEqual(include_args["start_stereo_image_proc"], "false")
        self.assertEqual(include_args["load_camera_info"], "false")
        self.assertNotIn("lidar", ET.tostring(root).decode("utf-8").lower())
        node = root.find("node")
        self.assertEqual(node.get("required"), "true")
        node_params = {param.get("name"): param.get("value")
                       for param in node.findall("param")}
        self.assertEqual(node_params["quality_min_views"], "5")
        self.assertEqual(node_params["sample_dir"], "$(arg sample_dir)")


if __name__ == "__main__":
    unittest.main()
