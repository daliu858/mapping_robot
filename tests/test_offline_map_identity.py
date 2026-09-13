"""PC-side regression tests for semantic-map provenance binding."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "merge_semantic_layers.py")
SPEC = importlib.util.spec_from_file_location("merge_semantic_layers", SCRIPT)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)
PREFLIGHT_SCRIPT = os.path.join(ROOT, "tools", "offline_preflight.py")
PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "offline_preflight", PREFLIGHT_SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT)


class OfflineMapIdentityTest(unittest.TestCase):
    def setUp(self):
        self.objects_path = os.path.join(
            ROOT, "maps", "mymap_smoke_objects.yaml")
        self.topology_path = os.path.join(
            ROOT, "maps", "mymap_smoke_topology.yaml")
        with open(self.objects_path,
                  encoding="utf-8") as stream:
            self.objects = yaml.safe_load(stream)
        with open(self.topology_path,
                  encoding="utf-8") as stream:
            self.topology = yaml.safe_load(stream)

    def test_matching_hash_is_preserved_in_final_map(self):
        result = MERGE.merge_layers(
            copy.deepcopy(self.objects), copy.deepcopy(self.topology),
            self.objects_path, self.topology_path)
        self.assertEqual(result["frame"], "map")
        self.assertEqual(result["map_image_sha256"],
                         self.objects["map_image_sha256"])
        self.assertEqual(len(result["poi"]), 3)
        self.assertEqual(
            ["chair", "table", "door"],
            result["poi_contract"]["required_labels"])

    def test_mismatched_hash_is_rejected(self):
        objects = copy.deepcopy(self.objects)
        objects["map_image_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            MERGE.merge_layers(
                objects, copy.deepcopy(self.topology),
                self.objects_path, self.topology_path)

    def test_missing_hash_is_rejected(self):
        objects = copy.deepcopy(self.objects)
        del objects["map_image_sha256"]
        with self.assertRaises(ValueError):
            MERGE.merge_layers(
                objects, copy.deepcopy(self.topology),
                self.objects_path, self.topology_path)

    def test_missing_required_poi_is_rejected(self):
        objects = copy.deepcopy(self.objects)
        objects["objects"] = [item for item in objects["objects"]
                              if item["label"] != "door"]
        with self.assertRaises(ValueError):
            MERGE.merge_layers(
                objects, copy.deepcopy(self.topology),
                self.objects_path, self.topology_path)

    def test_pc_preflight_accepts_copied_sibling_of_jetson_absolute_pgm(self):
        payload = b"P5\n1 1\n255\n\x00"
        with tempfile.TemporaryDirectory() as directory:
            pgm = os.path.join(directory, "site.pgm")
            yaml_path = os.path.join(directory, "site.yaml")
            with open(pgm, "wb") as stream:
                stream.write(payload)
            with open(yaml_path, "w", encoding="utf-8") as stream:
                yaml.safe_dump({
                    "image": "/home/jetbot/maps/site.pgm",
                    "resolution": 0.05,
                    "origin": [0.0, 0.0, 0.0],
                    "negate": 0,
                    "free_thresh": 0.196,
                    "occupied_thresh": 0.65,
                }, stream)
            self.assertEqual(
                PREFLIGHT.map_image_sha256(yaml_path),
                hashlib.sha256(payload).hexdigest())

    def test_melodic_legacy_static_tf_does_not_require_tf_static_topic(self):
        self.assertIn("/tf", PREFLIGHT.REQUIRED_BAG_TOPICS)
        self.assertNotIn("/tf_static", PREFLIGHT.REQUIRED_BAG_TOPICS)

    def test_preflight_requires_completed_recording_transaction(self):
        self.assertIn("/survey/recording_status",
                      PREFLIGHT.REQUIRED_BAG_TOPICS)
        token = "a" * 32
        map_hash = "b" * 64
        manifest_hash = "c" * 64
        camera_mount = [0.1, 0.0, 0.2, -1.5708, 0.0, -1.5708]
        payloads = [
            json.dumps({"schema": 1, "state": "started", "token": token,
                        "pose_source": "gmapping",
                        "map_image_sha256": map_hash,
                        "map_manifest_sha256": manifest_hash,
                        "camera_mount": camera_mount}),
            json.dumps({"schema": 1, "state": "completed", "token": token,
                        "pose_source": "gmapping",
                        "map_image_sha256": map_hash,
                        "map_manifest_sha256": manifest_hash,
                        "camera_mount": camera_mount}),
        ]
        result = PREFLIGHT.validate_recording_states(
            payloads, map_hash, manifest_hash)
        self.assertEqual("completed", result["state"])

    def test_preflight_rejects_failed_or_truncated_recording(self):
        token = "a" * 32
        map_hash = "b" * 64
        manifest_hash = "c" * 64
        started = json.dumps({"schema": 1, "state": "started",
                              "token": token,
                              "pose_source": "gmapping",
                              "map_image_sha256": map_hash,
                              "map_manifest_sha256": manifest_hash})
        with self.assertRaises(RuntimeError):
            PREFLIGHT.validate_recording_states([started], map_hash)
        failed = json.dumps({"schema": 1, "state": "failed",
                             "token": token,
                             "pose_source": "gmapping",
                             "map_image_sha256": map_hash,
                             "map_manifest_sha256": manifest_hash,
                             "reason": "sensor stale"})
        with self.assertRaises(RuntimeError):
            PREFLIGHT.validate_recording_states([started, failed], map_hash)

    def test_preflight_accepts_latched_republish_but_rejects_bad_order(self):
        token = "a" * 32
        map_hash = "b" * 64
        manifest_hash = "c" * 64
        camera_mount = [0.1, 0.0, 0.2, -1.5708, 0.0, -1.5708]

        def marker(state, digest=map_hash):
            return json.dumps({
                "schema": 1, "state": state, "token": token,
                "pose_source": "gmapping",
                "map_image_sha256": digest,
                "map_manifest_sha256": manifest_hash,
                "camera_mount": camera_mount})

        with self.assertRaises(RuntimeError):
            PREFLIGHT.validate_recording_states([
                marker("completed"), marker("started")], map_hash)
        result = PREFLIGHT.validate_recording_states([
            marker("started"), marker("started"),
            marker("completed")], map_hash)
        self.assertEqual(result["state"], "completed")
        with self.assertRaises(RuntimeError):
            PREFLIGHT.validate_recording_states([
                marker("started", "c" * 64), marker("completed")], map_hash)
        with self.assertRaises(RuntimeError):
            PREFLIGHT.validate_recording_states([
                marker("started"), marker("unexpected"),
                marker("completed")], map_hash)

    def test_preflight_uses_same_llmdet_large_default(self):
        self.assertEqual("iSEE-Laboratory/llmdet_large",
                         PREFLIGHT.DEFAULT_MODEL)

    def test_preflight_uses_formal_household_queries(self):
        self.assertEqual(
            ["chair", "table", "dining table", "closed door"],
            PREFLIGHT.POI_QUERIES)

    def test_preflight_checks_hf_home_volume_not_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "not-created" / "hf-cache"
            usage = SimpleNamespace(free=7 * 1024 ** 3)
            with mock.patch.object(
                    PREFLIGHT.shutil, "disk_usage",
                    return_value=usage) as disk_usage:
                free_gib, probe = PREFLIGHT.free_space_gib(cache)
            self.assertEqual(7.0, free_gib)
            self.assertEqual(Path(directory), probe)
            disk_usage.assert_called_once_with(str(Path(directory)))

    def test_preflight_rejects_hf_home_that_is_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "hf-cache"
            cache.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                PREFLIGHT.free_space_gib(cache)


if __name__ == "__main__":
    unittest.main()
