"""Adversarial provenance and atomic-commit tests for semantic artifacts."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = load_module(
    "semantic_artifact_contract_tested",
    ROOT / "tools" / "semantic_artifact_contract.py")
MERGE = load_module(
    "merge_semantic_layers_contract_tested",
    ROOT / "tools" / "merge_semantic_layers.py")
class SemanticArtifactContractTests(unittest.TestCase):
    def _objects(self):
        return {
            "schema": CONTRACT.OBJECTS_SCHEMA,
            "completed": True,
            "frame": "map",
            "recording_token": "recording-run-a",
            "source_bag_sha256": "c" * 64,
            "camera_mount": [0.10, 0.0, 0.20, -1.5708, 0.0, -1.5708],
            "map_image_sha256": "a" * 64,
            "map_manifest_sha256": "b" * 64,
            "objects": [
                {"label": "chair", "x": 1.0, "y": 1.0, "hits": 3},
                {"label": "table", "x": 1.2, "y": 1.0, "hits": 2},
                {"label": "door", "x": 1.4, "y": 1.0, "hits": 3},
            ],
        }

    def _topology(self, objects_sha):
        return {
            "schema": CONTRACT.TOPOLOGY_SCHEMA,
            "completed": True,
            "frame": "map",
            "recording_token": "recording-run-a",
            "source_bag_sha256": "c" * 64,
            "camera_mount": [0.10, 0.0, 0.20, -1.5708, 0.0, -1.5708],
            "map_image_sha256": "a" * 64,
            "map_manifest_sha256": "b" * 64,
            "source_objects_sha256": objects_sha,
            "resolution": 0.05,
            "origin": [-5.0, -6.0, 0.0],
            "rooms": [{
                "id": 1, "status": "room", "area_m2": 4.0,
                "centroid_map": [1.0, 1.0],
                "polygon_map": [[0.0, 0.0], [2.0, 0.0],
                                [2.0, 2.0], [0.0, 2.0]],
            }],
            "doors": [],
            "topology": [],
        }

    def _write_pair(self, directory, objects=None, topology_mutator=None):
        directory = Path(directory)
        objects_path = directory / "objects.yaml"
        topology_path = directory / "topology.yaml"
        objects = copy.deepcopy(objects if objects is not None
                                else self._objects())
        objects_path.write_text(
            yaml.safe_dump(objects, sort_keys=False), encoding="utf-8")
        topology = self._topology(hashlib.sha256(
            objects_path.read_bytes()).hexdigest())
        if topology_mutator:
            topology_mutator(topology)
        topology_path.write_text(
            yaml.safe_dump(topology, sort_keys=False), encoding="utf-8")
        return objects_path, topology_path

    def _merge_paths(self, objects_path, topology_path):
        objects = MERGE._load_yaml(str(objects_path), "objects")
        topology = MERGE._load_yaml(str(topology_path), "topology")
        return MERGE.merge_layers(
            objects, topology, str(objects_path), str(topology_path))

    def test_complete_pair_produces_self_checking_three_poi_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            objects_path, topology_path = self._write_pair(directory)
            result = self._merge_paths(objects_path, topology_path)
            self.assertEqual(result["objects_artifact_sha256"],
                             hashlib.sha256(
                                 objects_path.read_bytes()).hexdigest())
            self.assertEqual(result["topology_artifact_sha256"],
                             hashlib.sha256(
                                 topology_path.read_bytes()).hexdigest())
        self.assertEqual(result["schema"], CONTRACT.SEMANTIC_MAP_SCHEMA)
        self.assertTrue(result["completed"])
        self.assertEqual(
            ["chair", "table", "door"],
            result["poi_contract"]["required_labels"])
        self.assertTrue(CONTRACT.validate_semantic_map_document(result))

    def test_old_partial_and_truncated_objects_are_rejected(self):
        for mutation in ("old", "partial"):
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as directory:
                objects = self._objects()
                if mutation == "old":
                    objects.pop("schema")
                else:
                    objects["completed"] = False
                pair = self._write_pair(directory, objects=objects)
                with self.assertRaises(ValueError):
                    self._merge_paths(*pair)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.yaml"
            path.write_text("schema: [\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MERGE._load_yaml(str(path), "objects")

        topology_mutations = (
            lambda topology: topology.pop("schema"),
            lambda topology: topology.update(completed=False),
        )
        for mutation in topology_mutations:
            with self.subTest(topology_mutation=mutation), \
                    tempfile.TemporaryDirectory() as directory:
                pair = self._write_pair(
                    directory, topology_mutator=mutation)
                with self.assertRaises(ValueError):
                    self._merge_paths(*pair)

    def test_cross_run_bag_token_map_manifest_and_mount_are_rejected(self):
        mutations = (
            lambda topology: topology.update(recording_token="recording-run-b"),
            lambda topology: topology.update(source_bag_sha256="d" * 64),
            lambda topology: topology.update(map_image_sha256="d" * 64),
            lambda topology: topology.update(map_manifest_sha256="c" * 64),
            lambda topology: topology.update(
                camera_mount=[0.11, 0, 0.20, -1.5708, 0, -1.5708]),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as directory:
                pair = self._write_pair(directory, topology_mutator=mutation)
                with self.assertRaises(ValueError):
                    self._merge_paths(*pair)

    def test_modified_source_objects_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            objects_path, topology_path = self._write_pair(directory)
            with open(objects_path, "a", encoding="utf-8") as stream:
                stream.write("# changed after topology generation\n")
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._merge_paths(objects_path, topology_path)

    def test_missing_doors_topology_or_core_poi_is_rejected(self):
        for missing_field in ("doors", "topology"):
            with self.subTest(field=missing_field), \
                    tempfile.TemporaryDirectory() as directory:
                pair = self._write_pair(
                    directory,
                    topology_mutator=lambda topology, field=missing_field:
                    topology.pop(field))
                with self.assertRaises(ValueError):
                    self._merge_paths(*pair)
        with tempfile.TemporaryDirectory() as directory:
            objects = self._objects()
            objects["objects"] = [
                item for item in objects["objects"]
                if item["label"] != "door"]
            pair = self._write_pair(directory, objects=objects)
            with self.assertRaisesRegex(ValueError, "door"):
                self._merge_paths(*pair)

    def test_atomic_yaml_failure_preserves_old_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "semantic.yaml"
            target.write_text("old-complete-artifact\n", encoding="utf-8")
            with mock.patch.object(
                    CONTRACT.yaml, "safe_dump",
                    side_effect=RuntimeError("injected serialization error")):
                with self.assertRaises(RuntimeError):
                    CONTRACT.atomic_yaml_dump({"new": True}, str(target))
            self.assertEqual(target.read_text(encoding="utf-8"),
                             "old-complete-artifact\n")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_atomic_yaml_uses_same_directory_os_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "semantic.yaml"
            with mock.patch.object(
                    CONTRACT.os, "replace",
                    wraps=CONTRACT.os.replace) as replace:
                CONTRACT.atomic_yaml_dump({"completed": True}, str(target))
            replace.assert_called_once()
            temporary, replaced_target = replace.call_args[0]
            self.assertEqual(Path(temporary).parent, target.parent)
            self.assertEqual(Path(replaced_target), target)
            self.assertEqual(
                yaml.safe_load(target.read_text(encoding="utf-8")),
                {"completed": True})

    def test_final_map_rejects_changed_manifest_and_missing_poi(self):
        with tempfile.TemporaryDirectory() as directory:
            pair = self._write_pair(directory)
            result = self._merge_paths(*pair)
        changed = copy.deepcopy(result)
        changed["recording_token"] = "different-run"
        with self.assertRaisesRegex(ValueError, "checksum"):
            CONTRACT.validate_semantic_map_document(changed)

        changed_coordinate = copy.deepcopy(result)
        changed_coordinate["poi"][0]["x"] += 0.25
        with self.assertRaisesRegex(ValueError, "content checksum"):
            CONTRACT.validate_semantic_map_document(changed_coordinate)

        partial = copy.deepcopy(result)
        partial["completed"] = False
        with self.assertRaisesRegex(ValueError, "not a completed"):
            CONTRACT.validate_semantic_map_document(partial)

        missing = copy.deepcopy(result)
        missing["poi"] = [item for item in missing["poi"]
                          if item["label"] != "chair"]
        missing["poi_contract"]["observed_labels"] = sorted(
            item["label"] for item in missing["poi"])
        missing["semantic_manifest_sha256"] = \
            CONTRACT.semantic_manifest_sha256(missing)
        with self.assertRaisesRegex(ValueError, "missing"):
            CONTRACT.validate_semantic_map_document(missing)


if __name__ == "__main__":
    unittest.main()
