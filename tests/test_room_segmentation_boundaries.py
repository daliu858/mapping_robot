import copy
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOM = load_module("room_segment_under_test", "tools/room_segment.py")
VORONOI = load_module(
    "room_segment_voronoi_under_test", "tools/room_segment_voronoi.py")
BORMANN = load_module(
    "room_segment_bormann_under_test", "tools/room_segment_bormann.py")
MERGE = load_module(
    "merge_semantic_layers_under_test", "tools/merge_semantic_layers.py")


class EdgeAdjacencyTests(unittest.TestCase):
    def test_pixel_to_map_uses_ros_bottom_left_origin(self):
        top_left = ROOM.px_to_map(0, 0, 100, 0.05, [10.0, 20.0])
        bottom_left = ROOM.px_to_map(0, 99, 100, 0.05, [10.0, 20.0])
        self.assertEqual([10.025, 24.975], top_left)
        self.assertEqual([10.025, 20.025], bottom_left)

    def test_opposite_image_edges_are_not_room_neighbors(self):
        labels = np.zeros((3, 5), dtype=np.int32)
        labels[1, 0] = 1
        labels[1, -1] = 2

        self.assertEqual(ROOM.neighbors_table(labels), {})
        self.assertEqual(VORONOI.neighbors_table(labels), {})

    def test_real_in_image_adjacency_is_still_counted(self):
        labels = np.zeros((3, 5), dtype=np.int32)
        labels[1, 1] = 1
        labels[1, 2] = 2

        for table in (ROOM.neighbors_table(labels),
                      VORONOI.neighbors_table(labels)):
            self.assertEqual(table[1][2], 1)
            self.assertEqual(table[2][1], 1)

    def test_border_stats_do_not_wrap_neighbor_or_wall_contacts(self):
        labels = np.zeros((4, 5), dtype=np.int32)
        labels[0, 2] = 1
        labels[-1, 2] = 2
        free = np.ones(labels.shape, dtype=bool)
        free[-1, 1:4] = False
        free[-1, 2] = True  # label 2 remains valid free space

        neighbors, wall_border, total_border = VORONOI._border_stats(
            labels, free)

        self.assertNotIn(2, neighbors.get(1, {}))
        self.assertNotIn(1, neighbors.get(2, {}))
        self.assertEqual(wall_border.get(1, 0), 0)
        self.assertEqual(total_border.get(1, 0), 0)

    def test_connected_components_helper_really_uses_8_connectivity(self):
        diagonal = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        for helper in (ROOM._connected_components_8,
                       VORONOI._connected_components_8):
            count, labels = helper(diagonal)
            self.assertEqual(count, 2)  # background plus one diagonal component
            self.assertEqual(labels[0, 0], labels[1, 1])


class BormannSeedTests(unittest.TestCase):
    def test_empty_seed_set_is_rejected_before_wavefront(self):
        free = np.ones((3, 3), dtype=np.uint8)
        seeds = np.zeros_like(free, dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "no room-center seeds"):
            BORMANN.wavefront_propagation(seeds, free)

    def test_subpixel_distance_transform_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "one pixel"):
            BORMANN.choose_room_center_seeds(
                np.full((3, 3), 0.75, dtype=np.float32))

    def test_equal_component_counts_choose_highest_threshold(self):
        dist = np.array([[0, 2, 0, 2, 0]], dtype=np.float32)
        threshold, count, seeds, trace = BORMANN.choose_room_center_seeds(dist)
        self.assertEqual((threshold, count), (2, 2))
        self.assertEqual(trace, [(2, 2), (1, 2)])
        self.assertEqual(set(np.unique(seeds)), {0, 1, 2})


class RoomMapIdentityTests(unittest.TestCase):
    def test_watershed_topology_contains_exact_source_map_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            pgm = temp / "map.pgm"
            out = temp / "segmented"
            raw = np.zeros((80, 80), dtype=np.uint8)
            raw[20:60, 20:60] = 255
            Image.fromarray(raw).save(pgm)
            map_yaml = temp / "map.yaml"
            map_yaml.write_text(yaml.safe_dump({
                "image": "map.pgm", "resolution": 0.05,
                "origin": [0.0, 0.0, 0.0], "negate": 0,
                "occupied_thresh": 0.65, "free_thresh": 0.196,
            }), encoding="utf-8")
            identity = ROOM.load_map_identity(
                str(map_yaml), require_zero_yaw=True)
            objects_path = temp / "objects.yaml"
            objects_path.write_text(yaml.safe_dump({
                "schema": "slam-car-semantic-objects-v1",
                "completed": True,
                "frame": "map",
                "recording_token": "watershed-test-run",
                "source_bag_sha256": "c" * 64,
                "camera_mount": [0, 0, 0, 0, 0, 0],
                "map_image_sha256": identity["map_image_sha256"],
                "map_manifest_sha256": identity["map_manifest_sha256"],
                "objects": [],
            }), encoding="utf-8")

            argv = ["room_segment.py", str(pgm),
                    "--yaml", str(map_yaml),
                    "--objects", str(objects_path),
                    "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(ROOM, "visualize", return_value="unused"):
                ROOM.main()

            with open(str(out) + "_topology.yaml", "r", encoding="utf-8") as stream:
                topology = yaml.safe_load(stream)
            expected = hashlib.sha256(pgm.read_bytes()).hexdigest()
            self.assertEqual(topology["map_image_sha256"], expected)
            self.assertEqual(topology["map_manifest_sha256"],
                             identity["map_manifest_sha256"])
            self.assertEqual(topology["source_objects_sha256"],
                             hashlib.sha256(
                                 objects_path.read_bytes()).hexdigest())
            self.assertTrue(topology["completed"])


class SemanticTieBreakTests(unittest.TestCase):
    def _documents(self, labels):
        digest = "a" * 64
        objects = {
            "schema": "slam-car-semantic-objects-v1",
            "completed": True,
            "frame": "map",
            "recording_token": "tie-break-test-run",
            "source_bag_sha256": "c" * 64,
            "camera_mount": [0, 0, 0, 0, 0, 0],
            "map_image_sha256": digest,
            "map_manifest_sha256": "b" * 64,
            "objects": [
                {"label": label, "x": 1.0, "y": 1.0, "hits": 1}
                for label in labels
            ],
        }
        topology = {
            "schema": "slam-car-room-topology-v1",
            "completed": True,
            "frame": "map",
            "recording_token": "tie-break-test-run",
            "source_bag_sha256": "c" * 64,
            "camera_mount": [0, 0, 0, 0, 0, 0],
            "map_image_sha256": digest,
            "map_manifest_sha256": "b" * 64,
            "origin": [0.0, 0.0, 0.0],
            "resolution": 0.05,
            "rooms": [{
                "id": 1,
                "polygon_map": [[0.0, 0.0], [2.0, 0.0],
                                [2.0, 2.0], [0.0, 2.0]],
            }],
            "doors": [],
            "topology": [],
        }
        return objects, topology

    def _semantic_for(self, labels):
        objects, topology = self._documents(labels)
        with tempfile.TemporaryDirectory() as temp_dir:
            objects_path = Path(temp_dir) / "objects.yaml"
            topology_path = Path(temp_dir) / "topology.yaml"
            objects_path.write_text(
                yaml.safe_dump(objects), encoding="utf-8")
            topology["source_objects_sha256"] = hashlib.sha256(
                objects_path.read_bytes()).hexdigest()
            topology_path.write_text(
                yaml.safe_dump(topology), encoding="utf-8")
            merged = MERGE.merge_layers(
                copy.deepcopy(objects), copy.deepcopy(topology),
                str(objects_path), str(topology_path))
        return merged["rooms"][0]["semantic"]

    def test_equal_type_votes_do_not_depend_on_detection_order(self):
        # The current home-demo contract requires chair, table and door.
        # Include them while keeping this test focused on the deterministic
        # tie-break between the two type-defining POIs.
        required = ["chair", "table", "door"]
        forward = required + ["staircase", "elevator_door",
                   "vending_or_water", "exit_sign"]
        reverse = required + ["elevator_door", "staircase",
                   "vending_or_water", "exit_sign"]
        self.assertEqual(self._semantic_for(forward), "elevator_hall")
        self.assertEqual(self._semantic_for(reverse), "elevator_hall")


if __name__ == "__main__":
    unittest.main()
