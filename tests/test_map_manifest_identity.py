#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canonical map identity and rotated-map fail-closed regressions."""
from __future__ import print_function

import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JETSON_IDENTITY = _load(
    "jetson_map_identity_test",
    ROOT / "src" / "jetbot_pro" / "scripts" / "map_identity.py")
PC_IDENTITY = _load("pc_map_identity_test", ROOT / "tools" / "map_identity.py")


class CanonicalMapIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.image = self.directory / "map.pgm"
        self.image.write_bytes(b"P5\n2 2\n255\n\x00\xff\x80\x01")
        self.base = {
            "image": "map.pgm",
            "resolution": 0.05,
            "origin": [-1.25, 2.5, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name, document):
        path = self.directory / name
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return path

    def test_pc_and_jetson_hash_exactly_the_same_manifest(self):
        path = self._write("map.yaml", self.base)
        jetson = JETSON_IDENTITY.load_map_identity(path, require_zero_yaw=True)
        pc = PC_IDENTITY.load_map_identity(path, require_zero_yaw=True)
        self.assertEqual(jetson["map_image_sha256"], pc["map_image_sha256"])
        self.assertEqual(
            jetson["map_manifest_sha256"], pc["map_manifest_sha256"])
        self.assertEqual(jetson["manifest"], pc["manifest"])

    def test_yaml_key_order_and_equivalent_numbers_are_stable(self):
        first = self._write("first.yaml", self.base)
        reordered = {
            "free_thresh": 0.1960,
            "occupied_thresh": 0.6500,
            "negate": 0,
            "origin": [-1.250, 2.500, -0.0],
            "resolution": 0.0500,
            "image": "map.pgm",
        }
        second = self._write("second.yaml", reordered)
        one = JETSON_IDENTITY.load_map_identity(first)
        two = JETSON_IDENTITY.load_map_identity(second)
        self.assertEqual(one["map_manifest_sha256"],
                         two["map_manifest_sha256"])

    def test_same_pgm_with_different_geometry_has_different_identity(self):
        first = self._write("first.yaml", self.base)
        changed = dict(self.base)
        changed["resolution"] = 0.051
        second = self._write("second.yaml", changed)
        one = JETSON_IDENTITY.load_map_identity(first)
        two = JETSON_IDENTITY.load_map_identity(second)
        self.assertEqual(one["map_image_sha256"], two["map_image_sha256"])
        self.assertNotEqual(one["map_manifest_sha256"],
                            two["map_manifest_sha256"])
        artifact = {
            "map_image_sha256": one["map_image_sha256"],
            "map_manifest_sha256": one["map_manifest_sha256"],
        }
        with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
            JETSON_IDENTITY.verify_identity_fields(
                artifact, two, "same-PGM artifact")

    def test_each_map_server_geometry_field_is_bound(self):
        base_path = self._write("base.yaml", self.base)
        expected = JETSON_IDENTITY.load_map_identity(base_path)
        variants = [
            ("origin", [-1.25, 2.51, 0.0]),
            ("origin", [-1.25, 2.5, 0.01]),
            ("negate", 1),
            ("free_thresh", 0.20),
            ("occupied_thresh", 0.70),
        ]
        for index, (field, value) in enumerate(variants):
            document = dict(self.base)
            document[field] = value
            actual = JETSON_IDENTITY.load_map_identity(
                self._write("variant-%d.yaml" % index, document))
            self.assertNotEqual(expected["map_manifest_sha256"],
                                actual["map_manifest_sha256"])

    def test_nonzero_yaml_origin_yaw_is_rejected(self):
        document = dict(self.base)
        document["origin"] = [-1.25, 2.5, 0.001]
        with self.assertRaisesRegex(ValueError, "yaw must be zero"):
            JETSON_IDENTITY.load_map_identity(
                self._write("rotated.yaml", document),
                require_zero_yaw=True)

    def test_nonzero_occupancy_grid_yaw_is_rejected(self):
        yaw = 0.2
        orientation = types.SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0))
        with self.assertRaisesRegex(ValueError, "yaw must be zero"):
            JETSON_IDENTITY.require_zero_grid_yaw(orientation)

    def test_invalid_thresholds_fail_closed(self):
        document = dict(self.base)
        document["free_thresh"] = 0.8
        with self.assertRaisesRegex(ValueError, "thresholds"):
            JETSON_IDENTITY.load_map_identity(
                self._write("invalid.yaml", document))


if __name__ == "__main__":
    unittest.main()
