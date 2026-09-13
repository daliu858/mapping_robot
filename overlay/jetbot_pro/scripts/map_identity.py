#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canonical identity for a ROS ``map_server`` YAML + image pair.

The image digest alone is not a sufficient map identity: changing resolution,
origin or occupancy thresholds changes the metric world represented by the
same bytes.  This module is deliberately ROS-free and Python 2.7/3 compatible
so the Jetson recorder and PC offline tools hash exactly the same manifest.
"""
from __future__ import print_function

import hashlib
import json
import math
import os

import yaml


MANIFEST_SCHEMA = "slam-car-map-manifest-v1"
HEX_CHARS = set("0123456789abcdef")
YAW_EPSILON = 1e-12


def _finite(value):
    return not math.isnan(value) and not math.isinf(value)


def _canonical_number(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("map %s is not numeric" % field)
    if not _finite(number):
        raise ValueError("map %s is not finite" % field)
    # 17 significant digits round-trip an IEEE-754 double on Python 2 and 3.
    text = format(number, ".17g")
    if text in ("-0", "-0.0"):
        text = "0"
    return text


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value):
    value = str(value or "").strip().lower()
    return len(value) == 64 and all(char in HEX_CHARS for char in value)


def require_zero_origin_yaw(yaw, context="map"):
    try:
        value = float(yaw)
    except (TypeError, ValueError):
        raise ValueError("%s origin yaw is invalid" % context)
    if not _finite(value):
        raise ValueError("%s origin yaw is invalid" % context)
    if abs(value) > YAW_EPSILON:
        raise ValueError(
            "%s origin yaw must be zero; rotated occupancy maps are not "
            "supported" % context)
    return value


def quaternion_yaw(x, y, z, w):
    values = [float(value) for value in (x, y, z, w)]
    if not all(_finite(value) for value in values):
        raise ValueError("occupancy map origin quaternion is invalid")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("occupancy map origin quaternion has zero norm")
    x, y, z, w = [value / norm for value in values]
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def require_zero_grid_yaw(orientation, context="occupancy map"):
    values = [float(value) for value in (
        orientation.x, orientation.y, orientation.z, orientation.w)]
    if not all(_finite(value) for value in values):
        raise ValueError("%s origin quaternion is invalid" % context)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("%s origin quaternion has zero norm" % context)
    normalized = [value / norm for value in values]
    if abs(normalized[0]) > YAW_EPSILON or abs(normalized[1]) > YAW_EPSILON:
        raise ValueError(
            "%s origin must not contain roll or pitch" % context)
    yaw = quaternion_yaw(*normalized)
    return require_zero_origin_yaw(yaw, context)


def _resolve_image(map_file, image_value, allow_portable_sibling=False):
    map_file = os.path.abspath(os.path.expanduser(str(map_file)))
    image_text = str(image_value)
    image = os.path.expanduser(image_text)
    source_absolute = os.path.isabs(image) or image_text.startswith("/")
    if not source_absolute:
        image = os.path.join(os.path.dirname(map_file), image)
    image = os.path.abspath(image)
    if (not os.path.isfile(image) and allow_portable_sibling and
            source_absolute):
        sibling = os.path.join(os.path.dirname(map_file),
                               os.path.basename(image))
        if os.path.isfile(sibling):
            image = sibling
    if not os.path.isfile(image):
        raise ValueError("map image does not exist: %s" % image)
    return image


def load_map_identity(map_file, allow_portable_sibling=False,
                      require_zero_yaw=False):
    """Load, validate and hash a canonical ROS map manifest.

    Canonical numeric values are strings by design.  This avoids Python/YAML
    emitter differences while retaining every binary64-relevant digit.
    """
    map_file = os.path.abspath(os.path.expanduser(str(map_file)))
    if not os.path.isfile(map_file):
        raise ValueError("map YAML does not exist: %s" % map_file)
    with open(map_file, "r") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError("map YAML root must be an object")
    image_value = document.get("image")
    if not image_value:
        raise ValueError("map YAML has no image field: %s" % map_file)
    image_path = _resolve_image(
        map_file, image_value, allow_portable_sibling=allow_portable_sibling)
    resolution = float(_canonical_number(
        document.get("resolution"), "resolution"))
    if resolution <= 0.0:
        raise ValueError("map resolution must be positive")
    origin = document.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        raise ValueError("map origin must contain x, y and yaw")
    origin_text = [_canonical_number(origin[index], "origin[%d]" % index)
                   for index in range(3)]
    if require_zero_yaw:
        require_zero_origin_yaw(float(origin_text[2]), "map YAML")
    try:
        negate_value = float(document.get("negate"))
    except (TypeError, ValueError):
        raise ValueError("map negate must be 0 or 1")
    if not _finite(negate_value) or negate_value not in (0.0, 1.0):
        raise ValueError("map negate must be 0 or 1")
    negate = int(negate_value)
    free_text = _canonical_number(document.get("free_thresh"), "free_thresh")
    occupied_text = _canonical_number(
        document.get("occupied_thresh"), "occupied_thresh")
    free_value, occupied_value = float(free_text), float(occupied_text)
    if not (0.0 <= free_value < occupied_value <= 1.0):
        raise ValueError(
            "map thresholds must satisfy 0 <= free < occupied <= 1")
    image_sha = _sha256_file(image_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "map_image_sha256": image_sha,
        "resolution": _canonical_number(resolution, "resolution"),
        "origin": origin_text,
        "negate": negate,
        "free_thresh": free_text,
        "occupied_thresh": occupied_text,
    }
    serialized = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if not isinstance(serialized, bytes):
        serialized = serialized.encode("ascii")
    manifest_sha = hashlib.sha256(serialized).hexdigest()
    return {
        "map_file": map_file,
        "map_image_path": image_path,
        "map_image_sha256": image_sha,
        "map_manifest_sha256": manifest_sha,
        "manifest": manifest,
    }


def verify_identity_fields(document, identity, context="artifact"):
    image_sha = str(document.get("map_image_sha256", "")).strip().lower()
    manifest_sha = str(document.get("map_manifest_sha256", "")).strip().lower()
    if not is_sha256(image_sha) or image_sha != identity["map_image_sha256"]:
        raise ValueError("%s map image identity mismatch" % context)
    if (not is_sha256(manifest_sha) or
            manifest_sha != identity["map_manifest_sha256"]):
        raise ValueError("%s canonical map manifest identity mismatch" % context)
    return True
