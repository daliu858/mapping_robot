# -*- coding: utf-8 -*-
"""Merge one completed POI artifact and its bound room topology.

The output is an atomic, self-checking semantic-map manifest.  Cross-run,
stale, truncated, and partially populated inputs are rejected before the old
output (if any) is replaced.
"""
from __future__ import print_function

import argparse
import copy
import hashlib
import math
import os
import sys

import yaml

try:
    from semantic_artifact_contract import (
        SEMANTIC_MAP_SCHEMA, atomic_yaml_dump, build_poi_contract,
        camera_mount_equal, normalize_camera_mount,
        semantic_content_sha256, semantic_manifest_sha256,
        validate_objects_document,
        validate_topology_document)
except ImportError:  # imported from the repository root by unit tests
    from tools.semantic_artifact_contract import (
        SEMANTIC_MAP_SCHEMA, atomic_yaml_dump, build_poi_contract,
        camera_mount_equal, normalize_camera_mount,
        semantic_content_sha256, semantic_manifest_sha256,
        validate_objects_document,
        validate_topology_document)


TYPE_DEFINING = {
    "staircase": "stairwell",
    "elevator_door": "elevator_hall",
}


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _finite(value):
    return not math.isnan(value) and not math.isinf(value)


def _load_yaml_snapshot(path, context):
    """Parse and hash one immutable in-memory read of an artifact."""
    if not os.path.isfile(path):
        raise ValueError("%s artifact does not exist: %s" % (context, path))
    try:
        with open(path, "rb") as stream:
            payload = stream.read()
        document = yaml.safe_load(payload.decode("utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("%s artifact cannot be read: %s" %
                         (context, error))
    if not isinstance(document, dict):
        raise ValueError("%s root must be a mapping" % context)
    return document, hashlib.sha256(payload).hexdigest()


def _load_yaml(path, context):
    return _load_yaml_snapshot(path, context)[0]


def point_in_poly(x, y, polygon):
    """Return whether a map point is within a room polygon."""
    count = len(polygon)
    if count < 3:
        return False
    inside = False
    previous = count - 1
    for index in range(count):
        xi, yi = polygon[index][0], polygon[index][1]
        xj, yj = polygon[previous][0], polygon[previous][1]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        previous = index
    return inside


def choose_room_semantic(votes):
    """Choose the majority room type; ties are deterministic."""
    if not votes:
        return "room"
    highest = max(votes.values())
    return min(name for name, count in votes.items() if count == highest)


def _validated_poi_points(objects):
    points = []
    for index, item in enumerate(objects.get("objects", [])):
        if not isinstance(item, dict):
            raise ValueError("objects.objects[%d] must be a mapping" % index)
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError("objects.objects[%d].label is missing" % index)
        try:
            x, y = float(item["x"]), float(item["y"])
            hits = int(item.get("hits", 0))
        except (KeyError, TypeError, ValueError):
            raise ValueError("objects.objects[%d] coordinates are invalid" %
                             index)
        if not _finite(x) or not _finite(y) or hits < 0:
            raise ValueError("objects.objects[%d] values are invalid" % index)
        points.append({"label": label, "x": x, "y": y, "hits": hits})
    return points


def _require_same_provenance(objects_provenance, topology_provenance):
    for field in ("recording_token", "source_bag_sha256", "map_image_sha256",
                  "map_manifest_sha256"):
        if objects_provenance[field] != topology_provenance[field]:
            raise ValueError("objects/topology %s mismatch" % field)
    if not camera_mount_equal(objects_provenance["camera_mount"],
                              topology_provenance["camera_mount"]):
        raise ValueError("objects/topology camera_mount mismatch")


def merge_layers(objects, topology, objects_path, topology_path):
    """Validate and merge the exact two on-disk source artifacts."""
    disk_objects, objects_artifact_hash = _load_yaml_snapshot(
        objects_path, "objects")
    disk_topology, topology_artifact_hash = _load_yaml_snapshot(
        topology_path, "topology")
    if disk_objects != objects:
        raise ValueError("objects document differs from its source artifact")
    if disk_topology != topology:
        raise ValueError("topology document differs from its source artifact")

    objects_provenance = validate_objects_document(objects)
    topology_provenance = validate_topology_document(topology)
    _require_same_provenance(objects_provenance, topology_provenance)

    if topology_provenance["source_objects_sha256"] != objects_artifact_hash:
        raise ValueError(
            "topology is not bound to the supplied objects artifact")
    poi_points = _validated_poi_points(objects)
    required = None
    provenance = objects.get("provenance")
    if isinstance(provenance, dict):
        required = provenance.get("required_labels")
    poi_contract = build_poi_contract(poi_points, required_labels=required)
    rooms = copy.deepcopy(topology["rooms"])
    for room in rooms:
        if not isinstance(room, dict):
            raise ValueError("topology room entries must be mappings")
        room["semantic"] = "room"
        room["facilities"] = []
        room["poi"] = []
        room["_typevotes"] = {}

    unassigned = []
    for poi in poi_points:
        matched = None
        for room in rooms:
            # Phantom regions remain visible as topology diagnostics but are
            # never promoted to navigable rooms.
            if str(room.get("status", "room")).lower() == "phantom":
                continue
            polygon = room.get("polygon_map")
            if polygon and point_in_poly(poi["x"], poi["y"], polygon):
                matched = room
                break
        if matched is None:
            unassigned.append(poi)
            continue
        matched["poi"].append(poi)
        if poi["label"] in TYPE_DEFINING:
            room_type = TYPE_DEFINING[poi["label"]]
            votes = matched["_typevotes"]
            votes[room_type] = votes.get(room_type, 0) + 1
        elif poi["label"] not in matched["facilities"]:
            matched["facilities"].append(poi["label"])

    for room in rooms:
        votes = room.pop("_typevotes")
        if votes:
            room["semantic"] = choose_room_semantic(votes)

    output = {
        "schema": SEMANTIC_MAP_SCHEMA,
        "completed": True,
        "frame": "map",
        "recording_token": objects_provenance["recording_token"],
        "source_bag_sha256": objects_provenance["source_bag_sha256"],
        "camera_mount": normalize_camera_mount(
            objects_provenance["camera_mount"]),
        "map_image_sha256": objects_provenance["map_image_sha256"],
        "map_manifest_sha256": objects_provenance["map_manifest_sha256"],
        "objects_artifact_sha256": objects_artifact_hash,
        "topology_artifact_sha256": topology_artifact_hash,
        "source_objects": os.path.basename(objects_path),
        "source_topology": os.path.basename(topology_path),
        "poi_contract": poi_contract,
        "resolution": topology.get("resolution"),
        "origin": copy.deepcopy(topology.get("origin")),
        "poi": poi_points,
        "rooms": rooms,
        "doors": copy.deepcopy(topology["doors"]),
        "topology": copy.deepcopy(topology["topology"]),
        "unassigned_poi": unassigned,
    }
    output["semantic_content_sha256"] = semantic_content_sha256(output)
    output["semantic_manifest_sha256"] = semantic_manifest_sha256(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", required=True,
                        help="completed semantic_objects.yaml")
    parser.add_argument("--topology", required=True,
                        help="completed *_topology.yaml bound to --objects")
    parser.add_argument("--out", default="semantic_map.yaml")
    args = parser.parse_args()

    try:
        objects = _load_yaml(args.objects, "objects")
        topology = _load_yaml(args.topology, "topology")
        output = merge_layers(
            objects, topology, args.objects, args.topology)
        atomic_yaml_dump(output, args.out)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    print("=== semantic map merge complete ===")
    for room in output["rooms"]:
        facilities = ",".join(room["facilities"]) or "-"
        suffix = " (phantom)" if room.get("status") == "phantom" else ""
        print("  R%-3s %-14s area=%6.2f m2 facilities=[%s] POI=%d%s" %
              (room.get("id"), room["semantic"],
               room.get("area_m2", 0.0), facilities,
               len(room["poi"]), suffix))
    if output["unassigned_poi"]:
        print("  unassigned top-level POIs: %d" %
              len(output["unassigned_poi"]))
    print("wrote:", args.out)


if __name__ == "__main__":
    configure_utf8_stdio()
    main()
