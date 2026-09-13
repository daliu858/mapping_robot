# -*- coding: utf-8 -*-
"""Fail-closed contracts for PC-side semantic-map artifacts.

These helpers deliberately keep provenance fields at the top level of every
artifact.  A topology is valid only for the exact ``semantic_objects`` bytes
that were supplied while segmenting the map, and a final semantic map is valid
only for the exact objects/topology pair that was merged.
"""
from __future__ import print_function

import hashlib
import json
import math
import os
import tempfile

import yaml


OBJECTS_SCHEMA = "slam-car-semantic-objects-v1"
TOPOLOGY_SCHEMA = "slam-car-room-topology-v1"
SEMANTIC_MAP_SCHEMA = "slam-car-semantic-map-v1"
POI_CONTRACT_SCHEMA = "slam-car-core-poi-contract-v1"
CORE_POI_LABELS = (
    "chair",
    "table",
    "door",
)
DOOR_TRIAL_POI_LABELS = (
    "door",
)
ALLOWED_CORE_POI_TABLES = (
    CORE_POI_LABELS,
    DOOR_TRIAL_POI_LABELS,
)
HEX_CHARS = set("0123456789abcdef")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value, field):
    value = str(value or "").strip().lower()
    if len(value) != 64 or any(char not in HEX_CHARS for char in value):
        raise ValueError("%s must be a SHA-256 digest" % field)
    return value


def require_recording_token(value):
    token = str(value or "").strip()
    if not token or len(token) > 256 or any(ord(char) < 33 for char in token):
        raise ValueError("recording_token is missing or invalid")
    return token


def _finite(value):
    return not math.isnan(value) and not math.isinf(value)


def normalize_camera_mount(value):
    """Return ``[x, y, z, yaw, pitch, roll]`` as finite floats.

    Launch files naturally express the transform as one whitespace-separated
    string, while YAML producers may use a six-element sequence.  Both are
    accepted, but a missing transform is never inferred or defaulted.
    """
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(
            "camera_mount must contain x y z yaw pitch roll")
    if len(values) != 6:
        raise ValueError(
            "camera_mount must contain exactly x y z yaw pitch roll")
    try:
        normalized = [float(item) for item in values]
    except (TypeError, ValueError):
        raise ValueError("camera_mount contains a non-numeric value")
    if not all(_finite(item) for item in normalized):
        raise ValueError("camera_mount contains a non-finite value")
    return normalized


def camera_mount_equal(left, right, tolerance=1e-12):
    left_values = normalize_camera_mount(left)
    right_values = normalize_camera_mount(right)
    return all(abs(a - b) <= tolerance
               for a, b in zip(left_values, right_values))


def require_complete_document(document, schema, context):
    if not isinstance(document, dict):
        raise ValueError("%s root must be a mapping" % context)
    if document.get("schema") != schema:
        raise ValueError("%s schema is missing or unsupported" % context)
    if document.get("completed") is not True:
        raise ValueError("%s is not a completed artifact" % context)


def validate_objects_document(document):
    require_complete_document(document, OBJECTS_SCHEMA, "objects")
    if document.get("frame") != "map":
        raise ValueError("objects frame must be map")
    provenance = {
        "recording_token": require_recording_token(
            document.get("recording_token")),
        "source_bag_sha256": require_sha256(
            document.get("source_bag_sha256"),
            "objects.source_bag_sha256"),
        "map_image_sha256": require_sha256(
            document.get("map_image_sha256"),
            "objects.map_image_sha256"),
        "map_manifest_sha256": require_sha256(
            document.get("map_manifest_sha256"),
            "objects.map_manifest_sha256"),
        "camera_mount": normalize_camera_mount(
            document.get("camera_mount")),
    }
    if not isinstance(document.get("objects"), list):
        raise ValueError("objects.objects must be a list")
    return provenance


def load_bound_objects(objects_path, map_identity):
    """Load a completed objects artifact and bind it to one canonical map."""
    path = os.path.abspath(os.path.expanduser(str(objects_path)))
    if not os.path.isfile(path):
        raise ValueError("objects artifact does not exist: %s" % path)
    with open(path, "r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    provenance = validate_objects_document(document)
    if provenance["map_image_sha256"] != map_identity["map_image_sha256"]:
        raise ValueError("objects and occupancy-map image identities differ")
    if (provenance["map_manifest_sha256"] !=
            map_identity["map_manifest_sha256"]):
        raise ValueError(
            "objects and canonical occupancy-map manifests differ")
    return document, provenance, file_sha256(path)


def validate_topology_document(document):
    require_complete_document(document, TOPOLOGY_SCHEMA, "topology")
    if document.get("frame") != "map":
        raise ValueError("topology frame must be map")
    provenance = {
        "recording_token": require_recording_token(
            document.get("recording_token")),
        "source_bag_sha256": require_sha256(
            document.get("source_bag_sha256"),
            "topology.source_bag_sha256"),
        "map_image_sha256": require_sha256(
            document.get("map_image_sha256"),
            "topology.map_image_sha256"),
        "map_manifest_sha256": require_sha256(
            document.get("map_manifest_sha256"),
            "topology.map_manifest_sha256"),
        "camera_mount": normalize_camera_mount(
            document.get("camera_mount")),
        "source_objects_sha256": require_sha256(
            document.get("source_objects_sha256"),
            "topology.source_objects_sha256"),
    }
    if not isinstance(document.get("rooms"), list) or not document["rooms"]:
        raise ValueError("topology.rooms must be a non-empty list")
    for field in ("doors", "topology"):
        if field not in document or not isinstance(document[field], list):
            raise ValueError("topology.%s must be present as a list" % field)
    origin = document.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        raise ValueError("topology.origin must contain x, y and yaw")
    try:
        origin_values = [float(origin[index]) for index in range(3)]
    except (TypeError, ValueError):
        raise ValueError("topology.origin is invalid")
    if not all(_finite(item) for item in origin_values):
        raise ValueError("topology.origin is invalid")
    if abs(origin_values[2]) > 1e-12:
        raise ValueError("topology.origin yaw must be zero")
    try:
        resolution = float(document.get("resolution"))
    except (TypeError, ValueError):
        raise ValueError("topology.resolution is invalid")
    if not _finite(resolution) or resolution <= 0.0:
        raise ValueError("topology.resolution is invalid")
    return provenance


def build_poi_contract(poi_points, required_labels=None):
    labels = sorted(set(str(item.get("label", "")).strip()
                        for item in poi_points if isinstance(item, dict)))
    if required_labels is None:
        required = list(CORE_POI_LABELS)
    else:
        required = [str(item).strip() for item in required_labels
                    if str(item).strip()]
    if tuple(required) not in ALLOWED_CORE_POI_TABLES:
        raise ValueError("unsupported required POI table: %s" %
                         ",".join(required))
    missing = sorted(set(required) - set(labels))
    if missing:
        raise ValueError("missing required POIs: %s" % ", ".join(missing))
    return {
        "schema": POI_CONTRACT_SCHEMA,
        "complete": True,
        "required_labels": required,
        "observed_labels": labels,
    }


def semantic_manifest_payload(document):
    """Build the canonical, self-checking provenance manifest."""
    poi_contract = document.get("poi_contract")
    if not isinstance(poi_contract, dict):
        raise ValueError("semantic map poi_contract is missing")
    camera_mount = normalize_camera_mount(document.get("camera_mount"))
    # Decimal strings avoid Python 2/3 JSON float-rendering differences on the
    # Jetson validator while preserving the complete binary64 value.
    camera_mount_text = [format(value, ".17g") for value in camera_mount]
    return {
        "schema": document.get("schema"),
        "completed": document.get("completed"),
        "recording_token": require_recording_token(
            document.get("recording_token")),
        "source_bag_sha256": require_sha256(
            document.get("source_bag_sha256"),
            "semantic_map.source_bag_sha256"),
        "map_image_sha256": require_sha256(
            document.get("map_image_sha256"),
            "semantic_map.map_image_sha256"),
        "map_manifest_sha256": require_sha256(
            document.get("map_manifest_sha256"),
            "semantic_map.map_manifest_sha256"),
        "camera_mount": camera_mount_text,
        "objects_artifact_sha256": require_sha256(
            document.get("objects_artifact_sha256"),
            "semantic_map.objects_artifact_sha256"),
        "topology_artifact_sha256": require_sha256(
            document.get("topology_artifact_sha256"),
            "semantic_map.topology_artifact_sha256"),
        "semantic_content_sha256": require_sha256(
            document.get("semantic_content_sha256"),
            "semantic_map.semantic_content_sha256"),
        "poi_contract": poi_contract,
    }


def semantic_manifest_sha256(document):
    serialized = json.dumps(
        semantic_manifest_payload(document), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _canonical_content(value):
    """Canonicalize YAML values identically on Python 2.7 and Python 3."""
    if isinstance(value, dict):
        return {str(key): _canonical_content(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_content(item) for item in value]
    if isinstance(value, float):
        if not _finite(value):
            raise ValueError("semantic map contains a non-finite float")
        return {"$float64": format(value, ".17g")}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise ValueError("semantic map contains an unsupported value type")


def semantic_content_sha256(document):
    content = {
        key: value for key, value in document.items()
        if key not in ("semantic_content_sha256",
                       "semantic_manifest_sha256")
    }
    serialized = json.dumps(
        _canonical_content(content), sort_keys=True,
        separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def validate_semantic_map_document(document):
    """Validate the final map as a viewable, self-checking artifact.

    This contract deliberately belongs to the artifact layer rather than to a
    robot navigator.  The final semantic map is handed to people, so it still
    needs strict provenance and POI checks even though the project no longer
    drives the robot to semantic targets.
    """
    require_complete_document(document, SEMANTIC_MAP_SCHEMA, "semantic map")
    if str(document.get("frame", "")).lstrip("/") != "map":
        raise ValueError("semantic map frame must be map")

    origin = document.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        raise ValueError("semantic map origin must contain x, y and yaw")
    try:
        origin_values = [float(origin[index]) for index in range(3)]
    except (TypeError, ValueError):
        raise ValueError("semantic map origin is invalid")
    if (not all(_finite(item) for item in origin_values) or
            abs(origin_values[2]) > 1e-12):
        raise ValueError("semantic map origin is invalid")

    for field in ("rooms", "doors", "topology", "poi"):
        if field not in document or not isinstance(document[field], list):
            raise ValueError(
                "semantic map %s must be present as a list" % field)
    if not document["rooms"]:
        raise ValueError("semantic map has no rooms")

    contract = document.get("poi_contract")
    required = tuple(contract.get("required_labels") or ()) if isinstance(
        contract, dict) else ()
    if (not isinstance(contract, dict) or
            contract.get("schema") != POI_CONTRACT_SCHEMA or
            contract.get("complete") is not True or
            required not in ALLOWED_CORE_POI_TABLES):
        raise ValueError("semantic map POI contract is incomplete")

    actual_labels = set()
    for index, poi in enumerate(document["poi"]):
        if not isinstance(poi, dict):
            raise ValueError("semantic map poi[%d] is invalid" % index)
        label = str(poi.get("label", "")).strip()
        try:
            x, y = float(poi["x"]), float(poi["y"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("semantic map poi[%d] is invalid" % index)
        if not label or not _finite(x) or not _finite(y):
            raise ValueError("semantic map poi[%d] is invalid" % index)
        actual_labels.add(label)
    if not set(required).issubset(actual_labels):
        raise ValueError("semantic map is missing one or more core POIs")
    if contract.get("observed_labels") != sorted(actual_labels):
        raise ValueError("semantic map POI contract does not match its POIs")

    expected_content = require_sha256(
        document.get("semantic_content_sha256"),
        "semantic_map.semantic_content_sha256")
    if expected_content != semantic_content_sha256(document):
        raise ValueError("semantic map content checksum mismatch")
    expected_manifest = require_sha256(
        document.get("semantic_manifest_sha256"),
        "semantic_map.semantic_manifest_sha256")
    if expected_manifest != semantic_manifest_sha256(document):
        raise ValueError("semantic map provenance manifest checksum mismatch")
    return True


def _yaml_native(value):
    """Convert numpy scalars/tuples without making numpy a dependency."""
    if isinstance(value, dict):
        return {key: _yaml_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_native(item) for item in value]
    if (value.__class__.__module__.startswith("numpy") and
            callable(getattr(value, "item", None))):
        return value.item()
    return value


def atomic_yaml_dump(document, target_path):
    """Write YAML through a same-directory, fsynced temporary file."""
    target = os.path.abspath(os.path.expanduser(str(target_path)))
    directory = os.path.dirname(target) or "."
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                prefix=".%s." % os.path.basename(target), suffix=".tmp",
                dir=directory, delete=False) as stream:
            temporary = stream.name
            yaml.safe_dump(
                _yaml_native(document), stream,
                allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        # Persist the directory entry where the platform permits directory
        # fsync.  Windows rejects opening a directory as a normal file.
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return target
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
