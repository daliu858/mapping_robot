#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure fail-closed contracts shared by the formal offline replay nodes."""
from __future__ import print_function

import hashlib
import json
import math

try:
    STRING_TYPES = (basestring,)
except NameError:
    STRING_TYPES = (str, bytes)
try:
    INTEGER_TYPES = (int, long)
except NameError:
    INTEGER_TYPES = (int,)


FORMAL_STEREO_SOURCES = {
    "img_topic": "/stereo/left/image_raw/compressed",
    "left_image_base": "/stereo/left/image_raw",
    "right_image_base": "/stereo/right/image_raw",
    "camera_info_topic": "/stereo/left/camera_info",
    "disparity_topic": "/stereo/disparity",
    "optical_frame": "stereo_left_optical",
}

FORMAL_POI_CLASSES = (
    "chair",
    "table",
    "door",
)
# Household demo still uses the three-class table.  This AMCL second pass
# may publish door-only detections so the overlay can be judged on doors.
DOOR_TRIAL_POI_CLASSES = (
    "door",
)
# hall_v1 (2026-08-30): corridor+halls survey with labeled furniture POIs.
# door remains the only required/structure label; sofa/cabinet/spray are
# optional semantic POIs confirmed by the same hits/spread/anchor gates.
HALL_V1_POI_CLASSES = (
    "door",
    "sofa",
    "cabinet",
    "spray",
)
ALLOWED_POI_CLASS_TABLES = (
    FORMAL_POI_CLASSES,
    DOOR_TRIAL_POI_CLASSES,
    HALL_V1_POI_CLASSES,
)

# One-pass gmapping capture and two-pass AMCL survey both produce formal bags.
FORMAL_POSE_SOURCES = (
    "gmapping",
    "amcl",
)

# Offline map<-camera TF-miss classes.  Bag 19 postmortem: one '"map" ...
# does not exist' at bag open, then many few-ms 'extrapolation into the
# future' misses, each of which the old contract turned into a whole-bag
# INVALID -- 90 minutes of replay and no yaml.  None of these classes may
# invalidate the transaction; they cost at most the one frame involved.
TF_MISS_STARTUP = "startup"
TF_MISS_FUTURE = "future"
TF_MISS_BROKEN = "broken"


def classify_stereo_tf_miss(message, map_seen):
    """Classify one failed map<-camera lookup during offline replay.

    ``map_seen`` is whether this replay has already resolved the map chain
    once: the same error text is a normal warm-up condition before that
    proof and a real chain break after it.
    """
    text = str(message)
    if "does not exist" in text:
        # The frame is not registered in the buffer yet.  tf2 never
        # unregisters frames, so after one success this means the listener
        # or the chain is genuinely gone.
        return TF_MISS_BROKEN if map_seen else TF_MISS_STARTUP
    if "extrapolation into the future" in text:
        # Detection stamp slightly newer than the buffered chain.  The
        # caller may substitute the newest transform when the slip is
        # bounded (latest_tf_fallback_ok); otherwise the chain is stale.
        return TF_MISS_FUTURE
    if "extrapolation" in text:
        # Into the past / single-sample: expected while the replayed chain
        # is still filling, a real gap once the chain had been proven.
        return TF_MISS_BROKEN if map_seen else TF_MISS_STARTUP
    return TF_MISS_BROKEN


def latest_tf_fallback_ok(requested_sec, latest_sec, tolerance_s):
    """Accept the newest buffered transform only for bounded replay slip.

    The motion gate already limits the robot to slow twists, so a
    transform a few milliseconds older than the detection stamp moves the
    projected point by millimetres.  A stale chain (seconds old) must not
    be silently substituted -- the caller drops that frame instead.
    """
    if not (_finite(requested_sec) and _finite(latest_sec) and
            _finite(tolerance_s)):
        return False
    tolerance = float(tolerance_s)
    if tolerance < 0.0:
        return False
    return abs(float(requested_sec) - float(latest_sec)) <= tolerance


def replay_abort_reason(status_success, status_message):
    """Return why rosbag play must stop now, or None to keep playing.

    Only a mapper-locked INVALID aborts mid-replay: it is permanent, so
    every further bag second is wasted.  NOT_READY, probe failures and
    healthy states keep the replay running; the finalize barrier still
    fail-closes at the end.
    """
    message = str(status_message or "")
    if not status_success and message.startswith("INVALID:"):
        return message
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


def _strict_int(value, field):
    if isinstance(value, bool) or not isinstance(value, INTEGER_TYPES):
        raise ValueError("%s must be a JSON integer" % field)
    return int(value)


def canonical_camera_mount(value):
    """Return the static-transform xyz/yaw-pitch-roll as six finite floats."""
    if isinstance(value, STRING_TYPES):
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        parts = text.replace(",", " ").split()
    else:
        try:
            parts = list(value)
        except TypeError:
            raise ValueError("camera_mount must contain six numeric values")
    if len(parts) != 6 or not all(_finite(item) for item in parts):
        raise ValueError("camera_mount must contain six finite numeric values")
    return [float(item) for item in parts]


def validate_recording_states(payloads, expected_map_hash=None,
                              expected_manifest_hash=None):
    """Validate the canonical checked-recorder STARTED -> COMPLETED chain."""
    records = []
    for payload in payloads:
        try:
            record = json.loads(str(payload))
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict) and record.get("schema") == 1:
            records.append(record)
    failures = [item for item in records if item.get("state") == "failed"]
    if failures:
        return False, "recorder reported failure: %s" % \
            failures[-1].get("reason", "unspecified")
    unknown = sorted(set(
        str(item.get("state", "")) for item in records
        if item.get("state") not in ("started", "completed", "failed")))
    if unknown:
        return False, "recording markers contain unknown states: %s" % \
            ",".join(unknown)
    tokens = set(item.get("token") for item in records if item.get("token"))
    if len(tokens) != 1:
        return False, "recording markers contain multiple/missing sessions"
    completed = [(index, item) for index, item in enumerate(records)
                 if item.get("state") == "completed" and item.get("token")]
    if len(completed) != 1:
        return False, "expected exactly one recording completion marker"
    completion_index, completion = completed[0]
    matching_starts = [
        (index, item) for index, item in enumerate(records)
        if item.get("state") == "started" and
        item.get("token") == completion["token"]]
    # rosbag may record both the original and latched STARTED publication.
    if (not matching_starts or
            any(index >= completion_index for index, _item in matching_starts)):
        return False, "recording markers are out of order"
    map_hash = str(completion.get("map_image_sha256", "")).strip().lower()
    if (len(map_hash) != 64 or
            any(char not in "0123456789abcdef" for char in map_hash)):
        return False, "completion marker has no valid map SHA-256"
    start_hashes = set(
        str(item.get("map_image_sha256", "")).strip().lower()
        for _index, item in matching_starts)
    if start_hashes != set([map_hash]):
        return False, "start/completion map SHA-256 mismatch"
    if (expected_map_hash is not None and
            map_hash != str(expected_map_hash).strip().lower()):
        return False, "completion marker does not match expected map SHA-256"
    manifest_hash = str(
        completion.get("map_manifest_sha256", "")).strip().lower()
    if (len(manifest_hash) != 64 or
            any(char not in "0123456789abcdef" for char in manifest_hash)):
        return False, "completion marker has no valid map manifest SHA-256"
    start_manifests = set(
        str(item.get("map_manifest_sha256", "")).strip().lower()
        for _index, item in matching_starts)
    if start_manifests != set([manifest_hash]):
        return False, "start/completion map manifest SHA-256 mismatch"
    if (expected_manifest_hash is not None and
            manifest_hash != str(expected_manifest_hash).strip().lower()):
        return False, (
            "completion marker does not match expected map manifest SHA-256")
    try:
        camera_mount = canonical_camera_mount(completion.get("camera_mount"))
        start_mounts = set(
            tuple(canonical_camera_mount(item.get("camera_mount")))
            for _index, item in matching_starts)
    except (TypeError, ValueError) as error:
        return False, "recording marker has invalid camera_mount: %s" % error
    if start_mounts != set([tuple(camera_mount)]):
        return False, "start/completion camera_mount mismatch"
    pose_source = str(completion.get("pose_source", "")).strip().lower()
    if pose_source not in FORMAL_POSE_SOURCES:
        return False, (
            "formal recording marker has unknown pose_source")
    start_sources = set(
        str(item.get("pose_source", "")).strip().lower()
        for _index, item in matching_starts)
    if start_sources != set([pose_source]):
        return False, "start/completion pose_source mismatch"
    completion["camera_mount"] = camera_mount
    completion["pose_source"] = pose_source
    return True, completion


def camera_mount_quaternion(value):
    """Return xyzw for an xyz/yaw-pitch-roll static-transform argument."""
    mount = canonical_camera_mount(value)
    yaw, pitch, roll = mount[3], mount[4], mount[5]
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def camera_transform_matches(value, parent_frame, child_frame,
                             translation, quaternion, tolerance=1e-5,
                             expected_child="stereo_left_optical"):
    """Validate the actual TF edge against the sealed camera mount."""
    if str(parent_frame).lstrip("/") != "base_footprint":
        return False
    if str(child_frame).lstrip("/") != str(expected_child).lstrip("/"):
        return False
    mount = canonical_camera_mount(value)
    try:
        observed_translation = [float(item) for item in translation]
        observed_quaternion = [float(item) for item in quaternion]
    except (TypeError, ValueError):
        return False
    if (len(observed_translation) != 3 or len(observed_quaternion) != 4 or
            not all(_finite(item) for item in
                    observed_translation + observed_quaternion)):
        return False
    if any(abs(observed_translation[index] - mount[index]) > tolerance
           for index in range(3)):
        return False
    expected = camera_mount_quaternion(mount)
    observed_norm = math.sqrt(sum(item * item
                                  for item in observed_quaternion))
    if observed_norm <= 0.0:
        return False
    observed = [item / observed_norm for item in observed_quaternion]
    dot = abs(sum(left * right for left, right in zip(expected, observed)))
    return abs(1.0 - dot) <= tolerance


def validate_stereo_source_contract(range_source, actual):
    """Reject every configurable source combination outside formal left stereo."""
    if str(range_source).strip().lower() != "stereo":
        raise ValueError(
            "formal offline replay requires stereo disparity; legacy lidar "
            "bearing fusion is not a transactional release path")
    mismatches = []
    for name, expected in FORMAL_STEREO_SOURCES.items():
        observed = str(actual.get(name, ""))
        if observed != expected:
            mismatches.append("%s=%r (expected %r)" %
                              (name, observed, expected))
    if bool(actual.get("detections_rectified", False)):
        mismatches.append("detections_rectified must be false for raw-left input")
    if bool(actual.get("stereo_lidar_fallback", False)):
        mismatches.append("stereo_lidar_fallback must be false in formal stereo replay")
    if mismatches:
        raise ValueError("formal stereo source contract mismatch: %s" %
                         "; ".join(mismatches))
    return True


def validate_detection_frames(frames, classes=None):
    """Return stamp->(width,height), rejecting ambiguity and geometry drift."""
    if not isinstance(frames, list) or not frames:
        raise ValueError("detections JSON contains no frames")
    result = {}
    common_size = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("detection frame %d is not an object" % index)
        try:
            stamp = _strict_int(frame.get("stamp_ns"), "stamp_ns")
            width = _strict_int(frame.get("w"), "w")
            height = _strict_int(frame.get("h"), "h")
        except (TypeError, ValueError):
            raise ValueError("detection frame %d has invalid metadata" % index)
        if stamp <= 0:
            raise ValueError("detection frame %d has a nonpositive stamp" % index)
        if stamp in result:
            raise ValueError("detections JSON contains duplicate stamp %d" % stamp)
        if width <= 0 or height <= 0:
            raise ValueError("detection frame %d has invalid dimensions" % index)
        size = (width, height)
        if common_size is None:
            common_size = size
        elif size != common_size:
            raise ValueError("detection frame dimensions changed within one bag")
        detections = frame.get("dets")
        if not isinstance(detections, list):
            raise ValueError("detection frame %d has no dets list" % index)
        for det_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise ValueError(
                    "detection frame %d item %d is not an object" %
                    (index, det_index))
            try:
                class_id = _strict_int(detection.get("cls"), "cls")
                score = float(detection.get("score"))
                cx = float(detection.get("cx"))
                cy = float(detection.get("cy"))
                sx = float(detection.get("sx"))
                sy = float(detection.get("sy"))
            except (TypeError, ValueError):
                raise ValueError(
                    "detection frame %d item %d has invalid fields" %
                    (index, det_index))
            values = (score, cx, cy, sx, sy)
            if not all(_finite(value) for value in values):
                raise ValueError(
                    "detection frame %d item %d contains nonfinite values" %
                    (index, det_index))
            if not 0.0 <= score <= 1.0 or sx <= 0.0 or sy <= 0.0:
                raise ValueError(
                    "detection frame %d item %d has invalid score/size" %
                    (index, det_index))
            epsilon = 1e-6
            if (cx - sx / 2.0 < -epsilon or
                    cy - sy / 2.0 < -epsilon or
                    cx + sx / 2.0 > width + epsilon or
                    cy + sy / 2.0 > height + epsilon):
                raise ValueError(
                    "detection frame %d item %d lies outside the image" %
                    (index, det_index))
            if classes is not None:
                if class_id < 0 or class_id >= len(classes):
                    raise ValueError(
                        "detection frame %d item %d has invalid class id" %
                        (index, det_index))
                if str(detection.get("label", "")) != str(classes[class_id]):
                    raise ValueError(
                        "detection frame %d item %d class/label mismatch" %
                        (index, det_index))
        result[stamp] = size
    return result


def load_detection_contract(path, require_complete=True):
    with open(path, "r") as stream:
        data = json.load(stream)
    if data.get("schema") != "slam-car-detections-v1":
        raise ValueError("unsupported detections JSON schema")
    if require_complete and data.get("complete") is not True:
        raise ValueError("formal replay requires a complete detections JSON")
    raw_token = data.get("recording_token")
    if not isinstance(raw_token, STRING_TYPES):
        raise ValueError("detections JSON has no recording token")
    token = (raw_token.decode("ascii") if isinstance(raw_token, bytes)
             else raw_token).strip().lower()
    if (len(token) != 32 or
            any(char not in "0123456789abcdef" for char in token)):
        raise ValueError("detections JSON has an invalid recording token")
    data["recording_token"] = token
    for field in ("map_image_sha256", "map_manifest_sha256",
                  "source_bag_sha256"):
        value = str(data.get(field, "")).lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("detections JSON has invalid %s" % field)
        data[field] = value
    classes = data.get("classes")
    if (not isinstance(classes, list) or
            tuple(classes) not in ALLOWED_POI_CLASS_TABLES):
        raise ValueError("detections JSON has an invalid formal POI class table")
    data["camera_mount"] = canonical_camera_mount(data.get("camera_mount"))
    data["frame_sizes"] = validate_detection_frames(
        data.get("frames"), classes)
    return data


def pipeline_ready(expected_frames, stats, last_activity_wall,
                   now_wall, quiet_s, expected_stamps=None,
                   det_stamps=None, sync_stamps=None,
                   pipeline_error="", max_unsynced_frames=0):
    """Prove that the indexed detector frames reached stereo fusion.

    The detection side stays exact: every indexed stamp must have been
    published (the EOS service seals that before finalize is attempted).
    The stereo-fusion side tolerates up to ``max_unsynced_frames`` stamps
    whose exact-timestamp disparity never existed -- the Nano computes
    disparity at 9-10 Hz against a 7.5 Hz replayed pair, so a transient
    hiccup can drop a pair.  Once the input is sealed, callbacks are
    drained and the quiet period has passed, such a stamp can never sync;
    waiting for it forever would time out the finalize barrier and turn a
    22-minute replay into no yaml at all.  The shortfall is never silent:
    it is visible in the ready detail and in the mapper's sync_missing
    stat, which the yaml provenance records.
    """
    if pipeline_error:
        return False, "INVALID: %s" % pipeline_error
    expected = int(expected_frames)
    det_rx = int(stats.get("det_rx", 0))
    sync_rx = int(stats.get("sync_rx", 0))
    inflight = int(stats.get("callbacks_inflight", 0))
    infrastructure_drops = int(stats.get("infrastructure_drop", 0))
    allowed_unsynced = int(max_unsynced_frames)
    if allowed_unsynced < 0 or allowed_unsynced >= max(expected, 1):
        return False, "INVALID: unsynced-frame budget is invalid"
    if infrastructure_drops < 0:
        return False, "INVALID: infrastructure-drop count is invalid"
    if infrastructure_drops:
        return False, (
            "INVALID: %d synchronized frame(s) were dropped by mapper "
            "infrastructure" % infrastructure_drops)
    if expected <= 0:
        return False, "INVALID: expected detection-frame count is invalid"
    if expected_stamps is not None:
        expected_set = set(int(value) for value in expected_stamps)
        detected_set = set(int(value) for value in (det_stamps or []))
        synced_set = set(int(value) for value in (sync_stamps or []))
        if len(expected_set) != expected:
            return False, "INVALID: expected stamp set/count mismatch"
        unexpected_detections = detected_set - expected_set
        unexpected_sync = synced_set - expected_set
        if unexpected_detections or unexpected_sync:
            return False, (
                "INVALID: unexpected detection/fusion stamps: det=%s sync=%s" %
                (sorted(unexpected_detections)[:3],
                 sorted(unexpected_sync)[:3]))
        missing_detections = expected_set - detected_set
        if missing_detections:
            return False, "waiting for %d detection stamps" % \
                len(missing_detections)
        missing_sync = expected_set - synced_set
        if len(missing_sync) > allowed_unsynced:
            return False, "waiting for %d stereo-fusion stamps" % \
                len(missing_sync)
    if det_rx > expected or sync_rx > expected:
        return False, "INVALID: pipeline received duplicate frames"
    if det_rx != expected:
        return False, "waiting for detections: %d/%d" % (det_rx, expected)
    if sync_rx < expected - allowed_unsynced:
        return False, "waiting for stereo fusion: %d/%d" % (sync_rx, expected)
    if inflight != 0:
        return False, "waiting for %d in-flight fusion callbacks" % inflight
    if last_activity_wall is None:
        return False, "pipeline has no activity timestamp"
    quiet = float(now_wall) - float(last_activity_wall)
    if quiet < float(quiet_s):
        return False, "pipeline quiet period %.3f/%.3f s" % (quiet, quiet_s)
    return True, "pipeline complete: %d/%d frames (unsynced=%d, budget=%d)" % (
        sync_rx, expected, expected - sync_rx, allowed_unsynced)
