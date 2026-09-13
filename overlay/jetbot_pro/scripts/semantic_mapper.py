#!/usr/bin/env python
# -*- coding: utf-8 -*-
# semantic_mapper.py - Level 1 semantic mapping for JetBot (ROS Melodic, py2)
#
# Primary path (range_source=stereo): synchronize detections with a rectified
# stereo disparity image, estimate robust depth inside each bbox, back-project
# the object to the left optical frame, and transform the 3-D point to map.
#
# Legacy path (range_source=lidar): keep the original monocular bearing + 2-D
# lidar sector-range method for regression tests and emergency fallback only.
#
# Robustness rules (review feedback baked in):
#   - synchronize bbox and disparity by timestamp (ApproximateTime)
#   - rectify raw-image bbox coordinates using the left stereo calibration
#   - central-ROI median depth + IQR consistency check, not a single pixel
#   - TF lookup at detection stamp; observations gated by low twist
#   - per-class association radius; position = per-axis median
#   - optional map check: drop points that land in unknown space
#   - doors must sit on occupied structure; high-spread trails are not POIs
from __future__ import print_function
from collections import deque
import hmac
import math
import os
import sys
import tempfile
import threading
import time

import numpy as np
import message_filters
import rospy
import tf2_ros
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from vision_msgs.msg import Detection2DArray, VisionInfo
from visualization_msgs.msg import Marker, MarkerArray
from jetbot_pro.srv import (FinalizeSemanticMap,
                            FinalizeSemanticMapResponse)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from map_identity import load_map_identity, require_zero_grid_yaw
from replay_contract import (TF_MISS_FUTURE, TF_MISS_STARTUP,
                             classify_stereo_tf_miss, latest_tf_fallback_ok,
                             load_detection_contract, pipeline_ready)

try:
    from stereo_msgs.msg import DisparityImage
except ImportError:
    DisparityImage = None

try:
    import cv2
except ImportError:
    cv2 = None

# fallback only - real labels come from detectnet's vision_info rosparam
COCO91 = ["BACKGROUND", "person", "bicycle", "car", "motorcycle", "airplane",
          "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
          "street sign", "stop sign", "parking meter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
          "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "plate",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
          "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
          "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
          "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
          "oven", "toaster", "sink", "refrigerator", "blender", "book",
          "clock", "vase", "scissors", "teddy bear", "hair drier",
          "toothbrush"]

BIG = ["couch", "bed", "refrigerator", "dining table", "table", "tv",
       "desk", "toilet", "door"]
SMALL = ["cup", "bottle", "book", "mouse", "remote", "cell phone", "keyboard",
         "vase", "clock", "scissors", "bowl", "wine glass"]


def select_stereo_depth(depths, min_samples, iqr_max, gap_m=0.60):
    """Pick a depth that will not jump through an open doorway.

    A bbox on an open door is often mostly hallway. The global median then
    lands in the next room. If a nearer cluster has enough samples, use it;
    if the split is bimodal but the near side is too thin, reject.
    """
    samples = np.asarray(depths, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    needed = int(min_samples)
    if samples.size < needed:
        return None
    z_lo = float(np.min(samples))
    z_hi = float(np.max(samples))
    gap = float(gap_m)
    if gap > 0.0 and (z_hi - z_lo) >= gap:
        split = 0.5 * (z_lo + z_hi)
        near = samples[samples <= split]
        far = samples[samples > split]
        if near.size and far.size:
            near_z = float(np.median(near))
            far_z = float(np.median(far))
            if (far_z - near_z) >= gap:
                if near.size >= needed:
                    return near_z
                return None
    q25, z, q75 = np.percentile(samples, [25, 50, 75])
    if float(iqr_max) > 0.0 and (q75 - q25) > float(iqr_max):
        return None
    return float(z)


class TrackedObject(object):
    __slots__ = ("label", "cid", "obs", "hits", "last")

    def __init__(self, label, cid):
        self.label, self.cid = label, cid
        self.obs, self.hits, self.last = [], 0, rospy.Time(0)

    def add(self, x, y, stamp, cap=0):
        # Mapping must keep the full trail. A sliding window hides hallway
        # smears that would otherwise fail the compactness gate.
        self.obs.append((x, y))
        if cap and len(self.obs) > cap:
            self.obs.pop(0)
        self.hits += 1
        self.last = stamp

    def pos(self):
        a = np.array(self.obs)
        return float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    def std(self):
        a = np.array(self.obs)
        if len(a) <= 1:
            return 0.0
        # Mean of per-axis std hides a 1-D hallway smear: the empty axis
        # halves the number. RMS from the median is the mapping radius.
        centered = a - np.median(a, axis=0)
        return float(np.sqrt(np.mean(np.square(centered))))


class SemanticMapper(object):
    def __init__(self):
        gp = rospy.get_param
        self.range_source = str(gp("~range_source", "stereo")).lower()
        if self.range_source not in ("stereo", "lidar"):
            raise ValueError("~range_source must be 'stereo' or 'lidar'")
        self.score_min = gp("~score_min", 0.55)
        # Mirror compensation: the recorded frames are horizontally
        # mirrored (in-camera flip). Both eyes + calibration share that
        # mirrored space, so disparity/depth stay valid, but the
        # reconstructed optical-frame X has the wrong sign. Negating X
        # after all pixel-space checks and before the map TF restores
        # true bearings. Default off: only bags recorded with the
        # mirrored pipeline should enable it.
        self.mirror_x = bool(gp("~mirror_x", False))
        self.max_bearing = math.radians(gp("~max_bearing_deg", 40.0))
        self.max_ang_vel = gp("~max_ang_vel", 0.25)
        self.max_lin_vel = gp("~max_lin_vel", 0.30)

        # Stereo depth settings.  Detections normally come from the raw left
        # image; their bbox is rectified into disparity coordinates below.
        self.detections_rectified = gp("~detections_rectified", False)
        self.depth_roi_scale = float(gp("~depth_roi_scale", 0.40))
        # _19 live replay: surviving indoor disparity concentrates on
        # high-texture edges (door frames) at the bbox border, which the
        # shrunk center ROI excludes by design.  When the center ROI cannot
        # see enough positive disparity, sample the full rectified bbox
        # instead of rejecting the frame.
        self.depth_roi_expand = bool(gp("~depth_roi_expand", True))
        self.depth_min = float(gp("~depth_min_m", 0.30))
        self.depth_max = float(gp("~depth_max_m", 10.0))
        self.depth_iqr_max = float(gp("~depth_iqr_max_m", 1.50))
        self.depth_min_samples = int(gp("~depth_min_samples", 12))
        self.depth_near_gap = float(gp("~depth_near_gap_m", 0.60))
        self.sync_queue = int(gp("~stereo_sync_queue", 30))
        # Detection and disparity should carry the same left-image stamp.  A
        # few milliseconds tolerate serialization jitter without ever pairing
        # an adjacent camera frame.
        self.stereo_sync_slop = float(gp("~stereo_sync_slop_s", 0.005))
        self.lidar_sync_slop = float(gp("~lidar_sync_slop_s", 0.05))
        self.odom_max_age = float(gp("~odom_max_age_s", 0.12))
        self.tf_lookup_timeout = float(gp("~tf_lookup_timeout_s", 0.50))
        # Bounded replay slip: a detection stamped a few ms ahead of the
        # buffered map chain may use the newest transform instead of being
        # dropped.  Anything beyond this drops that one frame only.
        self.tf_future_tolerance = float(gp("~tf_future_tolerance_s", 0.25))
        self.fallback_scan_age = float(gp("~fallback_scan_max_age_s", 0.12))
        self.require_detection_camera_stamp = bool(
            gp("~require_detection_camera_stamp", False))

        # Legacy lidar range settings.  They are inactive in stereo mode
        # unless the explicit fallback is enabled.
        self.stereo_lidar_fallback = bool(
            gp("~stereo_lidar_fallback", False))
        self.sector_pad = math.radians(gp("~sector_pad_deg", 2.0))
        self.iqr_max = gp("~range_iqr_max", 0.5)
        self.min_hits = gp("~min_hits", 5)
        self.max_spread_m = float(gp("~max_spread_m", 0.25))
        self.assoc_default = gp("~assoc_radius", 0.5)
        self.use_map_check = gp("~use_map_check", True)
        self.structure_labels = set(
            label.strip() for label in
            str(gp("~structure_labels_csv", "door")).split(",")
            if label.strip())
        # Pixel-space image of the existing depth_max gate for structure
        # labels: a full-height door inside depth_max must subtend at least
        # this many raw pixels.  _19 full-bag evidence: 76 percent of gated
        # door boxes were a persistent ~86x41 px bottom-left detector
        # sliver whose floor points polluted association and map gating;
        # real doors sat above the 100-119 px histogram valley.  0 = off.
        self.min_structure_bbox_h = float(
            gp("~min_structure_bbox_h_px", 0.0))
        if self.min_structure_bbox_h < 0.0:
            raise ValueError("~min_structure_bbox_h_px must be >= 0")
        self.occupied_radius_m = float(gp("~occupied_radius_m", 0.30))
        self.occupied_threshold = int(gp("~occupied_threshold", 50))
        self.require_map_received = bool(
            gp("~require_map_received", False))
        self.exclude = set(gp("~exclude_labels", "person").split(","))
        self.required_labels = set(
            label.strip() for label in
            str(gp("~required_labels_csv", "")).split(",")
            if label.strip())
        self.save_path = os.path.expanduser(
            gp("~save_path", "~/catkin_ws/src/jetbot_pro/maps/semantic_objects.yaml"))
        self.source_bag = str(gp("~source_bag", ""))
        self.detections_file = str(gp("~detections_file", ""))
        self.transactional_finalize_only = bool(
            gp("~transactional_finalize_only", False))
        if (self.transactional_finalize_only and
                (self.range_source != "stereo" or
                 self.stereo_lidar_fallback)):
            raise ValueError(
                "transactional offline mapping requires stereo disparity "
                "with lidar fallback disabled")
        self.pipeline_quiet_s = float(gp("~pipeline_quiet_s", 1.0))
        self.expected_detection_frames = 0
        self.recording_token = ""
        self.camera_mount = []
        self.contract_labels = None
        if self.transactional_finalize_only:
            if not self.detections_file:
                raise ValueError(
                    "~detections_file is required in transactional offline mode")
            detection_contract = load_detection_contract(self.detections_file)
            self.expected_detection_stamps = set(
                detection_contract["frame_sizes"].keys())
            self.expected_detection_frames = len(
                self.expected_detection_stamps)
            self.recording_token = str(
                detection_contract["recording_token"])
            self.source_bag_sha256 = str(
                detection_contract["source_bag_sha256"])
            self.camera_mount = list(detection_contract["camera_mount"])
            # load_detection_contract() has already proved that this is an
            # allowed formal POI table (household three-class or door-only).
            # In formal replay it is the transaction-bound source of truth;
            # VisionInfo is transport metadata only and must never replace it.
            self.contract_labels = list(detection_contract["classes"])
            # required labels are the must-confirm SUBSET; the JSON may
            # carry additional optional POI classes (hall_v1: sofa/
            # cabinet/spray) that are written when they pass the gates
            # but never veto the transaction.
            if self.required_labels and not self.required_labels.issubset(
                    set(self.contract_labels)):
                raise ValueError(
                    "required_labels_csv must be a subset of detections "
                    "JSON classes")
            if not self.required_labels:
                self.required_labels = set(self.contract_labels)
            if self.pipeline_quiet_s <= 0.0:
                raise ValueError("~pipeline_quiet_s must be positive")
            # Bounded stereo-pairing shortfall: a stamp whose disparity was
            # never produced can no longer sync once the input is sealed.
            # Waiting for it would time out finalize and forfeit the yaml.
            fraction = float(gp("~max_unsynced_fraction", 0.05))
            if not 0.0 <= fraction < 1.0:
                raise ValueError(
                    "~max_unsynced_fraction must be in [0.0, 1.0)")
            self.max_unsynced_frames = int(math.floor(
                self.expected_detection_frames * fraction))
        else:
            self.expected_detection_stamps = set()
            self.source_bag_sha256 = ""
            self.max_unsynced_frames = 0
        self.map_file = os.path.abspath(os.path.expanduser(
            str(gp("~map_file", "")))) if gp("~map_file", "") else ""
        self.require_map_identity = bool(
            gp("~require_map_identity", False))
        self.map_identity_topic = str(
            gp("~map_identity_topic", "/survey/map_image_sha256"))
        if self.require_map_identity and not self.map_file:
            raise ValueError(
                "~map_file is required when ~require_map_identity=true")
        if self.transactional_finalize_only and not self.map_file:
            raise ValueError(
                "~map_file is required in transactional offline mode")
        if self.transactional_finalize_only and not self.require_map_identity:
            raise ValueError(
                "transactional offline mode requires ~require_map_identity=true")
        canonical_map = (load_map_identity(
            self.map_file, require_zero_yaw=True) if self.map_file else None)
        self.expected_map_hash = (
            canonical_map["map_image_sha256"] if canonical_map else "")
        self.expected_map_manifest_hash = (
            canonical_map["map_manifest_sha256"] if canonical_map else "")
        if self.transactional_finalize_only and canonical_map:
            if (str(detection_contract.get(
                    "map_image_sha256", "")).lower() !=
                    self.expected_map_hash or
                    str(detection_contract.get(
                        "map_manifest_sha256", "")).lower() !=
                    self.expected_map_manifest_hash):
                raise ValueError(
                    "detections JSON does not match the requested map YAML")
        self.recorded_map_hashes = set()
        self.map_manifest_identity_topic = str(gp(
            "~map_manifest_identity_topic",
            "/survey/map_manifest_sha256"))
        self.recorded_map_manifest_hashes = set()
        self.map_geometry_error = ""
        self.last_save_error = ""
        self.finalize_called = False
        self.finalize_lock = threading.Lock()
        self.accepting_observations = True
        self.finalize_token = str(gp("~finalize_token", ""))
        if self.transactional_finalize_only and not self.finalize_token:
            raise ValueError(
                "~finalize_token is required in transactional offline mode")
        if self.transactional_finalize_only and os.path.exists(self.save_path):
            raise ValueError(
                "transactional output already exists; choose a new "
                "~save_path or archive the old result: %s" % self.save_path)

        self.lock = threading.Lock()
        self.objects = []
        self.labels = (list(self.contract_labels)
                       if self.contract_labels is not None else None)
        self.K = self.D = self.R = self.P = None
        self.camera_stamps = deque(
            maxlen=max(10, int(gp("~camera_stamp_cache", 200))))
        self.scan = None
        # Do not map until an odometry sample matched to the observation has
        # passed the motion gate.
        self.twist_ok = False
        self.grid = None
        self.stats = {"eligible": 0, "stereo_ok": 0, "stereo_bad": 0,
                      "lidar_fallback": 0, "map_reject": 0,
                      "det_rx": 0, "disp_rx": 0, "sync_rx": 0,
                      "callbacks_inflight": 0,
                      "infrastructure_drop": 0,
                      "tf_startup_miss": 0, "tf_latest_fallback": 0,
                      "tf_stale_miss": 0,
                      # stereo_bad split: the _19 replay hit valid=0 and the
                      # aggregate counter could not say why.  Every stereo
                      # rejection lands in exactly one bucket below, and the
                      # yaml provenance snapshot records them all.
                      "stereo_roi_expanded": 0, "bbox_h_reject": 0,
                      "stereo_rej_decode": 0, "stereo_rej_rectify": 0,
                      "stereo_rej_roi_empty": 0, "stereo_rej_no_pos_d": 0,
                      "stereo_rej_few_samples": 0, "stereo_rej_calib": 0,
                      "stereo_rej_depth_gate": 0, "stereo_rej_iqr_or_gap": 0,
                      "stereo_rej_bearing": 0}
        # Set after the first successful map<-camera lookup; before that,
        # TF misses are the normal bag-open warm-up, not a chain break.
        self.map_tf_seen = False
        self.last_pipeline_activity_wall = None
        self.det_stamps_seen = set()
        self.sync_stamps_seen = set()
        self.pipeline_contract_error = ""
        self.health_last = (0, 0, 0)

        self.tfbuf = tf2_ros.Buffer(rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tfbuf)
        self.pub = rospy.Publisher("~markers", MarkerArray, queue_size=2,
                                   latch=True)
        rospy.Subscriber("camera_info", CameraInfo, self.cb_info,
                         queue_size=1)
        rospy.Subscriber("map", OccupancyGrid, self.cb_map, queue_size=1)
        rospy.Subscriber("vision_info", VisionInfo, self.cb_vinfo,
                         queue_size=1)
        if self.require_map_identity:
            rospy.Subscriber(self.map_identity_topic, String,
                             self.cb_map_identity, queue_size=10)
            rospy.Subscriber(self.map_manifest_identity_topic, String,
                             self.cb_map_manifest_identity, queue_size=10)
        if self.range_source == "stereo":
            if DisparityImage is None:
                raise RuntimeError(
                    "stereo_msgs is required for ~range_source=stereo")
            self.det_sub = message_filters.Subscriber(
                "detections", Detection2DArray)
            self.disp_sub = message_filters.Subscriber(
                "disparity", DisparityImage)
            self.odom_sub = message_filters.Subscriber("odom", Odometry)
            self.odom_cache = message_filters.Cache(
                self.odom_sub, max(50, self.sync_queue * 3))
            if self.transactional_finalize_only:
                self.measurement_sync = message_filters.TimeSynchronizer(
                    [self.det_sub, self.disp_sub], self.sync_queue)
            else:
                self.measurement_sync = \
                    message_filters.ApproximateTimeSynchronizer(
                        [self.det_sub, self.disp_sub], self.sync_queue,
                        self.stereo_sync_slop)
            self.measurement_sync.registerCallback(self.cb_det_stereo)
            self.det_sub.registerCallback(self.cb_det_seen)
            self.disp_sub.registerCallback(self.cb_disp_seen)
            # A missing or slow fallback scan must never hold up valid stereo.
            # Store it independently, then enforce a timestamp-age limit only
            # if disparity fails and fallback is actually needed.
            if self.stereo_lidar_fallback:
                rospy.Subscriber("scan", LaserScan, self.cb_scan, queue_size=5)
        else:
            self.det_sub = message_filters.Subscriber(
                "detections", Detection2DArray)
            self.scan_sub = message_filters.Subscriber("scan", LaserScan)
            self.odom_sub = message_filters.Subscriber("odom", Odometry)
            self.odom_cache = message_filters.Cache(
                self.odom_sub, max(50, self.sync_queue * 3))
            self.measurement_sync = message_filters.ApproximateTimeSynchronizer(
                [self.det_sub, self.scan_sub], self.sync_queue,
                self.lidar_sync_slop)
            self.measurement_sync.registerCallback(self.cb_det_lidar)
            self.det_sub.registerCallback(self.cb_det_seen)
        rospy.Timer(rospy.Duration(10.0), self.cb_timer)
        self.finalize_service = rospy.Service(
            "~finalize", FinalizeSemanticMap, self.cb_finalize)
        # Read-only probe: the replay runner polls it during rosbag play so
        # a permanently INVALID transaction stops the bag immediately
        # instead of burning the remaining replay wall-clock.
        self.status_service = rospy.Service(
            "~pipeline_status", Trigger, self.cb_pipeline_status)
        if not self.transactional_finalize_only:
            rospy.on_shutdown(self.save)
        rospy.loginfo(
            "semantic_mapper up (range_source=%s, bearing limit +/-%.0f deg)",
            self.range_source, math.degrees(self.max_bearing))

    # ---------------- callbacks ----------------
    def cb_info(self, m):
        self.camera_stamps.append(self.stamp_key(m.header.stamp))
        if self.K is None:
            k = np.array(m.K, dtype=np.float64).reshape(3, 3)
            d = np.array(m.D, dtype=np.float64).reshape(-1)
            r = np.array(m.R, dtype=np.float64).reshape(3, 3)
            p = np.array(m.P, dtype=np.float64).reshape(3, 4)
            if (d.size < 4 or not np.all(np.isfinite(d)) or
                    not np.all(np.isfinite(k)) or
                    not np.all(np.isfinite(r)) or
                    not np.all(np.isfinite(p)) or k[0, 0] <= 0.0 or
                    k[1, 1] <= 0.0 or p[0, 0] <= 0.0 or p[1, 1] <= 0.0 or
                    abs(np.linalg.det(r)) < 0.5):
                rospy.logwarn_throttle(
                    5.0, "camera_info is uncalibrated; stereo depth disabled")
                return
            self.K, self.D, self.R, self.P = k, d, r, p
            rospy.loginfo("camera_info: raw_fx=%.1f rect_fx=%.1f k1=%.2f",
                          self.K[0, 0], self.P[0, 0],
                          self.D[0] if len(self.D) else 0.0)

    @staticmethod
    def stamp_key(stamp):
        return int(stamp.secs) * 1000000000 + int(stamp.nsecs)

    def cb_det_seen(self, msg):
        self._pipeline_event(
            "det_rx", self.stamp_key(msg.header.stamp), "det")

    def cb_disp_seen(self, _msg):
        self._pipeline_event("disp_rx")

    def _record_pipeline_stamp_locked(self, stamp_ns, target):
        if int(stamp_ns) <= 0:
            self.pipeline_contract_error = (
                "%s callback has a nonpositive timestamp" % target)
            return
        seen = self.det_stamps_seen if target == "det" \
            else self.sync_stamps_seen
        if int(stamp_ns) in seen:
            self.pipeline_contract_error = (
                "duplicate %s callback stamp %d" % (target, int(stamp_ns)))
            return
        seen.add(int(stamp_ns))

    def _pipeline_event(self, stat_name=None, stamp_ns=None, target=None):
        with self.lock:
            if stat_name is not None:
                self.stats[stat_name] += 1
            if stamp_ns is not None and target is not None:
                self._record_pipeline_stamp_locked(stamp_ns, target)
            self.last_pipeline_activity_wall = time.time()

    def _pipeline_callback_begin(self, stat_name, stamp_ns):
        with self.lock:
            self.stats[stat_name] += 1
            self.stats["callbacks_inflight"] += 1
            self._record_pipeline_stamp_locked(stamp_ns, "sync")
            self.last_pipeline_activity_wall = time.time()

    def _pipeline_callback_end(self):
        with self.lock:
            self.stats["callbacks_inflight"] = max(
                0, self.stats["callbacks_inflight"] - 1)
            self.last_pipeline_activity_wall = time.time()

    def _increment_stat(self, name):
        with self.lock:
            self.stats[name] += 1

    def _pipeline_infrastructure_drop(self, reason):
        """Record a synchronized frame lost for a non-semantic reason.

        Confidence, geometry, motion and map-cell gates intentionally reject
        observations and are not errors.  Missing replay infrastructure is
        different: in formal mode one such loss makes the transaction
        permanently INVALID even though its sync stamp was already observed.
        """
        with self.lock:
            self.stats["infrastructure_drop"] = int(
                self.stats.get("infrastructure_drop", 0)) + 1
            if (getattr(self, "transactional_finalize_only", False) and
                    not getattr(self, "pipeline_contract_error", "")):
                self.pipeline_contract_error = (
                    "mapper infrastructure dropped a synchronized frame: %s" %
                    str(reason))
            self.last_pipeline_activity_wall = time.time()

    def _pipeline_contract_violation(self, reason):
        """Permanently invalidate a formal replay without changing labels."""
        with self.lock:
            # cb_finalize closes the observation gate only after every indexed
            # input and synchronized callback has drained.  A latched
            # VisionInfo callback delivered after that seal cannot affect any
            # classified observation and must not race the atomic commit.
            if not getattr(self, "accepting_observations", True):
                return
            if (getattr(self, "transactional_finalize_only", False) and
                    not getattr(self, "pipeline_contract_error", "")):
                self.pipeline_contract_error = str(reason)
            self.last_pipeline_activity_wall = time.time()

    def cb_pipeline_status(self, _request):
        """Report whether the replay transaction is already doomed.

        Success means the transaction can still commit; failure carries the
        locked INVALID reason.  This never mutates state, so polling it
        mid-replay cannot race the finalize commit.
        """
        with self.lock:
            error = self.pipeline_contract_error
            det_rx = int(self.stats.get("det_rx", 0))
            sync_rx = int(self.stats.get("sync_rx", 0))
        if error:
            return TriggerResponse(
                success=False, message="INVALID: %s" % error)
        return TriggerResponse(
            success=True,
            message="OK: det_rx=%d sync_rx=%d" % (det_rx, sync_rx))

    def nearest_odom(self, stamp):
        candidates = []
        try:
            before = self.odom_cache.getElemBeforeTime(stamp)
            after = self.odom_cache.getElemAfterTime(stamp)
            if before is not None:
                candidates.append(before)
            if after is not None:
                candidates.append(after)
        except Exception:
            return None
        if not candidates:
            return None
        odom = min(candidates,
                   key=lambda m: abs((m.header.stamp - stamp).to_sec()))
        age = abs((odom.header.stamp - stamp).to_sec())
        return odom if age <= self.odom_max_age else None

    def cb_det_stereo(self, detections, disparity):
        self._pipeline_callback_begin(
            "sync_rx", self.stamp_key(detections.header.stamp))
        try:
            odom = self.nearest_odom(detections.header.stamp)
            if odom is None:
                rospy.logwarn_throttle(
                    5.0, "no odometry within %.3fs of stereo observation",
                    self.odom_max_age)
                self._pipeline_infrastructure_drop(
                    "no timestamp-matched odometry for stereo observation")
                return
            self.cb_odom(odom)
            self.cb_det(detections, disparity)
        finally:
            self._pipeline_callback_end()

    def cb_det_lidar(self, detections, scan):
        self._pipeline_callback_begin(
            "sync_rx", self.stamp_key(detections.header.stamp))
        try:
            odom = self.nearest_odom(detections.header.stamp)
            if odom is None:
                rospy.logwarn_throttle(
                    5.0, "no odometry within %.3fs of lidar observation",
                    self.odom_max_age)
                self._pipeline_infrastructure_drop(
                    "no timestamp-matched odometry for lidar observation")
                return
            self.cb_scan(scan)
            self.cb_odom(odom)
            self.cb_det(detections)
        finally:
            self._pipeline_callback_end()

    def cb_timer(self, _event):
        if not self.transactional_finalize_only:
            self.save()
        if self.range_source == "stereo":
            with self.lock:
                now = (self.stats["det_rx"], self.stats["disp_rx"],
                       self.stats["sync_rx"])
            if (now[0] > self.health_last[0] and
                    now[1] > self.health_last[1] and
                    now[2] == self.health_last[2]):
                rospy.logwarn(
                    "detections and disparities are arriving but no new pair; "
                    "check timestamps, the DetectNet input-header patch, and "
                    "~stereo_sync_slop_s",
                )
            self.health_last = now

    def cb_scan(self, m):
        self.scan = m

    def cb_odom(self, m):
        self.twist_ok = (abs(m.twist.twist.angular.z) <= self.max_ang_vel and
                         abs(m.twist.twist.linear.x) <= self.max_lin_vel)

    def cb_map(self, m):
        try:
            require_zero_grid_yaw(
                m.info.origin.orientation, "semantic mapper occupancy map")
        except (TypeError, ValueError) as error:
            self.map_geometry_error = str(error)
            self.grid = None
            rospy.logfatal("semantic mapper rejected occupancy map: %s",
                           self.map_geometry_error)
            rospy.signal_shutdown(self.map_geometry_error)
            return
        self.grid = m

    def cb_vinfo(self, m):
        try:
            side_labels = rospy.get_param(m.database_location)
            if not isinstance(side_labels, (list, tuple)):
                raise ValueError("label parameter must be a list")
            side_labels = [str(label) for label in side_labels]
            if self.transactional_finalize_only:
                expected = list(self.contract_labels or [])
                if side_labels != expected:
                    reason = (
                        "VisionInfo label table disagrees with the validated "
                        "detections contract")
                    self._pipeline_contract_violation(reason)
                    rospy.logerr("%s: %s", reason, m.database_location)
                    return
                # Keep the independent copy derived from the validated JSON;
                # do not adopt mutable rosparam contents even when equal.
                rospy.loginfo("validated labels from %s (%d classes)",
                              m.database_location, len(expected))
                return
            self.labels = side_labels
            rospy.loginfo("labels from %s (%d classes)",
                          m.database_location, len(self.labels))
        except (KeyError, TypeError, ValueError) as error:
            if self.transactional_finalize_only:
                reason = (
                    "VisionInfo label table unavailable or invalid for the "
                    "validated detections contract: %s" % str(error))
                self._pipeline_contract_violation(reason)
                rospy.logerr("%s", reason)
            else:
                rospy.logwarn(
                    "vision_info param missing/invalid, using builtin COCO91: %s",
                    str(error))

    def cb_map_identity(self, message):
        value = str(message.data).strip().lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            rospy.logerr("invalid recorded map SHA-256 on %s: %r",
                         self.map_identity_topic, value)
            return
        with self.lock:
            self.recorded_map_hashes.add(value)

    def cb_map_manifest_identity(self, message):
        value = str(message.data).strip().lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            rospy.logerr("invalid recorded map manifest SHA-256 on %s: %r",
                         self.map_manifest_identity_topic, value)
            return
        with self.lock:
            self.recorded_map_manifest_hashes.add(value)

    def map_identity_status(self):
        if not getattr(self, "require_map_identity", False):
            return True, ""
        with self.lock:
            values = sorted(self.recorded_map_hashes)
        if not values:
            return False, "bag contains no recorded map identity"
        if len(values) != 1:
            return False, "bag contains multiple map identities: %s" % \
                ",".join(values)
        if values[0] != self.expected_map_hash:
            return False, (
                "bag/map identity mismatch: bag=%s map=%s" %
                (values[0], self.expected_map_hash))
        with self.lock:
            manifests = sorted(self.recorded_map_manifest_hashes)
        if not manifests:
            return False, "bag contains no recorded canonical map manifest"
        if len(manifests) != 1:
            return False, "bag contains multiple canonical map manifests: %s" % \
                ",".join(manifests)
        if manifests[0] != self.expected_map_manifest_hash:
            return False, (
                "bag/map manifest mismatch: bag=%s map=%s" %
                (manifests[0], self.expected_map_manifest_hash))
        return True, ""

    def label_of(self, cid):
        if getattr(self, "transactional_finalize_only", False):
            labels = getattr(self, "labels", None)
            expected = getattr(self, "contract_labels", None)
            if (not labels or expected is None or
                    list(labels) != list(expected)):
                reason = (
                    "formal mapper label table is not bound to the validated "
                    "detections contract")
                self._pipeline_contract_violation(reason)
                raise RuntimeError(reason)
        tab = self.labels if self.labels else COCO91
        return tab[cid] if 0 <= cid < len(tab) else "class_%d" % cid

    # ---------------- geometry ----------------
    def bearing(self, u, v):
        """Detection pixel -> bearing for the legacy lidar path."""
        if self.detections_rectified:
            xn = (float(u) - self.P[0, 2]) / self.P[0, 0]
        elif cv2 is not None and self.D is not None and np.any(self.D):
            p = cv2.undistortPoints(
                np.array([[[float(u), float(v)]]], dtype=np.float64),
                self.K, self.D)
            xn = float(p[0, 0, 0])
        else:
            xn = (float(u) - self.K[0, 2]) / self.K[0, 0]
        return -math.atan(xn)  # +x image right -> negative bearing

    def rectify_pixels(self, points):
        """Map raw-left pixels to the rectified-left/disparity image."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if self.detections_rectified:
            return pts.reshape(-1, 2)
        if cv2 is None:
            rospy.logwarn_throttle(
                5.0, "OpenCV missing: cannot rectify raw detection bboxes")
            return None
        if self.K is None or self.R is None or self.P is None \
                or self.K[0, 0] <= 0.0 or self.P[0, 0] <= 0.0:
            return None
        try:
            # OpenCV 3 (JetPack 4) accepts dst=None followed by R and P.
            distortion = self.D if self.D is not None and self.D.size else None
            out = cv2.undistortPoints(
                pts, self.K, distortion, None, self.R, self.P)
            return out.reshape(-1, 2)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "bbox rectification failed: %s",
                                   str(e))
            return None

    @staticmethod
    def disparity_array(msg):
        """Decode stereo_msgs/DisparityImage.image (32FC1) without cv_bridge."""
        image = msg.image
        if image.encoding not in ("32FC1", "32FC"):
            rospy.logwarn_throttle(
                5.0, "unsupported disparity encoding: %s", image.encoding)
            return None
        row_bytes = int(image.step)
        if image.height <= 0 or image.width <= 0 \
                or row_bytes < int(image.width) * 4 or row_bytes % 4:
            return None
        dtype = np.dtype(">f4" if image.is_bigendian else "<f4")
        row_values = int(row_bytes // 4)
        expected = int(image.height) * row_values
        try:
            flat = np.frombuffer(image.data, dtype=dtype, count=expected)
            return flat.reshape(int(image.height), row_values)[
                :, :int(image.width)]
        except (TypeError, ValueError) as e:
            rospy.logwarn_throttle(5.0, "bad disparity buffer: %s", str(e))
            return None

    def _bump_stat(self, name):
        """Tolerant counter for telemetry keys that may predate a fixture."""
        with self.lock:
            self.stats[name] = int(self.stats.get(name, 0)) + 1

    def _note_stereo_reject(self, reason):
        self._bump_stat("stereo_rej_" + reason)

    @staticmethod
    def _positive_roi_disparities(arr, disparity, x0, y0, x1, y1):
        """Positive, range-clamped disparities inside a clipped pixel ROI."""
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(arr.shape[1], int(x1))
        y1 = min(arr.shape[0], int(y1))
        if x1 <= x0 or y1 <= y0:
            return None
        d = arr[y0:y1, x0:x1].reshape(-1).astype(np.float64)
        valid = np.isfinite(d) & (d > 0.0)
        if disparity.max_disparity > disparity.min_disparity:
            valid &= (d >= float(disparity.min_disparity))
            valid &= (d <= float(disparity.max_disparity))
        return d[valid]

    def stereo_point(self, bbox, disparity):
        """Return robust (X,Y,Z) in the rectified left optical frame."""
        arr = self.disparity_array(disparity)
        if arr is None:
            self._note_stereo_reject("decode")
            return None

        u, v = float(bbox.center.x), float(bbox.center.y)
        sx, sy = float(bbox.size_x), float(bbox.size_y)
        raw_points = [
            (u, v),
            (u - sx / 2.0, v - sy / 2.0),
            (u + sx / 2.0, v - sy / 2.0),
            (u - sx / 2.0, v + sy / 2.0),
            (u + sx / 2.0, v + sy / 2.0),
        ]
        rect = self.rectify_pixels(raw_points)
        if rect is None or not np.all(np.isfinite(rect)):
            self._note_stereo_reject("rectify")
            return None

        uc, vc = float(rect[0, 0]), float(rect[0, 1])
        x_min, y_min = rect[1:, 0].min(), rect[1:, 1].min()
        x_max, y_max = rect[1:, 0].max(), rect[1:, 1].max()
        scale = min(1.0, max(0.10, self.depth_roi_scale))
        half_w = max(1.0, (x_max - x_min) * scale / 2.0)
        half_h = max(1.0, (y_max - y_min) * scale / 2.0)
        d = self._positive_roi_disparities(
            arr, disparity,
            math.floor(uc - half_w), math.floor(vc - half_h),
            math.ceil(uc + half_w) + 1, math.ceil(vc + half_h) + 1)
        # _19 live evidence: indoor disparity survives almost only on
        # high-texture edges such as the door frame, which sits on the bbox
        # border that the shrunk center ROI excludes by design.  A door
        # panel is textureless when closed and shows the next room when
        # open, so a starved center ROI falls back to the full rectified
        # bbox.  select_stereo_depth() still prefers the near cluster, so
        # an open doorway cannot pull the door into the far room, and the
        # depth/IQR/bearing/map/spread gates all stay active.
        expanded = False
        if getattr(self, "depth_roi_expand", True) and \
                (d is None or d.size < self.depth_min_samples):
            full = self._positive_roi_disparities(
                arr, disparity, math.floor(x_min), math.floor(y_min),
                math.ceil(x_max) + 1, math.ceil(y_max) + 1)
            if full is not None and (d is None or full.size > d.size):
                d = full
                expanded = True
        if d is None:
            # r2 live finding: whole rectified bboxes can land outside the
            # disparity frame (the alpha=0 rectification crop keeps only
            # the central ~2/3 of the raw FOV, zoomed 1.49x).  No depth
            # exists off-frame, so rejecting is correct -- but the numbers
            # must be visible to tell a cropped-away detection from a
            # coordinate-convention bug.
            self._note_stereo_reject("roi_empty")
            rospy.logwarn_throttle(
                5.0,
                "bbox outside disparity frame: raw=(%.0f,%.0f %.0fx%.0f) "
                "rect_center=(%.1f,%.1f) rect_x=[%.1f,%.1f] "
                "rect_y=[%.1f,%.1f] img=%dx%d",
                u, v, sx, sy, uc, vc, x_min, x_max, y_min, y_max,
                arr.shape[1], arr.shape[0])
            return None
        f = float(disparity.f)
        # stereo_image_proc reports a positive baseline T for the standard
        # left-to-right camera layout. A negative value indicates reversed or
        # invalid calibration and must not be hidden with abs().
        baseline = float(disparity.T)
        if f <= 0.0 or baseline <= 0.0:
            self._note_stereo_reject("calib")
            return None
        if d.size == 0:
            self._note_stereo_reject("no_pos_d")
            return None
        if d.size < self.depth_min_samples:
            self._note_stereo_reject("few_samples")
            return None
        if expanded:
            self._bump_stat("stereo_roi_expanded")

        depths = (f * baseline) / d
        depths = depths[np.isfinite(depths)]
        depths = depths[(depths >= self.depth_min) &
                        (depths <= self.depth_max)]
        if depths.size < self.depth_min_samples:
            self._note_stereo_reject("depth_gate")
            return None
        z = select_stereo_depth(
            depths, self.depth_min_samples, self.depth_iqr_max,
            getattr(self, "depth_near_gap", 0.60))
        if z is None:
            self._note_stereo_reject("iqr_or_gap")
            return None

        fx = float(self.P[0, 0])
        fy = float(self.P[1, 1])
        cx = float(self.P[0, 2])
        cy = float(self.P[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            self._note_stereo_reject("calib")
            return None
        bearing = math.atan2(uc - cx, fx)
        if self.max_bearing > 0.0 and abs(bearing) > self.max_bearing:
            self._note_stereo_reject("bearing")
            return None
        x = (uc - cx) * float(z) / fx
        y = (vc - cy) * float(z) / fy
        return x, y, float(z)

    def sector_range(self, th_lo, th_hi, scan=None):
        """median range in [th_lo,th_hi] (laser frame) + IQR check."""
        s = scan if scan is not None else self.scan
        if s is None:
            return None
        n = len(s.ranges)
        idx = []
        for i in range(n):
            a = s.angle_min + i * s.angle_increment
            a = math.atan2(math.sin(a), math.cos(a))
            if th_lo <= a <= th_hi:
                r = s.ranges[i]
                if s.range_min < r < s.range_max and not math.isinf(r) \
                        and not math.isnan(r):
                    idx.append(r)
        if len(idx) < 3:
            return None
        q25, q50, q75 = np.percentile(idx, [25, 50, 75])
        if (q75 - q25) > self.iqr_max:
            return None  # mixed foreground/background
        return float(q50)

    def _grid_cell(self, x, y):
        g = self.grid
        if g is None:
            return None
        c = int(math.floor(
            (x - g.info.origin.position.x) / g.info.resolution))
        r = int(math.floor(
            (y - g.info.origin.position.y) / g.info.resolution))
        if not (0 <= c < g.info.width and 0 <= r < g.info.height):
            return None
        return c, r

    def _near_occupied(self, x, y):
        g = self.grid
        cell = self._grid_cell(x, y)
        if cell is None:
            return False
        res = float(g.info.resolution)
        radius = float(self.occupied_radius_m)
        if res <= 0.0 or radius < 0.0:
            return False
        ox = g.info.origin.position.x
        oy = g.info.origin.position.y
        w, h = g.info.width, g.info.height
        thresh = int(self.occupied_threshold)
        span = int(math.ceil(radius / res))
        c0, r0 = cell
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                c, r = c0 + dc, r0 + dr
                if not (0 <= c < w and 0 <= r < h):
                    continue
                cx = ox + (c + 0.5) * res
                cy = oy + (r + 0.5) * res
                if math.hypot(cx - x, cy - y) > radius:
                    continue
                if int(g.data[r * w + c]) >= thresh:
                    return True
        return False

    def map_ok(self, x, y, label=""):
        g = self.grid
        if not self.use_map_check:
            return True
        if g is None:
            return not self.require_map_received
        # int() truncates toward zero and would map a point just below the
        # origin into cell zero.  floor() preserves the occupancy-grid
        # half-open bounds for negative coordinates.
        cell = self._grid_cell(x, y)
        if cell is None:
            return False
        c, r = cell
        value = int(g.data[r * g.info.width + c])
        if value < 0:
            return False
        if label in getattr(self, "structure_labels", set()):
            return self._near_occupied(x, y)
        return True

    def lidar_map_point(self, bbox, t_cl, t_ml, scan=None):
        """Legacy bbox-bearing + lidar-range projection to map (x,y)."""
        u, v = bbox.center.x, bbox.center.y
        b_c = self.bearing(u, v)
        if self.max_bearing > 0.0 and abs(b_c) > self.max_bearing:
            return None
        half = abs(self.bearing(u + bbox.size_x / 2.0, v) - b_c)
        th = laser_azimuth(t_cl, b_c)
        r = self.sector_range(th - half - self.sector_pad,
                              th + half + self.sector_pad, scan)
        if r is None:
            return None
        lx, ly = r * math.cos(th), r * math.sin(th)
        yaw_ml = yaw_from_quat(t_ml.transform.rotation)
        ox = t_ml.transform.translation.x
        oy = t_ml.transform.translation.y
        mx = ox + lx * math.cos(yaw_ml) - ly * math.sin(yaw_ml)
        my = oy + lx * math.sin(yaw_ml) + ly * math.cos(yaw_ml)
        return mx, my

    # ---------------- main ----------------
    def _stereo_map_transform(self, cam, stamp):
        """Look up map<-camera at the detection stamp, replay-tolerant.

        A TF miss costs at most this one frame and NEVER invalidates the
        transaction (bag 19 lost a 90-minute replay to a single bag-open
        miss).  Classes, per classify_stereo_tf_miss:
          - startup: map chain not replayed into the buffer yet
          - future:  stamp a few ms ahead of the buffer; the newest
            transform substitutes when within ~tf_future_tolerance_s
          - broken/stale: chain failed after being proven, or the slip is
            unbounded -- counted separately so finalize stats and the yaml
            provenance expose the gap
        Geometry only ever uses the exact or bounded-slip transform, so a
        dropped frame can reduce hits but never move an object.
        """
        try:
            transform = self.tfbuf.lookup_transform(
                "map", cam, stamp, rospy.Duration(self.tf_lookup_timeout))
            self.map_tf_seen = True
            return transform
        except Exception as error:
            miss = classify_stereo_tf_miss(str(error), self.map_tf_seen)
            detail = str(error)
        if miss == TF_MISS_STARTUP:
            self._increment_stat("tf_startup_miss")
            rospy.logwarn_throttle(
                5.0, "map TF not buffered yet; dropped one frame: %s",
                detail)
            return None
        if miss == TF_MISS_FUTURE:
            latest = None
            try:
                latest = self.tfbuf.lookup_transform(
                    "map", cam, rospy.Time(0))
            except Exception as late_error:
                detail = "%s; latest lookup also failed: %s" % (
                    detail, str(late_error))
            if latest is not None:
                try:
                    slip_ok = latest_tf_fallback_ok(
                        stamp.to_sec(), latest.header.stamp.to_sec(),
                        self.tf_future_tolerance)
                except Exception:
                    slip_ok = False
                if slip_ok:
                    self.map_tf_seen = True
                    self._increment_stat("tf_latest_fallback")
                    return latest
        self._increment_stat("tf_stale_miss")
        rospy.logwarn_throttle(
            5.0, "map TF chain stale or broken; dropped one frame: %s",
            detail)
        return None

    def cb_det(self, msg, disparity=None):
        if self.K is None:
            self._pipeline_infrastructure_drop(
                "camera calibration was unavailable")
            return
        # Excess motion is an intentional observation gate, not a replay
        # infrastructure failure.
        if not self.twist_ok:
            return
        if self.range_source == "stereo" and disparity is None:
            self._pipeline_infrastructure_drop(
                "stereo callback had no disparity image")
            return
        if self.range_source == "lidar" and self.scan is None:
            self._pipeline_infrastructure_drop(
                "lidar callback had no synchronized scan")
            return
        if (self.use_map_check and self.require_map_received and
                self.grid is None):
            self._pipeline_infrastructure_drop(
                "occupancy map was unavailable")
            return
        stamp = msg.header.stamp
        # A stereo disparity is itself produced from a camera image.  Accept
        # that proof directly so independent CameraInfo connection ordering
        # cannot drop an otherwise valid synchronized observation.  The
        # legacy lidar path still requires an explicitly seen camera stamp.
        camera_stamp_proven = (
            disparity is not None or
            self.stamp_key(stamp) in self.camera_stamps)
        if self.require_detection_camera_stamp and not camera_stamp_proven:
            rospy.logwarn_throttle(
                5.0, "synchronized detection stamp is not a camera stamp; "
                "online DetectNet must copy input->header, otherwise check "
                "the offline CameraInfo topic and bag timestamps")
            self._pipeline_infrastructure_drop(
                "detection timestamp lacked camera provenance")
            return
        disp_frame = ""
        if disparity is not None:
            disp_frame = (disparity.header.frame_id or
                          disparity.image.header.frame_id)
        det_frame = msg.header.frame_id
        if disp_frame and det_frame and disp_frame != det_frame:
            rospy.logwarn_throttle(
                5.0, "detection/disparity frame mismatch: %s != %s",
                det_frame, disp_frame)
            self._pipeline_infrastructure_drop(
                "detection/disparity frame mismatch: %s != %s" %
                (det_frame, disp_frame))
            return
        # stereo_point() is defined in the disparity image's rectified-left
        # optical frame, so that frame is authoritative for the 3-D TF.
        cam = disp_frame or det_frame or "stereo_left_optical"
        t_mc = t_cl = t_ml = None
        scan_for_lidar = None
        if self.range_source == "stereo":
            t_mc = self._stereo_map_transform(cam, stamp)
            if t_mc is None:
                return
        if self.range_source == "lidar" or self.stereo_lidar_fallback:
            scan = self.scan
            scan_close = scan is not None
            if self.range_source == "stereo" and scan_close:
                try:
                    scan_age = abs((scan.header.stamp - stamp).to_sec())
                    scan_close = scan_age <= self.fallback_scan_age
                except Exception:
                    scan_close = False
            if scan_close:
                try:
                    scan_for_lidar = scan
                    laser = scan.header.frame_id
                    scan_stamp = scan.header.stamp
                    t_cl = self.tfbuf.lookup_transform(
                        laser, cam, scan_stamp,
                        rospy.Duration(self.tf_lookup_timeout))
                    t_ml = self.tfbuf.lookup_transform(
                        "map", laser, scan_stamp,
                        rospy.Duration(self.tf_lookup_timeout))
                except Exception as e:
                    if self.range_source == "lidar":
                        rospy.logwarn_throttle(5.0, "lidar TF miss: %s",
                                               str(e))
                        self._pipeline_infrastructure_drop(
                            "lidar map transform unavailable: %s" % str(e))
                        return
                    rospy.logwarn_throttle(
                        5.0, "optional lidar fallback unavailable: %s", str(e))
                    t_cl = t_ml = None

        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            if hyp.score < self.score_min:
                continue
            label = self.label_of(hyp.id)
            if label in self.exclude or label == "BACKGROUND":
                continue
            min_bbox_h = float(getattr(self, "min_structure_bbox_h", 0.0))
            if (min_bbox_h > 0.0
                    and label in getattr(self, "structure_labels", set())
                    and float(det.bbox.size_y) < min_bbox_h):
                # A structure box shorter than a full door at depth_max is
                # detector noise, not an observable door.  Filtering it
                # here keeps sliver floor points from chaining into smear
                # tracks that absorb real door observations.
                self._bump_stat("bbox_h_reject")
                continue
            self._increment_stat("eligible")
            point = None
            if self.range_source == "stereo":
                xyz = self.stereo_point(det.bbox, disparity)
                if xyz is not None:
                    if self.mirror_x:
                        xyz = (-xyz[0], xyz[1], xyz[2])
                    self._increment_stat("stereo_ok")
                    mx, my, _mz = transform_xyz(t_mc, xyz)
                    point = (mx, my)
                elif self.stereo_lidar_fallback and t_cl is not None:
                    self._increment_stat("stereo_bad")
                    point = self.lidar_map_point(
                        det.bbox, t_cl, t_ml, scan_for_lidar)
                    if point is not None:
                        self._increment_stat("lidar_fallback")
                else:
                    self._increment_stat("stereo_bad")
            else:
                point = self.lidar_map_point(
                    det.bbox, t_cl, t_ml, scan_for_lidar)
            if point is None:
                continue
            mx, my = point
            if not self.map_ok(mx, my, label):
                self._increment_stat("map_reject")
                continue
            self.associate(label, int(hyp.id), mx, my, stamp)
        self.publish()
        if self.range_source == "stereo":
            rospy.loginfo_throttle(
                10.0,
                "stereo depth stats: valid=%d invalid=%d fallback=%d "
                "eligible=%d map_reject=%d tf_slip_used=%d tf_dropped=%d",
                self.stats["stereo_ok"], self.stats["stereo_bad"],
                self.stats["lidar_fallback"], self.stats["eligible"],
                self.stats["map_reject"], self.stats["tf_latest_fallback"],
                self.stats["tf_startup_miss"] + self.stats["tf_stale_miss"])
            rospy.loginfo_throttle(
                10.0,
                "stereo reject split: no_pos_d=%d few_samples=%d "
                "depth_gate=%d iqr_or_gap=%d bearing=%d rectify=%d "
                "roi_empty=%d calib=%d decode=%d roi_expanded=%d "
                "bbox_h=%d",
                self.stats.get("stereo_rej_no_pos_d", 0),
                self.stats.get("stereo_rej_few_samples", 0),
                self.stats.get("stereo_rej_depth_gate", 0),
                self.stats.get("stereo_rej_iqr_or_gap", 0),
                self.stats.get("stereo_rej_bearing", 0),
                self.stats.get("stereo_rej_rectify", 0),
                self.stats.get("stereo_rej_roi_empty", 0),
                self.stats.get("stereo_rej_calib", 0),
                self.stats.get("stereo_rej_decode", 0),
                self.stats.get("stereo_roi_expanded", 0),
                self.stats.get("bbox_h_reject", 0))

    def assoc_radius(self, label):
        if label in SMALL:
            return 0.35
        if label in BIG:
            return 0.9
        return self.assoc_default

    def associate(self, label, cid, x, y, stamp):
        with self.lock:
            if not getattr(self, "accepting_observations", True):
                return
            best, bd = None, 1e9
            for o in self.objects:
                if o.label != label:
                    continue
                px, py = o.pos()
                d = math.hypot(px - x, py - y)
                if d < bd:
                    best, bd = o, d
            if best is not None and bd <= self.assoc_radius(label):
                # Multiple prompts/detector proposals from one image must not
                # masquerade as independent temporal confirmations.
                if best.last == stamp:
                    return
                best.add(x, y, stamp)
            else:
                o = TrackedObject(label, cid)
                o.add(x, y, stamp)
                self.objects.append(o)

    # ---------------- output ----------------
    def confirmed(self):
        spread = float(getattr(self, "max_spread_m", 0.25))
        kept = []
        for tracked in self.objects:
            if tracked.hits < self.min_hits:
                continue
            if tracked.std() > spread:
                continue
            kept.append(tracked)
        return kept

    def _track_debrief(self, limit=6):
        """Name the binding gate when confirmation fails.

        r3 ended confirmed=0 with 28 map-accepted observations and no way
        to tell whether tracks died at min_hits or at the spread gate.
        This one line in the refusal message answers that without another
        90-minute replay.
        """
        ranked = sorted(self.objects, key=lambda o: o.hits, reverse=True)
        parts = []
        for o in ranked[:int(limit)]:
            if not o.obs:
                continue
            x, y = o.pos()
            parts.append("%s hits=%d spread=%.2f at (%.2f,%.2f)" %
                         (o.label, o.hits, o.std(), x, y))
        return "; ".join(parts) if parts else "no tracks"

    def publish(self):
        ma = MarkerArray()
        mid = 0
        for o in self.confirmed():
            x, y = o.pos()
            s = Marker()
            s.header.frame_id = "map"
            s.ns, s.id, s.type, s.action = "obj", mid, Marker.SPHERE, 0
            s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, 0.1
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 0.18
            s.color.r, s.color.g, s.color.b, s.color.a = 0.9, 0.2, 0.2, 0.9
            s.lifetime = rospy.Duration(0)
            mid += 1
            t = Marker()
            t.header.frame_id = "map"
            t.ns, t.id, t.type, t.action = "txt", mid, Marker.TEXT_VIEW_FACING, 0
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, 0.35
            t.pose.orientation.w = 1.0
            t.scale.z = 0.22
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = "%s(%d)" % (o.label, o.hits)
            mid += 1
            ma.markers.append(s)
            ma.markers.append(t)
        self.pub.publish(ma)

    def save(self):
        if (getattr(self, "require_map_received", False) and
                self.grid is None):
            self.last_save_error = "required occupancy map was never received"
            rospy.logerr_throttle(30.0, "semantic output refused: %s",
                                  self.last_save_error)
            return False
        identity_ok, identity_error = self.map_identity_status()
        if not identity_ok:
            self.last_save_error = identity_error
            rospy.logerr_throttle(30.0, "semantic output refused: %s",
                                  identity_error)
            return False
        objs = self.confirmed()
        confirmed_labels = set(item.label for item in objs)
        required_labels = getattr(self, "required_labels", set())
        missing_labels = sorted(required_labels - confirmed_labels)
        if missing_labels:
            self.last_save_error = (
                "missing required POIs: %s [tracks: %s]" %
                (",".join(missing_labels), self._track_debrief()))
            rospy.logerr_throttle(30.0, "semantic output refused: %s",
                                  self.last_save_error)
            return False
        if not objs:
            self.last_save_error = "no confirmed observations"
            rospy.logwarn_throttle(
                30.0, "no confirmed observations; preserving existing %s",
                self.save_path)
            return False
        with self.lock:
            recorded_hashes = sorted(self.recorded_map_hashes)
            recorded_manifest_hashes = sorted(
                self.recorded_map_manifest_hashes)
            stats_snapshot = dict(self.stats)
            pipeline_contract_error = self.pipeline_contract_error
        if (getattr(self, "transactional_finalize_only", False) and
                pipeline_contract_error):
            self.last_save_error = (
                "transactional pipeline is invalid: %s" %
                pipeline_contract_error)
            rospy.logerr("semantic output refused: %s", self.last_save_error)
            return False
        tmp_path = None
        try:
            import yaml
            data = {
                "frame": "map",
                "provenance": {
                    "source_bag": self.source_bag,
                    "detections_file": self.detections_file,
                    "recorded_map_image_sha256": recorded_hashes,
                    "recorded_map_manifest_sha256":
                        recorded_manifest_hashes,
                    "required_labels": sorted(required_labels),
                    "stats": stats_snapshot,
                    "pipeline_contract_error": pipeline_contract_error,
                },
                "objects": [
                {"label": o.label, "class_id": o.cid,
                 "x": round(o.pos()[0], 3), "y": round(o.pos()[1], 3),
                 "hits": o.hits, "spread_m": round(o.std(), 3)}
                for o in objs]}
            if self.transactional_finalize_only:
                data.update({
                    "schema": "slam-car-semantic-objects-v1",
                    "completed": True,
                    "recording_token": self.recording_token,
                    "source_bag_sha256": self.source_bag_sha256,
                    "camera_mount": list(self.camera_mount),
                })
            if self.expected_map_hash:
                data["map_image_sha256"] = self.expected_map_hash
                data["map_manifest_sha256"] = \
                    self.expected_map_manifest_hash
            d = os.path.dirname(self.save_path)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            target_dir = d if d else "."
            with tempfile.NamedTemporaryFile(
                    mode="w", prefix=".semantic_objects.", suffix=".tmp",
                    dir=target_dir, delete=False) as f:
                tmp_path = f.name
                yaml.safe_dump(data, f, default_flow_style=False)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, self.save_path)
            tmp_path = None
            rospy.loginfo_throttle(30.0, "saved %d objects -> %s",
                                   len(objs), self.save_path)
            self.last_save_error = ""
            return True
        except Exception as e:
            self.last_save_error = str(e)
            rospy.logerr("save failed: %s", str(e))
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def cb_finalize(self, request):
        """Commit one replay exactly once, by its per-launch token holder."""
        if not self.finalize_lock.acquire(False):
            return FinalizeSemanticMapResponse(
                success=False, message="finalize is already in progress")
        try:
            supplied = str(getattr(request, "token", ""))
            expected = str(getattr(self, "finalize_token", ""))
            if (not supplied or not expected or
                    not hmac.compare_digest(supplied, expected)):
                return FinalizeSemanticMapResponse(
                    success=False, message="invalid replay finalize token")
            if self.finalize_called:
                return FinalizeSemanticMapResponse(
                    success=False,
                    message="finalize already called; refusing a second commit")
            # Prove that every indexed detection reached fusion and that no
            # callback remains active.  Fixed post-bag sleeps can otherwise
            # commit a silently truncated Nano backlog.
            with self.lock:
                if getattr(self, "transactional_finalize_only", False):
                    ready, detail = pipeline_ready(
                        self.expected_detection_frames, self.stats,
                        self.last_pipeline_activity_wall, time.time(),
                        self.pipeline_quiet_s,
                        self.expected_detection_stamps,
                        self.det_stamps_seen, self.sync_stamps_seen,
                        self.pipeline_contract_error,
                        max_unsynced_frames=self.max_unsynced_frames)
                    if not ready:
                        prefix = "" if detail.startswith("INVALID:") \
                            else "NOT_READY: "
                        return FinalizeSemanticMapResponse(
                            success=False, message=prefix + detail)
                    # Any tolerated pairing shortfall is committed to the
                    # yaml provenance instead of disappearing silently.
                    self.stats["sync_missing"] = len(
                        self.expected_detection_stamps -
                        self.sync_stamps_seen)
                # Close the observation gate under the same lock used by
                # callback counters and associate().
                self.accepting_observations = False
            self.finalize_called = True
            confirmed_count = len(self.confirmed())
            success = self.save()
            message = (
                "confirmed=%d det_rx=%d sync_rx=%d stereo_ok=%d eligible=%d "
                "map_reject=%d output=%s reason=%s" %
                (confirmed_count, self.stats.get("det_rx", 0),
                 self.stats.get("sync_rx", 0),
                 self.stats.get("stereo_ok", 0),
                 self.stats.get("eligible", 0),
                 self.stats.get("map_reject", 0), self.save_path,
                 self.last_save_error or "ok"))
            if success:
                rospy.loginfo(
                    "offline semantic finalize succeeded: %s", message)
            else:
                rospy.logerr(
                    "offline semantic finalize failed: %s", message)
            return FinalizeSemanticMapResponse(
                success=bool(success), message=message)
        finally:
            self.finalize_lock.release()


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def transform_xyz(transform_stamped, point):
    """Apply a full 3-D geometry_msgs/TransformStamped to an xyz tuple."""
    vx, vy, vz = point
    q = transform_stamped.transform.rotation
    # Quaternion-vector rotation: v' = v + 2w(qv x v) + 2(qv x(qv x v))
    cx = q.y * vz - q.z * vy
    cy = q.z * vx - q.x * vz
    cz = q.x * vy - q.y * vx
    c2x = q.y * cz - q.z * cy
    c2y = q.z * cx - q.x * cz
    c2z = q.x * cy - q.y * cx
    rx = vx + 2.0 * (q.w * cx + c2x)
    ry = vy + 2.0 * (q.w * cy + c2y)
    rz = vz + 2.0 * (q.w * cz + c2z)
    t = transform_stamped.transform.translation
    return rx + t.x, ry + t.y, rz + t.z


def laser_azimuth(t_cl, bearing_cam):
    """Transform a ray at `bearing_cam` (optical frame, z forward,
    x right) into the laser frame, return its azimuth there."""
    # direction in optical frame
    dx, dy, dz = math.sin(-bearing_cam), 0.0, math.cos(-bearing_cam)
    q = t_cl.transform.rotation
    # rotate vector by quaternion (Hamilton)
    x, y, z, w = q.x, q.y, q.z, q.w
    # v' = v + 2*qv x (qv x v + w*v)
    qv = (x, y, z)
    cx = (qv[1] * dz - qv[2] * dy, qv[2] * dx - qv[0] * dz,
          qv[0] * dy - qv[1] * dx)
    cx = (cx[0] + w * dx, cx[1] + w * dy, cx[2] + w * dz)
    c2 = (qv[1] * cx[2] - qv[2] * cx[1], qv[2] * cx[0] - qv[0] * cx[2],
          qv[0] * cx[1] - qv[1] * cx[0])
    vx, vy = dx + 2.0 * c2[0], dy + 2.0 * c2[1]
    return math.atan2(vy, vx)


if __name__ == "__main__":
    rospy.init_node("semantic_mapper")
    SemanticMapper()
    rospy.spin()
