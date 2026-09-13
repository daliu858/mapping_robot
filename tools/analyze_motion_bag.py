#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize one bounded geometry-SLAM motion bag on a ROS 1 host."""
from __future__ import print_function

import argparse
import json
import math

import rosbag


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_delta(current, start):
    return math.atan2(math.sin(current - start), math.cos(current - start))


def pose(msg):
    value = msg.pose.pose
    return [float(value.position.x), float(value.position.y),
            yaw(value.orientation)]


def map_counts(msg):
    free = sum(1 for value in msg.data if 0 <= int(value) < 50)
    occupied = sum(1 for value in msg.data if int(value) >= 50)
    return {"width": int(msg.info.width), "height": int(msg.info.height),
            "free": free, "occupied": occupied, "known": free + occupied}


def summarize(path):
    topics = {}
    poses = {"/odom": [], "/odom_raw": []}
    cmd = []
    motors = {name: [] for name in
              ("/motor/lvel", "/motor/rvel", "/motor/lset", "/motor/rset")}
    safety = []
    maps = []
    scans = []
    last_map_to_odom = None
    map_to_odom_samples = []
    amcl = []
    particle_counts = []
    image_stamps = {
        "/stereo/left/image_raw/compressed": [],
        "/stereo/right/image_raw/compressed": [],
    }
    start = None
    end = None
    with rosbag.Bag(path, "r") as bag:
        for topic, msg, stamp in bag.read_messages():
            sec = stamp.to_sec()
            start = sec if start is None else min(start, sec)
            end = sec if end is None else max(end, sec)
            topics[topic] = topics.get(topic, 0) + 1
            if topic in poses:
                poses[topic].append((sec, pose(msg)))
            elif topic == "/cmd_vel":
                cmd.append((sec, float(msg.linear.x), float(msg.angular.z)))
            elif topic in motors:
                motors[topic].append((sec, int(msg.data)))
            elif topic == "/safety_stop":
                safety.append(bool(msg.data))
            elif topic == "/map":
                maps.append(map_counts(msg))
            elif topic == "/scan":
                valid = [float(value) for value in msg.ranges
                         if not math.isnan(float(value)) and
                         not math.isinf(float(value)) and
                         float(msg.range_min) <= float(value) <=
                         float(msg.range_max)]
                scans.append((len(msg.ranges), len(valid),
                              min(valid) if valid else None))
            elif topic == "/amcl_pose":
                value = msg.pose.pose
                covariance = list(msg.pose.covariance)
                amcl.append({
                    "stamp": msg.header.stamp.to_sec(),
                    "pose": [float(value.position.x),
                             float(value.position.y),
                             yaw(value.orientation)],
                    "variance": [float(covariance[0]),
                                 float(covariance[7]),
                                 float(covariance[35])],
                })
            elif topic == "/particlecloud":
                particle_counts.append(len(msg.poses))
            elif topic in image_stamps:
                image_stamps[topic].append(msg.header.stamp.to_sec())
            elif topic == "/tf":
                for transform in msg.transforms:
                    parent = transform.header.frame_id.lstrip("/")
                    child = transform.child_frame_id.lstrip("/")
                    if parent == "map" and child == "odom":
                        value = transform.transform
                        last_map_to_odom = [
                            float(value.translation.x),
                            float(value.translation.y),
                            yaw(value.rotation),
                        ]
                        map_to_odom_samples.append(last_map_to_odom)

    output = {
        "path": path,
        "duration_s": 0.0 if start is None else end - start,
        "topics": topics,
        "safety_values": sorted(set(safety)),
    }
    for name, samples in poses.items():
        if samples:
            first = samples[0][1]
            last = samples[-1][1]
            output[name] = {
                "first": first,
                "last": last,
                "translation_m": math.hypot(last[0] - first[0],
                                             last[1] - first[1]),
                "yaw_change_rad": angle_delta(last[2], first[2]),
                "samples": len(samples),
            }
    nonzero_cmd = [(sec, linear, angular) for sec, linear, angular in cmd
                   if abs(linear) > 1e-9 or abs(angular) > 1e-9]
    output["cmd_vel"] = {
        "samples": len(cmd),
        "nonzero_samples": len(nonzero_cmd),
        "max_abs_linear": max([abs(item[1]) for item in cmd] or [0.0]),
        "max_abs_angular": max([abs(item[2]) for item in cmd] or [0.0]),
        "nonzero_span_s": (nonzero_cmd[-1][0] - nonzero_cmd[0][0]
                           if nonzero_cmd else 0.0),
    }
    output["motors"] = {
        name: {"samples": len(samples),
               "min": min([value for _sec, value in samples] or [0]),
               "max": max([value for _sec, value in samples] or [0])}
        for name, samples in motors.items()
    }
    if maps:
        output["map"] = {"samples": len(maps), "first": maps[0],
                         "last": maps[-1],
                         "known_delta": maps[-1]["known"] -
                         maps[0]["known"]}
    if scans:
        output["scan"] = {
            "samples": len(scans),
            "points_per_scan": sorted(set(item[0] for item in scans)),
            "valid_ratio": sum(item[1] for item in scans) /
            float(sum(item[0] for item in scans)),
            "minimum_range_m": min(item[2] for item in scans
                                   if item[2] is not None),
        }
    if last_map_to_odom is not None:
        output["map_to_odom"] = last_map_to_odom
        output["map_to_odom_samples"] = len(map_to_odom_samples)
        output["map_to_odom_first"] = map_to_odom_samples[0]
        raw = output.get("/odom_raw", {}).get("last")
        if raw is not None:
            mx, my, ma = last_map_to_odom
            ox, oy, oa = raw
            output["initial_pose_map"] = [
                mx + math.cos(ma) * ox - math.sin(ma) * oy,
                my + math.sin(ma) * ox + math.cos(ma) * oy,
                ma + oa,
            ]
    if amcl:
        output["amcl"] = {"samples": len(amcl), "first": amcl[0],
                          "last": amcl[-1]}
    if particle_counts:
        output["particlecloud"] = {
            "samples": len(particle_counts),
            "first_count": particle_counts[0],
            "last_count": particle_counts[-1],
            "min_count": min(particle_counts),
            "max_count": max(particle_counts),
        }
    left = image_stamps["/stereo/left/image_raw/compressed"]
    right = image_stamps["/stereo/right/image_raw/compressed"]
    if left or right:
        offsets = []
        left = sorted(left)
        right = sorted(right)
        i = j = 0
        gate = 0.020
        while i < len(left) and j < len(right):
            delta = right[j] - left[i]
            if delta < -gate:
                j += 1
            elif delta > gate:
                i += 1
            else:
                offsets.append(abs(delta) * 1000.0)
                i += 1
                j += 1
        ordered = sorted(offsets)
        percentile_index = lambda fraction: min(
            len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        output["stereo"] = {
            "left": len(left), "right": len(right),
            "pairs_within_20ms": len(offsets),
            "left_acceptance": (float(len(offsets)) / len(left)
                                if left else 0.0),
            "median_dt_ms": (ordered[percentile_index(0.50)]
                             if ordered else None),
            "p95_dt_ms": (ordered[percentile_index(0.95)]
                          if ordered else None),
            "max_dt_ms": max(ordered) if ordered else None,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    args = parser.parse_args()
    print(json.dumps(summarize(args.bag), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
