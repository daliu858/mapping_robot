#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pure, ROS-free sector logic for scan_clip.py.

The rplidar sits at base yaw=pi (lidar.launch static TF), so the robot HEAD
lies near +-pi in scan angles.  The old clip rewrote EVERY return closer
than ~min_range_m to range_max regardless of direction: that hid the
chassis posts, but it also made a face-on obstacle inside 0.22 m invisible
to AMCL, the costmaps and the tour's braking cone.  This module keeps near
returns in a head sector and clips only the body sectors.

It must stay importable without rospy so the PC suite
(src/jetbot_pro/tests/test_scan_clip_sectors.py) can falsify it, and must
be deployed next to scan_clip.py in BOTH catkin trees (src/scripts and
devel/lib) -- see start_nav_survey.sh and releases/README.md.
"""
from __future__ import print_function

import math

# base_footprint -> laser_frame yaw (lidar.launch static TF).
LASER_YAW_OFFSET = math.pi

# Half-width of the head sector where near returns are KEPT.  Deliberately
# wider than the tour's 0.70 braking cone and at least as wide as its 1.05
# turn-sweep shoulder, so no gating sector can ever ask for a distance this
# clip already rewrote (test_scan_clip_sectors.py locks the ordering).
HEAD_KEEP_HALF_RAD = 1.05

# Below this, a head-sector return is the robot's own chassis/camera-board
# overhang (0.14 m footprint radius + 2 cm) and is still rewritten.
HEAD_KEEP_FLOOR_M = 0.16


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_faces_head(scan_angle,
                     laser_yaw=LASER_YAW_OFFSET,
                     half_width=HEAD_KEEP_HALF_RAD):
    """True when a scan angle points into the kept head sector."""
    return abs(wrap_angle(scan_angle + laser_yaw)) < half_width


def clip_ranges(ranges, angle_min, angle_increment, range_max, min_range):
    """Rewrite body-sector near returns to range_max, keep head-sector ones.

    A return closer than min_range is a chassis echo everywhere EXCEPT the
    head sector, where anything at or above HEAD_KEEP_FLOOR_M is a real
    obstacle about to touch the face and must stay visible.  Invalid
    samples (nan/inf) pass through untouched.
    """
    clipped = []
    angle = float(angle_min)
    increment = float(angle_increment)
    for value in ranges:
        sample = float(value)
        if (not math.isnan(sample) and not math.isinf(sample) and
                sample < min_range and
                not (sample >= HEAD_KEEP_FLOOR_M and
                     angle_faces_head(angle))):
            clipped.append(float(range_max))
        else:
            clipped.append(sample)
        angle += increment
    return clipped
