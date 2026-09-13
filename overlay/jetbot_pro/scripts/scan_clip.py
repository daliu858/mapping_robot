#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Drop rplidar hits on the robot body before AMCL/costmap see them.

Sector-aware since round 4: near returns are rewritten to range_max only in
the BODY sectors.  The head sector (laser yaw=pi, so scan angles near +-pi)
keeps them, otherwise a face-on obstacle inside ~min_range_m is invisible
to AMCL, the costmaps and the tour's braking cone.  The decision logic
lives in scan_clip_core.py (pure python, PC-tested by
tests/test_scan_clip_sectors.py); this file is only the rospy shell.
scan_clip_core.py must be deployed next to this script in BOTH catkin
trees (src/scripts and devel/lib) or the roslaunch-executed devel copy
dies on import.
"""
from __future__ import print_function

import rospy
from sensor_msgs.msg import LaserScan

from scan_clip_core import clip_ranges


def clip_scan(msg, min_range):
    clipped = LaserScan()
    clipped.header = msg.header
    clipped.angle_min = msg.angle_min
    clipped.angle_max = msg.angle_max
    clipped.angle_increment = msg.angle_increment
    clipped.time_increment = msg.time_increment
    clipped.scan_time = msg.scan_time
    clipped.range_min = msg.range_min
    clipped.range_max = msg.range_max
    clipped.intensities = msg.intensities
    clipped.ranges = clip_ranges(
        msg.ranges, float(msg.angle_min), float(msg.angle_increment),
        float(msg.range_max), float(min_range))
    return clipped


def main():
    rospy.init_node("scan_clip")
    min_range = float(rospy.get_param("~min_range_m", 0.22))
    publisher = rospy.Publisher("scan", LaserScan, queue_size=1)
    rospy.Subscriber(
        "scan_raw", LaserScan,
        lambda msg: publisher.publish(clip_scan(msg, min_range)),
        queue_size=1)
    rospy.spin()


if __name__ == "__main__":
    main()
