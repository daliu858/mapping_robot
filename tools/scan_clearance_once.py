#!/usr/bin/env python
from __future__ import print_function

import math

import rospy
from sensor_msgs.msg import LaserScan


def normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


rospy.init_node("scan_clearance_once", anonymous=True)
scan = rospy.wait_for_message("/scan", LaserScan, 3.0)
valid = [
    (float(value), scan.angle_min + index * scan.angle_increment)
    for index, value in enumerate(scan.ranges)
    if (not math.isnan(float(value)) and not math.isinf(float(value)) and
        scan.range_min <= float(value) <= scan.range_max)
]
distance, angle = min(valid)
print("min_m=%.3f angle_deg=%.1f" %
      (distance, angle * 180.0 / math.pi))
for name, center in (("front", 0.0), ("left", math.pi / 2.0),
                     ("back", math.pi), ("right", -math.pi / 2.0)):
    quadrant = min(
        value for value, ray_angle in valid
        if abs(normalize(ray_angle - center)) < math.pi / 4.0)
    print("%s=%.3f" % (name, quadrant))
