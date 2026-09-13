#!/usr/bin/env python
from __future__ import print_function
import math
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

state = {"pose": None, "scan": None, "stop": None, "map": None}

def pose_cb(msg):
    state["pose"] = msg

def scan_cb(msg):
    state["scan"] = msg

def stop_cb(msg):
    state["stop"] = msg

def map_cb(msg):
    state["map"] = msg

rospy.init_node("amcl_diag", anonymous=True)
rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, pose_cb, queue_size=1)
rospy.Subscriber("/scan", LaserScan, scan_cb, queue_size=1)
rospy.Subscriber("/safety_stop", Bool, stop_cb, queue_size=1)
rospy.Subscriber("/map", OccupancyGrid, map_cb, queue_size=1)
rospy.sleep(3.0)
pose = state["pose"]
scan = state["scan"]
stop = state["stop"]
grid = state["map"]
print("safety_stop", None if stop is None else stop.data)
print("map_received", grid is not None, "w" if grid is None else grid.info.width, "h" if grid is None else grid.info.height)
if scan is None:
    print("scan NONE")
else:
    finite = [r for r in scan.ranges if r == r and scan.range_min < r < scan.range_max]
    print("scan_n", len(scan.ranges), "finite", len(finite), "stamp", scan.header.stamp.to_sec())
if pose is None:
    print("amcl_pose NONE")
else:
    cov = pose.pose.covariance
    xy = math.sqrt(max(float(cov[0]), float(cov[7])))
    yaw = math.sqrt(max(0.0, float(cov[35])))
    p = pose.pose.pose.position
    print("amcl_frame", pose.header.frame_id)
    print("amcl_stamp", pose.header.stamp.to_sec(), "now", rospy.Time.now().to_sec())
    print("amcl_xy", round(p.x, 3), round(p.y, 3), "xy_std", round(xy, 3), "yaw_std", round(yaw, 3))
    print("amcl_good", xy <= 0.75 and yaw <= 0.40)
