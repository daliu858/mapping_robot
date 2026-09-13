#!/usr/bin/env python
from __future__ import print_function
import math
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

rospy.init_node("amcl_pose_once", anonymous=True)
print("update_min_d", rospy.get_param("/amcl/update_min_d", "MISSING"))
print("update_min_a", rospy.get_param("/amcl/update_min_a", "MISSING"))
msg = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=8.0)
cov = msg.pose.covariance
xy = math.sqrt(max(float(cov[0]), float(cov[7])))
yaw = math.sqrt(max(0.0, float(cov[35])))
now = rospy.Time.now().to_sec()
stamp = msg.header.stamp.to_sec()
print("frame", msg.header.frame_id)
print("stamp", stamp, "now", now, "age", now - stamp)
print("xy_std", round(xy, 3), "yaw_std", round(yaw, 3))
print("pos", round(msg.pose.pose.position.x, 3), round(msg.pose.pose.position.y, 3))
