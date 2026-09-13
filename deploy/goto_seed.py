#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Creep to the AMCL survey seed if the robot is already nearby.

AMCL is localization, not a taxi.  This script does not search the house.
If /amcl_pose is more than MAX_DIST_M from the seed, it exits and tells
the operator to carry the robot to the red start dot.
"""
from __future__ import print_function

import math
import sys

from survey_tour_logic import (GOAL_TOL_M, LIDAR_STOP_M, TURN_STOP_M, TURN_WZ,
                               front_cone_min, goal_reached, steer_command,
                               turn_sweep_min, yaw_from_quat)

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

SEED_X = -1.975
SEED_Y = 0.375
SEED_YAW = -1.5708
MAX_DIST_M = 2.0
TIMEOUT_S = 45.0
YAW_TOL_RAD = 0.35


class GotoSeed(object):
    def __init__(self):
        self.pose = None
        self.safety_stop = True
        self.min_front = 99.0
        self.min_sweep_ccw = 99.0
        self.min_sweep_cw = 99.0
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self._on_pose)
        rospy.Subscriber("/safety_stop", Bool, self._on_safety)
        rospy.Subscriber("/scan", LaserScan, self._on_scan)

    def _on_pose(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(o.z, o.w))

    def _on_safety(self, msg):
        self.safety_stop = bool(msg.data)

    def _on_scan(self, msg):
        args = (msg.ranges, float(msg.angle_min), float(msg.angle_increment),
                float(msg.range_min), float(msg.range_max))
        self.min_front = front_cone_min(*args)
        self.min_sweep_ccw = turn_sweep_min(*(args + (1.0,)))
        self.min_sweep_cw = turn_sweep_min(*(args + (-1.0,)))

    def stop(self):
        self.cmd_pub.publish(Twist())

    def drive(self, vx, wz):
        if self.safety_stop:
            self.stop()
            return False
        if abs(vx) > 0.02 and self.min_front < LIDAR_STOP_M:
            self.stop()
            return False
        if abs(vx) <= 0.02 and abs(wz) > 0.02:
            sweep = self.min_sweep_ccw if wz > 0.0 else self.min_sweep_cw
            if sweep < TURN_STOP_M:
                self.stop()
                return False
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)
        return True


def _yaw_align(yaw, target):
    err = math.atan2(math.sin(target - yaw), math.cos(target - yaw))
    if abs(err) <= YAW_TOL_RAD:
        return 0.0, True
    return (TURN_WZ if err > 0.0 else -TURN_WZ), False


def main():
    rospy.init_node("goto_seed")
    node = GotoSeed()
    deadline = rospy.Time.now() + rospy.Duration(8.0)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        if node.pose is not None and not node.safety_stop:
            break
        rospy.sleep(0.1)
    if node.pose is None:
        rospy.logerr("NO_POSE: AMCL has not published /amcl_pose")
        sys.exit(2)
    px, py, yaw = node.pose
    dist = math.hypot(SEED_X - px, SEED_Y - py)
    rospy.loginfo("amcl (%.3f, %.3f, %.3f) dist_to_seed=%.3f", px, py, yaw, dist)
    if dist <= GOAL_TOL_M:
        wz, aligned = _yaw_align(yaw, SEED_YAW)
        if aligned:
            node.stop()
            rospy.loginfo("ALREADY_AT_SEED")
            return
    if dist > MAX_DIST_M:
        rospy.logerr(
            "TOO_FAR: %.2f m from seed. Carry the robot to the red start "
            "dot. AMCL does not drive across the house.", dist)
        sys.exit(3)
    rospy.loginfo("creeping to seed; joystick will fight this if it is live")
    deadline = rospy.Time.now() + rospy.Duration(TIMEOUT_S)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        if node.pose is None:
            rate.sleep()
            continue
        px, py, yaw = node.pose
        if goal_reached(px, py, SEED_X, SEED_Y):
            wz, aligned = _yaw_align(yaw, SEED_YAW)
            if aligned:
                node.stop()
                rospy.loginfo("AT_SEED x=%.3f y=%.3f yaw=%.3f", px, py, yaw)
                return
            node.drive(0.0, wz)
            rate.sleep()
            continue
        vx, wz, state = steer_command(px, py, yaw, SEED_X, SEED_Y)
        if not node.drive(vx, wz):
            rospy.logwarn_throttle(2.0, "blocked state=%s min_front=%.2f",
                                   state, node.min_front)
        rate.sleep()
    node.stop()
    rospy.logerr("TIMEOUT: did not reach seed")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
