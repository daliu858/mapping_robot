#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-stop a live robot launch when its safety LiDAR stream goes stale."""
from __future__ import print_function

import math
import sys
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


def scan_stamp_key(stamp):
    """Return a comparable, non-zero ROS stamp or ``None`` when invalid."""
    try:
        secs = int(stamp.secs)
        nsecs = int(stamp.nsecs)
    except (AttributeError, TypeError, ValueError):
        return None
    if secs < 0 or nsecs < 0 or nsecs >= 1000000000:
        return None
    if secs == 0 and nsecs == 0:
        return None
    return secs, nsecs


def scan_stamp_advances(previous, current):
    """A repeated/backwards scan must not refresh the safety watchdog."""
    return current is not None and (previous is None or current > previous)


class ScanWatchdog(object):
    def __init__(self):
        self.startup_grace = float(rospy.get_param("~startup_grace_s", 10.0))
        self.stale_timeout = float(rospy.get_param("~stale_timeout_s", 1.0))
        self.started = time.time()
        self.last_scan = None
        self.last_scan_stamp = None
        # Once a safety failure is selected it is terminal for this process.
        # In particular, a late /scan callback must never release the latched
        # inhibit while roslaunch is still tearing the stack down.
        self.terminal_failure = None
        self.lock = threading.Lock()
        self.stop_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.inhibit_pub = rospy.Publisher(
            "/safety_stop", Bool, queue_size=1, latch=True)
        self.inhibit_pub.publish(Bool(data=True))
        self.healthy = False
        rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=2)

    def cb_scan(self, msg):
        increment = float(msg.angle_increment)
        stamp = scan_stamp_key(msg.header.stamp)
        valid = sum(1 for value in msg.ranges
                    if (not math.isnan(float(value)) and
                        not math.isinf(float(value)) and
                        float(msg.range_min) <= float(value) <=
                        float(msg.range_max)))
        minimum_valid = max(1, int(math.ceil(len(msg.ranges) * 0.01)))
        if (not msg.ranges or valid < minimum_valid or
                math.isnan(increment) or math.isinf(increment) or
                increment == 0.0):
            return
        with self.lock:
            if self.terminal_failure is not None:
                return
            if not scan_stamp_advances(self.last_scan_stamp, stamp):
                return
            self.last_scan_stamp = stamp
            self.last_scan = time.time()
            if not self.healthy:
                self.healthy = True
                self.inhibit_pub.publish(Bool(data=False))

    def latch_failure_if_due(self, now):
        """Atomically select a terminal failure against incoming scans."""
        with self.lock:
            if self.terminal_failure is not None:
                return self.terminal_failure
            if self.last_scan is None:
                if now - self.started > self.startup_grace:
                    self.terminal_failure = (
                        "no valid /scan within %.1fs" % self.startup_grace)
            elif now - self.last_scan > self.stale_timeout:
                self.terminal_failure = (
                    "/scan stale for %.2fs" % (now - self.last_scan))
            if self.terminal_failure is not None:
                self.healthy = False
            return self.terminal_failure

    def command_stop(self):
        self.inhibit_pub.publish(Bool(data=True))
        stop = Twist()
        for _index in range(10):
            self.stop_pub.publish(stop)
            time.sleep(0.02)

    def run(self):
        while not rospy.is_shutdown():
            now = time.time()
            failure = self.latch_failure_if_due(now)
            if failure is not None:
                self.command_stop()
                raise RuntimeError(failure)
            time.sleep(0.10)


def main():
    rospy.init_node("scan_watchdog")
    ScanWatchdog().run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("LiDAR safety watchdog stopped the stack: %s", str(error))
        sys.exit(1)
