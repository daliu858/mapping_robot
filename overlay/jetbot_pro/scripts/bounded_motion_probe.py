#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one tiny, fail-stop motion probe on a live JetBot.

The script is deliberately conservative: it requires fresh LiDAR, odometry,
and a released /safety_stop before publishing any non-zero /cmd_vel.  It also
checks both longitudinal directions because the physical motor sign may not
yet have been calibrated on a newly assembled robot.
"""
from __future__ import print_function

import argparse
import json
import math
import signal
import sys
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class OperatorStop(RuntimeError):
    pass


def finite(value):
    return not math.isnan(float(value)) and not math.isinf(float(value))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_delta(current, start):
    return math.atan2(math.sin(current - start), math.cos(current - start))


def angle_distance(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def clearance_summary(scan, longitudinal_half_angle=0.70,
                      laser_yaw_in_base=math.pi):
    """Return conservative clearances in metres for a 360-degree scan."""
    all_ranges = []
    longitudinal = []
    forward = []
    backward = []
    for index, raw in enumerate(scan.ranges):
        value = float(raw)
        if (not finite(value) or value < float(scan.range_min) or
                value > float(scan.range_max)):
            continue
        angle = float(scan.angle_min) + index * float(scan.angle_increment)
        base_angle = angle + float(laser_yaw_in_base)
        all_ranges.append(value)
        # Check both scan-frame +X and -X.  This intentionally avoids relying
        # on the current base_link->laser yaw or unverified motor polarity.
        if (angle_distance(angle, 0.0) <= longitudinal_half_angle or
                angle_distance(angle, math.pi) <= longitudinal_half_angle):
            longitudinal.append(value)
        if angle_distance(base_angle, 0.0) <= longitudinal_half_angle:
            forward.append(value)
        if angle_distance(base_angle, math.pi) <= longitudinal_half_angle:
            backward.append(value)
    if not all_ranges or not longitudinal or not forward or not backward:
        raise RuntimeError("LiDAR scan has no usable clearance samples")
    return {
        "valid": len(all_ranges),
        "minimum_all": min(all_ranges),
        "minimum_longitudinal": min(longitudinal),
        "minimum_forward": min(forward),
        "minimum_backward": min(backward),
    }


class LiveState(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.safety = None
        self.scan = None
        self.odom = None
        self.map_msg = None
        self.scan_wall = None
        self.odom_wall = None
        rospy.Subscriber("/safety_stop", Bool, self.on_safety, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber("/odom_raw", Odometry, self.on_odom, queue_size=2)
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1)

    def on_safety(self, msg):
        with self.lock:
            self.safety = bool(msg.data)

    def on_scan(self, msg):
        with self.lock:
            self.scan = msg
            self.scan_wall = time.time()

    def on_odom(self, msg):
        with self.lock:
            self.odom = msg
            self.odom_wall = time.time()

    def on_map(self, msg):
        with self.lock:
            self.map_msg = msg

    def snapshot(self):
        with self.lock:
            return (self.safety, self.scan, self.scan_wall,
                    self.odom, self.odom_wall, self.map_msg)


def pose_from_odom(msg):
    pose = msg.pose.pose
    return (float(pose.position.x), float(pose.position.y),
            yaw_from_quaternion(pose.orientation))


def map_counts(msg):
    if msg is None:
        return None
    occupied = sum(1 for value in msg.data if int(value) >= 50)
    free = sum(1 for value in msg.data if 0 <= int(value) < 50)
    return {"width": int(msg.info.width), "height": int(msg.info.height),
            "free": free, "occupied": occupied, "known": free + occupied}


def emit(event, **fields):
    payload = {"event": event}
    payload.update(fields)
    print(json.dumps(payload, sort_keys=True))
    sys.stdout.flush()


def stop_robot(pub):
    """Best-effort repeated stop; one closed-publisher error cannot skip it."""
    zero = Twist()
    published = 0
    for _index in range(12):
        try:
            pub.publish(zero)
            published += 1
        except Exception:
            # The base also has a one-second command timeout.  Continue all
            # attempts in case the publisher/master recovers during cleanup.
            pass
        time.sleep(0.025)
    return published


def require_ready(state, stale_s, longitudinal_clearance,
                  rotation_clearance, now=None, linear_speed=None):
    safety, scan, scan_wall, odom, odom_wall, map_msg = state.snapshot()
    now = time.time() if now is None else float(now)
    if safety is None:
        raise RuntimeError("no /safety_stop received")
    if safety:
        raise RuntimeError("/safety_stop is active")
    if scan is None or scan_wall is None or now - scan_wall > stale_s:
        raise RuntimeError("/scan is missing or stale")
    if odom is None or odom_wall is None or now - odom_wall > stale_s:
        raise RuntimeError("/odom_raw is missing or stale")
    clearance = clearance_summary(scan)
    if linear_speed is None:
        directional_key = "minimum_longitudinal"
    elif float(linear_speed) > 0.0:
        directional_key = "minimum_forward"
    else:
        directional_key = "minimum_backward"
    if clearance[directional_key] < longitudinal_clearance:
        raise RuntimeError(
            "%s clearance %.3fm is below %.3fm" %
            (directional_key.replace("minimum_", ""),
             clearance[directional_key], longitudinal_clearance))
    if clearance["minimum_all"] < rotation_clearance:
        raise RuntimeError(
            "all-around clearance %.3fm is below %.3fm" %
            (clearance["minimum_all"], rotation_clearance))
    return odom, map_msg, clearance


def command_until(state, pub, command, stop_condition, timeout_s,
                  stale_s, longitudinal_clearance, rotation_clearance,
                  phase, stop_event, linear_speed=None):
    started = time.time()
    rate = rospy.Rate(20)
    samples = 0
    while True:
        if stop_event.is_set():
            raise OperatorStop("operator requested stop")
        if rospy.is_shutdown():
            raise OperatorStop("ROS shutdown requested")
        odom, _map_msg, clearance = require_ready(
            state, stale_s, longitudinal_clearance, rotation_clearance,
            linear_speed=linear_speed)
        samples += 1
        if stop_condition(odom):
            emit("phase_complete", phase=phase, samples=samples,
                 elapsed_s=round(time.time() - started, 3),
                 clearance=clearance)
            return
        if time.time() - started >= timeout_s:
            raise RuntimeError("%s phase timed out" % phase)
        pub.publish(command)
        rate.sleep()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="publish the bounded non-zero commands")
    parser.add_argument("--linear-distance", type=float, default=0.10)
    parser.add_argument("--linear-stop-margin", type=float, default=0.015)
    parser.add_argument("--linear-speed", type=float, default=0.08)
    parser.add_argument("--turn-angle", type=float, default=0.28)
    parser.add_argument("--turn-speed", type=float, default=0.18)
    parser.add_argument("--linear-timeout", type=float, default=2.5)
    parser.add_argument("--turn-timeout", type=float, default=2.5)
    parser.add_argument("--stale", type=float, default=0.50)
    parser.add_argument("--longitudinal-clearance", type=float, default=0.45)
    parser.add_argument("--rotation-clearance", type=float, default=0.28)
    return parser.parse_args(argv)


def validate_args(args):
    nonnegative = (("linear-distance", args.linear_distance),
                   ("linear-stop-margin", args.linear_stop_margin),
                   ("turn-angle", args.turn_angle))
    positive = (("linear-timeout", args.linear_timeout),
                ("turn-timeout", args.turn_timeout),
                ("stale", args.stale),
                ("longitudinal-clearance", args.longitudinal_clearance),
                ("rotation-clearance", args.rotation_clearance))
    for name, value in nonnegative:
        if not finite(value) or value < 0.0:
            raise ValueError("--%s must be finite and nonnegative" % name)
    for name, value in positive:
        if not finite(value) or value <= 0.0:
            raise ValueError("--%s must be finite and positive" % name)
    if not finite(args.turn_speed) or args.turn_speed == 0.0:
        raise ValueError("--turn-speed must be finite and non-zero")
    if not finite(args.linear_speed) or args.linear_speed == 0.0:
        raise ValueError("--linear-speed must be finite and non-zero")


def install_stop_handlers(stop_event):
    def request_stop(_signum, _frame):
        stop_event.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    validate_args(args)
    # Keep rospy alive during SIGINT/SIGTERM cleanup so the repeated zero
    # commands are published before shutdown closes the topic.
    rospy.init_node("bounded_motion_probe", anonymous=True,
                    disable_signals=True)
    stop_event = threading.Event()
    install_stop_handlers(stop_event)
    state = LiveState()
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    try:
        deadline = time.time() + 6.0
        ready = None
        while (time.time() < deadline and not rospy.is_shutdown() and
               not stop_event.is_set()):
            try:
                ready = require_ready(
                    state, args.stale, args.longitudinal_clearance,
                    args.rotation_clearance,
                    linear_speed=(args.linear_speed
                                  if args.linear_distance > 0.0 else None))
                break
            except RuntimeError:
                time.sleep(0.10)
        if ready is None:
            if stop_event.is_set() or rospy.is_shutdown():
                raise OperatorStop("operator or ROS requested stop")
            # Re-run once so the precise final failure reaches the operator.
            ready = require_ready(
                state, args.stale, args.longitudinal_clearance,
                args.rotation_clearance,
                linear_speed=(args.linear_speed
                              if args.linear_distance > 0.0 else None))
        odom, map_msg, clearance = ready
        start_x, start_y, start_yaw = pose_from_odom(odom)
        emit("preflight_ok", execute=bool(args.execute), clearance=clearance,
             start_pose=[start_x, start_y, start_yaw],
             map_before=map_counts(map_msg))
        if not args.execute:
            return 0

        if pub.get_num_connections() < 1:
            raise RuntimeError("/cmd_vel has no subscriber")

        linear = Twist()
        linear.linear.x = float(args.linear_speed)
        linear_stop_target = max(
            0.0, args.linear_distance - args.linear_stop_margin)

        def linear_done(current):
            x, y, _yaw = pose_from_odom(current)
            return math.hypot(x - start_x, y - start_y) >= linear_stop_target

        if args.linear_distance > 0.0:
            command_until(
                state, pub, linear, linear_done, args.linear_timeout,
                args.stale, args.longitudinal_clearance,
                args.rotation_clearance, "linear", stop_event,
                linear_speed=args.linear_speed)
            stop_robot(pub)
            time.sleep(0.50)

        _safety, _scan, _scan_wall, odom, _odom_wall, _map = state.snapshot()
        turn_start = pose_from_odom(odom)[2]
        turn = Twist()
        turn.angular.z = float(args.turn_speed)

        def turn_done(current):
            return abs(angle_delta(pose_from_odom(current)[2], turn_start)) >= \
                args.turn_angle

        if args.turn_angle > 0.0:
            command_until(
                state, pub, turn, turn_done, args.turn_timeout, args.stale,
                args.longitudinal_clearance, args.rotation_clearance, "turn",
                stop_event)
            stop_robot(pub)
            time.sleep(1.0)

        _safety, _scan, _scan_wall, final_odom, _odom_wall, final_map = \
            state.snapshot()
        final_x, final_y, final_yaw = pose_from_odom(final_odom)
        emit("complete", final_pose=[final_x, final_y, final_yaw],
             translation_m=math.hypot(final_x - start_x, final_y - start_y),
             yaw_change_rad=angle_delta(final_yaw, start_yaw),
             map_after=map_counts(final_map))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        emit("abort", error=str(error))
        return 2
    finally:
        stop_robot(pub)
        rospy.signal_shutdown("bounded motion probe finished")


if __name__ == "__main__":
    sys.exit(main())
