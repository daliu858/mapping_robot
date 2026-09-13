#!/usr/bin/env python
from __future__ import print_function

import math
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


# Keep the pure motion-safety helpers testable on the PC, where ROS Melodic is
# intentionally not installed.  Live ROS integration is exercised on Jetson.
class _Vector(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Twist(object):
    def __init__(self):
        self.linear = _Vector()
        self.angular = _Vector()


def _stub_ros_message_package(package, names):
    module = types.ModuleType(package)
    message_module = types.ModuleType(package + ".msg")
    for name in names:
        setattr(message_module, name,
                _Twist if name == "Twist" else type(name, (object,), {}))
    module.msg = message_module
    sys.modules.setdefault(package, module)
    sys.modules.setdefault(package + ".msg", message_module)


sys.modules.setdefault("rospy", types.ModuleType("rospy"))
_stub_ros_message_package("geometry_msgs", ("Twist",))
_stub_ros_message_package("nav_msgs", ("OccupancyGrid", "Odometry"))
_stub_ros_message_package("sensor_msgs", ("LaserScan",))
_stub_ros_message_package("std_msgs", ("Bool",))

import bounded_motion_probe as probe


class Obj(object):
    def __init__(self, **values):
        self.__dict__.update(values)


class State(object):
    def __init__(self, safety=False, scan=None, scan_wall=10.0,
                 odom=None, odom_wall=10.0, map_msg=None):
        self.values = (safety, scan, scan_wall,
                       Obj() if odom is None else odom, odom_wall, map_msg)

    def snapshot(self):
        return self.values


def scan_with(ranges):
    return Obj(ranges=list(ranges), range_min=0.15, range_max=12.0,
               angle_min=-math.pi,
               angle_increment=(2.0 * math.pi) / (len(ranges) - 1))


class BoundedMotionProbeTest(unittest.TestCase):
    def test_clearance_ignores_invalid_and_out_of_range_returns(self):
        scan = scan_with([0.0, float("nan"), 0.70, float("inf"), 0.60,
                          13.0, 0.80, -1.0, 0.75])
        result = probe.clearance_summary(scan)
        self.assertEqual(result["valid"], 4)
        self.assertAlmostEqual(result["minimum_all"], 0.60)
        self.assertAlmostEqual(result["minimum_longitudinal"], 0.60)

    def test_clearance_rejects_scan_without_usable_longitudinal_data(self):
        with self.assertRaises(RuntimeError):
            probe.clearance_summary(scan_with([float("inf")] * 9))

    def test_ready_requires_explicit_released_safety(self):
        scan = scan_with([1.0] * 9)
        for safety in (None, True):
            with self.assertRaises(RuntimeError):
                probe.require_ready(State(safety=safety, scan=scan),
                                    0.5, 0.45, 0.28, now=10.0)
        odom, _map, clearance = probe.require_ready(
            State(safety=False, scan=scan), 0.5, 0.45, 0.28, now=10.0)
        self.assertIsNotNone(odom)
        self.assertEqual(clearance["valid"], 9)

    def test_stale_boundary_is_inclusive_but_later_data_fails(self):
        scan = scan_with([1.0] * 9)
        state = State(scan=scan, scan_wall=9.5, odom_wall=9.5)
        probe.require_ready(state, 0.5, 0.45, 0.28, now=10.0)
        with self.assertRaises(RuntimeError):
            probe.require_ready(state, 0.5, 0.45, 0.28, now=10.000001)

    def test_each_clearance_gate_fails_closed(self):
        longitudinal_blocked = scan_with(
            [1.0, 1.0, 1.0, 1.0, 0.30, 1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(RuntimeError):
            probe.require_ready(State(scan=longitudinal_blocked),
                                0.5, 0.45, 0.28, now=10.0)
        rotation_blocked = scan_with(
            [1.0, 1.0, 0.20, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(RuntimeError):
            probe.require_ready(State(scan=rotation_blocked),
                                0.5, 0.45, 0.28, now=10.0)

    def test_stop_attempts_continue_after_publish_errors(self):
        class Publisher(object):
            def __init__(self):
                self.calls = 0

            def publish(self, _msg):
                self.calls += 1
                if self.calls in (2, 7):
                    raise RuntimeError("closed once")

        publisher = Publisher()
        original_sleep = probe.time.sleep
        probe.time.sleep = lambda _duration: None
        try:
            self.assertEqual(probe.stop_robot(publisher), 10)
        finally:
            probe.time.sleep = original_sleep
        self.assertEqual(publisher.calls, 12)

    def test_arguments_reject_nonfinite_or_unsafe_values(self):
        args = probe.parse_args([])
        probe.validate_args(args)
        args.turn_speed = float("nan")
        with self.assertRaises(ValueError):
            probe.validate_args(args)
        args = probe.parse_args([])
        args.linear_distance = -0.01
        with self.assertRaises(ValueError):
            probe.validate_args(args)

    def test_clockwise_turn_speed_is_allowed_but_zero_is_rejected(self):
        args = probe.parse_args([])
        args.turn_speed = -0.18
        probe.validate_args(args)
        args.turn_speed = 0.0
        with self.assertRaises(ValueError):
            probe.validate_args(args)

    def test_reverse_uses_backward_clearance(self):
        scan = scan_with([0.25, 1.0, 1.0, 1.0, 0.90,
                          1.0, 1.0, 1.0, 0.25])
        clearance = probe.clearance_summary(scan)
        self.assertAlmostEqual(clearance["minimum_forward"], 0.25)
        self.assertAlmostEqual(clearance["minimum_backward"], 0.90)
        probe.require_ready(State(scan=scan), 0.5, 0.45, 0.20,
                            now=10.0, linear_speed=-0.08)
        with self.assertRaises(RuntimeError):
            probe.require_ready(State(scan=scan), 0.5, 0.45, 0.20,
                                now=10.0, linear_speed=0.08)


if __name__ == "__main__":
    unittest.main()
