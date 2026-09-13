#!/usr/bin/env python
"""Regression tests for the openloop tour's sign and head-cone logic.

Pure-python (no ROS) so they run on the PC.  These encode the 2026-08-27
field evidence: bag 11 proved +x is forward in the AMCL/odom frame, bag 17
proved 360-degree braking is wrong with a laser mounted at yaw=pi, and
bag 18 proved copying the joystick's negative scale strands the tour.
"""
from __future__ import print_function

import math
import os
import re
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(TESTS_DIR)))
RELEASES = os.path.join(REPO_ROOT, "releases")
if RELEASES not in sys.path:
    sys.path.insert(0, RELEASES)

import survey_tour_logic as logic


class TestAmclDrift(unittest.TestCase):
    def test_distance_toward_goal_does_not_drift(self):
        tracker = logic.DistanceDriftTracker()
        self.assertFalse(any(tracker.observe(distance)
                             for distance in (1.00, 0.95, 0.90, 0.85)))

    def test_bag18_distance_away_from_goal_is_drift(self):
        self.assertTrue(logic.amcl_drift_detected(
            (1.00, 1.10, 1.20, 1.30)))

    def test_non_drive_sample_breaks_the_run(self):
        tracker = logic.DistanceDriftTracker()
        self.assertFalse(tracker.observe(1.00))
        self.assertFalse(tracker.observe(1.10))
        tracker.reset()
        self.assertFalse(tracker.observe(1.20))
        self.assertFalse(tracker.observe(1.30))

    def test_invalid_distance_resets_the_run(self):
        tracker = logic.DistanceDriftTracker()
        self.assertFalse(tracker.observe(1.00))
        self.assertFalse(tracker.observe(1.10))
        self.assertFalse(tracker.observe(float("nan")))
        self.assertFalse(tracker.observe(1.20))
        self.assertFalse(tracker.observe(1.30))


class TestForwardSign(unittest.TestCase):
    def test_forward_sign_is_positive(self):
        # Bag 11: TEB's +x advanced the AMCL pose.  Bag 18: -x drifted away.
        self.assertGreater(logic.forward_sign(), 0.0)
        self.assertGreater(logic.FORWARD_VX, 0.0)

    def test_tour_does_not_copy_joystick_sign(self):
        # joystick.yaml's scale_linear is negative on purpose (inverted pad
        # axis).  The tour must NOT inherit that sign; if someone "aligns"
        # the two again, this test recreates the bag-18 failure report.
        joystick = os.path.join(REPO_ROOT, "src", "jetbot_pro", "config",
                                "teleop", "joystick.yaml")
        scale = None
        with open(joystick) as stream:
            for line in stream:
                text = line.split("#", 1)[0].strip()
                if text.startswith("scale_linear:"):
                    scale = float(text.split(":", 1)[1])
                    break
        self.assertIsNotNone(scale, "scale_linear missing from joystick.yaml")
        self.assertLess(scale, 0.0)
        self.assertGreater(logic.FORWARD_VX * scale, -1.0)  # sanity
        self.assertLess(math.copysign(1.0, logic.FORWARD_VX) *
                        math.copysign(1.0, scale), 0.0)


class TestFrontCone(unittest.TestCase):
    """Laser yaw=pi: robot head is near +-pi in scan angles."""

    def test_head_angles_are_in_cone(self):
        self.assertTrue(logic.angle_is_in_front_cone(math.pi))
        self.assertTrue(logic.angle_is_in_front_cone(-math.pi))
        self.assertTrue(logic.angle_is_in_front_cone(math.pi - 0.5))
        self.assertTrue(logic.angle_is_in_front_cone(-math.pi + 0.5))

    def test_tail_and_side_angles_are_outside_cone(self):
        self.assertFalse(logic.angle_is_in_front_cone(0.0))       # tail
        self.assertFalse(logic.angle_is_in_front_cone(0.3))
        self.assertFalse(logic.angle_is_in_front_cone(math.pi / 2.0))
        self.assertFalse(logic.angle_is_in_front_cone(-math.pi / 2.0))

    @staticmethod
    def _scan(hits, count=360, default=float("inf")):
        """Build a full-circle scan: hits maps angle(rad) -> range(m)."""
        angle_min = -math.pi
        increment = 2.0 * math.pi / count
        ranges = [default] * count
        for angle, value in hits.items():
            index = int(round((angle - angle_min) / increment)) % count
            ranges[index] = value
        return ranges, angle_min, increment

    def test_bag17_sofa_leg_behind_robot_is_ignored(self):
        # The 0.28 m return behind the robot (scan angle ~0) froze bag 17.
        ranges, amin, ainc = self._scan({0.0: 0.28})
        self.assertEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 12.0), 99.0)

    def test_obstacle_at_head_brakes(self):
        ranges, amin, ainc = self._scan({math.pi - 0.05: 0.20, 0.0: 0.10})
        self.assertAlmostEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 12.0), 0.20)

    def test_head_cone_wraps_across_pi(self):
        ranges, amin, ainc = self._scan({-math.pi + 0.10: 0.35})
        self.assertAlmostEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 12.0), 0.35)

    def test_invalid_and_body_returns_are_filtered(self):
        ranges, amin, ainc = self._scan({
            math.pi - 0.1: float("nan"),
            math.pi - 0.2: 0.05,   # below floor: robot body / noise
            math.pi - 0.3: 50.0,   # above cap
        })
        self.assertEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 60.0), 99.0)

    def test_clear_cone_reports_open_space(self):
        ranges, amin, ainc = self._scan({})
        self.assertEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 12.0), 99.0)

    def test_cone_half_width_boundary_is_strictly_exclusive(self):
        # The head cone is abs(offset) < half_width, strictly.  A beam
        # sitting exactly on the boundary must NOT brake; the comparison is
        # tested with laser_yaw=0 so no float error from the +-pi wrap can
        # blur the equality.
        half = logic.FRONT_CONE_HALF_RAD
        self.assertEqual(half, 0.70)
        self.assertFalse(logic.angle_is_in_front_cone(half, laser_yaw=0.0))
        self.assertFalse(logic.angle_is_in_front_cone(-half, laser_yaw=0.0))
        self.assertTrue(logic.angle_is_in_front_cone(
            half - 1e-9, laser_yaw=0.0))
        self.assertTrue(logic.angle_is_in_front_cone(
            -half + 1e-9, laser_yaw=0.0))

    def test_front_cone_min_brakes_inside_but_not_outside_half_width(self):
        # Same boundary through the full scan path, with the head at -pi:
        # a return 0.65 rad off the head brakes, 0.75 rad does not.
        ranges, amin, ainc = self._scan({-math.pi + 0.65: 0.25,
                                         math.pi - 0.75: 0.24})
        self.assertAlmostEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 12.0), 0.25)

    def test_stop_threshold_stays_above_scan_clip_blind_zone(self):
        # Body-sector returns closer than ~min_range_m are rewritten by
        # scan_clip.py, while the kept head sector preserves near obstacles.
        # LIDAR_STOP_M must keep a real braking band
        # ABOVE the blind zone ([clip, stop) is the only distance where the
        # tour can still brake); do not assume sub-clip protection exists.
        clip_path = os.path.join(REPO_ROOT, "src", "jetbot_pro",
                                 "scripts", "scan_clip.py")
        with open(clip_path) as stream:
            source = stream.read()
        match = re.search(r'get_param\("~min_range_m",\s*([0-9.]+)\)', source)
        self.assertIsNotNone(match, "scan_clip.py lost its ~min_range_m "
                                    "default; braking band unverifiable")
        blind = float(match.group(1))
        self.assertAlmostEqual(blind, 0.22)
        self.assertGreater(logic.LIDAR_STOP_M, blind + 0.05)
        # A body-radius return the clip rewrote to range_max is filtered by
        # the cap and must not brake the tour.
        ranges, amin, ainc = self._scan({math.pi - 0.1: 12.0})
        self.assertEqual(
            logic.front_cone_min(ranges, amin, ainc, 0.15, 12.0), 99.0)


class TestTurnSweepGate(unittest.TestCase):
    """In-place turns brake on the shoulder the head sweeps into.

    wz > 0 (CCW/left) sweeps the head toward positive base bearings; the
    gate watches only the strictly-open (0, SHOULDER_HALF_RAD) shoulder on
    that side, mirrored for wz < 0.  Nothing here may ever look behind the
    robot (bag 17).
    """

    def _sweep(self, hits, turn_sign):
        ranges, amin, ainc = TestFrontCone._scan(hits)
        return logic.turn_sweep_min(ranges, amin, ainc, 0.15, 12.0,
                                    turn_sign)

    def test_bag17_sofa_leg_never_gates_turns(self):
        # 0.28 m at scan angle 0 = base bearing +-pi, dead behind.  The
        # old 360-degree gate would have frozen every turn on it.
        self.assertEqual(self._sweep({0.0: 0.28}, 1.0), 99.0)
        self.assertEqual(self._sweep({0.0: 0.28}, -1.0), 99.0)

    def test_left_turn_brakes_on_left_shoulder_only(self):
        # Base bearing +0.5 -> scan angle 0.5 - pi.  A CCW turn sweeps the
        # head into it; a CW turn moves away and must not brake.
        hits = {0.5 - math.pi: 0.25}
        self.assertAlmostEqual(self._sweep(hits, 1.0), 0.25)
        self.assertEqual(self._sweep(hits, -1.0), 99.0)

    def test_right_turn_brakes_on_right_shoulder_only(self):
        # Base bearing -0.5 -> scan angle pi - 0.5.
        hits = {math.pi - 0.5: 0.24}
        self.assertAlmostEqual(self._sweep(hits, -1.0), 0.24)
        self.assertEqual(self._sweep(hits, 1.0), 99.0)

    def test_zero_turn_sign_reports_clear(self):
        self.assertEqual(self._sweep({0.5 - math.pi: 0.25}, 0.0), 99.0)

    def test_shoulder_obstacle_below_threshold_would_stop_a_turn(self):
        self.assertLess(self._sweep({0.5 - math.pi: 0.25}, 1.0),
                        logic.TURN_STOP_M)

    def test_turn_stop_threshold_ordering(self):
        # Above the 0.22 clip blind zone plus margin (the shoulder shares
        # the head keep sector, so those distances are actually visible),
        # and never stricter than the forward gate.
        self.assertGreater(logic.TURN_STOP_M, 0.22 + 0.05)
        self.assertLessEqual(logic.TURN_STOP_M, logic.LIDAR_STOP_M)
        self.assertGreater(logic.SHOULDER_HALF_RAD,
                           logic.FRONT_CONE_HALF_RAD)


class TestSteering(unittest.TestCase):
    def test_tour_wires_drift_to_fail_close(self):
        tour_path = os.path.join(RELEASES, "survey_tour_20260827.py")
        with open(tour_path) as stream:
            source = stream.read()
        self.assertIn("amcl-drift", source)
        self.assertIn('if state == "drive" and vx > 0.0:', source)
        self.assertIn('return "drift"', source)
        self.assertIn("raise RuntimeError(\"amcl-drift", source)
        drift_at = source.index('raise RuntimeError(\"amcl-drift')
        handler_end = source.index("        if result:", drift_at)
        self.assertNotIn("tour.complete()", source[drift_at:handler_end])
        self.assertLess(source.index("_abort_recorder", drift_at),
                        source.index('if __name__'))

    def test_goal_ahead_drives_forward(self):
        vx, wz, state = logic.steer_command(0.0, 0.0, 0.0, 1.0, 0.0)
        self.assertEqual(state, "drive")
        self.assertGreater(vx, 0.0)
        self.assertAlmostEqual(wz, 0.0)

    def test_goal_behind_turns_in_place(self):
        vx, wz, state = logic.steer_command(0.0, 0.0, 0.0, -1.0, 0.0)
        self.assertEqual(state, "turn")
        self.assertEqual(vx, 0.0)
        self.assertEqual(abs(wz), logic.TURN_WZ)

    def test_turn_direction_matches_shortest_heading_error(self):
        _, wz_left, _ = logic.steer_command(0.0, 0.0, 0.0, 0.0, 1.0)
        _, wz_right, _ = logic.steer_command(0.0, 0.0, 0.0, 0.0, -1.0)
        self.assertGreater(wz_left, 0.0)
        self.assertLess(wz_right, 0.0)

    def test_goal_reached_tolerance(self):
        self.assertTrue(logic.goal_reached(0.0, 0.0, 0.2, 0.1))
        self.assertFalse(logic.goal_reached(0.0, 0.0, 0.3, 0.3))

    def test_unicycle_sim_converges_to_waypoint(self):
        """End-to-end sign check: +x forward kinematics must reach goals.

        If FORWARD_SIGN regresses to the bag-18 value, this simulated
        robot drives away from the waypoint and the test fails.
        """
        px, py, yaw = -1.975, 0.375, -1.57
        goals = ((-1.975, -0.425), (-2.775, -0.425))
        dt = 0.1
        for gx, gy in goals:
            for _ in range(2000):
                if logic.goal_reached(px, py, gx, gy):
                    break
                vx, wz, _ = logic.steer_command(px, py, yaw, gx, gy)
                # REP-103 unicycle: +linear.x moves along the heading.
                px += vx * math.cos(yaw) * dt
                py += vx * math.sin(yaw) * dt
                yaw = logic.wrap_angle(yaw + wz * dt)
            self.assertTrue(
                logic.goal_reached(px, py, gx, gy),
                "sim never reached (%.2f, %.2f); forward sign wrong?" %
                (gx, gy))


class TestWrap(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(logic.wrap_angle(3.0 * math.pi), math.pi)
        self.assertAlmostEqual(abs(logic.wrap_angle(-3.0 * math.pi)),
                               math.pi, places=6)
        self.assertAlmostEqual(logic.wrap_angle(0.5), 0.5)

    def test_yaw_from_quat_round_trip(self):
        for yaw in (-3.0, -1.57, 0.0, 0.7, 2.5):
            z, w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            self.assertAlmostEqual(logic.yaw_from_quat(z, w), yaw, places=6)


if __name__ == "__main__":
    unittest.main()
