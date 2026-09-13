#!/usr/bin/env python
"""Falsify the sector-aware scan clip (round 4).

The old scan_clip.py rewrote EVERY return closer than ~min_range_m (0.22)
to range_max regardless of direction.  That hid the chassis posts but also
made a face-on obstacle at 0.19 m invisible to AMCL, the costmaps and the
tour's braking cone.  These tests pin the new head-sector keep rule and
the deployment registration that ships the pure module with the shell.
"""
from __future__ import print_function

import math
import os
import sys
import unittest
import xml.etree.ElementTree as ET

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
SCRIPTS = os.path.join(PKG_ROOT, "scripts")
RELEASES = os.path.join(REPO_ROOT, "releases")
for path in (SCRIPTS, RELEASES):
    if path not in sys.path:
        sys.path.insert(0, path)

import scan_clip_core as core  # noqa: E402
import survey_tour_logic as logic  # noqa: E402

MIN_RANGE_M = 0.22  # lidar.launch ~min_range_m; locked below against XML
RANGE_MAX = 12.0


def _scan(hits, count=360, default=6.0):
    """Build (ranges, angle_min, angle_increment) with hits at scan angles."""
    angle_min = -math.pi
    increment = 2.0 * math.pi / count
    ranges = [default] * count
    for angle, distance in hits.items():
        index = int(round((angle - angle_min) / increment)) % count
        ranges[index] = distance
    return ranges, angle_min, increment


def _clip(ranges, angle_min, increment):
    return core.clip_ranges(ranges, angle_min, increment,
                            RANGE_MAX, MIN_RANGE_M)


class TestSectorClip(unittest.TestCase):
    def test_rear_chassis_echo_is_still_clipped(self):
        # Scan angle 0 is the robot TAIL (laser yaw pi).  A 0.10 m return
        # there is the chassis and must keep vanishing.
        ranges, angle_min, inc = _scan({0.0: 0.10})
        clipped = _clip(ranges, angle_min, inc)
        self.assertEqual(clipped[ranges.index(0.10)], RANGE_MAX)

    def test_side_near_return_is_still_clipped(self):
        ranges, angle_min, inc = _scan({math.pi / 2.0: 0.15})
        clipped = _clip(ranges, angle_min, inc)
        self.assertEqual(clipped[ranges.index(0.15)], RANGE_MAX)

    def test_face_on_obstacle_survives_the_clip(self):
        # 0.19 m at scan angle pi = dead ahead.  The old clip erased it.
        ranges, angle_min, inc = _scan({math.pi: 0.19})
        clipped = _clip(ranges, angle_min, inc)
        self.assertEqual(clipped[ranges.index(0.19)], 0.19)

    def test_head_sector_self_echo_below_floor_is_clipped(self):
        # Below HEAD_KEEP_FLOOR_M even a head-sector return is the robot's
        # own overhang, not an obstacle.
        ranges, angle_min, inc = _scan({math.pi: 0.10})
        clipped = _clip(ranges, angle_min, inc)
        self.assertEqual(clipped[ranges.index(0.10)], RANGE_MAX)

    def test_keep_sector_boundary_is_strictly_exclusive(self):
        # Direct calls: the 360-beam helper quantizes angles and would
        # blur the 1.05 boundary (1.05/increment rounds to a beam at
        # 1.0472 rad).  Float note: pi - 1.05 is not exactly representable,
        # so feeding it through the default laser_yaw=pi lands 1 ulp
        # INSIDE the boundary and abs() < half comes out True.  Test the
        # strict comparison in the base-bearing domain (laser_yaw=0)
        # where the boundary value is exact, like test_survey_tour_logic
        # does for the 0.70 cone.
        half = core.HEAD_KEEP_HALF_RAD
        self.assertEqual(half, 1.05)
        eps = 1e-9
        self.assertTrue(core.angle_faces_head(half - eps, laser_yaw=0.0))
        self.assertFalse(core.angle_faces_head(half, laser_yaw=0.0))
        self.assertTrue(core.angle_faces_head(-half + eps, laser_yaw=0.0))
        self.assertFalse(core.angle_faces_head(-half, laser_yaw=0.0))
        # The real mapping through the default laser_yaw stays right: the
        # head direction (pi) faces head, the tail (0) does not.
        self.assertTrue(core.angle_faces_head(math.pi))
        self.assertFalse(core.angle_faces_head(0.0))

    def test_invalid_samples_pass_through_untouched(self):
        ranges, angle_min, inc = _scan(
            {0.0: float("nan"), math.pi: float("inf")})
        clipped = _clip(ranges, angle_min, inc)
        self.assertTrue(math.isnan(clipped[
            int(round((0.0 - angle_min) / inc)) % len(ranges)]))
        self.assertTrue(math.isinf(clipped[
            int(round((math.pi - angle_min) / inc)) % len(ranges)]))

    def test_far_returns_are_untouched_everywhere(self):
        ranges, angle_min, inc = _scan({}, default=6.0)
        self.assertEqual(_clip(ranges, angle_min, inc), ranges)


class TestClipToBrakingChain(unittest.TestCase):
    def test_face_on_obstacle_now_reaches_the_braking_cone(self):
        # End-to-end falsification of the round-3 blind spot: clip a scan
        # with a 0.19 m face-on obstacle, then feed the result to the
        # tour's front_cone_min.  Old behaviour: 99.0 (no brake).
        ranges, angle_min, inc = _scan({math.pi: 0.19})
        clipped = _clip(ranges, angle_min, inc)
        nearest = logic.front_cone_min(clipped, angle_min, inc,
                                       0.05, RANGE_MAX)
        self.assertAlmostEqual(nearest, 0.19)
        self.assertLess(nearest, logic.LIDAR_STOP_M)

    def test_rear_sofa_leg_still_never_brakes_forward_drive(self):
        # Bag 17 regression through the full chain: 0.28 m behind.
        # Open space in the head cone is modeled as inf, so the ONLY valid
        # return the cone can see is the rear leg; a real default of 6.0
        # would itself be the nearest cone return.
        ranges, angle_min, inc = _scan({0.0: 0.28}, default=float("inf"))
        clipped = _clip(ranges, angle_min, inc)
        nearest = logic.front_cone_min(clipped, angle_min, inc,
                                       0.05, RANGE_MAX)
        self.assertEqual(nearest, 99.0)


class TestConstantOrdering(unittest.TestCase):
    def test_keep_floor_sits_between_footprint_and_blind_zone(self):
        self.assertGreaterEqual(core.HEAD_KEEP_FLOOR_M, 0.14)
        self.assertLess(core.HEAD_KEEP_FLOOR_M, MIN_RANGE_M)

    def test_keep_sector_covers_every_gating_sector(self):
        # If a gating sector were wider than the keep sector, the gate
        # would ask for distances the clip already rewrote to range_max.
        self.assertLessEqual(logic.FRONT_CONE_HALF_RAD,
                             core.HEAD_KEEP_HALF_RAD)
        self.assertLessEqual(logic.SHOULDER_HALF_RAD,
                             core.HEAD_KEEP_HALF_RAD)

    def test_clip_and_tour_agree_on_the_laser_yaw(self):
        self.assertEqual(core.LASER_YAW_OFFSET, logic.LASER_YAW_OFFSET)


class TestDeploymentRegistration(unittest.TestCase):
    def test_lidar_launch_min_range_matches_this_suite(self):
        path = os.path.join(PKG_ROOT, "launch", "lidar.launch")
        root = ET.parse(path).getroot()
        node = next(n for n in root.findall("node")
                    if n.get("name") == "scan_clip")
        param = next(p for p in node.findall("param")
                     if p.get("name") == "min_range_m")
        self.assertEqual(float(param.get("value")), MIN_RANGE_M)

    def test_core_module_is_registered_for_catkin_install(self):
        with open(os.path.join(PKG_ROOT, "CMakeLists.txt"), "r") as stream:
            cmake = stream.read()
        self.assertIn("scripts/scan_clip_core.py", cmake)

    def test_entry_script_deploys_the_core_module_to_both_trees(self):
        path = os.path.join(RELEASES, "start_nav_survey.sh")
        with open(path, "r") as stream:
            start = stream.read()
        self.assertIn("scan_clip_core.py", start)
        self.assertIn("map_identity.py", start)
        self.assertIn("replay_contract.py", start)
        self.assertIn('cp "/tmp/$py" '
                      '"/home/jetbot/catkin_ws/src/jetbot_pro/scripts/$py"',
                      start)
        self.assertIn('cp "/tmp/$py" '
                      '"/home/jetbot/catkin_ws/devel/lib/jetbot_pro/$py"',
                      start)


if __name__ == "__main__":
    unittest.main()
