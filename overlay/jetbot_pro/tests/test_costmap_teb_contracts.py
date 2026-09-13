#!/usr/bin/env python
"""Static contracts for the costmap/TEB parameter stack (round 4).

The 8/27 field runs 01-10/12-16 died in "valid control could not be found"
TEB rejections while the robot sat ~7 cm from a mapped obstacle.  These tests
pin the mechanism-level fixes so a later edit cannot silently reintroduce
them:

* single source of truth -- costmap_common owns only the physical footprint;
  the inflation values that actually win at move_base.launch load order live
  exclusively in the per-costmap files;
* the global costmap stays optimistic enough to plan out of the 7 cm start
  pocket while local/TEB keep the physical 0.14 m footprint;
* TEB's soft obstacle margin and rear obstacle horizon match the measured
  doorway/sofa geometry instead of contradicting it.
"""
from __future__ import print_function

import os
import unittest
import xml.etree.ElementTree as ET

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFF = os.path.join(ROOT, "config", "diff")
REPO = os.path.dirname(os.path.dirname(ROOT))
RELEASES = os.path.join(REPO, "releases")

MAP_RESOLUTION_M = 0.05
START_OBSTACLE_GAP_M = 0.07
DOORWAY_MIN_WIDTH_M = 0.29
SOFA_BEHIND_START_M = 0.28


def load_diff_yaml(name):
    with open(os.path.join(DIFF, name), "r") as stream:
        return yaml.safe_load(stream)


class CostmapSingleSourceTest(unittest.TestCase):
    def test_common_file_owns_only_the_physical_footprint(self):
        common = load_diff_yaml("costmap_common_params.yaml")
        self.assertEqual(common["robot_radius"], 0.14)
        # The 8/27 tree carried a dead inflation_layer block (0.22) here plus
        # a misplaced inflation_radius key inside obstacle_layer.  Both read
        # as authoritative while being overridden (or ignored) at load time;
        # that file-level ambiguity must stay dead.
        self.assertNotIn("inflation_layer", common)
        self.assertNotIn("inflation_radius", common["obstacle_layer"])

    def test_move_base_load_order_lets_per_costmap_files_win(self):
        path = os.path.join(ROOT, "launch", "move_base.launch")
        node = next(n for n in ET.parse(path).getroot().findall("node")
                    if n.get("name") == "move_base")
        entries = [(os.path.basename(p.get("file", "")), p.get("ns"))
                   for p in node.findall("rosparam")]
        # rosparam load is last-writer-wins: the common file must come first
        # in each costmap namespace so the per-costmap inflation values are
        # the runtime truth.
        self.assertLess(
            entries.index(("costmap_common_params.yaml", "global_costmap")),
            entries.index(("global_costmap_params.yaml", None)))
        self.assertLess(
            entries.index(("costmap_common_params.yaml", "local_costmap")),
            entries.index(("local_costmap_params.yaml", None)))

    def test_global_planner_can_leave_the_seven_cm_start_pocket(self):
        common = load_diff_yaml("costmap_common_params.yaml")
        glob = load_diff_yaml("global_costmap_params.yaml")["global_costmap"]
        local = load_diff_yaml("local_costmap_params.yaml")["local_costmap"]
        teb = load_diff_yaml(
            "teb_local_planner_params.yaml")["TebLocalPlannerROS"]
        # costmap_2d marks every cell closer than robot_radius to an obstacle
        # INSCRIBED (253) and global_planner treats 253 as impassable.  The
        # start pose sits 0.07 m from a mapped obstacle on a 0.05 m grid, so
        # the first cell out of the pocket lies 0.07+0.05 m from it.  The
        # global radius must not swallow that cell; the physical 0.14 would.
        self.assertLessEqual(glob["robot_radius"],
                             START_OBSTACLE_GAP_M + MAP_RESOLUTION_M)
        self.assertLess(glob["robot_radius"], common["robot_radius"])
        # Execution stays physical: the local costmap must not override the
        # common 0.14 and TEB's own footprint matches it.
        self.assertNotIn("robot_radius", local)
        self.assertEqual(teb["footprint_model"]["radius"],
                         common["robot_radius"])

    def test_survey_inflation_tolerance_and_local_layers_stay_locked(self):
        glob_doc = load_diff_yaml("global_costmap_params.yaml")
        glob = glob_doc["global_costmap"]
        local = load_diff_yaml("local_costmap_params.yaml")["local_costmap"]
        planner = load_diff_yaml("base_global_planner_param.yaml")
        # Re-inflating the global map "to be safe" reproduces the 8/27
        # instant plan rejections at the start pocket.
        self.assertLessEqual(glob["inflation_layer"]["inflation_radius"],
                             0.10)
        self.assertLessEqual(local["inflation_layer"]["inflation_radius"],
                             0.10)
        self.assertGreaterEqual(
            glob_doc["GlobalPlanner"]["default_tolerance"], 0.50)
        self.assertGreaterEqual(
            planner["GlobalPlanner"]["default_tolerance"], 0.50)
        # The rolling odom-frame window must hold live obstacles only;
        # a copied StaticLayer leaves dragged edges that reject TEB plans.
        plugin_types = [plugin["type"] for plugin in local["plugins"]]
        self.assertNotIn("costmap_2d::StaticLayer", plugin_types)
        self.assertIn("costmap_2d::VoxelLayer", plugin_types)
        self.assertIn("costmap_2d::InflationLayer", plugin_types)


class TebRejectionFixTest(unittest.TestCase):
    def setUp(self):
        self.teb = load_diff_yaml(
            "teb_local_planner_params.yaml")["TebLocalPlannerROS"]

    def test_doorway_soft_margin_matches_measured_geometry(self):
        radius = self.teb["footprint_model"]["radius"]
        # Hard clearance: the 0.28 m chassis fits the measured 0.29 m door.
        self.assertLess(2 * radius, DOORWAY_MIN_WIDTH_M)
        # min_obstacle_dist is a soft g2o penalty margin.  A 0.12 m margin
        # (the old comment's claim) would demand a 2*(0.14+0.12)=0.52 m
        # doorway; 0.05 keeps the door penalty small enough to optimize
        # through while collision checking keeps the real footprint.
        self.assertGreater(self.teb["min_obstacle_dist"], 0.0)
        self.assertLessEqual(self.teb["min_obstacle_dist"], 0.05)
        # Keep the penalty soft rather than a de-facto hard wall.
        self.assertEqual(self.teb["penalty_epsilon"], 0.1)
        self.assertEqual(self.teb["weight_obstacle"], 50)

    def test_rear_obstacle_horizon_excludes_the_bag17_sofa(self):
        # Sofa legs sit ~0.28 m behind the start pose (bag 17 geometry).
        # They must not enter the TEB graph of a forward-drive plan.
        behind = self.teb["costmap_obstacles_behind_robot_dist"]
        self.assertGreater(behind, 0.0)
        self.assertLess(behind, SOFA_BEHIND_START_M)
        self.assertTrue(self.teb["include_costmap_obstacles"])

    def test_feasibility_horizon_stays_within_one_robot_length(self):
        # Each feasibility pose is a full-footprint costmap check spaced
        # ~dt_ref*max_vel_x apart.  Checking ~0.4 m ahead (5 poses) rejects
        # every doorway approach; one robot length still catches starts
        # inside an obstacle.
        horizon = (self.teb["feasibility_check_no_poses"] *
                   self.teb["dt_ref"] * self.teb["max_vel_x"])
        self.assertGreaterEqual(self.teb["feasibility_check_no_poses"], 1)
        self.assertLessEqual(
            horizon, 2 * self.teb["footprint_model"]["radius"] + 0.01)

    def test_velocity_limits_and_homotopy_stay_evidence_based(self):
        self.assertEqual(self.teb["max_vel_x"], 0.18)
        self.assertEqual(self.teb["max_vel_x_backwards"], 0.12)
        self.assertEqual(self.teb["max_vel_theta"], 0.40)
        # No evidence-free rescue attempts: homotopy stays off until someone
        # can show it helps on the real floor.
        self.assertFalse(self.teb["enable_homotopy_class_planning"])

    def test_survey_default_is_joystick_with_navigation_opt_in(self):
        path = os.path.join(RELEASES, "amcl_survey_20260827.launch")
        include = ET.parse(path).getroot().find("include")
        passed = {arg.get("name"): arg.get("value")
                  for arg in include.findall("arg")}
        # Openloop door-frame braking is off the live path. The operator
        # drives; navigation remains an explicit opt-in.
        self.assertEqual(passed.get("control_mode"), "joystick")
        self.assertEqual(passed.get("require_localization"), "false")
        with open(os.path.join(
                ROOT, "launch", "semantic_survey.launch"), "r") as stream:
            survey = stream.read()
        self.assertIn("arg('control_mode') == 'navigation'", survey)
        self.assertIn('name="require_enable_button" value="false"', survey)
        self.assertIn('name="scale_linear" value="0.15"', survey)


if __name__ == "__main__":
    unittest.main()
