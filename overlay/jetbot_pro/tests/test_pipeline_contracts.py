#!/usr/bin/env python
"""Static contracts that keep the launch/deployment pipeline connected."""
from __future__ import print_function

import os
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = os.path.join(ROOT, "launch")
REPO = os.path.dirname(os.path.dirname(ROOT))
RELEASES = os.path.join(REPO, "releases")


def launch_args(path):
    root = ET.parse(path).getroot()
    return set(node.get("name") for node in root.findall("arg"))


class PipelineContractsTest(unittest.TestCase):
    def test_in_package_include_arguments_are_declared_by_child(self):
        pattern = re.compile(
            r"^\$\(find jetbot_pro\)/launch/([A-Za-z0-9_.-]+\.launch)$")
        for name in os.listdir(LAUNCH):
            if not name.endswith(".launch"):
                continue
            path = os.path.join(LAUNCH, name)
            root = ET.parse(path).getroot()
            for include in root.iter("include"):
                match = pattern.match(include.get("file", ""))
                if not match:
                    continue
                child_name = match.group(1)
                child_path = os.path.join(LAUNCH, child_name)
                self.assertTrue(os.path.isfile(child_path),
                                "%s includes missing %s" % (name, child_name))
                declared = launch_args(child_path)
                passed = set(arg.get("name") for arg in include.findall("arg"))
                for arg in include.findall("arg"):
                    self.assertIn(
                        arg.get("name"), declared,
                        "%s passes unknown arg %s to %s" %
                        (name, arg.get("name"), child_name))
                child_root = ET.parse(child_path).getroot()
                required = set(
                    arg.get("name") for arg in child_root.findall("arg")
                    if arg.get("default") is None)
                if include.get("pass_all_args") != "true":
                    self.assertFalse(
                        required - passed,
                        "%s omits required args %s for %s" %
                        (name, sorted(required - passed), child_name))

    def test_required_physical_and_map_inputs_have_no_guessed_default(self):
        for name, required in {
                "nav.launch": ("map_file", "initial_pose_x", "initial_pose_y",
                               "initial_pose_a"),
                "semantic_map_offline.launch":
                    ("bag", "det_file", "map_file", "output_file"),
                "slam_capture.launch": ("bag", "map_file", "camera_mount"),
                "semantic_survey.launch":
                    ("map_file", "camera_mount", "initial_pose_x",
                     "initial_pose_y", "initial_pose_a"),
                "record_offline.launch": ("map_file", "camera_mount"),
                "semantic_map.launch":
                    ("map_file", "camera_mount", "initial_pose_x",
                     "initial_pose_y", "initial_pose_a"),
        }.items():
            root = ET.parse(os.path.join(LAUNCH, name)).getroot()
            args = {node.get("name"): node for node in root.findall("arg")}
            for arg in required:
                self.assertIn(arg, args, "%s lacks %s" % (name, arg))
                self.assertIsNone(args[arg].get("default"),
                                  "%s must require %s" % (name, arg))

    def test_capture_and_mapping_nodes_are_installed(self):
        with open(os.path.join(ROOT, "CMakeLists.txt"), "r") as stream:
            cmake = stream.read()
        for script in ("map_identity.py", "checked_slam_capture.py",
                       "checked_rosbag_record.py", "semantic_mapper.py"):
            self.assertIn("scripts/" + script, cmake)
        for removed in ("semantic_navigation_core.py",
                        "semantic_navigator.py"):
            self.assertNotIn("scripts/" + removed, cmake)

    def test_offline_fusion_requires_recorded_map_identity(self):
        path = os.path.join(LAUNCH, "semantic_map_offline.launch")
        root = ET.parse(path).getroot()
        launch_arg_nodes = {
            node.get("name"): node for node in root.findall("arg")}
        launch_args = set(launch_arg_nodes)
        self.assertNotIn("start_player", launch_args)
        self.assertNotIn("range_source", launch_args)
        self.assertNotIn("stereo_lidar_fallback", launch_args)
        mapper = next(node for node in root.findall("node")
                      if node.get("name") == "semantic_mapper")
        params = {node.get("name"): node.get("value")
                  for node in mapper.findall("param")}
        self.assertEqual(params.get("require_map_identity"), "true")
        self.assertEqual(params.get("map_file"), "$(arg map_file)")
        self.assertEqual(params.get("map_manifest_identity_topic"),
                         "/survey/map_manifest_sha256")
        self.assertEqual(params.get("transactional_finalize_only"), "true")
        self.assertEqual(params.get("range_source"), "stereo")
        self.assertEqual(params.get("stereo_lidar_fallback"), "false")
        self.assertEqual(params.get("require_map_received"), "true")
        self.assertEqual(params.get("finalize_token"),
                         "$(arg transaction_token)")
        self.assertEqual(float(params.get("tf_lookup_timeout_s")), 0.0)
        self.assertEqual(params.get("tf_future_tolerance_s"),
                         "$(arg tf_future_tolerance_s)")
        # r3: with valid-rate fixed (81 percent), the 443 tf-dropped frames
        # became the binding hits cost; 0.40 keeps the slip error (~12 cm
        # at the motion gate) inside the 0.25 m spread budget.
        self.assertEqual(
            launch_arg_nodes["tf_future_tolerance_s"].get("default"), "0.40")
        # r3: 58 percent of valid door depths failed near-occupied at
        # 0.30 m because an open door's bbox center lands mid-doorway-gap;
        # 0.45 reaches a jamb from anywhere inside a ~0.9 m doorway.
        self.assertEqual(params.get("occupied_radius_m"),
                         "$(arg occupied_radius_m)")
        self.assertEqual(
            launch_arg_nodes["occupied_radius_m"].get("default"), "0.45")
        self.assertEqual(params.get("max_unsynced_fraction"),
                         "$(arg max_unsynced_fraction)")
        self.assertEqual(
            launch_arg_nodes["max_unsynced_fraction"].get("default"), "0.05")
        self.assertEqual(params.get("score_min"), "0.40")
        self.assertEqual(params.get("min_hits"), "$(arg min_hits)")
        self.assertEqual(launch_arg_nodes["min_hits"].get("default"), "5")
        self.assertEqual(params.get("max_spread_m"),
                         "$(arg max_spread_m)")
        self.assertEqual(
            launch_arg_nodes["max_spread_m"].get("default"), "0.25")
        self.assertEqual(params.get("structure_labels_csv"), "door")
        map_server = next(node for node in root.findall("node")
                          if node.get("name") == "map_server")
        self.assertEqual(map_server.get("required"), "true")
        core_names = {
            "map_server", "left_decompress",
            "right_decompress", "stereo_pair_gate", "stereo_image_proc",
            "detection_publisher", "semantic_mapper",
            "offline_replay_runner"}
        nodes = {node.get("name"): node for node in root.iter("node")}
        self.assertFalse(core_names - set(nodes))
        self.assertEqual(
            launch_arg_nodes["disparity_range"].get("default"), "128")
        stereo_params = {
            node.get("name"): node.get("value")
            for node in nodes["stereo_image_proc"].findall("param")}
        self.assertEqual(stereo_params.get("disparity_range"),
                         "$(arg disparity_range)")
        # _19 depth valid=0: stock block-matcher admission thresholds
        # rejected ~99.3 percent of pixels on the zoomed low-texture
        # rectified pair.  The tuned values must stay wired and their
        # defaults pinned so a regression cannot silently starve depth.
        for name, default in (("prefilter_cap", "63"),
                              ("texture_threshold", "2"),
                              ("uniqueness_ratio", "8"),
                              ("correlation_window_size", "21"),
                              ("speckle_size", "100"),
                              ("speckle_range", "4"),
                              ("min_disparity", "0")):
            self.assertEqual(stereo_params.get(name), "$(arg %s)" % name)
            self.assertEqual(launch_arg_nodes[name].get("default"), default)
        self.assertEqual(params.get("depth_roi_expand"),
                         "$(arg depth_roi_expand)")
        self.assertEqual(
            launch_arg_nodes["depth_roi_expand"].get("default"), "true")
        # _19 full-bag: the bottom-left detector sliver (76 percent of
        # gated door boxes) must stay filtered by bbox height, and the
        # threshold stays pinned to the 100-119 px histogram valley.
        self.assertEqual(params.get("min_structure_bbox_h_px"),
                         "$(arg min_structure_bbox_h_px)")
        self.assertEqual(
            launch_arg_nodes["min_structure_bbox_h_px"].get("default"),
            "120")
        self.assertIsNone(nodes["offline_replay_runner"].get("if"))
        for name in core_names:
            self.assertEqual(nodes[name].get("required"), "true",
                             "%s must fail the whole transaction" % name)
        runner_params = {
            node.get("name"): node.get("value")
            for node in nodes["offline_replay_runner"].findall("param")}
        self.assertEqual(runner_params.get("finalize_token"),
                         "$(arg transaction_token)")
        self.assertEqual(runner_params.get("require_complete_marker"), "true")
        self.assertEqual(runner_params.get("range_source"), "stereo")
        self.assertEqual(
            runner_params.get("stereo_lidar_fallback"), "false")
        self.assertEqual(runner_params.get("detections_file"),
                         "$(arg det_file)")
        self.assertEqual(runner_params.get("detection_ready_service"),
                         "/detection_publisher/ready")
        self.assertEqual(runner_params.get("detection_complete_service"),
                         "/detection_publisher/complete_input")
        self.assertEqual(runner_params.get("pipeline_status_service"),
                         "/semantic_mapper/pipeline_status")
        self.assertNotIn("base_to_left_optical", nodes)
        detector_params = {
            node.get("name"): node.get("value")
            for node in nodes["detection_publisher"].findall("param")}
        self.assertEqual(detector_params.get("bag"), "$(arg bag)")
        self.assertEqual(detector_params.get("require_exact_timestamps"),
                         "true")
        self.assertEqual(params.get("required_labels_csv"),
                         "$(arg required_labels_csv)")
        self.assertEqual(
            launch_arg_nodes["required_labels_csv"].get("default"),
            "chair,table,door")

    def test_stereo_tf_miss_never_invalidates_the_whole_replay(self):
        # Bag 19: one bag-open TF miss made the entire 90-minute replay
        # INVALID and no yaml was written.  The stereo TF path must drop
        # frames through its dedicated counters and stay away from the
        # transaction-poisoning infrastructure/contract paths.
        path = os.path.join(ROOT, "scripts", "semantic_mapper.py")
        with open(path, "r") as stream:
            source = stream.read()
        self.assertNotIn("stereo map transform unavailable", source)
        method = source[source.index("def _stereo_map_transform"):
                        source.index("def cb_det(")]
        self.assertNotIn("_pipeline_infrastructure_drop", method)
        self.assertNotIn("_pipeline_contract_violation", method)
        for counter in ("tf_startup_miss", "tf_latest_fallback",
                        "tf_stale_miss"):
            self.assertIn(counter, method)
        self.assertIn("classify_stereo_tf_miss", method)
        self.assertIn("latest_tf_fallback_ok", method)

    def test_primary_three_poi_labels_are_stable(self):
        root = ET.parse(os.path.join(
            LAUNCH, "semantic_map_offline.launch")).getroot()
        mapper = next(node for node in root.findall("node")
                      if node.get("name") == "semantic_mapper")
        params = {node.get("name"): node.get("value")
                  for node in mapper.findall("param")}
        self.assertEqual(params.get("required_labels_csv"),
                         "$(arg required_labels_csv)")
        args = {node.get("name"): node.get("default")
                for node in root.findall("arg")}
        self.assertEqual(args.get("required_labels_csv"), "chair,table,door")

    def test_detection_source_topic_mismatch_is_fail_closed(self):
        path = os.path.join(ROOT, "scripts", "detection_publisher.py")
        with open(path, "rb") as stream:
            source = stream.read().decode("utf-8")
        self.assertIn("detection image_topic mismatch", source)
        self.assertIn("raise ValueError", source)

    def test_one_pass_capture_records_transaction_without_amcl(self):
        root = ET.parse(os.path.join(LAUNCH, "slam_capture.launch")).getroot()
        recorders = [node for node in root.iter("node")
                     if node.get("type") == "checked_slam_capture.py"]
        self.assertEqual(len(recorders), 1)
        for recorder in recorders:
            params = {node.get("name"): node.get("value")
                      for node in recorder.findall("param")}
            self.assertIn("/survey/recording_status",
                          params.get("topics_csv", ""))
            self.assertIn("/survey/map_manifest_sha256",
                          params.get("topics_csv", ""))
            self.assertEqual(params.get("recording_status_topic"),
                             "/survey/recording_status")
            self.assertEqual(params.get("camera_mount"),
                             "$(arg camera_mount)")
            self.assertEqual(params.get("finalize_free_space_gib"),
                             "$(arg finalize_free_space_gib)")
            self.assertEqual(params.get("max_pair_dt_s"),
                             "$(arg max_pair_dt_s)")
            self.assertEqual(params.get("rosbag_stop_timeout_s"),
                             "$(arg rosbag_stop_timeout_s)")
            self.assertNotIn("/amcl_pose", params.get("topics_csv", ""))
            self.assertNotIn("/map,", params.get("topics_csv", ""))
            self.assertIn("/map,", params.get("required_topics_csv", ""))
        transform_nodes = [
            node for node in root.iter("node")
            if node.get("type") == "static_transform_publisher"]
        self.assertEqual(len(transform_nodes), 1)
        for node in transform_nodes:
            self.assertIn("$(arg camera_mount)", node.get("args", ""))
            self.assertEqual(node.get("required"), "true")
        args = {node.get("name"): node.get("default")
                for node in root.findall("arg")}
        self.assertEqual(args.get("finalize_free_space_gib"), "0.8")
        self.assertEqual(args.get("max_pair_dt_s"), "0.020")
        self.assertEqual(args.get("rosbag_stop_timeout_s"), "45.0")

    def test_one_pass_capture_does_not_include_navigation(self):
        path = os.path.join(LAUNCH, "slam_capture.launch")
        with open(path, "r") as stream:
            source = stream.read()
        for forbidden in ("nav.launch", "slam_nav.launch", "amcl.launch",
                          "move_base.launch", "TebLocalPlannerROS"):
            self.assertNotIn(forbidden, source)
        self.assertIn("slam.launch", source)

    def test_amcl_survey_deploys_the_actual_overlay_recorder(self):
        with open(os.path.join(RELEASES, "start_nav_survey.sh"), "r") as stream:
            start = stream.read()
        source_recorder = (
            "/home/jetbot/catkin_ws/src/jetbot_pro/scripts/"
            "checked_rosbag_record.py")
        overlay_recorder = (
            "/home/jetbot/catkin_ws/devel/lib/jetbot_pro/"
            "checked_rosbag_record.py")
        self.assertIn("checked_rosbag_record.py", start)
        self.assertIn('cp "/tmp/$py" '
                      '"/home/jetbot/catkin_ws/src/jetbot_pro/scripts/$py"',
                      start)
        self.assertIn('cp "/tmp/$py" '
                      '"/home/jetbot/catkin_ws/devel/lib/jetbot_pro/$py"',
                      start)
        for sibling in ("checked_rosbag_record.py", "scan_clip.py",
                        "scan_clip_core.py", "map_identity.py",
                        "replay_contract.py"):
            self.assertIn(sibling, start)
        chmod = start[start.index("chmod +x " + source_recorder):]
        self.assertIn(overlay_recorder, chmod)
        self.assertIn("cp /tmp/run_amcl_survey_20260827.sh", start)

    def test_amcl_survey_restart_never_deletes_or_broad_matches_amcl(self):
        with open(os.path.join(RELEASES, "restart_amcl_survey.sh"), "r") as stream:
            restart = stream.read()
        self.assertNotRegex(
            restart, r"rm\s+-[^\n]*semantic_survey_home.*\.bag")
        self.assertNotIn("pkill -9 -f '/amcl'", restart)
        self.assertIn("/opt/ros/melodic/lib/amcl/amcl", restart)
        self.assertIn("SURVEY_BAG", restart)

    def test_amcl_survey_has_one_collision_checked_bag_argument(self):
        launch = ET.parse(os.path.join(
            RELEASES, "amcl_survey_20260827.launch")).getroot()
        args = {node.get("name"): node for node in launch.findall("arg")}
        self.assertEqual(args["bag"].get("default"), "$(env SURVEY_BAG)")
        include = launch.find("include")
        passed = {node.get("name"): node.get("value")
                  for node in include.findall("arg")}
        self.assertEqual(passed["bag"], "$(arg bag)")
        with open(os.path.join(
                RELEASES, "run_amcl_survey_20260827.sh"), "r") as stream:
            runner = stream.read()
        for suffix in ('"$SURVEY_BAG"', '"$SURVEY_BAG.active"',
                       '"$SURVEY_BAG.complete.json"',
                       '"$SURVEY_BAG.failed.json"'):
            self.assertIn(suffix, runner)
        self.assertIn('bag:="$SURVEY_BAG"', runner)

    def test_amcl_survey_entry_preflights_before_side_effects(self):
        path = os.path.join(RELEASES, "start_nav_survey.sh")
        with open(path, "r") as stream:
            start = stream.read()
        for word in ("DISK_LOW", "FPS_INVALID", "NVARGUS_STALE",
                     "df -P -k /", "1572864",
                     "sudo -n -l systemctl restart nvargus-daemon"):
            self.assertIn(word, start)
        preflight_at = start.index("DISK_AVAILABLE_KB=")
        self.assertLess(preflight_at, start.index("BAG_ROOT="))
        self.assertLess(preflight_at, start.index("cp /tmp/"))
        self.assertLess(preflight_at,
                        start.index("bash /tmp/restart_amcl_survey.sh"))
        self.assertLess(start.index("FPS_INVALID"),
                        start.index("BAG_ROOT="))
        self.assertLess(start.index("NVARGUS_STALE"),
                        start.index("BAG_ROOT="))
        self.assertIn("grep -Eq 'Failed to create CaptureSession|Could not get gstreamer sample'",
                      start)
        self.assertIn("this script does not sudo", start)
        self.assertIn("if ! DISK_AVAILABLE_KB=$(df -P -k /", start)

    def test_amcl_survey_forces_the_robot_tested_fps(self):
        release_path = os.path.join(RELEASES, "amcl_survey_20260827.launch")
        release_root = ET.parse(release_path).getroot()
        include = release_root.find("include")
        passed = {node.get("name"): node.get("value")
                  for node in include.findall("arg")}
        self.assertEqual(passed.get("fps"), "15")
        survey_root = ET.parse(os.path.join(
            LAUNCH, "semantic_survey.launch")).getroot()
        args = {node.get("name"): node.get("default")
                for node in survey_root.findall("arg")}
        self.assertEqual(args.get("fps"), "15")
        with open(os.path.join(RELEASES, "start_nav_survey.sh"), "r") as stream:
            start = stream.read()
        self.assertIn("grep -q 'name=\"fps\" value=\"15\"'", start)

    def test_amcl_survey_entry_exits_zero_only_for_joystick_ready(self):
        with open(os.path.join(RELEASES, "start_nav_survey.sh"), "r") as stream:
            start = stream.read()
        self.assertNotIn("goal 1 result=", start)
        self.assertNotIn("FIRST_GOAL", start)
        self.assertNotIn("starting waypoint tour", start)
        self.assertNotIn("TOUR_STARTED", start)
        self.assertNotIn("run_survey_tour.sh", start)
        self.assertEqual(start.count("exit 0"), 1)
        for word in ("JOYSTICK_READY", "FAILED_START", "NO_ACTIVE",
                     "exit 2", "exit 3"):
            self.assertIn(word, start)
        self.assertIn("flock -n 9", start)
        for line in start.splitlines():
            if "restart_amcl_survey.sh" in line and \
                    line.lstrip().startswith("bash"):
                self.assertIn("9>&-", line)

    def test_amcl_survey_entry_does_not_auto_start_the_openloop_tour(self):
        with open(os.path.join(RELEASES, "start_nav_survey.sh"), "r") as stream:
            start = stream.read()
        self.assertIn("JOYSTICK_READY", start)
        self.assertIn("complete_survey.sh", start)
        self.assertIn("joystick_teleop.launch", start)
        self.assertNotIn("starting openloop tour", start)
        self.assertNotIn("nohup /home/jetbot/run_survey_tour.sh", start)

    def test_tour_crash_fail_closes_recorder_but_interrupt_leaves_it_open(self):
        with open(os.path.join(
                RELEASES, "survey_tour_20260827.py"), "r") as stream:
            tour = stream.read()
        with open(os.path.join(
                ROOT, "scripts", "checked_rosbag_record.py"), "r") as stream:
            recorder = stream.read()
        # Route knowledge lives in the tour; the recorder owns only the
        # recording transaction.  .complete.json therefore never implies
        # route success -- a reached=0 tour still completes the bag and
        # reports the route result through its own exit code.
        self.assertIn('"/offline_recorder/complete"', tour)
        self.assertIn('"/offline_recorder/fail"', tour)
        self.assertIn("return 0 if reached >= 1 else 2", tour)
        self.assertIn('"~complete"', recorder)
        self.assertIn('"~fail"', recorder)
        self.assertNotIn("WAYPOINTS", recorder)
        self.assertNotIn("survey_tour", recorder)
        # Operator interrupt re-raises untouched (rerun_tour.sh may rerun
        # against the still-healthy recording); every other crash stops the
        # wheels and fail-closes the transaction explicitly.
        main_body = tour[tour.index("def main():"):tour.index("if __name__")]
        interrupt_at = main_body.index("except rospy.ROSInterruptException:")
        crash_at = main_body.index("except Exception as error:")
        self.assertLess(interrupt_at, crash_at)
        self.assertNotIn("_abort_recorder", main_body[interrupt_at:crash_at])
        self.assertIn("_crash_stop(tour)", main_body[crash_at:])
        self.assertIn("_abort_recorder(", main_body[crash_at:])

    def test_two_pass_survey_uses_amcl_and_checked_recorder(self):
        path = os.path.join(LAUNCH, "semantic_survey.launch")
        with open(path, "r") as stream:
            source = stream.read()
        self.assertIn("nav.launch", source)
        self.assertIn("record_offline.launch", source)
        self.assertIn("joystick_teleop.launch", source)
        for forbidden in ("slam.launch", "gmapping.launch",
                          "semantic_nav.launch", "slam_capture.launch"):
            self.assertNotIn(forbidden, source)
        root = ET.parse(os.path.join(LAUNCH, "record_offline.launch")).getroot()
        recorders = [node for node in root.iter("node")
                     if node.get("type") == "checked_rosbag_record.py"]
        self.assertEqual(len(recorders), 2)
        for recorder in recorders:
            params = {node.get("name"): node.get("value")
                      for node in recorder.findall("param")}
            self.assertIn("/amcl_pose", params.get("topics_csv", ""))
            self.assertEqual(params.get("require_localization"),
                             "$(arg require_localization)")
            self.assertEqual(params.get("camera_mount"),
                             "$(arg camera_mount)")
            self.assertNotIn("/map,", params.get("topics_csv", "") + ",")
        alias = os.path.join(LAUNCH, "semantic_map.launch")
        with open(alias, "r") as stream:
            alias_source = stream.read()
        self.assertIn("semantic_survey.launch", alias_source)
        self.assertNotIn("slam_capture.launch", alias_source)

    def test_gmapping_fails_closed_instead_of_resetting_map_frame(self):
        root = ET.parse(os.path.join(LAUNCH, "gmapping.launch")).getroot()
        node = next(node for node in root.findall("node")
                    if node.get("name") == "slam_gmapping")
        self.assertEqual(node.get("required"), "true")
        self.assertEqual(node.get("respawn"), "false")

    def test_teb_local_costmap_uses_only_live_obstacles(self):
        path = os.path.join(
            ROOT, "config", "diff", "local_costmap_params.yaml")
        with open(path, "r") as stream:
            source = stream.read()
        self.assertNotIn("type: \"costmap_2d::StaticLayer\"", source)
        self.assertIn("costmap_2d::VoxelLayer", source)
        self.assertIn("costmap_2d::InflationLayer", source)

    def test_global_costmap_keeps_saved_map_static_layer(self):
        path = os.path.join(
            ROOT, "config", "diff", "global_costmap_params.yaml")
        with open(path, "r") as stream:
            source = stream.read()
        self.assertIn("type: \"costmap_2d::StaticLayer\"", source)
        self.assertIn("costmap_2d::InflationLayer", source)
        self.assertNotIn("static_map: true", source)

    def test_finalize_token_service_is_generated(self):
        with open(os.path.join(ROOT, "CMakeLists.txt"), "r") as stream:
            cmake = stream.read()
        self.assertIn("FinalizeSemanticMap.srv", cmake)
        service = os.path.join(ROOT, "srv", "FinalizeSemanticMap.srv")
        self.assertTrue(os.path.isfile(service))
        with open(service, "r") as stream:
            source = stream.read()
        self.assertIn("string token", source)

    def test_base_serial_is_nonblocking_serialized_and_stops_before_close(self):
        source_path = os.path.join(ROOT, "src", "jetbot.cpp")
        with open(source_path, "r") as stream:
            source = stream.read()
        self.assertIn("O_NONBLOCK", source)
        self.assertIn("serial_io_mutex", source)
        self.assertIn("read_serial", source)
        self.assertIn("write_serial", source)
        self.assertNotIn("read(sp, buffer", source)
        self.assertNotIn("write(sp,buffer", source)
        stop_at = source.index("serial_stop_requested.store(true)")
        join_at = source.index("serial_thread.join()", stop_at)
        close_at = source.index("sp.close(close_error)", join_at)
        self.assertLess(stop_at, join_at)
        self.assertLess(join_at, close_at)

    def test_base_startup_exceptions_cannot_bypass_fail_stop_cleanup(self):
        source_path = os.path.join(ROOT, "src", "jetbot.cpp")
        with open(source_path, "r") as stream:
            source = stream.read()
        guarded_try = source.index("try\n  {", source.index("thread serial_thread"))
        callback_at = source.index("server.setCallback(f)")
        thread_at = source.index("serial_thread = thread", callback_at)
        std_catch_at = source.index("catch (const std::exception&", thread_at)
        final_zero_at = source.index("for (int i = 0; i < 3; ++i)", std_catch_at)
        self.assertLess(guarded_try, callback_at)
        self.assertLess(callback_at, thread_at)
        self.assertLess(thread_at, std_catch_at)
        self.assertLess(std_catch_at, final_zero_at)
        self.assertIn("return exit_code", source)

    def test_uncalibrated_hardware_imu_is_fail_closed_by_default(self):
        root = ET.parse(os.path.join(LAUNCH, "jetbot.launch")).getroot()
        args = {node.get("name"): node.get("default")
                for node in root.findall("arg")}
        self.assertNotIn("use_imu", args)
        ekf = next(node for node in root.findall("node")
                   if node.get("name") == "robot_pose_ekf")
        params = {node.get("name"): node.get("value")
                  for node in ekf.findall("param")}
        self.assertEqual(params.get("imu_used"), "false")

    def test_csi_gstreamer_pipeline_keeps_parseable_terminal_element(self):
        root = ET.parse(os.path.join(LAUNCH, "csi_camera.launch")).getroot()
        defaults = {node.get("name"): node.get("default")
                    for node in root.findall("arg")}
        config_node = next(node for node in root.findall("param")
                           if node.get("name") ==
                           "$(arg cam_name)/gscam_config")
        config = config_node.get("value", "")
        for name in ("sensor_id", "width", "height", "fps", "flip_method"):
            config = config.replace("$(arg %s)" % name, defaults[name])
        self.assertNotIn("$(arg ", config)
        self.assertNotIn("\n", config)
        self.assertNotIn("\r", config)
        stages = [stage.strip() for stage in config.split("!")]
        self.assertGreaterEqual(len(stages), 4)
        for stage in stages:
            if stage.startswith("video/"):
                self.assertIsNone(re.search(r",\s", stage), stage)
        self.assertEqual(stages[-1], "videoconvert")
        self.assertNotIn("/", stages[-1])

    def test_geometry_mapping_keeps_camera_compute_opt_in(self):
        for name in ("slam.launch", "slam_nav.launch"):
            root = ET.parse(os.path.join(LAUNCH, name)).getroot()
            args = {node.get("name"): node.get("default")
                    for node in root.findall("arg")}
            self.assertEqual(args.get("start_camera"), "false", name)
            camera_groups = [group for group in root.findall("group")
                             if group.get("if") == "$(arg start_camera)"]
            self.assertEqual(len(camera_groups), 1, name)
            includes = camera_groups[0].findall("include")
            self.assertEqual(len(includes), 1, name)
            self.assertTrue(includes[0].get("file", "").endswith(
                "/camera_select.launch"), name)

    def test_base_control_loop_sends_at_fixed_50hz_and_retains_fail_stop(self):
        source_path = os.path.join(ROOT, "src", "jetbot.cpp")
        with open(source_path, "r") as stream:
            source = stream.read()
        loop_at = source.index("ros::WallRate control_loop_rate")
        loop = source[loop_at:source.index(
            "catch (const boost::system::system_error&", loop_at)]
        self.assertIn("control_loop_rate(50.0)", loop)
        self.assertNotIn("current_time - last_time", loop)
        self.assertEqual(loop.count("SetVelocity(x,y,yaw)"), 1)
        self.assertLess(loop.index("ros::spinOnce()"),
                        loop.index("SetVelocity(x,y,yaw)"))
        self.assertIn("control_loop_rate.sleep()", loop)
        timeout_call = loop[loop.index("commandForCycle("):
                            loop.index("SetVelocity(x,y,yaw)")]
        self.assertIsNotNone(re.search(
            r"cmd_wall_time\)\.toSec\(\),\s*1\.0", timeout_call))
        self.assertIn("ros::WallTime current_wall_time", source)
        cleanup_at = source.index("// Fail-stop on every normal ROS shutdown")
        cleanup = source[cleanup_at:source.index("return exit_code", cleanup_at)]
        self.assertIn("for (int i = 0; i < 3; ++i)", cleanup)
        self.assertIn("SetVelocity(0.0, 0.0, 0.0)", cleanup)

    def test_bounded_probe_rechecks_health_and_stops_before_ros_shutdown(self):
        path = os.path.join(ROOT, "scripts", "bounded_motion_probe.py")
        with open(path, "r") as stream:
            source = stream.read()
        self.assertIn("disable_signals=True", source)
        self.assertIn("except (Exception, KeyboardInterrupt)", source)
        command_at = source.index("def command_until")
        command = source[command_at:source.index("def parse_args", command_at)]
        self.assertLess(command.index("stop_event.is_set()"),
                        command.index("require_ready("))
        self.assertLess(command.index("require_ready("),
                        command.index("pub.publish(command)"))
        final_at = source.index("finally:", source.index("def main"))
        final = source[final_at:source.index("if __name__", final_at)]
        self.assertLess(final.index("stop_robot(pub)"),
                        final.index("rospy.signal_shutdown"))

    def test_gscam_has_bounded_pull_and_idempotent_failure_cleanup(self):
        source_path = os.path.join(
            os.path.dirname(ROOT), "gscam", "src", "gscam.cpp")
        with open(source_path, "r") as stream:
            source = stream.read()
        self.assertIn("gst_app_sink_try_pull_sample", source)
        self.assertNotIn("gst_app_sink_pull_sample(GST_APP_SINK", source)
        self.assertIn("sample_timeout_s_", source)
        self.assertIn("failing camera closed", source)
        self.assertIn("gst_buffer_map(buf", source)
        self.assertNotIn("gst_buffer_get_memory(buf, 0)", source)
        self.assertIn("GST_STATE_CHANGE_ASYNC", source)
        self.assertIn("GST_CLOCK_TIME_IS_VALID", source)
        self.assertIn("gst_video_frame_map", source)
        self.assertIn("copyRawImageRows", source)
        self.assertIn("Invalid GStreamer raw image layout", source)
        destructor = source[source.index("GSCam::~GSCam"):
                            source.index("bool GSCam::configure")]
        self.assertIn("cleanup_stream()", destructor)
        init_failure = source[source.index("if(!this->init_stream())"):
                              source.index("// Block while publishing")]
        self.assertIn("cleanup_stream()", init_failure)

        nodelet_path = os.path.join(
            os.path.dirname(ROOT), "gscam", "src", "gscam_nodelet.cpp")
        with open(nodelet_path, "r") as stream:
            nodelet = stream.read()
        self.assertIn("gscam_driver_->stop()", nodelet)
        self.assertIn("stream_thread_->joinable()", nodelet)

    def test_base_safety_logic_has_a_real_cpp_test_target(self):
        with open(os.path.join(ROOT, "CMakeLists.txt"), "r") as stream:
            cmake = stream.read()
        self.assertIn("catkin_add_gtest(test_base_safety", cmake)
        test_path = os.path.join(ROOT, "tests", "test_base_safety.cpp")
        self.assertTrue(os.path.isfile(test_path))
        with open(test_path, "r") as stream:
            source = stream.read()
        for contract in ("RejectsEveryNonFiniteVelocity",
                         "StopsForTimeoutClockResetAndSafetyLatch",
                         "AcceptsOnlyTheDocumentedBoundedTelemetryFrame"):
            self.assertIn(contract, source)


if __name__ == "__main__":
    unittest.main()
