import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" /
    "build_jetson_overlay.py")
SPEC = importlib.util.spec_from_file_location(
    "build_jetson_overlay", str(MODULE_PATH))
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class BuildJetsonOverlayTest(unittest.TestCase):
    def test_missing_package_directory_cannot_produce_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gscam = root / "gscam"
            gscam.mkdir()
            output = root / "release.tar.gz"
            with mock.patch.object(
                    BUILDER, "PACKAGE", root / "missing-package"), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                with self.assertRaisesRegex(
                        RuntimeError, "package directory"):
                    BUILDER.build(output)
            self.assertFalse(output.exists())

    def test_missing_package_sentinel_cannot_produce_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "jetbot_pro"
            package.mkdir()
            (package / "package.xml").write_text(
                "<package/>\n", encoding="utf-8")
            gscam = root / "gscam"
            gscam.mkdir()
            output = root / "release.tar.gz"
            with mock.patch.object(BUILDER, "PACKAGE", package), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                with self.assertRaisesRegex(
                        RuntimeError, "sentinel members"):
                    BUILDER.build(output)
            self.assertFalse(output.exists())

    def test_complete_sentinel_set_is_present_in_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "jetbot_pro"
            gscam = root / "gscam"
            for archive_name in BUILDER.REQUIRED_PACKAGE_MEMBERS:
                relative = Path(*archive_name.parts[1:])
                base = package if archive_name.parts[0] == "jetbot_pro" \
                    else gscam
                source = base / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("test\n", encoding="utf-8")
            with mock.patch.object(BUILDER, "PACKAGE", package), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                payload = list(BUILDER.iter_payload())
                BUILDER.validate_payload(payload)
            names = {name for _path, name in payload}
            self.assertTrue(BUILDER.REQUIRED_PACKAGE_MEMBERS <= names)

    def test_release_sentinels_cover_one_pass_mapping_and_safety(self):
        required = {str(path) for path in BUILDER.REQUIRED_PACKAGE_MEMBERS}
        for member in (
                "jetbot_pro/launch/slam.launch",
                "jetbot_pro/launch/slam_capture.launch",
                "jetbot_pro/launch/semantic_survey.launch",
                "jetbot_pro/launch/record_offline.launch",
                "jetbot_pro/launch/semantic_map_offline.launch",
                "jetbot_pro/scripts/checked_slam_capture.py",
                "jetbot_pro/scripts/checked_rosbag_record.py",
                "jetbot_pro/scripts/scan_watchdog.py",
                "jetbot_pro/scripts/scan_clip.py",
                "jetbot_pro/scripts/stereo_pair_gate.py",
                "jetbot_pro/tests/test_base_safety.cpp",
                "jetbot_pro/tests/test_detection_timestamp_matching.py",
                "jetbot_pro/tests/test_offline_transactions.py",
                "jetbot_pro/tests/test_pipeline_contracts.py",
                "jetbot_pro/tests/test_replay_contract.py",
                "jetbot_pro/src/jetbot.cpp"):
            self.assertIn(member, required)
        for removed in (
                "jetbot_pro/launch/semantic_nav.launch",
                "jetbot_pro/launch/semantic_map_legacy_coco.launch",
                "jetbot_pro/scripts/multipoint_nav.py",
                "jetbot_pro/scripts/reactive_survey.py",
                "jetbot_pro/scripts/semantic_navigation_core.py",
                "jetbot_pro/scripts/semantic_navigator.py",
                "jetbot_pro/tests/test_semantic_navigation_core.py"):
            self.assertNotIn(removed, required)

    def test_deleted_semantic_navigation_cannot_enter_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "jetbot_pro"
            gscam = root / "gscam"
            for archive_name in BUILDER.REQUIRED_PACKAGE_MEMBERS:
                base = package if archive_name.parts[0] == "jetbot_pro" \
                    else gscam
                source = base / Path(*archive_name.parts[1:])
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("test\n", encoding="utf-8")
            forbidden = next(iter(BUILDER.FORBIDDEN_PACKAGE_MEMBERS))
            source = package / Path(*forbidden.parts[1:])
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("deleted feature\n", encoding="utf-8")
            with mock.patch.object(BUILDER, "PACKAGE", package), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                payload = list(BUILDER.iter_payload())
                with self.assertRaisesRegex(
                        RuntimeError, "deleted semantic-navigation"):
                    BUILDER.validate_payload(payload)

    def test_timestamped_calibration_backups_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "jetbot_pro"
            gscam = root / "gscam"
            calibration = package / "config" / "camera_calibration"
            calibration.mkdir(parents=True)
            gscam.mkdir()
            active = calibration / "left.yaml"
            backup = calibration / "left.yaml.bak.20260804T053000"
            active.write_text("active\n", encoding="utf-8")
            backup.write_text("backup\n", encoding="utf-8")
            with mock.patch.object(BUILDER, "PACKAGE", package), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                names = {str(name) for _path, name in BUILDER.iter_payload()}
            self.assertIn(
                "jetbot_pro/config/camera_calibration/left.yaml", names)
            self.assertNotIn(
                "jetbot_pro/config/camera_calibration/"
                "left.yaml.bak.20260804T053000", names)

    def test_timestamped_backup_directories_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "jetbot_pro"
            gscam = root / "gscam"
            active = package / "config" / "active" / "left.yaml"
            backup = (package / "config" /
                      "camera_calibration.bak.20260804" / "left.yaml")
            active.parent.mkdir(parents=True)
            backup.parent.mkdir(parents=True)
            gscam.mkdir()
            active.write_text("active\n", encoding="utf-8")
            backup.write_text("backup\n", encoding="utf-8")
            with mock.patch.object(BUILDER, "PACKAGE", package), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                names = {str(name) for _path, name in BUILDER.iter_payload()}
            self.assertIn("jetbot_pro/config/active/left.yaml", names)
            self.assertNotIn(
                "jetbot_pro/config/camera_calibration.bak.20260804/left.yaml",
                names)

    def test_archive_digest_does_not_depend_on_output_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "jetbot_pro"
            gscam = root / "gscam"
            for archive_name in BUILDER.REQUIRED_PACKAGE_MEMBERS:
                base = package if archive_name.parts[0] == "jetbot_pro" \
                    else gscam
                source = base / Path(*archive_name.parts[1:])
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("test\n", encoding="utf-8")
            with mock.patch.object(BUILDER, "PACKAGE", package), \
                    mock.patch.object(BUILDER, "GSCAM_PACKAGE", gscam):
                first = BUILDER.build(root / "first.tar.gz")
                second = BUILDER.build(root / "second.tar.gz")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
