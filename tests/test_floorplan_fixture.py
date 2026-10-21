"""使用合成占用栅格验证两种绘图入口，不读取真实住宅数据。"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "floorplan" / "fixtures"
VERSIONS = ("v31", "v32")


class FloorplanFixtureTests(unittest.TestCase):
    def run_floorplan(self, version, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "floorplan" / f"make_floorplan_{version}.py"),
             *map(str, arguments)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=90,
        )

    def inputs(self):
        return ["--map", FIXTURES / "synthetic_apartment.yaml",
                "--pois", FIXTURES / "synthetic_pois.yaml"]

    def test_synthetic_fixture_renders_without_bags(self):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                result = self.run_floorplan(
                    version, *self.inputs(), "--no-visibility", "--out", directory)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                png = Path(directory) / f"floorplan_{version}.png"
                self.assertTrue(png.is_file())
                with Image.open(png) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(min(image.size), 100)
                    image.verify()
                zones = yaml.safe_load(
                    (Path(directory) / f"zones_{version}.yaml").read_text(encoding="utf-8"))
                labels = {label for zone in zones["zones"] for label in zone["poi_labels"]}
                self.assertEqual(labels, {"door", "sofa", "cabinet", "spray"})
                self.assertTrue(zones["method"]["visibility"].startswith("off"))

    def test_missing_required_inputs_print_usage(self):
        for version in VERSIONS:
            with self.subTest(version=version):
                result = self.run_floorplan(version)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("usage:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_visibility_requires_an_explicit_existing_bag(self):
        for version in VERSIONS:
            for arguments in ([], ["--bag", FIXTURES / "missing.bag"]):
                with self.subTest(version=version, arguments=arguments):
                    result = self.run_floorplan(version, *self.inputs(), *arguments)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("--bag", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
