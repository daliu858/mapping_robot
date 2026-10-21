"""上游源码不随仓库转发；静态契约在显式组装的测试工作区检查。"""
import os
from pathlib import Path
import unittest


OVERLAY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = os.environ.get("JETBOT_TEST_SRC")
PACKAGE_ROOT = Path(SOURCE_ROOT) / "jetbot_pro" if SOURCE_ROOT else OVERLAY_ROOT
if SOURCE_ROOT and not (PACKAGE_ROOT / "CMakeLists.txt").is_file():
    raise RuntimeError("JETBOT_TEST_SRC must contain the packages assembled by setup_workspace.sh")

requires_upstream = unittest.skipUnless(
    (PACKAGE_ROOT / "CMakeLists.txt").is_file(),
    "Unredistributed upstream source required: run setup_workspace.sh and set JETBOT_TEST_SRC",
)
