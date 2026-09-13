# -*- coding: utf-8 -*-
"""Build a deterministic overlay archive for ~/catkin_ws/src on the Jetson."""
from __future__ import print_function

import argparse
import gzip
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "jetbot_pro"
GSCAM_PACKAGE = ROOT / "src" / "gscam"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}
EXCLUDED_NAMES = {"five_shot_recovery_report.json"}
REQUIRED_PACKAGE_MEMBERS = frozenset({
    PurePosixPath("jetbot_pro", "package.xml"),
    PurePosixPath("jetbot_pro", "CMakeLists.txt"),
    PurePosixPath("jetbot_pro", "launch", "csi_camera.launch"),
    PurePosixPath("jetbot_pro", "launch", "camera_select.launch"),
    PurePosixPath("jetbot_pro", "launch", "stereo_camera.launch"),
    PurePosixPath("jetbot_pro", "launch", "jetbot.launch"),
    PurePosixPath("jetbot_pro", "launch", "slam.launch"),
    PurePosixPath("jetbot_pro", "launch", "slam_nav.launch"),
    PurePosixPath("jetbot_pro", "launch", "nav.launch"),
    PurePosixPath("jetbot_pro", "launch", "slam_capture.launch"),
    PurePosixPath("jetbot_pro", "launch", "semantic_survey.launch"),
    PurePosixPath("jetbot_pro", "launch", "record_offline.launch"),
    PurePosixPath("jetbot_pro", "launch", "semantic_map.launch"),
    PurePosixPath("jetbot_pro", "launch", "semantic_map_offline.launch"),
    PurePosixPath("jetbot_pro", "scripts", "checked_slam_capture.py"),
    PurePosixPath("jetbot_pro", "scripts", "checked_rosbag_record.py"),
    PurePosixPath("jetbot_pro", "scripts", "bounded_motion_probe.py"),
    PurePosixPath("jetbot_pro", "scripts", "detection_publisher.py"),
    PurePosixPath("jetbot_pro", "scripts", "map_identity.py"),
    PurePosixPath("jetbot_pro", "scripts", "offline_replay_runner.py"),
    PurePosixPath("jetbot_pro", "scripts", "replay_contract.py"),
    PurePosixPath("jetbot_pro", "scripts", "semantic_mapper.py"),
    PurePosixPath("jetbot_pro", "scripts", "scan_watchdog.py"),
    PurePosixPath("jetbot_pro", "scripts", "scan_clip.py"),
    PurePosixPath("jetbot_pro", "scripts", "stereo_pair_gate.py"),
    PurePosixPath("jetbot_pro", "scripts", "stereo_calibrate_headless.py"),
    PurePosixPath("jetbot_pro", "scripts", "stereo_calibration_core.py"),
    PurePosixPath("jetbot_pro", "include", "jetbot_pro", "base_safety.hpp"),
    PurePosixPath("jetbot_pro", "src", "jetbot.cpp"),
    PurePosixPath("jetbot_pro", "config", "camera_calibration",
                  "stereo_left_640x480.yaml"),
    PurePosixPath("jetbot_pro", "config", "camera_calibration",
                  "stereo_right_640x480.yaml"),
    PurePosixPath("jetbot_pro", "srv", "FinalizeSemanticMap.srv"),
    PurePosixPath("jetbot_pro", "tests", "test_base_safety.cpp"),
    PurePosixPath("jetbot_pro", "tests", "test_bounded_motion_probe.py"),
    PurePosixPath("jetbot_pro", "tests", "test_camera_launch_defaults.py"),
    PurePosixPath("jetbot_pro", "tests", "test_detection_timestamp_matching.py"),
    PurePosixPath("jetbot_pro", "tests", "test_offline_transactions.py"),
    PurePosixPath("jetbot_pro", "tests", "test_pipeline_contracts.py"),
    PurePosixPath("jetbot_pro", "tests", "test_replay_contract.py"),
    PurePosixPath("jetbot_pro", "tests", "test_stereo_calibration_core.py"),
    PurePosixPath("jetbot_pro", "tests", "test_stereo_calibration_helpers.py"),
    PurePosixPath("jetbot_pro", "tests", "test_stereo_geometry.py"),
    PurePosixPath("jetbot_pro", "tests", "fixtures",
                  "minimal_three_poi_semantic.yaml"),
    PurePosixPath("gscam", "package.xml"),
    PurePosixPath("gscam", "CMakeLists.txt"),
    PurePosixPath("gscam", "include", "gscam", "gscam.h"),
    PurePosixPath("gscam", "include", "gscam", "raw_image_copy.h"),
    PurePosixPath("gscam", "src", "gscam.cpp"),
    PurePosixPath("gscam", "src", "gscam_node.cpp"),
    PurePosixPath("gscam", "src", "gscam_nodelet.cpp"),
    PurePosixPath("gscam", "src", "raw_image_copy.cpp"),
    PurePosixPath("gscam", "test", "test_raw_image_copy.cpp"),
})
FORBIDDEN_PACKAGE_MEMBERS = frozenset({
    PurePosixPath("jetbot_pro", "launch", "semantic_nav.launch"),
    PurePosixPath("jetbot_pro", "launch", "semantic_map_legacy_coco.launch"),
    PurePosixPath("jetbot_pro", "scripts", "semantic_navigator.py"),
    PurePosixPath("jetbot_pro", "scripts", "semantic_navigation_core.py"),
    PurePosixPath("jetbot_pro", "scripts", "reactive_survey.py"),
    PurePosixPath("jetbot_pro", "scripts", "multipoint_nav.py"),
    PurePosixPath("jetbot_pro", "tests", "test_semantic_navigation_core.py"),
})


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_payload():
    for package, prefix in ((PACKAGE, "jetbot_pro"),
                            (GSCAM_PACKAGE, "gscam")):
        for path in sorted(package.rglob("*")):
            relative = path.relative_to(package)
            if any(part in EXCLUDED_PARTS or part.endswith(".bak") or
                   ".bak." in part for part in relative.parts):
                continue
            if (path.is_dir() or path.name in EXCLUDED_NAMES or
                    path.suffix in (".pyc", ".pyo", ".bak") or
                    ".bak." in path.name):
                continue
            yield path, PurePosixPath(prefix, *relative.parts)


def executable_mode(archive_path):
    parts = archive_path.parts
    if archive_path.suffix == ".py" and "scripts" in parts:
        return 0o755
    if archive_path.suffix == ".sh":
        return 0o755
    return 0o644


def add_bytes(tar, archive_name, data, mode=0o644):
    info = tarfile.TarInfo(str(archive_name))
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    tar.addfile(info, io.BytesIO(data))


def verify_archive(path, snapshots, manifest):
    """Read the completed gzip/tar back and compare every byte and mode."""
    expected = {str(name): (data, mode)
                for _source, name, data, mode in snapshots}
    expected["RELEASE_MANIFEST.txt"] = (manifest, 0o644)
    with tarfile.open(str(path), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("release archive member set failed verification")
        for member in members:
            stream = archive.extractfile(member)
            payload = stream.read() if stream is not None else None
            expected_bytes, expected_mode = expected[member.name]
            if payload != expected_bytes:
                raise RuntimeError(
                    "release archive byte verification failed: %s" %
                    member.name)
            if member.mode != expected_mode:
                raise RuntimeError(
                    "release archive mode verification failed: %s" %
                    member.name)


def atomic_write_text(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="ascii", newline="\n",
                prefix=path.name + ".", suffix=".partial",
                dir=str(path.parent), delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_payload(payload):
    if not PACKAGE.is_dir():
        raise RuntimeError("missing release package directory: %s" % PACKAGE)
    if not GSCAM_PACKAGE.is_dir():
        raise RuntimeError(
            "missing release package directory: %s" % GSCAM_PACKAGE)
    missing_sources = [str(path) for path, _name in payload
                       if not path.is_file()]
    if missing_sources:
        raise RuntimeError("missing release inputs: %s" %
                           ", ".join(missing_sources))
    archive_names = {name for _path, name in payload}
    forbidden_members = sorted(FORBIDDEN_PACKAGE_MEMBERS & archive_names,
                               key=str)
    if forbidden_members:
        raise RuntimeError(
            "release package contains deleted semantic-navigation members: %s" %
            ", ".join(str(name) for name in forbidden_members))
    missing_members = sorted(REQUIRED_PACKAGE_MEMBERS - archive_names,
                             key=str)
    if missing_members:
        raise RuntimeError(
            "release package is incomplete; missing sentinel members: %s" %
            ", ".join(str(name) for name in missing_members))


def build(output):
    if not PACKAGE.is_dir():
        raise RuntimeError("missing release package directory: %s" % PACKAGE)
    payload = list(iter_payload())
    validate_payload(payload)
    # Snapshot every source exactly once.  The manifest and tar must describe
    # these same bytes even if a developer edits the tree during the build.
    snapshots = [
        (path, archive_path, path.read_bytes(), executable_mode(archive_path))
        for path, archive_path in payload
    ]

    manifest_lines = [
        "SLAM Car closure overlay",
        "Target: /home/jetbot/catkin_ws/src",
        "Extract: tar -xzf ARCHIVE -C /home/jetbot/catkin_ws/src",
        "Contents: complete jetbot_pro and patched gscam packages",
        "Generated paths and SHA-256:",
    ]
    for _path, archive_path, data, _mode in snapshots:
        manifest_lines.append("%s  %s" %
                              (sha256_bytes(data), str(archive_path)))
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb", prefix=output.name + ".", suffix=".partial",
            dir=str(output.parent), delete=False)
        temporary = Path(handle.name)
        with handle as raw:
            # Keep the gzip header independent of the caller's output basename.
            with gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb",
                    mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as tar:
                    add_bytes(tar, "RELEASE_MANIFEST.txt", manifest)
                    for _path, archive_path, data, mode in snapshots:
                        add_bytes(tar, archive_path, data, mode)
            raw.flush()
            os.fsync(raw.fileno())
        verify_archive(temporary, snapshots, manifest)
        digest = sha256_file(temporary)
        os.replace(str(temporary), str(output))
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    atomic_write_text(
        checksum_path, "%s  %s\n" % (digest, output.name))
    print("built:", output)
    print("sha256:", digest)
    print("files:", len(payload) + 1)
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "releases" /
                    "slam_car_closure_20260804_overlay.tar.gz"))
    args = parser.parse_args()
    build(Path(args.out).resolve())


if __name__ == "__main__":
    main()
