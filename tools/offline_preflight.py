# -*- coding: utf-8 -*-
"""电脑端开放词汇检测器与一次式 gmapping 录包的快速前置检查。"""
from __future__ import print_function

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from map_identity import load_map_identity
from rosbags_index_compat import patch_rosbags_index_headers
_ROBOT_SCRIPTS = os.path.join(
    os.path.dirname(_TOOLS_DIR), "src", "jetbot_pro", "scripts")
if _ROBOT_SCRIPTS not in sys.path:
    sys.path.insert(0, _ROBOT_SCRIPTS)
from replay_contract import (camera_transform_matches,
                             canonical_camera_mount as _canonical_camera_mount,
                             validate_recording_states as
                             _validate_recording_states)


REQUIRED_BAG_TOPICS = {
    # This Melodic stack uses tf/static_transform_publisher, which republishes
    # fixed transforms on /tf rather than tf2's latched /tf_static.
    "/tf", "/scan", "/odom", "/odom_raw",
    "/survey/map_image_sha256",
    "/survey/map_manifest_sha256",
    "/survey/recording_status",
    "/stereo/left/image_raw/compressed",
    "/stereo/right/image_raw/compressed",
    "/stereo/left/camera_info", "/stereo/right/camera_info",
}
POI_QUERIES = [
    "chair", "table", "dining table", "closed door",
]
DEFAULT_MODEL = "iSEE-Laboratory/llmdet_large"
DEFAULT_HF_HOME = (
    os.environ.get("HF_HOME") or
    (r"D:\hf_cache" if os.path.isdir(r"D:\hf_cache") else
     os.path.join(str(Path.home()), ".cache", "huggingface")))


def canonical_camera_mount(value):
    """Normalize the recorded base->camera pose to six finite floats."""
    try:
        return _canonical_camera_mount(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("录制标记 camera_mount 无效: %s" % error)


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def disk_usage_probe(path):
    """Return the nearest existing path on the same volume as ``path``."""
    requested = Path(os.path.abspath(os.path.expanduser(str(path))))
    if requested.exists() and not requested.is_dir():
        raise RuntimeError("HF_HOME 不是目录: %s" % requested)
    probe = requested
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise RuntimeError("无法定位 HF_HOME 所在磁盘: %s" % requested)
        probe = parent
    return probe


def free_space_gib(path):
    probe = disk_usage_probe(path)
    free = shutil.disk_usage(str(probe)).free / float(1024 ** 3)
    return free, probe


def map_image_sha256(map_file):
    return load_map_identity(
        map_file, allow_portable_sibling=True,
        require_zero_yaw=True)["map_image_sha256"]


def validate_recording_states(payloads, expected_map_hash=None,
                              expected_manifest_hash=None):
    """Require one checked STARTED -> COMPLETED survey transaction."""
    valid, detail = _validate_recording_states(
        payloads, expected_map_hash, expected_manifest_hash)
    if not valid:
        raise RuntimeError("录制事务无效: %s" % detail)
    return detail


def validate_recorded_camera_tf(reader, completion,
                                expected_child="stereo_left_optical"):
    """Require every recorded target camera edge to equal camera_mount."""
    connections = [connection for connection in reader.connections
                   if connection.topic == "/tf"]
    candidates = []
    for connection, _timestamp, raw in reader.messages(
            connections=connections):
        message = reader.deserialize(raw, connection.msgtype)
        for transform in getattr(message, "transforms", []):
            if (str(transform.child_frame_id).strip("/") !=
                    str(expected_child).strip("/")):
                continue
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            candidates.append((
                transform.header.frame_id, transform.child_frame_id,
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w)))
    if not candidates:
        raise RuntimeError("bag 中没有录到 base_footprint→双目左相机 TF")
    if any(not camera_transform_matches(
            completion["camera_mount"], parent, child,
            translation, quaternion, expected_child=expected_child)
            for parent, child, translation, quaternion in candidates):
        raise RuntimeError("bag 内相机 TF 与录制事务 camera_mount 不一致")
    return True


def validate_recorded_slam_tf(reader):
    """Require the gmapping pose chain used for offline image projection."""
    required = {("map", "odom"), ("odom", "base_footprint")}
    observed = set()
    connections = [connection for connection in reader.connections
                   if connection.topic == "/tf"]
    for connection, _timestamp, raw in reader.messages(
            connections=connections):
        message = reader.deserialize(raw, connection.msgtype)
        for transform in getattr(message, "transforms", []):
            observed.add((
                str(transform.header.frame_id).strip().lstrip("/"),
                str(transform.child_frame_id).strip().lstrip("/")))
    missing = sorted(required - observed)
    if missing:
        raise RuntimeError(
            "gmapping bag 缺少 TF: %s" %
            ", ".join("%s->%s" % edge for edge in missing))
    return True


def check_bag(path, map_file):
    patch_rosbags_index_headers()
    from rosbags.highlevel import AnyReader

    bag = Path(path)
    if not bag.is_file():
        raise RuntimeError("bag 不存在: %s" % bag)
    expected_identity = load_map_identity(
        map_file, allow_portable_sibling=True, require_zero_yaw=True)
    expected = expected_identity["map_image_sha256"]
    expected_manifest = expected_identity["map_manifest_sha256"]
    with AnyReader([bag]) as reader:
        topics = {connection.topic for connection in reader.connections}
        missing = sorted(REQUIRED_BAG_TOPICS - topics)
        if missing:
            raise RuntimeError(
                "bag 缺少闭环必需 topic: %s" % ", ".join(missing))
        identity_connections = [
            connection for connection in reader.connections
            if connection.topic == "/survey/map_image_sha256"]
        identities = set()
        for connection, _timestamp, raw in reader.messages(
                connections=identity_connections):
            message = reader.deserialize(raw, connection.msgtype)
            value = str(message.data).strip().lower()
            if len(value) != 64 or any(
                    char not in "0123456789abcdef" for char in value):
                raise RuntimeError("bag 中存在无效地图 SHA-256: %r" % value)
            identities.add(value)
        manifest_connections = [
            connection for connection in reader.connections
            if connection.topic == "/survey/map_manifest_sha256"]
        manifest_identities = set()
        for connection, _timestamp, raw in reader.messages(
                connections=manifest_connections):
            message = reader.deserialize(raw, connection.msgtype)
            value = str(message.data).strip().lower()
            if len(value) != 64 or any(
                    char not in "0123456789abcdef" for char in value):
                raise RuntimeError(
                    "bag 中存在无效 canonical map manifest SHA-256")
            manifest_identities.add(value)
        status_connections = [
            connection for connection in reader.connections
            if connection.topic == "/survey/recording_status"]
        status_payloads = []
        for connection, _timestamp, raw in reader.messages(
                connections=status_connections):
            message = reader.deserialize(raw, connection.msgtype)
            status_payloads.append(str(message.data))
        completion = validate_recording_states(
            status_payloads, expected, expected_manifest)
        pose_source = completion["pose_source"]
        validate_recorded_slam_tf(reader)
        validate_recorded_camera_tf(reader, completion)
    if len(identities) != 1:
        raise RuntimeError("bag 必须且只能包含一种地图身份，实际为: %s" %
                           sorted(identities))
    recorded = next(iter(identities))
    if recorded != expected:
        raise RuntimeError(
            "bag 与指定地图不匹配: bag=%s map=%s" % (recorded, expected))
    if manifest_identities != {expected_manifest}:
        raise RuntimeError(
            "bag 与指定地图 YAML 的 canonical manifest 不匹配: bag=%s map=%s" %
            (sorted(manifest_identities), expected_manifest))
    print("[OK] bag topic 合同完整: %d topics" % len(topics))
    print("[OK] bag 与地图 SHA-256 一致:", expected)
    print("[OK] canonical map manifest SHA-256:", expected_manifest)
    print("[OK] 录制事务完整: token=%s" %
          str(completion.get("token", ""))[:12])
    print("[OK] pose source:", completion["pose_source"])


def check_model(model_name, allow_cpu=False, local_only=False):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print("torch=%s CUDA build=%s" % (torch.__version__, torch.version.cuda))
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "PyTorch 未检测到 CUDA；离线流程不建议误用 CPU，请安装 CUDA wheel")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[OK] GPU:", torch.cuda.get_device_name(0))
    kwargs = {"local_files_only": True} if local_only else {}
    started = time.time()
    from groundingdino_detect import (
        apply_camera_processor_size, validated_text_labels)
    processor = apply_camera_processor_size(
        AutoProcessor.from_pretrained(model_name, **kwargs))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name, **kwargs).to(device)
    model.eval()
    image = Image.new("RGB", (640, 480), color=(127, 127, 127))
    # One short phrase is enough to prove CUDA + postprocess; the joint
    # household prompt can OOM an 8 GiB laptop during this smoke test.
    smoke_queries = ["door"]
    inputs = processor(
        images=image, text=smoke_queries, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.35, text_threshold=0.25,
        target_sizes=[(480, 640)])
    if len(results) != 1:
        raise RuntimeError("model preflight expected exactly one result")
    # Exercise the same API-output contract as the production detector. In
    # particular, current LLMDet may emit text_labels=[''] for a valid
    # zero-box frame, which the detector normalizes to [].
    validated_text_labels(results[0])
    if device == "cuda":
        torch.cuda.synchronize()
    print("[OK] 开放词汇模型联合 prompt 推理成功: model=%s time=%.2fs" %
          (model_name, time.time() - started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", help="可选：同时检查一次式 gmapping bag")
    parser.add_argument("--map-file", help="bag 对应的 map_server YAML")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="仅用于功能验证；正式离线流程不建议")
    parser.add_argument(
        "--local-only", action="store_true",
        help="只使用已经缓存的 Hugging Face 模型")
    args = parser.parse_args()

    if sys.version_info < (3, 9):
        raise RuntimeError("电脑端需要 Python 3.9+")
    os.environ["HF_HOME"] = os.path.abspath(os.path.expanduser(args.hf_home))
    print("HF_HOME=%s" % os.environ["HF_HOME"])
    free_gib, disk_probe = free_space_gib(os.environ["HF_HOME"])
    if free_gib < 3.0:
        raise RuntimeError("HF_HOME 所在磁盘剩余空间不足 3 GiB")
    print("[OK] Python %s，HF_HOME 所在磁盘剩余 %.1f GiB (%s)" %
          (sys.version.split()[0], free_gib, disk_probe))
    if args.bag:
        if not args.map_file:
            parser.error("使用 --bag 时必须同时提供 --map-file")
        check_bag(args.bag, args.map_file)
    check_model(args.model, args.allow_cpu, args.local_only)
    print("\nPRECHECK PASS：电脑端离线语义检测环境可用")


if __name__ == "__main__":
    configure_utf8_stdio()
    try:
        main()
    except Exception as error:
        print("\nPRECHECK FAIL:", str(error), file=sys.stderr)
        sys.exit(1)
