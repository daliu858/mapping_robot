# -*- coding: utf-8 -*-
"""在电脑端离线读取 ROS bag，用开放词汇模型检测三类导航 POI。

文件名为兼容旧命令保留；正式默认后端是 LLMDet-Large。
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


DEFAULT_HF_HOME = (
    os.environ.get("HF_HOME") or
    (r"D:\hf_cache" if os.path.isdir(r"D:\hf_cache") else
     os.path.join(os.path.expanduser("~"), ".cache", "huggingface")))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

POI_CLASSES = [
    ("chair", ["chair"]),
    ("table", ["table", "dining table"]),
    ("door", ["closed door"]),
]
# door_v2:门的多种表述 + 干扰类(下划线开头)。
# 干扰类只参与 cross-class ambiguity 竞争,把"开关面板/窗框/挂画"这类
# 误检从 door 手里抢走,随后在输出前被丢弃,不会进 JSON。
# bag20 人工复核显示误检 score(中位0.39)与真门(0.38)完全重叠,
# 分数阈值不可用;干扰类 + 几何门闩才是有效手段。
# 词表依据 bag20 60 帧误检逐张复核(swarm/KIMI_FP_TAXONOMY.md):
# A 类远处挂画/面板(80%)主杀靠 min-h,词表是双保险;
# B 类怼脸空白墙(几何闸全失效)只能靠 _wall 接走;
# 禁止加 "door frame" 之类含 door 的干扰短语——子串匹配会映射回 door。
DOOR_V2_CLASSES = [
    ("door", ["door", "closed door", "wooden door", "open doorway"]),
    ("_window", ["window"]),
    ("_switch", ["light switch", "wall socket", "doorbell", "thermostat",
                 "intercom", "control panel"]),
    ("_flat", ["picture frame", "poster", "television", "photo frame",
               "wall hanging", "wall decoration"]),
    ("_furniture", ["wardrobe", "cabinet", "curtain"]),
    ("_wall", ["wall", "blank wall", "white wall"]),
]
# hall_v1:大厅巡视四正主 + 干扰类。cabinet door 词组必须在 cabinet 类里,
# 靠子串匹配长词优先把柜门误火截胡回 cabinet(door1 橱柜冤案的疫苗)。
# 沙发盖着绗缝罩形似床,加 daybed 接住;故意不加 bed,防真床错标。
HALL_V1_CLASSES = [
    ("door", ["door", "closed door", "wooden door", "open doorway"]),
    ("sofa", ["sofa", "couch", "daybed"]),
    ("cabinet", ["blue cabinet", "sideboard", "cabinet", "cabinet door",
                 "cupboard"]),
    ("spray", ["spray can", "aerosol can", "spray bottle",
               "insecticide spray"]),
    ("_window", ["window"]),
    ("_switch", ["light switch", "wall socket", "doorbell", "thermostat",
                 "intercom", "control panel"]),
    ("_flat", ["picture frame", "poster", "television", "photo frame",
               "wall hanging", "wall decoration"]),
    ("_furniture", ["wardrobe", "curtain"]),
    ("_wall", ["wall", "blank wall", "white wall"]),
]
POI_SETS = {
    "household": POI_CLASSES,
    "door": [("door", ["door", "closed door"])],
    "door_v2": DOOR_V2_CLASSES,
    "hall_v1": HALL_V1_CLASSES,
}
CLASS_LABELS = [item[0] for item in POI_CLASSES]
POI_QUERY_TO_CLASS = {
    query.lower(): label
    for label, queries in POI_CLASSES for query in queries
}
JOINT_POI_QUERIES = list(POI_QUERY_TO_CLASS.keys())
PLATE_CAPTION = "door sign. room number plate. door number."
DEFAULT_MODEL = "iSEE-Laboratory/llmdet_large"
SUPPORTED_RAW_ENCODINGS = {"bgr8", "rgb8"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def stamp_to_ns(stamp):
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    return int(sec) * 1000000000 + int(nsec)


def resolve_poi_set(name):
    """Return labels, query map, and joint queries for one allowed POI table."""
    try:
        classes = POI_SETS[name]
    except KeyError:
        raise ValueError("unknown poi set: %s" % name)
    labels = [item[0] for item in classes]
    query_to_class = {
        query.lower(): label
        for label, queries in classes for query in queries
    }
    return labels, query_to_class, list(query_to_class.keys())


def canonical_poi_label(text_label, query_to_class=None):
    """把模型返回短语映射到当前 POI 表里的正式标签。"""
    mapping = POI_QUERY_TO_CLASS if query_to_class is None else query_to_class
    normalized = " ".join(str(text_label).lower().strip().split())
    if normalized in mapping:
        return mapping[normalized]
    for query in sorted(mapping, key=len, reverse=True):
        if query in normalized:
            return mapping[query]
    return None


def select_device(cuda_available, allow_cpu=False):
    if cuda_available:
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError(
        "CUDA 不可用；拒绝静默用 CPU 跑 LLMDet-Large。"
        "修复 GPU 环境，或仅在明确接受极慢速度时传 --allow-cpu")


# JetBot 双目是 640x480。GroundingDINO 默认 shortest_edge=800 会先
# 把图放大到 800x1066，8 GiB 笔记本在第一次前向就会 OOM。
CAMERA_IMAGE_SIZE = {"shortest_edge": 480, "longest_edge": 640}


def apply_camera_processor_size(processor):
    """Keep LLMDet at the recorded camera resolution instead of upscaling."""
    processor.image_processor.size = dict(CAMERA_IMAGE_SIZE)
    return processor


def validated_text_labels(result):
    """Return one textual label per box or fail on an API mismatch."""
    boxes = result.get("boxes", [])
    labels = result.get("text_labels")
    # transformers 5.12 + LLMDet may return the harmless placeholder ['']
    # when there are no boxes. Normalize every zero-box result to the only
    # semantically valid label list, while keeping nonzero-box mismatches
    # strictly fail-closed.
    if len(boxes) == 0:
        return []
    if labels is None:
        raise RuntimeError(
            "模型后处理未返回 text_labels；当前 Transformers/模型 API 不兼容")
    if len(labels) != len(boxes):
        raise RuntimeError(
            "模型后处理 boxes/text_labels 数量不一致: %d != %d" %
            (len(boxes), len(labels)))
    textual = [str(value) for value in labels]
    if any(not value.strip() for value in textual):
        raise RuntimeError("模型后处理返回了空 text_labels")
    return textual


def recording_identity(reader):
    """Extract and validate the checked survey identity from an open bag."""
    from offline_preflight import (validate_recorded_camera_tf,
                                   validate_recording_states)

    status_connections = [
        connection for connection in reader.connections
        if connection.topic == "/survey/recording_status"]
    hash_connections = [
        connection for connection in reader.connections
        if connection.topic == "/survey/map_image_sha256"]
    manifest_connections = [
        connection for connection in reader.connections
        if connection.topic == "/survey/map_manifest_sha256"]
    status_payloads = []
    for connection, _timestamp, raw in reader.messages(
            connections=status_connections):
        message = reader.deserialize(raw, connection.msgtype)
        status_payloads.append(str(message.data))
    completion = validate_recording_states(status_payloads)
    validate_recorded_camera_tf(reader, completion)
    map_hashes = set()
    for connection, _timestamp, raw in reader.messages(
            connections=hash_connections):
        message = reader.deserialize(raw, connection.msgtype)
        map_hashes.add(str(message.data).strip().lower())
    if len(map_hashes) != 1:
        raise RuntimeError("bag 必须且只能包含一个地图 SHA-256")
    map_hash = next(iter(map_hashes))
    if completion["map_image_sha256"].lower() != map_hash:
        raise RuntimeError("录制事务与 map identity topic 不一致")
    manifest_hashes = set()
    for connection, _timestamp, raw in reader.messages(
            connections=manifest_connections):
        message = reader.deserialize(raw, connection.msgtype)
        manifest_hashes.add(str(message.data).strip().lower())
    if len(manifest_hashes) != 1:
        raise RuntimeError("bag 必须且只能包含一个 canonical map manifest")
    manifest_hash = next(iter(manifest_hashes))
    if completion["map_manifest_sha256"].lower() != manifest_hash:
        raise RuntimeError("录制事务与 map manifest identity topic 不一致")
    return (str(completion["token"]), map_hash, manifest_hash,
            list(completion["camera_mount"]))


def detection_iou(left, right):
    """计算两个中心点/宽高格式检测框的 IoU。"""
    lx1 = float(left["cx"]) - float(left["sx"]) / 2.0
    ly1 = float(left["cy"]) - float(left["sy"]) / 2.0
    lx2 = float(left["cx"]) + float(left["sx"]) / 2.0
    ly2 = float(left["cy"]) + float(left["sy"]) / 2.0
    rx1 = float(right["cx"]) - float(right["sx"]) / 2.0
    ry1 = float(right["cy"]) - float(right["sy"]) / 2.0
    rx2 = float(right["cx"]) + float(right["sx"]) / 2.0
    ry2 = float(right["cy"]) + float(right["sy"]) / 2.0
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * \
        max(0.0, min(ly2, ry2) - max(ly1, ry1))
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def nms_detections(detections, iou_threshold=0.50):
    """同类别 NMS。"""
    remaining = sorted(
        detections, key=lambda item: float(item["score"]), reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [item for item in remaining
                     if detection_iou(best, item) <= iou_threshold]
    return kept


def resolve_cross_class_ambiguity(detections, iou_threshold=0.80,
                                  score_margin=0.10):
    """拒绝同一提议框被解码为多个 POI 且没有明确赢家的情况。"""
    count = len(detections)
    parents = list(range(count))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(count):
        for right in range(left + 1, count):
            if detections[left]["label"] == detections[right]["label"]:
                continue
            if detection_iou(detections[left], detections[right]) >= \
                    float(iou_threshold):
                union(left, right)

    groups = {}
    for index, item in enumerate(detections):
        groups.setdefault(find(index), []).append(item)

    kept, rejected = [], []
    for group in groups.values():
        labels = set(item["label"] for item in group)
        if len(labels) == 1:
            kept.extend(group)
            continue
        best_per_label = [
            max((item for item in group if item["label"] == label),
                key=lambda item: float(item["score"]))
            for label in labels
        ]
        best_per_label.sort(
            key=lambda item: float(item["score"]), reverse=True)
        public = [item for item in best_per_label
                  if not item["label"].startswith("_")]
        distractors = [item for item in best_per_label
                       if item["label"].startswith("_")]
        if public and distractors:
            # 非对称裁决:真假门 score 分布重叠,平局是常态。干扰类必须
            # 赢出 score_margin 才有资格把公开类(door)抢走;打平或
            # 公开类领先时保留最优公开类。否则玻璃门 vs window 这种
            # 平局会把真门连框带走。
            best_pub, best_dis = public[0], distractors[0]
            if float(best_dis["score"]) - float(best_pub["score"]) >= \
                    float(score_margin):
                winner = best_dis   # 干扰类明确获胜;随后会被输出过滤丢弃
            else:
                winner = best_pub
            kept.append(winner)
            rejected.extend(item for item in group if item is not winner)
            continue
        winner = best_per_label[0]
        runner_up = best_per_label[1]
        if float(winner["score"]) - float(runner_up["score"]) >= \
                float(score_margin):
            kept.append(winner)
            rejected.extend(item for item in group if item is not winner)
        else:
            rejected.extend(group)
    return kept, rejected


def run_gd(processor, model, image, caption, device, box_threshold,
           text_threshold, image_size):
    import torch

    inputs = processor(
        images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[image_size])[0]
    detections = []
    for box, score in zip(result["boxes"], result["scores"]):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        detections.append((x1, y1, x2, y2, float(score)))
    return detections


def run_gd_labeled(processor, model, image, captions, device,
                   box_threshold, text_threshold, image_size):
    import torch

    inputs = processor(
        images=image, text=captions, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[image_size])[0]
    detections = []
    text_labels = validated_text_labels(result)
    for box, score, text_label in zip(
            result["boxes"], result["scores"], text_labels):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        detections.append(
            (x1, y1, x2, y2, float(score), str(text_label)))
    return detections


def ocr_plate(reader, image_bgr, cx, cy, sx, sy, width, height):
    if reader is None:
        return None
    x1, y1 = max(0, int(cx - sx / 2)), max(0, int(cy - sy / 2))
    x2, y2 = min(width, int(cx + sx / 2)), min(height, int(cy + sy / 2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    try:
        candidates = reader.readtext(image_bgr[y1:y2, x1:x2])
    except Exception:
        return None
    best_text, best_confidence = None, 0.0
    for _box, text, confidence in candidates:
        cleaned = (text or "").strip()
        if cleaned and any(char.isalnum() for char in cleaned) and \
                confidence > best_confidence:
            best_text, best_confidence = cleaned, float(confidence)
    return best_text


def apply_image_orient(image, orient):
    """Flip bag pixels into the upright frame LLMDet was trained on.

    Boxes must be mapped back with ``xyxy_to_source_frame`` so JSON stays
    in raw bag coordinates for stereo fusion.
    """
    if orient in (None, "", "none"):
        return image
    if orient == "ud":
        return cv2.flip(image, 0)
    if orient == "lr":
        return cv2.flip(image, 1)
    if orient == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError("unknown image orient: %r" % orient)


def xyxy_to_source_frame(x1, y1, x2, y2, width, height, orient):
    """Map a box from the oriented detect frame back to bag pixels."""
    if orient in (None, "", "none"):
        return x1, y1, x2, y2
    if orient == "ud":
        return x1, height - y2, x2, height - y1
    if orient == "lr":
        return width - x2, y1, width - x1, y2
    if orient == "rot180":
        return width - x2, height - y2, width - x1, height - y1
    raise ValueError("unknown image orient: %r" % orient)


def read_camera_intrinsics(reader, img_topic):
    """从 bag 里 img_topic 对应的 camera_info 读 K。缺失时大声失败。"""
    info_topic = img_topic
    for suffix in ("/image_raw/compressed", "/image_raw"):
        if info_topic.endswith(suffix):
            info_topic = info_topic[:-len(suffix)] + "/camera_info"
            break
    connections = [c for c in reader.connections if c.topic == info_topic]
    if not connections:
        raise RuntimeError(
            "--virtual-level-pitch 需要 %s,但 bag 里没有" % info_topic)
    _, _, raw = next(reader.messages(connections=connections))
    message = reader.deserialize(raw, connections[0].msgtype)
    K = np.array(message.K, np.float64).reshape(3, 3)
    return K, int(message.width), int(message.height)


def orient_intrinsics(K, width, height, orient):
    """warp 发生在 orient 翻转之后,主点必须跟着翻,否则旋转中心错位。"""
    K = K.copy()
    if orient == "ud":
        K[1, 2] = (height - 1) - K[1, 2]
    elif orient == "lr":
        K[0, 2] = (width - 1) - K[0, 2]
    elif orient == "rot180":
        K[0, 2] = (width - 1) - K[0, 2]
        K[1, 2] = (height - 1) - K[1, 2]
    return K


def build_level_homography(K, pitch_up_deg):
    """虚拟平视:把上仰 pitch 的相机画面单应变换到水平视轴的虚拟相机。

    光学系约定 x右/y下/z前。相机上仰 θ 时,水平线出现在画面中心下方
    f·tanθ 处;H = K·Rx(θ)·K⁻¹ 把它搬回画面中心,竖直物体(门)恢复成
    接近竖直的矩形——LLMDet 训练分布内的形态。画面下缘会出现无数据黑区,
    这是诚实的:相机确实没拍到那里。"""
    theta = np.radians(float(pitch_up_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)
    H = K @ rot_x @ np.linalg.inv(K)
    return H, np.linalg.inv(H)


def warp_box_back(H_inv, x1, y1, x2, y2):
    """把虚拟平视画面里的框经逆单应变换映回(orient 后的)原像素系。"""
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                       np.float64).reshape(-1, 1, 2)
    back = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
    return (float(back[:, 0].min()), float(back[:, 1].min()),
            float(back[:, 0].max()), float(back[:, 1].max()))


def sharpness_of(image_bgr):
    """画面中央 60% 的 Laplacian 方差;怼墙糊帧会非常低。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop = gray[h // 5:h * 4 // 5, w // 5:w * 4 // 5]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def decode_image(message, compressed):
    if compressed:
        buffer = np.frombuffer(bytes(message.data), np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                "压缩检测源图像无法解码；正式检测拒绝静默跳帧")
        return image

    encoding = str(getattr(message, "encoding", "")).strip().lower()
    if encoding not in SUPPORTED_RAW_ENCODINGS:
        raise RuntimeError(
            "不支持 raw 图像编码 %r；仅接受 bgr8/rgb8，拒绝猜测像素布局" %
            encoding)

    channels = 3
    width = int(getattr(message, "width", 0))
    height = int(getattr(message, "height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError("raw 检测源图像尺寸无效")
    packed_row_bytes = width * channels
    row_bytes = int(getattr(message, "step", packed_row_bytes))
    if row_bytes < packed_row_bytes:
        raise RuntimeError("raw 检测源图像行跨度短于有效像素")
    array = np.frombuffer(bytes(message.data), np.uint8)
    if array.size < height * row_bytes:
        raise RuntimeError("raw 检测源图像数据被截断")
    rows = array[:height * row_bytes].reshape(height, row_bytes)
    image = rows[:, :packed_row_bytes].reshape(height, width, channels)
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _atomic_write_json(path, payload):
    """Durably replace a JSON artifact without exposing a partial file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        prefix=target.name + ".", suffix=".partial",
        dir=str(target.parent), delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def write_validated_output(path, output, allow_partial_pois=False,
                           class_labels=None):
    """Publish only complete POI results unless partial mode is explicit."""
    labels = list(CLASS_LABELS if class_labels is None else class_labels)
    counts = output.get("class_detection_counts", {})
    missing = [label for label in labels
               if int(counts.get(label, 0)) <= 0]
    if missing and not allow_partial_pois:
        raise RuntimeError(
            "检测结果不完整，完全缺少 POI 类别: %s" % ", ".join(missing))

    payload = dict(output)
    payload["complete"] = not missing
    payload["missing_poi_classes"] = missing
    _atomic_write_json(path, payload)
    return missing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="输入 ROS1 bag")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--local-only", action="store_true", help="仅使用本地模型缓存")
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="显式允许无 CUDA 时用 CPU；LLMDet-Large 会非常慢")
    parser.add_argument(
        "--img-topic", default="/stereo/left/image_raw/compressed")
    parser.add_argument(
        "--rectified", action="store_true",
        help="输入图片已经校正；检测框将按校正图坐标解释")
    parser.add_argument("--out", default="detections.json")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--box-th", type=float, default=0.35)
    parser.add_argument("--text-th", type=float, default=0.25)
    parser.add_argument("--cross-class-iou", type=float, default=0.80)
    parser.add_argument("--class-margin", type=float, default=0.10)
    parser.add_argument("--plate-box-th", type=float, default=0.30)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument(
        "--poi-set", choices=sorted(POI_SETS), default="door",
        help="household=chair/table/door；door=这次只把门投到地图上")
    parser.add_argument(
        "--skip-doorplates", action="store_true",
        help="跳过门牌检测和 OCR，最快完成所选 POI")
    parser.add_argument(
        "--allow-partial-pois", action="store_true",
        help="仅调试：即使某类 POI 完全没有检测也返回成功")
    parser.add_argument(
        "--image-orient", choices=("none", "ud", "lr", "rot180"), default="ud",
        help="读 bag 后、进 LLMDet 前的朝向。现有 flip_method=0 的包用 ud"
             "（上下翻转）；框会写回 bag 像素坐标。新包若采集已翻转则传 none")
    parser.add_argument(
        "--virtual-level-pitch", type=float, default=0.0,
        help="相机实测上仰角(度,tools/pitch_timeline.py 量出来的)。"
             "非 0 时进模型前做虚拟平视单应变换,框会逆变换写回原坐标。")
    parser.add_argument(
        "--pitch-segments", default="",
        help='分段仰角,用于录制中途支架被动过的 bag。格式 "0:5.8,140:13.2"'
             "(bag 相对秒:上仰角度),按时间取最后一个生效段。"
             "与 --virtual-level-pitch 互斥。")
    parser.add_argument(
        "--door-min-aspect", type=float, default=0.0,
        help="door 框的最小高宽比(sy/sx)。真门竖长;开关/窗条被拒。"
             "建议 1.2;默认 0 = 关闭,兼容旧命令")
    parser.add_argument(
        "--door-min-h-px", type=float, default=0.0,
        help="door 框的最小像素高。bag20 复核:真门中位 350px,误检 41px。"
             "建议 120;默认 0 = 关闭(融合端另有 min_structure_bbox_h_px)")
    parser.add_argument(
        "--min-sharpness", type=float, default=0.0,
        help="帧中央 Laplacian 方差低于此值判为糊帧(怼墙/剧烈运动),"
             "整帧跳过检测但保留空帧记录。建议 40;默认 0 = 关闭")
    parser.add_argument(
        "--emit-distractors", action="store_true",
        help="干扰类完成抢框后不丢弃,去掉下划线作为正式 POI 写入 JSON"
             "(_wall 仍丢弃——满屋都是墙)。融合端会按同一套门闩独立"
             "确认它们;completeness 检查仍只要求原公开类。")
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("--stride 必须是正整数")
    if not 0.0 <= args.cross_class_iou <= 1.0:
        parser.error("--cross-class-iou 必须在 [0,1] 内")
    if args.class_margin < 0.0:
        parser.error("--class-margin 不能为负数")
    if args.rectified and (args.virtual_level_pitch != 0.0 or
                           args.pitch_segments):
        parser.error(
            "--rectified 与虚拟平视不能同用:校正图的内参是 "
            "P 阵新 K,而本工具读的是原始 K,建出的单应会是错的")
    if args.pitch_segments and args.virtual_level_pitch != 0.0:
        parser.error("--pitch-segments 与 --virtual-level-pitch 互斥")
    if args.pitch_segments:
        try:
            segments = []
            for part in args.pitch_segments.split(","):
                t0, deg = part.split(":")
                segments.append((float(t0), float(deg)))
            if segments != sorted(segments) or not segments:
                raise ValueError
            args.pitch_segments = segments
        except ValueError:
            parser.error('--pitch-segments 格式应为 "0:5.8,140:13.2" 且按时间升序')
    else:
        args.pitch_segments = []
    return args


def main():
    args = parse_args()
    os.environ["HF_HOME"] = os.path.abspath(os.path.expanduser(args.hf_home))
    class_labels, query_to_class, poi_queries = resolve_poi_set(args.poi_set)
    # 下划线开头的是干扰类:参与检测与歧义竞争,默认不进输出、不算完整性;
    # --emit-distractors 时(除 _wall 外)去掉下划线作为正式 POI 输出
    public_labels = [l for l in class_labels if not l.startswith("_")]
    out_labels = list(public_labels)
    if args.emit_distractors:
        out_labels += [l[1:] for l in class_labels
                       if l.startswith("_") and l != "_wall"]

    try:
        import torch
        from PIL import Image as PILImage
        from transformers import (
            AutoModelForZeroShotObjectDetection, AutoProcessor)
        from rosbags_index_compat import patch_rosbags_index_headers
        patch_rosbags_index_headers()
        from rosbags.highlevel import AnyReader
    except Exception as error:
        sys.exit("缺少离线检测依赖: %r" % error)

    device = select_device(torch.cuda.is_available(), args.allow_cpu)
    print("device =", device, "| model =", args.model,
          "| poi-set =", args.poi_set, "| POI queries =", poi_queries,
          "| image-orient =", args.image_orient)
    load_options = {"local_files_only": True} if args.local_only else {}
    processor = apply_camera_processor_size(
        AutoProcessor.from_pretrained(args.model, **load_options))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model, **load_options).to(device)
    model.eval()

    ocr_reader = None
    if not args.no_ocr and not args.skip_doorplates:
        try:
            import easyocr
            ocr_reader = easyocr.Reader(["en"], gpu=(device == "cuda"))
        except Exception as error:
            print("OCR 未启用：%r" % error)

    frames, door_plates = [], []
    source_stamps = set()
    processed_size = None
    seen = processed = detection_count = plate_count = rejected_count = 0
    bag = Path(args.bag)
    source_bag_sha256 = file_sha256(bag)
    level_H = level_H_inv = None
    blur_skipped = distractor_dropped = door_gate_rejected = 0
    with AnyReader([bag]) as reader:
        (recording_token, map_hash, map_manifest_hash,
         camera_mount) = recording_identity(reader)
        pitch_plan = []          # [(起始秒, H, H_inv)],按时间升序
        if args.virtual_level_pitch != 0.0 or args.pitch_segments:
            K, cam_w, cam_h = read_camera_intrinsics(reader, args.img_topic)
            K = orient_intrinsics(K, cam_w, cam_h, args.image_orient)
            segs = (args.pitch_segments or
                    [(0.0, args.virtual_level_pitch)])
            for t0, deg in segs:
                H, H_inv = build_level_homography(K, deg)
                pitch_plan.append((t0, deg, H, H_inv))
            print("virtual-level pitch plan =",
                  [(t0, deg) for t0, deg, _, _ in pitch_plan],
                  "| K fx=%.1f fy=%.1f" % (K[0, 0], K[1, 1]))
        bag_start_ns = reader.start_time
        connections = [connection for connection in reader.connections
                       if connection.topic == args.img_topic]
        if not connections:
            sys.exit("bag 中没有图像 topic %s；实际 topic: %s" %
                     (args.img_topic,
                      sorted({item.topic for item in reader.connections})))
        compressed = "CompressedImage" in connections[0].msgtype
        for connection, _timestamp, raw in reader.messages(
                connections=connections):
            seen += 1
            if pitch_plan:
                t_rel = (_timestamp - bag_start_ns) / 1e9
                for t0, _deg, H, H_inv in pitch_plan:
                    if t_rel >= t0:
                        level_H, level_H_inv = H, H_inv
            message = reader.deserialize(raw, connection.msgtype)
            stamp_ns = stamp_to_ns(message.header.stamp)
            if stamp_ns <= 0:
                raise RuntimeError("检测源图像含非正时间戳")
            if stamp_ns in source_stamps:
                raise RuntimeError(
                    "检测源图像含重复时间戳: %d" % stamp_ns)
            source_stamps.add(stamp_ns)
            if (seen - 1) % args.stride:
                continue
            image_src = decode_image(message, compressed)
            image_bgr = apply_image_orient(image_src, args.image_orient)
            height, width = image_bgr.shape[:2]
            if processed_size is None:
                processed_size = (width, height)
            elif processed_size != (width, height):
                raise RuntimeError("同一录包中的检测图像尺寸发生变化")
            if args.min_sharpness > 0.0 and \
                    sharpness_of(image_bgr) < args.min_sharpness:
                # 糊帧(怼墙/急转):跳过检测但保留空帧,时间戳记账不受影响
                frames.append({
                    "stamp_ns": stamp_ns, "w": width, "h": height,
                    "dets": [], "ambiguous_rejected": 0,
                    "skipped_blur": True,
                })
                processed += 1
                blur_skipped += 1
                continue
            if level_H is not None:
                image_bgr = cv2.warpPerspective(
                    image_bgr, level_H, (width, height),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            image_pil = PILImage.fromarray(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            by_class = {label: [] for label in class_labels}
            proposals = run_gd_labeled(
                processor, model, image_pil, poi_queries, device,
                args.box_th, args.text_th, (height, width))
            for x1, y1, x2, y2, score, text_label in proposals:
                label = canonical_poi_label(text_label, query_to_class)
                if label is None:
                    continue
                if level_H_inv is not None:
                    x1, y1, x2, y2 = warp_box_back(
                        level_H_inv, x1, y1, x2, y2)
                x1, y1, x2, y2 = xyxy_to_source_frame(
                    x1, y1, x2, y2, width, height, args.image_orient)
                x1 = max(0.0, min(float(width), float(x1)))
                y1 = max(0.0, min(float(height), float(y1)))
                x2 = max(0.0, min(float(width), float(x2)))
                y2 = max(0.0, min(float(height), float(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                by_class[label].append({
                    "cls": class_labels.index(label),
                    "label": label,
                    "score": round(score, 4),
                    "cx": float((x1 + x2) / 2.0),
                    "cy": float((y1 + y2) / 2.0),
                    "sx": float(x2 - x1),
                    "sy": float(y2 - y1),
                })
            detections = []
            for label in class_labels:
                detections.extend(nms_detections(by_class[label]))
            detections, rejected = resolve_cross_class_ambiguity(
                detections, args.cross_class_iou, args.class_margin)
            rejected_count += len(rejected)
            # 干扰类完成"抢框"使命后丢弃,不进输出
            kept = []
            for item in detections:
                if item["label"].startswith("_"):
                    if not args.emit_distractors or item["label"] == "_wall":
                        distractor_dropped += 1
                        continue
                    item["label"] = item["label"][1:]
                if item["label"] == "door":
                    if args.door_min_h_px > 0.0 and \
                            item["sy"] < args.door_min_h_px:
                        door_gate_rejected += 1
                        continue
                    # aspect 闸只对四边都不触画面边界的框生效:怼脸真门
                    # 被裁剪后宽高比失真(实测 0.68~1.09),触边框的比例
                    # 不可信,不能拿来杀
                    touches_border = (
                        item["cx"] - item["sx"] / 2 <= 1.0 or
                        item["cy"] - item["sy"] / 2 <= 1.0 or
                        item["cx"] + item["sx"] / 2 >= width - 1.0 or
                        item["cy"] + item["sy"] / 2 >= height - 1.0)
                    if args.door_min_aspect > 0.0 and not touches_border \
                            and item["sy"] < args.door_min_aspect * item["sx"]:
                        door_gate_rejected += 1
                        continue
                item["cls"] = out_labels.index(item["label"])
                kept.append(item)
            detections = kept
            frames.append({
                "stamp_ns": stamp_ns,
                "w": width,
                "h": height,
                "dets": detections,
                "ambiguous_rejected": len(rejected),
            })
            processed += 1
            detection_count += len(detections)

            if not args.skip_doorplates:
                plate_proposals = run_gd(
                    processor, model, image_pil, PLATE_CAPTION, device,
                    args.plate_box_th, args.text_th, (height, width))
                for x1, y1, x2, y2, score in plate_proposals:
                    if level_H_inv is not None:
                        x1, y1, x2, y2 = warp_box_back(
                            level_H_inv, x1, y1, x2, y2)
                    x1, y1, x2, y2 = xyxy_to_source_frame(
                        x1, y1, x2, y2, width, height, args.image_orient)
                    x1 = max(0.0, min(float(width), float(x1)))
                    y1 = max(0.0, min(float(height), float(y1)))
                    x2 = max(0.0, min(float(width), float(x2)))
                    y2 = max(0.0, min(float(height), float(y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    sx, sy = x2 - x1, y2 - y1
                    door_plates.append({
                        "stamp_ns": stamp_ns,
                        "score": round(score, 4),
                        "cx": round(cx, 1), "cy": round(cy, 1),
                        "sx": round(sx, 1), "sy": round(sy, 1),
                        "text": ocr_plate(
                            ocr_reader, image_src, cx, cy, sx, sy,
                            width, height),
                    })
                    plate_count += 1
            if processed % 50 == 0:
                print("processed=%d, POI=%d, ambiguous_rejected=%d, plates=%d" %
                      (processed, detection_count, rejected_count, plate_count))

    if file_sha256(bag) != source_bag_sha256:
        raise RuntimeError("bag bytes changed while offline detection was running")

    output = {
        "schema": "slam-car-detections-v1",
        "classes": out_labels,
        "image_topic": args.img_topic,
        "detections_rectified": bool(args.rectified),
        "image_orient": args.image_orient,
        "virtual_level_pitch_deg": args.virtual_level_pitch,
        "virtual_level_pitch_segments": [
            [t0, deg] for t0, deg in (args.pitch_segments or [])],
        "door_min_aspect": args.door_min_aspect,
        "door_min_h_px": args.door_min_h_px,
        "min_sharpness": args.min_sharpness,
        "blur_skipped_frames": blur_skipped,
        "distractor_dropped_total": distractor_dropped,
        "door_gate_rejected_total": door_gate_rejected,
        "backend": "transformers/" + args.model,
        "recording_token": recording_token,
        "map_image_sha256": map_hash,
        "map_manifest_sha256": map_manifest_hash,
        "source_bag_sha256": source_bag_sha256,
        "camera_mount": camera_mount,
        "mode": "joint-caption-nms-cross-class-ambiguity-gate",
        "poi_set": args.poi_set,
        "poi_queries": poi_queries,
        "cross_class_iou": args.cross_class_iou,
        "class_margin": args.class_margin,
        "ambiguous_rejected_total": rejected_count,
        "frames": frames,
        "door_plates": door_plates,
    }
    class_counts = {
        label: sum(1 for frame in frames for item in frame["dets"]
                   if item["label"] == label)
        for label in out_labels
    }
    output["class_detection_counts"] = class_counts
    try:
        missing = write_validated_output(
            args.out, output, args.allow_partial_pois,
            class_labels=public_labels)
    except RuntimeError as error:
        sys.exit(str(error))
    if missing:
        print("警告：调试模式发布了不完整 POI 结果，缺少: %s" %
              ", ".join(missing), file=sys.stderr)
    print("done: seen=%d processed=%d POI=%d ambiguous_rejected=%d "
          "distractor_dropped=%d door_gate_rejected=%d blur_skipped=%d "
          "plates=%d -> %s" %
          (seen, processed, detection_count, rejected_count,
           distractor_dropped, door_gate_rejected, blur_skipped,
           plate_count, args.out))


if __name__ == "__main__":
    configure_utf8_stdio()
    main()
