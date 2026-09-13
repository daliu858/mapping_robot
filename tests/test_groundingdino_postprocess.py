"""开放词汇 POI 检测安全后处理回归测试。"""
from __future__ import print_function

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "groundingdino_detect.py")
SPEC = importlib.util.spec_from_file_location("groundingdino_detect", SCRIPT)
DETECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DETECTOR)


def detection(label, score, cx=100.0, cy=100.0, sx=50.0, sy=50.0):
    return {"label": label, "score": score, "cx": cx, "cy": cy,
            "sx": sx, "sy": sy}


class GroundingDinoPostprocessTest(unittest.TestCase):
    def test_formal_labels_and_query_aliases_are_stable(self):
        self.assertEqual(["chair", "table", "door"],
                         DETECTOR.CLASS_LABELS)
        self.assertEqual("chair", DETECTOR.canonical_poi_label("chair"))
        self.assertEqual("table", DETECTOR.canonical_poi_label("table"))
        self.assertEqual(
            "table", DETECTOR.canonical_poi_label("dining table"))
        self.assertEqual(
            "door", DETECTOR.canonical_poi_label("closed door"))

    def test_door_trial_poi_set_ignores_furniture(self):
        labels, mapping, queries = DETECTOR.resolve_poi_set("door")
        self.assertEqual(["door"], labels)
        self.assertEqual(["door", "closed door"], queries)
        self.assertEqual(
            "door", DETECTOR.canonical_poi_label("closed door", mapping))
        self.assertEqual("door", DETECTOR.canonical_poi_label("door", mapping))
        self.assertIsNone(DETECTOR.canonical_poi_label("chair", mapping))
        self.assertIsNone(DETECTOR.canonical_poi_label("table", mapping))

    def test_formal_default_is_llmdet_large(self):
        self.assertEqual(
            "iSEE-Laboratory/llmdet_large", DETECTOR.DEFAULT_MODEL)

    def test_detector_refuses_silent_large_model_cpu_fallback(self):
        with self.assertRaises(RuntimeError):
            DETECTOR.select_device(False, False)
        self.assertEqual("cpu", DETECTOR.select_device(False, True))
        self.assertEqual("cuda", DETECTOR.select_device(True, False))

    def test_cuda_keeps_jetbot_camera_resolution(self):
        self.assertEqual(
            {"shortest_edge": 480, "longest_edge": 640},
            DETECTOR.CAMERA_IMAGE_SIZE)
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"shortest_edge": 800}))
        DETECTOR.apply_camera_processor_size(processor)
        self.assertEqual(
            DETECTOR.CAMERA_IMAGE_SIZE, processor.image_processor.size)

    def test_missing_or_short_text_labels_fail_explicitly(self):
        with self.assertRaises(RuntimeError):
            DETECTOR.validated_text_labels({"boxes": [1], "scores": [0.5]})
        with self.assertRaises(RuntimeError):
            DETECTOR.validated_text_labels({
                "boxes": [1, 2], "scores": [0.5, 0.4],
                "text_labels": ["dining table"]})
        self.assertEqual(["dining table"], DETECTOR.validated_text_labels({
            "boxes": [1], "scores": [0.5],
            "text_labels": ["dining table"]}))

    def test_zero_box_placeholder_label_is_normalized(self):
        self.assertEqual([], DETECTOR.validated_text_labels({
            "boxes": [], "scores": [], "text_labels": [""]}))
        self.assertEqual([], DETECTOR.validated_text_labels({
            "boxes": [], "scores": []}))

    def test_bare_door_does_not_bypass_closed_door_query(self):
        self.assertIsNone(DETECTOR.canonical_poi_label("door"))

    def test_close_scores_on_same_box_are_rejected(self):
        items = [detection("chair", 0.51),
                 detection("door", 0.47, cx=100.5)]
        kept, rejected = DETECTOR.resolve_cross_class_ambiguity(items)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 2)

    def test_clear_winner_on_same_box_survives(self):
        winner = detection("table", 0.72)
        items = [winner, detection("door", 0.48, cx=100.5)]
        kept, rejected = DETECTOR.resolve_cross_class_ambiguity(items)
        self.assertEqual(kept, [winner])
        self.assertEqual(len(rejected), 1)

    def test_spatially_distinct_boxes_both_survive(self):
        items = [detection("chair", 0.51, cx=50.0),
                 detection("door", 0.50, cx=250.0)]
        kept, rejected = DETECTOR.resolve_cross_class_ambiguity(items)
        self.assertEqual(len(kept), 2)
        self.assertEqual(rejected, [])

    def test_raw_decode_accepts_only_exact_bgr8_or_rgb8(self):
        bgr = SimpleNamespace(
            encoding="bgr8", width=2, height=1, step=8,
            data=bytes([1, 2, 3, 4, 5, 6, 99, 99]))
        decoded = DETECTOR.decode_image(bgr, compressed=False)
        np.testing.assert_array_equal(
            decoded, np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))

        rgb = SimpleNamespace(
            encoding="rgb8", width=1, height=1, step=3,
            data=bytes([1, 2, 3]))
        np.testing.assert_array_equal(
            DETECTOR.decode_image(rgb, compressed=False),
            np.array([[[3, 2, 1]]], dtype=np.uint8))

        for encoding in ("mono8", "rgba8", "bgra8", "16UC1", "8UC3"):
            with self.subTest(encoding=encoding):
                message = SimpleNamespace(
                    encoding=encoding, width=1, height=1, step=4,
                    data=bytes([0, 0, 0, 0]))
                with self.assertRaises(RuntimeError):
                    DETECTOR.decode_image(message, compressed=False)

    def test_raw_decode_rejects_short_rows_and_truncated_data(self):
        short_step = SimpleNamespace(
            encoding="bgr8", width=2, height=1, step=5,
            data=bytes([0] * 6))
        with self.assertRaises(RuntimeError):
            DETECTOR.decode_image(short_step, compressed=False)
        truncated = SimpleNamespace(
            encoding="bgr8", width=2, height=2, step=6,
            data=bytes([0] * 11))
        with self.assertRaises(RuntimeError):
            DETECTOR.decode_image(truncated, compressed=False)

    def test_ud_orient_maps_boxes_back_to_bag_pixels(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[0, 0] = (1, 2, 3)
        flipped = DETECTOR.apply_image_orient(image, "ud")
        np.testing.assert_array_equal(flipped[3, 0], (1, 2, 3))
        self.assertEqual(
            DETECTOR.xyxy_to_source_frame(1, 0, 3, 1, 6, 4, "ud"),
            (1, 3, 3, 4))
        self.assertEqual(
            DETECTOR.xyxy_to_source_frame(1, 0, 3, 1, 6, 4, "none"),
            (1, 0, 3, 1))

    def test_corrupt_compressed_frame_is_terminal_not_silently_skipped(self):
        corrupt = SimpleNamespace(data=b"not-a-jpeg")
        with self.assertRaises(RuntimeError):
            DETECTOR.decode_image(corrupt, compressed=True)

    def test_missing_poi_does_not_replace_published_output(self):
        counts = {label: 1 for label in DETECTOR.CLASS_LABELS}
        counts["door"] = 0
        output = {"class_detection_counts": counts, "frames": []}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "detections.json"
            target.write_text("old-valid-result", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                DETECTOR.write_validated_output(target, output)
            self.assertEqual(
                "old-valid-result", target.read_text(encoding="utf-8"))
            self.assertEqual([], list(target.parent.glob("*.partial")))

    def test_partial_output_requires_opt_in_and_is_marked(self):
        counts = {label: 1 for label in DETECTOR.CLASS_LABELS}
        counts["door"] = 0
        output = {"class_detection_counts": counts, "frames": []}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "detections.json"
            missing = DETECTOR.write_validated_output(
                target, output, allow_partial_pois=True)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(["door"], missing)
            self.assertFalse(payload["complete"])
            self.assertEqual(["door"], payload["missing_poi_classes"])


if __name__ == "__main__":
    unittest.main()


class DoorV2AndVirtualLevelTest(unittest.TestCase):
    """door_v2 干扰类非对称裁决 + 虚拟平视内参修正的回归测试。"""

    def test_door_v2_poi_set_shape(self):
        labels, mapping, queries = DETECTOR.resolve_poi_set("door_v2")
        self.assertEqual("door", labels[0])
        self.assertTrue(all(l.startswith("_") for l in labels[1:]))
        self.assertEqual("door", DETECTOR.canonical_poi_label(
            "open doorway", mapping))
        self.assertEqual("_window", DETECTOR.canonical_poi_label(
            "window", mapping))
        # cabinet door 必须归 _furniture(长 query 优先),不能被 door 吃掉
        self.assertEqual("_furniture", DETECTOR.canonical_poi_label(
            "cabinet door", mapping))

    def test_tie_with_distractor_keeps_public_door(self):
        # 玻璃门场景:door 0.41 vs _window 0.44,差距 < margin=0.10
        # → 平局必须保 door,不能整组丢掉
        kept, rejected = DETECTOR.resolve_cross_class_ambiguity(
            [detection("door", 0.41), detection("_window", 0.44)],
            iou_threshold=0.80, score_margin=0.10)
        self.assertEqual(["door"], [item["label"] for item in kept])
        self.assertEqual(1, len(rejected))

    def test_distractor_clear_win_takes_the_box(self):
        # 干扰类赢出 margin:窗户就是窗户,door 框被抢走
        kept, rejected = DETECTOR.resolve_cross_class_ambiguity(
            [detection("door", 0.36), detection("_window", 0.55)],
            iou_threshold=0.80, score_margin=0.10)
        self.assertEqual(["_window"], [item["label"] for item in kept])
        self.assertEqual(1, len(rejected))

    def test_public_only_tie_still_rejected(self):
        # 老语义不变:两个公开类打平仍整组拒绝(chair vs table)
        kept, rejected = DETECTOR.resolve_cross_class_ambiguity(
            [detection("chair", 0.41), detection("table", 0.44)],
            iou_threshold=0.80, score_margin=0.10)
        self.assertEqual([], kept)
        self.assertEqual(2, len(rejected))

    def test_orient_intrinsics_flips_principal_point(self):
        K = np.array([[594.2, 0, 320.8], [0, 792.2, 235.2], [0, 0, 1.0]])
        K_ud = DETECTOR.orient_intrinsics(K, 640, 480, "ud")
        self.assertAlmostEqual(479 - 235.2, K_ud[1, 2])
        self.assertAlmostEqual(320.8, K_ud[0, 2])
        K_rot = DETECTOR.orient_intrinsics(K, 640, 480, "rot180")
        self.assertAlmostEqual(639 - 320.8, K_rot[0, 2])
        self.assertAlmostEqual(479 - 235.2, K_rot[1, 2])
        K_none = DETECTOR.orient_intrinsics(K, 640, 480, "none")
        self.assertTrue(np.allclose(K, K_none))

    def test_level_homography_roundtrip_and_horizon(self):
        K = np.array([[594.2, 0, 320.8], [0, 792.2, 235.2], [0, 0, 1.0]])
        H, H_inv = DETECTOR.build_level_homography(K, 15.0)
        # 地平线(cy + f·tanθ)必须映回主点行
        import cv2 as _cv2
        horizon_y = 235.2 + 792.2 * np.tan(np.radians(15.0))
        pt = np.array([[[320.8, horizon_y]]], np.float64)
        mapped = _cv2.perspectiveTransform(pt, H)[0, 0]
        self.assertAlmostEqual(235.2, mapped[1], places=6)
        # 框角点往返:warp_box_back(H⁻¹) 必须覆盖原框(保守外扩)
        x1, y1, x2, y2 = DETECTOR.warp_box_back(
            H_inv, *DETECTOR.warp_box_back(H, 100, 50, 500, 400))
        self.assertLessEqual(x1, 100 + 1e-6)
        self.assertLessEqual(y1, 50 + 1e-6)
        self.assertGreaterEqual(x2, 500 - 1e-6)
        self.assertGreaterEqual(y2, 400 - 1e-6)
