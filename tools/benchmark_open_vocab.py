#!/usr/bin/env python3
"""Small repeatable GPU benchmark for Transformers open-vocabulary detectors."""

from __future__ import annotations

import argparse
import os
import time


DEFAULT_QUERIES = [
    "staircase",
    "vending machine",
    "water dispenser",
    "exit sign",
    "elevator door",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="iSEE-Laboratory/llmdet_large")
    parser.add_argument("--image", required=True)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", r"D:\hf_cache"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    os.environ["HF_HOME"] = os.path.abspath(os.path.expanduser(args.hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = "cuda"
    load_options = {"local_files_only": True} if args.local_only else {}
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, **load_options)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model, **load_options
    ).to(device)
    model.eval()
    torch.cuda.synchronize()
    print("model={} gpu={} load_s={:.3f}".format(
        args.model, torch.cuda.get_device_name(0), time.perf_counter() - started
    ), flush=True)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=DEFAULT_QUERIES, return_tensors="pt").to(device)
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]
        torch.cuda.synchronize()
        labels = [str(value) for value in result.get("text_labels", [])]
        scores = [round(float(value), 4) for value in result.get("scores", [])]
        print(
            "run={} inference_s={:.3f} detections={} labels={} scores={} peak_vram_gib={:.3f}".format(
                index + 1,
                time.perf_counter() - started,
                len(result.get("boxes", [])),
                labels,
                scores,
                torch.cuda.max_memory_allocated() / float(1024 ** 3),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
