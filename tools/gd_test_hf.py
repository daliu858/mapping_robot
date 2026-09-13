# -*- coding: utf-8 -*-
r"""
gd_test_hf.py — 用 HuggingFace Transformers 开放词汇模型测试 POI 检测
(Windows/4060 友好: 纯 PyTorch, 无需编译 CUDA 算子; 模型与缓存放 D 盘, 不碰 C 盘)

对一个图片文件夹跑开放词汇零样本检测, 画框 + 打印, 看 5 类导航 POI 认得怎么样。
不依赖模型的独立源码仓库，不依赖 ROS。纯本地测试。

用法(在 D 盘 venv 里跑):
  D:\gd_env\Scripts\python.exe "D:\slam car\tools\gd_test_hf.py" \
      --images "D:\slam car\test_photos" --out "D:\slam car\gd_test_out"
  指定类别(逗号分隔): --classes "staircase,exit sign,elevator door"
"""
import os
# 模型缓存放 D 盘(C 盘快满)——必须在 import transformers 之前设
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
import argparse, glob, sys
import numpy as np
import cv2

# 导航 POI；正式后端使用 list prompt，让返回的 text_labels 可稳定映射。
DEFAULT_CLASSES = ["staircase", "vending machine", "water dispenser",
                   "exit sign", "elevator door", "door sign"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片文件或文件夹")
    ap.add_argument("--classes", default="", help="逗号分隔, 覆盖默认 5 类")
    ap.add_argument("--model", default="iSEE-Laboratory/llmdet_large")
    ap.add_argument("--out", default="gd_test_out")
    ap.add_argument("--box-th", type=float, default=0.35)
    ap.add_argument("--text-th", type=float, default=0.25)
    a = ap.parse_args()

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except Exception as ex:
        sys.exit("[依赖缺失] %r" % ex)

    classes = [c.strip() for c in a.classes.split(",") if c.strip()] or DEFAULT_CLASSES
    text = [c.lower() for c in classes]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device, "| model =", a.model)
    print("prompt =", text)

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to(device)

    if os.path.isdir(a.images):
        files = []
        for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            files += glob.glob(os.path.join(a.images, e))
            files += glob.glob(os.path.join(a.images, e.upper()))
        files = sorted(set(files))
    else:
        files = [a.images]
    if not files:
        sys.exit("没找到图片: %s" % a.images)
    if not os.path.isdir(a.out):
        os.makedirs(a.out)

    total = 0
    for fp in files:
        try:
            pil = Image.open(fp).convert("RGB")
        except Exception:
            print("跳过(打不开):", fp); continue
        inputs = proc(images=pil, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=a.box_th, text_threshold=a.text_th,   # transformers 5.x: box_threshold 改名 threshold
            target_sizes=[pil.size[::-1]])[0]   # (h, w)
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        labs = res.get("text_labels", res.get("labels"))   # 新版 text_labels / 旧版 labels
        hits = []
        for box, score, lab in zip(res["boxes"], res["scores"], labs):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tag = "%s %.2f" % (lab, float(score))
            cv2.putText(img, tag, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            hits.append("%s(%.2f)" % (lab, float(score)))
        total += len(hits)
        outp = os.path.join(a.out, os.path.basename(fp))
        ext = os.path.splitext(outp)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(outp)   # 绕过 cv2.imwrite 不支持中文路径(你的文件名是中文)
        print("%-34s 检出 %2d: %s" % (os.path.basename(fp), len(hits), ", ".join(hits) or "-"))

    print("\n完成: %d 图, 共 %d 框 -> 打开 %s 看画框效果。" % (len(files), total, a.out))
    if total == 0:
        print("⚠️ 一个都没检出 —— 可能阈值太高/图不清/prompt 不匹配, 调 --box-th 0.25 再试。")


if __name__ == "__main__":
    main()
