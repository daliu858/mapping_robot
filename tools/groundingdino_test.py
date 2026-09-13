# -*- coding: utf-8 -*-
"""
groundingdino_test.py — 在家快速验证 GroundingDINO 装好且能正常检测(4060, py3)

对一批照片(你在家随手拍的)跑 GroundingDINO, 用一组"家里常见物体"当 prompt,
画框存可视化图 + 打印检出。看到框画对了, 就说明 GroundingDINO 环境 OK,
之后才好放心跑正式的 POI 检测(groundingdino_detect.py)。

【在家怎么拍】把这些往桌上一摆 / 对着房间拍几张就行(都是 GroundingDINO 零样本
必然认得很好的): 椅子、笔记本、杯子、水瓶、显示器、键盘、书、背包、绿植、手机、门。
拍 5~10 张, 角度/远近混着来, 更能看出它稳不稳。

【依赖】(4060 PC, 不需要 ROS)
  pip install torch torchvision opencv-python pillow numpy
  + GroundingDINO(https://github.com/IDEA-Research/GroundingDINO)+ 权重/config

【用法】
  python groundingdino_test.py --images ./home_photos \
      --config GroundingDINO_SwinT_OGC.py --weights groundingdino_swint_ogc.pth \
      --out ./home_test_out
  跑完打开 ./home_test_out 看画了框的图。

【怎么算"能用"】
  - 大部分物体被框对、标签对、分数 >0.35 -> 环境 OK, 可以跑正式 POI 检测。
  - 一个都没框 / 报错 -> 多半是权重没加载/config 路径错/CUDA 没装好, 先弄通环境。
"""
import argparse, os, glob, sys
import numpy as np
import cv2


def _configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_stdio()

# 家里常见、GroundingDINO 零样本必然认得好的类, 仅用于"环境自检", 与正式 POI 无关
HOME_TEST_CLASSES = ["chair", "laptop", "cup", "bottle", "monitor",
                     "keyboard", "book", "backpack", "potted plant",
                     "cell phone", "door"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片文件或文件夹")
    ap.add_argument("--config", required=True, help="GroundingDINO config .py")
    ap.add_argument("--weights", required=True, help="GroundingDINO 权重 .pth")
    ap.add_argument("--out", default="home_test_out")
    ap.add_argument("--box-th", type=float, default=0.35)
    ap.add_argument("--text-th", type=float, default=0.25)
    a = ap.parse_args()

    try:
        import torch
        from PIL import Image as PILImage
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import load_model, predict
    except Exception as ex:
        sys.exit("[依赖缺失] 需要 torch + GroundingDINO: %r" % ex)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device, "(若不是 cuda, 检查 GPU/torch 安装)")
    model = load_model(a.config, a.weights, device=device)
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    caption = " . ".join(HOME_TEST_CLASSES) + " ."
    print("测试类别:", caption)

    # 收集图片
    if os.path.isdir(a.images):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files += glob.glob(os.path.join(a.images, ext))
            files += glob.glob(os.path.join(a.images, ext.upper()))
        files = sorted(set(files))
    else:
        files = [a.images]
    if not files:
        sys.exit("没找到图片: %s" % a.images)
    if not os.path.isdir(a.out):
        os.makedirs(a.out)

    total = 0
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            print("跳过(读不了):", fp); continue
        H, W = img.shape[:2]
        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_t, _ = transform(pil, None)
        boxes, logits, phrases = predict(
            model=model, image=image_t, caption=caption,
            box_threshold=a.box_th, text_threshold=a.text_th, device=device)
        n = 0
        hits = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = [float(v) for v in box]   # 归一化 cxcywh
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tag = "%s %.2f" % (phrase, float(logit))
            cv2.putText(img, tag, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            hits.append("%s(%.2f)" % (phrase, float(logit)))
            n += 1
        total += n
        outp = os.path.join(a.out, os.path.basename(fp))
        cv2.imwrite(outp, img)
        print("%-36s 检出 %2d: %s" % (os.path.basename(fp), n, ", ".join(hits) or "-"))

    print("\n完成: %d 张图, 共 %d 个框 -> 打开 %s 看画框效果。" % (len(files), total, a.out))
    if total == 0:
        print("⚠️ 一个都没检出 —— 多半是权重没加载/config 路径错/CUDA 没装好; "
              "先把环境弄通, 再跑正式 POI 检测 groundingdino_detect.py。")


if __name__ == "__main__":
    main()
