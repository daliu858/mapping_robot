# -*- coding: utf-8 -*-
"""
make_floorplan.py —— 占据栅格图(SLAM) → 带标签的建筑平面图

流程(全部确定性几何,LLM 不碰任何坐标):
  1. 读 map_server 的 PGM+YAML
  2. 清噪(形态学 + 连通域过滤)
  3. Manhattan 对齐(Hough 主方向 → 整图旋转)
  4. 分房间(距离变换 + watershed,门洞处自然断开)
  5. 房间矢量化(轮廓 → 多边形 → 直角化)
  6. 门:从语义 YAML 读坐标,吸附到最近墙边,画建筑门符号
  7. 起名:默认启发式(面积/形状/物体投票);--llm 时调 API 起名(仅起名)
  8. 输出 out/floorplan.png + floorplan.svg + floorplan.dxf + debug_rooms.png

用法:
  python make_floorplan.py                          # 用默认地图与示例门数据
  python make_floorplan.py --map ..\\maps\\xx.yaml --objects doors.yaml --llm
"""
import argparse
import json
import math
import re
import sys
import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage
from skimage.segmentation import watershed

HERE = Path(__file__).resolve().parent

# Windows 控制台默认编码可能打不出中文
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 1. 读地图


def load_map(yaml_path: Path):
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    img_ref = Path(str(cfg["image"]))
    # yaml 里常是车上的绝对路径(/home/jetbot/...),取文件名在 yaml 旁边找
    candidates = [img_ref, yaml_path.parent / img_ref.name]
    img_path = next((p for p in candidates if p.is_file()), None)
    if img_path is None:
        sys.exit(f"[错误] 找不到地图图像: {cfg['image']}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"[错误] 图像读取失败: {img_path}")
    if int(cfg.get("negate", 0)):
        img = 255 - img
    res = float(cfg["resolution"])
    ox, oy = float(cfg["origin"][0]), float(cfg["origin"][1])
    print(f"[地图] {img_path.name}  {img.shape[1]}x{img.shape[0]}px  "
          f"res={res}m/px  origin=({ox},{oy})")
    return img, res, (ox, oy)


def world_to_px(x, y, res, origin, img_h):
    """map 坐标系(米,y 向上) → 原始图像像素(col,row,y 向下)"""
    col = (x - origin[0]) / res
    row = img_h - 1 - (y - origin[1]) / res
    return col, row


# ---------------------------------------------------------------- 2. 清噪


def clean_masks(img, res):
    occ = (img <= 100).astype(np.uint8)      # 0   = 墙/障碍
    free = (img >= 240).astype(np.uint8)     # 254 = 自由空间
    # 墙:去掉 <5 格的孤立噪点(≈ 12cm² 的斑点)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 5:
            occ[lab == i] = 0
    # 自由空间:闭运算补小洞,然后只留最大连通域(屋内),
    # 玻璃/门缝漏出去的外部区域全部丢弃
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    free[occ > 0] = 0
    # 圆核开运算:磨掉激光噪声在墙边留下的毛刺(≈10cm),轮廓才像建筑图
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, 4)
    if n <= 1:
        sys.exit("[错误] 地图里没有自由空间")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    free = (lab == main).astype(np.uint8)
    return occ, free


def crop_to_content(occ, free, margin=20):
    content = (occ | free)
    rows = np.any(content, axis=1).nonzero()[0]
    cols = np.any(content, axis=0).nonzero()[0]
    r0 = max(0, rows[0] - margin)
    r1 = min(occ.shape[0], rows[-1] + margin + 1)
    c0 = max(0, cols[0] - margin)
    c1 = min(occ.shape[1], cols[-1] + margin + 1)
    return occ[r0:r1, c0:c1], free[r0:r1, c0:c1], (c0, r0)


# ---------------------------------------------------------------- 3. Manhattan 对齐


def dominant_angle(occ):
    """Hough 找墙线,按长度加权求主方向偏角(周期 90°),返回需要旋转的度数"""
    lines = cv2.HoughLinesP(occ * 255, 1, np.pi / 360, threshold=30,
                            minLineLength=25, maxLineGap=4)
    if lines is None or len(lines) < 3:
        return 0.0
    sx = sy = 0.0
    for x1, y1, x2, y2 in lines[:, 0]:
        length = math.hypot(x2 - x1, y2 - y1)
        a = math.atan2(y2 - y1, x2 - x1)      # 周期 90° → 乘 4 做圆均值
        sx += length * math.cos(4 * a)
        sy += length * math.sin(4 * a)
    dev = math.degrees(math.atan2(sy, sx)) / 4.0   # (-22.5, 22.5]
    return -dev


def rotate_expand(masks, ang_deg, pts):
    """整图旋转并扩大画布;pts(Nx2 像素点)同步变换"""
    h, w = masks[0].shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang_deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    outs = [cv2.warpAffine(m, M, (nw, nh), flags=cv2.INTER_NEAREST,
                           borderValue=0) for m in masks]
    if len(pts):
        pts = cv2.transform(np.asarray(pts, np.float64).reshape(-1, 1, 2),
                            M).reshape(-1, 2)
    return outs, np.asarray(pts), M


# ---------------------------------------------------------------- 4. 分房间


def segment_rooms(free, res, clearance_m, min_room_m2):
    """距离变换 + watershed:把门洞(半宽 < clearance)处自然分开"""
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    seeds = (dist > clearance_m / res).astype(np.uint8)
    markers, n = ndimage.label(seeds)
    # 丢掉太小的种子(< 0.5 m²)
    px_per_m2 = 1.0 / (res * res)
    for i in range(1, n + 1):
        if (markers == i).sum() < 0.5 * px_per_m2:
            markers[markers == i] = 0
    markers, n = ndimage.label(markers > 0)
    if n == 0:
        sys.exit("[错误] 没分出任何房间,试试调小 --clearance")
    labels = watershed(-dist, markers, mask=free.astype(bool))
    # 小房间并入相邻大房间
    for i in range(1, labels.max() + 1):
        m = labels == i
        if m.sum() >= min_room_m2 * px_per_m2:
            continue
        ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        neigh = labels[ring & ~m]
        neigh = neigh[neigh > 0]
        labels[m] = np.bincount(neigh).argmax() if len(neigh) else 0
    # 重新编号成 1..K
    out = np.zeros_like(labels)
    for k, i in enumerate(sorted(set(labels.ravel()) - {0}), start=1):
        out[labels == i] = k
    print(f"[分割] 共 {out.max()} 个房间")
    return out


# ---------------------------------------------------------------- 5. 矢量化


def rectify_polygon(poly, axis_tol_deg=30.0):
    """把接近水平/垂直的边掰成严格水平/垂直;斜墙(如玻璃)保留原方向"""
    pts = poly.astype(np.float64).tolist()
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        ang = math.degrees(math.atan2(dy, dx)) % 90.0
        dev = min(ang, 90.0 - ang)
        if dev > axis_tol_deg:
            continue                       # 真斜边,保留
        if abs(dx) >= abs(dy):             # 水平边:y 取平均
            m = (pts[i][1] + pts[j][1]) / 2
            pts[i][1] = pts[j][1] = m
        else:                              # 垂直边:x 取平均
            m = (pts[i][0] + pts[j][0]) / 2
            pts[i][0] = pts[j][0] = m
    # 去掉共线/重复点
    out = []
    for k in range(n):
        p0 = np.array(pts[(k - 1) % n])
        p1 = np.array(pts[k])
        p2 = np.array(pts[(k + 1) % n])
        if np.hypot(*(p1 - p0)) < 1.0:
            continue
        cross = np.cross(p1 - p0, p2 - p1)
        if abs(cross) < 1.0 and np.dot(p1 - p0, p2 - p1) > 0:
            continue
        out.append(pts[k])
    return np.array(out if len(out) >= 3 else pts)


def simplify_jogs(poly, tol_px):
    """正交理想化:把短于 tol 的碎边(锯齿台阶)收缩掉,墙变成长直线。
    这是有界的理想化——任何点的位移不超过 tol,容差会标在图注里。"""
    pts = poly.astype(np.float64).tolist()
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        best_i, best_len = -1, float(tol_px)
        for i in range(n):
            j = (i + 1) % n
            L = math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1])
            if L < best_len:
                best_len, best_i = L, i
        if best_i >= 0:
            j = (best_i + 1) % len(pts)
            pts[best_i] = [(pts[best_i][0] + pts[j][0]) / 2,
                           (pts[best_i][1] + pts[j][1]) / 2]
            del pts[j]
            changed = True
    return np.array(pts)


def extract_rooms(labels, res, eps_px=4.0, ideal_tol_m=0.3):
    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rooms = []
    for i in range(1, labels.max() + 1):
        mask = (labels == i).astype(np.uint8)
        if mask.sum() == 0:
            continue
        # 平滑房间边界(开+闭 ≈15cm),再矢量化
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, smooth_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_k)
        if mask.sum() == 0:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, eps_px, True).reshape(-1, 2)
        poly = rectify_polygon(poly)
        if ideal_tol_m > 0:
            poly = simplify_jogs(poly, ideal_tol_m / res)
            poly = rectify_polygon(poly)   # 收缩后再掰直一次
            poly = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2),
                                    2.5, True).reshape(-1, 2)
            poly = rectify_polygon(poly)   # 合并近共线长边
        area_m2 = cv2.contourArea(poly.astype(np.float32)) * res * res
        # 标签优先放多边形质心(居中好看);质心不在房间内(L形)则退回
        # 离墙最远点
        d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        r, c = np.unravel_index(int(d.argmax()), d.shape)
        mm = cv2.moments(poly.astype(np.float32))
        if mm["m00"] > 1e-6:
            ccx, ccy = mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]
            if cv2.pointPolygonTest(poly.astype(np.float32),
                                    (ccx, ccy), True) > 6:
                c, r = ccx, ccy
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        short, long_ = sorted([max(rw, 1), max(rh, 1)])
        rooms.append(dict(id=i, mask=mask, poly=poly, area=area_m2,
                          label_px=(c, r), short_m=short * res,
                          aspect=long_ / short, name="", objects=[]))
    return rooms


# ---------------------------------------------------------------- 6. 门


def snap_doors(door_pts, rooms, res, door_w=0.9, max_snap_m=1.0):
    """把门坐标吸附到最近的房间多边形边上,取边方向作为门的朝向"""
    doors = []
    for px, py in door_pts:
        best = None                                   # (距离, 落点, 边方向)
        for room in rooms:
            poly = room["poly"]
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-9:
                    continue
                t = float(np.clip(((px, py) - a) @ ab / L2, 0.0, 1.0))
                q = a + t * ab
                dd = math.hypot(px - q[0], py - q[1])
                if best is None or dd < best[0]:
                    best = (dd, q, math.atan2(ab[1], ab[0]))
        if best and best[0] * res <= max_snap_m:
            ang = best[2]
            qx, qy = best[1]
            # 门缝法向两侧各探 0.4m,门扇画向面积更大的一侧(向房间开)
            nx_, ny_ = -math.sin(ang), math.cos(ang)
            probe = 0.4 / res
            def side_area(sgn):
                px_, py_ = qx + sgn * nx_ * probe, qy + sgn * ny_ * probe
                for room in rooms:
                    m = room["mask"]
                    r_, c_ = int(round(py_)), int(round(px_))
                    if 0 <= r_ < m.shape[0] and 0 <= c_ < m.shape[1]                             and m[r_, c_]:
                        return room["area"]
                return -1.0
            swing = 1 if side_area(1) >= side_area(-1) else -1
            doors.append(dict(x=qx, y=qy, ang=ang, swing=swing,
                              w_px=door_w / res, ok=True))
        else:
            doors.append(dict(x=px, y=py, ang=0.0, w_px=door_w / res,
                              ok=False))   # 吸附失败:画警告标记,不瞎画门
    return doors


def load_objects(objects_path: Path):
    """读项目语义物体 YAML(slam-car-semantic-objects-v1),分出门和其它物体"""
    data = yaml.safe_load(objects_path.read_text(encoding="utf-8"))
    door_xy, others = [], []
    for o in data.get("objects", []):
        label = str(o.get("label", "")).lower()
        x, y = float(o["x"]), float(o["y"])
        if "door" in label:
            door_xy.append((x, y))
        else:
            others.append((label, x, y))
    print(f"[语义] 门 x{len(door_xy)},其它物体 x{len(others)}")
    return door_xy, others


# ---------------------------------------------------------------- 7. 起名

OBJ2ROOM = {
    "bed": "Bedroom", "sofa": "Living room", "couch": "Living room",
    "tv": "Living room", "television": "Living room",
    "refrigerator": "Kitchen", "fridge": "Kitchen",
    "oven": "Kitchen", "stove": "Kitchen", "microwave": "Kitchen",
    "sink": "Kitchen",
    "toilet": "Bathroom", "bathtub": "Bathroom", "shower": "Bathroom",
    "washing machine": "Balcony", "dining table": "Dining", "table": "Dining",
    "desk": "Study", "bookshelf": "Study",
}


def heuristic_names(rooms):
    """只做有证据的起名:屋内检测到的物体投票。没证据一律中性编号,不瞎猜。"""
    for room in rooms:
        votes = [OBJ2ROOM[o] for o in room["objects"] if o in OBJ2ROOM]
        if votes:
            room["name"] = max(set(votes), key=votes.count)
    unnamed = sorted((r for r in rooms if not r["name"]),
                     key=lambda r: -r["area"])
    for k, room in enumerate(unnamed):
        room["name"] = f"Room {chr(ord('A') + k)}"
    # 重名加编号
    seen = {}
    for room in rooms:
        seen[room["name"]] = seen.get(room["name"], 0) + 1
    idx = {}
    for room in rooms:
        if seen[room["name"]] > 1:
            idx[room["name"]] = idx.get(room["name"], 0) + 1
            room["name"] += str(idx[room["name"]])


def load_env(env_path: Path):
    env = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def llm_name_rooms(rooms, preview_png: Path, env):
    """可选:把预览图+房间统计发给 Claude,只要一个 {id: 名字} 的 JSON。
    LLM 只负责起名;几何、面积、门的位置它一个都改不了。失败就回退启发式。"""
    import base64
    import requests
    key = env.get("ANTHROPIC_API_KEY")
    if not key:
        print("[LLM] .env 里没有 ANTHROPIC_API_KEY,回退启发式起名")
        return False
    base = (env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
    model = (env.get("FLOORPLAN_LLM_MODEL") or env.get("FABLE_MODEL")
             or "claude-opus-5")
    stats = [dict(id=r["id"], area_m2=round(r["area"], 1),
                  aspect=round(r["aspect"], 1), objects=r["objects"])
             for r in rooms]
    prompt = (
        "This is a floor plan auto-segmented from a household SLAM map. "
        "Each room is labeled with a numeric id.\n"
        f"Room stats: {json.dumps(stats)}\n"
        "Name each room in English from shape, area, relative position, "
        "and objects (e.g. Living room / Bedroom / Kitchen / Bathroom / "
        "Hallway / Balcony / Entry).\n"
        'Output only one JSON object, e.g. {"1": "Living room", '
        '"2": "Hallway"}.')
    img_b64 = base64.standard_b64encode(preview_png.read_bytes()).decode()
    body = {
        "model": model, "max_tokens": 1024,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": img_b64}},
            {"type": "text", "text": prompt}]}],
    }
    try:
        resp = requests.post(f"{base}/v1/messages", json=body, timeout=120,
                             headers={"x-api-key": key,
                                      "anthropic-version": "2023-06-01",
                                      "content-type": "application/json"})
        resp.raise_for_status()
        text = "".join(b.get("text", "") for b in resp.json()["content"]
                       if b.get("type") == "text")
        m = re.search(r"\{[^{}]*\}", text, re.S)
        names = json.loads(m.group(0))
        for room in rooms:
            got = str(names.get(str(room["id"]), "")).strip()
            if got:
                room["name"] = got
        print(f"[LLM] {model} 起名成功: "
              + ", ".join(f'{r["id"]}={r["name"]}' for r in rooms))
        return True
    except Exception as e:  # 任何失败都不致命
        print(f"[LLM] 起名失败({e}),回退启发式")
        return False


# ---------------------------------------------------------------- 8. 绘图

WALL = "#1a1a1a"
WALL_W_M = 0.12      # 墙总厚(米)
EDGE_M = 0.025       # 双线墙每条黑边的宽(米)


def classify_edges(rooms, occ, res, near_m=0.30):
    """逐边判断是不是真墙:边中段采样,附近有激光占据点→实墙;
    否则是开放分界/未探明边界(画虚线,不冒充墙)。"""
    free_of_occ = (occ == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free_of_occ, cv2.DIST_L2, 5)
    thr = near_m / res
    h, w = dist.shape
    for room in rooms:
        poly = room["poly"]
        real = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            ts = np.linspace(0.15, 0.85, 9)
            cs = np.clip((a[0] + ts * (b[0] - a[0])).round().astype(int),
                         0, w - 1)
            rs = np.clip((a[1] + ts * (b[1] - a[1])).round().astype(int),
                         0, h - 1)
            real.append(bool(np.median(dist[rs, cs]) < thr))
        room["edge_real"] = real
        # 实测墙占周长比例:太低说明这块"房间"主要由激光漏缝的幻影自由
        # 空间构成(门关着根本没扫进去),必须降级为未探明区域
        seg_len = [float(np.hypot(*(poly[(k + 1) % len(poly)] - poly[k])))
                   for k in range(len(poly))]
        total = sum(seg_len) or 1.0
        room["explored_frac"] = sum(
            l for l, r_ in zip(seg_len, real) if r_) / total


def render(rooms, doors, others, res, canvas_hw, out_dir: Path,
           title="FLOOR PLAN (SLAM auto-vectorized)", labeled=True):
    """CAD 风渲染:白底、双线空心墙、黑色门弧、房间名+面积+map坐标"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Rectangle
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    H, W = canvas_hw
    def X(c): return c * res                 # 像素 → 米
    def Y(r): return (H - r) * res           # y 翻上

    ext_w, ext_h = W * res, H * res
    fig_w = 12.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * ext_h / ext_w))
    ax.set_xlim(0, ext_w)
    ax.set_ylim(0, ext_h)
    ax.set_aspect("equal")
    ax.axis("off")
    pt_per_m = fig_w * 72 / ext_w            # 米 → points
    wall_lw = WALL_W_M * pt_per_m
    inner_lw = max((WALL_W_M - 2 * EDGE_M), 0.02) * pt_per_m

    # 逐边画墙:实墙=双线灰芯,开放/未探明边界=细虚线
    pier_pts = []
    for room in rooms:
        poly = room["poly"]
        n = len(poly)
        real = room.get("edge_real", [True] * n)
        xy = np.column_stack([X(poly[:, 0]), Y(poly[:, 1])])
        ax.fill(xy[:, 0], xy[:, 1], facecolor="white", edgecolor="none",
                zorder=1)
        # 把连续的实墙边串成 polyline,拐角处线条相接不留豁口
        runs, cur = [], []
        for k in range(n):
            if real[k]:
                cur.append(k)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            if runs and real[n - 1] and real[0] and runs[0][0] == 0:
                runs[0] = cur + runs[0]      # 首尾相连的 run 合并
            else:
                runs.append(cur)
        for run in runs:
            ks = run + [(run[-1] + 1) % n]
            px_, py_ = xy[[k % n for k in ks], 0], xy[[k % n for k in ks], 1]
            ax.plot(px_, py_, color=WALL, linewidth=wall_lw,
                    solid_joinstyle="miter", solid_capstyle="butt", zorder=2)
            ax.plot(px_, py_, color="#d7d7d7", linewidth=inner_lw,
                    solid_joinstyle="miter", solid_capstyle="butt", zorder=3)
        for k in range(n):
            if not real[k]:
                j = (k + 1) % n
                ax.plot([xy[k, 0], xy[j, 0]], [xy[k, 1], xy[j, 1]],
                        color="#b0b0b0", linewidth=0.8,
                        linestyle=(0, (5, 4)), zorder=2)
        # 墙墩候选:两条相邻实墙相交、夹角接近直角、两边都够长
        for k in range(n):
            kp = (k - 1) % n
            if not (real[kp] and real[k]):
                continue
            p_prev, p, p_next = xy[kp], xy[k], xy[(k + 1) % n]
            v1, v2 = p - p_prev, p_next - p
            l1, l2 = np.hypot(*v1), np.hypot(*v2)
            if l1 < 0.7 or l2 < 0.7:
                continue
            cosang = abs(float(v1 @ v2) / (l1 * l2))
            if cosang < 0.35:                # 夹角 70°~110°
                pier_pts.append(p)
    pier = WALL_W_M * 2.2
    for p in pier_pts:
        ax.add_patch(Rectangle((p[0] - pier / 2, p[1] - pier / 2),
                               pier, pier, facecolor="#8f8f8f",
                               edgecolor=WALL, linewidth=0.5, zorder=4))

    for room in rooms:
        lx, ly = X(room["label_px"][0]), Y(room["label_px"][1])
        fs = float(np.clip(6.0 + 2.4 * math.sqrt(room["area"]), 9.0, 14.0))
        if labeled and room.get("explored_frac", 1.0) < 0.35:
            xy = np.column_stack([X(room["poly"][:, 0]), Y(room["poly"][:, 1])])
            ax.fill(xy[:, 0], xy[:, 1], facecolor="none", edgecolor="#bbb",
                    hatch="///", linewidth=0.0, zorder=1.5)
            ax.text(lx, ly, "未探明", ha="center", va="center",
                    fontsize=9, color="#999", zorder=8)
            continue
        if labeled:
            ax.text(lx, ly + 0.10, room["name"], ha="center", va="bottom",
                    fontsize=fs, color="#111", zorder=8)
            wx, wy = room.get("world_xy", (None, None))
            sub = f'{room["area"]:.1f} m²'
            if wx is not None:
                sub += f"  ({wx:.1f}, {wy:.1f})"
            ax.text(lx, ly - 0.08, sub, ha="center", va="top",
                    fontsize=max(fs - 4.5, 6.5), color="#555", zorder=8)
        else:
            ax.text(lx, ly, str(room["id"]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color="#c00", zorder=8)

    for d in doors:
        x, y = X(d["x"]), Y(d["y"])
        if not d["ok"]:
            ax.plot(x, y, "x", color="red", markersize=9, zorder=9)
            ax.text(x, y + 0.25, "door?", color="red", fontsize=9,
                    ha="center", zorder=9)
            continue
        w = d["w_px"] * res
        ang = -d["ang"]                      # y 翻转后角度取负
        ux, uy = math.cos(ang), math.sin(ang)
        nxv, nyv = -uy, ux                   # 墙法向
        # 1) 白色缺口把墙擦开
        ax.plot([x - ux * w / 2, x + ux * w / 2],
                [y - uy * w / 2, y + uy * w / 2],
                color="white", linewidth=wall_lw * 1.5,
                solid_capstyle="butt", zorder=4)
        # 2) 门垛:缺口两端各一小段横线封住墙截面
        jamb = WALL_W_M * 0.85
        for sgn in (-1, 1):
            jx, jy = x + sgn * ux * w / 2, y + sgn * uy * w / 2
            ax.plot([jx - nxv * jamb, jx + nxv * jamb],
                    [jy - nyv * jamb, jy + nyv * jamb],
                    color=WALL, linewidth=1.0, zorder=5)
        # 3) 门扇 + 四分之一圆弧(铰链在缺口一端)
        hx, hy = x - ux * w / 2, y - uy * w / 2
        a0 = math.degrees(ang)
        # 像素系 y 向下、图纸系 y 向上,swing 取反才是图纸上的同侧
        sw = -d.get("swing", 1)
        leaf = a0 + 90 * sw
        ax.add_patch(Arc((hx, hy), 2 * w, 2 * w, angle=0,
                         theta1=min(a0, leaf), theta2=max(a0, leaf),
                         color=WALL, linewidth=0.8, zorder=5))
        ax.plot([hx, hx + w * math.cos(math.radians(leaf))],
                [hy, hy + w * math.sin(math.radians(leaf))],
                color=WALL, linewidth=1.0, zorder=5)
        if d.get("world") is not None:
            ax.text(x + 0.15, y + 0.15, f'({d["world"][0]:.1f}, '
                    f'{d["world"][1]:.1f})', fontsize=6.5, color="#888",
                    zorder=8)

    # POI 分类配色:CAD 灰调为主,spray 是核心交付物,用红色突出
    poi_style = {
        "sofa":    dict(color="#c07820", marker="s", ms=5.0, fs=7.5),
        "cabinet": dict(color="#3a6ea5", marker="o", ms=4.0, fs=7),
        "spray":   dict(color="#c00000", marker="*", ms=13.0, fs=9),
    }
    for label, c, r in others:
        x, y = X(c), Y(r)
        st = poi_style.get(label, dict(color="#555", marker="o",
                                       ms=3.5, fs=7))
        ax.plot(x, y, st["marker"], color=st["color"], markersize=st["ms"],
                zorder=9, markeredgecolor="white", markeredgewidth=0.5)
        weight = "bold" if label == "spray" else "normal"
        ax.text(x + 0.15, y + 0.15, label, fontsize=st["fs"],
                color=st["color"], zorder=9, fontweight=weight)

    # 比例尺(左下)与指北针(右上)
    x0, y0 = 0.6, 0.6
    for i in range(3):
        ax.plot([x0 + i, x0 + i + 1], [y0, y0], color="#111",
                linewidth=3 if i % 2 == 0 else 1.0, solid_capstyle="butt")
    for i in range(4):
        ax.text(x0 + i, y0 + 0.15, str(i), fontsize=8, ha="center")
    ax.text(x0 + 3.4, y0 + 0.15, "m", fontsize=8)
    # 图例:实线=实测墙,虚线=开放/未探明边界
    ly0 = y0 + 0.75
    ax.plot([x0, x0 + 0.7], [ly0 + 0.42] * 2, color=WALL, linewidth=2.2,
            solid_capstyle="butt")
    ax.text(x0 + 0.85, ly0 + 0.42, "Measured wall", fontsize=7.5, va="center")
    ax.plot([x0, x0 + 0.7], [ly0] * 2, color="#b0b0b0", linewidth=0.9,
            linestyle=(0, (5, 4)))
    for i, (lab, st) in enumerate([
            ("sofa", ("#c07820", "s", 5.0)),
            ("cabinet", ("#3a6ea5", "o", 4.0)),
            ("insecticide spray (open-vocab)", ("#c00000", "*", 11.0))]):
        yl = ly0 + 0.84 + i * 0.42
        ax.plot(x0 + 0.35, yl, st[1], color=st[0], markersize=st[2],
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(x0 + 0.85, yl, lab, fontsize=7.5, va="center")
    ax.text(x0 + 0.85, ly0, "Open / unexplored boundary", fontsize=7.5,
            va="center")
    nx_, ny_ = ext_w - 1.0, ext_h - 1.6
    ax.annotate("", xy=(nx_, ny_ + 1.0), xytext=(nx_, ny_),
                arrowprops=dict(arrowstyle="-|>", color="#111", linewidth=1.2))
    ax.text(nx_, ny_ + 1.15, "N", fontsize=11, ha="center", fontweight="bold")

    # 双线图框 + 底部标题 + 元数据脚注(工程制图排版)
    for pad, lw in ((0.10, 1.4), (0.22, 0.5)):
        ax.add_patch(Rectangle((pad, pad), ext_w - 2 * pad, ext_h - 2 * pad,
                               facecolor="none", edgecolor="#111",
                               linewidth=lw, zorder=10))
    ax.text(ext_w / 2, 0.55, "Floor 1", fontsize=17, ha="center",
            va="bottom", color="#111", zorder=11)
    ax.text(ext_w - 0.45, 0.38,
            f"{title}   grid {res} m/px   idealized ±0.3 m   "
            + datetime.date.today().isoformat(),
            fontsize=6.5, ha="right", va="bottom", color="#777", zorder=11)

    fig.tight_layout()
    png = out_dir / ("floorplan.png" if labeled else "_preview_ids.png")
    fig.savefig(png, dpi=180, facecolor="white")
    if labeled:
        fig.savefig(out_dir / "floorplan.svg", facecolor="white")
    plt.close(fig)
    return png


def write_dxf(rooms, doors, res, canvas_hw, out_path: Path):
    import ezdxf
    H, _ = canvas_hw
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    msp = doc.modelspace()
    for name, color in [("WALLS", 7), ("DOORS", 3), ("LABELS", 5)]:
        doc.layers.add(name, color=color)
    def pt(p): return (p[0] * res, (H - p[1]) * res)
    for room in rooms:
        msp.add_lwpolyline([pt(p) for p in room["poly"]], close=True,
                           dxfattribs={"layer": "WALLS"})
        lx, ly = pt(room["label_px"])
        msp.add_text(f'{room["name"]} {room["area"]:.1f}m2',
                     dxfattribs={"layer": "LABELS", "height": 0.25}
                     ).set_placement((lx, ly))
    for d in doors:
        if not d["ok"]:
            continue
        x, y = pt((d["x"], d["y"]))
        w = d["w_px"] * res
        ang = -d["ang"]
        hx = x - math.cos(ang) * w / 2
        hy = y - math.sin(ang) * w / 2
        msp.add_arc((hx, hy), radius=w,
                    start_angle=math.degrees(ang),
                    end_angle=math.degrees(ang) + 90,
                    dxfattribs={"layer": "DOORS"})
    doc.saveas(out_path)


def debug_image(occ, free, labels, doors, out_path: Path):
    h, w = occ.shape
    vis = np.full((h, w, 3), 255, np.uint8)
    vis[free == 0] = (235, 235, 235)
    rng = np.random.default_rng(7)
    for i in range(1, labels.max() + 1):
        vis[labels == i] = rng.integers(120, 245, 3)
    vis[occ > 0] = (40, 40, 40)
    for d in doors:
        c = (0, 0, 255) if d["ok"] else (255, 0, 255)
        cv2.circle(vis, (int(d["x"]), int(d["y"])), 6, c, 2)
    cv2.imwrite(str(out_path), vis)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="占据栅格图 → 建筑平面图")
    ap.add_argument("--map", default=str(HERE.parent / "maps" /
                                         "home_floor_20260804_glass_fixed.yaml"))
    ap.add_argument("--objects", default=str(HERE / "doors_sample.yaml"),
                    help="语义物体 YAML(slam-car-semantic-objects-v1)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--clearance", type=float, default=0.45,
                    help="房间分割用的最小半宽(米),门洞半宽小于它就断开")
    ap.add_argument("--min-room", type=float, default=2.0,
                    help="小于此面积(m²)的碎块并入邻居")
    ap.add_argument("--door-width", type=float, default=0.9)
    ap.add_argument("--llm", action="store_true",
                    help="调用 Claude 给房间起名(仅起名,不碰几何)")
    ap.add_argument("--title", default="FLOOR PLAN (SLAM auto-vectorized)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2 读图清噪
    img, res, origin = load_map(Path(args.map))
    occ, free = clean_masks(img, res)
    occ, free, (c0, r0) = crop_to_content(occ, free)

    # 语义物体 → 像素坐标(先减裁剪偏移)
    door_xy, other_objs = load_objects(Path(args.objects))
    def to_px(x, y):
        c, r = world_to_px(x, y, res, origin, img.shape[0])
        return [c - c0, r - r0]
    pts = np.array([to_px(x, y) for x, y in door_xy] +
                   [to_px(x, y) for _, x, y in other_objs], np.float64)

    # 3 Manhattan 对齐
    ang = dominant_angle(occ)
    print(f"[对齐] 主方向偏角 {-ang:.2f}°,旋转 {ang:.2f}° 校正")
    rot_M = None
    if abs(ang) > 0.3:
        (occ, free), pts, rot_M = rotate_expand([occ, free], ang, pts)

    def px_to_world(c, r):
        """成图像素 → map 坐标(米):逆旋转 → 加裁剪偏移 → 米"""
        if rot_M is not None:
            inv = cv2.invertAffineTransform(rot_M)
            c, r = cv2.transform(np.array([[[c, r]]], np.float64),
                                 inv).ravel()
        c, r = c + c0, r + r0
        return (origin[0] + c * res,
                origin[1] + (img.shape[0] - 1 - r) * res)

    door_pts = pts[:len(door_xy)]
    other_pts = [(other_objs[i][0], *pts[len(door_xy) + i])
                 for i in range(len(other_objs))]

    # 4-5 分房间 + 矢量化
    labels = segment_rooms(free, res, args.clearance, args.min_room)
    rooms = extract_rooms(labels, res)

    # 物体落到所在房间(供起名投票)
    for label, c, r in other_pts:
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < labels.shape[0] and 0 <= ci < labels.shape[1]:
            k = labels[ri, ci]
            for room in rooms:
                if room["id"] == k:
                    room["objects"].append(label)

    # 房间/门挂上 map 真实坐标(标注用);逐边分类真墙/开放边界
    for room in rooms:
        room["world_xy"] = px_to_world(*room["label_px"])
    classify_edges(rooms, occ, res)

    # 6 门
    doors = snap_doors(door_pts, rooms, res, args.door_width)
    for d in doors:
        d["world"] = px_to_world(d["x"], d["y"])
        if not d["ok"]:
            print("[警告] 有一扇门离所有墙都超过 1 m,按红叉标出,未画门符号")

    # 7 起名:先启发式打底;--llm 则用带 id 的预览图请 LLM 覆盖
    heuristic_names(rooms)
    if args.llm:
        preview = render(rooms, doors, other_pts, res, free.shape, out_dir,
                         labeled=False)
        llm_name_rooms(rooms, preview, load_env(HERE.parent / ".env"))

    # 8 输出
    debug_image(occ, free, labels, doors, out_dir / "debug_rooms.png")
    render(rooms, doors, other_pts, res, free.shape, out_dir,
           title=args.title)
    write_dxf(rooms, doors, res, free.shape, out_dir / "floorplan.dxf")

    total = sum(r["area"] for r in rooms)
    print(f"[完成] {len(rooms)} 个房间,总面积 {total:.1f} m² → {out_dir}")
    for room in rooms:
        objs = f'  物体:{",".join(room["objects"])}' if room["objects"] else ""
        print(f'   - {room["name"]:<6} {room["area"]:6.1f} m²{objs}')


if __name__ == "__main__":
    main()
