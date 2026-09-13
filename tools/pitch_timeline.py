# -*- coding: utf-8 -*-
"""
pitch_timeline.py —— 从双目 bag 离线估计相机俯仰角时间线

原理:双目左右目是刚性支架(中途掰支架不影响立体标定),深度全程可信。
每隔 step 秒取一对帧 → SGBM 视差 → 画面下部反投影成点云 →
RANSAC 拟合地面平面 → 法向量反解 俯仰角/滚转角/相机离地高度。
掰支架的时刻会在俯仰角曲线上呈现清晰阶跃。

用法:
  python tools/pitch_timeline.py                      # 默认吃 bag20
  python tools/pitch_timeline.py --bag data/xxx.bag --step 2
输出: <bag>.pitch.csv + <bag>.pitch.png,并打印阶跃检测结果。
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rosbags_index_compat import patch_rosbags_index_headers  # noqa: E402
patch_rosbags_index_headers()
from rosbags.highlevel import AnyReader  # noqa: E402


def build_rectify(info):
    K = np.array(info.K, np.float64).reshape(3, 3)
    D = np.array(info.D, np.float64)
    R = np.array(info.R, np.float64).reshape(3, 3)
    P = np.array(info.P, np.float64).reshape(3, 4)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, R, P[:3, :3],
                                         (info.width, info.height),
                                         cv2.CV_16SC2)
    return (m1, m2), P


def plane_pose(n, d):
    """由平面 (n,d) 求 俯仰/滚转/高度;n 已归一化且 n_y<0"""
    pitch = math.degrees(math.atan2(n[2], -n[1]))
    roll = math.degrees(math.atan2(n[0], -n[1]))
    return pitch, roll, abs(d)


def plausible(n, d):
    """物理合理性:真正的地面必须满足——
    滚转 |roll|<15°(车不会侧翻着跑),俯仰 -10°~+50°(相机朝上装),
    相机离地高度 0.04~0.40 m(JetBot 很矮)。
    停车时拍到的沙发/床垫斜面会在这里被拒掉。"""
    pitch, roll, h = plane_pose(n, d)
    return (abs(roll) < 15.0) and (-10.0 < pitch < 50.0) and (0.04 < h < 0.40)


def fit_floor(points, iters=400, thr=0.02, min_inlier=300):
    """带约束的 RANSAC 平面拟合:只在物理上可能是地面的平面里选内点最多的"""
    n_pts = len(points)
    if n_pts < min_inlier:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ p0)
        if n[1] > 0:
            n, d = -n, -d
        if not plausible(n, d):
            continue
        dist = np.abs(points @ n + d)
        inl = int((dist < thr).sum())
        if best is None or inl > best[0]:
            best = (inl, n, d)
    if best is None or best[0] < min_inlier:
        return None
    # 用内点做最小二乘精修,并复查合理性
    inl_mask = np.abs(points @ best[1] + best[2]) < thr
    pts = points[inl_mask]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vt[2]
    if n[1] > 0:
        n = -n
    d = -float(n @ centroid)
    if not plausible(n, d):
        return None
    return n, d, int(inl_mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=str(HERE.parent / "data" /
                    "semantic_survey_home_20260829_20.bag"))
    ap.add_argument("--step", type=float, default=2.0, help="采样间隔(秒)")
    ap.add_argument("--pair-tol", type=float, default=0.02,
                    help="左右目时间配对容差(秒)")
    args = ap.parse_args()
    bag = Path(args.bag)

    records = []
    with AnyReader([bag]) as r:
        infos = {}
        for side in ("left", "right"):
            con = [c for c in r.connections
                   if c.topic == f"/stereo/{side}/camera_info"]
            _, _, raw = next(r.messages(connections=con))
            infos[side] = r.deserialize(raw, con[0].msgtype)
        (mapL, P_L) = build_rectify(infos["left"])
        (mapR, P_R) = build_rectify(infos["right"])
        fx = P_L[0, 0]
        cx, cy = P_L[0, 2], P_L[1, 2]
        fxB = -P_R[0, 3]                     # fx * baseline
        print(f"[标定] fx={fx:.1f} 基线={fxB / fx * 100:.1f}cm")

        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=7,
            P1=8 * 49, P2=32 * 49, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1)

        img_conns = [c for c in r.connections if c.topic in
                     ("/stereo/left/image_raw/compressed",
                      "/stereo/right/image_raw/compressed")]
        t0 = r.start_time
        last = {"left": None, "right": None}
        next_sample_t = 0.0
        n_done = 0
        for con, ts, raw in r.messages(connections=img_conns):
            side = "left" if "left" in con.topic else "right"
            t = (ts - t0) / 1e9
            last[side] = (t, raw, con.msgtype)
            if t < next_sample_t:
                continue
            if last["left"] is None or last["right"] is None:
                continue
            tl, tr = last["left"][0], last["right"][0]
            if abs(tl - tr) > args.pair_tol:
                continue
            # 解码 + 整流
            imgs = {}
            for s in ("left", "right"):
                m = r.deserialize(last[s][1], last[s][2])
                g = cv2.imdecode(np.frombuffer(m.data, np.uint8),
                                 cv2.IMREAD_GRAYSCALE)
                if g is None:
                    break
                imgs[s] = cv2.remap(g, *(mapL if s == "left" else mapR),
                                    cv2.INTER_LINEAR)
            if len(imgs) < 2:
                continue
            disp = sgbm.compute(imgs["left"], imgs["right"]).astype(
                np.float32) / 16.0
            # 画面下部 40% 找地面;深度 0.3~5 m
            h, w = disp.shape
            v0 = int(h * 0.60)
            vs, us = np.mgrid[v0:h, 0:w]
            dd = disp[v0:, :]
            ok = dd > fxB / 5.0
            if ok.sum() < 500:
                next_sample_t = t + args.step
                continue
            Z = fxB / dd[ok]
            good = (Z > 0.3) & (Z < 5.0)
            Z = Z[good]
            u = us[ok][good].astype(np.float64)
            v = vs[ok][good].astype(np.float64)
            pts = np.column_stack([(u - cx) / fx * Z, (v - cy) / fx * Z, Z])
            if len(pts) > 4000:
                pts = pts[np.random.default_rng(1).choice(
                    len(pts), 4000, replace=False)]
            fit = fit_floor(pts)
            next_sample_t = t + args.step
            if fit is None:
                continue
            n, d, inl = fit
            pitch, roll, height = plane_pose(n, d)
            records.append((t, pitch, roll, height, inl))
            n_done += 1
            if n_done % 50 == 0:
                print(f"  ... {n_done} 个采样,t={t:.0f}s")

    if len(records) < 10:
        sys.exit("[错误] 有效地面拟合样本太少,无法出时间线")
    arr = np.array(records)
    csv = bag.with_suffix(".pitch.csv")
    np.savetxt(csv, arr, delimiter=",", fmt="%.3f",
               header="t_s,pitch_up_deg,roll_deg,cam_height_m,inliers")

    # 阶跃检测:前后 30s 滑窗中位数差的最大值
    t, pitch = arr[:, 0], arr[:, 1]
    best_step = (0.0, 0.0, 0.0)   # |diff|, time, signed diff
    for i in range(len(t)):
        pre = pitch[(t > t[i] - 30) & (t <= t[i])]
        post = pitch[(t > t[i]) & (t <= t[i] + 30)]
        if len(pre) >= 5 and len(post) >= 5:
            diff = float(np.median(post) - np.median(pre))
            if abs(diff) > abs(best_step[0]):
                best_step = (abs(diff), t[i], diff)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axi, col, name, unit in ((0, 1, "俯仰角(上仰为正)", "°"),
                                 (1, 2, "滚转角", "°"),
                                 (2, 3, "相机离地高度", "m")):
        axes[axi].plot(t, arr[:, col], ".", markersize=3, color="#2b6cb0")
        med = np.median(arr[:, col])
        axes[axi].axhline(med, color="#aaa", linewidth=0.8, linestyle="--")
        axes[axi].set_ylabel(f"{name} ({unit})")
    if best_step[0] > 2.0:
        for ax in axes:
            ax.axvline(best_step[1], color="red", linewidth=1.2,
                       linestyle=":")
        axes[0].set_title(
            f"检测到俯仰角阶跃: t={best_step[1]:.0f}s 处变化 "
            f"{best_step[2]:+.1f}°(前后30s中位数差)")
    else:
        axes[0].set_title("未检测到明显俯仰角阶跃(前后30s中位数差 < 2°)")
    axes[2].set_xlabel("bag 时间 (s)")
    fig.tight_layout()
    png = bag.with_suffix(".pitch.png")
    fig.savefig(png, dpi=140)

    q1, q2 = np.percentile(t, [20, 80])
    print(f"[结果] 有效样本 {len(arr)} 个")
    print(f"  前20%时段 俯仰中位 {np.median(pitch[t < q1]):+.1f}°  "
          f"后20%时段 {np.median(pitch[t > q2]):+.1f}°")
    print(f"  滚转中位 {np.median(arr[:, 2]):+.1f}°  "
          f"相机高度中位 {np.median(arr[:, 3]):.2f} m")
    if best_step[0] > 2.0:
        print(f"  ★ 阶跃: t={best_step[1]:.0f}s,幅度 {best_step[2]:+.1f}°")
    else:
        print("  未见 >2° 的阶跃")
    print(f"[输出] {csv}\n[输出] {png}")


if __name__ == "__main__":
    main()
