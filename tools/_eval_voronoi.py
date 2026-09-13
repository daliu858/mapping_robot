# -*- coding: utf-8 -*-
"""
_eval_voronoi.py — compare A (room_segment.py, distance-transform core) vs
B (room_segment_voronoi.py, Voronoi core) on furnished Bormann maps against
the ground-truth segmentation, Bormann-style.

GT: Freiburg52_scan_gt_segmentation.png is a grayscale map where DOORS are
sealed with black lines; so free-space connected components (thr>=250) are
the ground-truth rooms.

Metrics (per Bormann Sec. IV-C, region-overlap based):
  - For each predicted region we find the GT room it overlaps most.
  - recall    = mean over GT rooms of (px of GT room covered by its
                best-matching predicted region) / (GT room px)
  - precision = mean over predicted regions of (px of pred region inside its
                best-matching GT room) / (pred region px)
  - under-segmentation count = # of GT-room PAIRS merged into one predicted
                region (a predicted region that is the best match / majority
                owner of >=2 GT rooms contributes that many merged pairs)
  - over-segmentation  count = # of GT rooms that are split across >=2
                predicted regions (each extra piece beyond the first counts).

Both algorithms receive the SAME single-channel input file so A's algorithm
is untouched (A only fails to *load* RGBA; the pixels are identical).
"""
import os, sys, subprocess, importlib.util
import numpy as np
from PIL import Image
import cv2

BASE = r"d:\slam car\bormann_maps"
TOOLS = r"d:\slam car\tools"
RES = 0.05
MIN_GT_PX = int(1.0 / (RES * RES))     # 1 m^2


def to_gray_png(src, dst):
    im = np.array(Image.open(src))
    if im.ndim == 3:
        im = im[..., 0]
    Image.fromarray(im.astype(np.uint8)).save(dst)
    return dst


def gt_labels(gt_path):
    g = np.array(Image.open(gt_path))
    if g.ndim == 3:
        g = g[..., 0]
    free = (g >= 250).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    out = np.zeros_like(lab, dtype=np.int32)
    nid = 0
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= MIN_GT_PX:
            nid += 1
            out[lab == k] = nid
    return out, nid


def run_A(inp, out_prefix):
    """Run room_segment.py as a module in-process to capture its label image.
    We import it and re-run its main pipeline with a thin wrapper so we can
    grab `lab` and the crop offset. Simpler: shell out, then re-derive labels
    from its colored output is lossy -> instead replicate by importing fns."""
    spec = importlib.util.spec_from_file_location("rs", os.path.join(TOOLS, "room_segment.py"))
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    raw = np.array(Image.open(inp))
    if raw.ndim == 3:
        raw = raw[..., 0]
    res, origin = RES, None
    Hf, Wf = raw.shape
    occ, free = raw <= 50, raw >= 250
    ys, xs = np.where(free | occ)
    m = 12
    y0, y1 = max(0, ys.min()-m), min(Hf, ys.max()+m)
    x0, x1 = max(0, xs.min()-m), min(Wf, xs.max()+m)
    crop = raw[y0:y1, x0:x1]
    free_c = (crop >= 250).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    free_c = cv2.morphologyEx(free_c, cv2.MORPH_OPEN, k)
    nlab, comps = cv2.connectedComponents(free_c)
    min_px = int(1.0/(res*res))
    sizes = np.bincount(comps.ravel()); sizes[0] = 0
    keep = np.isin(comps, np.where(sizes >= min_px)[0])
    free_c = keep.astype(np.uint8)
    dist = cv2.distanceTransform(free_c, cv2.DIST_L2, 5)
    from scipy import ndimage
    from skimage.segmentation import watershed
    t_m, _ = rs.auto_seed_threshold(dist, res)
    seeds, _ = ndimage.label(dist >= t_m/res)
    lab = watershed(-dist, seeds, mask=free_c.astype(bool)).astype(np.int32)
    lab = rs.merge_small(lab, res, 2.0, 12.5)
    full = np.zeros((Hf, Wf), np.int32)
    full[y0:y1, x0:x1] = lab
    return full


def run_B(inp, out_prefix):
    spec = importlib.util.spec_from_file_location("rv", os.path.join(TOOLS, "room_segment_voronoi.py"))
    rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
    raw = np.array(Image.open(inp))
    if raw.ndim == 3:
        raw = raw[..., 0]
    res = RES
    Hf, Wf = raw.shape
    occ, free = raw <= 50, raw >= 250
    ys, xs = np.where(free | occ)
    m = 12
    y0, y1 = max(0, ys.min()-m), min(Hf, ys.max()+m)
    x0, x1 = max(0, xs.min()-m), min(Wf, xs.max()+m)
    crop = raw[y0:y1, x0:x1]
    free_c = (crop >= 250).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    free_c = cv2.morphologyEx(free_c, cv2.MORPH_OPEN, k)
    nlab, comps = cv2.connectedComponents(free_c)
    min_px = int(1.0/(res*res))
    sizes = np.bincount(comps.ravel()); sizes[0] = 0
    keep = np.isin(comps, np.where(sizes >= min_px)[0])
    free_c = keep.astype(np.uint8)
    lab = rv.voronoi_segment(free_c, res, 2.0, 12.5, verbose=True)
    full = np.zeros((Hf, Wf), np.int32)
    full[y0:y1, x0:x1] = lab
    return full


def evaluate(pred, gt, ngt):
    """Region-overlap recall/precision + under/over counts on common free px."""
    common = (gt > 0) & (pred > 0)
    # recall: per GT room, fraction covered by its best single pred region
    recalls = []
    gt_to_pred = {}      # which pred region owns the majority of each GT room
    for r in range(1, ngt+1):
        gm = (gt == r)
        gtot = int(gm.sum())
        if gtot == 0:
            continue
        pvals, pc = np.unique(pred[gm & (pred > 0)], return_counts=True)
        if len(pvals) == 0:
            recalls.append(0.0); continue
        best = int(pvals[np.argmax(pc)])
        recalls.append(pc.max() / gtot)
        gt_to_pred[r] = best
    # precision: per pred region, fraction inside its best single GT room
    precs = []
    pred_to_gt = {}
    for p in [int(v) for v in np.unique(pred[pred > 0])]:
        pm = (pred == p)
        ptot_overlap = int((pm & (gt > 0)).sum())   # restrict to mapped GT area
        if ptot_overlap == 0:
            continue
        gvals, gc = np.unique(gt[pm & (gt > 0)], return_counts=True)
        best = int(gvals[np.argmax(gc)])
        precs.append(gc.max() / ptot_overlap)
        pred_to_gt[p] = best
    recall = float(np.mean(recalls)) if recalls else 0.0
    precision = float(np.mean(precs)) if precs else 0.0

    # under-segmentation: a predicted region that is the MAJORITY owner of >=2
    # GT rooms merges those rooms. Count merged pairs.
    from collections import defaultdict
    owner = defaultdict(list)
    for r, p in gt_to_pred.items():
        owner[p].append(r)
    under_pairs = 0
    under_groups = []
    for p, rs_ in owner.items():
        if len(rs_) >= 2:
            under_pairs += len(rs_) - 1     # n rooms merged = n-1 merged pairs
            under_groups.append(sorted(rs_))

    # over-segmentation: a GT room split across >=2 predicted regions that each
    # claim a meaningful share (>=15% of GT room).
    over_count = 0
    over_rooms = []
    for r in range(1, ngt+1):
        gm = (gt == r); gtot = int(gm.sum())
        if gtot == 0:
            continue
        pvals, pc = np.unique(pred[gm & (pred > 0)], return_counts=True)
        sig = int((pc >= 0.15 * gtot).sum())
        if sig >= 2:
            over_count += sig - 1
            over_rooms.append((r, sig))
    return dict(recall=recall, precision=precision, n_pred=len(precs),
                under_pairs=under_pairs, under_groups=under_groups,
                over_count=over_count, over_rooms=over_rooms)


def colorize(lab, gray):
    PAL = [(231,76,60),(52,152,219),(46,204,113),(155,89,182),(241,196,15),
           (230,126,34),(26,188,156),(149,165,166),(192,57,43),(41,128,185),
           (39,174,96),(142,68,173),(243,156,18),(211,84,0),(127,140,141)]
    vis = np.stack([gray]*3, -1).astype(np.uint8)
    ov = vis.copy()
    for i, v in enumerate([int(x) for x in np.unique(lab[lab>0])]):
        ov[lab==v] = PAL[i % len(PAL)]
    return (0.5*vis + 0.5*ov).astype(np.uint8)


def main():
    gt_path = os.path.join(BASE, "Freiburg52_scan_gt_segmentation.png")
    gt, ngt = gt_labels(gt_path)
    print("GT rooms:", ngt)
    g_gray = np.array(Image.open(gt_path))
    if g_gray.ndim == 3: g_gray = g_gray[...,0]

    maps = ["Freiburg52_scan_furnitures.png",
            "Freiburg52_scan_furnitures_trashbins.png"]
    results = {}
    for mp in maps:
        src = os.path.join(BASE, mp)
        gray = to_gray_png(src, os.path.join(BASE, "_gray_"+mp))
        graypix = np.array(Image.open(gray))
        print("\n==== MAP:", mp, "====")
        print("-- A (distance-transform) --")
        labA = run_A(gray, None)
        print("-- B (voronoi) --")
        labB = run_B(gray, None)
        rA = evaluate(labA, gt, ngt)
        rB = evaluate(labB, gt, ngt)
        results[mp] = (rA, rB)
        for nm, r in [("A", rA), ("B", rB)]:
            print(f"  {nm}: recall={r['recall']:.3f} precision={r['precision']:.3f} "
                  f"n_pred={r['n_pred']} under={r['under_pairs']} over={r['over_count']}")
            print(f"      under_groups(GT ids merged)={r['under_groups']}")
            print(f"      over_rooms(GT id, #pieces)={r['over_rooms']}")
        # side-by-side PNG: GT | A | B
        cA = colorize(labA, graypix)
        cB = colorize(labB, graypix)
        cG = colorize(gt, g_gray)
        def lab_label_in(img, txt):
            return cv2.putText(img.copy(), txt, (8,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cG = lab_label_in(cG, "GT (%d rooms)"%ngt)
        cA = lab_label_in(cA, "A dist-transform")
        cB = lab_label_in(cB, "B voronoi")
        gap = np.full((cG.shape[0], 6, 3), 255, np.uint8)
        combo = np.hstack([cG, gap, cA, gap, cB])
        s = 2
        combo = cv2.resize(combo, (combo.shape[1]*s, combo.shape[0]*s),
                           interpolation=cv2.INTER_NEAREST)
        tag = mp.replace("Freiburg52_scan_","").replace(".png","")
        outp = os.path.join(BASE, "cmp_voronoi_%s.png"%tag)
        Image.fromarray(combo[...,::-1] if False else combo).save(outp)
        print("  wrote:", outp)

    # summary table
    print("\n\n========== SUMMARY (recall / precision / under / over) ==========")
    print("%-40s | %-26s | %-26s" % ("map", "A dist-transform", "B voronoi"))
    for mp,(rA,rB) in results.items():
        print("%-40s | r=%.3f p=%.3f u=%d o=%d | r=%.3f p=%.3f u=%d o=%d" % (
            mp[:40], rA['recall'],rA['precision'],rA['under_pairs'],rA['over_count'],
            rB['recall'],rB['precision'],rB['under_pairs'],rB['over_count']))


if __name__ == "__main__":
    main()
