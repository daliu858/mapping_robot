# -*- coding: utf-8 -*-
"""
room_segment.py — Level 2: occupancy grid -> rooms + doors + topology (offline)

Pipeline (after Bormann et al., ICRA 2016, "Room Segmentation: Survey,
Implementation, and Analysis"): ternarize PGM -> keep main free component ->
distance transform -> auto-threshold seeds -> watershed -> merge rules
(<2 m^2 into neighbor; <12.5 m^2 single-neighbor merge) -> doorway detection
on medial-axis skeleton (clearance local minimum between two rooms,
width 0.4-1.4 m) -> YAML + colored PNG.

Usage:
  python room_segment.py "d:/slam car/room1.pgm" --yaml map.yaml
         --objects semantic_objects.yaml
         [--seed-dist auto|0.45] [--min-room 2.0] [--out prefix]
"""
import argparse, hashlib, os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import medial_axis
import yaml

try:
    from map_identity import load_map_identity
    from semantic_artifact_contract import (
        TOPOLOGY_SCHEMA, atomic_yaml_dump, load_bound_objects)
except ImportError:  # imported from the repository root by unit tests
    from tools.map_identity import load_map_identity
    from tools.semantic_artifact_contract import (
        TOPOLOGY_SCHEMA, atomic_yaml_dump, load_bound_objects)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _shift_without_wrap(array, shift, axis, fill_value=0):
    """Shift an array while treating pixels beyond the image as absent."""
    shifted = np.roll(array, shift, axis=axis)
    edge = [slice(None)] * shifted.ndim
    if shift > 0:
        edge[axis] = slice(0, shift)
    elif shift < 0:
        edge[axis] = slice(shift, None)
    else:
        return shifted
    shifted[tuple(edge)] = fill_value
    return shifted


def _connected_components_8(binary):
    """OpenCV's second positional argument is an output buffer, not connectivity."""
    return cv2.connectedComponents(binary, connectivity=8)


def load_map(pgm_path, yaml_path=None):
    img = np.array(Image.open(pgm_path))
    res, origin = 0.05, None
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path) as f:
            m = yaml.safe_load(f)
        res = float(m.get("resolution", 0.05))
        origin = [float(v) for v in m.get("origin", [0, 0, 0])[:2]]
    return img, res, origin


def px_to_map(col, row, H, res, origin):
    if origin is None:
        return [round(float(col), 1), round(float(row), 1)]  # pixel coords
    return [round(origin[0] + (col + 0.5) * res, 3),
            round(origin[1] + (H - row - 0.5) * res, 3)]


def auto_seed_threshold(dist, res, lo=0.3, hi=None, step=0.05):
    """Bormann-style: scan threshold, keep the one giving most components."""
    if hi is None:
        # ``dist`` is in pixels; the threshold loop is expressed in metres.
        hi = max(lo + step, float(dist.max()) * float(res) * 0.9)
    best_t, best_n = lo, 0
    t = lo
    while t <= hi:
        n = ndimage.label(dist >= t / res)[1]
        if n >= best_n:   # prefer larger t on ties -> tighter seeds
            best_t, best_n = t, n
        t += step
    return best_t, best_n


def neighbors_table(lab):
    """label -> {neighbor_label: shared_border_px} via 4-neighbor shifts."""
    nb = {}
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        a = lab
        b = _shift_without_wrap(lab, sh, axis=ax)
        m = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[m].ravel(), b[m].ravel()):
            nb.setdefault(int(u), {})
            nb[int(u)][int(v)] = nb[int(u)].get(int(v), 0) + 1
    return nb


def merge_small(lab, res, min_room=2.0, single_nb=12.5):
    """Iteratively apply Bormann merge rules."""
    px_area = res * res
    for _ in range(50):
        ids, cnts = np.unique(lab[lab > 0], return_counts=True)
        if len(ids) <= 1:
            break
        area = {int(i): c * px_area for i, c in zip(ids, cnts)}
        nb = neighbors_table(lab)
        changed = False
        for i in sorted(ids, key=lambda x: area[int(x)]):
            i = int(i)
            nbi = nb.get(i, {})
            tgt = None
            if area[i] < min_room and nbi:
                tgt = max(nbi, key=nbi.get)
            elif area[i] < single_nb and len(nbi) == 1:
                tgt = next(iter(nbi))
            if tgt:
                lab[lab == i] = tgt
                changed = True
                break  # recompute tables after each merge
        if not changed:
            break
    # drop isolated specks with no neighbors
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    for i, c in zip(ids, cnts):
        if c * px_area < 0.5:
            lab[lab == i] = 0
    return lab


def detect_doors(lab, free, res, w_lo=0.6, w_hi=1.4):
    """Doorways = clearance local minima on skeleton between two rooms."""
    skel, sdist = medial_axis(free, return_distance=True)
    H, W = lab.shape
    pairs = {}
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 3), min(H, y + 4)
        x0, x1 = max(0, x - 3), min(W, x + 4)
        ls = np.unique(lab[y0:y1, x0:x1])
        ls = ls[ls > 0]
        if len(ls) >= 2:
            a, b = int(ls[0]), int(ls[1])
            key = (min(a, b), max(a, b))
            width = 2.0 * float(sdist[y, x]) * res
            if key not in pairs or width < pairs[key][0]:
                pairs[key] = (width, int(x), int(y))
    doors = []
    for (a, b), (w, x, y) in sorted(pairs.items()):
        kind = "door" if w_lo <= w <= w_hi else ("narrow" if w < w_lo else "opening")
        doors.append({"rooms": [a, b], "px": [x, y], "width_m": round(w, 2),
                      "type": kind})
    return doors


PALETTE = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (155, 89, 182),
           (241, 196, 15), (230, 126, 34), (26, 188, 156), (149, 165, 166),
           (192, 57, 43), (41, 128, 185), (39, 174, 96), (142, 68, 173)]


def visualize(crop_gray, lab, doors, rooms, out_png):
    H, W = lab.shape
    vis = np.stack([crop_gray] * 3, axis=-1).astype(np.uint8)
    overlay = vis.copy()
    for i, r in enumerate(rooms):
        c = (200, 200, 200) if r.get("status") == "phantom" \
            else PALETTE[i % len(PALETTE)]
        overlay[lab == r["id"]] = c
    vis = (0.55 * vis + 0.45 * overlay).astype(np.uint8)
    # boundaries in black
    er = cv2.Canny((lab > 0).astype(np.uint8) * 80 + lab.astype(np.uint8), 1, 1)
    vis[er > 0] = (30, 30, 30)
    s = max(1, int(900 / max(H, W)))
    vis = cv2.resize(vis, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(vis)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    for i, r in enumerate(rooms):
        cx, cy = r["centroid_px"]
        tag = "R%d %.1fm2" % (r["id"], r["area_m2"])
        if r.get("status") == "phantom":
            tag = "(phantom) " + tag
        dr.text((cx * s, cy * s), tag, fill=(0, 0, 0), font=font, anchor="mm")
    for d in doors:
        x, y = d["px"]
        col = (255, 0, 0) if d["type"] == "door" else (255, 140, 0)
        rpx = 6
        dr.ellipse([x * s - rpx, y * s - rpx, x * s + rpx, y * s + rpx],
                   outline=col, width=3)
        dr.text((x * s, y * s - 14), "%.2fm" % d["width_m"], fill=col,
                font=font, anchor="mb")
    img.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm")
    ap.add_argument(
        "--yaml", required=True,
        help="canonical ROS map YAML for the supplied map image")
    ap.add_argument(
        "--objects", required=True,
        help="completed semantic_objects artifact from the same survey run")
    ap.add_argument("--seed-dist", default="auto")
    ap.add_argument("--min-room", type=float, default=2.0)
    ap.add_argument("--single-nb", type=float, default=12.5)
    ap.add_argument("--close-px", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    map_identity = load_map_identity(
        a.yaml, allow_portable_sibling=True, require_zero_yaw=True)
    pgm_hash = file_sha256(a.pgm)
    if pgm_hash != map_identity["map_image_sha256"]:
        raise ValueError(
            "input PGM and canonical map YAML image identities differ")
    _objects, object_provenance, objects_hash = load_bound_objects(
        a.objects, map_identity)
    raw, _res, _origin = load_map(a.pgm, a.yaml)
    res = float(map_identity["manifest"]["resolution"])
    origin = [float(value)
              for value in map_identity["manifest"]["origin"]]
    Hf, Wf = raw.shape
    occ, unk, free = raw <= 50, (raw > 50) & (raw < 250), raw >= 250
    if not free.any():
        sys.exit("no free space in map")
    ys, xs = np.where(free | occ)
    m = 12
    y0, y1 = max(0, ys.min() - m), min(Hf, ys.max() + m)
    x0, x1 = max(0, xs.min() - m), min(Wf, xs.max() + m)
    crop = raw[y0:y1, x0:x1]
    free_c = (crop >= 250).astype(np.uint8)

    # clean: close hairline gaps in free space, keep main component only
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close_px * 2 + 1,) * 2)
    free_c = cv2.morphologyEx(free_c, cv2.MORPH_OPEN, k)
    # keep every component >= 1 m^2 (not just the largest: a half-scanned
    # room may be shattered into pieces smaller than a phantom lobe)
    nlab, comps = _connected_components_8(free_c)
    min_px = int(1.0 / (res * res))
    sizes = np.bincount(comps.ravel()); sizes[0] = 0
    keep = np.isin(comps, np.where(sizes >= min_px)[0])
    free_c = keep.astype(np.uint8)

    dist = cv2.distanceTransform(free_c, cv2.DIST_L2, 5)
    if a.seed_dist == "auto":
        t_m, nseeds = auto_seed_threshold(dist, res)
    else:
        t_m, nseeds = float(a.seed_dist), -1
    seeds, nmark = ndimage.label(dist >= t_m / res)
    print("seed threshold = %.2f m -> %d seeds" % (t_m, nmark))
    lab = watershed(-dist, seeds, mask=free_c.astype(bool)).astype(np.int32)
    lab = merge_small(lab, res, a.min_room, a.single_nb)

    # rooms table; border touching unknown instead of walls => phantom
    # (lidar rays leaking through an opening, not a really mapped room)
    unk_c = (crop > 50) & (crop < 250)
    rooms = []
    for i in [int(v) for v in np.unique(lab[lab > 0])]:
        mask = (lab == i).astype(np.uint8)
        area = float(mask.sum()) * res * res
        ys_, xs_ = np.where(mask)
        cy, cx = float(ys_.mean()), float(xs_.mean())
        ring = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool) & ~mask.astype(bool)
        n_ring = max(1, int(ring.sum()))
        unk_ratio = float((ring & unk_c).sum()) / n_ring
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 2.0, True)
        rooms.append({
            "id": i, "area_m2": round(area, 2),
            "status": "phantom" if unk_ratio > 0.5 else "room",
            "unknown_border_ratio": round(unk_ratio, 2),
            "centroid_px": [int(cx), int(cy)],
            "centroid_map": px_to_map(cx + x0, cy + y0, Hf, res, origin),
            "polygon_map": [px_to_map(p[0][0] + x0, p[0][1] + y0, Hf, res,
                                      origin) for p in poly],
        })
    doors = detect_doors(lab, free_c.astype(bool), res)
    for d in doors:
        d["pos_map"] = px_to_map(d["px"][0] + x0, d["px"][1] + y0, Hf, res,
                                 origin)

    out = a.out or os.path.splitext(a.pgm)[0]
    data = {
        "schema": TOPOLOGY_SCHEMA,
        "completed": True,
        "frame": "map",
        "recording_token": object_provenance["recording_token"],
        "source_bag_sha256": object_provenance["source_bag_sha256"],
        "camera_mount": object_provenance["camera_mount"],
        "source_objects": os.path.basename(a.objects),
        "source_objects_sha256": objects_hash,
        "source": os.path.basename(a.pgm), "resolution": res,
        "map_image_sha256": map_identity["map_image_sha256"],
        "map_manifest_sha256": map_identity["map_manifest_sha256"],
        "origin": origin,
        "rooms": rooms,
        "doors": [{k: v for k, v in d.items() if k != "px"} for d in doors],
        "topology": [{"edge": d["rooms"], "via": d["type"],
                      "width_m": d["width_m"]} for d in doors],
    }
    ypath = out + "_topology.yaml"
    ppath = visualize(crop, lab, doors, rooms, out + "_rooms.png")
    atomic_yaml_dump(data, ypath)
    print("rooms=%d doors=%d" % (len(rooms), len(doors)))
    print("wrote:", ypath)
    print("wrote:", ppath)


if __name__ == "__main__":
    main()
