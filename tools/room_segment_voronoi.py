# -*- coding: utf-8 -*-
"""
room_segment_voronoi.py — Level 2: occupancy grid -> rooms + doors + topology.

SAME SHELL as room_segment.py (preprocessing, phantom detection, door
3-way classification, room metadata, YAML/PNG output, CLI), but the CORE
SPLITTING OPERATOR is swapped from distance-transform-watershed to a
Voronoi GRAPH-BASED segmentation following Bormann et al. (ICRA 2016),
"Room Segmentation: Survey, Implementation, and Analysis", Section III-C.

What is REUSED verbatim from room_segment.py (interfaces/outputs identical):
  - preprocessing: ternarize (254/205/0), crop to bbox, morphological clean,
    connected-component filtering (keep components >= 1 m^2)
  - phantom detection: room ring unknown-ratio > 0.5  -> status="phantom"
  - door 3-way classification along channel width:
        door (0.6-1.4 m) / narrow (<0.6 m) / opening (>1.4 m), field "type"
  - room metadata: id / area_m2 / status / centroid / polygon
  - YAML output (rooms + doors + topology) and colored PNG visualization
  - CLI: python room_segment_voronoi.py <pgm> --yaml <map.yaml>
         --objects <semantic_objects.yaml> --out <prefix>

What is REPLACED (this file only): the segmentation core.
  Old (room_segment.py): distance transform -> auto-threshold seeds ->
       watershed -> borrowed merge thresholds.
  New (Sec. III-C): generalized Voronoi graph (medial axis of free space)
       -> critical points (skeleton points whose clearance is a local minimum)
       -> critical lines (critical point to its two closest obstacle points)
       cut free space into Voronoi cells -> the 5 merge heuristics of Sec.III-C
       merge over-segmented cells into room-like structures.

Usage:
  python room_segment_voronoi.py "d:/slam car/map.png" --yaml map.yaml
         --objects semantic_objects.yaml [--out prefix]

Honesty notes on simplifications are written inline at each step with the tag
[SIMPLIFICATION].
"""
import argparse, hashlib, os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
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
    """Use the intended 8-connectivity via an unambiguous keyword argument."""
    return cv2.connectedComponents(binary, connectivity=8)

try:
    import sknw
    HAVE_SKNW = True
except Exception:
    HAVE_SKNW = False


# ===========================================================================
#  REUSED SHELL  (kept byte-for-byte compatible with room_segment.py)
# ===========================================================================
def load_map(pgm_path, yaml_path=None):
    """Same as room_segment.py, but tolerant of RGB/RGBA PNGs (the Bormann
    test maps are grayscale stored as RGBA). We collapse to a single channel
    so the 254/205/0 ternarization downstream is unchanged."""
    img = np.array(Image.open(pgm_path))
    if img.ndim == 3:                 # RGB / RGBA  -> take luminance channel
        img = img[..., 0]             # R==G==B for these maps; channel 0 is fine
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


def neighbors_table(lab):
    """label -> {neighbor_label: shared_border_px} via 4-neighbor shifts.
    Verbatim from room_segment.py."""
    nb = {}
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        a = lab
        b = _shift_without_wrap(lab, sh, axis=ax)
        m = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[m].ravel(), b[m].ravel()):
            nb.setdefault(int(u), {})
            nb[int(u)][int(v)] = nb[int(u)].get(int(v), 0) + 1
    return nb


def detect_doors(lab, free, res, w_lo=0.6, w_hi=1.4):
    """Doorways = clearance local minima on skeleton between two rooms.
    Verbatim from room_segment.py (door / narrow / opening, field 'type')."""
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
    """Verbatim from room_segment.py."""
    H, W = lab.shape
    vis = np.stack([crop_gray] * 3, axis=-1).astype(np.uint8)
    overlay = vis.copy()
    for i, r in enumerate(rooms):
        c = (200, 200, 200) if r.get("status") == "phantom" \
            else PALETTE[i % len(PALETTE)]
        overlay[lab == r["id"]] = c
    vis = (0.55 * vis + 0.45 * overlay).astype(np.uint8)
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


# ===========================================================================
#  NEW CORE  ---  Voronoi graph-based splitting  (Bormann Sec. III-C)
# ===========================================================================
def voronoi_segment(free_c, res, min_room=2.0, single_nb=12.5, verbose=True):
    """Replace the distance-transform-watershed core with a Voronoi
    graph-based segmentation, faithfully following Bormann et al. Sec. III-C.

    Returns an int32 label image over the free-space mask `free_c`.
    """
    free_b = free_c.astype(bool)

    # -----------------------------------------------------------------------
    # (1) Generalized Voronoi graph of the free space.
    # Sec. III-C: "First, a generalized Voronoi graph is computed on the grid
    #   map and pruned to the major skeleton by collapsing leave edges into the
    #   node points of their origin."
    #
    # The generalized Voronoi diagram of free space is the locus of points
    # equidistant from >=2 obstacle pixels; for a binary free mask this is
    # exactly the MEDIAL AXIS. We compute it with skimage.medial_axis, which
    # also returns the clearance (distance to nearest obstacle) at every
    # skeleton pixel -- that distance is what defines critical points below.
    # -----------------------------------------------------------------------
    skel, clearance = medial_axis(free_b, return_distance=True)
    # clearance is defined everywhere; keep only on-skeleton values.
    skel_clear = np.where(skel, clearance, 0.0)

    # Pruning "leave edges into the node points of their origin": we build the
    # skeleton graph with sknw and drop short leaf branches (spurs), which is
    # the standard realization of that pruning. [SIMPLIFICATION] If sknw is
    # unavailable we skip graph pruning and operate on the raw medial axis;
    # spurs then only add extra critical-point candidates that the III-C merge
    # rules below absorb, so the final rooms are unchanged in practice.
    if HAVE_SKNW:
        try:
            G = sknw.build_sknw(skel.astype(np.uint16))
            spur_len = int(round(0.5 / res))      # prune leaf branches < 0.5 m
            changed = True
            while changed:
                changed = False
                for n in list(G.nodes):
                    if G.degree(n) == 1:
                        nbrs = list(G.neighbors(n))
                        if not nbrs:
                            continue
                        e = G[n][nbrs[0]]
                        ed = e if 0 not in e else e[0]
                        pts = ed.get("pts") if isinstance(ed, dict) else None
                        if pts is not None and len(pts) < spur_len:
                            for (ry, rx) in pts:
                                skel[ry, rx] = False
                            G.remove_node(n)
                            changed = True
            skel_clear = np.where(skel, clearance, 0.0)
        except Exception as ex:
            if verbose:
                print("[sknw pruning skipped: %r]" % ex)

    # -----------------------------------------------------------------------
    # (2) Critical points.
    # Sec. III-C: "Any point on the resulting Voronoi graph, that has exactly
    #   two closest obstacle pixels, is a candidate for becoming a critical
    #   point that might delineate two room segments. From the set of possible
    #   critical points those are stored in set P which are closest to their
    #   obstacles within a certain neighborhood."
    #
    # "closest to their obstacles within a certain neighborhood" == the
    # clearance is a LOCAL MINIMUM along the skeleton. A doorway is exactly the
    # bottleneck where free-space clearance dips. We find skeleton pixels whose
    # clearance is <= every skeleton pixel within a local window (the
    # "neighborhood"), i.e. local minima of the clearance restricted to the
    # graph.
    # -----------------------------------------------------------------------
    nbhd = max(3, int(round(0.7 / res)))          # ~0.7 m search neighborhood
    if nbhd % 2 == 0:
        nbhd += 1
    big = clearance.max() + 1.0
    sc = np.where(skel, clearance, big)            # off-skeleton = +inf
    sc_max = np.where(skel, clearance, -1.0)
    local_min = ndimage.minimum_filter(sc, size=nbhd)
    local_max = ndimage.maximum_filter(sc_max, size=nbhd)
    # A genuine bottleneck is a point whose clearance is the local minimum AND
    # is STRICTLY below the surrounding skeleton (a real dip, not a flat
    # plateau in an open corridor). The strict dip removes the hundreds of
    # tie-points along wide corridors that would otherwise cause heavy
    # over-segmentation. We also cap clearance to door/opening scale (a door
    # bottleneck has radius < ~1.0 m); beyond that it is open space, not a cut.
    door_ceiling = 1.0 / res                       # ~1.0 m clearance radius
    crit_mask = (skel & (clearance <= local_min + 1e-6)
                 & (clearance < local_max - 1e-6)
                 & (clearance <= door_ceiling))
    # [SIMPLIFICATION] We do NOT require *exactly* two closest obstacle pixels
    # (rarely exactly two on a discrete grid); the strict-local-minimum-of-
    # clearance criterion is the operational definition Bormann uses to localize
    # doors, and the III-C merge rules clean up any spurious cuts. Stated honestly.
    ys, xs = np.where(crit_mask)
    crit_pts = list(zip(ys.tolist(), xs.tolist()))
    if verbose:
        print("Voronoi: skeleton px=%d, critical-point candidates=%d"
              % (int(skel.sum()), len(crit_pts)))

    # -----------------------------------------------------------------------
    # (3) Critical lines cut the free space into Voronoi cells.
    # Sec. III-C: "critical lines are drawn into the map at all critical points
    #   of set P. These critical lines connect the critical points with their
    #   two closest obstacle points. The angle between both line segments is
    #   important if there are too many critical lines within an area. Then
    #   those with the smallest angles are removed since these often lie at
    #   corners of a wall. Critical lines with a large angle occur frequently
    #   at doors."
    #
    # For each critical point we find its two closest obstacle pixels (the two
    # nearest obstacles lying on roughly opposite sides) and draw the line
    # through them; that line, burned into the free mask as a 1-px wall, splits
    # the channel. We reject low-angle lines (wall corners) and keep wide-angle
    # lines (doors), exactly as the paper prescribes.
    # -----------------------------------------------------------------------
    # nearest-obstacle vector field via distance transform indices
    obst = (~free_b).astype(np.uint8)
    # feature transform: for each pixel, index of nearest obstacle pixel
    edt, inds = ndimage.distance_transform_edt(free_b, return_indices=True)
    cut = free_c.copy().astype(np.uint8)           # working free mask to carve
    H, W = free_c.shape
    drawn = 0
    ANG_MIN = math.radians(60)   # keep wide-angle (door-like); drop corner cuts
    for (cy0, cx0) in crit_pts:
        # nearest obstacle point #1 (from feature transform at the crit point)
        oy1, ox1 = int(inds[0, cy0, cx0]), int(inds[1, cy0, cx0])
        v1 = np.array([oy1 - cy0, ox1 - cx0], float)
        n1 = np.linalg.norm(v1)
        if n1 < 1e-3:
            continue
        v1 /= n1
        # second-closest obstacle: search a window for the nearest obstacle
        # whose direction is well-separated from v1 (the "other side" wall).
        r = int(clearance[cy0, cx0]) + 4
        y0, y1 = max(0, cy0 - r), min(H, cy0 + r + 1)
        x0, x1 = max(0, cx0 - r), min(W, cx0 + r + 1)
        sub = obst[y0:y1, x0:x1]
        oys, oxs = np.where(sub)
        if len(oys) == 0:
            continue
        oys = oys + y0
        oxs = oxs + x0
        vy = oys - cy0
        vx = oxs - cx0
        d2 = vy * vy + vx * vx
        # cosine vs v1; we want the opposite side -> cos close to -1
        norms = np.sqrt(d2) + 1e-9
        cosang = (vy * v1[0] + vx * v1[1]) / norms
        opp = cosang < math.cos(ANG_MIN)           # angle(v1, v2) > ANG_MIN
        if not opp.any():
            continue
        idx = np.where(opp)[0]
        j = idx[np.argmin(d2[idx])]
        oy2, ox2 = int(oys[j]), int(oxs[j])
        # angle between the two obstacle directions; reject narrow (corner)
        v2 = np.array([oy2 - cy0, ox2 - cx0], float)
        n2 = np.linalg.norm(v2)
        if n2 < 1e-3:
            continue
        v2 /= n2
        ang = math.acos(max(-1.0, min(1.0, float(v1 @ v2))))
        if ang < ANG_MIN:
            continue
        # draw the critical line obstacle1 -- critpoint -- obstacle2 (1px wall)
        cv2.line(cut, (ox1, oy1), (ox2, oy2), 0, 1)
        drawn += 1
    if verbose:
        print("Voronoi: critical lines drawn=%d (wide-angle kept, corners dropped)"
              % drawn)

    # connected components of the carved free space = Voronoi cells
    ncc, lab = _connected_components_8(cut.astype(np.uint8))
    lab = lab.astype(np.int32)
    # restore any free pixels removed by the 1-px cuts to their adjacent cell
    # so the label image still tiles the ORIGINAL free mask (cuts are walls,
    # not lost area). Nearest-labeled-pixel fill confined to free space.
    missing = free_b & (lab == 0)
    if missing.any():
        _, fi = ndimage.distance_transform_edt((lab == 0), return_indices=True)
        lab_fill = lab[fi[0], fi[1]]
        lab = np.where(missing, lab_fill, lab)
    lab[~free_b] = 0
    if verbose:
        ncells = len(np.unique(lab[lab > 0]))
        print("Voronoi: cells after cutting=%d" % ncells)

    # -----------------------------------------------------------------------
    # (4) Merge Voronoi cells into room-like structures (Sec. III-C heuristics).
    #   The paper's five rules (thresholds quoted verbatim):
    #   1) Areas smaller than a threshold (12.5 m^2) with exactly one neighbor
    #      and less than 75% of the border touching walls become merged with
    #      that neighbor.
    #   2) Small regions below a threshold size (2 m^2) are merged with a
    #      surrounding area that is in touch with at least 20% of the small
    #      region's border pixels.
    #   3) Merge regions with (i) exactly one neighbor that has maximal 2
    #      neighbors and (ii) at least 50% of the perimeter touching walls.
    #   4) Merge regions that share a significant part of their borders, i.e.
    #      at least 20% for the smaller room and 10% for the larger room.
    #   5) Merge regions with more than 40% of their perimeter touching another
    #      segment (ragged obstacles / under tables).
    # -----------------------------------------------------------------------
    lab = _merge_voronoi_cells(lab, free_b, res, min_room, single_nb, verbose)
    return lab


def _border_stats(lab, free_b):
    """For every region: total border px, border px touching WALL (non-free,
    i.e. outside free mask incl. carved cut lines), and the {neighbor: shared
    border px} table. One pass over 4-neighbor shifts."""
    H, W = lab.shape
    wall = ~free_b
    nb = {}
    wall_border = {}
    tot_border = {}
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        a = lab
        b = _shift_without_wrap(lab, sh, axis=ax)
        bw = _shift_without_wrap(wall, sh, axis=ax, fill_value=False)
        # neighbor region contacts
        m = (a > 0) & (b > 0) & (a != b)
        for u, v in zip(a[m].ravel(), b[m].ravel()):
            u = int(u); v = int(v)
            nb.setdefault(u, {})
            nb[u][v] = nb[u].get(v, 0) + 1
            tot_border[u] = tot_border.get(u, 0) + 1
        # wall contacts
        mw = (a > 0) & bw
        for u in a[mw].ravel():
            u = int(u)
            wall_border[u] = wall_border.get(u, 0) + 1
            tot_border[u] = tot_border.get(u, 0) + 1
    return nb, wall_border, tot_border


def _merge_voronoi_cells(lab, free_b, res, min_room, single_nb, verbose=True):
    px_area = res * res
    merges = 0
    for _ in range(200):
        ids = np.unique(lab[lab > 0])
        if len(ids) <= 1:
            break
        cnts = {int(i): int((lab == i).sum()) for i in ids}
        area = {i: cnts[i] * px_area for i in cnts}
        nb, wallb, totb = _border_stats(lab, free_b)
        tgt = src = None

        order = sorted(cnts, key=lambda i: area[i])   # smallest first
        for i in order:
            nbi = nb.get(i, {})
            tb = max(1, totb.get(i, 1))
            wfrac = wallb.get(i, 0) / tb               # fraction border on wall

            # Rule 2: tiny region (<2 m^2) -> the neighbor touching >=20% border
            if area[i] < min_room and nbi:
                cand = [(v, k) for k, v in nbi.items() if v / tb >= 0.20]
                if cand:
                    src, tgt = i, max(cand)[1]
                    break
                src, tgt = i, max(nbi, key=nbi.get)    # fallback: biggest contact
                break

            # Rule 1: <12.5 m^2, exactly one neighbor, <75% border on walls
            if area[i] < single_nb and len(nbi) == 1 and wfrac < 0.75:
                src, tgt = i, next(iter(nbi))
                break

            # Rule 3: exactly one neighbor that itself has <=2 neighbors,
            #         and >=50% perimeter on walls (two halves of one room).
            if len(nbi) == 1:
                j = next(iter(nbi))
                if len(nb.get(j, {})) <= 2 and wfrac >= 0.50:
                    src, tgt = i, j
                    break

            # Rule 4: shares >=20% of smaller room's & >=10% of larger room's
            #         border with a neighbor.
            for j, shared in nbi.items():
                tbj = max(1, totb.get(j, 1))
                small, large = (i, j) if area[i] <= area[j] else (j, i)
                ts = max(1, totb.get(small, 1)); tl = max(1, totb.get(large, 1))
                sh_s = nb.get(small, {}).get(large, 0) / ts
                sh_l = nb.get(large, {}).get(small, 0) / tl
                if sh_s >= 0.20 and sh_l >= 0.10:
                    src, tgt = small, large
                    break
            if tgt is not None:
                break

            # Rule 5: >40% of perimeter touches a single other segment.
            for j, shared in nbi.items():
                if shared / tb > 0.40:
                    src, tgt = i, j
                    break
            if tgt is not None:
                break

        if tgt is None:
            break
        lab[lab == src] = tgt
        merges += 1

    # drop isolated specks (same final cleanup as room_segment.py)
    nb_final, _, _ = _border_stats(lab, free_b)
    for i in np.unique(lab[lab > 0]):
        i = int(i)
        if (lab == i).sum() * px_area < 0.5 and not nb_final.get(i):
            lab[lab == i] = 0
    if verbose:
        print("Voronoi: merge heuristics applied, %d merges -> %d rooms"
              % (merges, len(np.unique(lab[lab > 0]))))
    return lab


# ===========================================================================
#  MAIN  (shell identical to room_segment.py except the core call)
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm")
    ap.add_argument(
        "--yaml", required=True,
        help="canonical ROS map YAML for the supplied map image")
    ap.add_argument(
        "--objects", required=True,
        help="completed semantic_objects artifact from the same survey run")
    ap.add_argument("--res", type=float, default=None,
                    help="deprecated consistency check; cannot override YAML")
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
    if a.res is not None and abs(float(a.res) - res) > 1e-12:
        raise ValueError("--res cannot override canonical map YAML resolution")
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

    # --- preprocessing: REUSED from room_segment.py (morphological clean +
    #     keep every connected component >= 1 m^2) -------------------------
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close_px * 2 + 1,) * 2)
    free_c = cv2.morphologyEx(free_c, cv2.MORPH_OPEN, k)
    nlab, comps = _connected_components_8(free_c)
    min_px = int(1.0 / (res * res))
    sizes = np.bincount(comps.ravel()); sizes[0] = 0
    keep = np.isin(comps, np.where(sizes >= min_px)[0])
    free_c = keep.astype(np.uint8)

    # --- CORE SPLIT: Voronoi graph-based (Sec. III-C) replaces DT+watershed --
    lab = voronoi_segment(free_c, res, a.min_room, a.single_nb)

    # --- rooms table + phantom detection: REUSED from room_segment.py -------
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

    # --- doors: REUSED from room_segment.py --------------------------------
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
        "core": "voronoi-graph (Bormann Sec. III-C)",
        "rooms": rooms,
        "doors": [{kk: vv for kk, vv in d.items() if kk != "px"} for d in doors],
        "topology": [{"edge": d["rooms"], "via": d["type"],
                      "width_m": d["width_m"]} for d in doors],
    }
    # 把 numpy 标量(np.float64/int64, 来自 map 坐标计算)递归转原生, 否则带 --yaml
    # 时 yaml.safe_dump 会报 "cannot represent an object"。(实跑发现的 bug)
    import numpy as _np

    def _to_native(o):
        if isinstance(o, dict):
            return {k: _to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_native(v) for v in o]
        if isinstance(o, _np.generic):
            return o.item()
        return o
    ypath = out + "_topology.yaml"
    ppath = visualize(crop, lab, doors, rooms, out + "_rooms.png")
    atomic_yaml_dump(_to_native(data), ypath)
    print("rooms=%d doors=%d" % (len(rooms), len(doors)))
    print("wrote:", ypath)
    print("wrote:", ppath)


if __name__ == "__main__":
    main()
