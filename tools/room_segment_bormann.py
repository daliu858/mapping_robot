# -*- coding: utf-8 -*-
"""
room_segment_bormann.py — Faithful reimplementation of Bormann et al. (ICRA 2016)
Section III-B "Distance Transform-based Segmentation", and NOTHING else.

This file deliberately implements ONLY what Sec. III-B describes:
  1. distance transform of the free space
  2. threshold the DT, looping thresholds in DESCENDING order, to obtain
     room centers C; choose t* so that the number of room centers |C| is MAXIMAL
  3. label the room centers uniquely and extend them into the remaining
     unlabeled free space via WAVEFRONT PROPAGATION (NOT watershed)

No merging step is added: Sec. III-B contains NO merging subsection (the 5
merge heuristics in the paper belong to Sec. III-C Voronoi segmentation, and
Bormann explicitly presents them only there). No phantom rejection, no doorway
detection — none of that appears in III-B.

Every key step is annotated with the corresponding paper sentence.

Usage:
  python room_segment_bormann.py "d:/slam car/room1.pgm" --out "d:/slam car/cmp_B"
"""
import argparse, os, sys
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage


# ---------------------------------------------------------------------------
# Map loading. Same ternarization convention as the gmapping PGM:
# 254=free, 0=occupied, 205=unknown. III-B operates on the free space only.
# ---------------------------------------------------------------------------
def load_free(pgm_path, res=0.05):
    raw = np.array(Image.open(pgm_path))
    # "A distance transform represents the distance of each accessible (white)
    #  pixel to the closest border pixel (black)."  -> accessible = free space.
    free = (raw >= 250).astype(np.uint8)
    return raw, free, res


def wavefront_propagation(seeds, free):
    """III-B, final sentence:
       "Eventually, the found room centers are labeled uniquely and extended
        into remaining unlabeled space via wavefront propagation."

    Wavefront / grassfire propagation = simultaneous, equal-speed expansion of
    every labeled seed front through the free space. Each unlabeled free pixel
    is claimed by whichever wavefront reaches it FIRST, i.e. every free pixel is
    assigned the label of its GEODESICALLY NEAREST room center, with the geodesic
    confined to the free mask. This is implemented exactly as a multi-source
    breadth-first flood (4-connected fronts grown one ring per BFS layer), which
    is the literal meaning of an iso-distance advancing wavefront.

    NOTE: this is intentionally NOT skimage.watershed(-dist). Watershed floods
    the inverted-distance basins along its ridge lines; wavefront propagation
    floods by geodesic arrival time from the seeds. They are different operators
    and can place inter-room boundaries differently.
    """
    if seeds is None:
        raise ValueError("room-center seeds are missing")
    if seeds.shape != free.shape:
        raise ValueError("room-center seeds and free-space mask must have the same shape")
    if np.any((seeds > 0) & ~free.astype(bool)):
        raise ValueError("room-center seeds must lie inside free space")
    if free.any() and not np.any(seeds > 0):
        raise ValueError("no room-center seeds were found in non-empty free space")

    H, W = free.shape
    lab = seeds.astype(np.int32).copy()
    freeb = free.astype(bool)
    q = deque()
    ys, xs = np.where(seeds > 0)
    for y, x in zip(ys, xs):
        q.append((y, x))
    # 4-connected wavefront (matches a Manhattan-isotropic advancing front).
    while q:
        y, x = q.popleft()
        l = lab[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and freeb[ny, nx] and lab[ny, nx] == 0:
                lab[ny, nx] = l
                q.append((ny, nx))
    return lab


def choose_room_center_seeds(dist):
    """Choose the highest threshold attaining maximal |C|, with no empty result."""
    if dist.ndim != 2:
        raise ValueError("distance transform must be a 2-D array")
    dmax = int(np.floor(dist.max())) if dist.size else 0
    if dmax < 1:
        raise ValueError(
            "distance transform contains no threshold at or above one pixel")

    best_t, best_n, best_seeds = None, -1, None
    trace = []
    for t in range(dmax, 0, -1):
        seeds_t, n = ndimage.label(dist >= t)
        trace.append((t, n))
        # Thresholds are visited high-to-low, so strict improvement makes the
        # tie break explicit: the highest threshold wins equal component counts.
        if n > best_n:
            best_n, best_t, best_seeds = n, t, seeds_t
    if best_seeds is None or best_n <= 0 or not np.any(best_seeds > 0):
        raise ValueError("distance-transform scan produced no room-center seeds")
    return best_t, best_n, best_seeds, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    # CONTROLLED-EXPERIMENT KNOB (not part of III-B): force a specific DT
    # threshold in pixels, to compare wavefront-vs-watershed under IDENTICAL
    # seeds instead of letting the |C|-maximal rule pick a noise-dominated t*.
    ap.add_argument("--force-t", type=int, default=None)
    a = ap.parse_args()

    raw, free, res = load_free(a.pgm, a.res)
    Hf, Wf = raw.shape
    if not free.any():
        sys.exit("no free space in map")

    # Crop to the mapped extent purely for visualization/runtime (no algorithmic
    # effect — DT and propagation are translation-invariant). Keep free+occ bbox.
    occ = raw <= 50
    ys, xs = np.where(free | occ)
    m = 12
    y0, y1 = max(0, ys.min() - m), min(Hf, ys.max() + m)
    x0, x1 = max(0, xs.min() - m), min(Wf, xs.max() + m)
    crop = raw[y0:y1, x0:x1]
    free_c = (crop >= 250).astype(np.uint8)

    # -----------------------------------------------------------------------
    # STEP 1 — Distance transform.
    # III-B: "The distance transform-based segmentation initially requires a
    #         distance transform of the map. A distance transform represents the
    #         distance of each accessible (white) pixel to the closest border
    #         pixel (black)."
    # -----------------------------------------------------------------------
    dist = cv2.distanceTransform(free_c, cv2.DIST_L2, 5)  # pixels

    # -----------------------------------------------------------------------
    # STEP 2 — Threshold the DT to obtain room centers C, choosing t* so that
    #          the number of room centers |C| is maximal.
    # III-B: "Looping through all such possible thresholds t in descending order
    #         delivers a set of room centers C, which first increases in size
    #         until the threshold equals the local maxima at doors. From that
    #         point the number of room centers decreases again ... The
    #         segmentation algorithm chooses threshold t* in such a way that the
    #         number of retrieved room centers |C| is maximal."
    #
    # We scan ALL integer-pixel thresholds from high (dist.max) down to 1, label
    # the connected components of (dist >= t) as the candidate room centers, and
    # keep the t* that yields the maximal component count. This is the literal
    # "|C| maximal" criterion — no resolution-scaled magic threshold.
    # -----------------------------------------------------------------------
    try:
        best_t, best_n, best_seeds, trace = choose_room_center_seeds(dist)
    except ValueError as error:
        sys.exit(str(error))
    print("DT-threshold scan (t_px -> |C|):")
    for t, n in trace:
        mark = "  <- t* (|C| max)" if t == best_t else ""
        print("  t=%2d px (%.2f m)  |C|=%d%s" % (t, t * res, n, mark))
    print("chosen t* = %d px = %.2f m  ->  |C| = %d room centers"
          % (best_t, best_t * res, best_n))

    if a.force_t is not None:
        if a.force_t < 1:
            sys.exit("--force-t must be at least one pixel")
        seeds, n = ndimage.label(dist >= a.force_t)
        print("[FORCED] t = %d px = %.2f m  ->  |C| = %d room centers"
              % (a.force_t, a.force_t * res, n))
    else:
        seeds = best_seeds

    # -----------------------------------------------------------------------
    # STEP 3 — Label room centers uniquely and extend via WAVEFRONT PROPAGATION.
    # III-B: "Eventually, the found room centers are labeled uniquely and
    #         extended into remaining unlabeled space via wavefront propagation."
    # -----------------------------------------------------------------------
    try:
        lab = wavefront_propagation(seeds, free_c)
    except ValueError as error:
        sys.exit(str(error))

    # -----------------------------------------------------------------------
    # NO merging. III-B has no merging subsection. NO phantom rejection. NO door
    # detection. Section III-B ends at wavefront propagation.
    # -----------------------------------------------------------------------

    # rooms table (faithful: every wavefront region is a room, no filtering)
    px_area = res * res
    rooms = []
    for i in [int(v) for v in np.unique(lab[lab > 0])]:
        mask = (lab == i)
        area = float(mask.sum()) * px_area
        ys_, xs_ = np.where(mask)
        cy, cx = float(ys_.mean()), float(xs_.mean())
        rooms.append({"id": i, "area_m2": round(area, 2),
                      "centroid_px": [int(cx), int(cy)]})
    rooms.sort(key=lambda r: -r["area_m2"])
    print("\nBORMANN III-B result: %d rooms" % len(rooms))
    for r in rooms:
        print("  R%-3d area=%6.2f m^2  centroid_px=%s"
              % (r["id"], r["area_m2"], r["centroid_px"]))

    # visualization (same style as algorithm A for fair eyeball comparison)
    PALETTE = [(231, 76, 60), (52, 152, 219), (46, 204, 113), (155, 89, 182),
               (241, 196, 15), (230, 126, 34), (26, 188, 156), (149, 165, 166),
               (192, 57, 43), (41, 128, 185), (39, 174, 96), (142, 68, 173)]
    H, W = lab.shape
    vis = np.stack([crop] * 3, axis=-1).astype(np.uint8)
    overlay = vis.copy()
    for i, r in enumerate(rooms):
        overlay[lab == r["id"]] = PALETTE[i % len(PALETTE)]
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
    for r in rooms:
        cx, cy = r["centroid_px"]
        dr.text((cx * s, cy * s), "R%d %.1fm2" % (r["id"], r["area_m2"]),
                fill=(0, 0, 0), font=font, anchor="mm")
    out = a.out or os.path.splitext(a.pgm)[0] + "_bormann"
    ppath = out + "_rooms.png"
    img.save(ppath)
    print("wrote:", ppath)


if __name__ == "__main__":
    main()
