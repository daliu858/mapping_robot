# -*- coding: utf-8 -*-
"""floorplan v3: traced-outline floor plan (no Manhattan, no rectification).

Pipeline (geometry 100% from map data):
  1. free-space solid core via local density (kills glass-fan artifacts)
  2. dilate core back into closed free mask -> largest CC -> fill contour
  3. outer contour + light approxPolyDP smoothing (true shape kept)
  4. coarse zones: erosion seeds + nearest-seed Voronoi growth (3-5 zones)
  5. boundary doors -> dashed "unexplored" bubbles outside the outline
  6. POI overlay + scale bar + north arrow + legend + zone names/areas

LLM (me) only assigns zone names (ZONE_NAMES below, with rationale in the
output YAML). No geometry is invented.

Usage: python make_floorplan_v3.py
Outputs: floorplan/out_v3/floorplan_v3.png (+ .svg, zones_v3.yaml)
"""
import json
import math
import os

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAP_YAML = os.path.join(ROOT, "maps", "home_floor_20260830_v2.yaml")
POI_YAML = os.path.join(ROOT, "data", "semantic_objects_final_combined.yaml")
OUT_DIR = os.path.join(HERE, "out_v3")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- tunables ----------------
DENS_BOX = 25          # px, local free-density window
DENS_T = 0.65          # density threshold (fans are scan-line striped -> low)
REGROW = 11            # px, dilate core back toward closed-free mask
CLOSE_FREE = 9         # px, close lacy free mask before regrow
SMOOTH_EPS = 2.5       # px, approxPolyDP epsilon (light smoothing)
NOTCH_MIN_M = 0.45     # convexity-defect depth to count as a waist notch
CHORD_MIN_M = 0.8      # cut-chord length bounds (neck width)
CHORD_MAX_M = 4.6
ZONE_MIN_M2 = 3.0      # a cut must leave both pieces >= this
ZONE_MAX = 5           # stop cutting at this many zones
BUBBLE_R_M = 0.55      # unexplored bubble radius
FIG_DPI = 200

# LLM-assigned zone names (geometry untouched by naming). key = zone id.
ZONE_NAMES = {
    0: ("North Room",
        "NW lobe behind its own waist; holds a cabinet and a boundary door "
        "to unexplored space — a separate side room, not part of the hall."),
    1: ("Living Room / Main Hall",
        "Largest connected open area; contains both sofas, 9 of 10 cabinets "
        "and the insecticide spray — the main living space of the home."),
    2: ("SW Alcove",
        "Small lobe cut off by a narrow waist SW of the hall; no POI inside, "
        "spray POI sits right at its entrance."),
    3: ("Corridor",
        "Narrow SE stub holding the door cluster; every door opens to "
        "unexplored rooms — a hallway, not a living area."),
}

# ---------------- load map ----------------
cfg = yaml.safe_load(open(MAP_YAML))
RES = float(cfg["resolution"])
OX, OY = float(cfg["origin"][0]), float(cfg["origin"][1])
img_path = os.path.join(os.path.dirname(MAP_YAML),
                        os.path.basename(cfg["image"]))
raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
H0, W0 = raw.shape

content = raw != 205
rs, cs = np.nonzero(content)
R0, R1 = max(0, rs.min() - 20), rs.max() + 21
C0, C1 = max(0, cs.min() - 20), cs.max() + 21
crop = raw[R0:R1, C0:C1]
h, w = crop.shape


def w2p(x, y):
    """world m -> crop px (r, c, float)"""
    c = (x - OX) / RES - C0
    r = (H0 - 1 - (y - OY) / RES) - R0
    return r, c


def p2w(r, c):
    """crop px -> world m (x, y)"""
    return OX + (c + C0) * RES, OY + (H0 - 1 - (r + R0)) * RES


free = (crop == 254).astype(np.uint8)
occ = (crop == 0).astype(np.uint8)

# ---------------- 1. main free region ----------------
kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_FREE, CLOSE_FREE))
free_c = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kc)

dens = cv2.boxFilter(free.astype(np.float32), -1, (DENS_BOX, DENS_BOX),
                     normalize=True)
core = (dens > DENS_T).astype(np.uint8)
core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kc)

kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                               (2 * REGROW + 1, 2 * REGROW + 1))
main = (cv2.dilate(core, kr) & free_c).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(main, 8)
big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
main = (lab == big).astype(np.uint8)

# fill: keep only the outer contour (interior holes become part of region)
cnts, _ = cv2.findContours(main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea)
main_fill = np.zeros_like(main)
cv2.drawContours(main_fill, [cnt], -1, 1, thickness=cv2.FILLED)

smooth = cv2.approxPolyDP(cnt, SMOOTH_EPS, True).reshape(-1, 2)  # (x=c, y=r)
outline_w = np.array([p2w(r, c) for c, r in smooth])             # world m

# ---------------- 2. zones: cut at waist notches ----------------
# convexity defects of the smoothed outline = deep notches; a chord between
# two notches across a narrow waist splits off a lobe. Greedy, few cuts.
main_area_m2 = float(main_fill.sum() * RES * RES)

hull_idx = cv2.convexHull(smooth.reshape(-1, 1, 2), returnPoints=False)
defects = cv2.convexityDefects(smooth.reshape(-1, 1, 2), hull_idx)
notches = []
if defects is not None:
    for d in defects[:, 0]:
        dep = d[3] / 256.0 * RES
        if dep >= NOTCH_MIN_M:
            fx, fy = smooth[d[2]]
            notches.append((int(fx), int(fy)))


def chord_ok(m, p, q):
    """draw chord p-q on mask m; return (pieces, sizes) or None."""
    cm = m.copy()
    cv2.line(cm, p, q, 0, 3)
    n, l, s, _ = cv2.connectedComponentsWithStats(cm, 8)
    if n < 3:
        return None
    sizes = sorted((s[i, cv2.CC_STAT_AREA] for i in range(1, n)),
                   reverse=True)
    if sizes[1] * RES * RES < ZONE_MIN_M2:
        return None
    return cm, sizes


cands = []
for i in range(len(notches)):
    for j in range(i + 1, len(notches)):
        p, q = notches[i], notches[j]
        dist = math.hypot(p[0] - q[0], p[1] - q[1]) * RES
        if not (CHORD_MIN_M <= dist <= CHORD_MAX_M):
            continue
        mid = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        if not main_fill[mid[1], mid[0]]:
            continue
        cands.append((dist, p, q))
cands.sort()

cut_mask = main_fill.copy()
chords = []
for dist, p, q in cands:
    n0, _, _, _ = cv2.connectedComponentsWithStats(cut_mask, 8)
    if n0 - 1 >= ZONE_MAX:
        break
    res_cut = chord_ok(cut_mask, p, q)
    if res_cut is None:
        continue
    cut_mask, _sizes = res_cut
    chords.append((p, q))

nz, zone_lab, zstats, _ = cv2.connectedComponentsWithStats(cut_mask, 8)
zones = {}
for i in range(1, nz):
    zones[i - 1] = (zone_lab == i)
# reassign cut-line pixels (and any gaps) to nearest zone -> full tiling
zone_dists = np.stack([cv2.distanceTransform((~zm).astype(np.uint8),
                                             cv2.DIST_L2, 3)
                       for zm in zones.values()], 0)
assign = np.argmin(zone_dists, 0)
zkeys = list(zones.keys())
zones = {}
for zi, k in enumerate(zkeys):
    zones[k] = (assign == zi) & main_fill.astype(bool)
zone_lab = np.zeros_like(main_fill)
for k, zm in zones.items():
    zone_lab[zm] = k + 1

zone_info = {}
for zi, zm in zones.items():
    area = float(zm.sum() * RES * RES)
    # label anchor: deepest interior point (pole of inaccessibility)
    dt = cv2.distanceTransform(zm.astype(np.uint8), cv2.DIST_L2, 5)
    ar, ac = np.unravel_index(np.argmax(dt), dt.shape)
    cx, cy = p2w(ar, ac)
    zone_info[zi] = {"area_m2": area, "label_xy": (cx, cy), "mask": zm}

# ---------------- 3. POIs + boundary doors ----------------
poi_doc = yaml.safe_load(open(POI_YAML))
pois = poi_doc["objects"]

# doors: scan 24 directions around each door; "behind" = the direction whose
# probe points are most often unknown (205) map cells. >=2/3 unknown -> the
# door exits to unexplored space and gets a bubble; an interior door between
# explored areas has no such direction and gets none.
doors = []
ANGS = np.linspace(0, 2 * math.pi, 24, endpoint=False)
for o in pois:
    r, c = w2p(o["x"], o["y"])
    ri, ci = int(round(r)), int(round(c))
    o["_px"] = (r, c)
    if o["label"] != "door":
        continue
    best = None  # (n_unk, n_free, vx, vy)
    for a in ANGS:
        vx, vy = math.cos(a), math.sin(a)   # image coords (dc, dr)
        n_unk = n_free = 0
        for t in (0.4, 0.65, 0.9):
            pr = int(round(ri + vy * t / RES))
            pc = int(round(ci + vx * t / RES))
            if 0 <= pr < h and 0 <= pc < w:
                v = crop[pr, pc]
                if v == 205:
                    n_unk += 1
                elif v == 254:
                    n_free += 1
            else:
                n_unk += 1
        if best is None or (n_unk, -n_free) > (best[0], -best[1]):
            best = (n_unk, n_free, vx, vy)
    n_unk, n_free, vx, vy = best
    exit_door = n_unk >= 2 and n_unk > n_free
    doors.append({"poi": o, "unknown_behind": n_unk / 3.0,
                  "boundary": bool(exit_door), "out": (vx, vy)})

# assign POIs to zones (nearest zone pixel within 0.3 m: POIs sit on
# furniture, not necessarily on free floor)
zone_ids = list(zones.keys())
zone_dt = {k: cv2.distanceTransform((~zones[k]).astype(np.uint8),
                                    cv2.DIST_L2, 3) for k in zone_ids}
for o in pois:
    ri, ci = int(round(o["_px"][0])), int(round(o["_px"][1]))
    best, best_d = None, 1e9
    for k in zone_ids:
        dk = zone_dt[k][ri, ci]
        if dk < best_d:
            best, best_d = k, dk
    o["_zone"] = best if best_d <= 6 else None
for zi in zone_ids:
    zone_info[zi]["pois"] = sorted(o["label"] for o in pois
                                   if o.get("_zone") == zi)

# ---------------- report (for LLM naming pass) ----------------
report = {"main_area_m2": main_area_m2,
          "notches_px": notches,
          "chords_px": [[list(p), list(q)] for p, q in chords],
          "zones": {zi: {"area_m2": round(v["area_m2"], 1),
                         "label_xy": [round(v["label_xy"][0], 2),
                                      round(v["label_xy"][1], 2)],
                         "pois": v["pois"]}
                    for zi, v in zone_info.items()},
          "doors": [{"xy": [d["poi"]["x"], d["poi"]["y"]],
                     "unknown_behind": round(d["unknown_behind"], 2),
                     "exit_to_unexplored": d["boundary"]} for d in doors]}
print(json.dumps(report, indent=1))

# ---------------- 4. render ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

x0, x1 = OX + C0 * RES, OX + (C1 - 1) * RES
y0 = OY + (H0 - 1 - (R1 - 1)) * RES
y1 = OY + (H0 - 1 - R0) * RES

fig, ax = plt.subplots(figsize=(11, 10))
ax.set_aspect("equal")
ax.set_facecolor("#f7f5f0")

# unknown background inside bbox
ax.imshow(np.full((h, w, 3), 0.94), extent=[x0, x1, y0, y1],
          origin="upper", zorder=0, interpolation="nearest")

ZONE_COLORS = ["#cfe0f1", "#f6e2c5", "#d9ecd3", "#f3d5dc", "#e4ddf2"]
for zi, v in sorted(zone_info.items()):
    rgba = np.zeros((h, w, 4))
    rgb = matplotlib.colors.to_rgb(ZONE_COLORS[zi % len(ZONE_COLORS)])
    rgba[v["mask"]] = (*rgb, 1.0)
    ax.imshow(rgba, extent=[x0, x1, y0, y1], origin="upper", zorder=1,
              interpolation="nearest")

# occupied pixels near the explored region (real measurements)
occ_near = occ.astype(bool) & (cv2.dilate(main_fill, np.ones((9, 9),
                                          np.uint8)).astype(bool))
occ_img = np.zeros((h, w, 4))
occ_img[occ_near] = (0.15, 0.15, 0.18, 1.0)
ax.imshow(occ_img, extent=[x0, x1, y0, y1], origin="upper", zorder=2,
          interpolation="nearest")

# traced outline
poly = np.vstack([outline_w, outline_w[0]])
ax.plot(poly[:, 0], poly[:, 1], "-", color="#22242a", lw=2.6, zorder=3,
        solid_joinstyle="round")

# zone boundaries (waist chords)
for p, q in chords:
    wp = p2w(p[1], p[0])
    wq = p2w(q[1], q[0])
    ax.plot([wp[0], wq[0]], [wp[1], wq[1]], "-", color="#8a8f98", lw=1.1,
            alpha=0.9, zorder=3)

# unexplored bubbles behind boundary doors
labeled = []
for d in doors:
    if not d["boundary"]:
        continue
    x, y = d["poi"]["x"], d["poi"]["y"]
    ux, uy = d["out"][0], -d["out"][1]  # image (dc,dr) -> world (dx,dy)
    px_, py_ = -uy, ux                  # perpendicular
    ts = np.linspace(-1, 1, 25)
    arc = np.array([(x + px_ * BUBBLE_R_M * t +
                     ux * BUBBLE_R_M * math.sqrt(max(0.0, 1 - t * t)),
                     y + py_ * BUBBLE_R_M * t +
                     uy * BUBBLE_R_M * math.sqrt(max(0.0, 1 - t * t)))
                    for t in ts])
    bub = MplPolygon(arc, closed=True, facecolor="#f2f2f2",
                     edgecolor="#7a7a7a", lw=1.4, ls=(0, (4, 3)),
                     hatch="////", zorder=2)
    ax.add_patch(bub)
    lx = x + ux * (BUBBLE_R_M + 0.45)
    ly = y + uy * (BUBBLE_R_M + 0.45)
    if all(math.hypot(lx - a, ly - b) > 1.6 for a, b in labeled):
        ax.text(lx, ly, "unexplored", fontsize=7.5, style="italic",
                color="#666666", ha="center", va="center", zorder=6)
        labeled.append((lx, ly))

# POIs
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=3, foreground="white")]
POI_STYLE = {"door": ("o", "#2ca02c", 60),
             "sofa": ("s", "#ff8c1a", 90),
             "cabinet": ("^", "#1f77b4", 70),
             "spray": ("*", "#e00000", 420)}
for o in pois:
    mk, col, sz = POI_STYLE[o["label"]]
    ax.scatter([o["x"]], [o["y"]], marker=mk, s=sz, c=col,
               edgecolors="white", linewidths=1.2, zorder=5)
    if o["label"] == "spray":
        ax.annotate("insecticide spray", (o["x"], o["y"]),
                    xytext=(o["x"] - 2.15, o["y"] + 0.05),
                    fontsize=10, fontweight="bold", color="#b00000",
                    ha="center", zorder=6, path_effects=HALO,
                    arrowprops=dict(arrowstyle="->", color="#b00000",
                                    lw=1.2))

# zone names + areas
for zi, v in sorted(zone_info.items()):
    name, _why = ZONE_NAMES.get(zi, ("Zone %d" % (zi + 1), ""))
    cx, cy = v["label_xy"]
    ax.text(cx, cy, "%s\n%.1f m²" % (name, v["area_m2"]), fontsize=11,
            fontweight="bold", color="#333333", ha="center", va="center",
            zorder=4, path_effects=HALO)

# scale bar (2 m), bottom-right clear of the legend
bx, by = x1 - 3.1, y0 + 0.55
ax.plot([bx, bx + 2], [by, by], color="#222", lw=4, zorder=7,
        solid_capstyle="butt")
for dx in (0, 1, 2):
    ax.plot([bx + dx, bx + dx], [by, by + 0.09], color="#222", lw=1.6,
            zorder=7)
    ax.text(bx + dx, by + 0.18, "%d" % dx, fontsize=8, ha="center",
            color="#222", zorder=7)
ax.text(bx + 2.25, by + 0.18, "m", fontsize=8, ha="left", color="#222",
        zorder=7)

# north arrow (map +Y)
nxp, nyp = x1 - 0.7, y1 - 1.15
ax.annotate("", xy=(nxp, nyp + 0.7), xytext=(nxp, nyp),
            arrowprops=dict(arrowstyle="-|>", color="#222", lw=2.2),
            zorder=7)
ax.text(nxp, nyp + 0.82, "N", fontsize=12, fontweight="bold", ha="center",
        color="#222", zorder=7)

# legend
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c",
           markersize=8, label="door"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff8c1a",
           markersize=8, label="sofa"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#1f77b4",
           markersize=8, label="cabinet"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="#e00000",
           markersize=13, label="insecticide spray"),
    Line2D([0], [0], color="#22242a", lw=2.6, label="measured outline"),
    Line2D([0], [0], color="#7a7a7a", lw=1.4, ls="--",
           label="unexplored (behind door)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95,
          borderpad=0.8, labelspacing=0.5)

ax.set_xlim(x0 - 0.4, x1 + 0.4)
ax.set_ylim(y0 - 0.4, y1 + 0.4)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("Home floor plan (traced outline, SLAM survey 2026-08-30)",
             fontsize=13, pad=10)

png = os.path.join(OUT_DIR, "floorplan_v3.png")
fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(png.replace(".png", ".svg"), bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

# ---------------- 5. zones YAML (with naming provenance) ----------------
out = {
    "schema": "slam-car-floorplan-v3-zones",
    "map_image": os.path.basename(img_path),
    "resolution_m": RES,
    "method": {
        "outline": "free-density core (box%d>%.2f) -> regrow %dpx in "
                   "closed-free -> largest CC -> fill -> approxPolyDP %.1fpx"
                   % (DENS_BOX, DENS_T, REGROW, SMOOTH_EPS),
        "zones": "convexity-defect notches (depth>=%.2f m) + greedy waist "
                 "chords (%.1f-%.1f m, both pieces >= %.1f m2, <= %d zones); "
                 "LLM assigns names only, no geometry"
                 % (NOTCH_MIN_M, CHORD_MIN_M, CHORD_MAX_M, ZONE_MIN_M2,
                    ZONE_MAX),
        "unexplored": "doors whose probe points behind the door land in "
                      "unknown map cells get a dashed hatch bubble "
                      "r=%.2f m along the away-from-free-space normal"
                      % BUBBLE_R_M,
    },
    "main_area_m2": round(main_area_m2, 1),
    "zones": [],
}
for zi, v in sorted(zone_info.items()):
    name, why = ZONE_NAMES.get(zi, ("Zone %d" % (zi + 1),
                                    "unnamed (run naming pass)"))
    out["zones"].append({
        "id": int(zi), "name": name,
        "area_m2": round(float(v["area_m2"]), 1),
        "label_xy_m": [round(float(v["label_xy"][0]), 2),
                       round(float(v["label_xy"][1]), 2)],
        "poi_labels": v["pois"], "naming_rationale": why,
        "named_by": "LLM (name only; geometry from map data)",
    })
with open(os.path.join(OUT_DIR, "zones_v3.yaml"), "w",
          encoding="utf-8") as f:
    yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

print("saved", png)
