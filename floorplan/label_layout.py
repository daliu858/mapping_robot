"""仅调整标注位置；不修改地图、区域边界或 POI 坐标。"""
import math

import numpy as np
from matplotlib.transforms import Bbox


def place_labels(fig, ax, labels, pois, world_to_pixel):
    """用实际文字边框避开 POI；狭窄区域放不下时在旁边加引线。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    point_boxes = []
    for poi in pois:
        x, y = ax.transData.transform((poi["x"], poi["y"]))
        radius = (17 if poi["label"] == "spray" else 8) * fig.dpi / 72
        point_boxes.append(Bbox.from_extents(x - radius, y - radius,
                                            x + radius, y + radius))
    occupied = list(point_boxes)
    bounds = ax.get_window_extent(renderer).padded(-8)
    offsets = [(0, 0)]
    for radius in (30, 60, 100, 150, 220):
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))

    for label in labels:
        anchor = label["xy"]
        origin = ax.transData.transform(anchor)
        style = dict(label.get("style", {}))
        fontsize = style.pop("fontsize", 10)
        artist = ax.text(*anchor, label["text"], ha="center", va="center",
                         wrap=False, clip_on=False, zorder=8, **style)
        best = None
        for size in (fontsize, max(7, fontsize - 1), max(7, fontsize - 2)):
            artist.set_fontsize(size)
            artist.set_position(anchor)
            measured = artist.get_window_extent(renderer).padded(5)
            half_w, half_h = measured.width / 2, measured.height / 2
            for dx, dy in offsets:
                x, y = origin + np.array([dx, dy]) * fig.dpi / 100
                box = Bbox.from_extents(x - half_w, y - half_h,
                                        x + half_w, y + half_h)
                if not (bounds.contains(box.x0, box.y0)
                        and bounds.contains(box.x1, box.y1)):
                    continue
                if any(box.overlaps(other) for other in occupied):
                    continue
                inside = True
                mask = label.get("mask")
                if mask is not None:
                    corners = [(box.x0, box.y0), (box.x0, box.y1),
                               (box.x1, box.y0), (box.x1, box.y1), (x, y)]
                    for wx, wy in ax.transData.inverted().transform(corners):
                        row, col = world_to_pixel(wx, wy)
                        row, col = int(round(row)), int(round(col))
                        if not (0 <= row < mask.shape[0] and
                                0 <= col < mask.shape[1] and mask[row, col]):
                            inside = False
                            break
                score = (not inside, fontsize - size, math.hypot(dx, dy))
                if best is None or score < best[0]:
                    best = (score, size, (x, y), box)
        if best is None:
            raise ValueError("No clear position for floor-plan label: " + label["text"])
        _, size, position, box = best
        target = ax.transData.inverted().transform(position)
        artist.set_fontsize(size)
        artist.set_position(target)
        occupied.append(box)
        if np.linalg.norm(position - origin) > 5:
            ax.annotate("", xy=anchor, xytext=target, annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "lw": 0.8,
                                    "color": style.get("color", "#666666")},
                        zorder=6)
