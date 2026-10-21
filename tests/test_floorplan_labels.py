"""验证拥挤房间名、面积及门外标注不会遮挡 POI 或被裁切。"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox


path = Path(__file__).resolve().parents[1] / "floorplan" / "label_layout.py"
spec = importlib.util.spec_from_file_location("floorplan_label_layout", path)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


def test_crowded_room_names_areas_and_unexplored_remain_readable():
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    try:
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 7)
        corridor = np.zeros((80, 120), dtype=bool)
        corridor[10:70, 90:100] = True
        pois = [{"label": "spray", "x": 4, "y": 4},
                {"label": "door", "x": 8.5, "y": 2}]
        labels = [
            {"xy": (4, 4), "text": "insecticide spray"},
            {"xy": (4, 4), "text": "Main Hall\n17.7 m²"},
            {"xy": (8.5, 3), "text": "Corridor\n1.5 m²", "mask": corridor},
            {"xy": (8.5, 2), "text": "unexplored"},
        ]
        layout.place_labels(fig, ax, labels, pois,
                            lambda x, y: ((y + 1) * 10, (x + 1) * 10))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [artist for artist in ax.texts if artist.get_text()]
        assert [artist.get_text() for artist in texts] == [item["text"] for item in labels]
        boxes = [artist.get_window_extent(renderer) for artist in texts]
        bounds = ax.get_window_extent(renderer)
        for index, (artist, box) in enumerate(zip(texts, boxes)):
            assert not artist.get_wrap()
            assert not artist.get_clip_on()
            assert bounds.contains(box.x0, box.y0)
            assert bounds.contains(box.x1, box.y1)
            assert not any(box.overlaps(other) for other in boxes[index + 1:])
            for poi in pois:
                x, y = ax.transData.transform((poi["x"], poi["y"]))
                radius = (17 if poi["label"] == "spray" else 8) * fig.dpi / 72
                point = Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
                assert not box.overlaps(point)
    finally:
        plt.close(fig)
