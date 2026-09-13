# -*- coding: utf-8 -*-
"""Post-fusion readability pass: merge same-class POIs closer than a
per-class minimum spacing. Merge = hits-weighted centroid, hits summed.
Never invents geometry; only combines existing confirmed tracks."""
import argparse
import math

import yaml

DEFAULT_MIN_DIST = {"door": 1.0, "sofa": 1.0, "cabinet": 0.8, "spray": 0.6}


def dedupe(objects, min_dist):
    objs = [dict(o) for o in objects]
    for o in objs:
        o.setdefault("merged_from", 1)
    changed = True
    while changed:
        changed = False
        best = None
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                if a["label"] != b["label"]:
                    continue
                d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                thr = min_dist.get(a["label"], 0.8)
                if d < thr and (best is None or d < best[0]):
                    best = (d, i, j)
        if best:
            d, i, j = best
            a, b = objs[i], objs[j]
            w = float(a["hits"] + b["hits"])
            merged = {
                "label": a["label"],
                "class_id": a["class_id"],
                "x": round((a["x"] * a["hits"] + b["x"] * b["hits"]) / w, 3),
                "y": round((a["y"] * a["hits"] + b["y"] * b["hits"]) / w, 3),
                "hits": a["hits"] + b["hits"],
                "spread_m": round(max(a["spread_m"], b["spread_m"],
                                      d / 2.0), 3),
                "merged_from": a["merged_from"] + b["merged_from"],
            }
            objs = [o for k, o in enumerate(objs) if k not in (i, j)]
            objs.append(merged)
            changed = True
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--min-dist", action="append", default=[],
                    help="label=meters, e.g. sofa=1.2 (repeatable)")
    args = ap.parse_args()

    min_dist = dict(DEFAULT_MIN_DIST)
    for spec in args.min_dist:
        k, v = spec.split("=")
        min_dist[k] = float(v)

    doc = yaml.safe_load(open(args.input))
    before = doc["objects"]
    after = dedupe(before, min_dist)
    doc["objects"] = sorted(after, key=lambda o: (o["label"], -o["hits"]))
    doc["dedupe_min_dist_m"] = min_dist
    doc["dedupe_source"] = args.input.replace("\\", "/").rsplit("/", 1)[-1]

    def stats(objs):
        s = {}
        for o in objs:
            s[o["label"]] = s.get(o["label"], 0) + 1
        return s

    print("before:", stats(before), "-> after:", stats(after))
    for o in doc["objects"]:
        tag = " (merged x%d)" % o["merged_from"] if o["merged_from"] > 1 else ""
        print("  %-8s h=%-3d (%.2f, %.2f)%s" % (
            o["label"], o["hits"], o["x"], o["y"], tag))
    with open(args.output, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    print("saved", args.output)


if __name__ == "__main__":
    main()
