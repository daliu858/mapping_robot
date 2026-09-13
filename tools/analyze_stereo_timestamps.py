# -*- coding: utf-8 -*-
"""Measure one-to-one left/right header-stamp offsets in a ROS1 stereo bag."""
from __future__ import print_function

import argparse
from pathlib import Path
import sys

import numpy as np


def _configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_stdio()


def stamp_to_ns(stamp):
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    return int(sec) * 1000000000 + int(nsec)


def monotonic_gate_offsets_ms(left_ns, right_ns, gate_ms):
    """Return one-to-one offsets using a monotonic hard-window matcher."""
    left = np.asarray(sorted(left_ns), dtype=np.int64)
    right = np.asarray(sorted(right_ns), dtype=np.int64)
    if left.size == 0 or right.size == 0:
        return np.asarray([], dtype=np.float64)
    gate_ns = int(float(gate_ms) * 1e6)
    offsets = []
    i = j = 0
    while i < left.size and j < right.size:
        delta = int(right[j] - left[i])
        if delta < -gate_ns:
            j += 1
        elif delta > gate_ns:
            i += 1
        else:
            offsets.append(abs(delta) / 1e6)
            i += 1
            j += 1
    return np.asarray(offsets, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--left-topic",
                    default="/stereo/left/image_raw/compressed")
    ap.add_argument("--right-topic",
                    default="/stereo/right/image_raw/compressed")
    ap.add_argument("--gate-ms", type=float, default=20.0,
                    help="计算该时间门限下可接受的左帧比例")
    args = ap.parse_args()

    try:
        from rosbags_index_compat import patch_rosbags_index_headers
        patch_rosbags_index_headers()
        from rosbags.highlevel import AnyReader
    except Exception as ex:
        sys.exit("[依赖缺失] pip install rosbags: %r" % ex)

    stamps = {args.left_topic: [], args.right_topic: []}
    with AnyReader([Path(args.bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in stamps]
        present = {c.topic for c in conns}
        missing = sorted(set(stamps) - present)
        if missing:
            sys.exit("bag 缺少话题: %s" % ", ".join(missing))
        for conn, _bag_time, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            stamps[conn.topic].append(stamp_to_ns(msg.header.stamp))

    left_count = len(stamps[args.left_topic])
    right_count = len(stamps[args.right_topic])
    if left_count == 0 or right_count == 0:
        sys.exit("bag 中左右图像消息为空: left=%d right=%d" %
                 (left_count, right_count))
    offsets = monotonic_gate_offsets_ms(
        stamps[args.left_topic], stamps[args.right_topic], args.gate_ms)
    accepted = 100.0 * float(offsets.size) / float(left_count)
    print("left=%d right=%d" % (len(stamps[args.left_topic]),
                                len(stamps[args.right_topic])))
    if offsets.size:
        print("one-to-one accepted |dt|: median=%.2f ms, p95=%.2f ms, "
              "max=%.2f ms" %
              (np.percentile(offsets, 50), np.percentile(offsets, 95),
               offsets.max()))
    print("one-to-one gate %.2f ms: %d pairs, %.1f%% of left frames" %
          (args.gate_ms, offsets.size, accepted))


if __name__ == "__main__":
    main()
