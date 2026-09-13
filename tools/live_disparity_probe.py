#!/usr/bin/env python3
"""Print one-frame validity and depth statistics for a ROS stereo disparity."""

from __future__ import print_function

import argparse

import numpy as np
import rospy
from stereo_msgs.msg import DisparityImage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/stereo/disparity")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("live_disparity_probe", anonymous=True,
                    disable_signals=True)
    msg = rospy.wait_for_message(
        args.topic, DisparityImage, timeout=args.timeout)
    row_width = msg.image.step // np.dtype(np.float32).itemsize
    disparity = np.frombuffer(msg.image.data, dtype=np.float32).reshape(
        msg.image.height, row_width)[:, :msg.image.width]
    valid = disparity[
        np.isfinite(disparity) & (disparity > msg.min_disparity)]
    if valid.size == 0:
        raise RuntimeError("disparity frame contains no valid pixels")

    depth = (msg.f * msg.T) / valid
    print(
        "valid=%d total=%d ratio=%.4f disparity_median=%.3f "
        "depth_median_m=%.3f depth_p10_m=%.3f depth_p90_m=%.3f" % (
            valid.size,
            disparity.size,
            valid.size / float(disparity.size),
            np.median(valid),
            np.median(depth),
            np.percentile(depth, 10),
            np.percentile(depth, 90),
        ))


if __name__ == "__main__":
    main()
