# qcar2_autonomy/perception/depth.py

import numpy as np


def depth_plane(depth):
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[:, :, 0]
    return depth


def median_depth_from_bbox(depth, bbox_xyxy, inset=0.2):
    d = depth_plane(depth)
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    xx1 = max(0, int(x1 + w * inset))
    yy1 = max(0, int(y1 + h * inset))
    xx2 = min(d.shape[1], int(x2 - w * inset))
    yy2 = min(d.shape[0], int(y2 - h * inset))
    if xx2 <= xx1 or yy2 <= yy1:
        return float("inf")
    roi = d[yy1:yy2, xx1:xx2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid > 0) & (valid < 100)]
    if valid.size == 0:
        return float("inf")
    return float(np.median(valid))


def forward_obstacle_distance(depth):
    d = depth_plane(depth)
    h, w = d.shape[:2]
    y1 = int(0.35 * h)
    y2 = int(0.72 * h)
    x1 = int(0.42 * w)
    x2 = int(0.58 * w)
    roi = d[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid > 0) & (valid < 100)]
    if valid.size == 0:
        return float("inf")
    return float(np.percentile(valid, 25))