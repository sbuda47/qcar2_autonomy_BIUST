# qcar2_autonomy/perception/lane.py

import cv2
import numpy as np
from hal.utilities.image_processing import ImageProcessing


def compute_lane_steering_example(
    front_bgr,
    dt,
    steering_filter,
    lane_steer_limit,
    left_hand_traffic=False,
    yellow_line_target_ratio=-0.72,
    yellow_line_distance_gain=0.65,
    yellow_line_bottom_roi=0.65,
):
    h, w = front_bgr.shape[:2]
    row_start = int(np.clip(round((524.0 / 820.0) * h), 0, h - 1))
    row_end = int(np.clip(round((674.0 / 820.0) * h), row_start + 1, h))
    col_span = int(np.clip(round((820.0 / 1640.0) * w), 1, w))

    if left_hand_traffic:
        col_start = 0
        col_end = col_span
    else:
        col_start = max(0, w - col_span)
        col_end = w

    cropped = front_bgr[row_start:row_end, col_start:col_end]
    if cropped.size == 0:
        return None

    hsv_buf = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    binary = ImageProcessing.binary_thresholding(
        frame=hsv_buf,
        lowerBounds=np.array([10, 50, 100], dtype=np.uint8),
        upperBounds=np.array([45, 255, 255], dtype=np.uint8),
    )

    slope, intercept = ImageProcessing.find_slope_intercept_from_binary(binary=binary)
    raw_steering = 1.5 * (slope - 0.3419) + (1.0 / 150.0) * (intercept + 5.0)

    target_ratio = float(yellow_line_target_ratio)
    if target_ratio < 0.0:
        target_ratio = 0.70 if left_hand_traffic else 0.30

    rows, cols = np.where(binary > 0)
    if rows.size > 20:
        cutoff_row = int(np.clip(round(yellow_line_bottom_roi * binary.shape[0]), 0, binary.shape[0] - 1))
        near_mask = rows >= cutoff_row
        line_x = float(np.median(cols[near_mask])) if np.count_nonzero(near_mask) >= 10 else float(np.median(cols))
        target_x = target_ratio * float(max(binary.shape[1] - 1, 1))
        x_error_norm = (line_x - target_x) / float(max(binary.shape[1], 1))
        raw_steering += -float(yellow_line_distance_gain) * x_error_norm

    clipped = float(np.clip(raw_steering, -lane_steer_limit, lane_steer_limit))
    try:
        steering = float(steering_filter.send((clipped, dt)))
    except Exception:
        steering = clipped

    if np.isnan(steering):
        return None
    return steering