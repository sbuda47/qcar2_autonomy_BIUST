# qcar2_autonomy/perception/yolo_rules.py

import numpy as np
from .depth import median_depth_from_bbox, forward_obstacle_distance


class RuleState:
    def __init__(self):
        self.stop_state = "armed"
        self.stop_until = 0.0
        self.person_blocking = False


def classify_name(name):
    n = str(name).lower().replace("_", " ").strip()
    if ("stop" in n and "sign" in n) or n == "stop":
        return "stop"
    if "yield" in n:
        return "yield"
    if "traffic" in n and "light" in n:
        return "traffic_red"
    if "traffic" in n and "red" in n:
        return "traffic_red"
    if "red light" in n:
        return "traffic_red"
    if "person" in n or "pedestrian" in n:
        return "person"
    if "car" in n or "vehicle" in n:
        return "car"
    return None


def class_name_from_index(names, idx):
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
        return str(names[idx])
    return str(idx)


def detect_distances(model, rgb, depth, args):
    distances = {
        "stop": float("inf"),
        "person": float("inf"),
        "car": float("inf"),
        "obstacle": float("inf"),
        "yield": float("inf"),
        "traffic_red": float("inf"),
    }

    results = model.predict(
        source=rgb,
        conf=args.confidence,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
        device=args.device or None,
        half=False,
    )
    result = results[0]
    names = result.names

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.int32)
        cls_idx = result.boxes.cls.cpu().numpy().astype(np.int32)
        img_h = max(1, int(rgb.shape[0]))
        img_w = max(1, int(rgb.shape[1]))

        for i in range(len(xyxy)):
            mapped = classify_name(class_name_from_index(names, int(cls_idx[i])))
            if mapped is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(rgb.shape[1], x2); y2 = min(rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            if mapped == "car":
                cx = (x1 + x2) / (2.0 * img_w)
                cy = (y1 + y2) / (2.0 * img_h)
                area_ratio = ((x2 - x1) * (y2 - y1)) / float(img_w * img_h)
                in_forward_fov = args.car_forward_x_min <= cx <= args.car_forward_x_max and cy >= args.car_forward_y_min
                if not in_forward_fov or area_ratio < args.car_min_area_ratio:
                    continue

            d = median_depth_from_bbox(depth, (x1, y1, x2, y2), inset=0.2)
            if not np.isfinite(d) or d <= 0 or d > args.clipping_distance:
                continue
            distances[mapped] = min(distances[mapped], d)

    obs_d = forward_obstacle_distance(depth)
    if obs_d < args.forward_obstacle_max:
        distances["obstacle"] = min(distances["obstacle"], obs_d)

    return distances


def compute_rule_gain(now, dists, state: RuleState, args):
    if state.person_blocking:
        state.person_blocking = dists["person"] < args.person_resume
    else:
        state.person_blocking = dists["person"] < args.person_stop

    if state.stop_state == "cooldown" and now >= state.stop_until:
        state.stop_state = "armed"
    if state.stop_state == "armed" and dists["stop"] < args.stop_sign_trigger:
        state.stop_state = "stopping"
        state.stop_until = now + args.stop_hold
    if state.stop_state == "stopping" and now >= state.stop_until:
        state.stop_state = "cooldown"
        state.stop_until = now + args.stop_cooldown

    gain = 1.0
    if state.person_blocking or state.stop_state == "stopping":
        gain = 0.0
    if dists["traffic_red"] < args.traffic_light_stop:
        gain = 0.0

    if dists["car"] < args.car_slow:
        if args.car_slow <= args.car_stop:
            gain *= 0.0
        else:
            car_gain = np.clip((dists["car"] - args.car_stop) / (args.car_slow - args.car_stop), 0.0, 1.0)
            gain *= float(car_gain)

    if dists["yield"] < args.yield_trigger:
        conflicting = (dists["car"] < args.yield_clear_car_distance) or state.person_blocking
        if conflicting:
            gain *= float(np.clip(args.yield_gain, 0.0, 1.0))

    return float(np.clip(gain, 0.0, 1.0))