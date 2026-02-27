# qcar2_autonomy/control/limits.py

def rate_limit(value, target, dt, up_rate, down_rate):
    max_up = max(float(up_rate), 1e-3) * float(dt)
    max_down = max(float(down_rate), 1e-3) * float(dt)
    if target >= value:
        return min(target, value + max_up)
    return max(target, value - max_down)