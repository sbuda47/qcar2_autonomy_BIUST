# qcar2_autonomy/control/speed.py

import numpy as np


class SpeedController:
    
    def __init__(self, kp=0.04, ki=0.15, max_throttle=0.14):
        self.maxThrottle = float(max_throttle)
        self.kp = float(kp)
        self.ki = float(ki)
        self.ei = 0.0

    def reset(self):
        self.ei = 0.0

    def update(self, v, v_ref, dt):
        e = float(v_ref - v)
        self.ei += max(float(dt), 1e-3) * e
        self.ei = float(np.clip(self.ei, -0.2, 0.2))
        if v_ref <= 0.01:
            self.ei *= 0.9
        u = self.kp * e + self.ki * self.ei
        return float(np.clip(u, 0.0, self.maxThrottle))