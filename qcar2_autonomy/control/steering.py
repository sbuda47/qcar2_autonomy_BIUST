# qcar2_autonomy/control/steering.py

import numpy as np
from pal.utilities.math import wrap_to_pi


class PathSteeringController:
    def __init__(self, waypoints, k=0.6, cyclic=False, max_steer=0.52, switch_distance=0.5, search_window=60):
        self.maxSteeringAngle = float(max_steer)
        self.wp = waypoints
        self.N = int(waypoints.shape[1])
        self.wpi = 0
        self.k = float(k)
        self.cyclic = bool(cyclic)
        self.switchDistance = float(switch_distance)
        self.searchWindow = max(10, int(search_window))
        self.pathComplete = False

    def reanchor_to_position(self, p):
        if self.N < 2:
            self.wpi = 0
            self.pathComplete = True
            return
        p2 = np.asarray(p[:2], dtype=np.float64)
        pts = self.wp[:2, :].T
        self.wpi = int(np.argmin(np.linalg.norm(pts - p2, axis=1)))
        if not self.cyclic and self.wpi >= self.N - 1:
            self.wpi = self.N - 2
            self.pathComplete = True

    def _advance_to_local_nearest(self, p):
        if self.N < 2:
            return
        p2 = np.asarray(p[:2], dtype=np.float64)
        start = max(0, self.wpi - 2)
        end = min(self.N, self.wpi + self.searchWindow)
        if end - start < 2:
            return
        local_pts = self.wp[:2, start:end].T
        nearest_local = int(np.argmin(np.linalg.norm(local_pts - p2, axis=1)))
        nearest_idx = start + nearest_local
        if nearest_idx > self.wpi:
            self.wpi = nearest_idx

    def _advance_index(self):
        if self.cyclic:
            self.wpi = int((self.wpi + 1) % max(self.N - 1, 1))
            return
        if self.wpi < self.N - 2:
            self.wpi += 1
        else:
            self.pathComplete = True

    def update(self, p, th, speed):
        if self.N < 2:
            self.pathComplete = True
            return 0.0

        if not self.cyclic and self.wpi >= self.N - 2:
            self.pathComplete = True

        self._advance_to_local_nearest(p)

        i1 = int(self.wpi)
        i2 = int((i1 + 1) % max(self.N - 1, 1)) if self.cyclic else min(i1 + 1, self.N - 1)

        wp_1 = self.wp[:2, i1]
        wp_2 = self.wp[:2, i2]
        p2 = np.asarray(p[:2], dtype=np.float64)

        v_seg = wp_2 - wp_1
        v_mag = float(np.linalg.norm(v_seg))
        if v_mag < 1e-6:
            self._advance_index()
            return 0.0

        v_uv = v_seg / v_mag
        tangent = float(np.arctan2(v_uv[1], v_uv[0]))
        s = float(np.dot(p2 - wp_1, v_uv))
        dist_to_next = float(np.linalg.norm(p2 - wp_2))

        switch_threshold = min(v_mag, max(self.switchDistance, 0.08))
        if s >= v_mag or dist_to_next < switch_threshold:
            self._advance_index()
        else:
            heading = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
            to_next = wp_2 - p2
            if float(np.dot(heading, to_next)) < -0.1:
                self._advance_index()

        s_clamped = float(np.clip(s, 0.0, v_mag))
        ep = wp_1 + v_uv * s_clamped
        ct = ep - p2
        side_dir = float(wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent))
        ect = float(np.linalg.norm(ct) * np.sign(side_dir))
        psi = float(wrap_to_pi(tangent - th))
        calc_speed = max(float(abs(speed)), 0.2)

        steering = psi + np.arctan2(self.k * ect, calc_speed)
        return float(np.clip(wrap_to_pi(steering), -self.maxSteeringAngle, self.maxSteeringAngle))