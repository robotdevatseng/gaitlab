"""側面拍攝的合成驗證：已知站立期、雙腳支撐期、步長與膝關節角度的虛擬行走，從側面相機投影。

世界座標：X 左右、Y 前後（受試者朝 −Y 方向走）、Z 上。
側面相機放在走道旁（X = 走道中線 + 距離），看向 −X。
膝關節用兩節連桿（大腿、小腿各 0.45 m）逆運動學求得，因此有真實的膝屈曲角度可以比對。
"""
from __future__ import annotations

import numpy as np

from gaitlab import pose as P


class SideCamera:
    def __init__(self, dist=3.2, along=5.0, height=1.0, lane_x=0.5, f=1500.0, size=(1920, 1080),
                 yaw_deg=0.0, pitch_deg=0.0):
        self.pos = np.array([lane_x + dist, along, height], float)
        self.f, (self.w, self.h) = f, size
        yw, pt = np.radians(yaw_deg), np.radians(pitch_deg)
        fwd = np.array([-np.cos(yw) * np.cos(pt), np.sin(yw) * np.cos(pt), -np.sin(pt)])
        up = np.array([0, 0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])

    def project(self, Pw):
        pc = (np.asarray(Pw, float) - self.pos) @ self.R.T
        return np.stack([self.f * pc[..., 0] / pc[..., 2] + self.w / 2,
                         self.f * pc[..., 1] / pc[..., 2] + self.h / 2], -1)


def _knee(Hp, A, fwd, L1=0.45, L2=0.45):
    d = np.linalg.norm(A - Hp)
    e = (A - Hp) / d
    d = min(d, L1 + L2 - 1e-6)
    a = (L1 ** 2 - L2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(L1 ** 2 - a ** 2, 0))
    perp = fwd - np.dot(fwd, e) * e
    perp /= np.linalg.norm(perp)
    return Hp + a * e + h * perp


def _flex(Hp, K, A):
    v1, v2 = Hp - K, A - K
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def simulate_side(speed=1.0, cadence=110.0, stance_frac=0.62, step_width=0.12, fps=60.0, px_noise=2.0,
                  seed=0, y_start=9.5, y_end=0.5, cam=None, marker_y=(4.0, 6.0), lane_x=0.5,
                  direction=-1, swing_lift=0.12):
    rng = np.random.default_rng(seed)
    step_t = 60.0 / cadence
    step_len = speed * step_t
    stride_t = 2 * step_t
    if direction > 0:
        y_start, y_end = y_end, y_start
    n_steps = int(abs(y_start - y_end) / step_len) + 2
    fwd = np.array([0, float(np.sign(direction)), 0])
    places, sides, times = [], [], []
    for k in range(n_steps):
        s = "L" if k % 2 == 0 else "R"
        # 受試者朝 −Y 走時，左側 = +X；朝 +Y 走時，左側 = −X
        lat = step_width / 2 * (1 if s == "L" else -1) * (-np.sign(direction))
        places.append([lane_x + lat, y_start + direction * k * step_len])
        sides.append(s)
        times.append(k * step_t)
    places, times = np.array(places), np.array(times)
    T = int(times[-1] * fps)
    t = np.arange(T) / fps
    heel = {s: np.zeros((T, 3)) for s in "LR"}
    toe = {s: np.zeros((T, 3)) for s in "LR"}
    for s in "LR":
        idx = [k for k in range(n_steps) if sides[k] == s]
        for i, ti in enumerate(t):
            prev = [k for k in idx if times[k] <= ti]
            k0 = prev[-1] if prev else idx[0]
            nxt = [k for k in idx if k > k0]
            st_end = times[k0] + stance_frac * stride_t
            if (not prev) or ti < st_end or not nxt:
                pos, z = places[k0], 0.03
            else:
                ph = np.clip((ti - st_end) / (times[nxt[0]] - st_end), 0, 1)
                w = (1 - np.cos(np.pi * ph)) / 2
                pos = places[k0] + w * (places[nxt[0]] - places[k0])
                z = 0.03 + swing_lift * np.sin(np.pi * ph)
            heel[s][i] = [pos[0], pos[1], z]
            toe[s][i] = [pos[0], pos[1] + direction * 0.2, z - 0.01]
    yc = y_start + direction * (speed * t + step_len / 2)
    hip = {"L": np.stack([np.full(T, lane_x + 0.16 * (-np.sign(direction))), yc, np.full(T, 0.92)], -1),
           "R": np.stack([np.full(T, lane_x - 0.16 * (-np.sign(direction))), yc, np.full(T, 0.92)], -1)}
    ankle = {s: heel[s] + np.array([0, direction * 0.05, 0.07]) for s in "LR"}
    knee = {s: np.array([_knee(hip[s][i], ankle[s][i], fwd) for i in range(T)]) for s in "LR"}
    flex = {s: np.array([_flex(hip[s][i], knee[s][i], ankle[s][i]) for i in range(T)]) for s in "LR"}

    cam = cam or SideCamera(along=float(np.mean(marker_y)), lane_x=lane_x)
    pts = {P.L_HEEL: heel["L"], P.R_HEEL: heel["R"], P.L_TOE: toe["L"], P.R_TOE: toe["R"],
           P.L_ANKLE: ankle["L"], P.R_ANKLE: ankle["R"], P.L_KNEE: knee["L"], P.R_KNEE: knee["R"],
           P.L_HIP: hip["L"], P.R_HIP: hip["R"]}
    xy = np.zeros((T, P.N_LANDMARKS, 2))
    mid = (hip["L"] + hip["R"]) / 2 + [0, 0, 0.5]
    for k in range(P.N_LANDMARKS):
        xy[:, k] = cam.project(pts.get(k, mid))
    xy += rng.normal(0, px_noise, xy.shape)
    lm = {"xy": xy, "vis": np.ones((T, P.N_LANDMARKS)), "t": t, "fps": fps, "size": (cam.w, cam.h)}
    mk = cam.project(np.array([[lane_x, marker_y[0], 0], [lane_x, marker_y[1], 0]])) + rng.normal(0, 1.0, (2, 2))

    # 真值：只看步伐落在標記兩側各 1 m 範圍內
    lo, hi = min(marker_y) - 1.0, max(marker_y) + 1.0
    swing_peak = []
    for s in "LR":
        idx = [k for k in range(n_steps) if sides[k] == s]
        for k0, k1 in zip(idx, idx[1:]):
            if lo <= places[k0][1] <= hi and lo <= places[k1][1] <= hi:
                a = int((times[k0] + stance_frac * stride_t) * fps)
                b = int(times[k1] * fps)
                if b > a:
                    swing_peak.append(flex[s][a:b].max())
    truth = {"stance_pct": stance_frac * 100, "double_support_pct": (2 * stance_frac - 1) * 100,
             "cadence_steps_min": cadence, "step_length_cm": step_len * 100, "gait_speed_cm_s": speed * 100,
             "knee_peak_swing_deg": float(np.mean(swing_peak)) if swing_peak else np.nan}
    return lm, mk, truth
