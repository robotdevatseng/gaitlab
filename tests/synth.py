"""合成驗證：用已知步寬的虛擬行走，經過真實的相機投影，檢查演算法能否算回正確數值。

模擬條件：iPad 放在走道末端，高 1.1 m、略往下看；受試者從 8 m 外朝鏡頭走過來。
腳跟、腳尖、骨盆的 3D 位置 → 針孔相機投影成像素 → 加上像素雜訊 → 丟進 gait.analyze。
"""
from __future__ import annotations

import numpy as np

from gaitlab import pose as P
from gaitlab.calib import homography


class Camera:
    def __init__(self, pos=(0.5, -1.5, 1.1), pitch_deg=8.0, f=1500.0, size=(1920, 1080)):
        self.pos = np.array(pos, float)
        self.f = f
        self.w, self.h = size
        p = np.radians(pitch_deg)
        # 相機座標：x 右、y 下、z 前；世界：X 右、Y 前（遠離相機）、Z 上
        fwd = np.array([0, np.cos(p), -np.sin(p)])
        right = np.array([1.0, 0, 0])
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])

    def project(self, Pw: np.ndarray) -> np.ndarray:
        pc = (np.asarray(Pw, float) - self.pos) @ self.R.T
        u = self.f * pc[..., 0] / pc[..., 2] + self.w / 2
        v = self.f * pc[..., 1] / pc[..., 2] + self.h / 2
        return np.stack([u, v], -1)


def simulate(step_width=0.12, sw_sd=0.02, speed=1.0, cadence=110.0, stance_frac=0.62,
             foot_angle_deg=10.0, fps=60.0, px_noise=2.0, heel_z=0.03, seed=0,
             y_start=8.5, y_end=-0.8, cam=None, zone=(0.0, 6.0), lane_w=1.0):
    rng = np.random.default_rng(seed)
    step_t = 60.0 / cadence
    step_len = speed * step_t
    stride_t = 2 * step_t
    n_steps = int((y_start - y_end) / step_len) + 2
    # 著地點（真值）：左腳在行進方向左側；朝鏡頭走時，受試者左側 = 畫面右側 (+X)
    places, sides, times = [], [], []
    for k in range(n_steps):
        side = "L" if k % 2 == 0 else "R"
        lateral = (step_width / 2 + rng.normal(0, sw_sd / np.sqrt(2))) * (1 if side == "L" else -1)
        places.append([0.5 + lateral, y_start - k * step_len])
        sides.append(side)
        times.append(k * step_t)
    places = np.array(places)
    times = np.array(times)
    T = int((times[-1]) * fps)
    t = np.arange(T) / fps

    heel = {"L": np.zeros((T, 3)), "R": np.zeros((T, 3))}
    toe = {"L": np.zeros((T, 3)), "R": np.zeros((T, 3))}
    ang = np.radians(foot_angle_deg)
    for s in "LR":
        idx = [k for k in range(n_steps) if sides[k] == s]
        for i, ti in enumerate(t):
            # 這隻腳最近一次著地 k0 與下一次 k1
            prev = [k for k in idx if times[k] <= ti]
            if not prev:
                k0 = idx[0]
                pos, z = places[k0], heel_z
            else:
                k0 = prev[-1]
                nxt = [k for k in idx if k > k0]
                stance_end = times[k0] + stance_frac * stride_t
                if ti < stance_end or not nxt:
                    pos, z = places[k0], heel_z
                else:
                    k1 = nxt[0]
                    ph = (ti - stance_end) / (times[k1] - stance_end)
                    ph = np.clip(ph, 0, 1)
                    w = (1 - np.cos(np.pi * ph)) / 2
                    pos = places[k0] + w * (places[k1] - places[k0])
                    z = heel_z + 0.10 * np.sin(np.pi * ph)
            heel[s][i] = [pos[0], pos[1], z]
            out = 1 if s == "L" else -1  # 外八：左腳腳尖偏向 +X（受試者左）
            toe[s][i] = [pos[0] + out * 0.2 * np.sin(ang), pos[1] - 0.2 * np.cos(ang), 0.02]

    # 骨盆：等速前進 + 左右晃動
    yc = y_start - speed * t - step_len / 2
    sway = 0.02 * np.sin(np.pi * t / step_t)
    hipL = np.stack([0.5 + 0.16 + sway, yc, np.full(T, 0.9)], -1)
    hipR = np.stack([0.5 - 0.16 + sway, yc, np.full(T, 0.9)], -1)

    cam = cam or Camera()
    xy = np.zeros((T, P.N_LANDMARKS, 2))
    pts = {P.L_HEEL: heel["L"], P.R_HEEL: heel["R"], P.L_TOE: toe["L"], P.R_TOE: toe["R"],
           P.L_ANKLE: heel["L"] + [0, -0.03, 0.06], P.R_ANKLE: heel["R"] + [0, -0.03, 0.06],
           P.L_HIP: hipL, P.R_HIP: hipR}
    for k in range(P.N_LANDMARKS):
        xy[:, k] = cam.project(pts.get(k, (hipL + hipR) / 2))
    xy += rng.normal(0, px_noise, xy.shape)
    vis = np.ones((T, P.N_LANDMARKS))
    lm = {"xy": xy, "vis": vis, "t": t, "fps": fps, "size": (cam.w, cam.h)}

    # 校正：地板長方形 1.0 x 6.0 m 的四角，點選誤差 ±1.5 px
    z0, z1 = zone
    corners_w = np.array([[0, z0, 0], [lane_w, z0, 0], [lane_w, z1, 0], [0, z1, 0]], float)
    img_pts = cam.project(corners_w) + rng.normal(0, 1.5, (4, 2))
    H = homography(img_pts, lane_w, z1 - z0)

    # 真值（只取量測區 0–6 m 內的步，以 GAITRite 定義計算）
    inz = (places[:, 1] >= z0) & (places[:, 1] <= z1)
    zp = places[inz]
    sw_true = []
    for k in range(1, len(zp) - 1):
        a, b = zp[k - 1], zp[k + 1]
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        sw_true.append(abs(np.dot(zp[k] - a, n)) * 100)
    truth = {"step_width_mean_cm": float(np.mean(sw_true)), "step_width_sd_cm": float(np.std(sw_true, ddof=1)),
             "gait_speed_cm_s": speed * 100, "cadence_steps_min": cadence,
             "step_length_mean_cm": step_len * 100, "foot_angle_deg": foot_angle_deg,
             "double_support_pct": (2 * stance_frac - 1) * 100}
    return lm, H, truth
