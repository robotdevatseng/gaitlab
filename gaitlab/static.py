"""靜態站立測試：受試者兩腳跟站在已知距離的標記上不動，量左右距離。

用途：把「姿態模型的關鍵點偏差」跟「走路時有沒有踩準」分開。
人不動就沒有踩不準的問題，量到的偏差就是關鍵點本身的偏差。
做法：找出兩腳都靜止 ≥ min_still_s 的時段，取中位數的地板位置，
分別用腳跟、腳踝、腳尖算兩腳的左右距離（已做高度修正）。
"""
from __future__ import annotations

import warnings

import numpy as np

from . import pose as P
from .calib import to_floor
from .gait import Params, clean_landmarks, _runs

IDX = {"heel": (P.L_HEEL, P.R_HEEL), "ankle": (P.L_ANKLE, P.R_ANKLE), "toe": (P.L_TOE, P.R_TOE)}


def analyze_static(lm: dict, H: np.ndarray, prm: Params | None = None, min_still_s: float = 1.5,
                   still_hipwidth_per_s: float = 0.15) -> list[dict]:
    prm = prm or Params()
    fs = lm["fps"]
    xy = clean_landmarks(lm, prm)
    fl = {nm: [to_floor(H, xy[:, k]) for k in IDX[nm]] for nm in IDX}
    # 兩腳的腳踝、腳跟都幾乎不動 = 站立靜止。用像素判斷（遠處地板深度方向雜訊大），
    # 位移 = 前後各 0.25 秒平均位置的差，門檻以骨盆像素寬為尺度（約 0.15 骨盆寬/秒 ≈ 5 cm/s）
    hw = int(round(0.25 * fs))
    hip_w = np.nanmedian(np.abs(xy[:, P.L_HIP, 0] - xy[:, P.R_HIP, 0]))
    T = len(xy)
    disp = np.full(T, np.inf)
    pts = xy[:, [P.L_HEEL, P.R_HEEL, P.L_ANKLE, P.R_ANKLE]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(hw, T - hw):
            d = np.nanmean(pts[i:i + hw], axis=0) - np.nanmean(pts[i - hw:i], axis=0)
            disp[i] = np.nanmax(np.linalg.norm(d, axis=1)) / 0.25 if not np.isnan(d).all() else np.inf
    thr = still_hipwidth_per_s * hip_w if np.isfinite(hip_w) and hip_w > 0 else 20.0
    still = disp < thr
    hts = {"heel": prm.landmark_height_m, "ankle": prm.ankle_height_m, "toe": 0.02}
    out = []
    for a, b in _runs(still):
        if b - a < min_still_s * fs:
            continue
        a2, b2 = a + int(0.2 * (b - a)), b - int(0.2 * (b - a))  # 去頭尾
        row = {"t_start_s": float(lm["t"][a]), "t_end_s": float(lm["t"][b - 1])}
        L_ank = np.nanmedian(fl["ankle"][0][a2:b2], axis=0)
        R_ank = np.nanmedian(fl["ankle"][1][a2:b2], axis=0)
        row["dist_from_near_edge_m"] = float((L_ank[1] + R_ank[1]) / 2)
        for nm in IDX:
            L = np.nanmedian(fl[nm][0][a2:b2], axis=0)
            R = np.nanmedian(fl[nm][1][a2:b2], axis=0)
            ks = (prm.cam_height_m - hts[nm]) / prm.cam_height_m
            row[f"{nm}_lateral_cm"] = float(abs(L[0] - R[0]) * 100 * ks)
        out.append(row)
    return out
