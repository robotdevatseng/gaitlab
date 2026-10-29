"""側面拍攝分析：站立期、擺盪期、雙腳支撐期、步時、步長、膝關節屈曲角度。

拍攝：iPad 橫拍，與走道垂直，對準量測區中間；走道中線上貼兩張 ArUco 標記（ID 4、5，相距 gap 公尺）。
尺度：鏡頭與走道垂直時，走道上的前後位置在畫面上是線性的，所以用兩張標記中心的水平像素位置換算公尺。
事件（O'Connor 的足部速度法）：
  著地 = 腳跟往前速度降到門檻以下；離地 = 腳尖往前速度升到門檻以上。
  側面拍攝沒有正面那種「腳離地投影成往後退」的問題，速度法可以使用。
只分析骨盆在兩張標記外側各 1 m 範圍內的步伐（畫面中央、透視最小）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks

from . import pose as P
from .gait import Params as FrontParams, clean_landmarks


@dataclass
class SideParams:
    vis_min: float = 0.5
    max_gap_s: float = 0.20
    cutoff_hz: float = 6.0
    thr_frac: float = 0.3          # 速度門檻 = 此比例 x 走路速度
    region_margin_m: float = 1.0   # 標記外側各延伸多少公尺仍採用
    min_stance_s: float = 0.2
    hs_vertical: bool = True       # 著地時間改用「腳跟垂直落到地面」判斷（腳跟落地時仍有前進速度）
    hs_drop_m: float = 0.01        # 腳跟高於站立期地面高度此值以內 = 已著地


@dataclass
class SideResult:
    summary: dict
    events: dict                   # {"L": [(t_hs, t_to, hs_pos_m), ...], "R": [...]}
    steps: list
    t: np.ndarray
    sig: dict = field(default_factory=dict)


def _cross_time(t, v, i, thr, rising=None):
    """在 i-1 與 i 之間做線性內插，求 v 越過 thr 的時間。"""
    if i <= 0 or np.isnan(v[i - 1]) or np.isnan(v[i]) or v[i] == v[i - 1]:
        return float(t[i])
    return float(t[i - 1] + (thr - v[i - 1]) / (v[i] - v[i - 1]) * (t[i] - t[i - 1]))


def _angle(a, b, c):
    """b 點的夾角轉成屈曲角（180 − 夾角），逐格。"""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cosv = np.sum(v1 * v2, axis=1) / np.where(n == 0, np.nan, n)
    return 180.0 - np.degrees(np.arccos(np.clip(cosv, -1, 1)))


def _overlap(intervals_a, intervals_b, lo, hi):
    tot = 0.0
    for a0, a1 in intervals_a:
        for b0, b1 in intervals_b:
            s, e = max(a0, b0, lo), min(a1, b1, hi)
            if e > s:
                tot += e - s
    return tot


LEG_PAIRS = [(P.L_KNEE, P.R_KNEE), (P.L_ANKLE, P.R_ANKLE), (P.L_HEEL, P.R_HEEL), (P.L_TOE, P.R_TOE)]


def fix_lr_swaps(xy: np.ndarray, margin: float = 0.7) -> tuple[np.ndarray, int]:
    """側面拍攝時，姿態模型常把遠側、近側腳的左右標記對調幾格。
    逐格比較「維持原標記」與「左右交換」哪個比較接近依前兩格推估的位置；交換明顯較好（成本 < margin 倍）才交換。"""
    xy = xy.copy()
    n_fix = 0
    L = [a for a, _ in LEG_PAIRS]
    R = [b for _, b in LEG_PAIRS]
    for i in range(2, len(xy)):
        keep = swap = 0.0
        n_ok = 0
        for a, b in LEG_PAIRS:
            pa = 2 * xy[i - 1, a] - xy[i - 2, a]
            pb = 2 * xy[i - 1, b] - xy[i - 2, b]
            if np.isnan(pa).any() or np.isnan(pb).any() or np.isnan(xy[i, a]).any() or np.isnan(xy[i, b]).any():
                continue
            keep += np.linalg.norm(xy[i, a] - pa) + np.linalg.norm(xy[i, b] - pb)
            swap += np.linalg.norm(xy[i, b] - pa) + np.linalg.norm(xy[i, a] - pb)
            n_ok += 1
        if n_ok >= 2 and swap < margin * keep:
            tmp = xy[i, L].copy()
            xy[i, L] = xy[i, R]
            xy[i, R] = tmp
            n_fix += 1
    return xy, n_fix


def analyze_side(lm: dict, markers_px: np.ndarray, gap_m: float = 2.0, prm: SideParams | None = None) -> SideResult:
    prm = prm or SideParams()
    fs, t = lm["fps"], lm["t"]
    raw = lm["xy"].copy()
    raw[lm["vis"] < prm.vis_min] = np.nan
    fixed, n_swap = fix_lr_swaps(raw)
    lm2 = {**lm, "xy": np.where(np.isnan(fixed), lm["xy"], fixed)}
    lm2["xy"][np.isnan(fixed)] = np.nan
    xy = clean_landmarks({**lm2, "vis": np.where(np.isnan(fixed[..., 0]), 0.0, 1.0)},
                         FrontParams(vis_min=prm.vis_min, max_gap_s=prm.max_gap_s, cutoff_hz=prm.cutoff_hz))
    x4, x5 = float(markers_px[0][0]), float(markers_px[1][0])
    if abs(x5 - x4) < 10:
        raise ValueError("兩張側面標記在畫面上太近，無法換算距離")
    to_m = lambda x: (x - x4) / (x5 - x4) * gap_m  # noqa: E731  公尺，以 4 號標記為 0、往 5 號為正
    px_per_m = abs(x5 - x4) / gap_m
    pel = to_m((xy[:, P.L_HIP, 0] + xy[:, P.R_HIP, 0]) / 2)
    dpel = np.gradient(pel) * fs
    direction = 1.0 if np.nanmedian(dpel) > 0 else -1.0
    in_reg = (pel >= -prm.region_margin_m) & (pel <= gap_m + prm.region_margin_m)
    v_walk = float(np.nanmedian(direction * dpel[in_reg])) if in_reg.any() else np.nan
    if not np.isfinite(v_walk) or v_walk <= 0.05:
        raise ValueError("偵測不到穩定的行走（骨盆沒有經過兩張標記之間）")
    thr = prm.thr_frac * v_walk

    sig, events = {}, {}
    for s, (ih, it, ia, ik, ip) in {"L": (P.L_HEEL, P.L_TOE, P.L_ANKLE, P.L_KNEE, P.L_HIP),
                                    "R": (P.R_HEEL, P.R_TOE, P.R_ANKLE, P.R_KNEE, P.R_HIP)}.items():
        heel_m, toe_m = to_m(xy[:, ih, 0]), to_m(xy[:, it, 0])
        vh = direction * np.gradient(heel_m) * fs
        vt = direction * np.gradient(toe_m) * fs
        knee = _angle(xy[:, ip], xy[:, ik], xy[:, ia])
        sig[s] = {"heel_m": heel_m, "toe_m": toe_m, "vh": vh, "vt": vt, "knee": knee}
        vhn = np.nan_to_num(vh, nan=0.0)
        pk, _ = find_peaks(vhn, height=1.3 * v_walk, distance=max(1, int(0.5 * fs)))
        ev = []
        for p0, p1 in zip(pk, pk[1:]):
            # 著地：p0 之後腳跟速度第一次降到門檻以下
            i = p0
            while i < p1 and (np.isnan(vh[i]) or vh[i] >= thr):
                i += 1
            if i >= p1:
                continue
            t_hs = _cross_time(t, vh, i, thr, rising=False)
            # 離地：p1 之前腳尖速度最後一次在門檻以下 → 之後升上去的時刻
            j = p1
            while j > i and (np.isnan(vt[j]) or vt[j] >= thr):
                j -= 1
            if j <= i or j + 1 >= len(t):
                continue
            t_to = _cross_time(t, vt, j + 1, thr, rising=True)
            # 著地時間精修：腳跟「垂直」落到站立期地面高度的時刻。
            # 真實步態腳跟落地時仍帶前進速度（會滑 0.5–1 cm），單用速度門檻會晚 20–30 ms。
            if prm.hs_vertical:
                hy = xy[:, ih, 1]                      # 影像 y（向下為正）
                a_, b_ = i + int(0.3 * (j - i)), i + int(0.6 * (j - i)) + 1
                floor_y = np.nanmedian(hy[a_:b_]) if b_ > a_ else np.nan
                if np.isfinite(floor_y):
                    lvl = floor_y - prm.hs_drop_m * px_per_m
                    k = i
                    while k > p0 and not np.isnan(hy[k - 1]) and hy[k - 1] >= lvl:
                        k -= 1
                    if k > p0 and not np.isnan(hy[k - 1]) and hy[k] != hy[k - 1]:
                        t_v = float(t[k - 1] + (lvl - hy[k - 1]) / (hy[k] - hy[k - 1]) * (t[k] - t[k - 1]))
                        if t[p0] < t_v <= t_hs:
                            t_hs = t_v
            if t_to - t_hs < prm.min_stance_s:
                continue
            seg = heel_m[i:min(len(t), i + int(0.1 * fs))]
            pos = float(np.nanmedian(seg)) if np.isfinite(seg).any() else np.nan
            pel_i = pel[i] if not np.isnan(pel[i]) else np.nan
            ok = (not np.isnan(pos) and -prm.region_margin_m <= pel_i <= gap_m + prm.region_margin_m)
            # 期間缺值過多則不採用
            valid = np.isfinite(vh[i:j + 1]).mean() > 0.8 and np.isfinite(vt[i:j + 1]).mean() > 0.8
            ev.append((t_hs, t_to, pos, bool(ok and valid), int(i), int(j)))
        events[s] = ev

    # ---- 指標
    stance_pct, swing_s, stance_s, stride_s, ds_pct, knee_pk = [], [], [], [], [], {"L": [], "R": []}
    iv = {s: [(e[0], e[1]) for e in events[s]] for s in "LR"}
    for A, B in (("L", "R"), ("R", "L")):
        evA = events[A]
        for e0, e1 in zip(evA, evA[1:]):
            if not (e0[3] and e1[3]):
                continue
            stride = e1[0] - e0[0]
            if stride <= 0 or stride > 3.0:
                continue
            stride_s.append(stride)
            stance_s.append(e0[1] - e0[0])
            swing_s.append(e1[0] - e0[1])
            stance_pct.append((e0[1] - e0[0]) / stride * 100)
            # 雙腳支撐期：對側腳必須有一段站立期跨過這次著地，且這個週期內對側腳有著地（資料完整才算）
            span = any(e[0] < e0[0] < e[1] for e in events[B])
            hs_in = any(e0[0] < e[0] < e1[0] for e in events[B])
            if span and hs_in:
                ds_pct.append(_overlap(iv[A], iv[B], e0[0], e1[0]) / stride * 100)
            a, b = e0[5] + 1, e1[4]
            if b > a:
                seg = sig[A]["knee"][a:b]
                if np.isfinite(seg).any():
                    knee_pk[A].append(float(np.nanmax(seg)))

    allc = sorted([(e[0], s, e[2], e[3]) for s in "LR" for e in events[s]])
    steps, step_t, step_l = [], [], []
    for c0, c1 in zip(allc, allc[1:]):
        if c0[3] and c1[3] and c0[1] != c1[1]:
            dt = c1[0] - c0[0]
            dl = direction * (c1[2] - c0[2]) * 100
            steps.append({"side": c1[1], "t_contact_s": c1[0], "step_time_s": dt, "step_length_cm": dl})
            step_t.append(dt)
            step_l.append(dl)
    # 合理性：步時偏離中位數太多的剔除（多半是左右標記對調或漏偵測）
    if step_t:
        med = np.median(step_t)
        keep = [i for i, d in enumerate(step_t) if 0.6 * med <= d <= 1.5 * med]
        steps = [steps[i] for i in keep]
        step_t = [step_t[i] for i in keep]
        step_l = [step_l[i] for i in keep]

    # 步速：著地點位置對時間的斜率（不用骨盆，因為鏡頭若有下俯角，骨盆高度的換算比例會偏）
    good = sorted([(e[0], e[2]) for s_ in "LR" for e in events[s_] if e[3]])
    speed = np.nan
    if len(good) >= 3:
        tt = np.array([g[0] for g in good])
        pp = np.array([g[1] for g in good])
        speed = float(abs(np.polyfit(tt, pp, 1)[0]))

    # 近側腳：站立期腳跟在畫面上的位置較低（離鏡頭近、透視較下方）
    def stance_heel_y(s_):
        ys = [np.nanmedian(xy[e[4]:e[5] + 1, P.L_HEEL if s_ == "L" else P.R_HEEL, 1]) for e in events[s_] if e[5] > e[4]]
        ys = [y for y in ys if np.isfinite(y)]
        return float(np.median(ys)) if ys else np.nan
    yL, yR = stance_heel_y("L"), stance_heel_y("R")
    near = "L" if (np.isfinite(yL) and np.isfinite(yR) and yL > yR) else ("R" if np.isfinite(yL) and np.isfinite(yR) else "?")
    m = lambda x: float(np.mean(x)) if len(x) else np.nan  # noqa: E731
    cv = lambda x: float(np.std(x, ddof=1) / np.mean(x) * 100) if len(x) > 1 else np.nan  # noqa: E731
    summary = {
        "direction": "往 5 號標記" if direction > 0 else "往 4 號標記",
        "walking_speed_cm_s": speed * 100,
        "lr_swaps_fixed_frames": n_swap,
        "cadence_steps_min": 60.0 / m(step_t) if step_t else np.nan,
        "n_strides": len(stride_s),
        "n_steps": len(step_t),
        "stance_pct": m(stance_pct),
        "swing_pct": 100 - m(stance_pct) if stance_pct else np.nan,
        "double_support_pct": m(ds_pct),
        "stance_time_s": m(stance_s),
        "swing_time_s": m(swing_s),
        "stride_time_s": m(stride_s),
        "stride_time_cv_pct": cv(stride_s),
        "step_time_cv_pct": cv(step_t),
        "step_length_cm": m(step_l),
        "knee_peak_swing_L_deg": m(knee_pk["L"]),
        "knee_peak_swing_R_deg": m(knee_pk["R"]),
        "near_side": near,
        "speed_threshold_m_s": thr,
    }
    return SideResult(summary, events, steps, t, sig)


def aggregate_side(results: list[SideResult]) -> dict:
    keys_mean = ["walking_speed_cm_s", "stance_pct", "swing_pct", "double_support_pct", "stance_time_s",
                 "swing_time_s", "stride_time_s", "step_length_cm"]
    out = {"n_trials": len(results), "n_strides": sum(r.summary["n_strides"] for r in results),
           "n_steps": sum(r.summary["n_steps"] for r in results)}
    for k in keys_mean:
        vals = [r.summary[k] for r in results if np.isfinite(r.summary[k])]
        out[k] = float(np.mean(vals)) if vals else np.nan
    st = [s["step_time_s"] for r in results for s in r.steps]
    out["cadence_steps_min"] = 60.0 / float(np.mean(st)) if st else np.nan
    out["step_time_cv_pct"] = float(np.std(st, ddof=1) / np.mean(st) * 100) if len(st) > 1 else np.nan
    # 膝角度：只採用靠近鏡頭那一側（遠側腳被擋住，較不準）
    for s in "LR":
        vals = [r.summary[f"knee_peak_swing_{s}_deg"] for r in results
                if r.summary["near_side"] == s and np.isfinite(r.summary[f"knee_peak_swing_{s}_deg"])]
        out[f"knee_peak_swing_{s}_deg"] = float(np.mean(vals)) if vals else np.nan
    return out
