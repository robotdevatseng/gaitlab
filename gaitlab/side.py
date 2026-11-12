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
    event_method: str = "position"  # "position"：腳相對骨盆的前後位置極值（Zeni 2008，預設）；"velocity"：舊版足部速度法
    lr_method: str = "anchor"      # 左右腳整理方式："anchor"（預設）、"viterbi"、"greedy"（舊版）、"none"（完全照模型）
    mirrored: bool = False         # 影片是否左右鏡像（前鏡頭自拍模式）；能自動辨識 ArUco 標記就一定不是鏡像


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
    """舊版（2026-10）：逐格貪婪比較。保留供比較用，analyze_side 已不使用。"""
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


def track_legs(xy: np.ndarray, swap_prior_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """把每格的兩條腿整理成兩條前後連貫的軌跡（Viterbi 全域最佳化）。

    每格只有兩種可能：維持模型給的左右標記，或兩腿對調。成本 = 各腿關鍵點（膝、踝、跟、尖）
    與「前兩格等速外推位置」的距離總和；另外對「對調」加一點成本（模型大多數時候是對的）。
    與舊版逐格貪婪法不同，這裡找的是整段影片總成本最低的組合，兩腿交會時不會被單格雜訊帶偏後一路錯下去。
    回傳 (整理後的 xy, 每格是否對調的布林陣列)。
    """
    T = len(xy)
    L = [a for a, _ in LEG_PAIRS]
    R = [b for _, b in LEG_PAIRS]
    A, B = xy[:, L], xy[:, R]                      # (T, 4, 2)
    if T < 3:
        return xy.copy(), np.zeros(T, bool)

    def leg(i, s):                                 # 第 i 格在狀態 s 下的（軌跡1, 軌跡2）
        return (A[i], B[i]) if s == 0 else (B[i], A[i])

    def cost(i, s2, s1, s0):                       # s2,s1,s0 = 第 i-2, i-1, i 格的狀態
        c = 0.0
        for k in (0, 1):
            p2, p1, p0 = leg(i - 2, s2)[k], leg(i - 1, s1)[k], leg(i, s0)[k]
            d = np.linalg.norm(p0 - (2 * p1 - p2), axis=1)
            ok = ~np.isnan(d)
            if ok.any():
                c += float(d[ok].mean()) * 4
            else:                                  # 前兩格缺值 → 退而比較與前一格的距離
                d1 = np.linalg.norm(p0 - p1, axis=1)
                ok1 = ~np.isnan(d1)
                if ok1.any():
                    c += float(d1[ok1].mean()) * 4
        return c + (swap_prior_px if s0 else 0.0)

    states = [(a, b) for a in (0, 1) for b in (0, 1)]          # (s_{i-1}, s_i)
    INF = float("inf")
    score = {st: (swap_prior_px * (st[0] + st[1])) for st in states}
    back = []
    for i in range(2, T):
        new, bp = {}, {}
        for s1, s0 in states:
            best, arg = INF, None
            for s2 in (0, 1):
                v = score[(s2, s1)] + cost(i, s2, s1, s0)
                if v < best:
                    best, arg = v, (s2, s1)
            new[(s1, s0)], bp[(s1, s0)] = best, arg
        score = new
        back.append(bp)
    st = min(score, key=score.get)
    seq = [st[1], st[0]]
    for bp in reversed(back):
        st = bp[st]
        seq.append(st[0])
    seq = np.array(seq[::-1][-T:] if len(seq) >= T else seq[::-1], bool)
    seq = seq[:T]
    out = xy.copy()
    out[np.ix_(seq, L)] = xy[np.ix_(seq, R)]
    out[np.ix_(seq, R)] = xy[np.ix_(seq, L)]
    return out, seq


def anchor_legs(xy: np.ndarray, fs: float, win_s: float = 0.25, tol_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """以模型原本的左右標記為準，只修正「短暫」的對調與亂跳（預設方法，2026-11）。

    五位同仁的側面資料顯示：模型的左右標記大部分時間是對的，只在兩腿交會的幾格會對調或亂跳。
    做法：先對每條腿的關鍵點（相對骨盆的位置）做 0.25 秒中位數濾波，得到不受短暫錯誤影響的參考軌跡；
    再逐格比較「維持」與「對調」哪個比較接近參考軌跡。對調明顯較近 → 對調；
    兩種都離參考軌跡很遠（超過 tol_frac × 小腿長）→ 該格腿部點視為無效，交給後面的內插補上。
    不會像逐格追蹤那樣在交會後一路錯下去。
    回傳 (整理後的 xy, 每格是否對調, 每格是否作廢)。
    """
    from scipy.ndimage import median_filter
    T = len(xy)
    L = [a for a, _ in LEG_PAIRS]
    R = [b for _, b in LEG_PAIRS]
    hip = (xy[:, P.L_HIP] + xy[:, P.R_HIP]) / 2                      # (T, 2)
    k = max(3, int(round(win_s * fs)) | 1)

    def ref(idx):
        rel = xy[:, idx] - hip[:, None, :]                            # (T, 4, 2)
        out = np.full_like(rel, np.nan)
        for j in range(rel.shape[1]):
            for c in (0, 1):
                v = rel[:, j, c]
                ok = np.isfinite(v)
                if ok.sum() < k:
                    continue
                filled = np.interp(np.arange(T), np.nonzero(ok)[0], v[ok])
                out[:, j, c] = median_filter(filled, size=k, mode="nearest")
        return rel, out
    relL, refL = ref(L)
    relR, refR = ref(R)
    shank = np.nanmedian(np.linalg.norm(xy[:, P.L_KNEE] - xy[:, P.L_ANKLE], axis=1))
    tol = tol_frac * shank if np.isfinite(shank) and shank > 0 else 40.0

    def dist(a, b):
        d = np.linalg.norm(a - b, axis=2)                             # (T, 4)
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(d, axis=1)
    keep = dist(relL, refL) + dist(relR, refR)
    swap = dist(relR, refL) + dist(relL, refR)
    swapped = np.isfinite(keep) & np.isfinite(swap) & (swap < 0.6 * keep) & (keep > tol)
    best = np.where(swapped, swap, keep)
    bad = np.isfinite(best) & (best > 2 * tol) 
    out = xy.copy()
    out[np.ix_(swapped, L)] = xy[np.ix_(swapped, R)]
    out[np.ix_(swapped, R)] = xy[np.ix_(swapped, L)]
    out[np.ix_(bad, L + R)] = np.nan
    return out, swapped, bad


def assign_anatomical(xy: np.ndarray, direction_px: float, mirrored: bool = False) -> tuple[np.ndarray, dict]:
    """把兩條軌跡標成人體解剖側（受試者自己的左、右）。

    側面拍攝、畫面未鏡像時：受試者往畫面右邊走 → 靠鏡頭的是他的右腳；往左走 → 靠鏡頭的是左腳。
    靠鏡頭的腳站立時在畫面上位置較低（y 較大）。若目前標成「近側」的那條腿其實在較遠處，整段對調。
    """
    near_expected = "R" if (direction_px > 0) != mirrored else "L"

    def ground_y(idx):                             # 該腿踝、跟、尖的最低位置（第 90 百分位 ≈ 著地時）
        v = xy[:, idx, 1]
        v = v[~np.isnan(v)]
        return float(np.percentile(v, 90)) if len(v) > 10 else np.nan

    yL = np.nanmean([ground_y(P.L_HEEL), ground_y(P.L_TOE), ground_y(P.L_ANKLE)])
    yR = np.nanmean([ground_y(P.R_HEEL), ground_y(P.R_TOE), ground_y(P.R_ANKLE)])
    info = {"near_side_expected": near_expected, "near_far_gap_px": None, "whole_trial_flipped": False,
            "near_side_confident": False}
    if not (np.isfinite(yL) and np.isfinite(yR)):
        return xy, info
    near_now = "L" if yL > yR else "R"
    info["near_far_gap_px"] = float(abs(yL - yR))
    info["near_side_confident"] = bool(abs(yL - yR) >= 4.0)   # 兩腳高度差太小時無法確認
    if info["near_side_confident"] and near_now != near_expected:
        out = xy.copy()
        L = [a for a, _ in LEG_PAIRS]
        R = [b for _, b in LEG_PAIRS]
        out[:, L], out[:, R] = xy[:, R], xy[:, L]
        info["whole_trial_flipped"] = True
        return out, info
    return xy, info


def _refine_peak(t, y, k):
    """三點拋物線內插，回傳極值的時間（次影格）。"""
    if 0 < k < len(y) - 1 and np.isfinite(y[k - 1:k + 2]).all():
        den = y[k - 1] - 2 * y[k] + y[k + 1]
        if den != 0:
            off = float(np.clip(0.5 * (y[k - 1] - y[k + 1]) / den, -1, 1))
            return float(t[k] + off * (t[min(k + 1, len(t) - 1)] - t[k]))
    return float(t[k])


def _events_position(t, fs, heel_rel, toe_rel, heel_m, pel, gap_m, prm, vh=None, vt=None, hy=None,
                     thr=None, px_per_m=None) -> list:
    """兩階段事件偵測。

    第一階段（找到每一次站立）：Zeni 2008 座標法，著地 ≈ 腳跟相對骨盆最靠前、離地 ≈ 腳尖相對骨盆最靠後。
      這兩個時刻兩腳分得最開，離「兩腿交會、模型容易把左右腳標反」的時刻最遠，所以很耐雜訊。
      舊版直接在整段影片上找腳速峰值，交會處的假速度會造成多餘或漏掉的事件
      （2026-11 五位同仁：站立期每趟 59–75%、雙腳支撐期 14–66%）。
    第二階段（精修時間）：座標法的著地會偏早（腳到最前方後還要一小段時間才落地），站立期會高估約 3 個百分點。
      所以只在第一階段事件的 ±0.2 秒內，用原本的定義精修：
      著地 = 腳跟前進速度降到門檻以下，再往回找腳跟垂直落到地面高度的時刻；離地 = 腳尖前進速度升到門檻以上。
      找不到就保留座標法的時間。
    """
    def fill(a):
        a = a.copy()
        ok = np.isfinite(a)
        if ok.sum() < 5:
            return None
        return np.interp(np.arange(len(a)), np.nonzero(ok)[0], a[ok])
    h, o = fill(heel_rel), fill(toe_rel)
    if h is None or o is None:
        return []
    amp = np.nanpercentile(h, 95) - np.nanpercentile(o, 5)
    if not np.isfinite(amp) or amp <= 0.1:
        return []
    dist = max(1, int(0.4 * fs))
    hs, _ = find_peaks(h, distance=dist, prominence=0.25 * amp)
    to, _ = find_peaks(-o, distance=dist, prominence=0.25 * amp)
    ev = []
    for a, k in enumerate(hs):
        nxt = hs[a + 1] if a + 1 < len(hs) else len(t)
        cand = [j for j in to if k < j < nxt]
        if not cand:
            continue
        j = cand[0]
        t_hs, t_to = _refine_peak(t, h, k), _refine_peak(t, -o, j)
        i_hs, i_to = int(k), int(j)
        m = int(round(0.2 * fs))
        if vh is not None and thr is not None:
            # 著地：從座標法時刻前 0.1 秒起，腳跟速度第一次降到門檻以下
            a0 = max(1, k - m // 2)
            i = a0
            lim = min(len(t) - 1, k + m, j)
            while i < lim and (np.isnan(vh[i]) or vh[i] >= thr):
                i += 1
            if i < lim and i > a0:
                t_v = _cross_time(t, vh, i, thr)
                i_hs = i
                if prm.hs_vertical and hy is not None and px_per_m:
                    a_, b_ = i + int(0.3 * (j - i)), i + int(0.6 * (j - i)) + 1
                    floor_y = np.nanmedian(hy[a_:b_]) if b_ > a_ and np.isfinite(hy[a_:b_]).any() else np.nan
                    if np.isfinite(floor_y):
                        lvl = floor_y - prm.hs_drop_m * px_per_m
                        q = i
                        while q > a0 and not np.isnan(hy[q - 1]) and hy[q - 1] >= lvl:
                            q -= 1
                        if q > a0 and not np.isnan(hy[q - 1]) and hy[q] != hy[q - 1]:
                            t_q = float(t[q - 1] + (lvl - hy[q - 1]) / (hy[q] - hy[q - 1]) * (t[q] - t[q - 1]))
                            if t[a0] < t_q <= t_v:
                                t_v = t_q
                t_hs = t_v
        if vt is not None and thr is not None:
            # 離地：座標法時刻前後 0.2 秒內，腳尖速度最後一次在門檻以下 → 之後升上去的時刻
            b0 = min(len(t) - 2, j + m)
            q = b0
            lo = max(i_hs, j - m)
            while q > lo and (np.isnan(vt[q]) or vt[q] >= thr):
                q -= 1
            if lo < q < b0:
                t_to = _cross_time(t, vt, q + 1, thr)
                i_to = q
        k, j = i_hs, max(i_to, i_hs + 1)
        if t_to - t_hs < prm.min_stance_s:
            continue
        seg = heel_m[k:min(len(t), k + max(1, int(0.1 * fs)))]
        pos = float(np.nanmedian(seg)) if np.isfinite(seg).any() else np.nan
        pel_i = pel[k]
        ok = bool(np.isfinite(pos) and np.isfinite(pel_i) and -prm.region_margin_m <= pel_i <= gap_m + prm.region_margin_m)
        valid = bool(np.isfinite(heel_rel[k:j + 1]).mean() > 0.8 and np.isfinite(toe_rel[k:j + 1]).mean() > 0.8)
        ev.append((t_hs, t_to, pos, bool(ok and valid), int(k), int(j)))
    return ev


def analyze_side(lm: dict, markers_px: np.ndarray, gap_m: float = 2.04, prm: SideParams | None = None) -> SideResult:
    prm = prm or SideParams()
    fs, t = lm["fps"], lm["t"]
    raw = lm["xy"].copy()
    raw[lm["vis"] < prm.vis_min] = np.nan
    lr_info = {"lr_method": prm.lr_method, "mirrored": bool(prm.mirrored)}
    hip_px = (raw[:, P.L_HIP, 0] + raw[:, P.R_HIP, 0]) / 2
    dir_px = float(np.nanmedian(np.diff(hip_px))) if np.isfinite(hip_px).sum() > 5 else 0.0
    if prm.lr_method == "viterbi":
        fixed, swapped = track_legs(raw)
        fixed, a_info = assign_anatomical(fixed, dir_px, prm.mirrored)
        if a_info["whole_trial_flipped"]:
            swapped = ~swapped
        n_swap = int(swapped.sum())
        lr_info.update(a_info)
    elif prm.lr_method == "anchor":
        fixed, swapped, bad = anchor_legs(raw, fs)
        fixed, a_info = assign_anatomical(fixed, dir_px, prm.mirrored)
        n_swap = int(len(raw) - swapped.sum()) if a_info["whole_trial_flipped"] else int(swapped.sum())
        lr_info.update(a_info)
        lr_info.update({"frames_swapped_locally": int(swapped.sum()), "frames_discarded": int(bad.sum())})
    elif prm.lr_method == "greedy":
        fixed, n_swap = fix_lr_swaps(raw)
    else:
        fixed, n_swap = raw, 0
    lr_info.update({"frames_total": int(len(raw)), "frames_relabelled": int(n_swap),
                    "walking_direction_image": "往畫面右方" if dir_px > 0 else "往畫面左方"})
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
        ev = []
        if prm.event_method == "position":
            events[s] = _events_position(t, fs, direction * (heel_m - pel), direction * (toe_m - pel), heel_m, pel,
                                         gap_m, prm, vh=vh, vt=vt, hy=xy[:, ih, 1], thr=thr, px_per_m=px_per_m)
            continue
        vhn = np.nan_to_num(vh, nan=0.0)
        pk, _ = find_peaks(vhn, height=1.3 * v_walk, distance=max(1, int(0.5 * fs)))
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
    # 週期時間偏離中位數太多（漏掉一次著地會變成兩倍長）的週期不採用
    all_strides = [e1[0] - e0[0] for s_ in "LR" for e0, e1 in zip(events[s_], events[s_][1:]) if e0[3] and e1[3]]
    med_stride = float(np.median(all_strides)) if all_strides else np.nan
    # 參考週期優先用「左右交替相鄰著地的間隔中位數 × 2」（週期數很少時，中位數會被漏偵測的長週期帶偏）
    _c = sorted([(e[0], s_) for s_ in "LR" for e in events[s_]])
    _dt = [b[0] - a[0] for a, b in zip(_c, _c[1:]) if a[1] != b[1]]
    if len(_dt) >= 2:
        med_stride = 2 * float(np.median(_dt))
    n_stride_outliers = 0
    for A, B in (("L", "R"), ("R", "L")):
        evA = events[A]
        for e0, e1 in zip(evA, evA[1:]):
            if not (e0[3] and e1[3]):
                continue
            stride = e1[0] - e0[0]
            if stride <= 0 or stride > 3.0:
                continue
            if np.isfinite(med_stride) and not (0.7 * med_stride <= stride <= 1.4 * med_stride):
                n_stride_outliers += 1
                continue
            if not (40.0 <= (e0[1] - e0[0]) / stride * 100 <= 85.0):   # 站立期不在 40–85% → 事件偵測錯誤
                n_stride_outliers += 1
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
        "lr": lr_info,
        "cadence_steps_min": 60.0 / m(step_t) if step_t else np.nan,
        "n_strides": len(stride_s),
        "n_strides_excluded_irregular": n_stride_outliers,
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
