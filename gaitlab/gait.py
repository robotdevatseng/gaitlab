"""步態事件偵測與指標計算（正面拍攝、單鏡頭）。

輸入：逐格關鍵點（像素）+ 地板校正矩陣 H。
做法：
  1. 低可見度的點設為缺值 → 短缺口內插 → 6 Hz 零相位低通濾波
  2. 腳跟、腳尖投影到地板（公尺）
  3. 著地偵測：雙腳前後距離法（detect_contacts）；舊的腳速門檻法保留為 event_method="speed"
  4. 著地位置：前後方向用腳尖、左右方向用腳跟
  5. 以 GAITRite 定義計算步寬、步長、步幅、時間參數、變異度
  站立期與雙腳支撐期在正面拍攝下無法可靠量測，不提供。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import butter, filtfilt

from . import pose as P
from .calib import to_floor

SIDES = {"L": (P.L_HEEL, P.L_TOE, P.L_ANKLE), "R": (P.R_HEEL, P.R_TOE, P.R_ANKLE)}


@dataclass
class Params:
    vis_min: float = 0.5          # 低於此可見度視為缺值
    max_gap_s: float = 0.20       # 可內插的最長缺口
    cutoff_hz: float = 6.0        # 低通濾波截止頻率
    stance_frac: float = 0.25     # 站立期速度門檻 = 此比例 x 平均腳速（合成資料調校值）
    stance_min_thr: float = 0.12  # 門檻下限（m/s）
    min_stance_s: float = 0.15
    min_swing_s: float = 0.15
    same_spot_m: float = 0.15     # 同一腳兩段站立期著地點距離小於此值 → 視為同一個腳印（雜訊切斷）
    step_time_lo: float = 0.6     # 步時 < 0.6 x 中位數 → 視為異常（漏偵測或誤偵測），序列在此斷開
    step_time_hi: float = 1.5     # 步時 > 1.5 x 中位數 → 中間漏掉步伐，序列在此斷開
    zone_margin_m: float = 0.0    # 量測區 = 校正長方形向內/外延伸的距離
    lateral_tol_m: float = 1.0    # 左右超出長方形多少仍接受
    cam_height_m: float = 1.10    # 鏡頭離地高度；用來修正「腳跟點不在地面上」造成的放大
    landmark_height_m: float = 0.03  # 腳跟關鍵點離地高度（估計值，待驗證研究校正）
    event_method: str = "interfoot"  # "interfoot"（雙腳前後距離，預設）或 "speed"（舊的腳速門檻法）
    contact_prom_m: float = 0.08     # 雙腳前後距離峰值的最小突出量（公尺）
    dwell_r_m: float = 0.06          # 只用於標註影片顯示站立範圍
    min_advance_m: float = 0.10      # 相鄰兩次著地的最小前進距離
    edge_margin_frac: float = 0.015  # 腳尖距畫面邊緣小於此比例 → 該次著地不採用
    step_len_lo: float = 0.6        # 步長 < 0.6 x 中位數或 > 1.5 x → 序列斷開（位置異常）
    step_len_hi: float = 1.5
    contact_level: float = 0.8       # 著地時間 = 前後距離上升到振幅 80% 的時刻
    lateral_point: str = "ankle"     # 著地點左右位置用哪個關鍵點："ankle"（預設，2026-10 小驗證較準）或 "heel"
    ankle_height_m: float = 0.09     # 腳踝關鍵點離地高度（穿鞋，含鞋底；估計值）


BAREFOOT = {"ankle_height_m": 0.07, "landmark_height_m": 0.02}  # 赤腳：沒有鞋底，關鍵點較低（估計值）


def make_params(cam_height_m: float, lateral_point: str = "ankle", barefoot: bool = False) -> "Params":
    """依鞋子狀態建立參數。赤腳只影響高度修正（左右距離約 +2%）。"""
    kw = dict(BAREFOOT) if barefoot else {}
    return Params(cam_height_m=cam_height_m, lateral_point=lateral_point, **kw)


# ---------------------------------------------------------------- 前處理
def _interp_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    x = x.copy()
    n = len(x)
    isnan = np.isnan(x)
    if isnan.all() or not isnan.any():
        return x
    idx = np.arange(n)
    good = ~isnan
    filled = np.interp(idx, idx[good], x[good])
    # 只填「兩側都有資料且長度不超過 max_gap」的缺口
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = filled[i:j]
            i = j
        else:
            i += 1
    return x


def _lowpass_runs(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """對每段連續非缺值的資料做零相位濾波。"""
    out = x.copy()
    if fc <= 0 or fc >= fs / 2:
        return out
    b, a = butter(4, fc / (fs / 2))
    padlen = 3 * max(len(a), len(b))
    isnan = np.isnan(x)
    i, n = 0, len(x)
    while i < n:
        if isnan[i]:
            i += 1
            continue
        j = i
        while j < n and not isnan[j]:
            j += 1
        if j - i > padlen:
            out[i:j] = filtfilt(b, a, x[i:j])
        i = j
    return out


def clean_landmarks(lm: dict, prm: Params) -> np.ndarray:
    xy = lm["xy"].copy()
    xy[lm["vis"] < prm.vis_min] = np.nan
    fs = lm["fps"]
    max_gap = int(round(prm.max_gap_s * fs))
    for k in range(xy.shape[1]):
        for d in range(2):
            s = _interp_gaps(xy[:, k, d], max_gap)
            xy[:, k, d] = _lowpass_runs(s, fs, prm.cutoff_hz)
    return xy


# ---------------------------------------------------------------- 事件偵測
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """布林陣列中 True 的連續區段 [start, end)。"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


@dataclass
class Stance:
    side: str
    i_on: int      # 腳跟著地影格
    i_off: int     # 腳尖離地影格（不含）
    heel: np.ndarray   # 著地點（地板公尺）
    toe: np.ndarray
    complete: bool     # 前後都有擺盪期（不是被影片開頭/結尾切掉）


def detect_stances(floor: dict, fs: float, prm: Params) -> tuple[list[Stance], dict]:
    speeds: dict = {}
    for s in "LR":
        c = (floor[s]["heel"] + floor[s]["toe"]) / 2
        if len(c) < 3:
            speeds[s] = np.full(len(c), np.nan)
            continue
        speeds[s] = np.linalg.norm(np.gradient(c, axis=0), axis=1) * fs
    # 門檻只用「有在走」的影格計算，避免影片頭尾站著不動把門檻拉低
    vmax = np.fmax(np.nan_to_num(speeds["L"], nan=0), np.nan_to_num(speeds["R"], nan=0))
    active = vmax > 0.3
    pool = np.concatenate([speeds[s][active & ~np.isnan(speeds[s])] for s in "LR"])
    mean_v = float(np.mean(pool)) if len(pool) else 0.0
    thr = max(prm.stance_min_thr, prm.stance_frac * mean_v)
    min_st = max(1, int(round(prm.min_stance_s * fs)))
    min_sw = max(1, int(round(prm.min_swing_s * fs)))

    stances: list[Stance] = []
    for s in "LR":
        v = speeds[s]
        valid = ~np.isnan(v)
        runs = _runs(valid & (v < thr))
        # (1) 被極短擺盪隔開的站立期合併
        merged: list[list[int]] = []
        for a, b in runs:
            if merged and a - merged[-1][1] < min_sw and valid[merged[-1][1]:a].all():
                merged[-1][1] = b
            else:
                merged.append([a, b])
        # (2) 同一隻腳、著地點幾乎同一位置的兩段 → 同一個腳印被雜訊切斷，合併（不論間隔多久）
        def heel_of(a, b):
            k = max(a + 1, a + int(0.5 * (b - a)))
            return np.nanmedian(floor[s]["heel"][a:k], axis=0)
        spots: list[list] = []
        for a, b in merged:
            h = heel_of(a, b)
            if spots and not np.isnan(h).any() and not np.isnan(spots[-1][2]).any() \
                    and valid[spots[-1][1]:a].all() \
                    and np.linalg.norm(h - spots[-1][2]) < prm.same_spot_m:
                spots[-1][1] = b
            else:
                spots.append([a, b, h])
        for a, b, heel in spots:
            if b - a < min_st:
                continue
            # 完整 = 前後都看得到一段擺盪期（不是被影片頭尾或遮擋切斷的站立期）
            complete = (a >= min_sw and b + min_sw <= len(v)
                        and valid[a - min_sw:a].all() and valid[b:b + min_sw].all())
            mid0, mid1 = a + int(0.3 * (b - a)), a + int(0.7 * (b - a)) + 1
            toe = np.nanmedian(floor[s]["toe"][mid0:mid1], axis=0)
            heel_mid = np.nanmedian(floor[s]["heel"][mid0:mid1], axis=0)
            st = Stance(s, a, b, heel, toe, bool(complete))
            st.heel_mid = heel_mid  # type: ignore[attr-defined]
            st.lat_mid = heel_mid  # type: ignore[attr-defined]
            stances.append(st)
    stances.sort(key=lambda z: z.i_on)
    return stances, {"speed": speeds, "threshold": thr, "mean_foot_speed": mean_v}


def build_sequence(stances: list[Stance], usable, fs: float, prm: Params) -> list[list[Stance]]:
    """切出所有「左右交替、步時合理、中間沒有被排除的站立期」的連續片段（至少 3 次著地）。"""
    # 參考步時：所有相鄰、左右交替、都可用的著地之間的時間中位數
    dts = [(b.i_on - a.i_on) / fs for a, b in zip(stances, stances[1:])
           if usable(a) and usable(b) and a.side != b.side]
    med = float(np.median(dts)) if dts else np.nan
    dls = [float(np.linalg.norm(b.heel - a.heel)) for a, b in zip(stances, stances[1:])
           if usable(a) and usable(b) and a.side != b.side]
    med_l = float(np.median(dls)) if dls else np.nan
    seqs, cur = [], []
    for st in stances:
        ok = usable(st)
        if not ok:
            if cur:
                seqs.append(cur)
            cur = []
            continue
        if cur:
            prev = cur[-1]
            dt = (st.i_on - prev.i_on) / fs
            plausible = np.isnan(med) or (prm.step_time_lo * med <= dt <= prm.step_time_hi * med)
            dl = float(np.linalg.norm(st.heel - prev.heel))
            if not np.isnan(med_l) and not (prm.step_len_lo * med_l <= dl <= prm.step_len_hi * med_l):
                plausible = False
            if st.side == prev.side or not plausible:
                seqs.append(cur)
                cur = []
        cur.append(st)
    if cur:
        seqs.append(cur)
    return [q for q in seqs if len(q) >= 3]


def _progression_dir(floor: dict) -> np.ndarray:
    P_ = np.concatenate([floor["L"]["heel"], floor["R"]["heel"]])
    P_ = P_[~np.isnan(P_).any(axis=1)]
    if len(P_) < 10:
        return np.array([0.0, -1.0])
    _, _, vt = np.linalg.svd(P_ - P_.mean(axis=0), full_matrices=False)
    u = vt[0]
    for s in "LR":
        h = floor[s]["heel"]
        idx = np.nonzero(~np.isnan(h).any(axis=1))[0]
        if len(idx) > 1:
            if np.dot(h[idx[-1]] - h[idx[0]], u) < 0:
                u = -u
            break
    return u / np.linalg.norm(u)


def detect_contacts(floor: dict, fs: float, prm: Params, toe_px: dict | None = None,
                    size: tuple | None = None) -> tuple[list[Stance], dict]:
    """雙腳前後距離法（Zeni 2008 的改良版，適用正面拍攝）。

    著地時間：A 腳著地時「A 腳跟 − B 腳尖」的前後距離最大；取其上升段越過「谷底 + 80% 振幅」的時刻。
    著地位置：前後方向用腳尖（著地後 0.10–0.35 秒），左右方向用腳跟（0.05–0.25 秒）。
      正面拍攝時腳跟被腳背擋住，模型猜的腳跟前後位置有偏差且左右腳不同；腳尖則清楚可見。
    為什麼不用腳速：腳跟離地投影到地板會像在往後退，腳速會被這個假象干擾。
    站立期與雙腳支撐期在正面拍攝下無法可靠量測；i_off 只作顯示用途。
    以下情況該次著地標為不完整（不採用）：位置視窗有效資料 < 70%、峰值 0.1 秒內有缺值、
    腳尖缺值、腳尖靠近畫面邊緣（超出畫面時模型仍會硬猜位置）。
    """
    from scipy.signal import find_peaks
    T = len(floor["L"]["heel"])
    u = _progression_dir(floor)
    n_ = np.array([-u[1], u[0]])
    p = {s: {k: floor[s][k] @ u for k in ("heel", "toe")} for s in "LR"}
    w = {k: int(round(v * fs)) for k, v in {"05": 0.05, "10": 0.10, "25": 0.25, "35": 0.35, "nan": 0.10}.items()}
    edge = None
    if toe_px is not None and size is not None:
        edge = (prm.edge_margin_frac * size[0], prm.edge_margin_frac * size[1], size[0], size[1])

    def frac_valid(arr):
        return float(np.mean(~np.isnan(arr).any(axis=1))) if len(arr) else 0.0

    stances: list[Stance] = []
    dcurves = {}
    for A, B in (("L", "R"), ("R", "L")):
        d = p[A]["heel"] - p[B]["toe"]
        dcurves[A] = d
        if np.isnan(d).all():
            continue
        # 缺值以「前一個有效值」填補（不製造假的高低差），再找峰值
        dd = d.copy()
        last = np.nanmin(d)
        for i in range(T):
            if np.isnan(dd[i]):
                dd[i] = last
            else:
                last = dd[i]
        pk, _ = find_peaks(dd, distance=max(1, int(0.35 * fs)), prominence=prm.contact_prom_m)
        for k in pk:
            if np.isnan(d[k]):
                continue
            back = max(0, k - int(1.2 * fs))
            seg = d[back:k + 1]
            j0 = back + int(np.nanargmin(seg))
            lvl = d[j0] + prm.contact_level * (d[k] - d[j0])
            a = k
            while a > j0 and not np.isnan(d[a - 1]) and d[a - 1] >= lvl:
                a -= 1
            if a > 0 and not np.isnan(d[a - 1]) and d[a] != d[a - 1]:
                t_frac = (a - 1) + (lvl - d[a - 1]) / (d[a] - d[a - 1])
            else:
                t_frac = float(a)
            k_peak, k = k, a
            win_t = slice(min(T - 1, k + w["10"]), min(T, k + w["35"]))
            win_h = slice(min(T - 1, k + w["05"]), min(T, k + w["25"]))
            toe_seg, heel_seg = floor[A]["toe"][win_t], floor[A]["heel"][win_h]
            lat_seg = floor[A].get(prm.lateral_point, floor[A]["heel"])[win_h]
            toe = np.nanmedian(toe_seg, axis=0) if frac_valid(toe_seg) > 0 else np.full(2, np.nan)
            heel_mid = np.nanmedian(heel_seg, axis=0) if frac_valid(heel_seg) > 0 else np.full(2, np.nan)
            lat_mid = np.nanmedian(lat_seg, axis=0) if frac_valid(lat_seg) > 0 else np.full(2, np.nan)
            ok_pos = (frac_valid(toe_seg) >= 0.7 and frac_valid(heel_seg) >= 0.7 and frac_valid(lat_seg) >= 0.7
                      and k + w["35"] <= T and not np.isnan(toe).any() and not np.isnan(heel_mid).any()
                      and not np.isnan(lat_mid).any())
            near = slice(max(0, k - w["nan"]), min(T, k_peak + w["nan"] + 1))
            ok_gap = not np.isnan(d[near]).any()
            ok_edge = True
            if edge is not None:
                tp = toe_px[A][win_t]
                tp = tp[~np.isnan(tp).any(axis=1)]
                if len(tp):
                    mx, my, W, Hh = edge
                    ok_edge = bool(np.all((tp[:, 0] > mx) & (tp[:, 0] < W - mx) & (tp[:, 1] > my) & (tp[:, 1] < Hh - my)))
            m = int(round(0.2 * fs))
            ok_ctx = (k - m >= 0 and k + m <= T
                      and not np.isnan(p[A]["heel"][max(0, k - m):k + m]).any()
                      and not np.isnan(p[B]["toe"][max(0, k - m):k + m]).any())
            complete = bool(ok_pos and ok_gap and ok_edge and ok_ctx)
            # 顯示用的站立範圍：腳尖停在原地（±dwell_r）的最後一格
            b = k
            T0 = toe @ u if not np.isnan(toe).any() else np.nan
            while b < T - 1 and not np.isnan(p[A]["toe"][b + 1]) and abs(p[A]["toe"][b + 1] - T0) <= prm.dwell_r_m:
                b += 1
            st = Stance(A, int(k), int(max(b, k + 1)), heel_mid.copy(), toe, complete)
            st.heel_mid = heel_mid  # type: ignore[attr-defined]
            st.lat_mid = lat_mid  # type: ignore[attr-defined]
            st.reject = "、".join(n for n, ok in (("著地後腳部點不清楚", ok_pos), ("著地瞬間腳部點不清楚", ok_gap),  # type: ignore[attr-defined]
                                                  ("腳尖靠近畫面邊緣", ok_edge), ("著地前後腳部點不清楚，常是被另一腳擋住", ok_ctx)) if not ok)
            st.t_on = t_frac / fs  # type: ignore[attr-defined]
            st.edge_ok = ok_edge  # type: ignore[attr-defined]
            stances.append(st)
    # 組合著地點：前後 = 腳尖位置 − 全體中位足長，左右 = 腳跟（或腳踝）位置；缺值 → 位置 NaN
    foot_len = [np.dot(st.toe - st.heel_mid, u) for st in stances  # type: ignore[attr-defined]
                if st.complete]
    Lf = float(np.median(foot_len)) if foot_len else 0.0
    for st in stances:
        if np.isnan(st.toe).any() or np.isnan(st.heel_mid).any():  # type: ignore[attr-defined]
            st.heel = np.full(2, np.nan)
            st.complete = False
        else:
            lat = st.lat_mid if not np.isnan(st.lat_mid).any() else st.heel_mid  # type: ignore[attr-defined]
            st.heel = (np.dot(st.toe, u) - Lf) * u + np.dot(lat, n_) * n_  # type: ignore[attr-defined]
    # 同一腳、同一位置的重複峰值只留第一個
    stances.sort(key=lambda z: z.i_on)
    kept: list[Stance] = []
    for st in stances:
        dup = (not np.isnan(st.heel).any()) and any(
            o.side == st.side and not np.isnan(o.heel).any() and np.linalg.norm(o.heel - st.heel) < prm.same_spot_m
            for o in kept[-4:])
        if not dup:
            kept.append(st)
    speeds = {s: np.linalg.norm(np.gradient((floor[s]["heel"] + floor[s]["toe"]) / 2, axis=0), axis=1) * fs
              if T >= 3 else np.full(T, np.nan) for s in "LR"}
    return kept, {"speed": speeds, "threshold": np.nan, "mean_foot_speed": np.nan,
                  "interfoot": dcurves, "u": u}


# ---------------------------------------------------------------- 指標
def _ton(st, fs):
    """著地時間（秒）；新方法有次影格精度。"""
    return getattr(st, "t_on", st.i_on / fs)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cv(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.mean(x) == 0:
        return np.nan
    return float(np.std(x, ddof=1) / np.mean(x) * 100)


def _sd(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan


def _mean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else np.nan


@dataclass
class GaitResult:
    summary: dict
    steps: list[dict]
    stances: list[Stance]
    floor: dict
    xy: np.ndarray
    debug: dict = field(default_factory=dict)


def analyze(lm: dict, H: np.ndarray, floor_w: float, floor_l: float,
            prm: Params | None = None) -> GaitResult:
    prm = prm or Params()
    fs = lm["fps"]
    t = lm["t"]
    if len(t) < fs * 1.0:
        raise ValueError("影片太短（少於 1 秒）或讀不到影格，無法分析")
    xy = clean_landmarks(lm, prm)

    floor = {}
    for s, (ih, it, ia) in SIDES.items():
        floor[s] = {"heel": to_floor(H, xy[:, ih]), "toe": to_floor(H, xy[:, it]), "ankle": to_floor(H, xy[:, ia])}

    if prm.event_method == "speed":
        stances, dbg = detect_stances(floor, fs, prm)
    else:
        toe_px = {s: xy[:, SIDES[s][1]] for s in "LR"}
        stances, dbg = detect_contacts(floor, fs, prm, toe_px, lm.get("size"))
    # 腳跟點約在地面上方 h，經地板投影後所有距離會放大 Hc/(Hc-h)，這裡乘回來
    ks = 1.0
    if prm.cam_height_m > prm.landmark_height_m > 0:
        ks = (prm.cam_height_m - prm.landmark_height_m) / prm.cam_height_m
    # 左右方向用的關鍵點高度不同（腳踝約 9 cm），步寬另外修正
    h_lat = prm.ankle_height_m if prm.lateral_point == "ankle" else prm.landmark_height_m
    ks_lat = (prm.cam_height_m - h_lat) / prm.cam_height_m if prm.cam_height_m > h_lat > 0 else 1.0

    # 量測區：只採用落在校正長方形內的著地點
    y0, y1 = -prm.zone_margin_m, floor_l + prm.zone_margin_m
    x0, x1 = -prm.lateral_tol_m, floor_w + prm.lateral_tol_m

    def in_zone(st):
        x, y = st.heel
        return (not np.isnan(x)) and y0 <= y <= y1 and x0 <= x <= x1

    # 每一步的著地點必須比前一次「可用」著地點更往前（至少 min_advance_m）；否則是雜訊造成的假著地
    u_ev = dbg.get("u", _progression_dir(floor))
    clean: list[Stance] = []
    dropped: list[tuple] = []
    for st in stances:
        if in_zone(st) and st.complete:
            prev = next((c for c in reversed(clean) if in_zone(c) and c.complete), None)
            if prev is not None and np.dot(st.heel - prev.heel, u_ev) < prm.min_advance_m:
                dropped.append((st, "比前一步前進不到 10 cm（重複偵測）"))
                continue
        clean.append(st)
    stances = clean
    runs = build_sequence(stances, lambda st: in_zone(st) and st.complete, fs, prm)
    allc = [st for q in runs for st in q]
    # 量測區內但沒被採用的著地，記下原因（報告與 steps 檢查用）
    used_ids = {id(st) for st in allc}
    for st in stances:
        if id(st) in used_ids:
            continue
        x, y = st.heel if not np.isnan(st.heel).any() else (np.nan, np.nan)
        y_ok = (not np.isnan(y)) and y0 <= y <= y1
        if not y_ok:
            continue
        why = getattr(st, "reject", "") if not st.complete else "步時或步長與前後步不合，序列在此斷開"
        dropped.append((st, why or "不完整"))
    dbg["rejected"] = sorted(({"t_s": float(t[0] + _ton(st, fs)), "side": st.side,
                               "y_m": float(st.heel[1]) if not np.isnan(st.heel).any() else None, "reason": why}
                              for st, why in dropped), key=lambda d: d["t_s"])

    # 整體行進方向
    # 整體行進方向：所有著地點的主軸（不受左右腳交錯影響），方向依時間先後
    if len(allc) >= 2:
        P0 = np.array([s.heel for s in allc])
        _, _, vt = np.linalg.svd(P0 - P0.mean(axis=0))
        u_glob = _unit(vt[0])
        if np.dot(P0[-1] - P0[0], u_glob) < 0:
            u_glob = -u_glob
    else:
        u_glob = np.array([0.0, -1.0])

    steps = []
    n_run = 0
    for seq in runs:
        n_run += 1
        for k, st in enumerate(seq):
            row = {"n": len(steps) + 1, "segment": n_run, "side": st.side, "t_contact_s": float(t[0] + _ton(st, fs)),
                   "t_toeoff_s": float(t[min(st.i_off, len(t) - 1)]),
                   "heel_x_m": float(st.heel[0]), "heel_y_m": float(st.heel[1]),
                   "step_width_cm": np.nan, "step_length_cm": np.nan, "stride_length_cm": np.nan,
                   "step_time_s": np.nan, "stride_time_s": np.nan, "stance_time_s": float((st.i_off - st.i_on) / fs) if prm.event_method == "speed" else np.nan,
                   "foot_angle_deg": np.nan, "crossover": False}
            # 前一個與後一個（對側腳）
            if 0 < k < len(seq) - 1:
                a, b = seq[k - 1].heel, seq[k + 1].heel
                u = _unit(b - a)
                n = np.array([-u[1], u[0]])  # 行進方向的左側
                sd = float(np.dot(st.heel - a, n))
                # 受試者左腳應在對側連線的左邊；方向相反 = 交叉步（剪刀步態）或左右腳標記錯誤
                expect = 1.0 if st.side == "L" else -1.0
                row["crossover"] = bool(sd * expect < 0)
                row["step_width_cm"] = float(abs(sd) * 100 * ks_lat)
            if k > 0:
                u = u_glob
                if 0 < k < len(seq) - 1:
                    u = _unit(seq[k + 1].heel - seq[k - 1].heel)
                row["step_length_cm"] = float(np.dot(st.heel - seq[k - 1].heel, u) * 100 * ks)
                row["step_time_s"] = float(_ton(st, fs) - _ton(seq[k - 1], fs))
            if k > 1:
                row["stride_time_s"] = float(_ton(st, fs) - _ton(seq[k - 2], fs))
                row["stride_length_cm"] = float(np.linalg.norm(st.heel - seq[k - 2].heel) * 100 * ks)  # 同一腳的步幅
            # 足部行進角（外八為正）
            # 以該腳自己的步幅方向為基準（GAITRite 作法），避免左右腳交錯造成的偏差
            a_i, b_i = (k - 2 if k >= 2 else k), (k + 2 if k + 2 < len(seq) else k)
            u_f = _unit(seq[b_i].heel - seq[a_i].heel) if b_i != a_i else u_glob
            v = st.toe - st.heel_mid  # type: ignore[attr-defined]
            right = np.array([u_f[1], -u_f[0]])
            side_out = right if st.side == "R" else -right
            if np.linalg.norm(v) > 0.03:
                row["foot_angle_deg"] = float(np.degrees(np.arctan2(np.dot(v, side_out), np.dot(v, u_f))))
            steps.append(row)

    # 雙腳支撐期：在每個完整步態週期內兩腳同時站立的時間比例
    T = len(t)
    stance_mask = {"L": np.zeros(T, bool), "R": np.zeros(T, bool)}
    for st in stances:
        stance_mask[st.side][st.i_on:st.i_off] = True
    ds_pct, stance_pct = [], []
    for seq in runs:
        for k in range(len(seq) - 2):
            a, b = seq[k].i_on, seq[k + 2].i_on
            both = stance_mask["L"][a:b] & stance_mask["R"][a:b]
            ds_pct.append(both.mean() * 100)
            stance_pct.append((seq[k].i_off - seq[k].i_on) / (b - a) * 100)

    # 不需校正的備援指標：腳跟橫向距離 / 骨盆寬（像素比）
    ratios = []
    for st in allc:
        i = st.i_on
        hip_w = abs(xy[i, P.L_HIP, 0] - xy[i, P.R_HIP, 0])
        heel_d = abs(xy[i, P.L_HEEL, 0] - xy[i, P.R_HEEL, 0])
        if hip_w > 1 and not np.isnan(heel_d):
            ratios.append(heel_d / hip_w)

    # 骨盆左右晃動（相對骨盆寬）
    sway = np.nan
    if runs:
        seq = max(runs, key=len)
        a, b = seq[0].i_on, seq[-1].i_off
        mid = (xy[a:b, P.L_HIP, 0] + xy[a:b, P.R_HIP, 0]) / 2
        hipw = np.abs(xy[a:b, P.L_HIP, 0] - xy[a:b, P.R_HIP, 0])
        # 去掉朝鏡頭走時的透視漂移：以線性趨勢為基準
        ok = ~np.isnan(mid) & ~np.isnan(hipw)
        if ok.sum() > 10:
            tt = np.arange(len(mid))[ok]
            resid = mid[ok] - np.polyval(np.polyfit(tt, mid[ok], 1), tt)
            sway = float((resid.max() - resid.min()) / np.nanmedian(hipw[ok]))

    sw = [r["step_width_cm"] for r in steps]
    sl = [r["step_length_cm"] for r in steps]
    stt = [r["step_time_s"] for r in steps]
    sdt = [r["stride_time_s"] for r in steps]
    fa_l = [r["foot_angle_deg"] for r in steps if r["side"] == "L"]
    fa_r = [r["foot_angle_deg"] for r in steps if r["side"] == "R"]

    speed = np.nan
    if runs:
        dist = sum(np.dot(q[-1].heel - q[0].heel, u_glob) for q in runs)
        dur = sum(_ton(q[-1], fs) - _ton(q[0], fs) for q in runs)
        speed = dist / dur * 100 * ks if dur > 0 else np.nan

    valid_frames = (~np.isnan(lm["xy"][:, P.L_HEEL, 0])).mean() * 100

    # 左右標記對調檢查：朝鏡頭走時受試者左髖應在畫面右側（x 較大），背對鏡頭時相反
    swap_pct = np.nan
    if allc:
        a, b = allc[0].i_on, allc[-1].i_off
        dx = xy[a:b, P.L_HIP, 0] - xy[a:b, P.R_HIP, 0]
        dx = dx[~np.isnan(dx)]
        if len(dx):
            toward = u_glob[1] < 0
            swap_pct = float(np.mean(dx < 0 if toward else dx > 0) * 100)
    summary = {
        "n_contacts_used": len(allc),
        "n_segments": len(runs),
        "n_step_width_values": int(np.sum(~np.isnan(sw))),
        "gait_speed_cm_s": speed,
        "cadence_steps_min": 60.0 / _mean(stt) if not np.isnan(_mean(stt)) else np.nan,
        "step_width_mean_cm": _mean(sw),
        "step_width_sd_cm": _sd(sw),
        "step_width_cv_pct": _cv(sw),
        "step_length_mean_cm": _mean(sl),
        "stride_length_mean_cm": _mean([r["stride_length_cm"] for r in steps]),
        "step_length_cv_pct": _cv(sl),
        "step_time_mean_s": _mean(stt),
        "step_time_cv_pct": _cv(stt),
        "stride_time_cv_pct": _cv(sdt),
        "double_support_pct": _mean(ds_pct) if prm.event_method == "speed" else np.nan,
        "stance_pct": _mean(stance_pct) if prm.event_method == "speed" else np.nan,
        "foot_angle_L_deg": _mean(fa_l),
        "foot_angle_R_deg": _mean(fa_r),
        "heel_to_hip_width_ratio": float(np.median(ratios)) if ratios else np.nan,
        "pelvic_sway_hipwidths": sway,
        "pose_detected_frames_pct": float(valid_frames),
        "lr_swap_frames_pct": swap_pct,
        "n_crossover_steps": int(sum(1 for r in steps if r.get("crossover"))),
        # 起步點檢查：量測區遠端之外沒有任何著地 → 受試者在量測區內才起步，前幾步還在加速
        "started_inside_zone": bool(stances) and (
            not any((not np.isnan(st.heel[1])) and st.heel[1] > floor_l for st in stances) if u_glob[1] < 0
            else not any((not np.isnan(st.heel[1])) and st.heel[1] < 0 for st in stances)),
        "stance_speed_threshold_m_s": dbg["threshold"],
        "height_correction_factor": ks,
    }
    res = GaitResult(summary, steps, stances, floor, xy, dbg)
    res.debug_t = t  # type: ignore[attr-defined]
    return res


def aggregate(results: list[GaitResult]) -> dict:
    """多趟合併：所有步合在一起重算平均與變異。"""
    steps = [r for res in results for r in res.steps]
    sw = [r["step_width_cm"] for r in steps]
    sl = [r["step_length_cm"] for r in steps]
    stt = [r["step_time_s"] for r in steps]
    sdt = [r["stride_time_s"] for r in steps]
    pick = lambda key: _mean([res.summary[key] for res in results])
    return {
        "n_trials": len(results),
        "n_contacts_used": sum(res.summary["n_contacts_used"] for res in results),
        "n_step_width_values": int(np.sum(~np.isnan(sw))),
        "gait_speed_cm_s": pick("gait_speed_cm_s"),
        "cadence_steps_min": 60.0 / _mean(stt) if not np.isnan(_mean(stt)) else np.nan,
        "step_width_mean_cm": _mean(sw),
        "step_width_sd_cm": _sd(sw),
        "step_width_cv_pct": _cv(sw),
        "step_length_mean_cm": _mean(sl),
        "stride_length_mean_cm": _mean([r["stride_length_cm"] for r in steps]),
        "step_length_cv_pct": _cv(sl),
        "step_time_mean_s": _mean(stt),
        "step_time_cv_pct": _cv(stt),
        "stride_time_cv_pct": _cv(sdt),
        "double_support_pct": pick("double_support_pct"),
        "stance_pct": pick("stance_pct"),
        "foot_angle_L_deg": _mean([r["foot_angle_deg"] for r in steps if r["side"] == "L"]),
        "foot_angle_R_deg": _mean([r["foot_angle_deg"] for r in steps if r["side"] == "R"]),
        "heel_to_hip_width_ratio": pick("heel_to_hip_width_ratio"),
        "pelvic_sway_hipwidths": pick("pelvic_sway_hipwidths"),
    }
