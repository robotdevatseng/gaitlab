"""輸出：HTML 報告（圖表內嵌、單一檔案）、CSV、JSON、標註影片。"""
from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import io
import json
import math

import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import pose as P  # noqa: E402
from .calib import to_image  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans CJK TC",
                                   "Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_L, C_R = "#2a6fdb", "#e8743b"   # 左腳藍、右腳橘

METRICS = [
    # key, 中文, 單位, 小數位, 說明
    ("step_width_mean_cm", "步寬（平均）", "cm", 1, "對側兩次著地點連線到這一步的垂直距離（GAITRite 定義）"),
    ("step_width_sd_cm", "步寬變異（SD）", "cm", 1, "每一步步寬的標準差；本體感覺喪失時上升"),
    ("step_width_cv_pct", "步寬變異（CV）", "%", 1, ""),
    ("gait_speed_cm_s", "步速", "cm/s", 1, "Makino 2023：DCM 嚴重度分界約 100 cm/s"),
    ("cadence_steps_min", "步頻", "步/分", 1, ""),
    ("stride_length_mean_cm", "步幅（平均）", "cm", 1, "同一腳兩次著地的距離；正面拍攝最可靠的長度指標"),
    ("step_length_mean_cm", "步長（平均）", "cm", 1, "只看平均；單步與左右差在正面拍攝下不可解讀"),
    ("stride_time_cv_pct", "步幅時間變異（CV）", "%", 1, "時間變異以此為主；待側面與步道驗證"),
    ("step_time_cv_pct", "步時變異（CV）", "%", 1, "參考用：左右腳著地時間有系統差，會偏高"),
    ("double_support_pct", "雙腳支撐期", "% 週期", 1, "正面拍攝無法可靠量測，不提供；需加側面拍攝"),
    ("stance_pct", "站立期", "% 週期", 1, "正面拍攝無法可靠量測，不提供"),
    ("pelvic_sway_hipwidths", "骨盆左右晃動", "骨盆寬", 2, "去除趨勢後的晃動範圍；未驗證"),
]
# 足部行進角、腳跟距／骨盆寬：小驗證顯示正面拍攝量不到（三人足角都約 0°），只留在 summary.json


def _tc(res, st):
    """與 gait.analyze 相同定義的著地時間（秒），用來把站立期對應到逐步資料。"""
    t = res.debug_t
    fs = 1.0 / (t[1] - t[0]) if len(t) > 1 else 30.0
    return round(float(t[0] + getattr(st, "t_on", st.i_on / fs)), 4)


def _fmt(v, nd):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- 圖
def fig_footprints(results, floor_w, floor_l) -> str:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 0.6, 7.2), squeeze=False)
    for ax, (name, res) in zip(axes[0], results):
        ax.add_patch(plt.Rectangle((0, 0), floor_w, floor_l, fill=False, ls="--", color="#999"))
        for st in res.stances:
            col = C_L if st.side == "L" else C_R
            used = any(abs(r["t_contact_s"] - _tc(res, st)) < 1e-3 for r in res.steps)
            alpha = 1.0 if used else 0.25
            ax.plot(*st.heel, "o", color=col, alpha=alpha, ms=7)
            if not np.isnan(st.toe).any():
                ax.plot([st.heel_mid[0], st.toe[0]], [st.heel_mid[1], st.toe[1]], "-", color=col, alpha=alpha, lw=2)
        for r in res.steps:
            if not math.isnan(r["step_width_cm"]):
                ax.annotate(f'{r["step_width_cm"]:.0f}', (r["heel_x_m"], r["heel_y_m"]),
                            xytext=(8 if r["side"] == "L" else -8, 0), textcoords="offset points",
                            ha="left" if r["side"] == "L" else "right", va="center", fontsize=8, color="#333")
        ax.set_xlim(-0.6, floor_w + 0.6)
        ax.set_ylim(-0.5, floor_l + 0.5)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("左右（m，畫面右 = +）")
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel("距量測區近端（m）")
    fig.suptitle("腳印圖（藍 = 左腳、橘 = 右腳；數字 = 步寬 cm；淡色 = 量測區外未採用）", fontsize=10)
    return _png(fig)


def fig_stepwidth(results) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    x = 0
    for name, res in results:
        for r in res.steps:
            if not math.isnan(r["step_width_cm"]):
                ax.bar(x, r["step_width_cm"], color=C_L if r["side"] == "L" else C_R)
                x += 1
        x += 1
    vals = [r["step_width_cm"] for _, res in results for r in res.steps if not math.isnan(r["step_width_cm"])]
    if vals:
        ax.axhline(np.mean(vals), color="#333", lw=1, ls="--")
        ax.text(-0.5, np.mean(vals) + 0.4, f"平均 {np.mean(vals):.1f} cm", ha="left", va="bottom", fontsize=9, color="#333")
    ax.set_ylabel("步寬（cm）")
    ax.set_xticks([])
    ax.set_title("逐步步寬（各趟之間留空）", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def fig_speed(name, res) -> str:
    t = res.debug_t
    fig, ax = plt.subplots(figsize=(8, 2.6))
    if "interfoot" in res.debug:
        for s, col, lab in (("L", C_L, "左腳跟領先右腳尖"), ("R", C_R, "右腳跟領先左腳尖")):
            ax.plot(t, res.debug["interfoot"][s], color=col, lw=1.2, label=lab)
        used = {round(r["t_contact_s"], 4) for r in res.steps}
        for st in res.stances:
            col = C_L if st.side == "L" else C_R
            tc = _tc(res, st)
            ax.axvline(tc, color=col, lw=1.0 if tc in used else 0.6, ls="-" if tc in used else ":")
        ax.set_ylabel("m")
        ax.set_title(f"品質檢查：雙腳前後距離與偵測到的著地（實線 = 採用；{name}）", fontsize=10)
    else:
        for s, col in (("L", C_L), ("R", C_R)):
            ax.plot(t, res.debug["speed"][s], color=col, lw=1.2, label=f"{'左' if s == 'L' else '右'}腳速度")
        ax.set_ylabel("m/s")
        ax.set_title(f"品質檢查：腳速與站立期（{name}）", fontsize=10)
    ax.set_xlabel("時間（s）")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


# ---------------------------------------------------------------- 檔案
def write_steps_csv(path, results):
    rows = []
    for name, res in results:
        for r in res.steps:
            rows.append({"trial": name, **r})
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_json(path, meta, agg, results):
    def clean(d):
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}
    out = {"meta": meta, "overall": clean(agg),
           "trials": {name: clean(res.summary) for name, res in results}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def quality_flags(agg, results, meta=None) -> list[str]:
    flags = []
    geo = (meta or {}).get("calib_check") or {}
    hset = (meta or {}).get("cam_height_m")
    if geo and hset:
        dh = geo["cam_height_m"] - float(hset)
        if abs(dh) > 0.08:
            flags.append(f"地板校正與佈置不一致：從四個標記反推的鏡頭高度是 {geo['cam_height_m']:.2f} m，設定是 {float(hset):.2f} m。"
                         "請確認四張標記的左右間距、前後間距是否是標記中心到中心，以及鏡頭高度是否量對；不一致時步寬會等比例偏差。")
    # 2026-11 五位同仁的校正誤差多在 0.3–1.7 像素，對結果影響 < 0.2 cm；超過 3 像素才值得重點
    if geo and geo.get("resid_px", 0) > 3.0:
        flags.append(f"地板校正的四個點彼此不太吻合（誤差 {geo['resid_px']:.1f} 像素，正常 < 2）：可能有一個標記點點偏了。"
                     "請刪掉 calib.json 重新校正，點選時對準標記正中央。")
    if agg.get("n_step_width_values", 0) < 6:
        flags.append(f"可計算步寬的步數只有 {agg.get('n_step_width_values', 0)} 步（建議至少 6 步，請多走幾趟或加長量測區）。")
    for name, res in results:
        vs = res.summary.get("video_size_px")
        if vs and min(vs) not in (720, 1080, 1440, 2160):
            flags.append(f"{name}：影片解析度是 {vs[0]}×{vs[1]}，不是相機原始規格（例如 2160×3840），"
                         "表示傳檔或剪輯時被重新編碼、裁切過，畫質會下降。請用 AirDrop 或接線傳「原始檔」，不要經過 LINE、iMessage 或剪輯 App。")
        # 只看受試者在量測區內的影格；走進畫面前、走出畫面後偵測不到是正常的
        if res.summary.get("feet_visible_in_zone_pct", 100) < 80:
            flags.append(f"{name}：受試者在量測區內時，只有 {res.summary['feet_visible_in_zone_pct']:.0f}% 影格清楚看到腳踝與腳尖，"
                         "請檢查光線、褲管是否蓋住腳踝、鞋子與地板顏色是否太接近。")
        hv = res.summary.get("heel_visible_in_zone_pct")
        if hv is not None and not (isinstance(hv, float) and math.isnan(hv)) and hv < 70:
            flags.append(f"{name}：量測區內只有 {hv:.0f}% 影格看得到腳跟點（常見原因：褲管蓋住腳踝、深色鞋）。"
                         "著地時間已改用腳踝點計算，不影響步寬，但足部角度可能缺少。下次請受試者把褲管捲到腳踝以上。")
        sp = res.summary.get("lr_swap_frames_pct")
        if sp is not None and not (isinstance(sp, float) and math.isnan(sp)) and sp > 5:
            flags.append(f"{name}：{sp:.0f}% 影格的左右標記疑似對調，左右腳的指標（足角、左右步）可能不可靠，請看標註影片確認。")
        if res.summary.get("started_inside_zone"):
            flags.append(f"{name}：受試者似乎在量測區內才起步，前一兩步還在加速，步時變異會偏高。請從離鏡頭 9–10 m 處起步。")
        nc = res.summary.get("n_crossover_steps", 0)
        if nc:
            flags.append(f"{name}：有 {nc} 步的腳落在對側連線的另一邊（交叉步）。可能是剪刀步態，也可能是左右腳標記錯誤，請看標註影片確認。")
        # 最靠近鏡頭的一步常因腳尖出畫面而不採用，屬正常，不提醒
        rej = [d for d in ((res.debug or {}).get("rejected") or []) if "畫面邊緣" not in d["reason"]]
        if rej:
            items = "；".join(f"{d['t_s']:.2f} 秒{'左' if d['side'] == 'L' else '右'}腳，{d['reason']}" for d in rej[:4])
            flags.append(f"{name}：量測區內有 {len(rej)} 次著地未採用：{items}。偶爾一次不影響結果；常出現時請看影片該時間點。")
        if res.summary["n_contacts_used"] < 4:
            flags.append(f"{name}：量測區內只抓到 {res.summary['n_contacts_used']} 次著地，結果不可靠。")
    return flags


def _geo_txt(meta):
    g = meta.get("calib_check") or {}
    return f"（校正反推 {g['cam_height_m']:.2f} m）" if g else ""


def write_html(path, meta, agg, results, floor_w, floor_l):
    img_fp = fig_footprints(results, floor_w, floor_l)
    img_sw = fig_stepwidth(results)
    img_sp = [fig_speed(n, r) for n, r in results]
    flags = quality_flags(agg, results, meta)

    trial_names = [n for n, _ in results]
    head = "".join(f"<th>{html.escape(n)}</th>" for n in trial_names)
    rows = ""
    for key, label, unit, nd, note in METRICS:
        per = "".join(f"<td>{_fmt(res.summary.get(key), nd)}</td>" for _, res in results)
        rows += (f"<tr><td class='lab'>{label}</td><td class='big'>{_fmt(agg.get(key), nd)}</td>"
                 f"<td class='unit'>{unit}</td>{per if len(results) > 1 else ''}"
                 f"<td class='note'>{html.escape(note)}</td></tr>")
    step_rows = ""
    for name, res in results:
        for r in res.steps:
            step_rows += ("<tr>" + f"<td>{html.escape(name)}</td><td>{r['n']}</td><td>{'左' if r['side']=='L' else '右'}</td>"
                          f"<td>{r['t_contact_s']:.2f}</td><td>{_fmt(r['step_width_cm'],1)}</td>"
                          f"<td>{_fmt(r['step_length_cm'],1)}</td><td>{_fmt(r['step_time_s'],2)}</td>"
                          f"<td>{_fmt(r.get('stride_length_cm'),1)}</td></tr>")
    flag_html = ("<div class='flags'><b>品質提醒</b><ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in flags) + "</ul></div>") if flags else \
        "<div class='ok'>品質檢查通過：步數與人體偵測率足夠。</div>"
    speed_imgs = "".join(f"<img src='data:image/png;base64,{s}'>" for s in img_sp)
    per_head = head if len(results) > 1 else ""

    doc = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>步態分析報告 {html.escape(meta.get('subject',''))}</title>
<style>
body{{font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px;border-bottom:2px solid #eee;padding-bottom:4px}}
.meta{{color:#666;font-size:14px}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid #eee;padding:6px 8px;text-align:right}} th{{background:#fafafa}}
td.lab{{text-align:left;font-weight:600}} td.big{{font-size:18px;font-weight:700}} td.unit{{color:#666;text-align:left}}
td.note{{text-align:left;color:#666;font-size:12px}} img{{max-width:100%;display:block;margin:8px 0}}
.flags{{background:#fff4e5;border-left:4px solid #e8743b;padding:8px 12px;margin:12px 0}}
.ok{{background:#eef7ee;border-left:4px solid #3a9d4a;padding:8px 12px;margin:12px 0}}
.disc{{color:#888;font-size:12px;margin-top:32px}}
</style></head><body>
<h1>步態分析報告（研究原型）</h1>
<div class="meta">受試者：{html.escape(meta.get('subject','—'))}　|　分析時間：{meta['created']}　|　趟數：{len(results)}　|
地板量測區：{floor_w:g} m × {floor_l:g} m　|　鏡頭高度設定：{meta.get('cam_height_m')} m{_geo_txt(meta)}　|　步寬左右位置：{'腳踝點' if meta.get('lateral_point') == 'ankle' else '腳跟點'}　|　{meta.get('footwear', '穿鞋')}</div>
{flag_html}
<h2>主要指標</h2>
<table><tr><th style="text-align:left">指標</th><th>合併</th><th></th>{per_head}<th style="text-align:left">說明</th></tr>{rows}</table>
<h2>腳印圖</h2><img src="data:image/png;base64,{img_fp}">
<h2>逐步步寬</h2><img src="data:image/png;base64,{img_sw}">
<h2>品質檢查</h2>{speed_imgs}
<h2>逐步資料</h2>
<table><tr><th>趟</th><th>#</th><th>腳</th><th>著地時間 s</th><th>步寬 cm</th><th>步長 cm</th><th>步時 s</th><th>步幅 cm</th></tr>{step_rows}</table>
<div class="disc">本報告由研究用原型程式產生，尚未經過與參考標準（壓力步道或標記式動作分析）的驗證，數值僅供研究使用，不可作為臨床診斷依據。
所有運算在本機完成，未上傳任何資料。</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ---------------------------------------------------------------- 標註影片
LEG_LINKS = [(P.L_HIP, P.R_HIP), (P.L_HIP, P.L_KNEE), (P.L_KNEE, P.L_ANKLE), (P.L_ANKLE, P.L_HEEL),
             (P.L_HEEL, P.L_TOE), (P.L_ANKLE, P.L_TOE), (P.R_HIP, P.R_KNEE), (P.R_KNEE, P.R_ANKLE),
             (P.R_ANKLE, P.R_HEEL), (P.R_HEEL, P.R_TOE), (P.R_ANKLE, P.R_TOE),
             (P.L_SHOULDER, P.R_SHOULDER), (P.L_SHOULDER, P.L_HIP), (P.R_SHOULDER, P.R_HIP)]


def write_overlay(video_path, out_path, res, H, floor_w, floor_l):
    info = P.video_info(video_path)
    cap = P.open_video(video_path)
    fps = info["fps"]
    w, h = info["width"], info["height"]
    tt = res.debug_t
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    grid = []
    for y in np.arange(0, floor_l + 1e-6, 1.0):
        grid.append(to_image(H, [[0, y], [floor_w, y]]))
    for x in (0, floor_w):
        grid.append(to_image(H, [[x, 0], [x, floor_l]]))
    contacts = {}
    for st in res.stances:
        contacts.setdefault(st.side, []).append(st)
    used_t = {round(r["t_contact_s"], 4): r for r in res.steps}
    i = 0
    xy = res.xy
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 以時間對應到分析用的影格（變動影格率影片也能對上）
        ti = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if i == 0:
            t0 = ti
        j = int(np.clip(np.searchsorted(tt, ti - t0 - 0.5 / fps), 0, len(tt) - 1))
        for seg in grid:
            cv2.line(frame, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])), (0, 200, 255), 2)
        for a, b in LEG_LINKS:
            pa, pb = xy[j, a], xy[j, b]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                col = (219, 111, 42) if a in (P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_HEEL, P.L_SHOULDER) and b not in (P.R_HIP, P.R_SHOULDER) else (59, 116, 232)
                cv2.line(frame, tuple(np.int32(pa)), tuple(np.int32(pb)), col, 3)
        for s, sts in contacts.items():
            col = (219, 111, 42) if s == "L" else (59, 116, 232)
            for st in sts:
                if st.i_on <= j:
                    p = to_image(H, [st.toe if not np.isnan(st.toe).any() else st.heel])[0]
                    used = _tc(res, st) in used_t
                    cv2.circle(frame, tuple(np.int32(p)), 9, col, -1 if used else 2)
                    if st.i_on <= j < st.i_off:
                        cv2.circle(frame, tuple(np.int32(p)), 16, (0, 255, 0), 3)
                        if used and j - st.i_on < fps * 0.6:
                            r = used_t[_tc(res, st)]
                            if not math.isnan(r["step_width_cm"]):
                                cv2.putText(frame, f"SW {r['step_width_cm']:.1f} cm", (int(p[0]) + 20, int(p[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"t={tt[j]:5.2f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()


def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


# ================================================================ 側面報告
SIDE_METRICS = [
    ("stance_pct", "站立期", "% 週期", 1, "健康成人約 60%；側面速度法，待步道驗證"),
    ("swing_pct", "擺盪期", "% 週期", 1, ""),
    ("double_support_pct", "雙腳支撐期", "% 週期", 1, "健康成人約 20%；DCM 隨嚴重度上升"),
    ("stance_time_s", "站立時間", "s", 2, ""),
    ("swing_time_s", "擺盪時間", "s", 2, ""),
    ("stride_time_s", "步幅時間", "s", 2, ""),
    ("cadence_steps_min", "步頻", "步/分", 1, ""),
    ("step_time_cv_pct", "步時變異（CV）", "%", 1, ""),
    ("step_length_cm", "步長", "cm", 1, "側面量；與正面步幅/2 交叉比對"),
    ("walking_speed_cm_s", "步速", "cm/s", 1, "由著地點位置對時間的斜率計算"),
    ("knee_peak_swing_L_deg", "擺盪期最大膝屈曲 左", "°", 1, "只取靠近鏡頭那一側；健康成人約 60°"),
    ("knee_peak_swing_R_deg", "擺盪期最大膝屈曲 右", "°", 1, "只取靠近鏡頭那一側"),
]


def fig_side(name, res) -> str:
    t = res.t
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    ax = axes[0]
    for s, col in (("L", C_L), ("R", C_R)):
        ax.plot(t, res.sig[s]["vh"], color=col, lw=1.1, label=f"{'左' if s == 'L' else '右'}腳跟速度")
        ax.plot(t, res.sig[s]["vt"], color=col, lw=0.8, ls=":")
        for e in res.events[s]:
            ax.axvspan(e[0], e[1], color=col, alpha=0.12 if e[3] else 0.04)
    ax.axhline(res.summary["speed_threshold_m_s"], color="#666", ls=":", lw=1)
    ax.set_ylabel("往前速度 m/s")
    ax.set_ylim(-1, 6)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.set_title(f"品質檢查：足部速度與站立期（色塊；淡色 = 未採用）— {name}", fontsize=10)
    ax = axes[1]
    for s, col in (("L", C_L), ("R", C_R)):
        lw = 1.6 if res.summary["near_side"] == s else 0.8  # 近側線條較粗
        ax.plot(t, res.sig[s]["knee"], color=col, lw=lw, label=f"{'左' if s == 'L' else '右'}膝（{'近側' if res.summary['near_side'] == s else '遠側'}）")
    ax.set_ylabel("膝屈曲 °")
    ax.set_xlabel("時間（s）")
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def write_side_html(path, meta, agg, results):
    names = [n for n, _ in results]
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names) if len(results) > 1 else ""
    rows = ""
    for key, label, unit, nd, note in SIDE_METRICS:
        per = "".join(f"<td>{_fmt(r.summary.get(key), nd)}</td>" for _, r in results) if len(results) > 1 else ""
        rows += (f"<tr><td class='lab'>{label}</td><td class='big'>{_fmt(agg.get(key), nd)}</td>"
                 f"<td class='unit'>{unit}</td>{per}<td class='note'>{html.escape(note)}</td></tr>")
    flags = []
    for n, r in results:
        if r.summary["n_strides"] < 2:
            flags.append(f"{n}：只抓到 {r.summary['n_strides']} 個完整步態週期，請確認受試者有走過兩張標記之間、鏡頭沒有被擋住。")
        if r.summary["lr_swaps_fixed_frames"] > 0.1 * len(r.t):
            flags.append(f"{n}：有 {r.summary['lr_swaps_fixed_frames']} 格左右腳標記對調被自動修正，請看品質檢查圖確認。")
    flag_html = ("<div class='flags'><b>品質提醒</b><ul>" + "".join(f"<li>{html.escape(f)}</li>" for f in flags) + "</ul></div>") if flags else \
        "<div class='ok'>品質檢查通過。</div>"
    imgs = "".join(f"<img src='data:image/png;base64,{fig_side(n, r)}'>" for n, r in results)
    doc = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>側面步態報告 {html.escape(meta.get('subject',''))}</title>
<style>
body{{font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px;border-bottom:2px solid #eee;padding-bottom:4px}}
.meta{{color:#666;font-size:14px}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid #eee;padding:6px 8px;text-align:right}} th{{background:#fafafa}}
td.lab{{text-align:left;font-weight:600}} td.big{{font-size:18px;font-weight:700}} td.unit{{color:#666;text-align:left}}
td.note{{text-align:left;color:#666;font-size:12px}} img{{max-width:100%;display:block;margin:8px 0}}
.flags{{background:#fff4e5;border-left:4px solid #e8743b;padding:8px 12px;margin:12px 0}}
.ok{{background:#eef7ee;border-left:4px solid #3a9d4a;padding:8px 12px;margin:12px 0}}
.disc{{color:#888;font-size:12px;margin-top:32px}}
</style></head><body>
<h1>側面步態報告（研究原型）</h1>
<div class="meta">受試者：{html.escape(meta.get('subject','—'))}　|　分析時間：{meta['created']}　|　趟數：{len(results)}　|　側面標記間距：{meta.get('gap_m')} m</div>
{flag_html}
<h2>主要指標</h2>
<table><tr><th style="text-align:left">指標</th><th>合併</th><th></th>{head}<th style="text-align:left">說明</th></tr>{rows}</table>
<h2>品質檢查</h2>{imgs}
<div class="disc">研究用原型，側面分析目前只經過合成資料驗證，尚未與步道或動作擷取系統比對，不可作為臨床診斷依據。所有運算在本機完成。</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
