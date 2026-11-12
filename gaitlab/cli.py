"""命令列入口。

  python -m gaitlab markers                       # 產生列印用的地板標記 PDF
  python -m gaitlab calibrate 影片.mp4 --floor 1.0x6.0
  python -m gaitlab analyze 影片1.mp4 影片2.mp4 影片3.mp4 --floor 1.0x6.0 --subject P001
  python -m gaitlab auto [受試者資料夾]           # 一鍵分析（建議）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import calib as C
from . import pose as P
from . import gait as G
from . import report as R


def _floor(text: str) -> tuple[float, float]:
    try:
        w, l = (float(v) for v in text.lower().replace("×", "x").split("x"))
    except Exception:
        raise argparse.ArgumentTypeError("格式：寬x長（公尺），例如 1.0x6.0")
    return w, l


def do_calibrate(video: str, floor, method: str, points: str | None, out: str, prev: dict | None = None) -> dict:
    """對「這一支影片」做地板校正。prev = 同一次拍攝中前一支影片的校正：
    只有在解析度相同、且這支影片自動辨識到的標記位置與 prev 相差 < 3 像素（鏡頭確實沒動）時，
    才沿用 prev 中沒辨識到的那幾個點；否則請使用者重新點選。"""
    w, l = floor
    frame = P.read_frame(video, 0)
    size = (frame.shape[1], frame.shape[0])
    vname = os.path.basename(video)
    if method == "points":
        if not points:
            sys.exit("使用 --method points 時必須同時給 --points x0,y0;x1,y1;x2,y2;x3,y3")
        pts, used = C.parse_points(points), "points"
    else:
        pts, used, found = None, None, {}
        if method in ("auto", "aruco"):
            # 前 2 秒內多取幾格，累積偵測到的標記（遠端標記常因太扁而偵測不到）
            info = P.video_info(video)
            for i in range(0, min(info["n_frames"], int(info["fps"] * 2)), max(1, int(info["fps"] // 4))):
                for k, v in C.detect_aruco_ids(P.read_frame(video, i)).items():
                    found.setdefault(k, v)
                if len(found) == 4:
                    break
            if len(found) == 4:
                pts, used = np.float32([found[k] for k in range(4)]), "aruco"
            elif method == "aruco":
                sys.exit(f"只找到標記 {sorted(found)}，請改用 --method auto 或 click")
        if pts is None and prev is not None and found and list(prev.get("image_size_px") or []) == list(size):
            pp = np.asarray(prev["image_points_px"], float)
            dev = max(float(np.hypot(*(np.asarray(found[k]) - pp[k]))) for k in found)
            if dev < 3.0:
                pts = np.float32([found[k] if k in found else pp[k] for k in range(4)])
                used = f"aruco+沿用前一支（鏡頭未移動，標記位置相差 {dev:.1f} 像素）"
            else:
                print(f"這支影片的標記位置與前一支相差 {dev:.0f} 像素（鏡頭或畫面有變動），需要重新點選。")
        if pts is None:
            miss = [k for k in range(4) if k not in found]
            names = {0: "近左", 1: "近右", 2: "遠右", 3: "遠左"}
            if found:
                print(f"已自動找到標記 {sorted(found)}；請在視窗中依序點選：{' → '.join(names[k] for k in miss)}（會自動吸附到標記中心），按 Enter 確認")
            else:
                print("未偵測到 ArUco 標記，請在視窗中依序點選：近左 → 近右 → 遠右 → 遠左，按 Enter 確認")
            pts, used = C.click_points(frame, found), ("aruco+click" if found else "click")
    c = C.save_calib(out, pts, w, l, used, image_size=size, video=vname)
    print(f"校正完成（{used}），已存成 {out}")
    return C.load_calib(out)


def side_markers(video: str, method: str, points: str | None) -> np.ndarray:
    """側面標記（ID 4 = 靠近正面鏡頭那一張、ID 5）。每支影片各自校正（鏡頭可能移動）。"""
    if method == "points":
        if not points:
            sys.exit("使用 --method points 時必須給 --points x4,y4;x5,y5")
        try:
            pts = [[float(v) for v in p.split(",")] for p in points.split(";") if p.strip()]
        except ValueError:
            sys.exit("--points 格式錯誤，應為 x4,y4;x5,y5")
        if len(pts) != 2 or any(len(p) != 2 for p in pts):
            sys.exit("--points 需要剛好兩個點：x4,y4;x5,y5")
        return np.float32(pts)
    found = {}
    side_markers.last_auto = False  # type: ignore[attr-defined]
    if method == "auto":
        info = P.video_info(video)
        for i in range(0, min(info["n_frames"], int(info["fps"] * 2)), max(1, int(info["fps"] // 4))):
            for k, v in C.detect_aruco_ids(P.read_frame(video, i), ids=(4, 5)).items():
                found.setdefault(k, v)
            if len(found) == 2:
                side_markers.last_auto = True  # type: ignore[attr-defined]
                return np.float32([found[4], found[5]])
    known = {0: found[4]} if 4 in found else {}
    if 5 in found:
        known[1] = found[5]
    print(f"請在視窗中點選側面標記：{'、'.join(n for i, n in enumerate(['4 號', '5 號']) if i not in known)}（點附近即可，會自動吸附），按 Enter 確認")
    return C.click_points(P.read_frame(video, 0), known, labels=("4 side-near", "5 side-far"))


def run_side(args):
    from . import side as S
    import json
    os.makedirs(args.out, exist_ok=True)
    results, skipped = [], {}
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cache = os.path.join(args.out, name + "_pose.npz")
        mkfile = os.path.join(args.out, name + "_markers.json")
        have_cache = args.reuse and os.path.exists(cache)
        if os.path.exists(v):
            vi = P.video_info(v)
            size = (vi["width"], vi["height"])
        elif have_cache:
            size = tuple(int(x) for x in P.load_landmarks(cache)["size"])
        else:
            skipped[name] = "找不到影片檔"
            continue
        prob = P.size_problem(size)
        if prob:
            skipped[name] = prob
            print(f"[{name}] ✘ 停止分析：{prob}")
            continue
        if os.path.exists(v):
            try:
                mk = side_markers(v, args.method, args.points)
            except SystemExit as e:
                skipped[name] = f"側面標記沒有完成：{e}"
                continue
            auto = bool(getattr(side_markers, "last_auto", False))
            with open(mkfile, "w", encoding="utf-8") as f:
                json.dump({"markers_px": np.asarray(mk).tolist(), "auto_detected": auto,
                           "image_size_px": [int(size[0]), int(size[1])]}, f, ensure_ascii=False)
        elif os.path.exists(mkfile):
            d = json.load(open(mkfile, encoding="utf-8"))
            mk, auto = np.float32(d["markers_px"]), bool(d.get("auto_detected"))
        else:
            skipped[name] = "找不到影片檔，也沒有存下來的側面標記位置（舊版輸出沒有存），無法換算距離"
            print(f"[{name}] ✘ 不分析：{skipped[name]}")
            continue
        print(f"[{name}] 姿態估計中…")
        if have_cache:
            lm = P.load_landmarks(cache)
        else:
            lm = P.extract_landmarks(v, model_complexity=args.model)
            P.save_landmarks(lm, cache)
        try:
            r = S.analyze_side(lm, mk, args.gap)
        except ValueError as e:
            skipped[name] = str(e)
            print(f"[{name}] 無法分析：{e}")
            continue
        s = r.summary
        # ArUco 標記鏡像後無法辨識，所以自動辨識成功 = 影片確定沒有左右鏡像
        s["lr"]["mirrored_check"] = "已確認未鏡像（自動辨識到標記）" if auto else "未確認（標記為手動點選）；假設未鏡像"
        s["markers_px"] = np.asarray(mk).tolist()
        pc = lambda x: "—（算不出來）" if not np.isfinite(x) else f"{x:.1f}%"  # noqa: E731
        print(f"[{name}] 週期 {s['n_strides']}｜站立期 {pc(s['stance_pct'])}｜雙腳支撐 {pc(s['double_support_pct'])}｜步頻 {s['cadence_steps_min']:.1f}"
              f"｜{s['lr']['walking_direction_image']}，近側 = {s['lr'].get('near_side_expected')} 腳")
        results.append((name, r))
    _archive_old(args.out, ("side_report.html", "side_summary.json"))
    meta = {"subject": args.subject, "created": R.now_str(), "gap_m": args.gap,
            "gap_note": "側面標記間距只影響側面的「距離」指標（步長、步速），兩者與間距成正比；"
                        "不影響時間與比例指標（站立期、擺盪期、雙腳支撐期、步頻、步時、週期時間）與膝角度；也不影響正面分析的任何指標。",
            "validation_status": {"double_support_pct": "未驗證（尚未與參考標準比對；同一人各趟仍差 4–7 個百分點）",
                                  "stance_pct": "僅合成資料驗證；待與參考標準比對"},
            "not_analyzed": skipped}
    clean = R.clean_nan
    if not results:
        with open(os.path.join(args.out, "side_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "overall": None, "overall_null_reason": "沒有可分析的側面影片（原因見 meta.not_analyzed）",
                       "trials": {}}, f, ensure_ascii=False, indent=2)
        R.write_blocked_html(os.path.join(args.out, "side_report.html"), meta, "側面步態分析")
        sys.exit("沒有可分析的側面影片：" + "；".join(f"{k}：{v}" for k, v in skipped.items()))
    agg = S.aggregate_side([r for _, r in results])
    R.write_side_html(os.path.join(args.out, "side_report.html"), meta, agg, results)
    with open(os.path.join(args.out, "side_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "overall": clean(agg), "overall_null_reasons": R.side_null_reasons(agg),
                   "trials": {n: {**clean(r.summary), "null_reasons": R.side_null_reasons(r.summary)} for n, r in results}},
                  f, ensure_ascii=False, indent=2)
    print(f"完成。報告：{os.path.join(args.out, 'side_report.html')}")
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gaitlab", description="離線單鏡頭步態分析（研究原型）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("markers", help="產生列印用 ArUco 地板標記 PDF")
    m.add_argument("--size-cm", type=float, default=18.0)
    m.add_argument("--out", default="aruco_markers.pdf")

    c = sub.add_parser("calibrate", help="只做地板校正")
    c.add_argument("video")
    c.add_argument("--floor", type=_floor, required=True, help="量測長方形 寬x長（m），例如 1.0x6.0")
    c.add_argument("--method", choices=["auto", "aruco", "click", "points"], default="auto")
    c.add_argument("--points", help="近左;近右;遠右;遠左 的像素座標 x,y;x,y;x,y;x,y")
    c.add_argument("--out", default=None)

    a = sub.add_parser("analyze", help="分析一趟或多趟行走影片")
    a.add_argument("videos", nargs="+")
    a.add_argument("--floor", type=_floor, required=True)
    a.add_argument("--calib", help="舊版共用校正檔；只在沒有影片、只有姿態快取時才會用到（現在每支影片各自校正）")
    a.add_argument("--method", choices=["auto", "aruco", "click", "points"], default="auto")
    a.add_argument("--points")
    a.add_argument("--subject", default="")
    a.add_argument("--out", default="gait_output")
    a.add_argument("--cam-height", type=float, default=1.20, help="鏡頭離地高度（m）")
    a.add_argument("--model", type=int, choices=[0, 1], default=1, help="0 = 快、1 = 準（預設）")
    a.add_argument("--overlay", action="store_true", help="另外輸出標註影片（較慢）")
    a.add_argument("--reuse", action="store_true", help="若已有姿態快取檔就直接使用")
    a.add_argument("--barefoot", action="store_true", help="受試者赤腳（高度修正改用赤腳的數值）")
    a.add_argument("--lateral", choices=["heel", "ankle"], default="ankle",
                   help="步寬的左右位置用哪個關鍵點（預設 ankle；heel 為舊版）")

    sd = sub.add_parser("side", help="分析側面拍攝（站立期、雙腳支撐期、膝角度）")
    sd.add_argument("videos", nargs="+")
    sd.add_argument("--gap", type=float, default=2.04, help="側面兩張標記（ID 4、5）中心距離（m）；本院佈置實測 2.04")
    sd.add_argument("--method", choices=["auto", "click", "points"], default="auto")
    sd.add_argument("--points", help="4 號、5 號標記中心像素座標 x,y;x,y")
    sd.add_argument("--subject", default="")
    sd.add_argument("--out", default="side_output")
    sd.add_argument("--model", type=int, choices=[0, 1], default=1)
    sd.add_argument("--reuse", action="store_true")

    st = sub.add_parser("static", help="靜態站立測試：兩腳跟站在已知距離的標記上，量關鍵點左右偏差")
    st.add_argument("videos", nargs="+")
    st.add_argument("--calib", required=True, help="正面校正檔 calib.json（同一次佈置）")
    st.add_argument("--cam-height", type=float, default=1.20)
    st.add_argument("--barefoot", action="store_true", help="受試者赤腳")
    st.add_argument("--out", default="static_output")
    st.add_argument("--model", type=int, choices=[0, 1], default=1)
    st.add_argument("--reuse", action="store_true")

    au = sub.add_parser("auto", help="一鍵分析：選一個受試者資料夾，正面、側面自動跑完並打包")
    au.add_argument("folder", nargs="?", help="受試者資料夾（不給就跳出選擇視窗）")

    args = ap.parse_args(argv)

    if args.cmd == "auto":
        from .auto import run as run_auto
        return run_auto(args.folder)

    if args.cmd == "static":
        return run_static(args)

    if args.cmd == "side":
        return run_side(args)

    if args.cmd == "markers":
        print("已產生", C.make_markers(args.out, args.size_cm))
        side = os.path.splitext(args.out)[0] + "_side.pdf"
        print("已產生", C.make_markers(side, args.size_cm, ids=(4, 5),
                                      names=["4  側面近端 side-near", "5  側面遠端 side-far"]))
        return

    if args.cmd == "calibrate":
        out = args.out or os.path.splitext(args.video)[0] + "_calib.json"
        do_calibrate(args.video, args.floor, args.method, args.points, out)
        return

    return run_analyze(args)


def _archive_old(out_dir: str, names: tuple) -> str | None:
    """重跑時不覆寫舊輸出：把上一次的報告、摘要、逐步資料搬到 舊輸出/<時間>/。"""
    import shutil
    import time
    old = [n for n in names if os.path.exists(os.path.join(out_dir, n))]
    if not old:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(os.path.join(out_dir, old[0]))))
    dst = os.path.join(out_dir, "舊輸出", stamp)
    k = 1
    while os.path.exists(dst):
        k += 1
        dst = os.path.join(out_dir, "舊輸出", f"{stamp}_{k}")
    os.makedirs(dst)
    for n in old:
        shutil.move(os.path.join(out_dir, n), os.path.join(dst, n))
    print(f"上一次的輸出已保留在：{dst}")
    return dst


def _calib_for(video: str, name: str, args, floor, prev: dict | None, size) -> tuple[dict | None, str | None]:
    """取得這支影片自己的校正。回傳 (calib, 無法校正的原因)。"""
    own = os.path.join(args.out, name + "_calib.json")
    if os.path.exists(own):
        cal = C.load_calib(own)
        if list(cal.get("image_size_px") or []) == [int(size[0]), int(size[1])]:
            return cal, None
    if os.path.exists(video):
        try:
            return do_calibrate(video, floor, args.method, args.points, own, prev), None
        except SystemExit as e:
            return None, f"地板校正沒有完成：{e}"
        except Exception as e:  # 例如無法開啟點選視窗
            return None, f"地板校正失敗：{e}"
    # 沒有影片、只有姿態快取（重算別人寄來的資料）：只能用舊的共用校正檔
    shared = args.calib or os.path.join(args.out, "calib.json")
    if os.path.exists(shared):
        cal = C.load_calib(shared)
        cs = cal.get("image_size_px")
        if cs is None:
            cal["method"] = (cal.get("method") or "") + "｜舊版共用校正檔（未記錄是哪一支影片，無法確認與本影片相符）"
            return cal, None
        if list(cs) == [int(size[0]), int(size[1])]:
            return cal, None
        return None, f"共用校正檔的解析度 {cs[0]}×{cs[1]} 與本影片 {int(size[0])}×{int(size[1])} 不同"
    return None, "找不到影片檔，也沒有可用的校正檔"


def run_analyze(args):
    """正面分析（analyze 子命令與一鍵分析共用）。每支影片各自校正；解析度不是相機原始規格的影片不分析。"""
    os.makedirs(args.out, exist_ok=True)
    w, l = args.floor
    barefoot = bool(getattr(args, "barefoot", False))
    prm = G.make_params(args.cam_height, args.lateral, barefoot)
    results, skipped, calibs = [], {}, {}
    prev_cal, geo0 = None, {}
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cache = os.path.join(args.out, name + "_pose.npz")
        have_cache = args.reuse and os.path.exists(cache)
        if os.path.exists(v):
            vi = P.video_info(v)
            size = (vi["width"], vi["height"])
        elif have_cache:
            size = tuple(int(x) for x in P.load_landmarks(cache)["size"])
        else:
            skipped[name] = "找不到影片檔"
            print(f"[{name}] ✘ 不分析：{skipped[name]}")
            continue
        prob = P.size_problem(size)
        if prob:
            skipped[name] = prob
            print(f"[{name}] ✘ 停止分析：{prob}")
            continue
        cal, why = _calib_for(v, name, args, (w, l), prev_cal, size)
        if cal is None:
            skipped[name] = why
            print(f"[{name}] ✘ 不分析：{why}")
            continue
        prev_cal = cal if cal.get("image_size_px") else prev_cal
        geo = {}
        try:
            geo = C.check_geometry(cal["image_points_px"], w, l, size)
            if geo:
                print(f"[{name}] 校正檢查：反推鏡頭高度 {geo['cam_height_m']:.2f} m（設定 {args.cam_height:.2f} m），"
                      f"四點誤差 {geo.get('resid_px', float('nan')):.1f} 像素")
                if abs(geo["cam_height_m"] - args.cam_height) > 0.08:
                    print("  ⚠ 差距超過 8 cm：請確認標記間距是否為中心到中心、鏡頭高度是否量對。")
                if geo.get("resid_px", 0) > 3.0:
                    print(f"  ⚠ 四個標記點彼此不太吻合：可能有一點點偏了，建議刪掉 {name}_calib.json 重新校正。")
        except Exception as e:  # 檢查失敗不影響分析
            print("校正檢查略過：", e)
        geo0 = geo0 or geo
        calibs[name] = {"method": cal.get("method"), "image_size_px": [int(size[0]), int(size[1])],
                        "image_points_px": np.asarray(cal["image_points_px"]).tolist(), "check": geo}
        print(f"[{name}] 姿態估計中…")
        if have_cache:
            lm = P.load_landmarks(cache)
        else:
            lm = P.extract_landmarks(v, model_complexity=args.model)
            P.save_landmarks(lm, cache)
        try:
            res = G.analyze(lm, cal["H"], w, l, prm)
        except ValueError as e:
            skipped[name] = str(e)
            print(f"[{name}] ✘ 無法分析：{e}")
            continue
        res.calib_check = geo  # type: ignore[attr-defined]
        s = res.summary
        print(f"[{name}] 著地 {s['n_contacts_used']} 次｜步寬 {s['step_width_mean_cm']:.1f} cm｜"
              f"步速 {s['gait_speed_cm_s']:.1f} cm/s｜步頻 {s['cadence_steps_min']:.1f}")
        if args.overlay and os.path.exists(v):
            R.write_overlay(v, os.path.join(args.out, name + "_overlay.mp4"), res, cal["H"], w, l)
        results.append((name, res))

    _archive_old(args.out, ("report.html", "summary.json", "steps.csv"))
    meta = {"subject": args.subject, "created": R.now_str(), "videos": args.videos,
            "floor_m": [w, l], "cam_height_m": args.cam_height,
            "calibration": "每支影片各自校正", "calibration_per_video": calibs,
            "calib_check": geo0, "lateral_point": args.lateral, "footwear": "赤腳" if barefoot else "穿鞋",
            "validation_status": {"step_width": "未驗證（尚未與壓力步道或其他參考標準比對；僅膠帶小驗證）",
                                  "speed_cadence_stride": "膠帶小驗證：步長 60 → 59.3–60.2 cm、步幅 120 → 118.1–119.3 cm"},
            "not_analyzed": skipped}
    if not results:
        agg = {k: None for k in ("n_step_width_values", "gait_speed_cm_s", "cadence_steps_min", "step_width_mean_cm",
                                 "step_width_sd_cm", "step_length_mean_cm", "stride_length_mean_cm", "stride_time_cv_pct")}
        agg["n_trials"] = 0
        R.write_summary_json(os.path.join(args.out, "summary.json"), meta, agg, [])
        R.write_blocked_html(os.path.join(args.out, "report.html"), meta, "正面步態分析")
        print("沒有可分析的正面影片；原因已寫入 summary.json 與 report.html。")
        return agg, geo0
    agg = G.aggregate([r for _, r in results])
    R.write_steps_csv(os.path.join(args.out, "steps.csv"), results)
    R.write_summary_json(os.path.join(args.out, "summary.json"), meta, agg, results)
    R.write_html(os.path.join(args.out, "report.html"), meta, agg, results, w, l)
    print(f"完成。報告：{os.path.join(args.out, 'report.html')}")
    return agg, geo0


def run_static(args):
    import csv
    from .static import analyze_static
    os.makedirs(args.out, exist_ok=True)
    cal = C.load_calib(args.calib)
    prm = G.make_params(args.cam_height, "ankle", bool(getattr(args, "barefoot", False)))
    rows = []
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cache = os.path.join(args.out, name + "_pose.npz")
        if args.reuse and os.path.exists(cache):
            lm = P.load_landmarks(cache)
        else:
            print(f"[{name}] 姿態估計中…")
            lm = P.extract_landmarks(v, model_complexity=args.model)
            P.save_landmarks(lm, cache)
        for r in analyze_static(lm, cal["H"], prm):
            r = {"video": name, **r}
            rows.append(r)
            print(f"[{name}] {r['t_start_s']:.1f}–{r['t_end_s']:.1f} s，距量測區近端 {r['dist_from_near_edge_m']:.1f} m："
                  f"腳跟 {r['heel_lateral_cm']:.1f}｜腳踝 {r['ankle_lateral_cm']:.1f}｜腳尖 {r['toe_lateral_cm']:.1f} cm")
    if rows:
        path = os.path.join(args.out, "static.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("已輸出", path, "（請在旁邊填上每段實際的腳跟中心距離）")
    else:
        print("沒有找到靜止 1.5 秒以上的片段。請每個位置站穩 3 秒以上。")


if __name__ == "__main__":
    main()
