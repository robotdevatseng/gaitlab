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


def do_calibrate(video: str, floor, method: str, points: str | None, out: str) -> dict:
    w, l = floor
    frame = P.read_frame(video, 0)
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
        if pts is None:
            miss = [k for k in range(4) if k not in found]
            names = {0: "近左", 1: "近右", 2: "遠右", 3: "遠左"}
            if found:
                print(f"已自動找到標記 {sorted(found)}；請在視窗中依序點選：{' → '.join(names[k] for k in miss)}（會自動吸附到標記中心），按 Enter 確認")
            else:
                print("未偵測到 ArUco 標記，請在視窗中依序點選：近左 → 近右 → 遠右 → 遠左，按 Enter 確認")
            pts, used = C.click_points(frame, found), ("aruco+click" if found else "click")
    c = C.save_calib(out, pts, w, l, used)
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
    if method == "auto":
        info = P.video_info(video)
        for i in range(0, min(info["n_frames"], int(info["fps"] * 2)), max(1, int(info["fps"] // 4))):
            for k, v in C.detect_aruco_ids(P.read_frame(video, i), ids=(4, 5)).items():
                found.setdefault(k, v)
            if len(found) == 2:
                return np.float32([found[4], found[5]])
    known = {0: found[4]} if 4 in found else {}
    if 5 in found:
        known[1] = found[5]
    print(f"請在視窗中點選側面標記：{'、'.join(n for i, n in enumerate(['4 號', '5 號']) if i not in known)}（點附近即可，會自動吸附），按 Enter 確認")
    return C.click_points(P.read_frame(video, 0), known, labels=("4 side-near", "5 side-far"))


def run_side(args):
    from . import side as S
    os.makedirs(args.out, exist_ok=True)
    results = []
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cache = os.path.join(args.out, name + "_pose.npz")
        mk = side_markers(v, args.method, args.points)
        print(f"[{name}] 姿態估計中…")
        if args.reuse and os.path.exists(cache):
            lm = P.load_landmarks(cache)
        else:
            lm = P.extract_landmarks(v, model_complexity=args.model)
            P.save_landmarks(lm, cache)
        try:
            r = S.analyze_side(lm, mk, args.gap)
        except ValueError as e:
            print(f"[{name}] 無法分析：{e}")
            continue
        s = r.summary
        print(f"[{name}] 週期 {s['n_strides']}｜站立期 {s['stance_pct']:.1f}%｜雙腳支撐 {s['double_support_pct']:.1f}%｜步頻 {s['cadence_steps_min']:.1f}")
        results.append((name, r))
    if not results:
        sys.exit("沒有可分析的側面影片")
    agg = S.aggregate_side([r for _, r in results])
    meta = {"subject": args.subject, "created": R.now_str(), "gap_m": args.gap}
    R.write_side_html(os.path.join(args.out, "side_report.html"), meta, agg, results)
    import json
    clean = lambda d: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}  # noqa: E731
    with open(os.path.join(args.out, "side_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "overall": clean(agg), "trials": {n: clean(r.summary) for n, r in results}},
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
    a.add_argument("--calib", help="已存在的校正檔（鏡頭沒移動時各趟共用）")
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
    sd.add_argument("--gap", type=float, default=2.0, help="側面兩張標記（ID 4、5）中心距離（m）")
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


def run_analyze(args):
    """正面分析（analyze 子命令與一鍵分析共用）。"""
    os.makedirs(args.out, exist_ok=True)
    w, l = args.floor
    if args.calib:
        cal = C.load_calib(args.calib)
    else:
        cal = do_calibrate(args.videos[0], args.floor, args.method, args.points,
                           os.path.join(args.out, "calib.json"))
    H = cal["H"]
    barefoot = bool(getattr(args, "barefoot", False))
    prm = G.make_params(args.cam_height, args.lateral, barefoot)
    geo = {}
    try:
        cache0 = os.path.join(args.out, os.path.splitext(os.path.basename(args.videos[0]))[0] + "_pose.npz")
        if os.path.exists(args.videos[0]):
            vi = P.video_info(args.videos[0])
            size = (vi["width"], vi["height"])
        else:  # 只有姿態快取檔（例如重算別人寄來的資料）
            size = P.load_landmarks(cache0)["size"]
        geo = C.check_geometry(cal["image_points_px"], w, l, size)
        if geo:
            print(f"校正檢查：反推鏡頭高度 {geo['cam_height_m']:.2f} m（設定 {args.cam_height:.2f} m），"
                  f"鏡頭到量測區近端 {geo['near_dist_m']:.2f} m")
            if abs(geo["cam_height_m"] - args.cam_height) > 0.08:
                print("  ⚠ 差距超過 8 cm：請確認標記間距是否為中心到中心、鏡頭高度是否量對。")
            if geo.get("resid_px", 0) > 3.0:
                print(f"  ⚠ 四個標記點彼此不太吻合（{geo['resid_px']:.1f} 像素）：可能有一點點偏了，建議刪掉 calib.json 重新校正。")
    except Exception as e:  # 檢查失敗不影響分析
        print("校正檢查略過：", e)
    results = []
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        cache = os.path.join(args.out, name + "_pose.npz")
        print(f"[{name}] 姿態估計中…")
        if args.reuse and os.path.exists(cache):
            lm = P.load_landmarks(cache)
        else:
            lm = P.extract_landmarks(v, model_complexity=args.model)
            P.save_landmarks(lm, cache)
        res = G.analyze(lm, H, w, l, prm)
        s = res.summary
        print(f"[{name}] 著地 {s['n_contacts_used']} 次｜步寬 {s['step_width_mean_cm']:.1f} cm｜"
              f"步速 {s['gait_speed_cm_s']:.1f} cm/s｜步頻 {s['cadence_steps_min']:.1f}")
        if args.overlay:
            R.write_overlay(v, os.path.join(args.out, name + "_overlay.mp4"), res, H, w, l)
        results.append((name, res))

    agg = G.aggregate([r for _, r in results])
    meta = {"subject": args.subject, "created": R.now_str(), "videos": args.videos,
            "floor_m": [w, l], "cam_height_m": args.cam_height, "calibration": cal.get("method"),
            "calib_check": geo, "lateral_point": args.lateral, "footwear": "赤腳" if barefoot else "穿鞋"}
    R.write_steps_csv(os.path.join(args.out, "steps.csv"), results)
    R.write_summary_json(os.path.join(args.out, "summary.json"), meta, agg, results)
    R.write_html(os.path.join(args.out, "report.html"), meta, agg, results, w, l)
    print(f"完成。報告：{os.path.join(args.out, 'report.html')}")
    return agg, geo


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
