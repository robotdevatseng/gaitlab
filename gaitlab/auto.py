"""一鍵分析：選一個受試者資料夾，正面、側面全部自動跑完，並打包要寄給曾醫師的檔案。

資料夾放法（擇一）：
  A. 子資料夾：S02/正面/*.mov、S02/側面/*.mov（檔名不用改）
  B. 全放一起：檔名有 side 或「側」的當側面，其餘當正面
（另有 站立/ 子資料夾或檔名含 static、「站立」→ 靜態站立測試，平常不需要）

設定（量測區大小、鏡頭高度、側面標記距離）寫在程式資料夾的「設定.txt」，第一次執行會自動產生。
結果放在 S02/結果/；要寄的檔案自動打包成 S02/寄給曾醫師_S02.zip（不含影片，只有幾 MB）。
"""
from __future__ import annotations

import os
import sys
import traceback
import webbrowser
import zipfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(ROOT, "設定.txt")
VIDEO_EXT = (".mov", ".mp4", ".m4v", ".avi")
DEFAULTS = [
    ("量測區寬度", 1.0, "地板四張標記（0–3 號）左右中心距離，公尺"),
    ("量測區長度", 5.0, "地板四張標記前後中心距離，公尺"),
    ("正面鏡頭高度", 1.2, "正面 iPad 鏡頭離地高度，公尺"),
    ("側面標記距離", 2.0, "側面兩張標記（4、5 號）中心距離，公尺"),
    ("赤腳", "否", "受試者是否赤腳（是／否）。病人一律穿鞋；只有同仁驗證的赤腳組才改成「是」，或把資料夾名稱加上「赤腳」"),
]
YES, NO = ("是", "yes", "y", "true", "1", "赤腳"), ("否", "no", "n", "false", "0", "穿鞋")


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            f.write("# 一鍵分析的設定。只改等號右邊，存檔即可。\n")
            for k, v, note in DEFAULTS:
                f.write(f"# {note}\n{k} = {v}\n")
        print(f"已建立設定檔：{SETTINGS_FILE}（請確認數字跟現場佈置一樣）")
    vals = {k: v for k, v, _ in DEFAULTS}
    with open(SETTINGS_FILE, encoding="utf-8-sig") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = (x.strip() for x in line.split("=", 1))
            if v.lower() in YES or v.lower() in NO:
                vals[k] = "是" if v.lower() in YES else "否"
                continue
            try:
                vals[k] = float(v)
            except ValueError:
                print(f"設定檔「{k}」的值看不懂：{v}，改用預設 {vals.get(k)}")
    return vals


def pick_folder() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="選擇受試者資料夾（例如 S02）")
        root.destroy()
        if path:
            return path
    except Exception:
        pass
    raw = input("請把受試者資料夾拖進這個視窗，再按 Enter：").strip()
    return raw.strip("'\"").replace("\\ ", " ")


def _videos(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(VIDEO_EXT) and not f.startswith(".") and "_overlay" not in f)


def classify(folder: str) -> tuple[list, list, list]:
    subs = {n: os.path.join(folder, n) for n in ("正面", "側面", "站立")}
    if any(os.path.isdir(p) for p in subs.values()):
        return _videos(subs["正面"]), _videos(subs["側面"]), _videos(subs["站立"])
    front, side, static = [], [], []
    for v in _videos(folder):
        n = os.path.basename(v).lower()
        if "side" in n or "側" in n:
            side.append(v)
        elif "static" in n or "站立" in n:
            static.append(v)
        else:
            front.append(v)
    return front, side, static


def pack(folder: str, out_dir: str, subject: str) -> str:
    """結果資料夾裡除了影片以外的檔案（報告、pose 檔、校正檔、csv）打包。"""
    zpath = os.path.join(folder, f"寄給曾醫師_{subject}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, files in os.walk(out_dir):
            for f in files:
                if f.lower().endswith((".mp4", ".mov", ".m4v", ".avi")) or f.startswith("."):
                    continue
                full = os.path.join(dp, f)
                z.write(full, os.path.join(subject, os.path.relpath(full, out_dir)))
    return zpath


def run(folder: str | None = None, open_reports: bool = True) -> dict:
    from . import cli as CLI
    from . import calib as C

    print("=" * 60)
    print("GaitLab 一鍵分析（研究原型，全部在本機運算）")
    print("=" * 60)
    st = load_settings()
    folder = os.path.abspath(folder or pick_folder())
    if not os.path.isdir(folder):
        sys.exit(f"找不到資料夾：{folder}")
    subject = os.path.basename(folder.rstrip("/\\"))
    barefoot = st.get("赤腳") == "是" or "赤腳" in subject or "barefoot" in subject.lower()
    front, side, static = classify(folder)
    print(f"受試者：{subject}｜正面影片 {len(front)} 段｜側面影片 {len(side)} 段" + (f"｜站立測試 {len(static)} 段" if static else ""))
    print(f"設定：量測區 {st['量測區寬度']} × {st['量測區長度']} m，鏡頭高 {st['正面鏡頭高度']} m，側面標記距離 {st['側面標記距離']} m，"
          f"{'赤腳' if barefoot else '穿鞋'}")
    if not (front or side or static):
        sys.exit("資料夾裡沒有影片（.mov/.mp4）。正面放在「正面」子資料夾、側面放在「側面」子資料夾，或直接放在受試者資料夾。")

    out = os.path.join(folder, "結果")
    os.makedirs(out, exist_ok=True)
    done, problems, reports = {}, [], []
    floor = (st["量測區寬度"], st["量測區長度"])
    calib_path = os.path.join(out, "正面", "calib.json")

    if front or static:
        try:
            os.makedirs(os.path.dirname(calib_path), exist_ok=True)
            if os.path.exists(calib_path):
                print(f"\n使用上次的地板校正：{calib_path}（要重新校正就刪掉這個檔案）")
            else:
                print("\n【1】地板校正")
                CLI.do_calibrate((front or static)[0], floor, "auto", None, calib_path)
        except SystemExit as e:
            problems.append(f"地板校正沒有完成：{e}")
        except Exception as e:  # pragma: no cover
            problems.append(f"地板校正失敗：{e}")

    if front and os.path.exists(calib_path):
        print("\n【2】正面分析（每段約 1–2 分鐘）")
        try:
            agg, geo = CLI.run_analyze(SimpleNamespace(
                videos=front, floor=floor, calib=calib_path, method="auto", points=None, subject=subject,
                out=os.path.join(out, "正面"), cam_height=st["正面鏡頭高度"], model=1, overlay=False,
                reuse=True, lateral="ankle", barefoot=barefoot))
            done["front"] = agg
            reports.append(os.path.join(out, "正面", "report.html"))
            if geo and abs(geo["cam_height_m"] - st["正面鏡頭高度"]) > 0.08:
                problems.append(f"地板校正反推鏡頭高度 {geo['cam_height_m']:.2f} m，跟設定 {st['正面鏡頭高度']} m 不一致："
                                "請確認標記間距（中心到中心）與鏡頭高度")
        except Exception as e:
            traceback.print_exc()
            problems.append(f"正面分析失敗：{e}")

    if side:
        print("\n【3】側面分析")
        try:
            agg = CLI.run_side(SimpleNamespace(
                videos=side, gap=st["側面標記距離"], method="auto", points=None, subject=subject,
                out=os.path.join(out, "側面"), model=1, reuse=True))
            done["side"] = agg
            reports.append(os.path.join(out, "側面", "side_report.html"))
        except SystemExit as e:
            problems.append(f"側面分析沒有完成：{e}")
        except Exception as e:
            traceback.print_exc()
            problems.append(f"側面分析失敗：{e}")

    if static and os.path.exists(calib_path):
        print("\n【4】靜態站立測試")
        try:
            CLI.run_static(SimpleNamespace(videos=static, calib=calib_path, cam_height=st["正面鏡頭高度"], barefoot=barefoot,
                                           out=os.path.join(out, "站立"), model=1, reuse=True))
        except Exception as e:
            problems.append(f"站立測試失敗：{e}")

    with open(os.path.join(out, "使用的設定.txt"), "w", encoding="utf-8") as f:
        for k, v in st.items():
            f.write(f"{k} = {v}\n")
    zpath = pack(folder, out, subject)

    print("\n" + "=" * 60)
    if "front" in done:
        a = done["front"]
        print(f"正面：步寬 {a.get('step_width_mean_cm', float('nan')):.1f} cm｜步速 {a.get('gait_speed_cm_s', float('nan')):.1f} cm/s"
              f"｜步頻 {a.get('cadence_steps_min', float('nan')):.1f} 步/分")
    if "side" in done:
        a = done["side"]
        print(f"側面：站立期 {a.get('stance_pct', float('nan')):.1f}%｜雙腳支撐 {a.get('double_support_pct', float('nan')):.1f}%")
    for p in problems:
        print("⚠ " + p)
    print(f"\n報告在：{out}")
    print(f"要寄給曾醫師的檔案：{zpath}")
    print("（不含影片；影片需要時另外傳，請勿用 LINE）")
    print("=" * 60)
    if open_reports:
        for r in reports:
            if os.path.exists(r):
                webbrowser.open("file://" + os.path.abspath(r))
    return {"zip": zpath, "problems": problems, "done": list(done)}
