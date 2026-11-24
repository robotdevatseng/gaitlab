"""0.4.1 鎖版前小修的自我檢查。執行：python tests/test_v041.py

只用合成畫面與合成步態，不讀任何受試者影片或影格。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gaitlab import calib as C  # noqa: E402
from gaitlab import cli as CLI  # noqa: E402
from gaitlab import gait as G  # noqa: E402
from gaitlab import pose as P  # noqa: E402
from gaitlab import report as R  # noqa: E402
from gaitlab import version as V  # noqa: E402
from gaitlab.side import SideResult  # noqa: E402
from tests.synth import simulate  # noqa: E402


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        raise SystemExit(f"失敗：{msg}")


def _check_ver_css(html_text, name):
    check("{{" not in html_text, f"{name} 的 HTML 不含 {{{{")
    check(".ver{background:#143d66" in html_text, f"{name} 的版本列樣式是深藍底")


def _paste(img, marker, x, y):
    h, w = marker.shape
    img[y:y + h, x:x + w] = marker


def _marker(i, side=80):
    return cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(C.ARUCO_DICT), i, side)


def test_version_constant():
    print("版本號 0.4.2")
    check(V.PROGRAM_VERSION == "0.4.2", f"PROGRAM_VERSION 是 0.4.2（實際 {V.PROGRAM_VERSION}）")
    check("0.4.2" in V.version_line(V.get_version()), "報告用的版本行取自 PROGRAM_VERSION")
    check(V.version_line({}) == "程式版本 未取得　|　git commit 未取得", "沒有版本資訊時寫「未取得」")


def test_reports_lead_with_version():
    print("報告與 summary 的第一行是程式版本與 git commit")
    meta = {"subject": "SYN", "created": "2026-11-24 12:00", "cam_height_m": 1.2,
            "program_version": "0.4.1", "commit": "abc1234deadbeef000000000000000000000000",
            "lateral_point": "ankle", "footwear": "穿鞋", "gap_m": 2.04, "gap_note": "測試",
            "not_analyzed": {}}
    line = V.version_line(meta)
    d = tempfile.mkdtemp()
    try:
        blocked = os.path.join(d, "blocked.html")
        R.write_blocked_html(blocked, {**meta, "not_analyzed": {"T": "沒有影片"}}, "正面步態分析")
        blocked_html = open(blocked, encoding="utf-8").read()
        body = blocked_html.split("<body>", 1)[1].lstrip()
        check(body.startswith('<div class="ver">'), "無法分析的報告，body 第一行是版本列")
        check(body.find(line) < body.find("受試者"), "版本列在受試者之前")
        _check_ver_css(blocked_html, "無法分析的報告")

        lm, H, _truth = simulate(seed=1)
        res = G.analyze(lm, H, 1.0, 5.0)
        agg = G.aggregate([res])
        summary = os.path.join(d, "summary.json")
        R.write_summary_json(summary, meta, agg, [("t1", res)])
        raw = json.load(open(summary, encoding="utf-8"))
        check(list(raw)[:2] == ["program_version", "commit"], "summary.json 最前面兩個欄位是版本與 commit")
        check(raw["program_version"] == "0.4.1" and raw["commit"] == meta["commit"], "summary 寫入的版本與 commit")
        check(raw["meta"]["program_version"] == "0.4.1" and raw["meta"]["commit"] == meta["commit"],
              "meta 裡仍保留 program_version 與 commit")
        lines = open(summary, encoding="utf-8").read().splitlines()
        check('"program_version"' in lines[1] and '"commit"' in lines[2], "檔案前兩行資料是版本與 commit")

        report = os.path.join(d, "report.html")
        R.write_html(report, meta, agg, [("t1", res)], 1.0, 5.0)
        report_html = open(report, encoding="utf-8").read()
        body = report_html.split("<body>", 1)[1].lstrip()
        check(body.startswith('<div class="ver">'), "正面報告 body 第一行是版本列")
        check(body.find("git commit") < body.find("受試者"), "正面報告的版本在受試者那一行之前")
        check("程式版本：" not in body, "版本不再放在頁首次行")
        _check_ver_css(report_html, "正面報告")

        summ = {"n_strides": 4, "near_side": "L", "near_side_conflict": False, "near_side_expected": "L",
                "lr": {"walking_direction_image": "往右", "frames_total": 10, "frames_discarded": 0,
                       "near_side_confident": True, "frames_swapped_locally": 0, "whole_trial_flipped": False,
                       "frames_relabelled": 0, "mirrored_check": "已確認"}}
        side_agg = {key: 1.0 for key, *_rest in R.SIDE_METRICS}
        side_res = SideResult(summ, {"L": [], "R": []}, [], np.zeros(1), {})
        orig = R.fig_side
        R.fig_side = lambda *a, **k: ""
        try:
            side_html = os.path.join(d, "side.html")
            R.write_side_html(side_html, meta, side_agg, [("s1", side_res)])
        finally:
            R.fig_side = orig
        side_text = open(side_html, encoding="utf-8").read()
        body = side_text.split("<body>", 1)[1].lstrip()
        check(body.startswith('<div class="ver">') and body.find("git commit") < body.find("受試者"),
              "側面報告 body 第一行是版本列")
        _check_ver_css(side_text, "側面報告")
        side_json = os.path.join(d, "side_summary.json")
        R.write_json(side_json, meta, {"meta": meta, "overall": side_agg, "trials": {}})
        side_raw = json.load(open(side_json, encoding="utf-8"))
        check(list(side_raw)[:2] == ["program_version", "commit"], "side_summary.json 最前面是版本與 commit")
    finally:
        shutil.rmtree(d)


def _frames_near_shift():
    """三張合成畫面。ID 0 的 x、ID 1 的 y、ID 2 的 x 逐格移動；ID 3 不動。"""
    marks = {i: _marker(i) for i in range(4)}
    frames = []
    for dx0, dy1, dx2 in ((0, 0, 0), (8, -8, 10), (16, -16, 20)):
        img = np.full((520, 720), 255, np.uint8)
        _paste(img, marks[0], 80 + dx0, 400)
        _paste(img, marks[1], 280, 400 + dy1)
        _paste(img, marks[2], 80 + dx2, 40)
        _paste(img, marks[3], 560, 40)
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return frames


def test_near_center_is_median():
    print("近端 ID 0、1 用多格中位數，不用第 0 格；全畫面找到的遠端仍用第一次")
    frames = _frames_near_shift()
    dets = [C.detect_aruco_ids(f) for f in frames]
    for i, d in enumerate(dets):
        check(set(d) == {0, 1, 2, 3}, f"第 {i} 格四點都找得到（實際 {sorted(d)}）")
    med = {k: np.median([d[k] for d in dets], 0) for k in (0, 1, 2)}
    first = {k: np.asarray(dets[0][k], float) for k in range(4)}
    check(float(np.linalg.norm(med[0] - first[0])) > 3, "ID 0 的中位數離第 0 格超過 3 像素")
    check(float(np.linalg.norm(med[1] - first[1])) > 3, "ID 1 的中位數離第 0 格超過 3 像素")
    check(float(np.linalg.norm(med[2] - first[2])) > 3, "ID 2 有移動，用來確認全畫面遠端沒有改成中位數")

    orig = (P.read_frame, P.video_info, C.click_points)
    h, w = frames[0].shape[:2]

    def read_frame(path, index=0):
        return frames[int(index)]

    def video_info(path):
        return {"fps": 4.0, "n_frames": len(frames), "width": w, "height": h}

    def no_click(*a, **k):
        raise AssertionError("四點都找得到，不該點選")

    P.read_frame, P.video_info, C.click_points = read_frame, video_info, no_click
    fd, path = tempfile.mkstemp(suffix="_calib.json")
    os.close(fd)
    try:
        CLI.do_calibrate("synthetic.mov", (1.0, 5.0), "auto", None, path)
        raw = json.load(open(path, encoding="utf-8"))
    finally:
        P.read_frame, P.video_info, C.click_points = orig
        if os.path.exists(path):
            os.remove(path)
    pts = np.asarray(raw["image_points_px"], float)
    for i in (0, 1):
        err = float(np.linalg.norm(pts[i] - med[i]))
        check(err < 1e-6, f"ID {i} 中心是多格中位數（誤差 {err:.4f}）")
        check(float(np.linalg.norm(pts[i] - first[i])) > 3, f"ID {i} 不是第 0 格")
        check(raw["marker_detection"][str(i)]["n"] == 3, f"ID {i} 用了 3 格")
    for i in (2, 3):
        err = float(np.linalg.norm(pts[i] - first[i]))
        check(err < 1e-6, f"全畫面找到的 ID {i} 仍是第一次的位置（誤差 {err:.4f}）")
    check(raw["method"] == "aruco", f"四點都是全畫面偵測時 method 仍是 aruco（實際 {raw['method']}）")


def test_far_search_scans_whole_window():
    print("遠端放大不再在每個 ID 滿 15 次時停")
    img = np.full((360, 640), 255, np.uint8)
    for i, x in ((2, 40), (3, 420)):
        _paste(img, _marker(i, 100), x, 40)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    roi = (0, 0, bgr.shape[1], bgr.shape[0])
    one = C.search_far_markers([bgr], wanted=(2, 3), roi=roi)
    check(2 in one and 3 in one and one[2]["n"] >= 1, f"單格找得到遠端（實際 {sorted(one)}）")
    per = int(one[2]["n"])
    n_frames = max(12, (15 // per) + 3)
    pulled = {"n": 0}

    def gen():
        for _ in range(n_frames):
            pulled["n"] += 1
            yield bgr

    found = C.search_far_markers(gen(), wanted=(2, 3), roi=roi)
    check(pulled["n"] == n_frames, f"送進去的 {n_frames} 格都掃了（實際 {pulled['n']}）")
    check(found[2]["n"] == per * n_frames and found[3]["n"] == per * n_frames,
          f"次數是每格 {per} 次乘上全部格數（ID 2 實際 {found[2]['n']}）")
    check(found[2]["n"] > 15 and found[3]["n"] > 15, "次數超過舊的 15 次上限")
    check(C.FAR_SAMPLE_END == 240 and C.FAR_SAMPLE_STEP == 2, "正面校正的固定上限仍是前 240 格、每隔一格")


def test_recalc_keeps_point_methods():
    print("沿用前一支時 method 是 reused_previous，source_method 記下原本的方法")
    calib = {
        "method": "aruco+click",
        "image_points_px": [[0, 0], [1, 1], [2, 2], [3, 3]],
        "marker_detection": {
            "0": {"method": "aruco", "n": 3, "sd_px": [0.1, 0.2], "center_px": [0, 0]},
            "1": {"method": "aruco", "n": 3, "sd_px": [0.0, 0.1], "center_px": [1, 1]},
            "2": {"method": "aruco_roi_upscale", "n": 8, "sd_px": [0.3, 0.4], "center_px": [2, 2]},
            "3": {"method": "click", "n": 1, "sd_px": None, "center_px": [3, 3]},
        },
    }
    rec = C.records_from_saved_detection(calib)
    check(rec is not None, "讀得到各點的 method")
    check([rec[k]["method"] for k in range(4)] == ["aruco", "aruco", "aruco_roi_upscale", "click"],
          "保留 aruco、aruco_roi_upscale、click")
    text = C.format_calib_method(rec)
    check("aruco+click" not in text, f"重算出的 method 不是 aruco+click（{text}）")
    check("2:aruco_roi_upscale" in text and "3:click" in text, "逐點寫出放大與點選")
    check(C.records_from_saved_detection({"method": "aruco+click", "image_points_px": [[0, 0]] * 4}) is None,
          "只有整組 aruco+click、沒有逐點紀錄時，不臆造每個點的 method")

    img = np.full((480, 640), 255, np.uint8)
    _paste(img, _marker(0, 100), 40, 300)
    _paste(img, _marker(1, 100), 360, 300)
    frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    found = C.detect_aruco_ids(frame)
    check(set(found) == {0, 1}, f"這張合成圖只有近端（實際 {sorted(found)}）")
    h, w = frame.shape[:2]
    prev = {
        "image_size_px": [w, h],
        "image_points_px": [found[0].tolist(), found[1].tolist(), [12.0, 34.0], [56.0, 78.0]],
        "marker_detection": {
            "0": {"method": "aruco", "n": 2, "sd_px": [0.1, 0.2], "center_px": found[0].tolist()},
            "1": {"method": "aruco", "n": 2, "sd_px": [0.0, 0.1], "center_px": found[1].tolist()},
            "2": {"method": "aruco_roi_upscale", "n": 36, "sd_px": [0.3, 0.4], "center_px": [12.0, 34.0]},
            "3": {"method": "click", "n": 1, "sd_px": None, "center_px": [56.0, 78.0]},
        },
    }
    legacy = {
        "image_size_px": [w, h],
        "image_points_px": [found[0].tolist(), found[1].tolist(), [12.0, 34.0], [56.0, 78.0]],
        "method": "aruco+click",
    }
    orig = (P.read_frame, P.video_info, P.open_video, C.click_points)

    def read_frame(path, index=0):
        return frame

    def video_info(path):
        return {"fps": 30.0, "n_frames": 1, "width": w, "height": h}

    class _Cap:
        def read(self):
            return True, frame

        def release(self):
            pass

        def set(self, *a, **k):
            return True

        def get(self, prop):
            return 0.0

    def open_video(path):
        return _Cap()

    def no_click(*a, **k):
        raise AssertionError("前一支已有遠端的點，不該打開點選視窗")

    P.read_frame, P.video_info, P.open_video, C.click_points = read_frame, video_info, open_video, no_click

    def _run(previous):
        fd, path = tempfile.mkstemp(suffix="_calib.json")
        os.close(fd)
        try:
            CLI.do_calibrate("synthetic.mov", (1.0, 5.0), "auto", None, path, previous)
            return json.load(open(path, encoding="utf-8"))
        finally:
            if os.path.exists(path):
                os.remove(path)

    try:
        raw = _run(prev)
        legacy_raw = _run(legacy)
    finally:
        P.read_frame, P.video_info, P.open_video, C.click_points = orig
    det = raw["marker_detection"]
    check(det["2"]["method"] == "reused_previous" and det["3"]["method"] == "reused_previous",
          "沿用的點，method 維持 reused_previous")
    check(det["2"]["source_method"] == "aruco_roi_upscale(n=36)",
          f"ID 2 的 source_method 是原本的方法（實際 {det['2'].get('source_method')}）")
    check(det["3"]["source_method"] == "click(n=1)",
          f"ID 3 的 source_method 是 click(n=1)（實際 {det['3'].get('source_method')}）")
    check(det["0"]["method"] == "aruco" and det["1"]["method"] == "aruco", "這一支新找到的近端仍是 aruco")
    check("source_method" not in det["0"] and "source_method" not in det["1"], "新偵測的點不另寫 source_method")
    check("reused_previous" in raw["method"], f"校正檔 method 看得出是沿用（{raw['method']}）")
    check("aruco+click" not in raw["method"], f"校正檔 method 不是 aruco+click（{raw['method']}）")
    got = np.asarray(raw["image_points_px"], float)
    check(np.allclose(got[2], [12.0, 34.0]) and np.allclose(got[3], [56.0, 78.0]), "沿用的是前一支的座標")
    old = legacy_raw["marker_detection"]
    check(old["2"]["method"] == "reused_previous" and old["2"]["source_method"] is None,
          "舊檔沒有逐點紀錄時，source_method 是 null")
    check(old["3"]["source_method"] is None, "另一個沿用點的 source_method 也是 null")


def main():
    test_version_constant()
    test_reports_lead_with_version()
    test_near_center_is_median()
    test_far_search_scans_whole_window()
    test_recalc_keeps_point_methods()
    print("0.4.2 自我檢查通過 ✔")


if __name__ == "__main__":
    main()
