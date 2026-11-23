"""工單 v3（A–D）的自我檢查。執行：python tests/test_workorder_v3.py

不需要影片與姿態模型；只檢查記錄方式與衝突處理的邏輯。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from gaitlab import auto, gait as G, side as S, version as V  # noqa: E402
from gaitlab.side import SideResult  # noqa: E402
from tests.synth import simulate  # noqa: E402


def check(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        raise SystemExit(f"失敗：{msg}")


def test_a_barefoot():
    print("A. 赤腳旗標回寫")
    check(auto.resolve_barefoot("否", "S02_赤腳") == (True, "資料夾名稱"), "設定檔寫否＋資料夾名含赤腳 → 是，來源＝資料夾名稱")
    check(auto.resolve_barefoot("否", "S02") == (False, "設定檔"), "設定檔寫否＋資料夾名不含赤腳 → 否，來源＝設定檔")
    check(auto.resolve_barefoot("是", "S02") == (True, "設定檔"), "設定檔寫是＋資料夾名不含赤腳 → 是，來源＝設定檔")
    check(auto.resolve_barefoot("是", "S02_barefoot") == (True, "設定檔與資料夾名稱"), "設定檔寫是＋資料夾名含 barefoot → 是，來源＝設定檔與資料夾名稱")
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "使用的設定.txt")
        st = {"量測區寬度": 1.0, "量測區長度": 5.0, "正面鏡頭高度": 1.2, "側面標記距離": 2.04, "赤腳": "否"}
        auto.write_used_settings(p, st, True, "資料夾名稱", {"commit_short": "abc1234", "program_version": "x"})
        txt = open(p, encoding="utf-8").read()
        active = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
        check("赤腳 = 是" in active and "赤腳來源 = 資料夾名稱" in active, "使用的設定.txt 寫的是解析後的實際值與來源")
        check(sum(ln.startswith("赤腳 =") for ln in active) == 1, "有效的「赤腳 =」只有一行（原文以「# 原文｜」標示，不會被誤讀）")
        check("程式版本 = abc1234" in active, "使用的設定.txt 有程式版本")
    finally:
        shutil.rmtree(d)


def test_b_version():
    print("B. 版本寫入")
    v = V.get_version()
    root = V.ROOT
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = ""
    if head and os.path.exists(os.path.join(root, ".git")):
        check(v["commit"] == head, "commit 與 git rev-parse HEAD 相同")
        check(head.startswith(v["commit_short"]) and len(v["commit_short"]) == 7, "短 hash 是完整 hash 的前 7 碼")
        check(V._read_head_without_git(root) == head, "沒有 git 指令時，直接讀 .git 也得到同一個 hash")
    else:
        print("  （此資料夾不是 git 工作複本，略過 hash 比對）")
    d = tempfile.mkdtemp()
    try:
        nv = V.get_version(d)
        check(nv["commit"] == "未取得" and nv["commit_short"] == "未取得", "不是 git 目錄 → 寫明「未取得」，不空白、不虛構")
        check(bool(nv["program_version"]), "版本號仍有值")
    finally:
        shutil.rmtree(d)


def test_c_step_width_factor():
    print("C. 步寬係數寫入")
    for bare, kl, kw in ((False, 0.975, 0.925), (True, 1.18 / 1.2, (1.2 - 0.07) / 1.2)):
        lm, H, truth = simulate(seed=3)
        prm = G.make_params(1.2, "ankle", bare)
        res = G.analyze(lm, H, 1.0, 5.0, prm)
        s = res.summary
        check(abs(s["height_correction_factor"] - kl) < 1e-9, f"{'赤腳' if bare else '穿鞋'}：步長係數 {kl:.6f}")
        check(abs(s["step_width_height_correction_factor"] - kw) < 1e-9, f"{'赤腳' if bare else '穿鞋'}：步寬係數 {kw:.6f}（獨立欄位）")
        # 由逐步資料反推：檔案步寬 ÷ 幾何步寬 = 寫入的係數
        pos = np.array([[r["heel_x_m"], r["heel_y_m"]] for r in res.steps])
        n_ok = n_small = 0
        for i in range(1, len(res.steps) - 1):
            sw = res.steps[i]["step_width_cm"]
            if np.isnan(sw) or res.steps[i]["segment"] != res.steps[i - 1]["segment"] or res.steps[i]["segment"] != res.steps[i + 1]["segment"]:
                continue
            a, b, c = pos[i - 1], pos[i], pos[i + 1]
            dd = c - a
            geom = abs(dd[0] * (b - a)[1] - dd[1] * (b - a)[0]) / np.linalg.norm(dd) * 100
            if geom < 0.5:
                n_small += 1
                continue
            assert abs(sw / geom - kw) <= 1e-6 * kw, (sw, geom, kw)
            n_ok += 1
        check(n_ok >= 3, f"  反推 {n_ok} 步皆等於寫入係數（幾何步寬 < 0.5 cm 而排除 {n_small} 步）")


def _fake(near, expected, kL, kR, conflict=None):
    c = (near != expected) if conflict is None else conflict
    summ = {"walking_speed_cm_s": 100.0, "stance_pct": 60.0, "swing_pct": 40.0, "double_support_pct": 20.0,
            "stance_time_s": 0.6, "swing_time_s": 0.4, "stride_time_s": 1.0, "step_length_cm": 60.0,
            "n_strides": 4, "n_steps": 5, "knee_peak_swing_L_deg": kL, "knee_peak_swing_R_deg": kR,
            "near_side": near, "near_side_expected": expected, "near_side_conflict": c}
    return SideResult(summ, {}, [{"step_time_s": 0.5}] * 5, np.zeros(1), {})


def test_d_near_side_conflict():
    print("D. near_side 矛盾處理")
    rs = [_fake("R", "L", 62.1, 53.3), _fake("L", "L", 68.7, 65.1), _fake("R", "R", 59.1, 62.9), _fake("R", "R", 57.8, 70.1)]
    names = ["T1", "T2", "T3", "T4"]
    agg = S.aggregate_side(rs, names)
    check(agg["knee_peak_excluded_conflict_trials"] == ["T1"], "衝突趟被列為排除")
    check(agg["knee_peak_included_trials"] == {"L": ["T2"], "R": ["T3", "T4"]}, "寫明膝峰值納入的趟")
    check(abs(agg["knee_peak_swing_R_deg"] - (62.9 + 70.1) / 2) < 1e-9 and abs(agg["knee_peak_swing_L_deg"] - 68.7) < 1e-9,
          "衝突趟的膝峰值不進組平均（不依 near_side，也不依 expected）")
    check("T1" in agg["knee_peak_note"], "組平均附註寫出衝突趟")
    check(abs(agg["stance_pct"] - 60.0) < 1e-9 and agg["n_strides"] == 16 and agg["n_steps"] == 20, "衝突趟的其餘指標照常納入")
    rs2 = [_fake("L", "L", 68.7, 65.1), _fake("R", "R", 59.1, 62.9)]
    agg2 = S.aggregate_side(rs2, ["A", "B"])
    check(agg2["knee_peak_excluded_conflict_trials"] == [] and agg2["knee_peak_note"] == "", "沒有衝突時不排除任何趟、沒有附註")
    check(abs(agg2["knee_peak_swing_L_deg"] - 68.7) < 1e-9 and abs(agg2["knee_peak_swing_R_deg"] - 62.9) < 1e-9, "沒有衝突時膝峰值算法不變")


if __name__ == "__main__":
    test_a_barefoot()
    test_b_version()
    test_c_step_width_factor()
    test_d_near_side_conflict()
    print("工單 v3 自我檢查全部通過 ✔")
