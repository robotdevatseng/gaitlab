"""安裝後自我檢查：用 20 趟已知真值的合成行走驗證計算是否正確（不需要影片、不需要網路）。

執行：python tests/check_install.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from tests.synth import simulate, Camera  # noqa: E402
from gaitlab.gait import analyze, Params  # noqa: E402


def main():
    # (步寬 m, 步速 m/s, 步頻, fps, 像素雜訊) — 包含慢速、30 fps、較大雜訊的情境
    scen = [(0.10, 1.1, 112, 60, 2), (0.15, 0.9, 104, 60, 2), (0.20, 0.6, 90, 60, 4),
            (0.25, 0.45, 80, 60, 4), (0.08, 1.3, 120, 60, 2), (0.14, 0.9, 104, 30, 2)]
    e_sw, e_spd, e_cad = [], [], []
    for i, (sw, spd, cad, fps, noise) in enumerate(scen):
        for rep in range(4):
            cam = Camera(pos=(0.5, 0.0, 1.2), pitch_deg=5, f=1500, size=(1080, 1920))
            lm, H, tr = simulate(step_width=sw, sw_sd=0.03, speed=spd, cadence=cad, fps=fps, px_noise=noise,
                                 seed=100 * i + rep, cam=cam, zone=(2.5, 7.5), y_start=9.5, y_end=1.0)
            s = analyze(lm, H, 1.0, 5.0, Params(cam_height_m=1.2)).summary
            e_sw.append(s["step_width_mean_cm"] - tr["step_width_mean_cm"])
            e_spd.append((s["gait_speed_cm_s"] - tr["gait_speed_cm_s"]) / tr["gait_speed_cm_s"] * 100)
            e_cad.append(s["cadence_steps_min"] - tr["cadence_steps_min"])
    e_sw, e_spd, e_cad = np.array(e_sw), np.array(e_spd), np.array(e_cad)
    n = len(e_sw)
    print(f"步寬誤差：平均 {e_sw.mean():+.2f} cm，最大 {np.abs(e_sw).max():.2f} cm（{n} 趟合成行走）")
    print(f"步速誤差：平均 {e_spd.mean():+.1f}%，最大 {np.abs(e_spd).max():.1f}%")
    print(f"步頻誤差：平均 {e_cad.mean():+.1f} 步/分，最大 {np.abs(e_cad).max():.1f} 步/分")
    ok = (abs(e_sw.mean()) < 0.5 and np.abs(e_sw).max() < 1.5 and abs(e_spd.mean()) < 3
          and np.abs(e_spd).max() < 6 and np.abs(e_cad).max() < 5)
    # 側面：擬真合成資料（腳跟落地帶速度、站立末期腳跟上提、畫面邊界）
    from tests.synth_side_real import simulate_side_real
    from gaitlab.side import analyze_side
    e_st, e_ds = [], []
    for i, (spd, cad, sf, fps) in enumerate([(1.0, 110, 0.62, 60), (0.6, 90, 0.66, 60), (0.4, 76, 0.70, 60), (1.0, 110, 0.62, 30)]):
        for rep in range(2):
            lm, mk, tr, _ = simulate_side_real(seed=10 * i + rep, speed=spd, cadence=cad, stance_frac=sf, fps=fps,
                                               v_contact=0.6, direction=-1 if rep == 0 else 1)
            ss = analyze_side(lm, mk, 2.0).summary
            e_st.append(ss["stance_pct"] - tr["stance_pct"])
            e_ds.append(ss["double_support_pct"] - tr["double_support_pct"])
    e_st, e_ds = np.array(e_st), np.array(e_ds)
    print(f"側面 站立期誤差：平均 {e_st.mean():+.1f}，最大 {np.abs(e_st).max():.1f} 個百分點（8 趟）")
    print(f"側面 雙腳支撐期誤差：平均 {e_ds.mean():+.1f}，最大 {np.abs(e_ds).max():.1f} 個百分點")
    ok = ok and np.abs(e_st).max() < 3 and np.abs(e_ds).max() < 4
    print("檢查通過 ✔" if ok else "檢查未通過 ✘ — 請回報")
    try:
        import mediapipe  # noqa: F401
        print(f"mediapipe {mediapipe.__version__} 已安裝；姿態模型隨套件內建，可離線使用。")
    except Exception as e:  # pragma: no cover
        print("mediapipe 無法載入：", e)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
