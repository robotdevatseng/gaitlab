"""用合成資料產生一份示範報告（不需要影片），並印出真值 vs 估計。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.synth import simulate
from gaitlab import gait as G, report as R

out = sys.argv[1] if len(sys.argv) > 1 else "demo_output"
os.makedirs(out, exist_ok=True)
results = []
print(f"{'趟':6s}{'步寬真值':>10s}{'步寬估計':>10s}{'步速真值':>10s}{'步速估計':>10s}{'步頻真值':>10s}{'步頻估計':>10s}")
for k in range(3):
    lm, H, truth = simulate(step_width=0.18, sw_sd=0.04, speed=0.75, cadence=96, seed=10 + k)
    res = G.analyze(lm, H, 1.0, 6.0)
    s = res.summary
    print(f"trial{k+1:<2d}{truth['step_width_mean_cm']:10.1f}{s['step_width_mean_cm']:10.1f}"
          f"{truth['gait_speed_cm_s']:10.1f}{s['gait_speed_cm_s']:10.1f}{truth['cadence_steps_min']:10.1f}{s['cadence_steps_min']:10.1f}")
    results.append((f"trial{k+1}（合成）", res))
agg = G.aggregate([r for _, r in results])
meta = {"subject": "SYNTH-DEMO", "created": R.now_str(), "cam_height_m": 1.1}
R.write_steps_csv(os.path.join(out, "steps.csv"), results)
R.write_summary_json(os.path.join(out, "summary.json"), meta, agg, results)
R.write_html(os.path.join(out, "report.html"), meta, agg, results, 1.0, 6.0)
print("報告：", os.path.join(out, "report.html"))
