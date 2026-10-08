"""Run from ~/24PHD1314/ICAIN2026_NeuroClick_Hazard:  python verify_remaining.py"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = stats.wilcoxon(d, zero_method="wilcox").pvalue
    return m, m - t * se, m + t * se, p, n


print("=== (1) first1 ensemble MCC, exact ===")
c = pd.read_csv("crc_ensemble_mcc_threshold_matched.csv")
w = c.pivot_table(index=["horizon", "subject"], columns="model", values="mcc").reset_index()
x = w[w.horizon == "first1"]
for comp in ["gru_probability", "baseline_probability"]:
    m, lo, hi, p, n = paired(x["ensemble_average"], x[comp])
    print(f"avg vs {comp}: {m:+.4f} [{lo:.4f}, {hi:.4f}] raw p={p:.2e} n={n}")

print("\n=== (2) Table 5 secondary rows from per_subject_metrics.csv ===")
s = pd.read_csv("outputs/statistics/neuroclick_primary_v1/per_subject_metrics.csv")
print("models:", sorted(s.model.unique()))
print("horizons:", sorted(s.horizon.unique()))
P = s.pivot_table(index=["horizon", "subject"], columns="model", values="pr_auc").reset_index()
tests = [
    ("first2", "behavior", "logreg_dwell_propensity"),
    ("first3", "behavior", "logreg_dwell_propensity"),
    ("first1", "behavior_et", "behavior"),
    ("first2", "behavior_et", "behavior"),
]
rows = []
for h, a, b in tests:
    if a not in P.columns or b not in P.columns:
        print(f"  SKIP {h} {a} vs {b}: column missing")
        continue
    xx = P[P.horizon == h].dropna(subset=[a, b])
    m, lo, hi, p, n = paired(xx[a], xx[b])
    rows.append((h, a, b, m, lo, hi, p, n))
if rows:
    holm = multipletests([r[6] for r in rows], method="holm")[1]
    for r, hp in zip(rows, holm):
        print(f"{r[0]:6} {r[1]} vs {r[2]}: n={r[7]} {r[3]:+.4f} [{r[4]:.4f}, {r[5]:.4f}] Holm p={hp:.2f}")
print("Expected: +0.0176 [-0.0015,0.0368] 0.33 | +0.0150 [-0.0106,0.0406] 0.40 | "
      "+0.0037 [-0.0122,0.0195] 1.0 | -0.0065 [-0.0243,0.0113] 1.0")

print("\n=== (3) final_run_04 layout ===")
d = "cache/final_run_04"
print(os.listdir(d))
f = pd.read_csv(os.path.join(d, "losocv_predictions.csv"), nrows=3)
print(list(f.columns))
print(f.to_string())
