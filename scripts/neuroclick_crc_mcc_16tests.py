"""Sect. 4.5 threshold-matched MCC comparisons as reported in the paper: both ensemble strategies x both
components x four horizons = 16 paired two-sided Wilcoxon tests, Holm-corrected as one family, with
Student-t 95% intervals. Reads results/crc_extension/crc_ensemble_mcc_threshold_matched.csv (written by
neuroclick_crc_extension_tests.py) and writes results/crc_extension/crc_mcc_16_tests.csv.
Run from the repository root: python scripts/neuroclick_crc_mcc_16tests.py"""
import numpy as np, pandas as pd
from scipy import stats
SRC = "results/crc_extension/crc_ensemble_mcc_threshold_matched.csv"
OUT = "results/crc_extension/crc_mcc_16_tests.csv"

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

df = pd.read_csv(SRC); rows = []
for h in ["first1", "first2", "first3", "full"]:
    piv = df[df.horizon == h].pivot(index="subject", columns="model", values="mcc").dropna()
    for ens in ["ensemble_average", "ensemble_stacked"]:
        for comp in ["gru_probability", "baseline_probability"]:
            d = piv[ens] - piv[comp]; n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
            t = stats.t.ppf(0.975, n - 1); p = stats.wilcoxon(d).pvalue
            rows.append(dict(horizon=h, ensemble=ens, comparator=comp, n_subjects=n, mean_difference=mu,
                             ci95_low=mu - t * se, ci95_high=mu + t * se, raw_p=p))
res = pd.DataFrame(rows); res["holm_p"] = holm(res.raw_p)
res.to_csv(OUT, index=False)
print(res.round(4).to_string(index=False))
