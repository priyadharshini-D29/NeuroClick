"""Camera-ready extension analyses for NeuroClick (ICAIN 2026, Paper 456).
Consumes frozen outputs; writes nothing into them. Procedure matches the primary
manifest: paired two-sided Wilcoxon, 95% Student-t intervals, Holm per family."""
import numpy as np
import pandas as pd
from scipy import stats

METRICS_CSV  = "outputs/statistics/neuroclick_primary_v1/per_subject_metrics.csv"
ENSEMBLE_CSV = "cache/final_run_ensemble/ensemble_predictions.csv"

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj

def paired(df, horizon, model_a, model_b, metric="pr_auc"):
    sub = df[df.horizon == horizon]
    A = sub[sub.model == model_a].set_index("subject")[metric]
    B = sub[sub.model == model_b].set_index("subject")[metric]
    common = A.index.intersection(B.index)
    d = (A.loc[common] - B.loc[common]).values
    n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    _, p = stats.wilcoxon(d)
    return dict(n=n, diff=mu, lo=mu - tcrit * se, hi=mu + tcrit * se, p=p,
                mean_a=A.loc[common].mean(), mean_b=B.loc[common].mean())

def analysis_A_and_B():
    df = pd.read_csv(METRICS_CSV)
    tests = [("first2", "behavior", "logreg_dwell_propensity"),
             ("first3", "behavior", "logreg_dwell_propensity"),
             ("first1", "behavior_et", "behavior"),
             ("first2", "behavior_et", "behavior")]
    res = [paired(df, *t) for t in tests]
    adj = holm([r["p"] for r in res])
    print("=== A. New confirmatory family (PR AUC, Holm across 4) ===")
    for (h, a, b), r, pH in zip(tests, res, adj):
        print(f"{a} vs {b} @ {h}: n={r['n']}, means {r['mean_a']:.4f} vs {r['mean_b']:.4f}, "
              f"diff={r['diff']:+.4f}, CI [{r['lo']:+.4f},{r['hi']:+.4f}], "
              f"raw p={r['p']:.5f}, Holm p={pH:.5f}")
    print("\n=== B. Dispersion (participant PR AUC mean/SD) ===")
    print(df.groupby(["model", "horizon"]).pr_auc.agg(["mean", "std", "count"]).round(4).to_string())

def mcc_all_thresholds(y, s):
    order = np.argsort(s); y_s = y[order]; s_s = s[order]
    P = y.sum(); N = len(y) - P
    cum_pos = np.concatenate([[0], np.cumsum(y_s)]); idx = np.arange(len(y) + 1)
    FN = cum_pos; TN = idx - cum_pos; TP = P - FN; FP = N - TN
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = np.where(denom > 0, (TP * TN - FP * FN) / denom, 0.0)
    thr = np.concatenate([[-np.inf], (s_s[:-1] + s_s[1:]) / 2, [np.inf]])
    keep = np.concatenate([[True], s_s[1:] != s_s[:-1], [True]])
    return mcc[keep], thr[keep]

def mcc_at(y, s, t):
    pred = (s >= t).astype(int)
    TP = ((pred == 1) & (y == 1)).sum(); TN = ((pred == 0) & (y == 0)).sum()
    FP = ((pred == 1) & (y == 0)).sum(); FN = ((pred == 0) & (y == 1)).sum()
    d = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    return (TP * TN - FP * FN) / d if d > 0 else 0.0

def analysis_C():
    df = pd.read_csv(ENSEMBLE_CSV)
    models = ["gru_probability", "baseline_probability", "ensemble_average", "ensemble_stacked"]
    rows = []
    for h in ["first1", "first2", "first3", "full"]:
        sub = df[df.horizon == h]
        for m in models:
            for s_id in sub.subject.unique():
                pool = sub[sub.subject != s_id]; held = sub[sub.subject == s_id]
                if held.label.nunique() < 2:
                    continue
                mccs, thrs = mcc_all_thresholds(pool.label.values, pool[m].values)
                rows.append(dict(horizon=h, model=m, subject=s_id,
                                 mcc=mcc_at(held.label.values, held[m].values,
                                            thrs[np.argmax(mccs)])))
    res = pd.DataFrame(rows)
    res.to_csv("crc_ensemble_mcc_threshold_matched.csv", index=False)
    print("\n=== C. Threshold-matched MCC (participant mean/SD) ===")
    print(res.groupby(["horizon", "model"]).mcc.agg(["mean", "std", "count"]).round(4).to_string())
    tests, labels = [], []
    for h in ["first1", "first2", "first3", "full"]:
        piv = res[res.horizon == h].pivot(index="subject", columns="model", values="mcc").dropna()
        best = "ensemble_average" if piv["ensemble_average"].mean() >= piv["ensemble_stacked"].mean() \
               else "ensemble_stacked"
        for comp in ["gru_probability", "baseline_probability"]:
            d = piv[best] - piv[comp]
            _, p = stats.wilcoxon(d)
            n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
            tcrit = stats.t.ppf(0.975, n - 1)
            tests.append(p); labels.append((h, best, comp, n, mu, mu - tcrit * se, mu + tcrit * se))
    adj = holm(tests)
    print("\nPaired Wilcoxon, best ensemble vs component (Holm across 8):")
    for (h, best, comp, n, mu, lo, hi), praw, pH in zip(labels, tests, adj):
        print(f"{h}: {best} vs {comp}: n={n}, diff={mu:+.4f}, "
              f"CI [{lo:+.4f},{hi:+.4f}], raw p={praw:.4f}, Holm p={pH:.4f}")

if __name__ == "__main__":
    analysis_A_and_B()
    analysis_C()
