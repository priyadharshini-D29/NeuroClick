"""Sensitivity check for the information-matched baseline (camera-ready, ICAIN 2026 paper 456).

The classical behaviour summary used by `logreg_behavior` and `logreg_dwell_propensity`
(03_run_classical_losocv_benchmarks.py, behavior_features) carries ten values, the last two
being the mean EEG and eye-tracking validity fractions. Behaviour-only NeuroClick never sees
those two values (04_run_neuroclick_hazard.py, FeatureStore: "Modality-validity fractions
accompany only their own modality"). This script re-evaluates both logistic baselines with the
two validity columns removed, under exactly the same leave-one-participant-out procedure,
pipeline and cross-fitted propensity as script 03, and compares the result with NeuroClick's
saved per-participant PR AUC.

Inputs : the visit-feature cache written by script 03 (classical_visit_features_causal_v1.npz,
         SHA-256 41c486e9...) and results/neuroclick_primary_v1/per_subject_metrics.csv.
Usage  : python scripts/neuroclick_crc_matched_baseline.py <path to classical_visit_features_causal_v1.npz>
Output : results/crc_extension/matched_baseline_no_validity.csv and console summary.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "03_run_classical_losocv_benchmarks.py")
bench = importlib.util.module_from_spec(spec); sys.modules["bench"] = bench; spec.loader.exec_module(bench)

npz = Path(sys.argv[1])
with np.load(npz, allow_pickle=False) as data:
    arrays = {k: np.asarray(data[k]) for k in data.files}
config = bench.RunConfig(cache=Path("."), output=Path("."), feature_cache=npz, horizons=("first1", "first2", "first3", "full"),
                         models=("logreg_behavior", "logreg_dwell_propensity"))
ref = pd.read_csv(ROOT / "results" / "neuroclick_primary_v1" / "per_subject_metrics.csv")

def run(horizon, matrix, columns, with_propensity):
    y = matrix["label"].astype(np.uint8); subjects = matrix["subject"].astype(str)
    out_prob = np.empty(len(y));
    for held in sorted(np.unique(subjects)):
        test = subjects == held; train = ~test; y_train = y[train]; groups = subjects[train]
        x_train = matrix["behavior"][train][:, columns]; x_test = matrix["behavior"][test][:, columns]
        if with_propensity:
            p_train = bench.cross_fitted_propensity_feature(matrix["product_id"][train], y_train, groups, config.product_smoothing, config.inner_folds)
            p_test = bench.product_propensity_probability(matrix["product_id"][train], y_train, matrix["product_id"][test], config.product_smoothing)
            x_train = np.column_stack([x_train, bench.probability_to_logit(p_train)]); x_test = np.column_stack([x_test, bench.probability_to_logit(p_test)])
        est = clone(bench.make_logistic_pipeline(config, x_train.shape[1])); est.fit(x_train.astype(np.float32), y_train)
        out_prob[test] = est.predict_proba(x_test.astype(np.float32))[:, 1]
    pooled = float(average_precision_score(y, out_prob))
    per_subject = {s: float(average_precision_score(y[subjects == s], out_prob[subjects == s])) for s in np.unique(subjects)}
    return pooled, per_subject

rows = []
for horizon in config.horizons:
    matrix = bench.build_horizon_matrix(arrays, horizon)
    nc = ref[(ref.model == "behavior") & (ref.horizon == horizon)].set_index("subject").pr_auc
    for model, with_p in [("logreg_behavior", False), ("logreg_dwell_propensity", True)]:
        for variant, cols in [("as published (10 values)", list(range(10))), ("matched (validity fractions removed)", list(range(8)))]:
            pooled, per = run(horizon, matrix, cols, with_p)
            per = pd.Series(per); common = per.index.intersection(nc.index); d = (nc.loc[common] - per.loc[common]).values
            n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1); p = stats.wilcoxon(d).pvalue
            rows.append(dict(horizon=horizon, model=model, variant=variant, pooled_pr_auc=pooled, participant_mean_pr_auc=per.mean(),
                             neuroclick_minus_baseline=mu, ci95_low=mu - t * se, ci95_high=mu + t * se, wilcoxon_raw_p=p, n_subjects=n))
            print(f"{horizon:6s} {model:24s} {variant:38s} pooled {pooled:.4f} mean {per.mean():.4f} NC-baseline {mu:+.4f} [{mu - t*se:+.4f},{mu + t*se:+.4f}] p={p:.4f}", flush=True)
res = pd.DataFrame(rows); out = ROOT / "results" / "crc_extension" / "matched_baseline_no_validity.csv"
res.to_csv(out, index=False); print("written", out)
