#!/usr/bin/env python3
"""Combine NeuroClick's sequential risk with the dwell+propensity logistic
baseline and test, honestly, whether the combination beats either component
alone.

Two combination strategies are tried, both leakage-free with respect to the
held-out subject in every fold:

  average   - unweighted average of the two probabilities in logit space.
              Zero free parameters; cannot overfit.
  stacked   - a per-fold logistic regression on [gru_logit, baseline_logit],
              refit for every held-out subject using ONLY the other 41
              subjects' out-of-fold predictions. Because both input
              predictions were already produced out-of-fold with respect to
              every subject (strict LOSOCV in scripts 03/04), and the
              stacking model for held-out subject S is fit on rows from
              subjects other than S only, S's own label never contributes -
              directly or indirectly - to the stacked prediction made for S.

Join key: 04's `bag_index` and 03's `original_bag_index` both index the same
underlying feature cache (`classical_visit_features_causal_v1.npz` or
equivalent), so they refer to the same physical visit-history bag and are
joined directly, per horizon.

This script only produces new predictions and their significance tests. It
never touches or reruns the base models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

HORIZONS = ("first1", "first2", "first3", "full")
EPS = 1e-4


def to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def to_probability(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit))


def load_neural(path: Path, condition: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["condition"] == condition].copy()
    if frame.empty:
        raise ValueError(
            f"No rows with condition={condition!r} in {path}; available: "
            f"{sorted(pd.read_csv(path)['condition'].unique())}"
        )
    return frame.rename(
        columns={
            "heldout_subject": "subject",
            "bag_index": "bag_index",
            "label": "label",
            "probability": "gru_probability",
        }
    )[["subject", "horizon", "bag_index", "label", "gru_probability"]]


def load_classical(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["model"] == model].copy()
    if frame.empty:
        raise ValueError(
            f"No rows with model={model!r} in {path}; available: "
            f"{sorted(pd.read_csv(path)['model'].unique())}"
        )
    return frame.rename(
        columns={
            "held_out_subject": "subject",
            "original_bag_index": "bag_index",
            "label_buy": "label",
            "probability_buy": "baseline_probability",
        }
    )[["subject", "horizon", "bag_index", "label", "baseline_probability"]]


def merge_predictions(neural: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    merged = neural.merge(
        classical,
        on=["horizon", "bag_index"],
        suffixes=("_gru", "_baseline"),
        how="inner",
        validate="one_to_one",
    )
    mismatched_subject = merged["subject_gru"] != merged["subject_baseline"]
    mismatched_label = merged["label_gru"] != merged["label_baseline"]
    if mismatched_subject.any() or mismatched_label.any():
        raise ValueError(
            "Join sanity check failed: subject or label disagreed between "
            "the neural and classical predictions for the same bag_index. "
            "This means bag_index does not refer to the same underlying "
            "bag in both files - do not proceed with this join key."
        )
    merged = merged.rename(columns={"subject_gru": "subject", "label_gru": "label"})
    return merged[
        ["subject", "horizon", "bag_index", "label", "gru_probability", "baseline_probability"]
    ]


def average_ensemble(merged: pd.DataFrame) -> np.ndarray:
    gru_logit = to_logit(merged["gru_probability"].to_numpy())
    baseline_logit = to_logit(merged["baseline_probability"].to_numpy())
    return to_probability((gru_logit + baseline_logit) / 2.0)


def stacked_ensemble(merged: pd.DataFrame, seed: int) -> np.ndarray:
    gru_logit = to_logit(merged["gru_probability"].to_numpy())
    baseline_logit = to_logit(merged["baseline_probability"].to_numpy())
    features = np.column_stack([gru_logit, baseline_logit])
    labels = merged["label"].to_numpy()
    subjects = merged["subject"].to_numpy()
    output = np.empty(len(merged), dtype=float)
    for subject in np.unique(subjects):
        train_mask = subjects != subject
        test_mask = subjects == subject
        model = LogisticRegression(random_state=seed, max_iter=1000)
        model.fit(features[train_mask], labels[train_mask])
        output[test_mask] = model.predict_proba(features[test_mask])[:, 1]
    return output


def per_subject_metrics(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject", sort=True):
        y_true = group["label"].to_numpy(dtype=int)
        y_score = group[score_col].to_numpy(dtype=float)
        y_pred = (y_score >= 0.5).astype(int)
        if np.unique(y_true).size != 2:
            continue
        rows.append(
            {
                "subject": subject,
                "pr_auc": float(average_precision_score(y_true, y_score)),
                "roc_auc": float(roc_auc_score(y_true, y_score)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    difference = (a - b).to_numpy(dtype=float)
    if np.allclose(difference, 0.0):
        statistic, p_raw = 0.0, 1.0
    else:
        test = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        statistic, p_raw = float(test.statistic), float(test.pvalue)
    n = len(difference)
    mean_diff = float(np.mean(difference))
    sd_diff = float(np.std(difference, ddof=1))
    se_diff = sd_diff / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return {
        "n_subjects": n,
        "mean_difference": mean_diff,
        "ci95_low": mean_diff - t_crit * se_diff,
        "ci95_high": mean_diff + t_crit * se_diff,
        "cohen_dz": mean_diff / sd_diff if sd_diff > 0 else 0.0,
        "wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "losses": int(np.sum(difference < 0)),
        "wilcoxon_w": statistic,
        "p_raw": p_raw,
    }


def run_self_test() -> None:
    rng = np.random.default_rng(0)
    subjects = [f"S{i:02d}" for i in range(42)]
    rows = []
    for subject in subjects:
        n = 20
        label = rng.integers(0, 2, size=n)
        gru_prob = np.clip(label * 0.5 + rng.normal(0.3, 0.1, size=n), 0.01, 0.99)
        baseline_prob = np.clip(label * 0.3 + rng.normal(0.3, 0.1, size=n), 0.01, 0.99)
        for i in range(n):
            rows.append(
                {
                    "subject": subject,
                    "horizon": "first1",
                    "bag_index": len(rows),
                    "label": int(label[i]),
                    "gru_probability": float(gru_prob[i]),
                    "baseline_probability": float(baseline_prob[i]),
                }
            )
    merged = pd.DataFrame(rows)
    averaged = average_ensemble(merged)
    stacked = stacked_ensemble(merged, seed=0)
    if not (0.0 <= averaged.min() and averaged.max() <= 1.0):
        raise AssertionError("average ensemble probabilities out of range")
    if not (0.0 <= stacked.min() and stacked.max() <= 1.0):
        raise AssertionError("stacked ensemble probabilities out of range")
    merged["ensemble_average"] = averaged
    merged["ensemble_stacked"] = stacked
    metrics_avg = per_subject_metrics(merged, "ensemble_average")
    metrics_gru = per_subject_metrics(merged, "gru_probability")
    if len(metrics_avg) != 42 or len(metrics_gru) != 42:
        raise AssertionError("expected 42 subjects with both classes present")
    test = paired_test(metrics_avg["pr_auc"], metrics_gru["pr_auc"])
    if test["n_subjects"] != 42:
        raise AssertionError(test)
    print("SELF-TEST PASS")


def run_analysis(args: argparse.Namespace) -> None:
    neural = load_neural(args.neural_predictions, args.gru_condition)
    classical = load_classical(args.classical_predictions, args.baseline_model)

    all_rows = []
    comparison_rows = []
    for horizon in args.horizons:
        neural_h = neural.loc[neural["horizon"] == horizon]
        classical_h = classical.loc[classical["horizon"] == horizon]
        if neural_h.empty or classical_h.empty:
            print(f"NOTE: no rows for horizon={horizon}; skipping.", file=sys.stderr)
            continue
        merged = merge_predictions(neural_h, classical_h)
        merged["ensemble_average"] = average_ensemble(merged)
        merged["ensemble_stacked"] = stacked_ensemble(merged, seed=args.seed)
        merged["horizon"] = horizon
        all_rows.append(merged)

        metrics = {
            "gru": per_subject_metrics(merged, "gru_probability"),
            "baseline": per_subject_metrics(merged, "baseline_probability"),
            "ensemble_average": per_subject_metrics(merged, "ensemble_average"),
            "ensemble_stacked": per_subject_metrics(merged, "ensemble_stacked"),
        }
        for name, table in metrics.items():
            for _, row in table.iterrows():
                comparison_rows.append(
                    {"horizon": horizon, "model": name, **row.to_dict()}
                )

    if not all_rows:
        raise ValueError("No horizons produced any merged predictions.")

    per_subject = pd.DataFrame(comparison_rows)

    tests = []
    for horizon in args.horizons:
        subset = per_subject.loc[per_subject["horizon"] == horizon]
        if subset.empty:
            continue
        for metric in ("pr_auc", "mcc"):
            pivot = subset.pivot(index="subject", columns="model", values=metric).dropna()
            for candidate, comparator in (
                ("ensemble_average", "gru"),
                ("ensemble_average", "baseline"),
                ("ensemble_stacked", "gru"),
                ("ensemble_stacked", "baseline"),
            ):
                if candidate not in pivot or comparator not in pivot:
                    continue
                result = paired_test(pivot[candidate], pivot[comparator])
                tests.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "candidate": candidate,
                        "comparator": comparator,
                        "candidate_mean": float(pivot[candidate].mean()),
                        "comparator_mean": float(pivot[comparator].mean()),
                        **result,
                    }
                )
    tests_frame = pd.DataFrame(tests)
    tests_frame["p_holm"] = holm_adjust(tests_frame["p_raw"].to_numpy())
    tests_frame["significant_holm_0_05"] = tests_frame["p_holm"] < 0.05

    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(
        args.output / "ensemble_predictions.csv", index=False
    )
    per_subject.to_csv(args.output / "ensemble_per_subject_metrics.csv", index=False)
    tests_frame.to_csv(args.output / "ensemble_pairwise_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print("ENSEMBLE PAIRED TESTS (positive difference favors the ensemble)")
    print(
        tests_frame[
            [
                "horizon", "metric", "candidate", "comparator", "n_subjects",
                "candidate_mean", "comparator_mean", "mean_difference",
                "ci95_low", "ci95_high", "p_raw", "p_holm", "significant_holm_0_05",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-predictions", type=Path, required=False)
    parser.add_argument("--classical-predictions", type=Path, required=False)
    parser.add_argument("--gru-condition", type=str, default="behavior")
    parser.add_argument("--baseline-model", type=str, default="logreg_dwell_propensity")
    parser.add_argument("--horizons", nargs="+", default=list(HORIZONS), choices=HORIZONS)
    parser.add_argument("--output", type=Path, default=Path("cache/ensemble_results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        run_self_test()
    else:
        if parsed.neural_predictions is None or parsed.classical_predictions is None:
            raise ValueError(
                "--neural-predictions and --classical-predictions are required "
                "unless --self-test is given"
            )
        run_analysis(parsed)
