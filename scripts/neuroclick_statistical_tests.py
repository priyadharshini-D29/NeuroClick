#!/usr/bin/env python3
"""Subject-level paired statistical testing for NeuroClick-Hazard."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, matthews_corrcoef


SUBJECT_ALIASES = (
    "heldout_subject", "held_out_subject", "test_subject", "subject_id",
    "subject", "participant_id", "participant", "fold_subject",
)
HORIZON_ALIASES = ("horizon", "prefix", "endpoint", "visit_horizon")
MODEL_ALIASES = ("condition", "model", "model_name", "ablation", "feature_set")
TRUE_ALIASES = (
    "y_true", "true_label", "label", "label_buy", "target", "buy_label", "is_buy",
)
SCORE_ALIASES = (
    "y_prob", "y_score", "probability", "probability_buy", "prob_buy", "buy_probability",
    "risk", "score", "predicted_probability",
)
PRED_ALIASES = (
    "y_pred", "pred_label", "prediction_label", "prediction_buy", "predicted_label",
    "class_pred", "prediction", "predicted",
)
THRESHOLD_ALIASES = ("threshold", "decision_threshold", "selected_threshold")


@dataclass(frozen=True)
class Columns:
    subject: str
    horizon: str
    model: str
    y_true: str
    y_score: str
    y_pred: str | None
    threshold: str | None


def normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_column(frame: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> str | None:
    normalized = {normalized_name(column): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    if required:
        raise ValueError(
            f"Could not resolve any of {list(aliases)} from columns: "
            f"{list(map(str, frame.columns))}"
        )
    return None


def resolve_columns(frame: pd.DataFrame) -> Columns:
    return Columns(
        subject=resolve_column(frame, SUBJECT_ALIASES),
        horizon=resolve_column(frame, HORIZON_ALIASES),
        model=resolve_column(frame, MODEL_ALIASES),
        y_true=resolve_column(frame, TRUE_ALIASES),
        y_score=resolve_column(frame, SCORE_ALIASES),
        y_pred=resolve_column(frame, PRED_ALIASES, required=False),
        threshold=resolve_column(frame, THRESHOLD_ALIASES, required=False),
    )


def canonicalize_predictions(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    columns = resolve_columns(frame)
    result = pd.DataFrame(
        {
            "subject": frame[columns.subject].astype(str).str.strip(),
            "horizon": frame[columns.horizon].map(normalized_name),
            "model": frame[columns.model].map(normalized_name),
            "y_true": pd.to_numeric(frame[columns.y_true], errors="coerce"),
            "y_score": pd.to_numeric(frame[columns.y_score], errors="coerce"),
        }
    )

    if columns.y_pred is not None:
        result["y_pred"] = pd.to_numeric(frame[columns.y_pred], errors="coerce")
    elif columns.threshold is not None:
        threshold = pd.to_numeric(frame[columns.threshold], errors="coerce")
        result["y_pred"] = (result["y_score"] >= threshold).astype(float)
    else:
        raise ValueError(
            f"{source}: MCC requires saved outer-fold predictions or saved inner-fold "
            "thresholds. No prediction/threshold column was found. Do not substitute 0.5."
        )

    result["source"] = source
    before = len(result)
    result = result.dropna(subset=["subject", "horizon", "model", "y_true", "y_score", "y_pred"])
    if len(result) != before:
        raise ValueError(f"{source}: {before - len(result)} rows contain missing required values")
    result["y_true"] = result["y_true"].astype(int)
    result["y_pred"] = result["y_pred"].astype(int)
    for field in ("y_true", "y_pred"):
        values = set(result[field].unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{source}: {field} is not binary: {sorted(values)}")
    if not np.isfinite(result["y_score"]).all():
        raise ValueError(f"{source}: y_score contains non-finite values")
    return result


def read_prediction_files(paths: Iterable[Path], source_prefix: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_subjects_by_path: dict[Path, set[str]] = {}
    for path in paths:
        if not path.exists():
            print(f"WARNING: optional prediction file not found: {path}", file=sys.stderr)
            continue
        raw = pd.read_csv(path)
        canonical = canonicalize_predictions(raw, f"{source_prefix}:{path.name}")
        frames.append(canonical)
        seen_subjects_by_path[path] = set(canonical["subject"])

    if not frames:
        raise FileNotFoundError(f"No {source_prefix} prediction files were found")

    # Pilot and remaining-neural outputs must contain disjoint test subjects.
    if source_prefix == "neural" and len(seen_subjects_by_path) > 1:
        items = list(seen_subjects_by_path.items())
        for index, (left_path, left_subjects) in enumerate(items):
            for right_path, right_subjects in items[index + 1:]:
                overlap = sorted(left_subjects & right_subjects)
                if overlap:
                    raise ValueError(
                        f"Neural prediction files overlap for {len(overlap)} subjects "
                        f"({left_path}, {right_path}): {overlap}"
                    )
    return pd.concat(frames, ignore_index=True)


def per_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["horizon", "model", "subject"]
    for (horizon, model, subject), group in predictions.groupby(keys, sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_score = group["y_score"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=int)
        if np.unique(y_true).size != 2:
            raise ValueError(
                f"Subject {subject}, model {model}, horizon {horizon} does not contain both classes"
            )
        rows.append(
            {
                "horizon": horizon,
                "model": model,
                "subject": subject,
                "n": len(group),
                "n_buy": int(y_true.sum()),
                "pr_auc": float(average_precision_score(y_true, y_score)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
            }
        )
    return pd.DataFrame(rows)


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), math.nan, math.nan
    mean = float(values.mean())
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    return mean, mean - critical * sem, mean + critical * sem


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_test(
    metrics: pd.DataFrame,
    candidate: str,
    comparator: str,
    metric: str,
    horizon: str = "first1",
) -> dict[str, object]:
    subset = metrics.loc[
        (metrics["horizon"] == horizon) & metrics["model"].isin([candidate, comparator]),
        ["subject", "model", metric],
    ]
    pivot = subset.pivot(index="subject", columns="model", values=metric).dropna()
    if candidate not in pivot or comparator not in pivot:
        raise ValueError(
            f"Missing {candidate!r} or {comparator!r} for {metric}/{horizon}; "
            f"available models: {sorted(metrics['model'].unique())}"
        )
    difference = (pivot[candidate] - pivot[comparator]).to_numpy(dtype=float)
    if np.allclose(difference, 0.0):
        statistic, p_raw = 0.0, 1.0
    else:
        test = stats.wilcoxon(
            difference,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        statistic, p_raw = float(test.statistic), float(test.pvalue)

    mean_difference, ci_low, ci_high = mean_ci(difference)
    sd_difference = float(np.std(difference, ddof=1))
    cohen_dz = mean_difference / sd_difference if sd_difference > 0 else math.nan
    tolerance = 1e-12
    return {
        "horizon": horizon,
        "metric": metric,
        "candidate": candidate,
        "comparator": comparator,
        "n_subjects": len(pivot),
        "candidate_mean": float(pivot[candidate].mean()),
        "comparator_mean": float(pivot[comparator].mean()),
        "mean_difference": mean_difference,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohen_dz": cohen_dz,
        "wins": int(np.sum(difference > tolerance)),
        "ties": int(np.sum(np.abs(difference) <= tolerance)),
        "losses": int(np.sum(difference < -tolerance)),
        "wilcoxon_w": statistic,
        "p_raw": p_raw,
    }


def model_intervals(metrics: pd.DataFrame, horizon: str = "first1") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subset = metrics[metrics["horizon"] == horizon]
    for model, group in subset.groupby("model", sort=True):
        for metric in ("pr_auc", "mcc"):
            mean, low, high = mean_ci(group[metric].to_numpy(dtype=float))
            rows.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "metric": metric,
                    "n_subjects": len(group),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "std": float(group[metric].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def markdown_report(tests: pd.DataFrame, intervals: pd.DataFrame) -> str:
    lines = [
        "# NeuroClick-Hazard primary statistical analysis",
        "",
        "Inferential unit: held-out participant (strict LOSOCV). Primary endpoint: first valid product-ROI visit. ",
        "Tests: two-sided paired Wilcoxon signed-rank. Multiplicity: Holm correction across four ",
        "pre-specified comparisons (two comparators × PR-AUC/MCC). Confidence intervals are 95% ",
        "t-intervals over participant scores or participant-paired differences.",
        "",
        "## Confirmatory paired comparisons",
        "",
        "| Metric | Comparator | n | Candidate | Comparator | Difference (95% CI) | dz | W/T/L | p raw | p Holm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.comparator} | {row.n_subjects} | "
            f"{row.candidate_mean:.4f} | {row.comparator_mean:.4f} | "
            f"{row.mean_difference:+.4f} [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | "
            f"{row.cohen_dz:+.3f} | {row.wins}/{row.ties}/{row.losses} | "
            f"{row.p_raw:.6g} | {row.p_holm:.6g} |"
        )
    lines.extend(
        [
            "",
            "Positive differences favor NeuroClick behavior. Statistical significance is assessed at ",
            "Holm-adjusted p < 0.05. PR-AUC is the primary performance metric; MCC is confirmatory ",
            "support for thresholded predictions.",
            "",
            "## First-visit participant-level confidence intervals",
            "",
            "| Model | Metric | n | Mean (95% CI) | SD |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in intervals.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.metric} | {row.n_subjects} | "
            f"{row.mean:.4f} [{row.ci95_low:.4f}, {row.ci95_high:.4f}] | {row.std:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> None:
    neural = read_prediction_files(args.neural_predictions, "neural")
    classical = read_prediction_files(args.classical_predictions, "classical")

    neural_subjects = set(neural["subject"])
    if len(neural_subjects) != args.expected_subjects:
        raise ValueError(
            f"Neural merge contains {len(neural_subjects)} subjects, expected {args.expected_subjects}: "
            f"{sorted(neural_subjects)}"
        )

    predictions = pd.concat([neural, classical], ignore_index=True)
    metrics = per_subject_metrics(predictions)

    # The source files use the same canonical model labels after normalization.
    required_models = {"behavior", "logreg_behavior", "total_dwell"}
    missing_models = required_models - set(metrics["model"])
    if missing_models:
        raise ValueError(
            f"Required models missing: {sorted(missing_models)}. Available: "
            f"{sorted(metrics['model'].unique())}"
        )

    available_models = set(metrics["model"])
    comparators = ["logreg_behavior", "total_dwell"]
    # These two are optional so this script keeps working against older
    # prediction files that predate the propensity-leakage fix; when present
    # they answer the two outstanding first1 questions: whether NeuroClick
    # beats a simple dwell+propensity baseline, and how much of NeuroClick's
    # own gain comes from the propensity input specifically.
    for optional_comparator in ("logreg_dwell_propensity", "behavior_no_propensity"):
        if optional_comparator in available_models:
            comparators.append(optional_comparator)
        else:
            print(
                f"NOTE: '{optional_comparator}' not found in predictions; "
                "skipping that comparison. Re-run with the updated "
                "03/04 scripts to include it.",
                file=sys.stderr,
            )

    comparisons = []
    for metric in ("pr_auc", "mcc"):
        for comparator in comparators:
            comparisons.append(paired_test(metrics, "behavior", comparator, metric, "first1"))
    tests = pd.DataFrame(comparisons)
    tests["p_holm"] = holm_adjust(tests["p_raw"].to_numpy())
    tests["significant_holm_0_05"] = tests["p_holm"] < 0.05

    first1_models = [
        "behavior", "behavior_et", "behavior_eeg", "behavior_eeg_et",
        "logreg_behavior", "total_dwell",
    ]
    first1_models += [c for c in ("logreg_dwell_propensity", "behavior_no_propensity") if c in available_models]
    intervals = model_intervals(metrics[metrics["model"].isin(first1_models)], "first1")

    # Enforce complete pairing rather than silently analyzing a smaller set.
    if not (tests["n_subjects"] == args.expected_subjects).all():
        raise ValueError(
            "At least one confirmatory comparison is not paired over all expected subjects:\n"
            + tests[["metric", "comparator", "n_subjects"]].to_string(index=False)
        )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(["horizon", "model", "subject"]).to_csv(
        args.output / "per_subject_metrics.csv", index=False
    )
    tests.to_csv(args.output / "primary_pairwise_tests.csv", index=False)
    intervals.to_csv(args.output / "first1_model_intervals.csv", index=False)
    (args.output / "statistics_report.md").write_text(
        markdown_report(tests, intervals), encoding="utf-8"
    )
    manifest = {
        "expected_subjects": args.expected_subjects,
        "observed_neural_subjects": len(neural_subjects),
        "primary_horizon": "first1",
        "inferential_unit": "held-out subject",
        "test": "two-sided paired Wilcoxon signed-rank",
        "multiplicity": "Holm across four confirmatory tests",
        "confidence_interval": "95% Student-t interval",
        "neural_prediction_files": [str(path) for path in args.neural_predictions],
        "classical_prediction_files": [str(path) for path in args.classical_predictions],
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPRIMARY PAIRED TESTS")
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved statistical outputs to: {args.output}")


def self_test() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adjusted = holm_adjust(p)
    expected = np.array([0.04, 0.09, 0.09, 0.20])
    if not np.allclose(adjusted, expected):
        raise AssertionError((adjusted, expected))

    subjects = [f"S{index:02d}" for index in range(1, 43)]
    rows = []
    for index, subject in enumerate(subjects):
        rows.extend(
            [
                {"horizon": "first1", "model": "behavior", "subject": subject, "pr_auc": 0.30 + index / 1000, "mcc": 0.20 + index / 2000},
                {"horizon": "first1", "model": "logreg_behavior", "subject": subject, "pr_auc": 0.25 + index / 1000, "mcc": 0.18 + index / 2000},
            ]
        )
    result = paired_test(pd.DataFrame(rows), "behavior", "logreg_behavior", "pr_auc")
    if result["n_subjects"] != 42 or not np.isclose(result["mean_difference"], 0.05):
        raise AssertionError(result)
    print("SELF-TEST PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neural-predictions",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/benchmarks/neuroclick_hazard_pilot5_v1/losocv_predictions.csv"),
            Path("outputs/benchmarks/neuroclick_hazard_remaining37_v1/losocv_predictions.csv"),
        ],
    )
    parser.add_argument(
        "--classical-predictions",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/benchmarks/classical_first1_causal_v1/losocv_predictions.csv"),
            Path("outputs/benchmarks/classical_secondary_horizons_causal_v1/losocv_predictions.csv"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/statistics/neuroclick_primary_v1"),
    )
    parser.add_argument("--expected-subjects", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        run_analysis(parsed)
