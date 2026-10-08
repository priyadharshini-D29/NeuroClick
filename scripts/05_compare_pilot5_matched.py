#!/usr/bin/env python3
"""Compare the NeuroClick pilot with classical behavior on identical rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


SUBJECTS = {"S01", "S02", "S03", "S05", "S06"}
HORIZONS = ("first1", "first2", "first3", "full")
CONDITIONS = (
    "behavior",
    "behavior_et",
    "behavior_eeg",
    "behavior_eeg_et",
)
EXPECTED_COUNTS = {
    "first1": 675,
    "first2": 548,
    "first3": 366,
    "full": 675,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a strict five-subject matched comparison between the "
            "classical logreg_behavior baseline and NeuroClick-Hazard."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project directory containing outputs/ (default: current directory).",
    )
    return parser.parse_args()


def score(
    frame: pd.DataFrame,
    probability_column: str,
    prediction_column: str,
) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=int)

    subject_pr: list[float] = []
    subject_auc: list[float] = []
    subject_ba: list[float] = []
    subject_mcc: list[float] = []

    for _, group in frame.groupby("subject", sort=True):
        subject_labels = group["label"].to_numpy(dtype=int)
        subject_probability = group[probability_column].to_numpy(dtype=float)
        subject_prediction = group[prediction_column].to_numpy(dtype=int)

        subject_pr.append(
            average_precision_score(subject_labels, subject_probability)
        )
        if np.unique(subject_labels).size == 2:
            subject_auc.append(
                roc_auc_score(subject_labels, subject_probability)
            )
        subject_ba.append(
            balanced_accuracy_score(subject_labels, subject_prediction)
        )
        subject_mcc.append(
            matthews_corrcoef(subject_labels, subject_prediction)
        )

    return {
        "pr": float(average_precision_score(labels, probability)),
        "macro_pr": float(np.mean(subject_pr)),
        "auc": float(roc_auc_score(labels, probability)),
        "macro_auc": float(np.mean(subject_auc)),
        "ba": float(balanced_accuracy_score(labels, prediction)),
        "macro_ba": float(np.mean(subject_ba)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "macro_mcc": float(np.mean(subject_mcc)),
    }


def main() -> int:
    project = parse_args().project.resolve()
    benchmark_root = project / "outputs" / "benchmarks"

    classical_paths = (
        benchmark_root
        / "classical_first1_causal_v1"
        / "losocv_predictions.csv",
        benchmark_root
        / "classical_secondary_horizons_causal_v1"
        / "losocv_predictions.csv",
    )
    neural_path = (
        benchmark_root
        / "neuroclick_hazard_pilot5_v1"
        / "losocv_predictions.csv"
    )

    required_paths = (*classical_paths, neural_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n  " + "\n  ".join(missing))

    classical = pd.concat(
        [pd.read_csv(path) for path in classical_paths],
        ignore_index=True,
    )
    classical = classical.loc[
        classical["model"].eq("logreg_behavior")
        & classical["held_out_subject"].isin(SUBJECTS)
    ].rename(
        columns={
            "held_out_subject": "subject",
            "label_buy": "label_classical",
            "probability_buy": "probability_classical",
            "prediction_buy": "prediction_classical",
        }
    )

    neural = pd.read_csv(neural_path).rename(
        columns={
            "heldout_subject": "subject",
            "label": "label_neural",
            "probability": "probability_neural",
            "prediction": "prediction_neural",
            "page_audit": "page",
            "roi_audit": "roi",
            "product_id_audit": "product_id",
        }
    )

    keys = ["subject", "horizon", "page", "roi", "product_id"]
    if classical.duplicated(keys).any():
        raise RuntimeError("Duplicate classical prediction keys")
    if neural.duplicated(keys + ["condition"]).any():
        raise RuntimeError("Duplicate neural prediction keys")

    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        classical_horizon = classical.loc[
            classical["horizon"].eq(horizon),
            keys
            + [
                "label_classical",
                "probability_classical",
                "prediction_classical",
            ],
        ]

        for condition in CONDITIONS:
            neural_horizon = neural.loc[
                neural["horizon"].eq(horizon)
                & neural["condition"].eq(condition),
                keys
                + [
                    "label_neural",
                    "probability_neural",
                    "prediction_neural",
                ],
            ]

            matched = classical_horizon.merge(
                neural_horizon,
                on=keys,
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            unmatched = matched["_merge"].ne("both")
            if unmatched.any():
                preview = matched.loc[unmatched, keys + ["_merge"]].head(20)
                print(preview.to_string(index=False))
                raise RuntimeError(
                    f"Unmatched rows for {horizon}/{condition}: "
                    f"{int(unmatched.sum())}"
                )

            if not np.array_equal(
                matched["label_classical"].to_numpy(),
                matched["label_neural"].to_numpy(),
            ):
                raise RuntimeError(f"Label mismatch for {horizon}/{condition}")

            matched["label"] = matched["label_neural"].astype(int)
            classical_scores = score(
                matched,
                "probability_classical",
                "prediction_classical",
            )
            neural_scores = score(
                matched,
                "probability_neural",
                "prediction_neural",
            )

            row: dict[str, object] = {
                "horizon": horizon,
                "condition": condition,
                "n": len(matched),
                "n_buy": int(matched["label"].sum()),
                "prevalence": float(matched["label"].mean()),
            }
            for metric in (
                "pr",
                "macro_pr",
                "auc",
                "macro_auc",
                "ba",
                "macro_ba",
                "mcc",
                "macro_mcc",
            ):
                row[f"classical_{metric}"] = classical_scores[metric]
                row[f"neural_{metric}"] = neural_scores[metric]
                row[f"delta_{metric}"] = (
                    neural_scores[metric] - classical_scores[metric]
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    for horizon, expected in EXPECTED_COUNTS.items():
        observed = set(
            result.loc[result["horizon"].eq(horizon), "n"].astype(int)
        )
        if observed != {expected}:
            raise RuntimeError(
                f"{horizon}: expected n={expected}, observed={observed}"
            )

    output = (
        benchmark_root
        / "neuroclick_hazard_pilot5_v1"
        / "matched_classical_comparison.csv"
    )
    result.to_csv(output, index=False)

    display_columns = [
        "horizon",
        "condition",
        "n",
        "n_buy",
        "prevalence",
        "classical_pr",
        "neural_pr",
        "delta_pr",
        "classical_macro_pr",
        "neural_macro_pr",
        "delta_macro_pr",
        "classical_auc",
        "neural_auc",
        "delta_auc",
        "classical_mcc",
        "neural_mcc",
        "delta_mcc",
    ]
    print("\nMATCHED FIVE-SUBJECT COMPARISON")
    print(
        result[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved: {output}")
    print("MATCHING CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
