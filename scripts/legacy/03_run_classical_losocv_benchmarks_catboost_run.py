#!/usr/bin/env python3
"""Strict subject-independent classical benchmarks for NeuroClick-Hazard.

The script reads the frozen stage-2 pre-click sequence cache, extracts compact
EEG and eye-tracking visit descriptors, aggregates them at matched visit
horizons, and evaluates reviewer-facing baselines with leave-one-subject-out
cross-validation (LOSOCV).

Primary protocol
----------------
``first1`` evaluates every considered product immediately after its first
valid gaze visit.  This matched horizon prevents a classifier from succeeding
only because Buy sequences terminate at a click whereas NoBuy sequences are
right-censored at the end of exposure.

Secondary protocols
-------------------
``first2`` and ``first3`` use risk sets containing instances with at least two
or three visits.  ``full`` uses all available pre-decision visits and is
reported as a secondary full-information comparison because bag length is
potentially informative.

The primary metric is average precision (PR-AUC).  ROC-AUC, Buy-class F1,
precision, recall, balanced accuracy, MCC, Brier score, and ECE are also
reported.  All imputation, variance filtering, scaling, and feature selection
are fit using training subjects only.  Product/page/ROI and subject identifiers
remain audit metadata and are never model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import welch
from sklearn.base import clone
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
VALID_HORIZONS = ("first1", "first2", "first3", "full")
VALID_MODELS = (
    "prevalence",
    "product_propensity",
    "visit_count",
    "total_dwell",
    "logreg_behavior",
    "logreg_eeg",
    "logreg_et",
    "logreg_fusion",
    "catboost_fusion",
)


@dataclass(frozen=True)
class RunConfig:
    cache: str
    output: str
    feature_cache: str
    horizons: tuple[str, ...]
    models: tuple[str, ...]
    seed: int = 42
    select_percentile: int = 25
    logreg_c: float = 1.0
    catboost_iterations: int = 300
    catboost_depth: int = 5
    catboost_learning_rate: float = 0.05
    ece_bins: int = 10
    inner_folds: int = 3
    threshold_mode: str = "inner_mcc"
    product_smoothing: float = 10.0
    catboost_task_type: str = "CPU"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe classical LOSOCV Buy/NoBuy benchmarks."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional .npz path; defaults to OUTPUT/visit_features.npz.",
    )
    parser.add_argument(
        "--horizons", nargs="+", choices=VALID_HORIZONS, default=list(VALID_HORIZONS)
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=VALID_MODELS,
        default=[
            "prevalence",
            "product_propensity",
            "visit_count",
            "total_dwell",
            "logreg_behavior",
            "logreg_eeg",
            "logreg_et",
            "logreg_fusion",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-percentile", type=int, default=25)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=5)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--catboost-task-type", choices=("CPU", "GPU"), default="CPU"
    )
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--threshold-mode",
        choices=("inner_mcc", "fixed_0.5"),
        default="inner_mcc",
    )
    parser.add_argument("--product-smoothing", type=float, default=10.0)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _masked_fill_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1, keepdims=True)
    mean = np.divide(
        np.where(valid, values, 0.0).sum(axis=1, keepdims=True),
        count,
        out=np.zeros((values.shape[0], 1, values.shape[2]), dtype=np.float64),
        where=count > 0,
    )
    return np.where(valid, values, mean)


def eeg_visit_features(
    windows: np.ndarray,
    masks: np.ndarray,
    fs: float = 300.0,
) -> np.ndarray:
    """Five-band log-relative power plus Hjorth descriptors (152 features)."""
    x = _masked_fill_mean(windows, masks)
    x = x - x.mean(axis=1, keepdims=True)
    nperseg = min(256, x.shape[1])
    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        axis=1,
    )
    eps = np.finfo(np.float64).eps
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total_power = np.trapezoid(psd[:, total_mask, :], freqs[total_mask], axis=1)
    descriptors: list[np.ndarray] = []
    for low, high in EEG_BANDS.values():
        band_mask = (freqs >= low) & (freqs < high if high < 45.0 else freqs <= high)
        band_power = np.trapezoid(psd[:, band_mask, :], freqs[band_mask], axis=1)
        descriptors.append(np.log10(band_power / (total_power + eps) + eps))

    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    activity = np.var(x, axis=1)
    variance_dx = np.var(dx, axis=1)
    variance_ddx = np.var(ddx, axis=1)
    mobility = np.sqrt(variance_dx / (activity + eps))
    complexity = np.sqrt(variance_ddx / (variance_dx + eps)) / (mobility + eps)
    descriptors.extend([np.log10(activity + eps), mobility, complexity])

    features = np.concatenate(descriptors, axis=1).astype(np.float32)
    if features.shape[1] != 152:
        raise RuntimeError(f"Expected 152 EEG visit features, got {features.shape}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _masked_channel_statistics(
    windows: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(windows, dtype=np.float64)
    valid = np.asarray(masks, dtype=bool) & np.isfinite(x)
    masked = np.where(valid, x, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(masked, axis=1)
        std = np.nanstd(masked, axis=1)
        minimum = np.nanmin(masked, axis=1)
        maximum = np.nanmax(masked, axis=1)
        median = np.nanmedian(masked, axis=1)
    valid_fraction = valid.mean(axis=1)

    first = np.zeros((x.shape[0], x.shape[2]), dtype=np.float64)
    last = np.zeros_like(first)
    for visit in range(x.shape[0]):
        for channel in range(x.shape[2]):
            indices = np.flatnonzero(valid[visit, :, channel])
            if indices.size:
                first[visit, channel] = x[visit, indices[0], channel]
                last[visit, channel] = x[visit, indices[-1], channel]
    slope = last - first

    blocks = [mean, std, minimum, maximum, median, first, last, slope, valid_fraction]
    names = [
        f"ch{channel}_{statistic}"
        for statistic in (
            "mean", "std", "min", "max", "median", "first", "last", "delta", "valid"
        )
        for channel in range(x.shape[2])
    ]
    # The concatenation order is statistic-major, matching the generated names.
    features = np.concatenate(blocks, axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), names


def _masked_distribution(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.column_stack(
            [
                np.nanmean(masked, axis=1),
                np.nanstd(masked, axis=1),
                np.nanmax(masked, axis=1),
            ]
        )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def et_visit_features(
    windows: np.ndarray,
    masks: np.ndarray,
    fs: float = 120.0,
) -> np.ndarray:
    """ROI-relative gaze/pupil summaries, velocities, and binocular disparity."""
    base, _ = _masked_channel_statistics(windows, masks)
    x = np.asarray(windows, dtype=np.float64)
    valid = np.asarray(masks, dtype=bool) & np.isfinite(x)
    extras: list[np.ndarray] = []

    for x_channel, y_channel in ((0, 1), (3, 4)):
        pair_valid = (
            valid[:, 1:, x_channel]
            & valid[:, :-1, x_channel]
            & valid[:, 1:, y_channel]
            & valid[:, :-1, y_channel]
        )
        dx = np.diff(x[:, :, x_channel], axis=1)
        dy = np.diff(x[:, :, y_channel], axis=1)
        speed = np.sqrt(dx * dx + dy * dy) * fs
        extras.append(_masked_distribution(speed, pair_valid))

    binocular_valid = valid[:, :, 0] & valid[:, :, 1] & valid[:, :, 3] & valid[:, :, 4]
    disparity_x = x[:, :, 0] - x[:, :, 3]
    disparity_y = x[:, :, 1] - x[:, :, 4]
    disparity_distance = np.sqrt(disparity_x**2 + disparity_y**2)
    extras.extend(
        [
            _masked_distribution(disparity_x, binocular_valid),
            _masked_distribution(disparity_y, binocular_valid),
            _masked_distribution(disparity_distance, binocular_valid),
        ]
    )

    features = np.concatenate([base, *extras], axis=1).astype(np.float32)
    # 6 channels * 9 summaries + 2 eyes * 3 velocity summaries +
    # 3 binocular signals * 3 summaries = 69.
    if features.shape[1] != 69:
        raise RuntimeError(f"Expected 69 ET visit features, got {features.shape}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def aggregate_visit_features(features: np.ndarray) -> np.ndarray:
    """Chronology-aware fixed vector: mean, std, max, last, and endpoint delta."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Cannot aggregate visit features with shape {features.shape}")
    return np.concatenate(
        [
            features.mean(axis=0),
            features.std(axis=0),
            features.max(axis=0),
            features[-1],
            features[-1] - features[0],
        ]
    ).astype(np.float32)


def behavior_features(
    starts: np.ndarray,
    ends: np.ndarray,
    dwell: np.ndarray,
    eeg_valid_fraction: np.ndarray,
    et_valid_fraction: np.ndarray,
) -> np.ndarray:
    count = len(dwell)
    if count == 0:
        raise ValueError("Behavior aggregation requires at least one visit")
    gaps = np.maximum(0.0, starts[1:] - ends[:-1]) if count >= 2 else np.array([0.0])
    return np.asarray(
        [
            count,
            float(dwell.sum()),
            float(dwell.mean()),
            float(dwell.std()),
            float(dwell.max()),
            float(max(0.0, ends[-1] - starts[0])),
            float(gaps.mean()),
            float(gaps.max()),
            float(np.mean(eeg_valid_fraction)),
            float(np.mean(et_valid_fraction)),
        ],
        dtype=np.float32,
    )


def build_visit_feature_cache(cache: Path, target: Path) -> dict[str, np.ndarray]:
    index_path = cache / "cache_index.csv"
    summary_path = cache / "cache_summary.json"
    if not index_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Incomplete sequence cache: {cache}")
    index = pd.read_csv(index_path).sort_values(["subject", "bag_index"])
    subjects = sorted(index["subject"].astype(str).unique())

    all_eeg: list[np.ndarray] = []
    all_et: list[np.ndarray] = []
    all_start: list[np.ndarray] = []
    all_end: list[np.ndarray] = []
    all_dwell: list[np.ndarray] = []
    all_eeg_valid: list[np.ndarray] = []
    all_et_valid: list[np.ndarray] = []
    global_offsets = [0]
    bag_subject: list[str] = []
    labels: list[int] = []
    pages: list[int] = []
    rois: list[int] = []
    products: list[int] = []

    for subject in subjects:
        path = cache / f"{subject}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            eeg = np.asarray(data["eeg"], dtype=np.float32)
            eeg_mask = np.asarray(data["eeg_mask"], dtype=np.uint8)
            et = np.asarray(data["et_roi"], dtype=np.float32)
            et_mask = np.asarray(data["et_mask"], dtype=np.uint8)
            offsets = np.asarray(data["bag_offsets"], dtype=np.int64)
            subject_labels = np.asarray(data["labels"], dtype=np.uint8)
            subject_pages = np.asarray(data["page"], dtype=np.uint8)
            subject_rois = np.asarray(data["roi"], dtype=np.uint8)
            subject_products = np.asarray(data["product_id"], dtype=np.uint16)
            starts = np.asarray(data["visit_start"], dtype=np.float64)
            ends = np.asarray(data["visit_end"], dtype=np.float64)
            dwell = np.asarray(data["visit_dwell_s"], dtype=np.float32)

        if not (
            len(eeg) == len(et) == len(starts) == len(ends) == len(dwell) == offsets[-1]
        ):
            raise RuntimeError(f"{subject}: inconsistent visit arrays")
        if len(subject_labels) + 1 != len(offsets):
            raise RuntimeError(f"{subject}: inconsistent bag offsets")

        eeg_features = eeg_visit_features(eeg, eeg_mask)
        et_features = et_visit_features(et, et_mask)
        all_eeg.append(eeg_features)
        all_et.append(et_features)
        all_start.append(starts)
        all_end.append(ends)
        all_dwell.append(dwell)
        all_eeg_valid.append(eeg_mask.mean(axis=(1, 2)).astype(np.float32))
        all_et_valid.append(et_mask.mean(axis=(1, 2)).astype(np.float32))
        for bag in range(len(subject_labels)):
            n_visits = int(offsets[bag + 1] - offsets[bag])
            if n_visits <= 0:
                raise RuntimeError(f"{subject}/bag{bag}: empty considered bag")
            global_offsets.append(global_offsets[-1] + n_visits)
            bag_subject.append(subject)
            labels.append(int(subject_labels[bag]))
            pages.append(int(subject_pages[bag]))
            rois.append(int(subject_rois[bag]))
            products.append(int(subject_products[bag]))
        print(
            f"FEATURES {subject}: bags={len(subject_labels):3d}, "
            f"visits={len(eeg):4d}"
        )

    arrays = {
        "eeg_visit": np.concatenate(all_eeg).astype(np.float32),
        "et_visit": np.concatenate(all_et).astype(np.float32),
        "visit_start": np.concatenate(all_start).astype(np.float64),
        "visit_end": np.concatenate(all_end).astype(np.float64),
        "visit_dwell_s": np.concatenate(all_dwell).astype(np.float32),
        "visit_eeg_valid_fraction": np.concatenate(all_eeg_valid).astype(np.float32),
        "visit_et_valid_fraction": np.concatenate(all_et_valid).astype(np.float32),
        "bag_offsets": np.asarray(global_offsets, dtype=np.int64),
        "subject": np.asarray(bag_subject, dtype="U3"),
        "label": np.asarray(labels, dtype=np.uint8),
        "page": np.asarray(pages, dtype=np.uint8),
        "roi": np.asarray(rois, dtype=np.uint8),
        "product_id": np.asarray(products, dtype=np.uint16),
        "source_cache_sha256": np.asarray(sha256_file(summary_path)),
    }
    if arrays["bag_offsets"][-1] != len(arrays["eeg_visit"]):
        raise RuntimeError("Global bag offsets do not cover visit features")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(target)
    return arrays


def load_or_build_visit_features(
    cache: Path, target: Path, rebuild: bool
) -> dict[str, np.ndarray]:
    cache_summary = cache / "cache_summary.json"
    expected_hash = sha256_file(cache_summary)
    if target.exists() and not rebuild:
        with np.load(target, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        required = {
            "eeg_visit",
            "et_visit",
            "visit_start",
            "visit_end",
            "visit_dwell_s",
            "visit_eeg_valid_fraction",
            "visit_et_valid_fraction",
            "bag_offsets",
            "subject",
            "label",
            "source_cache_sha256",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise RuntimeError(
                f"Feature cache uses an older schema ({missing} missing). "
                "Pass --rebuild-features."
            )
        stored_hash = str(np.asarray(arrays["source_cache_sha256"]).item())
        if stored_hash != expected_hash:
            raise RuntimeError(
                "Feature cache belongs to a different sequence cache. "
                "Pass --rebuild-features."
            )
        print(f"Reusing visit features: {target}")
        return arrays
    return build_visit_feature_cache(cache, target)


def horizon_visit_count(horizon: str, available: int) -> int | None:
    if horizon == "full":
        return available
    if not horizon.startswith("first"):
        raise ValueError(horizon)
    requested = int(horizon.replace("first", ""))
    return requested if available >= requested else None


def build_horizon_matrix(
    arrays: dict[str, np.ndarray], horizon: str
) -> dict[str, np.ndarray]:
    offsets = arrays["bag_offsets"].astype(np.int64)
    eeg_rows: list[np.ndarray] = []
    et_rows: list[np.ndarray] = []
    behavior_rows: list[np.ndarray] = []
    retained: list[int] = []
    for bag in range(len(offsets) - 1):
        start, end = int(offsets[bag]), int(offsets[bag + 1])
        count = horizon_visit_count(horizon, end - start)
        if count is None:
            continue
        selected_end = start + count
        eeg_rows.append(aggregate_visit_features(arrays["eeg_visit"][start:selected_end]))
        et_rows.append(aggregate_visit_features(arrays["et_visit"][start:selected_end]))
        behavior_rows.append(
            behavior_features(
                arrays["visit_start"][start:selected_end],
                arrays["visit_end"][start:selected_end],
                arrays["visit_dwell_s"][start:selected_end],
                arrays["visit_eeg_valid_fraction"][start:selected_end],
                arrays["visit_et_valid_fraction"][start:selected_end],
            )
        )
        retained.append(bag)

    retained_array = np.asarray(retained, dtype=np.int64)
    result = {
        "eeg": np.stack(eeg_rows).astype(np.float32),
        "et": np.stack(et_rows).astype(np.float32),
        "behavior": np.stack(behavior_rows).astype(np.float32),
        "label": arrays["label"][retained_array].astype(np.uint8),
        "subject": arrays["subject"][retained_array].astype("U3"),
        "page": arrays["page"][retained_array],
        "roi": arrays["roi"][retained_array],
        "product_id": arrays["product_id"][retained_array],
        "original_bag_index": retained_array,
    }
    result["fusion"] = np.concatenate([result["eeg"], result["et"]], axis=1)
    result["fusion_behavior"] = np.concatenate(
        [result["eeg"], result["et"], result["behavior"]], axis=1
    )
    return result


def make_logistic_pipeline(config: RunConfig, n_features: int) -> Pipeline:
    percentile = 100 if n_features <= 16 else config.select_percentile
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectPercentile(score_func=f_classif, percentile=percentile)),
            (
                "model",
                LogisticRegression(
                    C=config.logreg_c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=config.seed,
                ),
            ),
        ]
    )


def make_catboost_pipeline(config: RunConfig) -> Pipeline:
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise RuntimeError(
            "catboost_fusion requested but catboost is not installed"
        ) from error
    catboost_options: dict[str, Any] = {
        "iterations": config.catboost_iterations,
        "depth": config.catboost_depth,
        "learning_rate": config.catboost_learning_rate,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "auto_class_weights": "Balanced",
        "random_seed": config.seed,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": config.catboost_task_type,
    }
    if config.catboost_task_type == "GPU":
        catboost_options["devices"] = "0"
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            (
                "select",
                SelectPercentile(score_func=f_classif, percentile=config.select_percentile),
            ),
            (
                "model",
                CatBoostClassifier(**catboost_options),
            ),
        ]
    )


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            error += selected.mean() * abs(
                float(y_true[selected].mean()) - float(probability[selected].mean())
            )
    return float(error if total else math.nan)


def safe_auc(metric, y_true: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(metric(y_true, probability))


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    prediction = (probability >= threshold).astype(np.uint8)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else math.nan
    prevalence = float(np.mean(y_true))
    pr_auc = safe_auc(average_precision_score, y_true, probability)
    return {
        "n": int(len(y_true)),
        "n_buy": int(np.sum(y_true)),
        "buy_prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_lift": float(pr_auc / prevalence) if prevalence > 0 else math.nan,
        "roc_auc": safe_auc(roc_auc_score, y_true, probability),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "buy_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "buy_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "buy_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability, ece_bins),
        "threshold": float(threshold),
    }


def heuristic_probability(
    train_score: np.ndarray, test_score: np.ndarray
) -> tuple[np.ndarray, float]:
    """Map a ranking heuristic to [0,1] and choose a train-prevalence quantile."""
    train_score = np.asarray(train_score, dtype=np.float64)
    test_score = np.asarray(test_score, dtype=np.float64)
    lower = float(np.nanmin(train_score))
    upper = float(np.nanmax(train_score))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return np.full(test_score.shape, 0.5, dtype=np.float64), 0.5
    probability = np.clip((test_score - lower) / (upper - lower), 0.0, 1.0)
    return probability, math.nan


def threshold_max_mcc(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select an MCC-maximizing threshold; deterministic ties prefer 0.5."""
    y_true = np.asarray(y_true, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float64)
    finite = np.isfinite(probability)
    if finite.sum() == 0 or np.unique(y_true[finite]).size < 2:
        return 0.5
    probability = np.clip(probability[finite], 0.0, 1.0)
    labels = y_true[finite]
    unique = np.unique(probability)
    if unique.size > 500:
        candidates = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 501)))
    else:
        candidates = unique
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], candidates]))
    scored = [
        (float(matthews_corrcoef(labels, probability >= threshold)), float(threshold))
        for threshold in candidates
    ]
    best_mcc = max(score for score, _ in scored)
    best = [threshold for score, threshold in scored if abs(score - best_mcc) <= 1e-12]
    return float(min(best, key=lambda value: (abs(value - 0.5), value)))


def inner_group_threshold(
    estimator: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    inner_folds: int,
) -> float:
    """Fit subject-grouped inner models and select threshold from OOF scores."""
    unique_groups = np.unique(groups)
    n_splits = min(inner_folds, len(unique_groups))
    if n_splits < 2:
        return 0.5
    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for inner_train, inner_valid in splitter.split(x, y, groups):
        if np.unique(y[inner_train]).size < 2:
            continue
        inner_model = clone(estimator)
        inner_model.fit(x[inner_train], y[inner_train])
        oof[inner_valid] = inner_model.predict_proba(x[inner_valid])[:, 1]
    valid = np.isfinite(oof)
    return threshold_max_mcc(y[valid], oof[valid]) if valid.any() else 0.5


def product_propensity_probability(
    train_product: np.ndarray,
    train_label: np.ndarray,
    test_product: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Training-only beta-smoothed purchase propensity for each catalogue item."""
    prevalence = float(np.mean(train_label))
    frame = pd.DataFrame(
        {"product_id": train_product.astype(int), "label": train_label.astype(int)}
    )
    grouped = frame.groupby("product_id")["label"].agg(["sum", "count"])
    propensity = (
        grouped["sum"] + smoothing * prevalence
    ) / (grouped["count"] + smoothing)
    return np.asarray(
        [float(propensity.get(int(product), prevalence)) for product in test_product],
        dtype=np.float64,
    )


def inner_product_propensity_threshold(
    products: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    inner_folds: int,
    smoothing: float,
) -> float:
    unique_groups = np.unique(groups)
    n_splits = min(inner_folds, len(unique_groups))
    if n_splits < 2:
        return 0.5
    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(y), 1), dtype=np.float32)
    for inner_train, inner_valid in splitter.split(dummy, y, groups):
        oof[inner_valid] = product_propensity_probability(
            products[inner_train], y[inner_train], products[inner_valid], smoothing
        )
    valid = np.isfinite(oof)
    return threshold_max_mcc(y[valid], oof[valid]) if valid.any() else 0.5


def evaluate_horizon(
    horizon: str,
    matrix: dict[str, np.ndarray],
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = matrix["label"].astype(np.uint8)
    subjects = matrix["subject"].astype(str)
    unique_subjects = sorted(np.unique(subjects))
    predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []

    modality_for_model = {
        "logreg_behavior": "behavior",
        "logreg_eeg": "eeg",
        "logreg_et": "et",
        "logreg_fusion": "fusion",
        "catboost_fusion": "fusion_behavior",
    }

    for fold_index, held_out in enumerate(unique_subjects, start=1):
        test_mask = subjects == held_out
        train_mask = ~test_mask
        y_train, y_test = y[train_mask], y[test_mask]
        train_groups = subjects[train_mask]
        if np.unique(y_train).size < 2:
            raise RuntimeError(f"{horizon}/{held_out}: training set has one class")

        print(
            f"{horizon} fold {fold_index:02d}/{len(unique_subjects)} "
            f"held={held_out} train={train_mask.sum()} test={test_mask.sum()}"
        )
        train_prevalence = float(y_train.mean())
        for model_name in config.models:
            if model_name == "prevalence":
                probability = np.full(int(test_mask.sum()), train_prevalence)
                threshold = 0.5
            elif model_name == "product_propensity":
                probability = product_propensity_probability(
                    matrix["product_id"][train_mask],
                    y_train,
                    matrix["product_id"][test_mask],
                    config.product_smoothing,
                )
                threshold = (
                    inner_product_propensity_threshold(
                        matrix["product_id"][train_mask],
                        y_train,
                        train_groups,
                        config.inner_folds,
                        config.product_smoothing,
                    )
                    if config.threshold_mode == "inner_mcc"
                    else 0.5
                )
            elif model_name in {"visit_count", "total_dwell"}:
                column = 0 if model_name == "visit_count" else 1
                train_score = matrix["behavior"][train_mask, column]
                test_score = matrix["behavior"][test_mask, column]
                if float(np.max(train_score)) <= float(np.min(train_score)):
                    probability = np.full(int(test_mask.sum()), train_prevalence)
                    threshold = 0.5
                else:
                    probability, _ = heuristic_probability(train_score, test_score)
                    train_probability, _ = heuristic_probability(train_score, train_score)
                    threshold = (
                        threshold_max_mcc(y_train, train_probability)
                        if config.threshold_mode == "inner_mcc"
                        else 0.5
                    )
            else:
                modality = modality_for_model[model_name]
                x = matrix[modality]
                if model_name.startswith("logreg"):
                    estimator = make_logistic_pipeline(config, x.shape[1])
                elif model_name == "catboost_fusion":
                    estimator = make_catboost_pipeline(config)
                else:
                    raise ValueError(model_name)
                estimator = clone(estimator)
                threshold = (
                    inner_group_threshold(
                        estimator,
                        x[train_mask],
                        y_train,
                        train_groups,
                        config.inner_folds,
                    )
                    if config.threshold_mode == "inner_mcc"
                    else 0.5
                )
                estimator.fit(x[train_mask], y_train)
                probability = estimator.predict_proba(x[test_mask])[:, 1]

            metrics = calculate_metrics(
                y_test, probability, threshold=threshold, ece_bins=config.ece_bins
            )
            fold_metrics.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "held_out_subject": held_out,
                    **metrics,
                }
            )
            test_indices = np.flatnonzero(test_mask)
            for local_index, global_index in enumerate(test_indices):
                predictions.append(
                    {
                        "horizon": horizon,
                        "model": model_name,
                        "held_out_subject": held_out,
                        "label_buy": int(y[global_index]),
                        "probability_buy": float(probability[local_index]),
                        "threshold": float(threshold),
                        "prediction_buy": int(probability[local_index] >= threshold),
                        "page": int(matrix["page"][global_index]),
                        "roi": int(matrix["roi"][global_index]),
                        "product_id": int(matrix["product_id"][global_index]),
                        "original_bag_index": int(matrix["original_bag_index"][global_index]),
                    }
                )

    return pd.DataFrame(predictions), pd.DataFrame(fold_metrics)


def summarize_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    ece_bins: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (horizon, model), group in predictions.groupby(["horizon", "model"], sort=False):
        pooled = calculate_metrics(
            group["label_buy"].to_numpy(dtype=np.uint8),
            group["probability_buy"].to_numpy(dtype=float),
            # Fold thresholds may differ for heuristics, so thresholded metrics
            # are read directly below rather than recomputed with one threshold.
            threshold=0.5,
            ece_bins=ece_bins,
        )
        direct_prediction = group["prediction_buy"].to_numpy(dtype=np.uint8)
        labels = group["label_buy"].to_numpy(dtype=np.uint8)
        tn, fp, fn, tp = confusion_matrix(labels, direct_prediction, labels=[0, 1]).ravel()
        pooled.update(
            {
                "accuracy": float(accuracy_score(labels, direct_prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, direct_prediction)
                ),
                "buy_f1": float(f1_score(labels, direct_prediction, zero_division=0)),
                "buy_precision": float(
                    precision_score(labels, direct_prediction, zero_division=0)
                ),
                "buy_recall": float(recall_score(labels, direct_prediction, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
                "mcc": float(matthews_corrcoef(labels, direct_prediction)),
                "threshold": math.nan,
            }
        )
        folds = fold_metrics.loc[
            (fold_metrics["horizon"] == horizon) & (fold_metrics["model"] == model)
        ]
        record: dict[str, Any] = {"horizon": horizon, "model": model, **pooled}
        for metric in (
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "buy_f1",
            "buy_recall",
            "mcc",
            "brier",
            "ece",
        ):
            record[f"mean_subject_{metric}"] = float(folds[metric].mean())
            record[f"std_subject_{metric}"] = float(folds[metric].std(ddof=1))
        records.append(record)
    return pd.DataFrame.from_records(records)


def run_self_test() -> None:
    rng = np.random.default_rng(42)
    eeg = rng.normal(size=(4, 300, 19)).astype(np.float32)
    eeg_mask = np.ones_like(eeg, dtype=np.uint8)
    eeg_mask[0, :20] = 0
    eeg_features = eeg_visit_features(eeg, eeg_mask)
    assert eeg_features.shape == (4, 152)
    assert np.isfinite(eeg_features).all()

    et = rng.normal(size=(4, 120, 6)).astype(np.float32)
    et_mask = np.ones_like(et, dtype=np.uint8)
    et_mask[1, :10, :3] = 0
    et_features = et_visit_features(et, et_mask)
    assert et_features.shape == (4, 69)
    assert np.isfinite(et_features).all()
    assert aggregate_visit_features(eeg_features[:2]).shape == (760,)
    assert aggregate_visit_features(et_features[:2]).shape == (345,)
    behavior = behavior_features(
        np.array([1.0, 2.0]),
        np.array([1.2, 2.3]),
        np.array([0.2, 0.3]),
        np.array([0.9, 1.0]),
        np.array([0.8, 0.9]),
    )
    assert behavior.shape == (10,)
    metrics = calculate_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), 0.5, 5
    )
    assert metrics["pr_auc"] == 1.0 and metrics["mcc"] == 1.0
    threshold = threshold_max_mcc(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )
    assert 0.4 < threshold <= 0.6
    propensity = product_propensity_probability(
        np.array([1, 1, 2, 2]),
        np.array([1, 1, 0, 0]),
        np.array([1, 2, 3]),
        smoothing=2.0,
    )
    assert propensity[0] > propensity[1] and math.isclose(propensity[2], 0.5)

    n = 24
    labels = np.tile(np.array([0, 0, 0, 1, 0, 1], dtype=np.uint8), 4)
    synthetic = {
        "label": labels,
        "subject": np.repeat(np.array(["S01", "S02", "S03", "S04"]), 6),
        "behavior": np.column_stack(
            [np.ones(n), labels + rng.normal(0.0, 0.25, n)]
        ).astype(np.float32),
        "product_id": np.tile(np.arange(1, 7), 4),
        "page": np.ones(n, dtype=np.uint8),
        "roi": np.tile(np.arange(1, 7), 4).astype(np.uint8),
        "original_bag_index": np.arange(n),
    }
    synthetic_config = RunConfig(
        cache="synthetic",
        output="synthetic",
        feature_cache="synthetic",
        horizons=("first1",),
        models=("prevalence", "product_propensity", "logreg_behavior"),
        inner_folds=3,
    )
    predictions, folds = evaluate_horizon("first1", synthetic, synthetic_config)
    assert len(predictions) == n * len(synthetic_config.models)
    assert len(folds) == 4 * len(synthetic_config.models)
    print("SELF-TEST: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    cache = args.cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_cache = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else output / "visit_features.npz"
    )
    if not (1 <= args.select_percentile <= 100):
        raise ValueError("--select-percentile must be between 1 and 100")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be at least 2")
    if args.inner_folds < 2:
        raise ValueError("--inner-folds must be at least 2")
    if args.product_smoothing < 0:
        raise ValueError("--product-smoothing cannot be negative")

    config = RunConfig(
        cache=str(cache),
        output=str(output),
        feature_cache=str(feature_cache),
        horizons=tuple(args.horizons),
        models=tuple(args.models),
        seed=args.seed,
        select_percentile=args.select_percentile,
        logreg_c=args.logreg_c,
        catboost_iterations=args.catboost_iterations,
        catboost_depth=args.catboost_depth,
        catboost_learning_rate=args.catboost_learning_rate,
        ece_bins=args.ece_bins,
        inner_folds=args.inner_folds,
        threshold_mode=args.threshold_mode,
        product_smoothing=args.product_smoothing,
        catboost_task_type=args.catboost_task_type,
    )

    print("=" * 78)
    print("NEUROCLICK CLASSICAL STRICT LOSOCV BENCHMARKS")
    print("=" * 78)
    print(f"Sequence cache        : {cache}")
    print(f"Output                : {output}")
    print(f"Visit feature cache   : {feature_cache}")
    print(f"Horizons              : {', '.join(config.horizons)}")
    print(f"Models                : {', '.join(config.models)}")
    print("Primary metric        : PR-AUC (average precision)")
    print("Primary horizon       : first1 matched-visit risk set")
    print("Outer validation      : strict leave-one-subject-out")
    print(
        "Decision threshold    : "
        + (
            f"MCC from {config.inner_folds}-fold subject-grouped inner OOF"
            if config.threshold_mode == "inner_mcc"
            else "fixed 0.5"
        )
    )

    arrays = load_or_build_visit_features(cache, feature_cache, args.rebuild_features)
    if len(arrays["label"]) != 5715 or int(arrays["label"].sum()) != 746:
        raise RuntimeError(
            f"Expected frozen cohort 5715/746, got "
            f"{len(arrays['label'])}/{int(arrays['label'].sum())}"
        )
    if len(np.unique(arrays["subject"])) != 42:
        raise RuntimeError("Expected 42 subjects in the frozen feature cache")

    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    horizon_counts: list[dict[str, Any]] = []
    for horizon in config.horizons:
        matrix = build_horizon_matrix(arrays, horizon)
        counts = {
            "horizon": horizon,
            "instances": int(len(matrix["label"])),
            "buys": int(matrix["label"].sum()),
            "nobuys": int(len(matrix["label"]) - matrix["label"].sum()),
            "subjects": int(len(np.unique(matrix["subject"]))),
        }
        horizon_counts.append(counts)
        print(
            f"\nHORIZON {horizon}: n={counts['instances']}, "
            f"Buy={counts['buys']}, NoBuy={counts['nobuys']}"
        )
        predictions, fold_metrics = evaluate_horizon(horizon, matrix, config)
        prediction_frames.append(predictions)
        fold_frames.append(fold_metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    summary = summarize_metrics(predictions, fold_metrics, config.ece_bins)
    horizon_frame = pd.DataFrame.from_records(horizon_counts)

    predictions.to_csv(output / "losocv_predictions.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    horizon_frame.to_csv(output / "horizon_counts.csv", index=False)

    run_record = {
        "decision": "BENCHMARK_GO",
        "config": asdict(config),
        "cohort": {
            "subjects": 42,
            "instances": 5715,
            "buys": 746,
            "nobuys": 4969,
        },
        "horizon_counts": horizon_counts,
        "hashes": {
            "sequence_cache_summary_sha256": sha256_file(cache / "cache_summary.json"),
            "visit_feature_cache_sha256": sha256_file(feature_cache),
            "benchmark_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    (output / "BENCHMARK_DECISION.txt").write_text(
        "BENCHMARK_GO\n\n"
        "Strict 42-fold LOSOCV completed.\n"
        "Primary horizon: first1.\n"
        "Primary metric: PR-AUC.\n",
        encoding="utf-8",
    )

    display_columns = [
        "horizon",
        "model",
        "n",
        "n_buy",
        "pr_auc",
        "pr_lift",
        "roc_auc",
        "balanced_accuracy",
        "buy_f1",
        "buy_recall",
        "mcc",
    ]
    print("\n" + "=" * 78)
    print("DECISION: BENCHMARK_GO")
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved outputs to: {output}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
