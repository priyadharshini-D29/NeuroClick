#!/usr/bin/env python3
"""Generate publication figures from saved NeuroClick LOSOCV outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# Publication palette. Data series never use grey.
TEAL = "#007C83"
BLUE = "#315F87"
MAGENTA = "#A23E8C"
GOLD = "#D89B2B"
VIOLET = "#6F4E9C"
CYAN = "#4A9297"
DEEP_TEAL = "#006C73"
ROSE = "#C44E62"
PALE_BLUE = "#DCEAF0"
PAIR_BLUE = "#B8D9E4"
TEXT = "#1F3540"
WHITE = "#FFFFFF"

MODEL_LABELS = {
    "behavior": "NeuroClick behavior",
    "logreg_behavior": "Logistic behavior",
    "total_dwell": "Total dwell",
    "logreg_dwell_propensity": "Dwell + propensity",
    "behavior_no_propensity": "NeuroClick (no propensity)",
    "behavior_et": "Behavior + ET",
    "behavior_eeg": "Behavior + EEG",
    "behavior_eeg_et": "Behavior + EEG + ET",
}

MODEL_COLORS = {
    "behavior": TEAL,
    "logreg_behavior": BLUE,
    "total_dwell": MAGENTA,
    "logreg_dwell_propensity": GOLD,
    "behavior_no_propensity": ROSE,
    "behavior_et": CYAN,
    "behavior_eeg": VIOLET,
    "behavior_eeg_et": DEEP_TEAL,
}

HORIZON_ORDER = ["first1", "first2", "first3", "full"]
HORIZON_LABELS = ["First visit", "First two", "First three", "Complete history"]
CONDITION_ORDER = ["behavior", "behavior_et", "behavior_eeg", "behavior_eeg_et"]

SUBJECT_ALIASES = (
    "heldout_subject",
    "held_out_subject",
    "test_subject",
    "subject_id",
    "subject",
    "participant_id",
    "participant",
    "fold_subject",
)
HORIZON_ALIASES = ("horizon", "prefix", "endpoint", "visit_horizon")
MODEL_ALIASES = ("condition", "model", "model_name", "ablation", "feature_set")
TRUE_ALIASES = (
    "y_true",
    "true_label",
    "label",
    "label_buy",
    "target",
    "buy_label",
    "is_buy",
)
SCORE_ALIASES = (
    "y_prob",
    "y_score",
    "probability",
    "probability_buy",
    "prob_buy",
    "buy_probability",
    "risk",
    "score",
    "predicted_probability",
)
PRED_ALIASES = (
    "y_pred",
    "pred_label",
    "prediction_label",
    "prediction_buy",
    "predicted_label",
    "class_pred",
    "prediction",
    "predicted",
)
THRESHOLD_ALIASES = ("threshold", "decision_threshold", "selected_threshold")
IDENTIFIER_ALIASES = (
    "bag_index",
    "instance_index",
    "sample_index",
    "product_index",
    "participant_product_index",
)


@dataclass(frozen=True)
class Columns:
    subject: str
    horizon: str
    model: str
    y_true: str
    y_score: str
    y_pred: str | None
    threshold: str | None
    identifier: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root of ICAIN2026_NeuroClick_Hazard.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <project-root>/outputs/manuscript_figures_v1",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="Optional explicit final causal .npz cache.",
    )
    return parser.parse_args()


def normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def resolve_column(
    frame: pd.DataFrame, aliases: Sequence[str], required: bool = True
) -> str | None:
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
        identifier=resolve_column(frame, IDENTIFIER_ALIASES, required=False),
    )


def canonicalize_predictions(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
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
        result["y_pred"] = np.nan

    if columns.identifier is not None:
        result["identifier"] = frame[columns.identifier].astype(str)
    else:
        # Stable within-file row identity. It is used only for duplicate audits.
        result["identifier"] = np.arange(len(frame)).astype(str)

    result["source"] = str(source)
    required = ["subject", "horizon", "model", "y_true", "y_score"]
    bad = result[required].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{source}: {int(bad.sum())} rows contain missing values")
    result["y_true"] = result["y_true"].astype(int)
    if not set(result["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{source}: y_true is not binary")
    finite_pred = result["y_pred"].notna()
    if finite_pred.any():
        result.loc[finite_pred, "y_pred"] = result.loc[finite_pred, "y_pred"].astype(int)
        if not set(result.loc[finite_pred, "y_pred"].unique()).issubset({0, 1}):
            raise ValueError(f"{source}: y_pred is not binary")
    if not np.isfinite(result["y_score"]).all():
        raise ValueError(f"{source}: y_score contains non-finite values")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_manifest_path(project: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(project / path)
    else:
        try:
            outputs_index = path.parts.index("outputs")
            candidates.append(project.joinpath(*path.parts[outputs_index:]))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Prediction path from manifest not found: {raw_path}")


def discover_prediction_paths(project: Path) -> tuple[list[Path], Path | None]:
    stats_dir = project / "outputs/statistics/neuroclick_primary_v1"
    manifest_path = stats_dir / "analysis_manifest.json"
    paths: list[Path] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_paths = list(manifest.get("neural_prediction_files", []))
        raw_paths += list(manifest.get("classical_prediction_files", []))
        paths = [resolve_manifest_path(project, item) for item in raw_paths]
    else:
        patterns = [
            "outputs/benchmarks/neuroclick_hazard_pilot5_v1/losocv_predictions.csv",
            "outputs/benchmarks/neuroclick_hazard_remaining37_v1/losocv_predictions.csv",
            "outputs/benchmarks/classical_first1*/losocv_predictions.csv",
            "outputs/benchmarks/classical_secondary_horizons*/losocv_predictions.csv",
        ]
        for pattern in patterns:
            paths.extend(sorted(project.glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError(
            "No LOSOCV prediction CSV files were found. Expected the statistical "
            "analysis manifest or final benchmark output folders."
        )
    return unique, manifest_path if manifest_path.exists() else None


def read_predictions(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        print(f"Reading predictions: {path}")
        frames.append(canonicalize_predictions(pd.read_csv(path), path))
    predictions = pd.concat(frames, ignore_index=True)

    # Overlapping neural pilot/remaining subjects indicate an invalid merge.
    neural_sources = [
        (source, set(group["subject"]))
        for source, group in predictions.groupby("source")
        if "neuroclick_hazard" in source
    ]
    for index, (left_name, left_subjects) in enumerate(neural_sources):
        for right_name, right_subjects in neural_sources[index + 1 :]:
            overlap = sorted(left_subjects & right_subjects)
            if overlap:
                raise ValueError(
                    "Neural prediction sources contain overlapping held-out subjects: "
                    f"{left_name}, {right_name}: {overlap}"
                )
    return predictions


def find_feature_cache(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates: list[Path] = []
    roots = [
        project / "cache",
        Path("/mnt/Neuma_Model/icain_neuroclick_work/cache/preclick_full_42"),
        project / "outputs",
    ]
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob("*.npz"))

    valid: list[Path] = []
    for path in candidates:
        try:
            with np.load(path, allow_pickle=False) as data:
                if {"bag_offsets", "label", "subject"}.issubset(data.files):
                    valid.append(path.resolve())
        except Exception:
            continue
    if not valid:
        raise FileNotFoundError(
            "Could not locate the final causal feature cache. Re-run with "
            "--feature-cache /absolute/path/to/final_cache.npz"
        )
    valid.sort(
        key=lambda path: (
            "preclick_full_42" not in str(path),
            "causal" not in path.name.lower(),
            len(str(path)),
        )
    )
    selected = valid[0]
    print(f"Using feature cache: {selected}")
    return selected


def configure_plotting() -> None:
    available = {font.name for font in mpl.font_manager.fontManager.ttflist}
    font = "Nimbus Sans" if "Nimbus Sans" in available else "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 8.4,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=PALE_BLUE, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)


def save_figure(fig: plt.Figure, output_base: Path) -> dict[str, str]:
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    with Image.open(png) as image:
        image.convert("RGB").save(png, format="PNG", optimize=True, dpi=(600, 600))
    return {"png": str(png), "pdf": str(pdf)}


def sequence_categories(lengths: np.ndarray) -> tuple[list[str], np.ndarray]:
    labels = ["1", "2", "3", "4", "5", "6-10", "11+"]
    category = np.full(len(lengths), 6, dtype=int)
    category[lengths == 1] = 0
    category[lengths == 2] = 1
    category[lengths == 3] = 2
    category[lengths == 4] = 3
    category[lengths == 5] = 4
    category[(lengths >= 6) & (lengths <= 10)] = 5
    return labels, category


def figure_dataset_profile(cache: Path, output: Path) -> tuple[dict[str, str], dict]:
    with np.load(cache, allow_pickle=False) as data:
        offsets = np.asarray(data["bag_offsets"], dtype=np.int64)
        labels = np.asarray(data["label"], dtype=int)
        subjects = np.asarray(data["subject"]).astype(str)
    lengths = np.diff(offsets)
    if len(lengths) != len(labels):
        raise ValueError("Feature-cache bag_offsets and labels disagree")

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.75), dpi=600)
    categories, encoded = sequence_categories(lengths)
    x = np.arange(len(categories))
    width = 0.36
    for shift, value, color, name in [
        (-width / 2, 0, BLUE, "NoBuy"),
        (width / 2, 1, ROSE, "Buy"),
    ]:
        subset = encoded[labels == value]
        percentage = np.bincount(subset, minlength=len(categories)).astype(float)
        percentage = percentage / percentage.sum() * 100.0
        axes[0].bar(
            x + shift,
            percentage,
            width,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            label=name,
            zorder=3,
        )
    axes[0].set_title("Sequence-length distribution", weight="bold")
    axes[0].set_xlabel("Observed visits per product sequence")
    axes[0].set_ylabel("Within-class percentage")
    axes[0].set_xticks(x, categories)
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    totals = np.asarray(
        [len(labels), int((lengths >= 2).sum()), int((lengths >= 3).sum()), len(labels)]
    )
    buys = np.asarray(
        [
            int(labels.sum()),
            int(((lengths >= 2) & (labels == 1)).sum()),
            int(((lengths >= 3) & (labels == 1)).sum()),
            int(labels.sum()),
        ]
    )
    x2 = np.arange(4)
    axes[1].bar(
        x2 - width / 2,
        totals,
        width,
        color=TEAL,
        edgecolor=WHITE,
        linewidth=0.5,
        label="All instances",
        zorder=3,
    )
    axes[1].bar(
        x2 + width / 2,
        buys,
        width,
        color=MAGENTA,
        edgecolor=WHITE,
        linewidth=0.5,
        label="Buy instances",
        zorder=3,
    )
    for xpos, value in zip(x2 - width / 2, totals):
        axes[1].text(xpos, value + totals.max() * 0.018, f"{value:,}", ha="center", va="bottom", fontsize=6.8, color=TEAL)
    for xpos, value in zip(x2 + width / 2, buys):
        axes[1].text(xpos, value + totals.max() * 0.018, f"{value:,}", ha="center", va="bottom", fontsize=6.8, color=MAGENTA)
    axes[1].set_title("Available instances by horizon", weight="bold")
    axes[1].set_ylabel("Number of participant-product instances")
    axes[1].set_xticks(x2, HORIZON_LABELS, rotation=16, ha="right")
    axes[1].set_ylim(0, totals.max() * 1.18)
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    fig.tight_layout(w_pad=1.25)

    info = {
        "n_instances": int(len(labels)),
        "n_buy": int(labels.sum()),
        "n_nobuy": int((labels == 0).sum()),
        "n_visits": int(lengths.sum()),
        "n_subjects": int(len(np.unique(subjects))),
        "horizon_total_counts": dict(zip(HORIZON_ORDER, totals.tolist())),
        "horizon_buy_counts": dict(zip(HORIZON_ORDER, buys.tolist())),
    }
    return save_figure(fig, output / "fig01_dataset_sequence_profile"), info


def locate_model_script(project: Path) -> Path:
    candidates = [
        project / "scripts/04_run_neuroclick_hazard.py",
        project / "04_run_neuroclick_hazard.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    found = sorted(project.rglob("04_run_neuroclick_hazard.py"))
    if not found:
        raise FileNotFoundError("04_run_neuroclick_hazard.py was not found")
    return found[0].resolve()


def inspect_architecture(project: Path, cache: Path) -> dict[str, int]:
    script = locate_model_script(project)
    spec = importlib.util.spec_from_file_location("neuroclick_model_for_figure", script)
    if spec is None or spec.loader is None:
        raise ImportError(script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    store = module.FeatureStore(cache)
    model = module.NeuroClickHazard(
        behavior_dim=int(store.behavior_visit.shape[1]),
        eeg_dim=int(store.eeg_model_visit.shape[1]),
        et_dim=int(store.et_model_visit.shape[1]),
        hidden_dim=64,
        modality_dim=64,
        dropout=0.20,
        condition="behavior_eeg_et",
        initial_hazard=float(np.clip(store.label.mean(), 1e-4, 1 - 1e-4)),
    )
    behavior_first = model.behavior_encoder.network[0]
    eeg_first = model.eeg_encoder.network[0]
    et_first = model.et_encoder.network[0]
    return {
        "behavior_input": int(behavior_first.in_features),
        "eeg_input": int(eeg_first.in_features),
        "et_input": int(et_first.in_features),
        "projection_dim": int(behavior_first.out_features),
        "gru_input": int(model.gru.input_size),
        "gru_hidden": int(model.gru.hidden_size),
    }


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str,
    edge: str,
    title: str,
    subtitle: str = "",
    title_size: float = 6.6,
    subtitle_size: float = 5.1,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.61 if subtitle else 0.50),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
        color=TEXT,
        linespacing=1.0,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.27,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=TEXT,
        )


def diagram_arrow(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.2,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def figure_architecture(
    architecture: dict[str, int], output: Path
) -> tuple[dict[str, str], dict]:
    fig, ax = plt.subplots(figsize=(6.25, 2.08), dpi=600)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_x = [0.095, 0.275, 0.485, 0.675, 0.82, 0.945]
    stage_names = ["VISIT INPUTS", "PROJECTIONS", "GATED FUSION", "TEMPORAL", "HAZARD", "BUY RISK"]
    for x, name in zip(stage_x, stage_names):
        ax.text(x, 0.955, name, ha="center", va="top", fontsize=5.6, weight="bold", color=TEXT)

    rows = [
        (0.69, "#DDF2F2", TEAL, "Behavior", f"{architecture['behavior_input']} inputs"),
        (0.395, "#E5ECF8", BLUE, "EEG", f"{architecture['eeg_input']} inputs"),
        (0.10, "#F0E8F7", VIOLET, "Eye tracking", f"{architecture['et_input']} inputs"),
    ]
    for y, face, edge, title, subtitle in rows:
        rounded_box(ax, 0.01, y, 0.165, 0.19, face, edge, title, subtitle)
        rounded_box(
            ax,
            0.205,
            y,
            0.145,
            0.19,
            WHITE,
            edge,
            f"{title}\nprojection",
            f"{architecture['projection_dim']}-D",
            title_size=6.0,
        )
        diagram_arrow(ax, (0.175, y + 0.095), (0.205, y + 0.095), edge)

    fusion = FancyBboxPatch(
        (0.398, 0.10),
        0.175,
        0.785,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor="#E5F3F3",
        edgecolor=DEEP_TEAL,
        linewidth=1.4,
    )
    ax.add_patch(fusion)
    ax.text(0.4855, 0.825, "Residual gated fusion", ha="center", va="center", fontsize=6.5, weight="bold", color=TEXT)
    gates = [
        (0.64, "#DDF2F2", TEAL, "Behavior base"),
        (0.405, "#E5ECF8", BLUE, r"$q_E \odot$ EEG"),
        (0.17, "#F0E8F7", VIOLET, r"$q_G \odot$ eye tracking"),
    ]
    for y, face, edge, label in gates:
        rounded_box(ax, 0.42, y, 0.131, 0.112, face, edge, label, title_size=5.6)
    ax.text(0.4855, 0.125, "LayerNorm", ha="center", va="center", fontsize=5.4, color=TEXT)

    targets = [0.696, 0.461, 0.226]
    for (y, _, edge, _, _), target in zip(rows, targets):
        diagram_arrow(ax, (0.35, y + 0.095), (0.42, target), edge)

    rounded_box(
        ax,
        0.625,
        0.30,
        0.105,
        0.38,
        "#E5ECF8",
        BLUE,
        "GRU",
        f"{architecture['gru_hidden']} hidden units",
    )
    diagram_arrow(ax, (0.573, 0.50), (0.625, 0.50), DEEP_TEAL)
    rounded_box(ax, 0.775, 0.34, 0.09, 0.30, "#F5EEDB", GOLD, "Sigmoid", r"hazard $h_t$")
    diagram_arrow(ax, (0.73, 0.50), (0.775, 0.50), BLUE)
    rounded_box(ax, 0.91, 0.29, 0.085, 0.42, "#E4F1F5", CYAN, "Cumulative\nrisk", "$p^{(t)}$\nmonotonic", title_size=5.8)
    diagram_arrow(ax, (0.865, 0.50), (0.91, 0.50), TEAL)
    return save_figure(fig, output / "fig02_neuroclick_architecture"), architecture


def pooled_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    y_true = group["y_true"].to_numpy(dtype=int)
    y_score = group["y_score"].to_numpy(dtype=float)
    result: dict[str, float | int] = {
        "n": int(len(group)),
        "n_buy": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    if group["y_pred"].notna().all():
        result["mcc"] = float(
            matthews_corrcoef(y_true, group["y_pred"].to_numpy(dtype=int))
        )
    return result


def require_group(
    predictions: pd.DataFrame, horizon: str, model: str
) -> pd.DataFrame:
    group = predictions[
        (predictions["horizon"] == horizon) & (predictions["model"] == model)
    ].copy()
    if group.empty:
        raise ValueError(
            f"Missing predictions for {model}/{horizon}. Available models: "
            f"{sorted(predictions['model'].unique())}"
        )
    return group


def figure_pr_roc(
    predictions: pd.DataFrame, output: Path
) -> tuple[dict[str, str], dict]:
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.85), dpi=600)
    metrics = {}
    styles = {
        "behavior": (TEAL, "-", "o"),
        "logreg_behavior": (BLUE, (0, (5, 2)), "s"),
        "total_dwell": (MAGENTA, (0, (2, 1.5)), "D"),
        "logreg_dwell_propensity": (GOLD, (0, (1, 1)), "^"),
    }
    for model in ["behavior", "logreg_behavior", "total_dwell", "logreg_dwell_propensity"]:
        group = require_group(predictions, "first1", model)
        y_true = group["y_true"].to_numpy(dtype=int)
        y_score = group["y_score"].to_numpy(dtype=float)
        metric = pooled_metrics(group)
        metrics[model] = metric
        color, linestyle, marker = styles[model]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        false_positive, true_positive, _ = roc_curve(y_true, y_score)
        axes[0].plot(
            recall,
            precision,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker=marker,
            markevery=max(1, len(recall) // 14),
            markersize=2.8,
            label=f"{MODEL_LABELS[model]} (AUC={metric['pr_auc']:.4f})",
            zorder=3,
        )
        axes[1].plot(
            false_positive,
            true_positive,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker=marker,
            markevery=max(1, len(false_positive) // 14),
            markersize=2.8,
            label=f"{MODEL_LABELS[model]} (AUC={metric['roc_auc']:.4f})",
            zorder=3,
        )

    prevalence = float(metrics["behavior"]["prevalence"])
    axes[0].axhline(
        prevalence,
        color=GOLD,
        linestyle=(0, (1.5, 1.5)),
        linewidth=1.6,
        label=f"Observed prevalence ({prevalence:.4f})",
    )
    axes[1].plot([0, 1], [0, 1], color=GOLD, linestyle=(0, (1.5, 1.5)), linewidth=1.5, label="Chance line")

    axes[0].set_title("Precision-recall curve", weight="bold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[1].set_title("Receiver-operating characteristic", weight="bold")
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.legend(frameon=False, loc="lower right" if ax is axes[1] else "upper right")
        style_axis(ax, grid_axis="both")
    fig.tight_layout(w_pad=1.25)
    return save_figure(fig, output / "fig03_first1_pr_roc_curves"), metrics


def compute_horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZON_ORDER:
        for model in ["behavior", "logreg_behavior"]:
            group = require_group(predictions, horizon, model)
            metric = pooled_metrics(group)
            rows.append({"horizon": horizon, "model": model, **metric})
    return pd.DataFrame(rows)


def figure_horizons(
    predictions: pd.DataFrame, output: Path
) -> tuple[dict[str, str], dict]:
    metrics = compute_horizon_metrics(predictions)
    fig, ax = plt.subplots(figsize=(6.25, 3.05), dpi=600)
    x = np.arange(4)
    behavior = metrics[metrics["model"] == "behavior"].set_index("horizon").loc[HORIZON_ORDER]
    logistic = metrics[metrics["model"] == "logreg_behavior"].set_index("horizon").loc[HORIZON_ORDER]
    prevalence = behavior["prevalence"].to_numpy(dtype=float)
    neural = behavior["pr_auc"].to_numpy(dtype=float)
    baseline = logistic["pr_auc"].to_numpy(dtype=float)
    difference = neural - baseline

    ax.plot(x, neural, color=TEAL, linewidth=2.0, marker="o", markersize=5.0, label="NeuroClick behavior", zorder=4)
    ax.plot(x, baseline, color=BLUE, linewidth=1.8, linestyle=(0, (5, 2)), marker="s", markersize=4.8, label="Logistic behavior", zorder=3)
    ax.plot(x, prevalence, color=GOLD, linewidth=1.6, linestyle=(0, (1.5, 1.5)), marker="D", markersize=4.1, label="Observed prevalence", zorder=3)
    for index, (value, delta) in enumerate(zip(neural, difference)):
        offset = 0.017 if delta >= 0 else -0.027
        ax.text(index, value + offset, f"{delta:+.4f}", color=TEAL if delta >= 0 else MAGENTA, ha="center", va="center", fontsize=7.2, weight="bold")
    ax.set_title("PR AUC across observation horizons", weight="bold")
    ax.set_ylabel("Pooled PR AUC")
    ax.set_xticks(x, HORIZON_LABELS)
    low = max(0.0, min(prevalence.min(), neural.min(), baseline.min()) - 0.04)
    high = min(1.0, max(prevalence.max(), neural.max(), baseline.max()) + 0.07)
    ax.set_ylim(low, high)
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    info = {
        "metrics": metrics.to_dict(orient="records"),
        "neuroclick_minus_logistic_pr_auc": dict(zip(HORIZON_ORDER, difference.tolist())),
    }
    return save_figure(fig, output / "fig04_horizon_pr_auc"), info


def figure_ablation(
    predictions: pd.DataFrame, output: Path
) -> tuple[dict[str, str], dict]:
    matrix = np.zeros((len(CONDITION_ORDER), len(HORIZON_ORDER)), dtype=float)
    counts = {}
    for row, model in enumerate(CONDITION_ORDER):
        for column, horizon in enumerate(HORIZON_ORDER):
            group = require_group(predictions, horizon, model)
            metric = pooled_metrics(group)
            matrix[row, column] = float(metric["pr_auc"])
            counts[f"{model}/{horizon}"] = {"n": metric["n"], "n_buy": metric["n_buy"]}

    cmap = LinearSegmentedColormap.from_list(
        "teal_blue_violet", ["#E6F5F5", CYAN, TEAL, BLUE, VIOLET]
    )
    fig, ax = plt.subplots(figsize=(6.25, 2.75), dpi=600)
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=float(matrix.min() - 0.01), vmax=float(matrix.max() + 0.01))
    ax.set_xticks(np.arange(4), HORIZON_LABELS)
    ax.set_yticks(np.arange(4), [MODEL_LABELS[item] for item in CONDITION_ORDER])
    ax.set_title("Modality ablation: pooled PR AUC", weight="bold", pad=8)
    threshold = float((matrix.min() + matrix.max()) / 2)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.4f}",
                ha="center",
                va="center",
                color=WHITE if matrix[row, column] > threshold else TEXT,
                fontsize=8.0,
                weight="bold",
            )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025)
    colorbar.set_label("Pooled PR AUC")
    fig.tight_layout()
    return save_figure(fig, output / "fig05_modality_ablation_heatmap"), {
        "row_order": CONDITION_ORDER,
        "column_order": HORIZON_ORDER,
        "pr_auc": matrix.tolist(),
        "counts": counts,
    }


def participant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required_models = ["behavior", "logreg_behavior", "total_dwell", "logreg_dwell_propensity"]
    subset = predictions[
        (predictions["horizon"] == "first1")
        & predictions["model"].isin(required_models)
    ]
    for (model, subject), group in subset.groupby(["model", "subject"], sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        if np.unique(y_true).size != 2:
            raise ValueError(f"Participant {subject}/{model} does not contain both classes")
        if not group["y_pred"].notna().all():
            raise ValueError(f"Saved outer predictions are missing for MCC: {subject}/{model}")
        rows.append(
            {
                "model": model,
                "subject": subject,
                "pr_auc": float(average_precision_score(y_true, group["y_score"].to_numpy(dtype=float))),
                "mcc": float(matthews_corrcoef(y_true, group["y_pred"].to_numpy(dtype=int))),
            }
        )
    frame = pd.DataFrame(rows)
    for model in required_models:
        count = frame.loc[frame["model"] == model, "subject"].nunique()
        if count != 42:
            raise ValueError(f"Expected 42 participant scores for {model}, found {count}")
    return frame


def holm_p_values(project: Path) -> dict[str, dict[str, float]]:
    path = project / "outputs/statistics/neuroclick_primary_v1/primary_pairwise_tests.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    output: dict[str, dict[str, float]] = {}
    for row in frame.to_dict(orient="records"):
        metric = normalized_name(row.get("metric", ""))
        comparator = normalized_name(row.get("comparator", ""))
        if metric and comparator and "p_holm" in row:
            output.setdefault(metric, {})[comparator] = float(row["p_holm"])
    return output


def add_pair_annotations(ax: plt.Axes, pvalues: dict[str, float], metric: str, ymax: float) -> None:
    label_map = {"logreg_behavior": "Logistic", "total_dwell": "Dwell", "logreg_dwell_propensity": "Dwell+Propensity"}
    text_items = []
    for comparator in ["logreg_behavior", "total_dwell", "logreg_dwell_propensity"]:
        value = pvalues.get(comparator)
        if value is not None:
            text_items.append(f"NeuroClick vs {label_map[comparator]}: Holm p={value:.4f}")
    if text_items:
        ax.text(0.99, 0.985, "\n".join(text_items), transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=TEXT)


def figure_participant_results(
    predictions: pd.DataFrame, project: Path, output: Path
) -> tuple[dict[str, str], dict]:
    metrics = participant_metrics(predictions)
    pvalues = holm_p_values(project)
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.25), dpi=600)
    models = ["behavior", "logreg_behavior", "total_dwell", "logreg_dwell_propensity"]
    x = np.arange(len(models))
    rng = np.random.default_rng(42)
    for ax, metric, title in zip(axes, ["pr_auc", "mcc"], ["Participant PR AUC", "Participant MCC"]):
        pivot = metrics.pivot(index="subject", columns="model", values=metric).loc[:, models]
        for _, row in pivot.iterrows():
            ax.plot(x, row.to_numpy(dtype=float), color=PAIR_BLUE, linewidth=0.55, alpha=0.72, zorder=1)
        for index, model in enumerate(models):
            values = pivot[model].to_numpy(dtype=float)
            jitter = rng.uniform(-0.065, 0.065, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=13,
                color=MODEL_COLORS[model],
                edgecolor=WHITE,
                linewidth=0.35,
                alpha=0.92,
                zorder=3,
            )
            mean = float(values.mean())
            ax.plot([index - 0.15, index + 0.15], [mean, mean], color=TEXT, linewidth=2.0, zorder=4)
            ax.text(index, mean, f" {mean:.4f}", ha="left", va="bottom", fontsize=6.7, weight="bold", color=TEXT)
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, ["NeuroClick", "Logistic", "Total dwell", "Dwell+Propensity"], rotation=14, ha="right")
        ax.set_xlim(-0.35, len(models) - 1 + 0.35)
        ax.set_ylabel(metric.upper().replace("_", " "))
        style_axis(ax)
        add_pair_annotations(ax, pvalues.get(metric, {}), metric, float(pivot.max().max()))
    fig.tight_layout(w_pad=1.25)
    summary = []
    for (model, metric), group in metrics.melt(id_vars=["model", "subject"], value_vars=["pr_auc", "mcc"], var_name="metric", value_name="value").groupby(["model", "metric"]):
        summary.append(
            {
                "model": model,
                "metric": metric,
                "n_subjects": int(group["subject"].nunique()),
                "mean": float(group["value"].mean()),
                "std": float(group["value"].std(ddof=1)),
            }
        )
    return save_figure(fig, output / "fig06_participant_paired_results"), {
        "summary": summary,
        "holm_p_values": pvalues,
    }


def validate_predictions(predictions: pd.DataFrame) -> dict:
    validation: dict[str, object] = {
        "available_models": sorted(predictions["model"].unique().tolist()),
        "available_horizons": sorted(predictions["horizon"].unique().tolist()),
    }
    primary = require_group(predictions, "first1", "behavior")
    validation["primary_first1"] = {
        "n_subjects": int(primary["subject"].nunique()),
        **pooled_metrics(primary),
    }
    if validation["primary_first1"]["n_subjects"] != 42:
        raise ValueError(validation["primary_first1"])
    if validation["primary_first1"]["n"] != 5715:
        raise ValueError(
            f"Expected 5715 first1 behavior instances, found {validation['primary_first1']['n']}"
        )
    if validation["primary_first1"]["n_buy"] != 746:
        raise ValueError(
            f"Expected 746 first1 Buy labels, found {validation['primary_first1']['n_buy']}"
        )

    group_checks = {}
    for model in ["behavior", "logreg_behavior", "total_dwell"]:
        group = require_group(predictions, "first1", model)
        group_checks[model] = pooled_metrics(group)
    validation["first1_model_checks"] = group_checks

    duplicate_keys = ["source", "horizon", "model", "identifier"]
    duplicate_count = int(predictions.duplicated(duplicate_keys).sum())
    validation["duplicate_rows_within_source"] = duplicate_count
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate prediction rows")
    return validation


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def make_zip(output: Path) -> Path:
    zip_path = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"{output.name}/{path.name}")
    return zip_path


def main() -> None:
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    if not (project / "outputs").exists():
        raise FileNotFoundError(
            f"{project} does not look like ICAIN2026_NeuroClick_Hazard: outputs/ is missing"
        )
    output = (args.output or project / "outputs/manuscript_figures_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    prediction_paths, analysis_manifest = discover_prediction_paths(project)
    predictions = read_predictions(prediction_paths)
    cache = find_feature_cache(project, args.feature_cache)

    validation = validate_predictions(predictions)
    print("\nComputed first1 checks")
    print(json.dumps(validation["first1_model_checks"], indent=2))

    architecture = inspect_architecture(project, cache)
    figures: dict[str, dict[str, str]] = {}
    figure_data: dict[str, object] = {}

    figures["fig01"], figure_data["fig01"] = figure_dataset_profile(cache, output)
    figures["fig02"], figure_data["fig02"] = figure_architecture(architecture, output)
    figures["fig03"], figure_data["fig03"] = figure_pr_roc(predictions, output)
    figures["fig04"], figure_data["fig04"] = figure_horizons(predictions, output)
    figures["fig05"], figure_data["fig05"] = figure_ablation(predictions, output)
    figures["fig06"], figure_data["fig06"] = figure_participant_results(predictions, project, output)

    validation["figures"] = figures
    validation["figure_data"] = figure_data
    write_json(output / "figure_validation.json", validation)

    source_files = list(prediction_paths) + [cache, locate_model_script(project)]
    if analysis_manifest is not None:
        source_files.append(analysis_manifest)
    for name in [
        "per_subject_metrics.csv",
        "primary_pairwise_tests.csv",
        "first1_model_intervals.csv",
    ]:
        path = project / "outputs/statistics/neuroclick_primary_v1" / name
        if path.exists():
            source_files.append(path.resolve())
    source_files = list(dict.fromkeys(source_files))
    source_manifest = {
        "project_root": str(project),
        "generated_without_retraining": True,
        "sources": [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
    }
    write_json(output / "figure_source_manifest.json", source_manifest)
    zip_path = make_zip(output)

    print(f"\nGenerated six publication figures in: {output}")
    print(f"Created archive: {zip_path}")
    print("No model was trained or refit.")


if __name__ == "__main__":
    main()
