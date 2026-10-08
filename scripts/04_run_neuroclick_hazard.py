#!/usr/bin/env python3
"""Leakage-aware cumulative-hazard modeling for NeuMa purchase prediction."""



from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from torch.utils.data import DataLoader, Dataset


CONDITIONS = (
    "behavior",
    "behavior_et",
    "behavior_eeg",
    "behavior_eeg_et",
    "behavior_no_propensity",
)
HORIZONS = ("first1", "first2", "first3", "full")
REQUIRED_KEYS = {
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
    "page",
    "roi",
    "product_id",
    "source_cache_sha256",
}


@dataclass(frozen=True)
class RunConfig:
    feature_cache: str
    output: str
    heldout_subjects: tuple[str, ...]
    conditions: tuple[str, ...]
    seed: int = 42
    epochs: int = 50
    min_epochs: int = 10
    patience: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    modality_dim: int = 64
    dropout: float = 0.20
    validation_fraction: float = 0.20
    product_smoothing: float = 10.0
    gradient_clip: float = 5.0
    max_pos_weight: float = 10.0
    workers: int = 0
    device: str = "cuda"
    deterministic: bool = True


@dataclass
class Normalizer:
    behavior_mean: np.ndarray
    behavior_std: np.ndarray
    eeg_mean: np.ndarray
    eeg_std: np.ndarray
    et_mean: np.ndarray
    et_std: np.ndarray


@dataclass
class PredictionBundle:
    bag_indices: np.ndarray
    labels: np.ndarray
    risks: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-aware NeuroClick cumulative-hazard LOSOCV."
    )
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--heldout-subjects", nargs="+", default=None)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--modality-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--product-smoothing", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Allow nondeterministic backend algorithms.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


class FeatureStore:
    """Validated in-memory representation of the causal visit-feature cache."""

    def __init__(self, path: Path):
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(REQUIRED_KEYS - set(data.files))
            if missing:
                raise ValueError(f"Feature cache is missing keys: {missing}")
            for key in REQUIRED_KEYS:
                setattr(self, key, np.asarray(data[key]).copy())

        self.path = path
        self.eeg_visit = np.asarray(self.eeg_visit, dtype=np.float32)
        self.et_visit = np.asarray(self.et_visit, dtype=np.float32)
        self.visit_start = np.asarray(self.visit_start, dtype=np.float64)
        self.visit_end = np.asarray(self.visit_end, dtype=np.float64)
        self.visit_dwell_s = np.asarray(self.visit_dwell_s, dtype=np.float32)
        self.visit_eeg_valid_fraction = np.asarray(
            self.visit_eeg_valid_fraction, dtype=np.float32
        )
        self.visit_et_valid_fraction = np.asarray(
            self.visit_et_valid_fraction, dtype=np.float32
        )
        self.bag_offsets = np.asarray(self.bag_offsets, dtype=np.int64)
        self.subject = np.asarray(self.subject).astype(str)
        self.label = np.asarray(self.label, dtype=np.uint8)
        self.page = np.asarray(self.page)
        self.roi = np.asarray(self.roi)
        self.product_id = np.asarray(self.product_id)
        self.source_cache_sha256 = str(np.asarray(self.source_cache_sha256).item())

        self.n_bags = int(self.label.size)
        self.n_visits = int(self.eeg_visit.shape[0])
        self.lengths = np.diff(self.bag_offsets).astype(np.int64)
        self.visit_to_bag = np.repeat(
            np.arange(self.n_bags, dtype=np.int64), self.lengths
        )
        self.behavior_visit = self._build_behavior_features()
        # Modality-validity fractions accompany only their own modality.  This
        # prevents the behavior-only ablation from receiving EEG/ET QC cues.
        self.eeg_model_visit = np.column_stack(
            [self.eeg_visit, self.visit_eeg_valid_fraction]
        ).astype(np.float32)
        self.et_model_visit = np.column_stack(
            [self.et_visit, self.visit_et_valid_fraction]
        ).astype(np.float32)
        self._validate()

    def _build_behavior_features(self) -> np.ndarray:
        output = np.zeros((self.n_visits, 5), dtype=np.float32)
        for bag_index in range(self.n_bags):
            start = int(self.bag_offsets[bag_index])
            end = int(self.bag_offsets[bag_index + 1])
            dwell = np.clip(
                self.visit_dwell_s[start:end].astype(np.float64), 0.0, None
            )
            visit_start = self.visit_start[start:end]
            visit_end = self.visit_end[start:end]
            rank = np.arange(1, end - start + 1, dtype=np.float64)
            elapsed = np.clip(visit_end - visit_start[0], 0.0, None)
            gaps = np.zeros(end - start, dtype=np.float64)
            if end - start > 1:
                gaps[1:] = np.clip(visit_start[1:] - visit_end[:-1], 0.0, None)
            output[start:end, 0] = np.log1p(dwell)
            output[start:end, 1] = np.log1p(np.cumsum(dwell))
            output[start:end, 2] = np.log1p(elapsed)
            output[start:end, 3] = np.log1p(gaps)
            output[start:end, 4] = np.log1p(rank)
        return output

    def _validate(self) -> None:
        if self.bag_offsets.shape != (self.n_bags + 1,):
            raise ValueError("bag_offsets must contain n_bags + 1 values")
        if self.bag_offsets[0] != 0 or self.bag_offsets[-1] != self.n_visits:
            raise ValueError("bag_offsets do not cover the visit arrays")
        if np.any(self.lengths <= 0):
            raise ValueError("Every product bag must have at least one visit")
        if self.et_visit.shape[0] != self.n_visits:
            raise ValueError("EEG and ET visit counts differ")
        for name in (
            "visit_start",
            "visit_end",
            "visit_dwell_s",
            "visit_eeg_valid_fraction",
            "visit_et_valid_fraction",
        ):
            if np.asarray(getattr(self, name)).shape != (self.n_visits,):
                raise ValueError(f"Invalid {name} shape")
        for name in ("subject", "page", "roi", "product_id"):
            if np.asarray(getattr(self, name)).shape != (self.n_bags,):
                raise ValueError(f"Invalid {name} shape")
        if not set(np.unique(self.label)).issubset({0, 1}):
            raise ValueError("Labels must be binary")
        if not (
            np.isfinite(self.eeg_visit).all()
            and np.isfinite(self.et_visit).all()
            and np.isfinite(self.eeg_model_visit).all()
            and np.isfinite(self.et_model_visit).all()
            and np.isfinite(self.behavior_visit).all()
        ):
            raise ValueError("Feature cache contains non-finite model inputs")
        for bag_index in range(self.n_bags):
            start = int(self.bag_offsets[bag_index])
            end = int(self.bag_offsets[bag_index + 1])
            if np.any(np.diff(self.visit_end[start:end]) < -1e-9):
                raise ValueError(f"Bag {bag_index} has non-chronological visits")

    @property
    def subjects(self) -> list[str]:
        return sorted(np.unique(self.subject).tolist())

    def visit_indices_for_bags(self, bag_indices: np.ndarray) -> np.ndarray:
        selected = np.zeros(self.n_bags, dtype=bool)
        selected[np.asarray(bag_indices, dtype=np.int64)] = True
        return np.flatnonzero(selected[self.visit_to_bag])


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_normalizer(store: FeatureStore, bag_indices: np.ndarray) -> Normalizer:
    visit_indices = store.visit_indices_for_bags(bag_indices)
    if visit_indices.size == 0:
        raise ValueError("Cannot fit normalizer without training visits")
    behavior_mean, behavior_std = mean_std(store.behavior_visit[visit_indices])
    eeg_mean, eeg_std = mean_std(store.eeg_model_visit[visit_indices])
    et_mean, et_std = mean_std(store.et_model_visit[visit_indices])
    return Normalizer(
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        eeg_mean=eeg_mean,
        eeg_std=eeg_std,
        et_mean=et_mean,
        et_std=et_std,
    )


def fit_product_propensity(
    store: FeatureStore,
    bag_indices: np.ndarray,
    smoothing: float,
) -> tuple[dict[int, float], float]:
    bag_indices = np.asarray(bag_indices, dtype=np.int64)
    labels = store.label[bag_indices].astype(np.float64)
    global_rate = float(labels.mean())
    frame = pd.DataFrame(
        {
            "product": store.product_id[bag_indices].astype(int),
            "label": labels,
        }
    )
    grouped = frame.groupby("product")["label"].agg(["sum", "count"])
    propensity = {
        int(product): float(
            (row["sum"] + smoothing * global_rate)
            / (row["count"] + smoothing)
        )
        for product, row in grouped.iterrows()
    }
    return propensity, global_rate


def propensity_logits_for_all(
    store: FeatureStore,
    propensity: dict[int, float],
    fallback: float,
) -> np.ndarray:
    probabilities = np.asarray(
        [propensity.get(int(product), fallback) for product in store.product_id],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def build_propensity_logits(
    store: FeatureStore,
    training_scope_indices: np.ndarray,
    smoothing: float,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Full-store propensity logits with leakage-free values for training bags.

    Bags OUTSIDE training_scope_indices (validation/test) get the standard
    propensity table fit on the whole training_scope_indices - this was
    already leakage-free, since those bags' own labels never entered the
    table. Bags INSIDE training_scope_indices instead get K-fold
    cross-fitted values: each bag's propensity is computed only from the
    OTHER folds of the training partition (grouped by subject), so no
    training bag's own label ever contributes to its own propensity
    feature. This removes the in-sample target-encoding leak while leaving
    the existing, already-safe validation/test propensity path unchanged.
    """
    training_scope_indices = np.asarray(training_scope_indices, dtype=np.int64)
    full_propensity, full_fallback = fit_product_propensity(
        store, training_scope_indices, smoothing
    )
    logits = propensity_logits_for_all(store, full_propensity, full_fallback)

    groups = store.subject[training_scope_indices]
    n_groups = np.unique(groups).size
    folds = min(n_folds, n_groups)
    if folds < 2:
        raise RuntimeError(
            "Need at least 2 participant groups in the training partition "
            "to cross-fit propensity without leakage"
        )
    gkf = GroupKFold(n_splits=folds)
    global_rate = float(store.label[training_scope_indices].mean())
    cross_fitted_probabilities = np.empty(training_scope_indices.size, dtype=np.float64)
    for fold_fit_rel, fold_heldout_rel in gkf.split(
        training_scope_indices, groups=groups
    ):
        fold_fit_idx = training_scope_indices[fold_fit_rel]
        fold_heldout_idx = training_scope_indices[fold_heldout_rel]
        fold_propensity, fold_fallback = fit_product_propensity(
            store, fold_fit_idx, smoothing
        )
        heldout_products = store.product_id[fold_heldout_idx].astype(int)
        probs = np.asarray(
            [fold_propensity.get(int(p), fold_fallback) for p in heldout_products],
            dtype=np.float64,
        )
        cross_fitted_probabilities[fold_heldout_rel] = probs

    cross_fitted_probabilities = np.clip(cross_fitted_probabilities, 1e-4, 1.0 - 1e-4)
    cross_fitted_logits = np.log(
        cross_fitted_probabilities / (1.0 - cross_fitted_probabilities)
    ).astype(np.float32)
    logits[training_scope_indices] = cross_fitted_logits
    return logits


class BagDataset(Dataset):
    def __init__(
        self,
        store: FeatureStore,
        bag_indices: np.ndarray,
        normalizer: Normalizer,
        propensity_logits: np.ndarray,
    ):
        self.store = store
        self.bag_indices = np.asarray(bag_indices, dtype=np.int64)
        self.normalizer = normalizer
        self.propensity_logits = np.asarray(propensity_logits, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.bag_indices.size)

    @staticmethod
    def _standardize(
        values: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        output = (values - mean) / std
        return np.clip(output, -8.0, 8.0).astype(np.float32)

    def __getitem__(self, item: int) -> dict[str, Any]:
        bag_index = int(self.bag_indices[item])
        start = int(self.store.bag_offsets[bag_index])
        end = int(self.store.bag_offsets[bag_index + 1])
        return {
            "bag_index": bag_index,
            "behavior": self._standardize(
                self.store.behavior_visit[start:end],
                self.normalizer.behavior_mean,
                self.normalizer.behavior_std,
            ),
            "eeg": self._standardize(
                self.store.eeg_model_visit[start:end],
                self.normalizer.eeg_mean,
                self.normalizer.eeg_std,
            ),
            "et": self._standardize(
                self.store.et_model_visit[start:end],
                self.normalizer.et_mean,
                self.normalizer.et_std,
            ),
            "propensity_logit": self.propensity_logits[bag_index],
            "label": float(self.store.label[bag_index]),
        }


def collate_bags(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(items)
    lengths = np.asarray([item["behavior"].shape[0] for item in items], dtype=np.int64)
    maximum = int(lengths.max())
    behavior_dim = int(items[0]["behavior"].shape[1])
    eeg_dim = int(items[0]["eeg"].shape[1])
    et_dim = int(items[0]["et"].shape[1])
    behavior = np.zeros((batch_size, maximum, behavior_dim), dtype=np.float32)
    eeg = np.zeros((batch_size, maximum, eeg_dim), dtype=np.float32)
    et = np.zeros((batch_size, maximum, et_dim), dtype=np.float32)
    mask = np.zeros((batch_size, maximum), dtype=bool)
    for row, item in enumerate(items):
        length = int(lengths[row])
        behavior[row, :length] = item["behavior"]
        eeg[row, :length] = item["eeg"]
        et[row, :length] = item["et"]
        mask[row, :length] = True
    return {
        "bag_index": torch.as_tensor(
            [item["bag_index"] for item in items], dtype=torch.long
        ),
        "behavior": torch.from_numpy(behavior),
        "eeg": torch.from_numpy(eeg),
        "et": torch.from_numpy(et),
        "propensity_logit": torch.as_tensor(
            [item["propensity_logit"] for item in items], dtype=torch.float32
        ),
        "label": torch.as_tensor(
            [item["label"] for item in items], dtype=torch.float32
        ),
        "lengths": torch.from_numpy(lengths),
        "mask": torch.from_numpy(mask),
    }


def condition_modalities(condition: str) -> tuple[bool, bool]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return "eeg" in condition, "et" in condition


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class NeuroClickHazard(nn.Module):
    def __init__(
        self,
        behavior_dim: int,
        eeg_dim: int,
        et_dim: int,
        hidden_dim: int,
        modality_dim: int,
        dropout: float,
        condition: str,
        initial_hazard: float,
    ):
        super().__init__()
        self.condition = condition
        self.use_eeg, self.use_et = condition_modalities(condition)
        self.use_propensity = condition != "behavior_no_propensity"
        behavior_input_dim = behavior_dim + 1 if self.use_propensity else behavior_dim
        self.behavior_encoder = Projection(behavior_input_dim, hidden_dim, dropout)
        if self.use_eeg:
            self.eeg_encoder = Projection(eeg_dim, modality_dim, dropout)
            self.eeg_to_hidden = nn.Linear(modality_dim, hidden_dim)
            self.eeg_gate = nn.Linear(hidden_dim + modality_dim, hidden_dim)
        if self.use_et:
            self.et_encoder = Projection(et_dim, modality_dim, dropout)
            self.et_to_hidden = nn.Linear(modality_dim, hidden_dim)
            self.et_gate = nn.Linear(hidden_dim + modality_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.hazard_head = nn.Linear(hidden_dim, 1)
        initial_hazard = float(np.clip(initial_hazard, 1e-4, 0.25))
        initial_bias = math.log(initial_hazard / (1.0 - initial_hazard))
        nn.init.constant_(self.hazard_head.bias, initial_bias)

    def forward(
        self,
        behavior: torch.Tensor,
        eeg: torch.Tensor,
        et: torch.Tensor,
        propensity_logit: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_steps = behavior.shape[1]
        if self.use_propensity:
            propensity = propensity_logit[:, None, None].expand(-1, time_steps, 1)
            base = self.behavior_encoder(torch.cat([behavior, propensity], dim=-1))
        else:
            base = self.behavior_encoder(behavior)
        fused = base
        if self.use_eeg:
            eeg_embedding = self.eeg_encoder(eeg)
            eeg_gate = torch.sigmoid(
                self.eeg_gate(torch.cat([base, eeg_embedding], dim=-1))
            )
            fused = fused + eeg_gate * self.eeg_to_hidden(eeg_embedding)
        if self.use_et:
            et_embedding = self.et_encoder(et)
            et_gate = torch.sigmoid(
                self.et_gate(torch.cat([base, et_embedding], dim=-1))
            )
            fused = fused + et_gate * self.et_to_hidden(et_embedding)
        fused = self.fusion_norm(fused)
        packed = pack_padded_sequence(
            fused,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=time_steps,
        )
        hazard_logits = self.hazard_head(self.output_dropout(output)).squeeze(-1)
        log_survival = torch.cumsum(F.logsigmoid(-hazard_logits), dim=1)
        cumulative_risk = -torch.expm1(log_survival)
        cumulative_risk = cumulative_risk.clamp(1e-7, 1.0 - 1e-7)
        return hazard_logits, log_survival, cumulative_risk


def prefix_hazard_loss(
    log_survival: torch.Tensor,
    cumulative_risk: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    labels_2d = labels[:, None]
    positive = -torch.log(cumulative_risk)
    negative = -log_survival
    per_prefix = (
        labels_2d * float(positive_weight) * positive
        + (1.0 - labels_2d) * negative
    )
    mask_float = mask.to(per_prefix.dtype)
    per_bag = (per_prefix * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
    return per_bag.mean()


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        if key not in {"lengths", "bag_index"}
        else value
        for key, value in batch.items()
    }


def make_loader(
    store: FeatureStore,
    bag_indices: np.ndarray,
    normalizer: Normalizer,
    propensity_logits: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = BagDataset(store, bag_indices, normalizer, propensity_logits)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_bags,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def predict_loader(
    model: NeuroClickHazard,
    loader: DataLoader,
    device: torch.device,
) -> PredictionBundle:
    model.eval()
    bag_indices: list[int] = []
    labels: list[int] = []
    risks: list[np.ndarray] = []
    for batch in loader:
        lengths = batch["lengths"].numpy()
        device_batch = move_batch(batch, device)
        _, _, cumulative = model(
            device_batch["behavior"],
            device_batch["eeg"],
            device_batch["et"],
            device_batch["propensity_logit"],
            batch["lengths"],
        )
        cumulative_np = cumulative.detach().cpu().numpy()
        for row, length in enumerate(lengths):
            bag_indices.append(int(batch["bag_index"][row]))
            labels.append(int(batch["label"][row]))
            risks.append(cumulative_np[row, : int(length)].astype(np.float64))
    order = np.argsort(np.asarray(bag_indices))
    return PredictionBundle(
        bag_indices=np.asarray(bag_indices, dtype=np.int64)[order],
        labels=np.asarray(labels, dtype=np.uint8)[order],
        risks=[risks[index] for index in order],
    )


def horizon_position(length: int, horizon: str) -> int | None:
    if horizon == "first1":
        return 0 if length >= 1 else None
    if horizon == "first2":
        return 1 if length >= 2 else None
    if horizon == "first3":
        return 2 if length >= 3 else None
    if horizon == "full":
        return length - 1 if length >= 1 else None
    raise ValueError(horizon)


def horizon_arrays(
    predictions: PredictionBundle, horizon: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bag_indices: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    for bag_index, label, risk in zip(
        predictions.bag_indices, predictions.labels, predictions.risks
    ):
        position = horizon_position(len(risk), horizon)
        if position is None:
            continue
        bag_indices.append(int(bag_index))
        labels.append(int(label))
        probabilities.append(float(risk[position]))
    return (
        np.asarray(bag_indices, dtype=np.int64),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(probabilities, dtype=np.float64),
    )


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    if labels.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if not selected.any():
            continue
        error += float(selected.mean()) * abs(
            float(labels[selected].mean()) - float(probabilities[selected].mean())
        )
    return float(error)


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5
    quantiles = np.quantile(probabilities, np.linspace(0.01, 0.99, 199))
    candidates = np.unique(
        np.concatenate([np.linspace(0.01, 0.99, 99), quantiles, [0.5]])
    )
    best_threshold = 0.5
    best_key = (-float("inf"), -float("inf"), -float("inf"))
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(np.uint8)
        mcc = float(matthews_corrcoef(labels, predicted))
        balanced = float(balanced_accuracy_score(labels, predicted))
        key = (mcc, balanced, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def metric_record(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype(np.uint8)
    return {
        "n": int(labels.size),
        "n_buy": int(labels.sum()),
        "pr_auc": safe_average_precision(labels, probabilities),
        "roc_auc": safe_roc_auc(labels, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "buy_f1": float(f1_score(labels, predicted, zero_division=0)),
        "buy_precision": float(precision_score(labels, predicted, zero_division=0)),
        "buy_recall": float(recall_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "ece": expected_calibration_error(labels, probabilities),
        "threshold": float(threshold),
    }


def choose_validation_split(
    store: FeatureStore,
    outer_train_indices: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = store.subject[outer_train_indices]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    fit_relative, validation_relative = next(
        splitter.split(outer_train_indices, store.label[outer_train_indices], groups)
    )
    fit_indices = outer_train_indices[fit_relative]
    validation_indices = outer_train_indices[validation_relative]
    if np.unique(store.label[fit_indices]).size < 2:
        raise RuntimeError("Inner fit partition contains only one class")
    if np.unique(store.label[validation_indices]).size < 2:
        raise RuntimeError("Inner validation partition contains only one class")
    validation_subjects = sorted(np.unique(store.subject[validation_indices]).tolist())
    return fit_indices, validation_indices, validation_subjects


def initial_hazard_for_indices(store: FeatureStore, bag_indices: np.ndarray) -> float:
    prevalence = float(store.label[bag_indices].mean())
    mean_visits = float(store.lengths[bag_indices].mean())
    return float(np.clip(prevalence / max(mean_visits, 1.0), 0.005, 0.20))


def positive_weight_for_indices(
    store: FeatureStore, bag_indices: np.ndarray, maximum: float
) -> float:
    positives = int(store.label[bag_indices].sum())
    negatives = int(bag_indices.size - positives)
    if positives == 0:
        raise ValueError("Training partition contains no Buy products")
    return float(min(negatives / positives, maximum))


def build_model(
    store: FeatureStore,
    config: RunConfig,
    condition: str,
    bag_indices: np.ndarray,
    device: torch.device,
) -> NeuroClickHazard:
    model = NeuroClickHazard(
        behavior_dim=store.behavior_visit.shape[1],
        eeg_dim=store.eeg_model_visit.shape[1],
        et_dim=store.et_model_visit.shape[1],
        hidden_dim=config.hidden_dim,
        modality_dim=config.modality_dim,
        dropout=config.dropout,
        condition=condition,
        initial_hazard=initial_hazard_for_indices(store, bag_indices),
    )
    return model.to(device)


def train_epoch(
    model: NeuroClickHazard,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    positive_weight: float,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_bags = 0
    for batch in loader:
        device_batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        _, log_survival, cumulative = model(
            device_batch["behavior"],
            device_batch["eeg"],
            device_batch["et"],
            device_batch["propensity_logit"],
            batch["lengths"],
        )
        loss = prefix_hazard_loss(
            log_survival,
            cumulative,
            device_batch["label"],
            device_batch["mask"],
            positive_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = int(batch["label"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_bags += batch_size
    return total_loss / max(total_bags, 1)


def fit_with_early_stopping(
    store: FeatureStore,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: RunConfig,
    condition: str,
    device: torch.device,
    seed: int,
) -> tuple[NeuroClickHazard, int, PredictionBundle, float]:
    seed_everything(seed, config.deterministic)
    normalizer = fit_normalizer(store, fit_indices)
    propensity_logits = build_propensity_logits(
        store, fit_indices, config.product_smoothing, seed=seed
    )
    pin_memory = device.type == "cuda"
    fit_loader = make_loader(
        store,
        fit_indices,
        normalizer,
        propensity_logits,
        config.batch_size,
        True,
        config.workers,
        seed,
        pin_memory,
    )
    validation_loader = make_loader(
        store,
        validation_indices,
        normalizer,
        propensity_logits,
        config.batch_size,
        False,
        config.workers,
        seed,
        pin_memory,
    )
    model = build_model(store, config, condition, fit_indices, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_weight = positive_weight_for_indices(
        store, fit_indices, config.max_pos_weight
    )
    best_score = -float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    best_predictions: PredictionBundle | None = None
    stale_epochs = 0
    final_loss = float("nan")

    for epoch in range(1, config.epochs + 1):
        final_loss = train_epoch(
            model,
            fit_loader,
            optimizer,
            device,
            positive_weight,
            config.gradient_clip,
        )
        validation_predictions = predict_loader(model, validation_loader, device)
        _, labels, probabilities = horizon_arrays(
            validation_predictions, "first1"
        )
        score = safe_average_precision(labels, probabilities)
        if not np.isfinite(score):
            raise RuntimeError("Validation PR-AUC is not finite")
        improved = score > best_score + 1e-5
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_predictions = validation_predictions
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            marker = "*" if improved else " "
            print(
                f"    epoch={epoch:03d} loss={final_loss:.5f} "
                f"val_first1_pr={score:.4f}{marker}",
                flush=True,
            )
        if epoch >= config.min_epochs and stale_epochs >= config.patience:
            break

    if best_state is None or best_predictions is None:
        raise RuntimeError("Early stopping did not retain a model")
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch, best_predictions, final_loss


def train_fixed_epochs(
    store: FeatureStore,
    training_indices: np.ndarray,
    config: RunConfig,
    condition: str,
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[NeuroClickHazard, Normalizer, np.ndarray, float]:
    seed_everything(seed, config.deterministic)
    normalizer = fit_normalizer(store, training_indices)
    propensity_logits = build_propensity_logits(
        store, training_indices, config.product_smoothing, seed=seed
    )
    loader = make_loader(
        store,
        training_indices,
        normalizer,
        propensity_logits,
        config.batch_size,
        True,
        config.workers,
        seed,
        device.type == "cuda",
    )
    model = build_model(store, config, condition, training_indices, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    positive_weight = positive_weight_for_indices(
        store, training_indices, config.max_pos_weight
    )
    final_loss = float("nan")
    for _ in range(int(epochs)):
        final_loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            positive_weight,
            config.gradient_clip,
        )
    return model, normalizer, propensity_logits, final_loss


def thresholds_from_validation(
    predictions: PredictionBundle,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for horizon in HORIZONS:
        _, labels, probabilities = horizon_arrays(predictions, horizon)
        thresholds[horizon] = select_mcc_threshold(labels, probabilities)
    return thresholds


def prediction_rows(
    store: FeatureStore,
    predictions: PredictionBundle,
    heldout_subject: str,
    condition: str,
    thresholds: dict[str, float],
    best_epoch: int,
    validation_subjects: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        bag_indices, labels, probabilities = horizon_arrays(predictions, horizon)
        threshold = thresholds[horizon]
        for bag_index, label, probability in zip(
            bag_indices, labels, probabilities
        ):
            rows.append(
                {
                    "heldout_subject": heldout_subject,
                    "condition": condition,
                    "horizon": horizon,
                    "bag_index": int(bag_index),
                    "label": int(label),
                    "probability": float(probability),
                    "threshold": float(threshold),
                    "prediction": int(probability >= threshold),
                    "page_audit": int(store.page[bag_index]),
                    "roi_audit": int(store.roi[bag_index]),
                    "product_id_audit": int(store.product_id[bag_index]),
                    "n_visits": int(store.lengths[bag_index]),
                    "best_epoch": int(best_epoch),
                    "validation_subjects": "|".join(validation_subjects),
                }
            )
    return rows


def fold_metric_rows(
    prediction_frame: pd.DataFrame,
    heldout_subject: str,
    condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = prediction_frame.loc[
        (prediction_frame["heldout_subject"] == heldout_subject)
        & (prediction_frame["condition"] == condition)
    ]
    for horizon in HORIZONS:
        frame = selected.loc[selected["horizon"] == horizon]
        if frame.empty:
            continue
        metrics = metric_record(
            frame["label"].to_numpy(dtype=np.uint8),
            frame["probability"].to_numpy(dtype=np.float64),
            float(frame["threshold"].iloc[0]),
        )
        rows.append(
            {
                "heldout_subject": heldout_subject,
                "condition": condition,
                "horizon": horizon,
                **metrics,
                "best_epoch": int(frame["best_epoch"].iloc[0]),
                "validation_subjects": str(frame["validation_subjects"].iloc[0]),
            }
        )
    return rows


def summarize_predictions(
    prediction_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition not in set(prediction_frame["condition"]):
            continue
        for horizon in HORIZONS:
            selected = prediction_frame.loc[
                (prediction_frame["condition"] == condition)
                & (prediction_frame["horizon"] == horizon)
            ]
            if selected.empty:
                continue
            labels = selected["label"].to_numpy(dtype=np.uint8)
            probabilities = selected["probability"].to_numpy(dtype=np.float64)
            predicted = selected["prediction"].to_numpy(dtype=np.uint8)
            prevalence = float(labels.mean())
            pooled = {
                "n": int(labels.size),
                "n_buy": int(labels.sum()),
                "pr_auc": safe_average_precision(labels, probabilities),
                "roc_auc": safe_roc_auc(labels, probabilities),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predicted)
                ),
                "buy_f1": float(f1_score(labels, predicted, zero_division=0)),
                "buy_recall": float(
                    recall_score(labels, predicted, zero_division=0)
                ),
                "mcc": float(matthews_corrcoef(labels, predicted)),
                "ece": expected_calibration_error(labels, probabilities),
            }
            pooled["pr_lift"] = (
                float(pooled["pr_auc"]) / prevalence if prevalence > 0 else float("nan")
            )
            fold_selected = fold_frame.loc[
                (fold_frame["condition"] == condition)
                & (fold_frame["horizon"] == horizon)
            ]
            row: dict[str, Any] = {
                "condition": condition,
                "horizon": horizon,
                **pooled,
            }
            for metric in (
                "pr_auc",
                "roc_auc",
                "balanced_accuracy",
                "mcc",
                "ece",
            ):
                values = fold_selected[metric].to_numpy(dtype=np.float64)
                row[f"mean_subject_{metric}"] = float(np.nanmean(values))
                row[f"std_subject_{metric}"] = float(np.nanstd(values, ddof=0))
            rows.append(row)
    return pd.DataFrame(rows)


def load_partial(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_progress(
    output: Path,
    prediction_rows_list: list[dict[str, Any]],
    fold_rows_list: list[dict[str, Any]],
    completed: set[tuple[str, str]],
) -> None:
    predictions = pd.DataFrame(prediction_rows_list)
    folds = pd.DataFrame(fold_rows_list)
    atomic_write_csv(predictions, output / "partial_predictions.csv")
    atomic_write_csv(folds, output / "partial_fold_metrics.csv")
    progress = {
        "completed": [
            {"heldout_subject": subject, "condition": condition}
            for subject, condition in sorted(completed)
        ],
        "updated_unix": time.time(),
    }
    atomic_write_text(output / "progress.json", json.dumps(progress, indent=2))


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(name)


def validate_args(args: argparse.Namespace) -> None:
    if args.feature_cache is None or args.output is None:
        raise ValueError("--feature-cache and --output are required")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not (1 <= args.min_epochs <= args.epochs):
        raise ValueError("--min-epochs must be in [1, epochs]")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not (0.05 <= args.validation_fraction <= 0.40):
        raise ValueError("--validation-fraction must be between 0.05 and 0.40")
    if args.hidden_dim < 8 or args.modality_dim < 8:
        raise ValueError("Model dimensions are too small")
    if not (0.0 <= args.dropout < 0.8):
        raise ValueError("--dropout must be in [0, 0.8)")


def run_self_test() -> None:
    seed_everything(7, True)

    # Cumulative risk must be monotonic.
    logits = torch.tensor([[0.2, -1.0, 0.5, -0.4]], dtype=torch.float32)
    log_survival = torch.cumsum(F.logsigmoid(-logits), dim=1)
    risk = -torch.expm1(log_survival)
    assert torch.all(risk[:, 1:] >= risk[:, :-1] - 1e-7)

    # Future visits must not change earlier prefix predictions.
    model = NeuroClickHazard(
        behavior_dim=5,
        eeg_dim=5,
        et_dim=4,
        hidden_dim=16,
        modality_dim=12,
        dropout=0.0,
        condition="behavior_eeg_et",
        initial_hazard=0.05,
    )
    model.eval()
    generator = torch.Generator().manual_seed(11)
    behavior = torch.randn(2, 5, 5, generator=generator)
    eeg = torch.randn(2, 5, 5, generator=generator)
    et = torch.randn(2, 5, 4, generator=generator)
    propensity = torch.tensor([-1.5, -2.0])
    lengths = torch.tensor([5, 5])
    with torch.no_grad():
        _, _, original = model(behavior, eeg, et, propensity, lengths)
        changed_behavior = behavior.clone()
        changed_eeg = eeg.clone()
        changed_et = et.clone()
        changed_behavior[:, 2:] += 100.0
        changed_eeg[:, 2:] -= 75.0
        changed_et[:, 2:] += 50.0
        _, _, changed = model(
            changed_behavior, changed_eeg, changed_et, propensity, lengths
        )
    assert torch.allclose(original[:, :2], changed[:, :2], atol=1e-6, rtol=1e-6)

    # Bag-normalized loss must remain finite and support backward propagation.
    labels = torch.tensor([1.0, 0.0])
    mask = torch.ones((2, 5), dtype=torch.bool)
    _, log_survival, cumulative = model(behavior, eeg, et, propensity, lengths)
    loss = prefix_hazard_loss(log_survival, cumulative, labels, mask, 3.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    # A training-only propensity map cannot react to held-out labels.
    class TinyStore:
        label = np.array([1, 0, 1, 0], dtype=np.uint8)
        product_id = np.array([10, 10, 20, 20], dtype=np.uint16)

    tiny = TinyStore()
    mapping_a, fallback_a = fit_product_propensity(
        tiny, np.array([0, 1]), smoothing=10.0
    )
    tiny.label[2:] = 1 - tiny.label[2:]
    mapping_b, fallback_b = fit_product_propensity(
        tiny, np.array([0, 1]), smoothing=10.0
    )
    assert mapping_a == mapping_b and fallback_a == fallback_b

    print("SELF-TEST: PASS - monotonicity, prefix causality, loss and propensity")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    validate_args(args)

    feature_cache = args.feature_cache.resolve()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite existing output {output}; use --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(feature_cache)
    available_subjects = store.subjects
    heldout_subjects = args.heldout_subjects or available_subjects
    unknown = sorted(set(heldout_subjects) - set(available_subjects))
    if unknown:
        raise ValueError(f"Unknown held-out subjects: {unknown}")

    config = RunConfig(
        feature_cache=str(feature_cache),
        output=str(output),
        heldout_subjects=tuple(heldout_subjects),
        conditions=tuple(args.conditions),
        seed=args.seed,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        modality_dim=args.modality_dim,
        dropout=args.dropout,
        validation_fraction=args.validation_fraction,
        product_smoothing=args.product_smoothing,
        gradient_clip=args.gradient_clip,
        max_pos_weight=args.max_pos_weight,
        workers=args.workers,
        device=args.device,
        deterministic=not args.non_deterministic,
    )
    device = resolve_device(config.device)
    feature_sha256 = sha256_file(feature_cache)
    script_sha256 = sha256_file(Path(__file__).resolve())
    config_record = json.loads(json.dumps(asdict(config)))
    run_record = {
        "config": config_record,
        "feature_cache_sha256": feature_sha256,
        "source_cache_sha256": store.source_cache_sha256,
        "script_sha256": script_sha256,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "CPU",
        "n_subjects_total": len(available_subjects),
        "n_bags": store.n_bags,
        "n_visits": store.n_visits,
        "n_buy": int(store.label.sum()),
        "identifier_policy": (
            "subject/page/ROI/raw product id are audit metadata only; "
            "product id is used solely for training-partition propensity"
        ),
    }
    run_config_path = output / "run_config.json"
    if args.resume and run_config_path.exists():
        previous = json.loads(run_config_path.read_text())
        if previous.get("config") != run_record["config"]:
            raise RuntimeError("Resume configuration differs from the saved run")
        if previous.get("feature_cache_sha256") != feature_sha256:
            raise RuntimeError("Resume feature-cache hash differs from the saved run")
    else:
        atomic_write_text(run_config_path, json.dumps(run_record, indent=2))

    partial_predictions_path = output / "partial_predictions.csv"
    partial_folds_path = output / "partial_fold_metrics.csv"
    existing_predictions = load_partial(partial_predictions_path) if args.resume else pd.DataFrame()
    existing_folds = load_partial(partial_folds_path) if args.resume else pd.DataFrame()
    completed: set[tuple[str, str]] = set()
    if not existing_predictions.empty and not existing_folds.empty:
        prediction_pairs = set(
            zip(
                existing_predictions["heldout_subject"].astype(str),
                existing_predictions["condition"].astype(str),
            )
        )
        fold_pairs = set(
            zip(
                existing_folds["heldout_subject"].astype(str),
                existing_folds["condition"].astype(str),
            )
        )
        completed = prediction_pairs & fold_pairs
        prediction_keep = [
            (str(subject), str(condition)) in completed
            for subject, condition in zip(
                existing_predictions["heldout_subject"],
                existing_predictions["condition"],
            )
        ]
        fold_keep = [
            (str(subject), str(condition)) in completed
            for subject, condition in zip(
                existing_folds["heldout_subject"],
                existing_folds["condition"],
            )
        ]
        existing_predictions = existing_predictions.loc[prediction_keep].copy()
        existing_folds = existing_folds.loc[fold_keep].copy()
    prediction_rows_list = existing_predictions.to_dict("records")
    fold_rows_list = existing_folds.to_dict("records")

    print("=" * 78, flush=True)
    print("NEUROCLICK CUMULATIVE-HAZARD STRICT LOSOCV", flush=True)
    print("=" * 78, flush=True)
    print(f"Feature cache        : {feature_cache}", flush=True)
    print(f"Source cache SHA256  : {store.source_cache_sha256}", flush=True)
    print(f"Output               : {output}", flush=True)
    print(f"Total subjects       : {len(available_subjects)}", flush=True)
    print(f"Held-out folds       : {', '.join(heldout_subjects)}", flush=True)
    print(f"Conditions           : {', '.join(config.conditions)}", flush=True)
    print(f"Bags / Buy / Visits  : {store.n_bags} / {int(store.label.sum())} / {store.n_visits}", flush=True)
    print(f"Device               : {run_record['device_name']}", flush=True)
    print("Primary metric       : first1 PR-AUC", flush=True)
    print("Outer validation     : strict leave-one-subject-out", flush=True)
    print("Inner validation     : subject-disjoint early stopping/thresholds", flush=True)
    print("Identifiers          : audit-only; train-fold propensity is the sole exception", flush=True)

    all_indices = np.arange(store.n_bags, dtype=np.int64)
    total_tasks = len(heldout_subjects) * len(config.conditions)
    task_number = 0
    for fold_number, heldout_subject in enumerate(heldout_subjects, start=1):
        test_indices = np.flatnonzero(store.subject == heldout_subject)
        outer_train_indices = np.flatnonzero(store.subject != heldout_subject)
        if test_indices.size == 0 or outer_train_indices.size + test_indices.size != all_indices.size:
            raise RuntimeError(f"Invalid outer fold for {heldout_subject}")
        fit_indices, validation_indices, validation_subjects = choose_validation_split(
            store,
            outer_train_indices,
            config.validation_fraction,
            config.seed + fold_number,
        )
        print("\n" + "-" * 78, flush=True)
        print(
            f"FOLD {fold_number:02d}/{len(heldout_subjects):02d} "
            f"held={heldout_subject} train={outer_train_indices.size} "
            f"test={test_indices.size}",
            flush=True,
        )
        print(
            f"Inner fit={fit_indices.size}, validation={validation_indices.size}, "
            f"validation subjects={','.join(validation_subjects)}",
            flush=True,
        )

        for condition_number, condition in enumerate(config.conditions, start=1):
            task_number += 1
            key = (heldout_subject, condition)
            if key in completed:
                print(f"  SKIP completed {heldout_subject}/{condition}", flush=True)
                continue
            task_seed = config.seed + fold_number * 1000 + condition_number * 100
            print(
                f"  TASK {task_number:03d}/{total_tasks:03d} condition={condition}",
                flush=True,
            )
            provisional, best_epoch, validation_predictions, _ = fit_with_early_stopping(
                store,
                fit_indices,
                validation_indices,
                config,
                condition,
                device,
                task_seed,
            )
            thresholds = thresholds_from_validation(validation_predictions)
            del provisional
            if device.type == "cuda":
                torch.cuda.empty_cache()

            final_model, final_normalizer, final_propensity, final_loss = train_fixed_epochs(
                store,
                outer_train_indices,
                config,
                condition,
                device,
                task_seed + 1,
                best_epoch,
            )
            test_loader = make_loader(
                store,
                test_indices,
                final_normalizer,
                final_propensity,
                config.batch_size,
                False,
                config.workers,
                task_seed,
                device.type == "cuda",
            )
            test_predictions = predict_loader(final_model, test_loader, device)
            new_prediction_rows = prediction_rows(
                store,
                test_predictions,
                heldout_subject,
                condition,
                thresholds,
                best_epoch,
                validation_subjects,
            )
            temporary_prediction_frame = pd.DataFrame(new_prediction_rows)
            new_fold_rows = fold_metric_rows(
                temporary_prediction_frame, heldout_subject, condition
            )
            for row in new_fold_rows:
                row["final_training_loss"] = float(final_loss)
            prediction_rows_list.extend(new_prediction_rows)
            fold_rows_list.extend(new_fold_rows)
            completed.add(key)
            save_progress(output, prediction_rows_list, fold_rows_list, completed)
            first1 = next(
                row for row in new_fold_rows if row["horizon"] == "first1"
            )
            print(
                f"  COMPLETE {condition}: best_epoch={best_epoch}, "
                f"first1 PR={first1['pr_auc']:.4f}, "
                f"AUC={first1['roc_auc']:.4f}, MCC={first1['mcc']:.4f}",
                flush=True,
            )
            del final_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    prediction_frame = pd.DataFrame(prediction_rows_list)
    fold_frame = pd.DataFrame(fold_rows_list)
    summary_frame = summarize_predictions(prediction_frame, fold_frame)
    atomic_write_csv(prediction_frame, output / "losocv_predictions.csv")
    atomic_write_csv(fold_frame, output / "fold_metrics.csv")
    atomic_write_csv(summary_frame, output / "summary_metrics.csv")
    decision = (
        "BENCHMARK_GO" if set(heldout_subjects) == set(available_subjects) else "PILOT_GO"
    )
    decision_lines = [
        f"DECISION: {decision}",
        f"heldout_folds={len(heldout_subjects)}",
        f"conditions={','.join(config.conditions)}",
        f"feature_cache_sha256={feature_sha256}",
        f"source_cache_sha256={store.source_cache_sha256}",
        "primary_metric=first1_pr_auc",
        "full_history=retrospective_upper_bound",
    ]
    atomic_write_text(
        output / "BENCHMARK_DECISION.txt", "\n".join(decision_lines) + "\n"
    )

    display_columns = [
        "condition",
        "horizon",
        "n",
        "n_buy",
        "pr_auc",
        "pr_lift",
        "roc_auc",
        "balanced_accuracy",
        "mcc",
        "mean_subject_pr_auc",
        "mean_subject_roc_auc",
        "mean_subject_mcc",
    ]
    print("\n" + "=" * 78, flush=True)
    print(f"DECISION: {decision}", flush=True)
    print(
        summary_frame[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
        flush=True,
    )
    print(f"Saved outputs to: {output}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed fold-condition checkpoints were preserved.")
        raise
