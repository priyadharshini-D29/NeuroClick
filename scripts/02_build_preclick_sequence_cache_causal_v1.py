#!/usr/bin/env python3
"""Build the leakage-safe NeuMa pre-click EEG/ET visit-sequence cache.

This is stage 2 of the ICAIN NeuroClick-Hazard pipeline.  It consumes the
frozen product-level audit manifest produced by ``01_audit_buy_nobuy_events``
and creates one chronological, variable-length bag per considered
subject-product instance.

Each gaze visit is represented by:

* a fixed one-second EEG window resampled at 300 Hz and harmonized to the
  verified 19-electrode production montage;
* a fixed one-second eye-tracking window resampled at 120 Hz with gaze
  coordinates expressed relative to the active product ROI; and
* visit timing/dwell metadata for a later discrete-time hazard or MIL model.

For Buy products, every sample is earlier than the frozen predecision cutoff
(the first click minus the audit margin).  For NoBuy products, evidence is
restricted to periods during which the product page was active.  This script
does not train a model and never uses subject/product identifiers as features.

Heavy outputs should be written below /mnt through the project's cache link.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


SUBJECT_COLUMNS = {
    "subject",
    "page",
    "roi",
    "product_id",
    "label_buy",
    "considered",
    "n_gaze_visits",
    "first_click_time",
    "predecision_cutoff_time",
}


@dataclass(frozen=True)
class CacheConfig:
    root: str
    audit_script: str
    output: str
    eeg_fs: float = 300.0
    et_fs: float = 120.0
    window_s: float = 1.0
    min_fixation_ms: float = 100.0
    merge_gap_ms: float = 75.0
    pre_click_margin_ms: float = 500.0
    max_visits: int = 0
    interpolation_gap_multiplier: float = 2.5
    roi_context_clip: float = 2.0
    screen_width: int = 1920
    screen_height: int = 1080


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chronological pre-click EEG/ET product bags."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument(
        "--audit-script",
        type=Path,
        default=Path("scripts/01_audit_buy_nobuy_events.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eeg-fs", type=float, default=300.0)
    parser.add_argument("--et-fs", type=float, default=120.0)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--min-fixation-ms", type=float, default=100.0)
    parser.add_argument("--merge-gap-ms", type=float, default=75.0)
    parser.add_argument("--pre-click-margin-ms", type=float, default=500.0)
    parser.add_argument(
        "--max-visits",
        type=int,
        default=0,
        help="Keep the last N visits per product; 0 preserves every visit.",
    )
    parser.add_argument("--interpolation-gap-multiplier", type=float, default=2.5)
    parser.add_argument("--roi-context-clip", type=float, default=2.0)
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--expect-instances", type=int, default=0)
    parser.add_argument("--expect-buys", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_harmonizer(root: Path):
    path = root / "src" / "model" / "data" / "channel_harmonizer.py"
    module = _load_module(path, "neuroclick_production_harmonizer")
    channels = [str(value) for value in module.CANONICAL_CHANNELS]
    if len(channels) != 19:
        raise RuntimeError(
            f"Production montage must have 19 channels; found {len(channels)}"
        )
    return module, channels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_rate(timestamps: np.ndarray) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size < 2:
        return math.nan
    duration = float(timestamps[-1] - timestamps[0])
    return float((timestamps.size - 1) / duration) if duration > 0 else math.nan


def validate_time_series(
    values: np.ndarray,
    timestamps: np.ndarray,
    label: str,
    expected_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"{label}: expected a 2-D array, got {values.shape}")
    if values.shape[0] != timestamps.size:
        raise ValueError(
            f"{label}: sample/timestamp mismatch {values.shape[0]}/{timestamps.size}"
        )
    if expected_channels is not None and values.shape[1] != expected_channels:
        raise ValueError(
            f"{label}: expected {expected_channels} channels, got {values.shape[1]}"
        )
    if timestamps.size < 2 or not np.isfinite(timestamps).all():
        raise ValueError(f"{label}: invalid timestamps")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{label}: timestamps are not strictly increasing")
    return values, timestamps


def resample_preboundary_window(
    values: np.ndarray,
    timestamps: np.ndarray,
    boundary: float,
    lower_bound: float,
    fs: float,
    window_s: float,
    interpolation_gap_multiplier: float,
    source_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a fixed window using only samples strictly before boundary.

    The returned mask is per time point and channel.  Interpolated values are
    accepted only when a real source sample lies within a small sampling-rate
    dependent distance, preventing long missing intervals from being silently
    bridged.  Invalid/padded values are exactly zero.
    """
    if fs <= 0 or window_s <= 0:
        raise ValueError("fs and window_s must be positive")
    n_samples = int(round(fs * window_s))
    if n_samples <= 0:
        raise ValueError("The requested window contains no samples")

    # Last grid point is one sample before the boundary: no boundary or future
    # sample can enter a pre-decision window.
    target = boundary - np.arange(n_samples, 0, -1, dtype=np.float64) / fs
    time_valid = (target >= lower_bound) & (target < boundary)
    source_keep = (
        (timestamps >= max(lower_bound, boundary - window_s - 2.0 / fs))
        & (timestamps < boundary)
    )
    source_t = timestamps[source_keep]
    source_x = np.asarray(values[source_keep], dtype=np.float64)
    if source_valid_mask is None:
        source_valid = np.ones(source_x.shape, dtype=bool)
    else:
        source_valid_mask = np.asarray(source_valid_mask, dtype=bool)
        if source_valid_mask.shape != values.shape:
            raise ValueError(
                "source_valid_mask must have the same shape as values: "
                f"{source_valid_mask.shape} != {values.shape}"
            )
        source_valid = source_valid_mask[source_keep]

    output = np.zeros((n_samples, values.shape[1]), dtype=np.float32)
    mask = np.zeros((n_samples, values.shape[1]), dtype=np.uint8)
    if source_t.size == 0:
        return output, mask, target

    max_nearest_distance = interpolation_gap_multiplier / fs
    for channel in range(values.shape[1]):
        finite = np.isfinite(source_x[:, channel]) & source_valid[:, channel]
        channel_t = source_t[finite]
        channel_x = source_x[finite, channel]
        if channel_t.size == 0:
            continue

        if channel_t.size == 1:
            nearest_distance = np.abs(target - channel_t[0])
            interpolated = np.full(target.shape, channel_x[0], dtype=np.float64)
        else:
            interpolated = np.interp(target, channel_t, channel_x)
            right = np.searchsorted(channel_t, target, side="left")
            right_clip = np.clip(right, 0, channel_t.size - 1)
            left_clip = np.clip(right - 1, 0, channel_t.size - 1)
            nearest_distance = np.minimum(
                np.abs(target - channel_t[left_clip]),
                np.abs(channel_t[right_clip] - target),
            )

        valid = (
            time_valid
            & (target >= channel_t[0])
            & (target <= channel_t[-1])
            & (nearest_distance <= max_nearest_distance + 1e-12)
            & np.isfinite(interpolated)
        )
        output[valid, channel] = interpolated[valid].astype(np.float32)
        mask[valid, channel] = 1

    return output, mask, target



def apply_causal_sos(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply an SOS filter along time without accessing future samples."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected non-empty (time, channel) data, got {values.shape}")

    initial = sosfilt_zi(sos)[:, :, None] * values[0][None, None, :]
    filtered, _ = sosfilt(sos, values, axis=0, zi=initial)
    return filtered


def causal_preprocess_eeg(
    eeg: np.ndarray,
    fs: float,
    lowcut: float = 1.0,
    highcut: float = 45.0,
    order: int = 4,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Causal CAR, 1-45 Hz band-pass and 50 Hz notch; no full-record ICA."""
    values = np.asarray(eeg, dtype=np.float64)

    # Forward filling is causal. Leading unavailable samples become zero.
    values = (
        pd.DataFrame(values)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    # Instantaneous CAR. Structurally zero-filled channels remain zero.
    present = np.abs(values) > 1e-12
    counts = np.maximum(present.sum(axis=1, keepdims=True), 1)
    reference = np.where(present, values, 0.0).sum(
        axis=1, keepdims=True
    ) / counts
    values = np.where(present, values - reference, 0.0)

    band_sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    values = apply_causal_sos(values, band_sos)

    notch_b, notch_a = iirnotch(notch_freq, notch_q, fs=fs)
    notch_sos = tf2sos(notch_b, notch_a)
    values = apply_causal_sos(values, notch_sos)

    return values.astype(np.float32)


def raw_et_channel_validity(et_values: np.ndarray) -> np.ndarray:
    """Build per-channel validity without trusting cleaned missing-value zeros."""
    values = np.asarray(et_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"Expected raw ET shape (time, >=6), got {values.shape}")
    valid = np.zeros((values.shape[0], 6), dtype=bool)
    for x_channel, y_channel, pupil_channel in ((0, 1, 2), (3, 4, 5)):
        eye_valid = (
            np.isfinite(values[:, x_channel])
            & np.isfinite(values[:, y_channel])
            & (values[:, x_channel] >= 0.0)
            & (values[:, x_channel] <= 1.0)
            & (values[:, y_channel] >= 0.0)
            & (values[:, y_channel] <= 1.0)
        )
        valid[:, x_channel] = eye_valid
        valid[:, y_channel] = eye_valid
        valid[:, pupil_channel] = (
            np.isfinite(values[:, pupil_channel]) & (values[:, pupil_channel] > 0.0)
        )
    return valid


def align_boolean_channels(
    source_times: np.ndarray,
    source_valid: np.ndarray,
    target_times: np.ndarray,
    max_distance_s: float,
) -> np.ndarray:
    """Nearest-neighbour align a Boolean channel mask to another time base."""
    source_times = np.asarray(source_times, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    source_valid = np.asarray(source_valid, dtype=bool)
    if source_valid.ndim != 2 or source_valid.shape[0] != source_times.size:
        raise ValueError("Invalid source validity shape")
    if source_times.size == 0:
        return np.zeros((target_times.size, source_valid.shape[1]), dtype=bool)

    right = np.searchsorted(source_times, target_times, side="left")
    right_clip = np.clip(right, 0, source_times.size - 1)
    left_clip = np.clip(right - 1, 0, source_times.size - 1)
    right_distance = np.abs(source_times[right_clip] - target_times)
    left_distance = np.abs(source_times[left_clip] - target_times)
    use_right = right_distance < left_distance
    nearest = np.where(use_right, right_clip, left_clip)
    distance = np.where(use_right, right_distance, left_distance)
    aligned = source_valid[nearest].copy()
    aligned[distance > max_distance_s] = False
    return aligned


def make_et_roi_relative(
    et_window: np.ndarray,
    et_mask: np.ndarray,
    roi: np.ndarray,
    image_width: int,
    image_height: int,
    context_clip: float,
) -> np.ndarray:
    """Convert normalized left/right gaze coordinates to product-relative axes."""
    output = np.asarray(et_window, dtype=np.float32).copy()
    x0, y0, width, height = np.asarray(roi, dtype=np.float64).reshape(4)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI dimensions: {roi}")

    for x_channel in (0, 3):
        valid = et_mask[:, x_channel].astype(bool)
        relative = (output[:, x_channel] * image_width - x0) / width
        output[valid, x_channel] = np.clip(
            relative[valid], -context_clip, 1.0 + context_clip
        )
        output[~valid, x_channel] = 0.0

    for y_channel in (1, 4):
        valid = et_mask[:, y_channel].astype(bool)
        relative = (output[:, y_channel] * image_height - y0) / height
        output[valid, y_channel] = np.clip(
            relative[valid], -context_clip, 1.0 + context_clip
        )
        output[~valid, y_channel] = 0.0

    output[et_mask == 0] = 0.0
    return output


def _resolve_existing(path: Path, base: Path | None = None) -> Path:
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _stream_dictionary(streams: Sequence[dict[str, Any]], audit) -> dict[str, Any]:
    return {audit._stream_name(stream): stream for stream in streams}


def process_subject(
    subject: str,
    rows: list[dict[str, Any]],
    config_dict: dict[str, Any],
    rois: dict[int, np.ndarray],
    image_width: int,
    image_height: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = CacheConfig(**config_dict)
    root = Path(config.root)
    output = Path(config.output)
    target = output / f"{subject}.npz"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")

    audit = _load_module(
        Path(config.audit_script), f"neuroclick_audit_support_{subject}"
    )
    harmonizer, channels = load_harmonizer(root)

    # Reconstruct the exact gaze visits used to create the frozen manifest.
    xdf_path = root / "DataSource" / f"{subject}.xdf"
    selectors = [
        {"name": "MyMarkerStream3"},
        {"name": "Tobii"},
        {"name": "WS-default"},
    ]
    streams, _ = audit.pyxdf.load_xdf(
        str(xdf_path), select_streams=selectors, verbose=False
    )
    by_name = _stream_dictionary(streams, audit)
    missing_streams = sorted(
        {"MyMarkerStream3", "Tobii", "WS-default"} - set(by_name)
    )
    if missing_streams:
        raise ValueError(f"{subject}: missing streams {missing_streams}")

    marker_times, marker_values = audit._event_stream(by_name["MyMarkerStream3"])
    raw_et_times, raw_et_values = audit._numeric_stream(by_name["Tobii"])
    raw_eeg_times, raw_eeg_values = audit._numeric_stream(
        by_name["WS-default"]
    )
    intervals, _ = audit.build_page_intervals(marker_times, marker_values)
    interval_lookup = {
        (int(interval.page), int(interval.visit_index)): interval
        for interval in intervals
    }
    gaze_x, gaze_y, gaze_valid = audit.binocular_gaze(raw_et_values)
    gaze_x_px = gaze_x * image_width
    gaze_y_px = gaze_y * image_height
    raw_et_rate = audit._effective_rate(raw_et_times)

    audit_config = audit.AuditConfig(
        root=str(root),
        output=str(output),
        screen_width=config.screen_width,
        screen_height=config.screen_height,
        min_fixation_ms=config.min_fixation_ms,
        merge_gap_ms=config.merge_gap_ms,
        pre_click_margin_ms=config.pre_click_margin_ms,
        max_mouse_staleness_s=60.0,
    )


    # Load signals directly from XDF. No legacy full-session clean arrays.
    eeg = np.asarray(raw_eeg_values, dtype=np.float64)
    eeg = (
        pd.DataFrame(eeg)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    eeg, eeg_times = validate_time_series(
        eeg,
        raw_eeg_times,
        f"{subject}/raw_EEG",
    )

    raw_et_values = np.asarray(raw_et_values[:, :6], dtype=np.float64)
    raw_et_valid = raw_et_channel_validity(raw_et_values)
    et = np.nan_to_num(
        raw_et_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    et, et_times = validate_time_series(
        et,
        raw_et_times,
        f"{subject}/raw_ET",
        6,
    )
    clean_et_valid = raw_et_valid

    eeg_19 = np.asarray(
        harmonizer.harmonize_eeg_channels(
            eeg,
            subject,
            channel_names=None,
            verbose=False,
        ),
        dtype=np.float32,
    )
    if eeg_19.shape != (len(eeg_times), 19):
        raise RuntimeError(
            f"{subject}: harmonizer returned {eeg_19.shape}; "
            f"expected ({len(eeg_times)}, 19)"
        )

    eeg_19 = causal_preprocess_eeg(eeg_19, fs=config.eeg_fs)

    eeg_windows: list[np.ndarray] = []
    eeg_masks: list[np.ndarray] = []
    et_windows: list[np.ndarray] = []
    et_masks: list[np.ndarray] = []
    visit_start: list[float] = []
    visit_end: list[float] = []
    visit_dwell_s: list[float] = []
    visit_page_index: list[int] = []
    visit_rank: list[int] = []
    bag_offsets = [0]
    index_records: list[dict[str, Any]] = []
    truncated_instances = 0

    rows = sorted(rows, key=lambda item: (int(item["page"]), int(item["roi"])))
    for bag_index, row in enumerate(rows):
        page = int(row["page"])
        roi_index = int(row["roi"])
        cutoff = float(row["predecision_cutoff_time"])
        page_intervals = [interval for interval in intervals if interval.page == page]
        usable_intervals = [interval for interval in page_intervals if interval.start < cutoff]

        visits = []
        for interval in usable_intervals:
            visits.extend(
                audit.gaze_visits_for_roi(
                    et_times=raw_et_times,
                    gaze_x_px=gaze_x_px,
                    gaze_y_px=gaze_y_px,
                    gaze_valid=gaze_valid,
                    interval=interval,
                    roi=rois[page][roi_index - 1],
                    cutoff=cutoff,
                    config=audit_config,
                    et_rate=raw_et_rate,
                )
            )
        visits.sort(key=lambda item: float(item.start))

        expected_visits = int(row["n_gaze_visits"])
        if len(visits) != expected_visits:
            raise RuntimeError(
                f"{subject}/P{page}/R{roi_index}: reconstructed {len(visits)} "
                f"visits but frozen audit records {expected_visits}. "
                "Use the same audit parameters that produced the manifest."
            )
        if not visits:
            raise RuntimeError(
                f"{subject}/P{page}/R{roi_index}: considered instance has no visits"
            )

        original_visit_count = len(visits)
        if config.max_visits > 0 and len(visits) > config.max_visits:
            visits = visits[-config.max_visits :]
            truncated_instances += 1

        first_global_visit = bag_offsets[-1]
        for local_rank, visit in enumerate(visits):
            interval_key = (int(visit.page), int(visit.page_visit_index))
            if interval_key not in interval_lookup:
                raise RuntimeError(f"{subject}: missing page interval {interval_key}")
            interval = interval_lookup[interval_key]
            boundary = min(float(visit.end), cutoff, float(interval.end))

            eeg_window, eeg_mask, _ = resample_preboundary_window(
                eeg_19,
                eeg_times,
                boundary=boundary,
                lower_bound=float(interval.start),
                fs=config.eeg_fs,
                window_s=config.window_s,
                interpolation_gap_multiplier=config.interpolation_gap_multiplier,
            )
            et_window, et_mask, _ = resample_preboundary_window(
                et,
                et_times,
                boundary=boundary,
                lower_bound=float(interval.start),
                fs=config.et_fs,
                window_s=config.window_s,
                interpolation_gap_multiplier=config.interpolation_gap_multiplier,
                source_valid_mask=clean_et_valid,
            )
            et_window = make_et_roi_relative(
                et_window,
                et_mask,
                rois[page][roi_index - 1],
                image_width,
                image_height,
                config.roi_context_clip,
            )

            if not eeg_mask.any():
                raise RuntimeError(
                    f"{subject}/P{page}/R{roi_index}/visit{local_rank}: empty EEG window"
                )
            if not et_mask.any():
                raise RuntimeError(
                    f"{subject}/P{page}/R{roi_index}/visit{local_rank}: empty ET window"
                )

            eeg_windows.append(eeg_window)
            eeg_masks.append(eeg_mask)
            et_windows.append(et_window)
            et_masks.append(et_mask)
            visit_start.append(float(visit.start))
            visit_end.append(float(boundary))
            visit_dwell_s.append(float(visit.dwell_s))
            visit_page_index.append(int(visit.page_visit_index))
            visit_rank.append(int(local_rank))

        bag_offsets.append(len(eeg_windows))
        index_records.append(
            {
                "subject": subject,
                "cache_file": target.name,
                "bag_index": bag_index,
                "page": page,
                "roi": roi_index,
                "product_id": int(row["product_id"]),
                "label_buy": int(row["label_buy"]),
                "n_visits_audit": original_visit_count,
                "n_visits_cached": len(visits),
                "first_visit_index": first_global_visit,
                "last_visit_index_exclusive": bag_offsets[-1],
                "first_click_time": float(row["first_click_time"]),
                "predecision_cutoff_time": cutoff,
            }
        )

    n_eeg = int(round(config.eeg_fs * config.window_s))
    n_et = int(round(config.et_fs * config.window_s))
    arrays = {
        "eeg": np.stack(eeg_windows).astype(np.float32),
        "eeg_mask": np.stack(eeg_masks).astype(np.uint8),
        "et_roi": np.stack(et_windows).astype(np.float32),
        "et_mask": np.stack(et_masks).astype(np.uint8),
        "bag_offsets": np.asarray(bag_offsets, dtype=np.int64),
        "labels": np.asarray([record["label_buy"] for record in index_records], dtype=np.uint8),
        "page": np.asarray([record["page"] for record in index_records], dtype=np.uint8),
        "roi": np.asarray([record["roi"] for record in index_records], dtype=np.uint8),
        "product_id": np.asarray([record["product_id"] for record in index_records], dtype=np.uint16),
        "visit_start": np.asarray(visit_start, dtype=np.float64),
        "visit_end": np.asarray(visit_end, dtype=np.float64),
        "visit_dwell_s": np.asarray(visit_dwell_s, dtype=np.float32),
        "visit_page_index": np.asarray(visit_page_index, dtype=np.uint8),
        "visit_rank": np.asarray(visit_rank, dtype=np.uint16),
        "canonical_channels": np.asarray(channels, dtype="U8"),
        "preprocessing_version": np.asarray("causal_v1"),
    }
    if arrays["eeg"].shape[1:] != (n_eeg, 19):
        raise RuntimeError(f"{subject}: invalid cached EEG shape {arrays['eeg'].shape}")
    if arrays["et_roi"].shape[1:] != (n_et, 6):
        raise RuntimeError(f"{subject}: invalid cached ET shape {arrays['et_roi'].shape}")
    if arrays["bag_offsets"][-1] != arrays["eeg"].shape[0]:
        raise RuntimeError(f"{subject}: bag offsets do not cover all visits")
    if np.any(np.diff(arrays["bag_offsets"]) <= 0):
        raise RuntimeError(f"{subject}: every considered bag must contain a visit")
    if not (np.isfinite(arrays["eeg"]).all() and np.isfinite(arrays["et_roi"]).all()):
        raise RuntimeError(f"{subject}: non-finite cached signal value")

    temporary = target.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, target)

    stats = {
        "subject": subject,
        "instances": len(index_records),
        "buys": int(sum(record["label_buy"] for record in index_records)),
        "nobuys": int(len(index_records) - sum(record["label_buy"] for record in index_records)),
        "visits": len(eeg_windows),
        "multivisit_instances": int(sum(record["n_visits_cached"] >= 2 for record in index_records)),
        "truncated_instances": truncated_instances,
        "eeg_effective_rate": effective_rate(eeg_times),
        "et_effective_rate": effective_rate(et_times),
        "raw_et_effective_rate": float(raw_et_rate),
        "raw_eeg_channels": int(eeg.shape[1]),
        "eeg_valid_fraction": float(arrays["eeg_mask"].mean()),
        "et_valid_fraction": float(arrays["et_mask"].mean()),
        "cache_bytes": int(target.stat().st_size),
    }
    return index_records, stats


def run_self_test() -> None:
    fs = 10.0
    timestamps = np.arange(0.0, 2.0, 1.0 / fs)
    values = np.column_stack([timestamps, timestamps * 2.0])
    window, mask, target = resample_preboundary_window(
        values,
        timestamps,
        boundary=1.5,
        lower_bound=0.8,
        fs=fs,
        window_s=1.0,
        interpolation_gap_multiplier=2.5,
    )
    assert window.shape == (10, 2)
    assert mask.shape == (10, 2)
    assert np.all(target < 1.5)
    assert not mask[:3].any(), "page-boundary padding was not applied"
    assert np.allclose(window[mask.astype(bool)], np.column_stack([target, target * 2.0])[mask.astype(bool)])

    et = np.array([[0.5, 0.5, 3.0, 0.6, 0.4, 4.0]], dtype=np.float32)
    et_mask = np.ones_like(et, dtype=np.uint8)
    relative = make_et_roi_relative(
        et,
        et_mask,
        roi=np.array([400.0, 300.0, 200.0, 100.0]),
        image_width=1000,
        image_height=800,
        context_clip=2.0,
    )
    assert np.allclose(relative[0, [0, 1, 3, 4]], [0.5, 1.0, 1.0, 0.2])
    assert np.allclose(relative[0, [2, 5]], [3.0, 4.0])

    raw_et = np.array(
        [
            [0.5, 0.5, 3.0, 0.6, 0.4, 4.0],
            [np.nan, np.nan, 0.0, 0.7, 0.3, 5.0],
        ]
    )
    raw_valid = raw_et_channel_validity(raw_et)
    assert raw_valid[0].all()
    assert not raw_valid[1, :3].any()
    assert raw_valid[1, 3:].all()
    aligned = align_boolean_channels(
        np.array([0.0, 0.1]), raw_valid, np.array([0.01, 0.09, 1.0]), 0.03
    )
    assert aligned[0].all()
    assert not aligned[1, :3].any() and aligned[1, 3:].all()
    assert not aligned[2].any()

    # Future perturbations must not alter any causal EEG prefix.
    rng = np.random.default_rng(42)
    test_eeg = rng.normal(size=(900, 19))
    cutoff_index = 600
    original = causal_preprocess_eeg(test_eeg, fs=300.0)
    changed = test_eeg.copy()
    changed[cutoff_index:] += rng.normal(
        loc=100.0,
        scale=20.0,
        size=changed[cutoff_index:].shape,
    )
    changed_output = causal_preprocess_eeg(changed, fs=300.0)
    assert np.allclose(
        original[:cutoff_index],
        changed_output[:cutoff_index],
        atol=1e-6,
        rtol=1e-6,
    ), "future EEG samples changed the causal prefix"

    print("SELF-TEST: PASS - boundary and causal-prefix checks")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    root = args.root.resolve()
    project = Path.cwd().resolve()
    audit_csv = _resolve_existing(args.audit_csv, project)
    audit_script = _resolve_existing(args.audit_script, project)
    output = args.output if args.output.is_absolute() else project / args.output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_visits < 0:
        raise ValueError("--max-visits cannot be negative")
    if args.pre_click_margin_ms < 0:
        raise ValueError("--pre-click-margin-ms cannot be negative")

    frame = pd.read_csv(audit_csv)
    missing_columns = sorted(SUBJECT_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Audit CSV is missing columns: {missing_columns}")
    frame = frame.loc[frame["considered"].astype(int) == 1].copy()
    if frame.empty:
        raise ValueError("Audit manifest contains no considered instances")
    if frame.duplicated(["subject", "product_id"]).any():
        raise ValueError("Audit manifest has duplicate subject-product rows")

    available_subjects = sorted(frame["subject"].astype(str).unique())
    subjects = args.subjects or available_subjects
    unknown = sorted(set(subjects) - set(available_subjects))
    if unknown:
        raise ValueError(f"Subjects absent from audit manifest: {unknown}")
    frame = frame.loc[frame["subject"].astype(str).isin(subjects)].copy()

    config = CacheConfig(
        root=str(root),
        audit_script=str(audit_script),
        output=str(output),
        eeg_fs=args.eeg_fs,
        et_fs=args.et_fs,
        window_s=args.window_s,
        min_fixation_ms=args.min_fixation_ms,
        merge_gap_ms=args.merge_gap_ms,
        pre_click_margin_ms=args.pre_click_margin_ms,
        max_visits=args.max_visits,
        interpolation_gap_multiplier=args.interpolation_gap_multiplier,
        roi_context_clip=args.roi_context_clip,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
    )

    audit = _load_module(audit_script, "neuroclick_audit_parent")
    rois, image_width, image_height = audit.load_rois(root)

    print("=" * 78)
    print("NEUROCLICK PRE-CLICK SEQUENCE CACHE")
    print("=" * 78)
    print(f"Root                 : {root}")
    print(f"Frozen audit         : {audit_csv}")
    print(f"Output               : {output}")
    print(f"Subjects             : {len(subjects)}")
    print(f"Considered instances : {len(frame)}")
    print(f"EEG visit tensor     : {round(args.eeg_fs * args.window_s)} x 19")
    print(f"ET visit tensor      : {round(args.et_fs * args.window_s)} x 6")
    print(f"Pre-click margin     : {args.pre_click_margin_ms:g} ms (frozen audit)")
    print(f"Maximum visits       : {'all' if args.max_visits == 0 else args.max_visits}")
    print("Normalization        : none; fit scaling on training subjects only")

    grouped = {
        subject: frame.loc[frame["subject"].astype(str) == subject].to_dict("records")
        for subject in subjects
    }
    all_records: list[dict[str, Any]] = []
    all_stats: list[dict[str, Any]] = []
    errors: list[str] = []

    if args.workers == 1:
        for subject in subjects:
            try:
                records, stats = process_subject(
                    subject,
                    grouped[subject],
                    asdict(config),
                    rois,
                    image_width,
                    image_height,
                    args.overwrite,
                )
                all_records.extend(records)
                all_stats.append(stats)
                print(
                    f"{subject}: instances={stats['instances']:3d}, "
                    f"Buy={stats['buys']:2d}, visits={stats['visits']:4d}, "
                    f"EEG-valid={stats['eeg_valid_fraction']:.3f}, "
                    f"ET-valid={stats['et_valid_fraction']:.3f}"
                )
            except Exception:
                errors.append(f"{subject}:\n{traceback.format_exc()}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_subject,
                    subject,
                    grouped[subject],
                    asdict(config),
                    rois,
                    image_width,
                    image_height,
                    args.overwrite,
                ): subject
                for subject in subjects
            }
            for future in as_completed(futures):
                subject = futures[future]
                try:
                    records, stats = future.result()
                    all_records.extend(records)
                    all_stats.append(stats)
                    print(
                        f"{subject}: instances={stats['instances']:3d}, "
                        f"Buy={stats['buys']:2d}, visits={stats['visits']:4d}, "
                        f"EEG-valid={stats['eeg_valid_fraction']:.3f}, "
                        f"ET-valid={stats['et_valid_fraction']:.3f}"
                    )
                except Exception:
                    errors.append(f"{subject}:\n{traceback.format_exc()}")

    if errors:
        (output / "CACHE_ERRORS.txt").write_text("\n\n".join(errors), encoding="utf-8")
        raise RuntimeError(
            f"Cache failed for {len(errors)} subject(s). See {output / 'CACHE_ERRORS.txt'}"
        )

    index = pd.DataFrame.from_records(all_records).sort_values(
        ["subject", "bag_index"]
    )
    subject_summary = pd.DataFrame.from_records(all_stats).sort_values("subject")
    index.to_csv(output / "cache_index.csv", index=False)
    subject_summary.to_csv(output / "subject_cache_summary.csv", index=False)

    total_instances = int(len(index))
    total_buys = int(index["label_buy"].sum())
    total_visits = int(subject_summary["visits"].sum())
    if args.expect_instances and total_instances != args.expect_instances:
        raise RuntimeError(
            f"Expected {args.expect_instances} instances, cached {total_instances}"
        )
    if args.expect_buys and total_buys != args.expect_buys:
        raise RuntimeError(f"Expected {args.expect_buys} Buy cases, cached {total_buys}")

    harmonizer_path = root / "src" / "model" / "data" / "channel_harmonizer.py"
    summary = {
        "decision": "CACHE_GO",
        "subjects": len(subjects),
        "instances": total_instances,
        "buys": total_buys,
        "nobuys": total_instances - total_buys,
        "buy_prevalence": total_buys / total_instances,
        "visits": total_visits,
        "multivisit_instances": int(subject_summary["multivisit_instances"].sum()),
        "truncated_instances": int(subject_summary["truncated_instances"].sum()),
        "config": asdict(config),
        "canonical_channels": [
            "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Pz",
            "Fp1", "Fp2", "T3", "T5", "O1", "O2", "F7", "F8", "T6", "T4",
        ],
        "hashes": {
            "audit_csv_sha256": sha256_file(audit_csv),
            "audit_script_sha256": sha256_file(audit_script),
            "harmonizer_sha256": sha256_file(harmonizer_path),
            "cache_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "cache_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "CACHE_DECISION.txt").write_text(
        "\n".join(
            [
                "CACHE_GO",
                "",
                f"Subjects: {len(subjects)}",
                f"Considered subject-product instances: {total_instances}",
                f"Buy: {total_buys}",
                f"NoBuy: {total_instances - total_buys}",
                f"Chronological gaze visits: {total_visits}",
                f"Truncated instances: {summary['truncated_instances']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("DECISION             : CACHE_GO")
    print(f"Subjects             : {len(subjects)}")
    print(f"Instances            : {total_instances}")
    print(f"Buy / NoBuy          : {total_buys} / {total_instances - total_buys}")
    print(f"Chronological visits : {total_visits}")
    print(f"Truncated instances  : {summary['truncated_instances']}")
    print(f"Saved outputs to     : {output}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
