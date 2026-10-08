#!/usr/bin/env python3
"""Build a leakage-aware NeuMa subject-product Buy/NoBuy audit.

The script reconstructs one decision instance per subject and product from the
raw XDF streams.  A product is labelled Buy when a left-button press occurs
inside that product's ROI while its brochure page is active.  Gaze evidence is
restricted to page-active samples before the first product click (with an
optional motor-artifact margin).  Products that receive sufficient pre-decision
gaze are marked as ``considered`` for the primary modelling task.

This script does not train a model and does not alter source data.

Expected NeuMa inputs under --root:
  DataSource/SXX.xdf
  DataSource/SXX.xlsx
  DataSource/Dependencies/BoundingBox_Coordinates/BoundingBoxPage_N.mat
  DataSource/Dependencies/Brochure_Pages/ImagePage_N.tif

Outputs:
  subject_product_audit.csv
  subject_summary.csv
  unmatched_clicks.csv
  audit_summary.json
  FEASIBILITY_DECISION.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyxdf
from PIL import Image
from scipy.io import loadmat


PAGE_RE = re.compile(r"FYLLADIO_(\d+)", flags=re.IGNORECASE)
SUBJECT_RE = re.compile(r"^S\d{2}$")


@dataclass(frozen=True)
class AuditConfig:
    root: str
    output: str
    screen_width: int = 1920
    screen_height: int = 1080
    min_fixation_ms: float = 100.0
    merge_gap_ms: float = 75.0
    pre_click_margin_ms: float = 500.0
    max_mouse_staleness_s: float = 10.0
    min_valid_subjects: int = 35
    min_usable_buys: int = 500
    min_positive_multivisit: int = 100


@dataclass(frozen=True)
class PageInterval:
    page: int
    start: float
    end: float
    visit_index: int


@dataclass(frozen=True)
class GazeVisit:
    page: int
    page_visit_index: int
    start: float
    end: float
    dwell_s: float
    n_samples: int


def _text(value: Any) -> str:
    """Flatten an XDF marker value to a readable string."""
    array = np.asarray(value, dtype=object).reshape(-1)
    return " ".join(str(item) for item in array)


def _stream_name(stream: dict[str, Any]) -> str:
    return str(stream["info"]["name"][0])


def _numeric_stream(stream: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(stream["time_stamps"], dtype=np.float64)
    values = np.asarray(stream["time_series"], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return timestamps, values


def _event_stream(stream: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    timestamps = np.asarray(stream["time_stamps"], dtype=np.float64)
    values = [_text(value) for value in stream["time_series"]]
    return timestamps, values


def _effective_rate(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return math.nan
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        return math.nan
    return float((timestamps.size - 1) / duration)


def load_rois(root: Path) -> tuple[dict[int, np.ndarray], int, int]:
    """Load six pages of 24 [x, y, width, height] brochure-pixel ROIs."""
    bbox_root = (
        root
        / "DataSource"
        / "Dependencies"
        / "BoundingBox_Coordinates"
    )
    image_root = root / "DataSource" / "Dependencies" / "Brochure_Pages"

    rois: dict[int, np.ndarray] = {}
    image_width: int | None = None
    image_height: int | None = None

    for page in range(1, 7):
        image_path = image_root / f"ImagePage_{page}.tif"
        mat_path = bbox_root / f"BoundingBoxPage_{page}.mat"

        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        with Image.open(image_path) as image:
            width, height = image.size

        if image_width is None:
            image_width, image_height = int(width), int(height)
        elif (width, height) != (image_width, image_height):
            raise ValueError(
                f"Inconsistent brochure size on page {page}: {(width, height)}"
            )

        data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if "ROI_list" not in data:
            raise KeyError(f"ROI_list missing from {mat_path}")

        raw = np.asarray(data["ROI_list"], dtype=object).reshape(-1)
        page_rois = np.stack(
            [np.asarray(item, dtype=np.float64).reshape(4) for item in raw]
        )

        if page_rois.shape != (24, 4):
            raise ValueError(
                f"Expected 24x4 ROIs for page {page}, got {page_rois.shape}"
            )
        if not np.isfinite(page_rois).all():
            raise ValueError(f"Non-finite ROI coordinate on page {page}")
        if (page_rois[:, 2:] <= 0).any():
            raise ValueError(f"Non-positive ROI width/height on page {page}")

        rois[page] = page_rois

    assert image_width is not None and image_height is not None
    return rois, image_width, image_height


def parse_page(marker: str) -> int | None:
    match = PAGE_RE.search(marker)
    if not match:
        return None
    page = int(match.group(1))
    return page if 1 <= page <= 6 else None


def build_page_intervals(
    marker_times: np.ndarray,
    marker_values: Sequence[str],
) -> tuple[list[PageInterval], float | None]:
    """Create page-active intervals, stopping at the first EOE marker."""
    events = sorted(
        [(float(t), str(v)) for t, v in zip(marker_times, marker_values)],
        key=lambda item: item[0],
    )
    first_eoe = next(
        (timestamp for timestamp, value in events if value.strip().upper() == "EOE"),
        None,
    )

    usable = [
        (timestamp, value)
        for timestamp, value in events
        if first_eoe is None or timestamp <= first_eoe
    ]
    intervals: list[PageInterval] = []
    page_visit_counts = {page: 0 for page in range(1, 7)}

    for index, (start, value) in enumerate(usable):
        page = parse_page(value)
        if page is None:
            continue

        if index + 1 < len(usable):
            end = float(usable[index + 1][0])
        elif first_eoe is not None:
            end = float(first_eoe)
        else:
            continue

        if end <= start:
            continue

        page_visit_counts[page] += 1
        intervals.append(
            PageInterval(
                page=page,
                start=float(start),
                end=end,
                visit_index=page_visit_counts[page],
            )
        )

    return intervals, first_eoe


def containing_roi_indices(rois: np.ndarray, x: float, y: float) -> list[int]:
    if not (math.isfinite(x) and math.isfinite(y)):
        return []
    x0 = rois[:, 0]
    y0 = rois[:, 1]
    x1 = x0 + rois[:, 2]
    y1 = y0 + rois[:, 3]
    mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return np.flatnonzero(mask).astype(int).tolist()


def active_interval(
    intervals: Sequence[PageInterval], timestamp: float
) -> PageInterval | None:
    for interval in intervals:
        if interval.start <= timestamp < interval.end:
            return interval
    return None


def map_clicks(
    subject: str,
    button_times: np.ndarray,
    button_values: Sequence[str],
    position_times: np.ndarray,
    position_values: np.ndarray,
    intervals: Sequence[PageInterval],
    rois: dict[int, np.ndarray],
    image_width: int,
    image_height: int,
    config: AuditConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Map left-button presses to page/product using the last cursor position."""
    scale_x = float(image_width / config.screen_width)
    scale_y = float(image_height / config.screen_height)

    mapped: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    counts = {
        "raw_left_presses": 0,
        "in_page_presses": 0,
        "mapped_product_presses": 0,
        "outside_page_presses": 0,
        "stale_or_missing_position": 0,
        "outside_roi_presses": 0,
        "ambiguous_roi_presses": 0,
    }

    for click_time, value in zip(button_times, button_values):
        if "left" not in value.lower() or "pressed" not in value.lower():
            continue

        counts["raw_left_presses"] += 1
        click_time = float(click_time)
        interval = active_interval(intervals, click_time)

        if interval is None:
            counts["outside_page_presses"] += 1
            unmatched.append(
                {
                    "subject": subject,
                    "click_time": click_time,
                    "reason": "outside_page_interval",
                    "page": np.nan,
                    "screen_x": np.nan,
                    "screen_y": np.nan,
                    "brochure_x": np.nan,
                    "brochure_y": np.nan,
                    "mouse_staleness_s": np.nan,
                }
            )
            continue

        counts["in_page_presses"] += 1
        position_index = int(np.searchsorted(position_times, click_time, side="right") - 1)

        if position_index < 0:
            counts["stale_or_missing_position"] += 1
            unmatched.append(
                {
                    "subject": subject,
                    "click_time": click_time,
                    "reason": "no_prior_mouse_position",
                    "page": interval.page,
                    "screen_x": np.nan,
                    "screen_y": np.nan,
                    "brochure_x": np.nan,
                    "brochure_y": np.nan,
                    "mouse_staleness_s": np.nan,
                }
            )
            continue

        screen_x, screen_y = position_values[position_index, :2]
        staleness = float(click_time - position_times[position_index])
        brochure_x = float(screen_x * scale_x)
        brochure_y = float(screen_y * scale_y)

        base = {
            "subject": subject,
            "click_time": click_time,
            "page": interval.page,
            "screen_x": float(screen_x),
            "screen_y": float(screen_y),
            "brochure_x": brochure_x,
            "brochure_y": brochure_y,
            "mouse_staleness_s": staleness,
        }

        if (
            not np.isfinite([screen_x, screen_y]).all()
            or staleness < 0
            or staleness > config.max_mouse_staleness_s
        ):
            counts["stale_or_missing_position"] += 1
            unmatched.append({**base, "reason": "stale_or_invalid_mouse_position"})
            continue

        candidates = containing_roi_indices(
            rois[interval.page], brochure_x, brochure_y
        )
        if not candidates:
            counts["outside_roi_presses"] += 1
            unmatched.append({**base, "reason": "outside_all_product_rois"})
            continue

        if len(candidates) > 1:
            counts["ambiguous_roi_presses"] += 1
            candidate_areas = [
                float(np.prod(rois[interval.page][index, 2:]))
                for index in candidates
            ]
            roi_zero_based = candidates[int(np.argmin(candidate_areas))]
        else:
            roi_zero_based = candidates[0]

        roi = int(roi_zero_based + 1)
        product_id = int((interval.page - 1) * 24 + roi)
        counts["mapped_product_presses"] += 1
        mapped.append(
            {
                **base,
                "roi": roi,
                "product_id": product_id,
                "page_visit_index": interval.visit_index,
                "ambiguous_roi": int(len(candidates) > 1),
            }
        )

    return mapped, unmatched, counts


def binocular_gaze(et_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized binocular mean gaze and a validity mask."""
    if et_values.ndim != 2 or et_values.shape[1] < 5:
        raise ValueError(f"Expected at least 5 ET channels, got {et_values.shape}")

    left_x, left_y = et_values[:, 0], et_values[:, 1]
    right_x, right_y = et_values[:, 3], et_values[:, 4]

    left_valid = (
        np.isfinite(left_x)
        & np.isfinite(left_y)
        & (left_x >= 0)
        & (left_x <= 1)
        & (left_y >= 0)
        & (left_y <= 1)
    )
    right_valid = (
        np.isfinite(right_x)
        & np.isfinite(right_y)
        & (right_x >= 0)
        & (right_x <= 1)
        & (right_y >= 0)
        & (right_y <= 1)
    )

    gaze_x = np.full(et_values.shape[0], np.nan, dtype=np.float64)
    gaze_y = np.full(et_values.shape[0], np.nan, dtype=np.float64)
    counts = left_valid.astype(int) + right_valid.astype(int)

    both_or_one = counts > 0
    gaze_x[both_or_one] = (
        np.where(left_valid, left_x, 0.0)[both_or_one]
        + np.where(right_valid, right_x, 0.0)[both_or_one]
    ) / counts[both_or_one]
    gaze_y[both_or_one] = (
        np.where(left_valid, left_y, 0.0)[both_or_one]
        + np.where(right_valid, right_y, 0.0)[both_or_one]
    ) / counts[both_or_one]
    return gaze_x, gaze_y, both_or_one


def gaze_visits_for_roi(
    et_times: np.ndarray,
    gaze_x_px: np.ndarray,
    gaze_y_px: np.ndarray,
    gaze_valid: np.ndarray,
    interval: PageInterval,
    roi: np.ndarray,
    cutoff: float,
    config: AuditConfig,
    et_rate: float,
) -> list[GazeVisit]:
    """Detect temporally ordered in-ROI gaze visits inside one page visit."""
    interval_end = min(float(interval.end), float(cutoff))
    if interval_end <= interval.start:
        return []

    in_time = (et_times >= interval.start) & (et_times < interval_end)
    x0, y0, width, height = [float(item) for item in roi]
    inside = (
        in_time
        & gaze_valid
        & (gaze_x_px >= x0)
        & (gaze_x_px <= x0 + width)
        & (gaze_y_px >= y0)
        & (gaze_y_px <= y0 + height)
    )
    indices = np.flatnonzero(inside)
    if indices.size == 0:
        return []

    merge_gap_s = float(config.merge_gap_ms / 1000.0)
    min_fixation_s = float(config.min_fixation_ms / 1000.0)
    nominal_dt = 1.0 / et_rate if math.isfinite(et_rate) and et_rate > 0 else 1 / 120

    gaps = np.diff(et_times[indices])
    split_points = np.flatnonzero(gaps > merge_gap_s) + 1
    groups = np.split(indices, split_points)

    visits: list[GazeVisit] = []
    for group in groups:
        if group.size == 0:
            continue
        dwell_s = float(group.size * nominal_dt)
        if dwell_s + 1e-12 < min_fixation_s:
            continue
        visits.append(
            GazeVisit(
                page=interval.page,
                page_visit_index=interval.visit_index,
                start=float(et_times[group[0]]),
                end=float(et_times[group[-1]] + nominal_dt),
                dwell_s=dwell_s,
                n_samples=int(group.size),
            )
        )
    return visits


def _xlsx_q76_count(path: Path) -> int | None:
    """Auxiliary questionnaire count only; never used to create labels."""
    if not path.exists():
        return None
    try:
        frame = pd.read_excel(path, usecols=lambda col: str(col) == "Q76")
        if "Q76" not in frame:
            return None
        return int(frame["Q76"].notna().sum())
    except Exception:
        return None


def process_subject(
    subject: str,
    config_dict: dict[str, Any],
    rois: dict[int, np.ndarray],
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    config = AuditConfig(**config_dict)
    root = Path(config.root)
    xdf_path = root / "DataSource" / f"{subject}.xdf"
    xlsx_path = root / "DataSource" / f"{subject}.xlsx"

    selectors = [
        {"name": "MyMarkerStream3"},
        {"name": "MouseButtons"},
        {"name": "MousePosition"},
        {"name": "Tobii"},
        {"name": "WS-default"},
    ]
    streams, _ = pyxdf.load_xdf(
        str(xdf_path), select_streams=selectors, verbose=False
    )
    by_name = {_stream_name(stream): stream for stream in streams}
    required = {item["name"] for item in selectors}
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(f"{subject}: missing streams {missing}")

    marker_times, marker_values = _event_stream(by_name["MyMarkerStream3"])
    button_times, button_values = _event_stream(by_name["MouseButtons"])
    position_times, position_values = _numeric_stream(by_name["MousePosition"])
    et_times, et_values = _numeric_stream(by_name["Tobii"])
    eeg_times, eeg_values = _numeric_stream(by_name["WS-default"])

    intervals, eoe_time = build_page_intervals(marker_times, marker_values)
    if not intervals:
        raise ValueError(f"{subject}: no usable brochure page intervals")

    mapped_clicks, unmatched_clicks, click_counts = map_clicks(
        subject=subject,
        button_times=button_times,
        button_values=button_values,
        position_times=position_times,
        position_values=position_values,
        intervals=intervals,
        rois=rois,
        image_width=image_width,
        image_height=image_height,
        config=config,
    )

    clicks_by_product: dict[int, list[dict[str, Any]]] = {}
    for click in mapped_clicks:
        clicks_by_product.setdefault(int(click["product_id"]), []).append(click)
    for product_clicks in clicks_by_product.values():
        product_clicks.sort(key=lambda item: float(item["click_time"]))

    gaze_x, gaze_y, gaze_valid = binocular_gaze(et_values)
    gaze_x_px = gaze_x * image_width
    gaze_y_px = gaze_y * image_height
    et_rate = _effective_rate(et_times)
    eeg_rate = _effective_rate(eeg_times)
    et_valid_fraction = float(gaze_valid.mean()) if gaze_valid.size else 0.0

    pre_click_margin_s = float(config.pre_click_margin_ms / 1000.0)
    records: list[dict[str, Any]] = []

    for page in range(1, 7):
        page_intervals = [interval for interval in intervals if interval.page == page]
        for roi in range(1, 25):
            product_id = int((page - 1) * 24 + roi)
            product_clicks = clicks_by_product.get(product_id, [])
            label = int(bool(product_clicks))
            first_click_time = (
                float(product_clicks[0]["click_time"]) if product_clicks else math.nan
            )
            cutoff = (
                first_click_time - pre_click_margin_s
                if label
                else max(interval.end for interval in page_intervals)
            )

            usable_page_intervals = [
                interval for interval in page_intervals if interval.start < cutoff
            ]
            exposure_s = float(
                sum(
                    max(0.0, min(interval.end, cutoff) - interval.start)
                    for interval in usable_page_intervals
                )
            )

            visits: list[GazeVisit] = []
            for interval in usable_page_intervals:
                visits.extend(
                    gaze_visits_for_roi(
                        et_times=et_times,
                        gaze_x_px=gaze_x_px,
                        gaze_y_px=gaze_y_px,
                        gaze_valid=gaze_valid,
                        interval=interval,
                        roi=rois[page][roi - 1],
                        cutoff=cutoff,
                        config=config,
                        et_rate=et_rate,
                    )
                )
            visits.sort(key=lambda item: item.start)

            total_dwell_s = float(sum(visit.dwell_s for visit in visits))
            considered = int(bool(visits))
            first_gaze_time = visits[0].start if visits else math.nan
            last_gaze_time = visits[-1].end if visits else math.nan
            latency_to_click_s = (
                float(first_click_time - first_gaze_time)
                if label and visits
                else math.nan
            )

            eeg_overlap_s = float(
                sum(
                    max(
                        0.0,
                        min(interval.end, cutoff, float(eeg_times[-1]))
                        - max(interval.start, float(eeg_times[0])),
                    )
                    for interval in usable_page_intervals
                )
            )
            et_overlap_s = float(
                sum(
                    max(
                        0.0,
                        min(interval.end, cutoff, float(et_times[-1]))
                        - max(interval.start, float(et_times[0])),
                    )
                    for interval in usable_page_intervals
                )
            )

            records.append(
                {
                    "subject": subject,
                    "page": page,
                    "roi": roi,
                    "product_id": product_id,
                    "label_buy": label,
                    "click_count": len(product_clicks),
                    "first_click_time": first_click_time,
                    "predecision_cutoff_time": cutoff,
                    "considered": considered,
                    "n_gaze_visits": len(visits),
                    "total_dwell_s": total_dwell_s,
                    "first_gaze_time": first_gaze_time,
                    "last_gaze_time": last_gaze_time,
                    "first_gaze_to_click_s": latency_to_click_s,
                    "n_page_visits_before_cutoff": len(usable_page_intervals),
                    "page_exposure_s_before_cutoff": exposure_s,
                    "eeg_overlap_s": eeg_overlap_s,
                    "et_overlap_s": et_overlap_s,
                    "et_valid_fraction_subject": et_valid_fraction,
                    "eeg_effective_rate": eeg_rate,
                    "et_effective_rate": et_rate,
                    "raw_eeg_channels": int(eeg_values.shape[1]),
                    "first_page_time": (
                        min(interval.start for interval in page_intervals)
                        if page_intervals
                        else math.nan
                    ),
                    "last_page_time": (
                        max(interval.end for interval in page_intervals)
                        if page_intervals
                        else math.nan
                    ),
                    "eoe_time": eoe_time if eoe_time is not None else math.nan,
                }
            )

    frame = pd.DataFrame.from_records(records)
    primary = frame.loc[frame["considered"] == 1]
    primary_buys = int(primary["label_buy"].sum())
    primary_nobuys = int(len(primary) - primary_buys)
    unique_buys = int(frame["label_buy"].sum())
    multivisit_buys = int(
        ((frame["label_buy"] == 1) & (frame["n_gaze_visits"] >= 2)).sum()
    )

    summary: dict[str, Any] = {
        "subject": subject,
        "n_page_intervals": len(intervals),
        "n_unique_pages": len({interval.page for interval in intervals}),
        **click_counts,
        "unique_buy_products": unique_buys,
        "duplicate_product_presses": int(
            click_counts["mapped_product_presses"] - unique_buys
        ),
        "considered_products": int(frame["considered"].sum()),
        "considered_buy_products": primary_buys,
        "considered_nobuy_products": primary_nobuys,
        "multivisit_buy_products": multivisit_buys,
        "et_valid_fraction": et_valid_fraction,
        "et_effective_rate": et_rate,
        "eeg_effective_rate": eeg_rate,
        "raw_eeg_channels": int(eeg_values.shape[1]),
        "xdf_q76_nonnull_aux": _xlsx_q76_count(xlsx_path),
        "has_primary_both_classes": int(primary_buys > 0 and primary_nobuys > 0),
    }
    return records, summary, unmatched_clicks


def discover_subjects(root: Path) -> list[str]:
    subjects = sorted(
        path.stem
        for path in (root / "DataSource").glob("S*.xdf")
        if SUBJECT_RE.match(path.stem)
    )
    return subjects


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value)}")


def aggregate_summary(
    products: pd.DataFrame,
    subjects: pd.DataFrame,
    unmatched: pd.DataFrame,
    config: AuditConfig,
    failed_subjects: dict[str, str],
) -> tuple[dict[str, Any], list[str], str]:
    primary = products.loc[products["considered"] == 1].copy()
    total_buys = int(products["label_buy"].sum())
    primary_buys = int(primary["label_buy"].sum())
    primary_nobuys = int(len(primary) - primary_buys)
    subjects_both = int(subjects["has_primary_both_classes"].sum())
    positive_multivisit = int(
        ((products["label_buy"] == 1) & (products["n_gaze_visits"] >= 2)).sum()
    )
    in_page = int(subjects["in_page_presses"].sum())
    mapped = int(subjects["mapped_product_presses"].sum())
    mapping_rate = float(mapped / in_page) if in_page else 0.0

    checks = {
        "all_requested_subjects_processed": len(failed_subjects) == 0,
        "at_least_min_valid_subjects": len(subjects) >= config.min_valid_subjects,
        "at_least_min_usable_buys": primary_buys >= config.min_usable_buys,
        "at_least_min_positive_multivisit": (
            positive_multivisit >= config.min_positive_multivisit
        ),
        "at_least_min_subjects_with_both_classes": (
            subjects_both >= config.min_valid_subjects
        ),
        "zero_post_click_by_construction": True,
    }

    warnings: list[str] = []
    if mapping_rate < 0.95:
        warnings.append(
            "Fewer than 95% of in-page presses landed in product ROIs; inspect "
            "unmatched_clicks.csv because navigation/UI presses may be legitimate."
        )
    if primary_buys < config.min_usable_buys:
        warnings.append(
            f"Only {primary_buys} Buy products have valid pre-decision gaze; "
            "prefer a smaller regularized model."
        )
    if positive_multivisit < config.min_positive_multivisit:
        warnings.append(
            f"Only {positive_multivisit} Buy products have multiple gaze visits; "
            "the chronological hazard claim is not sufficiently supported."
        )
    if subjects_both < config.min_valid_subjects:
        warnings.append(
            f"Only {subjects_both} subjects have both classes in the considered set."
        )

    hazard_go = all(
        checks[key]
        for key in (
            "all_requested_subjects_processed",
            "at_least_min_valid_subjects",
            "at_least_min_usable_buys",
            "at_least_min_positive_multivisit",
            "at_least_min_subjects_with_both_classes",
        )
    )
    decision = "HAZARD_GO" if hazard_go else "MIL_FALLBACK_REVIEW"

    summary = {
        "decision": decision,
        "configuration": asdict(config),
        "n_subjects_processed": int(len(subjects)),
        "failed_subjects": failed_subjects,
        "n_subject_product_instances": int(len(products)),
        "n_all_product_buys": total_buys,
        "all_product_buy_prevalence": (
            float(total_buys / len(products)) if len(products) else 0.0
        ),
        "n_considered_instances": int(len(primary)),
        "n_considered_buys": primary_buys,
        "n_considered_nobuys": primary_nobuys,
        "considered_buy_prevalence": (
            float(primary_buys / len(primary)) if len(primary) else 0.0
        ),
        "n_positive_multivisit": positive_multivisit,
        "n_subjects_primary_both_classes": subjects_both,
        "n_raw_left_presses": int(subjects["raw_left_presses"].sum()),
        "n_in_page_presses": in_page,
        "n_mapped_product_presses": mapped,
        "in_page_press_to_product_roi_rate": mapping_rate,
        "n_unmatched_click_records": int(len(unmatched)),
        "checks": checks,
        "warnings": warnings,
    }
    return summary, warnings, decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NeuMa pre-click Buy/NoBuy subject-product events."
    )
    parser.add_argument("--root", type=Path, default=Path("/mnt/Neuma_Model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--min-fixation-ms", type=float, default=100.0)
    parser.add_argument("--merge-gap-ms", type=float, default=75.0)
    parser.add_argument("--pre-click-margin-ms", type=float, default=500.0)
    parser.add_argument("--max-mouse-staleness-s", type=float, default=10.0)
    parser.add_argument("--min-valid-subjects", type=int, default=35)
    parser.add_argument("--min-usable-buys", type=int, default=500)
    parser.add_argument("--min-positive-multivisit", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = AuditConfig(
        root=str(root),
        output=str(output),
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        min_fixation_ms=args.min_fixation_ms,
        merge_gap_ms=args.merge_gap_ms,
        pre_click_margin_ms=args.pre_click_margin_ms,
        max_mouse_staleness_s=args.max_mouse_staleness_s,
        min_valid_subjects=args.min_valid_subjects,
        min_usable_buys=args.min_usable_buys,
        min_positive_multivisit=args.min_positive_multivisit,
    )

    discovered = discover_subjects(root)
    subjects = args.subjects if args.subjects else discovered
    subjects = [subject.upper() for subject in subjects]
    invalid = [subject for subject in subjects if not SUBJECT_RE.match(subject)]
    if invalid:
        raise ValueError(f"Invalid subject identifiers: {invalid}")
    missing = [
        subject
        for subject in subjects
        if not (root / "DataSource" / f"{subject}.xdf").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing XDF files for: {missing}")

    rois, image_width, image_height = load_rois(root)
    print("=" * 76)
    print("NEUROCLICK PRE-DECISION BUY/NOBUY AUDIT")
    print("=" * 76)
    print(f"Root                 : {root}")
    print(f"Output               : {output}")
    print(f"Subjects             : {len(subjects)}")
    print(f"Workers              : {max(1, args.workers)}")
    print(f"Brochure pixels      : {image_width} x {image_height}")
    print(f"Screen pixels        : {config.screen_width} x {config.screen_height}")
    print(f"Pre-click margin     : {config.pre_click_margin_ms:.0f} ms")
    print(f"Minimum gaze visit   : {config.min_fixation_ms:.0f} ms")
    print("Labels               : XDF mouse press inside active product ROI")
    print("XLSX                 : auxiliary audit only; never used as a label")

    all_records: list[dict[str, Any]] = []
    subject_summaries: list[dict[str, Any]] = []
    all_unmatched: list[dict[str, Any]] = []
    failed_subjects: dict[str, str] = {}
    config_dict = asdict(config)

    workers = max(1, min(int(args.workers), len(subjects)))
    if workers == 1:
        iterator: Iterable[tuple[str, Any]] = (
            (
                subject,
                process_subject(
                    subject, config_dict, rois, image_width, image_height
                ),
            )
            for subject in subjects
        )
        for subject, result in iterator:
            records, summary, unmatched = result
            all_records.extend(records)
            subject_summaries.append(summary)
            all_unmatched.extend(unmatched)
            print(
                f"{subject}: Buy={summary['unique_buy_products']:3d} | "
                f"considered Buy={summary['considered_buy_products']:3d} | "
                f"considered NoBuy={summary['considered_nobuy_products']:3d} | "
                f"multi-visit Buy={summary['multivisit_buy_products']:3d}"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_subject,
                    subject,
                    config_dict,
                    rois,
                    image_width,
                    image_height,
                ): subject
                for subject in subjects
            }
            for future in as_completed(futures):
                subject = futures[future]
                try:
                    records, summary, unmatched = future.result()
                    all_records.extend(records)
                    subject_summaries.append(summary)
                    all_unmatched.extend(unmatched)
                    print(
                        f"{subject}: Buy={summary['unique_buy_products']:3d} | "
                        f"considered Buy={summary['considered_buy_products']:3d} | "
                        f"considered NoBuy={summary['considered_nobuy_products']:3d} | "
                        f"multi-visit Buy={summary['multivisit_buy_products']:3d}"
                    )
                except Exception as exc:
                    failed_subjects[subject] = f"{type(exc).__name__}: {exc}"
                    print(f"{subject}: FAILED: {failed_subjects[subject]}", file=sys.stderr)
                    traceback.print_exc()

    products = pd.DataFrame.from_records(all_records)
    subject_frame = pd.DataFrame.from_records(subject_summaries)
    unmatched_frame = pd.DataFrame.from_records(all_unmatched)

    if products.empty or subject_frame.empty:
        raise RuntimeError("Audit produced no usable subject-product records")

    products = products.sort_values(["subject", "product_id"]).reset_index(drop=True)
    subject_frame = subject_frame.sort_values("subject").reset_index(drop=True)
    if not unmatched_frame.empty:
        unmatched_frame = unmatched_frame.sort_values(
            ["subject", "click_time"]
        ).reset_index(drop=True)

    products.to_csv(output / "subject_product_audit.csv", index=False)
    subject_frame.to_csv(output / "subject_summary.csv", index=False)
    unmatched_frame.to_csv(output / "unmatched_clicks.csv", index=False)

    summary, warnings, decision = aggregate_summary(
        products=products,
        subjects=subject_frame,
        unmatched=unmatched_frame,
        config=config,
        failed_subjects=failed_subjects,
    )
    with (output / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_scalar)

    decision_lines = [
        decision,
        "",
        f"Subjects processed: {summary['n_subjects_processed']}",
        f"Subject-product instances: {summary['n_subject_product_instances']}",
        f"All-product Buy count: {summary['n_all_product_buys']}",
        f"Considered Buy count: {summary['n_considered_buys']}",
        f"Considered NoBuy count: {summary['n_considered_nobuys']}",
        f"Positive multi-visit cases: {summary['n_positive_multivisit']}",
        f"Subjects with both primary classes: "
        f"{summary['n_subjects_primary_both_classes']}",
        "",
        "Warnings:",
        *(f"- {warning}" for warning in warnings),
    ]
    (output / "FEASIBILITY_DECISION.txt").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print(f"DECISION             : {decision}")
    print(f"Subjects processed   : {summary['n_subjects_processed']}")
    print(f"Instances            : {summary['n_subject_product_instances']}")
    print(f"All-product Buy      : {summary['n_all_product_buys']}")
    print(f"Considered Buy       : {summary['n_considered_buys']}")
    print(f"Considered NoBuy     : {summary['n_considered_nobuys']}")
    print(f"Positive multi-visit : {summary['n_positive_multivisit']}")
    print(
        "Subjects both class : "
        f"{summary['n_subjects_primary_both_classes']}"
    )
    print(f"Product click rate   : {summary['in_page_press_to_product_roi_rate']:.3f}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"Saved outputs to     : {output}")
    print("=" * 76)

    return 2 if failed_subjects else 0


if __name__ == "__main__":
    raise SystemExit(main())
