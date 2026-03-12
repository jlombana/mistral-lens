"""Metrics engine — compares extractions against ground truth and computes accuracy.

This module is independent of the extractor module.
It accepts generic dictionaries and computes field-level accuracy metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.utils import timestamp_now

logger = logging.getLogger(__name__)


class FieldMetrics(BaseModel):
    """Accuracy metrics for a single field."""

    field: str
    accuracy: float
    total: int
    correct: int


class MetricsReport(BaseModel):
    """Complete metrics report for an evaluation run."""

    timestamp: str
    model_version: str
    total_images: int
    fields: list[FieldMetrics]
    overall_accuracy: float


def _normalize_value(value: Any) -> str | None:
    """Normalize a value for comparison (lowercase, strip whitespace).

    Args:
        value: Raw value from extraction or ground truth.

    Returns:
        Normalized string or None if value is empty.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized else None


def compute_field_accuracy(
    extractions: list[dict],
    ground_truth: list[dict],
    field: str,
) -> FieldMetrics:
    """Compute accuracy for a single field across all samples.

    Args:
        extractions: List of extraction dicts (must have 'image_id' key).
        ground_truth: List of ground truth dicts (must have 'image_id' key).
        field: Field name to evaluate.

    Returns:
        FieldMetrics with accuracy stats for this field.
    """
    gt_by_id = {item["image_id"]: item for item in ground_truth}
    total = 0
    correct = 0

    for extraction in extractions:
        image_id = extraction.get("image_id")
        gt_item = gt_by_id.get(image_id)
        if gt_item is None:
            continue

        extracted_val = _normalize_value(extraction.get(field))
        gt_val = _normalize_value(gt_item.get(field))

        if gt_val is None:
            continue  # Skip if no ground truth for this field

        total += 1
        if extracted_val == gt_val:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return FieldMetrics(
        field=field,
        accuracy=accuracy,
        total=total,
        correct=correct,
    )


def compute_metrics(
    extractions: list[dict],
    ground_truth: list[dict],
    fields: list[str],
    model_version: str = "unknown",
) -> MetricsReport:
    """Compute accuracy metrics for all specified fields.

    Args:
        extractions: List of extraction dicts (keyed by image_id).
        ground_truth: List of ground truth dicts (keyed by image_id).
        fields: List of field names to evaluate.
        model_version: Model version string for the report.

    Returns:
        MetricsReport with per-field and overall accuracy.
    """
    field_metrics = [
        compute_field_accuracy(extractions, ground_truth, field)
        for field in fields
    ]

    total_correct = sum(fm.correct for fm in field_metrics)
    total_comparisons = sum(fm.total for fm in field_metrics)
    overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0

    report = MetricsReport(
        timestamp=timestamp_now(),
        model_version=model_version,
        total_images=len(extractions),
        fields=field_metrics,
        overall_accuracy=overall_accuracy,
    )

    logger.info(
        "Metrics computed: overall_accuracy=%.2f%% (%d/%d)",
        overall_accuracy * 100,
        total_correct,
        total_comparisons,
    )

    return report
