"""
Definition:
Utility helpers shared by cache-backed preprocessing and prompt rendering.
---
Results:
Provides stable formatting and normalization helpers for slice-position conditioning.
"""

from __future__ import annotations


def compute_normalized_slice_position(slice_idx: int, num_slices: int) -> float:
    """
    ########################################
    Definition:
    Normalize one slice index into a base-to-apex position in the `[0, 1]` range.
    ---
    Params:
    slice_idx: Selected slice index where `0` corresponds to the base.
    num_slices: Total number of slices in the volume.
    ---
    Results:
    Returns the normalized slice position.
    ########################################
    """
    if num_slices <= 1:
        return 0.0
    if slice_idx < 0 or slice_idx >= num_slices:
        raise ValueError(
            f"slice_idx must be in [0, {num_slices - 1}], got {slice_idx}."
        )
    return float(slice_idx) / float(num_slices - 1)


def format_slice_position(slice_position: float, precision: int = 2) -> str:
    """
    ########################################
    Definition:
    Format a normalized slice position into prompt-ready text.
    ---
    Params:
    slice_position: Normalized slice coordinate in the `[0, 1]` range.
    precision: Number of decimal places used in the formatted value.
    ---
    Results:
    Returns the prompt fragment describing the slice position.
    ########################################
    """
    return f"slice position {float(slice_position):.{precision}f}"


def format_bmi_group(bmi_group: str) -> str:
    """
    ########################################
    Definition:
    Render a BMI group label without duplicating the trailing `BMI` token.
    ---
    Params:
    bmi_group: Raw BMI-group label produced by dataset utilities.
    ---
    Results:
    Returns the prompt fragment describing the BMI group.
    ########################################
    """
    normalized = bmi_group.strip()
    if normalized.lower().endswith("bmi"):
        return normalized
    return f"{normalized} BMI"
