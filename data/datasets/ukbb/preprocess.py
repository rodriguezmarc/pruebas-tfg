"""
Definition:
Brief map of the UKBB preprocessing routine from raw cine images to export-ready slices.
---
Results:
Provides the dataset-specific preprocessing entrypoint.
"""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from data.datasets.ukbb import load_data
from data.preprocess.image_utilities import (
    crop_around_center,
    extract_slice,
    get_lv_center,
    get_mask_crop_size,
    normalize,
    resample_slice,
    resize_slice,
    select_mask_slice,
)
from data.preprocess.medical_utilities import compute_ef


def preprocess(config_path: Path) -> tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float | bool]]:
    """
    ########################################
    Definition:
    Run the full UKBB preprocessing pipeline for one case directory.
    ---
    Params:
    config_path: Path to the UKBB case directory.
    ---
    Results:
    Returns the processed ES slice, processed mask slice, EF value, and metadata.
    ########################################
    """
    (
        image_4d,
        ed_image,
        ed_mask,
        es_image,
        es_mask,
        metadata,
    ) = load_data(config_path)

    slice_idx = select_mask_slice(es_mask)
    es_slice, mask_slice = extract_slice(es_image, es_mask, slice_idx)

    es_slice = resample_slice(es_slice)
    mask_slice = resample_slice(mask_slice, is_label=True)

    center = get_lv_center(mask_slice)
    crop_size = get_mask_crop_size(mask_slice)
    es_slice = crop_around_center(es_slice, center, size=crop_size)
    mask_slice = crop_around_center(mask_slice, center, size=crop_size)

    es_slice = resize_slice(es_slice)
    mask_slice = resize_slice(mask_slice, is_label=True)
    es_slice = normalize(es_slice)

    metadata = dict(metadata)
    metadata["slice_idx"] = slice_idx
    metadata["n_frames"] = image_4d.GetSize()[-1] if image_4d.GetDimension() == 4 else 2
    ef = compute_ef(ed_mask, es_mask)

    return es_slice, mask_slice, ef, metadata
