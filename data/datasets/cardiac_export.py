"""
Definition:
Shared export helpers for cardiac dataset drivers.
---
Results:
Provides reusable image conversion and overlay-writing utilities for cardiac datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def image_to_uint8(image: sitk.Image) -> sitk.Image:
    """
    ########################################
    Definition:
    Convert a normalized scalar image into an 8-bit SimpleITK image.
    ---
    Params:
    image: Input image expected to contain values in the `[0, 1]` range.
    ---
    Results:
    Returns a uint8 image suitable for PNG export.
    ########################################
    """
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).round().astype(np.uint8)
    out = sitk.GetImageFromArray(image_np)
    out.CopyInformation(image)
    return out


def mask_overlay_to_uint8(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    ########################################
    Definition:
    Blend a binary or labeled mask on top of the grayscale image for inspection.
    ---
    Params:
    image: Base image to visualize.
    mask: Segmentation mask aligned with the image.
    ---
    Results:
    Returns an RGB uint8 overlay image.
    ########################################
    """
    image_np = sitk.GetArrayFromImage(image_to_uint8(image))
    mask_np = sitk.GetArrayFromImage(mask)

    rgb_np = np.stack([image_np, image_np, image_np], axis=-1)
    overlay_color = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.35

    mask_region = mask_np > 0
    rgb_np = rgb_np.astype(np.float32)
    rgb_np[mask_region] = (
        (1.0 - alpha) * rgb_np[mask_region] + alpha * overlay_color
    )

    out = sitk.GetImageFromArray(np.clip(rgb_np, 0, 255).astype(np.uint8), isVector=True)
    out.CopyInformation(image)
    return out


def save_processed_image(image: sitk.Image, output_path: Path) -> None:
    """
    ########################################
    Definition:
    Save a processed image slice to disk as an 8-bit image file.
    ---
    Params:
    image: Processed grayscale image to export.
    output_path: Destination file path.
    ---
    Results:
    Creates parent directories and writes the image.
    ########################################
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image_to_uint8(image), str(output_path))


def save_mask_overlay(image: sitk.Image, mask: sitk.Image, output_path: Path) -> None:
    """
    ########################################
    Definition:
    Save the inspection overlay that shows the mask on top of the processed image.
    ---
    Params:
    image: Processed base image.
    mask: Processed aligned mask.
    output_path: Destination file path for the overlay preview.
    ---
    Results:
    Creates parent directories and writes the RGB overlay image.
    ########################################
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_overlay_to_uint8(image, mask), str(output_path))
