"""
Definition:
Cache-backed ACDC export pipeline that saves images and builds CSV rows.
---
Results:
Provides the dataset-driver implementation that reads ACDC directly from the
local Hugging Face cache instead of a manually unpacked raw dataset folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import SimpleITK as sitk

from data.datasets.acdc.cache import (
    build_rows as build_cached_rows,
    discover_patient_dirs,
    preprocess_patient,
    render_prompt,
)
from data.datasets.driver_contract import BaseCardiacDatasetDriver, DatasetDriver
from data.export.row_contract import DataRow
from data.prompts.prompt_contract import PromptCapabilities, PromptField, PromptPayload


class ACDCCacheDatasetDriver(BaseCardiacDatasetDriver):
    """
    ########################################
    Definition:
    Dataset-driver implementation for cache-backed ACDC preprocessing and export.
    ---
    Results:
    Exposes the driver contract used by the top-level data pipeline.
    ########################################
    """

    dataset_name = "acdc"

    def discover_cases(self, data_path: Path) -> list[Path]:
        del data_path
        return discover_patient_dirs(split="train")

    def prompt_capabilities(self) -> PromptCapabilities:
        return {
            PromptField.MODALITY,
            PromptField.VIEW,
            PromptField.FRAME,
            PromptField.BMI,
            PromptField.EF,
            PromptField.GROUP,
        }

    def to_prompt_payload(self, metadata: dict, ef: float, modality: str) -> PromptPayload:
        raise NotImplementedError(
            "Cache-backed ACDC prompt rendering is implemented directly in data.datasets.acdc.cache.render_prompt."
        )

    def render_prompt(self, metadata: dict, ef: float, modality: str) -> str:
        return render_prompt(metadata, ef, modality=modality)

    def preprocess_case(
        self,
        case_path: Path,
    ) -> tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float | None]]:
        return preprocess_patient(
            patient_id=case_path.name,
            split=case_path.parent.name,
        )

    def save_case_outputs(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        metadata: dict[str, str | int | float | None],
        images_root: Path,
    ) -> str:
        from data.datasets.acdc.cache import _save_case_outputs

        return _save_case_outputs(image, mask, metadata, images_root)

    def build_rows(
        self,
        data_path: Path,
        images_root: Path,
        modality: str,
        preprocess_fn: Callable[[Path], tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float | None]]] | None = None,
        prompt_fn: Callable[[dict, float, str], str] | None = None,
    ) -> list[dict[str, str]]:
        del data_path, preprocess_fn
        rows = build_cached_rows(
            images_root=images_root,
            split="train",
            modality=modality,
            prompt_fn=prompt_fn,
        )
        return [row.to_dict() if isinstance(row, DataRow) else row for row in rows]


ACDC_DRIVER: DatasetDriver = ACDCCacheDatasetDriver()
