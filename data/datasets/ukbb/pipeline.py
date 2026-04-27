"""
Definition:
Brief map of the UKBB export pipeline that saves images and builds CSV rows.
---
Results:
Provides image-export helpers, case discovery, prompt generation, and row construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import SimpleITK as sitk

from data.datasets.cardiac_export import save_mask_overlay, save_processed_image
from data.datasets.driver_contract import BaseCardiacDatasetDriver, DatasetDriver
from data.export.row_contract import DataRow
from data.datasets.ukbb import discover_cases
from data.datasets.ukbb.preprocess import preprocess
from data.prompts.prompt_contract import PromptCapabilities, PromptField, PromptPayload


class UKBBDatasetDriver(BaseCardiacDatasetDriver):
    """
    ########################################
    Definition:
    Dataset-driver implementation for UKBB preprocessing and export.
    ---
    Results:
    Exposes the driver contract used by the top-level data pipeline.
    ########################################
    """

    dataset_name = "ukbb"

    def discover_cases(self, data_path: Path) -> list[Path]:
        return discover_cases(data_path)

    def prompt_capabilities(self) -> PromptCapabilities:
        return {
            PromptField.MODALITY,
            PromptField.VIEW,
            PromptField.FRAME,
            PromptField.EF,
            PromptField.BMI,
            PromptField.GROUP,
            PromptField.SEX,
            PromptField.AGE,
        }

    def capabilities_for_metadata(self, metadata: dict) -> PromptCapabilities:
        capabilities: PromptCapabilities = {
            PromptField.MODALITY,
            PromptField.VIEW,
            PromptField.FRAME,
            PromptField.EF,
        }
        if "height" in metadata and "weight" in metadata:
            capabilities.add(PromptField.BMI)
        if "pathology" in metadata or "disease" in metadata:
            capabilities.add(PromptField.GROUP)
        if "sex" in metadata or "Sex" in metadata or "gender" in metadata or "Gender" in metadata:
            capabilities.add(PromptField.SEX)
        if "age" in metadata or "Age" in metadata:
            capabilities.add(PromptField.AGE)
        return capabilities

    def to_prompt_payload(self, metadata: dict, ef: float, modality: str) -> PromptPayload:
        from data.prompts.cardiac_prompt import build_cardiac_prompt_payload

        return build_cardiac_prompt_payload(
            metadata,
            ef,
            modality=modality,
            view="short-axis view",
            frame="end-systolic frame",
        )

    def preprocess_case(
        self,
        case_path: Path,
    ) -> tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float | bool]]:
        return preprocess(case_path)

    def save_case_outputs(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        metadata: dict[str, str | int | float | bool],
        images_root: Path,
    ) -> str:
        img_filename = f"ukbb/{metadata['pid']}_es_mid.png"
        save_processed_image(image, Path(images_root) / img_filename)
        save_mask_overlay(
            image,
            mask,
            Path(images_root) / "ukbb" / "masked" / f"{metadata['pid']}_es_mid.png",
        )
        return img_filename

    def build_rows(
        self,
        data_path: Path,
        images_root: Path,
        modality: str,
        preprocess_fn: Callable[[Path], tuple[sitk.Image, sitk.Image, float, dict[str, str | int | float | bool]]] = preprocess,
        prompt_fn: Callable[[dict, float, str], str] | None = None,
    ) -> list[DataRow]:
        if preprocess_fn is preprocess:
            rows = super().build_rows(data_path=data_path, images_root=images_root, modality=modality, prompt_fn=prompt_fn)
        else:
            rows: list[DataRow] = []
            for case_path in self.discover_cases(data_path):
                es_slice, mask_slice, ef, metadata = preprocess_fn(case_path)
                img_filename = self.save_case_outputs(es_slice, mask_slice, metadata, images_root)
                prompt_text = self.render_prompt(metadata, ef, modality) if prompt_fn is None else prompt_fn(metadata, ef, modality)
                rows.append(self.build_row(img_filename, metadata, prompt_text, modality))
        return rows


UKBB_DRIVER: DatasetDriver = UKBBDatasetDriver()
