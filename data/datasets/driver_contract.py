"""
Definition:
Canonical dataset-driver contracts for dataset preprocessing and export.
---
Results:
Provides structural typing plus a reusable cardiac base driver for dataset pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

import SimpleITK as sitk

from data.export.row_contract import DataRow
from data.prompts.cardiac_prompt import generate_prompt
from data.prompts.driver_prompt_contract import PromptMetadataAdapter
from data.prompts.prompt_contract import PromptCapabilities, PromptPayload

Row = dict[str, str]


class DatasetDriver(PromptMetadataAdapter, Protocol):
    """
    ########################################
    Definition:
    Describe the structural contract protocol implemented by dataset drivers.
    ---
    Results:
    Requires dataset identification plus row-building behavior for CSV export.
    ########################################
    """

    dataset_name: str

    def discover_cases(self, data_path: Path) -> list[Path]:
        """
        ########################################
        Definition:
        Discover the dataset-specific case identifiers or config paths to process.
        ---
        Results:
        Returns the ordered case list for one dataset root.
        ########################################
        """

    def build_rows(self, data_path: Path, images_root: Path, modality: str) -> list[Row | DataRow]:
        """
        ########################################
        Definition:
        Build canonical dataset rows for one dataset root.
        ---
        Results:
        Returns rows ready for normalization and CSV export.
        ########################################
        """


class BaseCardiacDatasetDriver(ABC):
    """
    ########################################
    Definition:
    Reusable base class for cardiac dataset drivers that share the same export flow.
    ---
    Results:
    Centralizes the common pipeline from case discovery to row construction.
    ########################################
    """

    dataset_name: str

    @abstractmethod
    def discover_cases(self, data_path: Path) -> list[Path]:
        """
        ########################################
        Definition:
        Discover the ordered cases to process for one dataset root.
        ########################################
        """

    @abstractmethod
    def prompt_capabilities(self) -> PromptCapabilities:
        """
        ########################################
        Definition:
        Declare the superset of prompt capabilities exposed by the driver.
        ########################################
        """

    def capabilities_for_metadata(self, metadata: dict) -> PromptCapabilities:
        """
        ########################################
        Definition:
        Resolve the concrete capability set available for one case payload.
        ---
        Results:
        Returns the effective capability set used when rendering the prompt.
        ########################################
        """
        return self.prompt_capabilities()

    @abstractmethod
    def to_prompt_payload(self, metadata: dict, ef: float, modality: str) -> PromptPayload:
        """
        ########################################
        Definition:
        Adapt dataset metadata into the canonical prompt payload.
        ########################################
        """

    @abstractmethod
    def preprocess_case(self, case_path: Path) -> tuple[sitk.Image, sitk.Image, float, dict]:
        """
        ########################################
        Definition:
        Preprocess one case and return the export-ready image, mask, EF, and metadata.
        ########################################
        """

    @abstractmethod
    def save_case_outputs(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        metadata: dict,
        images_root: Path,
    ) -> str:
        """
        ########################################
        Definition:
        Persist the processed image artifacts for one case and return the relative image path.
        ########################################
        """

    def render_prompt(self, metadata: dict, ef: float, modality: str) -> str:
        """
        ########################################
        Definition:
        Render one prompt from the canonical payload and effective capabilities.
        ########################################
        """
        payload = self.to_prompt_payload(metadata, ef, modality)
        return generate_prompt(payload, self.capabilities_for_metadata(metadata))

    def build_row(self, img_filename: str, metadata: dict, prompt_text: str, modality: str) -> DataRow:
        """
        ########################################
        Definition:
        Build the canonical data row for one processed case.
        ########################################
        """
        return DataRow(
            path=img_filename,
            text=prompt_text,
            modality=modality,
            patient_id=str(metadata["pid"]),
            dataset=self.dataset_name,
        )

    def build_rows(
        self,
        data_path: Path,
        images_root: Path,
        modality: str,
        prompt_fn=None,
    ) -> list[Row | DataRow]:
        """
        ########################################
        Definition:
        Execute the common cardiac dataset export flow across all discovered cases.
        ########################################
        """
        rows: list[Row | DataRow] = []
        case_paths = self.discover_cases(data_path)
        print(f"Found {len(case_paths)} {self.dataset_name.upper()} cases.")

        preprocessed_count = 0  # counter that specifies the successfully preprocessed cases
        prompt_count = 0        # counter that specifies the cases with successful prompt generation

        for case_path in case_paths:
        
            es_slice, mask_slice, ef, metadata = self.preprocess_case(case_path)                # preprocessing
            preprocessed_count += 1

            img_filename = self.save_case_outputs(es_slice, mask_slice, metadata, images_root)  # outputs are stored for further examination

            prompt_text = self.render_prompt(metadata, ef, modality)                            # prompt rendering generates the prompt from metadata
            prompt_count += 1

            rows.append(self.build_row(img_filename, metadata, prompt_text, modality))          # row appending for further csv generation

        print(f"Preprocessing completed for {preprocessed_count} cases.")
        print(f"Prompt generation completed for {prompt_count} cases.")
        print(f"All rows were built successfully.")
        return rows
