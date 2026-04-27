"""
Definition:
Brief map of MINIM CSV writing and row validation utilities.
---
Results:
Provides helpers to write manifests and validate exported image rows.

"""

from __future__ import annotations

import csv
from pathlib import Path

MINIM_COLUMNS = ("path", "text", "modality")
REQUIRED_ROW_COLUMNS = ("path", "text", "modality", "patient_id", "dataset")


def write_minim_csv(rows: list[dict[str, str]], output_csv_path: Path) -> None:
    """
    ########################################
    Definition:
    Write manifest rows using the MINIM column schema.
    ---
    Params:
    rows: List of standardized row dictionaries to serialize.
    output_csv_path: Destination path for the CSV file.
    ---
    Results:
    Creates parent directories when needed and writes the CSV to disk.
    ---
    Other Information:
    Extra row keys are ignored to preserve a stable export schema.
    ########################################
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MINIM_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV created at {output_csv_path}.")


def validate_minim_csv(rows: list[dict[str, str]], images_root: Path) -> None:
    """
    ########################################
    Definition:
    Validate that MINIM rows are structurally correct and point to real images.
    ---
    Params:
    rows: List of manifest rows to inspect.
    images_root: Base directory used to resolve each relative image path.
    ---
    Results:
    Raises a ValueError when required columns, values, uniqueness, or files are invalid.
    ---
    Other Information:
    The function returns `None` when validation succeeds.
    ########################################
    """
    print("Validating CSV contents...")
    seen_paths: set[str] = set()  # rows which are validated 
    for idx, row in enumerate(rows):
        missing_cols = [col for col in REQUIRED_ROW_COLUMNS if col not in row]
        if missing_cols:
            raise ValueError(f"Row {idx} missing columns: {missing_cols}")

        rel_path = row["path"].strip()
        text = row["text"].strip()
        modality = row["modality"].strip()
        patient_id = row["patient_id"].strip()
        dataset = row["dataset"].strip()

        if not rel_path:
            raise ValueError(f"Row {idx} has empty path.")
        if not text:
            raise ValueError(f"Row {idx} has empty text.")
        if not modality:
            raise ValueError(f"Row {idx} has empty modality.")
        if not patient_id:
            raise ValueError(f"Row {idx} has empty patient_id.")
        if not dataset:
            raise ValueError(f"Row {idx} has empty dataset.")
        if rel_path in seen_paths:
            raise ValueError(f"Duplicated path in CSV rows: {rel_path}")

        image_path = images_root / rel_path
        if not image_path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")
        seen_paths.add(rel_path)  # confirm validation on this row
    print("CSV validated successfully.")
