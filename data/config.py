"""
Shared dataset input and output path configuration.
---
+ added build_output_csv_path and build_internal_csv_path functions
"""

from pathlib import Path

DATASET_PATHS = {
    "acdc": Path("HF_CACHE"),
    "ukbb": Path("UKBB"),
}

CACHE_BACKED_DATASETS = frozenset({"acdc"})

OUTPUT_PATHS = {
    "images": Path("outputs/data/images"),
    "csv": Path("outputs/data/csv"),
    "internal": Path("outputs/data/internal"),
}

def _build_output_csv_path(csv_root: Path, dataset: str) -> Path:
    """
    ########################################
    Definition:
    Build the output CSV path for a dataset.
    ---
    Params:
    csv_root: Root directory for generated CSV files.
    dataset: Dataset identifier.
    ---
    Results:
    Returns the CSV file path that should be written.
    ########################################
    """
    return csv_root / f"{dataset}_minim.csv"


def _build_internal_csv_path(internal_root: Path, dataset: str) -> Path:
    """
    ########################################
    Definition:
    Build the internal canonical-row CSV path for a dataset.
    ---
    Params:
    internal_root: Root directory for canonical split-capable manifests.
    dataset: Dataset identifier.
    ---
    Results:
    Returns the CSV path used to persist the full internal row contract.
    ########################################
    """
    return internal_root / f"{dataset}_rows.csv"
