"""
Definition:
Brief map of the top-level dataset pipeline orchestration.
---
Results:
Connects dataset drivers, validation, and CSV writing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from data.config import CACHE_BACKED_DATASETS, DATASET_PATHS, OUTPUT_PATHS, _build_internal_csv_path, _build_output_csv_path
from data.datasets.driver_contract import DatasetDriver
from data.datasets.acdc.pipeline import ACDC_DRIVER
from data.datasets.ukbb.pipeline import UKBB_DRIVER
from data.export.csv_utilities import validate_minim_csv, write_minim_csv
from data.export.row_contract import DataRow

Row = dict[str, str]
DATASET_DRIVERS: dict[str, DatasetDriver] = {
    "acdc": ACDC_DRIVER,
    "ukbb": UKBB_DRIVER,
}

def _normalize_rows(rows: list[Row | DataRow]) -> list[Row]:
    """
    ########################################
    Definition:
    Normalize driver outputs into the canonical plain-dictionary row contract.
    ---
    Params:
    rows: Driver outputs as dictionaries or `DataRow` instances.
    ---
    Results:
    Returns a list of dictionaries with canonical row fields.
    ########################################
    """
    normalized_rows: list[Row] = []
    for row in rows:
        if isinstance(row, DataRow):  # if the row is a DataRow, convert it to a dictionary
            normalized_rows.append(row.to_dict())
        else:
            normalized_rows.append(row)  # if the row is already a dictionary, add it to the list
    return normalized_rows


def write_internal_rows(rows: list[Row], output_csv_path: Path) -> None:
    """
    ########################################
    Definition:
    Persist the full canonical row contract required for downstream splits.
    ---
    Params:
    rows: Canonical rows including internal-only fields.
    output_csv_path: Destination path for the internal CSV.
    ---
    Results:
    Writes a CSV with all canonical fields.
    ########################################
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "text", "modality", "patient_id", "dataset"])
        writer.writeheader()
        writer.writerows(rows)


def load_internal_rows(internal_root: Path, dataset: str) -> list[Row]:
    """
    ########################################
    Definition:
    Load the persisted canonical row contract for one dataset.
    ---
    Params:
    internal_root: Root directory containing canonical row CSVs.
    dataset: Dataset identifier.
    ---
    Results:
    Returns the canonical rows required for split generation.
    ########################################
    """
    internal_csv_path = _build_internal_csv_path(internal_root, dataset)
    if not internal_csv_path.exists():
        raise FileNotFoundError(
            f"Missing internal manifest for dataset '{dataset}' at {internal_csv_path}. "
            f"Run `python prepare -d {dataset}` first."  # in case the load was performed from `python evaluate -d {dataset}`
        )
    with Path.open(internal_csv_path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_csv_pipeline(
    data_path: Path,                    # dataset driver path where training data is stored
    images_root: Path,                  # path where images will be stored (output)
    csv_root: Path,                     # path where csv manifest will be stored (output)
    internal_root: Path | None = None,  # path where internal csv will be stored (default to None, as decision is delegated to driver)
    dataset: str = "acdc",              # dataset identifier (default at acdc)
    modality: str = "Cardiac MRI",      # modality identifier (always Cardiac MRI)
) -> list[Row]:
    """
    ########################################
    Definition:
    Execute the full dataset-to-CSV export workflow.
    ---
    Params:
    data_path: Dataset input root.
    images_root: Directory where processed images are stored.
    csv_root: Directory where CSV manifests are stored.
    internal_root: Directory where internal CSVs are stored.
    dataset: Dataset identifier used to select the driver.
    modality: Modality label written into exported rows.
    Results:
    Returns the list of generated rows after validation and CSV writing.
    ---
    Other Information:
    Raises ValueError when the requested dataset driver is not registered.
    ########################################
    """
    try:
        dataset_driver = DATASET_DRIVERS[dataset]  # firstly read the dataset driver
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'.") from exc

    if dataset not in CACHE_BACKED_DATASETS and not data_path.exists():  # then check if dataset path is correct
        raise FileNotFoundError(
            f"Dataset root for '{dataset}' does not exist at {data_path}."
        )
    if dataset not in CACHE_BACKED_DATASETS and not data_path.is_dir():  # check that dataset path corresponds to a directory
        raise NotADirectoryError(
            f"Dataset root for '{dataset}' is not a directory: {data_path}."
        )

    output_csv_path = _build_output_csv_path(csv_root, dataset)

    if internal_root is None:
        internal_root = OUTPUT_PATHS["internal"]  # if not spcified, collect it from config.py
    internal_csv_path = _build_internal_csv_path(internal_root, dataset)

    print(f"Starting {dataset.upper()} preprocessing...")
    if dataset in CACHE_BACKED_DATASETS:
        print(f"Reading {dataset.upper()} directly from the local Hugging Face cache.")
    else:
        print(f"Reading data from {data_path}.")

    rows = _normalize_rows(  # normalize to dict form once build by the driver
        dataset_driver.build_rows(
            data_path=data_path,
            images_root=images_root,
            modality=modality,
        )
    )

    if not rows:  # in case no rows were generated
        raise ValueError(
            f"No rows were generated for dataset '{dataset}' from {data_path}. "
        )

    print(f"Prepared {len(rows)} rows. Writing outputs to {images_root}.")
    validate_minim_csv(rows, images_root)               # validation
    write_minim_csv(rows, output_csv_path)              # csv manifest write 
    write_internal_rows(rows, internal_csv_path)        # csv internal write
    print(f"{dataset.upper()} export completed successfully.")
    return rows  # end of the data pipeline
