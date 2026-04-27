"""
Definition:
Preprocess cached ACDC patient studies directly from the local Hugging Face Hub cache.
---
Results:
Provides patient discovery, raw case loading, preprocessing aligned with the
project ACDC pipeline, and helpers to expose the processed cases to Hugging Face
dataset workflows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm

from data.datasets.cardiac_export import save_mask_overlay, save_processed_image
from data.datasets.acdc import ACDC_LABEL_MAP, ACDC_SPACING, _read_nifti_image
from data.export.csv_utilities import validate_minim_csv, write_minim_csv
from data.export.row_contract import DataRow
from data.preprocess.dataset_utilities import compute_bmi_group, compute_disease_label
from data.preprocess.conditioning_utilities import (
    compute_normalized_slice_position,
    format_bmi_group,
    format_slice_position,
)
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
from data.preprocess.medical_utilities import classify_ef, compute_ef

DEFAULT_ACDC_REPO_ID = "msepulvedagodoy/acdc"
DEFAULT_IMAGES_ROOT = Path("outputs/data/images")
DEFAULT_OUTPUT_CSV = Path("outputs/data/csv/acdc_minim.csv")
DEFAULT_MODALITY = "Cardiac MRI"
_HF_CACHE_ENV_VARS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
_SPLIT_ALIASES = {
    "train": "training",
    "training": "training",
    "test": "testing",
    "testing": "testing",
}
_PATIENT_ID_KEYS = ("pid", "patient_id", "patient", "id")
_SPLIT_KEYS = ("split", "subset")

__all__ = [
    "ACDC_LABEL_MAP",
    "ACDC_SPACING",
    "DEFAULT_ACDC_REPO_ID",
    "build_hf_preprocess_transform",
    "build_rows",
    "discover_patient_dirs",
    "export_preprocessed_dataset",
    "iter_preprocessed_examples",
    "load_data",
    "load_metadata",
    "preprocess_patient",
    "render_prompt",
    "resolve_patient_dir",
]


def _resolve_hf_hub_cache_root() -> Path:
    """
    ########################################
    Definition:
    Resolve the local Hugging Face Hub cache directory.
    ---
    Results:
    Returns the cache root that stores dataset snapshots.
    ########################################
    """
    for env_var in _HF_CACHE_ENV_VARS:
        override = os.environ.get(env_var)
        if override:
            return Path(override).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _dataset_repo_cache_dir(repo_id: str, hub_cache_root: Path) -> Path:
    owner, name = repo_id.split("/", maxsplit=1)
    return hub_cache_root / f"datasets--{owner}--{name}"


def _normalize_split(split: str | None) -> str | None:
    if split is None:
        return None

    normalized = split.strip().lower()
    if normalized not in _SPLIT_ALIASES:
        raise ValueError(
            "split must be one of: 'train', 'training', 'test', 'testing'."
        )
    return _SPLIT_ALIASES[normalized]


def _resolve_snapshot_root(
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
) -> Path:
    """
    ########################################
    Definition:
    Resolve the active cached Hugging Face snapshot directory for the ACDC dataset.
    ---
    Params:
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    ---
    Results:
    Returns the snapshot directory that contains the cached raw files.
    ########################################
    """
    cache_root = _resolve_hf_hub_cache_root() if hub_cache_root is None else Path(hub_cache_root)
    repo_cache_dir = _dataset_repo_cache_dir(repo_id, cache_root)
    if not repo_cache_dir.exists():
        raise FileNotFoundError(
            f"Dataset cache not found for '{repo_id}' under '{repo_cache_dir}'."
        )

    refs_main = repo_cache_dir / "refs" / "main"
    snapshots_dir = repo_cache_dir / "snapshots"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_root = snapshots_dir / revision
        if snapshot_root.exists():
            return snapshot_root

    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"No cached snapshots were found for '{repo_id}' in '{repo_cache_dir}'."
        )

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(
            f"No cached snapshot directories were found for '{repo_id}' in '{snapshots_dir}'."
        )
    return max(snapshots, key=lambda path: path.stat().st_mtime_ns)


def discover_patient_dirs(
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
) -> list[Path]:
    """
    ########################################
    Definition:
    List cached patient directories from the ACDC Hugging Face snapshot.
    ---
    Params:
    split: Optional split filter. Accepted values are `training` and `testing`.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    ---
    Results:
    Returns the sorted patient directories found in cache.
    ########################################
    """
    snapshot_root = _resolve_snapshot_root(repo_id=repo_id, hub_cache_root=hub_cache_root)
    normalized_split = _normalize_split(split)
    split_names = (normalized_split,) if normalized_split is not None else ("training", "testing")

    patient_dirs: list[Path] = []
    for split_name in split_names:
        split_root = snapshot_root / split_name
        if not split_root.exists():
            continue
        patient_dirs.extend(path for path in split_root.glob("patient*") if path.is_dir())
    return sorted(patient_dirs, key=lambda path: (_patient_number(path.name), path.parent.name))


def resolve_patient_dir(
    patient_id: str,
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
) -> Path:
    """
    ########################################
    Definition:
    Resolve one cached ACDC patient directory by id.
    ---
    Params:
    patient_id: Patient folder name such as `patient001`.
    split: Optional split filter. Accepted values are `training` and `testing`.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    ---
    Results:
    Returns the patient directory path from the local cache.
    ########################################
    """
    normalized_id = patient_id.strip()
    if not normalized_id:
        raise ValueError("patient_id must be a non-empty string.")

    snapshot_root = _resolve_snapshot_root(repo_id=repo_id, hub_cache_root=hub_cache_root)
    normalized_split = _normalize_split(split)
    split_names = (normalized_split,) if normalized_split is not None else ("training", "testing")
    for split_name in split_names:
        candidate = snapshot_root / split_name / normalized_id
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Patient '{normalized_id}' was not found in cached split(s): {', '.join(split_names)}."
    )


def _extract_frame_numbers(patient_dir: Path) -> list[int]:
    frame_numbers: list[int] = []
    prefix = f"{patient_dir.name}_frame"
    suffix = "_gt.nii.gz"

    for mask_path in patient_dir.glob(f"{patient_dir.name}_frame*_gt.nii.gz"):
        name = mask_path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        frame_text = name[len(prefix):-len(suffix)]
        frame_numbers.append(int(frame_text))

    if len(frame_numbers) < 2:
        raise FileNotFoundError(
            f"Expected at least two labeled frames in '{patient_dir}', found {len(frame_numbers)}."
        )
    return sorted(frame_numbers)


def _patient_number(patient_id: str) -> int:
    digits = "".join(char for char in patient_id if char.isdigit())
    if not digits:
        raise ValueError(f"Patient id '{patient_id}' does not contain a numeric suffix.")
    return int(digits)


def _read_info_cfg(patient_dir: Path) -> dict[str, str] | None:
    info_path = patient_dir / "Info.cfg"
    if not info_path.exists():
        return None

    raw: dict[str, str] = {}
    with Path.open(info_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or ": " not in stripped:
                continue
            key, value = stripped.split(": ", maxsplit=1)
            raw[key.strip()] = value.strip()
    return raw


def load_metadata(
    patient_id: str,
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
) -> dict[str, str | int | float | None]:
    """
    ########################################
    Definition:
    Build canonical patient metadata from the cached Hugging Face ACDC snapshot.
    ---
    Params:
    patient_id: Patient folder name such as `patient001`.
    split: Optional split filter. Accepted values are `training` and `testing`.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    ---
    Results:
    Returns the normalized metadata dictionary for the patient.
    ---
    Other Information:
    The Hugging Face cached snapshot does not preserve the original `.cfg` files,
    so unavailable fields are returned as `None`.
    ########################################
    """
    patient_dir = resolve_patient_dir(
        patient_id=patient_id,
        split=split,
        repo_id=repo_id,
        hub_cache_root=hub_cache_root,
    )
    info_cfg = _read_info_cfg(patient_dir)
    if info_cfg is not None:
        return {
            "pid": patient_dir.name,
            "split": patient_dir.parent.name,
            "pathology": info_cfg.get("Group"),
            "height": float(info_cfg["Height"]) if "Height" in info_cfg else None,
            "weight": float(info_cfg["Weight"]) if "Weight" in info_cfg else None,
            "n_frames": int(info_cfg["NbFrame"]) if "NbFrame" in info_cfg else None,
            "ed_frame": int(info_cfg["ED"]) if "ED" in info_cfg else None,
            "es_frame": int(info_cfg["ES"]) if "ES" in info_cfg else None,
        }

    frame_numbers = _extract_frame_numbers(patient_dir)
    ed_frame = frame_numbers[0]
    es_frame = frame_numbers[1]
    image_4d = _read_nifti_image(patient_dir / f"{patient_dir.name}_4d.nii.gz")
    n_frames = int(image_4d.GetSize()[-1])

    return {
        "pid": patient_dir.name,
        "split": patient_dir.parent.name,
        "pathology": None,
        "height": None,
        "weight": None,
        "n_frames": n_frames,
        "ed_frame": ed_frame,
        "es_frame": es_frame,
    }


def load_data(
    patient_id: str,
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
) -> tuple[
    sitk.Image,
    sitk.Image,
    sitk.Image,
    sitk.Image,
    sitk.Image,
    dict[str, str | int | float | None],
]:
    """
    ########################################
    Definition:
    Load one ACDC patient directly from the local Hugging Face Hub cache.
    ---
    Params:
    patient_id: Patient folder name such as `patient001`.
    split: Optional split filter. Accepted values are `training` and `testing`.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    ---
    Results:
    Returns the 4D image, ED image, ED mask, ES image, ES mask, and metadata.
    ---
    Other Information:
    Mask images are loaded as uint8 to preserve label semantics.
    ########################################
    """
    metadata = load_metadata(
        patient_id=patient_id,
        split=split,
        repo_id=repo_id,
        hub_cache_root=hub_cache_root,
    )
    patient_dir = resolve_patient_dir(
        patient_id=patient_id,
        split=str(metadata["split"]),
        repo_id=repo_id,
        hub_cache_root=hub_cache_root,
    )
    pid = str(metadata["pid"])
    ed = int(metadata["ed_frame"])
    es = int(metadata["es_frame"])

    image_4d = _read_nifti_image(patient_dir / f"{pid}_4d.nii.gz")
    ed_image = _read_nifti_image(patient_dir / f"{pid}_frame{ed:02d}.nii.gz")
    ed_mask = _read_nifti_image(
        patient_dir / f"{pid}_frame{ed:02d}_gt.nii.gz",
        output_pixel_type=sitk.sitkUInt8,
    )
    es_image = _read_nifti_image(patient_dir / f"{pid}_frame{es:02d}.nii.gz")
    es_mask = _read_nifti_image(
        patient_dir / f"{pid}_frame{es:02d}_gt.nii.gz",
        output_pixel_type=sitk.sitkUInt8,
    )

    return image_4d, ed_image, ed_mask, es_image, es_mask, metadata


def preprocess_patient(
    patient_id: str,
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
) -> tuple[
    sitk.Image,
    sitk.Image,
    float,
    dict[str, str | int | float | None],
]:
    """
    ########################################
    Definition:
    Run the project ACDC preprocessing pipeline on one cached patient study.
    ---
    Params:
    patient_id: Patient folder name such as `patient001`.
    split: Optional split filter. Accepted values are `training` and `testing`.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    ---
    Results:
    Returns the processed ES slice, processed mask slice, EF value, and metadata.
    ---
    Other Information:
    This mirrors `data/datasets/acdc/preprocess.py`, but uses the Hugging Face
    cache-backed patient loader instead of `.cfg` files.
    ########################################
    """
    (
        image_4d,
        ed_image,
        ed_mask,
        es_image,
        es_mask,
        metadata,
    ) = load_data(
        patient_id=patient_id,
        split=split,
        repo_id=repo_id,
        hub_cache_root=hub_cache_root,
    )
    del image_4d, ed_image

    ed_mask = sitk.ChangeLabel(ed_mask, ACDC_LABEL_MAP)
    es_mask = sitk.ChangeLabel(es_mask, ACDC_LABEL_MAP)

    slice_idx = select_mask_slice(es_mask)
    num_slices = int(es_mask.GetSize()[-1])
    slice_position = compute_normalized_slice_position(slice_idx, num_slices)
    es_slice, mask_slice = extract_slice(es_image, es_mask, slice_idx)

    metadata = dict(metadata)
    metadata["slice_idx"] = int(slice_idx)
    metadata["num_slices"] = num_slices
    metadata["slice_position"] = slice_position

    es_slice = resample_slice(es_slice)
    mask_slice = resample_slice(mask_slice, is_label=True)

    center = get_lv_center(mask_slice)
    crop_size = get_mask_crop_size(mask_slice)
    es_slice = crop_around_center(es_slice, center, size=crop_size)
    mask_slice = crop_around_center(mask_slice, center, size=crop_size)

    es_slice = resize_slice(es_slice)
    mask_slice = resize_slice(mask_slice, is_label=True)

    es_slice = normalize(es_slice)

    ef = compute_ef(ed_mask, es_mask)

    return es_slice, mask_slice, ef, metadata


def _image_to_numpy(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    return np.asarray(array)


def render_prompt(
    metadata: dict[str, Any],
    ef: float,
    modality: str = "Cardiac MRI",
) -> str:
    """
    ########################################
    Definition:
    Render the cache-backed ACDC prompt text for one processed patient.
    ---
    Params:
    metadata: Cache-backed metadata dictionary for one patient.
    ef: Ejection fraction value for the processed case.
    modality: User-specified modality string.
    ---
    Results:
    Returns the final prompt text used in CSV export.
    ########################################
    """
    bmi_group = compute_bmi_group(metadata)
    ef_group = classify_ef(float(ef))
    pathology_label = compute_disease_label(metadata)
    slice_position_text = None
    if "slice_position" in metadata and metadata["slice_position"] is not None:
        slice_position_text = format_slice_position(float(metadata["slice_position"]))
    segments = [
        modality,
        "short-axis view",
        "end-systolic frame",
        slice_position_text,
        format_bmi_group(bmi_group),
        ef_group,
    ]
    if pathology_label:
        segments.append(pathology_label)
    return ", ".join(str(segment) for segment in segments if segment) + "."


def _save_case_outputs(
    image: sitk.Image,
    mask: sitk.Image,
    metadata: dict[str, Any],
    images_root: Path,
) -> str:
    """
    ########################################
    Definition:
    Save one processed ES slice plus its inspection overlay to disk.
    ---
    Params:
    image: Processed ES slice.
    mask: Processed aligned mask.
    metadata: Cache-backed metadata dictionary for one patient.
    images_root: Root directory for image export.
    ---
    Results:
    Returns the relative image path written to the CSV row.
    ########################################
    """
    img_filename = f"acdc/{metadata['pid']}_es_mid.png"
    save_processed_image(image, Path(images_root) / img_filename)
    save_mask_overlay(
        image,
        mask,
        Path(images_root) / "acdc" / "masked" / f"{metadata['pid']}_es_mid.png",
    )
    return img_filename


def _build_row(
    img_filename: str,
    metadata: dict[str, Any],
    prompt_text: str,
    modality: str,
) -> DataRow:
    return DataRow(
        path=img_filename,
        text=prompt_text,
        modality=modality,
        patient_id=str(metadata["pid"]),
        dataset="acdc",
    )


def _resolve_patient_ids_from_examples(examples: dict[str, Any]) -> list[str]:
    for key in _PATIENT_ID_KEYS:
        if key not in examples:
            continue
        values = examples[key]
        if isinstance(values, list):
            return [str(value) for value in values]
        return [str(values)]
    raise KeyError(
        f"Expected one of patient id keys {_PATIENT_ID_KEYS}, got {tuple(examples.keys())}."
    )


def _examples_are_batched(examples: dict[str, Any]) -> bool:
    for key in _PATIENT_ID_KEYS:
        if key in examples:
            return isinstance(examples[key], list)
    return False


def _resolve_splits_from_examples(
    examples: dict[str, Any],
    expected_size: int,
    default_split: str | None,
) -> list[str | None]:
    for key in _SPLIT_KEYS:
        if key not in examples:
            continue
        values = examples[key]
        if isinstance(values, list):
            return [None if value is None else str(value) for value in values]
        return [None if values is None else str(values)] * expected_size
    return [default_split] * expected_size


def iter_preprocessed_examples(
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
    include_arrays: bool = True,
) -> Iterator[dict[str, Any]]:
    """
    ########################################
    Definition:
    Yield every cached ACDC patient after running the project preprocessing pipeline.
    ---
    Params:
    split: Optional split filter. Accepted values are `training` and `testing`.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    include_arrays: Whether to include NumPy arrays for the processed image and mask.
    ---
    Results:
    Yields one dictionary per patient ready for conversion into a Hugging Face dataset.
    ########################################
    """
    for patient_dir in discover_patient_dirs(
        split=split,
        repo_id=repo_id,
        hub_cache_root=hub_cache_root,
    ):
        es_slice, mask_slice, ef, metadata = preprocess_patient(
            patient_id=patient_dir.name,
            split=patient_dir.parent.name,
            repo_id=repo_id,
            hub_cache_root=hub_cache_root,
        )

        example: dict[str, Any] = {
            "pid": metadata["pid"],
            "split": metadata["split"],
            "ef": float(ef),
            "text": render_prompt(metadata, ef),
            "metadata": metadata,
        }
        if include_arrays:
            example["image"] = _image_to_numpy(es_slice)
            example["mask"] = _image_to_numpy(mask_slice)
        yield example


def build_rows(
    images_root: Path,
    split: str | None = None,
    modality: str = "Cardiac MRI",
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
    prompt_fn: Callable[[dict[str, Any], float, str], str] | None = None,
) -> list[dict[str, str]]:
    """
    ########################################
    Definition:
    Preprocess all cached ACDC patients, export PNG artifacts, and build CSV rows.
    ---
    Params:
    images_root: Root directory where processed images will be written.
    split: Optional split filter. Accepted values are `training` and `testing`.
    modality: Modality label used for prompt generation and CSV export.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    prompt_fn: Optional custom prompt renderer.
    ---
    Results:
    Returns standardized CSV row dictionaries for the processed patients.
    ########################################
    """
    rows: list[dict[str, str]] = []
    patient_dirs = discover_patient_dirs(
        split=split,
        repo_id=repo_id,
        hub_cache_root=hub_cache_root,
    )

    for patient_dir in tqdm(patient_dirs, desc="Preprocessing ACDC", unit="patient"):
        es_slice, mask_slice, ef, metadata = preprocess_patient(
            patient_id=patient_dir.name,
            split=patient_dir.parent.name,
            repo_id=repo_id,
            hub_cache_root=hub_cache_root,
        )
        img_filename = _save_case_outputs(es_slice, mask_slice, metadata, images_root)
        prompt_text = render_prompt(metadata, ef, modality) if prompt_fn is None else prompt_fn(metadata, ef, modality)
        rows.append(_build_row(img_filename, metadata, prompt_text, modality).to_dict())

    return rows


def export_preprocessed_dataset(
    split: str | None = None,
    prompt_fn: Callable[[dict[str, Any], float, str], str] | None = None,
    validate: bool = True,
) -> list[dict[str, str]]:
    """
    ########################################
    Definition:
    Run the cache-backed ACDC preprocessing pipeline and export the MINIM CSV manifest.
    ---
    Params:
    split: Optional split filter. Accepted values are `training` and `testing`.
    prompt_fn: Optional custom prompt renderer.
    validate: Whether to validate the generated rows against the exported images.
    ---
    Results:
    Writes processed images and the CSV manifest, then returns the row dictionaries.
    ########################################
    """
    rows = build_rows(
        images_root=DEFAULT_IMAGES_ROOT,
        split=split,
        modality=DEFAULT_MODALITY,
        repo_id=DEFAULT_ACDC_REPO_ID,
        hub_cache_root=None,
        prompt_fn=prompt_fn,
    )
    if validate:
        validate_minim_csv(rows, DEFAULT_IMAGES_ROOT)
    write_minim_csv(rows, DEFAULT_OUTPUT_CSV)
    return rows


def build_hf_preprocess_transform(
    split: str | None = None,
    repo_id: str = DEFAULT_ACDC_REPO_ID,
    hub_cache_root: Path | None = None,
    image_key: str = "image",
    mask_key: str = "mask",
    text_key: str = "text",
    metadata_key: str = "metadata",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    ########################################
    Definition:
    Build a Hugging Face `with_transform` compatible callable for patient-level ACDC preprocessing.
    ---
    Params:
    split: Optional default split used when the examples do not provide one.
    repo_id: Hugging Face dataset repository id.
    hub_cache_root: Optional Hugging Face Hub cache root override.
    image_key: Output key for the processed ES image slice array.
    mask_key: Output key for the processed mask slice array.
    text_key: Output key for the rendered prompt text.
    metadata_key: Output key for the metadata dictionaries.
    ---
    Results:
    Returns a transform function that expects patient ids in the incoming examples.
    ---
    Other Information:
    The incoming dataset must be patient-level, not file-level. Each example should
    expose a patient id via one of: `pid`, `patient_id`, `patient`, or `id`.
    ########################################
    """
    normalized_default_split = _normalize_split(split)

    def transform(examples: dict[str, Any]) -> dict[str, Any]:
        is_batched = _examples_are_batched(examples)
        patient_ids = _resolve_patient_ids_from_examples(examples)
        splits = _resolve_splits_from_examples(
            examples,
            expected_size=len(patient_ids),
            default_split=normalized_default_split,
        )

        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        efs: list[float] = []
        texts: list[str] = []
        metadatas: list[dict[str, str | int | float | None]] = []

        for patient_id, patient_split in zip(patient_ids, splits, strict=True):
            es_slice, mask_slice, ef, metadata = preprocess_patient(
                patient_id=patient_id,
                split=patient_split,
                repo_id=repo_id,
                hub_cache_root=hub_cache_root,
            )
            images.append(_image_to_numpy(es_slice))
            masks.append(_image_to_numpy(mask_slice))
            efs.append(float(ef))
            texts.append(render_prompt(metadata, ef))
            metadatas.append(metadata)

        output = dict(examples)
        if is_batched:
            output[image_key] = images
            output[mask_key] = masks
            output["ef"] = efs
            output[text_key] = texts
            output[metadata_key] = metadatas
        else:
            output[image_key] = images[0]
            output[mask_key] = masks[0]
            output["ef"] = efs[0]
            output[text_key] = texts[0]
            output[metadata_key] = metadatas[0]
        return output

    return transform


def main() -> None:
    """
    ########################################
    Definition:
    Run the cache-backed ACDC preprocessing export with the default cached dataset.
    ---
    Results:
    Exports processed images and the CSV manifest using the single cached ACDC dataset.
    ########################################
    """
    rows = export_preprocessed_dataset(
        split=None,
        prompt_fn=None,
        validate=True,
    )
    print(f"Exported {len(rows)} rows to {DEFAULT_OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
