"""
Definition:
Brief map of UKBB dataset loading, metadata parsing, and frame discovery.
---
Results:
Exports label mappings plus helpers to discover cases and load UKBB cine data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data import LV_LABEL, MYO_LABEL, RV_LABEL
from data.preprocess.image_utilities import select_frame
from data.preprocess.medical_utilities import compute_frame_volume

UKBB_LABEL_MAP = {1: LV_LABEL, 2: MYO_LABEL, 3: RV_LABEL}

_CINE_IMAGE_CANDIDATES = (
    "sa.nii.gz",
    "sa.nii",
    "sax.nii.gz",
    "sax.nii",
)
_CINE_MASK_CANDIDATES = (
    "seg_sa.nii.gz",
    "seg_sa.nii",
    "sa_seg.nii.gz",
    "sa_seg.nii",
    "seg_sax.nii.gz",
    "seg_sax.nii",
)
_ED_IMAGE_CANDIDATES = (
    "sa_ED.nii.gz",
    "sa_ED.nii",
    "sa_ed.nii.gz",
    "sa_ed.nii",
)
_ED_MASK_CANDIDATES = (
    "seg_sa_ED.nii.gz",
    "seg_sa_ED.nii",
    "seg_sa_ed.nii.gz",
    "seg_sa_ed.nii",
    "sa_ED_seg.nii.gz",
    "sa_ED_seg.nii",
)
_ES_IMAGE_CANDIDATES = (
    "sa_ES.nii.gz",
    "sa_ES.nii",
    "sa_es.nii.gz",
    "sa_es.nii",
)
_ES_MASK_CANDIDATES = (
    "seg_sa_ES.nii.gz",
    "seg_sa_ES.nii",
    "seg_sa_es.nii.gz",
    "seg_sa_es.nii",
    "sa_ES_seg.nii.gz",
    "sa_ES_seg.nii",
)
_METADATA_CANDIDATES = (
    "metadata.json",
    "info.json",
    "Info.json",
    "manifest.json",
    "metadata.csv",
    "metadata.txt",
    "metadata.cfg",
    "Info.txt",
    "Info.cfg",
)


def _read_nifti_image(image_path: Path, output_pixel_type: int | None = None) -> sitk.Image:
    """
    ########################################
    Definition:
    Read a NIfTI image from disk with an optional output pixel type override.
    ---
    Params:
    image_path: Input NIfTI file path.
    output_pixel_type: Optional SimpleITK pixel type override.
    ---
    Results:
    Returns the loaded SimpleITK image.
    ########################################
    """
    if output_pixel_type is None:
        return sitk.ReadImage(str(image_path))
    return sitk.ReadImage(str(image_path), outputPixelType=output_pixel_type)


def _find_first_existing(
    case_path: Path,
    candidates: tuple[str, ...],
    recursive: bool = True,
) -> Path | None:
    """
    ########################################
    Definition:
    Return the first path that matches any supported filename candidate.
    ---
    Params:
    case_path: Root directory for one UKBB case.
    candidates: Supported filenames to probe in order.
    ---
    Results:
    Returns the first matching path or `None` when nothing is found.
    ########################################
    """
    for candidate in candidates:
        direct_path = case_path / candidate
        if direct_path.exists():
            return direct_path

    if recursive:
        for candidate in candidates:
            matches = sorted(case_path.rglob(candidate))
            if matches:
                return matches[0]

    return None


def _read_metadata_file(metadata_path: Path) -> dict[str, str]:
    """
    ########################################
    Definition:
    Read a supported UKBB metadata sidecar file into a plain dictionary.
    ---
    Params:
    metadata_path: Metadata file path.
    ---
    Results:
    Returns the parsed key-value dictionary.
    ########################################
    """
    if metadata_path.suffix == ".json":
        with Path.open(metadata_path, encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in {metadata_path}.")
        return {str(key): str(value) for key, value in raw.items() if value is not None}

    if metadata_path.suffix == ".csv":
        with Path.open(metadata_path, encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            try:
                row = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Metadata CSV {metadata_path} is empty.") from exc
        return {str(key): str(value) for key, value in row.items() if value not in (None, "")}

    parsed: dict[str, str] = {}
    with Path.open(metadata_path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ": " in stripped:
                key, value = stripped.split(": ", maxsplit=1)
            elif "=" in stripped:
                key, value = stripped.split("=", maxsplit=1)
            else:
                continue
            parsed[key.strip()] = value.strip()
    return parsed


def _lookup_metadata_value(metadata: dict[str, str], *keys: str) -> str | None:
    """
    ########################################
    Definition:
    Retrieve the first available metadata value from a list of possible keys.
    ---
    Params:
    metadata: Raw metadata dictionary.
    keys: Candidate keys to probe.
    ---
    Results:
    Returns the first matched value or `None`.
    ########################################
    """
    for key in keys:
        if key in metadata and metadata[key] not in ("", "None", "null"):
            return metadata[key]
    return None


def _parse_optional_float(value: str | None) -> float | None:
    """
    ########################################
    Definition:
    Convert a string value into float when available.
    ---
    Params:
    value: Optional raw metadata string.
    ---
    Results:
    Returns the parsed float or `None`.
    ########################################
    """
    if value is None:
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    """
    ########################################
    Definition:
    Convert a string value into int when available.
    ---
    Params:
    value: Optional raw metadata string.
    ---
    Results:
    Returns the parsed integer or `None`.
    ########################################
    """
    if value is None:
        return None
    return int(float(value))


def load_metadata(case_path: Path) -> dict[str, str | int | float | bool]:
    """
    ########################################
    Definition:
    Parse optional UKBB sidecar metadata into canonical project fields.
    ---
    Params:
    case_path: Root directory for one UKBB case.
    ---
    Results:
    Returns a normalized metadata dictionary for the case.
    ########################################
    """
    raw_metadata: dict[str, str] = {}
    metadata_path = _find_first_existing(case_path, _METADATA_CANDIDATES)
    if metadata_path is not None:
        raw_metadata = _read_metadata_file(metadata_path)

    pid = _lookup_metadata_value(raw_metadata, "pid", "patient_id", "subject_id", "eid")
    pathology = _lookup_metadata_value(raw_metadata, "pathology", "Group", "diagnosis", "disease")
    height = _parse_optional_float(_lookup_metadata_value(raw_metadata, "height", "Height", "height_cm"))
    weight = _parse_optional_float(_lookup_metadata_value(raw_metadata, "weight", "Weight", "weight_kg"))
    ed_frame = _parse_optional_int(_lookup_metadata_value(raw_metadata, "ed_frame", "ED", "ed"))
    es_frame = _parse_optional_int(_lookup_metadata_value(raw_metadata, "es_frame", "ES", "es"))
    labels_are_canonical_raw = _lookup_metadata_value(
        raw_metadata,
        "labels_are_canonical",
        "canonical_labels",
        "label_schema",
    )

    metadata: dict[str, str | int | float | bool] = {
        "pid": pid or case_path.name,
    }

    if pathology is not None:
        metadata["pathology"] = pathology
    if height is not None:
        metadata["height"] = height
    if weight is not None:
        metadata["weight"] = weight
    if ed_frame is not None:
        metadata["ed_frame"] = ed_frame
    if es_frame is not None:
        metadata["es_frame"] = es_frame
    if labels_are_canonical_raw is not None:
        value = labels_are_canonical_raw.strip().lower()
        metadata["labels_are_canonical"] = value in {"1", "true", "yes", "canonical"}

    return metadata


def _resolve_case_paths(case_path: Path) -> dict[str, Path]:
    """
    ########################################
    Definition:
    Resolve the available cine-image and segmentation paths for one UKBB case.
    ---
    Params:
    case_path: Root directory for one UKBB case.
    ---
    Results:
    Returns a dictionary of resolved image paths.
    ########################################
    """
    image_path = _find_first_existing(case_path, _CINE_IMAGE_CANDIDATES)
    mask_path = _find_first_existing(case_path, _CINE_MASK_CANDIDATES)
    if image_path is not None and mask_path is not None:
        return {
            "image_path": image_path,
            "mask_path": mask_path,
        }

    ed_image_path = _find_first_existing(case_path, _ED_IMAGE_CANDIDATES)
    ed_mask_path = _find_first_existing(case_path, _ED_MASK_CANDIDATES)
    es_image_path = _find_first_existing(case_path, _ES_IMAGE_CANDIDATES)
    es_mask_path = _find_first_existing(case_path, _ES_MASK_CANDIDATES)
    if all(path is not None for path in (ed_image_path, ed_mask_path, es_image_path, es_mask_path)):
        return {
            "ed_image_path": ed_image_path,
            "ed_mask_path": ed_mask_path,
            "es_image_path": es_image_path,
            "es_mask_path": es_mask_path,
        }

    raise FileNotFoundError(
        f"Could not resolve supported UKBB cine image/mask files under {case_path}."
    )


def _has_direct_case_files(case_path: Path) -> bool:
    """
    ########################################
    Definition:
    Check whether a directory itself, not its children, directly contains a supported UKBB case.
    ---
    Params:
    case_path: Directory to inspect.
    ---
    Results:
    Returns `True` when the directory directly contains a supported file layout.
    ########################################
    """
    image_path = _find_first_existing(case_path, _CINE_IMAGE_CANDIDATES, recursive=False)
    mask_path = _find_first_existing(case_path, _CINE_MASK_CANDIDATES, recursive=False)
    if image_path is not None and mask_path is not None:
        return True

    ed_image_path = _find_first_existing(case_path, _ED_IMAGE_CANDIDATES, recursive=False)
    ed_mask_path = _find_first_existing(case_path, _ED_MASK_CANDIDATES, recursive=False)
    es_image_path = _find_first_existing(case_path, _ES_IMAGE_CANDIDATES, recursive=False)
    es_mask_path = _find_first_existing(case_path, _ES_MASK_CANDIDATES, recursive=False)
    return all(path is not None for path in (ed_image_path, ed_mask_path, es_image_path, es_mask_path))


def _standardize_mask_labels(
    mask: sitk.Image,
    metadata: dict[str, str | int | float | bool],
) -> sitk.Image:
    """
    ########################################
    Definition:
    Convert UKBB segmentation labels into the project-wide canonical ids.
    ---
    Params:
    mask: Input mask image.
    metadata: Case metadata that may disable remapping when labels are already canonical.
    ---
    Results:
    Returns the standardized mask.
    ########################################
    """
    if bool(metadata.get("labels_are_canonical", False)):
        return mask
    return sitk.ChangeLabel(mask, UKBB_LABEL_MAP)


def _select_ed_es_frames_from_mask(
    mask_4d: sitk.Image,
    metadata: dict[str, str | int | float | bool],
) -> tuple[int, int]:
    """
    ########################################
    Definition:
    Identify ED and ES frame indices using metadata hints or labeled-frame volumes.
    ---
    Params:
    mask_4d: Four-dimensional segmentation mask.
    metadata: Canonical case metadata.
    ---
    Results:
    Returns the `(ed_frame, es_frame)` pair.
    ########################################
    """
    if "ed_frame" in metadata and "es_frame" in metadata:
        return int(metadata["ed_frame"]), int(metadata["es_frame"])

    mask_np = sitk.GetArrayFromImage(mask_4d)
    nonzero_frames = np.any(mask_np > 0, axis=tuple(range(1, mask_np.ndim)))
    frame_indices = [idx for idx, has_label in enumerate(nonzero_frames) if has_label]

    if len(frame_indices) < 2:
        raise ValueError(
            "UKBB frame selection requires either explicit ED/ES metadata or at least two labeled frames."
        )

    frame_volumes: list[tuple[float, int]] = []
    for frame_idx in frame_indices:
        frame_mask = _standardize_mask_labels(select_frame(mask_4d, frame_idx), metadata)
        frame_volumes.append((compute_frame_volume(frame_mask), frame_idx))

    ed_frame = max(frame_volumes)[1]
    es_frame = min(frame_volumes)[1]
    return ed_frame, es_frame


def discover_cases(data_path: Path) -> list[Path]:
    """
    ########################################
    Definition:
    Discover UKBB case directories that contain supported cine and mask files.
    ---
    Params:
    data_path: Root directory that contains UKBB subject folders.
    ---
    Results:
    Returns a sorted list of case-directory paths.
    ########################################
    """
    candidate_dirs: list[Path] = []

    if data_path.is_dir() and _has_direct_case_files(data_path):
        candidate_dirs.append(data_path)

    for child in sorted(path for path in data_path.iterdir() if path.is_dir()):
        try:
            _resolve_case_paths(child)
        except FileNotFoundError:
            continue
        candidate_dirs.append(child)

    return sorted({path.resolve(): path for path in candidate_dirs}.values())


def load_data(
    case_path: Path,
) -> tuple[
    sitk.Image,
    sitk.Image,
    sitk.Image,
    sitk.Image,
    sitk.Image,
    dict[str, str | int | float | bool],
]:
    """
    ########################################
    Definition:
    Load one UKBB case and resolve the ED and ES cine frames and masks.
    ---
    Params:
    case_path: Root directory for one UKBB case.
    ---
    Results:
    Returns the source cine image, ED image, ED mask, ES image, ES mask, and metadata.
    ########################################
    """
    metadata = load_metadata(case_path)
    resolved_paths = _resolve_case_paths(case_path)

    if "image_path" in resolved_paths and "mask_path" in resolved_paths:
        image_4d = _read_nifti_image(resolved_paths["image_path"])
        mask_4d = _read_nifti_image(resolved_paths["mask_path"], output_pixel_type=sitk.sitkUInt8)

        if image_4d.GetDimension() != 4 or mask_4d.GetDimension() != 4:
            raise ValueError(
                "UKBB cine-mode loading expects both image and mask to be 4D volumes."
            )

        ed_frame, es_frame = _select_ed_es_frames_from_mask(mask_4d, metadata)
        ed_image = select_frame(image_4d, ed_frame)
        es_image = select_frame(image_4d, es_frame)
        ed_mask = _standardize_mask_labels(select_frame(mask_4d, ed_frame), metadata)
        es_mask = _standardize_mask_labels(select_frame(mask_4d, es_frame), metadata)

        metadata = dict(metadata)
        metadata["ed_frame"] = ed_frame
        metadata["es_frame"] = es_frame
        return image_4d, ed_image, ed_mask, es_image, es_mask, metadata

    ed_image = _read_nifti_image(resolved_paths["ed_image_path"])
    ed_mask = _standardize_mask_labels(
        _read_nifti_image(resolved_paths["ed_mask_path"], output_pixel_type=sitk.sitkUInt8),
        metadata,
    )
    es_image = _read_nifti_image(resolved_paths["es_image_path"])
    es_mask = _standardize_mask_labels(
        _read_nifti_image(resolved_paths["es_mask_path"], output_pixel_type=sitk.sitkUInt8),
        metadata,
    )

    image_4d = sitk.JoinSeries([ed_image, es_image])
    metadata = dict(metadata)
    metadata.setdefault("ed_frame", 0)
    metadata.setdefault("es_frame", 1)
    return image_4d, ed_image, ed_mask, es_image, es_mask, metadata
