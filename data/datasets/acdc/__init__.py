"""
Definition:
Brief map of ACDC dataset loading, metadata parsing, and NIfTI header sanitation.
---
Results:
Exports label mappings plus helpers to read metadata and image volumes safely.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from data import RV_LABEL, MYO_LABEL, LV_LABEL

ACDC_LABEL_MAP = {3: LV_LABEL, 2: MYO_LABEL, 1: RV_LABEL}
ACDC_SPACING = (1.0, 1.0, 10.0)


def _resolve_cache_root() -> Path:
    """Return the cache directory for sanitized NIfTI files."""
    override = os.environ.get("TFG_NIFTI_CACHE_DIR")
    if override:
        return Path(override)

    try:
        return Path(tempfile.gettempdir()) / "tfg_nifti_cache"
    except (FileNotFoundError, OSError):
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "outputs" / "tmp" / "tfg_nifti_cache"


SANITIZED_NIFTI_CACHE_DIR = _resolve_cache_root()


def _get_nifti_axis_scales(affine: np.ndarray) -> np.ndarray:
    """
    ########################################
    Definition:
    Compute the scale magnitude of each spatial axis in an affine matrix.
    ---
    Params:
    affine: Affine transformation matrix from a NIfTI header.
    ---
    Results:
    Returns one scale value per spatial axis.
    ########################################
    """
    return np.linalg.norm(affine[:3, :3], axis=0)


def _nifti_has_unexpected_sform(header: nib.nifti1.Nifti1Header) -> bool:
    """
    ########################################
    Definition:
    Detect whether a NIfTI header has inconsistent sform scaling metadata.
    ---
    Params:
    header: NIfTI header to inspect.
    ---
    Results:
    Returns `True` when the sform scale disagrees with the voxel spacing fields.
    ########################################
    """
    if int(header["sform_code"]) <= 0:
        return False

    expected_scales = np.asarray(header["pixdim"][1:4], dtype=np.float64)
    sform = header.get_sform()
    sform_scales = _get_nifti_axis_scales(sform)
    return not np.allclose(sform_scales, expected_scales, atol=1e-5)


def _sanitized_nifti_cache_path(image_path: Path) -> Path:
    """
    ########################################
    Definition:
    Build a deterministic cache path for a sanitized NIfTI file.
    ---
    Params:
    image_path: Original image path.
    ---
    Results:
    Returns the cache file path associated with the image version on disk.
    ---
    Other Information:
    The cache key includes path, modification time, and file size.
    ########################################
    """
    stat = image_path.stat()
    cache_key = f"{image_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return SANITIZED_NIFTI_CACHE_DIR / f"{image_path.stem}.{digest}.nii.gz"


def _ensure_nifti_header_consistency(image_path: Path) -> Path:
    """
    ########################################
    Definition:
    Rewrite malformed NIfTI affine metadata when needed before reading with SimpleITK.
    ---
    Params:
    image_path: Input NIfTI file path.
    ---
    Results:
    Returns either the original path or a sanitized cached path.
    ---
    Other Information:
    Only files with inconsistent sform metadata are rewritten.
    ########################################
    """
    nii = nib.load(str(image_path))
    header = nii.header
    if not _nifti_has_unexpected_sform(header):
        return image_path

    cache_path = _sanitized_nifti_cache_path(image_path)
    if cache_path.exists():
        return cache_path

    SANITIZED_NIFTI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fixed_header = header.copy()
    affine = fixed_header.get_base_affine()
    sanitized = nib.Nifti1Image(nii.dataobj, affine, header=fixed_header)
    sanitized.set_sform(affine, code=2)
    sanitized.set_qform(affine, code=2)
    nib.save(sanitized, str(cache_path))
    return cache_path


def _read_nifti_image(image_path: Path, output_pixel_type: int | None = None) -> sitk.Image:
    """
    ########################################
    Definition:
    Read a NIfTI image after applying header-consistency safeguards.
    ---
    Params:
    image_path: NIfTI file to load.
    output_pixel_type: Optional SimpleITK pixel type override.
    ---
    Results:
    Returns the loaded SimpleITK image.
    ########################################
    """
    safe_path = _ensure_nifti_header_consistency(image_path)
    if output_pixel_type is None:
        return sitk.ReadImage(str(safe_path))
    return sitk.ReadImage(str(safe_path), outputPixelType=output_pixel_type)

def load_metadata(config_path: Path) -> dict[str, str | int | float]:
    """
    ########################################
    Definition:
    Parse an ACDC patient configuration file into canonical metadata fields.
    ---
    Params:
    config_path: Path to the `.cfg` metadata file.
    ---
    Results:
    Returns a normalized metadata dictionary for the patient.
    ---
    Other Information:
    The patient id is inferred from the parent directory name.
    ########################################
    """
    with Path.open(config_path, encoding="utf-8") as file:
        lines = file.read().splitlines()

    d: dict[str, str | int | float] = {}  # will contain .cfg info
    for l in lines:
        k, v = l.split(": ")  # each line is divided into key and value
        d[k] = v

    # dict with all the config parameters,
    return {
        "pid": config_path.parent.name,  # patient id is the name of the parent folder
        "pathology": d["Group"],  
        "height": float(d["Height"]),  # in cm
        "weight": float(d["Weight"]),  # in kg
        "n_frames": int(d["NbFrame"]),
        "ed_frame": int(d["ED"]),
        "es_frame": int(d["ES"]),
    }

def load_data(
    config_path: Path
) -> tuple[
    sitk.Image,  # image_4d
    sitk.Image,  # ed_image
    sitk.Image,  # ed_label
    sitk.Image,  # es_image
    sitk.Image,  # es_label
    dict[str, str | int | float],  # metadata
]:
    """
    ########################################
    Definition:
    Load the 4D ACDC image plus ED and ES images and masks for one patient.
    ---
    Params:
    config_path: Path to the patient configuration file.
    ---
    Results:
    Returns the 4D image, ED image, ED mask, ES image, ES mask, and metadata.
    ---
    Other Information:
    Mask images are loaded as uint8 to preserve label semantics.
    ########################################
    """
    metadata = load_metadata(config_path)
    pid = str(metadata["pid"])  # patient id
    ed = int(metadata["ed_frame"])  # end-diastole frame, starting at
    es = int(metadata["es_frame"])  # end-systole frame

    base = config_path.parent  # base path is the parent folder of the config file

    PATHS = {  # paths to the images and labels
        "image_path": base / f"{pid}_4d.nii.gz",  # complete images
        "ed_image_path": base / f"{pid}_frame{ed:02d}.nii.gz",  # end-diastole image
        "ed_mask_path": base / f"{pid}_frame{ed:02d}_gt.nii.gz",  # end-diastole label
        "es_image_path": base / f"{pid}_frame{es:02d}.nii.gz",  # end-systole image
        "es_mask_path": base / f"{pid}_frame{es:02d}_gt.nii.gz",  # end-systole label
    }

    # load the images and labels using SimpleITK
    image_4d = _read_nifti_image(PATHS["image_path"])
    ed_image = _read_nifti_image(PATHS["ed_image_path"])
    ed_mask = _read_nifti_image(PATHS["ed_mask_path"], output_pixel_type=sitk.sitkUInt8)
    es_image = _read_nifti_image(PATHS["es_image_path"])
    es_mask = _read_nifti_image(PATHS["es_mask_path"], output_pixel_type=sitk.sitkUInt8)
    
    return image_4d, ed_image, ed_mask, es_image, es_mask, metadata
