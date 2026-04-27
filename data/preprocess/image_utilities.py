"""
Definition:
Brief map of image utilities for frame extraction, resampling, cropping, and normalization.
---
Results:
Provides reusable image-processing helpers for the data pipeline.
"""

import numpy as np
import SimpleITK as sitk

from data import LV_LABEL

DEFAULT_CROP_SIZE = 512
DEFAULT_MIN_CROP_SIZE = 128
DEFAULT_CROP_MARGIN = 1.5

# --- slice extraction utilities ----

def select_frame(volume_4d: sitk.Image, frame_idx: int) -> sitk.Image:
    """
    ########################################
    Definition:
    Extract one 3D frame from a 4D image volume.
    ---
    Params:
    volume_4d: Input 4D image with time as the last dimension.
    frame_idx: Time index to extract.
    ---
    Results:
    Returns the selected 3D frame as a SimpleITK image.
    ########################################
    """
    size = list(volume_4d.GetSize())
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size[:3] + [0])  # keep x, y, z dimensions, set t to 0
    extractor.SetIndex([0, 0, 0, frame_idx])  # start at the desired time frame
    return extractor.Execute(volume_4d)

def extract_slice(
    image_3d: sitk.Image,
    mask_3d: sitk.Image,
    frame_idx: int,
) -> tuple[sitk.Image, sitk.Image]:
    """
    ########################################
    Definition:
    Extract the same 2D slice index from a 3D image and its aligned mask.
    ---
    Params:
    image_3d: Input 3D image volume.
    mask_3d: Input 3D segmentation mask aligned with the image.
    frame_idx: Slice index along the z axis.
    ---
    Results:
    Returns the image slice and mask slice pair.
    ########################################
    """
    size = list(image_3d.GetSize())
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size[:2] + [0])  # keep x, y dimensions, set z to 0
    extractor.SetIndex([0, 0, frame_idx])  # start at the desired frame index and time frame (0)
    return extractor.Execute(image_3d), extractor.Execute(mask_3d)


def select_mask_slice(mask_3d: sitk.Image, label: int = LV_LABEL) -> int:
    """
    ########################################
    Definition:
    Select the slice index with the largest labeled cardiac footprint.
    ---
    Params:
    mask_3d: Three-dimensional mask volume.
    label: Preferred label used to choose the slice.
    ---
    Results:
    Returns the selected slice index.
    ---
    Other Information:
    Falls back to any non-zero cardiac label if the preferred label is absent.
    ########################################
    """
    mask_np = sitk.GetArrayFromImage(mask_3d)

    label_region = mask_np == label
    if np.any(label_region):
        areas = label_region.sum(axis=(1, 2))
    else:
        cardiac_region = mask_np > 0
        if not np.any(cardiac_region):
            raise ValueError("No cardiac mask found in the 3D volume.")
        areas = cardiac_region.sum(axis=(1, 2))

    return int(np.argmax(areas))

# --- resampling utilities ---

def resample_slice(
    image: sitk.Image,
    target_spacing: tuple[float, float] = (1.0, 1.0),
    is_label: bool = False,
) -> sitk.Image:
    """
    ########################################
    Definition:
    Resample a 2D image slice to a target physical spacing.
    ---
    Params:
    image: Input 2D image.
    target_spacing: Desired output spacing in millimetres.
    is_label: Whether the image should be treated as a label map.
    ---
    Results:
    Returns the resampled image.
    ---
    Other Information:
    Label maps use nearest-neighbor interpolation while intensity images use linear interpolation.
    ########################################
    """
    original_spacing = image.GetSpacing() 
    original_size = image.GetSize()

    new_size = [int(round(original_size[i] * (original_spacing[i] / target_spacing[i]))) for i in range(2)] 

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    if is_label:
        # If the image is a mask, use nearest neighbor interpolation to preserve label values.
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # If the image is data, use linear interpolation for better quality.
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def resize_slice(
    image: sitk.Image,
    target_size: tuple[int, int] = (DEFAULT_CROP_SIZE, DEFAULT_CROP_SIZE),
    is_label: bool = False,
) -> sitk.Image:
    """
    ########################################
    Definition:
    Resize a 2D image to a fixed output canvas.
    ---
    Params:
    image: Input 2D image.
    target_size: Desired output size in pixels.
    is_label: Whether the image should be treated as a label map.
    ---
    Results:
    Returns the resized image.
    ---
    Other Information:
    The output spacing is adjusted so the physical extent remains consistent.
    ########################################
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    output_spacing = [
        (original_spacing[i] * original_size[i]) / target_size[i]
        for i in range(2)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(list(target_size))
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


# --- cropping utilities ---

def get_lv_center(mask_2d: sitk.Image) -> tuple[int, int]:
    """
    ########################################
    Definition:
    Compute the center of mass of the left-ventricle label in a 2D mask.
    ---
    Params:
    mask_2d: Two-dimensional mask containing the LV label.
    ---
    Results:
    Returns integer `(x, y)` center coordinates.
    ---
    Other Information:
    Raises ValueError when the LV label is missing.
    ########################################
    """
    mask_np = sitk.GetArrayFromImage(mask_2d)  # convert to numpy array
    coords = np.argwhere(mask_np == LV_LABEL)  # get coordinates of LV pixels

    if len(coords) == 0:
        raise ValueError("LV not found in mask.")
    
    y_mean, x_mean = coords.mean(axis=0)  # compute mean coordinates (center of mass)
    return int(x_mean), int(y_mean)


def get_mask_crop_size(
    mask_2d: sitk.Image,
    min_size: int = DEFAULT_MIN_CROP_SIZE,
    max_size: int = DEFAULT_CROP_SIZE,
    margin: float = DEFAULT_CROP_MARGIN,
) -> int:
    """
    ########################################
    Definition:
    Derive a square crop size from the occupied cardiac mask area.
    ---
    Params:
    mask_2d: Two-dimensional mask image.
    min_size: Lower crop-size bound.
    max_size: Upper crop-size bound.
    margin: Multiplicative margin applied around the mask footprint.
    ---
    Results:
    Returns the crop size in pixels.
    ---
    Other Information:
    Raises ValueError when the slice contains no cardiac labels.
    ########################################
    """
    mask_np = sitk.GetArrayFromImage(mask_2d)
    coords = np.argwhere(mask_np > 0)

    if len(coords) == 0:
        raise ValueError("No cardiac mask found in the 2D slice.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width = (x_max - x_min) + 1
    height = (y_max - y_min) + 1
    crop_size = int(np.ceil(max(width, height) * margin))

    return max(min_size, min(crop_size, max_size))

def crop_around_center(
    image: sitk.Image,
    center: tuple[int, int],
    size: int = DEFAULT_CROP_SIZE,
) -> sitk.Image:
    """
    ########################################
    Definition:
    Crop a square region around a given center and pad outside areas when required.
    ---
    Params:
    image: Input 2D image.
    center: Crop center in pixel coordinates.
    size: Desired square crop size.
    ---
    Results:
    Returns a cropped image of the requested size.
    ########################################
    """
    width, height = image.GetSize()
    cx, cy = center
    half = size // 2

    raw_start_x = cx - half
    raw_start_y = cy - half
    raw_end_x = raw_start_x + size
    raw_end_y = raw_start_y + size

    start_x = max(raw_start_x, 0)
    start_y = max(raw_start_y, 0)
    end_x = min(raw_end_x, width)
    end_y = min(raw_end_y, height)

    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetIndex([start_x, start_y])
    extractor.SetSize([end_x - start_x, end_y - start_y])
    cropped = extractor.Execute(image)

    return sitk.ConstantPad(
        cropped,
        [max(0, -raw_start_x), max(0, -raw_start_y)],
        [max(0, raw_end_x - width), max(0, raw_end_y - height)],
        0,
    )

# --- normalization utilities ---

def normalize(image: sitk.Image) -> sitk.Image:
    """
    ########################################
    Definition:
    Normalize image intensities to the `[0, 1]` range.
    ---
    Params:
    image: Input image to normalize.
    ---
    Results:
    Returns the normalized image.
    ---
    Other Information:
    Constant images are converted to all zeros.
    ########################################
    """
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)
    
    min_val = image_np.min()
    max_val = image_np.max()

    if max_val - min_val > 0:
        # Scale to [0, 1] when there is variation in intensities.
        image_np = (image_np - min_val) / (max_val - min_val)
    else:
        # For constant images, return zeros.
        image_np[:] = 0.0

    normalized = sitk.GetImageFromArray(image_np)
    normalized.CopyInformation(image)
    return normalized
