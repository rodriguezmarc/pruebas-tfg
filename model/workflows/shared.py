from __future__ import annotations

from model.config import RunConfig
from model.datasets.prompt_image import load_prompt_image_rows
from model.preprocess.adapter import CardiacPreprocessingAdapter
from model.runtime.device import select_device_and_dtype


def print_runtime_summary(config: RunConfig) -> None:
    device, dtype = select_device_and_dtype(config)
    print(f"Base Stable Diffusion checkpoint: {config.base_model_id}")
    print(f"MINIM repository: {config.minim_repo_id}")
    print(
        f"Configured modality: {config.modality} "
        f"(MINIM branch {config.minim_unet_subfolder})"
    )
    print(f"Runtime device: {device} ({dtype})")
    if device.type != "cuda":
        print("Warning: running without CUDA will work, but fine-tuning and inference may be very slow.")


def prepare_training_rows(config: RunConfig) -> tuple[list, object]:
    # Step 1. Export cached dataset
    print("1/4 Exporting cached ACDC dataset...")
    adapter = CardiacPreprocessingAdapter(config)

    # Step 2. Preprocess dataset
    print("2/4 Loading preprocessed dataset...")
    rows = adapter.export_and_load_rows()
    if not rows:
        raise ValueError("No prompts are available for generation.")
    print(f"Loaded {len(rows)} prompt-image pairs for LoRA fine-tuning.")
    if len(rows) < 500:
        print(
            "Warning: the current ACDC export contains a small number of training samples "
            f"({len(rows)}). This is likely too low for robust new-modality adaptation."
        )

    inference_row = select_inference_row(rows, config)
    return rows, inference_row


def load_exported_rows(config: RunConfig) -> list:
    rows = load_prompt_image_rows(config.csv_path, config.images_root)
    if not rows:
        raise ValueError("No prompts are available for generation.")
    print(f"Loaded {len(rows)} exported prompt-image pairs.")
    return rows


def select_inference_row(rows: list, config: RunConfig):
    if config.inference_row_index < 0 or config.inference_row_index >= len(rows):
        raise IndexError(
            f"inference_row_index={config.inference_row_index} is out of range for {len(rows)} rows."
        )
    inference_row = rows[config.inference_row_index]
    print(f"Selected inference row: {inference_row.image_path.name}")
    return inference_row


def load_inference_row(config: RunConfig):
    rows = load_exported_rows(config)
    return select_inference_row(rows, config)
