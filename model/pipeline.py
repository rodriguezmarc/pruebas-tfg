from __future__ import annotations

from data.datasets.acdc.cache import export_preprocessed_dataset
from model.config import RunConfig
from model.dataset import load_prompt_image_rows
from model.generation import generate_all_batches, save_generated_results
from model.training import build_inference_pipeline, finetune_lora


def run_pipeline(config: RunConfig | None = None) -> int:
    """
    Main method of the 'model' module, orchestrates de execution of the entire pipeline.
    It includes the following: preprocessing, finetuning, inference and evaluation.
    """
    active_config = RunConfig() if config is None else config  # activate config
    print(f"Base model: {active_config.base_model_id}")
    print(
        f"Configured modality: {active_config.modality} "
        f"(MINIM branch {active_config.minim_unet_subfolder})"
    )

    # Step 1. Export cached dataset
    print("1/4 Exporting cached ACDC dataset...")
    export_preprocessed_dataset(split=active_config.train_split)

    # Step 2. Preprocess dataset
    print("2/4 Loading preprocessed dataset...")
    rows = load_prompt_image_rows(active_config.csv_path, active_config.images_root)
    if active_config.inference_limit is not None:
        rows = rows[: active_config.inference_limit]
    if not rows:
        raise ValueError("No prompts are available for generation.")
    print(f"Loaded {len(rows)} prompt-image pairs for LoRA fine-tuning and generation.")

    # Step 3. Fine-tune using LoRA adapter
    print("3/4 Fine-tuning LoRA adapter...")
    artifacts = finetune_lora(active_config, rows)

    # Step 4. Inference results
    print("4/4 Generating result batch...")
    pipe = build_inference_pipeline(active_config, artifacts)
    images = generate_all_batches(pipe, rows, active_config)
    save_generated_results(images, rows, active_config)

    # Step 5. Evaluation

    return len(images)
