from __future__ import annotations

from model.config import RunConfig
from model.infer.runner import run_infer_from_row
from model.runtime.device import select_device_and_dtype
from model.training.lora import finetune_lora
from model.workflows.shared import prepare_training_rows, print_runtime_summary


def _resolve_config(config: RunConfig | None = None) -> RunConfig:
    return RunConfig() if config is None else config


def run_pipeline(config: RunConfig | None = None) -> int:
    """
    Main method of the 'model' module, orchestrates de execution of the entire pipeline.
    It includes the following: preprocessing, finetuning, inference and evaluation.
    """
    active_config = _resolve_config(config)  # activate config
    device, dtype = select_device_and_dtype(active_config)
    del device, dtype
    print_runtime_summary(active_config)
    rows, inference_row = prepare_training_rows(active_config)

    # Step 3. Fine-tune using LoRA adapter
    print("3/4 Fine-tuning LoRA adapter...")
    finetune_lora(active_config, rows)

    # Step 4. Inference results
    print("4/4 Running inference from saved LoRA adapters...")
    return run_infer_from_row(active_config, inference_row)
