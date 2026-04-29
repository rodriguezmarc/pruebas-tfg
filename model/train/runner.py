from __future__ import annotations

from model.config import RunConfig
from model.training.lora import finetune_lora
from model.workflows.shared import prepare_training_rows, print_runtime_summary


def run_train(config: RunConfig | None = None):
    active_config = RunConfig() if config is None else config
    print_runtime_summary(active_config)
    rows, _ = prepare_training_rows(active_config)

    # Step 3. Fine-tune using LoRA adapter
    print("3/3 Fine-tuning LoRA adapter...")
    artifacts = finetune_lora(active_config, rows)
    return artifacts


def main() -> None:
    run_train()
    print("LoRA adapters saved in results/lora_adapter.")


if __name__ == "__main__":
    main()
