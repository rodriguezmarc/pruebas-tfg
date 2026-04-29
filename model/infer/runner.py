from __future__ import annotations

from model.config import RunConfig
from model.datasets.prompt_image import PromptImageRow
from model.generation.batch import generate_inference_batch, save_generated_results
from model.training.lora import build_inference_pipeline_from_adapter
from model.workflows.shared import prepare_training_rows, print_runtime_summary


def run_infer_from_row(config: RunConfig, inference_row: PromptImageRow) -> int:
    # Step 3. Inference results
    print("3/3 Generating conditioned inference batch...")
    pipe = build_inference_pipeline_from_adapter(config)
    images = generate_inference_batch(pipe, inference_row, config)
    save_generated_results(images, inference_row, config)
    return len(images)


def run_infer(config: RunConfig | None = None) -> int:
    active_config = RunConfig() if config is None else config
    print_runtime_summary(active_config)
    _, inference_row = prepare_training_rows(active_config)
    return run_infer_from_row(active_config, inference_row)


def main() -> None:
    num = run_infer()
    print(f"Generated {num} images in results/generated.")


if __name__ == "__main__":
    main()
