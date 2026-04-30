from __future__ import annotations

from model.config import RunConfig
from model.datasets.prompt_image import PromptImageRow
from model.evaluation.metrics import evaluate_generated_batch, save_metrics
from model.generation.batch import generate_inference_batch, save_generated_results
from model.training.lora import build_inference_pipeline_from_adapter
from model.workflows.shared import load_exported_rows, print_runtime_summary, select_inference_row


def _print_metric_summary(metrics: dict[str, object], metrics_path: str) -> None:
    print(f"Saved evaluation metrics to {metrics_path}.")
    print(
        "Evaluation summary: "
        f"FID={metrics['fid']:.4f}, "
        f"IS={metrics['is']:.4f}, "
        f"MS-SSIM={metrics['ms_ssim'] if metrics['ms_ssim'] is not None else 'unavailable'}"
    )
    warnings = metrics.get("warnings")
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")


def run_infer_from_row(
    config: RunConfig,
    inference_row: PromptImageRow,
    reference_rows: list[PromptImageRow],
) -> int:
    # Step 3. Inference results
    print("3/3 Generating conditioned inference batch...")
    pipe = build_inference_pipeline_from_adapter(config)
    images = generate_inference_batch(pipe, inference_row, config)
    save_generated_results(images, inference_row, config)
    metrics = evaluate_generated_batch(reference_rows, inference_row.image_path, images, config)
    save_metrics(metrics, config.metrics_path)
    _print_metric_summary(metrics, str(config.metrics_path))
    return len(images)


def run_infer(config: RunConfig | None = None) -> int:
    active_config = RunConfig() if config is None else config
    print_runtime_summary(active_config)
    reference_rows = load_exported_rows(active_config)
    inference_row = select_inference_row(reference_rows, active_config)
    return run_infer_from_row(active_config, inference_row, reference_rows)


def main() -> None:
    num = run_infer()
    print(f"Generated {num} images in results/generated.")


if __name__ == "__main__":
    main()
