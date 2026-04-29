from __future__ import annotations

from pathlib import Path

from PIL import Image

from model.config import RunConfig
from model.datasets.prompt_image import PromptImageRow
from model.pipelines.minim import CardiacMINIMPipeline
from model.runtime.device import build_generators


def image_grid(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    width, height = images[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(image, box=(index % cols * width, index // cols * height))
    return grid


def generate_batch(
    pipe: CardiacMINIMPipeline,
    prompt: str,
    config: RunConfig,
    seed_offset: int = 0,
) -> list[Image.Image]:
    prompts = [prompt] * config.batch_size
    generator = build_generators(pipe.device, config.seed + seed_offset, config.batch_size)

    result = pipe(
        prompt=prompts,
        negative_prompt=[config.negative_prompt] * config.batch_size,
        generator=generator,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
    )
    return result.images


def _comparison_canvas(source: Image.Image, generated: Image.Image) -> Image.Image:
    source_rgb = source.convert("RGB")
    generated_rgb = generated.convert("RGB")
    width = source_rgb.width + generated_rgb.width
    height = max(source_rgb.height, generated_rgb.height)
    canvas = Image.new("RGB", size=(width, height))
    canvas.paste(source_rgb, box=(0, 0))
    canvas.paste(generated_rgb, box=(source_rgb.width, 0))
    return canvas


def generate_inference_batch(
    pipe: CardiacMINIMPipeline,
    row: PromptImageRow,
    config: RunConfig,
) -> list[Image.Image]:
    return generate_batch(pipe, row.prompt, config)


def save_generated_results(
    images: list[Image.Image],
    row: PromptImageRow,
    config: RunConfig,
) -> None:
    config.generated_dir.mkdir(parents=True, exist_ok=True)
    config.comparisons_dir.mkdir(parents=True, exist_ok=True)
    config.results_root.mkdir(parents=True, exist_ok=True)

    prompts = [row.prompt] * len(images)
    stem = row.image_path.stem
    for index, image in enumerate(images, start=1):
        generated_path = config.generated_dir / f"{stem}_generated_{index:02d}.png"
        image.save(generated_path)

        with Image.open(row.image_path) as source_image:
            comparison = _comparison_canvas(source_image, image)
        comparison.save(config.comparisons_dir / f"{stem}_comparison_{index:02d}.png")

    grid_rows = 1 if len(images) <= 2 else 2
    cols = (len(images) + grid_rows - 1) // grid_rows
    grid = image_grid(images, rows=grid_rows, cols=cols)
    grid.save(config.generated_grid_path)

    config.prompts_path.write_text("\n".join(prompts), encoding="utf-8")
