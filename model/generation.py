from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from model.config import RunConfig
from model.dataset import PromptImageRow, build_minim_prompt


def image_grid(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    width, height = images[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(image, box=(index % cols * width, index // cols * height))
    return grid


def generate_batch(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    config: RunConfig,
    seed_offset: int = 0,
) -> list[Image.Image]:
    conditioned_prompts = [
        build_minim_prompt(prompt, config.minim_modality_id) for prompt in prompts
    ]
    generator = None
    if pipe.device.type == "cuda":
        generator = [
            torch.Generator(pipe.device.type).manual_seed(config.seed + seed_offset + index)
            for index in range(len(prompts))
        ]

    result = pipe(
        prompt=conditioned_prompts,
        negative_prompt=[config.negative_prompt] * len(conditioned_prompts),
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


def generate_all_batches(
    pipe: StableDiffusionPipeline,
    rows: list[PromptImageRow],
    config: RunConfig,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for batch_start in range(0, len(rows), config.batch_size):
        batch_rows = rows[batch_start:batch_start + config.batch_size]
        prompts = [row.prompt for row in batch_rows]
        images.extend(generate_batch(pipe, prompts, config, seed_offset=batch_start))
    return images


def save_generated_results(
    images: list[Image.Image],
    rows: list[PromptImageRow],
    config: RunConfig,
) -> None:
    if len(images) != len(rows):
        raise ValueError("The number of generated images must match the number of prompt-image rows.")

    config.generated_dir.mkdir(parents=True, exist_ok=True)
    config.comparisons_dir.mkdir(parents=True, exist_ok=True)
    config.results_root.mkdir(parents=True, exist_ok=True)

    prompts: list[str] = []
    for row, image in zip(rows, images, strict=True):
        prompts.append(row.prompt)
        stem = row.image_path.stem
        generated_path = config.generated_dir / f"{stem}_generated.png"
        image.save(generated_path)

        with Image.open(row.image_path) as source_image:
            comparison = _comparison_canvas(source_image, image)
        comparison.save(config.comparisons_dir / f"{stem}_comparison.png")

    grid_rows = 1 if len(images) <= 2 else 2
    cols = (len(images) + grid_rows - 1) // grid_rows
    grid = image_grid(images, rows=grid_rows, cols=cols)
    grid.save(config.generated_grid_path)

    config.prompts_path.write_text("\n".join(prompts), encoding="utf-8")
