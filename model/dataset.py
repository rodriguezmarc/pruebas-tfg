from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class PromptImageRow:
    """
    Representation of a preprocessed .csv file row.
    Its definition helps when working with the dataset data.
    ------------------------------------------------------------------
    Row's structure: [ image_path, "prompt", modality ]
    """
    image_path: Path            # preprocessed image path
    prompt: str                 # preprocessed image resulting prompt
    modality: str               # preprocessed image modality: Cardiac MRI


def build_minim_prompt(prompt: str, modality_id: int) -> str:
    """
    Build the modality-conditioned prompt expected by MINIM.
    ------------------------------------------------------------------
    MINIM conditions prompts with a numeric modality prefix such as `4:`.
    """
    clean_prompt = prompt.strip()
    return f"{modality_id}:{clean_prompt}"


class PromptImageDataset(Dataset):
    """
    Representation of a preprocessed dataset, which contains the image's path, prompt and modality.
    Its definition helps when working with the dataset data it represents.
    """
    def __init__(self, rows: list[PromptImageRow], resolution: int) -> None:
        self.rows = rows
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.modality = "Cardiac MRI"

    def __len__(self) -> int:  # returns the length of the dataset
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        pixel_values = self.image_transform(image)
        return {
            "pixel_values": pixel_values,
            "prompt": row.prompt,
        }


def load_prompt_image_rows(csv_path: Path, images_root: Path) -> list[PromptImageRow]:
    """
    Method that recovers the preprocessed .csv file and injects it into the model.
    This step is esssential in the configuration prior to the finetuning and inference.
    ------------------------------------------------------------------------------------------
    - csv_path: Path, location of the preprocessed .csv file
    - images_root: Path, location of the preprocessed images
    """
    rows: list[PromptImageRow] = []

    with Path.open(csv_path, encoding="utf-8", newline="") as path:
        reader = csv.DictReader(path)  # each row is read as a dict
        for row in reader:
            rows.append(               # append to the rows construction
                PromptImageRow(        # conversion to PromptImageRow 
                    image_path= images_root / row["path"],
                    prompt=row["text"],
                    modality="Cardiac MRI"
                )
            )
    if not rows:
        raise ValueError(f"No contents were found in the {csv_path}.")

    return rows


def collate_prompt_image_batch(examples: list[dict[str, object]]) -> dict[str, object]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [str(example["prompt"]) for example in examples]
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
    }
