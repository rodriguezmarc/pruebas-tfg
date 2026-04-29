from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunConfig:
    base_model_id: str = "CompVis/stable-diffusion-v1-4"        # base Stable Diffusion checkpoint used by MINIM
    minim_repo_id: str = "CoheeY/MINIM"                         # MINIM repository that stores modality-specific UNets
    modality: str = "Cardiac MRI"                               # cardiac modality used across the project
    minim_modality_id: int = 4                                  # MINIM MRI-family modality identifier
    device_preference: Literal["auto", "cuda", "mps", "cpu"] = "auto"  # preferred execution device
    local_files_only: bool = False                              # whether model loading must stay offline
    cache_dir: Path | None = None                               # optional Hugging Face cache directory override
    enable_xformers: bool = False                               # whether to enable xformers attention when available

    images_root: Path = Path("outputs/data/images")             # path to store preprocessed images
    csv_path: Path = Path("outputs/data/csv/acdc_minim.csv")    # path to the preprocessing CSV
    results_root: Path = Path("results")                        # path to store inference results
    seed: int = 42                                              # generation seed

    train_split: str = "train"                                  # reference to dataset train split
    train_batch_size: int = 1                                   # training batch size
    num_train_epochs: int = 10                                  # training epochs
    max_train_steps: int = 50                                   # training steps maximum
    learning_rate: float = 5e-5                                 # training learning rate
    gradient_accumulation_steps: int = 1                        # training gradient accumulation steps

    batch_size: int = 1                                         # inference batch size
    num_inference_steps: int = 50                               # inference steps
    guidance_scale: float = 7.5                                 # guidance scale
    resolution: int = 512                                       # resolution
    inference_row_index: int = 0                                # selected prompt-image row used for single-image inference
    negative_prompt: str = "blurry, distorted, artifact"        # negative prompt aligned with MINIM examples

    adapter_dirname: str = "lora_adapter"                       # LoRA adapter directory
    lora_rank: int = 16                                         # fine-tuning LoRA rank
    lora_alpha: int = 16                                        # fine-tuning LoRA alpha
    lora_dropout: float = 0.1                                   # fine-tuning LoRA dropout

    @property
    def adapter_dir(self) -> Path:
        return self.results_root / self.adapter_dirname

    @property
    def generated_dir(self) -> Path:
        return self.results_root / "generated"

    @property
    def generated_grid_path(self) -> Path:
        return self.results_root / "generated_grid.png"

    @property
    def prompts_path(self) -> Path:
        return self.results_root / "prompts.txt"

    @property
    def comparisons_dir(self) -> Path:
        return self.results_root / "comparisons"

    @property
    def minim_unet_subfolder(self) -> str:
        return f"unets/{self.minim_modality_id}/unet"
