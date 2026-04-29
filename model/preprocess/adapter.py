from __future__ import annotations

from dataclasses import dataclass

from data.datasets.acdc.cache import export_preprocessed_dataset
from model.config import RunConfig
from model.datasets.prompt_image import PromptImageRow, load_prompt_image_rows


@dataclass(frozen=True)
class CardiacPreprocessingAdapter:
    config: RunConfig

    def export_and_load_rows(self) -> list[PromptImageRow]:
        export_preprocessed_dataset(split=self.config.train_split)
        return load_prompt_image_rows(self.config.csv_path, self.config.images_root)
