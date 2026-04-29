from __future__ import annotations

from typing import Any

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from model.config import RunConfig
from model.datasets.prompt_image import build_minim_prompt


class CardiacMINIMPipeline(StableDiffusionPipeline):
    @classmethod
    def from_cardiac_minim(
        cls,
        config: RunConfig,
        *,
        unet: UNet2DConditionModel | None = None,
        torch_dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> "CardiacMINIMPipeline":
        active_dtype = torch.float16 if torch_dtype is None else torch_dtype
        active_unet = unet
        if active_unet is None:
            active_unet = UNet2DConditionModel.from_pretrained(
                config.minim_repo_id,
                subfolder=config.minim_unet_subfolder,
                local_files_only=local_files_only,
                cache_dir=config.cache_dir,
                torch_dtype=active_dtype,
            )

        base_pipe = StableDiffusionPipeline.from_pretrained(
            config.base_model_id,
            unet=active_unet,
            safety_checker=None,
            local_files_only=local_files_only,
            cache_dir=config.cache_dir,
            torch_dtype=active_dtype,
        )
        pipe = cls(**base_pipe.components)
        pipe.register_to_config(
            modality=config.modality,
            minim_repo_id=config.minim_repo_id,
            minim_modality_id=config.minim_modality_id,
            negative_prompt=config.negative_prompt,
        )
        return pipe

    def _condition_prompt(self, prompt: str | list[str]) -> str | list[str]:
        modality_id = int(self.config.minim_modality_id)
        if isinstance(prompt, str):
            return build_minim_prompt(prompt, modality_id)
        return [build_minim_prompt(item, modality_id) for item in prompt]

    def _default_negative_prompt(self, batch_size: int) -> str | list[str]:
        negative_prompt = str(self.config.negative_prompt)
        if batch_size == 1:
            return negative_prompt
        return [negative_prompt] * batch_size

    def __call__(
        self,
        prompt: str | list[str],
        *args: Any,
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ):
        conditioned_prompt = self._condition_prompt(prompt)
        batch_size = 1 if isinstance(conditioned_prompt, str) else len(conditioned_prompt)
        if negative_prompt is None:
            negative_prompt = self._default_negative_prompt(batch_size)
        return super().__call__(
            conditioned_prompt,
            *args,
            negative_prompt=negative_prompt,
            **kwargs,
        )
