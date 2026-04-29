from __future__ import annotations

import torch

from model.config import RunConfig


def select_device_and_dtype(config: RunConfig) -> tuple[torch.device, torch.dtype]:
    preference = config.device_preference

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device_preference='cuda' was requested, but CUDA is not available.")
        return torch.device("cuda"), torch.float16

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device_preference='mps' was requested, but MPS is not available.")
        return torch.device("mps"), torch.float32

    if preference == "cpu":
        return torch.device("cpu"), torch.float16

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32

    return torch.device("cpu"), torch.float16


def build_generators(device: torch.device, seed: int, batch_size: int) -> list[torch.Generator]:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return [
        torch.Generator(device=generator_device).manual_seed(seed + index)
        for index in range(batch_size)
    ]
