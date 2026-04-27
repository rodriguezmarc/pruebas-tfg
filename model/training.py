from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from model.config import RunConfig
from model.dataset import PromptImageDataset, PromptImageRow, build_minim_prompt, collate_prompt_image_batch

DEFAULT_LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")


@dataclass
class TrainingArtifacts:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    device: torch.device
    dtype: torch.dtype


def _select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def _load_training_artifacts(config: RunConfig) -> TrainingArtifacts:
    device, dtype = _select_device()

    # Tokenizer / text encoder / VAE are loaded from the base MINIM repository.
    tokenizer = CLIPTokenizer.from_pretrained(
        config.base_model_id,
        subfolder="tokenizer",
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.base_model_id,
        subfolder="text_encoder",
        local_files_only=True,
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        config.base_model_id,
        subfolder="vae",
        local_files_only=True,
        torch_dtype=dtype,
    )

    # MINIM routes medical modalities through dedicated UNet branches.
    unet = UNet2DConditionModel.from_pretrained(
        config.base_model_id,
        subfolder=config.minim_unet_subfolder,
        local_files_only=True,
        torch_dtype=dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.base_model_id,
        subfolder="scheduler",
        local_files_only=True,
    )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(DEFAULT_LORA_TARGET_MODULES),
        bias="none",
    )
    unet.add_adapter(lora_config)

    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    return TrainingArtifacts(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
        device=device,
        dtype=dtype,
    )


def _encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
    config: RunConfig,
    device: torch.device,
) -> torch.Tensor:
    conditioned_prompts = [build_minim_prompt(prompt, config.minim_modality_id) for prompt in prompts]
    text_inputs = tokenizer(
        conditioned_prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    return text_encoder(input_ids)[0]


def _save_adapter(unet: UNet2DConditionModel, adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(adapter_dir)


def finetune_lora(
    config: RunConfig,
    rows: list[PromptImageRow],
) -> TrainingArtifacts:
    artifacts = _load_training_artifacts(config)
    torch.manual_seed(config.seed)
    if artifacts.device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    dataset = PromptImageDataset(rows, resolution=config.resolution)
    loader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_prompt_image_batch,
    )

    trainable_parameters = [param for param in artifacts.unet.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=config.learning_rate)

    artifacts.unet.train()
    global_step = 0
    progress_bar = tqdm(total=config.max_train_steps, desc="Fine-tuning LoRA", unit="step")
    for _epoch in range(config.num_train_epochs):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device=artifacts.device, dtype=artifacts.dtype)
            with torch.no_grad():
                latents = artifacts.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * artifacts.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                artifacts.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=artifacts.device,
            ).long()
            noisy_latents = artifacts.noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                encoder_hidden_states = _encode_prompts(
                    artifacts.tokenizer,
                    artifacts.text_encoder,
                    batch["prompts"],
                    config,
                    artifacts.device,
                )

            model_pred = artifacts.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if global_step >= config.max_train_steps:
                break
        if global_step >= config.max_train_steps:
            break
    progress_bar.close()

    _save_adapter(artifacts.unet, config.adapter_dir)
    artifacts.unet.eval()
    return artifacts


def build_inference_pipeline(
    config: RunConfig,
    artifacts: TrainingArtifacts,
) -> StableDiffusionPipeline:
    # Rebuild the full Stable Diffusion pipeline while keeping the MRI-family UNet
    # branch already adapted with LoRA.
    pipe = StableDiffusionPipeline.from_pretrained(
        config.base_model_id,
        local_files_only=True,
        torch_dtype=artifacts.dtype,
        tokenizer=artifacts.tokenizer,
        text_encoder=artifacts.text_encoder,
        vae=artifacts.vae,
        unet=artifacts.unet,
        safety_checker=None,
    )
    pipe = pipe.to(artifacts.device)
    pipe.enable_attention_slicing()
    return pipe
