from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

from model.config import RunConfig
from model.datasets.prompt_image import PromptImageRow
from model.runtime.device import select_device_and_dtype


def _build_inception_feature_model(device: torch.device) -> torch.nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def _build_inception_logits_model(device: torch.device) -> torch.nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False, transform_input=False)
    model.eval()
    return model.to(device)


def _image_tensor(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    return transform(image.convert("RGB"))


def _load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return _image_tensor(image)


def _feature_statistics(features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    features_np = features.detach().cpu().numpy()
    mu = np.mean(features_np, axis=0)
    sigma = np.cov(features_np, rowvar=False)
    return mu, sigma


def _compute_fid(real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    mu_real, sigma_real = _feature_statistics(real_features)
    mu_generated, sigma_generated = _feature_statistics(generated_features)

    covmean = sqrtm(sigma_real @ sigma_generated)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_real - mu_generated
    fid = diff @ diff + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return float(fid)


def _compute_inception_score(logits: torch.Tensor, splits: int = 1) -> float:
    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()
    num_images = probabilities.shape[0]
    effective_splits = max(1, min(splits, num_images))
    split_scores: list[float] = []

    for split_idx in range(effective_splits):
        start = split_idx * num_images // effective_splits
        end = (split_idx + 1) * num_images // effective_splits
        split_probs = probabilities[start:end]
        py = np.mean(split_probs, axis=0, keepdims=True)
        kl = split_probs * (np.log(split_probs + 1e-12) - np.log(py + 1e-12))
        split_scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(split_scores))


def _compute_ms_ssim(real_batch: torch.Tensor, generated_batch: torch.Tensor) -> float:
    value = ms_ssim(real_batch, generated_batch, data_range=1.0, size_average=True)
    return float(value.detach().cpu().item())


def _compute_pairwise_ms_ssim(generated_batch: torch.Tensor) -> float | None:
    if generated_batch.shape[0] < 2:
        return None

    scores: list[float] = []
    for index in range(generated_batch.shape[0] - 1):
        anchor = generated_batch[index : index + 1]
        others = generated_batch[index + 1 :]
        anchor_batch = anchor.repeat(others.shape[0], 1, 1, 1)
        scores.append(_compute_ms_ssim(anchor_batch, others))

    return float(np.mean(scores)) if scores else None


def _forward_in_batches(
    model: torch.nn.Module,
    batch: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, batch.shape[0], chunk_size):
        outputs.append(model(batch[start : start + chunk_size]))
    return torch.cat(outputs, dim=0)


def evaluate_generated_batch(
    reference_rows: list[PromptImageRow],
    reference_image_path: Path,
    generated_images: list[Image.Image],
    config: RunConfig,
) -> dict[str, object]:
    device, _ = select_device_and_dtype(config)
    eval_batch_size = max(1, min(16, config.batch_size))

    real_tensors = [_load_image_tensor(row.image_path) for row in reference_rows]
    generated_tensors = [_image_tensor(image) for image in generated_images]

    real_batch = torch.stack(real_tensors).to(device)
    generated_batch = torch.stack(generated_tensors).to(device)

    with torch.no_grad():
        feature_model = _build_inception_feature_model(device)
        real_features = _forward_in_batches(feature_model, real_batch, eval_batch_size)
        generated_features = _forward_in_batches(feature_model, generated_batch, eval_batch_size)
        del feature_model

        logits_model = _build_inception_logits_model(device)
        generated_logits = _forward_in_batches(logits_model, generated_batch, eval_batch_size)
        del logits_model

    pairwise_ms_ssim = _compute_pairwise_ms_ssim(generated_batch)

    metrics: dict[str, object] = {
        "fid": _compute_fid(real_features, generated_features),
        "is": _compute_inception_score(generated_logits),
        "ms_ssim": pairwise_ms_ssim,
        "reference_count": len(reference_rows),
        "generated_count": len(generated_images),
        "reference_image": str(reference_image_path),
    }
    warnings: list[str] = []
    if len(generated_images) < 10:
        warnings.append(
            "Inception Score and FID are unstable with very small generated batches; "
            f"current generated_count={len(generated_images)}."
        )
    if len(reference_rows) < 10:
        warnings.append(
            "FID is unstable with very small real reference cohorts; "
            f"current reference_count={len(reference_rows)}."
        )
    if pairwise_ms_ssim is None:
        warnings.append(
            "MS-SSIM requires at least two generated images in the batch; increase batch_size for diversity evaluation."
        )
    if warnings:
        metrics["warnings"] = warnings
    return metrics


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
