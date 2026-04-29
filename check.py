from safetensors.torch import load_file
import torch

sd = load_file("results/lora_adapter/pytorch_lora_weights.safetensors")
for k, v in sd.items():
    if torch.isnan(v).any() or torch.isinf(v).any():
        print("BAD", k)