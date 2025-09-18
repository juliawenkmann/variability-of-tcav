
from __future__ import annotations
from typing import Tuple, List, Callable
import torch
from torchvision import models

def available_models() -> List[str]:
    return ["resnet50", "googlenet", "mobilenet_v3_large", "vit_b_16"]

def choose_model(key: str) -> Tuple[torch.nn.Module, List[str], "Callable", torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    key = key.lower()
    if key == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2
        m = models.resnet50(weights=w)
        layers = ["layer2", "layer3", "layer4"]
        preprocess = w.transforms()
    elif key == "googlenet":
        w = models.GoogLeNet_Weights.IMAGENET1K_V1
        m = models.googlenet(weights=w)
        layers = ["inception4d", "inception5b"]
        preprocess = w.transforms()
    elif key == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        m = models.mobilenet_v3_large(weights=w)
        layers = ["features.10", "features.15"]
        preprocess = w.transforms()
    elif key == "vit_b_16":
        w = models.ViT_B_16_Weights.IMAGENET1K_V1
        m = models.vit_b_16(weights=w)
        layers = ["encoder.ln", "heads"]
        preprocess = w.transforms()
    else:
        raise ValueError(f"Unknown model key: {key}")
    m = m.eval().to(device)
    return m, layers, preprocess, device
