
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, torch

def available_multimodal_models() -> List[str]:
    return [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
    ]

def choose_multimodal_model(key: str, device: Optional[torch.device] = None):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import CLIPModel, CLIPProcessor, CLIPImageProcessor, AutoTokenizer  # type: ignore
    except Exception as e:
        raise ImportError("transformers is required for multimodal (CLIP). `pip install transformers pillow`") from e

    key = key.strip()
    tok = AutoTokenizer.from_pretrained(key, use_fast=True)
    img_proc = CLIPImageProcessor.from_pretrained(key, use_fast=True)
    processor = CLIPProcessor(tokenizer=tok, image_processor=img_proc)

    model = CLIPModel.from_pretrained(key, use_safetensors=True)
    model.eval()
    model.to(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    # Provide readable layer names for both towers
    L_v = getattr(model.config.vision_config, "num_hidden_layers", 12)
    L_t = getattr(model.config.text_config, "num_hidden_layers", 12)

    def picks(L):
        # choose lower/mid/upper layers
        s = sorted(set([max(1, L//3), max(1, 2*L//3), L]))
        return s

    vision_layers = [f"vision_hidden_{i}" for i in picks(L_v)]
    text_layers   = [f"text_hidden_{i}" for i in picks(L_t)]
    all_layers = vision_layers + text_layers

    info = {
        "vision_hidden_total": L_v,
        "text_hidden_total": L_t,
        "vision_layers": vision_layers,
        "text_layers": text_layers,
        "all_layers": all_layers,
    }
    return model, processor, device, info
