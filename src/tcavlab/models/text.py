
from __future__ import annotations
from typing import Tuple, List
import torch

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

def available_text_models() -> List[str]:
    return [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "bert-base-uncased",
    ]

def choose_text_model(key: str, device: torch.device | None = None):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ImportError("transformers is required for text mode. Please `pip install transformers`.")
    key = key.strip()
    tok = AutoTokenizer.from_pretrained(key)
    model = AutoModelForSequenceClassification.from_pretrained(key, output_hidden_states=True)
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    L = getattr(model.config, "num_hidden_layers", 6)
    picks = sorted(set([max(1, L//3), max(1, 2*L//3), L]))
    layer_names = [f"hidden_{i}" for i in picks]
    layer_indices = picks
    return model, layer_indices, layer_names, tok, device

def tokenize_texts(texts: List[str], tokenizer, max_length: int = 128, device: torch.device | None = None, pad: str = "max_length"):
    enc = tokenizer(texts, padding=pad, truncation=True, max_length=max_length, return_tensors="pt")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    return {k: v.to(device) for k, v in enc.items()}
