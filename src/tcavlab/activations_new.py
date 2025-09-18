# tcavlab/activations.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependency for multimodal neutral image
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# ---------------------------------------------------------------------
# Generic Torch models (images / vision / any nn.Module)
# ---------------------------------------------------------------------

class TorchModelWrapper(nn.Module):
    """Thin wrapper that pins a model to a device and eval()s it."""
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        super().__init__()
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x.to(self.device))


def get_activations_from_tensor(
    model: nn.Module,
    x: torch.Tensor,
    layer_name: str,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Capture activations at a named module and return a **flattened** CPU tensor [B, D_flat]."""
    device = device or next(model.parameters()).device
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"Layer '{layer_name}' not found. Available: {list(modules.keys())[:20]} ...")

    layer = modules[layer_name]
    captured: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, o):
        # ensure a detached, contiguous tensor for flattening
        captured["act"] = o.detach().to(memory_format=torch.contiguous_format).contiguous()

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x.to(device))
    h.remove()

    act = captured.get("act")
    if act is None:
        raise RuntimeError(f"Could not capture activations at layer '{layer_name}'")
    B = act.size(0)
    return act.reshape(B, -1).cpu()


def get_gradient_at_layer(
    model: nn.Module,
    x: torch.Tensor,
    layer_name: str,
    class_index: int,
    device: Optional[torch.device] = None
) -> Optional[np.ndarray]:
    """
    Gradient of the class logit wrt the **output** of a named module (layer_name).
    Returns a numpy array [B, D_flat] or None if hook didn't fire.
    """
    device = device or next(model.parameters()).device
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"Layer '{layer_name}' not found.")

    layer = modules[layer_name]
    captured: Dict[str, torch.Tensor] = {"out": None}

    def hook(_m, _i, o):
        # Keep graph node; make sure it's contiguous for later reshape()
        t = o.to(memory_format=torch.contiguous_format).contiguous()
        captured["out"] = t
        t.retain_grad()

    h = layer.register_forward_hook(hook)
    x = x.to(device).requires_grad_(True)
    logits = model(x)
    score = logits[:, class_index].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    h.remove()

    node = captured["out"]
    if node is None or node.grad is None:
        return None
    g = node.grad.detach().cpu()
    B = g.size(0)
    return g.reshape(B, -1).numpy()


# ---------------------------------------------------------------------
# Hugging Face TEXT models
# ---------------------------------------------------------------------

@torch.no_grad()
def get_text_activations(model: nn.Module, encodings: Dict[str, torch.Tensor], layer_index: int) -> np.ndarray:
    """
    Flattened hidden state for a Hugging Face text model at hidden_states[layer_index].
    Expects `encodings` already on the correct device.
    """
    outputs = model(**encodings, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states  # tuple: index 0 = embeddings; 1..L = after blocks
    H = hs[layer_index]         # [N, S, D]
    N = H.size(0)
    return H.detach().cpu().reshape(N, -1).numpy()


def get_text_gradients(
    model: nn.Module,
    encodings: Dict[str, torch.Tensor],
    layer_index: int,
    class_index: int
) -> Optional[np.ndarray]:
    """
    Gradient of the class logit wrt hidden_states[layer_index] for a Hugging Face text model.
    We DO NOT set requires_grad on integer inputs; we retain grad at the hidden-state node.
    """
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in encodings.items()}

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        logits = outputs.logits            # [N, C]
        hs = outputs.hidden_states
        H = hs[layer_index].contiguous()   # [N, S, D]
        H.retain_grad()
        score = logits[:, class_index].sum()
        score.backward()

    if H.grad is None:
        return None
    G = H.grad.detach().cpu()
    N = G.size(0)
    return G.reshape(N, -1).numpy()


# ---------------------------------------------------------------------
# Tabular helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def get_tabular_activations(model: nn.Module, x: torch.Tensor, layer_name: str) -> np.ndarray:
    """
    If layer_name == 'input', return x flattened.
    Otherwise, use the generic hook-based `get_activations_from_tensor`.
    """
    if layer_name.lower() == "input":
        z = x
        return z.detach().cpu().reshape(z.size(0), -1).numpy()
    act = get_activations_from_tensor(model, x, layer_name)
    return act.numpy()


def get_tabular_gradients(
    model: nn.Module,
    x: torch.Tensor,
    layer_name: str,
    class_index: int
) -> Optional[np.ndarray]:
    """
    If layer_name == 'input', compute grad(logit_c, x);
    else compute grad(logit_c, module_output) via the generic hook.
    """
    if layer_name.lower() == "input":
        device = next(model.parameters()).device
        x = x.to(device).requires_grad_(True)
        logits = model(x)
        score = logits[:, class_index].sum()
        model.zero_grad(set_to_none=True)
        x.retain_grad()
        score.backward()
        g = x.grad
        if g is None:
            return None
        return g.detach().cpu().reshape(g.size(0), -1).numpy()
    # generic
    return get_gradient_at_layer(model, x, layer_name, class_index)


# ---------------------------------------------------------------------
# Multimodal (CLIP) helpers
# ---------------------------------------------------------------------

def _parse_mm_layer(layer_name: str) -> Tuple[str, int]:
    # Accept "vision_hidden_6" or "text_hidden_6" (0 is embeddings; 1..L are blocks)
    parts = layer_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid multimodal layer name: {layer_name}")
    modality = parts[0]  # 'vision' or 'text'
    idx = int(parts[-1])
    return modality, idx


def _neutral_image(size: int = 224) -> Image.Image:
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for neutral image generation.")
    return Image.new("RGB", (size, size), color=(127, 127, 127))


@torch.no_grad()
def get_mm_activations(
    model,                 # CLIPModel
    processor,             # CLIPProcessor
    items: List,           # images if 'vision_*', texts if 'text_*'
    layer_name: str,
    device: torch.device,
    batch_size: int = 16,
    pad: str = "max_length",
    max_length: int = 77,
) -> np.ndarray:
    """
    Flattened hidden state from a CLIP tower at a given layer_index.
    We call **the tower directly** (vision_model or text_model) to avoid building a cross-tower graph.
    """
    modality, layer_idx = _parse_mm_layer(layer_name)
    outs: List[np.ndarray] = []

    if modality == "vision":
        for i in range(0, len(items), batch_size):
            enc = processor(images=items[i:i+batch_size], return_tensors="pt")
            pix = enc["pixel_values"].to(device).to(memory_format=torch.contiguous_format).contiguous()
            vout = model.vision_model(pixel_values=pix, output_hidden_states=True, return_dict=True)
            H = vout.hidden_states[layer_idx]  # [B, S, D]
            B = H.size(0)
            outs.append(H.detach().cpu().reshape(B, -1).numpy())

    elif modality == "text":
        for i in range(0, len(items), batch_size):
            tok = processor(text=items[i:i+batch_size], return_tensors="pt",
                            padding=pad, truncation=True, max_length=max_length)
            tok = {k: v.to(device).contiguous() for k, v in tok.items()}
            tout = model.text_model(**tok, output_hidden_states=True, return_dict=True)
            H = tout.hidden_states[layer_idx]  # [B, S, D]
            B = H.size(0)
            outs.append(H.detach().cpu().reshape(B, -1).numpy())

    else:
        raise ValueError(f"Unknown modality: {modality}")

    if not outs:
        return np.zeros((0, 0), dtype="float32")
    return np.vstack(outs)


@torch.no_grad()
def _frozen_text_embed(model, processor, target_text: str, device: torch.device) -> torch.Tensor:
    """Compute a single **detached** text embedding for a prompt (decouples towers)."""
    tok = processor(text=[target_text], return_tensors="pt", padding=True, truncation=True)
    tok = {k: v.to(device).contiguous() for k, v in tok.items()}
    tout = model.text_model(**tok, output_hidden_states=False, return_dict=True)
    tfeat = tout.pooler_output            # [1, D_text]
    temb  = model.text_projection(tfeat)  # [1, D_proj]
    temb  = F.normalize(temb, dim=-1)
    return temb.detach()


@torch.no_grad()
def _frozen_image_embed(model, processor, target_image: Image.Image, device: torch.device) -> torch.Tensor:
    """Compute a single **detached** image embedding (decouples towers)."""
    enc = processor(images=[target_image], return_tensors="pt")
    pix = enc["pixel_values"].to(device).to(memory_format=torch.contiguous_format).contiguous()
    vout = model.vision_model(pixel_values=pix, output_hidden_states=False, return_dict=True)
    vfeat = vout.pooler_output              # [1, D_vision]
    vemb  = model.visual_projection(vfeat)  # [1, D_proj]
    vemb  = F.normalize(vemb, dim=-1)
    return vemb.detach()


def get_mm_gradients(
    model,                 # CLIPModel
    processor,             # CLIPProcessor
    items: List,           # images if 'vision_*', texts if 'text_*'
    layer_name: str,
    device: torch.device,
    target_text: Optional[str] = None,      # required for 'vision_*'
    target_image: Optional["Image.Image"] = None,  # optional for 'text_*'
    batch_size: int = 8,
    pad: str = "max_length",
    max_length: int = 77,
) -> Optional[np.ndarray]:
    """
    Gradient of a **cosine similarity score** wrt the chosen CLIP tower's hidden state.
    We **decouple towers** so that only one tower participates in autograd:

      - For 'vision_*': score = cosine( image_embeds, frozen(text_embeds(target_text)) )
      - For 'text_*'  : score = cosine( text_embeds,  frozen(image_embeds(target_image)) )

    Returns an array of shape [N, D_flat], or None if no gradient was captured.
    """
    modality, layer_idx = _parse_mm_layer(layer_name)
    grads_batches: List[np.ndarray] = []

    if modality == "vision":
        if not target_text:
            raise ValueError("get_mm_gradients: 'target_text' is required for vision layers.")
        frozen_txt = _frozen_text_embed(model, processor, target_text, device)  # [1, D_proj]

        for i in range(0, len(items), batch_size):
            enc = processor(images=items[i:i+batch_size], return_tensors="pt")
            pix = enc["pixel_values"].to(device).to(memory_format=torch.contiguous_format).contiguous()

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                vout = model.vision_model(pixel_values=pix, output_hidden_states=True, return_dict=True)
                H = vout.hidden_states[layer_idx].contiguous()  # [B, S, D]
                H.retain_grad()

                vfeat = vout.pooler_output                 # [B, D_vision]
                vemb  = model.visual_projection(vfeat)     # [B, D_proj]
                vemb  = F.normalize(vemb, dim=-1)

                sim = (vemb @ frozen_txt.T).sum()          # scalar
                sim.backward()

            if H.grad is None:
                return None
            G = H.grad.detach()
            grads_batches.append(G.reshape(G.size(0), -1).cpu().numpy())

    elif modality == "text":
        if target_image is None:
            if Image is None:
                raise ValueError("Pillow is required to build a neutral image for text layers.")
            target_image = Image.new("RGB", (224, 224), color=(127, 127, 127))
        frozen_img = _frozen_image_embed(model, processor, target_image, device)  # [1, D_proj]

        for i in range(0, len(items), batch_size):
            tok = processor(text=items[i:i+batch_size], return_tensors="pt",
                            padding=pad, truncation=True, max_length=max_length)
            tok = {k: v.to(device).contiguous() for k, v in tok.items()}

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                tout = model.text_model(**tok, output_hidden_states=True, return_dict=True)
                H = tout.hidden_states[layer_idx].contiguous()  # [B, S, D]
                H.retain_grad()

                tfeat = tout.pooler_output               # [B, D_text]
                temb  = model.text_projection(tfeat)     # [B, D_proj]
                temb  = F.normalize(temb, dim=-1)

                sim = (temb @ frozen_img.T).sum()        # scalar
                sim.backward()

            if H.grad is None:
                return None
            G = H.grad.detach()
            grads_batches.append(G.reshape(G.size(0), -1).cpu().numpy())

    else:
        raise ValueError(f"Unknown modality: {modality}")

    if not grads_batches:
        return None
    return np.vstack(grads_batches)
