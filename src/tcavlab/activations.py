
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import torch, numpy as np
from PIL import Image

# IMAGES
class TorchModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        super().__init__()
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x.to(self.device))

def get_activations_from_tensor(model: nn.Module, x: torch.Tensor, layer_name: str, device: Optional[torch.device]=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    layer = dict(model.named_modules())[layer_name]
    captured = {}
    def hook(_m,_i,o): captured["act"]=o.detach()
    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x.to(device))
    h.remove()
    act = captured.get("act")
    if act is None:
        raise RuntimeError(f"Could not capture activations at layer {layer_name}")
    return act.view(act.size(0), -1).cpu()

def get_gradient_at_layer(model: nn.Module, x: torch.Tensor, layer_name: str, class_index: int, device: Optional[torch.device]=None):
    import numpy as np
    device = device or next(model.parameters()).device
    layer = dict(model.named_modules())[layer_name]
    captured = {"out": None}
    def hook(_m,_i,o):
        captured["out"]=o
        o.retain_grad()
    h = layer.register_forward_hook(hook)
    x = x.to(device).requires_grad_(True)
    out = model(x)
    score = out[:, class_index].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    h.remove()
    if captured["out"] is None or captured["out"].grad is None:
        return None
    grad = captured["out"].grad.detach().cpu().numpy()
    N = grad.shape[0]
    return grad.reshape(N, -1)


# TEXT
@torch.no_grad()
def get_text_activations(model, encodings: dict, layer_index: int) -> np.ndarray:
    outputs = model(**encodings, output_hidden_states=True)  # safe to force
    hs = outputs.hidden_states                              # tuple: 0=embeddings, 1..L
    H = hs[layer_index]                                     # (N, S, D)
    N, S, D = H.shape
    return H.detach().cpu().view(N, S*D).numpy()

def get_text_gradients(model, encodings: dict, layer_index: int, class_index: int) -> np.ndarray | None:
    """
    Gradient of the class logit wrt hidden_states[layer_index].
    Do NOT set requires_grad on integer inputs; retain grad on the hidden state tensor.
    """
    # Ensure on same device as model
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in encodings.items()}

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():  # enable autograd for forward
        outputs = model(**enc, output_hidden_states=True)
        logits = outputs.logits                    # (N, C)
        hs = outputs.hidden_states                 # (L+1) tuple
        H = hs[layer_index]                        # (N, S, D) non-leaf; fine
        H.retain_grad()                            # tell autograd to keep grad at this node
        score = logits[:, class_index].sum()       # scalar
        score.backward()

    if H.grad is None:
        return None
    G = H.grad.detach().cpu().numpy()              # (N, S, D)
    N, S, D = G.shape
    return G.reshape(N, S*D)


# TABULAR
@torch.no_grad()
def get_tabular_activations(model, x: torch.Tensor, layer_name: str):
    # Return activations at the requested layer, flattened per-example.
    if layer_name == "input":
        z = x
    else:
        z1 = F.relu(model.fc1(x))
        if layer_name == "fc1":
            z = z1
        else:
            z2 = F.relu(model.fc2(z1))
            z = z2
    return z.detach().cpu().view(z.size(0), -1).numpy()

def get_tabular_gradients(model, x: torch.Tensor, layer_name: str, class_index: int):
    # Gradient of class logit wrt the chosen layer output.
    # For 'input': grad(logit_c, x); 'fc1': grad(logit_c, z1); 'fc2': grad(logit_c, z2).
    device = next(model.parameters()).device
    x = x.to(device).requires_grad_(True)
    z1_pre = model.fc1(x)
    z1 = torch.relu(z1_pre)
    z2_pre = model.fc2(z1)
    z2 = torch.relu(z2_pre)
    logits = model.fc_out(z2)
    score = logits[:, class_index].sum()
    model.zero_grad(set_to_none=True)
    if layer_name == "input":
        target = x
    elif layer_name == "fc1":
        target = z1
    else:
        target = z2
    target.retain_grad()
    score.backward()
    g = target.grad
    if g is None:
        return None
    return g.detach().cpu().view(g.size(0), -1).numpy()


# MULTI-MODAL
def _parse_layer_name(layer_name: str) -> Tuple[str, int]:
    # "vision_hidden_6" or "text_hidden_6"
    parts = layer_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid multimodal layer name: {layer_name}")
    modality = parts[0]  # 'vision' or 'text'
    idx = int(parts[-1])
    return modality, idx

def _make_neutral_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(127, 127, 127))

def encode_batch(processor, images: List[Image.Image], texts: List[str], device, pad: str = "max_length", max_length: int = 77):
    enc = processor(
        text=texts,
        images=images,
        padding=pad,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k,v in enc.items()}

@torch.no_grad()
def get_mm_activations(model, processor, items: List, layer_name: str, device, neutral_text: str = "a photo.", neutral_image: Optional[Image.Image] = None, batch_size: int = 16) -> np.ndarray:
    modality, layer_idx = _parse_layer_name(layer_name)
    if modality == "vision":
        # items are images; pair them with neutral text
        texts = [neutral_text] * len(items)
        def get_batch(i,j):
            enc = encode_batch(processor, items[i:j], texts[i:j], device)
            outputs = model(**enc, output_hidden_states=True)
            # Vision hidden_states: tuple (L+1); index layer_idx
            hs = outputs.vision_model_output.hidden_states
            H = hs[layer_idx]  # [B, S, D]
            B,S,D = H.shape
            return H.detach().cpu().reshape(B, S*D).numpy()
    elif modality == "text":
        # items are texts; pair with neutral image
        if neutral_image is None:
            neutral_image = _make_neutral_image()
        images = [neutral_image] * len(items)
        def get_batch(i,j):
            enc = encode_batch(processor, images[i:j], items[i:j], device)
            outputs = model(**enc, output_hidden_states=True)
            hs = outputs.text_model_output.hidden_states
            H = hs[layer_idx]
            B,S,D = H.shape
            return H.detach().cpu().reshape(B, S*D).numpy()
    else:
        raise ValueError(f"Unknown modality: {modality}")
    outs=[]
    for i in range(0, len(items), batch_size):
        outs.append(get_batch(i, min(len(items), i+batch_size)))
    return np.vstack(outs) if outs else np.zeros((0,0), dtype="float32")

def _parse_mm_layer(layer_name: str) -> Tuple[str, int]:
    # Accept "vision_hidden_6" or "text_hidden_6"
    parts = layer_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid multimodal layer name: {layer_name}")
    modality = parts[0]  # 'vision' or 'text'
    idx = int(parts[-1])
    return modality, idx

def _neutral_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(127, 127, 127))

@torch.no_grad()
def _frozen_text_embed(model, processor, target_text: str, device):
    tok = processor(text=[target_text], return_tensors="pt", padding=True, truncation=True)
    tok = {k: v.to(device).contiguous() for k, v in tok.items()}
    tout = model.text_model(**tok, output_hidden_states=False, return_dict=True)
    tfeat = tout.pooler_output            # [1, D_text]
    temb  = model.text_projection(tfeat)  # [1, D_proj]
    return F.normalize(temb, dim=-1).detach()

@torch.no_grad()
def _frozen_image_embed(model, processor, target_image: Image.Image, device):
    enc = processor(images=[target_image], return_tensors="pt")
    pix = enc["pixel_values"].to(device).to(memory_format=torch.contiguous_format).contiguous()
    vout = model.vision_model(pixel_values=pix, output_hidden_states=False, return_dict=True)
    vfeat = vout.pooler_output             # [1, D_vision]
    vemb  = model.visual_projection(vfeat) # [1, D_proj]
    return F.normalize(vemb, dim=-1).detach()

def get_mm_gradients(
    model, processor, items: List, layer_name: str, device,
    target_text: Optional[str] = None,
    target_image: Optional[Image.Image] = None,
    batch_size: int = 8,
    force_float32: bool = True,
    fallback_to_proj: bool = True,
) -> np.ndarray | None:
    """
    Gradient of cosine similarity wrt the chosen tower's hidden state (decoupled towers).
      - 'vision_*': sim( image_embeds , frozen(text_embeds(target_text)) )
      - 'text_*'  : sim( text_embeds  , frozen(image_embeds(target_image)) )
    We only backprop through the analyzed tower, force contiguity, and reshape gradients.
    If a backend still throws a view/stride error, we can optionally fall back to the
    projected embedding node ('fallback_to_proj=True').
    """
    modality, layer_idx = _parse_mm_layer(layer_name)
    out_batches = []

    autocast_ctx = torch.cuda.amp.autocast(enabled=False) if (force_float32 and torch.cuda.is_available()) else nullcontext()  # type: ignore

    if modality == "vision":
        if not target_text:
            raise ValueError("get_mm_gradients: target_text is required for vision layers.")
        frozen_txt = _frozen_text_embed(model, processor, target_text, device)  # [1, D_proj]

        for i in range(0, len(items), batch_size):
            enc = processor(images=items[i:i+batch_size], return_tensors="pt")
            pix = enc["pixel_values"].to(device).to(memory_format=torch.contiguous_format).contiguous()

            model.zero_grad(set_to_none=True)
            with torch.enable_grad(), autocast_ctx:
                vout = model.vision_model(pixel_values=pix, output_hidden_states=True, return_dict=True)
                H = vout.hidden_states[layer_idx].contiguous()   # [B, S, D]
                H.retain_grad()

                vfeat = vout.pooler_output                 # [B, D_vision]
                vemb  = model.visual_projection(vfeat)     # [B, D_proj]
                vemb  = F.normalize(vemb, dim=-1)

                try:
                    sim = (vemb @ frozen_txt.T).sum()      # scalar
                    sim.backward()
                except RuntimeError as e:
                    if not fallback_to_proj:
                        raise
                    # Fallback: attach grad to vemb instead of hidden state
                    model.zero_grad(set_to_none=True)
                    with torch.enable_grad(), autocast_ctx:
                        vout2 = model.vision_model(pixel_values=pix, output_hidden_states=True, return_dict=True)
                        vfeat2 = vout2.pooler_output
                        vemb2  = F.normalize(model.visual_projection(vfeat2), dim=-1)
                        vemb2.retain_grad()
                        sim2 = (vemb2 @ frozen_txt.T).sum()
                        sim2.backward()
                    Gproj = vemb2.grad.detach()
                    out_batches.append(Gproj.reshape(Gproj.size(0), -1).cpu().numpy())
                    continue

            if H.grad is None: return None
            G = H.grad.detach()
            out_batches.append(G.reshape(G.size(0), -1).cpu().numpy())

    elif modality == "text":
        if target_image is None:
            target_image = Image.new("RGB", (224, 224), color=(127, 127, 127))
        frozen_img = _frozen_image_embed(model, processor, target_image, device)  # [1, D_proj]

        for i in range(0, len(items), batch_size):
            tok = processor(text=items[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=77)
            tok = {k: v.to(device).contiguous() for k, v in tok.items()}

            model.zero_grad(set_to_none=True)
            with torch.enable_grad(), autocast_ctx:
                tout = model.text_model(**tok, output_hidden_states=True, return_dict=True)
                H = tout.hidden_states[layer_idx].contiguous()   # [B, S, D]
                H.retain_grad()

                tfeat = tout.pooler_output                # [B, D_text]
                temb  = model.text_projection(tfeat)      # [B, D_proj]
                temb  = F.normalize(temb, dim=-1)

                try:
                    sim = (temb @ frozen_img.T).sum()     # scalar
                    sim.backward()
                except RuntimeError as e:
                    if not fallback_to_proj:
                        raise
                    # Fallback: attach grad to temb
                    model.zero_grad(set_to_none=True)
                    with torch.enable_grad(), autocast_ctx:
                        tout2 = model.text_model(**tok, output_hidden_states=True, return_dict=True)
                        tfeat2 = tout2.pooler_output
                        temb2  = F.normalize(model.text_projection(tfeat2), dim=-1)
                        temb2.retain_grad()
                        sim2 = (temb2 @ frozen_img.T).sum()
                        sim2.backward()
                    Gproj = temb2.grad.detach()
                    out_batches.append(Gproj.reshape(Gproj.size(0), -1).cpu().numpy())
                    continue

            if H.grad is None: return None
            G = H.grad.detach()
            out_batches.append(G.reshape(G.size(0), -1).cpu().numpy())

    else:
        raise ValueError(f"Unknown modality: {modality}")

    return np.vstack(out_batches) if out_batches else None

# small helper for autocast control without imports clutter
from contextlib import nullcontext