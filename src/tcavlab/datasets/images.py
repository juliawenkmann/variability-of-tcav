
from __future__ import annotations
from typing import List
import os, glob, torch
from PIL import Image

def list_image_paths(folder: str, exts=(".jpg",".jpeg",".png",".bmp",".webp")) -> List[str]:
    files=[]
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def load_images_as_tensor(folder: str, preprocess, device: torch.device) -> torch.Tensor:
    paths = list_image_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    imgs = [preprocess(Image.open(p).convert("RGB")) for p in paths]
    batch = torch.stack(imgs)
    return batch.to(device)
