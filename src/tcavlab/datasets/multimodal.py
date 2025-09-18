
from __future__ import annotations
from typing import List, Optional, Tuple, Iterable
import os, glob
from PIL import Image

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".webp")

def list_image_paths(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files = [p for p in files if not os.path.basename(p).startswith(".")]
    return sorted(files)

def load_images(folder: str, limit: Optional[int] = None) -> List[Image.Image]:
    paths = list_image_paths(folder)
    if limit is not None:
        paths = paths[:limit]
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    return imgs
