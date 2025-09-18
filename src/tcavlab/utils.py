import os, random, numpy as np, torch

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True); return path

def set_all_seeds(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception: pass

def l2_normalize(x, eps: float = 1e-8):
    import numpy as np
    n = np.linalg.norm(x) + eps
    return x / n

def device_auto() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
