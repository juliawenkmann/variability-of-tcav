import numpy as np
import torch
from typing import Optional, Tuple

def _to_2d_torch(x, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Accept torch.Tensor or np.ndarray and return a 2D torch.Tensor on device."""
    if isinstance(x, torch.Tensor):
        t = x
        if device is not None:
            t = t.to(device)
    else:  # assume numpy or array-like
        t = torch.as_tensor(x, device=device)
    if dtype is not None:
        t = t.to(dtype)
    # Ensure 2D
    if t.ndim == 1:
        t = t.unsqueeze(1)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    if t.ndim != 2:
        raise ValueError(f"Expected 2D data (n, d); got shape {tuple(t.shape)}")
    return t

@torch.no_grad()
def check_surround_assumption(
    data_tensor,                       # torch.Tensor or np.ndarray, shape (n, d)
    epsilon: float,
    delta: float,
    num_directions: int = 5000,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[str] = None,      # e.g. "cuda:0" or "cpu"
    dir_batch_size: Optional[int] = None,
) -> Tuple[bool, float, np.ndarray]:
    """
    Empirically checks the 'surround' assumption on a single set of points X:
      For many unit directions ω, proportion of centered points with (x - x̄)·ω > ε
      should exceed δ. Returns (holds, min_proportion, proportions_per_direction).
    """
    t = _to_2d_torch(data_tensor, device=device,
                     dtype=torch.float32 if isinstance(data_tensor, np.ndarray) else None)

    n_samples, n_features = t.shape
    if n_samples == 0:
        print("Warning: Data tensor is empty. Cannot perform check.")
        return False, 0.0, np.array([])
    if num_directions <= 0:
        raise ValueError("num_directions must be positive")

    mean_vec = t.mean(dim=0, keepdim=True)
    centered = t - mean_vec

    def _sample_dirs(k: int) -> torch.Tensor:
        omegas = torch.randn(k, n_features, device=t.device, generator=generator)
        return omegas / omegas.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if dir_batch_size is None or dir_batch_size <= 0 or dir_batch_size >= num_directions:
        omegas = _sample_dirs(num_directions)
        dots = centered @ omegas.T
        proportions = (dots > float(epsilon)).sum(dim=0).to(torch.float32) / float(n_samples)
        proportions = proportions.cpu().numpy()
    else:
        props = []
        done = 0
        while done < num_directions:
            k = min(dir_batch_size, num_directions - done)
            dirs = _sample_dirs(k)
            dots = centered @ dirs.T
            p = (dots > float(epsilon)).sum(dim=0).to(torch.float32) / float(n_samples)
            props.append(p.cpu().numpy())
            done += k
        proportions = np.concatenate(props, axis=0)

    min_prop = float(np.min(proportions)) if proportions.size else 0.0
    holds = bool(min_prop > float(delta))
    return holds, min_prop, proportions
