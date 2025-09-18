
from __future__ import annotations
from typing import List, Tuple, Optional
import torch, torch.nn as nn

class TabMLP(nn.Module):
    def __init__(self, in_features: int, hidden: List[int], num_classes: int):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden[0])
        self.act_in = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        prev = hidden[0]
        for i, h in enumerate(hidden[1:], start=1):
            self.hidden_layers.append(nn.Linear(prev, h))
            self.hidden_layers.append(nn.ReLU())
            prev = h
        self.fc_out = nn.Linear(prev, num_classes)
    def forward(self, x):
        x = self.act_in(self.fc_in(x))
        for m in self.hidden_layers:
            x = m(x)
        return self.fc_out(x)

def available_tabular_models() -> List[str]:
    return ["mlp_small", "mlp_medium"]

def choose_tabular_model(key: str, num_features: int, num_classes: int, device: Optional[torch.device] = None):
    key = key.strip().lower()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    if key == "mlp_small":
        hidden = [64, 64]
    elif key == "mlp_medium":
        hidden = [128, 64, 64]
    else:
        raise ValueError(f"Unknown tabular model key: {key}")
    model = TabMLP(num_features, hidden, num_classes).to(device).eval()
    # Layer names available for hooks
    layer_names = ["fc_in"]
    # enumerate hidden linear layers with indices 0,2,4,... (since ReLUs in between)
    for idx, m in enumerate(model.hidden_layers):
        if isinstance(m, torch.nn.Linear):
            layer_names.append(f"hidden_layers.{idx}")
    return model, layer_names, device, hidden
