from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence

@dataclass
class DatasetConfig:
    name: str
    concepts: List[str]
    target_class_name: str
    target_class_index: int
    concept_to_label: Dict[int, str] = field(default_factory=dict)
    data_paths: Dict[str, str] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    method: str = "dom"   # "dom", "logistic", "hinge"
    model_key: str = "resnet50"
    layers: Optional[Sequence[str]] = None
    n_values: Sequence[int] = (5, 10, 20, 40, 80)
    runs: int = 5
    sets_per_run: int = 5
    out_dir: str = "artifacts"
    seed: int = 42
