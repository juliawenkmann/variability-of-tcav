
from __future__ import annotations
from typing import Sequence, List, Dict
import numpy as np
import pandas as pd

def cav_pairwise_mean_angle_deg(cavs: Sequence[np.ndarray]) -> float:
    if len(cavs)<2: return 0.0
    ang=[]
    for i in range(len(cavs)):
        for j in range(i+1,len(cavs)):
            u,v = cavs[i], cavs[j]
            cs = abs(float(u@v) / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-8))
            cs = min(1.0, max(-1.0, cs))
            ang.append(float(np.degrees(np.arccos(cs))))
    return float(np.mean(ang))

def sensitivity_from_grad_and_cav(grad: np.ndarray, cav: np.ndarray) -> float:
    return float(grad@cav)

def tcav_score_from_grads_and_cavs(grads: np.ndarray, cavs: Sequence[np.ndarray]) -> List[float]:
    scores=[]
    for v in cavs:
        s = grads@v
        scores.append(float((s>0).mean()))
    return scores

def aggregate_variance_by_n(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    keys = ['layer', 'n']
    if 'concept' in df.columns:
        keys = ['layer', 'concept', 'n']
    agg = df.groupby(keys).agg(
        mean_value=('value','mean'),
        std_value=('value','std'),
        count=('value','size')
    ).reset_index()
    return agg
