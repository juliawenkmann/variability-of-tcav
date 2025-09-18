
from __future__ import annotations
from typing import Sequence, Optional, Dict, List, Tuple
import os, pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .cav import sample_train_cav, MethodName
from .metrics import cav_pairwise_mean_angle_deg, aggregate_variance_by_n
from .utils import ensure_dir

def precompute_cavs_for_layer(X_pos: np.ndarray, X_neg: np.ndarray, layer_name: str, method: MethodName, n_values: Sequence[int], runs: int, sets_per_run: int, out_dir: str, seed: int = 42, with_replacement: bool = False) -> None:
    rng = np.random.default_rng(seed)
    base_dir = ensure_dir(out_dir)
    for n in n_values:
        for run_id in range(runs):
            run_path = os.path.join(base_dir, str(n), f"run_{run_id}.pkl")
            if os.path.exists(run_path):
                print(f"CAV file already exists, skipping: {run_path}")
                continue 
            seeds = [int(rng.integers(0, 2**31-1)) for _ in range(sets_per_run)]
            cavs = Parallel(n_jobs=-1, backend="loky")(
                delayed(sample_train_cav)(X_pos, X_neg, n_examples=n, method=method, random_state=s)
                for s in seeds
            )
            run_path = os.path.join(base_dir, str(n), f"run_{run_id}.pkl")
            ensure_dir(os.path.dirname(run_path))
            with open(run_path, "wb") as f:
                pickle.dump(cavs, f)

def _load_cavs_directory(base_dir: str, n_values: Sequence[int], runs: int) -> Dict[Tuple[int,int], List[np.ndarray]]:
    out={}
    for n in n_values:
        for run_id in range(runs):
            p = os.path.join(base_dir, str(n), f"run_{run_id}.pkl")
            if not os.path.exists(p): continue
            try:
                with open(p, "rb") as f:
                    cavs = pickle.load(f)
                vecs=[]
                for item in cavs:
                    if isinstance(item, dict) and "vector" in item: vecs.append(item["vector"])
                    else: vecs.append(item)
                out[(n, run_id)] = [np.asarray(v, dtype=float) for v in vecs]
            except Exception:
                continue
    return out

def cav_variability_analysis(cav_root_for_layer_and_concept: str, layer_name: str, concept_name: str, n_values: Sequence[int], runs: int) -> pd.DataFrame:
    cavs_map = _load_cavs_directory(cav_root_for_layer_and_concept, n_values, runs)
    records=[]
    for (n, run_id), vecs in cavs_map.items():
        if len(vecs)<2: continue
        value = cav_pairwise_mean_angle_deg(vecs)
        records.append({"layer": layer_name, "concept": concept_name, "n": n, "run": run_id, "value": value})
    return aggregate_variance_by_n(records)

def sensitivity_variance_analysis(cav_root_for_layer_and_concept: str, layer_name: str, concept_name: str, gradients_for_layer: np.ndarray, n_values: Sequence[int], runs: int) -> pd.DataFrame:
    cavs_map = _load_cavs_directory(cav_root_for_layer_and_concept, n_values, runs)
    records=[]
    for (n, run_id), vecs in cavs_map.items():
        if not vecs: continue
        cav_means=[]
        for v in vecs:
            s = gradients_for_layer@v
            cav_means.append(float(np.mean(s)))
        value = float(np.var(cav_means, ddof=1)) if len(cav_means)>1 else 0.0
        records.append({"layer": layer_name, "concept": concept_name, "n": n, "run": run_id, "value": value})
    return aggregate_variance_by_n(records)

def tcav_score_variance_analysis(cav_root_for_layer_and_concept: str, layer_name: str, concept_name: str, gradients_for_layer: np.ndarray, n_values: Sequence[int], runs: int) -> pd.DataFrame:
    cavs_map = _load_cavs_directory(cav_root_for_layer_and_concept, n_values, runs)
    records=[]
    for (n, run_id), vecs in cavs_map.items():
        if not vecs: continue
        scores=[]
        for v in vecs:
            s = gradients_for_layer@v
            scores.append(float((s>0).mean()))
        value = float(np.var(scores, ddof=1)) if len(scores)>1 else 0.0
        records.append({"layer": layer_name, "concept": concept_name, "n": n, "run": run_id, "value": value})
    return aggregate_variance_by_n(records)

from .cache import save_df_bundle, try_load_df_bundle

def _cache_name(prefix: str, layer: str, concept: str) -> str:
    return f"{prefix}__{layer}__{concept}"

def cav_variability_analysis_cached(cav_root_for_layer_and_concept: str, layer_name: str, concept_name: str, n_values: Sequence[int], runs: int, cache_dir: Optional[str]=None, cache_key: Optional[str]=None, load_if_exists: bool=False, save: bool=True):
    params = {"type":"cav_variability","layer":layer_name,"concept":concept_name,"n_values":list(n_values),"runs":int(runs),"cav_root":str(cav_root_for_layer_and_concept)}
    name = cache_key or _cache_name("cav_variability", layer_name, concept_name)
    if cache_dir and load_if_exists:
        d = try_load_df_bundle(cache_dir, scope="analysis", name=name, params=params)
        if d is not None: return d
    df = cav_variability_analysis(cav_root_for_layer_and_concept, layer_name, concept_name, n_values, runs)
    if cache_dir and save and not df.empty:
        save_df_bundle(cache_dir, scope="analysis", name=name, params=params, df=df)
    return df

def sensitivity_variance_analysis_cached(cav_root_for_layer_and_concept: str, layer_name: str, concept_name: str, gradients_for_layer: np.ndarray, n_values: Sequence[int], runs: int, cache_dir: Optional[str]=None, cache_key: Optional[str]=None, load_if_exists: bool=False, save: bool=True):
    params = {"type":"sensitivity_variance","layer":layer_name,"concept":concept_name,"n_values":list(n_values),"runs":int(runs),"cav_root":str(cav_root_for_layer_and_concept),"grad_shape":list(gradients_for_layer.shape)}
    name = cache_key or _cache_name("sensitivity_variance", layer_name, concept_name)
    if cache_dir and load_if_exists:
        d = try_load_df_bundle(cache_dir, scope="analysis", name=name, params=params)
        if d is not None: return d
    df = sensitivity_variance_analysis(cav_root_for_layer_and_concept, layer_name, concept_name, gradients_for_layer, n_values, runs)
    if cache_dir and save and not df.empty:
        save_df_bundle(cache_dir, scope="analysis", name=name, params=params, df=df)
    return df

def tcav_score_variance_analysis_cached(cav_root_for_layer_and_concept: str, layer_name: str, concept_name: str, gradients_for_layer: np.ndarray, n_values: Sequence[int], runs: int, cache_dir: Optional[str]=None, cache_key: Optional[str]=None, load_if_exists: bool=False, save: bool=True):
    params = {"type":"tcav_score_variance","layer":layer_name,"concept":concept_name,"n_values":list(n_values),"runs":int(runs),"cav_root":str(cav_root_for_layer_and_concept),"grad_shape":list(gradients_for_layer.shape)}
    name = cache_key or _cache_name("tcav_score_variance", layer_name, concept_name)
    if cache_dir and load_if_exists:
        d = try_load_df_bundle(cache_dir, scope="analysis", name=name, params=params)
        if d is not None: return d
    df = tcav_score_variance_analysis(cav_root_for_layer_and_concept, layer_name, concept_name, gradients_for_layer, n_values, runs)
    if cache_dir and save and not df.empty:
        save_df_bundle(cache_dir, scope="analysis", name=name, params=params, df=df)
    return df
