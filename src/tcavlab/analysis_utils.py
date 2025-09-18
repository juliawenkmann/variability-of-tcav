
from __future__ import annotations
from typing import List, Dict
import os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from .activations import get_gradient_at_layer
from .cache import save_df_bundle, try_load_df_bundle

def load_cav_vector_variance_data(layer: str, cav_output_dir: str, concepts_to_load: List[str], n_values: List[int], runs: int) -> pd.DataFrame:
    records=[]
    for concept in concepts_to_load:
        for n_val in n_values:
            for run_id in range(runs):
                file_path = os.path.join(cav_output_dir, layer, concept, str(n_val), f'run_{run_id}.pkl')
                if not os.path.exists(file_path):
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        cavs_for_this_run = pickle.load(f)
                    vecs=[]
                    for item in cavs_for_this_run:
                        if isinstance(item, dict) and 'vector' in item: vecs.append(np.asarray(item['vector'], dtype=float))
                        else: vecs.append(np.asarray(item, dtype=float))
                    if vecs and len(vecs)>=2:
                        M = np.stack(vecs, axis=0)
                        trace_variance = float(np.sum(np.var(M, axis=0, ddof=1)))
                        records.append({"n": n_val, "concept": concept, "variance": trace_variance})
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")
    return pd.DataFrame.from_records(records)

def load_cav_vector_variance_data_cached(layer: str, cav_output_dir: str, concepts_to_load: List[str], n_values: List[int], runs: int, cache_dir: str | None = None, cache_key: str | None = None, load_if_exists: bool=False, save: bool=True) -> pd.DataFrame:
    params = {"type":"cav_vector_trace_variance","layer":layer,"concepts":list(concepts_to_load),"n_values":list(n_values),"runs":int(runs),"cav_output_dir":str(cav_output_dir)}
    name = cache_key or f"cav_vector_trace_variance__{layer}"
    if cache_dir and load_if_exists:
        d = try_load_df_bundle(cache_dir, scope="analysis", name=name, params=params)
        if d is not None: return d
    df = load_cav_vector_variance_data(layer, cav_output_dir, concepts_to_load, n_values, runs)
    if cache_dir and save and not df.empty:
        save_df_bundle(cache_dir, scope="analysis", name=name, params=params, df=df)
    return df

def precompute_gradients_for_class(model, tensors, layers, class_index, device) -> Dict[str, np.ndarray]:
    gradients_per_layer={}
    for layer in layers:
        grads=[]
        for t in tqdm(tensors, desc=f"Grads for {layer}", leave=False):
            g = get_gradient_at_layer(model, t.unsqueeze(0).to(device), layer, class_index, device=device)
            if g is not None:
                grads.append(g[0])
        if grads:
            gradients_per_layer[layer] = np.stack(grads, axis=0)
    return gradients_per_layer


# ===== User-defined sensitivity score variance (single gradient vector) =====
from .cache import _stable_hash as stable_hash

def load_sensitivity_score_variance_data(
    layer: str,
    gradient_vector: "np.ndarray",
    cav_output_dir: str,
    concepts_to_load: List[str],
    n_values: List[int],
    runs: int
) -> pd.DataFrame:
    """Loads CAVs, calculates the variance of sensitivity scores per run, returns DataFrame with ['n','concept','variance']."""
    records = []
    g = gradient_vector.flatten()
    for concept in concepts_to_load:
        for n_val in n_values:
            for run_id in range(runs):
                file_path = os.path.join(cav_output_dir, layer, concept, str(n_val), f'run_{run_id}.pkl')
                if not os.path.exists(file_path):
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        cavs_for_this_run = pickle.load(f)
                    vecs = []
                    for item in cavs_for_this_run:
                        if isinstance(item, dict) and 'vector' in item:
                            vecs.append(np.asarray(item['vector'], dtype=float))
                        else:
                            vecs.append(np.asarray(item, dtype=float))
                    if vecs:
                        M = np.stack(vecs, axis=0)  # [S, D]
                        sensitivity_scores = M @ g     # [S]
                        score_variance = float(np.var(sensitivity_scores, ddof=1)) if len(sensitivity_scores) > 1 else 0.0
                        records.append({"n": int(n_val), "concept": str(concept), "variance": score_variance})
                except Exception as e:
                    print(f"    - Could not process file {file_path}: {e}")
    if not records:
        print(f"Warning: No sensitivity score data loaded for layer '{layer}'.")
    return pd.DataFrame.from_records(records)

def load_sensitivity_score_variance_data_cached(
    layer: str,
    gradient_vector: "np.ndarray",
    cav_output_dir: str,
    concepts_to_load: List[str],
    n_values: List[int],
    runs: int,
    cache_dir: str | None = None,
    cache_key: str | None = None,
    load_if_exists: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    params = {
        "type": "sensitivity_score_variance_user",
        "layer": layer,
        "concepts": list(concepts_to_load),
        "n_values": list(n_values),
        "runs": int(runs),
        "cav_output_dir": str(cav_output_dir),
        "grad_digest": stable_hash(np.asarray(gradient_vector).astype(float).tolist()),
    }
    name = cache_key or f"sens_score_var__{layer}"
    if cache_dir and load_if_exists:
        d = try_load_df_bundle(cache_dir, scope="analysis", name=name, params=params)
        if d is not None:
            return d
    df = load_sensitivity_score_variance_data(layer, gradient_vector, cav_output_dir, concepts_to_load, n_values, runs)
    if cache_dir and save and df is not None and not df.empty:
        save_df_bundle(cache_dir, scope="analysis", name=name, params=params, df=df)
    return df

# ===== User-defined TCAV score variance across CAVs (per run), given many class gradients =====

def calculate_tcav_score_variance(
    layers: List[str],
    concepts_to_load: List[str],
    n_values: List[int],
    runs: int,
    gradients_per_layer: Dict[str, np.ndarray],
    cav_output_dir: str
) -> pd.DataFrame:
    """Loads CAVs, calculates TCAV scores against pre-computed gradients, returns DataFrame with columns ['layer','n','concept','run_id','score_variance']."""
    records = []
    for layer in layers:
        if layer not in gradients_per_layer:
            print(f"Skipping layer {layer}, no gradients found.")
            continue
        class_gradients = gradients_per_layer[layer]  # [M, D]
        M = class_gradients
        num_examples = int(M.shape[0])
        for concept in concepts_to_load:
            for n_val in n_values:
                for run_id in range(runs):
                    path = os.path.join(cav_output_dir, layer, concept, str(n_val), f'run_{run_id}.pkl')
                    if not os.path.exists(path):
                        continue
                    try:
                        with open(path, 'rb') as f:
                            cav_list = pickle.load(f)
                        scores = []
                        for item in cav_list:
                            v = np.asarray(item['vector'], dtype=float) if isinstance(item, dict) and 'vector' in item else np.asarray(item, dtype=float)
                            s = (M @ v)  # [M]
                            tcav = float((s > 0).sum()) / max(1, num_examples)
                            scores.append(tcav)
                        score_variance = float(np.var(scores, ddof=1)) if len(scores) > 1 else 0.0
                        records.append({"layer": layer, "n": int(n_val), "concept": str(concept), "run_id": int(run_id), "score_variance": score_variance})
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
    return pd.DataFrame.from_records(records)

def calculate_tcav_score_variance_cached(
    layers: List[str],
    concepts_to_load: List[str],
    n_values: List[int],
    runs: int,
    gradients_per_layer: Dict[str, np.ndarray],
    cav_output_dir: str,
    cache_dir: str | None = None,
    cache_key: str | None = None,
    load_if_exists: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    # Build a compact descriptor for gradients shapes
    grad_shapes = {k: list(v.shape) for k, v in gradients_per_layer.items()}
    params = {
        "type": "tcav_score_variance_user",
        "layers": list(layers),
        "concepts": list(concepts_to_load),
        "n_values": list(n_values),
        "runs": int(runs),
        "cav_output_dir": str(cav_output_dir),
        "grad_shapes": grad_shapes,
    }
    name = cache_key or "tcav_score_variance_user"
    if cache_dir and load_if_exists:
        d = try_load_df_bundle(cache_dir, scope="analysis", name=name, params=params)
        if d is not None:
            return d
    df = calculate_tcav_score_variance(layers, concepts_to_load, n_values, runs, gradients_per_layer, cav_output_dir)
    if cache_dir and save and df is not None and not df.empty:
        save_df_bundle(cache_dir, scope="analysis", name=name, params=params, df=df)
    return df
