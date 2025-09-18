
from __future__ import annotations
from typing import Optional, Dict, Any
import os, json, hashlib, time
import pandas as pd

CACHE_VERSION = "1"

def _stable_hash(obj: Any) -> str:
    try:
        js = json.dumps(obj, sort_keys=True, default=repr)
    except TypeError:
        js = repr(obj)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()[:16]

def _bundle_dir(base_dir: str, scope: str, name: str, param_hash: str) -> str:
    safe = name.replace(os.sep, "_")
    d = os.path.join(base_dir, scope, f"{safe}__{param_hash}")
    os.makedirs(d, exist_ok=True)
    return d

def _meta_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, "metadata.json")

def _data_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, "data.pkl")

def save_df_bundle(base_dir: str, scope: str, name: str, params: Dict[str, Any], df: pd.DataFrame, extra: Optional[Dict[str, Any]] = None) -> str:
    param_hash = _stable_hash(params)
    bdir = _bundle_dir(base_dir, scope, name, param_hash)
    meta = {
        "cache_version": CACHE_VERSION,
        "created_at": time.time(),
        "params": params,
        "param_hash": param_hash,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
    }
    if extra:
        meta["extra"] = extra
    with open(_meta_path(bdir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    df.to_pickle(_data_path(bdir))
    return bdir

def try_load_df_bundle(base_dir: str, scope: str, name: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    param_hash = _stable_hash(params)
    bdir = _bundle_dir(base_dir, scope, name, param_hash)
    meta_fp = _meta_path(bdir); data_fp = _data_path(bdir)
    if not (os.path.exists(meta_fp) and os.path.exists(data_fp)): return None
    try:
        with open(meta_fp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("param_hash") != param_hash:
            return None
    except Exception:
        return None
    try:
        return pd.read_pickle(data_fp)
    except Exception:
        return None

def save_plot_bundle(base_dir: str, name: str, params: Dict[str, Any], df: pd.DataFrame, extra: Optional[Dict[str, Any]] = None) -> str:
    return save_df_bundle(base_dir, scope="plots", name=name, params=params, df=df, extra=extra)

def load_plot_bundle(base_dir: str, name: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    return try_load_df_bundle(base_dir, scope="plots", name=name, params=params)

# Backward-compatibility aliases
stable_hash = _stable_hash
def save_df_cache(base_dir, scope, name, params, df): return save_df_bundle(base_dir, scope, name, params, df)
def load_df_cache(base_dir, scope, name, params): return try_load_df_bundle(base_dir, scope, name, params)
def compute_with_cache(base_dir, scope, name, params, compute_fn):
    cached = try_load_df_bundle(base_dir, scope, name, params)
    if cached is not None: return cached
    df = compute_fn()
    save_df_bundle(base_dir, scope, name, params, df)
    return df
