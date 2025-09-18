
from __future__ import annotations
from typing import List, Optional, Tuple, Iterable
import os, glob, pandas as pd, numpy as np

def list_table_paths(folder: str, patterns: Iterable[str] = ("*.csv","*.tsv","*.txt")) -> List[str]:
    paths=[]
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted([p for p in paths if not os.path.basename(p).startswith(".")])

def _read_any(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if sep is None:
        if ext == ".tsv": sep = "\t"
        elif ext == ".csv": sep = ","
        else: sep = None
    return pd.read_csv(path, sep=sep, engine="python")

def load_tabular_folder(folder: str, feature_cols: Optional[List[str]] = None, label_col: Optional[str] = None, sep: Optional[str] = None, dtype: str = "float32") -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    files = list_table_paths(folder)
    if not files: raise FileNotFoundError(f"No table files found in {folder}")
    frames = [_read_any(p, sep=sep) for p in files]
    df = pd.concat(frames, ignore_index=True)
    if feature_cols is None:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if label_col in numeric: numeric.remove(label_col)
        feature_cols = numeric
    X = df[feature_cols].astype(dtype).to_numpy(copy=False)
    y = None
    if label_col is not None and label_col in df.columns:
        y = df[label_col].to_numpy()
    return X, y, feature_cols

def standardize_fit(X_list: List[np.ndarray]):
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    allX = np.vstack(X_list) if X_list else np.zeros((0,0), dtype="float32")
    scaler = StandardScaler().fit(allX)
    return scaler

def standardize_apply(X: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(X).astype("float32")
