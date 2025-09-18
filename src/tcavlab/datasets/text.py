from __future__ import annotations
from typing import List, Optional, Iterable
import os, glob, re
import pandas as pd

# Heuristic candidates for a "text" column in CSVs
_PREFERRED_TEXT_COLS = ("text", "sentence", "utterance", "review", "content", "comment", "body")

def _list_paths(folder: str, exts: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    # exclude hidden/temp files
    paths = [p for p in paths if not os.path.basename(p).startswith(".")]
    return sorted(paths)

def _clean_line(
    text: str,
    drop_tokens: Iterable[str] = (".", "pad"),
    lowercase: bool = True,
) -> str:
    if text is None:
        return ""
    s = str(text)
    if lowercase:
        s = s.lower()
    toks = s.split()
    drop = {t.lower() for t in drop_tokens}
    kept = []
    for t in toks:
        tl = t.lower()
        if tl in drop:
            continue
        # drop tokens that are only made of dots (".", "..", "...")
        if set(t) <= {"."}:
            continue
        kept.append(t if not lowercase else tl)
    out = " ".join(kept)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def _load_ragged_token_file(path: str, drop_tokens: Iterable[str], lowercase: bool) -> List[str]:
    """Fallback: read any file line-by-line as whitespace-separated tokens (handles ragged rows)."""
    out: List[str] = []
    encodings = ("utf-8", "latin-1")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                for raw in f:
                    s = _clean_line(raw, drop_tokens, lowercase)
                    if s:
                        out.append(s)
            return out
        except Exception:
            continue
    return out

def _load_csv_one(
    path: str,
    csv_text_col: Optional[str],
    csv_sep: Optional[str],
    drop_tokens: Iterable[str],
    lowercase: bool,
) -> List[str]:
    # Try pandas first; if it fails (ragged rows), fall back to line-by-line.
    try:
        df = pd.read_csv(path, sep=csv_sep, engine="python")
    except Exception:
        return _load_ragged_token_file(path, drop_tokens, lowercase)

    # Case 1: explicit text column name
    if csv_text_col and csv_text_col in df.columns:
        series = df[csv_text_col].astype("string").fillna("")
        return [ _clean_line(s, drop_tokens, lowercase) for s in series.tolist() if str(s).strip() != "" ]

    # Case 2: try common text column names
    for col in _PREFERRED_TEXT_COLS:
        if col in df.columns:
            series = df[col].astype("string").fillna("")
            return [ _clean_line(s, drop_tokens, lowercase) for s in series.tolist() if str(s).strip() != "" ]

    # Case 3: single-column CSV -> each row is a line of tokens
    if df.shape[1] == 1:
        col = df.columns[0]
        series = df[col].astype("string").fillna("")
        out: List[str] = []
        for s in series.tolist():
            cleaned = _clean_line(s, drop_tokens, lowercase)
            if cleaned:
                out.append(cleaned)
        return out

    # Case 4: multi-column CSV -> treat row as list of tokens across columns
    out: List[str] = []
    for _, row in df.iterrows():
        cells = [str(x) for x in row.tolist() if pd.notna(x)]
        tokens = []
        for c in cells:
            tokens.extend(c.split())
        drop = {t.lower() for t in drop_tokens}
        tokens_clean = []
        for t in tokens:
            tl = t.lower()
            if tl in drop:
                continue
            if set(t) <= {"."}:
                continue
            tokens_clean.append(tl if lowercase else t)
        s = " ".join(tokens_clean).strip()
        if s:
            out.append(s)
    return out

def _load_from_csv_folder(
    folder: str,
    csv_text_col: Optional[str],
    csv_sep: Optional[str],
    drop_tokens: Iterable[str],
    lowercase: bool,
) -> List[str]:
    csvs = _list_paths(folder, (".csv", ".tsv"))
    texts: List[str] = []
    for p in csvs:
        texts.extend(_load_csv_one(p, csv_text_col, csv_sep, drop_tokens, lowercase))
    return texts

def _load_from_txt_folder(
    folder: str,
    drop_tokens: Iterable[str],
    lowercase: bool,
) -> List[str]:
    txts = _list_paths(folder, (".txt",))
    out: List[str] = []
    for p in txts:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except UnicodeDecodeError:
            with open(p, "r", encoding="latin-1", errors="ignore") as f:
                s = f.read()
        s = _clean_line(s, drop_tokens, lowercase)
        if s:
            out.append(s)
    return out

def load_texts(
    folder: str,
    *,
    csv_text_col: Optional[str] = None,  # set if your CSV has a specific text column name
    csv_sep: Optional[str] = None,       # None -> let pandas infer; else provide e.g. "," or "\\t" or r"\\s+"
    drop_tokens: Iterable[str] = (".", "pad"),
    lowercase: bool = True,
) -> List[str]:
    """
    Load texts from CSV/TSV or TXT files in a folder.

    Strategy:
      1) If .csv/.tsv files exist, try pandas; if parsing fails (ragged rows), fall back to line-by-line tokens.
      2) Otherwise, load .txt files (one document per file).

    Cleaning:
      - lowercases by default
      - drops tokens like '.' and 'pad'
      - removes tokens made solely of dots ('.', '..', '...')
      - collapses whitespace
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    csv_texts = _load_from_csv_folder(folder, csv_text_col, csv_sep, drop_tokens, lowercase)
    if csv_texts:
        return csv_texts

    txt_texts = _load_from_txt_folder(folder, drop_tokens, lowercase)
    if txt_texts:
        return txt_texts

    raise FileNotFoundError(f"No .csv/.tsv or .txt files found in {folder}")
