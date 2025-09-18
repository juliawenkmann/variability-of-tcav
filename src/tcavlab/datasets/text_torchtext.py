from __future__ import annotations
from typing import Iterator, List, Optional, Tuple
import os, glob
import torch
from torch.utils.data import IterableDataset, DataLoader

# TorchText import: support both old and legacy namespaces
try:
    from torchtext.legacy import data as ttdata      # torchtext <= 0.14
except Exception:
    try:
        from torchtext import data as ttdata         # very old torchtext
    except Exception:
        ttdata = None

# --- NEW: pick format from extension ---
def _ext_to_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "tsv" if ext == ".tsv" else "csv"

# --- NEW: choose proper field schema by file type ---
def _fields_for_path(path: str, TEXT, Label):
    fmt = _ext_to_format(path)
    # TSV: assume 3 columns -> [id, text, label]
    #     If Label is None, we still ignore that third column with None.
    if fmt == "tsv":
        return [("id", None), ("text", TEXT), ("label", Label if Label is not None else None)], fmt
    # CSV: your concepts are usually single-column with pre-tokenized lines
    return [("text", TEXT)], fmt

def get_tensor_from_filename(
    filename: str,
    TEXT, Label,
    device: torch.device,
    const_len: int = 7,
) -> Iterator[torch.Tensor]:
    assert ttdata is not None, "torchtext is required for this loader."

    fields, fmt = _fields_for_path(filename, TEXT, Label)
    ds = ttdata.TabularDataset(path=filename, fields=fields, format=fmt)

    pad_id = TEXT.vocab.stoi[TEXT.pad_token]
    unk_id = TEXT.vocab.stoi.get(getattr(TEXT, "unk_token", "<unk>"), 0)

    for example in ds:
        tokens = getattr(example, "text")
        numerical_tokens = [TEXT.vocab.stoi.get(tok, unk_id) for tok in tokens]
        processed_ids = numerical_tokens[:const_len]
        if len(processed_ids) < const_len:
            processed_ids += [pad_id] * (const_len - len(processed_ids))
        yield torch.tensor(processed_ids, device=device, dtype=torch.long)

class _FolderIterDataset(IterableDataset):
    def __init__(self, folder: str, TEXT, Label, device: torch.device,
                 const_len: int = 7,
                 patterns: Tuple[str,...] = ("*.csv", "*.tsv")):   # NEW: include *.tsv
        super().__init__()
        self.folder = folder
        self.TEXT = TEXT
        self.Label = Label
        self.device = device
        self.const_len = const_len
        self.patterns = patterns

    def __iter__(self):
        files = []
        for pat in self.patterns:
            files.extend(glob.glob(os.path.join(self.folder, pat)))
        for fp in sorted(files):
            for t in get_tensor_from_filename(fp, self.TEXT, self.Label, self.device, self.const_len):
                yield t

def dataset_to_dataloader(dataset: IterableDataset, batch_size: int = 1) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)

def assemble_concept(name: str, id: int, concepts_path: str, TEXT, Label, device: torch.device,
                     const_len: int = 7, batch_size: int = 1):
    folder = os.path.join(concepts_path, name)
    ds = _FolderIterDataset(folder, TEXT, Label, device, const_len=const_len)
    it = dataset_to_dataloader(ds, batch_size=batch_size)
    return {"id": id, "name": name, "data_iter": it}

def indices_to_tokens(idx_tensor: torch.Tensor, TEXT) -> List[str]:
    itos = TEXT.vocab.itos
    pad = TEXT.pad_token
    toks = [itos[int(i)] if 0 <= int(i) < len(itos) else getattr(TEXT, "unk_token", "<unk>") for i in idx_tensor.tolist()]
    return [t for t in toks if t != pad]

def print_concept_sample(concept_iter: DataLoader, TEXT, max_print: int = 10):
    print("--- Concept Samples ---")
    count = 0
    for batch in concept_iter:
        for row in batch:
            sent = " ".join(indices_to_tokens(row, TEXT))
            print(sent)
            count += 1
            if count >= max_print:
                print("-----------------------")
                return
    print("-----------------------")

def covert_text_to_tensor(input_texts: List[str], TEXT, nlp, device: torch.device) -> torch.Tensor:
    rows = []
    unk_id = TEXT.vocab.stoi.get(getattr(TEXT, "unk_token", "<unk>"), 0)
    for s in input_texts:
        ids = [TEXT.vocab.stoi.get(tok.text, unk_id) for tok in nlp.tokenizer(s)]
        rows.append(torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0))
    return torch.cat(rows, dim=0) if rows else torch.empty(0, dtype=torch.long, device=device)

def format_float(f: float) -> float:
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

# --- Build TEXT/Label vocab across concepts + random + class (supports csv/tsv) ---
def build_fields_and_vocab(concept_names: List[str], concepts_root: str, random_dir: str, class_dir: str,
                           *, lower: bool = True, pad_token: str = "pad", unk_token: str = "<unk>",
                           use_label: bool = False):
    assert ttdata is not None, "torchtext is required for this loader."
    TEXT = ttdata.Field(sequential=True, tokenize=str.split, lower=lower, pad_token=pad_token, unk_token=unk_token)
    Label = ttdata.Field(sequential=False, use_vocab=True) if use_label else None

    def one_tabular(file_path: str):
        fields, fmt = _fields_for_path(file_path, TEXT, Label)
        return ttdata.TabularDataset(path=file_path, fields=fields, format=fmt)

    def collect(folder):
        files = []
        for pat in ("*.csv", "*.tsv"):
            files.extend(glob.glob(os.path.join(folder, pat)))
        return sorted(files)

    datasets = []
    for cname in concept_names:
        for fp in collect(os.path.join(concepts_root, cname)):
            datasets.append(one_tabular(fp))
    for fp in collect(random_dir):
        datasets.append(one_tabular(fp))
    for fp in collect(class_dir):
        datasets.append(one_tabular(fp))

    if datasets:
        TEXT.build_vocab(*datasets)
        if Label is not None:
            Label.build_vocab(*datasets)
    return TEXT, Label

def collect_texts_from_concept(name: str, concepts_root: str, TEXT, Label, device: torch.device,
                               const_len: int = 7, max_docs: Optional[int] = None) -> List[str]:
    loader = assemble_concept(name, id=0, concepts_path=concepts_root,
                              TEXT=TEXT, Label=Label, device=device, const_len=const_len,
                              batch_size=1)["data_iter"]
    out: List[str] = []
    for batch in loader:
        for row in batch:
            toks = indices_to_tokens(row, TEXT)
            if toks:
                out.append(" ".join(toks))
        if max_docs is not None and len(out) >= max_docs:
            break
    return out
