
# On the Variability of TCAV

A clean, working scaffold for TCAV-style analyses across images (ready), with your **variability plot** (trace variance) and consistent styling, plus **caching** to avoid recomputing.

## Layout
```
tcavlab_ready/
├─ src/tcavlab/         # package (importable via the notebook shim)
│  ├─ __init__.py
│  ├─ activations.py
│  ├─ analysis_utils.py
│  ├─ cache.py
│  ├─ cav.py
│  ├─ config.py
│  ├─ datasets/images.py
│  ├─ metrics.py
│  ├─ models/vision.py
│  └─ plots.py          # your styling; plot_variance_vs_n == plot_stability_vs_n
└─ notebooks/
   └─ 01_images.ipynb   # ready to run (uses import shim, caching, your plots)
```

## How to run
1. Unzip so `notebooks/` and `src/` sit side by side.
2. Open `notebooks/01_images.ipynb`.
3. Update the three data folders under the config cell:
   - `concepts_root / striped, zigzagged, dotted`
   - `random_dir`
   - `class_dir` (target class, e.g. zebra)
4. Run all cells.

> The notebook saves CAVs under: `artifacts/images/{layer}/{concept}/{n}/run_{i}.pkl`
> Analyses and plots are cached under: `artifacts/images/cache/…`

## Notes
- Variability uses **your definition** (trace variance across CAV vectors) and your font sizes (title 38, labels 40, ticks 30, legend 34/30) with a **1/N + b** fit.
- `plot_variance_vs_n(...)` is an alias of `plot_stability_vs_n(...)` to match any older code.
- Parallel CAV training uses `joblib.delayed(...)` to avoid unpacking errors.

MIT License
# variability-of-tcav
