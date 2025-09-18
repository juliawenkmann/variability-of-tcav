
# On the Variability of TCAV

We provide notebooks and a clean framework to explore the variability of TCAV on four diffrent datasets. 

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
1. Clone this repository with: ```git clone https://github.com/juliawenkmann/variability-of-tcav```.
3. Open `notebooks/01_images.ipynb`.
4. Update the three data folders under the config cell:
   - `concepts_root / striped, zigzagged, dotted`
   - `random_dir`
   - `class_dir` (target class, e.g. zebra)
5. Run all cells.

> The notebook saves CAVs under: `artifacts/images/{layer}/{concept}/{n}/run_{i}.pkl`
> Analyses and plots are cached under: `artifacts/images/cache/…`

## Notes
- Variability uses our definition of variability (trace variance across CAV vectors) and with a **1/N + b** fit.
- `plot_variance_vs_n(...)` is an alias of `plot_stability_vs_n(...)` to match any older code.
- Parallel CAV training uses `joblib.delayed(...)` to avoid unpacking errors.

MIT License
# variability-of-tcav
