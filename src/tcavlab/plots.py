from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from .cache import save_plot_bundle, load_plot_bundle
import torch

TITLE_F = 50; LABEL_F = 50; TICK_F = 40; LEGEND_TITLE_F = 45; LEGEND_F = 40  # a bit larger

# ---------- helpers (same as before, abbreviated) ----------
from matplotlib.ticker import LogLocator, LogFormatterMathtext

def _apply_log_y_minor_labels(
    ax,
    minor_subs=(2, 3, 5),      # which minor decades to label (2×, 3×, 5×).
    major_size=LABEL_F,        # major tick label size
    minor_size=LEGEND_F       # minor tick label size (same as major for readability)
):
    """
    Show mathtext labels on both major and minor log ticks, e.g. 3×10^{-4},
    and make minor tick labels as large as majors.
    """
    # majors at 10^k
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    # minors at (2,3,5)×10^k (edit the tuple if you want all 2–9)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=minor_subs, numticks=100))

    # label both majors and minors with mathtext (gives "3×10^{-4}")
    fmt = LogFormatterMathtext(base=10.0, labelOnlyBase=False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)

    # make both sets of labels big
    ax.tick_params(axis="y", which="major", labelsize=major_size)
    ax.tick_params(axis="y", which="minor", labelsize=minor_size)

def _fmt_coef_tex(x: float, sig: int = 3, sci_lo: float = 1e-3, sci_hi: float = 1e3) -> str:
    if not np.isfinite(x) or x == 0: return "0"
    ax = abs(x)
    if sci_lo <= ax < sci_hi: return f"{x:.{sig}g}"
    exp = int(np.floor(np.log10(ax))); mant = x / (10 ** exp)
    return rf"{mant:.{sig}g}\times 10^{{{exp}}}"

def _fit_label_tex(a: float, b: float, thr_used: int | None) -> str:
    a_lbl = _fmt_coef_tex(a); b_lbl = _fmt_coef_tex(abs(b)); sign = "-" if b < 0 else "+"
    thr = f"(N≥{int(thr_used)}) " if thr_used is not None else ""
    return rf"Fit: $ {a_lbl}/N \; {sign} \; {b_lbl} $"

def _palette_map(names: List[str], palette: str | List[Tuple[float,float,float]]):
    cols = sns.color_palette(palette, n_colors=len(names)) if isinstance(palette, str) else palette
    return {name: cols[i] for i, name in enumerate(names)}

def _eps_from_positive(vals: np.ndarray) -> float:
    pos = vals[np.isfinite(vals) & (vals > 0)]
    return max(pos.min()*1e-3, 1e-12) if pos.size else 1e-12

def _summarize_bands(
    d: pd.DataFrame, ykey: str = "variance",
    group_keys: Tuple[str, str] = ("concept","n"),
    band: str = "auto", yscale: str = "log",
    conf: float = 0.68, qlo: float = 0.16, qhi: float = 0.84,
) -> pd.DataFrame:
    if band == "auto": band = "geom" if yscale == "log" else "sd"
    groups = d.groupby(list(group_keys), sort=True)
    eps = _eps_from_positive(d[ykey].to_numpy())
    rows = []
    for (concept, n), g in groups:
        y = g[ykey].to_numpy(); y = y[np.isfinite(y)]
        if y.size == 0: continue
        if band == "geom":
            y_clip = np.clip(y, eps, None); logy = np.log(y_clip)
            mu, sd = np.mean(logy), (np.std(logy, ddof=1) if y_clip.size > 1 else 0.0)
            z = 1.0 if abs(conf-0.68) < 1e-6 else 1.96 if abs(conf-0.95) < 1e-6 else 1.0
            center, lo, hi = float(np.exp(mu)), float(np.exp(mu - z*sd)), float(np.exp(mu + z*sd))
        elif band == "quantile":
            lo, hi = float(np.quantile(y, qlo)), float(np.quantile(y, qhi)); center = float(np.median(y))
        else:  # 'sd'
            m = float(np.mean(y)); s = float(np.std(y, ddof=1) if y.size > 1 else 0.0)
            center, lo, hi = m, max(m - s, eps if yscale == "log" else 0.0), m + s
        rows.append({"concept": concept, "n": n, "center": center, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).sort_values(["concept","n"])

# ---------- pretty, readable plots ----------
def plot_stability_vs_n(
    df: pd.DataFrame,
    layer: str,
    title: str,
    xscale: str = "linear",
    yscale: str = "log",
    ylabel: str = "Variance",
    palette: str = "viridis",
    fit_thresholds: List[int] = [0],
    save_path: str | None = None,
    cache_dir: str | None = None,
    cache_key: str | None = None,
    load_if_exists: bool = False,
    save_bundle: bool = True,
    band: str = "sd",
    conf: float = 0.68, qlo: float = 0.16, qhi: float = 0.84,
    xtick_every: int = 2,          
    ytick_every: Optional[int] = None,  
    legend_out: bool = False,
    showfit: bool = True,    
    showlabel: bool = False   
):
    if df is None or df.empty:
        print(f"Could not generate plot for layer '{layer}' because no data was provided.")
        return

    params_for_hash = {
        "fn": "plot_stability_vs_n",
        "layer": layer, "title": title, "ylabel": ylabel,
        "fit_thresholds": list(fit_thresholds) if fit_thresholds is not None else None,
        "palette": palette, "yscale": yscale, "band": band, "conf": conf,
        "qlo": qlo, "qhi": qhi, "xtick_every": xtick_every, "ytick_every": ytick_every,
    }
    if cache_dir and load_if_exists:
        cached = load_plot_bundle(cache_dir, cache_key or f"stability_vs_n__{layer}", params_for_hash)
        if cached is not None: df = cached

    d = df.copy()
    if "variance" not in d.columns:
        if {"mean_value","std_value"}.issubset(d.columns):
            d = d.rename(columns={"mean_value":"variance","std_value":"std"})
        elif "score_variance" in d.columns:
            d = d.rename(columns={"score_variance":"variance"})
        else:
            raise ValueError("plot_stability_vs_n expects 'variance' or aggregated 'mean_value'/'std_value'.")
    if "concept" not in d.columns:
        d["concept"] = "overall"
    d = d.sort_values("n")

    # Summaries for ribbons
    summary = _summarize_bands(d, ykey="variance", band=band, yscale=yscale, conf=conf, qlo=qlo, qhi=qhi)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(22, 12))  # bigger canvas for print

    # lines + ribbons
    concepts = list(summary["concept"].unique())
    cmap = _palette_map(concepts, palette)
    for c in concepts:
        g = summary[summary["concept"] == c].sort_values("n")
        ax.plot(g["n"], g["center"], marker="o", linestyle="-", linewidth=3.0, markersize=10,
                label=c, color=cmap[c], zorder=3)
        ax.fill_between(g["n"], g["lo"], g["hi"], color=cmap[c], alpha=0.22, zorder=2)

    # 1/N + b fit (same as before)
    def one_over_n_curve(n, a, b): return a / n + b
    mean_df = d.groupby("n")["variance"].mean().reset_index().sort_values("n")
    popt, successful_threshold = None, None
    for threshold in fit_thresholds:
        fit_df = mean_df[mean_df["n"] >= threshold]
        if len(fit_df) < 3: continue
        try:
            b_guess = fit_df.iloc[-1]["variance"]; a_guess = (fit_df.iloc[0]["variance"] - b_guess) * fit_df.iloc[0]["n"]
            p0 = [max(1e-12, a_guess), max(1e-12, b_guess)]
            bounds = ([0, 0], [np.inf, np.inf])
            popt, _ = curve_fit(one_over_n_curve, fit_df["n"], fit_df["variance"], p0=p0, bounds=bounds, maxfev=8000)
            successful_threshold = threshold; break
        except Exception:
            continue

    if popt is not None:
        a_fit, b_fit = popt
        x_smooth = np.linspace(d["n"].min(), d["n"].max(), 300)
        y_smooth = one_over_n_curve(x_smooth, a_fit, b_fit)
        print(_fit_label_tex(a_fit, b_fit, successful_threshold))
        if showfit and showlabel:
            ax.plot(x_smooth, y_smooth, linestyle="--", linewidth=3.0, label=_fit_label_tex(a_fit, b_fit, successful_threshold),
                    color="gray", zorder=1)
        if showfit and not showlabel:
            ax.plot(x_smooth, y_smooth, linestyle="--", linewidth=3.0,
                    color="gray", zorder=1)

    # axes styling (big + readable)
    ax.set_yscale(yscale); ax.set_xscale(xscale)
    ax.set_title(title, fontsize=TITLE_F, pad=24)
    ax.set_xlabel("Number of random examples per run (N)", fontsize=LABEL_F, labelpad=18)
    ax.set_ylabel(ylabel, fontsize=LABEL_F, labelpad=18)
    ax.tick_params(axis="both", which="major", labelsize=TICK_F, length=8, width=1.6, pad=10)

    # x ticks: every k-th label (no overlap)
    unique_n = np.array(sorted(d["n"].unique()), dtype=float)
    if unique_n.size:
        xticks = unique_n[::max(1, int(xtick_every))]
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        for lbl in ax.get_xticklabels():  # ensure no rotation overlap
            lbl.set_rotation(0)

    # y ticks: optional thinning + clean log labels
    if yscale == "log":
        _apply_log_y_minor_labels(ax, minor_subs=(2, 3, 5), major_size=TICK_F, minor_size=TICK_F)
    else:
        ax.tick_params(axis="y", which="major", labelsize=TICK_F)    
    if ytick_every and ytick_every > 1:
        labels = ax.get_yticklabels()
        for i, lbl in enumerate(labels):
            lbl.set_visible((i % ytick_every) == 0)

    # legend outside (no overlap with lines)
    if legend_out:
        leg = ax.legend(title="Concept", title_fontsize=LEGEND_TITLE_F, fontsize=LEGEND_F,
                        loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
    else:
        leg = ax.legend(title="Concept", title_fontsize=LEGEND_TITLE_F, fontsize=LEGEND_F)
    if leg and leg.get_frame() is not None:
        leg.get_frame().set_edgecolor("gray")

    fig.tight_layout(pad=1.2)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")  # high dpi for paper
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_variance_vs_n(*args, **kwargs):
    return plot_stability_vs_n(*args, **kwargs)

def plot_tcav_score_variance(
    df: pd.DataFrame,
    layer_name: str,
    xscale: str = "linear",
    yscale: str = "log",
    save_path: Optional[str] = None,
    palette: str = "plasma",
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    load_if_exists: bool = False,
    save_bundle: bool = True,
    **kwargs,   # passes xtick_every, ytick_every, legend_out, band, conf, ...
):
    if df is None or df.empty:
        print(f"Cannot plot for layer '{layer_name}', the DataFrame is empty.")
        return
    d = df.copy()
    if "variance" not in d.columns:
        if "score_variance" in d.columns: d = d.rename(columns={"score_variance":"variance"})
        elif "mean_value" in d.columns:   d = d.rename(columns={"mean_value":"variance","std_value":"std"})
        else: raise ValueError("tcav_score_variance plot expects 'variance', or 'score_variance', or aggregated 'mean_value'/'std_value'.")
    if "concept" not in d.columns: d["concept"] = "overall"
    title = f"TCAV Score Variance vs. Number of Random Examples: {layer_name}"
    return plot_variance_vs_n(
        df=d, layer=layer_name, title=title,
        xscale=xscale, yscale=yscale, ylabel="Variance of TCAV Scores",
        palette=palette, fit_thresholds=[0], save_path=save_path,
        cache_dir=cache_dir, cache_key=cache_key or f"tcav_score_variance__{layer_name}",
        load_if_exists=load_if_exists, save_bundle=save_bundle, **kwargs
    )



def _to_numpy_2d(a) -> np.ndarray:
    """Accept torch.Tensor or np.ndarray; return a 2D numpy array (n, d)."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    else:
        a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    return a

def plot_pca_projection(data_tensor, concept_name: str):
    """
    Reduces data to 2D using PCA and creates a scatter plot with large fonts.
    Accepts either torch.Tensor or np.ndarray of shape (n, d).
    """
    data_np = _to_numpy_2d(data_tensor)

    # 1) Fit PCA and transform the data
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_np)

    # 2) Transform the mean
    mean_vec = data_np.mean(axis=0, keepdims=True)
    mean_2d = pca.transform(mean_vec)

    # 3) Create the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Scatter of points
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                    alpha=0.7, label=f'{concept_name} Data Points')

    # Mean marker
    plt.scatter(mean_2d[0, 0], mean_2d[0, 1], marker='X',
                s=400, edgecolor='red', facecolor='red', linewidth=2, zorder=5,
                label='Mean of Data')

    # Labels & style
    plt.title(f"PCA Projection of '{concept_name}' Concept Embeddings", fontsize=30)
    plt.xlabel("Principal Component 1", fontsize=25)
    plt.ylabel("Principal Component 2", fontsize=25)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.axhline(y=mean_2d[0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=mean_2d[0, 0], color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"pca_projection_{concept_name.lower().replace(' ', '_')}.pdf", bbox_inches="tight")
    plt.show()


def plot_surround_assumption(proportions, min_prop, delta, label, bins=40):
    p = np.asarray(proportions, dtype=float).ravel()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.histplot(p, bins=bins, alpha=0.9)
    plt.axvline(min_prop, linestyle='--', linewidth=3.0, label=f"min={min_prop:.3f}")
    plt.axvline(float(delta), linestyle='--', linewidth=3.0, label=f"delta={float(delta):.3f}")
    plt.title(f"Surround Proportions — {label}", fontsize=30)
    plt.xlabel("Proportion of samples with dot(x-mean, ω) > ε", fontsize=25)
    plt.ylabel("Count", fontsize=25)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"surround_hist_{str(label).lower().replace(' ', '_')}.pdf", bbox_inches="tight")
    plt.show()
