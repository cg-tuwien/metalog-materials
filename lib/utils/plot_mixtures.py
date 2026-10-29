# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

from __future__ import annotations
from typing import Optional, Tuple, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt


def _unwrap_model(model):
    return getattr(model, "best_model", None) or model


def _model_display_name(model) -> str:
    """Return a friendly model name for the plot badge."""
    m = _unwrap_model(model)
    # Prefer an explicit name if the class provides one
    for attr in ("model_name", "name"):
        if hasattr(m, attr) and getattr(m, attr):
            return str(getattr(m, attr))

    cls = m.__class__.__name__
    # Lightweight heuristics to prettify common classes
    if "Gaussian" in cls and "Mixture" in cls:
        return "Gaussian Mixture Model"
    if "SPT" in cls and "Metalog" in cls:
        return "SPT Metalog Mixture Model"
    if "Metalog" in cls and "Mixture" in cls:
        return "Metalog Mixture Model"
    if "Mixture" in cls:
        return "Mixture Model"
    return cls


def _get_attr(obj, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    # duck-type Auto* wrappers that hold the real model in .best_model
    if hasattr(obj, "best_model") and obj.best_model is not None:
        inner = getattr(obj, "best_model")
        if hasattr(inner, name):
            return getattr(inner, name)
    return default


def _to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device) if x.device.type != device else x


def _nice_domain(
    samples: torch.Tensor,
    model,
    domain: Optional[Tuple[float, float]],
    n_grid: int,
) -> torch.Tensor:
    device = _get_attr(model, "device", "cpu")
    bounds = _get_attr(model, "bounds", None)

    if domain is not None:
        lo, hi = domain
    elif bounds is not None:
        lo, hi = float(bounds[0]), float(bounds[1])
    else:
        s = samples.detach().view(-1).cpu().numpy()
        lo_q, hi_q = np.quantile(s, [0.005, 0.995])
        span = max(1e-8, hi_q - lo_q)
        lo, hi = lo_q - 0.05 * span, hi_q + 0.05 * span

    # keep tiny gap off the edges to avoid numerical nasties at exact bounds
    eps = 1e-8
    x = torch.linspace(lo + eps, hi - eps, n_grid, device=device)
    return x


def _component_pdfs(model, x_grid: torch.Tensor) -> np.ndarray:
    # per-component pdfs: (N, K)
    pdf_t = model.eval_pdf(x_grid)
    if isinstance(pdf_t, torch.Tensor):
        pdf_np = pdf_t.detach().cpu().numpy()
    else:
        pdf_np = np.asarray(pdf_t)
    # weights
    pi = _get_attr(model, "mixing_coeffs", None)
    if pi is None:
        raise RuntimeError(
            "Model must have `mixing_coeffs` (or best_model.mixing_coeffs)."
        )
    pi_np = pi.detach().cpu().numpy()
    return pdf_np * pi_np  # (N, K) weighted


def _mixture_pdf(model, x_grid: torch.Tensor) -> np.ndarray:
    mix_t = model.eval_mixture_pdf(x_grid)
    return mix_t.detach().cpu().numpy()


def _compute_ymax(
    pdf_total: np.ndarray,
    weighted: np.ndarray,
    hist_vals: np.ndarray,
    *,
    mode: str = "robust",  # "peak" or "robust"
    quantile: float = 0.995,  # used for "robust"
    include_hist: bool = True,
) -> float:
    """Return a y-maximum using either simple peak or a robust quantile cap."""
    if mode == "robust":
        vals = [np.asarray(pdf_total).ravel()]
        if weighted.size:
            vals.append(np.asarray(weighted).ravel())
        if include_hist and hist_vals.size:
            vals.append(np.asarray(hist_vals).ravel())
        y_all = np.concatenate(vals) if vals else np.array([])
        y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
        return float(np.quantile(y_all, quantile)) if y_all.size else 0.0

    # "peak" mode
    hist_max = (
        float(hist_vals.max()) if (include_hist and hist_vals.size) else 0.0
    )
    comp_max = float(weighted.max()) if weighted.size else 0.0
    return float(max(np.max(pdf_total), comp_max, hist_max))


def plot_mixture_fit(
    model,
    samples: torch.Tensor,
    *,
    bins: str | int | Iterable[float] = "fd",
    domain: Optional[Tuple[float, float]] = None,
    n_grid: int = 1000,
    title: Optional[str] = None,
    show_components: bool = True,
    component_alpha: float = 0.4,
    show_hist: bool = True,
    hist_alpha: float = 0.3,
    show_legend: bool = True,
    show: bool = True,
    y_lim: Optional[Tuple[float, float]] = None,
    y_pad: float = 1.08,
    y_mode: str = "robust",
    y_quantile: float = 0.99,  # used when y_mode="robust"
    y_include_hist_in_auto: bool = True,  # include histogram heights in auto calc
) -> tuple[plt.Figure, np.ndarray, dict]:
    """
    Generic plotting helper for MixtureModel1D and Auto* selectors.

    Returns (fig, axes, info) where info contains 'x_grid', 'pdf_total', 'weighted_pdfs'.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    device = _get_attr(model, "device", "cpu")
    samples = _to_device(samples.view(-1).float(), device)

    x_grid = _nice_domain(samples, model, domain, n_grid)
    weighted = _component_pdfs(model, x_grid)  # (N, K)
    pdf_total = _mixture_pdf(model, x_grid)

    # Build figure. If it's an Auto* with scores, give it a second panel.
    is_auto = hasattr(model, "scores") and getattr(model, "scores") is not None
    if is_auto:
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]}
        )
        ax, ax_scores = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
        axes = np.array([ax])
        ax_scores = None

    # Left plot: histogram + mixture + components
    h = np.array([])
    if show_hist:
        h, b, _ = ax.hist(
            samples.detach().cpu().numpy(),
            bins=bins,
            density=True,
            alpha=hist_alpha,
            label="Data",
            color=None,
            edgecolor="white",
            linewidth=0.5,
        )

    # mixture
    ax.plot(
        x_grid.detach().cpu().numpy(),
        pdf_total,
        lw=4.0,
        label="Mix. PDF",
        alpha=0.5,
    )

    # components
    if show_components:
        K = weighted.shape[1]
        for k in range(K):
            ax.plot(
                x_grid.detach().cpu().numpy(),
                weighted[:, k],
                ls="--",
                lw=2.0,
                alpha=component_alpha,
                label=f"$PDF_{k+1}$",
            )

    # y-limits
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    else:
        y_max = _compute_ymax(
            pdf_total=pdf_total,
            weighted=weighted,
            hist_vals=h,
            mode=y_mode,
            quantile=y_quantile,
            include_hist=y_include_hist_in_auto,
        )
        if y_max > 0:
            ax.set_ylim(0, y_max * y_pad)

    # x-limits
    lo, hi = float(x_grid.min().item()), float(x_grid.max().item())
    ax.set_xlim(lo, hi)

    # Titles and labels
    bounds = _get_attr(model, "bounds", None)
    base_title = title
    model_title = _model_display_name(model)
    if base_title is None:
        # Compose a useful default title
        k = _get_attr(model, "K", None)  # plain models
        t = _get_attr(model, "metalog_term", None)
        best_k = getattr(model, "best_k", None)  # auto selector
        best_t = getattr(model, "best_term", None)  # auto selector
        best_l = getattr(model, "best_lambda", None)  # auto selector
        best_r = getattr(model, "best_resolution", None)  # auto selector
        crit = getattr(model, "criterion", None)
        if best_k is not None:
            crit = "$W_1$" if crit == "wasserstein1" else crit
            if best_l is not None:
                l_str = f"{best_l:.1e}" if isinstance(best_l, float) else best_l
                if best_r is not None:
                    base_title = f"Auto {model_title} (metric={crit}, K*={best_k}, T*={best_t}, L*={l_str}, R*={best_r})"
                else:
                    base_title = f"Auto {model_title} (metric={crit}, K*={best_k}, T*={best_t}, L*={l_str})"
            else:
                base_title = f"Auto {model_title} (metric={crit}, K*={best_k}, T*={best_t})"
        elif k is not None:
            t_label = "" if t is None else f", T={t}"
            base_title = f"{model_title} (K={k}{t_label})"
        else:
            base_title = "Mixture Model"
    ax.set_title(base_title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    if show_legend:
        ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25, linestyle=":")

    # Right plot: model selection panel
    if ax_scores is not None:
        import numpy as np
        import matplotlib.pyplot as plt

        crit_name = getattr(model, "criterion", "score").upper()
        crit_name = (
            "$W_1$" if crit_name == "wasserstein1".upper() else crit_name
        )

        # Aggregate scores_4d into scores_2d by taking the best score over L, R
        scores_4d = getattr(model, "scores_4d", None)
        scores_2d = getattr(model, "scores_2d", None)

        if isinstance(scores_4d, dict) and len(scores_4d) > 0:
            scores_2d = {}
            for (K, T, L, R), score in scores_4d.items():
                if (K, T) not in scores_2d or score < scores_2d[(K, T)]:
                    scores_2d[(K, T)] = score

        if isinstance(scores_2d, dict) and len(scores_2d) > 0:
            Ks = sorted({k for (k, _) in scores_2d.keys()})
            Ts = sorted({t for (_, t) in scores_2d.keys()})
            x_idx = np.arange(len(Ks))

            # --- sequential colormap for lines per T ---
            cmap = plt.cm.viridis
            nT = len(Ts)
            # avoid div-by-zero when only one T
            norm = plt.Normalize(vmin=0, vmax=max(1, nT - 1))

            for j, T in enumerate(Ts):
                ys = np.array(
                    [scores_2d.get((K, T), np.nan) for K in Ks], dtype=float
                )
                mask = np.isfinite(ys)
                if mask.any():
                    color = cmap(norm(j))
                    ax_scores.plot(
                        x_idx,
                        ys,
                        marker="o",
                        linestyle="--",
                        label=f"T={T}",
                        alpha=0.5,
                        color=color,
                    )

            # highlight global best (K*, T*) with a neutral marker
            best_k = getattr(model, "best_k", None)
            best_t = getattr(model, "best_term", None)
            if (
                (best_k is not None)
                and (best_t is not None)
                and ((best_k, best_t) in scores_2d)
            ):
                try:
                    k_pos = Ks.index(best_k)
                    best_v = scores_2d[(best_k, best_t)]
                    ax_scores.scatter(
                        [k_pos],
                        [best_v],
                        s=90,
                        zorder=3,
                        edgecolor="none",
                        marker="*",
                        label=f"K*={best_k}, T*={best_t}",
                    )
                except Exception:
                    pass

            ax_scores.set_title("Model selection")
            ax_scores.set_xlabel("K")
            ax_scores.set_ylabel(crit_name)
            ax_scores.set_xticks(x_idx)
            ax_scores.set_xticklabels(Ks)
            ax_scores.grid(True, alpha=0.25, linestyle=":")
            ax_scores.legend(frameon=False, title="Metalog terms")

        else:
            # Fallback: best-over-T per K only
            ks = sorted(getattr(model, "scores", {}).keys())
            if ks:
                vals = [model.scores[k] for k in ks]
                ax_scores.plot(ks, vals, marker="o")
                best_k = getattr(model, "best_k", None)
                if best_k in ks:
                    best_v = model.scores[best_k]
                    ax_scores.scatter(
                        [best_k], [best_v], s=90, zorder=3, edgecolor="none"
                    )
                    ax_scores.annotate(
                        f"K* = {best_k}",
                        xy=(best_k, best_v),
                        xytext=(best_k + 0.1, best_v),
                        textcoords="data",
                        fontsize=9,
                        va="center",
                    )
            ax_scores.set_title("Model selection")
            ax_scores.set_xlabel("K")
            ax_scores.set_ylabel(crit_name)
            ax_scores.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()

    info = {
        "x_grid": x_grid.detach().cpu().numpy(),
        "pdf_total": pdf_total,
        "weighted_pdfs": weighted,  # (N, K)
        "bounds": bounds,
    }

    if show:
        plt.show()
    return fig, axes, info


def quick_plot(
    model,
    samples: torch.Tensor,
    *,
    n_grid: int = 1000,
    y_lim: Optional[Tuple[float, float]] = None,
    y_pad: float = 1.08,
    y_mode: str = "peak",
    y_quantile: float = 0.998,
    y_include_hist_in_auto: bool = True,
    **kwargs,
):
    return plot_mixture_fit(
        model,
        samples,
        n_grid=n_grid,
        y_lim=y_lim,
        y_pad=y_pad,
        y_mode=y_mode,
        y_quantile=y_quantile,
        y_include_hist_in_auto=y_include_hist_in_auto,
        **kwargs,
    )


def plot_best_em_history(
    model,
    histories=None,
    *,
    title: str | None = None,
    semilogy_delta: bool = True,
    tol: float | None = None,  # draw a horizontal line at tol on Δ plot if set
    color: str = "tab:blue",
    show: bool = True,
):
    """
    Plot ONLY the best model's EM history:
      - Left : total log-likelihood per iteration
      - Right: |Δ log L| per iteration
    Selection:
      - If model.best_k & model.best_term exist, pick the best restart for (K*, T*).
      - Else pick the single run with the highest final log-likelihood.

    Returns (fig, axes, info)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- pull histories if not provided ---
    if histories is None:
        histories = (
            getattr(model, "fit_results", None)
            or getattr(model, "em_histories", None)
            or getattr(model, "em_history", None)  # single sequence
            or getattr(model, "history", None)  # single sequence
        )
    if histories is None:
        raise ValueError(
            "No EM histories provided and none found on model (fit_results/em_histories/em_history/history)."
        )

    # --- helpers to extract K, T, LL, Δ safely ---
    def _attr(obj, names):
        for n in names:
            if isinstance(obj, dict) and n in obj:
                return obj[n]
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    def _extract_ll(rec):
        ll = _attr(
            rec, ["log_likelihood_history", "ll", "log_likelihoods", "history"]
        )
        if ll is None and isinstance(rec, (list, tuple, np.ndarray)):
            ll = rec
        arr = np.asarray(ll, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("Empty log-likelihood history.")
        return arr

    def _extract_delta(rec, ll):
        d = _attr(rec, ["delta_history"])
        if d is None:
            d = np.abs(np.diff(ll, prepend=ll[0]))
        else:
            d = np.asarray(d, dtype=float).reshape(-1)
            if d.size < ll.size:
                d = np.pad(d, (0, ll.size - d.size), constant_values=np.nan)
        return d

    infer_K = getattr(model, "n_components", None) or getattr(model, "K", None)
    infer_T = getattr(model, "metalog_term", 3)

    def _extract_K(rec):
        k = _attr(rec, ["K", "k"])
        if k is None:
            k = _attr(_attr(rec, ["model"]), ["K", "k"])
        return (
            int(k if k is not None else infer_K)
            if (k is not None or infer_K is not None)
            else None
        )

    def _extract_T(rec):
        t = _attr(rec, ["T", "term", "metalog_term"])
        if t is None:
            t = _attr(_attr(rec, ["model"]), ["T", "term", "metalog_term"])
        return int(t if t is not None else infer_T)

    # Normalize to a list
    if not isinstance(histories, (list, tuple)):
        histories = [histories]

    records = []
    for i, rec in enumerate(histories):
        try:
            ll = _extract_ll(rec)
        except Exception:
            continue
        K = _extract_K(rec)
        T = _extract_T(rec)
        d = _extract_delta(rec, ll)
        records.append({"K": K, "T": T, "ll": ll, "delta": d, "idx": i})

    if not records:
        raise ValueError("Found no valid EM histories to plot.")

    # Pick the best run:
    best_k = getattr(model, "best_k", None)
    best_t = getattr(model, "best_term", None)

    if best_k is not None and best_t is not None:
        candidates = [
            r for r in records if r["K"] == best_k and r["T"] == best_t
        ] or records
    else:
        candidates = records

    best_run = max(candidates, key=lambda r: float(r["ll"][-1]))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax_ll, ax_d = axes

    it = np.arange(len(best_run["ll"]))
    ax_ll.plot(it, best_run["ll"], marker="o", ms=3, lw=1.6, color=color)
    ax_ll.set_title(f"log L (K={best_run['K']}, T={best_run['T']})")
    ax_ll.set_xlabel("EM iteration")
    ax_ll.set_ylabel("log L")
    ax_ll.grid(True, alpha=0.25, linestyle=":")

    d = best_run["delta"].copy()
    if semilogy_delta:
        ax_d.semilogy(
            it,
            np.maximum(d, np.finfo(float).tiny),
            ls="--",
            lw=1.4,
            color=color,
        )
    else:
        ax_d.plot(it, d, ls="--", lw=1.4, color=color)
    if tol is not None and tol > 0:
        if semilogy_delta:
            ax_d.axhline(tol, ls=":", color="gray", alpha=0.7)
        else:
            ax_d.axhline(tol, ls=":", color="gray", alpha=0.7)
    ax_d.set_title("|Δ log L|")
    ax_d.set_xlabel("EM iteration")
    ax_d.set_ylabel("|Δ log L|")
    ax_d.grid(True, alpha=0.25, linestyle=":")

    # Title
    if title is None:
        crit = getattr(model, "criterion", None)
        if best_k is not None and best_t is not None:
            title = (
                f"Best EM history (criterion={crit}, K*={best_k}, T*={best_t})"
            )
        else:
            title = "Best EM history"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if show:
        plt.show()

    return fig, axes
