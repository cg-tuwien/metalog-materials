# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from matplotlib import cm
from typing import Literal
import lib.utils.logit_transform as lt
import numpy as np
import logging
from pathlib import Path
import math
from lib.metalogs import (
    cos_theta_phi_to_unitvec,
    unitvec_to_cos_theta_phi,
    estimate_bivariate_metalog_pdf,
    eval_bivariate_metalog_pdf,
    eval_bounded_metalog_pdf,
    eval_bounded_metalog_quantile,
    eval_unbounded_metalog_pdf,
    eval_unbounded_metalog_quantile,
    get_metalog_coefficients,
    init_metalogs,
    make_probs,
    mean_unbounded_metalog,
    sample_bivariate_metalog,
    variance_unbounded_metalog,
    is_feasible_metalog,
)


import matplotlib.pyplot as plt
import torch

from lib.utils.sample_record import SampleRecord


@torch.no_grad
def plot_metalogs(
    samples: torch.Tensor,
    metalog_resolution: int,
    label: str,
    x_lim: list,
    metalog_coefficients: torch.Tensor,
    scores: torch.Tensor,
    num_terms: torch.Tensor,
    bounds: tuple[float, float] = None,
    y_lim_pdf: list | None = None,
    probs: torch.Tensor = None,
    orig_x: torch.Tensor = None,
    orig_cdf: torch.Tensor = None,
    orig_pdf: torch.Tensor = None,
    orig_samples: torch.Tensor = None,
    show_hist: bool = True,
) -> plt.Figure:
    """Plots the given metalog using coefficients.

    Args:
        samples (torch.Tensor): Samples in which metalogs were generated.
        label (str): Label of the plot.
        x_lim (list): Limits for the x-axis.
        metalog_coefficients (torch.Tensor): Metalog coefficients in a (max_num_terms - 3, max_num_terms - 1) shaped
         tensor.
        scores (torch.Tensor): Score metric of each metalog in a (max_num_terms - 3) tensor.
        num_terms (torch.Tensor): A tensor of metalog terms. Usually torch.arange(3, max_num_terms).
        bounds (float | None, optional): Lower and upper bound. Defaults to None.
        y_lim_pdf (list | None, optional): Limits the y-axis for the pdf. Defaults to None.
        probs (torch.Tensor | None, optional): CDF values of the samples. Defaults to None.
        show_hist (bool): Shows histogram of samples in the PDF plot. Defaults to True.
    """
    fig = plt.figure(
        constrained_layout=True, figsize=(12, 4 * num_terms.shape[0])
    )
    fig.suptitle(label, y=1.02)
    subfigs = fig.subfigures(nrows=num_terms.shape[0], ncols=1)

    # back to cpu
    samples = samples.cpu()

    if probs is not None:
        probs = probs.cpu()
        probs_len = probs.size(0)
        if probs_len != metalog_resolution:
            raise ValueError(
                "Metalog resolution and size of probabilities do not match."
            )
    else:
        probs = make_probs(metalog_resolution, device="cpu")

    if orig_x is not None:
        orig_x = orig_x.cpu()
    if orig_cdf is not None:
        orig_cdf = orig_cdf.cpu()
    if orig_pdf is not None:
        orig_pdf = orig_pdf.cpu()

    ys = make_probs(1000, device="cpu")
    metalog_coefficients = metalog_coefficients.cpu()
    num_terms = num_terms.cpu()
    scores = scores.cpu()
    is_bounded_metalog = bounds is not None

    samples_scatter = samples
    if is_bounded_metalog:
        samples_scatter = torch.cat(
            [
                torch.tensor([bounds[0]], device=samples.device),
                samples,
                torch.tensor([bounds[1]], device=samples.device),
            ]
        )
        probs = torch.cat(
            [
                torch.tensor([0.0], device=probs.device),
                probs,
                torch.tensor([1.0], device=probs.device),
            ]
        )

    for i, term in enumerate(num_terms):
        if torch.isnan(metalog_coefficients).all():
            ax = subfigs[i].subplots(nrows=1, ncols=1)
            ax.axis("off")  # Hide the axes box, ticks, and labels
            ax.text(
                0.5,
                0.5,
                f"{term}-Metalog is not feasible",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                fontweight="bold",
                transform=ax.transAxes,
            )
            continue

        metalog_coeffs = get_metalog_coefficients(metalog_coefficients, term)

        if not is_feasible_metalog(metalog_coeffs):
            ax = subfigs[i].subplots(nrows=1, ncols=1)
            ax.axis("off")  # Hide the axes box, ticks, and labels
            ax.text(
                0.5,
                0.5,
                f"{term}-Metalog is not feasible",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                fontweight="bold",
                transform=ax.transAxes,
            )
            continue

        if is_bounded_metalog:
            M_y, unbounded_My = eval_bounded_metalog_quantile(
                metalog_coeffs,
                bounds[0],
                bounds[1],
                ys,
            )
            m_y = eval_bounded_metalog_pdf(
                metalog_coeffs,
                bounds[0],
                bounds[1],
                ys,
                M_y=unbounded_My,
            )
        else:
            y = make_probs(1000, device=metalog_coeffs.device.type)
            M_y = eval_unbounded_metalog_quantile(metalog_coeffs, y)
            m_y = eval_unbounded_metalog_pdf(metalog_coeffs, y)

        subfigs[i].suptitle(
            f"{term}-Term Metalog, BIC/AIC Score: {scores[i]:.5f}"
        )
        axes = subfigs[i].subplots(nrows=1, ncols=2)

        # First column plots (CDF)
        ax = axes[0]
        ax.plot(
            M_y,
            ys,
            label="Metalog CDF",
        )
        if orig_x is not None and orig_cdf is not None:
            ax.plot(
                orig_x,
                orig_cdf,
                label="Original CDF",
                alpha=0.5,
            )
        ax.scatter(
            samples_scatter,
            probs,
            color="dimgray",
            label="Metalog Data",
            s=15,
        )
        ax.set_title(f"{term}-Term Metalog CDF")
        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel("$M(y)$")
        ax.set_ylabel("$y$")
        ax.grid(True)
        ax.legend(loc="best")

        # Second column plots (PDF)
        ax = axes[1]
        ax.plot(
            M_y,
            m_y,
            label="Metalog PDF",
        )
        if orig_x is not None and orig_pdf is not None:
            ax.plot(
                orig_x,
                orig_pdf,
                label="Original PDF",
                alpha=0.5,
            )

        if show_hist:
            hist_samples = samples
            if orig_samples is not None:
                hist_samples = orig_samples.cpu()

            ax.hist(
                hist_samples,
                density=True,
                bins="auto",
                histtype="stepfilled",
                alpha=0.2,
                label=f"Data",
            )

        ax.set_title(f"{term}-Term Metalog PDF")
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim_pdf is not None:
            ax.set_ylim(y_lim_pdf)
        ax.set_xlabel(f"$M(y)$")
        ax.set_ylabel(f"$m(y)$")
        ax.grid(True)
        ax.legend(loc="best")

    return fig


def _plot_ridge(
    n, M_y, m_y, ys, colors, metalog_coeffs, fig_suptitle, condition_values
):
    """Helper to plot 2.5D Ridge (joyplot) slices."""
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 6), constrained_layout=True
    )
    fig.suptitle(f"{fig_suptitle} (Ridge Plot)", y=1.05)

    max_pdf = torch.max(m_y[~torch.isnan(m_y)]).item()
    shift_pdf = max_pdf * 0.35
    shift_cdf = 1.0 * 0.35

    # Lists to store tick locations and labels
    y_ticks_pdf, y_ticks_cdf, y_labels = [], [], []

    for row in range(n - 1, -1, -1):
        if not is_feasible_metalog(metalog_coeffs[row].unsqueeze(0)):
            continue

        y_base_pdf = row * shift_pdf
        y_base_cdf = row * shift_cdf

        # Plot the curves and fills
        axes[1].plot(
            M_y[row, :], m_y[row, :] + y_base_pdf, color=colors[row], lw=1.5
        )
        axes[1].fill_between(
            M_y[row, :],
            y_base_pdf,
            m_y[row, :] + y_base_pdf,
            facecolor=colors[row],
            alpha=0.6,
            edgecolor="white",
            lw=0.5,
        )

        axes[0].plot(
            M_y[row, :], ys[row, :] + y_base_cdf, color=colors[row], lw=1.5
        )
        axes[0].fill_between(
            M_y[row, :],
            y_base_cdf,
            ys[row, :] + y_base_cdf,
            facecolor=colors[row],
            alpha=0.6,
            edgecolor="white",
            lw=0.5,
        )

        # Store ticks
        y_ticks_pdf.append(y_base_pdf)
        y_ticks_cdf.append(y_base_cdf)

        # Format label
        if condition_values is not None:
            val = (
                condition_values[row].item()
                if torch.is_tensor(condition_values[row])
                else condition_values[row]
            )
            y_labels.append(f"$i_{row}$: $x={val:.2f}$")
        else:
            y_labels.append(f"Idx {row}")

    # Reverse ticks/labels back to normal order (since we plotted in reverse)
    y_ticks_pdf.reverse()
    y_ticks_cdf.reverse()
    y_labels.reverse()

    # Apply ticks and labels
    axes[0].set_yticks(y_ticks_cdf)
    axes[0].set_yticklabels(y_labels, fontsize=9)
    axes[1].set_yticks(y_ticks_pdf)
    axes[1].set_yticklabels(y_labels, fontsize=9)

    axes[0].set_title("CDF Slices")
    axes[1].set_title("PDF Slices")
    for ax in axes:
        ax.set_xlabel("$M(y)$")
        ax.grid(axis="x", alpha=0.3)

    return fig


def _plot_waterfall(
    n, M_y, m_y, ys, colors, metalog_coeffs, fig_suptitle, condition_values
):
    """Helper to plot 3D Waterfall slices with filled in-between regions."""
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"{fig_suptitle} (3D Waterfall)", y=1.05)

    ax_cdf = fig.add_subplot(121, projection="3d")
    ax_pdf = fig.add_subplot(122, projection="3d")

    # Prepare labels for the Y-axis (Condition Index)
    y_ticks = list(range(n))
    y_labels = []
    for row in range(n):
        if condition_values is not None:
            val = (
                condition_values[row].item()
                if torch.is_tensor(condition_values[row])
                else condition_values[row]
            )
            y_labels.append(f"$i_{{{row}}}$: $x={val:.2f}$")
        else:
            y_labels.append(f"Idx {row}")

    for ax in (ax_cdf, ax_pdf):
        ax.view_init(elev=20, azim=-45)
        ax.set_xlabel("\n$M(y)$")
        ax.set_ylabel("\nCondition")

        # Transparent panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Faded 3D grid lines workaround
        grid_color = (0.8, 0.8, 0.8, 0.2)
        ax.xaxis._axinfo["grid"]["color"] = grid_color
        ax.yaxis._axinfo["grid"]["color"] = grid_color
        ax.zaxis._axinfo["grid"]["color"] = grid_color

        # Apply custom ticks
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)

    ax_cdf.set_title("CDF Slices")
    ax_cdf.set_zlabel("$y$")
    ax_pdf.set_title("PDF Slices")
    ax_pdf.set_zlabel("$m(y)$")

    # 1. Collect valid rows first to prevent bridging over skipped/infeasible rows
    valid_rows = []
    for row in range(n):
        if is_feasible_metalog(metalog_coeffs[row].unsqueeze(0)):
            valid_rows.append(row)

    # 2. Plot the solid 3D lines (original waterfall behavior)
    for row in valid_rows:
        y_index = np.full_like(M_y[row, :], row)
        ax_pdf.plot(M_y[row, :], y_index, m_y[row, :], color=colors[row], lw=2)
        ax_cdf.plot(M_y[row, :], y_index, ys[row, :], color=colors[row], lw=2)

    # 3. Fill the regions between consecutive valid slices
    fill_alpha = 0.25  # Adjust lower alpha here
    for i in range(len(valid_rows) - 1):
        r1 = valid_rows[i]
        r2 = valid_rows[i + 1]

        # Convert the RGBA color tuple to a list, modify alpha, and convert back
        fill_color = list(colors[r1])
        fill_color[3] = fill_alpha
        fill_color = tuple(fill_color)

        # Create 2D arrays (shape: 2 x N) to define the ribbon surface
        X_pdf = np.vstack([M_y[r1, :], M_y[r2, :]])
        Y_pdf = np.vstack(
            [np.full_like(M_y[r1, :], r1), np.full_like(M_y[r2, :], r2)]
        )
        Z_pdf = np.vstack([m_y[r1, :], m_y[r2, :]])

        X_cdf = np.vstack([M_y[r1, :], M_y[r2, :]])
        Z_cdf = np.vstack([ys[r1, :], ys[r2, :]])

        # Plot the connecting surfaces
        ax_pdf.plot_surface(
            X_pdf, Y_pdf, Z_pdf, color=fill_color, shade=False, edgecolor="none"
        )
        ax_cdf.plot_surface(
            X_cdf, Y_pdf, Z_cdf, color=fill_color, shade=False, edgecolor="none"
        )

    return fig


def _plot_superimposed(
    n, M_y, m_y, ys, colors, metalog_coeffs, fig_suptitle, orig_samples
):
    """Helper to plot superimposed lines."""
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 4), constrained_layout=True
    )
    fig.suptitle(fig_suptitle, y=1.05)

    if orig_samples is not None:
        axes[1].hist(
            orig_samples.cpu(),
            density=True,
            bins="auto",
            histtype="stepfilled",
            alpha=0.15,
            color="black",
            label="Data",
        )

    for row in range(n):
        if not is_feasible_metalog(metalog_coeffs[row].unsqueeze(0)):
            continue

        line_color = colors[row]
        line_label = f"$M^{{({row})}}$"

        axes[0].plot(
            M_y[row, :], ys[row, :], color=line_color, label=line_label
        )
        axes[1].plot(
            M_y[row, :], m_y[row, :], color=line_color, label=line_label
        )

    for i, title in enumerate(["CDF", "PDF"]):
        axes[i].set_xlabel("$M(y)$")
        axes[i].set_ylabel("$y$" if i == 0 else "$m(y)$")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(title)

    if n <= 15:
        axes[1].legend(loc="best", fontsize="small")

    return fig


def _plot_rows(
    n,
    term,
    M_y,
    m_y,
    ys,
    colors,
    metalog_coeffs,
    fig_suptitle,
    orig_samples,
    metalog_inverse_cdf,
    bounds,
    is_bounded_metalog,
    mean,
    std,
    safe_format_fn,
):
    """Helper to plot separate rows per term (your original logic)."""
    fig = plt.figure(constrained_layout=True, figsize=(12, 4 * n))
    subfigs = fig.subfigures(nrows=n, ncols=1) if n > 1 else None
    fig.suptitle(fig_suptitle, y=1.02)

    for row in range(n):
        is_feasible = is_feasible_metalog(metalog_coeffs[row].unsqueeze(0))
        current_subfig = subfigs[row] if n > 1 else fig
        current_axes = current_subfig.subplots(nrows=1, ncols=2)

        if not is_feasible:
            for a in current_axes:
                a.remove()
            ax = current_subfig.subplots(1, 1)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{term}-Metalog not feasible",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        if not is_bounded_metalog:
            row_title = f"{term}-Term ($\\mu={safe_format_fn(mean[row].item())}$, $\\sigma={safe_format_fn(std[row].item())}$)"
        else:
            row_title = f"{term}-Term Metalog"

        if n > 1:
            current_subfig.suptitle(row_title)
        else:
            current_axes[0].set_title("CDF " + row_title)

        ax_cdf, ax_pdf = current_axes[0], current_axes[1]
        line_color = colors[row]

        ax_cdf.plot(M_y[row, :], ys[row, :], color=line_color)

        # Original scatter point logic
        if metalog_inverse_cdf is not None:
            samples_scatter = metalog_inverse_cdf[row, :].cpu()
            y_scatter = make_probs(samples_scatter.shape[0], device="cpu")
            if is_bounded_metalog:
                samples_scatter = torch.cat(
                    [
                        torch.tensor(
                            [bounds[0]], device=samples_scatter.device
                        ),
                        samples_scatter,
                        torch.tensor(
                            [bounds[1]], device=samples_scatter.device
                        ),
                    ]
                )
                y_scatter = torch.cat(
                    [
                        torch.tensor([0.0], device=y_scatter.device),
                        y_scatter,
                        torch.tensor([1.0], device=y_scatter.device),
                    ]
                )
            ax_cdf.scatter(
                samples_scatter,
                y_scatter,
                color="dimgray",
                s=15,
                label="Metalog Data",
            )

        ax_pdf.plot(M_y[row, :], m_y[row, :], color=line_color)

        if orig_samples is not None:
            ax_pdf.hist(
                orig_samples.cpu(),
                density=True,
                bins="auto",
                histtype="stepfilled",
                alpha=0.2,
                label="Data",
            )
        elif metalog_inverse_cdf is not None:
            ax_pdf.hist(
                metalog_inverse_cdf[row].cpu(),
                density=True,
                bins="fd",
                histtype="stepfilled",
                alpha=0.2,
                label="Quantiles",
            )

        ax_cdf.set_xlabel("$M(y)$")
        ax_cdf.set_ylabel("$y$")
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_title("CDF")

        ax_pdf.set_xlabel("$M(y)$")
        ax_pdf.set_ylabel("$m(y)$")
        ax_pdf.grid(True, alpha=0.3)
        ax_pdf.set_title("PDF")

    return fig


@torch.no_grad
def plot_metalog_coefficients(
    metalog_coeffs: torch.Tensor,
    bounds: tuple[float, float] | None = None,
    metalog_inverse_cdf: torch.Tensor | None = None,
    fig_suptitle: str = "Metalogs",
    orig_samples: torch.Tensor = None,
    plot_type: Literal["rows", "superimposed", "ridge", "waterfall"] = "rows",
    condition_values: list[float] | torch.Tensor | None = None,
) -> plt.Figure:
    """
    Plots given metalog coefficients.

    Args:
        - metalog_coeffs (torch.Tensor): Metalog coefficients in a [n, term] shaped tensor.
        - bounds (float | None, optional): Lower and upper bound. Defaults to None.
        - metalog_inverse_cdf (torch.Tensor, optional): Quantiles that formed the metalog coefficients. Default to None.
        - fig_suptitle (str, optional): Suptitle of the plot. Defaults to "Metalogs".
        - orig_samples (torch.Tensor, optional): Original data samples for histogram background.
        - plot_type (str, optional): Options are "rows", "superimposed", "ridge", "waterfall". Defaults to "rows".
    """
    metalog_coeffs = metalog_coeffs.cpu()
    n = metalog_coeffs.size(0)
    term = metalog_coeffs.size(1)

    # Perform all calculations upfront
    y = make_probs(1000, device="cpu")
    ys = y.repeat(n, 1)
    unsqueezed_metalog_coeffs = metalog_coeffs.unsqueeze(-1)

    mean = None
    std = None

    def safe_format(val):
        return "undefined" if math.isnan(val) else f"{val:.2f}"

    is_bounded_metalog = bounds is not None
    if is_bounded_metalog:
        M_y, unbounded_My = eval_bounded_metalog_quantile(
            unsqueezed_metalog_coeffs, bounds[0], bounds[1], y
        )
        m_y = eval_bounded_metalog_pdf(
            unsqueezed_metalog_coeffs, bounds[0], bounds[1], y, M_y=unbounded_My
        )
    else:
        M_y = eval_unbounded_metalog_quantile(unsqueezed_metalog_coeffs, y)
        m_y = eval_unbounded_metalog_pdf(unsqueezed_metalog_coeffs, y)
        mean = mean_unbounded_metalog(metalog_coeffs, device="cpu")
        variance = variance_unbounded_metalog(metalog_coeffs, device="cpu")
        std = torch.sqrt(variance).cpu()

    # Setup Colors
    cmap = cm.get_cmap("GnBu")
    color_indices = np.linspace(0.3, 1.0, n)
    colors = [cmap(i) for i in color_indices]

    if plot_type == "ridge":
        return _plot_ridge(
            n,
            M_y,
            m_y,
            ys,
            colors,
            metalog_coeffs,
            fig_suptitle,
            condition_values,
        )

    elif plot_type == "waterfall":
        return _plot_waterfall(
            n,
            M_y,
            m_y,
            ys,
            colors,
            metalog_coeffs,
            fig_suptitle,
            condition_values,
        )

    elif plot_type == "superimposed":
        return _plot_superimposed(
            n, M_y, m_y, ys, colors, metalog_coeffs, fig_suptitle, orig_samples
        )

    else:
        # Defaults to "rows"
        return _plot_rows(
            n,
            term,
            M_y,
            m_y,
            ys,
            colors,
            metalog_coeffs,
            fig_suptitle,
            orig_samples,
            metalog_inverse_cdf,
            bounds,
            is_bounded_metalog,
            mean,
            std,
            safe_format,
        )


@torch.no_grad
def plot_marginal_conditional_waterfall(
    marginal_coeffs: torch.Tensor,
    conditional_coeffs: torch.Tensor,
    bounds: tuple[float, float] | None = None,
    fig_suptitle: str = "Marginal-Conditional Metalogs",
    condition_values: list[float] | torch.Tensor | None = None,
    marg_x_anchor_val: float | None = None,
    scale_to_joint: bool = False,
) -> plt.Figure:
    """
    Plots a 3D Waterfall of conditional metalog slices, and plots the
    marginal metalog perpendicularly on the back-left bounding wall.

    Args:
        scale_to_joint (bool): If True, multiplies the conditional PDF slices
        by the marginal density at that condition, visualizing slices of the joint PDF.
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    marginal_coeffs = marginal_coeffs.cpu()
    conditional_coeffs = conditional_coeffs.cpu()
    n = conditional_coeffs.size(0)

    # Ensure marginal_coeffs is shaped [1, term]
    if marginal_coeffs.dim() == 1:
        marginal_coeffs = marginal_coeffs.unsqueeze(0)

    # 1. Setup Probabilities
    y = make_probs(1000, device="cpu")
    ys_cond = y.repeat(n, 1)

    # 2. Evaluate Marginal and Conditional Metalogs
    unsqueezed_marg = marginal_coeffs.unsqueeze(-1)
    unsqueezed_cond = conditional_coeffs.unsqueeze(-1)

    is_bounded_metalog = bounds is not None
    if is_bounded_metalog:
        M_y_cond, unbounded_My_cond = eval_bounded_metalog_quantile(
            unsqueezed_cond, bounds[0], bounds[1], y
        )
        m_y_cond = eval_bounded_metalog_pdf(
            unsqueezed_cond, bounds[0], bounds[1], y, M_y=unbounded_My_cond
        )

        M_y_marg, unbounded_My_marg = eval_bounded_metalog_quantile(
            unsqueezed_marg, bounds[0], bounds[1], y
        )
        m_y_marg = eval_bounded_metalog_pdf(
            unsqueezed_marg, bounds[0], bounds[1], y, M_y=unbounded_My_marg
        )
    else:
        M_y_cond = eval_unbounded_metalog_quantile(unsqueezed_cond, y)
        m_y_cond = eval_unbounded_metalog_pdf(unsqueezed_cond, y)

        M_y_marg = eval_unbounded_metalog_quantile(unsqueezed_marg, y)
        m_y_marg = eval_unbounded_metalog_pdf(unsqueezed_marg, y)

    # 3. Setup the Figure & Axes
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"{fig_suptitle}", y=1.05)
    ax_cdf = fig.add_subplot(121, projection="3d")
    ax_pdf = fig.add_subplot(122, projection="3d")

    # Map the Y-axis to the true condition values (X1 space)
    if condition_values is not None:
        y_positions = [
            (cv.item() if torch.is_tensor(cv) else cv)
            for cv in condition_values
        ]
        y_ticks = y_positions
        y_labels = [f"$x_1={v:.2f}$" for v in y_positions]
    else:
        y_positions = list(range(n))
        y_ticks = y_positions
        y_labels = [f"Idx {i}" for i in range(n)]

    cmap = cm.get_cmap("GnBu")
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, n)]

    for ax in (ax_cdf, ax_pdf):
        ax.view_init(elev=20, azim=-45)
        ax.set_box_aspect((1, 1.0, 0.8))

        ax.set_xlabel("\n$M(y)$ (Conditional, $X_2$)")
        ax.set_ylabel("\nCondition ($X_1$)")
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        grid_color = (0.8, 0.8, 0.8, 0.6)
        ax.xaxis._axinfo["grid"]["color"] = grid_color
        ax.yaxis._axinfo["grid"]["color"] = grid_color
        ax.zaxis._axinfo["grid"]["color"] = grid_color

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)

    ax_cdf.set_title("CDF Slices")
    ax_cdf.set_zlabel("$y$")
    ax_pdf.set_title("Joint PDF Slices" if scale_to_joint else "PDF Slices")
    ax_pdf.set_zlabel("$f(x_1, x_2)$" if scale_to_joint else "$m(y)$")

    # --- Extract Marginal Arrays Early ---
    marg_M = (
        M_y_marg.squeeze().cpu().numpy()
        if torch.is_tensor(M_y_marg)
        else M_y_marg.squeeze()
    )
    marg_m = (
        m_y_marg.squeeze().cpu().numpy()
        if torch.is_tensor(m_y_marg)
        else m_y_marg.squeeze()
    )
    marg_y = y.squeeze().cpu().numpy() if torch.is_tensor(y) else y.squeeze()

    # --- Scale Conditionals to Joint PDF ---
    plot_m_y_cond = (
        m_y_cond.clone() if torch.is_tensor(m_y_cond) else m_y_cond.copy()
    )
    valid_rows = [
        r
        for r in range(n)
        if is_feasible_metalog(conditional_coeffs[r].unsqueeze(0))
    ]

    if scale_to_joint:
        for row in valid_rows:
            # Find the marginal density at this specific X1 condition using interpolation
            marginal_density_at_cond = np.interp(
                y_positions[row], marg_M, marg_m
            )
            plot_m_y_cond[row, :] = (
                plot_m_y_cond[row, :] * marginal_density_at_cond
            )

    # 4. Plot Conditionals & Fills along the true Y-axis
    for row in valid_rows:
        y_val = y_positions[row]
        y_array = np.full_like(M_y_cond[row, :], y_val)

        # Use our scaled (or unscaled) plotting array
        ax_pdf.plot(
            M_y_cond[row, :],
            y_array,
            plot_m_y_cond[row, :],
            color=colors[row],
            lw=6,
        )
        ax_cdf.plot(
            M_y_cond[row, :], y_array, ys_cond[row, :], color=colors[row], lw=6
        )

    fill_alpha = 0.25
    for i in range(len(valid_rows) - 1):
        r1, r2 = valid_rows[i], valid_rows[i + 1]
        fill_color = list(colors[r1])
        fill_color[3] = fill_alpha
        fill_color = tuple(fill_color)

        X_pdf = np.vstack([M_y_cond[r1, :], M_y_cond[r2, :]])
        Y_pdf = np.vstack(
            [
                np.full_like(M_y_cond[r1, :], y_positions[r1]),
                np.full_like(M_y_cond[r2, :], y_positions[r2]),
            ]
        )
        Z_pdf = np.vstack([plot_m_y_cond[r1, :], plot_m_y_cond[r2, :]])
        Z_cdf = np.vstack([ys_cond[r1, :], ys_cond[r2, :]])

        ax_pdf.plot_surface(
            X_pdf, Y_pdf, Z_pdf, color=fill_color, shade=False, edgecolor="none"
        )
        ax_cdf.plot_surface(
            X_pdf, Y_pdf, Z_cdf, color=fill_color, shade=False, edgecolor="none"
        )

    # 5. Plot Marginal perpendicularly on the back-left wall
    if marg_x_anchor_val is not None:
        x_min = marg_x_anchor_val
    elif torch.is_tensor(M_y_cond):
        x_min = M_y_cond[valid_rows, :].min().item()
    else:
        x_min = np.min(M_y_cond[valid_rows, :])

    marg_x_anchor = np.full_like(marg_M, x_min)

    if condition_values is not None:
        marg_Y = marg_M
    else:
        marg_Y = np.interp(marg_M, (marg_M.min(), marg_M.max()), (0, n - 1))

    marginal_color = "#424c55"

    ax_pdf.plot(marg_x_anchor, marg_Y, marg_m, color=marginal_color, lw=8)
    ax_cdf.plot(marg_x_anchor, marg_Y, marg_y, color=marginal_color, lw=8)

    # Labels
    peak_idx = np.argmax(marg_m)
    ax_pdf.text(
        marg_x_anchor[peak_idx],
        marg_Y[peak_idx],
        marg_m[peak_idx] + 0.03,
        "Marginal",
        color=marginal_color,
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    ax_cdf.text(
        marg_x_anchor[-1],
        marg_Y[-1],
        marg_y[-1] + 0.1,
        "Marginal",
        color=marginal_color,
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    for ax in (ax_cdf, ax_pdf):
        x_limits = ax.get_xlim()
        ax.set_xlim(
            x_limits[0] - (x_limits[1] - x_limits[0]) * 0.05, x_limits[1]
        )

    return fig


@torch.no_grad
def plot_marginal_conditional_results(
    sRec_orig: SampleRecord,
    orig_pdf_func,
    device,
    metalog_term,
    metalog_marginal,
    marginal_label,
    conditional_label,
    scatter_size,
    cmap,
    output_dir,
    samples_hist,
    metalogs_dict,
    max_num_terms,
    marginal_bounds: tuple[float, float] | None = None,
    conditional_bounds: tuple[float, float] | None = None,
    logit_transform: bool = False,
):
    bins = metalogs_dict["bins"]
    marginal_metalog_err = metalogs_dict["marginal_score"]
    marginal_metalogs = metalogs_dict["marginal"]
    marginal_metalog_dist_inverse_cdf = metalogs_dict["marginal_inverse_cdf"]
    cond_metalog_dist_inverse_cdf = metalogs_dict["conditional_inverse_cdf"]
    orig_samples_3d = sRec_orig.get("samples")
    orig_samples_pdf = sRec_orig.get("pdf")
    orig_unif2 = sRec_orig.get("unif2")

    # ensure output directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metalogs_filename = f"{output_dir}/metalogs.pt"
    bivariate_metalog_histogram_filename = f"{output_dir}/metalog_histogram.pt"

    if logit_transform:
        logit_marginal_filename = f"{output_dir}/logit_marginal.pt"
        logit_cond_filename = f"{output_dir}/logit_conditional.pt"

    plots_dir = f"{output_dir}/plots"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting original samples to 2D.")
    orig_marginal_conditional_samples = unitvec_to_cos_theta_phi(
        orig_samples_3d.to(device)
    )
    if metalog_marginal == "phi":
        orig_marginal_conditional_samples = (
            orig_marginal_conditional_samples.flip(1)
        )

    bins_marginal, bins_conditional = bins
    marginal_samples = orig_marginal_conditional_samples[:, 0]
    conditional_samples = orig_marginal_conditional_samples[:, 1]
    orig_marginal_conditional_samples_cpu = (
        orig_marginal_conditional_samples.cpu()
    )
    orig_marginal_samples_cpu = marginal_samples.cpu()
    orig_cond_samples_cpu = conditional_samples.cpu()
    bins_orig = bins
    bins_marginal_orig, bins_conditional_orig = bins_orig
    hist_orig = samples_hist

    if logit_transform:
        # if logit transform applied, the provided bins are according to the
        # logit transformed samples, hence for plotting we need to recompute it
        # according to the original (non-transformed) samples
        logging.info(f"Compute bins and 2D histogram for the original samples.")
        bins_marginal_orig = torch.from_numpy(
            np.histogram_bin_edges(
                orig_marginal_samples_cpu, bins="auto", range=marginal_bounds
            )
        )
        bins_conditional_orig = torch.from_numpy(
            np.histogram_bin_edges(
                orig_cond_samples_cpu,
                bins="auto",
                range=conditional_bounds,
            )
        )
        bins_orig = [bins_marginal_orig, bins_conditional_orig]
        hist_orig, _ = torch.histogramdd(
            orig_marginal_conditional_samples.cpu(),
            bins=bins_orig,
            density=True,
        )

        # apply logit transform to the original samples
        marginal_samples, conditional_samples = (
            lt.logit_transform_bivariate_samples(
                marginal_bounds,
                conditional_bounds,
                marginal_samples,
                conditional_samples,
            )
        )
        orig_marginal_conditional_samples = torch.stack(
            (marginal_samples, conditional_samples), dim=1
        )

        # if we work with a bounded metalog we have to update the bounds
        if marginal_bounds is not None and conditional_bounds is not None:
            marginal_bounds = lt.logit_transform_bounds(marginal_bounds)
            conditional_bounds = lt.logit_transform_bounds(conditional_bounds)

        """Plot logit transformed samples"""
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle("Histograms of the Logit Transformed Samples")
        # 1D Histogram of the original marginal Samples
        axs[0].hist(
            marginal_samples.cpu().numpy(),
            bins=bins_marginal,
            density=True,
        )
        axs[0].set_title(f"1D Histogram of the {marginal_label} Samples")
        axs[0].set_xlabel(marginal_label)
        axs[0].set_ylabel("Density")
        axs[0].grid(True)

        # 1D Histogram of the original conditional Samples
        axs[1].hist(
            conditional_samples.cpu().numpy(),
            bins=bins_conditional,
            density=True,
        )
        axs[1].set_title(f"1D Histogram of the {conditional_label} Samples")
        axs[1].set_xlabel(conditional_label)
        axs[1].set_ylabel("Density")
        axs[1].grid(True)

        im1 = axs[2].imshow(
            samples_hist.T.cpu().numpy(),
            origin="lower",
            cmap="inferno",
        )

        fig.colorbar(im1, ax=axs[2], label="Density")
        axs[2].set_xlabel(marginal_label)
        axs[2].set_ylabel(conditional_label)
        axs[2].set_title("2D Histogram")
        fig.savefig(f"{plots_dir}/hist_logit_samples.png", dpi=fig.dpi)
        plt.close(fig)

    marginal_samples_cpu = marginal_samples.cpu()
    metalog_res_marg = marginal_metalog_dist_inverse_cdf.shape[0]

    """Plot marginal metalogs"""
    x_offset = 0.1
    x_lim = [
        marginal_samples_cpu.min() - x_offset,
        marginal_samples_cpu.max() + x_offset,
    ]
    # modify xlim iff bounds are specified
    if marginal_bounds is not None:
        x_lim = [marginal_bounds[0] - x_offset, marginal_bounds[1] + x_offset]

    fig = plot_metalogs(
        marginal_metalog_dist_inverse_cdf,
        metalog_res_marg,
        f"Marginal Metalog Distribution {marginal_label}",
        x_lim,
        marginal_metalogs,
        marginal_metalog_err,
        torch.arange(3, max_num_terms),
        bounds=marginal_bounds,
        probs=make_probs(metalog_res_marg, device=device),
        orig_samples=marginal_samples_cpu,
    )
    fig.savefig(f"{plots_dir}/marginal_mls.png", dpi=fig.dpi)
    plt.close(fig)

    """Now plotting results for specified metalog term"""
    (
        marginal_metalog_coeffs,
        cond_metalog_coeffs,
        marginal_metalog_dist_inverse_cdf,
        _,
        __,
    ) = init_metalogs(metalogs_filename, metalog_term, marginal_bounds, device)

    fig = plot_metalog_coefficients(
        cond_metalog_coeffs,
        bounds=conditional_bounds,
        metalog_inverse_cdf=cond_metalog_dist_inverse_cdf,
        fig_suptitle=f"{metalog_term}-Term Conditional Metalogs",
        orig_samples=conditional_samples,
    )
    fig.savefig(f"{plots_dir}/conditional_mls.png", dpi=fig.dpi)
    plt.close(fig)

    # #### Sampling bivariate metalog
    logging.info("Sampling from bivariate metalog.")

    sRec = sample_bivariate_metalog(
        marginal_metalog_coeffs,
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs.to(device),
        orig_unif2,
        bounds_marginal=marginal_bounds,
        bounds_conditional=conditional_bounds,
    )
    gen_samples = sRec.get("samples")

    # inverse logit transform generated samples
    gen_samples_per_pdf = gen_samples.clone()
    logit_transform_marginal, logit_transform_cond = None, None
    if logit_transform:
        # save these for later use
        marginal_gen_samples_logit = gen_samples_per_pdf[:, 0].clone()
        cond_gen_samples_logit = gen_samples_per_pdf[:, 1].clone()

        logging.info(
            "Applying inverse logit transform to the generated samples."
        )
        logit_transform_marginal = torch.load(
            logit_marginal_filename, weights_only=True
        )
        gen_samples[:, 0] = lt.inverse_logit_samples(
            marginal_gen_samples_logit, logit_transform_marginal
        )
        logit_transform_cond = torch.load(
            logit_cond_filename, weights_only=True
        )
        gen_samples[:, 1] = lt.inverse_logit_samples(
            cond_gen_samples_logit, logit_transform_cond
        )

    metalog_samples_3d = cos_theta_phi_to_unitvec(
        gen_samples if metalog_marginal == "cos_theta" else gen_samples.flip(1)
    ).cpu()
    gen_samples_cpu = unitvec_to_cos_theta_phi(metalog_samples_3d).cpu()
    if metalog_marginal == "phi":
        gen_samples_cpu = gen_samples_cpu.flip(1)

    # First row of plots (samples from the normal distribution)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # 1D Histogram of X Samples
    axs[0, 0].hist(
        orig_marginal_samples_cpu.numpy(), bins=bins_marginal_orig, density=True
    )
    axs[0, 0].set_title(f"1D Histogram of {marginal_label} Samples")
    axs[0, 0].set_xlabel(marginal_label)
    axs[0, 0].set_ylabel("Density")
    axs[0, 0].grid(True)

    # 1D Histogram of Y Samples
    axs[0, 1].hist(
        orig_cond_samples_cpu.numpy(), bins=bins_conditional_orig, density=True
    )
    axs[0, 1].set_title(f"1D Histogram of {conditional_label} Samples")
    axs[0, 1].set_xlabel(conditional_label)
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].grid(True)

    # 2D Histogram for the original samples
    im1 = axs[0, 2].imshow(hist_orig.T, origin="lower", cmap=cmap)
    fig.colorbar(im1, ax=axs[0, 2], label="Density")
    axs[0, 2].set_xlabel(marginal_label)
    axs[0, 2].set_ylabel(conditional_label)
    axs[0, 2].set_title("Original 2D Histogram")

    # 1D Histogram from the generated marginal asis
    axs[1, 0].hist(gen_samples_cpu[:, 0], bins=bins_marginal_orig, density=True)
    axs[1, 0].set_title("1D Histogram of the Marginal Metalog Distribution")
    axs[1, 0].set_xlabel(marginal_label)
    axs[1, 0].set_ylabel("Density")
    axs[1, 0].grid(True)

    # 1D Histogram from the conditional axis
    axs[1, 1].hist(
        gen_samples_cpu[:, 1], bins=bins_conditional_orig, density=True
    )
    axs[1, 1].set_title("1D Histogram of the Conditional Metalog Distribution")
    axs[1, 1].set_xlabel(conditional_label)
    axs[1, 1].set_ylabel("Density")
    axs[1, 1].grid(True)

    # compute histogram of the generated samples
    hist_gen, _ = torch.histogramdd(
        gen_samples_cpu, bins=bins_orig, density=True
    )

    # save histogram for later use, using original edges
    torch.save(
        {
            "histogram": hist_gen,
            "xedges": bins_marginal_orig,
            "yedges": bins_conditional_orig,
        },
        bivariate_metalog_histogram_filename,
    )

    # 2D Histogram of Bivariate Metalog Distribution
    im2 = axs[1, 2].imshow(hist_gen.T, origin="lower", cmap=cmap)
    fig.colorbar(im2, ax=axs[1, 2], label="Density")
    axs[1, 2].set_xlabel(marginal_label)
    axs[1, 2].set_ylabel(conditional_label)
    axs[1, 2].set_title("2D Histogram of Bivariate Metalog Distribution")

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/orig_bivariate_ml_hist.png", dpi=fig.dpi)
    plt.close(fig)

    # Plotting both scatter plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(
        gen_samples_cpu[:, 0],
        gen_samples_cpu[:, 1],
        alpha=0.6,
        s=scatter_size,
    )
    axs[0].set_xlabel(marginal_label)
    axs[0].set_ylabel(conditional_label)
    axs[0].set_title("Generated metalog samples")
    axs[0].set_xlim(
        [
            gen_samples_cpu[:, 0].min().cpu() - 0.1,
            gen_samples_cpu[:, 0].max().cpu() + 0.1,
        ]
    )
    axs[0].set_ylim(
        [
            gen_samples_cpu[:, 1].min().cpu() - 0.1,
            gen_samples_cpu[:, 1].max().cpu() + 0.1,
        ]
    )

    axs[1].scatter(
        orig_marginal_samples_cpu,
        orig_cond_samples_cpu,
        alpha=0.6,
        s=scatter_size,
    )
    axs[1].set_xlabel(marginal_label)
    axs[1].set_ylabel(conditional_label)
    axs[1].set_title("Original samples")
    axs[1].set_xlim(
        [
            orig_marginal_samples_cpu.min() - 0.1,
            orig_marginal_samples_cpu.max() + 0.1,
        ]
    )
    axs[1].set_ylim(
        [
            orig_cond_samples_cpu.min() - 0.1,
            orig_cond_samples_cpu.max() + 0.1,
        ]
    )

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/orig_ml_samples_2d.png", dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        metalog_samples_3d[:, 0],
        metalog_samples_3d[:, 1],
        metalog_samples_3d[:, 2],
        c="k",
        alpha=0.5,
        label="Metalog Samples",
        s=scatter_size,
    )
    ax.scatter(
        orig_samples_3d[:, 0],
        orig_samples_3d[:, 1],
        orig_samples_3d[:, 2],
        c="r",
        alpha=0.3,
        label="Original Samples",
        s=scatter_size,
    )
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Generated Metalog Samples in 3D", y=1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/orig_ml_samples_3d.png", dpi=fig.dpi)
    plt.close(fig)

    # Create a scatter plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sc0 = axs[0].scatter(
        metalog_samples_3d[:, 0],
        metalog_samples_3d[:, 1],
        c=metalog_samples_3d[:, 2],
        cmap=cmap,
        s=scatter_size,
    )

    fig.colorbar(sc0, label="Z component")
    axs[0].set_xlabel("X component")
    axs[0].set_ylabel("Y component")
    axs[0].set_title("Generated Samples")

    sc1 = axs[1].scatter(
        orig_samples_3d[:, 0],
        orig_samples_3d[:, 1],
        c=orig_samples_3d[:, 2],
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc1, label="Z component")
    axs[1].set_xlabel("X component")
    axs[1].set_ylabel("Y component")
    axs[1].set_title("Original Samples")

    fig.suptitle("3D vectors as 2D scatter plots")
    fig.savefig(
        f"{plots_dir}/orig_ml_samples_3d_plane.png",
        dpi=fig.dpi,
    )
    plt.close(fig)

    # #### Evaluate bivariate metalog PDF
    logging.info(
        "Evaluating bivariate metalog PDF for the metalog generated samples."
    )
    orig_pdf = orig_pdf_func(metalog_samples_3d.numpy())
    metalog_pdf = sRec.get("pdf").cpu()
    estimated_metalog_pdf = estimate_bivariate_metalog_pdf(
        gen_samples_cpu, bins_marginal_orig, bins_conditional_orig, hist_gen
    )

    # Create a mask for values less than or equal to metalog_pdf
    mean_orig_pdf = orig_pdf.mean()
    mask = metalog_pdf <= orig_pdf.max() + mean_orig_pdf * 0.5

    # Apply the mask to your data and color values
    filtered_gen_samples_x = gen_samples_cpu[:, 0][mask]
    filtered_gen_samples_y = gen_samples_cpu[:, 1][mask]
    filtered_metalog_pdf = metalog_pdf[mask]

    # Count of filtered-out values is simply the sum of the inverted mask
    filtered_out_count = torch.count_nonzero(~mask)
    mean_nonnan_pdf = torch.mean(metalog_pdf[~torch.isnan(metalog_pdf)])
    logging.info(
        f"Mean bivariate metalog pdf: {mean_nonnan_pdf} (excluding non-nan values)."
    )
    logging.info(f"Mean of the original pdf: {mean_orig_pdf}.")

    # Create a scatter plot
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))
    sc0 = axs[0].scatter(
        gen_samples_cpu[:, 0],
        gen_samples_cpu[:, 1],
        c=orig_pdf,
        cmap=cmap,
        s=scatter_size,
    )

    fig.colorbar(sc0, label="PDF")
    axs[0].set_xlabel(marginal_label)
    axs[0].set_ylabel(conditional_label)
    axs[0].set_title("Using Original PDF")

    # Now plot only the filtered samples
    sc1 = axs[1].scatter(
        filtered_gen_samples_x,
        filtered_gen_samples_y,
        c=filtered_metalog_pdf,
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc1, label="PDF")
    axs[1].set_xlabel(marginal_label)
    axs[1].set_ylabel(conditional_label)
    axs[1].set_title(
        f"Using Bivariate Metalog PDF (# samples filtr.={filtered_out_count})"
    )

    sc2 = axs[2].scatter(
        gen_samples_cpu[:, 0],
        gen_samples_cpu[:, 1],
        c=estimated_metalog_pdf,
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc2, label="PDF")
    axs[2].set_xlabel(marginal_label)
    axs[2].set_ylabel(conditional_label)
    axs[2].set_title("Using Metalog Histogram")

    fig.suptitle("PDF of Generated samples")
    fig.savefig(
        f"{plots_dir}/orig_ml_pdf_gen_samples.png",
        dpi=fig.dpi,
    )
    plt.close(fig)

    logging.info("Evaluating bivariate metalog PDF for the original samples.")
    orig_pdf = orig_samples_pdf
    metalog_pdf = eval_bivariate_metalog_pdf(
        marginal_metalog_coeffs,
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs.to(device),
        SampleRecord(
            samples=orig_marginal_conditional_samples,
            unif2=None,
            pdf=None,
        ),
        bounds_marginal=marginal_bounds,
        bounds_conditional=conditional_bounds,
        logit_transform_marginal=logit_transform_marginal,
        logit_transform_cond=logit_transform_cond,
    ).cpu()
    estimated_metalog_pdf = estimate_bivariate_metalog_pdf(
        orig_marginal_conditional_samples_cpu,
        bins_marginal_orig,
        bins_conditional_orig,
        hist_gen,
    )

    # Create a mask for values less than or equal to metalog_pdf
    mean_orig_pdf = orig_pdf.mean()
    mask = metalog_pdf <= orig_pdf.max() + mean_orig_pdf * 0.5

    # Apply the mask to your data and color values
    filtered_gen_samples_x = orig_marginal_samples_cpu[mask]
    filtered_gen_samples_y = orig_cond_samples_cpu[mask]
    filtered_metalog_pdf = metalog_pdf[mask]

    # Count of filtered-out values is simply the sum of the inverted mask
    filtered_out_count = torch.count_nonzero(~mask)
    mean_nonnan_pdf = torch.mean(metalog_pdf[~torch.isnan(metalog_pdf)])
    logging.info(
        f"Mean bivariate metalog PDF: {mean_nonnan_pdf} (excluding non-nan values)."
    )
    logging.info(f"Mean of the original PDF: {mean_orig_pdf}.")

    # Create a scatter plot
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))
    sc0 = axs[0].scatter(
        orig_marginal_samples_cpu,
        orig_cond_samples_cpu,
        c=orig_pdf,
        cmap=cmap,
        s=scatter_size,
    )

    fig.colorbar(sc0, label="PDF")
    axs[0].set_xlabel(marginal_label)
    axs[0].set_ylabel(conditional_label)
    axs[0].set_title("Using Original PDF")

    # Now plot only the filtered samples
    sc1 = axs[1].scatter(
        filtered_gen_samples_x,
        filtered_gen_samples_y,
        c=filtered_metalog_pdf,
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc1, label="PDF")
    axs[1].set_xlabel(marginal_label)
    axs[1].set_ylabel(conditional_label)
    axs[1].set_title(
        f"Using Bivariate Metalog PDF (# samples filtr.={filtered_out_count})"
    )

    sc2 = axs[2].scatter(
        orig_marginal_samples_cpu,
        orig_cond_samples_cpu,
        c=estimated_metalog_pdf,
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc2, label="PDF")
    axs[2].set_xlabel(marginal_label)
    axs[2].set_ylabel(conditional_label)
    axs[2].set_title("Using Metalog Histogram")

    fig.suptitle("PDF of Original samples")
    fig.savefig(
        f"{plots_dir}/orig_ml_pdf_orig_samples.png",
        dpi=fig.dpi,
    )
    plt.close(fig)

    logging.info(f"Plots can be found in {plots_dir}.")


@torch.no_grad
def plot_epfl_metalog_material_results(
    device,
    metalog_term,
    metalog_marginal,
    marginal_label,
    conditional_label,
    scatter_size,
    cmap,
    output_dir,
    make_probs_method,
    metalogs_dict,
    max_num_terms,
):
    bins = metalogs_dict["bins"]
    marginal_metalog_err = metalogs_dict["marginal_score"]
    marginal_metalogs = metalogs_dict["marginal"]
    marginal_metalog_dist_inverse_cdf = metalogs_dict["marginal_inverse_cdf"]
    cond_metalog_dist_inverse_cdf = metalogs_dict["conditional_inverse_cdf"]
    marginal_bounds = metalogs_dict["marginal_bounds"]
    conditional_bounds = metalogs_dict["conditional_bounds"]

    from os.path import abspath

    # ensure output directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bivariate_metalog_histogram_filename = abspath(
        f"{output_dir}/metalog_histogram.pt"
    )

    plots_dir = abspath(f"{output_dir}/plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    metalog_res_marg = marginal_metalog_dist_inverse_cdf.shape[0]

    """Plot marginal metalogs"""
    x_offset = 0.1
    x_lim = [
        marginal_metalog_dist_inverse_cdf.min() - x_offset,
        marginal_metalog_dist_inverse_cdf.max() + x_offset,
    ]
    # modify xlim iff bounds are specified
    if marginal_bounds is not None:
        x_lim = [marginal_bounds[0] - x_offset, marginal_bounds[1] + x_offset]

    fig = plot_metalogs(
        marginal_metalog_dist_inverse_cdf,
        metalog_res_marg,
        f"Marginal Metalog Distribution {marginal_label}",
        x_lim,
        marginal_metalogs,
        marginal_metalog_err,
        torch.arange(3, max_num_terms),
        bounds=marginal_bounds,
        probs=make_probs(
            metalog_res_marg, device=device, method=make_probs_method
        ),
    )
    fig.savefig(f"{plots_dir}/marginal_mls.png", dpi=fig.dpi)
    plt.close(fig)

    """Now plotting results for specified metalog term"""
    marginal_metalog_coeffs = get_metalog_coefficients(
        marginal_metalogs, metalog_term
    )
    cond_metalog_coeffs = get_metalog_coefficients(
        metalogs_dict["conditional"], metalog_term
    )

    fig = plot_metalog_coefficients(
        cond_metalog_coeffs,
        bounds=conditional_bounds,
        metalog_inverse_cdf=cond_metalog_dist_inverse_cdf,
        fig_suptitle=f"{metalog_term}-Term Conditional Metalogs",
    )
    fig.savefig(f"{plots_dir}/conditional_mls.png", dpi=fig.dpi)
    plt.close(fig)

    # #### Sampling bivariate metalog
    logging.info("Sampling from bivariate metalog.")

    num_samples = 2**14
    sRec = sample_bivariate_metalog(
        marginal_metalog_coeffs,
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs.to(device),
        torch.rand(num_samples, 2),
        bounds_marginal=marginal_bounds,
        bounds_conditional=conditional_bounds,
    )
    gen_samples = sRec.get("samples")
    gen_samples_cpu = gen_samples.cpu()

    # First row of plots (samples from the normal distribution)
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))

    # 1D Histogram from the generated marginal asis
    axs[0].hist(gen_samples_cpu[:, 0], bins=bins[0], density=True)
    axs[0].set_title("1D Histogram of the Marginal Metalog Distribution")
    axs[0].set_xlabel(marginal_label)
    axs[0].set_ylabel("Density")
    axs[0].grid(True)

    # 1D Histogram from the conditional axis
    axs[1].hist(gen_samples_cpu[:, 1], bins=bins[1], density=True)
    axs[1].set_title("1D Histogram of the Conditional Metalog Distribution")
    axs[1].set_xlabel(conditional_label)
    axs[1].set_ylabel("Density")
    axs[1].grid(True)

    # compute histogram of the generated samples
    hist_gen, _ = torch.histogramdd(gen_samples_cpu, bins=bins, density=True)

    # save histogram for later use, using original edges
    torch.save(
        {
            "histogram": hist_gen,
            "xedges": bins[0],
            "yedges": bins[1],
        },
        bivariate_metalog_histogram_filename,
    )

    # 2D Histogram of Bivariate Metalog Distribution
    im2 = axs[2].imshow(hist_gen.T, origin="lower", cmap=cmap)
    fig.colorbar(im2, ax=axs[2], label="Density")
    axs[2].set_xlabel(marginal_label)
    axs[2].set_ylabel(conditional_label)
    axs[2].set_title("2D Histogram of Bivariate Metalog Distribution")

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/bivariate_ml_hist.png", dpi=fig.dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.scatter(
        gen_samples_cpu[:, 0],
        gen_samples_cpu[:, 1],
        alpha=0.6,
        s=scatter_size,
    )
    ax.set_xlabel(marginal_label)
    ax.set_ylabel(conditional_label)
    ax.set_title("Generated metalog samples")
    ax.set_xlim(
        [
            gen_samples_cpu[:, 0].min().cpu() - 0.1,
            gen_samples_cpu[:, 0].max().cpu() + 0.1,
        ]
    )
    ax.set_ylim(
        [
            gen_samples_cpu[:, 1].min().cpu() - 0.1,
            gen_samples_cpu[:, 1].max().cpu() + 0.1,
        ]
    )

    fig.tight_layout()
    fig.savefig(f"{plots_dir}/ml_samples_2d.png", dpi=fig.dpi)
    plt.close(fig)

    # #### Evaluate bivariate metalog PDF
    logging.info(
        "Evaluating bivariate metalog PDF for the metalog generated samples."
    )
    metalog_pdf = sRec.get("pdf").cpu()
    estimated_metalog_pdf = estimate_bivariate_metalog_pdf(
        gen_samples_cpu, bins[0], bins[1], hist_gen
    )

    gen_samples_x = gen_samples_cpu[:, 0]
    gen_samples_y = gen_samples_cpu[:, 1]

    # Create a scatter plot
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))
    sc0 = axs[0].imshow(
        metalogs_dict["target_data"],
        cmap=cmap,
    )
    fig.colorbar(sc0, label="PDF")
    axs[0].set_ylabel(marginal_label)
    axs[0].set_xlabel(conditional_label)
    axs[0].set_title("Target Data")

    # Now plot only the filtered samples
    sc1 = axs[1].scatter(
        gen_samples_y,
        gen_samples_x,
        c=metalog_pdf,
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc1, label="PDF")
    axs[1].set_ylabel(marginal_label)
    axs[1].set_xlabel(conditional_label)
    axs[1].set_title(f"Using Bivariate Metalog PDF")

    sc2 = axs[2].scatter(
        gen_samples_cpu[:, 1],
        gen_samples_cpu[:, 0],
        c=estimated_metalog_pdf,
        cmap=cmap,
        s=scatter_size,
    )
    fig.colorbar(sc2, label="PDF")
    axs[2].set_ylabel(marginal_label)
    axs[2].set_xlabel(conditional_label)
    axs[2].set_title("Using Metalog Histogram")

    fig.suptitle("PDF of Generated samples")
    fig.savefig(
        f"{plots_dir}/orig_ml_pdf_gen_samples.png",
        dpi=fig.dpi,
    )
    plt.close(fig)

    logging.info(f"Plots can be found in {plots_dir}.")
