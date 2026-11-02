# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import logging
import os
from pathlib import Path
import torch
from lib.fit.mixture_models.auto_metalog_mixture import AutoMetalogMixture1D
from lib.fit.mixture_models.metalog_mixture_model import MetalogMixtureModel
from lib.fit.mixture_models.spt_metalog_mixture_model import (
    SPTMetalogMixtureModel,
)
import lib.metalogs as ml
from lib.utils.plot_mixtures import (
    plot_mixture_fit,
    plot_best_em_history,
)
from matplotlib.pyplot import close as plt_close
import torch.nn.functional as F

MIX_CONVERGENCE_TOL = 1e-5
RESTARTS_PER_K = 1


@torch.no_grad()
def make_marginal_mixture_conditional_metalogs_samples(
    samples: torch.Tensor,
    device: str,
    min_regularization_term_marginal: float,
    max_regularization_term_marginal: float,
    regularization_term_conditional: float,
    min_metalog_res_marg: int,
    max_metalog_res_marg: int,
    metalog_res_cond: int,
    max_num_terms: int,
    metalog_marginal: str,
    plots_dir: Path,
    marginal_bounds: tuple[float, float] | None = None,
    conditional_bounds: tuple[float, float] | None = None,
    make_probs_method: str = "classic",
    measurement_dir_label: str = "",
    mix_K: int = 2,
    mix_alpha: float = 0.4,
    mix_max_iters: int = 30,
    mixture_model: str = "auto",  # auto | Metalog Mixture Model 1D (mmm),
    mix_metalog_term_marg: int = 5,
    n_conditionals: int = 6,
) -> tuple[torch.Tensor, dict]:
    """
    Makes marginal-mixture-conditional metalogs from samples (e.g., azimuth and eleveation).
    """
    if (marginal_bounds is None) ^ (conditional_bounds is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    samples = samples.to(device)
    x_marg = samples[:, 0].view(-1, 1)
    x_cond = samples[:, 1].view(-1, 1)

    # fit SPT mixture for the marginal samples
    if mixture_model == "auto":
        restarts_per_k = RESTARTS_PER_K
        model = AutoMetalogMixture1D(
            min_k=1,
            max_k=mix_K,
            metalog_term=mix_metalog_term_marg,
            tol=MIX_CONVERGENCE_TOL,
            bounds=marginal_bounds,
            restarts_per_k=restarts_per_k,
            max_iters=mix_max_iters,
            lambda_min=min_regularization_term_marginal,
            lambda_max=max_regularization_term_marginal,
            resolution_min=min_metalog_res_marg,
            resolution_max=max_metalog_res_marg,
            alpha=mix_alpha,
            k_penalty=min_regularization_term_marginal,
            device=device,
            make_probs_method=make_probs_method,
        )
        model_name = "AutoMetalogMixture1D"
    else:
        restarts_per_k = 1
        model = MetalogMixtureModel(
            K=mix_K,
            metalog_term=mix_metalog_term_marg,
            tol=MIX_CONVERGENCE_TOL,
            bounds=marginal_bounds,
            max_iters=mix_max_iters,
            lambda_=min_regularization_term_marginal,
            resolution=min_metalog_res_marg,
            device=device,
            make_probs_method=make_probs_method,
        )
        model_name = "MetalogMixtureModel"

    # fit data
    fit_results = model.fit(x_marg.view(-1))

    plot_enabled = plots_dir is not None
    plot_save_dir = None
    if plot_enabled:
        plot_save_dir = plots_dir / measurement_dir_label
        os.makedirs(plot_save_dir, exist_ok=True)
        fig, _, _ = plot_mixture_fit(model, x_marg.view(-1), show=False)
        fig.savefig(plot_save_dir / f"marginal_mixture_model.png")
        plt_close()

    if mixture_model == "auto":
        model_k = model.best_k
        model_t = model.best_term
        model_l = model.best_lambda
        model_r = model.best_resolution
        model = model.best_model
    else:
        model_k = mix_K
        model_t = mix_metalog_term_marg
        model_l = min_regularization_term_marginal
        model_r = min_metalog_res_marg

    # store mixing and metalog coeffs with zeros padding if necessary
    m_mix_coeffs = model.mixing_coeffs.detach().cpu()
    m_metalog_coeffs = model.metalog_coeffs.detach().cpu()

    # pad zeros to columns
    mixing_coeffs = F.pad(m_mix_coeffs, (0, mix_K - m_mix_coeffs.shape[0]))

    # pad zero rows and columns
    padding_bottom = mix_K - m_metalog_coeffs.shape[0]
    padding_right = mix_metalog_term_marg - model_t
    metalog_coeffs = F.pad(
        m_metalog_coeffs, (0, padding_right, 0, padding_bottom)
    )

    # create a dictionary per future use
    marginal_mixture = {
        "model_name": model_name,
        "K": mix_K,
        "bounds": marginal_bounds,
        "mixing_coeffs": mixing_coeffs,
        "coeffs": metalog_coeffs,
        "best_k": model_k,
        "best_t": model_t,
        "best_l": model_l,
        "best_r": model_r,
    }

    if plot_enabled:
        em_plot_title = f"{model_name} (T={mix_metalog_term_marg}, K={model_k}, {measurement_dir_label}) — EM History"
        (
            fig,
            _,
        ) = plot_best_em_history(
            model, fit_results, title=em_plot_title, show=False
        )
        fig.savefig(plot_save_dir / f"em_hist.png")
        plt_close()

    # get desired probs
    probs = ml.make_probs(n_conditionals, method="classic", device=device)
    smoothed_marginal_quantiles = model.eval_mixture_quantile(probs)

    # evaluate conditional metalogs
    cond_metalog_dists, cond_inv_cdf, cond_scores = (
        ml.eval_conditional_metalog_distributions_kernel(
            smoothed_marginal_quantiles,
            metalog_res_cond,
            x_marg,
            x_cond,
            max_num_terms=max_num_terms,
            bounds=conditional_bounds,
            lambda_=regularization_term_conditional,
            device=device,
        )
    )
    logging.info(
        f"({measurement_dir_label}/wo) Conditional metalog scores: \n{cond_scores}"
    )

    metalogs_dict = {
        "data": torch.cat([x_marg, x_cond], dim=1),
        "metalog_marginal": metalog_marginal,
        "marginal_mixture": marginal_mixture,
        "conditional": cond_metalog_dists,
        "marginal_bounds": marginal_bounds,
        "conditional_bounds": conditional_bounds,
        "marginal_inverse_cdf": smoothed_marginal_quantiles,
        "conditional_inverse_cdf": cond_inv_cdf,
        "marginal_quantile_cdf_range": [probs.min(), probs.max()],
    }

    return metalogs_dict


@torch.no_grad()
def make_marginal_bivariate_metalogs_samples(
    samples: torch.Tensor,
    device: str,
    min_regularization_term_marginal: float,
    max_regularization_term_marginal: float,
    min_regularization_term_conditional: float,
    max_regularization_term_conditional: float,
    min_metalog_res_marg: int,
    max_metalog_res_marg: int,
    min_metalog_res_cond: int,
    max_metalog_res_cond: int,
    metalog_marginal: str,
    plots_dir: Path,
    marginal_bounds: tuple[float, float] | None = None,
    conditional_bounds: tuple[float, float] | None = None,
    make_probs_method: str = "classic",
    measurement_dir_label: str = "",
    mix_K_marg: int = 2,
    mix_K_cond: int = 2,
    mix_alpha: float = 0.4,
    mix_max_iters: int = 30,
    mixture_model: str = "auto",  # auto | Metalog Mixture Model 1D (mmm),
    mix_metalog_term_marg: int = 5,
    mix_metalog_term_cond: int = 5,
    n_conditionals: int = 6,
) -> tuple[torch.Tensor, dict]:
    """
    Makes bivariate-mixture metalogs from samples (e.g., azimuth and eleveation).
    """
    if (marginal_bounds is None) ^ (conditional_bounds is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    samples = samples.to(device)
    x_marg = samples[:, 0].view(-1, 1)
    x_cond = samples[:, 1].view(-1, 1)

    # fit SPT mixture for the marginal samples
    if mixture_model == "auto":
        restarts_per_k = RESTARTS_PER_K
        model = AutoMetalogMixture1D(
            min_k=1,
            max_k=mix_K_marg,
            metalog_term=mix_metalog_term_marg,
            tol=MIX_CONVERGENCE_TOL,
            bounds=marginal_bounds,
            restarts_per_k=restarts_per_k,
            max_iters=mix_max_iters,
            lambda_min=min_regularization_term_marginal,
            lambda_max=max_regularization_term_marginal,
            resolution_min=min_metalog_res_marg,
            resolution_max=max_metalog_res_marg,
            alpha=mix_alpha,
            k_penalty=min_regularization_term_marginal,
            device=device,
            make_probs_method=make_probs_method,
        )
        model_name = "AutoMetalogMixture1D"
    else:
        restarts_per_k = 1
        if mix_metalog_term_marg == 3:
            model = SPTMetalogMixtureModel(
                n_components=mix_K_marg,
                max_iters=mix_max_iters,
                alpha=mix_alpha,
                tol=MIX_CONVERGENCE_TOL,
                bounds=marginal_bounds,
            )
            model_name = "SPTMetalogMixtureModel"
        else:
            model = MetalogMixtureModel(
                K=mix_K_marg,
                metalog_term=mix_metalog_term_marg,
                tol=MIX_CONVERGENCE_TOL,
                bounds=marginal_bounds,
                max_iters=mix_max_iters,
                lambda_=min_regularization_term_marginal,
                resolution=min_metalog_res_marg,
                device=device,
                make_probs_method=make_probs_method,
            )
            model_name = "MetalogMixtureModel"

    # fit data
    fit_results = model.fit(x_marg.view(-1))

    plot_enabled = plots_dir is not None
    plot_save_dir = None
    if plot_enabled:
        plot_save_dir = plots_dir / measurement_dir_label
        os.makedirs(plot_save_dir, exist_ok=True)

        fig, _, _ = plot_mixture_fit(model, x_marg.view(-1), show=False)
        fig.savefig(plot_save_dir / f"marginal_mixture_model.png")
        plt_close()

    if mixture_model == "auto":
        model_k = model.best_k
        model_t = model.best_term
        model_l = model.best_lambda
        model_r = model.best_resolution
        model = model.best_model
    else:
        model_k = mix_K_marg
        model_t = mix_metalog_term_marg
        model_l = min_regularization_term_marginal
        model_r = min_metalog_res_marg

    # store mixing and metalog coeffs with zeros padding if necessary
    m_mix_coeffs = model.mixing_coeffs.detach().cpu()
    m_metalog_coeffs = model.metalog_coeffs.detach().cpu()

    # pad zeros to columns
    mixing_coeffs = F.pad(m_mix_coeffs, (0, mix_K_marg - m_mix_coeffs.shape[0]))

    # pad zero rows and columns
    padding_bottom = mix_K_marg - m_metalog_coeffs.shape[0]
    padding_right = mix_metalog_term_marg - model_t
    metalog_coeffs = F.pad(
        m_metalog_coeffs, (0, padding_right, 0, padding_bottom)
    )

    # create a dictionary per future use
    marginal_mixture = {
        "model_name": model_name,
        "K": mix_K_marg,
        "bounds": marginal_bounds,
        "mixing_coeffs": mixing_coeffs,
        "coeffs": metalog_coeffs,
        "best_k": model_k,
        "best_t": model_t,
        "best_l": model_l,
        "best_r": model_r,
    }

    if plot_enabled:
        em_plot_title = f"{model_name} (T={mix_metalog_term_marg}, K={model_k}, {measurement_dir_label}) — EM History"
        (
            fig,
            _,
        ) = plot_best_em_history(
            model, fit_results, title=em_plot_title, show=False
        )
        fig.savefig(plot_save_dir / f"em_hist.png")
        plt_close()

    # get desired probs
    probs = ml.make_probs(n_conditionals, method="classic", device=device)
    smoothed_marginal_quantiles = model.eval_mixture_quantile(probs)

    # evaluate conditional metalogs
    cond_mix_model = fit_mixture_conditional_metalogs_kernel(
        smoothed_marginal_quantiles,
        x_marg,
        x_cond,
        min_metalog_res_cond,
        max_metalog_res_cond,
        mix_K_cond,
        mix_metalog_term_cond,
        mix_max_iters,
        plot_save_dir,
        bounds=conditional_bounds,
        lambda_min=min_regularization_term_conditional,
        lambda_max=max_regularization_term_conditional,
        mix_alpha=mix_alpha,
        mixture_model=mixture_model,
        measurement_dir_label=measurement_dir_label,
        n_samples=x_cond.size(0),
        make_probs_method=make_probs_method,
    )

    metalogs_dict = {
        "data": torch.cat([x_marg, x_cond], dim=1),
        "metalog_marginal": metalog_marginal,
        "marginal_mixture": marginal_mixture,
        "conditional_mixture": cond_mix_model,
        "marginal_bounds": marginal_bounds,
        "conditional_bounds": conditional_bounds,
        "marginal_inverse_cdf": smoothed_marginal_quantiles,
        "marginal_quantile_cdf_range": [probs.min(), probs.max()],
    }

    return metalogs_dict


@torch.no_grad()
def make_marginal_mixture_conditional_metalogs_tabular(
    device: str,
    target_data: torch.Tensor,
    min_regularization_term_marginal: float,
    max_regularization_term_marginal: float,
    regularization_term_conditional: float,
    min_metalog_res_marg: int,
    max_metalog_res_marg: int,
    metalog_res_cond: int,
    max_num_terms: int,
    plots_dir: Path,
    target_data_label: str,
    marginal_bounds: tuple[float, float] | None = None,
    conditional_bounds: tuple[float, float] | None = None,
    n_conditionals: int = 6,
    make_probs_method: str = "classic",
    measurement_dir_label: str = "",
    metalog_marginal: str = "phi",
    mix_metalog_term: int = 5,
    mix_K: int = 2,
    mix_alpha: float = 0.4,
    mix_max_iters: int = 30,
    n_marginal_samples: int = 128000,
    mixture_model: str = "auto",  # auto | Metalog Mixture Model 1D (mmm)
) -> tuple[torch.Tensor, dict]:
    """
    Makes marginal-mixture-conditional metalogs from tabulated data (e.g., VNDF/Luminance slice).
    """
    if (marginal_bounds is None) ^ (conditional_bounds is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    is_metalog_marginal_phi = metalog_marginal == "phi"

    # VNDFs are defined in range [0, 1]^2 - see also [Dupuy and Jakob 2018]
    theta_range_min, theta_range_max = 0, 1
    phi_range_min, phi_range_max = 0, 1

    theta_m_vndf_res = target_data.size(0)
    phi_m_vndf_res = target_data.size(1)

    phi_edges = torch.linspace(phi_range_min, phi_range_max, phi_m_vndf_res + 1)
    theta_edges = torch.linspace(
        theta_range_min, theta_range_max, theta_m_vndf_res + 1
    )

    if is_metalog_marginal_phi:
        assert (phi_range_min, phi_range_max) == marginal_bounds
        assert (theta_range_min, theta_range_max) == conditional_bounds
        marginal_edges = phi_edges
        cond_edges = theta_edges
    else:
        assert (theta_range_min, theta_range_max) == marginal_bounds
        assert (phi_range_min, phi_range_max) == conditional_bounds
        marginal_edges = theta_edges
        cond_edges = phi_edges
        target_data = target_data.T

    # to compute the marginal cdf, we need to sum along the other axis (i.e., 1)
    # note that marginal axis is always the x-axis (i.e., 0)
    marginal_counts = target_data.sum(axis=1)
    marginal_dist = marginal_counts / marginal_counts.sum()
    marginal_cdf = torch.cumsum(marginal_dist, dim=0)
    marginal_cdf = torch.cat(
        (torch.tensor([0.0], device=marginal_cdf.device), marginal_cdf)
    )

    # fit mixture for the marginal samples
    x_marg = ml.inverse_cdf(
        torch.rand(n_marginal_samples).to(device),
        marginal_cdf.to(device),
        marginal_edges.to(device),
    )
    marginal_mixture = {}
    if mixture_model == "auto":
        restarts_per_k = RESTARTS_PER_K
        model = AutoMetalogMixture1D(
            min_k=1,
            max_k=mix_K,
            metalog_term=mix_metalog_term,
            tol=MIX_CONVERGENCE_TOL,
            bounds=marginal_bounds,
            restarts_per_k=restarts_per_k,
            max_iters=mix_max_iters,
            lambda_min=min_regularization_term_marginal,
            lambda_max=max_regularization_term_marginal,
            resolution_min=min_metalog_res_marg,
            resolution_max=max_metalog_res_marg,
            alpha=mix_alpha,
            k_penalty=min_regularization_term_marginal,
            device=device,
            make_probs_method=make_probs_method,
        )
        model_name = "AutoMetalogMixture1D"
    else:
        restarts_per_k = 1
        model = MetalogMixtureModel(
            K=mix_K,
            metalog_term=mix_metalog_term,
            tol=MIX_CONVERGENCE_TOL,
            bounds=marginal_bounds,
            max_iters=mix_max_iters,
            lambda_=min_regularization_term_marginal,
            resolution=min_metalog_res_marg,
            device=device,
            make_probs_method=make_probs_method,
        )
        model_name = "MetalogMixtureModel"

    # fit data
    fit_results = model.fit(x_marg.view(-1))

    plot_enabled = plots_dir is not None
    plot_save_dir = None
    if plot_enabled:
        plot_save_dir = plots_dir / target_data_label / measurement_dir_label
        os.makedirs(plot_save_dir, exist_ok=True)

        fig, _, _ = plot_mixture_fit(model, x_marg.view(-1), show=False)
        fig.savefig(plot_save_dir / f"marginal_mixture_model.png")
        plt_close()

    if mixture_model == "auto":
        model_k = model.best_k
        model_t = model.best_term
        model_l = model.best_lambda
        model_r = model.best_resolution
        model = model.best_model
    else:
        model_k = mix_K
        model_t = mix_metalog_term
        model_l = min_regularization_term_marginal
        model_r = min_metalog_res_marg

    # store mixing and metalog coeffs with zeros padding if necessary
    m_mix_coeffs = model.mixing_coeffs.detach().cpu()
    m_metalog_coeffs = model.metalog_coeffs.detach().cpu()

    # pad zeros to columns
    mixing_coeffs = F.pad(m_mix_coeffs, (0, mix_K - m_mix_coeffs.shape[0]))

    # pad zero rows and columns
    padding_bottom = mix_K - m_metalog_coeffs.shape[0]
    padding_right = mix_metalog_term - model_t
    metalog_coeffs = F.pad(
        m_metalog_coeffs, (0, padding_right, 0, padding_bottom)
    )

    # create a dictionary per future use
    marginal_mixture = {
        "model_name": model_name,
        "K": mix_K,
        "bounds": marginal_bounds,
        "mixing_coeffs": mixing_coeffs,
        "coeffs": metalog_coeffs,
        "best_k": model_k,
        "best_t": model_t,
        "best_l": model_l,
        "best_r": model_r,
    }

    if plot_enabled:
        em_plot_title = f"{model_name} (T={mix_metalog_term}, K={model_k}, {measurement_dir_label}) — EM History"
        (
            fig,
            _,
        ) = plot_best_em_history(
            model, fit_results, title=em_plot_title, show=False
        )
        fig.savefig(plot_save_dir / f"em_hist.png")
        plt_close()

    # get desired probs
    probs = ml.make_probs(n_conditionals, method="classic", device=device)
    smoothed_marginal_quantiles = model.eval_mixture_quantile(probs)

    # evaluate conditional metalogs
    cond_metalog_dists, cond_inv_cdf, cond_scores = (
        ml.eval_conditional_metalog_distributions_tabular(
            smoothed_marginal_quantiles,
            metalog_res_cond,
            marginal_edges.to(device),
            target_data.to(device),
            max_num_terms,
            cond_edges.to(device),
            bounds=conditional_bounds,
            lambda_=regularization_term_conditional,
            device=device,
        )
    )
    logging.info(
        f"({measurement_dir_label}/wo) Conditional metalog scores: \n{cond_scores}"
    )

    metalogs_dict = {
        "data": target_data,
        "bins": [marginal_edges, cond_edges],
        "metalog_marginal": metalog_marginal,
        "marginal_mixture": marginal_mixture,
        "conditional": cond_metalog_dists,
        "marginal_bounds": marginal_bounds,
        "conditional_bounds": conditional_bounds,
        "marginal_inverse_cdf": smoothed_marginal_quantiles,
        "conditional_inverse_cdf": cond_inv_cdf,
        "marginal_quantile_cdf_range": [probs.min(), probs.max()],
    }

    return metalogs_dict


@torch.no_grad()
def make_mixture_bivariate_metalogs_tabular(
    device: str,
    target_data: torch.Tensor,
    min_regularization_term_marginal: float,
    max_regularization_term_marginal: float,
    min_regularization_term_conditional: float,
    max_regularization_term_conditional: float,
    min_metalog_res_marg: int,
    max_metalog_res_marg: int,
    min_metalog_res_cond: int,
    max_metalog_res_cond: int,
    plots_dir: Path,
    target_data_label: str,
    marginal_bounds: tuple[float, float] | None = None,
    conditional_bounds: tuple[float, float] | None = None,
    n_conditionals: int = 6,
    make_probs_method: str = "classic",
    measurement_dir_label: str = "",
    metalog_marginal: str = "phi",
    mix_metalog_term_marg: int = 5,
    mix_metalog_term_cond: int = 5,
    mix_K_marg: int = 2,
    mix_K_cond: int = 2,
    mix_alpha: float = 0.4,
    mix_max_iters: int = 30,
    mix_n_samples: int = 128000,
    mixture_model: str = "auto",  # auto | Metalog Mixture Model 1D (mmm)
) -> tuple[torch.Tensor, dict]:
    """
    Makes marginal-mixture-conditional metalogs from tabulated data (e.g., VNDF/Luminance slice).
    """
    if (marginal_bounds is None) ^ (conditional_bounds is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    is_metalog_marginal_phi = metalog_marginal == "phi"

    # VNDFs are defined in range [0, 1]^2 - see also [Dupuy and Jakob 2018]
    theta_range_min, theta_range_max = 0, 1
    phi_range_min, phi_range_max = 0, 1

    theta_m_vndf_res = target_data.size(0)
    phi_m_vndf_res = target_data.size(1)

    phi_edges = torch.linspace(phi_range_min, phi_range_max, phi_m_vndf_res + 1)
    theta_edges = torch.linspace(
        theta_range_min, theta_range_max, theta_m_vndf_res + 1
    )

    if is_metalog_marginal_phi:
        assert (phi_range_min, phi_range_max) == marginal_bounds
        assert (theta_range_min, theta_range_max) == conditional_bounds
        marginal_edges = phi_edges
        cond_edges = theta_edges
    else:
        assert (theta_range_min, theta_range_max) == marginal_bounds
        assert (phi_range_min, phi_range_max) == conditional_bounds
        marginal_edges = theta_edges
        cond_edges = phi_edges
        target_data = target_data.T

    # to compute the marginal cdf, we need to sum along the other axis (i.e., 1)
    # note that marginal axis is always the x-axis (i.e., 0)
    marginal_counts = target_data.sum(axis=1)
    marginal_dist = marginal_counts / marginal_counts.sum()
    marginal_cdf = torch.cumsum(marginal_dist, dim=0)
    marginal_cdf = torch.cat(
        (torch.tensor([0.0], device=marginal_cdf.device), marginal_cdf)
    )

    # fit mixture for the marginal samples
    x_marg = ml.inverse_cdf(
        torch.rand(mix_n_samples).to(device),
        marginal_cdf.to(device),
        marginal_edges.to(device),
    )

    if mixture_model == "auto":
        restarts_per_k = RESTARTS_PER_K
        model = AutoMetalogMixture1D(
            min_k=1,
            max_k=mix_K_marg,
            metalog_term=mix_metalog_term_marg,
            tol=MIX_CONVERGENCE_TOL,
            bounds=marginal_bounds,
            restarts_per_k=restarts_per_k,
            max_iters=mix_max_iters,
            lambda_min=min_regularization_term_marginal,
            lambda_max=max_regularization_term_marginal,
            resolution_min=min_metalog_res_marg,
            resolution_max=max_metalog_res_marg,
            alpha=mix_alpha,
            k_penalty=min_regularization_term_marginal,
            device=device,
            make_probs_method=make_probs_method,
        )
        model_name = "AutoMetalogMixture1D"
    else:
        restarts_per_k = 1
        if mix_metalog_term_marg == 3:
            model = SPTMetalogMixtureModel(
                n_components=mix_K_marg,
                max_iters=mix_max_iters,
                alpha=mix_alpha,
                tol=MIX_CONVERGENCE_TOL,
                bounds=marginal_bounds,
            )
            model_name = "SPTMetalogMixtureModel"
        else:
            model = MetalogMixtureModel(
                K=mix_K_marg,
                metalog_term=mix_metalog_term_marg,
                tol=MIX_CONVERGENCE_TOL,
                bounds=marginal_bounds,
                max_iters=mix_max_iters,
                lambda_=min_regularization_term_marginal,
                resolution=min_metalog_res_marg,
                device=device,
                make_probs_method=make_probs_method,
            )
            model_name = "MetalogMixtureModel"

    # fit data
    fit_results = model.fit(x_marg.view(-1))

    plot_enabled = plots_dir is not None
    plot_save_dir = None
    if plot_enabled:
        plot_save_dir = plots_dir / target_data_label / measurement_dir_label
        os.makedirs(plot_save_dir, exist_ok=True)

        fig, _, _ = plot_mixture_fit(model, x_marg.view(-1), show=False)
        fig.savefig(plot_save_dir / f"marginal_mixture_model.png")
        plt_close()

    if mixture_model == "auto":
        model_k = model.best_k
        model_t = model.best_term
        model_l = model.best_lambda
        model_r = model.best_resolution
        model = model.best_model
    else:
        model_k = mix_K_marg
        model_t = mix_metalog_term_marg
        model_l = min_regularization_term_marginal
        model_r = min_metalog_res_marg

    # store mixing and metalog coeffs with zeros padding if necessary
    m_mix_coeffs = model.mixing_coeffs.detach().cpu()
    m_metalog_coeffs = model.metalog_coeffs.detach().cpu()

    # pad zeros to columns
    mixing_coeffs = F.pad(m_mix_coeffs, (0, mix_K_marg - m_mix_coeffs.shape[0]))

    # pad zero rows and columns
    padding_bottom = mix_K_marg - m_metalog_coeffs.shape[0]
    padding_right = mix_metalog_term_marg - model_t
    metalog_coeffs = F.pad(
        m_metalog_coeffs, (0, padding_right, 0, padding_bottom)
    )

    # create a dictionary per future use
    marginal_mixture = {
        "model_name": model_name,
        "K": mix_K_marg,
        "bounds": marginal_bounds,
        "mixing_coeffs": mixing_coeffs,
        "coeffs": metalog_coeffs,
        "best_k": model_k,
        "best_t": model_t,
        "best_l": model_l,
        "best_r": model_r,
    }

    if plot_enabled:
        em_plot_title = f"{model_name} (T={mix_metalog_term_marg}, K={model_k}, {measurement_dir_label}) — EM History"
        (
            fig,
            _,
        ) = plot_best_em_history(
            model, fit_results, title=em_plot_title, show=False
        )
        fig.savefig(plot_save_dir / f"em_hist.png")
        plt_close()

    # get desired probs
    probs = ml.make_probs(n_conditionals, method="classic", device=device)
    smoothed_marginal_quantiles = model.eval_mixture_quantile(probs)

    cond_mix_model = fit_mixture_conditional_metalogs_tabular(
        smoothed_marginal_quantiles,
        marginal_edges.to(device),
        target_data.to(device),
        cond_edges.to(device),
        min_metalog_res_cond,
        max_metalog_res_cond,
        mix_K_cond,
        mix_metalog_term_cond,
        mix_max_iters,
        plot_save_dir,
        bounds=conditional_bounds,
        lambda_min=min_regularization_term_conditional,
        lambda_max=max_regularization_term_conditional,
        mix_alpha=mix_alpha,
        mixture_model=mixture_model,
        measurement_dir_label=measurement_dir_label,
        n_samples=mix_n_samples,
        make_probs_method=make_probs_method,
    )

    metalogs_dict = {
        "data": target_data,
        "bins": [marginal_edges, cond_edges],
        "metalog_marginal": metalog_marginal,
        "marginal_mixture": marginal_mixture,
        "conditional_mixture": cond_mix_model,
        "marginal_bounds": marginal_bounds,
        "conditional_bounds": conditional_bounds,
        "marginal_inverse_cdf": smoothed_marginal_quantiles,
        "marginal_quantile_cdf_range": [probs.min(), probs.max()],
    }

    return metalogs_dict


@torch.no_grad
def fit_mixture_conditional_metalogs_tabular(
    marginal_quantiles: torch.Tensor,
    xedges: torch.Tensor,
    data2d: torch.Tensor,
    yedges: torch.Tensor,
    min_resolution: int,
    max_resolution: int,
    K: int,
    metalog_term: int,
    max_iter: int,
    plot_save_dir: Path,
    bounds: tuple[float, float] | None = None,
    lambda_min: float = 1e-5,
    lambda_max: float = 1e-5,
    mix_alpha: float = 0.3,
    mixture_model: str = "auto",  # auto | Metalog Mixture Model 1D (mmm)
    measurement_dir_label: str = "",
    n_samples: int = 128000,
    make_probs_method: str = "classic",
) -> dict:
    """
    Fits conditional distributions into n Metalog Mixture Models using tabulated data approach.

    Adapted after `ml.eval_conditional_metalog_distributions_tabular()`.
    """
    plot_enabled = plot_save_dir is not None
    device = marginal_quantiles.device

    # to compute the marginal cdf, we need to sum along the other axis (i.e., 1)
    # note that marginal axis is always the x-axis (i.e., 0)
    marginal_x_counts = data2d.sum(dim=1)
    xmidpoints = (xedges[:-1] + xedges[1:]) / 2
    num_bins = xmidpoints.size(0)

    # determine the appropriate bin for x by finding where it fits in
    # xedges, then adjusting for array index
    x_lower_bin_idx = (
        torch.searchsorted(xedges, marginal_quantiles, right=True) - 1
    )

    # clamp the lower bin index to the valid range [0, num_bins - 2]
    x_lower_bin_idx.clamp_(0, num_bins - 2)
    x_upper_bin_idx = x_lower_bin_idx + 1

    # get marginal_x_counts at the lower and upper bin, shape = [N, 1]
    marginal_x_counts_lower = marginal_x_counts[x_lower_bin_idx].unsqueeze(1)
    marginal_x_counts_upper = marginal_x_counts[x_upper_bin_idx].unsqueeze(1)

    # get data slices at the lower and upper bin, shape = [N, num_bins_y]
    data2d_lower = data2d[x_lower_bin_idx, :]
    data2d_upper = data2d[x_upper_bin_idx, :]

    # compute conditional distributions
    cond_dist_lower = torch.where(
        marginal_x_counts_lower > 0,
        data2d_lower / marginal_x_counts_lower,
        torch.zeros_like(data2d_lower),
    )
    cond_dist_upper = torch.where(
        marginal_x_counts_upper > 0,
        data2d_upper / marginal_x_counts_upper,
        torch.zeros_like(data2d_upper),
    )

    # fill empty x-rows with uniform rows
    def fill_empty_rows(cond_rows):
        empty = cond_rows.sum(dim=1) <= 0
        if empty.any():
            logging.warning(
                "Empty conditional distribution row; filling it with uniform rows."
            )
            cond_rows[empty] = uniform_row
        return cond_rows

    M = cond_dist_lower.shape[1]
    uniform_row = torch.full((1, M), 1.0 / M, device=device)
    cond_dist_lower = fill_empty_rows(cond_dist_lower)
    cond_dist_upper = fill_empty_rows(cond_dist_upper)

    # compute lower and upper conditional CDFs
    lower_cond_cdfs = torch.cumsum(cond_dist_lower, dim=1)
    upper_cond_cdfs = torch.cumsum(cond_dist_upper, dim=1)

    # add a zero at the start of the CDFs
    zero_col = torch.zeros((lower_cond_cdfs.size(0), 1), device=device)
    lower_cond_cdfs = torch.cat((zero_col, lower_cond_cdfs), dim=1)
    upper_cond_cdfs = torch.cat((zero_col, upper_cond_cdfs), dim=1)

    # ensure last value is 1
    EPSILON = ml._eps()
    lower_cond_cdfs /= lower_cond_cdfs[:, -1:].clamp_min(EPSILON)
    upper_cond_cdfs /= upper_cond_cdfs[:, -1:].clamp_min(EPSILON)

    # calculate bin width for interpolation and handle zero bin widths
    x_lower_midpoint = xmidpoints[x_lower_bin_idx]
    x_upper_midpoint = xmidpoints[x_upper_bin_idx]
    interp_width = (x_upper_midpoint - x_lower_midpoint).clamp_(min=1e-8)

    # compute interpolation parameter t, shape [N, 1]
    t = (marginal_quantiles - x_lower_midpoint) / interp_width
    t = torch.clamp(t, min=0, max=1).unsqueeze(1)

    # interpolate lower & upper CDFs and normalize
    cond_cdf = (1 - t) * lower_cond_cdfs + t * upper_cond_cdfs
    cond_cdf /= cond_cdf[:, -1:].clamp_min(EPSILON)

    n_conditionals = cond_dist_lower.shape[0]
    mixing_coeffs = torch.empty((n_conditionals, K))
    metalog_coeffs = torch.empty((n_conditionals, K, metalog_term))
    for n in range(n_conditionals):
        cond_samples = ml.inverse_cdf(
            torch.rand(n_samples).to(device),
            cond_cdf[n],
            yedges,
        )

        if mixture_model == "auto":
            restarts_per_k = RESTARTS_PER_K
            model = AutoMetalogMixture1D(
                min_k=1,
                max_k=K,
                metalog_term=metalog_term,
                tol=MIX_CONVERGENCE_TOL,
                bounds=bounds,
                restarts_per_k=restarts_per_k,
                max_iters=max_iter,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                resolution_min=min_resolution,
                resolution_max=max_resolution,
                alpha=mix_alpha,
                k_penalty=lambda_min,
                device=str(device),
                make_probs_method=make_probs_method,
            )
            model_name = "AutoMetalogMixture1D"
        else:
            restarts_per_k = 1
            if metalog_term == 3:
                model = SPTMetalogMixtureModel(
                    n_components=K,
                    max_iters=max_iter,
                    alpha=mix_alpha,
                    tol=MIX_CONVERGENCE_TOL,
                    bounds=bounds,
                )
                model_name = "SPTMetalogMixtureModel"
            else:
                model = MetalogMixtureModel(
                    K=K,
                    metalog_term=metalog_term,
                    tol=MIX_CONVERGENCE_TOL,
                    bounds=bounds,
                    max_iters=max_iter,
                    lambda_=lambda_min,
                    resolution=min_resolution,
                    device=str(device),
                    make_probs_method=make_probs_method,
                )
                model_name = "MetalogMixtureModel"

        fit_results = model.fit(cond_samples.view(-1))

        if plot_enabled:
            fig, _, _ = plot_mixture_fit(
                model, cond_samples.view(-1), show=False
            )
            fig.savefig(plot_save_dir / f"cond_mixture_model_{n}.png")
            plt_close()

        if mixture_model == "auto":
            model_k = model.best_k
            model_t = model.best_term
            model_l = model.best_lambda
            model_r = model.best_resolution
            model = model.best_model
        else:
            model_k = K
            model_t = metalog_term
            model_l = lambda_min
            model_r = min_resolution

        if plot_enabled:
            em_plot_title = f"{model_name} (T={metalog_term}, K={model_k}, N={n}, {measurement_dir_label}) — EM History"
            (
                fig,
                _,
            ) = plot_best_em_history(
                model, fit_results, title=em_plot_title, show=False
            )
            fig.savefig(plot_save_dir / f"cond_em_hist_{n}.png")
            plt_close()

        # store mixing and metalog coeffs with zeros padding if necessary
        m_mix_coeffs = model.mixing_coeffs.detach().cpu()
        m_metalog_coeffs = model.metalog_coeffs.detach().cpu()

        # pad zeros to columns
        mixing_coeffs[n] = F.pad(m_mix_coeffs, (0, K - m_mix_coeffs.shape[0]))

        # pad zero rows and columns
        padding_bottom = K - m_metalog_coeffs.shape[0]
        padding_right = metalog_term - model_t
        metalog_coeffs[n] = F.pad(
            m_metalog_coeffs, (0, padding_right, 0, padding_bottom)
        )

    # create a dictionary per future use
    cond_mixture = {
        "model_name": model_name,
        "K": K,
        "bounds": bounds,
        "mixing_coeffs": mixing_coeffs,
        "coeffs": metalog_coeffs,
        "best_k": model_k,
        "best_t": model_t,
        "best_l": model_l,
        "best_r": model_r,
    }

    return cond_mixture


def fit_mixture_conditional_metalogs_kernel(
    marginal_quantiles: torch.Tensor,
    x_marg_samples: torch.Tensor,
    x_cond_samples: torch.Tensor,
    min_resolution: int,
    max_resolution: int,
    K: int,
    metalog_term: int,
    max_iter: int,
    plot_save_dir: Path,
    bounds: tuple[float, float] | None = None,
    lambda_min: float = 1e-5,
    lambda_max: float = 1e-5,
    mix_alpha: float = 0.3,
    mixture_model: str = "auto",  # "auto" | "mmm"
    measurement_dir_label: str = "",
    n_samples: int = 128000,
    bandwidth: float | None = None,
    make_probs_method: str = "classic",
) -> dict:
    """
    Fits conditional distributions into n Metalog Mixture Models using the kernel approach.

    For each q_j in `marginal_quantiles`, it builds a kernel-weighted empirical conditional distribution over
    `x_cond_samples`, draws `n_samples` from this discrete distribution and fits a 1D metalog mixture model to those
    samples.

    Adapted after `ml.eval_conditional_metalog_distributions_kernel()`.
    """
    plot_enabled = plot_save_dir is not None
    device = marginal_quantiles.device
    marginal_quantiles = marginal_quantiles.view(-1).to(device)

    x_marg_samples = x_marg_samples.view(-1).to(device)
    x_cond_samples = x_cond_samples.view(-1).to(device)
    assert (
        x_marg_samples.shape == x_cond_samples.shape
    ), "x_marg_samples and x_cond_samples must have the same shape."

    # kernel weights over marginal dimension, shape: (J, N)
    W = ml.marginal_kernel_weights(
        x_marg_samples, marginal_quantiles, bandwidth
    )

    J = marginal_quantiles.shape[0]
    mixing_coeffs = torch.empty((J, K))
    metalog_coeffs = torch.empty((J, K, metalog_term))
    slice_info = {}
    for j in range(J):
        # sample from the discrete conditional p(i | x_marg ≈ q_j)
        idx = torch.multinomial(W[j], num_samples=n_samples, replacement=True)
        cond_samples = x_cond_samples[idx]

        # fit mixture model
        if mixture_model == "auto":
            restarts_per_k = RESTARTS_PER_K
            model = AutoMetalogMixture1D(
                min_k=1,
                max_k=K,
                metalog_term=metalog_term,
                tol=MIX_CONVERGENCE_TOL,
                bounds=bounds,
                restarts_per_k=restarts_per_k,
                max_iters=max_iter,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                resolution_min=min_resolution,
                resolution_max=max_resolution,
                alpha=mix_alpha,
                k_penalty=lambda_min,
                device=device,
                make_probs_method=make_probs_method,
            )
            model_name = "AutoMetalogMixture1D"
        else:
            restarts_per_k = 1
            if metalog_term == 3:
                model = SPTMetalogMixtureModel(
                    n_components=K,
                    max_iters=max_iter,
                    alpha=mix_alpha,
                    tol=MIX_CONVERGENCE_TOL,
                    bounds=bounds,
                )
                model_name = "SPTMetalogMixtureModel"
            else:
                model = MetalogMixtureModel(
                    K=K,
                    metalog_term=metalog_term,
                    tol=MIX_CONVERGENCE_TOL,
                    bounds=bounds,
                    max_iters=max_iter,
                    lambda_=lambda_min,
                    resolution=min_resolution,
                    device=str(device),
                    make_probs_method=make_probs_method,
                )
                model_name = "MetalogMixtureModel"

        fit_results = model.fit(cond_samples.view(-1))

        if plot_enabled:
            fig, _, _ = plot_mixture_fit(
                model,
                cond_samples.view(-1),
                show=False,
            )
            fig.savefig(plot_save_dir / f"cond_mixture_model_{j}.png")
            plt_close()

        if mixture_model == "auto":
            model_k = model.best_k
            model_t = model.best_term
            model_l = model.best_lambda
            model_r = model.best_resolution
            model = model.best_model
        else:
            model_k = K
            model_t = metalog_term
            model_l = lambda_min
            model_r = min_resolution

        if plot_enabled:
            em_plot_title = (
                f"{model_name} (T={metalog_term}, K={model_k}, "
                f"N={j}, {measurement_dir_label}) — EM History"
            )
            fig, _ = plot_best_em_history(
                model,
                fit_results,
                title=em_plot_title,
                show=False,
            )
            fig.savefig(plot_save_dir / f"cond_em_hist_{j}.png")
            plt_close()

        # store mixing and metalog coeffs with zero padding if necessary
        m_mix_coeffs = model.mixing_coeffs.detach().cpu()
        m_metalog_coeffs = model.metalog_coeffs.detach().cpu()

        # pad mixture weights to length K
        mixing_coeffs[j] = F.pad(
            m_mix_coeffs,
            (0, K - m_mix_coeffs.shape[0]),
        )

        # pad metalog coeffs to shape (K, metalog_term)
        padding_bottom = K - m_metalog_coeffs.shape[0]
        padding_right = metalog_term - model_t
        metalog_coeffs[j] = F.pad(
            m_metalog_coeffs,
            (0, padding_right, 0, padding_bottom),
        )

        slice_info[j] = {
            "best_k": model_k,
            "best_t": model_t,
            "best_l": model_l,
            "best_r": model_r,
        }

    cond_mixture = {
        "model_name": model_name,
        "K": K,
        "bounds": bounds,
        "mixing_coeffs": mixing_coeffs,
        "coeffs": metalog_coeffs,
        "slice_info": slice_info,
    }

    return cond_mixture
