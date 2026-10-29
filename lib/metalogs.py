# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import torch
from pathlib import Path
import logging
from typing import TypedDict
from enum import Enum
from lib.utils.sample_record import SampleRecord
import lib.utils.logit_transform as lt


class UnivariateMetalogCDFMethod(Enum):
    CDF_NEWTON = 1
    CDF_INTERPOLATION = 2


class BivariateMetalogPDFMethod(TypedDict):
    marginal_method: UnivariateMetalogCDFMethod
    conditional_method: UnivariateMetalogCDFMethod


@torch.jit.script
def _eps() -> float:
    return 1e-6


@torch.no_grad
@torch.jit.script
def make_probs(
    resolution: int,
    method: str = "classic",
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns 'resolution' amount of values in [0, 1] range.

    Args:
        resolution (int): Amount of values.
        method (str, optional): If "classic", then `(np.arange(1, resolution + 1) - 0.5) / resolution`, if "chebyshev",
          then chebyshev points in [0, 1], if "linspace", then `np.linspace(EPSILON, 1 - EPSILON, resolution)`.
          Defaults to "classic".
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Returns:
        torch.Tensor: 'resolution' amount of values in [0, 1] range.
    """
    EPSILON = _eps()
    if method == "classic":
        return (
            torch.arange(1, resolution + 1, device=device) - 0.5
        ) / resolution
    elif method == "linspace":
        return torch.linspace(EPSILON, 1 - EPSILON, resolution, device=device)
    elif method == "chebyshev":
        a, b = 1, 0
        i = torch.arange(resolution, device=device)
        # compute Chebyshev points in [-1, 1]
        x_chebyshev = torch.cos((2 * i + 1) / (2 * resolution) * torch.pi)
        # transform the points to [a, b]
        return (b - a) / 2 * x_chebyshev + (a + b) / 2
    else:
        # Default "classic"
        return (torch.arange(1, resolution + 1) - 0.5) / resolution


@torch.jit.script
def eval_unbounded_metalog_pdf(
    a: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Computes metalog PDF for given CDF values using PyTorch. This is the Metalog PDF implementation.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [1|N, T].
        y (torch.Tensor): CDF values in shape [N,].

    Raises:
        ValueError: If the number of metalog terms is lower than 3.

    Returns:
        torch.Tensor: PDF value(s) from the given metalog coefficients and CDF value in shape [N,].
    """
    T = a.size(1)
    if T <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    EPSILON = _eps()
    ys = torch.clamp(y, EPSILON, 1 - EPSILON)
    y_term = ys * (1 - ys)
    log_coeff = torch.log(ys / (1 - ys))
    sub_coeff = ys - 0.5

    # base terms
    dx_dy = ((a[:, 1] / y_term)) + a[:, 2] * ((sub_coeff / y_term) + log_coeff)

    # 4th term
    if T >= 4:
        dx_dy += a[:, 3]

    # add terms starting from the 5th term
    for n in range(5, T + 1):
        if n % 2 == 0:  # even terms
            dx_dy += a[:, n - 1] * (
                ((sub_coeff ** (n // 2 - 1)) / y_term)
                + (n / 2 - 1) * sub_coeff ** (n // 2 - 2) * log_coeff
            )
        else:  # odd terms
            dx_dy += a[:, n - 1] * ((n - 1) / 2) * sub_coeff ** ((n - 3) // 2)

    return 1.0 / torch.clamp(dx_dy, min=EPSILON)


@torch.jit.script
def eval_derivative_unbounded_metalog_quantile(
    a: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Computes the derivative of the unbounded Metalog Quantile function. Identical to the unbounded metalog PDF, but its
    inverse. See also the definition of the Metalog PDF, equation 9 in The Metalog Distributions by Tom Keelin 2016.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [n, num_metalog_coeffs].
        y (torch.Tensor): CDF values.

    Raises:
        ValueError: If the number of metalog terms is lower than 3.

    Returns:
        torch.Tensor: Derivative of the Metalog Quantile function, M'(y).
    """
    num_terms = a.size(1)
    if num_terms <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    EPSILON = _eps()
    ys = torch.clamp(y, EPSILON, 1 - EPSILON)
    y_term = ys * (1 - ys)
    log_coeff = torch.log(ys / (1 - ys))
    sub_coeff = ys - 0.5

    # base terms
    dx_dy = ((a[:, 1] / y_term)) + a[:, 2] * ((sub_coeff / y_term) + log_coeff)

    # 4th term
    if num_terms >= 4:
        dx_dy += a[:, 3]

    # add terms starting from the 5th term
    for n in range(5, num_terms + 1):
        if n % 2 == 0:  # even terms
            dx_dy += a[:, n - 1] * (
                ((sub_coeff ** (n // 2 - 1)) / y_term)
                + (n / 2 - 1) * sub_coeff ** (n // 2 - 2) * log_coeff
            )
        else:  # odd terms
            dx_dy += a[:, n - 1] * ((n - 1) / 2) * sub_coeff ** ((n - 3) // 2)

    return dx_dy


@torch.jit.script
def eval_unbounded_metalog_quantile(
    a: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates metalog quantile function for the given metalog coefficients.
    Essentially inverse CDF.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [1|N, T].
        y (torch.Tensor): CDF values in shape [N,].

    Returns:
        torch.Tensor: Metalog quantiles from the given coefficients in shape [N,].
    """
    T = a.size(1)
    if T <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    EPSILON = _eps()
    ys = torch.clamp(y, EPSILON, 1 - EPSILON)
    log_coeff = torch.log(ys / (1 - ys))
    sub_coeff = ys - 0.5

    # base terms
    M_y = a[:, 0] + a[:, 1] * log_coeff + a[:, 2] * sub_coeff * log_coeff

    # 4th term
    if T >= 4:
        M_y += a[:, 3] * sub_coeff

    # add terms starting from the 5th term
    for n in range(5, T + 1):
        if n % 2 == 0:  # even terms
            M_y += a[:, n - 1] * (sub_coeff ** ((n // 2) - 1)) * log_coeff
        else:  # odd terms
            M_y += a[:, n - 1] * (sub_coeff ** ((n - 1) // 2))

    return M_y


@torch.no_grad
def is_feasible_metalog(
    a: torch.Tensor,
    bounds: tuple[float, float] | None = None,
) -> bool:
    """
    Checks metalog feasability. A metalog is feasible, if M'(y) > 0 for all y in (0,1), see Eq. 5, or equivalently
    m(y) > 0 for all y in (0,1), see Eq. 10.

    Even though metalogs might be bounded, we use Eq. 10 to check feasibility. See also Keelin 2016: "Since the quantity
    ... is always positive, this condition is equivalent to (10)".

    When bounded metalogs are used, to further ensure feasibility, optionally provide the bounds to test whether
    m(y)^{bounded} > 0 for all y in (0, 1).


    Args:
        a (torch.Tensor): Metalog coefficients in shape [n, num_metalog_coeffs].
        bounds (tuple): Lower and upper bound. Defaults to None.

    Raises:
        ValueError: If a does not have the expected shape, or metalog term is not >= 3.

    Returns:
        bool: True, if the provided metalog is feasible otherwise False.
    """
    if a.ndim != 2:
        raise ValueError("Expected a [n, num_metalog_coeffs] shaped tensor.")

    num_terms = a.size(1)
    if num_terms <= 2:
        raise ValueError(f"At least 3 terms is necessary.")
    
    if bounds is not None and bounds[0] >= bounds[1]:
        raise ValueError(f"Invalid bounds: {bounds}")
    
    # if batch is passed, evaluate sequentially
    if a.size(0) > 1:
        for i in range(a.size(0)):
            if not is_feasible_metalog(a[i:i+1, :], bounds):
                return False
        return True

    # we evaluate in float64, but we check against float32 limits
    compute_dtype = torch.float64
    a_calc = a.to(dtype=compute_dtype)

    # check deep into the tails
    f32_eps = torch.finfo(torch.float32).eps
    tail = f32_eps

    # create uniformly spaced ys
    grid_size = 4096
    y_lin = torch.linspace(
        tail, 1 - tail, grid_size, device=a.device, dtype=compute_dtype
    )

    # create a logit grid for denser samples in tails
    L = torch.logit(
        torch.tensor(1 - tail, dtype=compute_dtype, device=a.device)
    ).item()
    t = torch.linspace(-L, L, grid_size, device=a.device, dtype=compute_dtype)
    y_log = torch.sigmoid(t)

    # combine linspace and logit grid
    y = torch.cat([y_lin, y_log])

    # evaluate the derivative M'(y)
    dx_dy = eval_derivative_unbounded_metalog_quantile(a_calc, y)

    if dx_dy.ndim == 1:
        dx_dy = dx_dy.unsqueeze(0)

    # check finiteness
    if not torch.isfinite(dx_dy).all():
        return False

    # check M'(y) > 0 for all y in (0,1). See also Equation 5
    min_derivative = 1e-5
    if not (dx_dy >= min_derivative).all():
        return False

    # bounded feasibility, if bounds are provided
    if bounds is not None:
        # evaluate unbounded metalog quantile/PDF only if we need it for bounds
        M_y = eval_unbounded_metalog_quantile(a_calc, y)

        # float32 exp(x) overflows to inf at ~88, we cap at 85.0
        max_exp_limit = 85.0
        if not (M_y.abs() <= max_exp_limit).all():
            return False

        m_y = eval_unbounded_metalog_pdf(a_calc, y)
        if m_y.ndim == 1:
            m_y = m_y.unsqueeze(0)

        # check finiteness of m_y
        if not torch.isfinite(m_y).all():
            return False

        b_pdf = eval_bounded_metalog_pdf(a_calc, bounds[0], bounds[1], y, M_y)

        # check finiteness of bounded PDF
        if not torch.isfinite(b_pdf).all():
            return False

        # check bounded m(y) > 0 for all y in (0,1)
        if not (b_pdf >= 0.0).all():
            return False

    return True


def feasibilize_metalog_coeffs(
    a_new: torch.Tensor,
    a_old: torch.Tensor | None = None,
    lambda_: float = 0.0,
    feasibility_max_backtracks: int = 10,
) -> torch.Tensor:
    """
    If infeasible, backtrack between `a_old` and `a_new`. Otherwise return `a_new`. If no `a_old` is provided
    (e.g., initializer), softly inflate ridge (`lambda_`) until feasible.

    Args:
        a_new (torch.Tensor): Metalog coefficients to feasibilize, shape: [n, num_metalog_coeffs].
        a_old (torch.Tensor): Old metalog coefficients, shape [n, num_metalog_coeffs]. Defaults to None.
        lambda_ (float, optional): Regularization term. Defaults to 1e-5.
        feasibility_max_backtracks (int): Maximum number of iterations to backtrack. Defaults to 10.

    Returns:
        torch.Tensor: If successful, feasbilized metalog coefficients, shape: [n, num_metalog_coeffs],
    """
    if is_feasible_metalog(a_new):
        return a_new

    if a_old is None:
        # inflate lambda progressively
        lam = max(lambda_, 1e-4)
        a_try = a_new.clone()

        for i in range(feasibility_max_backtracks):
            lam *= 10.0

            # small Tikhonov push towards "shorter" a
            shrink = 1.0 / (1.0 + lam)
            a_try[:, 3:] *= shrink
            a_try[:, 1] = a_try[:, 1].abs() + 1e-6

            if is_feasible_metalog(a_try):
                return a_try

        # fallback: 3-term projection
        a_proj = a_new.clone()
        a_proj[:, 3:] = 0.0
        a_proj[:, 1] = a_proj[:, 1].abs() + 1e-6

        return a_proj

    # backtrack between `a_old` and `a_new`: line search from a_old -> a_new
    t = 0.5
    lam = max(lambda_, 1e-6)
    for _ in range(feasibility_max_backtracks):
        a_try = a_old + t * (a_new - a_old)

        if is_feasible_metalog(a_try):
            return a_try

        t *= 0.5

    # fallback: return old coeffs
    if a_old is not None:
        return a_old

    # fallback: 3-term projection
    a_proj = a_new.clone()
    a_proj[:, 3:] = 0.0
    a_proj[:, 1] = a_proj[:, 1].abs() + 1e-6

    return a_proj


@torch.no_grad
def make_unbounded_metalogs(
    samples: torch.Tensor,
    probs: torch.Tensor = None,
    lambda_: float = 1e-5,
    max_num_terms: int = 12,
    model_criterion: str = "bic",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Makes unbounded metalogs by solving for metalog coefficients. See also equation 7, and 8 in The Metalog
    Distributions by Tom Keelin 2016.

    Args:
        samples (torch.Tensor): Sorted data points in shape [metalog_resolution] or
          [metalog_resolution, metalog_resolution].
        probs (torch.Tensor | None, optional): CDF values of the samples in shape [metalog_resolution].
          Defaults to None.
        lambda_ (float, optional): Regularization term. Defaults to 1e-5.
        max_num_terms (int, optional): Defines maximum number of metalog terms (i.e., np.arange(3, max_num_terms)).
          Defaults to 12.
        model_criterion (str, optional): Criterion for model scoring ('aic', 'bic', 'sse'). Defaults to 'bic'.
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Model Criterion (source Wikipedia):
        - Akaike information criterion (AIC) is a statistical measure used to evaluate the quality of different
        statistical models for a given dataset. It helps in model selection by balancing goodness-of-fit with the number
        of parameters in the model. Lower AIC values generally indicate a better-fitting model.
        - Bayesian information criterion (BIC) is a criterion for model selection among a finite set of models; models
        with lower BIC are generally preferred. It is based, in part, on the likelihood function and it is closely
        related to the AIC.

    Returns:
        tuple(torch.Tensor, torch.Tensor): Computed metalog coefficients in a (n_rows, max_num_terms - 3, max_num_terms - 1)
          tensor. Beware that each metalog coefficient tensor may contain NaNs due to fitting them into a one big
          tensor. Model scores for each metalog in a (max_num_terms - 3) tensor.
    """
    # ensure valid number of metalog terms
    if max_num_terms < 3:
        raise ValueError("At least 3 metalog terms is necessary.")

    # determine samples shape
    samples_dim = samples.dim()
    if samples_dim > 2:
        raise ValueError("Unexpected shape for samples.")

    if samples_dim == 1:
        # expand samples to have the shape [1, metalog_resolution]
        samples = samples.unsqueeze(0)

    # ensure valid metalog resolution
    n_rows, ml_resolution = samples.shape
    if ml_resolution < 2:
        raise ValueError("At least two samples is necessary.")

    # use double precision for stability
    dtype = torch.float64
    samples = samples.to(dtype=dtype, device=device)

    # initialize y (i.e., CDFs of samples)
    if probs is not None:
        if probs.ndim != 1:
            raise ValueError(f"Unexpected shape for probabilities.")

        # ensure metalog and CDF resolutions match
        probs_res = probs.size(0)
        if probs.size(0) != ml_resolution:
            raise ValueError(
                f"Resolution mismatch (quantiles: {[ml_resolution]}, probabilities: {[probs_res]})."
            )

        y = probs.to(dtype=dtype, device=device)
    else:
        # this approach is less accurate
        y = (
            torch.arange(1, ml_resolution + 1, device=device, dtype=dtype) - 0.5
        ) / ml_resolution

    # initialize results
    result = torch.full(
        (n_rows, max_num_terms - 3, max_num_terms - 1),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    scores = torch.full(
        (n_rows, max_num_terms - 3),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    num_terms = torch.arange(3, max_num_terms, device=device)

    # precompute components
    log_coeff = torch.log(y / (1 - y)).unsqueeze(1)
    sub_coeff = (y - 0.5).unsqueeze(1)

    # iterate over each set of samples
    for row_idx in range(n_rows):
        samples_i = samples[row_idx, :]

        # iterate over number of terms requested
        for term_idx, term in enumerate(num_terms):

            ### Find matrix Y and then metalog coefficients ###
            Y = torch.ones((ml_resolution, 1), device=device, dtype=dtype)

            # 2nd Term
            Y = torch.cat([Y, log_coeff], dim=1)

            # 3rd Term
            Y = torch.cat([Y, sub_coeff * log_coeff], dim=1)

            # 4th Term
            if term >= 4:
                Y = torch.cat([Y, sub_coeff], dim=1)

            for n in range(5, int(term.item()) + 1):
                if n % 2 == 0:  # even terms
                    Y = torch.cat(
                        [Y, (sub_coeff ** ((n // 2) - 1)) * log_coeff], dim=1
                    )
                else:  # odd terms
                    Y = torch.cat([Y, sub_coeff ** ((n - 1) // 2)], dim=1)

            ### Solve for metalog coefficients ###
            if lambda_ >= 1e-5:
                # Modify the normal equation to include the regularization term
                # mentioned by Tom Keelin, 3.2. Metalog Panel
                lambda_identity = lambda_ * torch.eye(
                    term, device=device, dtype=dtype
                )
                # exclude regularization for the intercept (a_1), penalize the
                # complexity and shape coefficients, but not overall position of
                # the distribution on the number line
                lambda_identity[0, 0] = 0
                A = Y.T @ Y + lambda_identity
                b = Y.T @ samples_i
                a = torch.linalg.solve(A, b)
            else:
                # ordinary least squares: min_a ||Ya - x||^2
                a = torch.linalg.lstsq(Y, samples_i).solution

            # if not feasible try to feasibilize it
            a = feasibilize_metalog_coeffs(a.unsqueeze(0), lambda_=lambda_)
            a = a.squeeze()

            result[row_idx, term_idx, : a.size(0)] = a

            # compute sum of squares (SSE)
            predicted_x = Y @ a
            residuals = samples_i - predicted_x
            sse = torch.sum(residuals**2)

            # evaluate and store the model score
            if sse > 1e-6:
                # avoid log(0)
                if model_criterion == "aic":
                    score = (
                        ml_resolution * torch.log(sse / ml_resolution)
                        + 2 * term
                    )
                elif model_criterion == "bic":
                    score = ml_resolution * torch.log(
                        sse / ml_resolution
                    ) + term * torch.log(torch.tensor(ml_resolution))
                else:
                    # Defaults to SSE
                    score = sse
            else:
                # "perfect" fit
                score = torch.round(sse, decimals=4)

            scores[row_idx, term_idx] = score

    # remove the extra dimension if samples were 1D
    if samples_dim == 1:
        result = result.squeeze(0)
        scores = scores.squeeze(0)

    return result.to(dtype=torch.float32), scores.to(dtype=torch.float32)


@torch.no_grad
@torch.jit.script
def get_metalog_coefficients(
    tensors: torch.Tensor, metalog_term: int
) -> torch.Tensor:
    """
    Returns the metalog coefficients from the given tensors.

    Args:
        tensors (torch.Tensor): Computed metalog coefficients in a [a, b] or [n, a, b] shaped tensor,
          where a is max_num_terms - 3, and b is max_num_terms - 1 and n is a positive integer.
        metalog_term (int): Requested metalog term.

    Returns:
        torch.Tensor: The metalog coefficients from the given tensors.
    """
    # ensure the tensor is three-dimensional [n, a, b]
    if tensors.dim() == 2:
        tensors = tensors.unsqueeze(0)

    # Extract the specified row across all batches.
    extracted_rows = tensors[:, metalog_term - 3, :]

    # create a mask for NaNs
    non_nan_mask = ~torch.isnan(extracted_rows)

    # remove only NaN values
    non_nan_values = extracted_rows[non_nan_mask]

    # reshape the result to ensure it is a 2D tensor with each row representing
    # the non-NaN values
    result_tensor = non_nan_values.view(-1, non_nan_mask.sum(dim=1).max())

    return result_tensor


@torch.no_grad
@torch.jit.script
def inverse_cdf(
    probs: torch.Tensor, cdf: torch.Tensor, bin_edges: torch.Tensor
) -> torch.Tensor:
    """
    Evaluates inverse CDF of a given CDF. This function is used for non metalog distributions.

    Args:
        probs (torch.Tensor): Probabilities to evaluate inverse CDF.
        cdf (torch.Tensor): CDF values.
        bin_edges (torch.Tensor): Bins of the CDF.

    Returns:
        torch.Tensor: Returns the inverse CDF values for the provided probabilities.
    """
    probs = probs.to(torch.float64)
    cdf = cdf.to(torch.float64)
    bin_edges = bin_edges.to(torch.float64)

    # ensure monotonicity of the CDF
    cdf = cdf.cummax(dim=-1).values

    # check that cdf and bin_edges have compatible lengths
    if cdf.shape[-1] != bin_edges.shape[0]:
        raise ValueError("CDF and bin_edges must have the same length.")

    # clamp probs to be within the range of the cdf, and,
    # find indices where each prob value falls in the CDF
    cdf_ndim = cdf.ndim
    if cdf_ndim == 1:
        probs = probs.clamp(min=cdf[0], max=cdf[-1])
        indices = torch.searchsorted(cdf, probs, right=True)
    else:
        # expand to the shapes per broadcasting
        n_conditionals, n_bins = cdf.shape
        bin_edges = bin_edges.expand(n_conditionals, n_bins)
        metalog_cond_res = probs.size(0)
        probs = probs.expand(n_conditionals, metalog_cond_res)
        probs = probs.clamp(
            min=cdf[:, 0].unsqueeze(1), max=cdf[:, -1].unsqueeze(1)
        )
        indices = torch.searchsorted(cdf, probs, right=True)

    # clamp indices to ensure they remain within the valid range
    indices = indices.clamp(min=1, max=cdf.shape[-1] - 1)

    # get lower and upper cdf/bin for interpolation
    indices_lower = indices - 1
    indices_upper = indices
    if cdf_ndim == 1:
        lower_cdf = cdf[indices_lower]
        upper_cdf = cdf[indices_upper]
        lower_bin = bin_edges[indices_lower]
        upper_bin = bin_edges[indices_upper]
    else:
        lower_cdf = torch.gather(cdf, 1, indices_lower)
        upper_cdf = torch.gather(cdf, 1, indices_upper)
        lower_bin = torch.gather(bin_edges, 1, indices_lower)
        upper_bin = torch.gather(bin_edges, 1, indices_upper)

    # interpolate to get the quantiles
    t = torch.zeros_like(probs)
    denom = upper_cdf - lower_cdf
    nonzero_mask = denom != 0
    t[nonzero_mask] = (probs[nonzero_mask] - lower_cdf[nonzero_mask]) / denom[
        nonzero_mask
    ]
    t = torch.clamp(t, min=0, max=1)
    quantiles = lower_bin + t * (upper_bin - lower_bin)

    return quantiles.to(torch.float32)


@torch.no_grad
def estimate_marginal_metalog_cdf(
    a: torch.Tensor,
    X: torch.Tensor,
    bounds: tuple[float, float] | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Estimate the CDF value for random variables using the metalog quantile values.

    Args:
        a (torch.Tensor): Marginal metalog coefficients, shape [1, term].
        X (torch.Tensor): Random variables for which to find the CDF, shape: [sample_size].
        bounds (float | None, optional): Lower and upper bound. Defaults to None.
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Returns:
        torch.Tensor: The estimated CDF values, shape [sample_size].
    """
    # evaluate marginal metalog quantiles to interpolate, shape [1000]
    y = make_probs(1000, device=a.device.type)
    if bounds is None:
        metalog_quantiles = eval_unbounded_metalog_quantile(a, y)
    else:
        metalog_quantiles, _ = eval_bounded_metalog_quantile(
            a,
            bounds[0],
            bounds[1],
            y,
        )

    # ensure X and quantiles are in the same device
    X = X.to(device)
    metalog_quantiles = metalog_quantiles.to(device)

    # init resolution and probs
    ml_resolution = metalog_quantiles.size(0)
    probs = (
        torch.arange(1, ml_resolution + 1, device=device) - 0.5
    ) / ml_resolution

    # create masks for out of bounds values
    if bounds is None:
        below_lower_mask = X < metalog_quantiles[0]
        above_upper_mask = X > metalog_quantiles[-1]
    else:
        below_lower_mask = X <= bounds[0]
        above_upper_mask = X >= bounds[1]

    # initialize result
    interpolated_probs = torch.empty_like(X, device=device)

    # assing probs for out of bounds values
    interpolated_probs[below_lower_mask] = 0.0
    interpolated_probs[above_upper_mask] = 1.0

    # process values within bounds
    within_bounds_mask = ~(below_lower_mask | above_upper_mask)
    X_within = X[within_bounds_mask]

    # get indices for interpolation
    idx_right = torch.searchsorted(metalog_quantiles, X_within, side="right")
    idx_left = idx_right - 1

    # handle edge cases
    idx_right = idx_right.clamp(max=ml_resolution - 1)
    idx_left = idx_left.clamp(min=0)

    # gather necessary quantile and pdf values for interpolation
    x0 = metalog_quantiles[idx_left]
    x1 = metalog_quantiles[idx_right]
    y0 = probs[idx_left]
    y1 = probs[idx_right]

    # compute interpolated probs
    denom = x1 - x0
    interpolated_probs_within = y0 + (X_within - x0) * (y1 - y0) / denom

    # assign computed probs to the result
    interpolated_probs[within_bounds_mask] = interpolated_probs_within

    return interpolated_probs


@torch.no_grad
def estimate_conditional_metalog_cdf(
    a: torch.Tensor,
    X: torch.Tensor,
    bounds: tuple[float, float] | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Estimate CDF values for random variables using the metalog quantiles.

    Args:
        a (torch.Tensor): Conditional metalog coefficients, shape [sample_size, term].
        X (torch.Tensor): Random variables for which to find the CDF, shape: [sample_size].
        bounds (float | None, optional): Lower and upper bound. Defaults to None.
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Returns:
        torch.Tensor: The estimated CDF values.
    """
    # evaluate cond. metalog quantiles to interpolate
    a = a.unsqueeze(-1)  # shape: [sample_size, term, 1]
    y = make_probs(1000, device=a.device.type)
    if bounds is None:
        metalog_quantiles = eval_unbounded_metalog_quantile(
            a, y
        )  # shape [sample_size, 1000]
    else:
        metalog_quantiles, _ = eval_bounded_metalog_quantile(
            a,
            bounds[0],
            bounds[1],
            y,
        )  # shape [sample_size, 1000]

    # ensure X and quantiles are in the same device
    X = X.to(device)
    metalog_quantiles = metalog_quantiles.to(device)

    # init resolution and probs
    n, ml_resolution = metalog_quantiles.shape
    probs = (
        torch.arange(1, ml_resolution + 1, device=device) - 0.5
    ) / ml_resolution

    # expand to the shape [n, ml_resolution]
    probs = probs.expand(n, ml_resolution)

    if bounds is None:
        lower_bounds = metalog_quantiles[:, 0]
        upper_bounds = metalog_quantiles[:, -1]
    else:
        lower_bounds = torch.full((n,), bounds[0], device=device)
        upper_bounds = torch.full((n,), bounds[1], device=device)

    # create masks for out of bounds values
    below_lower_mask = X <= lower_bounds
    above_upper_mask = X >= upper_bounds

    # init the result
    interpolated_probs = torch.empty_like(X, device=device)

    # assign probabilities for out of bounds values
    interpolated_probs[below_lower_mask] = 0.0
    interpolated_probs[above_upper_mask] = 1.0

    # process values within bounds
    within_bounds_mask = ~(below_lower_mask | above_upper_mask)

    # shape of within values [k], where k <= n
    X_within = X[within_bounds_mask]

    # shape of metalog quantile within values [k, ml_resolution]
    metalog_quantile_within = metalog_quantiles[within_bounds_mask, :]

    # for each in X_within, perform interpolation
    # align X's shape for broadcasting, the new shape is [k, 1]
    X_within_expanded = X_within.unsqueeze(1)

    # get indices for interpolation
    idx_right = torch.searchsorted(
        metalog_quantile_within, X_within_expanded, side="right"
    )
    idx_left = idx_right - 1

    # handle edge cases
    idx_right = idx_right.clamp(max=ml_resolution - 1)
    idx_left = idx_left.clamp(min=0)

    # gather necessary quantile and pdf values for interpolation
    x0 = torch.gather(metalog_quantile_within, 1, idx_left)
    x1 = torch.gather(metalog_quantile_within, 1, idx_right)
    y0 = torch.gather(probs[within_bounds_mask], 1, idx_left)
    y1 = torch.gather(probs[within_bounds_mask], 1, idx_right)

    # compute denominator and avoid division by zero
    denom = x1 - x0
    zero_denom_mask = denom == 0
    denom[zero_denom_mask] = 1

    # compute interpolation
    interpolated_probs_within = (
        y0 + (X_within_expanded - x0) * (y1 - y0) / denom
    )
    interpolated_probs_within[zero_denom_mask] = y0[zero_denom_mask]

    # get back to shape [k]
    interpolated_probs_within = interpolated_probs_within.squeeze(1)

    # assign computed probs to the result
    interpolated_probs[within_bounds_mask] = interpolated_probs_within

    return interpolated_probs


@torch.no_grad
def eval_newton_raphson_for_metalog_cdf(
    a: torch.Tensor,
    x: torch.Tensor,
    bounds: tuple[float, float] | None = None,
    max_iter: int = 30,
) -> torch.Tensor:
    """
    Evaluates the metalog CDF value 'y' for a given 'x' using a combination of bisection and Newton-Raphson. See
    also Numerical Recipes: The Art of Scientific Computing, Ch. 9.4 Newton-Raphson Method Using Derivative.

    The metalog quantile function M(y) provides the value 'x' for a given cumulative probability 'y' (i.e., M(y) = x).
    To find the CDF value 'y' corresponding to a specific 'x', we solve the equation:

        M(y) - x = 0

    The Newton-Raphson iterative method updates 'y' using:

        y_new = y - [M(y) - x] / M'(y)

    where:
        M(y) is the metalog quantile function evaluated at 'y'.

    Differentiating the metalog quantile function M(y) with respect to 'y' and inverting the result yields the metalog
    PDF m(y) = 1 / M'(y). That means M'(y) = 1 / m(y).

    Therefore, we can further simplify and get:

        y_new = y - [M(y) - x] * m(y).

    Args:
        a (torch.Tensor): Metalog coefficients in shape [1|N, T].
        x (torch.Tensor): Values at which to evaluate the CDF, shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds of the metalog distribution.
        max_iter (int, optional): Maximum number of iterations for the Newton-Raphson method. Defaults to 30.

    Returns:
        torch.Tensor: Estimated CDF values 'y' corresponding to input 'x', shape: [N].
    """

    def My(a, y):
        if bounds is None:
            M_y = eval_unbounded_metalog_quantile(a, y)
            unbounded_My = None
        else:
            M_y, unbounded_My = eval_bounded_metalog_quantile(
                a, bounds[0], bounds[1], y
            )

        return M_y, unbounded_My

    def my(a, y, unbounded_My):
        if bounds is None:
            m_y = eval_unbounded_metalog_pdf(a, y)
        else:
            m_y = eval_bounded_metalog_pdf(
                a, bounds[0], bounds[1], y, M_y=unbounded_My
            )

        return m_y

    EPSILON = _eps()

    # yl and yh are the lower and upper bounds for the probability 'y'
    yl = torch.full_like(x, EPSILON)
    yh = torch.full_like(x, 1.0 - EPSILON)

    # pick starting values at the midpoint
    y = 0.5 * (yl + yh)
    dx_old = yh - yl
    dx = dx_old.clone()
    converged = torch.zeros_like(x, dtype=torch.bool)

    i = 0
    while (i < max_iter) and torch.any(~converged):
        # active lanes are the ones that are yet not converged
        active = ~converged

        # evaluate the function f(y) = M(y) - x
        M_y, unbounded_My = My(a, y)
        f_y = M_y - x

        # evaluate f'(y) = M'(y) = 1 / m(y), where m(y) is the metalog PDF
        m_y = my(a, y, unbounded_My)
        f_prime = torch.reciprocal(m_y + EPSILON)

        # new y using Newton-Raphson - recall m(y) = 1 / M'(y)
        y_nr = y - f_y * m_y

        """
        Check if the Newton-Raphson step is acceptable. Use Newton-Raphson if
        the conditions met and the lane is `active`. See also `rtsafe()`, Ch.
        9.4 Newton-Raphson Method Using Derivative.
        - Condition 1 (inside): Check if the Newton step remains within the
        bracket.
        - Condition 2 (shrink): Check if bracket is shrinking fast enough.
        """
        inside = (y_nr > yl) & (y_nr < yh)
        shrink = torch.abs(2.0 * f_y) <= torch.abs(dx_old * f_prime)
        use_nr = inside & shrink & active

        # new y using Bisection (fallback)
        y_bi = 0.5 * (yl + yh)

        # assign new y using Newton-Raphson if possible, else use bisection
        y_new = torch.where(use_nr, y_nr, y_bi)

        # ensure new y is in [EPSILON, 1 - EPSILON]
        y_new = torch.clamp(y_new, EPSILON, 1.0 - EPSILON)

        # calculate the step dx based on the method
        dx_newton = y_new - y
        dx_bi = 0.5 * (yh - yl)
        dx_new = torch.where(use_nr, dx_newton, dx_bi)

        # update steps
        dx_old = torch.where(active, dx, dx_old)
        dx = torch.where(active, dx_new, dx)
        y = torch.where(active, y_new, y)

        # update the bracket [yl, yh] using the new position of y
        M_y_new, _ = My(a, y)
        f_new = M_y_new - x
        f_neg = f_new < 0.0
        yl = torch.where(active & f_neg, y, yl)
        yh = torch.where(active & ~f_neg, y, yh)

        # update the convergence mask
        converged |= torch.abs(dx) < 1e-4
        i += 1

    return y


@torch.no_grad
def eval_metalog_inverse_cdf(
    probs: torch.Tensor,
    metalog_coeffs: torch.Tensor,
    bounds: tuple[float, float] | None = None,
) -> torch.Tensor:
    """
    Computes the inverse CDF using the metalog quantile function.

    Args:
        probs (torch.Tensor): Probability for evaluating inverse CDF.
        metalog_coeffs (torch.Tensor): Metalog coefficients.
        bounds (float | None, optional): Lower and upper bound. Defaults to None.

    Returns:
        torch.Tensor: Returns the random variable(s) X for the given probability.
    """
    # Ensure the probability is within the valid range.
    EPSILON = _eps()
    probs = torch.clamp(probs, EPSILON, 1 - EPSILON)

    if bounds is None:
        result = eval_unbounded_metalog_quantile(metalog_coeffs, probs)
    else:
        result, _ = eval_bounded_metalog_quantile(
            metalog_coeffs, bounds[0], bounds[1], probs
        )

    if not torch.all(result[:-1] <= result[1:]):
        logging.warning("Metalog data is not monotonically increasing.")

    return result


@torch.no_grad
def eval_conditional_metalog_distributions_tabular(
    marginal_quantiles: torch.Tensor,
    conditional_metalog_resolution: int,
    xedges: torch.Tensor,
    data2d: torch.Tensor,
    max_num_terms: int,
    yedges: torch.Tensor,
    bounds: tuple[float, float] | None = None,
    device: str = "cuda",
    lambda_: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates conditional metalog distributions for each quantile from the given `marginal_metalog_dist_inverse_cdf`
    using the tabulated data approach.

    Args:
        marginal_quantiles (torch.Tensor): Quantiles from the marginal metalog distribution.
        conditional_metalog_resolution (int): Number of CDF values to evaluate the empirical conditional inverse CDFs.
        xedges (torch.Tensor): The bin edges along the first dimension.
        data2d (torch.Tensor): A 2D tabulated data (e.g., VNDF slice).
        max_num_terms (int): Defines maximum number of metalog terms (e.g., usually np.arange(3, max_num_terms)).
        yedges (torch.Tensor): The bin edges along the second dimension.
        bounds (float | None, optional): Lower and upper bound. Defaults to None.
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".
        lambda_: (float, optional): Regularization term. Defaults to 1e-5.

    NOTE: If `data2d` is a density histogram, then `yedges` must be uniform bins (i.e., y-bins widths should not vary)
    otherwise the result might become biased.

    Returns:
        tuple(torch.Tensor, torch.Tensor, torch.Tensor):
            - A large torch tensor of metalog coefficients shaped [n, a, b], where n is metalog resolution
            (`metalog_resolution = marginal_metalog_dist_inverse_cdf.shape[0]`), where a is max_num_terms - 3, and b is
            max_num_terms - 1. See also `make_metalogs()` for a and b. Each metalog coefficient refers to a conditional
            metalog distribution and corresponds to a quantile from `marginal_metalog_dist_inverse_cdf`.
            - Quantiles that formed conditional metalog distributions in a [n, n] shaped tensor.
            - Conditional metalog scores.
    """
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

    # get marginal_x_counts at the lower and upper bin, shape = [n_samples, 1]
    marginal_x_counts_lower = marginal_x_counts[x_lower_bin_idx].unsqueeze(1)
    marginal_x_counts_upper = marginal_x_counts[x_upper_bin_idx].unsqueeze(1)

    # get histogram slices at the lower and upper bin, shape = [n_samples, num_bins_y]
    hist2d_lower = data2d[x_lower_bin_idx, :]
    hist2d_upper = data2d[x_upper_bin_idx, :]

    # compute conditional distributions
    cond_dist_lower = torch.where(
        marginal_x_counts_lower > 0,
        hist2d_lower / marginal_x_counts_lower,
        torch.zeros_like(hist2d_lower),
    )
    cond_dist_upper = torch.where(
        marginal_x_counts_upper > 0,
        hist2d_upper / marginal_x_counts_upper,
        torch.zeros_like(hist2d_upper),
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
    EPSILON = _eps()
    lower_cond_cdfs /= lower_cond_cdfs[:, -1:].clamp_min(EPSILON)
    upper_cond_cdfs /= upper_cond_cdfs[:, -1:].clamp_min(EPSILON)

    # calculate bin width for interpolation and handle zero bin widths
    x_lower_midpoint = xmidpoints[x_lower_bin_idx]
    x_upper_midpoint = xmidpoints[x_upper_bin_idx]
    interp_width = (x_upper_midpoint - x_lower_midpoint).clamp_(min=1e-8)

    # compute interpolation parameter t, shape [n_samples, 1]
    t = (marginal_quantiles - x_lower_midpoint) / interp_width
    t = torch.clamp(t, min=0, max=1).unsqueeze(1)

    # interpolate lower & upper CDFs and normalize
    cond_cdf = (1 - t) * lower_cond_cdfs + t * upper_cond_cdfs
    cond_cdf /= cond_cdf[:, -1:].clamp_min(EPSILON)

    # invert interpolated CDFs to get the conditional quantiles
    probs = make_probs(
        conditional_metalog_resolution, method="classic", device=device
    )
    cond_quantiles = inverse_cdf(probs, cond_cdf, yedges)

    if not torch.all(cond_quantiles[:, :-1] <= cond_quantiles[:, 1:]):
        logging.warning(
            "Conditional metalog data is not monotonically increasing."
        )

    # fit metalog to the interpolated conditional distribution
    if bounds is None:
        cond_metalog_dist, scores = make_unbounded_metalogs(
            cond_quantiles,
            probs=probs,
            max_num_terms=max_num_terms,
            lambda_=lambda_,
            device=device,
        )
    else:
        cond_metalog_dist, scores = make_bounded_metalogs(
            cond_quantiles,
            probs=probs,
            bounds=bounds,
            max_num_terms=max_num_terms,
            lambda_=lambda_,
            device=device,
        )

    return cond_metalog_dist, cond_quantiles, scores


@torch.no_grad
def sample_bivariate_metalog(
    marginal_metalog_coeffs: torch.Tensor,
    marginal_metalog_dist_inverse_cdf: torch.Tensor,
    cond_metalog_coeffs: torch.Tensor,
    unif2: torch.Tensor,
    marginal_quantile_cdf_range: tuple[float, float],
    bounds_marginal: tuple[float, float] | None = None,
    bounds_conditional: tuple[float, float] | None = None,
) -> SampleRecord:
    """
    Generates samples from the bivariate metalog distribution.

    Note regarding the `compute_interpolated_coeffs()` step:
      We assume that the values of `marginal_quantile_cdfs` are uniformly spaced, when the `metalog_quantiles` is
      initialized. That means, `marginal_quantile_cdfs` is initialized using the `classic` or `linspace` method, when
      `make_probs()` is evaluated. This simplifies the index calculation without the need of a binary search. See also:
      `compute_interpolated_coeffs()`, `init_metalogs()` and `make_probs()`.

    Args:
        marginal_metalog_coeffs (torch.Tensor): Coeffiecients of the marginal metalog distribution in a
        [1, metalog_terms] shaped tensor.
        marginal_metalog_dist_inverse_cdf (torch.Tensor): Quantiles from the marginal metalog distribution in a
        [metalog_resolution] tensor.
        cond_metalog_coeffs (torch.Tensor): Coeffiecients of the conditional metalog distribution in a
        [metalog_resolution, metalog_terms] tensor.
        unif2 (torch.Tensor): Uniform random numbers in range [0, 1] in a [n, 2] tensor.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.
        bounds_marginal (float | None, optional): Lower and upper bound for the marginal. Defaults to None.
        bounds_conditional (float | None, optional): Lower and upper bound for the conditional. Defaults to None.

    Raises:
        ValueError: If unif_sample is not shaped as [n, 2] or marginal metalog coefficients have smaller than 3 metalog
        terms.

    Returns:
        SampleRecord: A sample record that at least contains the samples of the bivariate metalog.
    """
    if (bounds_marginal is None) ^ (bounds_conditional is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    # Ensure tensor is of shape (n, 2)
    if unif2.ndim != 2 or unif2.size(1) != 2:
        raise ValueError("Uniform sample must be shaped (n, 2).")

    term = marginal_metalog_coeffs.size(-1)
    if term <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    # sample x and evaluate the PDF for the marginal
    unif_x = unif2[:, 0]
    if bounds_marginal is None:
        x = eval_unbounded_metalog_quantile(marginal_metalog_coeffs, unif_x)
        pdf_marginal = eval_unbounded_metalog_pdf(
            marginal_metalog_coeffs,
            unif_x,
        )
    else:
        x, unbounded_My = eval_bounded_metalog_quantile(
            marginal_metalog_coeffs,
            bounds_marginal[0],
            bounds_marginal[1],
            unif_x,
        )
        pdf_marginal = eval_bounded_metalog_pdf(
            marginal_metalog_coeffs,
            bounds_marginal[0],
            bounds_marginal[1],
            unif_x,
            M_y=unbounded_My,
        )

    interpolated_coeffs = compute_interpolated_coeffs(
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs,
        x,
        unif_x,
        marginal_quantile_cdf_range,
    )

    # sample y and evaluate the PDF for the conditional
    unif_y = unif2[:, 1]
    if bounds_conditional is None:
        y = eval_unbounded_metalog_quantile(interpolated_coeffs, unif_y)
        pdf_cond = eval_unbounded_metalog_pdf(
            interpolated_coeffs,
            unif_y,
        )
    else:
        y, unbounded_My = eval_bounded_metalog_quantile(
            interpolated_coeffs,
            bounds_conditional[0],
            bounds_conditional[1],
            unif_y,
        )
        pdf_cond = eval_bounded_metalog_pdf(
            interpolated_coeffs,
            bounds_conditional[0],
            bounds_conditional[1],
            unif_y,
            M_y=unbounded_My,
        )

    sRec = SampleRecord(
        unif2=unif2,
        samples=torch.column_stack((x, y)),
        pdf=pdf_marginal * pdf_cond,
    )

    return sRec


@torch.jit.script
def compute_interpolated_coeffs(
    marginal_metalog_dist_inverse_cdf: torch.Tensor,
    cond_metalog_coeffs: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    marginal_quantile_cdf_range: tuple[float, float],
) -> torch.Tensor:
    """
    Computes interpolated metalog coefficients.

    Important:
      We assume that the values of `marginal_quantile_cdfs` are uniformly spaced, when the `metalog_quantiles` is
      initialized. That means, `marginal_quantile_cdfs` is initialized using the `classic` or `linspace` method, when
      `make_probs()` is evaluated. This simplifies the index calculation without the need of a binary search. See also:
      `init_metalogs()` and `make_probs()`.

    Args:
        marginal_metalog_dist_inverse_cdf (torch.Tensor):  Quantiles from the marginal metalog distribution in a
          [metalog_resolution] or [sample_size, metalog_resolution] tensor.
        cond_metalog_coeffs (torch.Tensor): Coeffiecients of the conditional metalog distribution in a
          [metalog_resolution, metalog_terms] tensor.
        x (torch.Tensor): Sample or quantile x from the marginal metalog distribution in a [sample_size] tensor.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.

    Returns:
        torch.Tensor: Interpolated metalog coefficients in a [sample_size, metalog_terms] tensor.
    """
    ndim = cond_metalog_coeffs.ndim
    if ndim not in [2, 3]:
        raise ValueError(
            "Unexpected number of dimensions for the conditional metalogs."
        )

    metalog_res = cond_metalog_coeffs.shape[-2]

    # for each slice, find the index where x would be
    y_min, y_max = marginal_quantile_cdf_range
    idx = (
        torch.floor((y - y_min) / (y_max - y_min) * (metalog_res - 1))
    ).long()
    idx = idx.unsqueeze(1)  # [sample_size, 1]

    # clamp the indices to ensure they fall within the valid range, shape: [sample_size]
    lower_idx = idx.clamp(0, metalog_res - 2).squeeze(1)
    upper_idx = lower_idx + 1

    if cond_metalog_coeffs.ndim == 2:  # Unbatched case
        # Gather directly from the 1D quantile tensor
        lower_quantiles = marginal_metalog_dist_inverse_cdf[lower_idx]
        upper_quantiles = marginal_metalog_dist_inverse_cdf[upper_idx]

        # Gather rows directly from the 2D coefficient tensor
        lower_coeffs = cond_metalog_coeffs[lower_idx]
        upper_coeffs = cond_metalog_coeffs[upper_idx]
    else:  # Batched case
        idx_sample_size = torch.arange(
            marginal_metalog_dist_inverse_cdf.size(0), device=x.device
        )
        lower_quantiles = marginal_metalog_dist_inverse_cdf[
            idx_sample_size, lower_idx
        ]
        upper_quantiles = marginal_metalog_dist_inverse_cdf[
            idx_sample_size, upper_idx
        ]
        lower_coeffs = cond_metalog_coeffs[idx_sample_size, lower_idx]
        upper_coeffs = cond_metalog_coeffs[idx_sample_size, upper_idx]

    # calculate the interpolation ratios
    t = (x - lower_quantiles) / (upper_quantiles - lower_quantiles)
    t = torch.clamp(t, min=0, max=1).unsqueeze(-1)  # [sample_size, 1]

    # linearly interpolate between lower and upper coefficients
    interpolated_coeffs = (
        1 - t
    ) * lower_coeffs + t * upper_coeffs  # [sample_size, metalag_terms]

    return interpolated_coeffs


@torch.no_grad
@torch.jit.script
def spherical_to_unitvec(
    samples2D: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a batch samples in spherical coordinates (theta, phi) to unit vectors (3D).

    Args:
        samples2D (torch.Tensor): An array of shape (n_samples, 2) containing 2D samples.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 3) after conversion.
    """
    theta = samples2D[:, 0]
    phi = samples2D[:, 1]

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.column_stack((x, y, z))


@torch.no_grad
@torch.jit.script
def unitvec_to_spherical(
    samples3D: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a batch samples in spherical coordinates (theta, phi) to unit vectors (3D).

    Args:
        samples3D (torch.Tensor): A tensor of shape (n_samples, 3) containing 3D samples.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) after conversion.
    """
    samples3D = samples3D.reshape(-1, 3)
    x = samples3D[:, 0]
    y = samples3D[:, 1]
    z = samples3D[:, 2]

    theta = torch.acos(z)
    phi = torch.atan2(y, x)

    return torch.column_stack((theta, phi))


@torch.no_grad
@torch.jit.script
def cos_theta_phi_to_unitvec(samples2D: torch.Tensor) -> torch.Tensor:
    """
    Convert samples in (cos_theta, azimuth) coordinates to unit vectors (3D).

    Args:
        samples2D (torch.Tensor): Samples in (cos_theta, phi) coordinates.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 3) after conversion.
    """
    cos_theta = samples2D[:, 0]
    phi = samples2D[:, 1]

    # Clamp cos_theta to valid range
    cos_theta = torch.clamp(cos_theta, 0.0, 1.0)
    r = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0))

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = cos_theta

    return torch.stack((x, y, z), dim=-1)


@torch.no_grad
@torch.jit.script
def unitvec_to_cos_theta_phi(samples3D: torch.Tensor) -> torch.Tensor:
    """
    Convert unit vectors (3D) to samples in (cos_theta, azimuth).

    Args:
        samples3D (torch.Tensor): A tensor of shape (n_samples, 3) containing 3D samples.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2) after conversion.
    """
    s = samples3D.reshape(-1, 3)
    x, y, z = s[:, 0], s[:, 1], s[:, 2]

    cos_theta = torch.clamp(z, 0.0, 1.0)
    phi = torch.atan2(y, x)

    return torch.stack((cos_theta, phi), dim=-1)


@torch.no_grad
def eval_bivariate_metalog_pdf(
    marginal_metalog_coeffs: torch.Tensor,
    marginal_metalog_dist_inverse_cdf: torch.Tensor,
    cond_metalog_coeffs: torch.Tensor,
    sRec: SampleRecord,
    marginal_quantile_cdf_range: tuple[float, float],
    max_newton_iteration: int = 3,
    bounds_marginal: tuple[float, float] | None = None,
    bounds_conditional: tuple[float, float] | None = None,
    logit_transform_marginal: lt.LogitTransform | None = None,
    logit_transform_cond: lt.LogitTransform | None = None,
) -> torch.Tensor:
    """
    Evaluates bivariate metalog pdf.

    Args:
        marginal_metalog_coeffs (torch.Tensor): Coeffiecients of the marginal metalog distribution in a
          [1, metalog_terms] shaped tensor.
        marginal_metalog_dist_inverse_cdf (torch.Tensor): Quantiles from the marginal metalog distribution in a
          [metalog_resolution] shaped tensor.
        cond_metalog_coeffs (torch.Tensor): Coeffiecients of the conditional metalog distribution in a
          [metalog_resolution, metalog_terms] tensor.
        sRec (SampleRecord): Sample record.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.
        max_newton_iteration (int): Maximum number of Newton-Raphson iteration for Metalog CDF. Defaults to 3.
        bounds_marginal (float | None, optional): Lower and upper bound for the marginal. Defaults to None.
        bounds_conditional (float | None, optional): Lower and upper bound for the conditional. Defaults to None.
        logit_transform_marginal (LogitTransform): Transformation applied to marginal metalogs. Currently only
          `LogitTransform` is available. Defaults to None. If specified, then the marginal metalog PDF is transformed
          back to the original domain.
        logit_transform_cond (LogitTransform): Transformation applied to conditional metalogs. Currently only
          `LogitTransform` is available. Defaults to None. If specified, then the conditional metalog PDF is transformed
          back to the original domain.

    Raises:
        ValueError: If `term` is lower than 3.

    Returns:
        torch.Tensor: Computed bivariate metalog PDF in a (num_samples) tensor.
    """
    return eval_bivariate_metalog_cdf_pdf(
        marginal_metalog_coeffs,
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs,
        sRec,
        marginal_quantile_cdf_range,
        max_newton_iteration,
        bounds_marginal,
        bounds_conditional,
        logit_transform_marginal,
        logit_transform_cond,
    )[1]


@torch.no_grad
def eval_bivariate_metalog_cdf_pdf(
    marginal_metalog_coeffs: torch.Tensor,
    marginal_metalog_dist_inverse_cdf: torch.Tensor,
    cond_metalog_coeffs: torch.Tensor,
    sRec: SampleRecord,
    marginal_quantile_cdf_range: tuple[float, float],
    max_newton_iteration: int = 3,
    bounds_marginal: tuple[float, float] | None = None,
    bounds_conditional: tuple[float, float] | None = None,
    logit_transform_marginal: lt.LogitTransform | None = None,
    logit_transform_cond: lt.LogitTransform | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates bivariate metalog CDF and PDF.

    Note regarding the `compute_interpolated_coeffs()` step:
      We assume that the values of `marginal_quantile_cdfs` are uniformly spaced, when the `metalog_quantiles` is
      initialized. That means, `marginal_quantile_cdfs` is initialized using the `classic` or `linspace` method, when
      `make_probs()` is evaluated. This simplifies the index calculation without the need of a binary search. See also:
      `compute_interpolated_coeffs()`, `init_metalogs()` and `make_probs()`.

    Args:
        marginal_metalog_coeffs (torch.Tensor): Coeffiecients of the marginal metalog distribution in a
          [1, metalog_terms] shaped tensor.
        marginal_metalog_dist_inverse_cdf (torch.Tensor): Quantiles from the marginal metalog distribution in a
          [metalog_resolution] shaped tensor.
        cond_metalog_coeffs (torch.Tensor): Coeffiecients of the conditional metalog distribution in a
          [metalog_resolution, metalog_terms] tensor.
        sRec (SampleRecord): Sample record.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.
        max_newton_iteration (int): Maximum number of Newton-Raphson iteration for Metalog CDF. Defaults to 3.
        bounds_marginal (float | None, optional): Lower and upper bound for the marginal. Defaults to None.
        bounds_conditional (float | None, optional): Lower and upper bound for the conditional. Defaults to None.
        logit_transform_marginal (LogitTransform): Transformation applied to marginal metalogs. Currently only
          `LogitTransform` is available. Defaults to None. If specified, then the marginal metalog PDF is transformed
          back to the original domain.
        logit_transform_cond (LogitTransform): Transformation applied to conditional metalogs. Currently only
          `LogitTransform` is available. Defaults to None. If specified, then the conditional metalog PDF is transformed
          back to the original domain.

    Raises:
        ValueError: If `term` is lower than 3.

    Returns:
        tuple:
        - [0]: Marginal-conditional CDF (i.e., uniform variates) of the samples in a [n, 2] shaped tensor.
        - [1]: Computed bivariate metalog PDF in a (num_samples) tensor.
    """
    if (bounds_marginal is None) ^ (bounds_conditional is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    sample2 = sRec.get("samples")
    if sample2 is None:
        raise ValueError("No samples found.")

    if sample2.ndim != 2 or sample2.size(1) != 2:
        raise ValueError("Samples must be shaped (n, 2).")

    # check if uniform variates exist
    use_unif2 = False
    unif2 = sRec.get("unif2")
    if unif2 is not None:
        use_unif2 = True

        if unif2.ndim != 2 or unif2.size(1) != 2:
            raise ValueError("Uniform variates must be shaped (n, 2).")
    else:
        y = sample2[:, 1].contiguous()

    x = sample2[:, 0].contiguous()

    n, term = marginal_metalog_coeffs.shape
    if term <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    if use_unif2:
        unif_x = unif2[:, 0]
    else:
        unif_x = eval_newton_raphson_for_metalog_cdf(
            marginal_metalog_coeffs,
            x,
            bounds_marginal,
            max_newton_iteration,
        )

    # evaluate marginal metalog PDF for x
    if bounds_marginal is None:
        pdf_marginal = eval_unbounded_metalog_pdf(
            marginal_metalog_coeffs,
            unif_x,
        )
    else:
        pdf_marginal = eval_bounded_metalog_pdf(
            marginal_metalog_coeffs,
            bounds_marginal[0],
            bounds_marginal[1],
            unif_x,
        )

    # if logit transform provided, compute inverse logit of marginal PDF
    is_logit_transform_marginal = (
        logit_transform_marginal is not None
        and logit_transform_marginal.get("dimension")
        == lt.TransformDimension.MARGINAL
    )
    if is_logit_transform_marginal:
        x_orig_domain = lt.inverse_logit_samples(
            x,
            logit_transform_marginal,
        )
        pdf_marginal = lt.inverse_logit_pdf(
            x_orig_domain,
            pdf_marginal,
            logit_transform_marginal,
        )

    interpolated_coeffs = compute_interpolated_coeffs(
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs,
        x,
        unif_x,
        marginal_quantile_cdf_range,
    )

    if use_unif2:
        unif_y = unif2[:, 1]
    else:
        unif_y = eval_newton_raphson_for_metalog_cdf(
            interpolated_coeffs,
            y,
            bounds_conditional,
            max_newton_iteration,
        )

    # evaluate conditional metalog PDF for y
    if bounds_conditional is None:
        pdf_cond = eval_unbounded_metalog_pdf(
            interpolated_coeffs,
            unif_y,
        )
    else:
        pdf_cond = eval_bounded_metalog_pdf(
            interpolated_coeffs,
            bounds_conditional[0],
            bounds_conditional[1],
            unif_y,
        )

    # if logit transform provided, compute inverse logit of conditional PDF
    is_logit_transform_conditional = (
        logit_transform_cond is not None
        and logit_transform_cond.get("dimension")
        == lt.TransformDimension.CONDITIONAL
    )
    if is_logit_transform_conditional:
        y_orig_domain = lt.inverse_logit_samples(y, logit_transform_cond)
        pdf_cond = lt.inverse_logit_pdf(
            y_orig_domain,
            pdf_cond,
            logit_transform_cond,
        )

    # compute bivariate pdf
    pdf_joint = pdf_marginal * pdf_cond

    return torch.column_stack((unif_x, unif_y)), pdf_joint


@torch.no_grad
def estimate_bivariate_metalog_pdf(
    samples: torch.Tensor,
    x_edges: torch.Tensor,
    y_edges: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate the PDF using bilinear interpolation over a 2D histogram.

    Args:
        samples (torch.Tensor): An array of sample points, shape (n_samples, 2) or (n_samples, 1, 2).
        x_edges (torch.Tensor): Bin edges along the first dimension.
        y_edges (torch.Tensor): Bin edges along the second dimension.
        H (torch.Tensor): The 2D histogram array.

    Returns:
        torch.Tensor: An array of interpolated densities.
    """

    # ensure samples are in shape (N, 2)
    samples = samples.reshape(-1, 2)

    # identify bins for each sample
    x = samples[:, 0].contiguous()
    y = samples[:, 1].contiguous()
    x_indices = torch.bucketize(x, x_edges) - 1
    y_indices = torch.bucketize(y, y_edges) - 1

    # clip indices to ensure they fall within the valid range for interpolation
    x_indices = torch.clip(x_indices, 0, H.shape[0] - 2)
    y_indices = torch.clip(y_indices, 0, H.shape[1] - 2)

    # compute bin edges for interpolation
    x1, x2 = x_edges[x_indices], x_edges[x_indices + 1]
    y1, y2 = y_edges[y_indices], y_edges[y_indices + 1]

    # extract the four corner values of H for each sample
    H00 = H[x_indices, y_indices]
    H10 = H[x_indices + 1, y_indices]
    H01 = H[x_indices, y_indices + 1]
    H11 = H[x_indices + 1, y_indices + 1]

    # compute weights for bilinear interpolation
    wx1 = (x2 - x) / (x2 - x1)
    wx2 = (x - x1) / (x2 - x1)
    wy1 = (y2 - y) / (y2 - y1)
    wy2 = (y - y1) / (y2 - y1)

    # perform bilinear interpolation
    interpolated_density = (
        H00 * wx1 * wy1 + H10 * wx2 * wy1 + H01 * wx1 * wy2 + H11 * wx2 * wy2
    )

    # ensure valid values
    EPSILON = _eps()
    interpolated_density = torch.clamp(interpolated_density, min=EPSILON)

    return interpolated_density


@torch.no_grad
@torch.jit.script
def mean_unbounded_metalog(
    a: torch.Tensor, device: str = "cuda"
) -> torch.Tensor:
    """
    Computes the mean of provided unbounded metalog. The closed form formulation is taken from
    http://metalogdistributions.com/images/first_four_moments_of_unbounded_metalogs_with_up_to_16_terms.pdf by
    Tom Keelin.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [n, num_metalog_coeffs]
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Returns:
        torch.Tensor: Mean of the unbounded metalog.
    """
    n = a.size(0)
    term = a.size(1)

    if term <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    if term > 16:
        raise ValueError(f"Currently metalogs up to 16 terms are supported.")

    # as suggested in the closed form formulation document, for fewer terms,
    # we set higher order a coefficients to zero

    a0 = a[:, 0]
    a2 = a[:, 2]

    zeros = torch.zeros(n).to(device)
    a4 = a[:, 4] if term >= 5 else zeros
    a7 = a[:, 7] if term >= 8 else zeros
    a8 = a[:, 8] if term >= 9 else zeros
    a11 = a[:, 11] if term >= 12 else zeros
    a12 = a[:, 12] if term >= 13 else zeros
    a15 = a[:, 15] if term == 16 else zeros

    result = a0 + (
        (
            10080 * a2
            + 1680 * a4
            + 1680 * a7
            + 252 * a8
            + 322 * a11
            + 45 * a12
            + 66 * a15
        )
        / 20160
    )

    result = result.unsqueeze(-1)

    return result


@torch.no_grad
@torch.jit.script
def variance_unbounded_metalog(
    a: torch.Tensor, device: str = "cuda"
) -> torch.Tensor:
    """
    Computes the variance of provided unbounded metalog. The closed form formulation is taken from
    http://metalogdistributions.com/images/first_four_moments_of_unbounded_metalogs_with_up_to_16_terms.pdf by
    Tom Keelin.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [n, num_metalog_coeffs]
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Returns:
        torch.Tensor: Variance of the unbounded metalog.
    """
    n = a.size(0)
    term = a.size(1)

    if term <= 2:
        raise ValueError(f"At least 3 terms is necessary.")

    if term > 16:
        raise ValueError(f"Currently metalogs up to 16 terms are supported.")

    # as suggested in the closed form formulation document, for fewer terms,
    # we set higher order a coefficients to zero
    # unlike mean formulation, we name the variables like in the formulation
    zeros = torch.zeros(n).to(device)
    pisqr = torch.pi**2
    a2 = a[:, 1]
    a2sqr = a2**2
    a3 = a[:, 2]
    a3sqr = a3**2
    a4 = a[:, 3] if term >= 4 else zeros
    a4sqr = a4**2
    a5 = a[:, 4] if term >= 5 else zeros
    a5sqr = a5**2
    a6 = a[:, 5] if term >= 6 else zeros
    a6sqr = a6**2
    a7 = a[:, 6] if term >= 7 else zeros
    a7sqr = a7**2
    a8 = a[:, 7] if term >= 8 else zeros
    a8sqr = a8**2
    a9 = a[:, 8] if term >= 9 else zeros
    a9sqr = a9**2
    a10 = a[:, 9] if term >= 10 else zeros
    a10sqr = a10**2
    a11 = a[:, 10] if term >= 11 else zeros
    a11sqr = a11**2
    a12 = a[:, 11] if term >= 12 else zeros
    a12sqr = a12**2
    a13 = a[:, 12] if term >= 13 else zeros
    a13sqr = a13**2
    a14 = a[:, 13] if term >= 14 else zeros
    a14sqr = a14**2
    a15 = a[:, 14] if term >= 15 else zeros
    a15sqr = a15**2
    a16 = a[:, 15] if term == 16 else zeros
    a16sqr = a16**2

    result = (
        1 / 3 * pisqr * a2sqr
        + a3sqr / 12
        + 1 / 36 * pisqr * a3sqr
        + a2 * a4
        + a4sqr / 12
        + a3 * a5 / 12
        + a3 * a5 / 12
        + a5sqr / 180
        + 2 * a2 * a6 / 3
        + 1 / 18 * pisqr * a2 * a6
        + a4 * a6 / 6
        + a6sqr / 12
        + 1 / 240 * pisqr * a6sqr
        + a2 * a7 / 6
        + a4 * a7 / 40
        + 23 * a6 * a7 / 720
        + a7sqr / 448
        + a3 * a8 / 12
        + 1 / 120 * pisqr * a3 * a8
        + 13 * a5 * a8 / 720
        + a8sqr / 80
        + pisqr * a8sqr / 1344
        + 7 * a3 * a9 / 360
        + a5 * a9 / 420
        + a8 * a9 / 224
        + a9sqr / 3600
        + a2 * a10 / 6
        + 1 / 120 * pisqr * a2 * a10
        + 23 * a4 * a10 / 720
        + 7 * a6 * a10 / 180
        + 1 / 672 * pisqr * a6 * a10
        + 11 * a7 * a10 / 1680
        + 409 * a10sqr / 90720
        + pisqr * a10sqr / 6912
        + 23 * a2 * a11 / 720
        + a4 * a11 / 224
        + 11 * a6 * a11 / 1680
        + a7 * a11 / 1152
        + 563 * a10 * a11 / 403200
        + a11sqr / 11264
        + 11 * a3 * a12 / 480
        + 1 / 672 * pisqr * a3 * a12
        + 47 * a5 * a12 / 12096
        + 1153 * a8 * a12 / 181440
        + pisqr * a8 * a12 / 3456
        + 67 * a9 * a12 / 67200
        + 461 * a12sqr / 580608
        + pisqr * a12sqr / 33792
        + 29 * a3 * a13 / 6720
        + a5 * a13 / 2016
        + 59 * a8 * a13 / 57600
        + 3 * a9 * a13 / 24640
        + 499 * a12 * a13 / 2128896
        + 9 * a13sqr / 652288
        + 7 * a2 * a14 / 180
        + 1 / 672 * pisqr * a2 * a14
        + 11 * a4 * a14 / 1680
        + 409 * a6 * a14 / 45360
        + pisqr * a6 * a14 / 3456
        + 563 * a7 * a14 / 403200
        + 47 * a10 * a14 / 22400
        + pisqr * a10 * a14 / 16896
        + 1627 * a11 * a14 / 5322240
        + 13063 * a14sqr / 53222400
        + pisqr * a14sqr / 159744
        + 11 * a2 * a15 / 1680
        + a4 * a15 / 1152
        + 563 * a6 * a15 / 403200
        + a7 * a15 / 5632
        + 1627 * a10 * a15 / 5322240
        + a11 * a15 / 26624
        + 88069 * a14 * a15 / 1291530240
        + a15sqr / 245760
        + 521 * a3 * a16 / 90720
        + pisqr * a3 * a16 / 3456
        + 49 * a5 * a16 / 57600
        + 313 * a8 * a16 / 201600
        + pisqr * a8 * a16 / 16896
        + 851 * a9 * a16 / 3801600
        + 257 * a12 * a16 / 665280
        + pisqr * a12 * a16 / 79872
        + 69193 * a13 * a16 / 1291530240
        + 34847 * a16sqr / 741312000
        + pisqr * a16sqr / 737280
    )
    result = result.unsqueeze(-1)

    return result


@torch.no_grad
def init_metalogs(
    metalogs_filename: str,
    metalog_term: int,
    bounds_marginal: tuple[float, float],
    torch_device: str,
    make_probs_method: str = "classic",
) -> dict:
    """
    Initializes bivariate metalog from the given filename and metalog term.

    Args:
        metalog_filename (str): Filename of the bivariate metalog. Expected file name must have the `.pt` extension.
          The file will be loaded using torch. The loaded file must contain a dictionary with the keys "marginal",
          and "conditional". The "marginal" must correspond to a torch tensor of [max_term - 2, max_term] shaped tensor.
          The "conditional" must correspond to a torch tensor of [n_conditionals, [max_term - 2, max_term] shaped tensor.
        metalog_term (int): Requested metalog term.
        bounds_marginal (float | None, optional): Lower and upper bound for the marginal. Defaults to None.
        torch_device (str): Requested torch device.
        make_probs_method (str, optional): If "classic", then `(np.arange(1, resolution + 1) - 0.5) / resolution`,
          if "chebyshev", then chebyshev points in [0, 1],
          if "linspace", then `np.linspace(EPSILON, 1 - EPSILON, resolution)`. Defaults to "classic".

    Raises:
        ValueError: If the requested metalog term is not available.

    Returns:
        dict: A dictionary where the keys are  "marginal", "conditional", "marginal-quantile-cdfs" and
        "marginal-quantiles" with their corresponding values in torch tensors. The marginal and conditional values
        contains only the specified metalog terms.
    """
    loaded_metalogs = torch.load(metalogs_filename, weights_only=True)

    loaded_marg_coeffs = loaded_metalogs["marginal"]
    marginal_max_term = get_maximum_marginal_term(loaded_marg_coeffs)
    if metalog_term > marginal_max_term:
        raise ValueError(
            f"{metalog_term}-term metalog is not available."
            f"The loaded marginal metalog has maximum {marginal_max_term} term."
        )

    # extract only requested marginal metalog coefficients
    marginal_metalog_coeffs = get_metalog_coefficients(
        loaded_marg_coeffs, metalog_term
    ).to(torch_device)

    loaded_cond_coeffs = loaded_metalogs["conditional"]
    n_conditionals, cond_max_term = get_num_conditionals_and_max_term(
        loaded_cond_coeffs
    )
    if metalog_term > cond_max_term:
        raise ValueError(
            f"{metalog_term}-term metalog is not available."
            f"The loaded conditional metalogs has maximum {cond_max_term} term."
        )

    # extract only requested conditional metalog coefficients
    cond_metalog_coeffs = get_metalog_coefficients(
        loaded_cond_coeffs, metalog_term
    ).to(torch_device)

    marginal_quantile_cdfs = make_probs(
        n_conditionals, method=make_probs_method, device=torch_device
    )
    min_marginal_quantile_cdf = marginal_quantile_cdfs.min()
    max_marginal_quantile_cdf = marginal_quantile_cdfs.max()
    marginal_quantiles = eval_metalog_inverse_cdf(
        marginal_quantile_cdfs,
        marginal_metalog_coeffs,
        bounds_marginal,
    ).to(torch_device)

    return {
        "marginal": marginal_metalog_coeffs,
        "conditional": cond_metalog_coeffs,
        "marginal-quantile-cdfs": marginal_quantile_cdfs,
        "marginal-quantiles": marginal_quantiles,
        "marginal-quantile-cdf-range": [
            min_marginal_quantile_cdf.item(),
            max_marginal_quantile_cdf.item(),
        ],
    }


@torch.no_grad
def init_metalogs_nd(
    metalogs_filename: str,
    metalog_term: int,
    bounds_marginal: tuple[float, float],
    torch_device: str,
):
    """
    Initializes multivariate metalog from the given filename and metalog terms.

    Args:
        metalog_filename (str): Filename of the multivariate metalog. Expected file name must have the `.pt` extension.
          The file will be loaded using torch. The loaded file must contain a dictionary, with the keys (e.g.,
          incident directions (0.0, 0.0)). The corresponding values must contain a dictionary with the keys "marginal",
          and "conditional". The "marginal" must correspond to a torch tensor of [max_term - 2, max_term] shaped tensor.
          The "conditional" must correspond to a torch tensor of [n_conditionals, [max_term - 2, max_term] shaped tensor.
        metalog_term (int): Requested metalog term.
        bounds_marginal (float | None, optional): Lower and upper bound for the marginal. Defaults to None.
        torch_device (str): Requested torch device.

    Raises:
        ValueError: If the requested metalog term is not available.

    Returns:
        dict: A dictionary where the keys are identical to the keys of the loaded metalogs. The corresponding values are
        a dictionary consisting of four keys "marginal", "conditional", "marginal-quantile-cdfs" and
        "marginal-quantiles" with their corresponding values in torch tensors. The marginal and conditional values
        contains only the specified metalog terms.
        int: Metalog resolution in use.
        tuple: Min and max value of the marginal quantile CDF.
    """

    def get_marginal_metalogs(metalog_term, val):
        marginal_metalog = val["marginal"]
        if marginal_metalog.dim() != 2:
            raise ValueError(
                "Expected marginal metalog in a [max_term - 2, max_term] shaped tensor."
            )

        marginal_max_term = marginal_metalog.shape[1]
        if metalog_term > marginal_max_term:
            raise ValueError(
                f"{metalog_term}-term metalog is not available."
                f"The loaded marginal metalog has maximum {marginal_max_term} term."
            )

        # extract only requested marginal metalog coefficients
        marginal_metalog = get_metalog_coefficients(
            marginal_metalog, metalog_term
        ).to(torch_device)

        return marginal_metalog

    loaded_metalogs = torch.load(metalogs_filename, weights_only=True)
    metalogs = {}

    # initialize metalogs dict without the "conditional" metalogs
    if "conditional" not in list(loaded_metalogs[(0.0, 0.0)].keys()):
        # we assume we only have "marginal" metalogs

        for key, val in loaded_metalogs.items():
            marginal_metalog = get_marginal_metalogs(metalog_term, val)

            # put marginal metalogs into a dict for current given key
            metalogs[key] = {
                "marginal": marginal_metalog,
            }

        return metalogs, None, None

    # initialize metalog resolution and CDF range
    metalog_resolution = loaded_metalogs[(0.0, 0.0)]["conditional"].size(0)
    marginal_quantile_cdfs = make_probs(metalog_resolution, device=torch_device)
    marginal_quantile_cdf_range = [
        marginal_quantile_cdfs.min().item(),
        marginal_quantile_cdfs.max().item(),
    ]

    # initialize metalogs dict with the "conditional" metalogs
    for key, val in loaded_metalogs.items():

        marginal_metalog = get_marginal_metalogs(metalog_term, val)

        cond_metalogs = val["conditional"]

        if cond_metalogs.dim() != 3:
            raise ValueError(
                "Expected conditional metalogs in a"
                " [n_conditionals, [max_term - 2, max_term] shaped tensor."
            )

        n_conditionals, _, cond_max_term = cond_metalogs.shape
        if metalog_term > cond_max_term:
            raise ValueError(
                f"{metalog_term}-term metalog is not available."
                f"The loaded conditional metalogs has maximum {cond_max_term} term."
            )

        # extract only requested conditional metalog coefficients
        cond_metalogs = get_metalog_coefficients(
            cond_metalogs, metalog_term
        ).to(torch_device)

        if "marginal_inverse_cdf" in val.keys():
            marginal_quantiles = val["marginal_inverse_cdf"].to(torch_device)
        else:
            marginal_quantiles = eval_metalog_inverse_cdf(
                marginal_quantile_cdfs,
                marginal_metalog,
                bounds_marginal,
            ).to(torch_device)

        # put marginal-conditional metalogs into a dict for current given key
        metalogs[key] = {
            "marginal": marginal_metalog,
            "conditional": cond_metalogs,
            "marginal-quantile-cdfs": marginal_quantile_cdfs,
            "marginal-quantiles": marginal_quantiles,
        }

    return metalogs, metalog_resolution, marginal_quantile_cdf_range


@torch.no_grad
def make_bounded_metalogs(
    samples: torch.Tensor,
    bounds: tuple[float, float],
    probs: torch.Tensor = None,
    lambda_: float = 1e-5,
    max_num_terms: int = 12,
    model_criterion: str = "bic",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Makes bounded metalogs by solving for metalog coefficients.

    Args:
        samples (torch.Tensor): Sorted data points in shape [metalog_resolution] or
          [metalog_resolution, metalog_resolution].
        bounds (float | None, optional): Lower and upper bound. Defaults to None.
        probs (torch.Tensor | None, optional): CDF values of the samples in shape [metalog_resolution].
          Defaults to None.
        lambda_ (float, optional): Regularization term. Defaults to 1e-5.
        max_num_terms (int, optional): Defines maximum number of metalog terms (i.e., np.arange(3, max_num_terms)).
          Defaults to 12.
        model_criterion (str, optional): Criterion for model scoring ('aic', 'bic', 'sse'). Defaults to 'bic'.
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".

    Returns:
        tuple(torch.Tensor, torch.Tensor): Computed metalog coefficients in a (metalog_resolution, max_num_terms - 3, max_num_terms - 1)
          tensor. Model scores for each metalog in a (max_num_terms - 3) tensor.
    """

    lower_bound, upper_bound = bounds
    if lower_bound >= upper_bound:
        raise ValueError(
            f"Lower bound {lower_bound} must be less than upper bound {upper_bound}."
        )

    if torch.any(samples < lower_bound) or torch.any(samples > upper_bound):
        raise ValueError("All samples must be within the specified bounds.")

    samples = samples.to(dtype=torch.float64)
    EPSILON = _eps()

    # transform the data from the bounded space to the unbounded space
    expr = (samples - lower_bound) / (upper_bound - samples)

    # clamp the results to a safe range for the logarithm
    expr_clamped = torch.clamp(expr, min=EPSILON)

    z = torch.log(expr_clamped)

    return make_unbounded_metalogs(
        z, probs, lambda_, max_num_terms, model_criterion, device
    )


@torch.no_grad
def eval_bounded_metalog_quantile(
    a: torch.Tensor,
    b_l: float,
    b_u: float,
    y: torch.Tensor,
):
    """
    Evaluates the bounded metalog quantile function for the given metalog coefficients.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [1|N, T].
        b_l (float): Lower bound.
        b_u (float): Upper bound.
        y (torch.Tensor): CDF values in shape [N,].

    Returns:
        tuple:
        - [0]: Bounded metalog quantiles from the given coefficients.
        - [1]: Unbounded metalog quantiles from the given coefficients.
    """

    M_y = eval_unbounded_metalog_quantile(a, y)

    # bounded metalog (or, logit metalog) for 0 < y < 1
    eM_y = torch.exp(M_y)
    result = (b_l + b_u * eM_y) / (1 + eM_y)

    # handle boundary conditions
    EPSILON = _eps()
    result = torch.where(y <= EPSILON, b_l, result)
    result = torch.where(y >= 1.0 - EPSILON, b_u, result)

    return result, M_y


@torch.no_grad
def eval_bounded_metalog_pdf(
    a: torch.Tensor,
    b_l: float,
    b_u: float,
    y: torch.Tensor,
    M_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes bounded metalog PDF for given CDF values. This is the Metalog PDF implementation.

    Args:
        a (torch.Tensor): Metalog coefficients in shape [1|N, T].
        b_l (float): Lower bound.
        b_u (float): Upper bound.
        y (torch.Tensor): CDF values in shape [N,].
        M_y (torch.Tensor | None, optional): Evaluated unbounded metalog quantile function for y.

    Raises:
        ValueError: If the number of metalog terms is lower than 3.

    Returns:
        torch.Tensor: Bounded PDF value(s) from the given metalog coefficients and CDF value in shape [N,].
    """

    m_y = eval_unbounded_metalog_pdf(a, y)

    # evaluate unbounded metalog quantile if necessary
    if M_y is None:
        M_y = eval_unbounded_metalog_quantile(a, y)

    # bounded metalog (or, logit metalog) pdf for 0 < y < 1
    eM_y = torch.exp(M_y)
    result = m_y * ((1 + eM_y) ** 2) / ((b_u - b_l) * eM_y)

    # handle boundary conditions
    EPSILON = _eps()
    result = torch.where(
        torch.logical_or(y <= EPSILON, y >= 1.0 - EPSILON), 0, result
    )

    return result


@torch.no_grad()
@torch.jit.script
def compute_marginal_cdf(H: torch.Tensor) -> torch.Tensor:
    """
    Computes the marginal CDF from a given 2D histogram.

    Args:
        H (torch.Tensor): 2D histogram.

    Returns:
        torch.Tensor: Marginal (x-axis) CDF of a 2D Histogram.
    """
    marginal_counts = H.sum(dim=1).to(torch.float64)
    total_counts = marginal_counts.sum()
    marginal_dist = marginal_counts / (total_counts + 1e-9)
    marginal_cdf = torch.cumsum(marginal_dist, dim=0)
    marginal_cdf = torch.cat(
        (torch.tensor([0.0], device=marginal_cdf.device), marginal_cdf)
    )
    marginal_cdf[-1] = 1.0
    return marginal_cdf.to(torch.float32)


@torch.no_grad
def make_marginal_conditional_metalogs(
    samples: torch.Tensor,
    device: str,
    regularization_term_marginal: float,
    regularization_term_conditional: float,
    metalog_res_marg: int,
    metalog_res_cond: int,
    max_num_terms: int,
    marginal_bounds: tuple[float, float] | None = None,
    conditional_bounds: tuple[float, float] | None = None,
    logit_transform: bool = False,
    make_probs_method: str = "classic",
    n_conditionals: int = 6,
) -> dict:
    """
    Generates marginal-conditional metalogs from samples.

    Args:
        samples (torch.Tensor): Samples in shape [num_samples, 2].
        device (str): The desired device of returned tensor.
        regularization_term_marginal (float): Regularization term for marginal metalogs.
        regularization_term_conditional (float): Regularization term for conditional metalogs.
        metalog_res_marg (int): Marginal metalog resolution.
        metalog_res_cond (int): Conditional metalog resolution.
        max_num_terms (int): Maximum term of a metalog.
        marginal_bounds (tuple[float, float] | None): Bounds for the marginal axis. Defaults to None.
        conditional_bounds (tuple[float, float] | None): Bounds for the conditional axis. Defaults to None.
        logit_transform (bool): If True, applies logit transform on samples. Default: False.
        make_probs_method (str): Method do generate the CDF values. See also `make_probs()`. Defaults to "classic".
        n_conditionals (int): Number of conditional metalogs.

    Returns:
        metalogs_dict (dict): A dictionary of possible resources relevant for bivariate metalogs.
    """

    if (marginal_bounds is None) ^ (conditional_bounds is None):
        raise ValueError(
            "Bounds must be provided for both marginal and conditionals."
        )

    marginal_conditional_samples = samples.to(device)
    is_unbounded = marginal_bounds is None
    marginal_samples = marginal_conditional_samples[:, 0].view(-1, 1)
    conditional_samples = marginal_conditional_samples[:, 1].view(-1, 1)

    # if specified, logit transform samples
    logit_transform_marginal = None
    logit_transform_cond = None
    if logit_transform:
        (
            marginal_samples,
            conditional_samples,
            logit_transform_marginal,
            logit_transform_cond,
        ) = lt.logit_transform_bivariate_samples(
            marginal_bounds,
            conditional_bounds,
            marginal_samples,
            conditional_samples,
        )

        # if bounds are specified we need to update the bounds to logit domain
        if not is_unbounded:
            marginal_bounds = lt.logit_transform_bounds(marginal_bounds)
            conditional_bounds = lt.logit_transform_bounds(conditional_bounds)

    # #### Compute marginal metalog distribution
    logging.info(f"Computing marginal metalog distribution.")
    marginal_quantile_cdfs = make_probs(
        metalog_res_marg, method=make_probs_method, device=device
    )
    marginal_quantile_cdf_range = [
        marginal_quantile_cdfs.min(),
        marginal_quantile_cdfs.max(),
    ]

    # empirical marginal quantiles at desired CDFs
    empirical_marginal_quantiles = torch.quantile(
        marginal_samples.view(-1).to(device),
        marginal_quantile_cdfs,
    )

    if not torch.all(
        empirical_marginal_quantiles[:-1] <= empirical_marginal_quantiles[1:]
    ):
        logging.warning(
            "Marginal metalog data is not monotonically increasing."
        )

    if is_unbounded:
        marginal_metalogs, marginal_metalog_score = make_unbounded_metalogs(
            empirical_marginal_quantiles,
            probs=marginal_quantile_cdfs,
            max_num_terms=max_num_terms,
            lambda_=regularization_term_marginal,
            device=device,
        )
    else:
        marginal_metalogs, marginal_metalog_score = make_bounded_metalogs(
            empirical_marginal_quantiles,
            bounds=marginal_bounds,
            probs=marginal_quantile_cdfs,
            max_num_terms=max_num_terms,
            lambda_=regularization_term_marginal,
            device=device,
        )

    # select the feasible term with the best score
    metalog_term = find_best_feasible_marginal_metalog(
        marginal_metalogs, marginal_metalog_score
    )

    if metalog_term < 3:
        raise RuntimeError("No feasible marginal metalog model could be found.")

    # #### For all x, compute conditional metalog distributions.
    logging.info(
        f"Computing conditional metalog distributions using {metalog_term}-Term marginal metalogs."
    )
    probs = make_probs(n_conditionals, method=make_probs_method, device=device)
    smoothed_marginal_quantiles = eval_metalog_inverse_cdf(
        probs,
        get_metalog_coefficients(marginal_metalogs, metalog_term),
        bounds=marginal_bounds,
    )

    cond_metalog_dists, cond_metalog_dist_inverse_cdf, cond_metalog_scores = (
        eval_conditional_metalog_distributions_kernel(
            smoothed_marginal_quantiles,
            metalog_res_cond,
            marginal_samples,
            conditional_samples,
            max_num_terms=max_num_terms,
            bounds=conditional_bounds,
            lambda_=regularization_term_conditional,
            device=device,
        )
    )

    metalogs_dict = {
        "data": marginal_conditional_samples,
        "marginal": marginal_metalogs,
        "conditional": cond_metalog_dists,
        "marginal_bounds": marginal_bounds,
        "conditional_bounds": conditional_bounds,
        "marginal_score": marginal_metalog_score,
        "conditional_score": cond_metalog_scores,
        "empirical_marginal_quantiles": empirical_marginal_quantiles,
        "marginal_quantile_cdfs": marginal_quantile_cdfs,
        "marginal_inverse_cdf": smoothed_marginal_quantiles,
        "conditional_inverse_cdf": cond_metalog_dist_inverse_cdf,
        "marginal_quantile_cdf_range": marginal_quantile_cdf_range,
        "logit_transform_marginal": logit_transform_marginal,
        "logit_transform_conditional": logit_transform_cond,
    }

    return metalogs_dict


def save_marginal_conditional_metalogs(metalogs_dict: dict, output_dir: str):
    """
    Saves marginal-conditional metalogs created by `make_marginal_conditional_metalogs()`.

    Args:
        metalogs_dict (dict): A dictionary of possible resources relevant for bivariate metalogs.
        output_dir (str): Output directory to save.
    """

    from os.path import abspath

    # ensure output directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    marginal_metalog_filename = abspath(
        f"{output_dir}/marginal_metalog_dist.pt"
    )
    conditional_metalog_filename = abspath(
        f"{output_dir}/conditional_metalog_dist.pt"
    )
    metalogs_full_filename = abspath(f"{output_dir}/metalogs_vis.pt")
    metalogs_lightweight_filename = abspath(
        f"{output_dir}/metalogs_lightweight.pt"
    )

    marginal_metalogs = metalogs_dict["marginal"]
    cond_metalog_dists = metalogs_dict["conditional"]

    # save marginal metalogs
    torch.save(marginal_metalogs, marginal_metalog_filename)
    logging.info(
        f"Marginal metalog coefficients are saved to {marginal_metalog_filename}."
    )

    # save conditional metalogs
    torch.save(cond_metalog_dists, conditional_metalog_filename)
    logging.info(
        f"Conditional metalog coefficients are saved to {conditional_metalog_filename}."
    )

    # save metalogs dict and its lighter version
    torch.save(
        metalogs_dict,
        metalogs_full_filename,
    )
    metalogs_lightweight = {
        "marginal": marginal_metalogs,
        "conditional": cond_metalog_dists,
        "marginal_bounds": metalogs_dict["marginal_bounds"],
        "conditional_bounds": metalogs_dict["conditional_bounds"],
    }
    torch.save(
        metalogs_lightweight,
        metalogs_lightweight_filename,
    )
    logging.info(f"Metalogs can be found in {metalogs_full_filename}.")

    logit_transform_marginal = metalogs_dict["logit_transform_marginal"]
    logit_transform_cond = metalogs_dict["logit_transform_conditional"]

    if logit_transform_marginal is None or logit_transform_cond is None:
        return

    logit_marginal_filename = abspath(f"{output_dir}/logit_marginal.pt")
    logit_cond_filename = abspath(f"{output_dir}/logit_conditional.pt")

    # save marginal and logit transforms
    marginal_logit_dict = {
        "std": logit_transform_marginal.get("std"),
        "mean": logit_transform_marginal.get("mean"),
        "dimension": logit_transform_marginal.get("dimension"),
        "range_min": logit_transform_marginal.get("range_min"),
        "range_max": logit_transform_marginal.get("range_max"),
    }
    cond_logit_dict = {
        "std": logit_transform_cond.get("std"),
        "mean": logit_transform_cond.get("mean"),
        "dimension": logit_transform_cond.get("dimension"),
        "range_min": logit_transform_cond.get("range_min"),
        "range_max": logit_transform_cond.get("range_max"),
    }
    torch.save(
        marginal_logit_dict,
        logit_marginal_filename,
    )
    logging.info(
        f"Marginal logit transform is saved to {logit_marginal_filename}."
    )
    torch.save(
        cond_logit_dict,
        logit_cond_filename,
    )
    logging.info(
        f"Conditional logit transform is saved to {logit_cond_filename}."
    )


def get_maximum_marginal_term(marginal_metalog):
    if marginal_metalog.dim() != 2:
        raise ValueError(
            "Expected marginal metalog in a [max_term - 2, max_term] shaped tensor."
        )

    return marginal_metalog.shape[1]


def get_num_conditionals_and_max_term(
    conditional_metalog: torch.Tensor,
) -> tuple[int, int, int]:
    """
    Returns the number of conditionals and maximum available terms of the conditional metalogs after checking its
    validity.

    Args:
        conditional_metalog (torch.Tensor): Conditional metalogs in a [n_conditionals, max_term - 2, max_term] shaped
        tensor.

    Raises:
        ValueError: If the dimensionality of the conditional metalogs is invalid.

    Returns:
        tuple[int, int]: Number of conditional metalogs, maximum available terms.
    """
    if conditional_metalog.dim() != 3:
        raise ValueError(
            "Expected conditional metalogs in a"
            " [n_conditionals, max_term - 2, max_term] shaped tensor."
        )

    n_conditionals, _, max_term = conditional_metalog.shape
    return (n_conditionals, max_term)


@torch.no_grad
@torch.jit.script
def find_metalogs(
    wi2D: torch.Tensor,
    metalogs_keys: torch.Tensor,
    marginal_metalogs: torch.Tensor,
    marginal_quantiles: torch.Tensor,
    conditional_metalogs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given an incident direction in spherical domain, calculate the interpolated marginal-conditional coefficients and
    marginal quantiles.

    Args:
        wi2D (torch.Tensor): Incident direction in spherical coordinates, shape: [sample_size, 2].
        metalogs_keys (torch.Tensor): Measured angles in spherical coordinates (i.e., [phi_i, theta_i], shape: [n_slices, 2].
        marginal_metalogs (torch.Tensor): Marginal metalog coefficients, shape: [n_slices, metalog_term].
        marginal_quantiles (torch.Tensor): Marginal quantiles, shape: [n_slices, metalog_resolution].
        conditional_metalogs (torch.Tensor): Conditional metalog coefficients, shape: [n_slices, metalog_resolution, metalog_term].

    Returns:
        tuple:
        - [0]: Interpolated marginal metalog coefficients for the given wi2D.
        - [1]: Interpolated conditional metalog coefficients for the given wi2D.
        - [2]: Interpolated marginal quantiles for the given wi2D.
    """
    # find the eucledian distance between wi and the metalog keys
    dists = torch.cdist(wi2D, metalogs_keys)  # shape: [sample_size, n_slices]

    # for each wi, find the index of the closest dictionary key, shape: [sample_size]
    _, top2_indices = torch.topk(dists, k=2, dim=1, largest=False)
    idx0 = top2_indices[:, 0]
    idx1 = top2_indices[:, 1]

    # gather marginal metalogs, shape: [sample_size, metalog_term]
    lower_wi_marginal_metalogs = marginal_metalogs[idx0]
    upper_wi_marginal_metalogs = marginal_metalogs[idx1]

    # gather conditional metalogs, shape: [sample_size, metalog_resolution, metalog_term]
    lower_wi_conditional_metalogs = conditional_metalogs[idx0]
    upper_wi_conditional_metalogs = conditional_metalogs[idx1]

    # gather marginal quantiles, shape: [sample_size, metalog_resolution]
    lower_wi_marginal_quantiles = marginal_quantiles[idx0]
    upper_wi_marginal_quantiles = marginal_quantiles[idx1]

    # get lower / upper keys
    lower_keys = metalogs_keys[idx0]  # shape: [sample_size, 2]
    upper_keys = metalogs_keys[idx1]

    # calculate the interpolation ratios
    v = upper_keys - lower_keys
    w = wi2D - lower_keys
    dot_vv = torch.sum(v * v, dim=1)
    dot_wv = torch.sum(w * v, dim=1)
    t = torch.clamp(dot_wv / dot_vv, min=0.0, max=1.0)

    # linearly interpolate between lower and upper values
    t_marginal = t.unsqueeze(1)  # shape: [sample_size, 1]
    t_conditional = t.unsqueeze(1).unsqueeze(2)  # shape: [sample_size, 1, 1]
    interpolated_marginal_metalogs = (
        lower_wi_marginal_metalogs * (1 - t_marginal)
        + upper_wi_marginal_metalogs * t_marginal
    )  # shape: [sample_size, metalog_term]
    interpolated_conditional_metalogs = (
        lower_wi_conditional_metalogs * (1 - t_conditional)
        + upper_wi_conditional_metalogs * t_conditional
    )  # shape: [sample_size, metalog_resolution, metalog_term]
    interpolated_marginal_quantiles = (
        lower_wi_marginal_quantiles * (1 - t_marginal)
        + upper_wi_marginal_quantiles * t_marginal
    )  # shape: [sample_size, metalog_resolution]

    return (
        interpolated_marginal_metalogs,
        interpolated_conditional_metalogs,
        interpolated_marginal_quantiles,
    )


def find_best_feasible_marginal_metalog(
    marginal_metalogs: torch.Tensor,
    marginal_metalog_score: torch.Tensor,
) -> int:
    """
    Find the best feasible marginal metalog term.

    Args:
        marginal_metalogs (torch.Tensor): Marginal metalog coefficients in [1, max_num_terms - 3, max_num_terms - 1].
        marginal_metalog_score (torch.Tensor): Marginal metalog model scores for each metalog in [max_num_terms - 3].

    Returns:
        int: Best marginal metalog terms according to the given marginal metalog score. If none of the terms is
        feasible, returns -1.
    """
    best_marginal_term = -1
    _, best_scoring_metalog_terms = torch.sort(marginal_metalog_score)
    best_scoring_metalog_terms = best_scoring_metalog_terms + 3
    for term_idx in best_scoring_metalog_terms:
        coeffs = get_metalog_coefficients(marginal_metalogs, term_idx)
        if is_feasible_metalog(coeffs):
            best_marginal_term = term_idx
            break
    return best_marginal_term


def solid_angle_pdf(
    rectangular_pdf: torch.Tensor, cos_theta: torch.Tensor
) -> torch.Tensor:
    """
    Converts rectangular (theta,phi) density to per-steradian density p_w, where w refers to omega.

    Mitsuba PDF is per unit solid angle p_w(w)dw. We reparameterize w by spherical angles (theta,phi):

        w(theta,phi)=(sin(theta)cos(phi),sin(theta)sin(phi),cos(theta))

    On the unit sphere the surface/solid angle element is dw = sin(theta) dthetadphi, that is, sin(theta) is the surface
    Jacobian. As such, the PDFs become p_(theta,phi) = p_w sin(theta), and p_w = p_(theta,phi) / sin(theta).

    Args:
        rectangular_pdf (torch.Tensor): Tensor of p_(theta,phi) evaluated at samples.
        cos_theta (torch.Tensor): Tensor of mu=cos(theta) for the samples; values in [-1, 1], or [0, 1] for a hemisphere.

    Returns:
        torch.Tensor: Rectangular p_(theta,phi) density to solid angle density: p_w = p_(theta,phi) / sin(theta).
    """
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
    denom = torch.clamp(sin_theta, min=1e-12)
    p_omega = rectangular_pdf * torch.reciprocal(denom)
    return p_omega


def weighted_quantiles(
    X_sorted: torch.Tensor, W_sorted: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute weighted empirical quantiles for multiple weight-vectors.

    Args:
        X_sorted (torch.Tensor): 1D tensor of sorted samples, shape: (N, ).
        W_sorted (torch.Tensor): A 2D tensor of non negative weights corresponding to `X_sorted`. Each row is a separate
        weight-vector over the same `X_sorted`, shape: (K, N).
        probs (torch.Tensor): 1D cumulative probabilities for target quantiles, shape: (R, ).

    Returns:
        torch.Tensor: Weighted quantiles for each row of `W_sorted` and each prob in `probs`. shape: (K, R), where
        output[k, r] is the `probs[r]`-quantile under weights `W_sorted[k]`.
    """
    K, N = W_sorted.shape

    # exact normalization (row-wise), shape: (K, N)
    # w[k, :] is a discrete probability distribution over X_sorted for row k
    w = W_sorted
    w = w / w.sum(dim=1, keepdim=True)

    # row-wise CDF
    cdf = torch.cumsum(w, dim=1).clamp(0.0, 1.0)

    Q_cols = []
    K_i = torch.arange(K, device=X_sorted.device)

    # for each target probability p, perform a weighted 1D inverse-CDF lookup
    for p in probs:
        # first index where cdf >= p, clamped to [1, N-1] for interpolation
        j = (cdf >= p).float().argmax(dim=1).clamp(1, N - 1)

        c0 = cdf[K_i, j - 1]
        c1 = cdf[K_i, j]
        x0 = X_sorted[j - 1]
        x1 = X_sorted[j]

        # lerp in CDF space between (c0, x0) and (c1, x1)
        t = ((p - c0) / (c1 - c0)).clamp(0.0, 1.0)
        Q_cols.append(x0 + t * (x1 - x0))

    return torch.stack(Q_cols, dim=1)


def marginal_kernel_weights(
    x_marg_samples: torch.Tensor,
    marginal_quantiles: torch.Tensor,
    bandwidth: float | torch.Tensor | None = None,
    cutoff: float = 0.0,
) -> torch.Tensor:
    """
    Compute Gaussian kernel weights in the marginal dimension.

    Args:
        x_marg_samples (torch.Tensor): 1D tensor of marginal samples, shape (N,).
        marginal_quantiles (torch.Tensor): 1D tensor of marginal locations to condition on, shape (J,)
        bandwidth (float | torch.Tensor | None, optional): Scalar bandwidth h. Defaults to None.
        cutoff (float, optional): If > 0, zero out samples with |(x - x*)/h| > cutoff. Defaults to 3.0.

    Returns:
        torch.Tensor: (J, N) weight matrix, each row normalized to sum ~1.
    """
    device = x_marg_samples.device
    x_marg_samples = x_marg_samples.view(-1).to(device)
    x_marg_query = marginal_quantiles.view(-1).to(device)

    if bandwidth is None:
        """
        Simple Silverman style bandwith for 1D data.
        See also https://stats.stackexchange.com/q/6671.
        """
        N = x_marg_samples.numel()
        std = x_marg_samples.std()
        if N <= 1:
            h = torch.tensor(1.0, device=x_marg_samples.device)
        else:
            h = (1.06 * std * (N ** (-1.0 / 5.0))).clamp_min(1e-8)
    else:
        h = torch.as_tensor(bandwidth, device=device).clamp_min(1e-8)

    # create a mask for cutoff |(x - x*)/h|
    diff = (x_marg_query.unsqueeze(1) - x_marg_samples.unsqueeze(0)) / h
    if cutoff is not None and cutoff > 0.0:
        mask = diff.abs() <= cutoff
    else:
        mask = None

    # compute weight using a Gaussian kernel
    w = torch.exp(-0.5 * diff * diff)
    if mask is not None:
        w = w * mask

    # normalize weights
    EPSILON = _eps()
    w = w + EPSILON
    w = w / (w.sum(dim=1, keepdim=True) + EPSILON)

    return w


@torch.no_grad
def eval_conditional_metalog_distributions_kernel(
    marginal_quantiles: torch.Tensor,
    conditional_metalog_resolution: int,
    x_marg_samples: torch.Tensor,
    x_cond_samples: torch.Tensor,
    max_num_terms: int,
    bounds: tuple[float, float] | None = None,
    device: str | torch.device = "cuda",
    lambda_: float = 1e-5,
    bandwidth: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates conditional metalog distributions for each quantile from the given `marginal_metalog_dist_inverse_cdf`
    based on the kernel approach.

    Fits a family of conditional metalog distributions p(Y | X = x_j) from paired samples (X, Y) using kernel-weidhted
    empirical quantiles. For each quantile x_j in `marginal_quantiles`:
        1. Define a local conditional distribution over the observed Y samples by assigning kernel weights to each (X_i,
        Y_i) based on the distance between X_i and x_j in the marginal dimension.
        2. Compute kernel-weighted empirical quantiles of Y at a fixed set of CDFs of length
        `conditional_metalog_resolution`.
        3. Fit a 1D metalog distribution to those empirical conditional quantiles.

    The result is a collection of 1D metalogs, one for each x_j such that p(Y | X = x_j) obtained smoothly.

    Args:
        marginal_quantiles (torch.Tensor): Quantiles from the marginal metalog distribution, shape: (J,).
        conditional_metalog_resolution (int): Number of CDF values to evaluate the empirical conditional inverse CDFs.
        x_marg_samples: Raw marginal samples, shape (N,).
        x_cond_samples: Raw conditional samples, shape (N,).
        max_num_terms (int): Defines maximum number of metalog terms (e.g., usually np.arange(3, max_num_terms)).
        bounds (float | None, optional): Lower and upper bound. Defaults to None.
        device (str, optional): The desired device of returned tensor. Defaults to "cuda".
        lambda_ (float, optional): Regularization term. Defaults to 1e-5.
        bandwith (float, optional): Kernel bandwidth h in the marginal dimension. Defaults to None.

    Returns:
        - cond_metalog_dist (torch.Tensor): Metalog coefficient tensor of shape [J, max_num_terms-3, max_num_terms-1].
        - cond_quantiles (torch.Tensor): Empirical conditional quantiles used for fitting, shape [J, R], where R is
         `conditional_metalog_resolution`.
        - scores (torch.Tensor): Conditional metalog scores.
    """
    marginal_quantiles = marginal_quantiles.to(device).view(-1)

    x_marg_samples = x_marg_samples.to(device).view(-1)
    x_cond_samples = x_cond_samples.to(device).view(-1)
    assert (
        x_marg_samples.shape == x_cond_samples.shape
    ), "x_marg_samples and x_cond_samples must have the same shape."

    # sort conditional samples
    x_cond_sorted, sort_idx = torch.sort(x_cond_samples)

    # kernel weights over marginal dimension, shape: (J, N)
    W = marginal_kernel_weights(x_marg_samples, marginal_quantiles, bandwidth)

    # reorder weights to align with sorted conditional samples
    W_sorted = W[:, sort_idx]

    # cumulative probabilities for conditionals, shape: (R, )
    probs = make_probs(
        conditional_metalog_resolution, method="classic", device=device
    )

    # for each row compute weighted quantiles, shape: (J, R)
    cond_quantiles = weighted_quantiles(x_cond_sorted, W_sorted, probs)

    if not torch.all(cond_quantiles[:, :-1] <= cond_quantiles[:, 1:]):
        logging.warning(
            "Conditional metalog data is not monotonically increasing."
        )

    # fit metalogs row-wise
    if bounds is None:
        cond_metalog_dist, scores = make_unbounded_metalogs(
            cond_quantiles,
            probs=probs,
            max_num_terms=max_num_terms,
            lambda_=lambda_,
            device=device,
        )
    else:
        cond_metalog_dist, scores = make_bounded_metalogs(
            cond_quantiles,
            probs=probs,
            bounds=bounds,
            max_num_terms=max_num_terms,
            lambda_=lambda_,
            device=device,
        )

    return cond_metalog_dist, cond_quantiles, scores
