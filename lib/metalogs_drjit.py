# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import drjit as dr
import mitsuba as mi
from lib.utils.sample_record import SampleRecord

from lib.metalogs import _eps

EPSILON = _eps()
HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS = 13


@dr.syntax
def gather_metalog_coeffs(a: mi.TensorXf, row_idx: mi.UInt) -> list[mi.Float]:
    """
    Gather coefficients at `idx` (row-major), then return the per-sample coefficients as a list of Float arrays, one
    entry per term.

    Args:
        a (mi.TensorXf): Metalog coefficients, shape: [R, T] or [N, R, T].
        row_idx (mi.UInt): Flattened row-major indices per sample, shape: [N].

    Returns:
        list[mi.Float]: Return per-sample coefficients in a list, one entry per term (i.e., len = T, where each entry
        has shape [N]).
    """
    T = a.shape[-1]
    a = a.array
    base_idx = row_idx * T

    a_terms: list[mi.Float] = []
    """
    NOTE: Setting n to `n = mi.UInt(0)` leads to:
    RuntimeError: dr.while_loop(): internal error: the JIT variable type of state variable 'a_interp[0]' (of type
    'drjit.cuda.ad.Float') changed from 'float32' to 'uint32'. Please review the interface and assumptions of
    'drjit.while_loop()' as explained in the documentation.
    """
    n = 0
    while n < T:
        a_terms.append(dr.gather(mi.Float, a, base_idx + n))
        n += 1

    return a_terms


@dr.syntax
def gather_lerp_metalog_coeffs(
    a: mi.TensorXf, lower_idx: mi.UInt, upper_idx: mi.UInt, t: mi.Float
) -> list[mi.Float]:
    """
    Gather coefficients at `lower_idx` and `upper_idx` (row-major), then return the per-sample interpolated coefficients
    as a list of Float arrays, one entry per term.

    Args:
        a (mi.TensorXf): Metalog coefficients, shape: [R, T] or [N, R, T].
        lower_idx (mi.UInt): Flattened lower row-major indices per sample, shape: [N].
        upper_idx (mi.UInt): Flattened upper row-major indices per sample, shape: [N].
        t (mi.Float): Interpolation weights per sample, shape: [N].

    Returns:
        list[mi.Float]: Return per-sample interpolated coefficients in a list, one entry per term (i.e., len = T, where
        each entry has shape [N]).
    """
    T = a.shape[-1]
    a = a.array
    base_lo, base_hi = lower_idx * T, upper_idx * T

    a_interp: list[mi.Float] = []
    """
    NOTE: Setting n to `n = mi.UInt(0)` leads to:
    RuntimeError: dr.while_loop(): internal error: the JIT variable type of state variable 'a_interp[0]' (of type
    'drjit.cuda.ad.Float') changed from 'float32' to 'uint32'. Please review the interface and assumptions of
    'drjit.while_loop()' as explained in the documentation.
    """
    n = 0
    while n < T:
        a_lo = dr.gather(mi.Float, a, base_lo + n)
        a_hi = dr.gather(mi.Float, a, base_hi + n)
        a_interp.append(dr.lerp(a_lo, a_hi, t))
        n += 1

    return a_interp


@dr.syntax
def gather_bilerp_marginal_metalog_coeffs(
    a: mi.TensorXf,
    idx00: mi.UInt,
    idx01: mi.UInt,
    idx10: mi.UInt,
    idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
) -> list[mi.Float]:
    """
    Term-wise bilinearly interpolate marginal metalog coeffiecients across four neighboring slices on the (phi, theta)
    grid.

    Args:
        a (mi.TensorXf): Marginal metalog coefficients, shape: [S, T].
        idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].

    Returns:
        list[mi.Float]: Length-T list, each entry is an `mi.Float` of shape [N] containing the bilinearly interpolated
        coefficient for that term.
    """
    T = a.shape[-1]
    a = a.array

    # build base indices for each slice
    base_idx00 = idx00 * T
    base_idx01 = idx01 * T
    base_idx10 = idx10 * T
    base_idx11 = idx11 * T

    result = []
    """
    NOTE: Setting n to `n = mi.UInt(0)` leads to:
    RuntimeError: dr.while_loop(): internal error: the JIT variable type of state variable 'a_interp[0]' (of type
    'drjit.cuda.ad.Float') changed from 'float32' to 'uint32'. Please review the interface and assumptions of
    'drjit.while_loop()' as explained in the documentation.
    """
    n = 0
    while n < T:
        # gather term n at the four corners
        a_00 = dr.gather(mi.Float, a, base_idx00 + n)
        a_01 = dr.gather(mi.Float, a, base_idx01 + n)
        a_10 = dr.gather(mi.Float, a, base_idx10 + n)
        a_11 = dr.gather(mi.Float, a, base_idx11 + n)

        # interpolate across theta then across phi
        a_0 = dr.lerp(a_00, a_01, t_theta)
        a_1 = dr.lerp(a_10, a_11, t_theta)
        result.append(dr.lerp(a_0, a_1, t_phi))
        n += 1

    return result


@dr.syntax
def gather_slice_lerp_metalog_coeffs(
    metalog_coeffs: mi.TensorXf,
    idx0: mi.UInt,
    idx1: mi.UInt,
    lower_idx: mi.UInt,
    upper_idx: mi.UInt,
    t0: mi.Float,
    t1: mi.Float,
    t: mi.Float,
) -> list[mi.Float]:
    """
    Row-wise linearly interpolate inside each of 2 slices, then linearly interpolate across slices.

    For each sample we gather:
        - slice 0 at lower (`lower_idx`) and upper (`upper_idx`) row,
        - slice 1 at lower (`lower_idx`) and upper (`upper_idx`) row,
    then we:
        1. Interpolate within each slice using t0/t1 (row-wise ratio at x).
        2. Interpolate across slices using t (Shepard weight for slice 1).

    Args:
        metalog_coeffs (mi.TensorXf): Conditional metalog coefficients, shape: [S, R, T].
        idx0 (mi.UInt): Per-sample slice 0 indices, shape: [N].
        idx1 (mi.UInt): Per-sample slice 1 indices, shape: [N].
        lower_idx (mi.UInt): Per-sample lower row indices, shape: [N].
        upper_idx (mi.UInt): Per-sample upper row indices, shape: [N].
        t0 (mi.Float): Within-slice interpolation ratio for slice 0, shape: [N].
        t1 (mi.Float): Within-slice interpolation ratio for slice 1, shape: [N].
        t: Cross-slice interpolation ratio for slice 1 (slice 0 uses 1 - t).

    Returns:
        list[mi.Float]: List of length T. Each entry is an `mi.Float` of shape [N] holding the interpolated coefficient
        for that term.
    """
    S, R, T = metalog_coeffs.shape
    metalog_coeffs = metalog_coeffs.array

    # build row-major indices: row = s * R + r
    base_idx0 = idx0 * R
    base_idx1 = idx1 * R
    row_idx0_lo = base_idx0 + lower_idx
    row_idx0_hi = base_idx0 + upper_idx
    row_idx1_lo = base_idx1 + lower_idx
    row_idx1_hi = base_idx1 + upper_idx

    # convert row indices to flat term base indices: row * T
    base_idx0_lo, base_idx0_hi = row_idx0_lo * T, row_idx0_hi * T
    base_idx1_lo, base_idx1_hi = row_idx1_lo * T, row_idx1_hi * T

    result = []
    for n in range(T):
        # gather lower and upper rows for each slice
        slice0_row_lo = dr.gather(mi.Float, metalog_coeffs, base_idx0_lo + n)
        slice0_row_hi = dr.gather(mi.Float, metalog_coeffs, base_idx0_hi + n)
        slice1_row_lo = dr.gather(mi.Float, metalog_coeffs, base_idx1_lo + n)
        slice1_row_hi = dr.gather(mi.Float, metalog_coeffs, base_idx1_hi + n)

        # interpolate within slice 0 and 1 (row-wise)
        slice0_coeffs = dr.lerp(slice0_row_lo, slice0_row_hi, t0)
        slice1_coeffs = dr.lerp(slice1_row_lo, slice1_row_hi, t1)

        # interpolate across slices
        result.append(dr.lerp(slice0_coeffs, slice1_coeffs, t))

    return result


@dr.syntax
def gather_slice_bilerp_metalog_coeffs(
    metalog_coeffs: mi.TensorXf,
    idx00: mi.UInt,
    idx01: mi.UInt,
    idx10: mi.UInt,
    idx11: mi.UInt,
    lower_idx: mi.UInt,
    upper_idx: mi.UInt,
    t00: mi.Float,
    t01: mi.Float,
    t10: mi.Float,
    t11: mi.Float,
    t_theta: mi.Float,
    t_phi: mi.Float,
) -> list[mi.Float]:
    """
    Row-wise linearly interpolate inside each of 4 slices, then bilinearly interpolate across slices.

    For each sample we gather:
        - slice 00 at lower (`lower_idx`) and upper (`upper_idx`) row,
        - slice 01 at lower (`lower_idx`) and upper (`upper_idx`) row,
        - slice 10 at lower (`lower_idx`) and upper (`upper_idx`) row,
        - slice 11 at lower (`lower_idx`) and upper (`upper_idx`) row,
    then we:
        1. Row-wise linearly interpolate each slice using its own row ratio (t00, t01, t10, t11).
        2. Lineraly interpolate across theta within each phi row.
        3. Lineraly interpolate across phi between rows.

    Args:
        metalog_coeffs (mi.TensorXf): Conditional metalog coefficients, shape: [S, R, T].
        idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        lower_idx (mi.UInt): Per-sample lower row indices, shape: [N].
        upper_idx (mi.UInt): Per-sample upper row indices, shape: [N].
        t00 (mi.Float): Within-slice interpolation ratio for slice 00, shape: [N].
        t01 (mi.Float): Within-slice interpolation ratio for slice 01, shape: [N].
        t10 (mi.Float): Within-slice interpolation ratio for slice 10, shape: [N].
        t11 (mi.Float): Within-slice interpolation ratio for slice 11, shape: [N].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].

    Returns:
        list[mi.Float]: Length T list, each entry is [N]-shaped `mi.Float` holding the bilinearly interpolated
        coefficient for that term.
    """
    S, R, T = metalog_coeffs.shape
    metalog_coeffs = metalog_coeffs.array

    def calc_lower_and_upper_row_idx(slice_idx: mi.UInt) -> mi.UInt:
        """Build lower and upper row-major bases for each slice"""
        base_idx = slice_idx * R
        row_lo = (base_idx + lower_idx) * T
        row_hi = (base_idx + upper_idx) * T
        return row_lo, row_hi

    base_idx00_lo, base_idx00_hi = calc_lower_and_upper_row_idx(idx00)
    base_idx01_lo, base_idx01_hi = calc_lower_and_upper_row_idx(idx01)
    base_idx10_lo, base_idx10_hi = calc_lower_and_upper_row_idx(idx10)
    base_idx11_lo, base_idx11_hi = calc_lower_and_upper_row_idx(idx11)

    result = []
    for n in range(T):
        # gather lower and upper rows for each slice with term n
        slice00_row_lo = dr.gather(mi.Float, metalog_coeffs, base_idx00_lo + n)
        slice00_row_hi = dr.gather(mi.Float, metalog_coeffs, base_idx00_hi + n)
        slice01_row_lo = dr.gather(mi.Float, metalog_coeffs, base_idx01_lo + n)
        slice01_row_hi = dr.gather(mi.Float, metalog_coeffs, base_idx01_hi + n)
        slice10_row_lo = dr.gather(mi.Float, metalog_coeffs, base_idx10_lo + n)
        slice10_row_hi = dr.gather(mi.Float, metalog_coeffs, base_idx10_hi + n)
        slice11_row_lo = dr.gather(mi.Float, metalog_coeffs, base_idx11_lo + n)
        slice11_row_hi = dr.gather(mi.Float, metalog_coeffs, base_idx11_hi + n)

        # row-wise interpolation in each slice
        slice00_coeffs = dr.lerp(slice00_row_lo, slice00_row_hi, t00)
        slice01_coeffs = dr.lerp(slice01_row_lo, slice01_row_hi, t01)
        slice10_coeffs = dr.lerp(slice10_row_lo, slice10_row_hi, t10)
        slice11_coeffs = dr.lerp(slice11_row_lo, slice11_row_hi, t11)

        # bilinear interpolation across slices
        slice0_coeffs = dr.lerp(slice00_coeffs, slice01_coeffs, t_theta)
        slice1_coeffs = dr.lerp(slice10_coeffs, slice11_coeffs, t_theta)
        result.append(dr.lerp(slice0_coeffs, slice1_coeffs, t_phi))

    return result


@dr.syntax
def eval_unbounded_metalog_pdf(
    a: list[mi.Float],
    y: mi.Float,
) -> mi.Float:
    """
    Evaluates the unbounded metalog PDF from per-sample term arrays.

    Args:
        a (list[mi.Float]): Metalog coefficients one entry per term (i.e., len = T, where each entry has shape [N]).
        y (mi.Float): CDF values in shape [n_samples].

    Returns:
        mi.Float: The unbounded metalog PDF from per-sample term arrays.
    """
    T = len(a)

    # clip y into the valid range
    ys = dr.clip(y, EPSILON, 1 - EPSILON)

    # precompute terms
    one_min_ys = 1 - ys
    y_term = ys * one_min_ys
    log_coeff = dr.log(ys * dr.rcp(one_min_ys))
    sub_coeff = ys - 0.5

    # Base terms
    y_term_rcp = dr.rcp(y_term)
    dx_dy = dr.fma(
        a[2], dr.fma(sub_coeff, y_term_rcp, log_coeff), a[1] * y_term_rcp
    )

    # 4th term
    if T >= 4:
        dx_dy += a[3]

    """
    NOTE: Setting n to `n = mi.UInt(5)` leads to:
    RuntimeError: jit_var_call(): an indirect call to instance 2 accesses an evaluated variable r150 of type float32 and
    size 48. However, only *scalar* (size == 1) evaluated variables can be accessed while recording indirect function
    calls.
    """
    n = 5
    # Add terms starting from the 5th term
    while n < T + 1:
        n_min_1 = n - 1
        an = a[n_min_1]
        if n % 2 == 0:  # even terms
            n_int_div_2 = n // 2
            dx_dy = dr.fma(
                an,
                (
                    dr.fma(
                        dr.fma(n, 0.5, -1),
                        dr.power(sub_coeff, n_int_div_2 - 2) * log_coeff,
                        dr.power(sub_coeff, n_int_div_2 - 1) * y_term_rcp,
                    )
                ),
                dx_dy,
            )
        else:  # odd terms
            dx_dy = dr.fma(
                an * n_min_1 * 0.5, dr.power(sub_coeff, (n - 3) // 2), dx_dy
            )

        n += 1

    return dr.rcp(dr.maximum(dx_dy, EPSILON))


@dr.syntax
def eval_bounded_metalog_pdf(
    a: list[mi.Float],
    b_l: float,
    b_u: float,
    y: mi.Float,
    M_y: mi.Float,
) -> mi.Float:
    """
    Evaluates the bounded metalog PDF from per-sample term arrays.

    Args:
        a (list[mi.Float]): Metalog coefficients one entry per term (i.e., len = T, where each entry has shape [N]).
        b_l (float): Lower bound.
        b_u (float): Upper bound.
        y (mi.Float): CDF values in shape [n_samples].
        M_y (mi.Float): Evaluated unbounded metalog quantile function for y, shape [n_samples].

    Returns:
        mi.Float: Bounded PDF value(s) from the given metalog coefficients and CDF value.
    """

    m_y = eval_unbounded_metalog_pdf(a, y)

    # bounded metalog (or, logit metalog) pdf for 0 < y < 1
    eM_y = dr.exp(M_y)
    result = m_y * dr.square(1 + eM_y) * dr.rcp((b_u - b_l) * eM_y)

    # handle boundary conditions
    result = dr.select(y <= EPSILON, 0, result)
    result = dr.select(y >= 1.0 - EPSILON, 0, result)

    return result


@dr.syntax
def eval_unbounded_metalog_quantile(a: list[mi.Float], y: mi.Float) -> mi.Float:
    """
    Evaluates the unbounded metalog quantile function from per-sample term arrays.

    Args:
        a (list[mi.Float]): Metalog coefficients one entry per term (i.e., len = T, where
        each entry has shape [N]).
        y (mi.Float): CDF values in shape [n_samples].

    Returns:
        Y (mi.Float): Metalog quantiles from the given coefficients.
    """
    T = len(a)

    # clip y into the valid range
    ys = dr.clip(y, EPSILON, 1 - EPSILON)

    # precompute terms
    log_coeff = dr.log(ys * dr.rcp(1 - ys))
    sub_coeff = ys - 0.5

    # base terms
    M_y = dr.fma(a[2], sub_coeff * log_coeff, dr.fma(a[1], log_coeff, a[0]))

    # 4th term
    if T >= 4:
        a3 = a[3]
        M_y = dr.fma(a3, sub_coeff, M_y)

    """
    NOTE: Setting n to `n = mi.UInt(5)` leads to:
    RuntimeError: jit_var_call(): an indirect call to instance 2 accesses an evaluated variable r150 of type float32 and
    size 48. However, only *scalar* (size == 1) evaluated variables can be accessed while recording indirect function
    calls.
    """
    n = 5
    # add terms starting from the 5th term
    while n < T + 1:
        n_min_1 = n - 1
        a_n = a[n_min_1]
        if n % 2 == 0:  # even terms
            M_y = dr.fma(dr.power(sub_coeff, n // 2 - 1) * log_coeff, a_n, M_y)
        else:  # odd terms
            M_y = dr.fma(a_n, dr.power(sub_coeff, n_min_1 // 2), M_y)

        n += 1

    return M_y


@dr.syntax
def eval_bounded_metalog_quantile(
    a: list[mi.Float],
    b_l: float,
    b_u: float,
    y: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates the unbounded metalog quantile function from per-sample term arrays.

    Args:
        a (list[mi.Float]): Metalog coefficients one entry per term (i.e., len = T, where each entry has shape [N]).
        b_l (float): Lower bound.
        b_u (float): Upper bound.
        y (mi.Float): CDF values in shape [n_samples].

    Returns:
        tuple:
        - [0]: Bounded metalog quantiles from the given coefficients.
        - [1]: Unbounded metalog quantiles from the given coefficients.
    """
    y = dr.clip(y, EPSILON, 1 - EPSILON)
    M_y = eval_unbounded_metalog_quantile(a, y)

    # bounded metalog (or, logit metalog) for 0 < y < 1
    eM_y = dr.exp(M_y)
    result = dr.fma(eM_y, b_u, b_l) * dr.rcp(1 + eM_y)

    # handle boundary conditions
    result = dr.select(y <= EPSILON, b_l, result)
    result = dr.select(y >= 1.0 - EPSILON, b_u, result)

    return result, M_y


@dr.syntax
def row_major_flat_idx(row_idx, col_idx, n_cols):
    """
    Compute the 1D offset of a 2D element in row-major (C-order) layout.
    The mapping follows C-style lexicographical access order with zero-based
    indexing:

        flat_index = row_idx * n_cols + col_idx
    """
    return dr.fma(row_idx, n_cols, col_idx)


def eval_interpolation_ratios(
    metalog_quantiles: mi.Float,
    x: mi.Float,
    lower_idx: mi.UInt,
    upper_idx: mi.UInt,
) -> mi.Float:
    """
    Calculates the interpolation ratio (t) from the given x and lower/upper metalog quantiles.

    Args:
        metalog_quantiles (mi.Float): Quantiles from a metalog distribution, shape: [R].
        x (mi.Float): Sample or quantile x from the marginal metalog distribution, shape: [N].
        lower_idx (mi.UInt): Indices to gather lower quantiles, shape: [N].
        upper_idx (mi.UInt): Indices to gather upper quantiles, shape: [N]

    Returns:
        mi.Float: Interpolation ratios in shape [N].
    """
    lower_quantiles = dr.gather(mi.Float, metalog_quantiles, lower_idx)
    upper_quantiles = dr.gather(mi.Float, metalog_quantiles, upper_idx)

    # calculate the interpolation ratios, shape: [N]
    denom = dr.maximum(upper_quantiles - lower_quantiles, EPSILON)
    t = (x - lower_quantiles) * dr.rcp(denom)
    t = dr.clip(t, 0, 1)
    return t


def eval_lower_upper_indices(
    metalog_resolution: int,
    y: mi.Float,
    marginal_quantile_cdf_range: tuple[float, float],
) -> tuple[mi.UInt, mi.UInt]:
    """
    Compute the lower and upper indices to gather from the conditional metalogs according to the given y.

    We assume that the values of `marginal_quantile_cdfs` are uniformly spaced, when the `metalog_quantiles` is
    initialized. That means, `marginal_quantile_cdfs` is initialized using the `classic` or `linspace` method, when
    `make_probs()` is evaluated. This simplifies the index calculation without the need of a binary search. See also:
    `init_metalogs()` and `make_probs()`.

    Args:
        metalog_res (int): Metalog resolution.
        y (mi.Float): CDF values in shape [n_samples].
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.

    Returns:
        tuple[mi.UInt, mi.UInt]: Lower and upper indices to gather from the conditional metalogs based on y, shape:
        [n_samples].
    """
    # for each sample x find where the index of the closest entry in
    # marginal_metalog_dist_inverse_cdf is
    y_min, y_max = marginal_quantile_cdf_range

    if metalog_resolution > 1:
        idx = mi.UInt(
            dr.floor(
                (y - y_min) * dr.rcp(y_max - y_min) * (metalog_resolution - 1)
            )
        )

        # clamp the indices to ensure they fall within the valid range, shape: [N]
        lower_idx = dr.clip(idx, 0, metalog_resolution - 2)
        upper_idx = lower_idx + 1
    else:
        zeros = dr.zeros(mi.UInt, dr.width(y))
        lower_idx, upper_idx = zeros, zeros

    return (lower_idx, upper_idx)


@dr.syntax
def sample_bounded_bivariate_metalog(
    marginal_metalog_coeffs: mi.TensorXf,
    marginal_metalog_dist_inverse_cdf: mi.Float | mi.TensorXf,
    cond_metalog_coeffs: mi.TensorXf,
    unif2: mi.Point2f,
    bounds_marginal: tuple[float, float],
    bounds_conditional: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
) -> SampleRecord:
    """
    Generates samples from the bivariate bounded metalog distribution.

    Note regarding the index calculation to gather from the conditional metalogs:
      We assume that the values of `marginal_quantile_cdfs` are uniformly spaced, when the `metalog_quantiles` is
      initialized. That means, `marginal_quantile_cdfs` is initialized using the `classic` or `linspace` method, when
      `make_probs()` is evaluated. This simplifies the index calculation without the need of a binary search. See also:
      `eval_lower_upper_indices()`, `compute_interpolated_coeffs()`, `init_metalogs()` and `make_probs()`.

    Note regarding the differences to the PyTorch implementation of `sample_bivariate_metalog()`:
      Unlike the `sample_bivariate_metalog()` implementation (PyTorch) in `metalogs.py`, this implementation only
      considers bounded marginal-conditional metalogs.

    Args:
        marginal_metalog_coeffs (mi.TensorXf): Coeffiecients of the marginal metalog distribution in a
        [1, metalog_terms] or [num_samples, metalog_terms] shaped tensor.
        marginal_metalog_dist_inverse_cdf (mi.Float): Quantiles from the marginal metalog distribution in a
        [metalog_resolution] or [num_samples, metalog_resolution] tensor.
        cond_metalog_coeffs (mi.TensorXf): Coeffiecients of the conditional metalog distribution in a
        [metalog_resolution, metalog_terms] or [num_samples, metalog_resolution, metalog_terms] tensor.
        unif2 (mi.Point2f): Uniform random numbers in range [0, 1] in a [num_samples, 2] tensor.
        bounds_marginal (float): Lower and upper bound for the marginal.
        bounds_conditional (float): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.

    Returns:
        SampleRecord: A sample record that contains the provided uniform variates (`unif2`), warped samples (`sample2`),
        and the bivariate pdf (`pdf = marginal_pdf * conditional_pdf`).
    """
    # sample x and evaluate the PDF for the marginal
    if marginal_metalog_coeffs.shape[0] == 1:
        # use the same coefficient row for every sample
        row_indices = dr.reinterpret_array(mi.UInt, dr.zeros_like(unif2.x))
    else:
        # a.shape[0] == n_samples, use one row per sample
        row_indices = dr.arange(mi.UInt, marginal_metalog_coeffs.shape[0])
    marginal_terms = gather_metalog_coeffs(marginal_metalog_coeffs, row_indices)
    x, unbounded_My = eval_bounded_metalog_quantile(
        marginal_terms,
        bounds_marginal[0],
        bounds_marginal[1],
        unif2.x,
    )
    pdf_marginal = eval_bounded_metalog_pdf(
        marginal_terms,
        bounds_marginal[0],
        bounds_marginal[1],
        unif2.x,
        unbounded_My,
    )

    y, pdf_cond = sample_bounded_conditional_metalogs(
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs,
        unif2,
        bounds_conditional,
        marginal_quantile_cdf_range,
        x,
    )

    sRec = SampleRecord(
        unif2=unif2,
        samples=mi.Point2f([x, y]),
        pdf=pdf_marginal * pdf_cond,
    )

    return sRec


def sample_bounded_conditional_metalogs(
    marginal_metalog_dist_inverse_cdf: mi.Float,
    cond_metalog_coeffs: mi.TensorXf,
    unif2: mi.Point2f,
    bounds_conditional: tuple,
    marginal_quantile_cdf_range: tuple[float, float],
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a bounded conditional metalog using interpolation within quantile rows.

    Args:
        marginal_metalog_dist_inverse_cdf (mi.Float): Quantiles from the marginal metalog distribution in a
        [metalog_resolution] or [num_samples, metalog_resolution] tensor.
        cond_metalog_coeffs (mi.TensorXf): Coeffiecients of the conditional metalog distribution in a
        [metalog_resolution, metalog_terms] or [num_samples, metalog_resolution, metalog_terms] tensor.
        unif2 (mi.Point2f): Uniform random numbers in range [0, 1] in a [num_samples, 2] tensor.
        bounds_conditional (float): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.
        x (mi.Float): Sample from the bounded marginal metalog.

    Returns:
        tuple[mi.Float, mi.Float]:
        - [0]: Sample from the conditional metalog given x.
        - [1]: Conditional PDF of the sample.
    """
    # gather lower and upper indices to get conditional metalogs correspond to x
    R = marginal_metalog_dist_inverse_cdf.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif2.x, marginal_quantile_cdf_range
    )

    # compute interpolation ratio
    t = eval_interpolation_ratios(
        marginal_metalog_dist_inverse_cdf, x, lower_idx, upper_idx
    )

    # interpolate conditional coefficients
    interp_cond_terms = gather_lerp_metalog_coeffs(
        cond_metalog_coeffs, lower_idx, upper_idx, t
    )

    # evaluate conditional metalog quantile/PDF functions
    y, unbounded_My = eval_bounded_metalog_quantile(
        interp_cond_terms,
        bounds_conditional[0],
        bounds_conditional[1],
        unif2.y,
    )
    pdf_cond = eval_bounded_metalog_pdf(
        interp_cond_terms,
        bounds_conditional[0],
        bounds_conditional[1],
        unif2.y,
        unbounded_My,
    )

    return y, pdf_cond


@dr.syntax
def sample_bounded_mixture_metalog(
    mixing_coeffs: mi.TensorXf,
    metalog_coeffs: mi.TensorXf,
    unif: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a bounded metalog mixture.

    Args:
        mixing_coeffs (mi.TensorXf): Mixing coefficients, shape: [K].
        metalog_coeffs (mi.TensorXf): Per-component metalog coefficients, shape: [K, T].
        unif (mi.Float): Uniform variates in [0, 1], shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the metalog mixture model, shape: [N].
          - [1]: Mixture PDF of the generated samples, shape: [N].
    """
    N = dr.width(unif)
    K, T = metalog_coeffs.shape
    mixing_coeffs = mixing_coeffs.array
    metalog_coeffs = metalog_coeffs.array

    """
    Pass 1: Vectorized component selection. Build mixing, metalog coeffs and
    select according to the first component whose cumulative covers the uniform
    variate.
    """

    # build a cumulative array [pi_0, pi_0 + pi_1, ..., 1.0]
    mix_cdf = dr.cumsum(mixing_coeffs)

    """
    Binary search to find the component index for every sample. Index is found
    such that mix_cdf[idx-1] <= unif < mix_cdf[idx].
    """
    idx = dr.binary_search(
        0, K - 1, lambda i: dr.gather(mi.Float, mix_cdf, i) < unif
    )

    # gather mixing coeffs for the selected component idx
    pi_k = dr.gather(mi.Float, mixing_coeffs, idx)

    # get the cumulative probability up to the previous component
    cum_pi_at_idx = dr.gather(mi.Float, mix_cdf, idx)
    prev_cum_pi = cum_pi_at_idx - pi_k

    # gather metalog coeffs for the selected component
    base_offset = idx * T
    a_terms_selected = [
        dr.gather(mi.Float, metalog_coeffs, base_offset + mi.UInt(ti))
        for ti in range(T)
    ]

    """
    Compute local uniform and sample x. Renormalize unif to [0, 1] within the
    selected components range.
    """
    denom = dr.maximum(pi_k, EPSILON)
    u_local = dr.clip((unif - prev_cum_pi) / denom, EPSILON, 1.0 - EPSILON)
    x, _ = eval_bounded_metalog_quantile(
        a_terms_selected, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # gather mixing and metalog coeffs
        pi = dr.gather(mi.Float, mixing_coeffs, kU)
        a_k = [
            dr.gather(mi.Float, metalog_coeffs, kU * T + mi.UInt(ti))
            for ti in range(T)
        ]

        """
        If this is the component we just sampled from (k == idx), then we know
        the exact CDF value (u_local). For others we use Newton-Raphson.
        """
        is_sampled_comp = idx == kU

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        u_k = dr.select(is_sampled_comp, u_local, u_k)
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi * pdf_k

    return x, mix_pdf


def sample_bounded_bivariate_metalog_slice_lerp(
    marginal_metalogs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    marginal_bounds: tuple[float, float],
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    cross_slice_interp_ratio: mi.Float,
) -> SampleRecord:
    """
    Sample from a bounded bivariate metalog using interpolation across two slices and within-slice quantile rows.

    Args:
        marginal_metalogs (mi.TensorXf): Per-slice marginal metalog terms, shape: [S, T].
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        marginal_bounds (tuple[float, float]): Lower and upper bounds for the marginal.
        conditional_bounds (tuple[float, float]): Lower and upper bounds for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx0 (mi.UInt): First slice index used for interpolation.
        slice_idx1 (mi.UInt): Second slice index used for interpolation.
        cross_slice_interp_ratio (mi.Float): Cross-slice interpolation ratio.

    Returns:
        SampleRecord: A sample record that contains the provided uniform variates (`unif2`), warped
        samples (`sample2`), and the bivariate pdf (`pdf = marginal_pdf * conditional_pdf`).
    """

    # interpolate marginal coefficients across slices
    interp_marg_terms = gather_lerp_metalog_coeffs(
        marginal_metalogs, slice_idx0, slice_idx1, cross_slice_interp_ratio
    )

    # sample from the marginal
    x, Mx_unbounded = eval_bounded_metalog_quantile(
        interp_marg_terms,
        marginal_bounds[0],
        marginal_bounds[1],
        sample2.x,
    )
    pdf_marginal = eval_bounded_metalog_pdf(
        interp_marg_terms,
        marginal_bounds[0],
        marginal_bounds[1],
        sample2.x,
        Mx_unbounded,
    )

    # sample conditional by lerping slices and within slice quantile rows
    y, pdf_cond = sample_bounded_conditional_metalog_slice_lerp(
        marginal_quantiles,
        conditional_metalogs,
        sample2,
        conditional_bounds,
        marginal_quantile_cdf_range,
        slice_idx0,
        slice_idx1,
        cross_slice_interp_ratio,
        x,
    )

    return SampleRecord(
        unif2=sample2,
        samples=mi.Point2f([x, y]),
        pdf=pdf_marginal * pdf_cond,
    )


def sample_bounded_conditional_metalog_slice_lerp(
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    cross_slice_interp_ratio: mi.Float,
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a bounded conditional metalog using interpolation across two slices and within-slice quantile rows.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        conditional_bounds (tuple[float, float]): Lower and upper bounds for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx0 (mi.UInt): First slice index used for interpolation.
        slice_idx1 (mi.UInt): Second slice index used for interpolation.
        cross_slice_interp_ratio (mi.Float): Cross-slice interpolation ratio.
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across two slices.

    Returns:
        tuple[mi.Float, mi.Float]:
        - [0]: Sample from the conditional metalog given x.
        - [1]: Conditional PDF of the sample.
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, sample2.x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t0 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx0, lower_idx, upper_idx
    )
    t1 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx1, lower_idx, upper_idx
    )

    # gather interpolated conditional terms across two slices
    interp_cond_terms = gather_slice_lerp_metalog_coeffs(
        conditional_metalogs,
        slice_idx0,
        slice_idx1,
        lower_idx,
        upper_idx,
        t0,
        t1,
        cross_slice_interp_ratio,
    )

    # sample from the conditional
    y, unbounded_My = eval_bounded_metalog_quantile(
        interp_cond_terms,
        conditional_bounds[0],
        conditional_bounds[1],
        sample2.y,
    )
    pdf_cond = eval_bounded_metalog_pdf(
        interp_cond_terms,
        conditional_bounds[0],
        conditional_bounds[1],
        sample2.y,
        unbounded_My,
    )

    return y, pdf_cond


def sample_bounded_bivariate_metalog_slice_bilerp(
    marginal_metalogs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    marginal_bounds: tuple[float, float],
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
) -> SampleRecord:
    """
    Sample from a bounded bivariate metalog using interpolation across four slices and within-slice quantile rows.

    Args:
        marginal_metalogs (mi.TensorXf): Per-slice marginal metalog terms, shape: [S, T].
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        marginal_bounds (tuple[float, float]): Lower and upper bounds for the marginal.
        conditional_bounds (tuple[float, float]): Lower and upper bounds for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].

    Returns:
        SampleRecord: A sample record that contains the provided uniform variates (`unif2`), warped
        samples (`sample2`), and the bivariate pdf (`pdf = marginal_pdf * conditional_pdf`).
    """

    # bilinearly interpolate marginal coefficients across slices
    interp_marg_terms = gather_bilerp_marginal_metalog_coeffs(
        marginal_metalogs,
        slice_idx00,
        slice_idx01,
        slice_idx10,
        slice_idx11,
        t_theta,
        t_phi,
    )

    # sample from the marginal
    x, Mx_unbounded = eval_bounded_metalog_quantile(
        interp_marg_terms,
        marginal_bounds[0],
        marginal_bounds[1],
        sample2.x,
    )
    pdf_marginal = eval_bounded_metalog_pdf(
        interp_marg_terms,
        marginal_bounds[0],
        marginal_bounds[1],
        sample2.x,
        Mx_unbounded,
    )

    # sample conditional by lerping slices and within slice quantile rows
    y, pdf_cond = sample_bounded_conditional_metalog_slice_bilerp(
        marginal_quantiles,
        conditional_metalogs,
        sample2,
        conditional_bounds,
        marginal_quantile_cdf_range,
        slice_idx00,
        slice_idx01,
        slice_idx10,
        slice_idx11,
        t_theta,
        t_phi,
        x,
    )

    return SampleRecord(
        unif2=sample2,
        samples=mi.Point2f([x, y]),
        pdf=pdf_marginal * pdf_cond,
    )


def sample_bounded_conditional_metalog_slice_bilerp(
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a bounded conditional metalog using interpolation across four slices and within-slice quantile rows.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        conditional_bounds (tuple[float, float]): Lower and upper bounds for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across four slices.

    Returns:
        tuple[mi.Float, mi.Float]:
        - [0]: Sample from the conditional metalog given x.
        - [1]: Conditional PDF of the sample.
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, sample2.x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t00 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx00, lower_idx, upper_idx
    )
    t01 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx01, lower_idx, upper_idx
    )
    t10 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx10, lower_idx, upper_idx
    )
    t11 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx11, lower_idx, upper_idx
    )

    # gather interpolated conditional terms across four slices
    interp_cond_terms = gather_slice_bilerp_metalog_coeffs(
        conditional_metalogs,
        slice_idx00,
        slice_idx01,
        slice_idx10,
        slice_idx11,
        lower_idx,
        upper_idx,
        t00,
        t01,
        t10,
        t11,
        t_theta,
        t_phi,
    )

    # sample from the conditional
    y, unbounded_My = eval_bounded_metalog_quantile(
        interp_cond_terms,
        conditional_bounds[0],
        conditional_bounds[1],
        sample2.y,
    )
    pdf_cond = eval_bounded_metalog_pdf(
        interp_cond_terms,
        conditional_bounds[0],
        conditional_bounds[1],
        sample2.y,
        unbounded_My,
    )

    return y, pdf_cond


@dr.syntax
def cdist(A: mi.Point2f, B: mi.Point2f) -> mi.TensorXf:
    """
    Computes euclidean distance between A and B. Similar to `torch.cdist()`.

    Args:
        A (mi.Point2f): 2D points of shape N.
        B (mi.Point2f): 2D points of shape M.

    Returns:
        mi.TensorXf: Euclidean distance between A and B in a [N, M] tensor.
    """
    N = A.shape[1]
    M = B.shape[1]

    # cast into TensorXf so we can do 2-D slicing; A -> [N, 1], B -> [1, M]
    Ax = mi.TensorXf(A.x, shape=(N, 1))
    Ay = mi.TensorXf(A.y, shape=(N, 1))
    Bx = mi.TensorXf(B.x, shape=(1, M))
    By = mi.TensorXf(B.y, shape=(1, M))

    # elementwise difference, shape [N, M]
    dx = (Ax - Bx).array
    dy = (Ay - By).array

    return mi.TensorXf(dr.norm(mi.Point2f(dx, dy)), shape=(N, M))


@dr.syntax
def argmin_axis1(dists: mi.TensorXf) -> mi.UInt:
    """
    Argmin along the axis 1. Similar to `torch.argmin(dists, dim=1)`.

    Args:
        dists (mi.TensorXf): [N, M] shaped tensor.

    Returns:
        mi.UInt: For each of the N rows, the index of the minimum value (UInt[N]).
    """
    N, M = dists.shape

    # flatten distances for direct memory access
    dists = dists.array

    # calculate starting memory index for each row - base pointer
    row_starts = dr.arange(mi.UInt, N) * M

    # initialize best distances and indices
    best_dist = dr.gather(mi.Float, dists, row_starts + 0)
    best_idx = dr.zeros(mi.UInt, shape=N)

    # initialize pointer to the first (index 1) of each row
    current_pointers = row_starts + 1

    # loop through rest of the columns
    j = mi.UInt(1)
    while j < M:
        # distances and indices in column j
        col = dr.gather(mi.Float, dists, current_pointers)
        j_idx = dr.zeros(mi.UInt, N) + j

        # the new value is smaller than the current best
        mask = col < best_dist

        # the new value becomes the new best
        best_idx = dr.select(mask, j_idx, best_idx)
        best_dist = dr.select(mask, col, best_dist)

        j += 1

    return best_idx


@dr.syntax
def gather_rows(tensor: mi.TensorXf, idx: mi.UInt) -> mi.TensorXf:
    """
    Gathers the rows along the first axis of tensor.

    Args:
        tensor (mi.TensorXf): A tensor of shape [n_slices, R, T] or [n_slices, T] or [n_slices, R].
        idx (mi.UInt): Indices to gather the tensor.

    Returns:
        mi.TensorXf: A tensor of shape [sample_size, ...].
    """
    N = idx.shape[0]

    # flatten the tensor to a 1D Float array of length n_slices * prod(shape[1:])
    flat = tensor.array

    # compute the row‐stride, e.g., R*T, T, or R
    stride = dr.prod(tensor.shape[1:])

    # build flat indices using row and column offset
    row_off = mi.TensorXu(idx * stride, shape=(N, 1))
    col_off = mi.TensorXu(dr.arange(mi.UInt, stride), shape=(1, stride))

    # broadcast to [sample_size, stride], then flatten to [sample_size*stride]
    flat_idx = (row_off + col_off).array

    # gather all values into a Float[sample_size * stride] array
    gathered = dr.gather(mi.Float, flat, flat_idx)

    # reshape back into a TensorXf of shape [N, *shape[1:]]
    return mi.TensorXf(gathered, shape=(N, *tensor.shape[1:]))


@dr.syntax
def argtop2_axis1(dists: mi.TensorXf) -> tuple[mi.UInt, mi.UInt]:
    N, M = dists.shape

    # flatten distances for direct memory access
    dists = dists.array

    # calculate starting memory index for each row - base pointer
    row_starts = dr.arange(mi.UInt, N) * M

    # initialize first and second column
    col0 = dr.gather(mi.Float, dists, row_starts + 0)
    col1 = dr.gather(mi.Float, dists, row_starts + 1)

    # create masks
    mask = col0 < col1
    zeros = dr.zeros(mi.UInt, N)
    ones = dr.ones(mi.UInt, N)

    # initialize best and second best distances and indices
    best_dist = dr.select(mask, col0, col1)
    best_idx = dr.select(mask, zeros, ones)
    second_best_dist = dr.select(mask, col1, col0)
    second_best_idx = dr.select(mask, ones, zeros)

    # initialize pointer to the 3rd element (index 2) of each row
    current_pointers = row_starts + 2

    # loop through rest of the columns
    j = mi.UInt(2)
    while j < M:
        # distances and indices in column j
        col = dr.gather(mi.Float, dists, current_pointers)
        j_idx = dr.zeros(mi.UInt, N) + j

        # case 1: the new value is smaller than the current best
        mask1 = col < best_dist

        # the old best becomes the new second best
        second_best_dist = dr.select(mask1, best_dist, second_best_dist)
        second_best_idx = dr.select(mask1, best_idx, second_best_idx)

        # the new value becomes the new best
        best_dist = dr.select(mask1, col, best_dist)
        best_idx = dr.select(mask1, j_idx, best_idx)

        # case 2: the new value is not smaller than the best, but is smaller
        # than the second best
        mask2 = (~mask1) & (col < second_best_dist)
        second_best_dist = dr.select(mask2, col, second_best_dist)
        second_best_idx = dr.select(mask2, j_idx, second_best_idx)

        current_pointers += 1
        j += 1

    return best_idx, second_best_idx


def argmin(arr):
    min_array = dr.min(arr)
    return dr.compress(arr == min_array)[0]


def phi_theta_cell_corners(
    wi2D: mi.Point2f, metalogs_keys: mi.Point2f, n_phi: int, n_theta: int
) -> tuple[
    mi.Float,
    mi.Float,
    mi.Float,
    mi.Float,
    mi.UInt,
    mi.UInt,
    mi.UInt,
    mi.UInt,
]:
    """
    Locate the enclosing (phi, theta) grid cell for each sample and return its four corner coordinates together with
    flat (row-major) slice indices. The `metalogs_keys` assumed to be laid out with phi-major order:

        flat_idx = phi_idx * n_theta + theta_idx

    Steps:
        1. Build the 1D phi and theta grids from `metalog_keys`.
        2. For each `wi2D` (phi*, theta*) find bracketing indices (phi0<=phi*<=phi1, theta0<=theta*<=theta1) using
        binary search.
        3. Return (phi0, phi1, theta0, theta1) and the four flat indices idx00=(phi0, theta0), idx10=(phi1, theta0),
        idx01=(phi0, theta1), idx11=(phi1, theta1).

    Maps 2D (phi, theta) indices to flat row indices.

    Args:
        wi2D (mi.Point2f): Incident direction in spherical coordinates, shape: [sample_size, 2].
        metalogs_keys (mi.Point2f): Measured angles in spherical coordinates (i.e., [phi_i, theta_i], shape: [n_slices, 2].
        n_phi (int): Number of unique phi_i values.
        n_theta (int): Number of unique theta_i values.

    Returns:
        - [mi.Float, mi.Float, mi.Float, mi.Float]: Gathered four corner coordinates: phi0, phi1, theta0, theta1.
        - [mi.UInt, mi.UInt, mi.UInt, mi.UInt]: Flat row-major indices of the four corners: idx00, idx10, idx01, idx11.
    """

    # get 1D coordinate grids from the flattened keys
    phi_idx = dr.arange(mi.UInt, n_phi) * n_theta  # 0, T, 2T, ...
    theta_idx = dr.arange(mi.UInt, n_theta)  # 0 ... T-1
    phi_grid = dr.gather(mi.Float, metalogs_keys.x, phi_idx)
    theta_grid = dr.gather(mi.Float, metalogs_keys.y, theta_idx)

    # for wi2D find the lower and upper bounds indices for phi and theta
    pred_phi = lambda index: dr.gather(mi.Float, phi_grid, index) <= wi2D.x
    i_phi0 = dr.binary_search(0, n_phi - 1, pred_phi) - 1
    pred_theta = lambda index: dr.gather(mi.Float, theta_grid, index) <= wi2D.y
    i_theta0 = dr.binary_search(0, n_theta - 1, pred_theta) - 1

    # clamp indices to valid intervals and define upper bound indices
    i_phi0 = dr.clip(i_phi0, 0, n_phi - 2)
    i_theta0 = dr.clip(i_theta0, 0, n_theta - 2)
    i_phi1 = i_phi0 + 1
    i_theta1 = i_theta0 + 1

    # gather the corner coordinates
    phi0 = dr.gather(mi.Float, phi_grid, i_phi0)
    phi1 = dr.gather(mi.Float, phi_grid, i_phi1)
    theta0 = dr.gather(mi.Float, theta_grid, i_theta0)
    theta1 = dr.gather(mi.Float, theta_grid, i_theta1)

    # map 2D (phi, theta) indices to flat row indices
    idx00 = mi.UInt(dr.fma(i_phi0, n_theta, i_theta0))
    idx10 = mi.UInt(dr.fma(i_phi1, n_theta, i_theta0))
    idx01 = mi.UInt(dr.fma(i_phi0, n_theta, i_theta1))
    idx11 = mi.UInt(dr.fma(i_phi1, n_theta, i_theta1))

    return phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11


@dr.syntax
def argtop2_nearest_keys(
    wi2D: mi.Point2f, keys: mi.Point2f
) -> tuple[mi.UInt, mi.UInt]:
    """
    Finds the row indices of the smallest and second-smallest distance to the point in `keys`.

    Args:
        keys (mi.Point2f): Key positions (e.g., spherical coords [phi, theta]), shape: [n_slices, 2].
        wi2D (mi.Point2f): Per-sample query positions, shape [n_samples].

    Returns:
        tuple[mi.UInt, mi.UInt]: The row indices of the smallest and second-smallest distance to the point in `keys`.
    """
    # N := number of samples, M := number of key pairs
    N, M = wi2D.shape[1], keys.shape[1]

    # distance squared to key 0
    dx0 = wi2D.x - keys.x[0]
    dy0 = wi2D.y - keys.y[0]
    d0 = dr.sqr(dx0) + dr.sqr(dy0)

    # distance squared to key 1
    dx1 = wi2D.x - keys.x[1]
    dy1 = wi2D.y - keys.y[1]
    d1 = dr.sqr(dx1) + dr.sqr(dy1)

    # initialize best and second-best distances and indices
    N_zeros = dr.zeros(mi.UInt, N)
    N_ones = dr.ones(mi.UInt, N)
    mask = d0 < d1
    best_dist = dr.select(mask, d0, d1)
    best_idx = dr.select(mask, N_zeros, N_ones)
    second_best_dist = dr.select(mask, d1, d0)
    second_best_idx = dr.select(mask, N_ones, N_zeros)

    # loop through rest of the keys
    """
    NOTE: Setting j to `j = mi.UInt(2)` leads to:
    RuntimeError: jit_var_call(): an indirect call to instance 2 accesses an evaluated variable r172 of type float32 and
    size 8. However, only *scalar* (size == 1) evaluated variables can be accessed while recording indirect function
    calls.
    """
    j = 2
    while j < M:
        # new value: distance squared to key j
        dx = wi2D.x - keys.x[j]
        dy = wi2D.y - keys.y[j]
        dj = dr.sqr(dx) + dr.sqr(dy)

        j_idx = dr.zeros(mi.UInt, N) + j

        # case 1: new value becomes the new best
        mask1 = dj < best_dist
        second_best_dist = dr.select(mask1, best_dist, second_best_dist)
        second_best_idx = dr.select(mask1, best_idx, second_best_idx)
        best_dist = dr.select(mask1, dj, best_dist)
        best_idx = dr.select(mask1, j_idx, best_idx)

        # case 2: new value is between best and second best
        mask2 = (~mask1) & (dj < second_best_dist)
        second_best_dist = dr.select(mask2, dj, second_best_dist)
        second_best_idx = dr.select(mask2, j_idx, second_best_idx)

        j += 1

    return best_idx, second_best_idx


@dr.syntax
def sph_to_dir(theta: mi.Float, phi: mi.Float) -> mi.Vector3f:
    """Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)


@dr.syntax
def cos_theta_phi_to_unitvec(cos_theta_phi: mi.Point2f) -> mi.Vector3f:
    """
    Convert samples in (cos_theta, azimuth) coordinates to unit vectors (3D).

    Args:
        samples2D (mi.Point2f): Samples in (cos_theta, phi) coordinates.

    Returns:
        mi.Vector: A tensor of shape (n_samples, 3) after conversion.
    """
    cos_theta = dr.clip(cos_theta_phi.x, 0.0, 1.0)
    phi = cos_theta_phi.y

    # Compute the radial component in the xy-plane
    r = dr.safe_sqrt(1 - cos_theta**2)
    s, c = dr.sincos(phi)

    return mi.Vector3f(r * c, r * s, cos_theta)


def unitvec_to_cos_theta_phi(w: mi.Vector3f) -> mi.Point2f:
    """
    Convert unit vectors (3D) to samples in (cos_theta, azimuth).

    Args:
        samples3D (mi.Vector3f): A tensor of shape (n_samples, 3) containing 3D samples.

    Returns:
        mi.Point2f: A tensor of shape (n_samples, 2) after conversion.
    """
    phi = dr.atan2(w.y, w.x)
    cos_theta = dr.clamp(w.z, 0, 1)
    return mi.Point2f(cos_theta, phi)


@dr.syntax
def spherical_to_unitvec(
    samples2D: mi.Point2f,
) -> mi.Vector3f:
    """
    Convert samples in spherical coordinates (theta, phi) to unit vectors (3D).

    Args:
        samples2D (mi.Point2f): An array of shape (n_samples, 2) containing 2D samples.

    Returns:
        mi.Vector3f: A tensor of shape (n_samples, 3) after conversion.
    """
    return sph_to_dir(theta=samples2D.x, phi=samples2D.y)


@dr.syntax
def unitvec_to_spherical(
    samples3D: mi.Vector3f,
) -> mi.Vector2f:
    """
    Convert unit vectors (3D) to samples in spherical coordinates (theta, phi).

    Args:
        samples3D (mi.Vector3f): A tensor of shape (n_samples, 3) containing 3D samples.

    Returns:
        mi.Vector2f: A tensor of shape (n_samples, 2) after conversion.
    """
    phi = dr.atan2(samples3D.y, samples3D.x)
    theta = dr.safe_acos(samples3D.z)
    return mi.Vector2f(theta, phi)


@dr.syntax
def eval_newton_raphson_for_metalog_cdf(
    a: list[mi.Float],
    x: mi.Float,
    y: mi.Float,
    bounds: tuple[float, float],
    max_iter: int = 30,
) -> mi.Float:
    """
    Evaluates the bounded metalog CDF value 'y' for a given 'x' using a combination of bisection and Newton-Raphson. See
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
        a (list[mi.Float]): Metalog coefficients one entry per term (i.e., len = T, where each entry has shape [N]).
        x (mi.Float): Values at which to evaluate the CDF, shape: [N].
        y (mi.Float): Starting values for the CDF, shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds of the metalog distribution.
        max_iter (int, optional): Maximum number of iterations for the Newton-Raphson method. Defaults to 30.

    Returns:
        mi.Float: Estimated CDF values 'y' corresponding to input 'x', shape: [N].
    """
    # yl and yh are the lower and upper bounds for the probability 'y'
    yl = dr.zeros_like(x) + EPSILON
    yh = dr.zeros_like(x) + (1.0 - EPSILON)

    """
    NOTE: dx = dx_old leads to:
    RuntimeError: dr.while_loop(): jit_var_loop_end(): loop state variable 4
    (r2906) was presumed to be constant, but it changed (to r2782) when
    re-recording the loop a second time.
    """
    dx_old = yh - yl
    dx = yh - yl
    converged = dr.zeros_like(x) > 0

    """
    NOTE: Megakernel mode does not allow the following symbolic while loop:
        while (i < max_iter) & (~converged):
            # loop body as below
    The error is:
        RuntimeError: jit_var_call(): an indirect call to instance 2 accesses an
        evaluated variable r150 of type float32 and size 4. However, only
        *scalar* (size == 1) evaluated variables can be accessed while recording
        indirect function calls.
    As such, we use while loop in the 'scalar' mode. See also:
    https://drjit.readthedocs.io/en/latest/reference.html#drjit.while_loop.
    """
    i = mi.UInt(0)
    while dr.hint(i < max_iter, mode="scalar", max_iterations=max_iter):
        # active lanes are the ones that are yet not converged
        active = ~converged

        # evaluate the function f(y) = M(y) - x
        M_y, unbounded_My = eval_bounded_metalog_quantile(
            a, bounds[0], bounds[1], y
        )
        f_y = M_y - x

        # evaluate f'(y) = M'(y) = 1 / m(y), where m(y) is the metalog PDF
        m_y = eval_bounded_metalog_pdf(
            a,
            bounds[0],
            bounds[1],
            y,
            M_y=unbounded_My,
        )
        f_prime = dr.rcp(m_y + EPSILON)

        # new y using Newton-Raphson - recall m(y) = 1 / M'(y)
        y_nr = dr.fma(-f_y, m_y, y)

        """
        Check if the Newton-Raphson step is acceptable. Use Newton-Raphson if
        the conditions met and the lane is `active`. See also `rtsafe()`, Ch.
        9.4 Newton-Raphson Method Using Derivative.
        - Condition 1 (inside): Check if the Newton step remains within the
        bracket.
        - Condition 2 (shrink): Check if bracket is shrinking fast enough.
        """
        inside = (y_nr > yl) & (y_nr < yh)
        shrink = dr.abs(2.0 * f_y) <= dr.abs(dx_old * f_prime)
        use_nr = inside & shrink & active

        # new y using Bisection (fallback)
        y_bi = 0.5 * (yl + yh)

        # assign new y using Newton-Raphson if possible, else use bisection
        y_new = dr.select(use_nr, y_nr, y_bi)

        # ensure new y is in [EPSILON, 1 - EPSILON]
        y_new = dr.clip(y_new, EPSILON, 1.0 - EPSILON)

        # calculate the step dx based on the method
        dx_newton = y_new - y
        dx_bi = 0.5 * (yh - yl)
        dx_new = dr.select(use_nr, dx_newton, dx_bi)

        # update steps
        dx_old = dr.select(active, dx, dx_old)
        dx = dr.select(active, dx_new, dx)
        y = dr.select(active, y_new, y)

        # update the bracket [yl, yh] using the new position of y
        M_y_new, _ = eval_bounded_metalog_quantile(a, bounds[0], bounds[1], y)
        f_new = M_y_new - x
        f_neg = f_new < 0.0
        yl = dr.select(active & f_neg, y, yl)
        yh = dr.select(active & ~f_neg, y, yh)

        # update the convergence mask
        converged |= dr.abs(dx) < 1e-4
        i += 1

    return y


@dr.syntax
def eval_bivariate_metalog_cdf_pdf(
    marginal_metalog_coeffs: mi.TensorXf,
    marginal_metalog_dist_inverse_cdf: mi.Float | mi.TensorXf,
    cond_metalog_coeffs: mi.TensorXf,
    sample2: mi.Point2f,
    marginal_quantile_cdf_range: tuple[float, float],
    bounds_marginal: tuple[float, float],
    bounds_conditional: tuple[float, float],
) -> tuple[mi.Point2f, mi.Float]:
    """
    Evaluates bivariate bounded metalog CDF and PDF.

    Note regarding the `compute_interpolated_coeffs()` step:
      We assume that the values of `marginal_quantile_cdfs` are uniformly spaced, when the `metalog_quantiles` is
      initialized. That means, `marginal_quantile_cdfs` is initialized using the `classic` or `linspace` method, when
      `make_probs()` is evaluated. This simplifies the index calculation without the need of a binary search. See also:
      `compute_interpolated_coeffs()`, `init_metalogs()` and `make_probs()`.

    Note regarding the CDF method:
      We always use the Newton-Raphson method.

    Args:
        marginal_metalog_coeffs (mi.TensorXf): Coeffiecients of the marginal metalog distribution in a
        [1, metalog_terms] or [num_samples, metalog_terms] shaped tensor.
        marginal_metalog_dist_inverse_cdf (mi.Float): Quantiles from the marginal metalog distribution in a
        [metalog_resolution] or [num_samples, metalog_resolution] tensor.
        cond_metalog_coeffs (mi.TensorXf): Coeffiecients of the conditional metalog distribution in a
        [metalog_resolution, metalog_terms] or [num_samples, metalog_resolution, metalog_terms] tensor.
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.
        bounds_marginal (float): Lower and upper bound for the marginal.
        bounds_conditional (float): Lower and upper bound for the conditional.

    Returns:
        tuple:
        - [0]: Marginal-conditional CDF (i.e., uniform variates) of the samples.
        - [1]: Computed bivariate metalog PDF in a (num_samples) tensor.
    """

    x = sample2.x
    if marginal_metalog_coeffs.shape[0] == 1:
        # use the same coefficient row for every sample
        row_indices = dr.reinterpret_array(mi.UInt, dr.zeros_like(x))
    else:
        # a.shape[0] == n_samples, use one row per sample
        row_indices = dr.arange(mi.UInt, marginal_metalog_coeffs.shape[0])
    marginal_terms = gather_metalog_coeffs(marginal_metalog_coeffs, row_indices)

    init_unif_x = dr.zeros_like(x) + 0.5
    unif_x = eval_newton_raphson_for_metalog_cdf(
        marginal_terms,
        x,
        init_unif_x,
        bounds_marginal,
        max_iter=HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS,
    )

    # evaluate marginal metalog PDF for x
    unbounded_My = eval_unbounded_metalog_quantile(marginal_terms, unif_x)
    pdf_marginal = eval_bounded_metalog_pdf(
        marginal_terms,
        bounds_marginal[0],
        bounds_marginal[1],
        unif_x,
        M_y=unbounded_My,
    )

    # gather lower and upper indices to get conditional metalogs correspond to x
    unif_y, pdf_cond = eval_bounded_conditional_metalog_cdf_pdf(
        marginal_metalog_dist_inverse_cdf,
        cond_metalog_coeffs,
        sample2,
        marginal_quantile_cdf_range,
        bounds_conditional,
        unif_x,
    )

    # compute bivariate pdf
    pdf_joint = pdf_marginal * pdf_cond

    return mi.Point2f(unif_x, unif_y), pdf_joint


def eval_bounded_conditional_metalog_cdf_pdf(
    marginal_metalog_dist_inverse_cdf: mi.Float | mi.TensorXf,
    cond_metalog_coeffs: mi.TensorXf,
    sample2: mi.Point2f,
    marginal_quantile_cdf_range: tuple[float, float],
    bounds_conditional: tuple[float, float],
    unif_x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog CDF and PDF using within-slice quantile rows.

    Args:
        marginal_metalog_dist_inverse_cdf (mi.Float): Quantiles from the marginal metalog distribution in a
        [metalog_resolution] or [num_samples, metalog_resolution] tensor.
        cond_metalog_coeffs (mi.TensorXf): Coeffiecients of the conditional metalog distribution in a
        [metalog_resolution, metalog_terms] or [num_samples, metalog_resolution, metalog_terms] tensor.
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        marginal_quantile_cdf_range (tuple[float, float]): Min and max value of the marginal quantile CDF.
        bounds_conditional (float): Lower and upper bound for the conditional.
        unif_x (mi.Float): Per-sample marginal metalog CDFs.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs within quantile interpolated conditional metalogs.
    """
    R = marginal_metalog_dist_inverse_cdf.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif_x, marginal_quantile_cdf_range
    )

    # compute interpolation ratio
    t = eval_interpolation_ratios(
        marginal_metalog_dist_inverse_cdf, sample2.x, lower_idx, upper_idx
    )

    # interpolate conditional coefficients
    interp_cond_terms = gather_lerp_metalog_coeffs(
        cond_metalog_coeffs, lower_idx, upper_idx, t
    )

    # evaluate interpolated conditional metalog CDF
    init_unif_y = dr.zeros_like(sample2.y) + 0.5
    unif_y = eval_newton_raphson_for_metalog_cdf(
        interp_cond_terms,
        sample2.y,
        init_unif_y,
        bounds_conditional,
        max_iter=HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS,
    )

    # evaluate interpolated conditional metalog PDF for y
    unbounded_My = eval_unbounded_metalog_quantile(interp_cond_terms, unif_y)
    pdf_cond = eval_bounded_metalog_pdf(
        interp_cond_terms,
        bounds_conditional[0],
        bounds_conditional[1],
        unif_y,
        unbounded_My,
    )

    return unif_y, pdf_cond


@dr.syntax
def eval_bounded_bivariate_metalog_cdf_pdf_slice_lerp(
    marginal_metalogs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    marginal_bounds: tuple[float, float],
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    cross_slice_interp_ratio: mi.Float,
) -> tuple[mi.Point2f, mi.Float]:
    """
    Evaluates bounded metalog CDF and PDF using interpolation across two slices and within-slice quantile rows.

    Note regarding the CDF method:
      We always use the Newton-Raphson method.

    Args:
        marginal_metalogs (mi.TensorXf): Per-slice marginal metalog terms, shape: [S, T].
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        marginal_bounds (tuple[float, float]): Lower and upper bound for the marginal.
        conditional_bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx0 (mi.UInt): First slice index used for interpolation.
        slice_idx1 (mi.UInt): Second slice index used for interpolation.
        cross_slice_interp_ratio (mi.Float): Cross-slice interpolation ratio.

    Returns:
        tuple[mi.Point2f, mi.Float]:
            - [0]: Marginal-conditional CDF (i.e., uniform variates), shape: [N, 2].
            - [1]: Bivariate metalog PDF, shape: [N].
    """
    # interpolate marginal coefficients across slices
    interp_marg_terms = gather_lerp_metalog_coeffs(
        marginal_metalogs, slice_idx0, slice_idx1, cross_slice_interp_ratio
    )

    # compute marginal CDF
    init_unif_x = dr.zeros_like(sample2.x) + 0.5
    unif_x = eval_newton_raphson_for_metalog_cdf(
        interp_marg_terms,
        sample2.x,
        init_unif_x,
        marginal_bounds,
        max_iter=HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS,
    )

    # evaluate marginal metalog PDF for x
    unbounded_My = eval_unbounded_metalog_quantile(interp_marg_terms, unif_x)
    pdf_marginal = eval_bounded_metalog_pdf(
        interp_marg_terms,
        marginal_bounds[0],
        marginal_bounds[1],
        unif_x,
        M_y=unbounded_My,
    )

    # evaluate conditional metalog PDF for y given x
    unif_y, pdf_cond = eval_bounded_conditional_metalog_cdf_pdf_slice_lerp(
        marginal_quantiles,
        conditional_metalogs,
        sample2,
        conditional_bounds,
        marginal_quantile_cdf_range,
        slice_idx0,
        slice_idx1,
        cross_slice_interp_ratio,
        unif_x,
    )

    # compute joint pdf
    pdf_joint = pdf_marginal * pdf_cond

    return mi.Point2f(unif_x, unif_y), pdf_joint


def eval_bounded_conditional_metalog_cdf_pdf_slice_lerp(
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    cross_slice_interp_ratio: mi.Float,
    unif_x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog CDF and PDF using interpolation across two slices and within-slice quantile
    rows.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        conditional_bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx0 (mi.UInt): First slice index used for interpolation.
        slice_idx1 (mi.UInt): Second slice index used for interpolation.
        cross_slice_interp_ratio (mi.Float): Cross-slice interpolation ratio.
        unif_x (mi.Float): Per-sample marginal metalog CDFs, where metalogs interpolated across slices.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of across slice and within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of across slice and within quantile interpolated conditional metalogs.
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif_x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t0 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx0, lower_idx, upper_idx
    )
    t1 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx1, lower_idx, upper_idx
    )

    # gather interpolated conditional terms across two slices
    interp_cond_terms = gather_slice_lerp_metalog_coeffs(
        conditional_metalogs,
        slice_idx0,
        slice_idx1,
        lower_idx,
        upper_idx,
        t0,
        t1,
        cross_slice_interp_ratio,
    )

    # evaluate lower and upper conditional metalog CDFs
    init_unif_y = dr.zeros_like(sample2.y) + 0.5
    unif_y = eval_newton_raphson_for_metalog_cdf(
        interp_cond_terms,
        sample2.y,
        init_unif_y,
        conditional_bounds,
        max_iter=HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS,
    )

    # evaluate conditional metalog PDF for y
    unbounded_My = eval_unbounded_metalog_quantile(interp_cond_terms, unif_y)
    pdf_cond = eval_bounded_metalog_pdf(
        interp_cond_terms,
        conditional_bounds[0],
        conditional_bounds[1],
        unif_y,
        unbounded_My,
    )

    return unif_y, pdf_cond


@dr.syntax
def eval_bounded_bivariate_metalog_cdf_pdf_slice_bilerp(
    marginal_metalogs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    marginal_bounds: tuple[float, float],
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
) -> tuple[mi.Point2f, mi.Float]:
    """
    Evaluates bounded metalog CDF and PDF using interpolation across four slices and within-slice quantile rows.

    Note regarding the CDF method:
      We always use the Newton-Raphson method.

    Args:
        marginal_metalogs (mi.TensorXf): Per-slice marginal metalog terms, shape: [S, T].
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        marginal_bounds (tuple[float, float]): Lower and upper bound for the marginal.
        conditional_bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].

    Returns:
        tuple[mi.Point2f, mi.Float]:
            - [0]: Marginal-conditional CDF (i.e., uniform variates), shape: [N, 2].
            - [1]: Bivariate metalog PDF, shape: [N].
    """
    # bilinearly interpolate marginal coefficients across slices
    interp_marg_terms = gather_bilerp_marginal_metalog_coeffs(
        marginal_metalogs,
        slice_idx00,
        slice_idx01,
        slice_idx10,
        slice_idx11,
        t_theta,
        t_phi,
    )

    # compute marginal CDF
    init_unif_x = dr.zeros_like(sample2.x) + 0.5
    unif_x = eval_newton_raphson_for_metalog_cdf(
        interp_marg_terms,
        sample2.x,
        init_unif_x,
        marginal_bounds,
        max_iter=HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS,
    )

    # evaluate marginal metalog PDF for x
    unbounded_My = eval_unbounded_metalog_quantile(interp_marg_terms, unif_x)
    pdf_marginal = eval_bounded_metalog_pdf(
        interp_marg_terms,
        marginal_bounds[0],
        marginal_bounds[1],
        unif_x,
        M_y=unbounded_My,
    )

    # evaluate conditional metalog PDF for y
    unif_y, pdf_cond = eval_bounded_conditional_metalog_cdf_pdf_slice_bilerp(
        marginal_quantiles,
        conditional_metalogs,
        sample2,
        conditional_bounds,
        marginal_quantile_cdf_range,
        slice_idx00,
        slice_idx01,
        slice_idx10,
        slice_idx11,
        t_theta,
        t_phi,
        unif_x,
    )

    # compute joint pdf
    pdf_joint = pdf_marginal * pdf_cond

    return mi.Point2f(unif_x, unif_y), pdf_joint


def eval_bounded_conditional_metalog_cdf_pdf_slice_bilerp(
    marginal_quantiles: mi.TensorXf,
    conditional_metalogs: mi.TensorXf,
    sample2: mi.Point2f,
    conditional_bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
    unif_x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog CDF and PDF using interpolation across four slices and within-slice quantile
    rows.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        conditional_metalogs (mi.TensorXf): Per-slice, per-row conditional metalog terms, shape: [S, R, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        conditional_bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        unif_x (mi.Float): Per-sample marginal metalog CDFs, where metalogs interpolated across four slices.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of across slice and within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of across slice and within quantile interpolated conditional metalogs.
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif_x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t00 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx00, lower_idx, upper_idx
    )
    t01 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx01, lower_idx, upper_idx
    )
    t10 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx10, lower_idx, upper_idx
    )
    t11 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx11, lower_idx, upper_idx
    )

    # gather interpolated conditional terms across four slices
    interp_cond_terms = gather_slice_bilerp_metalog_coeffs(
        conditional_metalogs,
        slice_idx00,
        slice_idx01,
        slice_idx10,
        slice_idx11,
        lower_idx,
        upper_idx,
        t00,
        t01,
        t10,
        t11,
        t_theta,
        t_phi,
    )

    # evaluate lower and upper conditional metalog CDFs
    init_unif_y = dr.zeros_like(sample2.y) + 0.5
    unif_y = eval_newton_raphson_for_metalog_cdf(
        interp_cond_terms,
        sample2.y,
        init_unif_y,
        conditional_bounds,
        max_iter=HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS,
    )

    # evaluate conditional metalog PDF for y
    unbounded_My = eval_unbounded_metalog_quantile(interp_cond_terms, unif_y)
    pdf_cond = eval_bounded_metalog_pdf(
        interp_cond_terms,
        conditional_bounds[0],
        conditional_bounds[1],
        unif_y,
        unbounded_My,
    )

    return unif_y, pdf_cond


def solid_angle_pdf(rectangular_pdf: mi.Float, cos_theta: mi.Float) -> mi.Float:
    """
    Converts rectangular (theta,phi) density to per-steradian density p_w, where w refers to omega.

    Mitsuba PDF is per unit solid angle p_w(w)dw. We reparameterize w by spherical angles (theta,phi):

        w(theta,phi)=(sin(theta)cos(phi),sin(theta)sin(phi),cos(theta))

    On the unit sphere the surface/solid angle element is dw = sin(theta) dthetadphi, that is, sin(theta) is the surface
    Jacobian. As such, the PDFs become p_(theta,phi) = p_w sin(theta), and p_w = p_(theta,phi) / sin(theta).

    Args:
        rectangular_pdf (mi.Float): Tensor of p_(theta,phi) evaluated at samples.
        cos_theta (mi.Float): Tensor of mu=cos(theta) for the samples; values in [-1, 1], or [0, 1] for a hemisphere.

    Returns:
        mi.Float: Rectangular p_(theta,phi) density to solid angle density: p_w = p_(theta,phi) / sin(theta).
    """
    sin_theta = dr.sqrt(dr.maximum(1.0 - cos_theta * cos_theta, 0.0))
    denom = dr.maximum(sin_theta, 1e-12)
    p_omega = rectangular_pdf * dr.rcp(denom)
    return p_omega


def eval_projection_weights(
    keys: mi.Point2f,
    wi2D: mi.Point2f,
    idx0: mi.UInt,
    idx1: mi.UInt,
) -> mi.Float:
    """
    Computes linear interpolation weight `t` by projecting the query direction onto the vector segment connecting the
    two nearest keys.

    We treat the parameter space between the two nearest keys (`k0` and `k1`) as a linear segment, then project the
    query point `wi2d` onto this segment to find the relative position `t`. Mathematically:

        v = k1 - k0
        w = wi - k0
        t = dot(w, v) / dot(v, v)

    This result in a scalar `t` where 0.0 corresponds to `k0` and 1.0 to `k1`.

    Args:
        keys (mi.Point2f): Key positions (e.g., spherical coords [phi, theta]), shape: [n_slices, 2].
        wi2D (mi.Point2f): Per-sample query positions, shape [n_samples, 2].
        idx0 (mi.UInt): Indices of the nearest key for each sample, shape [n_samples].
        idx1 (mi.UInt): Indices of the second-nearest key for each sample, shape [n_samples].

    Returns:
        mi.Float: Per-sample interpolation ratio `t` for the second key (`idx1`). The weight for the first key (`idx0`)
        is implicitly `1-t`, shape: [n_samples].
    """
    # gather lower and upper keys
    kx0 = dr.gather(mi.Float, keys.x, idx0)
    ky0 = dr.gather(mi.Float, keys.y, idx0)
    kx1 = dr.gather(mi.Float, keys.x, idx1)
    ky1 = dr.gather(mi.Float, keys.y, idx1)

    # compute the vector from closest to 2nd closest (v = upper - lower)
    v = mi.Vector2f(kx1 - kx0, ky1 - ky0)

    # compute the vector from closest to query (w = wi - lower)
    w = mi.Vector2f(wi2D.x - kx0, wi2D.y - ky0)

    # project w onto v: t = dot(w, v) / dot(v, v)
    t = dr.dot(w, v) * dr.rcp(dr.dot(v, v))

    return dr.clamp(t, 0.0, 1.0)


def eval_interpolation_ratio_per_slice(
    marginal_quantiles: mi.TensorXf,
    x: mi.Float,
    slice_idx: mi.UInt,
    lower_idx: mi.UInt,
    upper_idx: mi.UInt,
) -> mi.Float:
    """
    Computes the per-sample linear interpolation weight within a slice's marginal quantiles.

    Given a per-slice table of marginal quantiles and row indices `lower_idx`/`upper_idx` for each sample, compute the
    normalized position of `x` between those two quantiles for the selected `slice_idx`.

    Args:
        marginal_quantiles (mi.TensorXf): Quantile gird from the marginal metalog per slice, shape: [S, R].
        x (mi.Float): Marginal sample in the bounded domain, shape: [N].
        slice_idx (mi.UInt): Per-sample slice indices selecting which row grid to use, shape: [N].
        lower_idx (mi.UInt): Per-sample lower row indices in the quantile grid, shape: [N].
        upper_idx (mi.UInt): Per-sample upper row indices in the quantile grid, shape: [N].

    Returns:
        mi.Float: Interpolation ratio t per sample (shape [N]), representing the relative location of `x` between lower
        and upper quantiles for the chosen slice.
    """
    S, R = marginal_quantiles.shape
    base_idx = slice_idx * R
    marginal_quantiles = marginal_quantiles.array

    # gather lower and upper quantiles for the slice indices
    q_lo = dr.gather(mi.Float, marginal_quantiles, base_idx + lower_idx)
    q_hi = dr.gather(mi.Float, marginal_quantiles, base_idx + upper_idx)

    # evaluate interpolation ratio
    denom = q_hi - q_lo
    t = dr.clip((x - q_lo) * dr.rcp(denom + EPSILON), 0.0, 1.0)
    return t


@dr.syntax
def sample_bounded_mixture_metalog_slice_lerp(
    mixing_coeffs: mi.TensorXf,
    metalog_coeffs: mi.TensorXf,
    unif: mi.Float,
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    t: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a bounded metalog mixture using interpolation across two slices.

    Args:
        mixing_coeffs (mi.TensorXf): Per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs (mi.TensorXf): Per-slice, per-component metalog coefficients, shape: [S, K, T].
        unif (mi.Float): Uniform variates in [0, 1], shape: [N].
        slice_idx0 (mi.UInt): First slice index used for interpolation, shape: [N].
        slice_idx1 (mi.UInt): Second slice index used for interpolation, shape: [N].
        t (mi.Float): Cross-slice interpolation ratio, shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the metalog mixture model, shape: [N].
          - [1]: Mixture PDF of the generated samples, shape: [N].
    """
    N = dr.width(unif)
    S, K, T = metalog_coeffs.shape
    mixing_coeffs = mixing_coeffs.array
    metalog_coeffs = metalog_coeffs.array

    # compute base indices for components
    base_idx0 = slice_idx0 * K
    base_idx1 = slice_idx1 * K

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component  whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # lerp mixing coeffs for component k between two slices
        pi0 = dr.gather(mi.Float, mixing_coeffs, base_idx0 + kU)
        pi1 = dr.gather(mi.Float, mixing_coeffs, base_idx1 + kU)
        pi_lerp_k = dr.lerp(pi0, pi1, t)
        pi_list.append(pi_lerp_k)

        # lerp per-component metalog terms between two slices
        r0 = (base_idx0 + kU) * T
        r1 = (base_idx1 + kU) * T
        a_k = [
            dr.lerp(
                dr.gather(mi.Float, metalog_coeffs, r0 + mi.UInt(ti)),
                dr.gather(mi.Float, metalog_coeffs, r1 + mi.UInt(ti)),
                t,
            )
            for ti in range(T)
        ]
        a_terms.append(a_k)

        # accumulate mixing coeffs
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_lerp_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_lerp_k, sel_pi_k)

        # pick the metalog terms according to `take`
        a_k = a_terms[k]
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    x, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather lerped mixing and metalog coeffs
        pi_lerp_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_lerp_k * pdf_k

    return x, mix_pdf


@dr.syntax
def sample_bounded_mixture_metalog_slice_lerp_blend(
    mixing_coeffs_A: mi.TensorXf,
    metalog_coeffs_A: mi.TensorXf,
    mixing_coeffs_B: mi.TensorXf,
    metalog_coeffs_B: mi.TensorXf,
    unif: mi.Float,
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    lambda_val: mi.Float,
    t: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from blended two bounded metalog mixture using interpolation across two slices.

    Args:
        mixing_coeffs_A (mi.TensorXf): Material A, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per-component metalog coefficients, shape: [S, K, T].
        mixing_coeffs_B (mi.TensorXf): Material B, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per-component metalog coefficients, shape: [S, K, T].
        unif (mi.Float): Uniform variates in [0, 1], shape: [N].
        slice_idx0 (mi.UInt): First slice index used for interpolation, shape: [N].
        slice_idx1 (mi.UInt): Second slice index used for interpolation, shape: [N].
        lambda_val (mi.Float): Blending interpolation ratio between material A and B.
        t (mi.Float): Cross-slice interpolation ratio, shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the metalog mixture model, shape: [N].
          - [1]: Mixture PDF of the generated samples, shape: [N].
    """
    N = dr.width(unif)
    S, K, T = metalog_coeffs_A.shape
    mixing_coeffs_A = mixing_coeffs_A.array
    metalog_coeffs_A = metalog_coeffs_A.array
    mixing_coeffs_B = mixing_coeffs_B.array
    metalog_coeffs_B = metalog_coeffs_B.array

    # compute base indices for components
    base_idx0 = slice_idx0 * K
    base_idx1 = slice_idx1 * K

    def lerp_mixing_weights(mixing_coeffs, kU):
        # lerp mixing coeffs for component k between two slices
        pi0 = dr.gather(mi.Float, mixing_coeffs, base_idx0 + kU)
        pi1 = dr.gather(mi.Float, mixing_coeffs, base_idx1 + kU)
        return dr.lerp(pi0, pi1, t)

    def lerp_metalog_coeffs(metalog_coeffs, r0, r1, tiU):
        c0 = dr.gather(mi.Float, metalog_coeffs, r0 + tiU)
        c1 = dr.gather(mi.Float, metalog_coeffs, r1 + tiU)
        return dr.lerp(c0, c1, t)

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component  whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing coeffs between material A and B
        pi_lerp_k_A = lerp_mixing_weights(mixing_coeffs_A, kU)
        pi_lerp_k_B = lerp_mixing_weights(mixing_coeffs_B, kU)
        pi_lerp_k = dr.lerp(pi_lerp_k_A, pi_lerp_k_B, lambda_val)
        pi_list.append(pi_lerp_k)

        r0 = (base_idx0 + kU) * T
        r1 = (base_idx1 + kU) * T

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in range(T):
            tiU = mi.UInt(ti)
            term_A = lerp_metalog_coeffs(metalog_coeffs_A, r0, r1, tiU)
            term_B = lerp_metalog_coeffs(metalog_coeffs_B, r0, r1, tiU)
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        a_terms.append(a_k)

        # accumulate mixing coeffs
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_lerp_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_lerp_k, sel_pi_k)

        # pick the metalog terms according to `take`
        a_k = a_terms[k]
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    x, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather lerped mixing and metalog coeffs
        pi_lerp_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_lerp_k * pdf_k

    return x, mix_pdf


@dr.syntax
def sample_bounded_mixture_metalog_slice_bilerp(
    mixing_coeffs: mi.TensorXf,
    metalog_coeffs: mi.TensorXf,
    unif: mi.Float,
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a bounded metalog mixture using interpolation across four slices.

    Args:
        mixing_coeffs (mi.TensorXf): Per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs (mi.TensorXf): Per-slice, per-component metalog coefficients, shape: [S, K, T].
        unif (mi.Float): Uniform variates in [0, 1], shape: [N].
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the metalog mixture model, shape: [N].
          - [1]: Mixture PDF of the generated samples, shape: [N].
    """
    N = dr.width(unif)
    S, K, T = metalog_coeffs.shape
    mixing_coeffs = mixing_coeffs.array
    metalog_coeffs = metalog_coeffs.array

    # compute base indices per-corner
    base_idx00 = slice_idx00 * K
    base_idx01 = slice_idx01 * K
    base_idx10 = slice_idx10 * K
    base_idx11 = slice_idx11 * K

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # bilerp mixing coeffs for component k between four slices
        pi00 = dr.gather(mi.Float, mixing_coeffs, base_idx00 + kU)
        pi01 = dr.gather(mi.Float, mixing_coeffs, base_idx01 + kU)
        pi10 = dr.gather(mi.Float, mixing_coeffs, base_idx10 + kU)
        pi11 = dr.gather(mi.Float, mixing_coeffs, base_idx11 + kU)
        pi_k = bilerp(pi00, pi01, pi10, pi11)
        pi_list.append(pi_k)

        # bilerp per-component metalog terms between four slices
        r00 = (base_idx00 + kU) * T
        r01 = (base_idx01 + kU) * T
        r10 = (base_idx10 + kU) * T
        r11 = (base_idx11 + kU) * T

        a_k = [
            bilerp(
                dr.gather(mi.Float, metalog_coeffs, r00 + mi.UInt(ti)),
                dr.gather(mi.Float, metalog_coeffs, r01 + mi.UInt(ti)),
                dr.gather(mi.Float, metalog_coeffs, r10 + mi.UInt(ti)),
                dr.gather(mi.Float, metalog_coeffs, r11 + mi.UInt(ti)),
            )
            for ti in range(T)
        ]
        a_terms.append(a_k)

        # cumulative for picking the component
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_k, sel_pi_k)

        # pick the metalog terms according to `take`
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen components bilerped metalogs
    x, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather bilerped mixing and metalog coeffs
        pi_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_k * pdf_k

    return x, mix_pdf


@dr.syntax
def sample_bounded_mixture_metalog_slice_bilerp_blend(
    mixing_coeffs_A: mi.TensorXf,
    metalog_coeffs_A: mi.TensorXf,
    mixing_coeffs_B: mi.TensorXf,
    metalog_coeffs_B: mi.TensorXf,
    unif: mi.Float,
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    lambda_val: mi.Float,
    t_theta: mi.Float,
    t_phi: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from blended two bounded metalog mixtures using interpolation across four slices.

    Args:
        mixing_coeffs_A (mi.TensorXf): Material A, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per-component metalog coefficients, shape: [S, K, T].
        mixing_coeffs_B (mi.TensorXf): Material B, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per-component metalog coefficients, shape: [S, K, T].
        unif (mi.Float): Uniform variates in [0, 1], shape: [N].
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        lambda_val (mi.Float): Blending interpolation ratio between material A and B.
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the metalog mixture model, shape: [N].
          - [1]: Mixture PDF of the generated samples, shape: [N].
    """
    N = dr.width(unif)
    S, K, T = metalog_coeffs_A.shape
    mixing_coeffs_A = mixing_coeffs_A.array
    metalog_coeffs_A = metalog_coeffs_A.array
    mixing_coeffs_B = mixing_coeffs_B.array
    metalog_coeffs_B = metalog_coeffs_B.array

    # compute base indices per-corner
    base_idx00 = slice_idx00 * K
    base_idx01 = slice_idx01 * K
    base_idx10 = slice_idx10 * K
    base_idx11 = slice_idx11 * K

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    def bilerp_mixing_weights(mixing_coeffs, kU):
        # bilerp mixing coeffs for component k between four slices
        pi00 = dr.gather(mi.Float, mixing_coeffs, base_idx00 + kU)
        pi01 = dr.gather(mi.Float, mixing_coeffs, base_idx01 + kU)
        pi10 = dr.gather(mi.Float, mixing_coeffs, base_idx10 + kU)
        pi11 = dr.gather(mi.Float, mixing_coeffs, base_idx11 + kU)
        return bilerp(pi00, pi01, pi10, pi11)

    def bilerp_metalog_coeffs(metalog_coeffs, r00, r01, r10, r11, tiU):
        # bilerp per-component metalog terms between four slices
        c00 = dr.gather(mi.Float, metalog_coeffs, r00 + tiU)
        c01 = dr.gather(mi.Float, metalog_coeffs, r01 + tiU)
        c10 = dr.gather(mi.Float, metalog_coeffs, r10 + tiU)
        c11 = dr.gather(mi.Float, metalog_coeffs, r11 + tiU)
        return bilerp(c00, c01, c10, c11)

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing coeffs between material A and B
        pi_k_A = bilerp_mixing_weights(mixing_coeffs_A, kU)
        pi_k_B = bilerp_mixing_weights(mixing_coeffs_B, kU)
        pi_k = dr.lerp(pi_k_A, pi_k_B, lambda_val)
        pi_list.append(pi_k)

        r00 = (base_idx00 + kU) * T
        r01 = (base_idx01 + kU) * T
        r10 = (base_idx10 + kU) * T
        r11 = (base_idx11 + kU) * T

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in range(T):
            tiU = mi.UInt(ti)
            term_A = bilerp_metalog_coeffs(
                metalog_coeffs_A, r00, r01, r10, r11, tiU
            )
            term_B = bilerp_metalog_coeffs(
                metalog_coeffs_B, r00, r01, r10, r11, tiU
            )
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        a_terms.append(a_k)

        # cumulative for picking the component
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_k, sel_pi_k)

        # pick the metalog terms according to `take`
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen components blended bilerped metalogs
    x, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather blended bilerped mixing and metalog coeffs
        pi_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_k * pdf_k

    return x, mix_pdf


@dr.syntax
def sample_bounded_conditional_mixture_metalog(
    cond_mix_coeffs: mi.TensorXf,
    cond_mix_metalog_coeffs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    marginal_quantile_cdf_range: tuple[float, float],
    sample2: mi.Point2f,
    bounds: tuple[mi.Float, mi.Float],
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a conditional bounded metalog mixture using interpolation within quantile rows.

    Args:
        cond_mix_coeffs (mi.TensorXf): Per conditional mixing coefficients, shape: [R, K].
        cond_mix_metalog_coeffs (mi.TensorXf): Per conditional, per-component conditional metalog coefficients,
        shape: [R, K, T].
        marginal_quantiles (mi.TensorXf): Marginal quantiles, shape: [R].
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        bounds (tuple[mi.Float, mi.Float]): Lower and upper bounds.
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across slices.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the conditional metalog mixture model, shape: [N].
          - [1]: Conditional Mixture PDF of the generated samples, shape: [N].
    """
    unif = sample2.y
    N = dr.width(unif)
    R, K, T = cond_mix_metalog_coeffs.shape

    cond_mix_coeffs = cond_mix_coeffs.array
    cond_mix_metalog_coeffs = cond_mix_metalog_coeffs.array
    marginal_quantiles = marginal_quantiles.array

    """Compute row indices and interpolation ratio."""
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, sample2.x, marginal_quantile_cdf_range
    )

    # compute interpolation ratio
    t = eval_interpolation_ratios(marginal_quantiles, x, lower_idx, upper_idx)

    """Pass 1: Component Selection"""
    base_lo = lower_idx * K
    base_hi = upper_idx * K
    cum_pi = dr.zeros(mi.Float, N)
    selected_mask = dr.full(mi.Bool, False, N)

    sel_idx = dr.zeros(mi.UInt, N)
    sel_pi = dr.zeros(mi.Float, N)
    sel_prev_cum = dr.zeros(mi.Float, N)
    cached_pi = []
    for k in range(K):
        kU = mi.UInt(k)

        # interpolate mixing coeffs
        w_lo = dr.gather(mi.Float, cond_mix_coeffs, base_lo + kU)
        w_hi = dr.gather(mi.Float, cond_mix_coeffs, base_hi + kU)
        pi_k = dr.lerp(w_lo, w_hi, t)
        cached_pi.append(pi_k)

        # update cumulative mixing coeffs
        prev_cum = cum_pi
        cum_pi = cum_pi + pi_k

        # select if this is the first component crossing the threshold
        take = (~selected_mask) & (unif <= cum_pi)

        sel_idx = dr.select(take, kU, sel_idx)
        sel_pi = dr.select(take, pi_k, sel_pi)
        sel_prev_cum = dr.select(take, prev_cum, sel_prev_cum)

        selected_mask = selected_mask | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected_mask
    sel_idx = dr.select(take_last, K - 1, sel_idx)
    sel_pi = dr.maximum(dr.select(take_last, cached_pi[-1], sel_pi), EPSILON)

    """Pass 2: Gather metalog coeffs for the selected component and sample"""
    sel_a_terms = []

    # base offsets for the selected component
    sel_base_lo_T = (base_lo + sel_idx) * T
    sel_base_hi_T = (base_hi + sel_idx) * T
    for ti in range(T):
        tiU = mi.UInt(ti)
        a_lo = dr.gather(mi.Float, cond_mix_metalog_coeffs, sel_base_lo_T + tiU)
        a_hi = dr.gather(mi.Float, cond_mix_metalog_coeffs, sel_base_hi_T + tiU)
        a_k_t = dr.lerp(a_lo, a_hi, t)
        sel_a_terms.append(a_k_t)

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum) * dr.rcp(sel_pi), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    y, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 3: Compute exact mixture PDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)
        pi_k = cached_pi[k]

        # interpolate metalog coeffs for component k
        curr_base_lo_T = (base_lo + kU) * T
        curr_base_hi_T = (base_hi + kU) * T
        a_k = []
        for ti in range(T):
            tiU = mi.UInt(ti)
            a_lo = dr.gather(
                mi.Float, cond_mix_metalog_coeffs, curr_base_lo_T + tiU
            )
            a_hi = dr.gather(
                mi.Float, cond_mix_metalog_coeffs, curr_base_hi_T + tiU
            )
            a_val = dr.lerp(a_lo, a_hi, t)
            a_k.append(a_val)

        """
        If this is the component we just sampled from (k == idx), then we know
        the exact CDF value (u_local). For others we use Newton-Raphson.
        """
        is_sampled = sel_idx == kU
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, y, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        u_k = dr.select(is_sampled, u_local, u_k)

        # evaluate the metalog PDF for the component
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_k * pdf_k

    return y, mix_pdf


@dr.syntax
def sample_bounded_conditional_mixture_metalog_slice_lerp(
    cond_mix_coeffs: mi.TensorXf,
    cond_mix_metalog_coeffs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    marginal_quantile_cdf_range: tuple[float, float],
    sample2: mi.Point2f,
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    t: mi.Float,
    bounds: tuple[mi.Float, mi.Float],
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a conditional bounded metalog mixture using interpolation across two slices and within-slice quantile
    rows.

    Args:
        cond_mix_coeffs (mi.TensorXf): Per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs (mi.TensorXf): Per-slice, per conditional, per-component conditional metalog
        coefficients, shape: [S, R,  K, T].
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        slice_idx0 (mi.UInt): First slice index used for interpolation, shape: [N].
        slice_idx1 (mi.UInt): Second slice index used for interpolation, shape: [N].
        t (mi.Float): Cross-slice interpolation ratio, shape: [N].
        bounds (tuple[mi.Float, mi.Float]): Lower and upper bounds.
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across slices.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the conditional metalog mixture model, shape: [N].
          - [1]: Conditional Mixture PDF of the generated samples, shape: [N].
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, sample2.x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t0 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx0, lower_idx, upper_idx
    )
    t1 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx1, lower_idx, upper_idx
    )

    unif = sample2.y
    N = dr.width(unif)
    S, R, K, T = cond_mix_metalog_coeffs.shape
    cond_mix_coeffs = cond_mix_coeffs.array
    cond_mix_metalog_coeffs = cond_mix_metalog_coeffs.array

    """
    Base flat indices for (S,R,K,T) and (S,R,K) tensors:
    - Mixing Coeffs: idx = ((s * R + r) * K + k)
    - Metalog Coeffs: idx = (((s * R + r) * K + k) * T + tci)
    """
    base0_lo = (slice_idx0 * R + lower_idx) * K
    base0_hi = (slice_idx0 * R + upper_idx) * K
    base1_lo = (slice_idx1 * R + lower_idx) * K
    base1_hi = (slice_idx1 * R + upper_idx) * K

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # row-lerp mixing coeffs in each slice then slice-lerp between two slice
        w00 = dr.gather(mi.Float, cond_mix_coeffs, base0_lo + kU)
        w01 = dr.gather(mi.Float, cond_mix_coeffs, base0_hi + kU)
        pi0 = dr.lerp(w00, w01, t0)

        w10 = dr.gather(mi.Float, cond_mix_coeffs, base1_lo + kU)
        w11 = dr.gather(mi.Float, cond_mix_coeffs, base1_hi + kU)
        pi1 = dr.lerp(w10, w11, t1)

        pi_lerp_k = dr.lerp(pi0, pi1, t)
        pi_list.append(pi_lerp_k)

        # row-lerp metalog coeffs in each slice then slice-lerp
        a_k = []
        r00 = (base0_lo + kU) * T
        r01 = (base0_hi + kU) * T
        r10 = (base1_lo + kU) * T
        r11 = (base1_hi + kU) * T

        for t_idx in range(T):
            a00 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r00 + t_idx)
            a01 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r01 + t_idx)
            a0r = dr.lerp(a00, a01, t0)

            a10 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r10 + t_idx)
            a11 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r11 + t_idx)
            a1r = dr.lerp(a10, a11, t1)

            a_k_t = dr.lerp(a0r, a1r, t)
            a_k.append(a_k_t)
        a_terms.append(a_k)

        # accumulate mixing coeffs
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_lerp_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_lerp_k, sel_pi_k)

        # pick the metalog terms according to `take`
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    y, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather lerped mixing and metalog coeffs
        pi_lerp_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, y, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_lerp_k * pdf_k

    return y, mix_pdf


@dr.syntax
def sample_bounded_conditional_mixture_metalog_slice_lerp_blend(
    cond_mix_coeffs_A: mi.TensorXf,
    cond_mix_metalog_coeffs_A: mi.TensorXf,
    marginal_quantiles_A: mi.TensorXf,
    cond_mix_coeffs_B: mi.TensorXf,
    cond_mix_metalog_coeffs_B: mi.TensorXf,
    marginal_quantiles_B: mi.TensorXf,
    sample2: mi.Point2f,
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    lambda_val: mi.Float,
    t: mi.Float,
    bounds: tuple[mi.Float, mi.Float],
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a conditional bounded metalog mixture using interpolation across two slices and within-slice quantile
    rows.

    Args:
        cond_mix_coeffs_A (mi.TensorXf): Material A, per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R,  K, T].
        marginal_quantiles_A (mi.TensorXf): Material A, per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R,  K, T].
        marginal_quantiles_B (mi.TensorXf): Material B, per-slice marginal quantiles, shape: [S, R].
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        slice_idx0 (mi.UInt): First slice index used for interpolation, shape: [N].
        slice_idx1 (mi.UInt): Second slice index used for interpolation, shape: [N].
        lambda_val (mi.Float): Interpolation ratio between material A and B.
        t (mi.Float): Cross-slice interpolation ratio, shape: [N].
        bounds (tuple[mi.Float, mi.Float]): Lower and upper bounds.
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across slices.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the conditional metalog mixture model, shape: [N].
          - [1]: Conditional Mixture PDF of the generated samples, shape: [N].
    """
    unif = sample2.y
    N = dr.width(unif)
    S, R_A, K, T = cond_mix_metalog_coeffs_A.shape
    _, R_B, _, _ = cond_mix_metalog_coeffs_B.shape
    cond_mix_coeffs_A = cond_mix_coeffs_A.array
    cond_mix_metalog_coeffs_A = cond_mix_metalog_coeffs_A.array
    cond_mix_coeffs_B = cond_mix_coeffs_B.array
    cond_mix_metalog_coeffs_B = cond_mix_metalog_coeffs_B.array

    """Material A: Find unique bracketing indices and interpolation ratios"""
    lo0_A, hi0_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx0, x
    )
    lo1_A, hi1_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx1, x
    )
    t0_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, x, slice_idx0, lo0_A, hi0_A
    )
    t1_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, x, slice_idx1, lo1_A, hi1_A
    )
    base0_lo_A = (slice_idx0 * R_A + lo0_A) * K
    base0_hi_A = (slice_idx0 * R_A + hi0_A) * K
    base1_lo_A = (slice_idx1 * R_A + lo1_A) * K
    base1_hi_A = (slice_idx1 * R_A + hi1_A) * K

    """Material B: Find unique bracketing indices and interpolation ratios"""
    lo0_B, hi0_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx0, x
    )
    lo1_B, hi1_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx1, x
    )
    t0_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, x, slice_idx0, lo0_B, hi0_B
    )
    t1_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, x, slice_idx1, lo1_B, hi1_B
    )
    base0_lo_B = (slice_idx0 * R_B + lo0_B) * K
    base0_hi_B = (slice_idx0 * R_B + hi0_B) * K
    base1_lo_B = (slice_idx1 * R_B + lo1_B) * K
    base1_hi_B = (slice_idx1 * R_B + hi1_B) * K

    def lerp_cond_mixing_weights(
        cond_mix_coeffs,
        base0_lo,
        base0_hi,
        base1_lo,
        base1_hi,
        t0,
        t1,
        kU,
    ):
        # row-lerp mixing coeffs in each slice then slice-lerp between two slice
        w00 = dr.gather(mi.Float, cond_mix_coeffs, base0_lo + kU)
        w01 = dr.gather(mi.Float, cond_mix_coeffs, base0_hi + kU)
        pi0 = dr.lerp(w00, w01, t0)

        w10 = dr.gather(mi.Float, cond_mix_coeffs, base1_lo + kU)
        w11 = dr.gather(mi.Float, cond_mix_coeffs, base1_hi + kU)
        pi1 = dr.lerp(w10, w11, t1)

        return dr.lerp(pi0, pi1, t)

    def lerp_cond_metalog_coeffs(
        cond_mix_metalog_coeffs,
        base0_lo,
        base0_hi,
        base1_lo,
        base1_hi,
        t0,
        t1,
        kU,
        tiU,
    ):
        r00 = (base0_lo + kU) * T
        r01 = (base0_hi + kU) * T
        r10 = (base1_lo + kU) * T
        r11 = (base1_hi + kU) * T

        a00 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r00 + tiU)
        a01 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r01 + tiU)
        a0r = dr.lerp(a00, a01, t0)

        a10 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r10 + tiU)
        a11 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r11 + tiU)
        a1r = dr.lerp(a10, a11, t1)

        return dr.lerp(a0r, a1r, t)

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing weights between material A and B
        pi_k_A = lerp_cond_mixing_weights(
            cond_mix_coeffs_A,
            base0_lo_A,
            base0_hi_A,
            base1_lo_A,
            base1_hi_A,
            t0_A,
            t1_A,
            kU,
        )
        pi_k_B = lerp_cond_mixing_weights(
            cond_mix_coeffs_B,
            base0_lo_B,
            base0_hi_B,
            base1_lo_B,
            base1_hi_B,
            t0_B,
            t1_B,
            kU,
        )
        pi_lerp_k = dr.lerp(pi_k_A, pi_k_B, lambda_val)
        pi_list.append(pi_lerp_k)

        # blend metalog coeffs between material A and B
        a_k = []
        for t_idx in range(T):
            tiU = mi.UInt(t_idx)
            term_A = lerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_A,
                base0_lo_A,
                base0_hi_A,
                base1_lo_A,
                base1_hi_A,
                t0_A,
                t1_A,
                kU,
                tiU,
            )
            term_B = lerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_B,
                base0_lo_B,
                base0_hi_B,
                base1_lo_B,
                base1_hi_B,
                t0_B,
                t1_B,
                kU,
                tiU,
            )
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        a_terms.append(a_k)

        # accumulate mixing coeffs
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_lerp_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_lerp_k, sel_pi_k)

        # pick the metalog terms according to `take`
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    y, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather lerped mixing and metalog coeffs
        pi_lerp_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, y, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_lerp_k * pdf_k

    return y, mix_pdf


def find_interval_binary_search(
    marginal_quantiles: mi.TensorXf, slice_idx: mi.UInt, x: mi.Float
) -> tuple[mi.UInt, mi.UInt]:
    """
    Performs a vectorized binary search to find the interval enclosing `x` within specific slices of `marginal_quantiles`.

    Mathematically, it attempts to satisfy:
        marginal_quantiles[s, l] <= x < marginal_quantiles[s, u]
    where $s$ corresponds to `slice_idx`.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        slice_idx (mi.UInt): Per-sample corner slice indices for a cell (phi, theta).
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across four slices.

    Returns:
        tuple[mi.UInt, mi.UInt]: A tuple `(lower_idx, upper_idx)` containing the result indices. `lower_idx` is
        guaranteed to be in the range [0, R-2].
    """
    R = marginal_quantiles.shape[-1]

    # is the quantile value at this index <= x?
    pred = (
        lambda index: dr.gather(
            mi.Float, marginal_quantiles.array, slice_idx * R + index
        )
        <= x
    )

    # find the first index where pred is false
    idx = dr.binary_search(0, R - 1, pred)

    # clamp to valid range
    lower_idx = dr.clip(idx - 1, 0, R - 2)
    upper_idx = lower_idx + 1

    return lower_idx, upper_idx


@dr.syntax
def sample_bounded_conditional_mixture_metalog_slice_bilerp(
    cond_mix_coeffs: mi.TensorXf,
    cond_mix_metalog_coeffs: mi.TensorXf,
    marginal_quantiles: mi.TensorXf,
    marginal_quantile_cdf_range: tuple[float, float],
    sample2: mi.Point2f,
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
    bounds: tuple[mi.Float, mi.Float],
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a conditional bounded metalog mixture using interpolation across four slices and within-slice quantile
    rows.

    Args:
        cond_mix_coeffs (mi.TensorXf): Per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs (mi.TensorXf): Per-slice, per conditional, per-component conditional metalog
        coefficients, shape: [S, R, K, T].
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        bounds (tuple[mi.Float, mi.Float]): Lower and upper bounds.
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across four slices.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the conditional metalog mixture model, shape: [N].
          - [1]: Conditional Mixture PDF of the generated samples, shape: [N].
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]

    # find unique bracketing indices for each of the 4 corner slices
    lo00, hi00 = find_interval_binary_search(marginal_quantiles, slice_idx00, x)
    lo01, hi01 = find_interval_binary_search(marginal_quantiles, slice_idx01, x)
    lo10, hi10 = find_interval_binary_search(marginal_quantiles, slice_idx10, x)
    lo11, hi11 = find_interval_binary_search(marginal_quantiles, slice_idx11, x)

    # compute interpolation ratios (t) using slice-specific indices at x
    t00 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx00, lo00, hi00
    )
    t01 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx01, lo01, hi01
    )
    t10 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx10, lo10, hi10
    )
    t11 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, x, slice_idx11, lo11, hi11
    )

    unif = sample2.y
    N = dr.width(unif)
    S, R, K, T = cond_mix_metalog_coeffs.shape
    cond_mix_coeffs = cond_mix_coeffs.array
    cond_mix_metalog_coeffs = cond_mix_metalog_coeffs.array

    """
    Base flat indices for (S,R,K,T) and (S,R,K) tensors:
    - Mixing Coeffs: idx = ((s * R + r) * K + k)
    - Metalog Coeffs: idx = (((s * R + r) * K + k) * T + tci)
    """
    base00_lo = (slice_idx00 * R + lo00) * K
    base00_hi = (slice_idx00 * R + hi00) * K
    base01_lo = (slice_idx01 * R + lo01) * K
    base01_hi = (slice_idx01 * R + hi01) * K
    base10_lo = (slice_idx10 * R + lo10) * K
    base10_hi = (slice_idx10 * R + hi10) * K
    base11_lo = (slice_idx11 * R + lo11) * K
    base11_hi = (slice_idx11 * R + hi11) * K

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # row-lerp mixing coeffs within each corner slice using t** then bilerp
        # across four slices
        w00_lo = dr.gather(mi.Float, cond_mix_coeffs, base00_lo + kU)
        w00_hi = dr.gather(mi.Float, cond_mix_coeffs, base00_hi + kU)
        pi00 = dr.lerp(w00_lo, w00_hi, t00)

        w01_lo = dr.gather(mi.Float, cond_mix_coeffs, base01_lo + kU)
        w01_hi = dr.gather(mi.Float, cond_mix_coeffs, base01_hi + kU)
        pi01 = dr.lerp(w01_lo, w01_hi, t01)

        w10_lo = dr.gather(mi.Float, cond_mix_coeffs, base10_lo + kU)
        w10_hi = dr.gather(mi.Float, cond_mix_coeffs, base10_hi + kU)
        pi10 = dr.lerp(w10_lo, w10_hi, t10)

        w11_lo = dr.gather(mi.Float, cond_mix_coeffs, base11_lo + kU)
        w11_hi = dr.gather(mi.Float, cond_mix_coeffs, base11_hi + kU)
        pi11 = dr.lerp(w11_lo, w11_hi, t11)

        pi_lerp_k = bilerp(pi00, pi01, pi10, pi11)
        pi_list.append(pi_lerp_k)

        # row-lerp metalog coeffs within each corner slice per term, then bilerp
        # across four slices
        r00_lo = (base00_lo + kU) * T
        r00_hi = (base00_hi + kU) * T
        r01_lo = (base01_lo + kU) * T
        r01_hi = (base01_hi + kU) * T
        r10_lo = (base10_lo + kU) * T
        r10_hi = (base10_hi + kU) * T
        r11_lo = (base11_lo + kU) * T
        r11_hi = (base11_hi + kU) * T

        a_k = []
        for ti in range(T):
            tiU = mi.UInt(ti)
            a00 = dr.lerp(
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r00_lo + tiU),
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r00_hi + tiU),
                t00,
            )
            a01 = dr.lerp(
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r01_lo + tiU),
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r01_hi + tiU),
                t01,
            )
            a10 = dr.lerp(
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r10_lo + tiU),
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r10_hi + tiU),
                t10,
            )
            a11 = dr.lerp(
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r11_lo + tiU),
                dr.gather(mi.Float, cond_mix_metalog_coeffs, r11_hi + tiU),
                t11,
            )
            a_k_t = bilerp(a00, a01, a10, a11)
            a_k.append(a_k_t)
        a_terms.append(a_k)

        # accumulate mixing coeffs
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_lerp_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_lerp_k, sel_pi_k)

        # pick the metalog terms according to `take`
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    y, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather lerped mixing and metalog coeffs
        pi_lerp_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, y, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_lerp_k * pdf_k

    return y, mix_pdf


@dr.syntax
def sample_bounded_conditional_mixture_metalog_slice_bilerp_blend(
    cond_mix_coeffs_A: mi.TensorXf,
    cond_mix_metalog_coeffs_A: mi.TensorXf,
    marginal_quantiles_A: mi.TensorXf,
    cond_mix_coeffs_B: mi.TensorXf,
    cond_mix_metalog_coeffs_B: mi.TensorXf,
    marginal_quantiles_B: mi.TensorXf,
    sample2: mi.Point2f,
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    lambda_val: mi.Float,
    t_theta: mi.Float,
    t_phi: mi.Float,
    bounds: tuple[mi.Float, mi.Float],
    x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Sample from a conditional bounded metalog mixture blending two models using interpolation across four slices and
    within-slice quantile rows.

    Args:
        cond_mix_coeffs_A (mi.TensorXf): Material A,per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R, K, T].
        marginal_quantiles_A (mi.TensorXf): Material A, per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R, K, T].
        marginal_quantiles_B (mi.TensorXf): Material B, per-slice marginal quantiles, shape: [S, R].
        sample2 (mi.Point2f): Uniform variates in [0, 1]^2, shape: [N, 2].
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        lambda_val (mi.Float): Interpolation ratio between material A and B.
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        bounds (tuple[mi.Float, mi.Float]): Lower and upper bounds.
        x (mi.Float): Sample from the bounded marginal metalog using interpolation across four slices.

    Returns:
        tuple[mi.Float, mi.Float]:
          - [0]: Samples from the conditional metalog mixture model, shape: [N].
          - [1]: Conditional Mixture PDF of the generated samples, shape: [N].
    """
    unif = sample2.y
    N = dr.width(unif)
    S, R_A, K, T = cond_mix_metalog_coeffs_A.shape
    _, R_B, _, _ = cond_mix_metalog_coeffs_B.shape
    cond_mix_coeffs_A = cond_mix_coeffs_A.array
    cond_mix_metalog_coeffs_A = cond_mix_metalog_coeffs_A.array
    cond_mix_coeffs_B = cond_mix_coeffs_B.array
    cond_mix_metalog_coeffs_B = cond_mix_metalog_coeffs_B.array

    """Material A: Find unique bracketing indices and interpolation ratios"""
    lo00_A, hi00_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx00, x
    )
    lo01_A, hi01_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx01, x
    )
    lo10_A, hi10_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx10, x
    )
    lo11_A, hi11_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx11, x
    )
    t00_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, x, slice_idx00, lo00_A, hi00_A
    )
    t01_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, x, slice_idx01, lo01_A, hi01_A
    )
    t10_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, x, slice_idx10, lo10_A, hi10_A
    )
    t11_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, x, slice_idx11, lo11_A, hi11_A
    )
    base00_lo_A = (slice_idx00 * R_A + lo00_A) * K
    base00_hi_A = (slice_idx00 * R_A + hi00_A) * K
    base01_lo_A = (slice_idx01 * R_A + lo01_A) * K
    base01_hi_A = (slice_idx01 * R_A + hi01_A) * K
    base10_lo_A = (slice_idx10 * R_A + lo10_A) * K
    base10_hi_A = (slice_idx10 * R_A + hi10_A) * K
    base11_lo_A = (slice_idx11 * R_A + lo11_A) * K
    base11_hi_A = (slice_idx11 * R_A + hi11_A) * K

    """Material B: Find unique bracketing indices and interpolation ratios"""
    lo00_B, hi00_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx00, x
    )
    lo01_B, hi01_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx01, x
    )
    lo10_B, hi10_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx10, x
    )
    lo11_B, hi11_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx11, x
    )
    t00_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, x, slice_idx00, lo00_B, hi00_B
    )
    t01_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, x, slice_idx01, lo01_B, hi01_B
    )
    t10_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, x, slice_idx10, lo10_B, hi10_B
    )
    t11_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, x, slice_idx11, lo11_B, hi11_B
    )
    base00_lo_B = (slice_idx00 * R_B + lo00_B) * K
    base00_hi_B = (slice_idx00 * R_B + hi00_B) * K
    base01_lo_B = (slice_idx01 * R_B + lo01_B) * K
    base01_hi_B = (slice_idx01 * R_B + hi01_B) * K
    base10_lo_B = (slice_idx10 * R_B + lo10_B) * K
    base10_hi_B = (slice_idx10 * R_B + hi10_B) * K
    base11_lo_B = (slice_idx11 * R_B + lo11_B) * K
    base11_hi_B = (slice_idx11 * R_B + hi11_B) * K

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    def bilerp_cond_mixing_weights(
        cond_mix_coeffs,
        b00_lo,
        b00_hi,
        b01_lo,
        b01_hi,
        b10_lo,
        b10_hi,
        b11_lo,
        b11_hi,
        t00,
        t01,
        t10,
        t11,
        kU,
    ):
        # row-lerp mixing coeffs within each corner slice using t** then bilerp
        # across four slices
        w00_lo = dr.gather(mi.Float, cond_mix_coeffs, b00_lo + kU)
        w00_hi = dr.gather(mi.Float, cond_mix_coeffs, b00_hi + kU)
        pi00 = dr.lerp(w00_lo, w00_hi, t00)

        w01_lo = dr.gather(mi.Float, cond_mix_coeffs, b01_lo + kU)
        w01_hi = dr.gather(mi.Float, cond_mix_coeffs, b01_hi + kU)
        pi01 = dr.lerp(w01_lo, w01_hi, t01)

        w10_lo = dr.gather(mi.Float, cond_mix_coeffs, b10_lo + kU)
        w10_hi = dr.gather(mi.Float, cond_mix_coeffs, b10_hi + kU)
        pi10 = dr.lerp(w10_lo, w10_hi, t10)

        w11_lo = dr.gather(mi.Float, cond_mix_coeffs, b11_lo + kU)
        w11_hi = dr.gather(mi.Float, cond_mix_coeffs, b11_hi + kU)
        pi11 = dr.lerp(w11_lo, w11_hi, t11)

        return bilerp(pi00, pi01, pi10, pi11)

    def bilerp_cond_metalog_coeffs(
        cond_metalog_coeffs,
        b00_lo,
        b00_hi,
        b01_lo,
        b01_hi,
        b10_lo,
        b10_hi,
        b11_lo,
        b11_hi,
        t00,
        t01,
        t10,
        t11,
        kU,
        tiU,
    ):
        r00_lo = (b00_lo + kU) * T
        r00_hi = (b00_hi + kU) * T
        a00 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r00_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r00_hi + tiU),
            t00,
        )

        r01_lo = (b01_lo + kU) * T
        r01_hi = (b01_hi + kU) * T
        a01 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r01_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r01_hi + tiU),
            t01,
        )

        r10_lo = (b10_lo + kU) * T
        r10_hi = (b10_hi + kU) * T
        a10 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r10_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r10_hi + tiU),
            t10,
        )

        r11_lo = (b11_lo + kU) * T
        r11_hi = (b11_hi + kU) * T
        a11 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r11_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r11_hi + tiU),
            t11,
        )

        return bilerp(a00, a01, a10, a11)

    """
    Pass 1: Build interpolated mixing, metalog coeffs and select according to
    the first component whose cumulative covers the uniform variate.
    """
    pi_list = []
    a_terms = []
    cum_pi = dr.zeros(mi.Float, N)
    selected = dr.full(mi.Bool, False, N)
    sel_prev_cum_pi = dr.zeros(mi.Float, N)
    sel_pi_k = dr.zeros(mi.Float, N)
    sel_a_terms = [dr.zeros(mi.Float, N) for _ in range(T)]

    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing weights between material A and B
        pi_k_A = bilerp_cond_mixing_weights(
            cond_mix_coeffs_A,
            base00_lo_A,
            base00_hi_A,
            base01_lo_A,
            base01_hi_A,
            base10_lo_A,
            base10_hi_A,
            base11_lo_A,
            base11_hi_A,
            t00_A,
            t01_A,
            t10_A,
            t11_A,
            kU,
        )
        pi_k_B = bilerp_cond_mixing_weights(
            cond_mix_coeffs_B,
            base00_lo_B,
            base00_hi_B,
            base01_lo_B,
            base01_hi_B,
            base10_lo_B,
            base10_hi_B,
            base11_lo_B,
            base11_hi_B,
            t00_B,
            t01_B,
            t10_B,
            t11_B,
            kU,
        )
        pi_lerp_k = dr.lerp(pi_k_A, pi_k_B, lambda_val)
        pi_list.append(pi_lerp_k)

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in range(T):
            tiU = mi.UInt(ti)

            term_A = bilerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_A,
                base00_lo_A,
                base00_hi_A,
                base01_lo_A,
                base01_hi_A,
                base10_lo_A,
                base10_hi_A,
                base11_lo_A,
                base11_hi_A,
                t00_A,
                t01_A,
                t10_A,
                t11_A,
                kU,
                tiU,
            )
            term_B = bilerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_B,
                base00_lo_B,
                base00_hi_B,
                base01_lo_B,
                base01_hi_B,
                base10_lo_B,
                base10_hi_B,
                base11_lo_B,
                base11_hi_B,
                t00_B,
                t01_B,
                t10_B,
                t11_B,
                kU,
                tiU,
            )
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        a_terms.append(a_k)

        # accumulate mixing coeffs
        prev_cum_pi = cum_pi
        cum_pi = cum_pi + pi_lerp_k

        # take the first component whose cumulative covers the unif
        take = (~selected) & (unif <= cum_pi)

        # pick the previous cumulative and mix coeffients according to `take`
        sel_prev_cum_pi = dr.select(take, prev_cum_pi, sel_prev_cum_pi)
        sel_pi_k = dr.select(take, pi_lerp_k, sel_pi_k)

        # pick the metalog terms according to `take`
        for t_idx in range(T):
            sel_a_terms[t_idx] = dr.select(take, a_k[t_idx], sel_a_terms[t_idx])

        selected = selected | take

    # ensure a component is chosen even with tiny numerical drift
    take_last = ~selected
    pi_last = pi_list[-1]
    a_last = a_terms[-1]
    sel_prev_cum_pi = dr.select(take_last, cum_pi - pi_last, sel_prev_cum_pi)
    sel_pi_k = dr.maximum(dr.select(take_last, pi_last, sel_pi_k), EPSILON)
    for ti in range(T):
        sel_a_terms[ti] = dr.select(take_last, a_last[ti], sel_a_terms[ti])

    # local uniform inside the chosen components interval [0, 1]
    u_local = dr.clip(
        (unif - sel_prev_cum_pi) * dr.rcp(sel_pi_k), EPSILON, 1.0 - EPSILON
    )

    # sample from the chosen component
    y, _ = eval_bounded_metalog_quantile(
        sel_a_terms, bounds[0], bounds[1], u_local
    )

    """Pass 2: Compute exact mixture PDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    for k in range(K):
        # gather lerped mixing and metalog coeffs
        pi_lerp_k = pi_list[k]
        a_k = a_terms[k]

        # evaluate the metalog PDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, y, u_local, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF (Eq. 3.111, Bishop 2023)
        mix_pdf += pi_lerp_k * pdf_k

    return y, mix_pdf


@dr.syntax
def eval_bounded_mixture_metalog_cdf_pdf(
    mixing_coeffs: mi.TensorXf,
    metalog_coeffs: mi.TensorXf,
    x: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded metalog-mixture CDF and PDF.

    Args:
        mixing_coeffs (mi.TensorXf): Mixing coefficients, shape: [K].
        metalog_coeffs (mi.TensorXf): Per-component metalog coefficients, shape: [K, T].
        x (mi.Float): Samples to evaluate the metalog-mixture CDF/PDF.
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Metalog-mixture CDF for given x, shape: [N].
            - [1]: Metalog-mixture PDF for given x, shape: [N]
    """

    N = dr.width(x)
    K, T = metalog_coeffs.shape
    mixing_coeffs = mixing_coeffs.array
    metalog_coeffs = metalog_coeffs.array

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((x - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    """Compute exact mixture PDF/CDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # gather mixing and metalog coeffs
        pi_k = dr.gather(mi.Float, mixing_coeffs, kU)
        a_k = [dr.gather(mi.Float, metalog_coeffs, kU * T + ti) for ti in tiU]

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf = dr.fma(pi_k, u_k, mix_cdf)
        mix_pdf = dr.fma(pi_k, pdf_k, mix_pdf)

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_mixture_metalog_cdf_pdf_slice_lerp_blend(
    mixing_coeffs_A: mi.TensorXf,
    metalog_coeffs_A: mi.TensorXf,
    mixing_coeffs_B: mi.TensorXf,
    metalog_coeffs_B: mi.TensorXf,
    x: mi.Float,
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    lambda_val: mi.Float,
    t: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded metalog-mixture CDF and PDF blending two models using interpolation across two slices.

    Args:
        mixing_coeffs_A (mi.TensorXf): Material A, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per-component metalog coefficients, shape: [S, K, T].
        mixing_coeffs_B (mi.TensorXf): Material B, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per-component metalog coefficients, shape: [S, K, T].
        x (mi.Float): Samples to evaluate the metalog-mixture CDF/PDF.
        slice_idx0 (mi.UInt): First slice index used for interpolation, shape: [N].
        slice_idx1 (mi.UInt): Second slice index used for interpolation, shape: [N].
        lambda_val (mi.Float): Blending interpolation ratio between material A and B.
        t (mi.Float): Cross-slice interpolation ratio, shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Metalog-mixture CDF for given x, shape: [N].
            - [1]: Metalog-mixture PDF for given x, shape: [N]
    """

    N = dr.width(x)
    S, K, T = metalog_coeffs_A.shape
    mixing_coeffs_A = mixing_coeffs_A.array
    metalog_coeffs_A = metalog_coeffs_A.array
    mixing_coeffs_B = mixing_coeffs_B.array
    metalog_coeffs_B = metalog_coeffs_B.array

    # compute base indices for two slices
    base_idx0 = slice_idx0 * K
    base_idx1 = slice_idx1 * K

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((x - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    def lerp_mixing_weights(mixing_coeffs, kU):
        # lerp mixing coeffs for component k between two slices
        pi0 = dr.gather(mi.Float, mixing_coeffs, base_idx0 + kU)
        pi1 = dr.gather(mi.Float, mixing_coeffs, base_idx1 + kU)
        return dr.lerp(pi0, pi1, t)

    def lerp_metalog_coeffs(metalog_coeffs, r0, r1, tiU):
        c0 = dr.gather(mi.Float, metalog_coeffs, r0 + tiU)
        c1 = dr.gather(mi.Float, metalog_coeffs, r1 + tiU)
        return dr.lerp(c0, c1, t)

    """Compute exact mixture PDF/CDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing coeffs between material A and B
        pi_lerp_k_A = lerp_mixing_weights(mixing_coeffs_A, kU)
        pi_lerp_k_B = lerp_mixing_weights(mixing_coeffs_B, kU)
        pi_k = dr.lerp(pi_lerp_k_A, pi_lerp_k_B, lambda_val)

        r0 = (base_idx0 + kU) * T
        r1 = (base_idx1 + kU) * T

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in tiU:
            term_A = lerp_metalog_coeffs(metalog_coeffs_A, r0, r1, ti)
            term_B = lerp_metalog_coeffs(metalog_coeffs_B, r0, r1, ti)
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_mixture_metalog_cdf_pdf_slice_lerp(
    mixing_coeffs: mi.TensorXf,
    metalog_coeffs: mi.TensorXf,
    x: mi.Float,
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    t: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded metalog-mixture CDF and PDF using interpolation across two slices.

    Args:
        mixing_coeffs (mi.TensorXf): Per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs (mi.TensorXf): Per-slice, per-component metalog coefficients, shape: [S, K, T].
        x (mi.Float): Samples to evaluate the metalog-mixture CDF/PDF.
        slice_idx0 (mi.UInt): First slice index used for interpolation, shape: [N].
        slice_idx1 (mi.UInt): Second slice index used for interpolation, shape: [N].
        t (mi.Float): Cross-slice interpolation ratio, shape: [N].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Metalog-mixture CDF for given x, shape: [N].
            - [1]: Metalog-mixture PDF for given x, shape: [N]
    """

    N = dr.width(x)
    S, K, T = metalog_coeffs.shape
    mixing_coeffs = mixing_coeffs.array
    metalog_coeffs = metalog_coeffs.array

    # compute base indices for two slices
    base_idx0 = slice_idx0 * K
    base_idx1 = slice_idx1 * K

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((x - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    """Compute exact mixture PDF/CDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # lerp mixing coeffs across two slices
        pi0 = dr.gather(mi.Float, mixing_coeffs, base_idx0 + kU)
        pi1 = dr.gather(mi.Float, mixing_coeffs, base_idx1 + kU)
        pi_k = dr.lerp(pi0, pi1, t)

        # lerp metalog coeffs across two slices
        r0 = (base_idx0 + kU) * T
        r1 = (base_idx1 + kU) * T
        a_k = [
            dr.lerp(
                dr.gather(mi.Float, metalog_coeffs, r0 + ti),
                dr.gather(mi.Float, metalog_coeffs, r1 + ti),
                t,
            )
            for ti in tiU
        ]

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_conditional_mixture_metalog_cdf_pdf(
    marginal_quantiles: mi.TensorXf,
    cond_mix_coeffs: mi.TensorXf,
    cond_mix_metalog_coeffs: mi.TensorXf,
    sample2: mi.Point2f,
    bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    unif_x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog-mixture CDF and PDF using interpolation across two slices.

    Args:
        marginal_quantiles (mi.TensorXf): Marginal quantiles, shape: [R].
        cond_mix_coeffs (mi.TensorXf): Per conditional mixing coefficients, shape: [R, K].
        cond_mix_metalog_coeffs (mi.TensorXf): Per conditional, per-component conditional metalog coefficients,
        shape: [R, K, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        unif_x (mi.Float): Per-sample marginal metalog CDFs for the x-axis.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of within quantile interpolated conditional metalogs.
    """
    N = dr.width(sample2.y)
    R, K, T = cond_mix_metalog_coeffs.shape
    cond_mix_coeffs = cond_mix_coeffs.array
    cond_mix_metalog_coeffs = cond_mix_metalog_coeffs.array

    # compute lower and upper row indices to interpolate conditionals
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif_x, marginal_quantile_cdf_range
    )

    # compute interpolation ratio
    t = eval_interpolation_ratios(
        marginal_quantiles, sample2.x, lower_idx, upper_idx
    )

    # base flat indices for (R,K,T) and (R,K) tensors:
    base_lo = lower_idx * K
    base_hi = upper_idx * K

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((sample2.y - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    """Compute exact mixture PDF/CDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # lerp mixing coeffs within slice
        w_lo = dr.gather(mi.Float, cond_mix_coeffs, base_lo + kU)
        w_hi = dr.gather(mi.Float, cond_mix_coeffs, base_hi + kU)
        pi_k = dr.lerp(w_lo, w_hi, t)

        # lerp metalog coeffs within slice
        r_lo = (base_lo + kU) * T
        r_hi = (base_hi + kU) * T

        a_k = []
        for ti in tiU:
            a_lo = dr.gather(mi.Float, cond_mix_metalog_coeffs, r_lo + ti)
            a_hi = dr.gather(mi.Float, cond_mix_metalog_coeffs, r_hi + ti)
            a_k.append(dr.lerp(a_lo, a_hi, t))

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, sample2.y, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf = dr.fma(pi_k, u_k, mix_cdf)
        mix_pdf = dr.fma(pi_k, pdf_k, mix_pdf)

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_lerp(
    marginal_quantiles: mi.TensorXf,
    cond_mix_coeffs: mi.TensorXf,
    cond_mix_metalog_coeffs: mi.TensorXf,
    sample2: mi.Point2f,
    bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    t: mi.Float,
    unif_x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog-mixture CDF and PDF using interpolation across two slices.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs (mi.TensorXf): Per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs (mi.TensorXf): Per-slice, per conditional, per-component conditional metalog
        coefficients, shape: [S, R, K, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        uniformly spaced over this range.
        slice_idx0 (mi.UInt): First slice index used for interpolation.
        slice_idx1 (mi.UInt): Second slice index used for interpolation.
        t (mi.Float): Cross-slice interpolation ratio.
        unif_x (mi.Float): Per-sample marginal metalog CDFs, where metalogs interpolated across slices.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of across slice and within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of across slice and within quantile interpolated conditional metalogs.
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif_x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t0 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx0, lower_idx, upper_idx
    )
    t1 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx1, lower_idx, upper_idx
    )

    N = dr.width(sample2.y)
    S, R, K, T = cond_mix_metalog_coeffs.shape
    cond_mix_coeffs = cond_mix_coeffs.array
    cond_mix_metalog_coeffs = cond_mix_metalog_coeffs.array

    """
    Base flat indices for (S,R,K,T) and (S,R,K) tensors:
    - Mixing Coeffs: idx = ((s * R + r) * K + k)
    - Metalog Coeffs: idx = (((s * R + r) * K + k) * T + tci)
    """
    base0_lo = (slice_idx0 * R + lower_idx) * K
    base0_hi = (slice_idx0 * R + upper_idx) * K
    base1_lo = (slice_idx1 * R + lower_idx) * K
    base1_hi = (slice_idx1 * R + upper_idx) * K

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((sample2.y - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    """Compute exact mixture PDF/CDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # lerp mixing coeffs within slice then across slices
        w00 = dr.gather(mi.Float, cond_mix_coeffs, base0_lo + kU)
        w01 = dr.gather(mi.Float, cond_mix_coeffs, base0_hi + kU)
        pi0 = dr.lerp(w00, w01, t0)

        w10 = dr.gather(mi.Float, cond_mix_coeffs, base1_lo + kU)
        w11 = dr.gather(mi.Float, cond_mix_coeffs, base1_hi + kU)
        pi1 = dr.lerp(w10, w11, t1)

        pi_k = dr.lerp(pi0, pi1, t)

        # lerp metalog coeffs within slice then across slices
        r00 = (base0_lo + kU) * T
        r01 = (base0_hi + kU) * T
        r10 = (base1_lo + kU) * T
        r11 = (base1_hi + kU) * T

        a_k = []
        for ti in tiU:
            a00 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r00 + ti)
            a01 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r01 + ti)
            a0 = dr.lerp(a00, a01, t0)

            a10 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r10 + ti)
            a11 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r11 + ti)
            a1 = dr.lerp(a10, a11, t1)

            a_k.append(dr.lerp(a0, a1, t))

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, sample2.y, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_lerp_blend(
    marginal_quantiles_A: mi.TensorXf,
    cond_mix_coeffs_A: mi.TensorXf,
    cond_mix_metalog_coeffs_A: mi.TensorXf,
    marginal_quantiles_B: mi.TensorXf,
    cond_mix_coeffs_B: mi.TensorXf,
    cond_mix_metalog_coeffs_B: mi.TensorXf,
    sample2: mi.Point2f,
    bounds: tuple[float, float],
    slice_idx0: mi.UInt,
    slice_idx1: mi.UInt,
    lambda_val: mi.Float,
    t: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog-mixture CDF and PDF blending two models using interpolation across two slices.

    Args:
        marginal_quantiles_A (mi.TensorXf): Material A, per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs_A (mi.TensorXf): Material A, per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R,  K, T].
        marginal_quantiles_B (mi.TensorXf): Material B, per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R,  K, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        bounds (tuple[float, float]): Lower and upper bound for the conditional.
        slice_idx0 (mi.UInt): First slice index used for interpolation.
        slice_idx1 (mi.UInt): Second slice index used for interpolation.
        lambda_val (mi.Float): Interpolation ratio between material A and B.
        t (mi.Float): Cross-slice interpolation ratio.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of across slice and within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of across slice and within quantile interpolated conditional metalogs.
    """
    N = dr.width(sample2.y)
    S, R_A, K, T = cond_mix_metalog_coeffs_A.shape
    _, R_B, _, _ = cond_mix_metalog_coeffs_B.shape
    cond_mix_coeffs_A = cond_mix_coeffs_A.array
    cond_mix_metalog_coeffs_A = cond_mix_metalog_coeffs_A.array
    cond_mix_coeffs_B = cond_mix_coeffs_B.array
    cond_mix_metalog_coeffs_B = cond_mix_metalog_coeffs_B.array

    """Material A: Find unique bracketing indices and interpolation ratios"""
    lo0_A, hi0_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx0, sample2.x
    )
    lo1_A, hi1_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx1, sample2.x
    )
    t0_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, sample2.x, slice_idx0, lo0_A, hi0_A
    )
    t1_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, sample2.x, slice_idx1, lo1_A, hi1_A
    )
    base0_lo_A = (slice_idx0 * R_A + lo0_A) * K
    base0_hi_A = (slice_idx0 * R_A + hi0_A) * K
    base1_lo_A = (slice_idx1 * R_A + lo1_A) * K
    base1_hi_A = (slice_idx1 * R_A + hi1_A) * K

    """Material B: Find unique bracketing indices and interpolation ratios"""
    lo0_B, hi0_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx0, sample2.x
    )
    lo1_B, hi1_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx1, sample2.x
    )
    t0_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, sample2.x, slice_idx0, lo0_B, hi0_B
    )
    t1_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, sample2.x, slice_idx1, lo1_B, hi1_B
    )
    base0_lo_B = (slice_idx0 * R_B + lo0_B) * K
    base0_hi_B = (slice_idx0 * R_B + hi0_B) * K
    base1_lo_B = (slice_idx1 * R_B + lo1_B) * K
    base1_hi_B = (slice_idx1 * R_B + hi1_B) * K

    def lerp_cond_mixing_weights(
        cond_mix_coeffs,
        base0_lo,
        base0_hi,
        base1_lo,
        base1_hi,
        t0,
        t1,
        kU,
    ):
        # row-lerp mixing coeffs in each slice then slice-lerp between two slice
        w00 = dr.gather(mi.Float, cond_mix_coeffs, base0_lo + kU)
        w01 = dr.gather(mi.Float, cond_mix_coeffs, base0_hi + kU)
        pi0 = dr.lerp(w00, w01, t0)

        w10 = dr.gather(mi.Float, cond_mix_coeffs, base1_lo + kU)
        w11 = dr.gather(mi.Float, cond_mix_coeffs, base1_hi + kU)
        pi1 = dr.lerp(w10, w11, t1)

        return dr.lerp(pi0, pi1, t)

    def lerp_cond_metalog_coeffs(
        cond_mix_metalog_coeffs,
        base0_lo,
        base0_hi,
        base1_lo,
        base1_hi,
        t0,
        t1,
        kU,
        tiU,
    ):
        r00 = (base0_lo + kU) * T
        r01 = (base0_hi + kU) * T
        r10 = (base1_lo + kU) * T
        r11 = (base1_hi + kU) * T

        a00 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r00 + tiU)
        a01 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r01 + tiU)
        a0r = dr.lerp(a00, a01, t0)

        a10 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r10 + tiU)
        a11 = dr.gather(mi.Float, cond_mix_metalog_coeffs, r11 + tiU)
        a1r = dr.lerp(a10, a11, t1)

        return dr.lerp(a0r, a1r, t)

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((sample2.y - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    """Compute exact mixture PDF/CDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing weights between material A and B
        pi_k_A = lerp_cond_mixing_weights(
            cond_mix_coeffs_A,
            base0_lo_A,
            base0_hi_A,
            base1_lo_A,
            base1_hi_A,
            t0_A,
            t1_A,
            kU,
        )
        pi_k_B = lerp_cond_mixing_weights(
            cond_mix_coeffs_B,
            base0_lo_B,
            base0_hi_B,
            base1_lo_B,
            base1_hi_B,
            t0_B,
            t1_B,
            kU,
        )
        pi_k = dr.lerp(pi_k_A, pi_k_B, lambda_val)

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in tiU:
            term_A = lerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_A,
                base0_lo_A,
                base0_hi_A,
                base1_lo_A,
                base1_hi_A,
                t0_A,
                t1_A,
                kU,
                ti,
            )
            term_B = lerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_B,
                base0_lo_B,
                base0_hi_B,
                base1_lo_B,
                base1_hi_B,
                t0_B,
                t1_B,
                kU,
                ti,
            )
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, sample2.y, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_mixture_metalog_cdf_pdf_slice_bilerp(
    mixing_coeffs: mi.TensorXf,
    metalog_coeffs: mi.TensorXf,
    x: mi.Float,
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded metalog-mixture CDF and PDF using interpolation across four slices.

    Args:
        mixing_coeffs (mi.TensorXf): Per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs (mi.TensorXf): Per-slice, per-component metalog coefficients, shape: [S, K, T].
        x (mi.Float): Samples to evaluate the metalog-mixture CDF/PDF.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Metalog-mixture CDF for given x, shape: [N].
            - [1]: Metalog-mixture PDF for given x, shape: [N]
    """
    N = dr.width(x)
    S, K, T = metalog_coeffs.shape
    mixing_coeffs = mixing_coeffs.array
    metalog_coeffs = metalog_coeffs.array

    # compute base indices per-corner
    base_idx00 = slice_idx00 * K
    base_idx01 = slice_idx01 * K
    base_idx10 = slice_idx10 * K
    base_idx11 = slice_idx11 * K

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y = dr.clip((x - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    """Compute exact mixture PDF/CDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # bilerp mixing coeffs for component k between four slices
        pi00 = dr.gather(mi.Float, mixing_coeffs, base_idx00 + kU)
        pi01 = dr.gather(mi.Float, mixing_coeffs, base_idx01 + kU)
        pi10 = dr.gather(mi.Float, mixing_coeffs, base_idx10 + kU)
        pi11 = dr.gather(mi.Float, mixing_coeffs, base_idx11 + kU)
        pi_k = bilerp(pi00, pi01, pi10, pi11)

        # bilerp per-component metalog terms between four slices
        r00 = (base_idx00 + kU) * T
        r01 = (base_idx01 + kU) * T
        r10 = (base_idx10 + kU) * T
        r11 = (base_idx11 + kU) * T
        a_k = [
            bilerp(
                dr.gather(mi.Float, metalog_coeffs, r00 + ti),
                dr.gather(mi.Float, metalog_coeffs, r01 + ti),
                dr.gather(mi.Float, metalog_coeffs, r10 + ti),
                dr.gather(mi.Float, metalog_coeffs, r11 + ti),
            )
            for ti in tiU
        ]

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, y, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_mixture_metalog_cdf_pdf_slice_bilerp_blend(
    mixing_coeffs_A: mi.TensorXf,
    metalog_coeffs_A: mi.TensorXf,
    mixing_coeffs_B: mi.TensorXf,
    metalog_coeffs_B: mi.TensorXf,
    x: mi.Float,
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    lambda_val: mi.Float,
    t_theta: mi.Float,
    t_phi: mi.Float,
    bounds: tuple[float, float],
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded metalog-mixture CDF and PDF blending two models using interpolation across four slices.

    Args:
        mixing_coeffs_A (mi.TensorXf): Material A, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per-component metalog coefficients, shape: [S, K, T].
        mixing_coeffs_B (mi.TensorXf): Material B, per-slice mixing coefficients, shape: [S, K].
        metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per-component metalog coefficients, shape: [S, K, T].
        x (mi.Float): Samples to evaluate the metalog-mixture CDF/PDF.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        lambda_val (mi.Float): Blending interpolation ratio between material A and B.
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        bounds (tuple[float, float]): Lower and upper bounds.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Metalog-mixture CDF for given x, shape: [N].
            - [1]: Metalog-mixture PDF for given x, shape: [N]
    """
    N = dr.width(x)
    S, K, T = metalog_coeffs_A.shape
    mixing_coeffs_A = mixing_coeffs_A.array
    metalog_coeffs_A = metalog_coeffs_A.array
    mixing_coeffs_B = mixing_coeffs_B.array
    metalog_coeffs_B = metalog_coeffs_B.array

    # compute base indices per-corner
    base_idx00 = slice_idx00 * K
    base_idx01 = slice_idx01 * K
    base_idx10 = slice_idx10 * K
    base_idx11 = slice_idx11 * K

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y = dr.clip((x - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    def bilerp_mixing_weights(mixing_coeffs, kU):
        # bilerp mixing coeffs for component k between four slices
        pi00 = dr.gather(mi.Float, mixing_coeffs, base_idx00 + kU)
        pi01 = dr.gather(mi.Float, mixing_coeffs, base_idx01 + kU)
        pi10 = dr.gather(mi.Float, mixing_coeffs, base_idx10 + kU)
        pi11 = dr.gather(mi.Float, mixing_coeffs, base_idx11 + kU)
        return bilerp(pi00, pi01, pi10, pi11)

    def bilerp_metalog_coeffs(metalog_coeffs, r00, r01, r10, r11, tiU):
        # bilerp per-component metalog terms between four slices
        c00 = dr.gather(mi.Float, metalog_coeffs, r00 + tiU)
        c01 = dr.gather(mi.Float, metalog_coeffs, r01 + tiU)
        c10 = dr.gather(mi.Float, metalog_coeffs, r10 + tiU)
        c11 = dr.gather(mi.Float, metalog_coeffs, r11 + tiU)
        return bilerp(c00, c01, c10, c11)

    """Compute exact mixture PDF/CDF at x (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing coeffs between material A and B
        pi_k_A = bilerp_mixing_weights(mixing_coeffs_A, kU)
        pi_k_B = bilerp_mixing_weights(mixing_coeffs_B, kU)
        pi_k = dr.lerp(pi_k_A, pi_k_B, lambda_val)

        # bilerp per-component metalog terms between four slices
        r00 = (base_idx00 + kU) * T
        r01 = (base_idx01 + kU) * T
        r10 = (base_idx10 + kU) * T
        r11 = (base_idx11 + kU) * T

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in tiU:
            term_A = bilerp_metalog_coeffs(
                metalog_coeffs_A, r00, r01, r10, r11, ti
            )
            term_B = bilerp_metalog_coeffs(
                metalog_coeffs_B, r00, r01, r10, r11, ti
            )
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        # evaluate the metalog PDF/CDF for the component
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, x, y, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_bilerp(
    marginal_quantiles: mi.TensorXf,
    cond_mix_coeffs: mi.TensorXf,
    cond_mix_metalog_coeffs: mi.TensorXf,
    sample2: mi.Point2f,
    bounds: tuple[float, float],
    marginal_quantile_cdf_range: tuple[float, float],
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    t_theta: mi.Float,
    t_phi: mi.Float,
    unif_x: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog-mixture CDF and PDF using interpolation across four slices.

    Args:
        marginal_quantiles (mi.TensorXf): Per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs (mi.TensorXf): Per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs (mi.TensorXf): Per-slice, per conditional, per-component conditional metalog
        coefficients, shape: [S, R, K, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        bounds (tuple[float, float]): Lower and upper bound for the conditional.
        marginal_quantile_cdf_range (tuple[float, float]): CDF range for the marginal grid. Assume the grid is
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].
        unif_x (mi.Float): Per-sample marginal metalog CDFs, where metalogs interpolated across four slices.

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of across slice and within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of across slice and within quantile interpolated conditional metalogs.
    """
    # compute lower and upper row indices to interpolate conditionals
    R = marginal_quantiles.shape[-1]
    lower_idx, upper_idx = eval_lower_upper_indices(
        R, unif_x, marginal_quantile_cdf_range
    )

    # compute per-slice interpolation ratios at x
    t00 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx00, lower_idx, upper_idx
    )
    t01 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx01, lower_idx, upper_idx
    )
    t10 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx10, lower_idx, upper_idx
    )
    t11 = eval_interpolation_ratio_per_slice(
        marginal_quantiles, sample2.x, slice_idx11, lower_idx, upper_idx
    )

    N = dr.width(sample2.y)
    S, R, K, T = cond_mix_metalog_coeffs.shape
    cond_mix_coeffs = cond_mix_coeffs.array
    cond_mix_metalog_coeffs = cond_mix_metalog_coeffs.array

    """
    Base flat indices for (S,R,K,T) and (S,R,K) tensors:
    - Mixing Coeffs: idx = ((s * R + r) * K + k)
    - Metalog Coeffs: idx = (((s * R + r) * K + k) * T + tci)
    """
    base00_lo = (slice_idx00 * R + lower_idx) * K
    base00_hi = (slice_idx00 * R + upper_idx) * K
    base01_lo = (slice_idx01 * R + lower_idx) * K
    base01_hi = (slice_idx01 * R + upper_idx) * K
    base10_lo = (slice_idx10 * R + lower_idx) * K
    base10_hi = (slice_idx10 * R + upper_idx) * K
    base11_lo = (slice_idx11 * R + lower_idx) * K
    base11_hi = (slice_idx11 * R + upper_idx) * K

    def rowlerp_then_bilerp(
        v00_lo, v00_hi, v01_lo, v01_hi, v10_lo, v10_hi, v11_lo, v11_hi
    ):
        # row lerp inside each corner slice
        v00 = dr.lerp(v00_lo, v00_hi, t00)
        v01 = dr.lerp(v01_lo, v01_hi, t01)
        v10 = dr.lerp(v10_lo, v10_hi, t10)
        v11 = dr.lerp(v11_lo, v11_hi, t11)

        # bilerp: lerp across theta then across phi
        a0 = dr.lerp(v00, v01, t_theta)
        a1 = dr.lerp(v10, v11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((sample2.y - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    """Compute exact mixture PDF/CDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # lerp mixing coeffs within slice then across slices
        w00_lo = dr.gather(mi.Float, cond_mix_coeffs, base00_lo + kU)
        w00_hi = dr.gather(mi.Float, cond_mix_coeffs, base00_hi + kU)
        w01_lo = dr.gather(mi.Float, cond_mix_coeffs, base01_lo + kU)
        w01_hi = dr.gather(mi.Float, cond_mix_coeffs, base01_hi + kU)
        w10_lo = dr.gather(mi.Float, cond_mix_coeffs, base10_lo + kU)
        w10_hi = dr.gather(mi.Float, cond_mix_coeffs, base10_hi + kU)
        w11_lo = dr.gather(mi.Float, cond_mix_coeffs, base11_lo + kU)
        w11_hi = dr.gather(mi.Float, cond_mix_coeffs, base11_hi + kU)

        pi_k = rowlerp_then_bilerp(
            w00_lo, w00_hi, w01_lo, w01_hi, w10_lo, w10_hi, w11_lo, w11_hi
        )

        # lerp metalog coeffs within slice then across slices
        r00_lo = (base00_lo + kU) * T
        r00_hi = (base00_hi + kU) * T
        r01_lo = (base01_lo + kU) * T
        r01_hi = (base01_hi + kU) * T
        r10_lo = (base10_lo + kU) * T
        r10_hi = (base10_hi + kU) * T
        r11_lo = (base11_lo + kU) * T
        r11_hi = (base11_hi + kU) * T

        a_k = []
        for ti in tiU:
            a_k.append(
                rowlerp_then_bilerp(
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r00_lo + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r00_hi + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r01_lo + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r01_hi + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r10_lo + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r10_hi + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r11_lo + ti),
                    dr.gather(mi.Float, cond_mix_metalog_coeffs, r11_hi + ti),
                )
            )

        # evaluate the metalog PDF/CDF for the component at y
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, sample2.y, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


@dr.syntax
def eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_bilerp_blend(
    marginal_quantiles_A: mi.TensorXf,
    cond_mix_coeffs_A: mi.TensorXf,
    cond_mix_metalog_coeffs_A: mi.TensorXf,
    marginal_quantiles_B: mi.TensorXf,
    cond_mix_coeffs_B: mi.TensorXf,
    cond_mix_metalog_coeffs_B: mi.TensorXf,
    sample2: mi.Point2f,
    bounds: tuple[float, float],
    slice_idx00: mi.UInt,
    slice_idx01: mi.UInt,
    slice_idx10: mi.UInt,
    slice_idx11: mi.UInt,
    lambda_val: mi.Float,
    t_theta: mi.Float,
    t_phi: mi.Float,
) -> tuple[mi.Float, mi.Float]:
    """
    Evaluates bounded conditional metalog-mixture CDF and PDF blending two models using interpolation across four
    slices.

    Args:
        marginal_quantiles_A (mi.TensorXf): Material A, per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs_A (mi.TensorXf): Material A,per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_A (mi.TensorXf): Material A, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R, K, T].
        marginal_quantiles_B (mi.TensorXf): Material B, per-slice marginal quantiles, shape: [S, R].
        cond_mix_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional mixing coefficients, shape: [S, R, K].
        cond_mix_metalog_coeffs_B (mi.TensorXf): Material B, per-slice, per conditional, per-component conditional
        metalog coefficients, shape: [S, R, K, T].
        sample2 (mi.Point2f): Marginal-conditional samples to evaluate the metalog CDF/PDF.
        bounds (tuple[float, float]): Lower and upper bound for the conditional.
        slice_idx00 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta0).
        slice_idx01 (mi.UInt): Per-sample corner slice indices for cell (phi0, theta1).
        slice_idx10 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta0).
        slice_idx11 (mi.UInt): Per-sample corner slice indices for cell (phi1, theta1).
        lambda_val (mi.Float): Interpolation ratio between material A and B.
        t_theta (mi.Float): Per-sample interpolation coordinate along theta in [0, 1].
        t_phi (mi.Float): Per-sample interpolation coordinate along phi in [0, 1].

    Returns:
        tuple[mi.Float, mi.Float]:
            - [0]: Per-sample CDFs of across slice and within quantile interpolated conditional metalogs.
            - [1]: Per-sample PDFs of across slice and within quantile interpolated conditional metalogs.
    """
    N = dr.width(sample2.y)
    S, R_A, K, T = cond_mix_metalog_coeffs_A.shape
    _, R_B, _, _ = cond_mix_metalog_coeffs_B.shape
    cond_mix_coeffs_A = cond_mix_coeffs_A.array
    cond_mix_metalog_coeffs_A = cond_mix_metalog_coeffs_A.array
    cond_mix_coeffs_B = cond_mix_coeffs_B.array
    cond_mix_metalog_coeffs_B = cond_mix_metalog_coeffs_B.array

    """Material A: Find unique bracketing indices and interpolation ratios"""
    lo00_A, hi00_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx00, sample2.x
    )
    lo01_A, hi01_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx01, sample2.x
    )
    lo10_A, hi10_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx10, sample2.x
    )
    lo11_A, hi11_A = find_interval_binary_search(
        marginal_quantiles_A, slice_idx11, sample2.x
    )
    t00_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, sample2.x, slice_idx00, lo00_A, hi00_A
    )
    t01_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, sample2.x, slice_idx01, lo01_A, hi01_A
    )
    t10_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, sample2.x, slice_idx10, lo10_A, hi10_A
    )
    t11_A = eval_interpolation_ratio_per_slice(
        marginal_quantiles_A, sample2.x, slice_idx11, lo11_A, hi11_A
    )
    base00_lo_A = (slice_idx00 * R_A + lo00_A) * K
    base00_hi_A = (slice_idx00 * R_A + hi00_A) * K
    base01_lo_A = (slice_idx01 * R_A + lo01_A) * K
    base01_hi_A = (slice_idx01 * R_A + hi01_A) * K
    base10_lo_A = (slice_idx10 * R_A + lo10_A) * K
    base10_hi_A = (slice_idx10 * R_A + hi10_A) * K
    base11_lo_A = (slice_idx11 * R_A + lo11_A) * K
    base11_hi_A = (slice_idx11 * R_A + hi11_A) * K

    """Material B: Find unique bracketing indices and interpolation ratios"""
    lo00_B, hi00_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx00, sample2.x
    )
    lo01_B, hi01_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx01, sample2.x
    )
    lo10_B, hi10_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx10, sample2.x
    )
    lo11_B, hi11_B = find_interval_binary_search(
        marginal_quantiles_B, slice_idx11, sample2.x
    )
    t00_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, sample2.x, slice_idx00, lo00_B, hi00_B
    )
    t01_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, sample2.x, slice_idx01, lo01_B, hi01_B
    )
    t10_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, sample2.x, slice_idx10, lo10_B, hi10_B
    )
    t11_B = eval_interpolation_ratio_per_slice(
        marginal_quantiles_B, sample2.x, slice_idx11, lo11_B, hi11_B
    )
    base00_lo_B = (slice_idx00 * R_B + lo00_B) * K
    base00_hi_B = (slice_idx00 * R_B + hi00_B) * K
    base01_lo_B = (slice_idx01 * R_B + lo01_B) * K
    base01_hi_B = (slice_idx01 * R_B + hi01_B) * K
    base10_lo_B = (slice_idx10 * R_B + lo10_B) * K
    base10_hi_B = (slice_idx10 * R_B + hi10_B) * K
    base11_lo_B = (slice_idx11 * R_B + lo11_B) * K
    base11_hi_B = (slice_idx11 * R_B + hi11_B) * K

    def bilerp(a00, a01, a10, a11):
        a0 = dr.lerp(a00, a01, t_theta)
        a1 = dr.lerp(a10, a11, t_theta)
        return dr.lerp(a0, a1, t_phi)

    def bilerp_cond_mixing_weights(
        cond_mix_coeffs,
        b00_lo,
        b00_hi,
        b01_lo,
        b01_hi,
        b10_lo,
        b10_hi,
        b11_lo,
        b11_hi,
        t00,
        t01,
        t10,
        t11,
        kU,
    ):
        # row-lerp mixing coeffs within each corner slice using t** then bilerp
        # across four slices
        w00_lo = dr.gather(mi.Float, cond_mix_coeffs, b00_lo + kU)
        w00_hi = dr.gather(mi.Float, cond_mix_coeffs, b00_hi + kU)
        pi00 = dr.lerp(w00_lo, w00_hi, t00)

        w01_lo = dr.gather(mi.Float, cond_mix_coeffs, b01_lo + kU)
        w01_hi = dr.gather(mi.Float, cond_mix_coeffs, b01_hi + kU)
        pi01 = dr.lerp(w01_lo, w01_hi, t01)

        w10_lo = dr.gather(mi.Float, cond_mix_coeffs, b10_lo + kU)
        w10_hi = dr.gather(mi.Float, cond_mix_coeffs, b10_hi + kU)
        pi10 = dr.lerp(w10_lo, w10_hi, t10)

        w11_lo = dr.gather(mi.Float, cond_mix_coeffs, b11_lo + kU)
        w11_hi = dr.gather(mi.Float, cond_mix_coeffs, b11_hi + kU)
        pi11 = dr.lerp(w11_lo, w11_hi, t11)

        return bilerp(pi00, pi01, pi10, pi11)

    def bilerp_cond_metalog_coeffs(
        cond_metalog_coeffs,
        b00_lo,
        b00_hi,
        b01_lo,
        b01_hi,
        b10_lo,
        b10_hi,
        b11_lo,
        b11_hi,
        t00,
        t01,
        t10,
        t11,
        kU,
        tiU,
    ):
        r00_lo = (b00_lo + kU) * T
        r00_hi = (b00_hi + kU) * T
        a00 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r00_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r00_hi + tiU),
            t00,
        )

        r01_lo = (b01_lo + kU) * T
        r01_hi = (b01_hi + kU) * T
        a01 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r01_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r01_hi + tiU),
            t01,
        )

        r10_lo = (b10_lo + kU) * T
        r10_hi = (b10_hi + kU) * T
        a10 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r10_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r10_hi + tiU),
            t10,
        )

        r11_lo = (b11_lo + kU) * T
        r11_hi = (b11_hi + kU) * T
        a11 = dr.lerp(
            dr.gather(mi.Float, cond_metalog_coeffs, r11_lo + tiU),
            dr.gather(mi.Float, cond_metalog_coeffs, r11_hi + tiU),
            t11,
        )

        return bilerp(a00, a01, a10, a11)

    # per-sample initial guess for the hybrid Newton-Raphson
    inv_span = dr.rcp(dr.maximum(bounds[1] - bounds[0], EPSILON))
    y0 = dr.clip((sample2.y - bounds[0]) * inv_span, EPSILON, 1.0 - EPSILON)

    # [0, 1, ..., T-1] as mi.UInt
    tiU = tuple(mi.UInt(ti) for ti in range(T))

    """Compute exact mixture PDF/CDF at y (see Bishop 2023, Eq. 3.111)"""
    mix_pdf = dr.zeros(mi.Float, N)
    mix_cdf = dr.zeros(mi.Float, N)
    for k in range(K):
        kU = mi.UInt(k)

        # blend mixing weights between material A and B
        pi_k_A = bilerp_cond_mixing_weights(
            cond_mix_coeffs_A,
            base00_lo_A,
            base00_hi_A,
            base01_lo_A,
            base01_hi_A,
            base10_lo_A,
            base10_hi_A,
            base11_lo_A,
            base11_hi_A,
            t00_A,
            t01_A,
            t10_A,
            t11_A,
            kU,
        )
        pi_k_B = bilerp_cond_mixing_weights(
            cond_mix_coeffs_B,
            base00_lo_B,
            base00_hi_B,
            base01_lo_B,
            base01_hi_B,
            base10_lo_B,
            base10_hi_B,
            base11_lo_B,
            base11_hi_B,
            t00_B,
            t01_B,
            t10_B,
            t11_B,
            kU,
        )
        pi_k = dr.lerp(pi_k_A, pi_k_B, lambda_val)

        # blend metalog coeffs between material A and B
        a_k = []
        for ti in tiU:
            term_A = bilerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_A,
                base00_lo_A,
                base00_hi_A,
                base01_lo_A,
                base01_hi_A,
                base10_lo_A,
                base10_hi_A,
                base11_lo_A,
                base11_hi_A,
                t00_A,
                t01_A,
                t10_A,
                t11_A,
                kU,
                ti,
            )
            term_B = bilerp_cond_metalog_coeffs(
                cond_mix_metalog_coeffs_B,
                base00_lo_B,
                base00_hi_B,
                base01_lo_B,
                base01_hi_B,
                base10_lo_B,
                base10_hi_B,
                base11_lo_B,
                base11_hi_B,
                t00_B,
                t01_B,
                t10_B,
                t11_B,
                kU,
                ti,
            )
            a_k.append(dr.lerp(term_A, term_B, lambda_val))

        # evaluate the metalog PDF/CDF for the component at y
        u_k = eval_newton_raphson_for_metalog_cdf(
            a_k, sample2.y, y0, bounds, HYBRID_NEWTON_RAPHSON_MAX_ITERATIONS
        )
        My_k = eval_unbounded_metalog_quantile(a_k, u_k)
        pdf_k = eval_bounded_metalog_pdf(a_k, bounds[0], bounds[1], u_k, My_k)

        # accumulate for the mixture PDF/CDF (Eq. 3.111, Bishop 2023)
        mix_cdf += pi_k * u_k
        mix_pdf += pi_k * pdf_k

    mix_cdf = dr.clip(mix_cdf, 0.0, 1.0)

    return mix_cdf, mix_pdf


def eval_derivative_metalog_quantile(
    metalog_coeffs: mi.TensorXf, n_grid: int = 1024
):
    """
    Evaluates the derivative of metalog quantile function with respect to y.

    Supports arbitrary batched metalog coefficients (e.g., [S, K, T] or [S, R, K, T]).

    See also `eval_unbounded_metalog_pdf()` for similarities.
    See also Eq. 5/9 in The Metalog Distributions by Tom Keelin 2016.

    Args:
        metalog_coeffs (mi.TensorXf): Metalog coefficients of shape [..., T].
        n_grid (int): Number of grid points for cumulative probabilities in (0, 1).
    """
    from math import prod

    shape = metalog_coeffs.shape
    T = shape[-1]

    # flatten coefficients
    flat_coeffs = metalog_coeffs.array

    # calculate total number of independent distributions
    N = prod(shape[:-1])

    # compute indices
    idx = dr.arange(mi.UInt, N * n_grid)
    dist_idx = idx // n_grid  # distribution index from 0 to N - 1
    y_idx = idx % n_grid  # y grid index from 0 to n_grid - 1

    # create a grid of y values in [0.01, 0.99]
    y_vals = dr.linspace(mi.Float, 0.01, 0.99, n_grid)
    ys = dr.gather(mi.Float, y_vals, y_idx)

    # precompute terms
    one_min_ys = 1 - ys
    y_term = ys * one_min_ys
    y_term_rcp = dr.rcp(y_term)
    log_coeff = dr.log(ys * dr.rcp(one_min_ys))
    y_min_half = ys - 0.5

    def get_a(i: mi.Int) -> mi.Float:
        return dr.gather(mi.Float, flat_coeffs, dist_idx * T + i)

    # evaluate the base terms
    result = get_a(1) * y_term_rcp  # 2-term
    result += get_a(2) * dr.fma(y_min_half, y_term_rcp, log_coeff)  # 3-term

    # 4-term
    if T >= 4:
        result += get_a(3)

    # Add terms starting from the 5th term
    n = mi.UInt(5)
    while n < T + 1:
        n_min_1 = n - 1
        an = get_a(n_min_1)
        if n % 2 == 0:  # even terms
            n_int_div_2 = n // 2
            result = dr.fma(
                an,
                (
                    dr.fma(
                        dr.fma(n, 0.5, -1),
                        dr.power(y_min_half, n_int_div_2 - 2) * log_coeff,
                        dr.power(y_min_half, n_int_div_2 - 1) * y_term_rcp,
                    )
                ),
                result,
            )
        else:  # odd terms
            result = dr.fma(
                an * n_min_1 * 0.5, dr.power(y_min_half, (n - 3) // 2), result
            )

        n += 1

    return result


def eval_feasibility_penalty(
    metalog_coeffs: mi.TensorXf, n_grid: int = 1024, epsilon: float = 1e-3
) -> mi.Float:
    """
    Computes a differentiable soft penalty for metalog coefficients that approach the infeasibile region.

    See also Eq. 5 in The Metalog Distributions by Tom Keelin 2016.

    Args:
        metalog_coeffs (mi.TensorXf): Metalog coefficients of shape [S, K, T].
        n_grid (int): Number of grid points for cumulative probabilities in (0, 1).
        epsilon (float): The margin of safety for metalog feasibility. Denominators below this value incur a penalty.
    """
    denom = eval_derivative_metalog_quantile(metalog_coeffs, n_grid)

    """
    Evaluate the penalty: max(0, epsilon - denom)^2
    If denom > epsiolon, penalty is 0. Otherwise, penalty is positive.
    """
    violation = dr.maximum(0.0, epsilon - denom)
    sqr_penalty = dr.square(violation)

    # sum penalities across all grid points and distribution to return a scalar
    return dr.sum(sqr_penalty)
