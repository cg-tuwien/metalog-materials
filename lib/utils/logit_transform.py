# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import torch
from enum import Enum
from typing import TypedDict
import logging

EPSILON = 1e-6


class TransformDimension(Enum):
    MARGINAL = 1
    CONDITIONAL = 2


class LogitTransform(TypedDict):
    std: torch.Tensor
    mean: torch.Tensor
    dimension: TransformDimension
    range_min: float
    range_max: float


@torch.no_grad
def logit_transform_bivariate_samples(
    marginal_bounds,
    conditional_bounds,
    marginal_samples,
    conditional_samples,
):
    logging.info(f"Applying logit transform.")

    # logit transform marginal samples
    marginal_lower_bound = (
        marginal_samples.min().item() - EPSILON
        if marginal_bounds is None
        else marginal_bounds[0]
    )
    marginal_upper_bound = (
        marginal_samples.max().item() + EPSILON
        if marginal_bounds is None
        else marginal_bounds[1]
    )
    logit_transform_marginal, marginal_samples = logit_transform_samples(
        marginal_samples,
        marginal_lower_bound,
        marginal_upper_bound,
        transform_dimension=TransformDimension.MARGINAL,
    )

    # logit transform conditional samples
    cond_lower_bound = (
        conditional_samples.min().item() - EPSILON
        if conditional_bounds is None
        else conditional_bounds[0]
    )
    cond_upper_bound = (
        conditional_samples.max().item() + EPSILON
        if conditional_bounds is None
        else conditional_bounds[1]
    )
    logit_transform_cond, conditional_samples = logit_transform_samples(
        conditional_samples,
        cond_lower_bound,
        cond_upper_bound,
        transform_dimension=TransformDimension.CONDITIONAL,
    )

    return (
        marginal_samples,
        conditional_samples,
        logit_transform_marginal,
        logit_transform_cond,
    )


@torch.no_grad
def logit_transform_samples(
    samples: torch.Tensor,
    range_min: float | None = None,
    range_max: float | None = None,
    logit_transform: LogitTransform | None = None,
    transform_dimension: TransformDimension | None = None,
    EPSILON: float = 1e-6,
) -> tuple[LogitTransform, torch.Tensor]:
    """Transforms the provided samples using logit.

    Args:
        - samples (torch.Tensor): Samples to be transformed.
        - range_min (float | None, optional): Minimum value that a sample can take. If `logit_transform` provided, then
          uses `range_min` value of the transform. Defaults to None.
        - range_max (float | None, optional): Maximum value that a sample can take. If `logit_transform` provided, then
          uses `range_max` value of the transform. Defaults to None.
        - logit_transform (LogitTransform | None, optional): If provided, then uses LogitTransform for the
          standardization. Defaults to None.
        - transform_dimension (TransformDimension | None, optional): If provided, and assuming `logit_transform` is
          None, then initializes `logit_transform` with the provided dimension. Otherwise not used. Defaults to None.
        - EPSILON (float): Epsilon to avoid division by 0 or NaN values. Defaults to 1e-6.

    Returns:
        - tuple[LogitTransform, torch.Tensor]: A tuple of LogitTransform and transformed samples. If logit_transform
          provided then returns back the same LogitTransform instance in the first entry.
    """

    # get range_min and range_max if logit_transform exist
    logit_exist = logit_transform is not None
    if logit_exist:
        range_min = logit_transform.get("range_min")
        range_max = logit_transform.get("range_max")

    # check range, normalize to [0, 1] range if necessary
    if range_min < 0 or range_max > 1:
        samples = (samples - range_min) / (range_max - range_min)

    # clip to secure range [EPSILON, 1 - EPSILON]
    samples = torch.clamp(samples, min=EPSILON, max=1 - EPSILON)

    # logit transformation
    logit_transformed_samples = torch.log(samples / (1 - samples))

    # create logit transform if necessary
    if not logit_exist:
        logit_transform = LogitTransform(
            std=torch.std(logit_transformed_samples),
            mean=torch.mean(logit_transformed_samples),
            dimension=transform_dimension,
            range_min=range_min,
            range_max=range_max,
        )

    # standardization
    standardized_logit_transformed_samples = (
        logit_transformed_samples - logit_transform.get("mean")
    ) / logit_transform.get("std")

    return logit_transform, standardized_logit_transformed_samples


@torch.no_grad
def inverse_logit_samples(
    logit_transformed_samples: torch.Tensor, logit_transform: LogitTransform
) -> torch.Tensor:
    """
    Function that reverses the standardization and logit transformation to convert logit-transformed samples back to the
    original domain.

    Args:
        - logit_transformed_samples (torch.Tensor): Samples in logit transformed domain.
        - logit_transform (LogitTransform): Logit transform instance to reverse and rescale the samples.

    Returns:
        - torch.Tensor: Samples transformed back to the original domain.
    """
    # reverse standardization
    restandardized_samples = logit_transformed_samples * logit_transform.get(
        "std"
    ) + logit_transform.get("mean")

    # reverse Logit transformation
    result = 1 / (1 + torch.exp(-restandardized_samples))

    # rescale back to the original range, if necessary
    range_min = logit_transform.get("range_min")
    range_max = logit_transform.get("range_max")
    if range_min < 0 or range_max > 1:
        result = result * (range_max - range_min) + range_min

    return result


def inverse_logit_pdf(
    samples: torch.Tensor,
    pdf_values: torch.Tensor,
    logit_transform: LogitTransform,
    EPSILON: float = 1e-6,
) -> torch.Tensor:
    """
    Function that computes the PDF in the original domain from the PDF values computed in the standardized logit domain.

    Args:
        - samples (torch.Tensor): Samples in the original domain.
        - pdf_values (torch.Tensor): PDF values computed in the standardized logit domain.
        - logit_transform (LogitTransform): Logit transform instance to (re)normalize and standardization.
        - EPSILON (float): Epsilon to avoid division by 0 or NaN values. Defaults to 1e-6.

    Returns:
        - torch.Tensor: PDF values in the original domain.
    """
    range_min = logit_transform.get("range_min")
    range_max = logit_transform.get("range_max")
    orig_std = logit_transform.get("std")

    # normalize samples to [0, 1] range, if necessary
    min_samples, max_samples = samples.min().item(), samples.max().item()
    if min_samples < 0 or max_samples > 1:
        samples = (samples - range_min) / (range_max - range_min)

    # clamp the samples to avoid numerical issues with the logit function
    samples_clamped = torch.clamp(samples, min=EPSILON, max=1 - EPSILON)

    # jacobian for logit transformation: d(logit(x))/dx = 1/(x*(1-x))
    jacobian_logit = 1 / (samples_clamped * (1 - samples_clamped))

    # jacobian for standardization
    jacobian_standardization = 1 / orig_std

    # jacobian for normalization from [range_min, range_max] to [0, 1]
    jacobian_normalization = 1 / (range_max - range_min)

    # combined jacobian
    jacobian = (
        jacobian_logit * jacobian_standardization * jacobian_normalization
    )

    # transform the PDF values using the Jacobian
    transformed_pdfs = pdf_values * torch.abs(jacobian)

    return transformed_pdfs


@torch.no_grad
def logit_transform_bounds(bounds):
    bounds = torch.tensor(bounds)
    bounds = torch.clamp(bounds, EPSILON, 1 - EPSILON)
    bounds = torch.log(bounds / (1 - bounds))
    return tuple(bounds.tolist())
