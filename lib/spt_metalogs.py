# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import torch
from lib.metalog_strategies.metalog_strategy import (
    TorchBoundedMetalogs,
    MetalogsStrategy,
    TorchUnboundedMetalogs,
)
from lib.metalog_strategies.spt_metalogs_strategy import (
    SPTMetalogsStrategy,
    TorchSPTBoundedMetalogs,
    TorchSPTUnboundedMetalogs,
)


class SPTMetalogs:
    bounds: tuple[float, float] | None
    spt_metalogs_strategy: SPTMetalogsStrategy
    metalog_strategy: MetalogsStrategy
    metalog_coeffs: torch.Tensor
    alpha: float = 0.3
    quantiles: torch.Tensor
    spt_probs: torch.Tensor

    def __init__(
        self, bounds: tuple[float, float] | None = None, alpha: float = 0.3
    ):
        self.bounds = bounds
        self.alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)
        self.spt_probs = torch.tensor([self.alpha, 0.5, 1 - self.alpha])

        if self.bounds == None:
            self.metalog_strategy = TorchUnboundedMetalogs()
            self.spt_metalogs_strategy = TorchSPTUnboundedMetalogs()
        else:
            self.metalog_strategy = TorchBoundedMetalogs(self.bounds)
            self.spt_metalogs_strategy = TorchSPTBoundedMetalogs(self.bounds)

    def update_spt_metalogs(self, quantiles: torch.Tensor):
        self.quantiles = quantiles
        a = self.spt_metalogs_strategy.eval_spt_metalog(
            quantiles[0], quantiles[1], quantiles[2], self.alpha
        )
        self.metalog_coeffs = a

    def eval_metalog_cdf_pdf(self, X: torch.Tensor):
        return self.metalog_strategy.eval_metalog_cdf_pdf(
            self.metalog_coeffs, X
        )

    def eval_metalog_quantile(self, unif: torch.tensor) -> torch.Tensor:
        return self.metalog_strategy.eval_metalog_quantile(
            self.metalog_coeffs, unif
        )
