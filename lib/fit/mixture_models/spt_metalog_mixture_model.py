# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import torch
from lib.fit.mixture_models.metalog_mixture_model import MetalogMixtureModel1D
from lib.metalog_strategies.metalog_strategy import (
    TorchBoundedMetalogs,
    TorchUnboundedMetalogs,
)
from lib.metalog_strategies.spt_metalogs_strategy import (
    SPTMetalogsStrategy,
    TorchSPTBoundedMetalogs,
    TorchSPTUnboundedMetalogs,
)


class SPTMetalogMixtureModel(MetalogMixtureModel1D):
    """
    EM with an SPT (symmetric-percentile triplet) M-step for 3-term SPT metalogs.

    - Unbounded: Proposition 1 (a1,a2,a3 from q_α, q_0.5, q_{1-α})
    - Logit (two bounds): Proposition 4 (apply formulas in ln((x-bl)/(bu-x)) space)
    - Feasibility: Proposition 2 (project r into (k_α, 1-k_α)).
    """

    spt_metalogs_strategy: SPTMetalogsStrategy
    alpha: float = 0.3
    spt_probs: torch.Tensor  # shape: (3, )

    def __init__(
        self,
        n_components=2,
        max_iters=100,
        tol=1e-6,
        bounds: tuple[float, float] | None = None,
        alpha: float = 0.3,
        dirichlet_prior: float = 1.0,
        device: str | None = None,
    ):
        super().__init__(
            n_components,
            max_iters,
            tol,
            lambda_=0,
            dirichlet_prior=dirichlet_prior,
            device=device,
        )

        self.bounds = bounds
        self.alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)
        self.spt_probs = torch.tensor(
            [self.alpha, 0.5, 1 - self.alpha], device=self.device
        )

        if self.bounds == None:
            self.metalog_strategy = TorchUnboundedMetalogs()
            self.spt_metalogs_strategy = TorchSPTUnboundedMetalogs()
        else:
            self.metalog_strategy = TorchBoundedMetalogs(self.bounds)
            self.spt_metalogs_strategy = TorchSPTBoundedMetalogs(self.bounds)

        # initialize SPT metalog coefficients
        self.metalog_term = 3
        self.metalog_coeffs = torch.empty(
            (self.K, self.metalog_term), device=self.device
        )

    def initialize_parameters_concrete(self, W_sorted: torch.Tensor):
        # compute weighted quantiles at (α, 0.5, 1-α), shape: (K, 3)
        Q = self.weighted_quantiles(W_sorted, self.spt_probs)

        # tiny jitter to keep components distinct, shape: (K, 3)
        Q = self.jitter_quantiles(Q, self.bounds)

        # map quantiles to new metalog coefficients
        self.metalog_coeffs = self.spt_metalogs_strategy.eval_spt_metalog(
            Q[:, 0], Q[:, 1], Q[:, 2], self.alpha
        )

    def jitter_quantiles(
        self, Q: torch.Tensor, bounds: tuple | None
    ) -> torch.Tensor:
        """
        Row-wise jitter and clamp
        """
        K = Q.shape[0]
        span = (Q[:, 2] - Q[:, 0]).clamp_min(self.eps).unsqueeze(1)  # (K, 1)

        # initialize jitter, shape (K, 1)
        jitter = span * torch.tensor(
            [-self.eps, 0.0, self.eps], device=self.device
        ).unsqueeze(0)

        rnd = torch.rand(K, 1, device=self.device) * 2.0 - 1.0
        Q = Q + jitter * rnd

        if bounds != None:
            Q = Q.clamp(min=bounds[0] + self.eps, max=bounds[1] - self.eps)

        return torch.sort(Q, dim=1).values

    def maximization_step_concrete(self, W_sorted_T: torch.Tensor):
        """
        Compute weighted quantiles and solve for new metalog coeffiecients using Proposition 1 (SPT Metalog Constants),
        Eq. 16/17.
        """

        # compute weighted quantiles for all components at (α, 0.5, 1-α)
        Q = self.weighted_quantiles(W_sorted_T, self.spt_probs)  # (K, 3)

        # map quantiles to new metalog coefficients
        self.metalog_coeffs = self.spt_metalogs_strategy.eval_spt_metalog(
            Q[:, 0], Q[:, 1], Q[:, 2], self.alpha
        )
