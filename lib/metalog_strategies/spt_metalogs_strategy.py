# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import torch
from abc import ABC, abstractmethod


class SPTMetalogsStrategy(ABC):
    eps = 1e-9

    def eval_k_alpha(self, alpha: float) -> float:
        # k_α is defined in Proposition 2 (Keelin 2016, Eq. 18)
        return 0.5 * (1.0 - 1.66711 * (0.5 - alpha))

    @abstractmethod
    def eval_feasible_r(
        self,
        q_low: torch.Tensor,
        q_med: torch.Tensor,
        q_high: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_spt_metalog(
        self,
        q_low: torch.Tensor,
        q_med: torch.Tensor,
        q_high: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        pass


class TorchSPTMetalogs(SPTMetalogsStrategy):
    def eval_feasible_r(
        self,
        q_low: torch.Tensor,
        q_med: torch.Tensor,
        q_high: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        # r is defined in Eq. 17, and Proposition 2 SPT Metalog Feasibility (Keelin 2016)
        r = ((q_med - q_low) / (q_high - q_low + self.eps)).clamp(0.0, 1.0)
        k_alpha = self.eval_k_alpha(alpha)

        # ensure a feasible SPT parameterization by k_α < r < k_{1-α}
        margin = 0.001
        feasible_r = r.clamp(min=k_alpha + margin, max=(1.0 - k_alpha) - margin)

        return feasible_r


class TorchSPTBoundedMetalogs(TorchSPTMetalogs):
    bounds: tuple | None = None

    def __init__(self, bounds: tuple | None = None):
        self.bounds = bounds

    def eval_spt_metalog(
        self,
        q_low: torch.Tensor,
        q_med: torch.Tensor,
        q_high: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        # Keelin 2016, Proposition 4
        def gamma_alpha(q_alpha):
            return torch.log(
                (
                    (q_alpha - self.bounds[0]) / (self.bounds[1] - q_alpha)
                ).clamp_min(self.eps)
            )

        # evaluate gamma alpha
        gamma_low = gamma_alpha(q_low)
        gamma_med = gamma_alpha(q_med)
        gamma_high = gamma_alpha(q_high)

        # ensure a feasible SPT parameterization
        r = self.eval_feasible_r(gamma_low, gamma_med, gamma_high, alpha)

        # refers to the (1-α/α) term to get bounded SPT metalog coefficients
        L = torch.log(
            torch.tensor(
                (1.0 - alpha) / alpha, dtype=q_low.dtype, device=q_low.device
            )
        )

        # obtain SPT bounded metalog coefficients
        a1 = gamma_med
        a2 = 0.5 * ((gamma_high - gamma_low) / (L + self.eps))
        a3 = ((1.0 - 2.0 * r) * (gamma_high - gamma_low)) / (
            (1.0 - 2.0 * alpha) * (L + self.eps)
        )

        # ensure a2 > 0 (strictly increasing quantile function)
        return torch.column_stack([a1, a2.clamp_min(self.eps), a3])


class TorchSPTUnboundedMetalogs(TorchSPTMetalogs):
    def eval_spt_metalog(
        self,
        q_low: torch.Tensor,
        q_med: torch.Tensor,
        q_high: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        # Proposition 1 (SPT Metalog Constants), Eq. 16/17

        # ensure a feasible SPT parameterization
        r = self.eval_feasible_r(q_low, q_med, q_high, alpha)

        # refers to the (1-α/α) term to get unbounded SPT metalog coefficients
        L = torch.log(
            torch.tensor(
                (1.0 - alpha) / alpha, dtype=q_low.dtype, device=q_low.device
            )
        )

        # obtain SPT unbounded metalog coefficients
        q_range = (q_high - q_low).clamp_min(self.eps)
        a1 = q_med
        a2 = 0.5 * (q_range / (L + self.eps))
        a3 = ((1.0 - 2.0 * r) * q_range) / (
            (1.0 - 2.0 * alpha) * (L + self.eps)
        )

        # ensure a2 > 0 (strictly increasing quantile function)
        return torch.column_stack([a1, a2.clamp_min(self.eps), a3])
