# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import lib.metalogs as ml
from torch import Tensor
from abc import ABC, abstractmethod


class MetalogsStrategy(ABC):
    MAX_HYBRID_NEWTON_RAPHSON_ITERATIONS = 50

    def __init__(self, bounds: tuple | None = None):
        self.bounds = bounds

    @abstractmethod
    def eval_metalog_pdf(
        metalog_coeffs: Tensor,
        cdf: Tensor,
        M_y: Tensor,
    ) -> Tensor:
        pass

    @abstractmethod
    def eval_metalog_quantile(
        self, metalog_coeffs: Tensor, unif: Tensor
    ) -> Tensor:
        pass

    @abstractmethod
    def eval_metalog_cdf(self, metalog_coeffs: Tensor, X: Tensor) -> Tensor:
        pass

    @abstractmethod
    def eval_metalog_cdf_pdf(
        self, metalog_coeffs: Tensor, X: Tensor
    ) -> tuple[Tensor, Tensor]:
        pass


class TorchMetalogs(MetalogsStrategy):
    def eval_metalog_cdf(self, metalog_coeffs, X):
        return ml.eval_newton_raphson_for_metalog_cdf(
            metalog_coeffs,
            X,
            bounds=self.bounds,
            max_iter=self.MAX_HYBRID_NEWTON_RAPHSON_ITERATIONS,
        )

    def eval_metalog_cdf_pdf(
        self, metalog_coeffs: Tensor, X: Tensor
    ) -> tuple[Tensor, Tensor]:
        cdf = ml.eval_newton_raphson_for_metalog_cdf(
            metalog_coeffs,
            X,
            bounds=self.bounds,
            max_iter=self.MAX_HYBRID_NEWTON_RAPHSON_ITERATIONS,
        )
        M_y = ml.eval_unbounded_metalog_quantile(metalog_coeffs, cdf)
        pdf = self.eval_metalog_pdf(metalog_coeffs, cdf, M_y)

        return cdf, pdf


class TorchBoundedMetalogs(TorchMetalogs):
    def eval_metalog_pdf(
        self, metalog_coeffs: Tensor, cdf: Tensor, M_y: Tensor
    ) -> Tensor:
        return ml.eval_bounded_metalog_pdf(
            metalog_coeffs, self.bounds[0], self.bounds[1], cdf, M_y
        )

    def eval_metalog_quantile(
        self, metalog_coeffs: Tensor, unif: Tensor
    ) -> Tensor:
        return ml.eval_bounded_metalog_quantile(
            metalog_coeffs, self.bounds[0], self.bounds[1], unif
        )[0]


class TorchUnboundedMetalogs(TorchMetalogs):
    def eval_metalog_pdf(
        self, metalog_coeffs: Tensor, cdf: Tensor, M_y: Tensor
    ) -> Tensor:
        return ml.eval_unbounded_metalog_pdf(metalog_coeffs, cdf)

    def eval_metalog_quantile(
        self, metalog_coeffs: Tensor, unif: Tensor
    ) -> Tensor:
        return ml.eval_unbounded_metalog_quantile(metalog_coeffs, unif)
