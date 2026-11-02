# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from lib.metalog_strategies.metalogs_variant import MetalogsVariant
import drjit as dr
import lib.metalogs as ml
from lib.utils.sample_record import SampleRecord
import mitsuba as mi
import torch
from typing import Tuple
import warnings


class TorchMetalogs(MetalogsVariant):
    def __init__(self):
        super().__init__()
        warnings.warn(
            "Initializing MetalogsVariant::TorchMetalogs. Beware this class has not been maintained for a while."
        )

    def sample_metalog(
        self,
        sample2: mi.Point2f,
    ) -> Tuple[mi.Point2f, mi.Float]:
        sRec = ml.sample_bivariate_metalog(
            self.metalogs["marginal"],
            self.metalogs["marginal-quantiles"],
            self.metalogs["conditional"],
            sample2.torch().to(self.torch_device).T,
            self.metalogs["marginal-quantile-cdf-range"],
            self.marginal_bounds,
            self.conditional_bounds,
        )

        # convert sample to euclidean coordinates
        sRecSample = sRec.get("samples")
        if self.metalog_marginal == "phi":
            sRecSample = sRecSample.flip(1)
        w = mi.Vector3f(ml.cos_theta_phi_to_unitvec(sRecSample).T)
        w = dr.normalize(w)

        return w, mi.Float(sRec.get("pdf"))

    def sample_nd_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")

    def sample_nd_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_spt_coeffs,
        marginal_mix_coeffs,
        marginal_quantiles,
        conditional_metalogs,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")

    def sample_nd_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs,
        marginal_mix_coeffs,
        marginal_quantiles,
        cond_mix_metalog_coeffs,
        cond_mix_coeffs,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")
    
    def sample_nd_blended_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")

    def eval_metalog_pdf(self, w2D: mi.Vector2f) -> mi.Float:
        if self.metalog_marginal == "phi":
            w2D = w2D.flip(1)

        # evaluate metalog pdf
        pdf = ml.eval_bivariate_metalog_pdf(
            self.metalogs["marginal"],
            self.metalogs["marginal-quantiles"],
            self.metalogs["conditional"],
            SampleRecord(
                samples=w2D,
                unif2=None,
                pdf=None,
            ),
            self.metalogs["marginal-quantile-cdf-range"],
            bounds_marginal=self.marginal_bounds,
            bounds_conditional=self.conditional_bounds,
        )

        return mi.Float(pdf)

    def eval_nd_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_metalog_pdf() instead.")

    def eval_nd_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_metalog_pdf() instead.")
    
    def eval_nd_blended_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs_A: mi.TensorXf,
        marginal_mix_coeffs_A: mi.TensorXf,
        marginal_quantiles_A: mi.TensorXf,
        cond_mix_metalog_coeffs_A: mi.TensorXf,
        cond_mix_coeffs_A: mi.TensorXf,
        marginal_mix_metalog_coeffs_B: mi.TensorXf,
        marginal_mix_coeffs_B: mi.TensorXf,
        marginal_quantiles_B: mi.TensorXf,
        cond_mix_metalog_coeffs_B: mi.TensorXf,
        cond_mix_coeffs_B: mi.TensorXf,
        lambda_val: mi.Float,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_metalog_pdf() instead.")

    def eval_nd_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_metalog_pdf() instead.")

    def sample_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector3f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def sample_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector3f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def eval_mixture_metalog_pdf(
        self,
        w: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def eval_marginal_mixture_metalog_pdf(
        self,
        w: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def __str__(self):
        return "TorchMetalogs"


class TorchNDMetalogs(TorchMetalogs):
    def __init__(self):
        super().__init__()

    def sample_metalog(
        self,
        sample2: mi.Point2f,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")

    def sample_nd_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_spt_coeffs,
        marginal_mix_coeffs,
        marginal_quantiles,
        conditional_metalogs,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")

    def sample_nd_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs,
        marginal_mix_coeffs,
        marginal_quantiles,
        cond_mix_metalog_coeffs,
        cond_mix_coeffs,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")
    
    def sample_nd_blended_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")

    def sample_nd_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ) -> Tuple[mi.Vector2f, mi.Float]:

        wi_marginal_metalogs, wi_conditional_metalogs, wi_marginal_quantiles = (
            ml.find_metalogs(
                wi2D.torch().T,
                self.metalogs_keys,
                marginal_metalogs,
                marginal_quantiles,
                conditional_metalogs,
            )
        )

        sample2_torch = sample2.torch().to(self.torch_device).T
        sRec = ml.sample_bivariate_metalog(
            wi_marginal_metalogs,
            wi_marginal_quantiles,
            wi_conditional_metalogs,
            sample2_torch,
            self.marginal_quantile_cdf_range,
            self.marginal_bounds,
            self.conditional_bounds,
        )

        warped_sample2 = mi.Point2f(sRec.get("samples").T)

        # flip x and y if the metalog marginal is defined as phi (azimuth)
        if self.metalog_marginal == "phi":
            warped_sample2 = mi.Point2f(warped_sample2.y, warped_sample2.x)

        return warped_sample2, mi.Float(sRec.get("pdf"))

    def eval_metalog_pdf(self, w: mi.Vector2f) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_nd_metalog_pdf() instead.")

    def eval_nd_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ) -> tuple[mi.Point2f, mi.Float]:
        wi_marginal_metalogs, wi_conditional_metalogs, wi_marginal_quantiles = (
            ml.find_metalogs(
                wi2D.torch().T,
                self.metalogs_keys,
                marginal_metalogs,
                marginal_quantiles,
                conditional_metalogs,
            )
        )

        # flip x and y if the metalog marginal is defined as phi (azimuth)
        if self.metalog_marginal == "phi":
            w2D = mi.Point2f(w2D.y, w2D.x)

        unif2, pdf = ml.eval_bivariate_metalog_cdf_pdf(
            wi_marginal_metalogs,
            wi_marginal_quantiles,
            wi_conditional_metalogs,
            SampleRecord(
                samples=w2D.torch().T,
                unif2=None,
                pdf=None,
            ),
            self.marginal_quantile_cdf_range,
            bounds_marginal=self.marginal_bounds,
            bounds_conditional=self.conditional_bounds,
        )

        return mi.Point2f(unif2.T), mi.Float(pdf)

    def eval_nd_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_nd_metalog_pdf() instead.")

    def eval_nd_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_nd_metalog_pdf() instead.")
    
    def eval_nd_blended_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs_A: mi.TensorXf,
        marginal_mix_coeffs_A: mi.TensorXf,
        marginal_quantiles_A: mi.TensorXf,
        cond_mix_metalog_coeffs_A: mi.TensorXf,
        cond_mix_coeffs_A: mi.TensorXf,
        marginal_mix_metalog_coeffs_B: mi.TensorXf,
        marginal_mix_coeffs_B: mi.TensorXf,
        marginal_quantiles_B: mi.TensorXf,
        cond_mix_metalog_coeffs_B: mi.TensorXf,
        cond_mix_coeffs_B: mi.TensorXf,
        lambda_val: mi.Float,
    ) -> tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use eval_nd_metalog_pdf() instead.")

    def sample_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector3f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def sample_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector3f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def eval_mixture_metalog_pdf(
        self,
        w: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def eval_marginal_mixture_metalog_pdf(
        self,
        w: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Not supported. Please use DrJit instead.")

    def __str__(self):
        return "TorchNDMetalogs"
