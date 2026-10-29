# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import lib.metalogs_drjit as dr_ml
from lib.metalog_strategies.metalogs_variant import MetalogsVariant
import drjit as dr
import mitsuba as mi
import torch
from typing import Tuple


class DrMetalogs(MetalogsVariant):
    @dr.syntax
    def sample_metalog(
        self,
        sample2: mi.Point2f,
    ) -> Tuple[mi.Point2f, mi.Float]:
        sRec = dr_ml.sample_bounded_bivariate_metalog(
            self.metalogs["marginal"],
            self.metalogs["marginal-quantiles"],
            self.metalogs["conditional"],
            sample2,
            self.marginal_bounds,
            self.conditional_bounds,
            self.metalogs["marginal-quantile-cdf-range"],
        )

        return sRec.get("samples"), sRec.get("pdf")

    @dr.syntax
    def sample_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        # sample the marginal (mixture) metalog
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            sample2.x,
            self.marginal_bounds,
        )

        # sample conditional by interpolating within slice quantile rows
        y, pdf_cond = dr_ml.sample_bounded_conditional_metalogs(
            marginal_quantiles,
            cond_metalogs,
            sample2,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            x,
        )

        # assemble samples and evaluate the joint pdf
        w2D = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return w2D, pdf

    @dr.syntax
    def sample_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        # sample the marginal (mixture) metalog
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            sample2.x,
            self.marginal_bounds,
        )

        # sample conditional by interpolating within slice quantile rows
        y, pdf_cond = dr_ml.sample_bounded_conditional_mixture_metalog(
            cond_mix_coeffs,
            cond_mix_metalog_coeffs,
            marginal_quantiles,
            self.marginal_quantile_cdf_range,
            sample2,
            self.conditional_bounds,
            x,
        )

        # assemble samples and evaluate the joint pdf
        w2D = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return w2D, pdf

    @dr.syntax
    def eval_metalog_pdf(self, w2D: mi.Point2f) -> Tuple[mi.Vector2f, mi.Float]:
        # evaluate metalog pdf
        unif2, pdf = dr_ml.eval_bivariate_metalog_cdf_pdf(
            self.metalogs["marginal"],
            self.metalogs["marginal-quantiles"],
            self.metalogs["conditional"],
            w2D,
            self.metalogs["marginal-quantile-cdf-range"],
            bounds_marginal=self.marginal_bounds,
            bounds_conditional=self.conditional_bounds,
        )

        return unif2, pdf

    @dr.syntax
    def eval_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # evaluate the marginal (mixture) PDF/CDF
        unif_x, pdf_marginal = dr_ml.eval_bounded_mixture_metalog_cdf_pdf(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            w2D.x,
            self.marginal_bounds,
        )

        # evaluate conditional (mixture) PDF/CDF lerping between the two slices
        unif_y, pdf_cond = dr_ml.eval_bounded_conditional_metalog_cdf_pdf(
            marginal_quantiles,
            cond_metalogs,
            w2D,
            self.marginal_quantile_cdf_range,
            self.conditional_bounds,
            unif_x,
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    @dr.syntax
    def eval_mixture_metalog_pdf(
        self,
        w2D: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # evaluate the marginal (mixture) PDF/CDF
        unif_x, pdf_marginal = dr_ml.eval_bounded_mixture_metalog_cdf_pdf(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            w2D.x,
            self.marginal_bounds,
        )

        # evaluate conditional (mixture) PDF/CDF lerping between the two slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_mixture_metalog_cdf_pdf(
                marginal_quantiles,
                cond_mix_coeffs,
                cond_mix_metalog_coeffs,
                w2D,
                self.conditional_bounds,
                self.marginal_quantile_cdf_range,
                unif_x,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

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

    def init_metalogs(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        metalogs = super().init_metalogs(
            metalogs_filename,
            metalog_term,
            marginal_bounds,
            conditional_bounds,
            metalog_marginal,
            torch_device,
        )

        # convert to Dr types
        self.metalogs["marginal"] = mi.TensorXf(metalogs["marginal"])
        self.metalogs["conditional"] = mi.TensorXf(metalogs["conditional"])
        self.metalogs["marginal-quantiles"] = mi.Float(
            metalogs["marginal-quantiles"]
        )

        return self.metalogs

    def init_marginal_mixture_metalogs(
        self,
        metalogs_filename: str,
        cond_metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_metalogs,
        ) = super().init_marginal_mixture_metalogs(
            metalogs_filename,
            cond_metalog_term,
            marginal_bounds,
            conditional_bounds,
            metalog_marginal,
            torch_device,
        )

        # convert to Dr types
        marginal_quantiles = mi.Float(marginal_quantiles)
        marg_mix_metalog_coeffs = mi.TensorXf(marg_mix_metalog_coeffs)
        marg_mix_weights = mi.TensorXf(marg_mix_weights)
        cond_metalogs = mi.TensorXf(cond_metalogs)

        return (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_metalogs,
        )

    def init_mixture_metalogs(
        self,
        metalogs_filename: str,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_mix_weights,
            cond_mix_metalog_coeffs,
        ) = super().init_mixture_metalogs(
            metalogs_filename,
            marginal_bounds,
            conditional_bounds,
            metalog_marginal,
            torch_device,
        )

        # convert to Dr types
        marginal_quantiles = mi.Float(marginal_quantiles)
        marg_mix_metalog_coeffs = mi.TensorXf(marg_mix_metalog_coeffs)
        marg_mix_weights = mi.TensorXf(marg_mix_weights)
        cond_mix_weights = mi.TensorXf(cond_mix_weights)
        cond_mix_metalog_coeffs = mi.TensorXf(cond_mix_metalog_coeffs)

        return (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_mix_weights,
            cond_mix_metalog_coeffs,
        )

    def init_metalog_slices(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        raise NotImplementedError("Use init_metalogs() instead.")

    def init_marginal_mixture_conditional_slices(
        self,
        metalogs_filename: str,
        cond_metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        raise NotImplementedError("Use init_metalogs() instead.")

    def sample_nd_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")

    @dr.syntax
    def sample_nd_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_spt_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")

    def sample_nd_mixture_metalog(
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

    def sample_nd_blended_mixture_metalog(
        self,
        sample2: mi.Point2f,
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
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use sample_metalog() instead.")

    @dr.syntax
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

    def __str__(self):
        return "DrMetalogs"


class DrNDMetalogs(MetalogsVariant):
    def __init__(self):
        super().__init__()

    def sample_metalog(
        self,
        sample2: mi.Point2f,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")

    @dr.syntax
    def sample_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")

    @dr.syntax
    def sample_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        raise NotImplementedError("Use sample_nd_metalog() instead.")

    @dr.syntax
    def sample_nd_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        if self.is_isotropic:
            samples, pdf = self.sample_3D_metalogs(
                sample2,
                wi2D,
                marginal_metalogs,
                marginal_quantiles,
                conditional_metalogs,
            )
        else:
            samples, pdf = self.sample_4D_metalogs(
                sample2,
                wi2D,
                marginal_metalogs,
                marginal_quantiles,
                conditional_metalogs,
            )

        return samples, pdf

    @dr.syntax
    def sample_nd_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        if self.is_isotropic:
            samples, pdf = self.sample_3D_marginal_mixture_metalog(
                sample2,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                conditional_metalogs,
            )
        else:
            samples, pdf = self.sample_4D_marginal_mixture_metalog(
                sample2,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                conditional_metalogs,
            )

        return samples, pdf

    @dr.syntax
    def sample_nd_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        if self.is_isotropic:
            samples, pdf = self.sample_3D_mixture_metalog(
                sample2,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                cond_mix_metalog_coeffs,
                cond_mix_coeffs,
            )
        else:
            samples, pdf = self.sample_4D_mixture_metalog(
                sample2,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                cond_mix_metalog_coeffs,
                cond_mix_coeffs,
            )

        return samples, pdf

    @dr.syntax
    def sample_nd_blended_mixture_metalog(
        self,
        sample2: mi.Point2f,
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
    ) -> Tuple[mi.Vector2f, mi.Float]:
        if self.is_isotropic:
            samples, pdf = self.sample_3D_blended_mixture_metalog(
                sample2,
                wi2D,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_A,
                marginal_quantiles_A,
                cond_mix_metalog_coeffs_A,
                cond_mix_coeffs_A,
                marginal_mix_metalog_coeffs_B,
                marginal_mix_coeffs_B,
                marginal_quantiles_B,
                cond_mix_metalog_coeffs_B,
                cond_mix_coeffs_B,
                lambda_val,
            )
        else:
            samples, pdf = self.sample_4D_blended_mixture_metalog(
                sample2,
                wi2D,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_A,
                marginal_quantiles_A,
                cond_mix_metalog_coeffs_A,
                cond_mix_coeffs_A,
                marginal_mix_metalog_coeffs_B,
                marginal_mix_coeffs_B,
                marginal_quantiles_B,
                cond_mix_metalog_coeffs_B,
                cond_mix_coeffs_B,
                lambda_val,
            )

        return samples, pdf

    @dr.syntax
    def sample_3D_blended_mixture_metalog(
        self,
        sample2: mi.Point2f,
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
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi, find the two closest index in terms of euclidian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        # sample the marginal (mixture) by lerping between the two slices
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog_slice_lerp_blend(
            marginal_mix_coeffs_A,
            marginal_mix_metalog_coeffs_A,
            marginal_mix_coeffs_B,
            marginal_mix_metalog_coeffs_B,
            sample2.x,
            idx0,
            idx1,
            lambda_val,
            t,
            self.marginal_bounds,
        )

        # sample conditional by lerping slices and within slice quantile rows
        y, pdf_cond = (
            dr_ml.sample_bounded_conditional_mixture_metalog_slice_lerp_blend(
                cond_mix_coeffs_A,
                cond_mix_metalog_coeffs_A,
                marginal_quantiles_A,
                cond_mix_coeffs_B,
                cond_mix_metalog_coeffs_B,
                marginal_quantiles_B,
                sample2,
                idx0,
                idx1,
                lambda_val,
                t,
                self.conditional_bounds,
                x,
            )
        )

        # assemble samples and evaluate the joint pdf
        samples = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return samples, pdf

    @dr.syntax
    def sample_4D_blended_mixture_metalog(
        self,
        sample2: mi.Point2f,
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
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        # sample the marginal (mixture) by lerping between the two slices
        x, pdf_marginal = (
            dr_ml.sample_bounded_mixture_metalog_slice_bilerp_blend(
                marginal_mix_coeffs_A,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_B,
                marginal_mix_metalog_coeffs_B,
                sample2.x,
                idx00,
                idx01,
                idx10,
                idx11,
                lambda_val,
                t_theta,
                t_phi,
                self.marginal_bounds,
            )
        )

        # sample conditional by lerping slices and within slice quantile rows
        y, pdf_cond = (
            dr_ml.sample_bounded_conditional_mixture_metalog_slice_bilerp_blend(
                cond_mix_coeffs_A,
                cond_mix_metalog_coeffs_A,
                marginal_quantiles_A,
                cond_mix_coeffs_B,
                cond_mix_metalog_coeffs_B,
                marginal_quantiles_B,
                sample2,
                idx00,
                idx01,
                idx10,
                idx11,
                lambda_val,
                t_theta,
                t_phi,
                self.conditional_bounds,
                x,
            )
        )

        # assemble samples and evaluate the joint pdf
        samples = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return samples, pdf

    @dr.syntax
    def sample_3D_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi, find the two closest index in terms of euclidian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        # sample the marginal (mixture) by lerping between the two slices
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog_slice_lerp(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            sample2.x,
            idx0,
            idx1,
            t,
            self.marginal_bounds,
        )

        # sample conditional by lerping slices and within slice quantile rows
        y, pdf_cond = (
            dr_ml.sample_bounded_conditional_mixture_metalog_slice_lerp(
                cond_mix_coeffs,
                cond_mix_metalog_coeffs,
                marginal_quantiles,
                self.marginal_quantile_cdf_range,
                sample2,
                idx0,
                idx1,
                t,
                self.conditional_bounds,
                x,
            )
        )

        # assemble samples and evaluate the joint pdf
        samples = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return samples, pdf

    @dr.syntax
    def sample_4D_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        # sample the marginal (mixture) by lerping between the two slices
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog_slice_bilerp(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            sample2.x,
            idx00,
            idx01,
            idx10,
            idx11,
            t_theta,
            t_phi,
            self.marginal_bounds,
        )

        # sample conditional by lerping slices and within slice quantile rows
        y, pdf_cond = (
            dr_ml.sample_bounded_conditional_mixture_metalog_slice_bilerp(
                cond_mix_coeffs,
                cond_mix_metalog_coeffs,
                marginal_quantiles,
                self.marginal_quantile_cdf_range,
                sample2,
                idx00,
                idx01,
                idx10,
                idx11,
                t_theta,
                t_phi,
                self.conditional_bounds,
                x,
            )
        )

        # assemble samples and evaluate the joint pdf
        samples = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return samples, pdf

    @dr.syntax
    def sample_3D_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi, find the two closest index in terms of euclidian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        # sample the marginal (mixture) by lerping between the two slices
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog_slice_lerp(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            sample2.x,
            idx0,
            idx1,
            t,
            self.marginal_bounds,
        )

        # sample conditional by lerping slices and within slice quantile rows
        y, pdf_cond = dr_ml.sample_bounded_conditional_metalog_slice_lerp(
            marginal_quantiles,
            conditional_metalogs,
            sample2,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            idx0,
            idx1,
            t,
            x,
        )

        # assemble samples and evaluate the joint pdf
        samples = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return samples, pdf

    @dr.syntax
    def sample_4D_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        # sample the marginal (mixture) by bilerping between the four slices
        x, pdf_marginal = dr_ml.sample_bounded_mixture_metalog_slice_bilerp(
            marginal_mix_coeffs,
            marginal_mix_metalog_coeffs,
            sample2.x,
            idx00,
            idx01,
            idx10,
            idx11,
            t_theta,
            t_phi,
            self.marginal_bounds,
        )

        # sample conditional by bilerping slices and within slice quantile rows
        y, pdf_cond = dr_ml.sample_bounded_conditional_metalog_slice_bilerp(
            marginal_quantiles,
            conditional_metalogs,
            sample2,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            idx00,
            idx01,
            idx10,
            idx11,
            t_theta,
            t_phi,
            x,
        )

        # assemble samples and evaluate the joint pdf
        w2D = mi.Point2f([x, y])
        pdf = pdf_marginal * pdf_cond

        return w2D, pdf

    def sample_4D_metalogs(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ):
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        sRec = dr_ml.sample_bounded_bivariate_metalog_slice_bilerp(
            marginal_metalogs,
            marginal_quantiles,
            conditional_metalogs,
            sample2,
            self.marginal_bounds,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            idx00,
            idx01,
            idx10,
            idx11,
            t_theta,
            t_phi,
        )

        return sRec.get("samples"), sRec.get("pdf")

    def sample_3D_metalogs(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ):
        # for each wi, find the two closest index in terms of euclidian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        sRec = dr_ml.sample_bounded_bivariate_metalog_slice_lerp(
            marginal_metalogs,
            marginal_quantiles,
            conditional_metalogs,
            sample2,
            self.marginal_bounds,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            idx0,
            idx1,
            t,
        )

        return sRec.get("samples"), sRec.get("pdf")

    def eval_metalog_pdf(self, w: mi.Point2f) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use eval_pdf_nd_metalog() instead.")

    @dr.syntax
    def eval_nd_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ) -> tuple[mi.Point2f, mi.Float]:
        if self.is_isotropic:
            unif2, pdf = self.eval_3D_metalog_pdf(
                w2D,
                wi2D,
                marginal_metalogs,
                marginal_quantiles,
                conditional_metalogs,
            )
        else:
            unif2, pdf = self.eval_4D_metalog_pdf(
                w2D,
                wi2D,
                marginal_metalogs,
                marginal_quantiles,
                conditional_metalogs,
            )

        return unif2, pdf

    @dr.syntax
    def eval_mixture_metalog_pdf(
        self,
        w: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use eval_pdf_nd_metalog() instead.")

    @dr.syntax
    def eval_marginal_mixture_metalog_pdf(
        self,
        w: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        raise NotImplementedError("Use eval_pdf_nd_metalog() instead.")

    @dr.syntax
    def eval_nd_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> tuple[mi.Point2f, mi.Float]:
        if self.is_isotropic:
            unif2, pdf = self.eval_3D_marginal_mixture_metalog_pdf(
                w2D,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                conditional_metalogs,
            )
        else:
            unif2, pdf = self.eval_4D_marginal_mixture_metalog_pdf(
                w2D,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                conditional_metalogs,
            )

        return unif2, pdf

    @dr.syntax
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
        if self.is_isotropic:
            unif2, pdf = self.eval_3D_mixture_metalog_pdf(
                w2D,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                cond_mix_metalog_coeffs,
                cond_mix_coeffs,
            )
        else:
            unif2, pdf = self.eval_4D_mixture_metalog_pdf(
                w2D,
                wi2D,
                marginal_mix_metalog_coeffs,
                marginal_mix_coeffs,
                marginal_quantiles,
                cond_mix_metalog_coeffs,
                cond_mix_coeffs,
            )

        return unif2, pdf

    @dr.syntax
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
        if self.is_isotropic:
            unif2, pdf = self.eval_3D_blended_mixture_metalog_pdf(
                w2D,
                wi2D,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_A,
                marginal_quantiles_A,
                cond_mix_metalog_coeffs_A,
                cond_mix_coeffs_A,
                marginal_mix_metalog_coeffs_B,
                marginal_mix_coeffs_B,
                marginal_quantiles_B,
                cond_mix_metalog_coeffs_B,
                cond_mix_coeffs_B,
                lambda_val,
            )
        else:
            unif2, pdf = self.eval_4D_blended_mixture_metalog_pdf(
                w2D,
                wi2D,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_A,
                marginal_quantiles_A,
                cond_mix_metalog_coeffs_A,
                cond_mix_coeffs_A,
                marginal_mix_metalog_coeffs_B,
                marginal_mix_coeffs_B,
                marginal_quantiles_B,
                cond_mix_metalog_coeffs_B,
                cond_mix_coeffs_B,
                lambda_val,
            )

        return unif2, pdf

    @dr.syntax
    def eval_3D_blended_mixture_metalog_pdf(
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
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi, find the two closest index in terms of eucledian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        # evaluate the marginal (mixture) PDF/CDF lerping between the two slices
        unif_x, pdf_marginal = (
            dr_ml.eval_bounded_mixture_metalog_cdf_pdf_slice_lerp_blend(
                marginal_mix_coeffs_A,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_B,
                marginal_mix_metalog_coeffs_B,
                w2D.x,
                idx0,
                idx1,
                lambda_val,
                t,
                self.marginal_bounds,
            )
        )

        # evaluate conditional (mixture) PDF/CDF lerping between the two slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_lerp_blend(
                marginal_quantiles_A,
                cond_mix_coeffs_A,
                cond_mix_metalog_coeffs_A,
                marginal_quantiles_B,
                cond_mix_coeffs_B,
                cond_mix_metalog_coeffs_B,
                w2D,
                self.conditional_bounds,
                idx0,
                idx1,
                lambda_val,
                t,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    @dr.syntax
    def eval_4D_blended_mixture_metalog_pdf(
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
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        # evaluate marginal (mixture) PDF/CDF lerping between four slices
        unif_x, pdf_marginal = (
            dr_ml.eval_bounded_mixture_metalog_cdf_pdf_slice_bilerp_blend(
                marginal_mix_coeffs_A,
                marginal_mix_metalog_coeffs_A,
                marginal_mix_coeffs_B,
                marginal_mix_metalog_coeffs_B,
                w2D.x,
                idx00,
                idx01,
                idx10,
                idx11,
                lambda_val,
                t_theta,
                t_phi,
                self.marginal_bounds,
            )
        )

        # evaluate conditional (mixture) PDF/CDF lerping between four slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_bilerp_blend(
                marginal_quantiles_A,
                cond_mix_coeffs_A,
                cond_mix_metalog_coeffs_A,
                marginal_quantiles_B,
                cond_mix_coeffs_B,
                cond_mix_metalog_coeffs_B,
                w2D,
                self.conditional_bounds,
                idx00,
                idx01,
                idx10,
                idx11,
                lambda_val,
                t_theta,
                t_phi,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    @dr.syntax
    def eval_3D_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi, find the two closest index in terms of eucledian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        # evaluate the marginal (mixture) PDF/CDF lerping between the two slices
        unif_x, pdf_marginal = (
            dr_ml.eval_bounded_mixture_metalog_cdf_pdf_slice_lerp(
                marginal_mix_coeffs,
                marginal_mix_metalog_coeffs,
                w2D.x,
                idx0,
                idx1,
                t,
                self.marginal_bounds,
            )
        )

        # evaluate conditional (mixture) PDF/CDF lerping between the two slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_lerp(
                marginal_quantiles,
                cond_mix_coeffs,
                cond_mix_metalog_coeffs,
                w2D,
                self.conditional_bounds,
                self.marginal_quantile_cdf_range,
                idx0,
                idx1,
                t,
                unif_x,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    @dr.syntax
    def eval_4D_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        # evaluate marginal (mixture) PDF/CDF lerping between four slices
        unif_x, pdf_marginal = (
            dr_ml.eval_bounded_mixture_metalog_cdf_pdf_slice_bilerp(
                marginal_mix_coeffs,
                marginal_mix_metalog_coeffs,
                w2D.x,
                idx00,
                idx01,
                idx10,
                idx11,
                t_theta,
                t_phi,
                self.marginal_bounds,
            )
        )

        # evaluate conditional (mixture) PDF/CDF lerping between four slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_mixture_metalog_cdf_pdf_slice_bilerp(
                marginal_quantiles,
                cond_mix_coeffs,
                cond_mix_metalog_coeffs,
                w2D,
                self.conditional_bounds,
                self.marginal_quantile_cdf_range,
                idx00,
                idx01,
                idx10,
                idx11,
                t_theta,
                t_phi,
                unif_x,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    @dr.syntax
    def eval_3D_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi, find the two closest index in terms of eucledian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        # evaluate the marginal (mixture) PDF/CDF lerping between the two slices
        unif_x, pdf_marginal = (
            dr_ml.eval_bounded_mixture_metalog_cdf_pdf_slice_lerp(
                marginal_mix_coeffs,
                marginal_mix_metalog_coeffs,
                w2D.x,
                idx0,
                idx1,
                t,
                self.marginal_bounds,
            )
        )

        # evaluate condtional PDF/CDF lerping between the two slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_metalog_cdf_pdf_slice_lerp(
                marginal_quantiles,
                conditional_metalogs,
                w2D,
                self.conditional_bounds,
                self.marginal_quantile_cdf_range,
                idx0,
                idx1,
                t,
                unif_x,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    @dr.syntax
    def eval_4D_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        # evaluate marginal (mixture) PDF/CDF lerping between four slices
        unif_x, pdf_marginal = (
            dr_ml.eval_bounded_mixture_metalog_cdf_pdf_slice_bilerp(
                marginal_mix_coeffs,
                marginal_mix_metalog_coeffs,
                w2D.x,
                idx00,
                idx01,
                idx10,
                idx11,
                t_theta,
                t_phi,
                self.marginal_bounds,
            )
        )

        # evaluate conditional (mixture) PDF/CDF lerping between four slices
        unif_y, pdf_cond = (
            dr_ml.eval_bounded_conditional_metalog_cdf_pdf_slice_bilerp(
                marginal_quantiles,
                conditional_metalogs,
                w2D,
                self.conditional_bounds,
                self.marginal_quantile_cdf_range,
                idx00,
                idx01,
                idx10,
                idx11,
                t_theta,
                t_phi,
                unif_x,
            )
        )

        # compute joint pdf
        pdf_joint = pdf_marginal * pdf_cond

        return mi.Point2f(unif_x, unif_y), pdf_joint

    def eval_3D_metalog_pdf(
        self,
        w2D,
        wi2D,
        marginal_metalogs,
        marginal_quantiles,
        conditional_metalogs,
    ):
        # for each wi, find the two closest index in terms of eucledian distance
        idx0, idx1 = dr_ml.argtop2_nearest_keys(wi2D, self.metalogs_keys)

        # compute interpolation ratio
        t = dr_ml.eval_projection_weights(self.metalogs_keys, wi2D, idx0, idx1)

        unif2, pdf = dr_ml.eval_bounded_bivariate_metalog_cdf_pdf_slice_lerp(
            marginal_metalogs,
            marginal_quantiles,
            conditional_metalogs,
            w2D,
            self.marginal_bounds,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            idx0,
            idx1,
            t,
        )

        return unif2, pdf

    def eval_4D_metalog_pdf(
        self,
        w2D,
        wi2D,
        marginal_metalogs,
        marginal_quantiles,
        conditional_metalogs,
    ):
        # for each wi2D, find (phi, theta) grid cell for each sample and its
        # four corner coordinates together with flat slice indices
        phi0, phi1, theta0, theta1, idx00, idx10, idx01, idx11 = (
            dr_ml.phi_theta_cell_corners(
                wi2D,
                self.metalogs_keys,
                n_phi=self.n_phi_i,
                n_theta=self.n_theta_i,
            )
        )

        # get local bilinear coordinates (t_phi, t_theta) in [0,1]
        denom_phi = dr.maximum(phi1 - phi0, dr_ml.EPSILON)
        denom_theta = dr.maximum(theta1 - theta0, dr_ml.EPSILON)
        t_phi = dr.clip((wi2D.x - phi0) * dr.rcp(denom_phi), 0.0, 1.0)
        t_theta = dr.clip((wi2D.y - theta0) * dr.rcp(denom_theta), 0.0, 1.0)

        unif2, pdf = dr_ml.eval_bounded_bivariate_metalog_cdf_pdf_slice_bilerp(
            marginal_metalogs,
            marginal_quantiles,
            conditional_metalogs,
            w2D,
            self.marginal_bounds,
            self.conditional_bounds,
            self.marginal_quantile_cdf_range,
            idx00,
            idx01,
            idx10,
            idx11,
            t_theta,
            t_phi,
        )

        return unif2, pdf

    def init_metalog_slices(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        (
            marginal_metalogs,
            marginal_quantiles,
            conditional_metalogs,
        ) = super().init_metalog_slices(
            metalogs_filename,
            metalog_term,
            marginal_bounds,
            conditional_bounds,
            metalog_marginal,
            torch_device,
        )

        # convert to Dr types
        marginal_metalogs = mi.TensorXf(marginal_metalogs)
        marginal_quantiles = mi.TensorXf(marginal_quantiles)
        conditional_metalogs = mi.TensorXf(conditional_metalogs)
        self.metalogs_keys = mi.Point2f(self.metalogs_keys.T)

        return (
            marginal_metalogs,
            marginal_quantiles,
            conditional_metalogs,
        )

    def init_marginal_mixture_conditional_slices(
        self,
        metalogs_filename: str,
        cond_metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        (
            marginal_quantiles,
            conditional_metalogs,
            mix_weights,
            mix_spt_coeffs,
        ) = super().init_marginal_mixture_conditional_slices(
            metalogs_filename,
            cond_metalog_term,
            marginal_bounds,
            conditional_bounds,
            metalog_marginal,
            torch_device,
        )

        # convert to Dr types
        marginal_mix_spt_coeffs = mi.TensorXf(mix_spt_coeffs)
        marginal_mix_weights = mi.TensorXf(mix_weights)
        marginal_quantiles = mi.TensorXf(marginal_quantiles)
        conditional_metalogs = mi.TensorXf(conditional_metalogs)
        self.metalogs_keys = mi.Point2f(self.metalogs_keys.T)

        return (
            marginal_mix_spt_coeffs,
            marginal_mix_weights,
            marginal_quantiles,
            conditional_metalogs,
        )

    def init_bivariate_mixture_slices(
        self,
        metalogs_filename: str,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_mix_weights,
            cond_mix_metalog_coeffs,
        ) = super().init_bivariate_mixture_slices(
            metalogs_filename,
            marginal_bounds,
            conditional_bounds,
            metalog_marginal,
            torch_device,
        )

        # convert to Dr types
        marginal_quantiles = mi.TensorXf(marginal_quantiles)
        marg_mix_metalog_coeffs = mi.TensorXf(marg_mix_metalog_coeffs)
        marg_mix_weights = mi.TensorXf(marg_mix_weights)
        cond_mix_weights = mi.TensorXf(cond_mix_weights)
        cond_mix_metalog_coeffs = mi.TensorXf(cond_mix_metalog_coeffs)
        self.metalogs_keys = mi.Point2f(self.metalogs_keys.T)

        return (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_mix_weights,
            cond_mix_metalog_coeffs,
        )

    def init_metalogs(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        raise NotImplementedError("Use init_metalog_slices() instead.")

    def init_marginal_mixture_metalogs(
        self,
        metalogs_filename: str,
        cond_metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        raise NotImplementedError("Use init_metalog_slices() instead.")

    def __str__(self):
        return "DrNDMetalogs"
