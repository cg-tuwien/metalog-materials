# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from typing import Tuple
import mitsuba as mi
import drjit as dr
from lib.metalog_strategies.metalogs_drjit_variant import DrMetalogs
from lib.metalog_strategies.metalogs_torch_variant import (
    TorchNDMetalogs,
    TorchMetalogs,
)
from lib.metalog_strategies.metalogs_drjit_variant import DrNDMetalogs
from lib.metalog_strategies.sample_variant import (
    SampleMarginalMixtureMetalogs,
    SampleMixtureMetalogs,
    SampleVanillaMetalogs,
)
import lib.metalogs as ml
from math import pi


class MetalogBSDF(mi.BSDF):
    def __init__(
        self: mi.BSDF,
        props: mi.Properties,
        use_nd_metalog: bool = False,
        is_meta_metalog: bool = False,
    ) -> None:
        mi.BSDF.__init__(self, props)

        self.use_nd_metalog = use_nd_metalog
        self.is_meta_metalog = is_meta_metalog
        self.read_metalog_scene_props(props)

        if self.use_nd_metalog:
            self.init_metalog_measured()
        else:
            self.init_metalogs()

    def init_metalogs(self):
        if self.marginal_mode == "metalog":
            metalogs = self.metalogs_variant.init_metalogs(
                self.metalogs_filename,
                self.metalog_term,
                self.marginal_bounds,
                self.conditional_bounds,
                self.metalog_marginal,
                self.torch_device,
            )
            self.metalog_sampler = SampleVanillaMetalogs(
                self.metalogs_variant, self.metalog_marginal, metalogs
            )
        else:
            if self.conditional_mode == "metalog":
                (
                    marginal_quantiles,
                    marg_mix_weights,
                    marg_mix_metalog_coeffs,
                    cond_metalogs,
                ) = self.metalogs_variant.init_marginal_mixture_metalogs(
                    self.metalogs_filename,
                    self.metalog_term,
                    self.marginal_bounds,
                    self.conditional_bounds,
                    self.metalog_marginal,
                    self.torch_device,
                )
                self.metalog_sampler = SampleMarginalMixtureMetalogs(
                    self.metalogs_variant,
                    self.metalog_marginal,
                    marg_mix_metalog_coeffs,
                    marg_mix_weights,
                    cond_metalogs,
                    marginal_quantiles,
                )
            else:
                (
                    marginal_quantiles,
                    marg_mix_weights,
                    marg_mix_metalog_coeffs,
                    cond_mix_weights,
                    cond_mix_metalog_coeffs,
                ) = self.metalogs_variant.init_mixture_metalogs(
                    self.metalogs_filename,
                    self.marginal_bounds,
                    self.conditional_bounds,
                    self.metalog_marginal,
                    self.torch_device,
                )
                self.metalog_sampler = SampleMixtureMetalogs(
                    self.metalogs_variant,
                    self.metalog_marginal,
                    marg_mix_metalog_coeffs,
                    marg_mix_weights,
                    cond_mix_metalog_coeffs,
                    cond_mix_weights,
                    marginal_quantiles,
                )

    def init_metalog_measured(self):
        pass

    def init_metalog_slices(self, metalogs_filename):
        return self.metalogs_variant.init_metalog_slices(
            metalogs_filename,
            self.metalog_term,
            self.marginal_bounds,
            self.conditional_bounds,
            self.metalog_marginal,
            self.torch_device,
        )

    def init_marginal_mixture_conditional_slices(self, metalogs_filename):
        return self.metalogs_variant.init_marginal_mixture_conditional_slices(
            metalogs_filename,
            self.metalog_term,
            self.marginal_bounds,
            self.conditional_bounds,
            self.metalog_marginal,
            self.torch_device,
        )

    def init_bivariate_mixture_slices(self, metalogs_filename):
        return self.metalogs_variant.init_bivariate_mixture_slices(
            metalogs_filename,
            self.marginal_bounds,
            self.conditional_bounds,
            self.metalog_marginal,
            self.torch_device,
        )

    def init_marginal_metalog_slices(self, metalogs_filename):
        return self.metalogs_variant.init_marginal_metalog_slices(
            metalogs_filename,
            self.metalog_term,
            self.marginal_bounds,
            self.metalog_marginal,
            self.torch_device,
        )

    def get_bounds(self):
        cos_theta_bounds = (0.0, 1.0)
        phi_bounds = (-pi, pi)
        if self.metalog_marginal == "cos_theta":
            marginal_bounds = cos_theta_bounds
            conditional_bounds = phi_bounds
        else:
            marginal_bounds = phi_bounds
            conditional_bounds = cos_theta_bounds
        return marginal_bounds, conditional_bounds

    def read_metalog_scene_props(self, props):
        # read metalog terms
        tmp_metalog_term = props.get("metalog_terms", 3)
        self.metalog_term = tmp_metalog_term if tmp_metalog_term >= 3 else 3

        # read metalog marginal
        self.metalog_marginal = props.get("metalog_marginal", None)
        if self.metalog_marginal is None:
            mi.Log(
                mi.LogLevel.Warn,
                f"No `metalog_marginal` value is found, `phi` will be used instead.",
            )
            self.metalog_marginal = "phi"

        if (
            self.metalog_marginal != "cos_theta"
            and self.metalog_marginal != "phi"
        ):
            raise ValueError(
                "Expected `metalog_marginal` either `cos_theta` or `phi`."
            )

        # initialize marginal and conditional bounds
        # NOTE assuming bounded metalogs
        self.marginal_bounds, self.conditional_bounds = self.get_bounds()

        # read metalogs filename
        if self.is_meta_metalog:
            # get metalogs name for sample metalogs
            self.metalogs_filename = props.get("sample_metalogs_filename", None)
            if self.metalogs_filename is None:
                raise ValueError(
                    "Please provide `sample_metalogs_filename` in the scene file. For example: "
                    '`<string name="sample_metalogs_filename" value="sample_metalogs.pt" />`.'
                )
            # get metalogs name for directional-hemispherical-reflectance
            self.directional_hemispherical_reflectance_filename = props.get(
                "directional_hemispherical_reflectance_filename", None
            )
            if self.directional_hemispherical_reflectance_filename is None:
                raise ValueError(
                    "Please provide `directional_hemispherical_reflectance_filename` in the scene file. For example: "
                    '`<string name="directional_hemispherical_reflectance_filename" value="brdf.pt" />`.'
                )

        else:
            self.metalogs_filename = props.get("metalogs_filename", None)
            if self.metalogs_filename is None:
                raise ValueError(
                    "Please provide `metalogs_filename` in the scene file. For example: "
                    '`<string name="metalogs_filename" value="metalogs.pt" />`.'
                )

        # set marginal mode: metalog or mixture
        self.marginal_mode = props.get("marginal_mode", None)
        options_marginal_mode = ["metalog", "mixture"]
        if (
            self.marginal_mode is None
            or self.marginal_mode not in options_marginal_mode
        ):
            raise ValueError(
                "Marginal mode can be either `metalog` or `mixture`."
            )

        # set conditional mode: metalog or mixture
        self.conditional_mode = props.get("conditional_mode", None)
        options_conditional_mode = ["metalog", "mixture"]
        if (
            self.conditional_mode is None
            or self.conditional_mode not in options_conditional_mode
        ):
            raise ValueError(
                "Conditional mode can be either `metalog` or `mixture`."
            )

        # initialize `torch_device`
        self.torch_device = "cuda" if "cuda" in mi.variant() else "cpu"
        mi.Log(
            mi.LogLevel.Info,
            f"Using {self.torch_device} for torch operations.",
        )

        # read cdf method
        tmp_cdf_method = props.get("cdf_method", "newton")
        if tmp_cdf_method == "newton":
            self.cdf_method = ml.BivariateMetalogPDFMethod(
                marginal_method=ml.UnivariateMetalogCDFMethod.CDF_NEWTON,
                conditional_method=ml.UnivariateMetalogCDFMethod.CDF_NEWTON,
            )
        else:
            self.cdf_method = ml.BivariateMetalogPDFMethod(
                marginal_method=ml.UnivariateMetalogCDFMethod.CDF_INTERPOLATION,
                conditional_method=ml.UnivariateMetalogCDFMethod.CDF_INTERPOLATION,
            )

        mi.Log(
            mi.LogLevel.Info,
            f"Using {self.cdf_method} as the CDF method.",
        )

        # read Dr Metalogs flag
        dr_metalogs = props.get("dr_metalogs", True)
        if self.use_nd_metalog:
            self.metalogs_variant = (
                DrNDMetalogs() if dr_metalogs else TorchNDMetalogs()
            )
        else:
            if self.marginal_mode == "mixture" and not dr_metalogs:
                return ValueError(
                    "Mixture mode is currently only supported with DrJit."
                )

            self.metalogs_variant = (
                DrMetalogs() if dr_metalogs else TorchMetalogs()
            )
        mi.Log(
            mi.LogLevel.Info,
            f"Using Dr Metalogs: {dr_metalogs}",
        )

    def elevation(self, d: mi.Vector3f) -> mi.Float:
        """
        Numerically stable method to compute the elevation of the given normalized vector in local frame.
        Conceptually equivalent to: safe_acos(Frame3f.cos_theta(d)).

        Taken from Mitsuba 3.

        Args:
            d (mi.Vector3f): Normalized vector in local frame.

        Returns:
            mi.Float: Elevation term of the given vector.
        """
        dist = dr.sqrt(dr.square(d.x) + dr.square(d.y) + dr.square(d.z - 1.0))
        return 2.0 * dr.safe_asin(0.5 * dist)
