# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from lib.metalog_strategies.metalogs_variant import MetalogsVariant
import mitsuba as mi
from abc import ABC, abstractmethod
from typing import Tuple


class SampleVariant(ABC):

    metalogs_variant: MetalogsVariant
    metalog_marginal: str = "phi"

    def __init__(
        self, metalogs_variant: MetalogsVariant, metalog_marginal: str = "phi"
    ):
        super().__init__()

        self.metalogs_variant = metalogs_variant
        self.metalog_marginal = metalog_marginal

    def sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        """
        Warp a uniformly distributed sample (`sample2`) to metalog distribution.

        Args:
            sample2 (mi.Point2f): Uniformly distributed sample to warp.
            wi2D (mi.Vector2f): Direction to locate which metalog slice to warp from. Used only with ND metalogs.

        Returns: tuple[mi.Point2f, mi.Float]:
            - Samples warped to metalog distribution. If `metalog_marginal` is `phi`, x and y values are flipped.
            - PDF of the warped samples.

        """
        w2D, pdf = self.warp_sample(sample2, wi2D)

        # flip x and y if the metalog marginal is defined as phi (azimuth)
        if self.metalog_marginal == "phi":
            w2D = mi.Point2f(w2D.y, w2D.x)

        return w2D, pdf

    def eval_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        """
        Evaluates the metalog PDF of a given sample (`w2D`).

        Args:
            w2D (mi.Vector2f): Sample to evaluate the metalog PDF.
            wi2D (mi.Vector2f): Direction to locate which metalog slice to use. Used only with ND metalogs.

        Returns: Tuple[mi.Vector2f, mi.Float]:
            - Inverse warp that yields the "random numbers" that would have generated the given sample (`w2D`). If
            `metalog_marginal` is `phi`, x and y values are flipped.
            - PDF of the given samples.
        """
        w2D_ = w2D
        if self.metalog_marginal == "phi":
            w2D_ = mi.Point2f(w2D.y, w2D.x)

        unif2, pdf = self.pdf(w2D_, wi2D)

        if self.metalog_marginal == "phi":
            unif2 = mi.Point2f(unif2.y, unif2.x)

        return unif2, pdf

    @abstractmethod
    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        pass

    @abstractmethod
    def warp_sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        """Implementation specific sampling logic."""
        pass

    @abstractmethod
    def pdf(
        self, w2D: mi.Vector2f, wi2D: mi.Vector2f
    ) -> Tuple[mi.Vector2f, mi.Float]:
        """Implementation specific PDF evaluation logic."""
        pass


class SampleVanillaMetalogs(SampleVariant):

    metalogs: dict

    def __init__(
        self,
        metalogs_variant: MetalogsVariant,
        metalog_marginal: str,
        metalogs: dict,
    ):
        super().__init__(metalogs_variant, metalog_marginal)
        self.metalogs = metalogs

    def warp_sample(
        self, sample2: mi.Vector3f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        return self.metalogs_variant.sample_metalog(sample2)

    def pdf(
        self,
        w2D: mi.Point2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.metalogs_variant.eval_metalog_pdf(w2D)

    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        cb.put(
            "marginal_metalogs",
            self.metalogs_variant.metalogs["marginal"],
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "conditional_metalogs",
            self.metalogs_variant.metalogs["conditional"],
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marginal_quantiles",
            self.metalogs_variant.metalogs["marginal-quantiles"],
            mi.ParamFlags.Differentiable,
        )


class SampleMarginalMixtureMetalogs(SampleVariant):

    marg_mix_metalog_coeffs: mi.TensorXf
    marg_mix_coeffs: mi.TensorXf
    cond_metalogs: mi.TensorXf
    marginal_quantiles: mi.TensorXf

    def __init__(
        self,
        metalogs_variant: MetalogsVariant,
        metalog_marginal: str,
        marg_mix_metalog_coeffs: mi.TensorXf,
        marg_mix_coeffs: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
    ):
        super().__init__(metalogs_variant, metalog_marginal)

        self.marg_mix_metalog_coeffs = marg_mix_metalog_coeffs
        self.marg_mix_coeffs = marg_mix_coeffs
        self.cond_metalogs = cond_metalogs
        self.marginal_quantiles = marginal_quantiles

    def warp_sample(
        self, sample2: mi.Vector3f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        return self.metalogs_variant.sample_marginal_mixture_metalog(
            sample2,
            self.marg_mix_metalog_coeffs,
            self.marg_mix_coeffs,
            self.marginal_quantiles,
            self.cond_metalogs,
        )

    def pdf(
        self,
        w2D: mi.Point2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.metalogs_variant.eval_marginal_mixture_metalog_pdf(
            w2D,
            self.marg_mix_metalog_coeffs,
            self.marg_mix_coeffs,
            self.marginal_quantiles,
            self.cond_metalogs,
        )

    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        cb.put(
            "marg_mix_metalog_coeffs",
            self.marg_mix_metalog_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_mix_coeffs",
            self.marg_mix_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_metalogs",
            self.cond_metalogs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marginal_quantiles",
            self.marginal_quantiles,
            mi.ParamFlags.Differentiable,
        )


class SampleMixtureMetalogs(SampleVariant):

    marg_mix_metalog_coeffs: mi.TensorXf
    marg_mix_coeffs: mi.TensorXf
    cond_mix_metalog_coeffs: mi.TensorXf
    cond_mix_coeffs: mi.TensorXf
    marginal_quantiles: mi.TensorXf

    def __init__(
        self,
        metalogs_variant: MetalogsVariant,
        metalog_marginal: str,
        marg_mix_metalog_coeffs: mi.TensorXf,
        marg_mix_coeffs: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
    ):
        super().__init__(metalogs_variant, metalog_marginal)

        self.marg_mix_metalog_coeffs = marg_mix_metalog_coeffs
        self.marg_mix_coeffs = marg_mix_coeffs
        self.cond_mix_metalog_coeffs = cond_mix_metalog_coeffs
        self.cond_mix_coeffs = cond_mix_coeffs
        self.marginal_quantiles = marginal_quantiles

    def warp_sample(
        self, sample2: mi.Vector3f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        return self.metalogs_variant.sample_mixture_metalog(
            sample2,
            self.marg_mix_metalog_coeffs,
            self.marg_mix_coeffs,
            self.marginal_quantiles,
            self.cond_mix_metalog_coeffs,
            self.cond_mix_coeffs,
        )

    def pdf(
        self,
        w2D: mi.Point2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.metalogs_variant.eval_mixture_metalog_pdf(
            w2D,
            self.marg_mix_metalog_coeffs,
            self.marg_mix_coeffs,
            self.marginal_quantiles,
            self.cond_mix_metalog_coeffs,
            self.cond_mix_coeffs,
        )

    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        cb.put(
            "marg_mix_metalog_coeffs",
            self.marg_mix_metalog_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_mix_coeffs",
            self.marg_mix_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_metalog_coeffs",
            self.cond_mix_metalog_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_coeffs",
            self.cond_mix_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marginal_quantiles",
            self.marginal_quantiles,
            mi.ParamFlags.Differentiable,
        )


class SampleNDVanillaMetalogs(SampleVariant):

    marginal_metalogs: mi.TensorXf
    marginal_quantiles: mi.TensorXf
    conditional_metalogs: mi.TensorXf

    def __init__(
        self,
        metalogs_variant: MetalogsVariant,
        metalog_marginal: str,
        marginal_metalogs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ):
        super().__init__(metalogs_variant, metalog_marginal)

        self.marginal_metalogs = marginal_metalogs
        self.marginal_quantiles = marginal_quantiles
        self.conditional_metalogs = conditional_metalogs

    def warp_sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        return self.metalogs_variant.sample_nd_metalog(
            sample2,
            wi2D,
            self.marginal_metalogs,
            self.marginal_quantiles,
            self.conditional_metalogs,
        )

    def pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.metalogs_variant.eval_nd_metalog_pdf(
            w2D,
            wi2D,
            self.marginal_metalogs,
            self.marginal_quantiles,
            self.conditional_metalogs,
        )

    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        cb.put(
            "marginal_metalogs",
            self.marginal_metalogs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "conditional_metalogs",
            self.conditional_metalogs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marginal_quantiles",
            self.marginal_quantiles,
            mi.ParamFlags.Differentiable,
        )


class SampleNDMarginalMixtureConditionalMetalogs(SampleVariant):

    mix_metalog_coeffs: mi.TensorXf
    mix_coeffs: mi.TensorXf
    marginal_quantiles: mi.TensorXf
    conditional_metalogs: mi.TensorXf

    def __init__(
        self,
        metalogs_variant: MetalogsVariant,
        metalog_marginal: str,
        mix_metalog_coeffs: mi.TensorXf,
        mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ):
        super().__init__(metalogs_variant, metalog_marginal)

        self.mix_metalog_coeffs = mix_metalog_coeffs
        self.mix_coeffs = mix_coeffs
        self.marginal_quantiles = marginal_quantiles
        self.conditional_metalogs = conditional_metalogs

    def warp_sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        return self.metalogs_variant.sample_nd_marginal_mixture_metalog(
            sample2,
            wi2D,
            self.mix_metalog_coeffs,
            self.mix_coeffs,
            self.marginal_quantiles,
            self.conditional_metalogs,
        )

    def pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.metalogs_variant.eval_nd_marginal_mixture_metalog_pdf(
            w2D,
            wi2D,
            self.mix_metalog_coeffs,
            self.mix_coeffs,
            self.marginal_quantiles,
            self.conditional_metalogs,
        )

    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        cb.put(
            "mix_metalog_coeffs",
            self.mix_metalog_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "mix_coeffs",
            self.mix_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "conditional_metalogs",
            self.conditional_metalogs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marginal_quantiles",
            self.marginal_quantiles,
            mi.ParamFlags.Differentiable,
        )


class SampleNDMixtureMetalogs(SampleVariant):

    marg_mix_metalog_coeffs: mi.TensorXf
    marg_mix_coeffs: mi.TensorXf
    cond_mix_metalog_coeffs: mi.TensorXf
    cond_mix_coeffs: mi.TensorXf
    marginal_quantiles: mi.TensorXf

    def __init__(
        self,
        metalogs_variant: MetalogsVariant,
        metalog_marginal: str,
        marg_mix_metalog_coeffs: mi.TensorXf,
        marg_mix_coeffs: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
    ):
        super().__init__(metalogs_variant, metalog_marginal)

        self.marg_mix_metalog_coeffs = marg_mix_metalog_coeffs
        self.marg_mix_coeffs = marg_mix_coeffs
        self.cond_mix_metalog_coeffs = cond_mix_metalog_coeffs
        self.cond_mix_coeffs = cond_mix_coeffs
        self.marginal_quantiles = marginal_quantiles

    def warp_sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        return self.metalogs_variant.sample_nd_mixture_metalog(
            sample2,
            wi2D,
            self.marg_mix_metalog_coeffs,
            self.marg_mix_coeffs,
            self.marginal_quantiles,
            self.cond_mix_metalog_coeffs,
            self.cond_mix_coeffs,
        )

    def pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.metalogs_variant.eval_nd_mixture_metalog_pdf(
            w2D,
            wi2D,
            self.marg_mix_metalog_coeffs,
            self.marg_mix_coeffs,
            self.marginal_quantiles,
            self.cond_mix_metalog_coeffs,
            self.cond_mix_coeffs,
        )

    def traverse(self, cb: mi.TraversalCallback):
        """
        Update internal state after applying changes to parameters.

        See also:
            - https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.Object.traverse
            - https://mitsuba.readthedocs.io/en/latest/src/others/custom_plugin.html#Custom-Python-plugin
        """
        cb.put(
            "marg_mix_metalog_coeffs",
            self.marg_mix_metalog_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_mix_coeffs",
            self.marg_mix_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_metalog_coeffs",
            self.cond_mix_metalog_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_coeffs",
            self.cond_mix_coeffs,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marginal_quantiles",
            self.marginal_quantiles,
            mi.ParamFlags.Differentiable,
        )


class SampleBlendedNDMixtureMetalogs(SampleVariant):

    def __init__(
        self,
        metalogs_variant,
        metalog_marginal,
        marg_mix_metalog_coeffs_A,
        marg_mix_coeffs_A,
        cond_mix_metalog_coeffs_A,
        cond_mix_coeffs_A,
        marg_quantiles_A,
        marg_mix_metalog_coeffs_B,
        marg_mix_coeffs_B,
        cond_mix_metalog_coeffs_B,
        cond_mix_coeffs_B,
        marg_quantiles_B,
    ):
        super().__init__(metalogs_variant, metalog_marginal)

        # Material A
        self.marg_mix_metalog_coeffs_A = marg_mix_metalog_coeffs_A
        self.marg_mix_coeffs_A = marg_mix_coeffs_A
        self.cond_mix_metalog_coeffs_A = cond_mix_metalog_coeffs_A
        self.cond_mix_coeffs_A = cond_mix_coeffs_A
        self.marg_quantiles_A = marg_quantiles_A

        # Material B
        self.marg_mix_metalog_coeffs_B = marg_mix_metalog_coeffs_B
        self.marg_mix_coeffs_B = marg_mix_coeffs_B
        self.cond_mix_metalog_coeffs_B = cond_mix_metalog_coeffs_B
        self.cond_mix_coeffs_B = cond_mix_coeffs_B
        self.marg_quantiles_B = marg_quantiles_B

    def sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f, lambda_val: mi.Float
    ) -> tuple[mi.Point2f, mi.Float]:
        """
        SampleVariant override. Adapted to SVBRDF case.
        """
        w2D, pdf = self.metalogs_variant.sample_nd_blended_mixture_metalog(
            sample2,
            wi2D,
            self.marg_mix_metalog_coeffs_A,
            self.marg_mix_coeffs_A,
            self.marg_quantiles_A,
            self.cond_mix_metalog_coeffs_A,
            self.cond_mix_coeffs_A,
            self.marg_mix_metalog_coeffs_B,
            self.marg_mix_coeffs_B,
            self.marg_quantiles_B,
            self.cond_mix_metalog_coeffs_B,
            self.cond_mix_coeffs_B,
            lambda_val,
        )

        # flip x and y if the metalog marginal is defined as phi (azimuth)
        if self.metalog_marginal == "phi":
            w2D = mi.Point2f(w2D.y, w2D.x)

        return w2D, pdf

    def warp_sample(
        self, sample2: mi.Point2f, wi2D: mi.Vector2f
    ) -> tuple[mi.Point2f, mi.Float]:
        pass

    def pdf(
        self, w2D: mi.Vector2f, wi2D: mi.Vector2f
    ) -> Tuple[mi.Vector2f, mi.Float]:
        pass

    def eval_pdf(
        self, w2D: mi.Vector2f, wi2D: mi.Vector2f, lambda_val: mi.Float
    ) -> Tuple[mi.Vector2f, mi.Float]:
        """
        SampleVariant override. Adapted to SVBRDF case.
        """

        w2D_ = w2D
        if self.metalog_marginal == "phi":
            w2D_ = mi.Point2f(w2D.y, w2D.x)

        unif2, pdf = self.metalogs_variant.eval_nd_blended_mixture_metalog_pdf(
            w2D_,
            wi2D,
            self.marg_mix_metalog_coeffs_A,
            self.marg_mix_coeffs_A,
            self.marg_quantiles_A,
            self.cond_mix_metalog_coeffs_A,
            self.cond_mix_coeffs_A,
            self.marg_mix_metalog_coeffs_B,
            self.marg_mix_coeffs_B,
            self.marg_quantiles_B,
            self.cond_mix_metalog_coeffs_B,
            self.cond_mix_coeffs_B,
            lambda_val,
        )

        if self.metalog_marginal == "phi":
            unif2 = mi.Point2f(unif2.y, unif2.x)

        return unif2, pdf

    def traverse(self, cb: mi.TraversalCallback):
        # Material A
        cb.put(
            "marg_mix_metalog_coeffs_A",
            self.marg_mix_metalog_coeffs_A,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_mix_coeffs_A",
            self.marg_mix_coeffs_A,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_metalog_coeffs_A",
            self.cond_mix_metalog_coeffs_A,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_coeffs_A",
            self.cond_mix_coeffs_A,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_quantiles_A",
            self.marg_quantiles_A,
            mi.ParamFlags.Differentiable,
        )

        # Material B
        cb.put(
            "marg_mix_metalog_coeffs_B",
            self.marg_mix_metalog_coeffs_B,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_mix_coeffs_B",
            self.marg_mix_coeffs_B,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_metalog_coeffs_B",
            self.cond_mix_metalog_coeffs_B,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "cond_mix_coeffs_B",
            self.cond_mix_coeffs_B,
            mi.ParamFlags.Differentiable,
        )
        cb.put(
            "marg_quantiles_B",
            self.marg_quantiles_B,
            mi.ParamFlags.Differentiable,
        )
