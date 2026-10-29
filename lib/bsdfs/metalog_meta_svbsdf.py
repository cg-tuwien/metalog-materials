# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import mitsuba as mi
import drjit as dr
from lib.bsdfs import metalog_meta_bsdf as m_bsdf
from lib.metalog_strategies.metalogs_drjit_variant import DrNDMetalogs
from lib.metalog_strategies.sample_variant import (
    SampleBlendedNDMixtureMetalogs,
)
from lib.utils.meta_material import MetaMaterialTypes
from lib.metalogs_drjit import (
    cos_theta_phi_to_unitvec,
    EPSILON,
    unitvec_to_cos_theta_phi,
)

mi.register_bsdf(
    "MetaBSDF",
    lambda props: m_bsdf.MetaBSDF(props),
)


class MetaSVBRDF(m_bsdf.MetaBSDF):

    def __init__(
        self, props: mi.Properties, use_nd_metalog: bool = True
    ) -> None:
        mi.BSDF.__init__(self, props)
        self.use_nd_metalog = use_nd_metalog
        self.is_meta_metalog = True

        self.read_metalog_scene_props(props)
        self.init_svbrdf_metalogs()

        reflection_flags = self.get_reflection_flags()
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: float,
        sample2: mi.Point2f,
        active: bool = True,
    ):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        bs = mi.BSDFSample3f()

        active &= cos_theta_i > 0.0
        if not self.is_ctx_enabled(ctx):
            return (bs, 0.0)

        # transform wi into spherical coordinates
        theta_i = self.elevation(si.wi)
        phi_i = dr.atan2(si.wi.y, si.wi.x)
        wi2D = mi.Vector2f(phi_i, theta_i)

        # evaluate spatial lambda
        lambda_val = self.blend_texture.eval(si.uv)[0]

        w2D, bs.pdf = self.sampler.sample(sample2, wi2D, lambda_val)
        if self.rusinkiewicz_reparam:
            if self.is_isotropic:
                # shift by phi_i as it is isotropic - see also EPFL Measured BSDF
                w2D.y = self.wrap_to_phi_bounds(w2D.y + phi_i)

            # transform from spherical to cartesian coordinates
            h = cos_theta_phi_to_unitvec(w2D)

            # reflect based on the halfway vector h
            bs.wo = dr.fma(h, 2.0 * dr.dot(h, si.wi), -si.wi)

            # jacobian of the half-direction mapping
            bs.pdf *= dr.rcp(4.0 * dr.maximum(dr.dot(bs.wo, h), EPSILON))
        else:
            # transform from spherical to cartesian coordinates
            bs.wo = cos_theta_phi_to_unitvec(w2D)

        # ensure that this is a valid sample
        cos_theta_o = mi.Frame3f.cos_theta(bs.wo)
        active &= (bs.pdf > 0.0) & (cos_theta_o > 0.0)

        # interpolate between directional-hemispherical reflectances
        dhr_uv = mi.Point2f((phi_i + dr.pi) * dr.inv_two_pi, 1.0 - cos_theta_i)
        dhr_A = mi.Color3f(self.dhr_texture_A.eval(dhr_uv))
        dhr_B = mi.Color3f(self.dhr_texture_B.eval(dhr_uv))
        dhr_mixed = dr.lerp(dhr_A, dhr_B, lambda_val)

        # evaluate spatial color
        spatial_color = mi.Color3f(self.base_color_texture.eval(si.uv))

        bs.eta = 1.0
        bs.sampled_type = self.get_sampled_type()
        bs.sampled_component = 0

        # evaluate the final MC weight
        weight = dr.select(
            active,
            dhr_mixed * spatial_color,
            0.0,
        )

        return (
            bs,
            weight,
        )

    def eval(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool = True,
    ) -> mi.Color3f:
        if not self.is_ctx_enabled(ctx):
            return 0.0

        # transform wi into spherical coordinates
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        theta_i = self.elevation(si.wi)
        phi_i = dr.atan2(si.wi.y, si.wi.x)
        wi2D = mi.Vector2f(phi_i, theta_i)

        # evaluate spatial lambda
        lambda_val = self.blend_texture.eval(si.uv)[0]

        # interpolate between directional-hemispherical reflectances
        dhr_uv = mi.Point2f((phi_i + dr.pi) * dr.inv_two_pi, 1.0 - cos_theta_i)
        dhr_A = mi.Color3f(self.dhr_texture_A.eval(dhr_uv))
        dhr_B = mi.Color3f(self.dhr_texture_B.eval(dhr_uv))
        dhr_mixed = dr.lerp(dhr_A, dhr_B, lambda_val)

        # evaluate spatial color
        spatial_color = mi.Color3f(self.base_color_texture.eval(si.uv))

        # evaluate metalog PDF
        pdf = self.eval_metalog_pdf(si, wo, phi_i, wi2D, active)

        """Evaluate the BRDF. See also `MetaBRDF`."""
        result = spatial_color * dhr_mixed * pdf

        return result

    def eval_metalog_pdf(
        self,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        phi_i: mi.Float,
        wi2D: mi.Point2f,
        active: bool = True,
    ):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        # evaluate spatial lambda
        lambda_val = self.blend_texture.eval(si.uv)[0]

        if self.rusinkiewicz_reparam:
            # calculate the halfway vector
            h: mi.Vector3f = dr.normalize(wo + si.wi)

            """
            Filter cases where the micro/macro-surface do not aggre on the side.
            """
            active &= (
                (cos_theta_i > 0.0)
                & (cos_theta_o > 0.0)
                & (dr.dot(si.wi, h) > 0.0)
                & (dr.dot(wo, h) > 0)
            )

            # transform h to spherical coordinates and unshift azimuth
            w2D = unitvec_to_cos_theta_phi(h)
            if self.is_isotropic:
                w2D.y = self.wrap_to_phi_bounds(w2D.y - phi_i)

            # evaluate the metalog density
            _, pdf_h = self.sampler.eval_pdf(w2D, wi2D, lambda_val)

            # jacobian of the half-direction mapping
            result = pdf_h * dr.rcp(4.0 * dr.maximum(dr.dot(wo, h), EPSILON))
        else:
            # filter valid cases
            active &= (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

            # transform wo to spherical coordinates
            wo2D = unitvec_to_cos_theta_phi(wo)

            # evaluate metalog density having sampled wo2D
            _, result = self.sampler.eval_pdf(wo2D, wi2D, lambda_val)

        return dr.select(active, result, 0.0)

    def read_metalog_scene_props(self, props):
        self.metalog_marginal = props.get("metalog_marginal", "phi")
        if self.metalog_marginal not in ["cos_theta", "phi"]:
            raise ValueError(
                "Expected `metalog_marginal` either `cos_theta` or `phi`."
            )

        self.marginal_bounds, self.conditional_bounds = self.get_bounds()

        # Read Two Metalog Files
        self.metalogs_filename_0 = props.get("sample_metalogs_filename_0", None)
        self.metalogs_filename_1 = props.get("sample_metalogs_filename_1", None)
        if not self.metalogs_filename_0 or not self.metalogs_filename_1:
            raise ValueError(
                "Please provide `sample_metalogs_filename_0` and `sample_metalogs_filename_1`."
            )

        # Read Two DHR Files
        self.dhr_filename_0 = props.get(
            "directional_hemispherical_reflectance_filename_0", None
        )
        self.dhr_filename_1 = props.get(
            "directional_hemispherical_reflectance_filename_1", None
        )
        if not self.dhr_filename_0 or not self.dhr_filename_1:
            raise ValueError(
                "Please provide `directional_hemispherical_reflectance_filename_0` and `..._1`."
            )

        # read Blend Texture
        self.blend_texture_filename = props.get("blend_texture_filename", None)

        self.marginal_mode = "mixture"
        self.conditional_mode = "mixture"
        self.torch_device = "cuda" if "cuda" in mi.variant() else "cpu"

        # set material types
        meta_material_type_A = props.get("meta_material_type_A", "None")
        try:
            self.m_meta_material_type_A = MetaMaterialTypes(
                meta_material_type_A
            )
        except ValueError:
            mi.Log(
                mi.LogLevel.Error,
                f"Undefined `meta_material_type_A`, setting to `MetaMaterialTypes.META_ROUGHCONDUCTOR`.",
            )
            self.m_meta_material_type_A = MetaMaterialTypes.META_ROUGHCONDUCTOR

        meta_material_type_B = props.get("meta_material_type_B", "None")
        try:
            self.m_meta_material_type_B = MetaMaterialTypes(
                meta_material_type_B
            )
        except ValueError:
            mi.Log(
                mi.LogLevel.Error,
                f"Undefined `meta_material_type_B`, setting to `MetaMaterialTypes.META_ROUGHCONDUCTOR`.",
            )
            self.m_meta_material_type_B = MetaMaterialTypes.META_ROUGHCONDUCTOR

        # read halfway vector reparameterization (default True)
        self.rusinkiewicz_reparam = props.get("rusinkiewicz_reparam", True)

        mi.Log(
            mi.LogLevel.Info,
            f"Rusinkiewicz reparameterization is set to {self.rusinkiewicz_reparam} for both materials.",
        )

        # set base color
        self.init_base_color_texture = props.get(
            "init_base_color_texture", False
        )
        base_color_filename = props.get("base_color_filename", None)
        self.base_color_filename = (
            None if base_color_filename == "None" else base_color_filename
        )

    def init_blend_texture(self) -> mi.Texture:
        import numpy as np

        if self.blend_texture_filename is not None:
            bitmap = mi.Bitmap(self.blend_texture_filename)
            bitmap = bitmap.convert(
                pixel_format=bitmap.pixel_format(),
                component_format=mi.Struct.Type.Float32,
                srgb_gamma=not bitmap.srgb_gamma(),
            )
            tensor = mi.TensorXf(bitmap)
        else:
            tensor = mi.TensorXf(np.full((1, 1, 1), 0.5, dtype=np.float32))

        return mi.Texture2f(
            tensor,
            filter_mode=dr.FilterMode.Linear,
            wrap_mode=dr.WrapMode.Clamp,
            use_accel=True,
        )

    def init_svbrdf_metalogs(self):
        self.metalogs_variant = DrNDMetalogs()

        # load from metalog material A
        (
            marg_quantiles_A,
            marg_mix_coeffs_A,
            marg_mix_metalog_coeffs_A,
            cond_mix_coeffs_A,
            cond_mix_metalog_coeffs_A,
        ) = self.metalogs_variant.init_bivariate_mixture_slices(
            self.metalogs_filename_0,
            self.marginal_bounds,
            self.conditional_bounds,
            self.metalog_marginal,
            self.torch_device,
        )

        # load from metalog material A
        (
            marg_quantiles_B,
            marg_mix_coeffs_B,
            marg_mix_metalog_coeffs_B,
            cond_mix_coeffs_B,
            cond_mix_metalog_coeffs_B,
        ) = self.metalogs_variant.init_bivariate_mixture_slices(
            self.metalogs_filename_1,
            self.marginal_bounds,
            self.conditional_bounds,
            self.metalog_marginal,
            self.torch_device,
        )

        """
        Pad parameters metalog coeffiecients and mixing weights, if dimensions
        between material A and B do not match.

        Quantiles resolution may differ, so they are untouched unless number of
        slices differ. In that case, we warn the user and pad using `edge`
        method.
        """
        from lib.utils.material_interpolation.utils import pad_tensor_for_interp

        marg_mix_coeffs_A, marg_mix_coeffs_B = pad_tensor_for_interp(
            marg_mix_coeffs_A, marg_mix_coeffs_B, "constant"
        )
        marg_mix_metalog_coeffs_A, marg_mix_metalog_coeffs_B = (
            pad_tensor_for_interp(
                marg_mix_metalog_coeffs_A, marg_mix_metalog_coeffs_B, "constant"
            )
        )
        cond_mix_coeffs_A, cond_mix_coeffs_B = pad_tensor_for_interp(
            cond_mix_coeffs_A, cond_mix_coeffs_B, "constant"
        )
        cond_mix_metalog_coeffs_A, cond_mix_metalog_coeffs_B = (
            pad_tensor_for_interp(
                cond_mix_metalog_coeffs_A, cond_mix_metalog_coeffs_B, "constant"
            )
        )

        if marg_quantiles_A.shape[0] != marg_quantiles_B.shape[0]:
            mi.Log(
                mi.LogLevel.Warn,
                "Quantile slices do not match. We recomend using same number of slices. Repeating quantiles as fallback.",
            )
            marg_quantiles_A, marg_quantiles_B = pad_tensor_for_interp(
                marg_quantiles_A, marg_quantiles_B, "edge"
            )

        # initialize the sampler
        self.sampler = SampleBlendedNDMixtureMetalogs(
            self.metalogs_variant,
            self.metalog_marginal,
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
        )

        # initialize blend, base color and DHR textures
        self.blend_texture = self.init_blend_texture()
        self.base_color_texture = self.init_spatial_base_color()
        self.dhr_texture_A, shape_A = (
            self.init_directional_hemispherical_reflectance(self.dhr_filename_0)
        )
        self.dhr_texture_B, shape_B = (
            self.init_directional_hemispherical_reflectance(self.dhr_filename_1)
        )

        """
        If dimensions between directional-reflectances differ, then use the
        reflectance from material A.
        """
        if shape_A != shape_B:
            mi.Log(
                mi.LogLevel.Warn,
                f"DHR shape mismatch between material A and B. Using DHR from material A.",
            )
            self.dhr_texture_B = self.dhr_texture_A

        self.is_isotropic = self.metalogs_variant.is_isotropic
        if self.is_isotropic and not self.rusinkiewicz_reparam:
            mi.Log(
                mi.LogLevel.Warn,
                "Material is isotropic (n_phi_i == 1) but Rusinkiewicz reparameterization is disabled.",
            )

        # save phi bounds
        if self.metalog_marginal == "phi":
            self.phi_bounds = self.marginal_bounds
        else:
            self.phi_bounds = self.conditional_bounds

    def get_reflection_flags(self):

        def get_reflection_flag(material_type):
            if material_type == MetaMaterialTypes.META_ROUGHCONDUCTOR:
                return mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
            elif material_type == MetaMaterialTypes.META_DIFFUSE:
                return mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
            elif material_type == MetaMaterialTypes.META_EPFL_MEASURED:
                return mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
            else:
                raise ValueError(
                    f"Unexpected meta material type: {material_type}"
                )

        reflect_flags_A = get_reflection_flag(self.m_meta_material_type_A)
        reflect_flags_B = get_reflection_flag(self.m_meta_material_type_B)
        return reflect_flags_A | reflect_flags_B

    def is_ctx_enabled(self, ctx):

        def ctx_enabled(material_type):
            if material_type == MetaMaterialTypes.META_ROUGHCONDUCTOR:
                return ctx.is_enabled(mi.BSDFFlags.GlossyReflection)
            elif material_type == MetaMaterialTypes.META_DIFFUSE:
                return ctx.is_enabled(mi.BSDFFlags.DiffuseReflection)
            elif material_type == MetaMaterialTypes.META_EPFL_MEASURED:
                return ctx.is_enabled(mi.BSDFFlags.GlossyReflection)
            else:
                raise ValueError(
                    f"Unexpected meta material type: {material_type}"
                )

        ctx_enabled_A = ctx_enabled(self.m_meta_material_type_A)
        ctx_enabled_B = ctx_enabled(self.m_meta_material_type_B)
        return ctx_enabled_A and ctx_enabled_B

    def get_sampled_type(self):

        def sampled_type(material_type):
            if material_type == MetaMaterialTypes.META_ROUGHCONDUCTOR:
                return mi.BSDFFlags.GlossyReflection
            elif material_type == MetaMaterialTypes.META_DIFFUSE:
                return mi.BSDFFlags.DiffuseReflection
            elif material_type == MetaMaterialTypes.META_EPFL_MEASURED:
                return mi.BSDFFlags.GlossyReflection
            else:
                raise ValueError(
                    f"Unexpected meta material type: {material_type}"
                )

        """NOTE: Not sure how accurate this is but that's life."""
        sampled_type_A = sampled_type(self.m_meta_material_type_A)
        sampled_type_B = sampled_type(self.m_meta_material_type_B)
        sampled_type_AB = sampled_type_A | sampled_type_B
        return mi.UInt32(+sampled_type_AB)
