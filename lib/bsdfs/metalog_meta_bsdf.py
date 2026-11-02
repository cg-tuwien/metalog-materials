# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import mitsuba as mi
from lib.bsdfs import metalog_bsdf as m_bsdf
from lib.metalog_strategies.sample_variant import (
    SampleNDMarginalMixtureConditionalMetalogs,
    SampleNDMixtureMetalogs,
    SampleNDVanillaMetalogs,
)
from lib.utils.meta_material import MetaMaterialTypes
from lib.metalogs_drjit import (
    cos_theta_phi_to_unitvec,
    unitvec_to_cos_theta_phi,
    EPSILON,
)
import drjit as dr

mi.register_bsdf(
    "MetalogBSDF",
    lambda props: m_bsdf.MetalogBSDF(props),
)


class MetaBSDF(m_bsdf.MetalogBSDF):
    def __init__(
        self: mi.BSDF, props: mi.Properties, use_nd_metalog: bool = True
    ) -> None:
        m_bsdf.MetalogBSDF.__init__(
            self, props, use_nd_metalog=use_nd_metalog, is_meta_metalog=True
        )

        reflection_flags = self.get_reflection_flags()
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(
        self: mi.BSDF,
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

        w2D, bs.pdf = self.metalog_sampler.sample(sample2, wi2D)
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
        active &= (bs.pdf != 0.0) & (cos_theta_o > 0.0)

        # evaluate directional-hemispherical reflectance
        hemispherical_reflectance = self.eval_directional_reflectance(
            phi_i, cos_theta_i
        )

        # evaluate spatial color
        spatial_color = mi.Color3f(self.base_color_texture.eval(si.uv))

        bs.eta = 1.0
        bs.sampled_type = self.get_sampled_type()
        bs.sampled_component = 0

        # evaluate the final MC weight
        weight = dr.select(
            active & (bs.pdf > 0),
            hemispherical_reflectance * spatial_color,
            0.0,
        )

        return (
            bs,
            weight,
        )

    def eval_directional_reflectance(
        self, phi_i: mi.Float, cos_theta_i: mi.Float
    ) -> mi.Color3f:
        # map phi [-pi, pi] -> U [0, 1]
        u = (phi_i + dr.pi) * dr.inv_two_pi

        """
        Map theta [0, pi/2] -> V[0, 1].
        We generate DHR texture rows using linear steps in cos(theta) but not in
        theta. This ensures every pixel covers an equal solid angle on the
        hemisphere. This provides higher angular resolution at grazing angles
        where Fresnel effects are strongest.
        Since cos(theta) evaluates to 1.0 at the pole and 0.0 at the equator, we
        subtract it from 1.0. This maps the pole to V=0.0 (top row of the
        texture) and the equator to V=1.0 (bottom row of the texture).
        """
        v = 1.0 - cos_theta_i

        result = self.dhr_texture.eval(mi.Point2f(u, v))

        return mi.Color3f(result)

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

        # evaluate directional-hemispherical reflectance
        directional_reflectance = self.eval_directional_reflectance(
            phi_i, cos_theta_i
        )

        # evaluate spatial color
        spatial_color = mi.Color3f(self.base_color_texture.eval(si.uv))

        # evaluate metalog PDF
        pdf = self.eval_metalog_pdf(si, wo, phi_i, wi2D, active)

        """
        Evaluate the BRDF.

        We store the directional hemispherical reflectance rho(omega_i):
            rho(omega_i) = \\int_{H^2} f_r cos(theta_o) d omega_o
        In the thesis, we showed (Eq. 5.2):
            weight = f_r cos(theta_o)/p_metalog(omega_o) = rho(omega_i).
        The `eval()` method must return the `f_r  cos(theta)`. As such, we can
        reformulate and get:
            rho(omega_i) =  f_r * cos(theta_o)/p_metalog(omega_o),
            f_r * cos(theta_o) = rho(omega_i) * p_metalog(omega_o).
        """
        result = spatial_color * directional_reflectance * pdf

        return result

    def pdf(
        self: mi.BSDF,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool = True,
    ) -> float:
        if not self.is_ctx_enabled(ctx):
            return 0.0

        # transform wi into spherical coordinates
        theta_i = self.elevation(si.wi)
        phi_i = dr.atan2(si.wi.y, si.wi.x)
        wi2D = mi.Vector2f(phi_i, theta_i)

        return self.eval_metalog_pdf(si, wo, phi_i, wi2D, active)

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

        if self.rusinkiewicz_reparam:
            # calculate the halfway vector
            h: mi.Vector3f = dr.normalize(wo + si.wi)

            """
            Filter cases where the micro/macro-surface do not aggre on the side.
            """
            wo_dot_h = dr.dot(wo, h)
            active &= (
                (cos_theta_i > 0.0)
                & (cos_theta_o > 0.0)
                & (dr.dot(si.wi, h) > 0.0)
                & (wo_dot_h > 0)
            )

            # transform h to spherical coordinates and unshift azimuth
            w2D = unitvec_to_cos_theta_phi(h)

            if self.is_isotropic:
                w2D.y = self.wrap_to_phi_bounds(w2D.y - phi_i)

            # evaluate the metalog density
            _, pdf_h = self.metalog_sampler.eval_pdf(w2D, wi2D)

            # jacobian of the half-direction mapping
            result = pdf_h * dr.rcp(4.0 * dr.maximum(wo_dot_h, EPSILON))
        else:
            # filter valid cases
            active &= (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

            # transform wo to spherical coordinates
            wo2D = unitvec_to_cos_theta_phi(wo)

            # evaluate metalog density having sampled wo2D
            _, result = self.metalog_sampler.eval_pdf(wo2D, wi2D)

        return dr.select(active, result, 0.0)

    def wrap_to_phi_bounds(self, phi: mi.Float) -> mi.Float:
        span = self.phi_bounds[1] - self.phi_bounds[0]
        return phi - span * dr.floor((phi - self.phi_bounds[0]) / span)

    def init_metalog_measured(self):
        # initialize sample metalog slices
        mi.Log(mi.LogLevel.Info, "Initialize ND Metalogs for sampling.")

        if self.marginal_mode == "metalog":
            (
                sample_marginal_metalogs,
                sample_marginal_quantiles,
                sample_conditional_metalogs,
            ) = self.init_metalog_slices(self.metalogs_filename)

            self.metalog_sampler = SampleNDVanillaMetalogs(
                self.metalogs_variant,
                self.metalog_marginal,
                sample_marginal_metalogs,
                sample_marginal_quantiles,
                sample_conditional_metalogs,
            )
        else:
            if self.conditional_mode == "metalog":
                (
                    sample_marginal_mix_metalog_coeffs,
                    sample_marginal_mix_coeffs,
                    sample_marginal_quantiles,
                    sample_conditional_metalogs,
                ) = self.init_marginal_mixture_conditional_slices(
                    self.metalogs_filename
                )

                self.metalog_sampler = (
                    SampleNDMarginalMixtureConditionalMetalogs(
                        self.metalogs_variant,
                        self.metalog_marginal,
                        sample_marginal_mix_metalog_coeffs,
                        sample_marginal_mix_coeffs,
                        sample_marginal_quantiles,
                        sample_conditional_metalogs,
                    )
                )
            else:
                (
                    sample_marginal_quantiles,
                    sample_marginal_mix_coeffs,
                    sample_marginal_mix_metalog_coeffs,
                    sample_cond_mix_coeffs,
                    sample_cond_mix_metalog_coeffs,
                ) = self.init_bivariate_mixture_slices(self.metalogs_filename)

                self.metalog_sampler = SampleNDMixtureMetalogs(
                    self.metalogs_variant,
                    self.metalog_marginal,
                    sample_marginal_mix_metalog_coeffs,
                    sample_marginal_mix_coeffs,
                    sample_cond_mix_metalog_coeffs,
                    sample_cond_mix_coeffs,
                    sample_marginal_quantiles,
                )

        self.is_isotropic = self.metalogs_variant.is_isotropic

        if self.is_isotropic and not self.rusinkiewicz_reparam:
            mi.Log(
                mi.LogLevel.Warn,
                "Material is isotropic (n_phi_i == 1) but Rusinkiewicz reparameterization is disabled.",
            )

        if self.metalog_marginal == "phi":
            self.phi_bounds = self.marginal_bounds
        else:
            self.phi_bounds = self.conditional_bounds

        # initialize directional-hemispherical-reflectance
        mi.Log(
            mi.LogLevel.Info,
            "Initialize directional-hemispherical-reflectance for evaluating BRDF.",
        )
        self.dhr_texture, shape = (
            self.init_directional_hemispherical_reflectance(
                self.directional_hemispherical_reflectance_filename
            )
        )
        self.dhr_tensor = self.dhr_texture.tensor()

        # initialize spatial base color
        self.base_color_texture = self.init_spatial_base_color()
        self.base_color_tensor = self.base_color_texture.tensor()

    def init_directional_hemispherical_reflectance(
        self, filename: str
    ) -> tuple[mi.Texture, tuple]:
        import numpy as np

        # load the EXR file into a bitmap
        bitmap = mi.Bitmap(filename)

        np_img = np.array(bitmap)
        width = np_img.shape[1]

        if width == 1:
            """Case: isotropy."""
            padded_tensor = mi.TensorXf(np_img)
        else:
            """
            Pad the texture to avoid discontinuties.
            Extract the first column for all channels.
            Pad by appending the first column to the far right.
            Convert back to mitsuba tensor.
            """
            first_col = np_img[:, 0:1, :]
            padded_np_img = np.concatenate([np_img, first_col], axis=1)
            padded_tensor = mi.TensorXf(padded_np_img)

        """
        Configure wrap modes.
            - Axis 0 (Width/Phi): "repeat", since phi wraps -pi to pi
            - Axis 1 (Height/Theta), "clamp", since theta stops at 0 and pi/2
        Following does not work: 
            `wrap_modes = [dr.WrapMode.Repeat, dr.WrapMode.Clamp]`
        Therefore, we pad the texture - copy the first column on to the last
        colum (far right side). Mitsuba will blend the original right edge with
        the padded column when necessary.
        """
        dhr_texture = mi.Texture2f(
            padded_tensor,
            filter_mode=dr.FilterMode.Linear,
            wrap_mode=dr.WrapMode.Clamp,
            use_accel=True,
        )

        return dhr_texture, padded_tensor.shape

    def init_spatial_base_color(self) -> mi.Texture:
        """
        If `self.base_color_filename` is provided, then load and returns the texture.
        Otherwise, returns a 3 channel resxres texture. If `self.init_base_color_texture`, then res=512, else res=1.

        This is a convenience function to initialize a texture w.r.t. UV coordinates of the mesh that can be optimized
        in a differentiable rendering setting.
        """
        import numpy as np

        # if filename is provided, load the texture
        if self.base_color_filename is not None:
            bitmap = mi.Bitmap(self.base_color_filename)
            init_color_tensor = mi.TensorXf(bitmap)
        else:
            # initialize blank white texture
            if self.init_base_color_texture:
                res = 512
            else:
                res = 1

            init_color = np.full((res, res, 3), 1, dtype=np.float32)
            init_color_tensor = mi.TensorXf(init_color)

        texture = mi.Texture2f(
            init_color_tensor,
            filter_mode=dr.FilterMode.Linear,
            wrap_mode=dr.WrapMode.Repeat,
            use_accel=True,
        )

        return texture

    def traverse(self, cb):
        self.metalog_sampler.traverse(cb)
        cb.put(
            "dhr_tensor",
            self.dhr_tensor,
            mi.ParamFlags.Differentiable,
        )
        if self.init_base_color_texture:
            cb.put(
                "base_color_tensor",
                self.base_color_tensor,
                mi.ParamFlags.Differentiable,
            )

    def parameters_changed(self, keys=...):
        if "dhr_tensor" in keys or len(keys) == 0:
            self.dhr_texture.update_inplace()
        if self.init_base_color_texture and (
            "base_color_tensor" in keys or len(keys) == 0
        ):
            self.base_color_texture.update_inplace()

    def to_string(self):
        return (
            "MetaBRDF[\n"
            "    metalog-term=%s,\n"
            "    metalog-marginal=%s,\n"
            "    sample-metalog-filename=%s,\n"
            "    directional-hemispherical-reflectance-filename=%s,\n"
            "    meta-material-type=%s,\n"
            "    cdf-method=%s,\n"
            "    torch-device=%s,\n"
            "    metalogs-variant=%s,\n"
            "    marginal-mode=%s,\n"
            "    rusinkiewicz_reparam=%s,\n"
            "]"
            % (
                self.metalog_term,
                self.metalog_marginal,
                self.metalogs_filename,
                self.directional_hemispherical_reflectance_filename,
                self.m_meta_material_type,
                self.cdf_method,
                self.torch_device,
                self.metalogs_variant,
                self.marginal_mode,
                self.rusinkiewicz_reparam,
            )
        )

    def get_reflection_flags(self):
        if self.m_meta_material_type == MetaMaterialTypes.META_ROUGHCONDUCTOR:
            return mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        elif self.m_meta_material_type == MetaMaterialTypes.META_DIFFUSE:
            return mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        elif self.m_meta_material_type == MetaMaterialTypes.META_EPFL_MEASURED:
            return mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        else:
            raise ValueError(
                f"Unexpected meta material type: {self.meta_material_type}"
            )

    def is_ctx_enabled(self, ctx):
        if self.m_meta_material_type == MetaMaterialTypes.META_ROUGHCONDUCTOR:
            return ctx.is_enabled(mi.BSDFFlags.GlossyReflection)
        elif self.m_meta_material_type == MetaMaterialTypes.META_DIFFUSE:
            return ctx.is_enabled(mi.BSDFFlags.DiffuseReflection)
        elif self.m_meta_material_type == MetaMaterialTypes.META_EPFL_MEASURED:
            return ctx.is_enabled(mi.BSDFFlags.GlossyReflection)
        else:
            raise ValueError(
                f"Unexpected meta material type: {self.meta_material_type}"
            )

    def get_sampled_type(self):
        if self.m_meta_material_type == MetaMaterialTypes.META_ROUGHCONDUCTOR:
            return mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        elif self.m_meta_material_type == MetaMaterialTypes.META_DIFFUSE:
            return mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
        elif self.m_meta_material_type == MetaMaterialTypes.META_EPFL_MEASURED:
            return mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        else:
            raise ValueError(
                f"Unexpected meta material type: {self.meta_material_type}"
            )

    def read_metalog_scene_props(self, props):
        super().read_metalog_scene_props(props)

        # read halfway vector reparameterization (default True)
        self.rusinkiewicz_reparam = props.get("rusinkiewicz_reparam", True)

        meta_material_type = props.get("meta_material_type", "None")
        # check meta material type validity
        try:
            self.m_meta_material_type = MetaMaterialTypes(meta_material_type)
        except ValueError:
            raise ValueError(
                f"{meta_material_type} is not a valid MetaMaterialTypes."
            )

        # check base color texture flag (used for diff. rendering example)
        self.init_base_color_texture = props.get(
            "init_base_color_texture", False
        )

        # read optional base color filename
        base_color_filename = props.get("base_color_filename", None)
        self.base_color_filename = (
            None if base_color_filename == "None" else base_color_filename
        )
