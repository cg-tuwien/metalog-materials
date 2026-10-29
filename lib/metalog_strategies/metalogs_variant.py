# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import lib.metalogs as ml
import mitsuba as mi
import torch
from abc import ABC, abstractmethod
from typing import Tuple


class MetalogsVariant(ABC):

    def __init__(self):
        super().__init__()
        self.make_probs_method = "classic"

    @abstractmethod
    def sample_metalog(
        self,
        sample2: mi.Point2f,
    ) -> Tuple[mi.Point2f, mi.Float]:
        pass

    @abstractmethod
    def sample_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        pass

    @abstractmethod
    def sample_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        pass

    @abstractmethod
    def sample_nd_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor | mi.TensorXf,
        marginal_quantiles: torch.Tensor | mi.TensorXf,
        conditional_metalogs: torch.Tensor | mi.TensorXf,
    ) -> Tuple[mi.Point2f, mi.Float]:
        pass

    @abstractmethod
    def sample_nd_marginal_mixture_metalog(
        self,
        sample2: mi.Point2f,
        wi2D: mi.Vector2f,
        marginal_mix_spt_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def eval_metalog_pdf(self, w: mi.Point2f) -> Tuple[mi.Vector2f, mi.Float]:
        pass

    @abstractmethod
    def eval_mixture_metalog_pdf(
        self,
        w: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_mix_metalog_coeffs: mi.TensorXf,
        cond_mix_coeffs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        pass

    @abstractmethod
    def eval_marginal_mixture_metalog_pdf(
        self,
        w: mi.Point2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        cond_metalogs: mi.TensorXf,
    ) -> Tuple[mi.Vector2f, mi.Float]:
        pass

    @abstractmethod
    def eval_nd_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ) -> tuple[mi.Point2f, mi.Float]:
        pass

    @abstractmethod
    def eval_nd_marginal_mixture_metalog_pdf(
        self,
        w2D: mi.Vector2f,
        wi2D: mi.Vector2f,
        marginal_mix_metalog_coeffs: mi.TensorXf,
        marginal_mix_coeffs: mi.TensorXf,
        marginal_quantiles: mi.TensorXf,
        conditional_metalogs: mi.TensorXf,
    ) -> tuple[mi.Point2f, mi.Float]:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def init_metalogs(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        self.metalogs = ml.init_metalogs(
            metalogs_filename,
            metalog_term,
            marginal_bounds,
            torch_device,
            self.make_probs_method,
        )
        self.metalog_resolution = self.metalogs["marginal-quantiles"].size(0)
        self.metalogs.pop("marginal-quantile-cdfs", None)  # NOTE: not used
        self.metalog_term = metalog_term
        self.torch_device = torch_device
        self.marginal_bounds = marginal_bounds
        self.conditional_bounds = conditional_bounds
        self.metalog_marginal = metalog_marginal

        self.check_metalogs_feasibility(self.metalogs)

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
        self.metalog_term = cond_metalog_term
        self.torch_device = torch_device
        self.metalog_marginal = metalog_marginal
        self.marginal_bounds = marginal_bounds
        self.conditional_bounds = conditional_bounds

        loaded = torch.load(
            metalogs_filename,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        self.marginal_quantile_cdf_range = loaded["marginal_quantile_cdf_range"]
        marginal_quantiles = loaded["marginal_inverse_cdf"].to(
            self.torch_device
        )

        mm = loaded["marginal_mixture"]
        marg_mix_weights = mm["mixing_coeffs"].to(self.torch_device)
        marg_mix_metalog_coeffs = mm["coeffs"].to(self.torch_device)

        # assemble conditional tensors

        cm = loaded["conditional"]
        _, _, cond_max_term = cm.shape
        if cond_metalog_term > cond_max_term:
            raise ValueError(
                f"{cond_metalog_term}-term metalog is not available."
                f"The loaded conditional metalogs has maximum {cond_max_term} term."
            )

        cond_metalogs = ml.get_metalog_coefficients(cm, cond_metalog_term).to(
            torch_device
        )

        self.check_marginal_quantiles(marginal_quantiles)
        self.check_marginal_mixture_metalog_model_feasibility(
            marg_mix_weights, marg_mix_metalog_coeffs
        )
        self.check_conditional_metalogs_feasibility(cond_metalogs)

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
        self.torch_device = torch_device
        self.metalog_marginal = metalog_marginal
        self.marginal_bounds = marginal_bounds
        self.conditional_bounds = conditional_bounds

        loaded = torch.load(
            metalogs_filename,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        self.marginal_quantile_cdf_range = loaded["marginal_quantile_cdf_range"]
        marginal_quantiles = loaded["marginal_inverse_cdf"].to(
            self.torch_device
        )

        mm = loaded["marginal_mixture"]
        marg_mix_weights = mm["mixing_coeffs"].to(self.torch_device)
        marg_mix_metalog_coeffs = mm["coeffs"].to(self.torch_device)

        cm = loaded["conditional_mixture"]
        cond_mix_weights = cm["mixing_coeffs"].to(self.torch_device)
        cond_mix_metalog_coeffs = cm["coeffs"].to(self.torch_device)

        self.check_marginal_quantiles(marginal_quantiles)
        self.check_marginal_mixture_metalog_model_feasibility(
            marg_mix_weights, marg_mix_metalog_coeffs
        )
        self.check_conditional_mixture_metalog_model_feasibility(
            cond_mix_weights, cond_mix_metalog_coeffs
        )

        return (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_mix_weights,
            cond_mix_metalog_coeffs,
        )

    def check_metalogs_feasibility(self, metalogs: dict):

        # check if marginal inverse CDF is monotonically increase
        inverse_cdf = metalogs["marginal-quantiles"]
        if not torch.all(inverse_cdf[:-1] <= inverse_cdf[1:]):
            raise ValueError(
                f"Marginal inverse CDF values is not monotonically increasing."
            )

        # check if marginal metalogs are feasible
        marginal_metalogs = metalogs["marginal"]
        is_marginal_metalog_feasible = ml.is_feasible_metalog(marginal_metalogs)
        if not is_marginal_metalog_feasible:
            raise ValueError(
                f"{self.metalog_term}-Term marginal metalog is not feasible."
            )

        self.check_conditional_metalogs_feasibility(metalogs["conditional"])

    def check_conditional_metalogs_feasibility(self, conditional_metalogs):
        # check if conditional metalogs are feasible
        metalog_resolution = conditional_metalogs.size(0)
        for res in range(metalog_resolution):
            res_conditional_metalogs = conditional_metalogs[res].unsqueeze(0)
            is_cond_metalog_feasible = ml.is_feasible_metalog(
                res_conditional_metalogs
            )
            if not is_cond_metalog_feasible:
                raise ValueError(
                    f"(Resolution {res}) {self.metalog_term}-Term conditional metalog is not feasible."
                )

    def init_marginal_metalog_slices(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        self.metalog_term = metalog_term
        self.torch_device = torch_device
        self.marginal_bounds = marginal_bounds
        self.metalog_marginal = metalog_marginal

        metalogs, _, _ = ml.init_metalogs_nd(
            metalogs_filename,
            self.metalog_term,
            self.marginal_bounds,
            self.torch_device,
        )

        keys_list = list(metalogs.keys())
        self.metalogs_keys = torch.tensor(keys_list).to(self.torch_device)
        self.n_phi_i = torch.unique(self.metalogs_keys[:, 0]).size(0)
        self.n_theta_i = torch.unique(self.metalogs_keys[:, 1]).size(0)
        self.is_isotropic = self.n_phi_i == 1
        assert self.is_isotropic, "Loaded metalogs indicate anisotropy."

        n_slices = len(keys_list)
        marginal_metalogs = torch.zeros((n_slices, 1, self.metalog_term)).to(
            self.torch_device
        )
        for i, key in enumerate(keys_list):
            marginal_metalogs[i] = metalogs[key]["marginal"]

        marginal_metalogs = marginal_metalogs.squeeze(
            1
        )  # shape: [n_slices, metalog_term]

        self.check_nd_marginal_metalogs_feasability(marginal_metalogs)

        return marginal_metalogs

    def init_metalog_slices(
        self,
        metalogs_filename: str,
        metalog_term: int,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        self.metalog_term = metalog_term
        self.torch_device = torch_device
        self.marginal_bounds = marginal_bounds
        self.conditional_bounds = conditional_bounds
        self.metalog_marginal = metalog_marginal

        metalogs, self.metalog_resolution, self.marginal_quantile_cdf_range = (
            ml.init_metalogs_nd(
                metalogs_filename,
                self.metalog_term,
                self.marginal_bounds,
                self.torch_device,
            )
        )

        keys_list = list(metalogs.keys())
        self.metalogs_keys = torch.tensor(keys_list).to(self.torch_device)
        self.n_phi_i = torch.unique(self.metalogs_keys[:, 0]).size(0)
        self.n_theta_i = torch.unique(self.metalogs_keys[:, 1]).size(0)
        self.is_isotropic = self.n_phi_i == 1

        n_slices = len(keys_list)
        marginal_metalogs = torch.zeros((n_slices, 1, self.metalog_term)).to(
            self.torch_device
        )
        for i, key in enumerate(keys_list):
            marginal_metalogs[i] = metalogs[key]["marginal"]

        marginal_metalogs = marginal_metalogs.squeeze(
            1
        )  # shape: [n_slices, metalog_term]

        metalog_resolution = metalogs[key]["marginal-quantiles"].size(0)
        marginal_quantiles = torch.zeros((n_slices, 1, metalog_resolution)).to(
            self.torch_device
        )
        for i, key in enumerate(keys_list):
            marginal_quantiles[i] = metalogs[key]["marginal-quantiles"]

        marginal_quantiles = marginal_quantiles.squeeze(
            1
        )  # shape: [n_slices, metalog_resolution]

        n = metalogs[key]["conditional"].size(0)
        conditional_metalogs = torch.zeros((n_slices, n, self.metalog_term)).to(
            self.torch_device
        )
        for i, key in enumerate(keys_list):
            conditional_metalogs[i] = metalogs[key]["conditional"]

        self.check_nd_metalogs_feasability(
            marginal_metalogs, marginal_quantiles, conditional_metalogs
        )

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
        self.metalog_term = cond_metalog_term
        self.torch_device = torch_device
        self.metalog_marginal = metalog_marginal
        self.marginal_bounds = marginal_bounds
        self.conditional_bounds = conditional_bounds

        loaded = torch.load(
            metalogs_filename,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        keys_list = list(loaded.keys())
        self.metalogs_keys = torch.tensor(keys_list).to(self.torch_device)
        self.n_phi_i = torch.unique(self.metalogs_keys[:, 0]).size(0)
        self.n_theta_i = torch.unique(self.metalogs_keys[:, 1]).size(0)
        self.is_isotropic = self.n_phi_i == 1

        # infer resolution R and terms from the first slice
        first = loaded[keys_list[0]]
        R = first["conditional"].size(0)
        coeffs_label = "coeffs"
        T = first["marginal_mixture"]["coeffs"].size(1)
        K = int(first["marginal_mixture"]["K"])

        # assemble tensors
        n_slices = len(keys_list)
        marginal_quantiles = torch.empty(
            (n_slices, R), device=self.torch_device
        )
        conditional_metalogs = torch.empty(
            (n_slices, R, self.metalog_term), device=self.torch_device
        )
        mix_weights = torch.empty((n_slices, K), device=self.torch_device)
        mix_metalog_coeffs = torch.empty(
            (n_slices, K, T), device=self.torch_device
        )

        # assume a single CDF-grid range
        self.marginal_quantile_cdf_range = first["marginal_quantile_cdf_range"]

        for i, key in enumerate(keys_list):
            ek = loaded[key]
            marginal_quantiles[i] = ek["marginal_inverse_cdf"].to(
                self.torch_device
            )

            # extract only requested conditional metalog coefficients
            cond_metalogs = ek["conditional"]
            _, _, cond_max_term = cond_metalogs.shape
            if cond_metalog_term > cond_max_term:
                raise ValueError(
                    f"{cond_metalog_term}-term metalog is not available."
                    f"The loaded conditional metalogs has maximum {cond_max_term} term."
                )
            cond_metalogs = ml.get_metalog_coefficients(
                cond_metalogs, cond_metalog_term
            ).to(torch_device)
            conditional_metalogs[i] = cond_metalogs

            mm = ek["marginal_mixture"]
            # sanity check
            assert (
                int(mm["K"]) == K
            ), "Per-slice K mismatch in marginal mixture."

            mix_weights[i] = mm["mixing_coeffs"].to(self.torch_device)
            mix_metalog_coeffs[i] = mm[coeffs_label].to(self.torch_device)

        self.check_nd_marginal_quantiles(marginal_quantiles)
        self.check_nd_conditional_metalogs_feasability(conditional_metalogs)
        self.check_nd_marginal_mixture_metalog_model_feasibility(
            mix_weights, mix_metalog_coeffs
        )

        return (
            marginal_quantiles,
            conditional_metalogs,
            mix_weights,
            mix_metalog_coeffs,
        )

    def init_bivariate_mixture_slices(
        self,
        metalogs_filename: str,
        marginal_bounds: tuple,
        conditional_bounds: tuple,
        metalog_marginal: str,
        torch_device: str,
    ):
        self.torch_device = torch_device
        self.metalog_marginal = metalog_marginal
        self.marginal_bounds = marginal_bounds
        self.conditional_bounds = conditional_bounds

        loaded = torch.load(
            metalogs_filename,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        keys_list = list(loaded.keys())
        self.metalogs_keys = torch.tensor(keys_list).to(self.torch_device)
        self.n_phi_i = torch.unique(self.metalogs_keys[:, 0]).size(0)
        self.n_theta_i = torch.unique(self.metalogs_keys[:, 1]).size(0)
        self.is_isotropic = self.n_phi_i == 1

        # get the first slice
        first = loaded[keys_list[0]]

        # assume a single CDF-grid range
        self.marginal_quantile_cdf_range = first["marginal_quantile_cdf_range"]

        # infer resolution R, conditional metalog term / components
        coeffs_label = "coeffs"
        R, K_cond, T_cond = first["conditional_mixture"][coeffs_label].shape

        # infer marginal metalog term and marginal components
        T_marg = first["marginal_mixture"][coeffs_label].size(1)
        K_marg = int(first["marginal_mixture"]["K"])

        # assemble tensors
        n_slices = len(keys_list)
        marginal_quantiles = torch.empty(
            (n_slices, R), device=self.torch_device
        )
        marg_mix_weights = torch.empty(
            (n_slices, K_marg), device=self.torch_device
        )
        marg_mix_metalog_coeffs = torch.empty(
            (n_slices, K_marg, T_marg), device=self.torch_device
        )
        cond_mix_weights = torch.empty(
            (n_slices, R, K_cond), device=self.torch_device
        )
        cond_mix_metalog_coeffs = torch.empty(
            (n_slices, R, K_cond, T_cond), device=self.torch_device
        )

        for i, key in enumerate(keys_list):
            ek = loaded[key]
            marginal_quantiles[i] = ek["marginal_inverse_cdf"].to(
                self.torch_device
            )

            mm = ek["marginal_mixture"]

            marg_mix_weights[i] = mm["mixing_coeffs"].to(self.torch_device)
            marg_mix_metalog_coeffs[i] = mm[coeffs_label].to(self.torch_device)

            cm = ek["conditional_mixture"]

            cond_mix_weights[i] = cm["mixing_coeffs"].to(self.torch_device)
            cond_mix_metalog_coeffs[i] = cm[coeffs_label].to(self.torch_device)

        self.check_nd_marginal_quantiles(marginal_quantiles)
        self.check_nd_marginal_mixture_metalog_model_feasibility(
            marg_mix_weights, marg_mix_metalog_coeffs
        )
        self.check_nd_conditional_mixture_metalog_model_feasibility(
            cond_mix_weights, cond_mix_metalog_coeffs
        )

        return (
            marginal_quantiles,
            marg_mix_weights,
            marg_mix_metalog_coeffs,
            cond_mix_weights,
            cond_mix_metalog_coeffs,
        )

    def check_marginal_mixture_metalog_model_feasibility(
        self, mix_weights, mix_metalog_coeffs
    ):
        # for each slice check if all mixing coefficients sum up to one
        mix_weights_sum = mix_weights.sum(dim=0)
        if not torch.allclose(
            mix_weights_sum,
            torch.ones_like(mix_weights_sum),
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError("Not all mixing coeffiecients sum up to 1.")

        self.check_marginal_mixture_metalogs_feasability(
            mix_weights, mix_metalog_coeffs
        )

    def check_nd_marginal_mixture_metalog_model_feasibility(
        self, mix_weights, mix_metalog_coeffs
    ):
        # for each slice check if all mixing coefficients sum up to one
        mix_weights_sum = mix_weights.sum(dim=1)
        if not torch.allclose(
            mix_weights_sum,
            torch.ones_like(mix_weights_sum),
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError("Not all mixing coeffiecients sum up to 1.")

        self.check_nd_marginal_mixture_metalogs_feasability(
            mix_weights, mix_metalog_coeffs
        )

    def check_conditional_mixture_metalog_model_feasibility(
        self, mix_weights, mix_metalog_coeffs
    ):
        # for each slice for each conditional check if all mixing coefficients
        # sum up to one
        mix_weights_sum = mix_weights.sum(dim=1)
        if not torch.allclose(
            mix_weights_sum,
            torch.ones_like(mix_weights_sum),
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError("Not all mixing coeffiecients sum up to 1.")

        self.check_conditional_mixture_metalogs_feasability(
            mix_weights, mix_metalog_coeffs
        )

    def check_nd_conditional_mixture_metalog_model_feasibility(
        self, mix_weights, mix_metalog_coeffs
    ):
        # for each slice for each conditional check if all mixing coefficients
        # sum up to one
        mix_weights_sum = mix_weights.sum(dim=2)
        if not torch.allclose(
            mix_weights_sum,
            torch.ones_like(mix_weights_sum),
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError("Not all mixing coeffiecients sum up to 1.")

        self.check_nd_conditional_mixture_metalogs_feasability(
            mix_weights, mix_metalog_coeffs
        )

    def check_nd_metalogs_feasability(
        self,
        marginal_metalogs: torch.Tensor,
        marginal_quantiles: torch.Tensor,
        conditional_metalogs: torch.Tensor,
    ):
        self.check_nd_marginal_quantiles(marginal_quantiles)
        self.check_nd_marginal_metalogs_feasability(marginal_metalogs)
        self.check_nd_conditional_metalogs_feasability(conditional_metalogs)

    def check_marginal_quantiles(self, marginal_quantiles):
        # for each slice check if marginal inverse CDF is monotonically increase
        if not torch.all(marginal_quantiles[:-1] <= marginal_quantiles[1:]):
            raise ValueError(
                f"(Slice {slice}) Marginal inverse CDF values is not monotonically increasing."
            )

    def check_nd_marginal_quantiles(self, marginal_quantiles):
        # for each slice check if marginal inverse CDF is monotonically increase
        n_slices, metalog_resolution = marginal_quantiles.shape
        for slice in range(n_slices):
            slice_quantiles = marginal_quantiles[slice]
            if not torch.all(slice_quantiles[:-1] <= slice_quantiles[1:]):
                raise ValueError(
                    f"(Slice {slice}) Marginal inverse CDF values is not monotonically increasing."
                )

    def check_nd_conditional_metalogs_feasability(self, conditional_metalogs):
        # for each slice check if conditional metalogs are feasible
        n_slices, metalog_resolution, _ = conditional_metalogs.shape
        for slice in range(n_slices):
            slice_conditional_metalogs = conditional_metalogs[slice]
            for res in range(metalog_resolution):
                res_conditional_metalogs = slice_conditional_metalogs[
                    res
                ].unsqueeze(0)
                is_cond_metalog_feasible = ml.is_feasible_metalog(
                    res_conditional_metalogs
                )
                if not is_cond_metalog_feasible:
                    raise ValueError(
                        f"(Slice {slice}, Resolution {res}) {self.metalog_term}-Term conditional metalog is not feasible."
                    )

    def check_marginal_mixture_metalogs_feasability(
        self, mix_weights, mix_metalogs
    ):
        # even though Proposition 2 exist for SPT Metalogs feasibility, for
        # simplicity we use equation 10

        # for each component check if metalogs are feasible
        K, _ = mix_metalogs.shape
        for k in range(K):
            # skip padded/ignored components
            if mix_weights[k] <= 1e-8:
                continue
            metalogs_k = mix_metalogs[k].unsqueeze(0)
            is_metalog_k_feasible = ml.is_feasible_metalog(metalogs_k)
            if not is_metalog_k_feasible:
                raise ValueError(f"(Component {k}) Metalog is not feasible.")

    def check_nd_marginal_mixture_metalogs_feasability(
        self, mix_weights, mix_metalogs
    ):
        # even though Proposition 2 exist for SPT Metalogs feasibility, for
        # simplicity we use equation 10

        # for each slice check if SPT metalogs are feasible
        n_slices, K, _ = mix_metalogs.shape
        for slice in range(n_slices):
            slice_weights = mix_weights[slice]
            slice_metalogs = mix_metalogs[slice]
            for k in range(K):
                # skip padded/ignored components
                if slice_weights[k] <= 1e-8:
                    continue

                slice_metalogs_k = slice_metalogs[k].unsqueeze(0)
                is_metalog_k_feasible = ml.is_feasible_metalog(slice_metalogs_k)
                if not is_metalog_k_feasible:
                    raise ValueError(
                        f"(Slice {slice}, Component {k}) Metalog is not feasible."
                    )

    def check_conditional_mixture_metalogs_feasability(
        self, mix_weights, mix_metalogs
    ):
        # even though Proposition 2 exist for SPT Metalogs feasibility, for
        # simplicity we use equation 10

        # for each slice check if SPT metalogs are feasible
        R, K, _ = mix_metalogs.shape
        for r in range(R):
            weight_r = mix_weights[r]
            metalogs_r = mix_metalogs[r]
            for k in range(K):
                # skip padded/ignored components
                if weight_r[k] <= 1e-8:
                    continue
                metalogs_rk = metalogs_r[k].unsqueeze(0)
                is_metalog_k_feasible = ml.is_feasible_metalog(metalogs_rk)
                if not is_metalog_k_feasible:
                    raise ValueError(
                        f"(Conditional {r}, Component {k}) Metalog is not feasible."
                    )

    def check_nd_conditional_mixture_metalogs_feasability(
        self, mix_weights, mix_metalogs
    ):
        # even though Proposition 2 exist for SPT Metalogs feasibility, for
        # simplicity we use equation 10

        # for each slice check if SPT metalogs are feasible
        n_slices, R, K, _ = mix_metalogs.shape
        for slice in range(n_slices):
            slice_weights = mix_weights[slice]
            slice_metalogs = mix_metalogs[slice]
            for r in range(R):
                slice_weights_r = slice_weights[r]
                slice_metalogs_r = slice_metalogs[r]
                for k in range(K):
                    # skip padded/ignored components
                    if slice_weights_r[k] <= 1e-8:
                        continue

                    slice_metalogs_k = slice_metalogs_r[k].unsqueeze(0)
                    is_metalog_k_feasible = ml.is_feasible_metalog(
                        slice_metalogs_k
                    )
                    if not is_metalog_k_feasible:
                        raise ValueError(
                            f"(Slice {slice}, Conditional {r}, Component {k}) Metalog is not feasible."
                        )

    def check_nd_marginal_metalogs_feasability(self, marginal_metalogs):
        # for each slice check if marginal metalogs are feasible
        n_slices, _ = marginal_metalogs.shape
        for slice in range(n_slices):
            slice_marginal_metalogs = marginal_metalogs[slice].unsqueeze(0)
            is_marginal_metalog_feasible = ml.is_feasible_metalog(
                slice_marginal_metalogs
            )
            if not is_marginal_metalog_feasible:
                raise ValueError(
                    f"(Slice {slice}) {self.metalog_term}-Term marginal metalog is not feasible."
                )
