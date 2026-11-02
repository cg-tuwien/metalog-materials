# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import math
import torch
from torch import Tensor
from lib.fit.mixture_models.mixture_model import FitResult, MixtureModel1D
from lib.metalog_strategies.metalog_strategy import (
    MetalogsStrategy,
    TorchBoundedMetalogs,
    TorchUnboundedMetalogs,
)
import lib.metalogs as ml


class MetalogMixtureModel1D(MixtureModel1D):
    metalog_coeffs: torch.Tensor
    bounds: tuple[float, float] | None
    metalog_strategy: MetalogsStrategy
    metalog_term: int
    make_probs_method: str = "classic"

    """These are stored only during training/fitting"""
    X_sorted: Tensor
    sort_idx: Tensor

    def rank_kernel_weights(
        self,
        N: int,
        K: int,
        rho: float = 0.5,
        kind: str = "triangular",
    ) -> torch.Tensor:
        """
        Builds KxN weights in rank, or cumulative probability space telling each component which slice of the sorted
        data to focus on.

        Details:
        1. Initialize K component centers at c_k = (k + 0.5) / K. For example, when K=2, then c_0 = 0.25, c_1 = 0.75.
        2. Assign each data point an empirical CDF rank r_i = (i + 0.5) / N, i = 0, ..., N-1.
        3. For each component k, place a kernel centered at c_k in [0, 1] using a bandwith of rho / K.
        4. Normalize each row such that weights for a component sum to 1.

        This makes initialization scale-free, robust to outliers and gives each component rouhgly a 1/K slice of the
        ordered data.

        Args:
            N (int): Number of data points.
            K (int): Number of mixture components.
            rho (float, optional): Overlap control in probability space. We compute the bandwith by rho / K. Larger rho
            leads to more overlap, smaller rho leads to more localized weights. Defaults to 0.5.
            kind (str, optional):  Kernel shape: "triangular" (compact support) or "gaussian" (infinite support).
            Defaults to "triangular".

        Returns:
            torch.Tensor: Row-normalized weight matrix (K, N) for each component over to weight quantiles. See also
            `weighted_quantiles()`.
        """
        # center for each component in probability space: (k + 0.5) / K
        k = torch.arange(K, device=self.device)
        centers = (k + 0.5) / K  # (K, )

        # bandwidth in probability space
        bandwidth = rho / K

        N_i = torch.arange(N, device=self.device)
        r = ((N_i + 0.5) / N).unsqueeze(0)  # (1, N)
        centers = centers.unsqueeze(1)  # (K, 1)

        if kind == "triangular":
            d = (r - centers).abs() / bandwidth
            W = (1.0 - d).clamp_min(0.0)
        else:
            d = (r - centers) / (bandwidth + 1e-12)
            W = torch.exp(-0.5 * d * d)

        return W / (W.sum(dim=1, keepdim=True) + self.eps)

    def weighted_quantiles(
        self,
        W_sorted: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted empirical quantiles for all components at once.

        Args:
            W_sorted: weights per component over *sorted* data, shape (K, N).
            probs: probabilities to extract, shape (R).

        Returns:
            Q: quantiles per component, shape (K, R).
        """
        return ml.weighted_quantiles(self.X_sorted, W_sorted, probs)

    def initialize_parameters(self, X: Tensor):
        """
        Rank-kernel initializer:
          - compute per-component weights in probability space,
          - call `initialize_parameters_concrete()`
        """
        N = X.numel()
        K = self.K

        # store sorted data and indices required to sort the data
        self.X_sorted, self.sort_idx = torch.sort(X.view(-1))  # N

        # initialize mixing weights equally, which sum to 1 (Eq. 3.112), see
        # also the book of Bishop 2023, Deep Learning
        self.mixing_coeffs = torch.full((K,), 1.0 / K, device=self.device)

        # rank-kernel weights for all components, shape: (K, N)
        W_sorted = self.rank_kernel_weights(N, K, rho=0.5, kind="gaussian")

        self.initialize_parameters_concrete(W_sorted)

    def initialize_parameters_concrete(self, W_sorted: Tensor):
        raise NotImplementedError("Subclasses responsibility.")

    def eval_pdf(self, X):
        """
        NOTE: Current implementation of `unbounded_metalog_quantile()` expects
        `a` in [N, T] and `y` in [N]. As a result, giving an `X` in
        [n_samples] with `a` in [n_components, n_terms=3] does not work. The
        following is a possible way to avoid the for loop, but if N is large
        X_rep becomes too heavy:
            ```
            K, N = self.n_components, X.size(0)
            # build cartesian (KxN) as a flattened paired batch (K*N)
            coeffs_rep = self.spt_metalogs_coeffs.repeat_interleave(N, dim=0) (KxN, 3)
            X_rep = X.repeat(K) # (KxN, )
            _, pdf_flat = self.metalog_strategy.eval_metalog_cdf_pdf(coeffs_rep, X_rep)
            pdf = pdf_flat.view(K, N).T.contiguous() # (N, K)
            return pdf
            ```
        """
        pdf_values = torch.empty((X.shape[0], self.K), device=self.device)

        # for each component compute metalog PDFs
        for k in range(self.K):
            coeffs_k = self.metalog_coeffs[k].unsqueeze(0)
            cdf, pdf = self.metalog_strategy.eval_metalog_cdf_pdf(coeffs_k, X)
            pdf_values[:, k] = pdf

        return pdf_values

    def eval_log_pdf(self, X):
        return torch.log(self.eval_pdf(X).clamp_min(self.eps))

    @torch.no_grad()
    def maximization_step(self, X: Tensor, responsibilities: Tensor) -> None:
        """
        - π_k via Nk/N (Eq. 15.21 from the book of Bishop),
        - Per k: normalize responsibilities to weights over data,
        - call `maximization_step_concrete()`
        """
        N, K = responsibilities.shape

        # π_k = Nk / N (Eq. 15.21 from the book of Bishop)
        Nk = responsibilities.sum(dim=0).clamp_min(self.eps)  # (K, )
        pi = (Nk + (self.dirichlet_prior - 1.0)).clamp_min(self.eps)
        self.mixing_coeffs = (pi / pi.sum()).clamp_min(self.eps)

        # normalize responsibilities column-wise to get weights per component
        W = responsibilities / Nk.clamp_min(self.eps)  # (N, K)

        # handle near-empty components with uniform weights over data
        near_empty = Nk <= self.empty_component_frac * N  # (K, )
        if near_empty.any():
            W[:, near_empty] = 1.0 / N

        # get sorted weights
        W_sorted_T = W[self.sort_idx, :].T  # (K, N)

        self.maximization_step_concrete(W_sorted_T)

    def maximization_step_concrete(self, W_sorted_T: Tensor):
        raise NotImplementedError("Subclasses responsibility.")

    def fit(self, X: torch.Tensor) -> FitResult:
        result = super().fit(X)

        # sort components
        self.sort_components()

        # clear sorted data
        self.X_sorted, self.sort_idx = None, None

        return result

    def sample(self, unif: Tensor) -> Tensor:
        N, K = unif.size(0), self.K

        # choose a component via mixture weights π
        cumulative_π = torch.cumsum(self.mixing_coeffs, dim=0)
        k = torch.searchsorted(cumulative_π, unif, right=True).clamp_max(K - 1)

        # map u to the conditional uniform inside the chosen component's interval
        #    u_cond = (u - C_{k-1}) / π_k, where C_{-1} := 0
        c_prev = torch.cat([cumulative_π.new_zeros(1), cumulative_π[:-1]])  # K
        u_cond = (unif - c_prev[k]) / (self.mixing_coeffs[k] + self.eps)  # N

        # gather metalog coefficients for each chosen component and sample
        coeffs = self.metalog_coeffs[k]
        samples = self.metalog_strategy.eval_metalog_quantile(coeffs, u_cond)

        return samples

    def eval_mixture_cdf(self, X: Tensor) -> Tensor:
        X = X.to(self.device).view(-1)
        N = X.numel()

        mix_cdf = torch.zeros((N,), device=self.device)
        for k in range(self.K):
            coeffs_k = self.metalog_coeffs[k].unsqueeze(0)
            cdf_k = self.metalog_strategy.eval_metalog_cdf(coeffs_k, X)
            mix_cdf += self.mixing_coeffs[k] * cdf_k.squeeze(0)

        return mix_cdf.clamp(0.0, 1.0)

    def eval_mixture_quantile(
        self, probs: torch.Tensor, max_iter: int = 30, tol: float = 1e-4
    ) -> torch.Tensor:
        p = probs.to(self.device)

        # Build scalar bracket [lo, hi] shared by all p
        if self.bounds is not None:
            # keep a tiny gap away from hard bounds
            lo_s = self.bounds[0] + self.eps
            hi_s = self.bounds[1] - self.eps
        else:
            # Use component quantiles to bracket the mixture quantile
            u_lo = torch.full((self.K,), self.eps, device=self.device)
            u_hi = torch.full((self.K,), 1.0 - self.eps, device=self.device)
            q_lo_k = self.metalog_strategy.eval_metalog_quantile(
                self.metalog_coeffs, u_lo
            )
            q_hi_k = self.metalog_strategy.eval_metalog_quantile(
                self.metalog_coeffs, u_hi
            )
            lo_s = q_lo_k.min().item()
            hi_s = q_hi_k.max().item()

            # add a small padding
            pad = 0.05 * max(1e-8, hi_s - lo_s)
            lo_s -= pad
            hi_s += pad

        # initialize brackets for the bisection
        lo = torch.full_like(p, lo_s)
        hi = torch.full_like(p, hi_s)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            cdf_mid = self.eval_mixture_cdf(mid)
            go_right = cdf_mid < p
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)

            if torch.max(hi - lo) <= tol:
                break

        return 0.5 * (lo + hi)

    def sort_components(self):
        """
        Reoreder components based on their median (0.5 quantile) to ensure consistency across different slices for
        interpolation.
        """

        # evaluate the median
        medians = self.metalog_strategy.eval_metalog_quantile(
            self.metalog_coeffs, torch.full((self.K,), 0.5, device=self.device)
        )

        # get sorted indices wrt medians
        sort_idx = torch.argsort(medians)

        # permute the parameters
        self.metalog_coeffs = self.metalog_coeffs[sort_idx]
        self.mixing_coeffs = self.mixing_coeffs[sort_idx]

    def is_feasible(self) -> bool:
        """
        Checks if all components in the mixture model are feasible.
        """
        for k in range(self.metalog_coeffs.shape[0]):
            if not ml.is_feasible_metalog(
                self.metalog_coeffs[k].unsqueeze(0), bounds=self.bounds
            ):
                return False
        return True


class MetalogMixtureModel(MetalogMixtureModel1D):
    """
    EM for a K-component mixture whose components are n-term metalogs (n >= 3).

    - Unbounded: fit quantiles directly.
    - Bounded: fit in Keelin's logit space logit(x) = ln((x-bl)/(bu-x)) (Sec. 4.3).
    - M-step: per-component weighted-quantile -> linear solve for 'a' (n terms).
    - Feasibility: enforce m(y) > 0 for all y in (0,1) approximately by grid test (Keelin Eq. 5 / Eq. 10). If violated,
    damp (line search) and/or increase lambda.
    """

    def __init__(
        self,
        K: int = 2,
        metalog_term: int = 6,
        max_iters: int = 100,
        tol: float = 1e-6,
        bounds: tuple[float, float] | None = None,
        lambda_: float = 1e-4,
        resolution: int = 16,
        feas_max_backtracks: int = 10,
        dirichlet_prior: float = 1.0,
        device: str | None = None,
        make_probs_method: str = "classic",
    ):
        super().__init__(
            K=K,
            max_iters=max_iters,
            tol=tol,
            lambda_=lambda_,
            dirichlet_prior=dirichlet_prior,
            device=device,
        )
        assert metalog_term >= 3, "Metalog requires at least 3 terms"
        self.metalog_term = int(metalog_term)
        self.bounds = bounds
        self.resolution = int(max(resolution, metalog_term))
        self.feasibility_max_backtracks = int(feas_max_backtracks)
        self.make_probs_method = make_probs_method

        self.metalog_strategy: MetalogsStrategy = (
            TorchUnboundedMetalogs()
            if bounds is None
            else TorchBoundedMetalogs(bounds)
        )

        # component parameters: (K, T)
        self.metalog_coeffs = torch.empty(
            (self.K, self.metalog_term), device=self.device
        )

        # cache for init/M-step
        self.X_sorted = None
        self.sort_idx = None

    @staticmethod
    def build_Y(y: Tensor, n_terms: int) -> Tensor:
        """
        Construct Keelin Metalog design matrix Y_n (Definition 1), Keelin 2016.

        Columns: [1, L, (y-0.5)L, (y-0.5), (y-0.5)^(3/2), (y-0.5)L, ...] pattern:
          - n=2: 1, L
          - n=3: + (y-0.5) L
          - n=4: + (y-0.5)
          - odd n>=5: + (y-0.5)^(n-1)/2
          - even n>=6: + (y-0.5)^{n/2 - 1} * L

        where L = ln(y/(1-y)).

        Note: Similar to Y construction in `ml.make_unbounded_metalog()`.
        """
        eps = 1e-12
        L = torch.log(y.clamp_min(eps) / (1.0 - y).clamp_min(eps))
        t = y - 0.5
        cols = [torch.ones_like(y), L]

        if n_terms >= 3:
            cols.append(t * L)

        if n_terms >= 4:
            cols.append(t)

        for j in range(5, n_terms + 1):
            if j % 2 == 1:
                # odd -> t**((j-1)/2)
                pow = (j - 1) // 2
                cols.append(t.pow(pow))
            else:
                # even -> t**(j/2 - 1) * L
                pow = j // 2 - 1
                cols.append(t.pow(pow) * L)

        return torch.stack(cols, dim=1)

    def logit_metalog_transform(self, x: Tensor) -> Tensor:
        """
        Keelin 2016, Sec. 4.3, logit metalog transform: z = logit(x) = ln((x - b_l)/(b_u - x))
        """
        eps = 1e-12
        return torch.log(
            ((x - self.bounds[0]).clamp_min(eps))
            / ((self.bounds[1] - x).clamp_min(eps))
        )

    def transform_quantiles(self, q: Tensor) -> Tensor:
        """
        If unbounded, the transform is identity (i.e., return q), else use `logit_metalog_transform()`. See also Keelin
        2016, Sec. 4.3.
        """
        if self.bounds is None:
            return q
        return self.logit_metalog_transform(q)

    def initialize_parameters_concrete(self, W_sorted: Tensor):
        # probability grid for linear solve
        y = ml.make_probs(self.resolution, self.make_probs_method, self.device)

        # build Metalog Y_n matrix (Def. 1, Keelin 2016), shape: (R, T)
        Y = self.build_Y(y, self.metalog_term)

        """
        Modify the normal equation to include the regularization term mentioned
        by Tom Keelin, 3.2. Metalog Panel. See also `make_unbounded_metalogs()`.
        """
        ridge = self.lambda_ * torch.diag_embed(
            torch.tensor(
                [0.0, 1e-6] + [1.0] * (self.metalog_term - 2),
                device=self.device,
            )
        )

        # compute weighted quantiles
        Q = self.weighted_quantiles(W_sorted, y)

        for k in range(self.K):
            q = Q[k]
            t = self.transform_quantiles(q)

            # solve for metalog coefficients: (Y^T Y + ridge) a = Y^T t
            A = Y.T @ Y + ridge
            b = Y.T @ t
            a = torch.linalg.solve(A, b)

            # basic a2 positivity nudge (helps tails)
            if self.metalog_term >= 2 and a[1] <= 0:
                a[1] = a[1].abs() + 1e-6

            a = ml.feasibilize_metalog_coeffs(a.unsqueeze(0))
            self.metalog_coeffs[k] = a

    def maximization_step_concrete(self, W_sorted_T: Tensor):
        """
        Build the Y matrix (Def. 1, Keelin 2016), compute weighted quantiles and solve the linear system for 'a_k' +
        ridge/lambda and enforce feasibility.
        """
        K, N = W_sorted_T.shape

        # build Metalog Y_n matrix (Def. 1, Keelin 2016), shape: (R, T)
        y = ml.make_probs(self.resolution, self.make_probs_method, self.device)
        Y = self.build_Y(y, self.metalog_term)

        """
        Modify the normal equation to include the regularization term mentioned
        by Tom Keelin, 3.2. Metalog Panel. See also `make_unbounded_metalogs()`.
        """
        ridge = self.lambda_ * torch.diag_embed(
            torch.tensor(
                [0.0, 1e-6] + [1.0] * (self.metalog_term - 2),
                device=self.device,
            )
        )

        # compute weighted quantiles for all components
        Q = self.weighted_quantiles(W_sorted_T, y)

        if self.bounds is not None:
            margin = 1e-4 * (self.bounds[1] - self.bounds[0])
            Q = Q.clamp(self.bounds[0] + margin, self.bounds[1] - margin)

        for k in range(K):
            q = Q[k]
            t = self.transform_quantiles(q)

            # solve for metalog coefficients: (Y^T Y + ridge) a = Y^T t
            A = Y.T @ Y + ridge
            b = Y.T @ t
            a_old = (
                self.metalog_coeffs[k]
                if torch.isfinite(self.metalog_coeffs[k]).all()
                else None
            )

            try:
                a_new = torch.linalg.solve(A, b)
            except:
                a_new = torch.linalg.lstsq(A, b).solution

            # basic a2 positivity nudge (helps tails)
            if self.metalog_term >= 2 and a_new[1] <= 0:
                a_new[1] = a_new[1].abs() + 1e-6

            self.metalog_coeffs[k] = ml.feasibilize_metalog_coeffs(
                a_new.unsqueeze(0), a_old=a_old.unsqueeze(0)
            )
