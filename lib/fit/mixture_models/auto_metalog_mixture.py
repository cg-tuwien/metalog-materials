# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import logging
import math
import torch
from lib.fit.mixture_models.metalog_mixture_model import (
    MetalogMixtureModel,
    MetalogMixtureModel1D,
)
from lib.fit.mixture_models.mixture_model import FitResult
from lib.fit.mixture_models.spt_metalog_mixture_model import (
    SPTMetalogMixtureModel,
)


class AutoMetalogMixture1D:
    """
    Model selector that composes `AutoMetalogMixture1D` and choses K (i.e., number of components).
    It tries K in range `[min_k, max_k]` with restarts and picks the best by criterion.
    It delegates `eval_pdf` / `eval_mixture_pdf` / `sample` to the best model.
    """

    EPS: float = 1e-12

    # available `criterion`
    WASSERSTEIN1 = "wasserstein1"
    AIC = "AIC"
    BIC = "BIC"
    ICL = "ICL"
    CRITERIONS = [WASSERSTEIN1, AIC, BIC, ICL]

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 4,
        metalog_term: int = 5,
        resolution_min: int = 6,
        resolution_max: int = 32,
        criterion: str = WASSERSTEIN1,
        restarts_per_k: int = 1,
        max_iters: int = 100,
        tol: float = 1e-6,
        bounds: tuple[float, float] | None = None,
        alpha: float = 0.3,
        k_penalty: float = 0.0,
        lambda_min: float = 0.0,
        lambda_max: float = 0.0,
        lambda_steps: int = 3,
        device: str | None = None,
        make_probs_method: str = "classic",
        dirichlet_prior: float = 1.0,
    ):
        if criterion not in self.CRITERIONS:
            logging.warning(
                f"Criterion `{criterion}` is not available. "
                f"Using default `{self.WASSERSTEIN1}`. Available are: `{self.CRITERIONS}`."
            )
            criterion = self.WASSERSTEIN1

        self.min_k = min_k
        self.max_k = max_k
        self.metalog_term = max(3, int(metalog_term))
        self.resolution_min = max(resolution_min, 4)
        self.resolution_max = min(resolution_max, 256)
        self.criterion = criterion
        self.restarts_per_k = restarts_per_k
        self.bounds = bounds
        self.alpha = alpha  # used only for SPT
        self.dirichlet_prior = dirichlet_prior

        # parameters of MetalogMixtureModel1D
        self.model_kwargs = dict(
            max_iters=max_iters,
            tol=tol,
            bounds=bounds,
            dirichlet_prior=self.dirichlet_prior,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model = None
        self.best_k = None
        self.best_term = None
        self.best_lambda = None
        self.best_resolution = None
        self.best_score = None
        self.scores = {}  # {K: score}
        self.scores_4d = {}  # {(K, T, L, R): score}
        self.k_penalty = k_penalty
        self.make_probs_method = make_probs_method
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # set up lambda grid in log-scale
        if self.lambda_max > self.lambda_min and lambda_steps > 1:
            # ensure strictly positive values
            l_min = max(self.lambda_min, 1e-12)
            l_max = max(self.lambda_max, 1e-12)
            base_min = math.log10(l_min)
            base_max = math.log10(l_max)
            self.lambda_grid = torch.logspace(
                base_min, base_max, lambda_steps, device=self.device
            ).tolist()
        else:
            # single value
            self.lambda_grid = [self.lambda_min]

        # set up resolution grid
        if self.resolution_max > self.resolution_min:
            exponent = torch.arange(2, 9)
            resolutions = 2**exponent
            self.resolution_grid = torch.clamp(
                resolutions, self.resolution_min, self.resolution_max
            ).tolist()
            self.resolution_grid = list(set(self.resolution_grid))
        else:
            # single value
            self.resolution_grid = [self.resolution_min]

    def fit_once(
        self, X: torch.Tensor, K: int, T: int, L: float, R: int
    ) -> tuple[MetalogMixtureModel1D, FitResult]:
        if T == 3:
            m = SPTMetalogMixtureModel(
                n_components=K, alpha=self.alpha, **self.model_kwargs
            )
        else:
            m = MetalogMixtureModel(
                K=K,
                resolution=R,
                metalog_term=T,
                lambda_=L,
                make_probs_method=self.make_probs_method,
                **self.model_kwargs,
            )

        em_history = m.fit(X)
        return m, em_history

    def log_likelihood(
        self, model: MetalogMixtureModel1D, X: torch.Tensor
    ) -> float:
        # reuse mixture PDF then sum logs
        mix_pdf = model.eval_mixture_pdf(X).clamp_min(self.EPS)
        return torch.log(mix_pdf).sum().item()

    def parameter_count(self, K: int, T: int) -> int:
        # T-Terms per component and (K-1) mixture weights
        return T * K + (K - 1)

    def bic(self, log_likelihood: float, N: int, K: int, T: int) -> float:
        p = self.parameter_count(K, T)
        return -2.0 * log_likelihood + p * math.log(N)

    def aic(self, log_likelihood: float, K: int, T: int) -> float:
        p = self.parameter_count(K, T)
        return -2.0 * log_likelihood + 2.0 * p

    def classification_entropy(self, model, X: torch.Tensor) -> float:
        """
        Entropy is:

            H(gamma_n) = - \\sum_{k=1}^K gamma_{nk} log(gamma_{nk}),

        where gamma_{nk} are the responsibilities (posterior component probabilities) which is computed in the E-step.
        """

        # E-step with the fitted parameters to get gamma_{nk}
        with torch.no_grad():
            responsibilities, _ = model.expectation_step(X)  # (N, K)
            responsibilities = responsibilities.clamp_min(self.EPS)
            class_entropy = (
                -(responsibilities * torch.log(responsibilities)).sum().item()
            )

        return class_entropy

    def icl(
        self, log_likelihood: float, N: int, K: int, T: int, entropy: float
    ) -> float:
        """
        BIC plus a penalty for soft assignments:

            BIC + 2 \\sum_{n=1}^N H(gamma_n),

        where H is the entropy:

            H(gamma_n) = - \\sum_{k=1}^K gamma_{nk} log(gamma_{nk}),

        and gamma_{nk} are the responsibilities (posterior component probabilities) which is computed in the E-step.
        """
        p = self.parameter_count(K, T)
        bic = -2.0 * log_likelihood + p * math.log(N)
        return bic + 2.0 * entropy

    def wasserstein1_ecdf(
        self,
        model: MetalogMixtureModel1D,
        X_sorted: torch.Tensor,
        grid_n: int = 2048,
    ) -> float:
        """
        Compute the 1-Wasserstein distance between an empirical distribution and a metalog mixture model using the CDF
        integral form:

            W1 = \\int |F_emp(x) - F_model(x)| dx.

        We estimate this integral by sampling both CDFs on a 1D grid and integrating with the trapezoid rule.

        See also https://en.wikipedia.org/wiki/Wasserstein_metric#One-dimensional_distributions.

        Args:
            model (MetalogMixtureModel1D): Metalog mixture model to compute the mixture CDF.
            X_sorted (torch.Tensor):1D tensor of sample data sorted in ascending order to compute the empirical CDF.
            grid_n (int, optional): Number of grid points for numerical integration. Defaults to 2048.

        Returns:
            float: The estimated 1-Wasserstein distance.
        """
        N = X_sorted.numel()

        # choose integration range
        if self.bounds != None:
            lo, hi = map(
                lambda t: torch.tensor(t, device=self.device), self.bounds
            )
        else:
            lo, hi = X_sorted[0], X_sorted[-1]
            span = (hi - lo).abs().clamp_min(1e-8)
            lo, hi = lo - 0.05 * span, hi + 0.05 * span

        # half of the points for a uniform grid
        uniform_grid = torch.linspace(lo, hi, grid_n // 2, device=self.device)

        # other half are sampled evenly from the empirical data indices
        idx = torch.linspace(0, N - 1, grid_n // 2, device=self.device).long()
        quantile_grid = X_sorted[idx]

        # combine, sort and ensure unique values for the grid
        grid = torch.cat([uniform_grid, quantile_grid])
        grid = torch.sort(grid).values
        grid = torch.unique(grid)

        # model CDF and empirical CDF on the same grid
        model_cdf = model.eval_mixture_cdf(grid)
        counts = torch.searchsorted(X_sorted, grid, right=True)
        empirical_cdf = counts.float() / float(N)

        # evaluate Wasserstein 1 distance
        w1 = torch.trapezoid((empirical_cdf - model_cdf).abs(), grid).item()

        return w1

    @torch.no_grad
    def fit(
        self,
        X: torch.Tensor,
        val_split: float = 0.2,
        n_trials: int | None = None,
    ) -> list[FitResult]:
        """
        Fit randomly sampled configurations of K, T, L and R, computing the log-likelihood and penalizing complexity
        according to BIC, AIC, ICL or Wasserstein1.

        BIC, AIC, or ICL:
            - BIC = -2 log_likelihood_k + p log N (see also `bic()`),
            - AIC = -2 log_likelihood_k + 2 p (see also `aic()`),
            - ICL = BIC + 2 gamma_{n=1}^N H(gamma_n), ("BIC plus a penalty for soft assignments")

        Here p is the parameter count. For 3-term SPT components p per K is:

            p = 3K + (K - 1) = 4K - 1,

        where we have 3 SPT metalog coefficients per component and (K-1) independent mixture weights. H is:

            H(gamma_n) = - gamma_{k=1}^K gamma_{nk} log(gamma_{nk}),

        where gamma_{nk} are the responsibilities (posterior component probabilities) which is computed in the E-step.

        If criterion is `wasserstein1`, we perform a single-fold cross-validation:
            1. Train on (1 - val_split) of data.
            2. Score on `val_split` of data.
            3. After finding best K/T, refit best configuration on FULL data.

        If criterion is `bic` / `aic` / `icl`:
            1. Train on full data.
            2. Score on full data using analytical penalty terms.

        Args:
            X (torch.Tensor): The 1D data tensor.
            val_split (float): Ratio of validation data (used if `criterion="wasserstein1"`).
            n_trials (int | None): Maximum number of random hyperparameter combinations to try. If none, falls back to a
            full grid search.
        """
        import random

        X = X.to(self.device).view(-1)

        x_full_sorted = X.sort().values
        q25 = torch.quantile(x_full_sorted, 0.25)
        q75 = torch.quantile(x_full_sorted, 0.75)
        iqr = (q75 - q25).clamp_min(1e-8).item()

        X_train, X_val_sorted = self.split_dataset(X, val_split)

        N_train = X_train.shape[0]
        self.scores.clear()
        self.scores_4d.clear()
        (
            self.best_model,
            self.best_k,
            self.best_term,
            self.best_lambda,
            self.best_resolution,
        ) = (None, None, None, None, None)
        self.best_score = float("inf")

        fit_results: list[FitResult] = []

        # build the full parameter space
        all_configs = []
        for K in range(self.min_k, self.max_k + 1):
            for T in range(3, self.metalog_term + 1):
                if T == 3:
                    # SPT ignores L and R, only add it once per K
                    all_configs.append(
                        (K, T, self.lambda_grid[0], self.resolution_grid[0])
                    )
                else:
                    for L in self.lambda_grid:
                        for R in self.resolution_grid:
                            all_configs.append((K, T, L, R))

        # randomly sample from the parameter space
        if n_trials is not None and n_trials < len(all_configs):
            configs_to_eval = random.sample(all_configs, n_trials)
        else:
            configs_to_eval = all_configs

        # track the best score for each K independently
        best_scores_per_k = {
            k: float("inf") for k in range(self.min_k, self.max_k + 1)
        }

        # iterate through the chosen configurations
        for K, T, L, R in configs_to_eval:

            # find best restart on training data
            best_log_likelihood_ktlr, best_model_ktlr = -float("inf"), None
            i = 0

            while i < self.restarts_per_k:
                m, em_hist_k = self.fit_once(X_train, K, T, L, R)

                # only consider this model if it is feasible
                if m.is_feasible():
                    fit_results.append(em_hist_k)

                    log_likelihood = self.log_likelihood(m, X_train)
                    if log_likelihood > best_log_likelihood_ktlr:
                        (
                            best_log_likelihood_ktlr,
                            best_model_ktlr,
                        ) = (
                            log_likelihood,
                            m,
                        )

                i += 1

            if best_model_ktlr is None:
                logging.info(
                    f"Skipping, could not fit model (K={K}, T={T}, L={L}, R={R})"
                )
                continue

            """Evaluate the score for the current configuration"""
            score = 0.0
            if self.criterion == self.BIC:
                score = self.bic(best_log_likelihood_ktlr, N_train, K, T)
                score += 2.0 * self.k_penalty * K * math.log(N_train)

            elif self.criterion == self.AIC:
                score = self.aic(best_log_likelihood_ktlr, K, T)
                score += 2.0 * self.k_penalty * K * math.log(N_train)

            elif self.criterion == self.ICL:
                entropy = self.classification_entropy(best_model_ktlr, X_train)
                score = self.icl(
                    best_log_likelihood_ktlr, N_train, K, T, entropy
                )
                score += 2.0 * self.k_penalty * K * math.log(N_train)

            elif self.criterion == self.WASSERSTEIN1:
                # Evaluate on validation set
                w1 = self.wasserstein1_ecdf(
                    best_model_ktlr, X_val_sorted, grid_n=2**16
                )

                # Normalize by global IQR
                score = w1 / iqr
            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

            # track 4D score and best per-K score
            self.scores_4d[(K, T, L, R)] = score
            if score < best_scores_per_k[K]:
                best_scores_per_k[K] = score

            # update global best
            if score < self.best_score:
                self.best_score = score
                self.best_model = best_model_ktlr
                self.best_k = K
                self.best_term = T
                self.best_lambda = L
                self.best_resolution = R

        # drop any K values that failed entirely or were not sampled
        self.scores = {
            k: v for k, v in best_scores_per_k.items() if v != float("inf")
        }

        # refit step if cross-validation is used
        if self.criterion == self.WASSERSTEIN1 and self.best_model is not None:

            # use multiple restarts
            best_refit_ll = -float("inf")
            final_model = None

            attempts = 0
            while attempts < max(self.restarts_per_k, 1):
                m_refit, _ = self.fit_once(
                    X,
                    self.best_k,
                    self.best_term,
                    self.best_lambda,
                    self.best_resolution,
                )

                if m_refit.is_feasible():
                    ll_refit = self.log_likelihood(m_refit, X)
                    if ll_refit > best_refit_ll:
                        best_refit_ll = ll_refit
                        final_model = m_refit

                attempts += 1

            if final_model is not None:
                self.best_model = final_model
            else:
                logging.info(
                    "Refit failed to find a feasible model; keeping best validation model, if available."
                )

        if self.best_model is None:
            raise ValueError(
                f"Fit failed for all sampled combinations of K in [{self.min_k, self.max_k + 1}], "
                f"T in [3, {self.metalog_term + 1}], L in [{self.lambda_min}, {self.lambda_max}] "
                f"and R in [{self.resolution_min}, {self.resolution_max}]"
            )

        return fit_results

    def split_dataset(self, X: torch.Tensor, val_split: float) -> tuple:
        """
        Splits dataset into training and validation if `self.criterion == "wasserstein1"`.

        Args:
            X (torch.Tensor): Full data.
            val_split (float): Ratio of the validation set.

        Returns:
            tuple:
                - [0]: Training data according to the `val_split`, if `wasserstein1` is used, otherwise full data `X`.
                - [1]: Validation data according to the `val_split`, if `wasserstein1` is used, otherwise None.
        """
        N_full = X.shape[0]

        # split dataset / use cross-validation if `wasserstein1`
        use_cv = self.criterion == self.WASSERSTEIN1

        if use_cv:
            # shuffle and split
            perm = torch.randperm(N_full, device=self.device)
            n_val = int(N_full * val_split)

            # ensure at least minimal samples in both
            n_val = max(1, min(n_val, N_full - 5))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            X_train = X[train_idx]
            X_val_sorted = X[val_idx].sort().values
        else:
            # we fit on full data otherwise
            X_train = X
            X_val_sorted = None

        return X_train, X_val_sorted

    def eval_pdf(self, X: torch.Tensor) -> torch.Tensor:
        return self.best_model.eval_pdf(X)

    def eval_mixture_pdf(self, X: torch.Tensor) -> torch.Tensor:
        return self.best_model.eval_mixture_pdf(X)

    def sample(self, unif: torch.Tensor) -> torch.Tensor:
        return self.best_model.sample(unif)

    def eval_mixture_cdf(self, X: torch.Tensor) -> torch.Tensor:
        return self.best_model.eval_mixture_cdf(X)

    def eval_mixture_quantile(
        self, probs: torch.Tensor, max_iter: int = 30, tol: float = 1e-4
    ) -> torch.Tensor:
        return self.best_model.eval_mixture_quantile(probs, max_iter, tol)
