# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import torch
from dataclasses import dataclass
from typing import List


@dataclass
class FitResult:
    log_likelihood_history: List[float]
    delta_history: List[float]


class MixtureModel1D:
    """
    This class runs EM for a 1D mixture model.
    Fitting structure follows Bishop's EM for Gaussian mixtures:

        1. E-step -> responsibilities gamma(znk).
        2. M-step -> re-estimate pik (mixing coeffiecients), and distribution parameters (e.g., muk, Sigmak^2).
        3. Track log-likelihood L(X) until convergence (Alg. 15.2 / Sec. 15.3).

    See also general EM formulation Q(theta, theta_old) in (15.23).
    See also the book of Bishop 2023, Deep Learning, Chapter 3.2.9 and Chapter
    15.2 and onwards.
    """

    """
    Setting Dirichlet prior (alpha) on the mixing coefficients pi = (pi_1, ..., pi_k),
    p(pi) is proportional to \\prod_{k=1}^K pi_k^{prior_k - 1} then the M-Step
    becomes a MAP update with pseudo-counts:
        pi_k \\propto N_k + (alpha_k - 1),
    where N_k = sum_n gamma_{nk} is the effective count from the E step. We set
    alpha_k = alpha for simplicity. When alpha=1, we recover pi_k = N_k / N,
    hence no effect. Setting alpha > 1 nudges pi toward uniform and prevents
    vanishing components. Setting alpha < 1 encourages sparsity, some pi_k may
    drop to 0.
    See also the book of Bishop 2023, Deep Learning, Chapter 15.3.
    """
    dirichlet_prior: float = 1.0

    def __init__(
        self,
        K=2,
        max_iters=100,
        tol=1e-6,
        lambda_=0,
        dirichlet_prior: float = 1.0,
        device: str | None = None,
    ):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_ = lambda_
        self.dirichlet_prior = dirichlet_prior

        # small positive floor to avoid division by 0 and log(0)
        self.eps = 1e-8

        # variance clamp to avoid collapse
        self.var_floor = 1e-8

        # heuristically "near-empty" if Nk/N <= this
        self.empty_component_frac = 1e-3

    def initialize_parameters(self, X):
        N = X.shape[0]

        """
        Randomly choose means from the data (simple heuristic).
        
        NOTE: Bishop suggests K-means initialization as a robust choice before
        EM. See also the K-means -> EM handoff discussion.
        """
        indices = torch.randperm(N)[: self.K]
        self.means = X[indices].clone().to(self.device)

        # initialize all component variances to the global data variance
        self.variances = torch.full((self.K,), torch.var(X), device=self.device)

        # initialize mixing weights equally, which  sum to 1 (constraint (3.112)
        self.mixing_coeffs = torch.full(
            (self.K,), 1 / self.K, device=self.device
        )

    def eval_pdf(self, X):
        # Implemented by subclasses (e.g., GaussianMixtureModel1D)
        raise NotImplementedError("Subclasses should implement this method.")

    def maximization_step(self, X, responsibilities):
        # Implemented by subclasses: should update p_ik, mu_k, Sigma_k^2 per
        # (15.21), (15.16), (15.18)
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(self, unif: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def fit(self, X: torch.Tensor) -> FitResult:
        """
        Implements the main loop for the expectation-maximization algorithm. See
        also (Alg. 15.2 / 15.3) from the book of Bishop 2023, Deep Learning.

        Args:
            X (torch.Tensor): Dataset.
        """
        X = X.to(self.device).float()

        self.initialize_parameters(X)

        log_likelihood_hist: list[float] = []
        delta_hist: list[float] = []

        old_log_likelihood = None
        converged = False
        for i in range(self.max_iters):
            responsibilities, log_marginal = self.expectation_step(X)
            self.maximization_step(X, responsibilities)

            # evaluate log-likelihood (Eq. 15.13)
            log_likelihood = log_marginal.sum()
            log_likelihood_hist.append(log_likelihood.detach().cpu().item())

            """
            Check for convergence: likelihood monotonicity guaranteed by EM
            until a local max.
            """
            if old_log_likelihood == None:
                delta_hist.append(float("nan"))
            else:
                # compute average log-likelihood
                N = X.shape[0]
                avg_log_likelihood = log_likelihood / N
                prev_avg_log_likelihood = old_log_likelihood / N

                # compare change in average log-likelihood
                abs_delta = torch.abs(
                    avg_log_likelihood - prev_avg_log_likelihood
                )
                delta = abs_delta.detach().cpu().item()
                delta_hist.append(delta)
                if delta < self.tol:
                    break

            old_log_likelihood = log_likelihood

        return FitResult(
            log_likelihood_history=log_likelihood_hist, delta_history=delta_hist
        )

    def expectation_step(self, X: torch.Tensor) -> tuple:
        """
        E-step: responsibilities gamma(znk), normalized over k. This implements the
        posterior responsibility formula, see also Eq. 15.29 from the book of
        Bishop 2023, Deep Learning.

        Args:
            X (torch.Tensor): Dataset.

        Returns:
            tuple: Normalized responsibilities and their over components.
        """

        # compute component PDFs for all points, shape: (N, K)
        if hasattr(self, "eval_log_pdf"):
            # prefer a log-PDF for numerical stability
            log_pdf = self.eval_log_pdf(X)
        else:
            # fall back if log eval_log_pdf() is not implemented
            pdf_values = self.eval_pdf(X).clamp_min(self.eps)
            log_pdf = torch.log(pdf_values)

        log_pi = torch.log(self.mixing_coeffs.clamp_min(self.eps)).unsqueeze(
            0
        )  # (1, K)
        log_joint = log_pdf + log_pi  # (N, K)

        # log-sum-exp gives log ∑_k pi_k N(x|mu_k,Sigma_k)  (Eq. 15.13)
        log_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)  # (N, 1)

        # compute the responsibilities gamma_{nk} = exp(log pi_k N - log ∑_j pi_j N)
        responsibilities = torch.exp(log_joint - log_marginal)  # (N, K)

        return responsibilities, log_marginal.squeeze(1)

    def eval_mixture_pdf(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the mixture density, see also Mixture definition Eq. 3.111
        from the book of Bishop 2023, Deep Learning.

        Args:
            X (torch.Tensor): Dataset.

        Returns:
            torch.Tensor: Mixture density for X.
        """
        pdf_values = self.eval_pdf(X)
        weighted_pdfs = pdf_values * self.mixing_coeffs
        return weighted_pdfs.sum(dim=1)

    def eval_mixture_cdf(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def eval_mixture_quantile(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")
