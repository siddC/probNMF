"""Test double for the base class: a rank-one Poisson model, X ~ Poisson(w hᵀ).

It is not a model of this package. It exists so the API contract, the estimator checks and
the six test layers can be exercised before any real model lands, and it stays as a fast
check of the harness itself. Fit by KL multiplicative updates over observed entries, which
never increase the negative log-likelihood.
"""

import numpy as np
from scipy.special import gammaln, xlogy
from sklearn.utils import check_random_state

from probnmf import BaseFactorization
from probnmf._core import EPS, ConvergenceMonitor, safe_divide


class ToyRankOne(BaseFactorization):
    def __init__(self, max_iter=500, tol=1e-10, random_state=None):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _weights(X, mask):
        return np.ones(X.shape) if mask is None else mask.astype(np.float64)

    @staticmethod
    def _nll(X, M, mu):
        return float(np.sum(M * (mu - xlogy(X, mu) + gammaln(X + 1.0))))

    def _fit(self, X, mask):
        X = X.astype(np.float64)
        M = self._weights(X, mask)
        rng = check_random_state(self.random_state)
        w = rng.uniform(0.5, 1.5, X.shape[0])
        h = rng.uniform(0.5, 1.5, X.shape[1])
        monitor = ConvergenceMonitor(tol=self.tol, max_iter=self.max_iter)
        while True:
            R = M * safe_divide(X, np.outer(w, h))
            h = h * (w @ R) / np.maximum(w @ M, EPS)
            R = M * safe_divide(X, np.outer(w, h))
            w = w * (R @ h) / np.maximum(M @ h, EPS)
            total = h.sum()
            if total > 0:
                h, w = h / total, w * total
            if monitor.update(self._nll(X, M, np.outer(w, h))):
                break
        self.components_ = h[None, :]
        self.n_components_ = 1
        self.latent_train_ = w.copy()
        self.objective_ = monitor.history
        self.n_iter_ = monitor.n_iter
        self.converged_ = monitor.converged
        return w[:, None]

    def _transform(self, X, mask):
        # closed-form maximum likelihood for one component with h held fixed
        M = self._weights(X, mask)
        h = self.components_[0]
        return ((M * X).sum(axis=1) / np.maximum(M @ h, EPS))[:, None]

    def _inverse_transform(self, Z):
        return Z @ self.components_

    def _score_samples(self, X, mask):
        X = X.astype(np.float64)
        M = self._weights(X, mask)
        mu = self._inverse_transform(self._transform(X, mask))
        return -np.sum(M * (mu - xlogy(X, mu) + gammaln(X + 1.0)), axis=1)

    def _sample(self, n_samples, random_state):
        w = random_state.choice(self.latent_train_, size=n_samples)
        return random_state.poisson(np.outer(w, self.components_[0])).astype(np.float64)


def simulate(seed, n=80, p=30, missing=0.1):
    """Rank-one Poisson counts with a random mask; returns (X, mask, true components)."""
    rng = np.random.default_rng(seed)
    h = rng.gamma(2.0, 1.0, p)
    h /= h.sum()
    w = rng.gamma(5.0, 100.0, n)
    X = rng.poisson(np.outer(w, h)).astype(np.float64)
    mask = rng.random((n, p)) >= missing
    return X, mask, h[None, :]


def stationarity(est, X, mask):
    """Gradient of the negative log-likelihood in h at the fitted point (w from transform)."""
    M = np.ones(X.shape) if mask is None else mask.astype(np.float64)
    h = est.components_[0]
    w = est.transform(X, mask=mask)[:, 0]
    mu = np.outer(w, h)
    positive = M.T @ w  # the gradient is positive part minus (M * X / mu)ᵀ w
    grad = positive - (M * safe_divide(X, mu)).T @ w
    return [(h, grad, True, float(positive.max()))]
