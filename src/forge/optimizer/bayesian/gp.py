"""Gaussian Process regression with ARD kernel optimisation."""

import math
from typing import Optional

import numpy as np

from .kernels import _EPS, _NUGGET, rbf_kernel_ard


class _GaussianProcess:
    """
    Manual Gaussian Process regressor with ARD-RBF kernel.

    Implements Cholesky-based GP inference without external ML libraries.
    Supports noise-free and noisy observations via nugget regularisation.
    """

    def __init__(self, length_scales: Optional[np.ndarray] = None) -> None:
        """
        Initialize the GP model.

        Args:
            length_scales: Per-dimension length scales. If None, defaults to 1.0
                           for all dimensions (set on first fit).
        """
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self.length_scales = length_scales
        self._dims: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GP to training data via Cholesky decomposition.

        Args:
            X: Training inputs, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64)).flatten()

        n, d = X.shape
        self._dims = d

        if self.length_scales is None or len(self.length_scales) != d:
            self.length_scales = np.ones(d, dtype=np.float64)

        K = rbf_kernel_ard(X, X, self.length_scales)
        K += _NUGGET * np.eye(n, dtype=np.float64)

        try:
            self._L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-3 * np.eye(n, dtype=np.float64)
            self._L = np.linalg.cholesky(K)

        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._X_train = X
        self._y_train = y

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior mean and variance at test points.

        Args:
            X: Test inputs, shape (n_samples, n_features).

        Returns:
            Tuple of (mu, sigma^2) each of shape (n_samples,).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        if self._X_train is None or self._alpha is None:
            raise RuntimeError("GP must be fit before predict()")

        K_s = rbf_kernel_ard(self._X_train, X, self.length_scales)
        mu = K_s.T @ self._alpha

        v = np.linalg.solve(self._L, K_s)
        sigma2 = np.ones(len(X), dtype=np.float64) - np.sum(v ** 2, axis=0)
        sigma2 = np.maximum(sigma2, _EPS)

        return mu, sigma2


class _GaussianProcessARD(_GaussianProcess):
    """
    Gaussian Process with Automatic Relevance Determination and length scale optimisation.

    Extends _GaussianProcess with gradient-ascent optimisation of per-dimension
    length scales via marginal likelihood maximisation.  Learns which parameters
    (total_cores vs omp_threads vs mode) are most relevant to walltime.
    """

    def __init__(
        self,
        length_scales: Optional[np.ndarray] = None,
        learning_rate: float = 0.05,
        n_opt_steps: int = 50,
        min_length_scale: float = 0.1,
        max_length_scale: float = 10.0,
    ) -> None:
        super().__init__(length_scales=length_scales)
        self._learning_rate = learning_rate
        self._n_opt_steps = n_opt_steps
        self._min_ls = min_length_scale
        self._max_ls = max_length_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        super().fit(X, y)
        self._optimize_length_scales()

    def _optimize_length_scales(self) -> None:
        if self._X_train is None or self._y_train is None or len(self._y_train) < 3:
            return

        X = self._X_train
        y = self._y_train.reshape(-1, 1)
        n, d = X.shape
        current_ls = self.length_scales.copy()
        best_ls = current_ls.copy()
        best_nll = float("inf")

        for _step in range(self._n_opt_steps):
            K = rbf_kernel_ard(X, X, current_ls)
            K += _NUGGET * np.eye(n, dtype=np.float64)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(n, dtype=np.float64)
                L = np.linalg.cholesky(K)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))

            quad = float((y.T @ alpha).item())
            nll = 0.5 * quad + float(np.sum(np.log(np.diag(L)))) + \
                  0.5 * n * math.log(2.0 * math.pi)
            if nll < best_nll:
                best_nll = nll
                best_ls = current_ls.copy()

            grad = np.zeros(d, dtype=np.float64)
            outer = alpha @ alpha.T - Kinv

            for j in range(d):
                ls_j = max(current_ls[j], _EPS)
                diff = (X[:, j].reshape(-1, 1) - X[:, j].reshape(1, -1)) ** 2
                dK = K * (diff / (ls_j ** 3))
                grad[j] = 0.5 * float(np.trace(outer @ dK)) / ls_j

            grad = np.clip(grad, -1.0, 1.0)
            current_ls = current_ls - self._learning_rate * grad
            current_ls = np.clip(current_ls, self._min_ls, self._max_ls)

        self.length_scales = best_ls.copy()

        K = rbf_kernel_ard(X, X, self.length_scales)
        K += _NUGGET * np.eye(n, dtype=np.float64)
        try:
            self._L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-3 * np.eye(n, dtype=np.float64)
            self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y.flatten()))

    def get_relevance(self) -> dict[str, float]:
        """
        Return per-dimension relevance scores.

        Shorter length scale → higher relevance (the parameter matters more).

        Returns:
            Dict mapping dimension names to relevance scores (0-1, higher = more relevant).
        """
        if self.length_scales is None:
            return {}
        names = ["total_cores", "omp_threads", "mode_kpoint", "mode_hybrid", "mode_mpi"]
        inv_ls = 1.0 / np.maximum(self.length_scales, _EPS)
        norm = float(np.sum(inv_ls)) if np.sum(inv_ls) > 0 else 1.0
        return {names[i]: round(float(inv_ls[i] / norm), 4) for i in range(min(len(names), len(self.length_scales)))}


__all__ = ["_GaussianProcess", "_GaussianProcessARD"]
