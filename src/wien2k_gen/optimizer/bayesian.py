"""
Bayesian Optimisation Framework for Tuning WIEN2k Parallel Execution Parameters.
Production features:
• Gaussian Process regression with manual RBF kernel (NumPy only, no scikit-learn)
• Expected Improvement (EI) acquisition function for exploration-exploitation trade-off
• Mixed parameter space: continuous (total_cores, omp_threads) + categorical (mode)
• One-hot encoding for categorical mode parameter with valid dimension handling
• History-driven warm-start via ExecutionHistory integration
• Regularised GP with jitter (nugget) for numerical stability on small datasets
• Adaptive length scales per dimension and Cholesky-based GP inference
• Full type hints, structured logging, and English documentation
All documentation and inline comments are in English per project standards.
"""

import math
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from ..logging_config import get_logger
from .history import ExecutionHistory, ExecutionRecord

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

_NUGGET = 1e-6       # Jitter for Cholesky numerical stability
_EPS = 1e-8           # Small epsilon for division safety
_CATEGORICAL_MODES = ["kpoint", "hybrid", "mpi"]


# =============================================================================
# Kernel & Acquisition Functions
# =============================================================================

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """
    Radial Basis Function (squared exponential) kernel for Gaussian Process.

    k(x1, x2) = exp(-0.5 * ||x1 - x2||^2 / length_scale^2)

    Args:
        x1: First input vector or matrix (shape (n,) or (n, d)).
        x2: Second input vector or matrix (shape (m,) or (m, d)).
        length_scale: Characteristic length scale determining smoothness.

    Returns:
        Kernel matrix of shape (n, m) or scalar for vector inputs.
    """
    x1 = np.atleast_1d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_1d(np.asarray(x2, dtype=np.float64))

    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)
    if x2.ndim == 1:
        x2 = x2.reshape(1, -1)

    sqdist = np.sum(x1 ** 2, axis=1).reshape(-1, 1) + \
             np.sum(x2 ** 2, axis=1) - \
             2.0 * np.dot(x1, x2.T)

    return np.exp(-0.5 * sqdist / max(length_scale ** 2, _EPS))


def rbf_kernel_ard(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scales: np.ndarray,
) -> np.ndarray:
    """
    RBF kernel with Automatic Relevance Determination (per-dimension length scales).

    k(x1, x2) = exp(-0.5 * sum_d ((x1_d - x2_d)^2 / length_scale[d]^2))

    Args:
        x1: First input matrix (shape (n, d)).
        x2: Second input matrix (shape (m, d)).
        length_scales: Per-dimension length scale vector (shape (d,)).

    Returns:
        Kernel matrix of shape (n, m).
    """
    x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
    length_scales = np.atleast_1d(np.asarray(length_scales, dtype=np.float64))

    d = x1.shape[1]
    K = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float64)
    for i in range(d):
        ls = max(length_scales[i], _EPS)
        diff = (x1[:, i].reshape(-1, 1) - x2[:, i].reshape(1, -1)) ** 2
        K -= 0.5 * diff / (ls ** 2)
    return np.exp(K)


def compute_expected_improvement(
    mu: float,
    sigma: float,
    best_y: float,
    xi: float = 0.01,
) -> float:
    """
    Expected Improvement acquisition function.

    EI(x) = sigma(x) * (z * Phi(z) + phi(z))
    where z = (mu(x) - best_y - xi) / sigma(x)
    and Phi, phi are the standard normal CDF and PDF respectively.

    For sigma == 0, returns 0.0 (no uncertainty = no improvement potential).

    Args:
        mu: Predicted mean at candidate point.
        sigma: Predicted standard deviation at candidate point.
        best_y: Best observed value so far (minimisation).
        xi: Exploration parameter (small positive value encourages exploration).

    Returns:
        Expected improvement value (non-negative).
    """
    if sigma < _EPS:
        return 0.0

    improvement = mu - best_y - xi
    z = improvement / sigma

    # Standard normal PDF and CDF
    pdf_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    ei = improvement * cdf_z + sigma * pdf_z
    return max(0.0, ei)


# =============================================================================
# Gaussian Process (Manual Implementation)
# =============================================================================

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
        self._L: Optional[np.ndarray] = None       # Lower-triangular Cholesky of K + sigma^2 I
        self._alpha: Optional[np.ndarray] = None    # K^{-1} y
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
            # Increase jitter on numerical failure
            K += 1e-3 * np.eye(n, dtype=np.float64)
            self._L = np.linalg.cholesky(K)

        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._X_train = X
        self._y_train = y

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        K_ss = rbf_kernel_ard(X, X, self.length_scales)

        # Posterior mean: K_s^T * alpha
        mu = K_s.T @ self._alpha

        # Posterior variance: K_ss - v^T v where v = L^{-1} K_s
        v = np.linalg.solve(self._L, K_s)
        cov = K_ss - v.T @ v

        # Extract diagonal and ensure non-negative
        sigma2 = np.diag(cov)
        sigma2 = np.maximum(sigma2, _EPS)

        return mu, sigma2


# =============================================================================
# Parameter Space Encoding
# =============================================================================

def _encode_config(mode: str, total_cores: int, omp_threads: int) -> np.ndarray:
    """
    Encode a configuration into a numeric feature vector.

    Encoding scheme:
    - total_cores:     normalised by dividing by 256.0
    - omp_threads:     normalised by dividing by 64.0
    - mode:            one-hot encoded (3 categories -> 3 features)

    Args:
        mode: Parallelisation mode ('kpoint', 'hybrid', 'mpi').
        total_cores: Total CPU cores.
        omp_threads: OpenMP threads per rank.

    Returns:
        Feature vector of shape (6,).
    """
    vec = np.zeros(2 + len(_CATEGORICAL_MODES), dtype=np.float64)
    vec[0] = float(total_cores) / 256.0
    vec[1] = float(omp_threads) / 64.0

    if mode in _CATEGORICAL_MODES:
        vec[2 + _CATEGORICAL_MODES.index(mode)] = 1.0

    return vec


def _decode_config(vec: np.ndarray, min_cores: int, max_cores: int) -> Dict[str, Any]:
    """
    Decode a feature vector back to a configuration dictionary.

    Returns:
        Dict with keys 'mode', 'total_cores', 'omp_threads'.
    """
    total_cores = max(min_cores, min(max_cores, int(round(vec[0] * 256.0))))
    omp_threads = max(1, min(64, int(round(vec[1] * 64.0))))

    one_hot = vec[2:2 + len(_CATEGORICAL_MODES)]
    mode_idx = int(np.argmax(one_hot))
    mode = _CATEGORICAL_MODES[mode_idx]

    return {
        "mode": mode,
        "total_cores": total_cores,
        "omp_threads": omp_threads,
    }


# =============================================================================
# BayesianOptimizer Class
# =============================================================================

class BayesianOptimizer:
    """
    Bayesian parameter optimiser for WIEN2k parallel execution tuning.

    Maintains a Gaussian Process surrogate model of the objective function
    (walltime) over the configuration space and suggests new configurations
    to try using the Expected Improvement acquisition criterion.

    Usage:
        history = ExecutionHistory()
        opt = BayesianOptimizer(history, backend="wien2k")
        suggestion = opt.suggest_next(nmat=5000, nkpt=4)
        # ... execute the suggested config and record the result ...
        record = ExecutionRecord(walltime_sec=elapsed, ...)
        opt.update(record)
    """

    def __init__(
        self,
        history: ExecutionHistory,
        backend: str = "wien2k",
        min_cores: int = 1,
        max_cores: int = 256,
        length_scales: Optional[np.ndarray] = None,
        exploration_xi: float = 0.01,
        n_random_restarts: int = 50,
    ) -> None:
        """
        Initialize the Bayesian optimiser.

        Args:
            history: ExecutionHistory instance for warm-starting from past runs.
            backend: Target DFT backend name.
            min_cores: Lower bound on total_cores.
            max_cores: Upper bound on total_cores (clamped by topology).
            length_scales: Per-dimension length scales for ARD kernel.
            exploration_xi: Exploration parameter for EI acquisition.
            n_random_restarts: Number of random restarts for global optimisation
                               of the acquisition function.
        """
        self._history = history
        self.backend = backend
        self.min_cores = min_cores
        self.max_cores = max_cores
        self._exploration_xi = exploration_xi
        self._n_random_restarts = n_random_restarts
        self._gp = _GaussianProcess(length_scales=length_scales)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._best_y: Optional[float] = None
        self._lock = threading.Lock()
        self._n_dims = 2 + len(_CATEGORICAL_MODES)

        self._warm_start()

    def _warm_start(self) -> None:
        """Seed the GP with data from the execution history."""
        records = self._history.query(
            filters={"backend": self.backend, "success": True},
            order_by="timestamp DESC",
            limit=200,
        )
        with self._lock:
            for rec in records:
                if rec.walltime_sec > 0:
                    self._add_observation_no_lock(rec)
            if self._X:
                self._refit_gp()
                logger.info(
                    f"Warm-started BayesianOptimizer with {len(self._X)} historical records"
                )

    def _add_observation_no_lock(self, record: ExecutionRecord) -> None:
        """Internal: add a record to the observation list without locking."""
        x = _encode_config(record.mode, record.total_cores, record.omp_threads)
        self._X.append(x)
        self._y.append(record.walltime_sec)

        if self._best_y is None or record.walltime_sec < self._best_y:
            self._best_y = record.walltime_sec

    def _refit_gp(self) -> None:
        """Re-fit the GP surrogate to all accumulated observations."""
        if len(self._X) < 2:
            return
        X_arr = np.array(self._X, dtype=np.float64)
        y_arr = np.array(self._y, dtype=np.float64)
        self._gp.fit(X_arr, y_arr)

    # =========================================================================
    # Public API
    # =========================================================================

    def update(self, record: ExecutionRecord) -> None:
        """
        Add a new observation (from a completed run) to the model.

        Args:
            record: ExecutionRecord from a completed (or failed) run.
        """
        with self._lock:
            self._add_observation_no_lock(record)
            self._refit_gp()
        logger.debug(
            f"BayesianOptimizer updated: best_y={self._best_y:.2f}s, n_obs={len(self._X)}"
        )

    def suggest_next(
        self,
        nmat: int,
        nkpt: int,
        user_max_cores: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Suggest the next configuration to evaluate using Expected Improvement.

        Performs a randomised search over the parameter space, evaluating
        the EI acquisition function at each candidate. Returns the config
        with the highest expected improvement.

        Args:
            nmat: Hamiltonian matrix size (for context, passed through).
            nkpt: Number of k-points (for context, passed through).
            user_max_cores: Override max_cores (e.g. from topology limit).

        Returns:
            Dictionary with keys:
            - mode: str
            - total_cores: int
            - omp_threads: int
            - expected_improvement: float
            - predicted_mean: float
            - predicted_std: float
            - source: str ('model' or 'random')
        """
        max_cores = self.max_cores
        if user_max_cores is not None:
            max_cores = min(self.max_cores, max(1, user_max_cores))

        with self._lock:
            n_obs = len(self._X)

            # If too few observations, return a random diverse suggestion
            if n_obs < 2:
                return self._random_suggestion(max_cores)

            current_best = self._best_y or float("inf")
            best_vec: Optional[np.ndarray] = None
            best_ei = -1.0
            best_mu = float("inf")
            best_sigma = 0.0

            rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))

            for _ in range(self._n_random_restarts):
                # Generate a random candidate
                cores = rng.randint(self.min_cores, max_cores + 1)
                omp = rng.randint(1, min(65, cores + 1))
                mode = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]

                candidate = _encode_config(mode, cores, omp)

                try:
                    mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                    mu_val = float(mu[0])
                    sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))

                    ei = compute_expected_improvement(
                        mu_val, sigma_val, current_best, xi=self._exploration_xi
                    )

                    if ei > best_ei:
                        best_ei = ei
                        best_vec = candidate.copy()
                        best_mu = mu_val
                        best_sigma = sigma_val
                except Exception:
                    continue

            if best_vec is not None:
                config = _decode_config(best_vec, self.min_cores, max_cores)
                return {
                    "mode": config["mode"],
                    "total_cores": config["total_cores"],
                    "omp_threads": config["omp_threads"],
                    "expected_improvement": round(best_ei, 6),
                    "predicted_mean": round(best_mu, 3),
                    "predicted_std": round(best_sigma, 3),
                    "source": "model",
                }

            return self._random_suggestion(max_cores)

    def _random_suggestion(self, max_cores: int) -> Dict[str, Any]:
        """Fallback: return a randomly generated plausible configuration."""
        rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))
        cores = rng.randint(self.min_cores, max_cores + 1)
        omp = rng.randint(1, min(65, cores + 1))
        mode = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]

        return {
            "mode": mode,
            "total_cores": cores,
            "omp_threads": omp,
            "expected_improvement": 0.0,
            "predicted_mean": 0.0,
            "predicted_std": 0.0,
            "source": "random",
        }

    def get_best_observed(self) -> Optional[Dict[str, Any]]:
        """
        Return the best observed configuration so far.

        Returns:
            Dict with 'mode', 'total_cores', 'omp_threads', 'walltime_sec',
            or None if no observations exist.
        """
        with self._lock:
            if not self._X or self._best_y is None:
                return None

            idx = self._y.index(self._best_y)
            best_vec = self._X[idx]
            config = _decode_config(best_vec, self.min_cores, self.max_cores)
            return {
                "mode": config["mode"],
                "total_cores": config["total_cores"],
                "omp_threads": config["omp_threads"],
                "walltime_sec": self._best_y,
            }

    @property
    def n_observations(self) -> int:
        """Return the number of observations accumulated so far."""
        with self._lock:
            return len(self._X)


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "BayesianOptimizer",
    "compute_expected_improvement",
    "rbf_kernel",
    "rbf_kernel_ard",
]
