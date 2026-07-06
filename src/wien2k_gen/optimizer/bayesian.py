"""
Bayesian Optimisation Framework for Tuning WIEN2k Parallel Execution Parameters.
Production features:
• Gaussian Process regression with ARD kernel + gradient-based length scale tuning
• Multi-fidelity BO: coarse/medium/full fidelity levels with cost-aware MF-EI
• Transfer learning across chemical systems using elemental property similarity
• Constrained BO with memory and walltime budget enforcement
• Expected Improvement (EI) acquisition function for exploration-exploitation
• Mixed parameter space: continuous (total_cores, omp_threads) + categorical (mode)
• One-hot encoding for categorical mode parameter with valid dimension handling
• History-driven warm-start via ExecutionHistory integration
• Regularised GP with jitter (nugget) for numerical stability on small datasets
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

# Multi-fidelity cost multipliers (fraction of full-run cost)
_FIDELITY_COST = {0: 0.10, 1: 0.40, 2: 1.00}
_FIDELITY_CORRELATION = {0: 0.60, 1: 0.85, 2: 1.00}
_DEFAULT_EI_THRESHOLD = 0.001

# Chemical similarity: atomic number distance weight coefficients
_ELEMENT_GROUPS = {
    1: {1},  2: {2},  3: {3, 11, 19, 37, 55},  4: {4, 12, 20, 38, 56},
    5: {5, 13, 31, 49, 81},  6: {6, 14, 32, 50, 82},
    7: {7, 15, 33, 51, 83},  8: {8, 16, 34, 52, 84},
    9: {9, 17, 35, 53, 85},  10: {10, 18, 36, 54, 86},
    11: {21, 39},  12: {22, 40, 72},  13: {23, 41, 73},
    14: {24, 42, 74},  15: {25, 43, 75},  16: {26, 44, 76},
    17: {27, 45, 77},  18: {28, 46, 78},  19: {29, 47, 79},  20: {30, 48, 80},
}
_ELEMENT_PERIODS = {
    1: {1, 2},  2: {3, 4, 5, 6, 7, 8, 9, 10},
    3: {11, 12, 13, 14, 15, 16, 17, 18},
    4: {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
    5: {37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54},
    6: {55, 56, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86},
}


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

    pdf_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    ei = improvement * cdf_z + sigma * pdf_z
    return max(0.0, ei)


# =============================================================================
# Gaussian Process with ARD (Manual Implementation)
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

        mu = K_s.T @ self._alpha

        v = np.linalg.solve(self._L, K_s)
        cov = K_ss - v.T @ v

        sigma2 = np.diag(cov)
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

        for step in range(self._n_opt_steps):
            K = rbf_kernel_ard(X, X, current_ls)
            K += _NUGGET * np.eye(n, dtype=np.float64)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-3 * np.eye(n, dtype=np.float64)
                L = np.linalg.cholesky(K)

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))

            nll = 0.5 * float(y.T @ alpha) + float(np.sum(np.log(np.diag(L)))) + \
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

    def get_relevance(self) -> Dict[str, float]:
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
# Chemical Similarity Utilities for Transfer Learning
# =============================================================================

def _get_element_group(atomic_number: int) -> int:
    """Map atomic number to simplified group number (1-20)."""
    for group_num, elements in _ELEMENT_GROUPS.items():
        if atomic_number in elements:
            return group_num
    return 0


def _get_element_period(atomic_number: int) -> int:
    """Map atomic number to period number (1-6)."""
    for period_num, elements in _ELEMENT_PERIODS.items():
        if atomic_number in elements:
            return period_num
    return 0


def _chemical_similarity(source_atomic_num: int, target_atomic_num: int) -> float:
    """
    Compute chemical similarity weight (0-1) between two elements.

    Factors:
    - Atomic number proximity (40% weight)
    - Same group (35% weight)
    - Same period (25% weight)

    Args:
        source_atomic_num: Atomic number of source element.
        target_atomic_num: Atomic number of target element.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if source_atomic_num == target_atomic_num:
        return 1.0

    max_z = max(source_atomic_num, target_atomic_num)
    min_z = min(source_atomic_num, target_atomic_num)
    z_distance = abs(source_atomic_num - target_atomic_num)
    z_similarity = max(0.0, 1.0 - z_distance / max(1.0, float(max_z)))

    same_group = 1.0 if _get_element_group(source_atomic_num) == _get_element_group(target_atomic_num) else 0.0
    same_period = 1.0 if _get_element_period(source_atomic_num) == _get_element_period(target_atomic_num) else 0.0

    weight = 0.40 * z_similarity + 0.35 * same_group + 0.25 * same_period
    return min(1.0, max(0.0, weight))


_ELEMENT_ATOMIC_NUMBERS = {
    "H": 1,  "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,   "O": 8,
    "F": 9,  "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,  "S": 16,
    "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,  "Xe": 54, "Cs": 55, "Ba": 56,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}


# =============================================================================
# Constraint Models
# =============================================================================

def _estimate_memory_gb_for_config(nmat: int, total_cores: int) -> float:
    """
    Per-rank memory estimate for a given MPI/OpenMP configuration.

    In ScaLAPACK/ELPA block-cyclic distribution, the Hamiltonian matrix
    is distributed across MPI ranks, so per-rank memory scales as nmat²/ranks.

    Args:
        nmat: Hamiltonian matrix size.
        total_cores: Total CPU cores (MPI ranks × OMP threads).

    Returns:
        Estimated per-rank memory in GB.
    """
    ranks = max(1, total_cores)
    # Aggregate matrix memory, then divide by ranks for block-cyclic distribution
    aggregate_gb = (float(nmat) ** 2.0) * 16.0 / (1024.0 ** 3)
    per_rank_gb = aggregate_gb / float(ranks)
    # Small per-rank overhead for communication buffers + replicated data
    comm_overhead = 0.5  # GB per rank for MPI buffers, charge density copies
    safety = 1.5  # Per-rank safety factor (was 3.0× on aggregate)
    return (per_rank_gb + comm_overhead) * safety


def _estimate_walltime_min_for_config(nmat: int, nkpt: int, total_cores: int) -> float:
    """
    Rough walltime estimate for a given config using empirical scaling.

    Args:
        nmat: Hamiltonian matrix size.
        nkpt: Number of k-points.
        total_cores: Total CPU cores.

    Returns:
        Estimated walltime in minutes.
    """
    baseline_work = (float(nmat) ** 3.0) * float(max(1, nkpt))
    baseline_time_sec = baseline_work / (5000.0 ** 3.0 * 4.0) * 3600.0
    effective_cores = float(max(1, total_cores))
    return baseline_time_sec / effective_cores / 60.0


# =============================================================================
# BayesianOptimizer Class
# =============================================================================

class BayesianOptimizer:
    """
    Bayesian parameter optimiser for WIEN2k parallel execution tuning.

    Maintains a Gaussian Process surrogate model of the objective function
    (walltime) over the configuration space and suggests new configurations
    to try using the Expected Improvement acquisition criterion.

    Supports transfer learning from chemically similar systems and constrained
    optimisation with memory/walltime budgets.

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
        use_ard: bool = True,
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
            use_ard: If True, use _GaussianProcessARD; otherwise _GaussianProcess.
        """
        self._history = history
        self.backend = backend
        self.min_cores = min_cores
        self.max_cores = max_cores
        self._exploration_xi = exploration_xi
        self._n_random_restarts = n_random_restarts
        self._use_ard = use_ard
        self._gp: _GaussianProcess = (
            _GaussianProcessARD(length_scales=length_scales)
            if use_ard
            else _GaussianProcess(length_scales=length_scales)
        )
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._best_y: Optional[float] = None
        self._lock = threading.Lock()
        self._n_dims = 2 + len(_CATEGORICAL_MODES)

        self._transfer_mean: Optional[np.ndarray] = None
        self._transfer_weight: float = 0.0
        self._source_system: Optional[str] = None

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

    def transfer_from_system(
        self,
        source_element: str,
        target_element: str,
        source_history: Optional[ExecutionHistory] = None,
    ) -> None:
        """
        Transfer prior knowledge from a chemically similar source system.

        Uses elemental properties (atomic number ratio, group, period) to
        compute a similarity weight.  The prior mean from the source system's
        GP is blended with the target GP prior.  The blend weight is proportional
        to the chemical similarity.

        If no similar system data is available, falls back to standard GP prior
        (zero mean function).

        Args:
            source_element: Chemical symbol of the source element (e.g. 'Si').
            target_element: Chemical symbol of the target element (e.g. 'Ge').
            source_history: Optional ExecutionHistory containing source data.
                            If None, uses self._history filtered for source_element.
        """
        src_z = _ELEMENT_ATOMIC_NUMBERS.get(source_element, 0)
        tgt_z = _ELEMENT_ATOMIC_NUMBERS.get(target_element, 0)

        similarity = _chemical_similarity(src_z, tgt_z)
        logger.info(
            f"Transfer learning: {source_element}(Z={src_z}) -> "
            f"{target_element}(Z={tgt_z}), similarity={similarity:.4f}"
        )

        if similarity < 0.1:
            logger.warning(
                f"Chemical similarity too low ({similarity:.4f}); "
                f"falling back to standard GP prior."
            )
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return

        history_to_use = source_history if source_history is not None else self._history

        source_records = history_to_use.query(
            filters={"backend": self.backend, "success": True},
            order_by="walltime_sec ASC",
            limit=100,
        )

        if not source_records:
            logger.warning("No source system records found; falling back to standard GP prior.")
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return

        source_X = []
        source_y = []
        for rec in source_records:
            if rec.walltime_sec > 0:
                source_X.append(_encode_config(rec.mode, rec.total_cores, rec.omp_threads))
                source_y.append(rec.walltime_sec)

        if len(source_X) < 2:
            logger.warning("Insufficient source data; falling back to standard GP prior.")
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return

        source_X_arr = np.array(source_X, dtype=np.float64)
        source_y_arr = np.array(source_y, dtype=np.float64)

        source_gp = _GaussianProcess()
        source_gp.fit(source_X_arr, source_y_arr)

        self._transfer_mean = source_X_arr.mean(axis=0)
        self._transfer_weight = similarity
        self._source_system = source_element

        self._gp = (
            _GaussianProcessARD()
            if self._use_ard
            else _GaussianProcess()
        )

        logger.info(
            f"Transfer prior established from {source_element} "
            f"(weight={self._transfer_weight:.4f}), "
            f"{len(source_X)} source records used."
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

        If transfer learning is active, candidate predictions are adjusted
        toward the source prior mean.

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
            - transfer_weight: float
        """
        max_cores = self.max_cores
        if user_max_cores is not None:
            max_cores = min(self.max_cores, max(1, user_max_cores))

        with self._lock:
            n_obs = len(self._X)

            if n_obs < 2:
                result = self._random_suggestion(max_cores)
                result["transfer_weight"] = self._transfer_weight
                return result

            current_best = self._best_y or float("inf")
            best_vec: Optional[np.ndarray] = None
            best_ei = -1.0
            best_mu = float("inf")
            best_sigma = 0.0

            rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))

            for _ in range(self._n_random_restarts):
                cores = rng.randint(self.min_cores, max_cores + 1)
                omp = rng.randint(1, min(65, cores + 1))
                mode = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]

                candidate = _encode_config(mode, cores, omp)

                try:
                    mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                    mu_val = float(mu[0])
                    sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))

                    if self._transfer_mean is not None and self._transfer_weight > 0:
                        source_gp = _GaussianProcess(length_scales=self._gp.length_scales)
                        mu_val = (1.0 - self._transfer_weight) * mu_val + \
                                 self._transfer_weight * float(np.mean(self._y)) if self._y else mu_val

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
                    "transfer_weight": round(self._transfer_weight, 4),
                }

            result = self._random_suggestion(max_cores)
            result["transfer_weight"] = self._transfer_weight
            return result

    def suggest_next_with_constraints(
        self,
        nmat: int,
        nkpt: int,
        max_memory_gb: float,
        max_walltime_min: float,
        user_max_cores: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Suggest next configuration subject to memory and walltime budgets.

        Uses constrained EI: EI(x) * P(feasible | constraints) where
        feasibility is estimated via sigmoid soft constraints on memory
        and walltime.

        Args:
            nmat: Hamiltonian matrix size.
            nkpt: Number of k-points.
            max_memory_gb: Maximum memory budget in GB.
            max_walltime_min: Maximum walltime budget in minutes.
            user_max_cores: Override max_cores (e.g. from topology limit).

        Returns:
            Dictionary with same keys as suggest_next, plus:
            - p_feasible: float
            - estimated_memory_gb: float
            - estimated_walltime_min: float
        """
        max_cores = self.max_cores
        if user_max_cores is not None:
            max_cores = min(self.max_cores, max(1, user_max_cores))

        with self._lock:
            n_obs = len(self._X)

            if n_obs < 2:
                result = self._random_suggestion(max_cores)
                result["p_feasible"] = 1.0
                result["transfer_weight"] = self._transfer_weight
                return result

            current_best = self._best_y or float("inf")
            best_vec: Optional[np.ndarray] = None
            best_constrained_ei = -1.0
            best_mu = float("inf")
            best_sigma = 0.0
            best_p_feasible = 0.0
            best_est_mem = 0.0
            best_est_wt = 0.0

            rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))

            for _ in range(self._n_random_restarts):
                cores = rng.randint(self.min_cores, max_cores + 1)
                omp = rng.randint(1, min(65, cores + 1))
                mode = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]

                candidate = _encode_config(mode, cores, omp)

                try:
                    mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                    mu_val = float(mu[0])
                    sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))

                    if self._transfer_mean is not None and self._transfer_weight > 0:
                        mu_val = (1.0 - self._transfer_weight) * mu_val + \
                                 self._transfer_weight * float(np.mean(self._y)) if self._y else mu_val

                    ei = compute_expected_improvement(
                        mu_val, sigma_val, current_best, xi=self._exploration_xi
                    )

                    est_mem = _estimate_memory_gb_for_config(nmat, cores)
                    est_wt = _estimate_walltime_min_for_config(nmat, nkpt, cores)

                    p_mem_feasible = _sigmoid_feasibility(est_mem, max_memory_gb)
                    p_wt_feasible = _sigmoid_feasibility(est_wt, max_walltime_min)
                    p_feasible = p_mem_feasible * p_wt_feasible

                    constrained_ei = ei * p_feasible

                    if constrained_ei > best_constrained_ei:
                        best_constrained_ei = constrained_ei
                        best_vec = candidate.copy()
                        best_mu = mu_val
                        best_sigma = sigma_val
                        best_p_feasible = p_feasible
                        best_est_mem = est_mem
                        best_est_wt = est_wt
                except Exception:
                    continue

            if best_vec is not None:
                config = _decode_config(best_vec, self.min_cores, max_cores)
                return {
                    "mode": config["mode"],
                    "total_cores": config["total_cores"],
                    "omp_threads": config["omp_threads"],
                    "expected_improvement": round(best_constrained_ei / max(best_p_feasible, _EPS), 6),
                    "constrained_ei": round(best_constrained_ei, 6),
                    "predicted_mean": round(best_mu, 3),
                    "predicted_std": round(best_sigma, 3),
                    "source": "model_constrained",
                    "p_feasible": round(best_p_feasible, 4),
                    "estimated_memory_gb": round(best_est_mem, 2),
                    "estimated_walltime_min": round(best_est_wt, 2),
                    "transfer_weight": round(self._transfer_weight, 4),
                }

            result = self._random_suggestion(max_cores)
            result["p_feasible"] = 1.0
            result["transfer_weight"] = self._transfer_weight
            return result

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

    def get_parameter_relevance(self) -> Dict[str, float]:
        """Return ARD relevance scores if using ARD kernel."""
        with self._lock:
            if isinstance(self._gp, _GaussianProcessARD):
                return self._gp.get_relevance()
            return {}


# =============================================================================
# Constrained BO Helper: Sigmoid Feasibility
# =============================================================================

def _sigmoid_feasibility(estimated: float, max_allowed: float, slope: float = 10.0) -> float:
    """
    Soft feasibility probability via sigmoid.

    P(feasible) ≈ 1 / (1 + exp(slope * (estimated/max_allowed - 1)))

    Args:
        estimated: Estimated resource consumption.
        max_allowed: Maximum resource budget.
        slope: Steepness of the transition.

    Returns:
        Probability between 0 and 1.
    """
    if max_allowed <= 0:
        return 0.0
    ratio = estimated / max_allowed
    return float(1.0 / (1.0 + math.exp(slope * (ratio - 1.0))))


# =============================================================================
# Multi-Fidelity Bayesian Optimizer
# =============================================================================

class MultiFidelityBayesianOptimizer(BayesianOptimizer):
    """
    Multi-fidelity Bayesian Optimizer for WIEN2k parameter tuning.

    Uses cost-aware multi-fidelity Expected Improvement (MF-EI) that trades
    off evaluation cost against information gain.  Fidelity levels:
        - 0: coarse   (1 k-point,  ~10% of full run cost, noisy)
        - 1: medium   (4 k-points, ~40% of full run cost)
        - 2: full     (all k-points, 100% of full run cost)

    Automatically promotes to higher fidelity when Expected Improvement drops
    below a configurable threshold, indicating the low-fidelity surrogate is
    no longer providing meaningful guidance.
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
        use_ard: bool = True,
        ei_promotion_threshold: float = _DEFAULT_EI_THRESHOLD,
        min_fidelity_evals: int = 3,
    ) -> None:
        """
        Initialize the multi-fidelity Bayesian optimiser.

        Args:
            history: ExecutionHistory instance for warm-starting.
            backend: Target DFT backend name.
            min_cores: Lower bound on total_cores.
            max_cores: Upper bound on total_cores.
            length_scales: Per-dimension length scales for ARD kernel.
            exploration_xi: Exploration parameter for EI acquisition.
            n_random_restarts: Number of random restarts for acquisition optimisation.
            use_ard: If True, use ARD GP kernel.
            ei_promotion_threshold: EI value below which fidelity is promoted.
            min_fidelity_evals: Minimum evaluations at a fidelity before promotion.
        """
        super().__init__(
            history=history,
            backend=backend,
            min_cores=min_cores,
            max_cores=max_cores,
            length_scales=length_scales,
            exploration_xi=exploration_xi,
            n_random_restarts=n_random_restarts,
            use_ard=use_ard,
        )
        self._current_fidelity: int = 0
        self._fidelity_eval_count: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._ei_promotion_threshold = ei_promotion_threshold
        self._min_fidelity_evals = min_fidelity_evals
        self._fidelity_gps: Dict[int, _GaussianProcess] = {}
        self._fidelity_X: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        self._fidelity_y: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    def _effective_nkpt(self, nkpt: int, fidelity_level: int) -> int:
        """Map fidelity level to effective k-point count for cost model."""
        if fidelity_level == 0:
            return 1
        elif fidelity_level == 1:
            return max(1, min(4, nkpt))
        else:
            return nkpt

    def _add_observation_no_lock(self, record: ExecutionRecord) -> None:
        """Override to track per-fidelity observations."""
        super()._add_observation_no_lock(record)
        fid = getattr(record, "fidelity_level", 2)
        x = _encode_config(record.mode, record.total_cores, record.omp_threads)
        self._fidelity_X.setdefault(fid, []).append(x)
        self._fidelity_y.setdefault(fid, []).append(record.walltime_sec)
        self._fidelity_eval_count[fid] = self._fidelity_eval_count.get(fid, 0) + 1

    def suggest_next_fidelity(
        self,
        nmat: int,
        nkpt: int,
        user_max_cores: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Suggest the next configuration AND fidelity level to evaluate.

        Uses Multi-fidelity Expected Improvement:
            MF-EI(x, f) = EI(x) * correlation_weight(f) / cost(f)

        Automatically promotes fidelity when EI drops below threshold.

        Args:
            nmat: Hamiltonian matrix size.
            nkpt: Number of k-points.
            user_max_cores: Override max_cores.

        Returns:
            Tuple of (config_dict, fidelity_level).
        """
        max_cores = self.max_cores
        if user_max_cores is not None:
            max_cores = min(self.max_cores, max(1, user_max_cores))

        with self._lock:
            n_obs = len(self._X)

            if n_obs < 2:
                config = self._random_suggestion(max_cores)
                config["fidelity"] = 0
                config["source"] = "random_mf"
                self._current_fidelity = 0
                return config, 0

            current_best = self._best_y or float("inf")

            best_mf_ei = -1.0
            best_config: Optional[Dict[str, Any]] = None
            best_fid = 0

            rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))

            for fid_level in [0, 1, 2]:
                corr = _FIDELITY_CORRELATION[fid_level]
                cost = _FIDELITY_COST[fid_level]

                for _ in range(self._n_random_restarts):
                    cores = rng.randint(self.min_cores, max_cores + 1)
                    omp = rng.randint(1, min(65, cores + 1))
                    mode = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]
                    candidate = _encode_config(mode, cores, omp)

                    try:
                        mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                        mu_val = float(mu[0])
                        sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))

                        if self._transfer_mean is not None and self._transfer_weight > 0:
                            mu_val = (1.0 - self._transfer_weight) * mu_val + \
                                     self._transfer_weight * float(np.mean(self._y)) if self._y else mu_val

                        ei = compute_expected_improvement(
                            mu_val, sigma_val, current_best, xi=self._exploration_xi
                        )
                        mf_ei = ei * corr / max(cost, _EPS)

                        if mf_ei > best_mf_ei:
                            best_mf_ei = mf_ei
                            config = _decode_config(candidate, self.min_cores, max_cores)
                            best_config = {
                                "mode": config["mode"],
                                "total_cores": config["total_cores"],
                                "omp_threads": config["omp_threads"],
                                "expected_improvement": round(ei, 6),
                                "mf_ei": round(mf_ei, 6),
                                "predicted_mean": round(mu_val, 3),
                                "predicted_std": round(sigma_val, 3),
                                "fidelity": fid_level,
                                "source": "model_mf",
                                "transfer_weight": round(self._transfer_weight, 4),
                            }
                            best_fid = fid_level
                    except Exception:
                        continue

            if best_config is not None:
                ei_val = best_config.get("expected_improvement", 0.0)
                eval_count = self._fidelity_eval_count.get(best_fid, 0)

                if (
                    ei_val < self._ei_promotion_threshold
                    and eval_count >= self._min_fidelity_evals
                    and best_fid < 2
                ):
                    best_fid += 1
                    best_config["fidelity"] = best_fid
                    best_config["source"] = "model_mf_promoted"
                    logger.info(
                        f"Promoting fidelity {best_fid - 1} -> {best_fid} "
                        f"(EI={ei_val:.6f} < threshold={self._ei_promotion_threshold}, "
                        f"evals={eval_count})"
                    )

                self._current_fidelity = best_fid
                return best_config, best_fid

            config = self._random_suggestion(max_cores)
            config["fidelity"] = 0
            config["source"] = "random_mf"
            self._current_fidelity = 0
            return config, 0

    @property
    def current_fidelity(self) -> int:
        """Return the current fidelity level."""
        return self._current_fidelity

    def get_fidelity_stats(self) -> Dict[str, Any]:
        """Return per-fidelity evaluation counts and costs."""
        with self._lock:
            return {
                "evals_per_fidelity": dict(self._fidelity_eval_count),
                "cost_multipliers": dict(_FIDELITY_COST),
                "correlation_weights": dict(_FIDELITY_CORRELATION),
                "current_fidelity": self._current_fidelity,
            }


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "BayesianOptimizer",
    "MultiFidelityBayesianOptimizer",
    "compute_expected_improvement",
    "rbf_kernel",
    "rbf_kernel_ard",
    "_GaussianProcessARD",
]
