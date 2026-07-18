"""
Kernel functions for Gaussian Process regression.

Nugget and Stability Contract
-----------------------------
This module defines two global stability constants used across the GP stack:

  ``_NUGGET`` (1e-6)
      Provides numerical stability for Cholesky decomposition.
      Callers (GP modules) **must** add ``_NUGGET * eye(n)`` to every
      kernel matrix before factorization.  The kernel functions themselves
      do **not** apply the nugget — that responsibility belongs to the
      decomposed-kernel call-site.

  ``_EPS``   (1e-8)
      Prevents division by zero in length-scale operations (e.g., squared
      length-scale denominators).  All per-dimension length scales are
      clipped to at least ``_EPS`` inside the kernels.

Failure to apply the nugget in callers that factor the kernel matrix will
result in ``LinAlgError`` for near-singular or rank-deficient matrices.
"""

import math

import numpy as np

_NUGGET = 1e-6  # additive jitter for Cholesky stability — see module docstring
_EPS = 1e-8    # floor for length-scale denominators — see module docstring


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    r"""
    Radial Basis Function (squared exponential) kernel for Gaussian Process.

    k(x1, x2) = exp(-0.5 * ||x1 - x2||^2 / length_scale^2)

    Note
    ----
    This function returns the **raw** kernel matrix — it does **not** add
    the nugget term ``_NUGGET * eye(n)``.  Callers that factor the kernel
    (e.g., via Cholesky) must add the nugget themselves; see the module
    docstring for the full contract.

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
    r"""
    RBF kernel with Automatic Relevance Determination (per-dimension length scales).

    k(x1, x2) = exp(-0.5 * sum_d ((x1_d - x2_d)^2 / length_scale[d]^2))

    Note
    ----
    This function returns the **raw** kernel matrix — it does **not** add
    the nugget term ``_NUGGET * eye(n)``.  Callers that factor the kernel
    (e.g., via Cholesky) must add the nugget themselves; see the module
    docstring for the full contract.

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


def matern_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    length_scales: np.ndarray,
    nu: float = 2.5,
) -> np.ndarray:
    """
    Matern kernel with v = 2.5 (twice differentiable).

    k(r) = (1 + sqrt(5)·r/l + 5r^2/3l^2) · exp(-sqrt(5)·r/l)

    Preferred over RBF for modelling non-smooth objective surfaces
    such as SCF convergence behaviour (Snoek et al. 2012, NIPS 25, 2951-2959;
    Rasmussen & Williams 2006, Gaussian Processes for Machine Learning).

    Args:
        x1: First input matrix (shape (n, d)).
        x2: Second input matrix (shape (m, d)).
        length_scales: Per-dimension length scale vector (shape (d,)).
        nu: Smoothness parameter (not used directly, fixed v=2.5 formula above).

    Returns:
        Kernel matrix of shape (n, m).
    """
    x1 = np.atleast_2d(np.asarray(x1, dtype=np.float64))
    x2 = np.atleast_2d(np.asarray(x2, dtype=np.float64))
    length_scales = np.atleast_1d(np.asarray(length_scales, dtype=np.float64))

    d = x1.shape[1]
    r_sq = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float64)
    for i in range(d):
        ls = max(length_scales[i], _EPS)
        diff = (x1[:, i].reshape(-1, 1) - x2[:, i].reshape(1, -1)) ** 2
        r_sq += diff / (ls ** 2)

    r = np.sqrt(r_sq + _EPS)
    sqrt5_r = math.sqrt(5.0) * r
    K = (1.0 + sqrt5_r + (5.0 / 3.0) * r_sq) * np.exp(-sqrt5_r)
    return K


__all__ = ["_EPS", "_NUGGET", "matern_kernel", "rbf_kernel", "rbf_kernel_ard"]
