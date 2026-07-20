"""DPP Batch Selection for diverse BO batches.

Implements Determinantal Point Process (DPP) based batch selection
for Bayesian Optimisation, following Kathuria, Deshpande & Kohli
(NeurIPS 2016).

Given a set of N candidates with quality scores q_i and a kernel
k(x_i, x_j), the DPP selects a diverse subset S of size q that
maximises P(S) ∝ det(L_S), where L_ij = q_i · k(x_i, x_j) · q_j.

A greedy MAP approximation is used with Cholesky-based conditional
variance (Schur complement) updates — O(q·N²) instead of O(N³).

Reference:
  Kathuria, T., Deshpande, A., & Kohli, P. (2016).
  Batched Gaussian Process Bandit Optimization via Determinantal
  Point Processes.  NeurIPS 29.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class DPPBatchSelector:
    """Select a diverse batch of candidates for parallel BO evaluation.

    Uses quality scores (typically exp(EI_i / 2)) and a GP kernel to
    build the DPP kernel L_ij = q_i * k(x_i, x_j) * q_j, then greedily
    selects the subset maximising the determinant.

    Usage::

        selector = DPPBatchSelector()
        indices = selector.select(
            candidates=X_candidates,
            quality=ei_values,
            gp=fitted_gp,
            q=4,
        )
    """

    def __init__(self) -> None:
        pass

    def select(
        self,
        candidates: np.ndarray,
        quality: np.ndarray,
        gp: Any,
        q: int = 4,
        kernel_func: Any = None,
    ) -> list[int]:
        """Select q candidate indices via greedy DPP.

        Args:
            candidates: (N, D) candidate points.
            quality: (N,) quality scores (e.g. EI values).
            gp: Fitted GP with a ``kernel`` method (k(x, x')), or None to
                use *kernel_func* directly.
            q: Number of points to select in the batch.
            kernel_func: Callable k(x, x') → scalar.  Overrides *gp*.

        Returns:
            List of q indices into *candidates*, in selection order.
        """
        n = len(candidates)
        if n == 0 or q <= 0:
            return []
        q = min(q, n)

        q_vec = np.asarray(quality, dtype=np.float64).ravel()
        if q_vec.shape[0] != n:
            raise ValueError("quality length must match candidates")

        # --- Build L_ii (diagonal of DPP kernel) ---
        q_norm = np.exp(q_vec / 2.0)  # q_i = exp(EI_i / 2)
        L_diag = q_norm ** 2  # k(x_i, x_i) = 1 for stationary kernels

        # --- Kernel function for off-diagonal ---
        k_fn = kernel_func if kernel_func is not None else _extract_kernel_fn(gp)

        # --- Greedy DPP selection ---
        selected: list[int] = []
        rem = list(range(n))

        # Cholesky factor of L_S (built incrementally)
        L_chol: np.ndarray | None = None  # (k, k)

        for _step in range(q):
            best_idx_in_rem = -1
            best_marginal = -np.inf

            for r_idx, i in enumerate(rem):
                if L_chol is None:
                    marginal = float(L_diag[i])
                else:
                    v = np.zeros(len(selected), dtype=np.float64)
                    for a_idx, a in enumerate(selected):
                        kern_val = float(k_fn(candidates[a:a+1], candidates[i:i+1]))
                        v[a_idx] = q_norm[a] * q_norm[i] * kern_val
                    solved = np.linalg.solve(L_chol, v)
                    cond_var = float(np.dot(solved, solved))
                    marginal = float(L_diag[i]) - cond_var

                if marginal > best_marginal:
                    best_marginal = marginal
                    best_idx_in_rem = r_idx

            if best_idx_in_rem < 0:
                break

            chosen = rem.pop(best_idx_in_rem)
            selected.append(int(chosen))

            # --- Update Cholesky of L_S ---
            L_chol = _chol_insert(
                L_chol,
                q_norm,
                selected,
                candidates,
                k_fn,
            )

        return selected


# ---------------------------------------------------------------------------
# Cholesky insertion
# ---------------------------------------------------------------------------

def _chol_insert(
    L_old: np.ndarray | None,
    q_norm: np.ndarray,
    selected: list[int],
    candidates: np.ndarray,
    k_fn,
) -> np.ndarray:
    """Insert the last element of *selected* into the Cholesky factor.

    Given the Cholesky L_{k-1} of L_{S} for the first k-1 selected points,
    produce L_k for all k points.

    Args:
        L_old: Cholesky factor (k-1, k-1) or None for k=1.
        q_norm: quality vector (N,).
        selected: indices of selected points so far (k elements).
        candidates: candidate point array.
        k_fn: kernel function k(x, x') → scalar.

    Returns:
        Cholesky factor (k, k).
    """
    k = len(selected)
    i_new = selected[-1]

    if k == 1:
        return np.array([[float(q_norm[i_new])]], dtype=np.float64)

    L_new = np.zeros((k, k), dtype=np.float64)
    L_new[:k-1, :k-1] = L_old

    # Compute L_{k, 1:k-1}
    for r in range(k - 1):
        i_r = selected[r]
        kern_val = float(k_fn(
            candidates[i_r:i_r+1], candidates[i_new:i_new+1]
        ))
        L_val = q_norm[i_r] * q_norm[i_new] * kern_val
        # Solve triangular system
        L_new[k-1, r] = (L_val - np.dot(L_old[r, :r], L_new[k-1, :r])) / L_old[r, r]

    # Diagonal: L_kk² = L_ii - Σ_{j<k} L_kj²
    diag_val = float(q_norm[i_new] ** 2)
    for j in range(k - 1):
        diag_val -= L_new[k-1, j] ** 2
    L_new[k-1, k-1] = math.sqrt(max(diag_val, 1e-12))

    return L_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_kernel_fn(gp: Any):
    """Extract a kernel function k(x, x') → scalar from a fitted GP."""
    if hasattr(gp, "kernel"):
        def _gp_kernel(a: np.ndarray, b: np.ndarray) -> float:
            return float(gp.kernel(a, b))  # type: ignore[union-attr]
        return _gp_kernel

    default_length_scales = np.ones(2) * 0.5

    from .kernels import rbf_kernel_ard

    def _kernel(a: np.ndarray, b: np.ndarray) -> float:
        a2 = np.atleast_2d(a)
        b2 = np.atleast_2d(b)
        # Use a default RBF kernel if no explicit kernel available
        return float(rbf_kernel_ard(a2, b2, default_length_scales)[0, 0])

    return _kernel


__all__ = ["DPPBatchSelector"]
