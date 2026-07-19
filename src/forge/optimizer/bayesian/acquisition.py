"""Acquisition functions: Expected Improvement and q-EI."""

import math

import numpy as np

_DEFAULT_EI_THRESHOLD = 0.001


def compute_q_expected_improvement(
    gp,                       # _GaussianProcess — fitted GP model
    X_candidates: np.ndarray,  # candidate points (n, d)
    best_y: float,
    q: int = 4,
    xi: float = 0.01,
    n_mc_samples: int = 500,
) -> list[int]:
    """
    q-batch Expected Improvement via greedy Monte Carlo joint posterior sampling.

    Instead of selecting one point at a time (single EI), q-EI selects a batch of q
    points simultaneously, accounting for the joint posterior correlation between them.
    This prevents selecting redundant points that cluster in the same region.

    Algorithm (Ginsbourger et al. 2010):
      1. Build joint posterior covariance K_joint over all candidates
      2. Greedy selection: at each step, pick the candidate that maximises
         E[max(0, max_{s in selected+candidate} f_s - best_y - xi)]
      3. Estimate the expectation via Cholesky-based Monte Carlo sampling from
         the multivariate normal posterior of selected + candidate points

    Args:
        gp: Fitted _GaussianProcess with _X_train, _alpha, length_scales.
        X_candidates: Candidate points (n_candidates, d).
        best_y: Best observed value (minimisation).
        q: Number of points to select in the batch.
        xi: Exploration parameter.
        n_mc_samples: Number of Monte Carlo samples per candidate evaluation.

    Returns:
        List of q indices into X_candidates, selected greedily.
    """
    from .kernels import rbf_kernel_ard

    n_candidates = len(X_candidates)
    if n_candidates == 0 or q <= 0:
        return []

    q = min(q, n_candidates)

    mu_full, _var_full = gp.predict(X_candidates)
    mu_full = mu_full.ravel()

    # Posterior covariance between candidates:
    #   cov(f_test) = K_tt - K_ttrain @ inv(K_train) @ K_traint
    # Compute from stored GP decomposition for numerical stability.
    K_tt = rbf_kernel_ard(X_candidates, X_candidates, gp.length_scales)

    if gp._X_train is not None and gp._alpha is not None:
        K_t_train = rbf_kernel_ard(X_candidates, gp._X_train, gp.length_scales)
        v = np.linalg.solve(gp._L, K_t_train.T)
        K_posterior = K_tt - v.T @ v
    else:
        K_posterior = K_tt

    jitter = max(1e-6, np.max(np.diag(np.abs(K_posterior))) * 1e-8)
    K_posterior += jitter * np.eye(n_candidates, dtype=np.float64)

    selected: list[int] = []
    rng = np.random.RandomState()

    for _step in range(q):
        best_qei = -float("inf")
        best_idx = -1

        for i in range(n_candidates):
            if i in selected:
                continue

            joint_idx = np.array([*selected, i], dtype=int)
            mu_joint = mu_full[joint_idx].astype(np.float64)
            K_joint = K_posterior[joint_idx][:, joint_idx].astype(np.float64)

            try:
                L = np.linalg.cholesky(K_joint)
            except np.linalg.LinAlgError:
                K_joint += jitter * np.eye(len(joint_idx), dtype=np.float64)
                L = np.linalg.cholesky(K_joint)

            samples = L @ rng.randn(len(joint_idx), n_mc_samples) + mu_joint.reshape(-1, 1)
            max_per_sample = np.max(samples, axis=0)
            improvements = np.maximum(0, max_per_sample - best_y - xi)
            q_ei_val = float(np.mean(improvements))

            if q_ei_val > best_qei:
                best_qei = q_ei_val
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)
        else:
            break

    return selected


def compute_expected_improvement(
    mu: float,
    sigma: float,
    best_y: float,
    xi: float = 0.01,
) -> float:
    """
    Expected Improvement acquisition function.

    EI(x) = sigma(x) * (z * Phi(z) + phi(z))
    where z = (best_y - mu(x) - xi) / sigma(x)   (minimisation)
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
    _EPS = 1e-8

    if sigma < _EPS:
        return 0.0

    improvement = best_y - mu - xi
    z = improvement / sigma

    pdf_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    ei = improvement * cdf_z + sigma * pdf_z
    return max(0.0, ei)


__all__ = ["compute_expected_improvement", "compute_q_expected_improvement"]
