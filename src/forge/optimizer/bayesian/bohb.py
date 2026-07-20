"""BOHB: Bayesian Optimization Hyperband for WIEN2k multi-fidelity tuning.

Implements the BOHB algorithm (Falkner, Klein & Hutter, ICML 2018) adapted
for WIEN2k SCF convergence optimisation.  Combines Successive Halving
(Hyperband brackets) with a TPE-style KDE model — NOT a GP.

Algorithm:
  - Hyperband brackets with successive halving (eta=3)
  - At each budget, observations are split: top gamma fraction -> D_good, rest -> D_bad
  - Build KDEs l(x) = p(x|D_good), g(x) = p(x|D_bad)
  - Suggest: sample N candidates from l(x), pick argmax l(x)/g(x)
  - With probability rho, sample uniformly random for exploration
  - Categorical parameters use Aitchison-Aitken kernel

Reference:
  Falkner, S., Klein, A. & Hutter, F. (2018).  BOHB: Robust and Efficient
  Hyperparameter Optimization at Scale.  PMLR 80:1437-1446.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...logging_config import get_logger
from .sampling import _CATEGORICAL_MODES, _decode_config, _encode_config

logger = get_logger(__name__)

_DEFAULT_ETA = 3
_DEFAULT_MIN_BUDGET = 1
_DEFAULT_N_CONFIGS = 16

# KDE / TPE defaults
_DEFAULT_TOP_FRACTION = 0.15
_DEFAULT_BANDWIDTH_FACTOR = 0.25
_DEFAULT_N_EI_CANDIDATES = 128
_DEFAULT_RANDOM_RATIO = 1.0 / 3.0
_DEFAULT_CAT_LAMBDA = 0.8


# ---------------------------------------------------------------------------
# Aitchison-Aitken kernel for categorical variables
# ---------------------------------------------------------------------------

def _aitchison_aitken_kernel(
    x: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    n_categories: int,
) -> np.ndarray:
    """Pairwise Aitchison-Aitken kernel values.

    Returns p(x_i | y_j) for each pair (x_i, y_j):
        p = lambda_ if x_i == y_j else (1 - lambda_) / (n_categories - 1)
    """
    # x: (M, 1), y: (N, 1) or (M,) vs (N,)
    x_flat = x.reshape(-1, 1)
    y_flat = y.reshape(1, -1)
    match = (x_flat == y_flat).astype(np.float64)
    not_match = 1.0 - match
    return lambda_ * match + not_match * (1.0 - lambda_) / max(n_categories - 1, 1)


# ---------------------------------------------------------------------------
# Gaussian KDE (pure numpy)
# ---------------------------------------------------------------------------

def _gaussian_kde_pdf(
    x: np.ndarray,
    samples: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Gaussian KDE log-probability density for points *x* given *samples*.

    Args:
        x: (M, D) evaluation points.
        samples: (N, D) KDE reference samples.
        bandwidth: scalar bandwidth (same for all dimensions).

    Returns:
        (M,) log-pdf values (unnormalised, for ratio comparison).
    """
    if samples.shape[0] == 0:
        return np.full(x.shape[0], -np.inf)

    # (M, N) = squared dist / (2 * h^2)
    diff = x[:, np.newaxis, :] - samples[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    inv_var = 1.0 / (2.0 * bandwidth * bandwidth + 1e-12)

    # stable log-sum-exp
    max_val = np.max(-sq_dist * inv_var, axis=1, keepdims=True)
    log_sum = max_val.ravel() + np.log(
        np.sum(np.exp(-sq_dist * inv_var - max_val), axis=1) + 1e-300
    )
    return log_sum - math.log(max(samples.shape[0], 1))


def _mean_aa_kernel(
    val: int, values: np.ndarray, lambda_: float, n_categories: int,
) -> float:
    """Average Aitchison-Aitken kernel p(val | each value_i)."""
    n = values.shape[0]
    if n == 0:
        return 1e-12
    match = (values == val).astype(np.float64)
    not_match = 1.0 - match
    probs = lambda_ * match + not_match * ((1.0 - lambda_) / max(n_categories - 1, 1))
    return float(np.mean(probs))


def _estimate_bandwidth(samples: np.ndarray, factor: float = _DEFAULT_BANDWIDTH_FACTOR) -> float:
    """Silverman-style bandwidth estimate scaled by *factor*."""
    n, d = samples.shape
    if n < 2:
        return 1.0
    std = np.std(samples, axis=0)
    iqr = np.subtract(*np.percentile(samples, [75, 25], axis=0))
    sigma = np.where(std > iqr / 1.34, std, iqr / 1.34)
    h = np.mean(sigma) * (n ** (-1.0 / (d + 4.0)))
    return max(float(h * factor), 1e-6)


# ---------------------------------------------------------------------------
# BOHB Optimiser
# ---------------------------------------------------------------------------

class BOHBOptimizer:
    """BOHB optimiser for multi-fidelity WIEN2k parameter tuning.

    TPE-style model per budget level:
      - Split observations into D_good (top gamma) and D_bad (rest)
      - Fit KDEs l(x) and g(x)
      - Suggest via l(x)/g(x) ratio maximisation

    Usage::

        bohb = BOHBOptimizer(nkpt=42, min_budget=1, max_budget=42)
        for _ in range(total_iters):
            config, budget = bohb.suggest()
            walltime = evaluate_config_at_budget(config, budget)
            bohb.observe(config, walltime, budget)
    """

    def __init__(
        self,
        nkpt: int,
        min_budget: int = _DEFAULT_MIN_BUDGET,
        max_budget: int | None = None,
        eta: int = _DEFAULT_ETA,
        n_configs: int = _DEFAULT_N_CONFIGS,
        min_cores: int = 1,
        max_cores: int = 256,
        top_fraction: float = _DEFAULT_TOP_FRACTION,
        bandwidth_factor: float = _DEFAULT_BANDWIDTH_FACTOR,
        n_ei_candidates: int = _DEFAULT_N_EI_CANDIDATES,
        random_ratio: float = _DEFAULT_RANDOM_RATIO,
        cat_lambda: float = _DEFAULT_CAT_LAMBDA,
    ) -> None:
        self._nkpt = nkpt
        self._min_budget = max(1, min_budget)
        self._max_budget = max_budget if max_budget is not None else max(1, nkpt)
        self._eta = max(2, eta)
        self._n_configs = n_configs
        self._min_cores = min_cores
        self._max_cores = max_cores
        self._gamma = max(0.01, min(0.5, top_fraction))
        self._bandwidth_factor = bandwidth_factor
        self._n_ei_candidates = n_ei_candidates
        self._random_ratio = random_ratio
        self._cat_lambda = cat_lambda
        self._n_dims = 2 + len(_CATEGORICAL_MODES)  # 5: cores_norm, omp_norm, one_hot[3]

        self._s_max = math.floor(
            math.log(self._max_budget / self._min_budget) / math.log(self._eta)
        )

        self._observations: list[dict[str, Any]] = []
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._budgets: list[int] = []

        self._best_y: float | None = None
        self._best_config: dict[str, Any] | None = None

        self._active_bracket: dict[str, Any] | None = None
        self._bracket_queue: list[dict[str, Any]] = []
        self._pending_rung_configs: list[np.ndarray] = []
        self._pending_rung_budgets: list[int] = []

        self._current_suggestion: dict[str, Any] | None = None
        self._current_budget: int = self._min_budget

        # TPE KDE caches — rebuilt lazily per call to _tpe_suggest
        self._tpe_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, float]] = {}

        logger.info(
            f"BOHB initialised: nkpt={nkpt}, budget_range=[{self._min_budget}, {self._max_budget}], "
            f"eta={eta}, s_max={self._s_max}, gamma={self._gamma:.2f}"
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def suggest(self) -> tuple[dict[str, Any], int]:
        if self._pending_rung_configs:
            config_vec = self._pending_rung_configs.pop(0)
            budget = self._pending_rung_budgets.pop(0)
            config = _decode_config(config_vec, self._min_cores, self._max_cores)
            self._current_suggestion = config
            self._current_budget = budget
            return config, budget

        bracket = self._next_bracket()
        if bracket is None:
            config = self._random_config()
            self._current_suggestion = config
            self._current_budget = self._max_budget
            return config, self._max_budget

        config_vec = bracket["configs"].pop(0)
        budget = bracket["budgets"][0]
        config = _decode_config(config_vec, self._min_cores, self._max_cores)
        self._current_suggestion = config
        self._current_budget = budget
        return config, budget

    def observe(self, config: dict[str, Any], walltime: float, budget: int) -> None:
        x = _encode_config(
            config.get("mode", "kpoint"),
            config.get("total_cores", 1),
            config.get("omp_threads", 1),
        )
        self._X.append(x)
        self._y.append(walltime)
        self._budgets.append(budget)
        self._tpe_cache = {}

        if self._best_y is None or walltime < self._best_y:
            self._best_y = walltime
            self._best_config = dict(config)

        self._observations.append({
            "config": dict(config),
            "walltime": walltime,
            "budget": budget,
        })

        bracket = self._active_bracket
        if bracket is not None:
            bracket["results"].append((x, walltime, budget))
            bracket["evaluated"] += 1

            configs_remain = len(bracket["configs"])
            pending_remain = len(self._pending_rung_configs)
            rung_total = configs_remain + pending_remain

            if rung_total == 0 and bracket["evaluated"] > 0:
                if len(bracket["budgets"]) > 1:
                    self._advance_rung(bracket)
                else:
                    self._finish_bracket(bracket)
                    self._active_bracket = None

    # -----------------------------------------------------------------
    # Internal: Bracket Management  (identical to previous version)
    # -----------------------------------------------------------------

    def _next_bracket(self) -> dict[str, Any] | None:
        if self._active_bracket is not None:
            return self._active_bracket

        if not self._bracket_queue:
            self._bracket_queue = self._generate_brackets()

        if not self._bracket_queue:
            return None

        self._active_bracket = self._bracket_queue.pop(0)
        bracket = self._active_bracket
        s = bracket["s"]
        n_configs = bracket["n_total"]
        budget = bracket["budgets"][0]

        bracket["configs"] = self._sample_configs_tpe(n_configs)
        bracket["results"] = []
        bracket["evaluated"] = 0

        logger.info(
            f"BOHB bracket start: s={s}, n_configs={n_configs}, "
            f"rung_0_budget={budget} kpts, rungs={len(bracket['budgets'])}"
        )
        return bracket

    def _advance_rung(self, bracket: dict[str, Any]) -> None:
        results = bracket["results"]
        n_keep = max(1, len(results) // self._eta)

        results.sort(key=lambda t: t[1])
        top_configs = [t[0] for t in results[:n_keep]]

        bracket["results"] = []
        bracket["evaluated"] = n_keep

        if len(bracket["budgets"]) <= 1:
            return

        bracket["budgets"].pop(0)
        next_budget = bracket["budgets"][0]

        self._pending_rung_configs = list(top_configs)
        self._pending_rung_budgets = [next_budget] * len(top_configs)

        logger.info(
            f"BOHB rung advance: {n_keep}/{len(results)} configs promoted "
            f"to budget={next_budget} kpts"
        )

    def _finish_bracket(self, bracket: dict[str, Any]) -> None:
        logger.info(
            f"BOHB bracket finished: {bracket.get('evaluated', 0)} evals, "
            f"best observed={self._best_y}"
        )

    def _generate_brackets(self) -> list[dict[str, Any]]:
        brackets = []
        for s in range(self._s_max, -1, -1):
            n_configs = max(
                1,
                math.ceil(float(self._n_configs) * (self._eta**s) / (s + 1)),
            )
            budgets = []
            r = self._max_budget * (self._eta ** (-s))
            for _ in range(s + 1):
                budgets.append(max(self._min_budget, int(r)))
                r = max(self._min_budget, int(r * self._eta))
            brackets.append({
                "s": s,
                "n_total": n_configs,
                "budgets": budgets,
            })
        return brackets

    # -----------------------------------------------------------------
    # Internal: TPE / KDE Sampling
    # -----------------------------------------------------------------

    def _sample_configs_tpe(self, n: int) -> list[np.ndarray]:
        """Sample *n* configurations using TPE (KDE ratio maximisation)."""
        configs: list[np.ndarray] = []
        for _ in range(n):
            configs.append(self._tpe_suggest())
        return configs

    def _tpe_suggest(self) -> np.ndarray:
        """Return a single config vector via TPE: sample from l(x), rank by l/g.

        With probability random_ratio, return a purely random config (exploration).
        Uses continuous KDE on dims [:2] and Aitchison-Aitken on mode index.
        """
        rng = np.random.RandomState(int(id(self) + len(self._X) * 37) % (2**31))

        if rng.random() < self._random_ratio or len(self._X) < 2 * self._eta:
            return self._random_config_vec(rng)

        budget = (
            self._active_bracket["budgets"][0] if self._active_bracket
            else self._min_budget
        )
        X_arr, y_arr = self._observations_at_budget(budget)
        if X_arr.shape[0] < self._eta:
            return self._random_config_vec(rng)

        _good_mask, good, bad, bw = self._build_tpe_kdes(X_arr, y_arr)
        if good.shape[0] == 0 or bad.shape[0] == 0:
            return self._random_config_vec(rng)

        best_vec: np.ndarray | None = None
        best_ratio = -np.inf
        n_cat = len(_CATEGORICAL_MODES)

        for _ in range(self._n_ei_candidates):
            candidate_vec = self._sample_from_kde(good, bw, rng)
            cand_cont = candidate_vec[:2].reshape(1, -1)
            log_l = _gaussian_kde_pdf(cand_cont, good[:, :2], bw)
            log_g = _gaussian_kde_pdf(cand_cont, bad[:, :2], bw)

            mode_idx_l = int(np.argmax(candidate_vec[2:2 + n_cat]))
            mode_idx_arr = np.argmax(good[:, 2:2 + n_cat], axis=1)
            cat_l = _mean_aa_kernel(mode_idx_l, mode_idx_arr, self._cat_lambda, n_cat)

            mode_idx_arr_bad = np.argmax(bad[:, 2:2 + n_cat], axis=1)
            cat_g = _mean_aa_kernel(mode_idx_l, mode_idx_arr_bad, self._cat_lambda, n_cat)

            ratio = (log_l + math.log(max(cat_l, 1e-12))) - (
                log_g + math.log(max(cat_g, 1e-12))
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_vec = candidate_vec

        if best_vec is not None:
            return best_vec
        return self._random_config_vec(rng)

    def _sample_from_kde(
        self, samples: np.ndarray, bw: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """Sample one point from a Gaussian KDE: pick random sample + add noise.

        Continuous dims [:2] get Gaussian noise.  Categorical dims [2:] are
        one-hot, so we set the one-hot column corresponding to the sample's mode.
        """
        idx = rng.randint(0, samples.shape[0])
        base = samples[idx].copy()
        noise = rng.randn(2) * bw
        base[:2] = np.clip(base[:2] + noise, 0.0, 1.0 - 1e-6)
        return base

    def _observations_at_budget(
        self, budget: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays for all observations at the given budget level."""
        if not self._X:
            dim = 2 + len(_CATEGORICAL_MODES)
            return (
                np.empty((0, dim), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        X_arr = np.array(self._X, dtype=np.float64)
        y_arr = np.array(self._y, dtype=np.float64)
        b_arr = np.array(self._budgets, dtype=np.float64)
        # Use observations at the exact budget or nearby
        mask = np.abs(b_arr - budget) <= max(1, budget * 0.1)
        if mask.sum() < self._eta:
            mask = np.abs(b_arr - budget) <= max(2, budget * 0.25)
        if mask.sum() < 2:
            return X_arr, y_arr  # fallback: all observations
        return X_arr[mask], y_arr[mask]

    def _build_tpe_kdes(
        self, X_arr: np.ndarray, y_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Build D_good / D_bad split and return (good_mask, good, bad, bandwidth)."""
        n_good = max(2, int(len(y_arr) * self._gamma))
        cut = np.partition(y_arr, n_good - 1)[n_good - 1]
        good_mask = y_arr <= cut
        good = X_arr[good_mask]
        bad = X_arr[~good_mask]
        bw = _estimate_bandwidth(
            np.vstack([good[:, :2], bad[:, :2]]) if bad.shape[0] > 0 else good[:, :2],
            self._bandwidth_factor,
        )
        return good_mask, good, bad, bw

    # -----------------------------------------------------------------
    # Internal: Random configs
    # -----------------------------------------------------------------

    def _random_config_vec(self, rng: np.random.RandomState) -> np.ndarray:
        cores_val = rng.randint(self._min_cores, self._max_cores + 1)
        mode_val = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]
        omp_val = 1
        if mode_val == "hybrid" and cores_val > 1:
            omp_val = _random_divisor(cores_val, rng)
        return _encode_config(mode_val, cores_val, omp_val)

    def _random_config(self) -> dict[str, Any]:
        rng = np.random.RandomState(int(id(self) * 13) % (2**31))
        cores_val = rng.randint(self._min_cores, self._max_cores + 1)
        mode_val = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]
        omp_val = 1
        if mode_val == "hybrid" and cores_val > 1:
            omp_val = _random_divisor(cores_val, rng)
        return {
            "mode": mode_val,
            "total_cores": cores_val,
            "omp_threads": omp_val,
        }

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def best_observed(self) -> dict[str, Any] | None:
        return self._best_config

    @property
    def n_observations(self) -> int:
        return len(self._X)

    @property
    def current_budget(self) -> int:
        return self._current_budget


def _random_divisor(n: int, rng: np.random.RandomState) -> int:
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i and i != 1:
                divisors.append(n // i)
    if not divisors:
        return 1
    return divisors[rng.randint(0, len(divisors))]


__all__ = ["BOHBOptimizer"]
