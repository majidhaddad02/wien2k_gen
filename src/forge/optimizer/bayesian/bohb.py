"""BOHB: Bayesian Optimization Hyperband for WIEN2k multi-fidelity tuning.

Implements the BOHB algorithm (Falkner, Klein & Hutter, ICML 2018) adapted
for WIEN2k SCF convergence optimisation.  Combines Successive Halving
(Hyperband brackets) with Bayesian Optimisation using Expected Improvement.

In WIEN2k, fidelity maps to k-point count:
  - Low budget  (1-4 k-points)  → ~10-40% cost, noisy but cheap
  - High budget (full k-points) → 100% cost, accurate

Algorithm (per bracket, s ∈ {s_max, ..., 0}):
  1. Sample n = ceil(n_configs * η^s / (s+1)) configurations via BO
  2. Evaluate all at budget r₀ = max_budget * η^(-s)
  3. Select top 1/η performers; evaluate them at budget r₁ = r₀ * η
  4. Repeat until r_i > max_budget or only 1 config remains
  5. Report the best configuration across all brackets

Reference:
  Falkner, S., Klein, A. & Hutter, F. (2018).  BOHB: Robust and Efficient
  Hyperparameter Optimization at Scale.  PMLR 80:1437-1446.
"""

import math
from typing import Any, Optional

import numpy as np

from ...logging_config import get_logger
from .acquisition import compute_expected_improvement
from .gp import _GaussianProcess, _GaussianProcessARD
from .kernels import _EPS
from .sampling import _CATEGORICAL_MODES, _decode_config, _encode_config

logger = get_logger(__name__)

_DEFAULT_ETA = 3
_DEFAULT_MIN_BUDGET = 1
_DEFAULT_N_CONFIGS = 16


class BOHBOptimizer:
    """BOHB optimiser for multi-fidelity WIEN2k parameter tuning.

    Usage::

        bohb = BOHBOptimizer(
            nkpt=42,
            min_budget=1,
            max_budget=42,
            eta=3,
        )
        for _ in range(total_iters):
            config, budget = bohb.suggest()
            walltime = evaluate_config_at_budget(config, budget)
            bohb.observe(config, walltime, budget)
    """

    def __init__(
        self,
        nkpt: int,
        min_budget: int = _DEFAULT_MIN_BUDGET,
        max_budget: Optional[int] = None,
        eta: int = _DEFAULT_ETA,
        n_configs: int = _DEFAULT_N_CONFIGS,
        min_cores: int = 1,
        max_cores: int = 256,
        use_ard: bool = True,
        exploration_xi: float = 0.01,
        n_random_restarts: int = 50,
    ) -> None:
        """Initialise BOHB optimiser.

        Args:
            nkpt: Number of irreducible k-points (maps to max_budget).
            min_budget: Minimum k-point budget per evaluation (≥ 1).
            max_budget: Maximum k-point budget (default: nkpt).
            eta: Halving factor — fraction of configs kept per rung (≥ 2).
            n_configs: Nominal number of random configs per bracket.
            min_cores: Minimum total cores.
            max_cores: Maximum total cores.
            use_ard: Use ARD kernel for GP.
            exploration_xi: EI exploration parameter.
            n_random_restarts: Random restarts for acquisition optimisation.
        """
        self._nkpt = nkpt
        self._min_budget = max(1, min_budget)
        self._max_budget = max_budget if max_budget is not None else max(1, nkpt)
        self._eta = max(2, eta)
        self._n_configs = n_configs
        self._min_cores = min_cores
        self._max_cores = max_cores
        self._use_ard = use_ard
        self._exploration_xi = exploration_xi
        self._n_random_restarts = n_random_restarts

        self._s_max = math.floor(math.log(self._max_budget / self._min_budget) / math.log(self._eta))

        self._observations: list[dict[str, Any]] = []
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._budgets: list[int] = []

        self._best_y: Optional[float] = None
        self._best_config: Optional[dict[str, Any]] = None

        self._gp: _GaussianProcess = (
            _GaussianProcessARD() if use_ard else _GaussianProcess()
        )
        self._n_dims = 2 + len(_CATEGORICAL_MODES)

        self._active_bracket: Optional[dict[str, Any]] = None
        self._bracket_queue: list[dict[str, Any]] = []
        self._pending_rung_configs: list[np.ndarray] = []
        self._pending_rung_budgets: list[int] = []

        self._current_suggestion: Optional[dict[str, Any]] = None
        self._current_budget: int = self._min_budget

        logger.info(
            f"BOHB initialised: nkpt={nkpt}, budget_range=[{self._min_budget}, {self._max_budget}], "
            f"eta={eta}, s_max={self._s_max}"
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def suggest(self) -> tuple[dict[str, Any], int]:
        """Return the next (config, budget) to evaluate.

        If a bracket is active, returns the next rung config.  Otherwise
        starts a new Hyperband bracket with BO-sampled configs.

        Returns:
            (config_dict, budget_in_kpoints) — config has keys mode,
            total_cores, omp_threads; budget is effective k-point count.
        """
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
        """Record the result of evaluating *config* at *budget* k-points.

        Args:
            config: The configuration that was evaluated.
            walltime: Observed walltime in seconds (lower is better).
            budget: Effective k-point count used.
        """
        x = _encode_config(config.get("mode", "kpoint"), config.get("total_cores", 1), config.get("omp_threads", 1))
        self._X.append(x)
        self._y.append(walltime)
        self._budgets.append(budget)

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

        if self._pending_rung_configs and bracket is not None:
            full_count = len(bracket["configs"]) + len(self._pending_rung_configs) + bracket["evaluated"]
            if bracket["evaluated"] >= full_count:
                self._advance_rung(bracket)
                bracket = None

        if bracket is not None and not self._pending_rung_configs:
            self._finish_bracket(bracket)
            self._active_bracket = None

        self._refit_gp()

    # -----------------------------------------------------------------
    # Internal: Bracket Management
    # -----------------------------------------------------------------

    def _next_bracket(self) -> Optional[dict[str, Any]]:
        """Return the next active bracket, creating one if necessary."""
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

        bracket["configs"] = self._sample_configs_bohb(n_configs)
        bracket["results"] = []
        bracket["evaluated"] = 0

        logger.info(
            f"BOHB bracket start: s={s}, n_configs={n_configs}, "
            f"rung_0_budget={budget} kpts, rungs={len(bracket['budgets'])}"
        )
        return bracket

    def _advance_rung(self, bracket: dict[str, Any]) -> None:
        """Promote the top 1/eta configs to the next rung budget."""
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
        """Mark a bracket as complete."""
        logger.info(
            f"BOHB bracket finished: {bracket.get('evaluated', 0)} evals, "
            f"best observed={self._best_y}"
        )

    def _generate_brackets(self) -> list[dict[str, Any]]:
        """Generate Hyperband brackets from s_max down to 0."""
        brackets = []
        for s in range(self._s_max, -1, -1):
            n_configs = max(1, math.ceil(
                float(self._n_configs) * (self._eta ** s) / (s + 1)
            ))
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
    # Internal: Configuration Sampling
    # -----------------------------------------------------------------

    def _sample_configs_bohb(self, n: int) -> list[np.ndarray]:
        """Sample *n* configurations using BO-EI when observations exist.

        If few observations (< 2), falls back to random sampling.
        Otherwise, repeatedly selects the best EI point and adds it as an
        artificial "pending" observation to encourage diversity.
        """
        rng = np.random.RandomState(int(id(self) + len(self._X) * 37) % (2 ** 31))
        configs: list[np.ndarray] = []

        if len(self._X) < 2:
            for _ in range(n):
                cores_val = rng.randint(self._min_cores, self._max_cores + 1)
                mode_val = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]
                omp_val = 1
                if mode_val == "hybrid" and cores_val > 1:
                    omp_val = _random_divisor(cores_val, rng)
                configs.append(_encode_config(mode_val, cores_val, omp_val))
            return configs

        for _ in range(n):
            best_vec: Optional[np.ndarray] = None
            best_ei = -1.0

            for _ in range(self._n_random_restarts):
                cores_val = rng.randint(self._min_cores, self._max_cores + 1)
                omp_val = rng.randint(1, min(65, cores_val + 1))
                mode_val = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]
                candidate = _encode_config(mode_val, cores_val, omp_val)

                try:
                    mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                    mu_val = float(mu[0])
                    sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))
                    ei = compute_expected_improvement(
                        mu_val, sigma_val, self._best_y or float("inf"),
                        xi=self._exploration_xi,
                    )
                    if ei > best_ei:
                        best_ei = ei
                        best_vec = candidate.copy()
                except Exception:
                    continue

            if best_vec is not None:
                configs.append(best_vec)
            else:
                cores_val = rng.randint(self._min_cores, self._max_cores + 1)
                mode_val = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]
                configs.append(_encode_config(mode_val, cores_val, 1))

        return configs

    def _refit_gp(self) -> None:
        """Re-fit the GP to all accumulated observations."""
        if len(self._X) < 2:
            return
        X_arr = np.array(self._X, dtype=np.float64)
        y_arr = np.array(self._y, dtype=np.float64)
        self._gp = (
            _GaussianProcessARD() if self._use_ard else _GaussianProcess()
        )
        self._gp.fit(X_arr, y_arr)

    def _random_config(self) -> dict[str, Any]:
        """Return a random valid configuration."""
        rng = np.random.RandomState(int(id(self) * 13) % (2 ** 31))
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
    def best_observed(self) -> Optional[dict[str, Any]]:
        """Return the best configuration observed so far."""
        return self._best_config

    @property
    def n_observations(self) -> int:
        """Total number of evaluations performed."""
        return len(self._X)

    @property
    def current_budget(self) -> int:
        """Budget of the most recently suggested configuration."""
        return self._current_budget


def _random_divisor(n: int, rng: np.random.RandomState) -> int:
    """Return a random divisor of n (excluding n itself for parallelism)."""
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
