"""Core Bayesian optimizer: orchestrator, warm-start, and main optimization loop."""

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ...logging_config import get_logger
from ...optimizer.history import ExecutionHistory, ExecutionRecord
from .acquisition import (
    _DEFAULT_EI_THRESHOLD,
    compute_expected_improvement,
    compute_q_expected_improvement,
)
from .constraints import (
    _estimate_memory_gb_for_config,
    _estimate_walltime_min_for_config,
    _sigmoid_feasibility,
)
from .elements import _chemical_similarity
from .gp import _GaussianProcess, _GaussianProcessARD
from .kernels import _EPS
from .sampling import _CATEGORICAL_MODES, _decode_config, _encode_config, latin_hypercube_sampling

logger = get_logger(__name__)

_FIDELITY_COST = {0: 0.10, 1: 0.40, 2: 1.00}
_FIDELITY_CORRELATION = {0: 0.60, 1: 0.85, 2: 1.00}

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


def add_physics_priors(
    structure: dict[str, Any],
    nmat: int = 0,
    is_soc: bool = False,
    is_metallic: bool = False,
) -> dict[str, Any]:
    """
    Enforce physically-motivated parameter constraints.

    Based on Blaha et al. (2020), J. Chem. Phys. 152, 074101 and
    WIEN2k User Guide 2023:
      - RKMAX ≥ 7.0 for light hard elements (O, F, N) — hard potentials
      - RKMAX ≥ 7.0 for SOC calculations — spin-orbit requires high cutoff
      - mixing ≤ 0.3 for metallic systems — Kerker preconditioning needed
      - kpt density ≥ 1000 kpts/Å⁻³ for metals — Fermi surface resolution

    Returns dict of constraints with min, max, and recommended default.
    """
    atoms = structure.get("atoms", [])
    atomic_numbers = [a.get("z_num", 1) for a in atoms]
    hard_elements = {8, 9, 7, 16, 17}
    has_hard = any(z in hard_elements for z in atomic_numbers)

    constraints = {
        "rkmax": {"min": 5.0, "max": 9.0, "default": 7.0},
        "mixing_beta": {"min": 0.05, "max": 1.0, "default": 0.30},
        "kpoint_density": {"min": 100, "max": 2000, "default": 500},
        "gmax": {"min": 10.0, "max": 20.0, "default": 14.0},
        "lmax_apw": {"min": 8, "max": 12, "default": 10},
        "warnings": [],
    }

    if has_hard:
        constraints["rkmax"]["min"] = 7.0
        constraints["rkmax"]["default"] = 7.0
        constraints["warnings"].append(
            "Light hard elements (O/F/N) detected — RKMAX set to ≥ 7.0"
        )

    if is_soc:
        constraints["rkmax"]["min"] = max(constraints["rkmax"]["min"], 7.0)
        constraints["warnings"].append(
            "SOC calculation — RKMAX ≥ 7.0 required for reliable results"
        )

    if is_metallic:
        constraints["mixing_beta"]["max"] = 0.30
        constraints["mixing_beta"]["default"] = 0.10
        constraints["kpoint_density"]["min"] = 1000
        constraints["warnings"].append(
            "Metallic system — mixing ≤ 0.30, kpt density ≥ 1000 recommended"
        )

    if nmat > 15000:
        constraints["warnings"].append(
            f"Large basis (nmat={nmat}) — consider parallel BO with q-EI batch evaluation"
        )

    return constraints


def load_warm_start_history(
    history_file: str = ".bo_history.json",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load previous Bayesian optimization results for warm starting.

    Reads .bo_history.json and extracts (X, y) from completed evaluations.
    Returns (X, y) or (None, None) if no history exists.
    """
    path = Path(history_file)
    if not path.exists():
        return None, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "evaluations" not in data or not data["evaluations"]:
            return None, None

        X_list, y_list = [], []
        for ev in data["evaluations"]:
            params = ev.get("params", {})
            cost = ev.get("cost", 0.0)
            if params and cost > 0:
                X_list.append([
                    params.get("rkmax", 7.0),
                    params.get("mixing_beta", 0.30),
                    params.get("kpoint_density", 500),
                    params.get("gmax", 14.0),
                ])
                y_list.append(cost)

        if X_list and y_list:
            X = np.array(X_list, dtype=np.float64)
            y = np.array(y_list, dtype=np.float64)
            logger.info(f"Warm start loaded: {len(X_list)} previous evaluations")
            return X, y
    except Exception as e:
        logger.warning(f"Failed to load warm start history: {e}")

    return None, None


def save_bo_history(history_file: str, evaluations: list[dict[str, Any]]) -> None:
    """Save Bayesian optimization results for future warm starts."""
    path = Path(history_file)
    data = {"evaluations": evaluations}
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info(f"BO history saved: {len(evaluations)} evaluations → {history_file}")


# =============================================================================
# Bayesian Optimization Loop
# =============================================================================

def define_search_space(structure: dict[str, Any]) -> dict[str, Any]:
    """Define search space bounds and types for WIEN2k parameters.

    Returns dict with bounds, types, and defaults for each parameter.
    """
    atoms = structure.get("atoms", [])
    numeric_z = [int(a.get("z_num", 1)) for a in atoms]

    # Detect system type for starting point heuristics
    has_heavy = any(z > 50 for z in numeric_z)
    has_light_hard = any(z in {7, 8, 9} for z in numeric_z)

    defaults = {
        "rkmax": 7.5 if has_heavy else (7.0 if has_light_hard else 6.5),
        "mixing_beta": 0.10,
        "kpoint_density": 500,
        "gmax": 14.0,
        "lmax_apw": 10,
    }

    return {
        "rkmax": {"bounds": (5.0, 9.0), "type": "continuous", "default": defaults["rkmax"]},
        "mixing_beta": {"bounds": (0.05, 1.0), "type": "continuous", "default": defaults["mixing_beta"]},
        "kpoint_density": {"bounds": (100, 2000), "type": "integer", "default": defaults["kpoint_density"]},
        "gmax": {"bounds": (10.0, 20.0), "type": "continuous", "default": defaults["gmax"]},
        "lmax_apw": {"bounds": (8, 12), "type": "discrete", "default": defaults["lmax_apw"]},
    }


def bayesian_optimize_scf_params(
    structure: dict[str, Any],
    eval_objective: Callable[[dict[str, float]], float],
    budget: int = 20,
    initial_samples: int = 10,
    kernel_type: str = "matern",
    acquisition: str = "EI",
    parallel_batch: int = 4,
    warm_start: bool = True,
    history_file: str = ".bo_history.json",
) -> dict[str, Any]:
    """Full Bayesian optimization loop for WIEN2k SCF parameters.

    Algorithm:
      1. Define search space from structure
      2. LHS initial sampling (or warm start from previous results)
      3. Evaluate objective for initial points
      4. Fit GP with Matérn kernel
      5. Maximize EI/q-EI acquisition function
      6. Evaluate and update GP
      7. Repeat until budget exhausted
      8. Return best parameters with convergence diagnostics

    Args:
        structure: Crystal structure dict from case_parser.
        eval_objective: Function mapping params → cost (lower is better).
        budget: Maximum number of total evaluations.
        initial_samples: Number of initial LHS samples.
        kernel_type: "matern" or "rbf".
        acquisition: "EI" or "qEI" acquisition function.
        parallel_batch: Batch size for q-EI.
        warm_start: Load previous results if available.
        history_file: Path to BO history JSON for warm start.

    Returns:
        Dict with best_params, best_cost, evaluations, convergence_info.
    """
    space = define_search_space(structure)
    bounds = [space[k]["bounds"] for k in ["rkmax", "mixing_beta", "kpoint_density", "gmax"]]
    param_names = ["rkmax", "mixing_beta", "kpoint_density", "gmax"]
    dims = len(bounds)
    evaluations: list[dict[str, Any]] = []

    # Initial samples: warm start or LHS
    X_init = None
    y_init = None
    if warm_start:
        X_warm, y_warm = load_warm_start_history(history_file)
        if X_warm is not None and len(X_warm) >= 2:
            X_init = X_warm
            y_init = y_warm
            logger.info(f"Using {len(X_init)} warm start points")

    if X_init is None:
        n_init = min(initial_samples, budget)
        X_init = latin_hypercube_sampling(bounds, n_init)
        y_init = np.array([eval_objective(_params_dict(X_init[i], param_names))
                           for i in range(n_init)])
        for i in range(n_init):
            evaluations.append({
                "iteration": i,
                "params": _params_dict(X_init[i], param_names),
                "cost": float(y_init[i]),
            })

    # Main BO loop
    X = X_init.copy()
    y = y_init.copy()
    best_idx = int(np.argmin(y))
    best_params = _params_dict(X[best_idx], param_names)
    best_cost = float(y[best_idx])

    for iteration in range(len(evaluations), budget):
        # Fit GP
        gp = _GaussianProcess(length_scales=np.ones(dims))
        gp.fit(X, y)

        # Acquisition function
        if acquisition == "qEI":
            n_test = min(2000, 50 * dims)
            X_test = latin_hypercube_sampling(bounds, n_test)
            batch_indices = compute_q_expected_improvement(
                gp, X_test, best_cost, q=parallel_batch)
            next_indices = batch_indices if batch_indices else [int(np.argmin(gp.predict(X_test)[0]))]
        else:
            n_test = min(2000, 50 * dims)
            X_test = latin_hypercube_sampling(bounds, n_test)
            mu_test, var_test = gp.predict(X_test)
            ei_vals = np.array([
                compute_expected_improvement(float(mu_test[i]), float(np.sqrt(max(var_test[i], 0))), best_cost)
                for i in range(n_test)
            ])
            next_indices = [int(np.argmax(ei_vals))]

        for next_idx in next_indices:
            next_params = _params_dict(X_test[next_idx], param_names)
            next_y = eval_objective(next_params)

            X = np.vstack([X, X_test[next_idx].reshape(1, -1)])
            y = np.append(y, next_y)

            evaluations.append({
                "iteration": iteration,
                "params": next_params,
                "cost": float(next_y),
            })

            if next_y < best_cost:
                best_cost = float(next_y)
                best_params = next_params

        save_bo_history(history_file, evaluations)
        logger.info(
            f"BO iter {iteration}: cost={next_y:.4f}, "
            f"best={best_cost:.4f}"
        )

    return {
        "best_params": best_params,
        "best_cost": best_cost,
        "evaluations": evaluations,
        "n_evaluations": len(evaluations),
        "kernel_type": kernel_type,
        "acquisition": acquisition,
    }


def _divisors(n: int) -> list[int]:
    """All positive divisors of n (including 1 and n), for valid OMP splits."""
    result = []
    for i in range(1, math.isqrt(n) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)
    return sorted(result)


def _params_dict(x: np.ndarray, names: list[str]) -> dict[str, float]:
    """Convert numpy array to parameter dict."""
    out = {}
    for i, name in enumerate(names):
        if i < len(x):
            out[name] = float(x[i])
    return out


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
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
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
        with self._lock:
            best_y_val = self._best_y
            n_x = len(self._X)
        logger.debug(
            f"BayesianOptimizer updated: best_y={best_y_val:.2f}s, n_obs={n_x}"
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

    def auto_transfer_learn(  # noqa: C901
        self,
        target_elements: list[str],
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> float:
        """Automatically transfer knowledge from chemically similar systems.

        Scans the execution history for all records with known elements,
        computes chemical similarity using the multi-factor scheme from
        :func:`_chemical_similarity`, and builds a weighted prior for
        the GP.  The prior mean is blended with the GP prediction via
        ``transfer_weight * prior + (1 - transfer_weight) * gp`.

        The method returns the aggregate similarity weight (0-1).  If no
        sufficiently similar systems are found, the transfer weight is
        set to zero (standard zero-mean GP prior).

        Args:
            target_elements: Chemical symbols of the target system's
                             elements (e.g. ``["Fe", "O"]``).
            top_k: Number of most similar historical systems to use.
            min_similarity: Minimum chemical similarity to consider a
                            source system useful.

        Returns:
            Float similarity weight used for blending.
        """
        target_z = {_ELEMENT_ATOMIC_NUMBERS.get(e, 0) for e in target_elements}
        target_z.discard(0)

        if not target_z:
            logger.warning("No valid target elements for transfer learning.")
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return 0.0

        all_records = self._history.query(
            filters={"backend": self.backend, "success": True},
            order_by="walltime_sec ASC",
            limit=500,
        )

        if not all_records:
            logger.warning("Empty execution history — skipping transfer learning.")
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return 0.0

        system_groups: dict[tuple, list[Any]] = {}

        for rec in all_records:
            rec_tags = getattr(rec, "tags", None)
            if not rec_tags:
                continue
            if isinstance(rec_tags, str):
                try:
                    rec_tags = json.loads(rec_tags)
                except Exception:
                    continue
            rec_elements = [t for t in rec_tags if isinstance(t, str) and t in _ELEMENT_ATOMIC_NUMBERS]
            if not rec_elements:
                continue
            rec_z = {_ELEMENT_ATOMIC_NUMBERS[e] for e in rec_elements}
            rec_z.discard(0)
            if not rec_z:
                continue

            sims = []
            for tz in target_z:
                for sz in rec_z:
                    sims.append(_chemical_similarity(sz, tz))
            if not sims:
                continue
            max_sim = max(sims)

            if max_sim < min_similarity:
                continue

            key = tuple(sorted(rec_elements))
            if key not in system_groups:
                system_groups[key] = []
            system_groups[key].append(rec)

        system_scores: list[tuple[float, list[Any]]] = []
        for key, recs in system_groups.items():
            key_z = {_ELEMENT_ATOMIC_NUMBERS[e] for e in key}
            key_z.discard(0)
            sims = []
            for tz in target_z:
                for kz in key_z:
                    sims.append(_chemical_similarity(kz, tz))
            score = max(sims) if sims else 0.0
            if score >= min_similarity:
                system_scores.append((score, recs))

        if not system_scores:
            logger.info(
                f"No chemically similar systems found (min similarity={min_similarity}). "
                f"Using standard GP prior."
            )
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return 0.0

        system_scores.sort(key=lambda x: -x[0])
        system_scores = system_scores[:top_k]

        source_X_all = []
        source_y_all = []
        weights_all = []

        for sim_score, recs in system_scores:
            for rec in recs:
                if rec.walltime_sec <= 0:
                    continue
                source_X_all.append(_encode_config(rec.mode, rec.total_cores, rec.omp_threads))
                source_y_all.append(rec.walltime_sec)
                weights_all.append(sim_score)

        if len(source_X_all) < 2:
            logger.warning("Insufficient transfer data — falling back to standard GP prior.")
            self._transfer_mean = None
            self._transfer_weight = 0.0
            return 0.0

        source_X_arr = np.array(source_X_all, dtype=np.float64)
        source_y_arr = np.array(source_y_all, dtype=np.float64)
        weights_arr = np.array(weights_all, dtype=np.float64)

        w_sum = weights_arr.sum()
        transfer_weight = min(1.0, w_sum / len(weights_arr))

        source_gp = _GaussianProcess()
        source_gp.fit(source_X_arr, source_y_arr)

        self._transfer_mean = source_X_arr.mean(axis=0)
        self._transfer_weight = transfer_weight
        self._source_system = ",".join(target_elements)

        self._gp = (
            _GaussianProcessARD() if self._use_ard else _GaussianProcess()
        )

        logger.info(
            f"Auto-transfer: {len(system_scores)} similar systems found, "
            f"blend weight={transfer_weight:.4f}, "
            f"{len(source_X_all)} source records used."
        )
        return transfer_weight

    def suggest_next(
        self,
        nmat: int,
        nkpt: int,
        user_max_cores: Optional[int] = None,
    ) -> dict[str, Any]:
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
                        mu_val = (1.0 - self._transfer_weight) * mu_val + \
                                 self._transfer_weight * (float(np.mean(self._y)) if len(self._y) > 0 else mu_val)

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
    ) -> dict[str, Any]:
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
                mode, omp = self._random_valid_mode_omp(cores, rng)

                candidate = _encode_config(mode, cores, omp)

                try:
                    mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                    mu_val = float(mu[0])
                    sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))

                    if self._transfer_mean is not None and self._transfer_weight > 0:
                        mu_val = (1.0 - self._transfer_weight) * mu_val + \
                                 self._transfer_weight * (float(np.mean(self._y)) if len(self._y) > 0 else mu_val)

                    ei = compute_expected_improvement(
                        mu_val, sigma_val, current_best, xi=self._exploration_xi
                    )

                    est_mem = _estimate_memory_gb_for_config(nmat, cores, omp)
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

    @staticmethod
    def _random_valid_mode_omp(cores: int, rng: np.random.RandomState) -> tuple[str, int]:
        """Generate a valid (mode, omp) pair consistent with physical core count.

        Rules:
          - cores == 1 → kpoint mode only, omp = 1
          - hybrid mode → omp must divide cores evenly
          - mpi/kpoint → omp = 1
        """
        if cores <= 1:
            return ("kpoint", 1)

        mode = _CATEGORICAL_MODES[rng.randint(0, len(_CATEGORICAL_MODES))]

        if mode == "hybrid" and cores > 1:
            divisors = _divisors(cores)
            omp = divisors[rng.randint(0, len(divisors))]
        else:
            omp = 1

        return (mode, omp)

    def _random_suggestion(self, max_cores: int) -> dict[str, Any]:
        """Fallback: return a randomly generated plausible configuration."""
        rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))
        cores = max(1, rng.randint(self.min_cores, max_cores + 1))
        mode, omp = self._random_valid_mode_omp(cores, rng)

        return {
            "mode": mode,
            "total_cores": cores,
            "omp_threads": omp,
            "expected_improvement": 0.0,
            "predicted_mean": 0.0,
            "predicted_std": 0.0,
            "source": "random",
        }

    def get_best_observed(self) -> Optional[dict[str, Any]]:
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

    def get_parameter_relevance(self) -> dict[str, float]:
        """Return ARD relevance scores if using ARD kernel."""
        with self._lock:
            if isinstance(self._gp, _GaussianProcessARD):
                return self._gp.get_relevance()
            return {}


# =============================================================================
# MultiFidelityBayesianOptimizer Class
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
        self._fidelity_eval_count: dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._ei_promotion_threshold = ei_promotion_threshold
        self._min_fidelity_evals = min_fidelity_evals
        self._fidelity_gps: dict[int, _GaussianProcess] = {}
        self._fidelity_X: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
        self._fidelity_y: dict[int, list[float]] = {0: [], 1: [], 2: []}

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
    ) -> tuple[dict[str, Any], int]:
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
            best_config: Optional[dict[str, Any]] = None
            best_fid = 0

            rng = np.random.RandomState(int(time.time() * 1e6) % (2 ** 31))

            for fid_level in [0, 1, 2]:
                corr = _FIDELITY_CORRELATION[fid_level]
                cost = _FIDELITY_COST[fid_level]

                for _ in range(self._n_random_restarts):
                    cores = rng.randint(self.min_cores, max_cores + 1)
                    mode, omp = BayesianOptimizer._random_valid_mode_omp(cores, rng)

                    candidate = _encode_config(mode, cores, omp)

                    try:
                        mu, sigma2 = self._gp.predict(candidate.reshape(1, -1))
                        mu_val = float(mu[0])
                        sigma_val = float(math.sqrt(max(sigma2[0], _EPS)))

                        if self._transfer_mean is not None and self._transfer_weight > 0:
                            mu_val = (1.0 - self._transfer_weight) * mu_val + \
                                     self._transfer_weight * (float(np.mean(self._y)) if len(self._y) > 0 else mu_val)

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

    def get_fidelity_stats(self) -> dict[str, Any]:
        """Return per-fidelity evaluation counts and costs."""
        with self._lock:
            return {
                "evals_per_fidelity": dict(self._fidelity_eval_count),
                "cost_multipliers": dict(_FIDELITY_COST),
                "correlation_weights": dict(_FIDELITY_CORRELATION),
                "current_fidelity": self._current_fidelity,
            }


__all__ = [
    "BayesianOptimizer",
    "MultiFidelityBayesianOptimizer",
    "add_physics_priors",
    "bayesian_optimize_scf_params",
    "define_search_space",
    "load_warm_start_history",
    "save_bo_history",
]
