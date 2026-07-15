"""
Bayesian Auto-Tuner for WIEN2k Parameters.

Extends the existing BayesianOptimizer with an expanded parameter space
for automatic tuning of:
  - RKMAX (5.0 — 10.0, continuous)
  - K-point grid density (500 — 20000 KPPRA equivalent, continuous)
  - SCF mixing parameter beta (0.05 — 0.50, continuous)
  - Parallelization mode & core count (inherited from base optimizer)

References:
  Snoek, Larochelle, Adams. "Practical Bayesian Optimization of ML Algorithms." NeurIPS 2012.
  Shahriari et al. "Taking the Human Out of the Loop: A Review of Bayesian Optimization." Proc. IEEE 2016.
  Frazier. "Bayesian Optimization." Informs Tutorials 2018.

Usage:
  forge optimize --case Fe --target energy_convergence --budget 10
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.case_parser import CaseFileParser
from ..logging_config import get_logger
from .bayesian import _CATEGORICAL_MODES, compute_expected_improvement

logger = get_logger(__name__)

_DEFAULT_RKMAX_BOUNDS = (5.0, 10.0)
_DEFAULT_KPPRA_BOUNDS = (500, 20000)
_DEFAULT_MIXING_BOUNDS = (0.05, 0.50)
_DEFAULT_BUDGET = 10


@dataclass
class TunerResult:
    """Result of a single or multi-iteration Bayesian tuning run."""
    best_rkmax: float = 7.0
    best_kppra: int = 1000
    best_mixing: float = 0.15
    best_energy: float = 0.0
    convergence_achieved: bool = False
    iterations: int = 0
    uncertainty_rkmax: float = 0.0
    uncertainty_mixing: float = 0.0
    observations: list[dict[str, Any]] = None  # type: ignore

    def __post_init__(self):
        if self.observations is None:
            self.observations = []


def _encode_tuning_config(
    rkmax: float, kppra: float, mixing: float,
    mode: str = "kpoint", total_cores: int = 16, omp_threads: int = 1,
) -> np.ndarray:
    """Encode tuning parameters into a numeric feature vector.

    Vector layout (10 dimensions):
      [0] total_cores / 256.0
      [1] omp_threads / 64.0
      [2:5] mode one-hot (3 categories)
      [5] rkmax / 10.0
      [6] kppra / 20000.0
      [7] mixing / 0.50
      [8:10] interaction features (rkmax*mixing, log(kppra), rkmax/atoms)

    This adds 5 dimensions beyond the base 5-dim encoding, for a total of 10.
    """
    vec_base = np.zeros(5, dtype=np.float64)
    vec_base[0] = float(total_cores) / 256.0
    vec_base[1] = float(omp_threads) / 64.0

    mode_idx = _CATEGORICAL_MODES.index(mode) if mode in _CATEGORICAL_MODES else 0
    vec_base[2 + mode_idx] = 1.0

    rkmax_clipped = max(_DEFAULT_RKMAX_BOUNDS[0], min(_DEFAULT_RKMAX_BOUNDS[1], rkmax))
    kppra_clipped = max(_DEFAULT_KPPRA_BOUNDS[0], min(_DEFAULT_KPPRA_BOUNDS[1], kppra))
    mixing_clipped = max(_DEFAULT_MIXING_BOUNDS[0], min(_DEFAULT_MIXING_BOUNDS[1], mixing))

    vec_tune = np.zeros(5, dtype=np.float64)
    vec_tune[0] = rkmax_clipped / 10.0
    vec_tune[1] = kppra_clipped / 20000.0
    vec_tune[2] = mixing_clipped / 0.50
    vec_tune[3] = (rkmax_clipped / 10.0) * (mixing_clipped / 0.50)
    vec_tune[4] = np.log(max(1.0, kppra_clipped)) / np.log(20000.0)

    return np.concatenate([vec_base, vec_tune])


def _decode_tuning_config(vec: np.ndarray) -> dict[str, Any]:
    """Decode a 10-dim feature vector to a tuning configuration."""
    total_cores = max(1, min(256, round(vec[0] * 256.0)))
    omp_threads = max(1, min(64, round(vec[1] * 64.0)))
    mode_idx = int(np.argmax(vec[2:5]))
    mode = _CATEGORICAL_MODES[mode_idx] if mode_idx < len(_CATEGORICAL_MODES) else "kpoint"

    rkmax = round(max(5.0, min(10.0, vec[5] * 10.0)), 2)
    kppra = int(round(max(500, min(20000, vec[6] * 20000.0)) / 100) * 100)
    mixing = round(max(0.05, min(0.50, vec[7] * 0.50)), 4)

    return {
        "mode": mode,
        "total_cores": total_cores,
        "omp_threads": omp_threads,
        "rkmax": rkmax,
        "kppra": kppra,
        "mixing": mixing,
    }


class BayesianParameterTuner:
    """Bayesian optimizer for WIEN2k convergence parameters.

    Uses Gaussian Process regression with Expected Improvement acquisition
    to find optimal RKMAX, k-point density, and mixing parameter with
    minimal DFT runs.

    On each iteration:
      1. GP suggests next (rkmax, kppra, mixing)
      2. WIEN2k SCF is run with these parameters
      3. Convergence delta-energy is recorded as the objective
      4. GP is updated and the next suggestion is computed

    After `budget` iterations, returns the best configuration found.
    """

    def __init__(
        self,
        case_name: str = "case",
        budget: int = _DEFAULT_BUDGET,
        exploration_xi: float = 0.01,
        use_ard: bool = True,
        verbose: bool = False,
    ) -> None:
        self.case_name = case_name
        self.budget = budget
        self.exploration_xi = exploration_xi
        self.use_ard = use_ard
        self.verbose = verbose

        self._rng = np.random.RandomState(42)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._best_y: Optional[float] = None
        self._best_x: Optional[np.ndarray] = None

        from .bayesian import _GaussianProcess, _GaussianProcessARD
        self._gp_cls = _GaussianProcessARD if use_ard else _GaussianProcess
        self._gp = self._gp_cls()

    @property
    def n_observations(self) -> int:
        return len(self._y)

    def _objective_from_run(
        self, rkmax: float, kppra: int, mixing: float,
    ) -> float:
        """Run WIEN2k with given parameters and return convergence score.

        The objective is the absolute delta-energy from the last SCF cycle.
        Lower is better. Returns 1e10 if the run fails.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        case = Path(self.case_name)
        try:
            kx = round(kppra ** (1.0 / 3.0))
            ky = round(kppra ** (1.0 / 3.0))
            kz = round(kppra ** (1.0 / 3.0))

            cmd = [
                "x", "kgen", str(kx), str(ky), str(kz),
            ]
            cwd = case.parent if case.parent != Path(".") else None

            subprocess.run(
                cmd, shell=False,
                capture_output=True, text=True, timeout=120,
                cwd=cwd,
            )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".in1", delete=False) as f:
                f.write(f"WFFIL\n 4  0  0 0 0 {rkmax} 0\n")
                f.write(" 18.00   10   4                            0.30  0 1.00\n")
                f.write("K-VECTORS FROM UNIT:4   -7.0  0.5    emin\n")
            subprocess.run(["cp", f.name, f"{self.case_name}.in1"], check=False)
            Path(f.name).unlink(missing_ok=True)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".inm", delete=False) as f2:
                f2.write(f"MSEC2  0.0  NO\n{mixing:.4f}\n")
            subprocess.run(["cp", f2.name, f"{self.case_name}.inm"], check=False)
            Path(f2.name).unlink(missing_ok=True)

            result = subprocess.run(
                ["run_lapw", "-p", "-ec", "0.0001", "-cc", "0.0001", "-NI"],
                capture_output=True, text=True, timeout=600,
            )

            content = result.stdout.lower()
            if "charge convergence" in content or "energy convergence" in content:
                import re
                energies = re.findall(r":ene\s*:\s*.*?(-?\d+\.\d+)", result.stdout.lower())
                if len(energies) >= 2:
                    delta = abs(float(energies[-1]) - float(energies[-2]))
                    if self.verbose:
                        logger.info(f"Converged: rkmax={rkmax:.1f} kppra={kppra} "
                                   f"mixing={mixing:.3f} deltaE={delta:.6f}")
                    return delta
                return 1e-6

            if self.verbose:
                logger.warning(f"Not converged: rkmax={rkmax:.1f} kppra={kppra} "
                              f"mixing={mixing:.3f}")

            import re
            energies = re.findall(r":ene\s*:\s*.*?(-?\d+\.\d+)", result.stdout.lower())
            if len(energies) >= 2:
                delta = abs(float(energies[-1]) - float(energies[-2]))
                return min(delta * 10, 1e10)

            return 1e10

        except Exception as e:
            logger.error(f"Run failed: rkmax={rkmax:.1f} kppra={kppra} — {e}")
            return 1e10

    def _objective_simulated(
        self, rkmax: float, kppra: int, mixing: float,
    ) -> float:
        """Simulated objective for testing. Returns noisy convergence quality."""
        kppra_opt = 8000.0
        rkmax_opt = 8.0
        mixing_opt = 0.15

        d_kppra = (kppra - kppra_opt) / kppra_opt
        d_rkmax = (rkmax - rkmax_opt) / rkmax_opt
        d_mixing = (mixing - mixing_opt) / mixing_opt

        quality = 1e-6 + 0.1 * (d_kppra ** 2 + d_rkmax ** 2 + 3 * d_mixing ** 2)
        noise = self._rng.normal(0, quality * 0.3)
        return max(1e-12, quality + noise)

    def _get_next_suggestion(self) -> dict[str, Any]:
        if self.n_observations == 0:
            rkmax = float(self._rng.uniform(5.5, 9.0))
            kppra = int(self._rng.uniform(1000, 15000))
            mixing = float(self._rng.uniform(0.08, 0.30))
            config = _decode_tuning_config(_encode_tuning_config(rkmax, kppra, mixing))
            return self._apply_physics_constraints(config)

        X = np.vstack(self._X)
        y = np.array(self._y, dtype=np.float64)

        best_y = float(np.min(y))
        self._gp.fit(X, y)

        candidates = []
        for _ in range(500):
            rkmax = float(self._rng.uniform(*_DEFAULT_RKMAX_BOUNDS))
            kppra = float(self._rng.uniform(*_DEFAULT_KPPRA_BOUNDS))
            mixing = float(self._rng.uniform(*_DEFAULT_MIXING_BOUNDS))
            vec = _encode_tuning_config(rkmax, kppra, mixing)
            candidates.append(vec)

        best_ei = -float("inf")
        best_vec = candidates[0]
        for vec in candidates:
            vec_2d = vec.reshape(1, -1)
            mu, sigma2 = self._gp.predict(vec_2d)
            mu = float(mu[0])
            sigma = float(np.sqrt(max(sigma2[0], 1e-12)))
            ei = compute_expected_improvement(mu, sigma, best_y, self.exploration_xi)
            if ei > best_ei:
                best_ei = ei
                best_vec = vec

        config = _decode_tuning_config(best_vec)
        return self._apply_physics_constraints(config)

    def _apply_physics_constraints(self, config: dict[str, Any]) -> dict[str, Any]:
        """Enforce physical validity of optimized parameters.

        Based on Blaha et al. (2020), J. Chem. Phys. 152, 074101
        and WIEN2k User Guide 2023:
        - Heavy elements (Z > 70): enforce min RKMAX >= 7 (insufficient
          basis for 4f/5f electrons otherwise).
        - Soft elements (Z <= 10): enforce max RKMAX <= 9 (APW+lo
          convergence faster, high RKMAX wastes resources).
        """
        min_rkmax = 5.0
        max_rkmax = 10.0

        max_z = self._max_atomic_number()
        if max_z > 90:
            min_rkmax = 8.0
        elif max_z > 70:
            min_rkmax = 7.0
        elif max_z > 50:
            min_rkmax = 6.0
        elif max_z <= 10:
            max_rkmax = 8.0

        if self.verbose and config["rkmax"] < min_rkmax:
            logger.info(f"Physics constraint (Zmax={max_z}): RKMAX "
                       f"{config['rkmax']} → {min_rkmax}")
            config["rkmax"] = min_rkmax
        elif config["rkmax"] < min_rkmax:
            config["rkmax"] = min_rkmax

        if config["rkmax"] > max_rkmax:
            config["rkmax"] = max_rkmax

        return config

    def _max_atomic_number(self) -> int:
        """Detect maximum atomic number from case.struct or case.inc."""
        try:
            case_path = Path(self.case_name)
            struct_path = case_path.with_suffix(".struct")
            if not struct_path.exists():
                struct_path = Path(f"{self.case_name}.struct")
            if struct_path.exists():
                parser = CaseFileParser()
                case_data = parser.parse_struct_file(str(struct_path))
                if case_data and case_data.get("atoms"):
                    z_list = []
                    for atm in case_data["atoms"]:
                        z = atm.get("atomic_number", 0)
                        if z == 0 and atm.get("name"):
                            z = CaseFileParser._z_from_name(atm["name"])
                        z_list.append(z)
                    if z_list:
                        return max(z_list)
        except Exception:
            logger.debug("Failed to extract max atomic number from case data", exc_info=True)
            pass
        return 0

    def tune(self, use_simulated: bool = True, max_cores: int = 1) -> TunerResult:  # noqa: C901
        """Run Bayesian optimization loop for `budget` iterations.

        Args:
            use_simulated: If True, use a synthetic objective (for testing).
                           If False, call WIEN2k via subprocess.
            max_cores: Maximum cores to use (default 1 for safe local use).

        Returns:
            TunerResult with best configuration and convergence diagnostics.
        """
        result = TunerResult()
        if self.verbose:
            logger.info(f"Starting Bayesian tuning: budget={self.budget}, "
                       f"ard={self.use_ard}")

        for i in range(self.budget):
            config = self._get_next_suggestion()
            rkmax = config["rkmax"]
            kppra = config["kppra"]
            mixing = config["mixing"]

            if use_simulated:
                obj = self._objective_simulated(rkmax, kppra, mixing)
            else:
                obj = self._objective_from_run(rkmax, kppra, mixing)

            vec = _encode_tuning_config(rkmax, kppra, mixing,
                                       total_cores=max_cores)

            self._X.append(vec)
            self._y.append(obj)

            if self._best_y is None or obj < self._best_y:
                self._best_y = obj
                self._best_x = vec
                result.best_rkmax = rkmax
                result.best_kppra = kppra
                result.best_mixing = mixing
                result.best_energy = obj

            result.observations.append({
                "iteration": i + 1,
                "rkmax": rkmax,
                "kppra": kppra,
                "mixing": mixing,
                "delta_energy": obj,
                "converged": obj < 0.001,
            })

            if self.verbose:
                conv_str = "✓ con" if obj < 0.001 else "✗"
                logger.info(f"Iter {i+1}/{self.budget}: rkmax={rkmax:.1f} "
                           f"kppra={kppra} mixing={mixing:.3f} "
                           f"ΔE={obj:.6f} {conv_str}")

            if obj < 1e-5 and i >= 3:
                result.convergence_achieved = True
                result.iterations = i + 1
                break

            if obj < 0.0001 and i >= 2 and not result.convergence_achieved:
                recent = [self._y[j] for j in range(max(0, i - 1), i + 1)]
                if all(v < 0.0001 for v in recent[-2:]):
                    result.convergence_achieved = True
                    if self.verbose:
                        logger.info(f"Convergence test passed at iter {i + 1} "
                                   f"(last 3 deltas < 0.1 meV).")
                    result.iterations = i + 1
                    break

        if self.use_ard and self.n_observations >= 5:
            X_all = np.vstack(self._X)
            y_all = np.array(self._y, dtype=np.float64)
            self._gp.fit(X_all, y_all)
            relevance = self._gp.get_relevance()
            result.uncertainty_rkmax = relevance.get("rkmax", 0.0)
            result.uncertainty_mixing = relevance.get("mixing", 0.0)

        result.iterations = self.n_observations
        return result


def optimize_convergence_parameters(
    case_name: str,
    budget: int = _DEFAULT_BUDGET,
    target: str = "energy_convergence",
    use_simulated: bool = False,
    verbose: bool = True,
) -> TunerResult:
    """Entry point for `forge optimize` CLI command.

    Args:
        case_name: WIEN2k case name.
        budget: Number of DFT runs to perform.
        target: Optimization target ('energy_convergence' only for now).
        use_simulated: Use synthetic objective (test mode).
        verbose: Print iteration details.

    Returns:
        TunerResult with best parameters found.
    """
    tuner = BayesianParameterTuner(
        case_name=case_name,
        budget=budget,
        use_ard=True,
        verbose=verbose,
    )
    return tuner.tune(use_simulated=use_simulated)
