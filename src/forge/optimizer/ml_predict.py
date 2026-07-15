"""
ML-Based SCF Convergence Prediction for WIEN2k.

Predicts SCF convergence time and probability using features from:
  - Crystal structure (atoms, volume, spacegroup, max_z)
  - Electronic parameters (nmat, nbands, rkmax, nkpt)
  - Hardware context (CPU, memory bandwidth, NUMA, interconnect)
  - I/O characteristics (scratch filesystem type)

Model: scikit-learn RandomForestRegressor with cross-validation and persistence.
Fallback: physics-based scaling laws (O(N^3) diagonalization + O(N log N) FFT).

Usage:
  forge predict --struct Fe.struct
  # Output: Estimated SCF time: 4.2 \u00b1 0.8 hours
  #         Convergence probability: 92%
"""

import contextlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)

_MODEL_CACHE_DIR = Path.home() / ".forge" / "models"
_MODEL_CACHE_PATH = _MODEL_CACHE_DIR / "scf_time_rf.pkl"


@dataclass
class StructureFeatures:
    """Features extracted from crystal structure for ML prediction."""
    atoms: int = 1
    atoms_inequiv: int = 1
    ntype: int = 1
    max_z: int = 1
    avg_z: float = 1.0
    volume_bohr3: float = 100.0
    packing_fraction: float = 0.5
    spacegroup_number: int = 1


@dataclass
class ElectronicFeatures:
    """Features from WIEN2k electronic structure parameters."""
    nmat: int = 100
    nbands: int = 10
    rkmax: float = 7.0
    nkpt: int = 8
    is_soc: bool = False
    is_hybrid: bool = False
    complexity_index: float = 1.0


@dataclass
class HardwareContext:
    """Hardware features captured at job runtime."""
    cpu_arch: str = ""
    cpu_generation: str = ""
    peak_gflops: float = 0.0
    mem_bandwidth_gbs: float = 0.0
    numa_nodes: int = 1
    interconnect_type: str = ""
    scratch_fs: str = ""


@dataclass
class ConvergencePrediction:
    """Predicted SCF convergence characteristics."""
    estimated_time_hours: float = 0.0
    time_uncertainty_hours: float = 0.0
    convergence_probability: float = 1.0
    estimated_cycles: int = 20
    recommended_mixing: float = 0.15
    convergence_difficulty: str = "easy"
    feature_importance: dict[str, float] = None  # type: ignore

    def __post_init__(self):
        if self.feature_importance is None:
            self.feature_importance = {}


class SCFTimePredictor:
    """Predict SCF convergence time with hybrid ML + physics approach.

    Features (20-D):
      Structural:  atoms, volume_bohr3, packing_fraction, spacegroup,
                    max_z, avg_z, ntype
      Electronic:  nmat, nbands, rkmax, nkpt, is_soc, is_hybrid
      Complexity:  complexity_index, log(nmat * nkpt)
      Hardware:    peak_gflops, mem_bandwidth_gbs, numa_nodes,
                    interconnect_speed_gbps
      I/O:         scratch_is_local (bool)

    When trained data >= 20 records: uses RandomForest with CV-tuned params.
    Otherwise: falls back to physics-based O(N^3) estimate with safety margin.
    """

    def __init__(self) -> None:
        self._model = None
        self._trained = False
        self._feature_names = [
            "atoms", "nmat", "nbands", "rkmax", "nkpt",
            "is_soc", "is_hybrid", "spacegroup", "max_z", "avg_z",
            "volume_bohr3", "packing_fraction", "complexity",
            "log_nmat_nkpt", "peak_gflops", "mem_bandwidth_gbs",
            "numa_nodes", "interconnect_gbps", "scratch_is_local",
        ]
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    def train_from_history(self, history_path: Optional[str] = None) -> None:
        """Train ML model from ExecutionHistory records with CV hyperparameter tuning."""
        records = self._load_training_data(history_path)
        if not records or len(records) < 20:
            logger.info(f"Only {len(records)} training records (need 20+); "
                       "using physics-based estimates")
            return

        X, y_time = self._prepare_training_data(records)
        if len(X) < 10:
            return

        X = np.array(X, dtype=np.float64)
        y_time = np.array(y_time, dtype=np.float64)

        self._scaler_mean = np.mean(X, axis=0)
        self._scaler_std = np.std(X, axis=0)
        self._scaler_std = np.where(self._scaler_std < 1e-8, 1.0, self._scaler_std)
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV, cross_val_score

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [6, 8, 12, None],
                "min_samples_leaf": [1, 3, 5],
            }

            base = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = GridSearchCV(
                base, param_grid, cv=min(5, max(2, len(X) // 5)),
                scoring="neg_mean_absolute_error", n_jobs=-1,
            )
            search.fit(X_scaled, y_time)
            self._model = search.best_estimator_

            cv_scores = cross_val_score(
                self._model, X_scaled, y_time,
                cv=min(5, max(2, len(X) // 5)),
                scoring="neg_mean_absolute_error",
            )
            cv_mae = -np.mean(cv_scores)

            self._trained = True
            importances = dict(zip(self._feature_names, self._model.feature_importances_))
            sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
            logger.info(
                f"RF trained: {len(X)} records, CV-MAE={cv_mae:.3f}h, "
                f"best_params={search.best_params_}, "
                f"top features: {sorted_imp[:5]}"
            )

            _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(_MODEL_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "feature_names": self._feature_names,
                    "scaler_mean": self._scaler_mean,
                    "scaler_std": self._scaler_std,
                }, f)

        except ImportError:
            logger.info("scikit-learn not available; using physics-based estimates only")

    def load_cached_model(self) -> bool:
        """Load a previously saved trained model from disk."""
        if not _MODEL_CACHE_PATH.exists():
            return False
        try:
            with open(_MODEL_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._feature_names = data.get("feature_names", self._feature_names)
            self._scaler_mean = data.get("scaler_mean")
            self._scaler_std = data.get("scaler_std")
            self._trained = True
            logger.info(f"Loaded cached RF model from {_MODEL_CACHE_PATH}")
            return True
        except Exception as e:
            logger.debug(f"Failed to load cached model: {e}")
            return False

    def _load_training_data(self, history_path: Optional[str]) -> list[dict[str, Any]]:
        records = []
        try:
            from ..optimizer.history import ExecutionHistory
            with ExecutionHistory(path=history_path) as hist:
                if hasattr(hist, '_db_path') and hist._db_path:
                    import sqlite3
                    conn = sqlite3.connect(str(hist._db_path))
                    try:
                        rows = conn.execute(
                            "SELECT * FROM execution_history WHERE success = 1 "
                            "AND walltime_sec > 0 AND nmat > 0 LIMIT 1000"
                        ).fetchall()
                        cols = [d[0] for d in conn.execute("PRAGMA table_info(execution_history)")]
                        for row in rows:
                            records.append(dict(zip(cols, row)))
                    finally:
                        conn.close()
        except Exception as e:
            logger.debug(f"Could not load training data: {e}")
        return records

    def _prepare_training_data(self, records: list[dict[str, Any]]) -> tuple[Any, Any]:
        X_list = []
        y_time = []
        for r in records:
            try:
                features = _encode_features_from_record(r)
                X_list.append(features)
                y_time.append(float(r.get("walltime_sec", 3600)) / 3600.0)
            except Exception:
                continue
        return np.array(X_list, dtype=np.float64), np.array(y_time, dtype=np.float64)

    def predict(
        self,
        struct_features: StructureFeatures,
        electronic_features: ElectronicFeatures,
        hardware: Optional[HardwareContext] = None,
    ) -> ConvergencePrediction:
        """Predict SCF convergence time and probability."""
        sf = struct_features
        ef = electronic_features
        hw = hardware or HardwareContext()

        complexity = ef.complexity_index
        if complexity < 1.0:
            complexity = _compute_complexity(sf, ef)

        physics_time = _physics_estimate(sf, ef, hw)

        X = np.array([_encode_features(sf, ef, hw, complexity)], dtype=np.float64)

        if self._trained and self._model is not None:
            try:
                if self._scaler_mean is not None and self._scaler_std is not None:
                    X_scaled = (X - self._scaler_mean) / self._scaler_std
                else:
                    X_scaled = X
                ml_time = float(self._model.predict(X_scaled)[0])
                predicted_time = 0.3 * physics_time + 0.7 * ml_time
                uncertainty = abs(physics_time - ml_time) * 0.4
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")
                predicted_time = physics_time
                uncertainty = physics_time * 0.3
        else:
            predicted_time = physics_time
            uncertainty = physics_time * 0.3

        conv_probability = _estimate_convergence_probability(sf, ef)
        mixing = _recommend_mixing(sf, ef)
        difficulty = _classify_difficulty(sf, ef)

        feature_importance = {}
        if self._trained:
            with contextlib.suppress(Exception):
                feature_importance = dict(zip(
                    self._feature_names,
                    self._model.feature_importances_,
                ))

        return ConvergencePrediction(
            estimated_time_hours=round(predicted_time, 2),
            time_uncertainty_hours=round(uncertainty, 2),
            convergence_probability=round(conv_probability, 3),
            estimated_cycles=_estimate_cycles(sf, ef),
            recommended_mixing=mixing,
            convergence_difficulty=difficulty,
            feature_importance=feature_importance,
        )


def _encode_features(
    sf: StructureFeatures,
    ef: ElectronicFeatures,
    hw: HardwareContext,
    complexity: float,
) -> list[float]:
    """Encode all features into a 19-D vector (matching _feature_names)."""
    interconnect_gbps = _interconnect_speed_lookup(hw.interconnect_type)
    scratch_local = 1.0 if hw.scratch_fs not in ("nfs", "unknown", "") else 0.0
    peak_gflops = hw.peak_gflops if hw.peak_gflops > 0 else 50.0
    mem_bw = hw.mem_bandwidth_gbs if hw.mem_bandwidth_gbs > 0 else 50.0
    numa = hw.numa_nodes if hw.numa_nodes > 0 else 1

    return [
        float(sf.atoms),
        float(ef.nmat),
        float(ef.nbands),
        float(ef.rkmax),
        float(ef.nkpt),
        float(ef.is_soc),
        float(ef.is_hybrid),
        float(sf.spacegroup_number),
        float(sf.max_z),
        sf.avg_z,
        sf.volume_bohr3,
        sf.packing_fraction,
        complexity,
        math.log(max(float(ef.nmat) * float(ef.nkpt), 1.0)),
        peak_gflops,
        mem_bw,
        float(numa),
        interconnect_gbps,
        scratch_local,
    ]


def _encode_features_from_record(r: dict[str, Any]) -> list[float]:
    """Encode features from a raw SQLite record (may have extended columns)."""
    nmat = int(r.get("nmat", 100))
    nkpt = int(r.get("nkpt", 8))
    atoms = int(r.get("atoms", 1))
    rkmax = float(r.get("rkmax", 7.0))
    nbands = int(r.get("nbands", 0)) or (nmat // 10) or 10
    complexity = max(0.1, (nmat ** 1.5 * nkpt * atoms * rkmax / 7.0) / 1e6)
    is_soc = int(r.get("is_soc", 0))
    is_hybrid = int(r.get("is_hybrid", 0))
    spacegroup = int(r.get("spacegroup", 1))
    max_z = int(r.get("max_z", 26))
    avg_z = float(r.get("avg_z", 26.0))
    volume_bohr3 = float(r.get("volume_bohr3", 100.0))
    packing = min(1.0, atoms * 15.0 / volume_bohr3) if volume_bohr3 > 0 else 0.5

    peak_gflops = float(r.get("peak_gflops", 0.0)) or 50.0
    mem_bw = float(r.get("mem_bandwidth_gbs", 0.0)) or 50.0
    numa = int(r.get("numa_nodes", 1)) or 1
    interconnect = str(r.get("interconnect_type", ""))
    scratch = str(r.get("scratch_fs", ""))

    interconnect_gbps = _interconnect_speed_lookup(interconnect)
    scratch_local = 1.0 if scratch not in ("nfs", "unknown", "") else 0.0

    return [
        float(atoms), float(nmat), float(nbands), rkmax, float(nkpt),
        float(is_soc), float(is_hybrid), float(spacegroup), float(max_z), avg_z,
        volume_bohr3, packing, complexity,
        math.log(max(nmat * nkpt, 1.0)),
        peak_gflops, mem_bw, float(numa), interconnect_gbps, scratch_local,
    ]


def _interconnect_speed_lookup(interconnect_type: str) -> float:
    mapping = {
        "infiniband": 100.0,
        "omni_path": 100.0,
        "ethernet": 25.0,
        "tcp": 10.0,
    }
    return mapping.get(interconnect_type.lower(), 10.0)


# ---------------------------------------------------------------------------
# Physics-based fallback
# ---------------------------------------------------------------------------

def _compute_complexity(sf: StructureFeatures, ef: ElectronicFeatures) -> float:
    return max(0.1, (ef.nmat ** 1.5 * ef.nkpt * sf.atoms * ef.rkmax / 7.0) / 1e6)


def _physics_estimate(
    sf: StructureFeatures,
    ef: ElectronicFeatures,
    hw: HardwareContext,
) -> float:
    """Physics-based walltime estimate (hours).

    Scaling:
      Diagonalization: O(nmat^3)
      FFT: O(nmat * log(nmat))
      k-points: O(nkpt)
      Hardware: uses peak GFLOPS and core count for throughput estimate
    """
    nmat = max(ef.nmat, 10)
    nkpt = max(ef.nkpt, 1)
    rkmax_factor = (ef.rkmax / 7.0) ** 2.5

    diag_ops = nmat ** 3
    fft_ops = nmat * math.log(nmat + 1) * 100
    total_ops = (diag_ops + fft_ops) * nkpt * rkmax_factor

    if hw.peak_gflops > 0:
        ops_per_core_per_sec = hw.peak_gflops * 1e9 * 0.15
    else:
        ops_per_core_per_sec = 50e9

    cores = max(1, hw.numa_nodes * 4) if hw.numa_nodes > 1 else 8
    cycles = 30
    seconds = total_ops * cycles / (ops_per_core_per_sec * cores)
    hours = seconds / 3600.0
    if ef.is_soc:
        hours *= 4.0
    if ef.is_hybrid:
        hours *= 6.0
    return max(0.01, hours)


def _estimate_convergence_probability(
    sf: StructureFeatures, ef: ElectronicFeatures,
) -> float:
    prob = 1.0
    if ef.is_soc:
        prob *= 0.85
    if ef.is_hybrid:
        prob *= 0.75
    if ef.rkmax > 8.5:
        prob *= 0.92
    if sf.atoms > 50:
        prob *= 0.90
    if sf.spacegroup_number == 1:
        prob *= 0.95
    rkmax_quality = min(1.0, ef.rkmax / 8.0)
    prob *= (0.7 + 0.3 * rkmax_quality)
    return min(1.0, max(0.1, prob))


def _estimate_cycles(sf: StructureFeatures, ef: ElectronicFeatures) -> int:
    base = 20
    if ef.is_soc:
        base += 10
    if ef.is_hybrid:
        base += 15
    if sf.atoms > 100:
        base += 10
    if sf.spacegroup_number == 1:
        base += 5
    if ef.rkmax > 8.0:
        base -= 3
    return max(5, base)


def _recommend_mixing(sf: StructureFeatures, ef: ElectronicFeatures) -> float:
    mixing = 0.20
    if ef.is_hybrid:
        mixing = 0.10
    elif ef.is_soc:
        mixing = 0.12
    elif sf.atoms > 50:
        mixing = 0.15
    elif sf.max_z > 70:
        mixing = 0.12
    return round(mixing, 3)


def _classify_difficulty(sf: StructureFeatures, ef: ElectronicFeatures) -> str:
    score = 0
    if ef.is_soc:
        score += 2
    if ef.is_hybrid:
        score += 3
    if sf.atoms > 100:
        score += 2
    if ef.rkmax > 9.0:
        score += 1
    if sf.max_z > 70:
        score += 1
    if sf.spacegroup_number == 1:
        score += 1
    if score <= 2:
        return "easy"
    elif score <= 5:
        return "moderate"
    elif score <= 8:
        return "hard"
    else:
        return "very_hard"


# ---------------------------------------------------------------------------
# Hardware context capture
# ---------------------------------------------------------------------------

def capture_hardware_context() -> HardwareContext:
    """Capture current hardware state for ML feature encoding.

    Tries to load hardware info; graceful fallback to defaults if unavailable.
    """
    try:
        from ..core.hardware import (
            calculate_peak_fp64_gflops,
            get_cpu_architecture,
            get_cpu_generation,
            get_interconnect_info,
            get_memory_bandwidth_gb_s,
            get_numa_node_count,
            get_scratch_filesystem_type,
        )
        return HardwareContext(
            cpu_arch=get_cpu_architecture(),
            cpu_generation=get_cpu_generation(),
            peak_gflops=calculate_peak_fp64_gflops(),
            mem_bandwidth_gbs=get_memory_bandwidth_gb_s(),
            numa_nodes=get_numa_node_count(),
            interconnect_type=get_interconnect_info().get("type", ""),
            scratch_fs=get_scratch_filesystem_type(),
        )
    except Exception as e:
        logger.debug(f"Hardware context capture skipped: {e}")
        return HardwareContext()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def predict_convergence(
    case_name: str = "case",
    struct_path: Optional[str] = None,
    use_history: bool = True,
) -> ConvergencePrediction:
    """CLI entry point: predict SCF convergence for a WIEN2k case."""
    struct_file = Path(struct_path) if struct_path else Path(f"{case_name}.struct")
    sf = StructureFeatures()
    if struct_file.exists():
        sf = _extract_structure_features(struct_file)

    ef = ElectronicFeatures()
    try:
        from ..core.case_parser import CaseFileParser
        parser_arg = struct_file if struct_file.exists() else Path(case_name)
        parser = CaseFileParser(parser_arg)
        data = parser.parse_all()
        ef = ElectronicFeatures(
            nmat=data.nmat or 100,
            nbands=data.nbands or 10,
            rkmax=data.rkmax or 7.0,
            nkpt=data.kpoints or 8,
            is_soc=data.is_soc,
            is_hybrid=data.is_hybrid,
            complexity_index=_compute_complexity(sf, ElectronicFeatures(
                nmat=data.nmat or 100,
                nbands=data.nbands or 10,
                rkmax=data.rkmax or 7.0,
                nkpt=data.kpoints or 8,
                is_soc=data.is_soc,
                is_hybrid=data.is_hybrid,
            )),
        )
    except Exception as e:
        logger.debug(f"Case parsing skipped: {e}")

    hw = capture_hardware_context()

    predictor = SCFTimePredictor()
    if use_history and not predictor.load_cached_model():
        predictor.train_from_history()

    return predictor.predict(sf, ef, hw)


def _extract_structure_features(struct_path: Path) -> StructureFeatures:
    sf = StructureFeatures()
    try:
        content = struct_path.read_text(encoding="utf-8", errors="replace")
        lines = content.strip().splitlines()
        if len(lines) >= 3:
            lattice_line = lines[2]
            parts = lattice_line.split()
            if len(parts) >= 6:
                a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
                alpha, beta, gamma = float(parts[3]), float(parts[4]), float(parts[5])
                import math
                ca, cb, cg = math.cos(math.radians(alpha)), math.cos(math.radians(beta)), math.cos(math.radians(gamma))
                vol = a * b * c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
                sf.volume_bohr3 = vol
        z_set: set = set()
        atoms = 0
        atom_lines = [line for line in lines if line.strip().startswith("X=") or ": X=" in line]
        for line in atom_lines:
            import re
            z_match = re.search(r"Z\s*[:=]\s*(\d+\.?\d*)", line)
            if z_match:
                z = int(float(z_match.group(1)))
                z_set.add(z)
                atoms += 1
            else:
                atoms += 1
        if atoms == 0:
            atoms = len([line for line in lines if "ATOM" in line.upper()])
            atoms = max(atoms, 1) * 5
        sf.atoms = atoms
        sf.atoms_inequiv = atoms
        sf.ntype = max(len(z_set), 1)
        sf.max_z = max(z_set) if z_set else 26
        sf.avg_z = sum(z_set) / len(z_set) if z_set else 26.0
        if sf.volume_bohr3 > 0:
            atom_volume = 15.0
            sf.packing_fraction = min(1.0, (atoms * atom_volume) / sf.volume_bohr3)
    except Exception:
        pass
    return sf
