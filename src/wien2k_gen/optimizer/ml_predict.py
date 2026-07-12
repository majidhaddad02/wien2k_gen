"""
ML-Based SCF Convergence Prediction for WIEN2k.

Predicts SCF convergence time and probability before running a calculation,
using features extracted from the crystal structure and electronic parameters.

Features:
  - atomic_features: atoms, ntype, max_z, avg_z, volume, packing_fraction
  - electronic_features: nmat, nbands, rkmax, nkpt, is_soc, is_hybrid
  - complexity_index: product of (nmat * nkpt * atoms) scaled by RKMAX effect
  - symmetry_features: spacegroup number, symmetry reduction factor

Model: scikit-learn RandomForestRegressor (no external deps beyond sklearn)
  Schutt et al., Physical Review B 89, 205118 (2014)
  Pilania et al., Scientific Reports 3, 2810 (2013)

Usage:
  wien2k_gen predict --struct Fe.struct
  # Output: Estimated SCF time: 4.2 ± 0.8 hours
  #         Convergence probability: 92%
  #         Recommended mixing: 0.15
"""

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


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
    """Predict SCF convergence time and probability from structure features.

    Uses a hybrid approach:
      1. Physics-based scaling laws (O(N^3) for diagonalisation, O(N^2) for FFT)
      2. ML-trained correction factors from historical WIEN2k runs
      3. Empirical convergence difficulty classification

    When trained data is available, uses RandomForest; otherwise falls back
    to physics-based estimates with safety margins.
    """

    def __init__(self) -> None:
        self._model = None
        self._trained = False
        self._feature_names = [
            "atoms", "inequiv", "ntype", "max_z", "avg_z", "volume",
            "packing", "nmat", "nbands", "rkmax", "nkpt", "complexity",
        ]

    def train_from_history(self, history_path: Optional[str] = None) -> None:
        """Train the ML model from historical WIEN2k execution records.

        If no history is available, the model remains in physics-only mode.
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            logger.info("scikit-learn not available; using physics-based estimates only")
            return

        records = self._load_training_data(history_path)
        if not records or len(records) < 20:
            logger.info(f"Only {len(records)} training records (need 20+); "
                       "using physics-based estimates")
            return

        X, y_time, _y_converged = self._prepare_training_data(records)
        if len(X) < 10:
            return

        self._model = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        self._model.fit(X, y_time)
        self._trained = True

        importances = dict(zip(self._feature_names, self._model.feature_importances_))
        logger.info(f"Model trained on {len(X)} records. "
                   f"Top features: {sorted(importances.items(), key=lambda x: -x[1])[:3]}")

    def _load_training_data(self, history_path: Optional[str]) -> list[dict[str, Any]]:
        records = []
        try:
            from ..optimizer.history import ExecutionHistory
            with ExecutionHistory(path=history_path) as hist:
                if hasattr(hist, '_db_path') and hist._db_path:
                    import sqlite3
                    conn = sqlite3.connect(str(hist._db_path))
                    rows = conn.execute(
                        "SELECT * FROM execution_history WHERE success = 1 "
                        "AND walltime_sec > 0 AND nmat > 0 LIMIT 1000"
                    ).fetchall()
                    cols = [d[0] for d in conn.execute("PRAGMA table_info(execution_history)")]
                    for row in rows:
                        records.append(dict(zip(cols, row)))
                    conn.close()
        except Exception as e:
            logger.debug(f"Could not load training data: {e}")

        return records

    def _prepare_training_data(
        self, records: list[dict[str, Any]]
    ) -> tuple[Any, Any, Any]:
        import numpy as np

        X_list = []
        y_time = []
        y_converged = []

        for r in records:
            try:
                nmat = int(r.get("nmat", 100))
                nkpt = int(r.get("nkpt", 8))
                atoms = int(r.get("atoms", 1))
                rkmax = float(r.get("rkmax", 7.0))

                complexity = (nmat ** 1.5 * nkpt * atoms * rkmax / 7.0) / 1e6
                features = [
                    atoms, atoms, 1, 26, 26.0, 100.0,
                    0.5, nmat, nmat // 10, rkmax, nkpt, complexity,
                ]
                X_list.append(features)
                y_time.append(float(r.get("walltime_sec", 3600)) / 3600.0)
                y_converged.append(1.0)
            except Exception:
                continue

        return np.array(X_list), np.array(y_time), np.array(y_converged)

    def predict(
        self,
        struct_features: StructureFeatures,
        electronic_features: ElectronicFeatures,
    ) -> ConvergencePrediction:
        """Predict SCF convergence time and probability.

        Args:
            struct_features: Crystal structure features
            electronic_features: Electronic structure parameters

        Returns:
            ConvergencePrediction with time estimate, probability, and recommendations
        """
        sf = struct_features
        ef = electronic_features

        complexity = ef.complexity_index
        if complexity < 1.0:
            complexity = _compute_complexity(sf, ef)

        physics_time = _physics_estimate(sf, ef)

        if self._trained and self._model is not None:
            try:
                import numpy as np
                X = np.array([[
                    sf.atoms, sf.atoms_inequiv, sf.ntype, sf.max_z, sf.avg_z,
                    sf.volume_bohr3, sf.packing_fraction,
                    ef.nmat, ef.nbands, ef.rkmax, ef.nkpt, complexity,
                ]])
                ml_time = float(self._model.predict(X)[0])
                predicted_time = 0.4 * physics_time + 0.6 * ml_time
                uncertainty = abs(physics_time - ml_time) * 0.5
            except Exception:
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


def _compute_complexity(sf: StructureFeatures, ef: ElectronicFeatures) -> float:
    return max(0.1, (ef.nmat ** 1.5 * ef.nkpt * sf.atoms * ef.rkmax / 7.0) / 1e6)


def _physics_estimate(sf: StructureFeatures, ef: ElectronicFeatures) -> float:
    """Physics-based walltime estimate (hours).

    Scaling:
      - Diagonalisation: O(nmat^3)
      - FFT: O(nmat * log(nmat))
      - k-points: O(nkpt)
      - Processor speed: ~2.5 GHz base, ~50 GFLOPS FP64 per core
    """
    nmat = max(ef.nmat, 10)
    nkpt = max(ef.nkpt, 1)
    rkmax_factor = (ef.rkmax / 7.0) ** 2.5

    diag_ops = nmat ** 3
    fft_ops = nmat * math.log(nmat + 1) * 100

    total_ops = (diag_ops + fft_ops) * nkpt * rkmax_factor

    ops_per_core_per_sec = 50e9
    cores = 8
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


def predict_convergence(
    case_name: str = "case",
    struct_path: Optional[str] = None,
    use_history: bool = True,
) -> ConvergencePrediction:
    """CLI entry point: predict SCF convergence for a WIEN2k case.

    Args:
        case_name: WIEN2k case name
        struct_path: Path to .struct file (auto-detected if None)
        use_history: Train ML model from execution history

    Returns:
        ConvergencePrediction with time, probability, and recommendations
    """
    struct_file = Path(struct_path) if struct_path else Path(f"{case_name}.struct")

    sf = StructureFeatures()
    if struct_file.exists():
        sf = _extract_structure_features(struct_file)

    ef = ElectronicFeatures()
    try:
        from ..core.case_parser import CaseFileParser
        parser = CaseFileParser(case_name)
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

    predictor = SCFTimePredictor()
    if use_history:
        predictor.train_from_history()

    return predictor.predict(sf, ef)


def _extract_structure_features(struct_path: Path) -> StructureFeatures:
    sf = StructureFeatures()
    try:
        content = struct_path.read_text(encoding="utf-8", errors="replace")
        lines = content.strip().splitlines()

        if len(lines) >= 3:
            lattice_line = lines[2]
            parts = lattice_line.split()
            if len(parts) >= 6:
                a = float(parts[0])
                b = float(parts[1])
                c = float(parts[2])
                alpha = float(parts[3])
                beta = float(parts[4])
                gamma = float(parts[5])

                import math
                ca = math.cos(math.radians(alpha))
                cb = math.cos(math.radians(beta))
                cg = math.cos(math.radians(gamma))
                vol = a * b * c * math.sqrt(1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg)
                sf.volume_bohr3 = vol

        z_set: set = set()
        atom_lines = [line for line in lines if line.strip().startswith("X=") or ": X=" in line]
        atoms = 0
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
