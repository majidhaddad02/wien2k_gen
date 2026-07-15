"""Monitor data types, enums, and shared global state."""

import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    from filelock import FileLock
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False
    FileLock = None  # type: ignore


class MonitorEvent(Enum):
    """Types of events that can trigger monitoring actions or state transitions."""
    NMAT_INCREASE = "nmat_increase"
    NK_CHANGE = "nk_change"
    RKMAX_CHANGE = "rkmax_change"
    SOC_TOGGLE = "soc_toggle"
    CONVERGENCE_STALL = "convergence_stall"
    ERROR_DETECTED = "error_detected"
    CYCLE_COMPLETED = "cycle_completed"
    PREEMPTION_SIGNAL = "preemption_signal"
    CHARGE_SLOSHING = "charge_sloshing"
    BROYDEN_STUCK = "broyden_stuck"
    ANDERSON_STUCK = "anderson_stuck"
    DIIS_DIVERGENCE = "diis_divergence"


@dataclass
class ProblemVector:
    """
    Snapshot of problem parameters for change detection and state tracking.
    Designed for deterministic comparison and adaptive thresholding.
    """
    nmat: int = 0
    kpoints: int = 0
    atoms: int = 0
    rkmax: float = 7.0
    is_soc: bool = False
    is_hybrid: bool = False
    complexity: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def significant_change(
        self,
        other: 'ProblemVector',
        adaptive: bool = True
    ) -> dict[str, Any]:
        """
        Detect significant changes between two problem vectors.
        Uses adaptive thresholds: larger matrices are more sensitive to relative changes,
        while smaller systems tolerate wider fluctuations to avoid noisy rebuilds.

        Returns:
            dict with 'changes' (field diffs), 'severity' (0.0-1.0), and 'should_rebuild' (bool)
        """
        import math
        changes: dict[str, dict[str, Any]] = {}
        severity = 0.0

        # NMAT: Dominates Hamiltonian diagonalization cost & memory footprint
        if self.nmat > 0 and other.nmat > 0:
            rel_change = abs(other.nmat - self.nmat) / max(self.nmat, other.nmat)
            threshold = 0.20 / (1.0 + math.log10(max(1000, self.nmat)) / 10) if adaptive else 0.20
            if rel_change > threshold:
                changes["nmat"] = {"old": self.nmat, "new": other.nmat, "rel_change": round(rel_change, 4)}
                severity += min(0.5, rel_change * 2)  # Cap contribution at 0.5

        # K-points: Directly impacts parallelization mode & k-point distribution
        if self.kpoints != other.kpoints:
            changes["kpoints"] = {"old": self.kpoints, "new": other.kpoints}
            severity += 0.2

        # RKMAX: Quadratic scaling of basis set size & memory
        if abs(self.rkmax - other.rkmax) > 0.5:
            changes["rkmax"] = {"old": self.rkmax, "new": other.rkmax}
            severity += 0.15

        # SOC Toggle: Fundamentally alters wavefunction dimensionality & Hamiltonian structure
        if self.is_soc != other.is_soc:
            changes["is_soc"] = {"old": self.is_soc, "new": other.is_soc}
            severity += 0.3

        # Hybrid Functional: Switches to exact exchange & significantly increases cost
        if self.is_hybrid != other.is_hybrid:
            changes["is_hybrid"] = {"old": self.is_hybrid, "new": other.is_hybrid}
            severity += 0.25

        return {
            "changes": changes,
            "severity": round(min(1.0, severity), 4),
            "should_rebuild": severity > 0.3
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging or JSON storage."""
        return asdict(self)


@dataclass
class MonitorState:
    """
    Thread-safe state container for the SCF monitor.
    Uses RLock for reentrant protection during nested rebuild/callback operations.
    """
    last_problem: Optional[ProblemVector] = None
    last_dayfile_mtime: float = 0.0
    rebuild_count: int = 0
    last_rebuild_time: float = 0.0
    paused: bool = False
    preemption_handled: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def update_problem(self, prob: ProblemVector) -> None:
        """Thread-safe update of last problem vector."""
        with self.lock:
            self.last_problem = prob

    def is_rebuild_cooldown(self, min_interval: int = 300) -> bool:
        """Check if enough time has passed since last rebuild to avoid thrashing."""
        with self.lock:
            return (time.time() - self.last_rebuild_time) < min_interval

    def record_rebuild(self) -> None:
        """Record a successful rebuild with high-resolution timestamp."""
        with self.lock:
            self.rebuild_count += 1
            self.last_rebuild_time = time.time()

    def mark_preemption(self) -> None:
        """Mark that preemption signal was received and handled."""
        with self.lock:
            self.preemption_handled = True


@dataclass
class ConvergenceAnalysis:
    """
    Structured analysis of SCF convergence history from case.scf or dayfile.
    Provides convergence type classification, mixing recommendations, and raw history.
    """
    convergence_type: str = "unknown"  # "monotonic", "oscillatory", "stalled", "divergent"
    mixing_recommendation: str = ""
    estimated_cycles_to_converge: int = -1
    charge_distance_history: list[float] = field(default_factory=list)
    energy_history: list[float] = field(default_factory=list)


_monitor_state = MonitorState()
_monitor_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_file_lock: Optional[Any] = None

if _HAS_FILELOCK and FileLock is not None:
    _file_lock = FileLock(".wien2k_monitor.lock", timeout=5)


def _get_current_backend():
    try:
        from ...backend_manager import get_current_backend
        return get_current_backend()
    except ImportError:
        return None


__all__ = [
    "_HAS_FILELOCK",
    "ConvergenceAnalysis",
    "MonitorEvent",
    "MonitorState",
    "ProblemVector",
    "_file_lock",
    "_get_current_backend",
    "_monitor_state",
    "_monitor_thread",
    "_stop_event",
]
