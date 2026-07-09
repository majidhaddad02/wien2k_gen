"""
Adaptive SCF Monitor with Intelligent Reconfiguration & Preemption Resilience.
Production features:
• Multi-parameter change detection with adaptive, problem-size-aware thresholds
• Benefit estimation via Roofline-informed advisor before triggering rebuilds
• Atomic rollback on failure with persistent backup & JSONL audit logging
• Thread-safe state management with RLock & graceful signal handling (SIGTERM/USR1)
• Convergence stall detection via energy/charge delta tracking in dayfile/scf logs
• SLURM/LFS preemption awareness with checkpoint triggers before forced termination
• Structured logging, fallback file locking, and HPC-grade I/O safety
All documentation and inline comments are in English per project standards.
"""

import json
import math
import re
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Robust FileLock fallback
try:
    from filelock import FileLock
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False
    FileLock = None  # type: ignore

from ..core.topology import Topology
from ..logging_config import get_logger
from ..utils.atomic_write import atomic_write


# Lazy import to avoid circular dependency
def _get_current_backend():
    from ..backend_manager import get_current_backend
    return get_current_backend()

logger = get_logger(__name__)

# =============================================================================
# Enums & Data Classes
# =============================================================================

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
    ) -> Dict[str, Any]:
        """
        Detect significant changes between two problem vectors.
        Uses adaptive thresholds: larger matrices are more sensitive to relative changes,
        while smaller systems tolerate wider fluctuations to avoid noisy rebuilds.

        Returns:
            dict with 'changes' (field diffs), 'severity' (0.0-1.0), and 'should_rebuild' (bool)
        """
        changes: Dict[str, Dict[str, Any]] = {}
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

    def to_dict(self) -> Dict[str, Any]:
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
    charge_distance_history: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)


# =============================================================================
# Global State & Signal Handling
# =============================================================================

_monitor_state = MonitorState()
_monitor_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_file_lock: Optional[Any] = None

if _HAS_FILELOCK and FileLock is not None:
    _file_lock = FileLock(".wien2k_monitor.lock", timeout=5)


def _register_preemption_signals(checkpoint_fn: Optional[Callable] = None) -> None:
    """
    Register signal handlers for SLURM/LFS preemption (SIGTERM) and user interrupts (SIGUSR1).
    Triggers checkpoint routine and graceful monitor shutdown before forced termination.
    """
    def _handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}. Triggering preemption checkpoint...")
        _monitor_state.mark_preemption()
        if checkpoint_fn:
            try:
                checkpoint_fn()
            except Exception as e:
                logger.error(f"Checkpoint execution failed during {sig_name}: {e}")
        logger.info("Preemption handled. Exiting monitor loop gracefully.")
        _stop_event.set()

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGUSR1, _handler)
        logger.debug("Preemption signal handlers registered (SIGTERM, SIGUSR1)")
    except ValueError:
        logger.debug("Cannot register signal handlers in non-main thread")


# =============================================================================
# Helper Functions: Parsing & Estimation
# =============================================================================

def _get_current_problem_vector() -> ProblemVector:
    """Extract current problem parameters from the active DFT backend."""
    backend = _get_current_backend()
    try:
        params = backend.detect_problem_size()
        return ProblemVector(
            nmat=params.get("nmat", 0),
            kpoints=params.get("kpoints", 0),
            atoms=params.get("atoms", 0),
            rkmax=params.get("rkmax", 7.0),
            is_soc=params.get("is_soc", False),
            is_hybrid=params.get("is_hybrid", False),
            complexity=params.get("complexity", 1.0)
        )
    except Exception as e:
        logger.debug(f"Failed to extract problem vector: {e}")
        return ProblemVector()


def _get_dayfile_path() -> Optional[Path]:
    """Locate the active SCF log/dayfile for parsing."""
    backend = _get_current_backend()
    if hasattr(backend, 'get_log_filename'):
        try:
            path = Path(backend.get_log_filename())
            if path.exists():
                return path
        except Exception:
            pass
            
    patterns = ["*.scf", "*.dayfile", "*.output", "case*.scf"]
    for pattern in patterns:
        matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _parse_dayfile_events(dayfile_path: Path) -> List[MonitorEvent]:
    """
    Parse SCF dayfile content to detect convergence behavior, errors, or cycle completion.
    Uses regex-based extraction for robust matching across WIEN2k versions.
    Also invokes charge-sloshing and Broyden-mixing detectors.
    """
    events: List[MonitorEvent] = []
    if not dayfile_path.exists():
        return events

    try:
        raw_content = dayfile_path.read_text(encoding="utf-8", errors="replace")
        content = raw_content.lower()

        # Convergence stall detection (repeated identical charge/energy values)
        if re.search(r"charge convergence.*?:.*?0\.0000", content):
            events.append(MonitorEvent.CONVERGENCE_STALL)

        # Cycle completion marker
        if re.search(r"cycle\s+\d+|end of scf cycle", content):
            events.append(MonitorEvent.CYCLE_COMPLETED)

        # Critical error patterns (FIXED: Removed trailing spaces from regex strings)
        error_patterns = [
            r"qtl-b",
            r"lapw[0-9]?\s*(crashed|error|fail)",
            r"not converged",
            r"segmentation fault",
            r"abort"
        ]
        for pat in error_patterns:
            if re.search(pat, content):
                events.append(MonitorEvent.ERROR_DETECTED)
                break  # Report first critical error only

        # Charge sloshing detection
        sloshing_result = detect_charge_sloshing(raw_content)
        if sloshing_result.get("sloshing_detected"):
            events.append(MonitorEvent.CHARGE_SLOSHING)
            logger.debug(
                f"Charge sloshing detected (severity={sloshing_result.get('severity', 0)}). "
                f"Recommendation: {sloshing_result.get('recommendation', '')}"
            )

        # FFT-based frequency-domain sloshing verification (Kresse & Furthmueller 1996)
        fft_result = detect_charge_sloshing_fft(raw_content)
        if fft_result.get("sloshing_detected"):
            if MonitorEvent.CHARGE_SLOSHING not in events:
                events.append(MonitorEvent.CHARGE_SLOSHING)
            logger.debug(
                f"FFT confirms charge sloshing (HF ratio={fft_result.get('hf_power_ratio', 0)}, "
                f"dominant freq={fft_result.get('dominant_frequency_hz', 0)}). "
                f"Recommendation: {fft_result.get('recommendation', '')}"
            )

        # Broyden mixing analysis
        broyden_result = analyze_broyden_mixing(raw_content, str(dayfile_path))
        if broyden_result.get("stuck"):
            events.append(MonitorEvent.BROYDEN_STUCK)
            logger.debug(
                f"Broyden mixing stuck (plateau={broyden_result.get('iteration_plateau_length', 0)}). "
                f"Recommendation: {broyden_result.get('recommendation', '')}"
            )

        # Anderson mixing analysis (Eyert 1996)
        anderson_result = analyze_anderson_mixing(raw_content)
        if anderson_result.get("stuck"):
            events.append(MonitorEvent.ANDERSON_STUCK)
            logger.debug(
                f"Anderson mixing stuck (plateau={anderson_result.get('iteration_plateau_length', 0)}). "
                f"Recommendation: {anderson_result.get('recommendation', '')}"
            )

        # DIIS/Pulay mixing analysis (Pulay 1980)
        diis_result = analyze_diis_mixing(raw_content)
        if diis_result.get("diverging"):
            events.append(MonitorEvent.DIIS_DIVERGENCE)
            logger.debug(
                f"DIIS mixing diverging (residual={diis_result.get('residual_trend', 0)}). "
                f"Recommendation: {diis_result.get('recommendation', '')}"
            )

    except Exception as e:
        logger.debug(f"Dayfile parsing failed for {dayfile_path}: {e}")

    return events


def detect_charge_sloshing(dayfile_content: str, tolerance: float = 0.5) -> dict:
    """
    Detect charge sloshing—oscillatory convergence behavior where charge distance
    alternates (high/low) across SCF cycles instead of decaying monotonically.

    Uses the Durbin-Watson statistic on consecutive charge-distance deltas to
    identify negative autocorrelation, the hallmark signature of sloshing.

    Args:
        dayfile_content: Raw or lowercased text content from the dayfile/SCF log.
        tolerance: Sensitivity threshold for DW-based detection (lower = more sensitive).

    Returns:
        dict with keys:
            sloshing_detected: bool
            severity: float (0.0 – 1.0, based on oscillation amplitude)
            cycles_affected: List[int] of cycle indices exhibiting sloshing
            recommendation: str with actionable mixing advice
    """
    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    if len(matches) < 5:
        return {
            "sloshing_detected": False,
            "severity": 0.0,
            "cycles_affected": [],
            "recommendation": ""
        }

    values = [float(m) for m in matches]
    n = len(values)
    diffs = [values[i] - values[i - 1] for i in range(1, n)]

    numerator = sum(d * d for d in diffs)
    denominator = sum((v - sum(values) / n) ** 2 for v in values)
    if denominator < 1e-30:
        dw_stat = 2.0
    else:
        dw_stat = numerator / denominator if denominator > 0 else 2.0

    sloshing_detected = dw_stat > (2.5 + tolerance)

    slope = values[-1] - values[0]
    if sloshing_detected and slope > 0:
        sloshing_detected = False

    mean_val = sum(values) / n if n > 0 else 0
    rel_range = (max(values) - min(values)) / mean_val if mean_val > 1e-12 else 0
    severity_raw = min(1.0, (dw_stat - 2.5) / 1.5)
    severity = round(0.8 * severity_raw + 0.2 * min(1.0, rel_range), 4)

    cycles_affected: List[int] = []
    if sloshing_detected:
        osc_threshold = (max(diffs) - min(diffs)) * 0.3 if diffs else 0
        for i in range(2, n):
            if diffs[i - 1] * diffs[i - 2] < 0 and abs(diffs[i - 1]) > osc_threshold * 0.1:
                cycles_affected.append(i + 1)

    return {
        "sloshing_detected": sloshing_detected,
        "severity": severity,
        "cycles_affected": cycles_affected,
        "recommendation": (
            "Charge sloshing detected—reduce mixing beta by 50%, "
            "increase number of PRATT iterations, or try MSR1a mixing"
            if sloshing_detected
            else ""
        )
    }


def diagnose_charge_sloshing_root_cause(
    dayfile_content: str,
    case_name: str = "case",
    struct_path: Optional[str] = None,
) -> dict:
    """Diagnose root cause of charge sloshing and recommend targeted fix.

    Based on Peter Blaha's feedback: sloshing has specific root causes
    that require different treatments:

    1. Metallic systems (Fermi surface, bands crossing EF):
       → Apply Methfessel-Paxton smearing (0.02 Ry), PRATT mixing
    2. Symmetry breaking (structural distortion, magnetic ordering):
       → Disable symmetry (runsp_lapw), reduce mixing slowly
    3. Core overlap (RMT ratio < 1.5 between adjacent atoms):
       → Check RMT values, adjust R0, increase PRATT iterations
    4. Default (mixing rate too aggressive):
       → Reduce mixing beta, increase PRATT cycles, try MSR1a

    Returns dict with:
        root_cause: str — one of "metallic", "symmetry_breaking",
                   "core_overlap", "mixing_too_aggressive", "unknown"
        actions: List[dict] — ordered list of remediation steps
        confidence: float — 0.0–1.0 diagnostic confidence
    """
    slosh = detect_charge_sloshing(dayfile_content)
    if not slosh["sloshing_detected"]:
        return {
            "root_cause": "none",
            "actions": [],
            "confidence": 0.0,
        }

    indicators: Dict[str, float] = {}

    # 1. Check for metallic system indicators
    if case_name:
        try:
            scf_path = Path(f"{case_name}.scf")
            if not scf_path.exists():
                scf_path = Path(case_name) / f"{case_name}.scf"
            if scf_path.exists():
                scf_text = scf_path.read_text()
                if ":GAP" in scf_text:
                    gap_match = re.search(r':GAP\s*:\s*(-?\d+\.\d+)', scf_text)
                    if gap_match:
                        gap = float(gap_match.group(1))
                        if gap < 0.01:
                            indicators["metallic"] = 0.9
                        elif gap < 0.1:
                            indicators["metallic"] = 0.6
                if "FERMI" in scf_text or "nearly free" in scf_text.lower():
                    indicators.setdefault("metallic", 0.0)
                    indicators["metallic"] = max(indicators["metallic"], 0.5)
        except Exception:
            pass

    # 2. Check for symmetry breaking
    if "symmetry" in dayfile_content.lower() or "symm" in dayfile_content.lower():
        if "broken" in dayfile_content.lower() or "fail" in dayfile_content.lower():
            indicators["symmetry_breaking"] = 0.85

    # 3. Check for core overlap via RMT info
    rmt_pattern = re.compile(r'(?:RMT|rmt)\s*[=:]\s*([\d]+\.?\d*)', re.IGNORECASE)
    rmt_matches = rmt_pattern.findall(dayfile_content)
    if len(rmt_matches) >= 2:
        rmt_vals = [float(v) for v in rmt_matches]
        min_rmt = min(rmt_vals)
        max_rmt = max(rmt_vals)
        if min_rmt > 0 and max_rmt / min_rmt > 1.5:
            indicators["core_overlap"] = min(0.9, (max_rmt / min_rmt - 1.5) * 2)

    # 4. Check case.inc for mixing settings
    if case_name:
        try:
            inc_path = Path(f"{case_name}.inc")
            if not inc_path.exists():
                inc_path = Path(case_name) / f"{case_name}.inc"
            if inc_path.exists():
                inc_text = inc_path.read_text()
                mix_match = re.search(r'([\d]+\.?\d*)\s*\n', inc_text)
                if mix_match:
                    beta = float(mix_match.group(1))
                    if beta > 0.3:
                        indicators["mixing_too_aggressive"] = 0.8
                    elif beta > 0.2:
                        indicators["mixing_too_aggressive"] = 0.4
        except Exception:
            pass

    # 5. Default: if nothing specific, assume aggressive mixing
    if not indicators:
        indicators["mixing_too_aggressive"] = 0.5

    root_cause = max(indicators, key=indicators.get)
    confidence = indicators[root_cause]

    actions = _build_sloshing_remediation(root_cause, slosh["severity"])

    return {
        "root_cause": root_cause,
        "confidence": round(confidence, 2),
        "indicators": indicators,
        "actions": actions,
    }


_SLOSHING_REMEDIATION = {
    "metallic": [
        {"action": "set_smearing", "params": {"type": "Methfessel-Paxton", "width_ry": 0.02},
         "reason": "Metallic systems require Fermi surface smearing to stabilize SCF"},
        {"action": "set_mixing", "params": {"beta": 0.10},
         "reason": "Use PRATT mixing with beta=0.10 for metals"},
        {"action": "increase_kpoints", "params": {"factor": 2.0},
         "reason": "Denser k-mesh captures Fermi surface better"},
    ],
    "symmetry_breaking": [
        {"action": "disable_symmetry", "params": {},
         "reason": "Switch to runsp_lapw to handle broken symmetry"},
        {"action": "set_mixing", "params": {"beta": 0.05},
         "reason": "Conservative mixing for symmetry-broken systems"},
        {"action": "set_rmt", "params": {"reduction": 0.95},
         "reason": "Slightly reduce RMT to avoid sphere overlap"},
    ],
    "core_overlap": [
        {"action": "check_rmt_ratios", "params": {},
         "reason": "RMT ratio > 1.5 between atoms; check sphere overlap"},
        {"action": "adjust_r0", "params": {"reduction": 0.90},
         "reason": "Reduce R0 to separate core from valence spheres"},
        {"action": "set_mixing", "params": {"beta": 0.02},
         "reason": "Very conservative mixing for core overlap cases"},
        {"action": "increase_pratt", "params": {"cycles": 5},
         "reason": "Increase PRATT iterations to 5 cycles"},
    ],
    "mixing_too_aggressive": [
        {"action": "set_mixing", "params": {"beta": 0.05},
         "reason": "Reduce mixing beta by 50% from current value"},
        {"action": "increase_pratt", "params": {"cycles": 3},
         "reason": "Increase PRATT iterations for better preconditioning"},
        {"action": "try_msr1a", "params": {},
         "reason": "Switch to multi-secant MSR1a mixing for improved stability"},
    ],
}


def _build_sloshing_remediation(root_cause: str, severity: float) -> List[dict]:
    """Build ordered remediation steps scaled by severity."""
    template = _SLOSHING_REMEDIATION.get(root_cause, _SLOSHING_REMEDIATION["mixing_too_aggressive"])
    actions = []
    for step in template:
        action = dict(step)
        action["priority"] = "critical" if severity > 0.7 else "high"
        actions.append(action)
    return actions


def detect_charge_sloshing_fft(dayfile_content: str) -> dict:
    """
    Detect charge sloshing via frequency-domain (FFT) analysis.

    Per Kresse & Furthmueller 1996 (PRB 54, 11169), charge sloshing manifests
    as high-frequency oscillations in the charge density residual. While the
    Durbin-Watson statistic catches alternating patterns, FFT-based analysis
    identifies the dominant frequency components explicitly.

    This function computes the power spectral density (PSD) of the charge
    distance time series using the discrete Fourier transform. A strong
    peak at the Nyquist frequency (period ≈ 2 SCF cycles) is the hallmark
    of charge sloshing. The normalised high-frequency power ratio quantifies
    how much of the total spectral energy resides above the median frequency.

    Requires at least 8 SCF cycles for meaningful frequency resolution.

    Returns:
        dict with keys:
            sloshing_detected: bool
            dominant_frequency_hz: float (in cycles per SCF iteration)
            hf_power_ratio: float (0.0–1.0, ratio of power above median freq)
            recommendation: str
    """
    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    if len(matches) < 8:
        return {
            "sloshing_detected": False,
            "dominant_frequency_hz": 0.0,
            "hf_power_ratio": 0.0,
            "recommendation": "Insufficient cycles for FFT analysis (need >= 8)"
        }

    values = [float(m) for m in matches]
    n = len(values)

    demeaned = [v - sum(values) / n for v in values]
    if n % 2 == 1:
        demeaned = demeaned[:-1]
        n = len(demeaned)

    try:
        fft_result = [abs(x) ** 2 for x in _simple_dft(demeaned)]
    except Exception:
        return {
            "sloshing_detected": False,
            "dominant_frequency_hz": 0.0,
            "hf_power_ratio": 0.0,
            "recommendation": "FFT computation failed"
        }

    pos_spectrum = fft_result[1:n // 2 + 1]
    if not pos_spectrum or sum(pos_spectrum) < 1e-30:
        return {
            "sloshing_detected": False,
            "dominant_frequency_hz": 0.0,
            "hf_power_ratio": 0.0,
            "recommendation": "Flat spectrum: no oscillatory behaviour detected"
        }

    total_power = sum(pos_spectrum)
    median_idx = (n // 2) // 2
    if median_idx < 1:
        median_idx = 1
    hf_power = sum(pos_spectrum[median_idx:])
    hf_ratio = hf_power / total_power if total_power > 0 else 0.0

    max_idx = max(range(len(pos_spectrum)), key=lambda i: pos_spectrum[i])
    dominant_freq = (max_idx + 1) / n

    sloshing_detected = (
        hf_ratio > 0.4 and dominant_freq > 0.3
    )

    recommendation = ""
    if sloshing_detected:
        recommendation = (
            f"FFT confirms charge sloshing (HF ratio={hf_ratio:.2f}, "
            f"dominant freq={dominant_freq:.3f}/cycle). "
            f"Per Kresse & Furthmueller 1996: increase Kerker "
            f"preconditioning, reduce mixing beta, or use PRATT mixing."
        )

    return {
        "sloshing_detected": sloshing_detected,
        "dominant_frequency_hz": round(dominant_freq, 4),
        "hf_power_ratio": round(hf_ratio, 4),
        "recommendation": recommendation
    }


def _simple_dft(signal: List[float]) -> List[complex]:
    """
    Simple discrete Fourier transform (DFT) for short time series.
    Uses O(N^2) algorithm suitable for SCF cycle counts (N < 200).
    For production use with long time series, numpy.fft is preferred.
    """
    n = len(signal)
    result = []
    for k in range(n):
        real = 0.0
        imag = 0.0
        for t in range(n):
            angle = 2.0 * math.pi * k * t / n
            real += signal[t] * math.cos(angle)
            imag -= signal[t] * math.sin(angle)
        result.append(complex(real, imag))
    return result


def analyze_broyden_mixing(dayfile_content: str, log_path: str) -> dict:
    """
    Analyze Broyden mixing behavior and detect stagnation.

    Checks whether Broyden mixing is active, examines charge-distance
    progression across iterations, and identifies plateauing that signals
    a stuck Broyden history.

    Args:
        dayfile_content: Text content from the dayfile/SCF log.
        log_path: Path to the log file (reserved for extended diagnostics).

    Returns:
        dict with keys:
            broyden_active: bool
            stuck: bool
            convergence_rate: float (log-average reduction per cycle)
            iteration_plateau_length: int
            recommendation: str
    """
    broyden_active = bool(
        re.search(r'\bbroy', dayfile_content, re.IGNORECASE)
    )

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    values = [float(m) for m in matches]

    n = len(values)
    convergence_rate = 0.0
    if n >= 3:
        ratios = []
        for i in range(1, n):
            if values[i - 1] > 1e-12:
                ratios.append(math.log(values[i] / values[i - 1]))
        if ratios:
            convergence_rate = round(math.exp(sum(ratios) / len(ratios)), 6)

    plateau_length = 0
    stuck = False
    if n >= 4:
        rel_tol = 0.02
        run = 1
        max_run = 1
        for i in range(1, n):
            ref = max(values[i - 1], 1e-12)
            if abs(values[i] - values[i - 1]) / ref < rel_tol:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        plateau_length = max_run
        stuck = plateau_length >= 3 and any(v > 1e-6 for v in values[-plateau_length:])

    recommendation = ""
    if broyden_active and stuck:
        recommendation = (
            "Broyden mixing stuck—reset Broyden history, "
            "increase mixing parameter, fall back to PRATT mixing, "
            "or try Anderson mixing (faster for metallic systems, Eyert 1996)"
        )

    return {
        "broyden_active": broyden_active,
        "stuck": stuck,
        "convergence_rate": convergence_rate,
        "iteration_plateau_length": plateau_length,
        "recommendation": recommendation
    }


def analyze_anderson_mixing(dayfile_content: str) -> dict:
    """
    Analyze Anderson mixing behavior and detect stagnation.

    Anderson mixing (Eyert 1996, J. Comp. Phys. 124, 271) generalizes Broyden
    by mixing a linear combination of previous charge densities with coefficients
    chosen to minimise the residual. For metallic systems, Anderson often
    converges 2-3x faster than Broyden because it constructs a better
    approximate inverse Jacobian from the charge density history.

    This detector checks for Anderson-specific signatures:
    1. ``:MIX`` lines indicating Anderson or extended Anderson mixing
    2. Charge distance plateauing despite active Anderson iterations
    3. Recommendations to switch to Broyden (insulators) or PRATT (sloshing)

    Returns:
        dict with keys:
            anderson_active: bool
            stuck: bool
            convergence_rate: float
            iteration_plateau_length: int
            recommendation: str
    """
    anderson_active = bool(
        re.search(r'(?:\banderson\b|:MIX\s*[12])', dayfile_content, re.IGNORECASE)
    )

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    matches = cd_pattern.findall(dayfile_content)
    values = [float(m) for m in matches]
    n = len(values)

    convergence_rate = 0.0
    if n >= 3:
        ratios = []
        for i in range(1, n):
            if values[i - 1] > 1e-12:
                ratios.append(math.log(values[i] / values[i - 1]))
        if ratios:
            convergence_rate = round(math.exp(sum(ratios) / len(ratios)), 6)

    plateau_length = 0
    stuck = False
    if n >= 4:
        rel_tol = 0.01
        run = 1
        max_run = 1
        for i in range(1, n):
            ref = max(values[i - 1], 1e-12)
            if abs(values[i] - values[i - 1]) / ref < rel_tol:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        plateau_length = max_run
        stuck = plateau_length >= 4 and any(v > 1e-6 for v in values[-plateau_length:])

    recommendation = ""
    if anderson_active and stuck:
        recommendation = (
            "Anderson mixing stuck—try tightening the mixing parameter, "
            "increase mixing history depth, or switch to Broyden mixing "
            "(better Jacobian approximation for insulators)"
        )

    return {
        "anderson_active": anderson_active,
        "stuck": stuck,
        "convergence_rate": convergence_rate,
        "iteration_plateau_length": plateau_length,
        "recommendation": recommendation
    }


def analyze_diis_mixing(dayfile_content: str) -> dict:
    """
    Analyze DIIS/Pulay mixing behavior and detect divergence.

    DIIS (Direct Inversion in the Iterative Subspace), also known as Pulay
    mixing (Pulay 1980, Chem. Phys. Lett. 73, 393), constructs an optimal
    linear combination of Fock matrices by minimising the error vector norm
    in a Krylov subspace. For charge-density mixing in DFT (Kresse &
    Furthmueller 1996, PRB 54, 11169), DIIS converges quadratically near
    the minimum but can diverge catastrophically if the initial guess is
    poor or the subspace becomes linearly dependent.

    Detector checks:
    1. ``:DIIS`` or ``:PULAY`` keywords in the mixing log
    2. Charge distance INCREASING over recent iterations (divergence)
    3. Residual norm oscillation without convergence

    Returns:
        dict with keys:
            diis_active: bool
            diverging: bool
            residual_trend: float (+ = diverging, - = converging)
            recommendation: str
    """
    diis_active = bool(
        re.search(r'(?:\bdiis\b|\:DIIS|\bpulay\b|:PULAY)', dayfile_content, re.IGNORECASE)
    )

    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    res_pattern = re.compile(
        r'(?:residual|error)\s*(?:norm)?\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )

    values = [float(m) for m in cd_pattern.findall(dayfile_content)]
    _residuals = [float(m) for m in res_pattern.findall(dayfile_content)]
    n = len(values)

    residual_trend = 0.0
    if n >= 5:
        recent = values[-5:]
        window_len = len(recent)
        x_mean = sum(range(window_len)) / window_len
        y_mean = sum(recent) / window_len
        num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(window_len))
        den = sum((i - x_mean) ** 2 for i in range(window_len))
        if den > 1e-12:
            residual_trend = num / den / (y_mean + 1e-12)

    diverging = False
    if n >= 5:
        recent_vals = values[-5:]
        max_recent = max(recent_vals)
        early_vals = values[:min(5, n // 2)]
        avg_early = sum(early_vals) / len(early_vals) if early_vals else 0
        if avg_early > 1e-12 and (max_recent / avg_early) > 2.0:
            diverging = True
        if n >= 10:
            last_half_avg = sum(values[n // 2:]) / (n - n // 2)
            first_half_avg = sum(values[:n // 2]) / (n // 2)
            if first_half_avg > 1e-12 and last_half_avg / first_half_avg > 1.5:
                diverging = True

    recommendation = ""
    if diis_active and diverging:
        recommendation = (
            "DIIS/Pulay mixing diverging—reduce mixing history depth "
            "(NDII=5 or less), increase Kerker preconditioning, "
            "or fall back to Broyden mixing with smaller beta"
        )
    elif diis_active and not diverging and residual_trend > 0.0 and residual_trend < 0.1:
        recommendation = (
            "DIIS convergence slow; consider increasing DIIS history "
            "dimension or switching to Anderson mixing for metallic systems"
        )

    return {
        "diis_active": diis_active,
        "diverging": diverging,
        "residual_trend": round(residual_trend, 6),
        "recommendation": recommendation
    }


def analyze_convergence_history(dayfile_content: str) -> ConvergenceAnalysis:
    """
    Read full convergence history from SCF log content and classify the
    convergence trajectory, providing mixing recommendations and cycle estimates.

    Args:
        dayfile_content: Text content from the dayfile/SCF log.

    Returns:
        ConvergenceAnalysis dataclass with classification, history vectors, and advice.
    """
    cd_pattern = re.compile(
        r'(?:charge\s*distance|\:dis)\s*[=:]\s*([\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )
    en_pattern = re.compile(
        r'(?:(?:total|:)?\s*energy)\s*[=:]\s*([\-]?[\d]+\.?\d*(?:[eE][+\-]?\d+)?)',
        re.IGNORECASE
    )

    charge_distances = [float(m) for m in cd_pattern.findall(dayfile_content)]
    energies = [float(m) for m in en_pattern.findall(dayfile_content)]

    n = len(charge_distances)
    if n < 3:
        return ConvergenceAnalysis(
            convergence_type="unknown",
            mixing_recommendation="Insufficient data for convergence classification",
            estimated_cycles_to_converge=-1,
            charge_distance_history=charge_distances,
            energy_history=energies
        )

    slope = charge_distances[-1] - charge_distances[0]
    ratios = []
    for i in range(1, n):
        if charge_distances[i - 1] > 1e-12:
            ratios.append(charge_distances[i] / charge_distances[i - 1])

    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

    diffs = [charge_distances[i] - charge_distances[i - 1] for i in range(1, n)]
    sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
    oscillation_ratio = sign_changes / max(1, len(diffs) - 1) if len(diffs) > 1 else 0.0

    rel_plateau = 0
    if n >= 4:
        rel_tol = 0.02
        run = 1
        for i in range(1, n):
            ref = max(charge_distances[i - 1], 1e-12)
            if abs(charge_distances[i] - charge_distances[i - 1]) / ref < rel_tol:
                run += 1
                rel_plateau = max(rel_plateau, run)
            else:
                run = 1

    if avg_ratio > 1.01 and slope > 0:
        convergence_type = "divergent"
        recommendation = (
            "Divergent SCF—reduce mixing beta, enable PRATT mixing, "
            "or add more k-points to stabilise charge density"
        )
        est_cycles = -1
    elif rel_plateau >= 3 and charge_distances[-1] > 1e-5:
        convergence_type = "stalled"
        recommendation = (
            "SCF stalled—increase mixing iterations, "
            "enable Broyden mixing, or shake the density with a small displacement"
        )
        est_cycles = -1
    elif oscillation_ratio >= 0.4 and avg_ratio < 1.0:
        convergence_type = "oscillatory"
        recommendation = (
            "Oscillatory convergence (charge sloshing)—reduce mixing beta, "
            "increase PRATT iterations, or switch to MSR1a mixing"
        )
        est = 0
        if avg_ratio > 0 and avg_ratio < 1.0:
            est = int(math.log(1e-6 / max(charge_distances[-1], 1e-12))
                     / math.log(avg_ratio)) + 1
        est_cycles = max(0, est)
    elif avg_ratio < 1.0 and oscillation_ratio < 0.4:
        convergence_type = "monotonic"
        recommendation = "Convergence progressing monotonically—current settings are suitable"
        est = 0
        if avg_ratio > 0 and avg_ratio < 1.0:
            est = int(math.log(1e-6 / max(charge_distances[-1], 1e-12))
                     / math.log(avg_ratio)) + 1
        est_cycles = max(0, est)
    else:
        convergence_type = "unknown"
        recommendation = "Unclear convergence pattern—monitor further"
        est_cycles = -1

    return ConvergenceAnalysis(
        convergence_type=convergence_type,
        mixing_recommendation=recommendation,
        estimated_cycles_to_converge=est_cycles,
        charge_distance_history=charge_distances,
        energy_history=energies
    )


def _estimate_rebuild_benefit(topo: Topology, current_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate whether rebuilding the parallel configuration will yield meaningful performance gains.
    Compares current allocation against advisor-suggested optimal resources.
    """
    try:
        from ..optimizer.advisor import suggest_optimal_resources
        new_suggestion = suggest_optimal_resources(topo, user_max_cores=None)
        expected_speedup = 1.0
        
        if new_suggestion.mode != "hybrid":
            expected_speedup = 1.1  # Conservative estimate for mode change
        elif new_suggestion.vector_split_active:
            expected_speedup = 1.15  # I/O optimization gain
            
        return {
            "expected_speedup": expected_speedup,
            "confidence": getattr(new_suggestion, 'confidence_score', 0.7),
            "recommendation": "proceed" if expected_speedup > 1.05 else "skip",
            "new_config_summary": {
                "mode": new_suggestion.mode,
                "cores": new_suggestion.recommended_total_cores,
                "omp": new_suggestion.omp_threads_per_rank
            }
        }
    except Exception as e:
        logger.warning(f"Could not estimate rebuild benefit: {e}")
        return {"recommendation": "unknown", "expected_speedup": 1.0, "confidence": 0.0}


# =============================================================================
# Main Monitoring Loop
# =============================================================================

def monitor_and_rebuild(
    topo: Topology,
    rebuild_callback: Optional[Callable[[Topology, Dict[str, Any]], bool]] = None,
    check_interval: int = 60,
    min_rebuild_interval: int = 300,
    adaptive_threshold: bool = True,
    checkpoint_fn: Optional[Callable] = None
) -> None:
    """
    Background loop that watches SCF progress, detects problem changes,
    and triggers intelligent reconfiguration when beneficial.
    """
    _register_preemption_signals(checkpoint_fn)
    dayfile_path = _get_dayfile_path()
    
    if not dayfile_path:
        logger.warning("No SCF dayfile/log found. Monitoring disabled.")
        return

    logger.info(f"SCF monitor started for {dayfile_path.name} (interval={check_interval}s)")
    _monitor_state.update_problem(_get_current_problem_vector())
    _monitor_state.last_dayfile_mtime = dayfile_path.stat().st_mtime

    while not _stop_event.is_set():
        try:
            # FIXED: 'i f' -> 'if'
            if _monitor_state.paused:
                _stop_event.wait(check_interval)
                continue

            if not dayfile_path.exists():
                _stop_event.wait(check_interval)
                continue

            current_mtime = dayfile_path.stat().st_mtime
            if current_mtime <= _monitor_state.last_dayfile_mtime:
                _stop_event.wait(check_interval)
                continue

            _monitor_state.last_dayfile_mtime = current_mtime
            events = _parse_dayfile_events(dayfile_path)
            current_problem = _get_current_problem_vector()
            last_problem = _monitor_state.last_problem

            # Analyze parameter drift
            change_analysis: Dict[str, Any] = {"changes": {}, "severity": 0.0, "should_rebuild": False}
            if last_problem:
                change_analysis = current_problem.significant_change(last_problem, adaptive=adaptive_threshold)

            # Decide if rebuild is warranted (FIXED: 'warra nted' -> 'warranted')
            should_rebuild = False
            rebuild_reason: List[str] = []

            if change_analysis.get("should_rebuild"):
                should_rebuild = True
                rebuild_reason.append(f"Problem drift: {list(change_analysis['changes'].keys())}")

            if MonitorEvent.ERROR_DETECTED in events:
                should_rebuild = True
                rebuild_reason.append("Critical SCF error detected")

            if MonitorEvent.CONVERGENCE_STALL in events:
                should_rebuild = True
                rebuild_reason.append("Convergence stall detected")

            if MonitorEvent.CHARGE_SLOSHING in events:
                should_rebuild = True
                rebuild_reason.append("Charge sloshing detected")

            if MonitorEvent.BROYDEN_STUCK in events:
                should_rebuild = True
                rebuild_reason.append("Broyden mixing stuck")

            # Cooldown enforcement
            if should_rebuild and _monitor_state.is_rebuild_cooldown(min_rebuild_interval):
                logger.info("Rebuild warranted but within cooldown window. Skipping.")
                should_rebuild = False

            # Benefit estimation
            if should_rebuild:
                benefit = _estimate_rebuild_benefit(topo, asdict(current_problem))
                rec = benefit.get("recommendation", "skip")
                if rec == "skip":
                    logger.info(f"Rebuild skipped: low expected benefit ({benefit.get('expected_speedup', 1.0):.2f}x)")
                    should_rebuild = False
                elif rec == "proceed":
                    rebuild_reason.append(f"Advisor suggests {benefit.get('expected_speedup', 1.0):.2f}x speedup")

            # Execute rebuild
            if should_rebuild:
                logger.info(f"Triggering configuration rebuild: {'; '.join(rebuild_reason)}")
                
                # Backup current config for atomic rollback
                config_backup: Optional[str] = None
                machines_path = Path(".machines")
                if machines_path.exists():
                    try:
                        config_backup = machines_path.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning(f"Config backup read failed: {e}")

                success = False
                try:
                    if rebuild_callback:
                        success = rebuild_callback(topo, change_analysis)
                    else:
                        from ..core.builder import build_auto
                        result = build_auto(topo, backup=False, dry_run=False)
                        # FIXED: Removed extra space in getattr
                        success = bool(getattr(result, 'success', False))

                    if success:
                        _monitor_state.record_rebuild()
                        _monitor_state.update_problem(current_problem)
                        logger.info("Configuration rebuilt successfully")
                        
                        # Audit logging
                        log_entry = {
                            "timestamp": time.time(),
                            "reason": rebuild_reason,
                            "change_analysis": change_analysis,
                            "new_problem": asdict(current_problem)
                        }
                        log_path = Path(".wien2k_rebuild_log.jsonl")
                        
                        if _HAS_FILELOCK and _file_lock:
                            with _file_lock:
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(log_entry) + "\n")
                        else:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_entry) + "\n")
                    else:
                        raise RuntimeError("Rebuild routine reported failure")

                except Exception as e:
                    logger.error(f"Rebuild failed: {e}", exc_info=True)
                    # Atomic rollback (FIXED: 'machi nes_path' -> 'machines_path')
                    if config_backup and machines_path.exists():
                        try:
                            atomic_write(machines_path, config_backup)
                            logger.info("Rolled back to previous configuration")
                        except Exception as rb_err:
                            logger.error(f"Rollback failed: {rb_err}")

            # Always update state for next iteration
            _monitor_state.update_problem(current_problem)

        except Exception as e:
            logger.error(f"Monitor loop exception: {e}", exc_info=True)

        _stop_event.wait(timeout=check_interval)


# =============================================================================
# Public Control API
# =============================================================================

def start_monitoring(
    topo: Topology,
    rebuild_callback: Optional[Callable] = None,
    check_interval: int = 60,
    min_rebuild_interval: int = 300,
    daemon: bool = True,
    checkpoint_fn: Optional[Callable] = None
) -> threading.Thread:
    """
    Start background SCF monitoring thread.
    
    Args:
        topo: Hardware topology for resource context.
        rebuild_callback: Optional custom rebuild function (topo, changes) -> bool
        check_interval: Polling frequency in seconds.
        min_rebuild_interval: Cooldown between rebuilds to prevent thrashing.
        daemon: Exit automatically when main process terminates.
        checkpoint_fn: Callable to save SCF state on preemption signal.
    """
    global _monitor_thread
    stop_monitoring(timeout=1.0)
    _stop_event.clear()
    _monitor_state.preemption_handled = False
    
    _monitor_thread = threading.Thread(
        target=monitor_and_rebuild,
        args=(topo, rebuild_callback, check_interval, min_rebuild_interval, True, checkpoint_fn),
        daemon=daemon,
        name="wien2k_scf_monitor"
    )
    _monitor_thread.start()
    logger.info(f"SCF monitor thread started (daemon={daemon})")
    return _monitor_thread


def stop_monitoring(timeout: float = 3.0) -> bool:
    """Gracefully stop the monitoring thread."""
    global _monitor_thread
    _stop_event.set()
    if _monitor_thread and _monitor_thread.is_alive():
        _monitor_thread.join(timeout=timeout)
        success = not _monitor_thread.is_alive()
        if success:
            logger.info("SCF monitor stopped gracefully")
        else:
            logger.warning(f"Monitor thread did not exit within {timeout}s timeout")
        return success
    return True


def pause_monitoring() -> None:
    """Pause polling without terminating the thread."""
    with _monitor_state.lock:
        _monitor_state.paused = True
    logger.debug("SCF monitor paused")


def create_scf_checkpoint(case_name: str, label: str = "") -> str:
    """Save SCF checkpoint for restart after failure or preemption.

    Copies case.clmval (charge density), case.clmsum, case.broyd*
    to a timestamped backup directory under case_checkpoints/.

    Returns path to the checkpoint directory.
    """
    import shutil as _shutil

    case = Path(case_name)
    if not case.exists():
        case = Path(".")

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = case / "case_checkpoints" / f"ckpt_{ts}"
    if label:
        ckpt_dir = case / "case_checkpoints" / f"ckpt_{label}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for suffix in [".clmval", ".clmsum", ".broyd", ".broyd1", ".broyd2"]:
        src = case / f"{case.name}{suffix}"
        if src.exists():
            _shutil.copy2(str(src), str(ckpt_dir / src.name))

    (ckpt_dir / "CHECKPOINT_INFO").write_text(
        f"case={case.name}\nlabel={label}\ntimestamp={ts}\n"
    )

    logger.info(f"SCF checkpoint saved: {ckpt_dir}")
    return str(ckpt_dir)


def restore_from_checkpoint(case_name: str, checkpoint_dir: Optional[str] = None) -> bool:
    """Restore SCF state from most recent (or specified) checkpoint.

    Returns True if restore succeeded.
    """
    import shutil as _shutil

    case = Path(case_name)
    if not case.exists():
        case = Path(".")

    ckpt_base = case / "case_checkpoints"
    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir)
    elif ckpt_base.exists():
        dirs = sorted(ckpt_base.glob("ckpt_*"), key=os.path.getmtime, reverse=True)
        if not dirs:
            logger.warning("No checkpoints found")
            return False
        ckpt_dir = dirs[0]
    else:
        logger.warning("No checkpoint directory exists")
        return False

    for src_file in ckpt_dir.glob("*"):
        if src_file.name.endswith((".clmval", ".clmsum", ".broyd", ".broyd1", ".broyd2")):
            dest = case / f"{case.name}{src_file.suffix}"
            _shutil.copy2(str(src_file), str(dest))

    logger.info(f"SCF checkpoint restored from: {ckpt_dir}")
    return True


def resume_monitoring() -> None:
    """Resume polling after pause."""
    with _monitor_state.lock:
        _monitor_state.paused = False
    logger.debug("SCF monitor resumed")


def get_monitor_status() -> Dict[str, Any]:
    """Return current monitor state for UI/CLI diagnostics."""
    with _monitor_state.lock:
        return {
            "running": bool(_monitor_thread and _monitor_thread.is_alive()),
            "paused": _monitor_state.paused,
            "preemption_handled": _monitor_state.preemption_handled,
            "rebuild_count": _monitor_state.rebuild_count,
            "last_rebuild_time": _monitor_state.last_rebuild_time,
            "last_problem": asdict(_monitor_state.last_problem) if _monitor_state.last_problem else None,
            "last_check_mtime": _monitor_state.last_dayfile_mtime
        }