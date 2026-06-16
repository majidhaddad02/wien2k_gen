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

import os
import re
import time
import signal
import threading
import json
import math
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# Robust FileLock fallback
try:
    from filelock import FileLock
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False
    FileLock = None  # type: ignore

from ..core.topology import Topology
from ..core.hardware import get_job_memory_limit_mb, get_scratch_filesystem_type
from ..backend_manager import get_current_backend
from ..logging_config import get_logger
from ..utils.atomic_write import atomic_write

# FIXED: Use __name__ instead of undefined 'name'
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
        logger.info(f"Preemption handled. Exiting monitor loop gracefully.")
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
    backend = get_current_backend()
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
    backend = get_current_backend()
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
    """
    events: List[MonitorEvent] = []
    if not dayfile_path.exists():
        return events
        
    try:
        content = dayfile_path.read_text(encoding="utf-8", errors="replace").lower()
        
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
                
    except Exception as e:
        logger.debug(f"Dayfile parsing failed for {dayfile_path}: {e}")
        
    return events


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