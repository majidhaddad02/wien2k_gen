"""Monitoring engine: real-time SCF loop, rebuild triggers, preemption, threading."""

import json
import re
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from ...core.topology import Topology
from ...logging_config import get_logger
from ...utils.atomic_write import atomic_write
from .convergence import (
    analyze_anderson_mixing,
    analyze_broyden_mixing,
    analyze_diis_mixing,
    detect_charge_sloshing,
    detect_charge_sloshing_fft,
)
from .types import (
    _HAS_FILELOCK,
    MonitorEvent,
    ProblemVector,
    _file_lock,
    _get_current_backend,
    _monitor_state,
    _monitor_thread,
    _stop_event,
)

logger = get_logger(__name__)


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
            logger.debug("Failed to find current dayfile via backend log", exc_info=True)
            
    patterns = ["*.scf", "*.dayfile", "*.output", "case*.scf"]
    for pattern in patterns:
        matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _parse_dayfile_events(dayfile_path: Path) -> list[MonitorEvent]:  # noqa: C901
    """
    Parse SCF dayfile content to detect convergence behavior, errors, or cycle completion.
    Uses regex-based extraction for robust matching across WIEN2k versions.
    Also invokes charge-sloshing and Broyden-mixing detectors.
    """
    events: list[MonitorEvent] = []
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


def _estimate_rebuild_benefit(topo: Topology, current_params: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate whether rebuilding the parallel configuration will yield meaningful performance gains.
    Compares current allocation against advisor-suggested optimal resources.
    """
    try:
        from ..advisor import suggest_optimal_resources
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


def monitor_and_rebuild(  # noqa: C901
    topo: Topology,
    rebuild_callback: Optional[Callable[[Topology, dict[str, Any]], bool]] = None,
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
            change_analysis: dict[str, Any] = {"changes": {}, "severity": 0.0, "should_rebuild": False}
            if last_problem:
                change_analysis = current_problem.significant_change(last_problem, adaptive=adaptive_threshold)

            # Decide if rebuild is warranted (FIXED: 'warra nted' -> 'warranted')
            should_rebuild = False
            rebuild_reason: list[str] = []

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
                            with _file_lock, open(log_path, "a", encoding="utf-8") as f:
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


def get_monitor_status() -> dict[str, Any]:
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


def estimate_remaining_walltime(job_id: str, scheduler: str = "slurm") -> dict[str, Any]:  # noqa: C901
    """Estimate remaining walltime for a running job.

    Reads job info from SLURM (scontrol) or PBS (qstat -f).
    Returns dict with:
        walltime_limit_sec: float — total walltime requested
        elapsed_sec: float — elapsed runtime
        remaining_sec: float — remaining walltime
        remaining_pct: float — remaining fraction (0-100)
        scheduler: str
    """
    import subprocess as _sp

    result = {
        "walltime_limit_sec": 3600.0,
        "elapsed_sec": 0.0,
        "remaining_sec": 3600.0,
        "remaining_pct": 100.0,
        "scheduler": scheduler,
        "job_id": job_id,
    }

    try:
        if scheduler == "slurm":
            proc = _sp.run(
                ["scontrol", "show", "jobid", job_id],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout:
                tlimit = re.search(r'TimeLimit=(\d+):(\d+):(\d+)', proc.stdout)
                elapsed = re.search(r'RunTime=(\d+):(\d+):(\d+)', proc.stdout)
                if tlimit:
                    h, m, s = int(tlimit.group(1)), int(tlimit.group(2)), int(tlimit.group(3))
                    result["walltime_limit_sec"] = float(h * 3600 + m * 60 + s)
                if elapsed:
                    h, m, s = int(elapsed.group(1)), int(elapsed.group(2)), int(elapsed.group(3))
                    result["elapsed_sec"] = float(h * 3600 + m * 60 + s)
        elif scheduler == "pbs":
            proc = _sp.run(
                ["qstat", "-f", job_id],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0 and proc.stdout:
                w_match = re.search(r'Resource_List\.walltime\s*=\s*(\d+):(\d+):(\d+)', proc.stdout)
                e_match = re.search(r'resources_used\.walltime\s*=\s*(\d+):(\d+):(\d+)', proc.stdout)
                if w_match:
                    h, m, s = int(w_match.group(1)), int(w_match.group(2)), int(w_match.group(3))
                    result["walltime_limit_sec"] = float(h * 3600 + m * 60 + s)
                if e_match:
                    h, m, s = int(e_match.group(1)), int(e_match.group(2)), int(e_match.group(3))
                    result["elapsed_sec"] = float(h * 3600 + m * 60 + s)
    except Exception:
        logger.debug("Job time parsing failed from .time file", exc_info=True)

    result["remaining_sec"] = max(0.0, result["walltime_limit_sec"] - result["elapsed_sec"])
    if result["walltime_limit_sec"] > 0:
        result["remaining_pct"] = round(result["remaining_sec"] / result["walltime_limit_sec"] * 100, 1)

    return result


__all__ = [
    "estimate_remaining_walltime",
    "get_monitor_status",
    "monitor_and_rebuild",
    "pause_monitoring",
    "resume_monitoring",
    "start_monitoring",
    "stop_monitoring",
]
