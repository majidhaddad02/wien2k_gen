"""
SCF monitoring: real-time convergence tracking, charge sloshing detection, and checkpoint management.
"""

from .checkpoint import (
    calculate_checkpoint_interval,
    cleanup_old_checkpoints,
    create_scf_checkpoint,
    perform_incremental_checkpoint,
    restore_from_checkpoint,
    resume_from_checkpoint,
)
from .convergence import (
    analyze_anderson_mixing,
    analyze_broyden_mixing,
    analyze_convergence_history,
    analyze_diis_mixing,
    detect_charge_sloshing,
    detect_charge_sloshing_fft,
    diagnose_charge_sloshing_root_cause,
)
from .engine import (
    estimate_remaining_walltime,
    get_monitor_status,
    pause_monitoring,
    resume_monitoring,
    start_monitoring,
    stop_monitoring,
)
from .types import ConvergenceAnalysis, MonitorEvent, MonitorState, ProblemVector

__all__ = [
    "ConvergenceAnalysis",
    "MonitorEvent",
    "MonitorState",
    "ProblemVector",
    "analyze_anderson_mixing",
    "analyze_broyden_mixing",
    "analyze_convergence_history",
    "analyze_diis_mixing",
    "calculate_checkpoint_interval",
    "cleanup_old_checkpoints",
    "create_scf_checkpoint",
    "detect_charge_sloshing",
    "detect_charge_sloshing_fft",
    "diagnose_charge_sloshing_root_cause",
    "estimate_remaining_walltime",
    "get_monitor_status",
    "pause_monitoring",
    "perform_incremental_checkpoint",
    "restore_from_checkpoint",
    "resume_from_checkpoint",
    "resume_monitoring",
    "start_monitoring",
    "stop_monitoring",
]
