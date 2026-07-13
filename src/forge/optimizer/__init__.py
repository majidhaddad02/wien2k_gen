"""
Optimizer module for WIEN2k parallel configuration.
Submodules:
• advisor: Scientific resource recommendation with Roofline modeling
• monitor: Self-healing SCF execution monitor with adaptive reconfiguration
• profiler: Statistical performance profiling with async execution
• history: SQLite-based execution history store for learning from past runs
• bayesian: Bayesian optimization for parallel execution parameter tuning
"""

from .advisor import (
    BACKEND_OPERATIONAL_INTENSITY,
    OptimizationTarget,
    ResourceSuggestion,
    estimate_amdahl_saturation,
    estimate_arithmetic_intensity,
    estimate_memory_footprint_gb,
    get_optimization_report,
    recommend,
    roofline_crossover_analysis,
    suggest_optimal_resources,
)
from .bayesian import (
    BayesianOptimizer,
    MultiFidelityBayesianOptimizer,
    compute_expected_improvement,
)
from .history import (
    ExecutionHistory,
    ExecutionRecord,
    compute_efficiency,
    suggest_from_history,
)
from .monitor import (
    ConvergenceAnalysis,
    MonitorEvent,
    ProblemVector,
    analyze_anderson_mixing,
    analyze_broyden_mixing,
    analyze_convergence_history,
    analyze_diis_mixing,
    detect_charge_sloshing,
    detect_charge_sloshing_fft,
    get_monitor_status,
    pause_monitoring,
    resume_monitoring,
    start_monitoring,
    stop_monitoring,
)
from .profiler import (
    AutoProfiler,
    ProfileResult,
    ProfilingReport,
    profile_and_select,
    profile_and_select_async,
)

# Explicit public API declaration.
# Controls `from forge.optimizer import *` and IDE auto-completion.
__all__ = [
    "BACKEND_OPERATIONAL_INTENSITY",
    "AutoProfiler",
    # Bayesian
    "BayesianOptimizer",
    "ConvergenceAnalysis",
    "ExecutionHistory",
    # History
    "ExecutionRecord",
    "MonitorEvent",
    "MultiFidelityBayesianOptimizer",
    "OptimizationTarget",
    "ProblemVector",
    "ProfileResult",
    "ProfilingReport",
    "ResourceSuggestion",
    "analyze_anderson_mixing",
    "analyze_broyden_mixing",
    "analyze_convergence_history",
    "analyze_diis_mixing",
    "compute_efficiency",
    "compute_expected_improvement",
    "detect_charge_sloshing",
    "detect_charge_sloshing_fft",
    "estimate_amdahl_saturation",
    "estimate_arithmetic_intensity",
    "estimate_memory_footprint_gb",
    "get_monitor_status",
    "get_optimization_report",
    "pause_monitoring",
    # Profiler
    "profile_and_select",
    "profile_and_select_async",
    "recommend",
    "resume_monitoring",
    "roofline_crossover_analysis",
    # Monitor
    "start_monitoring",
    "stop_monitoring",
    "suggest_from_history",
    # Advisor
    "suggest_optimal_resources",
]