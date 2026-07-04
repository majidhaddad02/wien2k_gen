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
    suggest_optimal_resources,
    recommend,
    OptimizationTarget,
    ResourceSuggestion,
    estimate_memory_footprint_gb,
    roofline_crossover_analysis,
    estimate_arithmetic_intensity,
    get_optimization_report,
    BACKEND_OPERATIONAL_INTENSITY,
)
from .monitor import (
    start_monitoring,
    stop_monitoring,
    pause_monitoring,
    resume_monitoring,
    get_monitor_status,
    MonitorEvent,
    ProblemVector,
    ConvergenceAnalysis,
    detect_charge_sloshing,
    detect_charge_sloshing_fft,
    analyze_broyden_mixing,
    analyze_anderson_mixing,
    analyze_diis_mixing,
    analyze_convergence_history,
)
from .profiler import (
    profile_and_select,
    profile_and_select_async,
    AutoProfiler,
    ProfileResult,
    ProfilingReport,
)
from .history import (
    ExecutionRecord,
    ExecutionHistory,
    suggest_from_history,
    compute_efficiency,
)
from .bayesian import (
    BayesianOptimizer,
    MultiFidelityBayesianOptimizer,
    compute_expected_improvement,
)

# Explicit public API declaration.
# Controls `from wien2k_gen.optimizer import *` and IDE auto-completion.
__all__ = [
    # Advisor
    "suggest_optimal_resources",
    "recommend",
    "OptimizationTarget",
    "ResourceSuggestion",
    "estimate_memory_footprint_gb",
    "roofline_crossover_analysis",
    "estimate_arithmetic_intensity",
    "get_optimization_report",
    "BACKEND_OPERATIONAL_INTENSITY",
    # Monitor
    "start_monitoring",
    "stop_monitoring",
    "pause_monitoring",
    "resume_monitoring",
    "get_monitor_status",
    "MonitorEvent",
    "ProblemVector",
    "ConvergenceAnalysis",
    "detect_charge_sloshing",
    "detect_charge_sloshing_fft",
    "analyze_broyden_mixing",
    "analyze_anderson_mixing",
    "analyze_diis_mixing",
    "analyze_convergence_history",
    # Profiler
    "profile_and_select",
    "profile_and_select_async",
    "AutoProfiler",
    "ProfileResult",
    "ProfilingReport",
    # History
    "ExecutionRecord",
    "ExecutionHistory",
    "suggest_from_history",
    "compute_efficiency",
    # Bayesian
    "BayesianOptimizer",
    "MultiFidelityBayesianOptimizer",
    "compute_expected_improvement",
]