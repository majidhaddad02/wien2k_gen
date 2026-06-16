"""
Optimizer module for WIEN2k parallel configuration.
Submodules:
• advisor: Scientific resource recommendation with Roofline modeling
• monitor: Self-healing SCF execution monitor with adaptive reconfiguration
• profiler: Statistical performance profiling with async execution
"""

from .advisor import (
    suggest_optimal_resources,
    recommend,
    OptimizationTarget,
    ResourceSuggestion,
    estimate_memory_footprint_gb,
)
from .monitor import (
    start_monitoring,
    stop_monitoring,
    pause_monitoring,
    resume_monitoring,
    get_monitor_status,
    MonitorEvent,
    ProblemVector,
)
from .profiler import (
    profile_and_select,
    profile_and_select_async,
    AutoProfiler,
    ProfileResult,
    ProfilingReport,
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
    # Monitor
    "start_monitoring",
    "stop_monitoring",
    "pause_monitoring",
    "resume_monitoring",
    "get_monitor_status",
    "MonitorEvent",
    "ProblemVector",
    # Profiler
    "profile_and_select",
    "profile_and_select_async",
    "AutoProfiler",
    "ProfileResult",
    "ProfilingReport",
]