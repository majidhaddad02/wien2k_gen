"""
Benchmark Package Initialization for FORGE.
Exports synthetic workload simulation, real-world cluster execution,
calibration utilities, and report generation for HPC/DFT performance validation.

Submodules:
• synthetic: Roofline-based DFT workload simulation, strong/weak scaling suites
• real: Cluster job orchestration, empirical timing collection, output parsing
• report: Speedup/efficiency charts, text reports, YAML loading
• calibration: Real-vs-predicted deviation analysis & model tuning

Designed for seamless integration with optimizer/profiler.py and analysis.py.
"""

# =============================================================================
# Synthetic Simulation Engine
# =============================================================================
# =============================================================================
# Real-World Execution & Calibration
# =============================================================================
from .real import (
    BenchmarkExecutionState,
    RealBenchmarkConfig,
    RealBenchmarkResult,
    RealBenchmarkRunner,
    calibrate_real_vs_synthetic,
)

# =============================================================================
# Report Generation
# =============================================================================
from .report import (
    ScalingDataPoint,
    ScalingSeries,
    generate_charts,
    generate_report,
    generate_text_report,
    load_series_from_yaml,
)
from .synthetic import (
    BenchmarkResult,
    SimulationConfig,
    SyntheticWorkloadParams,
    WorkloadSimulator,
    generate_strong_scaling_suite,
    generate_weak_scaling_suite,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    "BenchmarkExecutionState",
    "BenchmarkResult",
    # Real & Calibration
    "RealBenchmarkConfig",
    "RealBenchmarkResult",
    "RealBenchmarkRunner",
    # Reports
    "ScalingDataPoint",
    "ScalingSeries",
    "SimulationConfig",
    # Synthetic
    "SyntheticWorkloadParams",
    "WorkloadSimulator",
    "calibrate_real_vs_synthetic",
    "generate_charts",
    "generate_report",
    "generate_strong_scaling_suite",
    "generate_text_report",
    "generate_weak_scaling_suite",
    "load_series_from_yaml",
]