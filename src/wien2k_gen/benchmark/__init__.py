"""
Benchmark Package Initialization for Wien2kGen.
Exports synthetic workload simulation, real-world cluster execution,
and calibration utilities for HPC/DFT performance validation.

Submodules:
• synthetic: Roofline-based DFT workload simulation, strong/weak scaling suites
• real: Cluster job orchestration, empirical timing collection, output parsing
• calibration: Real-vs-predicted deviation analysis & model tuning

Designed for seamless integration with optimizer/profiler.py and analysis.py.
"""

# =============================================================================
# Synthetic Simulation Engine
# =============================================================================
from .synthetic import (
    SyntheticWorkloadParams,
    BenchmarkResult,
    SimulationConfig,
    WorkloadSimulator,
    generate_strong_scaling_suite,
    generate_weak_scaling_suite,
)

# =============================================================================
# Real-World Execution & Calibration
# =============================================================================
from .real import (
    RealBenchmarkConfig,
    RealBenchmarkResult,
    BenchmarkExecutionState,
    RealBenchmarkRunner,
    calibrate_real_vs_synthetic,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    # Synthetic
    "SyntheticWorkloadParams",
    "BenchmarkResult",
    "SimulationConfig",
    "WorkloadSimulator",
    "generate_strong_scaling_suite",
    "generate_weak_scaling_suite",
    # Real & Calibration
    "RealBenchmarkConfig",
    "RealBenchmarkResult",
    "BenchmarkExecutionState",
    "RealBenchmarkRunner",
    "calibrate_real_vs_synthetic",
]