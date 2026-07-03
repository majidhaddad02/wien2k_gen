"""
Core Package Initialization for WIEN2k Parallel Execution Engine.
Exposes topology detection, hardware profiling, scheduler integration,
configuration building, and pipeline orchestration.
Designed for exascale HPC environments with robust error handling,
type safety, and explicit public API declaration.
"""

# =============================================================================
# Topology & Hardware Detection
# =============================================================================
from .topology import (
    Topology,
    TopologyValidationError,
    NUMANode,
    NodeSpec,
    GPUInfo,
    GPUTopology,
    detect_gpu_topology,
)
from .hardware import (
    get_physical_cores,
    get_logical_cores,
    is_hyperthreading_active,
    get_total_mem_kb,
    get_job_memory_limit_mb,
    get_numa_topology_detailed,
    get_cache_topology,
    get_cpu_frequency_info,
    get_cpu_governor,
    get_cpu_architecture,
    get_scratch_filesystem_type,
    get_interconnect_info,
    get_memory_bandwidth_gb_s,
    is_containerized,
    check_elpa_available,
    check_mkl_available,
    get_hardware_profile
)

# =============================================================================
# Scheduler & Environment Detection
# =============================================================================
from .scheduler import (
    detect,
    SchedulerHints
)

# =============================================================================
# Configuration Builder & Pipeline
# =============================================================================
from .builder import (
    build_auto,
    build_mpi,
    build_hybrid,
    build_kpoint,
    BuildResult
)
from .pipeline import (
    run_pipeline,
    preflight_check,
    detect_wien2k_version,
    get_total_ram_gb,
    get_numa_node_count
)

# =============================================================================
# Performance Counters & Energy
# =============================================================================
from .perf_counters import (
    get_real_roofline_data,
    measure_memory_bandwidth,
    measure_peak_flops,
    measure_cache_bandwidth,
    PerfCounterCache,
    HAS_PERF_COUNTERS,
)
from .energy import (
    EnergyMeasurement,
    measure_energy_section,
    get_rapl_energy_uj,
    estimate_energy_per_scf_cycle,
    get_power_cap,
    HAS_RAPL,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    # Topology & Hardware
    "Topology",
    "TopologyValidationError",
    "NUMANode",
    "NodeSpec",
    "GPUInfo",
    "GPUTopology",
    "detect_gpu_topology",
    "get_physical_cores",
    "get_logical_cores",
    "is_hyperthreading_active",
    "get_total_mem_kb",
    "get_job_memory_limit_mb",
    "get_numa_topology_detailed",
    "get_cache_topology",
    "get_cpu_frequency_info",
    "get_cpu_governor",
    "get_cpu_architecture",
    "get_scratch_filesystem_type",
    "get_interconnect_info",
    "get_memory_bandwidth_gb_s",
    "is_containerized",
    "check_elpa_available",
    "check_mkl_available",
    "get_hardware_profile",
    
    # Scheduler & Environment
    "detect",
    "SchedulerHints",
    
    # Builder & Pipeline
    "build_auto",
    "build_mpi",
    "build_hybrid",
    "build_kpoint",
    "BuildResult",
    "run_pipeline",
    "preflight_check",
    "detect_wien2k_version",
    "get_total_ram_gb",
    "get_numa_node_count",
    
    # Performance Counters & Energy
    "get_real_roofline_data",
    "measure_memory_bandwidth",
    "measure_peak_flops",
    "measure_cache_bandwidth",
    "PerfCounterCache",
    "HAS_PERF_COUNTERS",
    "EnergyMeasurement",
    "measure_energy_section",
    "get_rapl_energy_uj",
    "estimate_energy_per_scf_cycle",
    "get_power_cap",
    "HAS_RAPL",
]