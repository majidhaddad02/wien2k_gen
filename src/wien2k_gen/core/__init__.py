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
    GPUInfo
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
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    # Topology & Hardware
    "Topology",
    "TopologyValidationError",
    "NUMANode",
    "NodeSpec",
    "GPUInfo",
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
]