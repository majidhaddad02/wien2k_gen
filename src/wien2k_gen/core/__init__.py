"""
Core Package Initialization for WIEN2k Parallel Execution Engine.
Exposes topology detection, hardware profiling, scheduler integration,
configuration building, pipeline orchestration, and workflow provenance.
Designed for exascale HPC environments with robust error handling,
type safety, and explicit public API declaration.
"""

# =============================================================================
# Topology & Hardware Detection
# =============================================================================
# =============================================================================
# Configuration Builder & Pipeline
# =============================================================================
from .builder import BuildResult, build_auto, build_hybrid, build_kpoint, build_mpi

# =============================================================================
# Case File Parser
# =============================================================================
from .case_parser import (
    CaseData,
    CaseFileParser,
    LDAUData,
    parse_case_directory,
)
from .constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV, RYDBERG_TO_EV
from .energy import (
    HAS_RAPL,
    EnergyMeasurement,
    estimate_energy_per_scf_cycle,
    get_power_cap,
    get_rapl_energy_uj,
    measure_energy_section,
)
from .hardware import (
    check_elpa_available,
    check_mkl_available,
    get_cache_topology,
    get_cpu_architecture,
    get_cpu_frequency_info,
    get_cpu_generation,
    get_cpu_governor,
    get_hardware_profile,
    get_interconnect_info,
    get_job_memory_limit_mb,
    get_logical_cores,
    get_memory_bandwidth_gb_s,
    get_numa_topology_detailed,
    get_physical_cores,
    get_scratch_filesystem_type,
    get_system_type,
    get_total_mem_kb,
    is_containerized,
    is_hyperthreading_active,
)

# =============================================================================
# Performance Counters & Energy
# =============================================================================
from .perf_counters import (
    HAS_PERF_COUNTERS,
    PerfCounterCache,
    get_real_roofline_data,
    measure_cache_bandwidth,
    measure_memory_bandwidth,
    measure_peak_flops,
)
from .pipeline import (
    detect_wien2k_version,
    get_numa_node_count,
    get_total_ram_gb,
    preflight_check,
    run_pipeline,
)

# =============================================================================
# Scheduler & Environment Detection
# =============================================================================
from .scheduler import (
    SchedulerHints,
    _detect_scheduler,
    auto_detect_memory,
    detect,
)
from .topology import (
    GPUInfo,
    GPUTopology,
    NodeSpec,
    NUMANode,
    Topology,
    TopologyValidationError,
    detect_gpu_topology,
)

# =============================================================================
# Workflow Provenance
# =============================================================================
from .workflow import (
    NodeStatus,
    WorkflowDAG,
    WorkflowNode,
    WorkflowStore,
    create_band_structure_workflow,
    create_convergence_workflow,
    create_wien2k_workflow,
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
    "get_cpu_generation",
    "get_system_type",
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
    "_detect_scheduler",
    "auto_detect_memory",
    
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
    "RYDBERG_TO_EV",
    "HARTREE_TO_EV",
    "BOHR_TO_ANGSTROM",

    # Workflow Provenance
    "NodeStatus",
    "WorkflowNode",
    "WorkflowDAG",
    "WorkflowStore",
    "create_wien2k_workflow",
    "create_convergence_workflow",
    "create_band_structure_workflow",

    # Case File Parser
    "CaseData",
    "LDAUData",
    "CaseFileParser",
    "parse_case_directory",
]