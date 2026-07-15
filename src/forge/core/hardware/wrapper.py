"""Module-level cached wrappers and provider management for hardware detection."""

from functools import cache
from typing import Optional, Union

from ...logging_config import get_logger
from .detection import SysFSHardwareInfo
from .types import (
    CacheLevel,
    HardwareNUMANode,
    HardwareProfile,
    InterconnectInfo,
)

logger = get_logger(__name__)

_provider: Optional[SysFSHardwareInfo] = None


def _get_provider() -> SysFSHardwareInfo:
    global _provider
    if _provider is None:
        _provider = SysFSHardwareInfo()
    return _provider


def get_provider() -> SysFSHardwareInfo:
    """Return the current module-level hardware info provider."""
    return _get_provider()


def set_provider(provider: SysFSHardwareInfo) -> None:
    """Inject a custom hardware info provider (e.g., for testing)."""
    global _provider
    _provider = provider


# Clear any cached results when switching provider for tests
def _reset_provider_caches() -> None:
    for fn in [
        get_logical_cores, get_physical_cores, is_hyperthreading_active,
        get_vector_isa_and_width, get_fma_units_per_core, calculate_peak_fp64_gflops,
        get_cpu_governor, get_cpu_frequency_info, get_job_memory_limit_mb,
        get_numa_topology_detailed, get_cache_topology, get_total_mem_kb,
        get_scratch_filesystem_type, get_interconnect_info, get_cpu_architecture,
        get_memory_bandwidth_gb_s, is_containerized, check_elpa_available,
        check_mkl_available, get_hardware_profile, get_numa_node_count,
    ]:
        fn.cache_clear()


@cache
def get_logical_cores() -> int:
    return _get_provider().get_logical_cores()


@cache
def get_physical_cores() -> int:
    return _get_provider().get_physical_cores()


@cache
def is_hyperthreading_active() -> bool:
    return _get_provider().is_hyperthreading_active()


@cache
def get_vector_isa_and_width() -> dict[str, Union[str, int]]:
    return _get_provider().get_vector_isa_and_width()


@cache
def get_fma_units_per_core() -> int:
    return _get_provider().get_fma_units_per_core()


@cache
def calculate_peak_fp64_gflops() -> float:
    return _get_provider().calculate_peak_fp64_gflops()


@cache
def get_cpu_governor(cpu_id: int = 0) -> Optional[str]:
    return _get_provider().get_cpu_governor(cpu_id)


@cache
def get_cpu_frequency_info() -> dict[str, float]:
    return _get_provider().get_cpu_frequency_info()


@cache
def get_job_memory_limit_mb() -> Optional[int]:
    return _get_provider().get_job_memory_limit_mb()


@cache
def get_numa_topology_detailed() -> list[HardwareNUMANode]:
    return _get_provider().get_numa_topology_detailed()


@cache
def get_cache_topology() -> list[CacheLevel]:
    return _get_provider().get_cache_topology()


@cache
def get_total_mem_kb() -> int:
    return _get_provider().get_total_mem_kb()


@cache
def get_scratch_filesystem_type() -> str:
    return _get_provider().get_scratch_filesystem_type()


@cache
def get_interconnect_info() -> InterconnectInfo:
    return _get_provider().get_interconnect_info()


@cache
def get_cpu_architecture() -> str:
    return _get_provider().get_cpu_architecture()


@cache
def get_cpu_generation() -> str:
    return _get_provider().get_cpu_generation()


@cache
def get_system_type() -> str:
    return _get_provider().get_system_type()


@cache
def get_memory_bandwidth_gb_s() -> float:
    return _get_provider().get_memory_bandwidth_gb_s()


@cache
def is_containerized() -> bool:
    return _get_provider().is_containerized()


@cache
def check_elpa_available() -> bool:
    return _get_provider().check_elpa_available()


@cache
def check_mkl_available() -> bool:
    return _get_provider().check_mkl_available()


@cache
def get_hardware_profile() -> HardwareProfile:
    return _get_provider().get_hardware_profile()


@cache
def get_numa_node_count() -> int:
    return _get_provider().get_numa_node_count()


__all__ = [
    "calculate_peak_fp64_gflops",
    "check_elpa_available",
    "check_mkl_available",
    "get_cache_topology",
    "get_cpu_architecture",
    "get_cpu_frequency_info",
    "get_cpu_generation",
    "get_cpu_governor",
    "get_fma_units_per_core",
    "get_hardware_profile",
    "get_interconnect_info",
    "get_job_memory_limit_mb",
    "get_logical_cores",
    "get_memory_bandwidth_gb_s",
    "get_numa_node_count",
    "get_numa_topology_detailed",
    "get_physical_cores",
    "get_provider",
    "get_scratch_filesystem_type",
    "get_system_type",
    "get_total_mem_kb",
    "get_vector_isa_and_width",
    "is_containerized",
    "is_hyperthreading_active",
    "set_provider",
]
