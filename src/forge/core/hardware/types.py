"""Hardware detection types, TypedDicts, ABC, and utility parsers."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypedDict, Union

from ...logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Structured Return Types for Type Safety & Static Analysis
# =============================================================================

class HardwareNUMANode(TypedDict):
    node_id: int
    cpus: str
    cpu_ids: list[int]
    mem_kb: int
    cores: int
    distance: dict[int, int]

class CacheLevel(TypedDict):
    level: int
    size_kb: int
    type: str
    cores_sharing: list[int]

class InterconnectInfo(TypedDict):
    type: str          # e.g., infiniband, omni_path, ethernet, tcp
    provider: str      # e.g., mlx5, verbs, tcp, ucx
    speed_gbps: float
    latency_ns: float  # Estimated based on type
    numa_aware: bool
    active_rate_gbps: float  # Measured link speed from hardware counters

class HardwareProfile(TypedDict):
    physical_cores: int
    logical_cores: int
    hyperthreading: bool
    sockets: int
    cores_per_socket: int
    threads_per_core: int
    numa_nodes: list[HardwareNUMANode]
    cache_topology: list[CacheLevel]
    memory_total_gb: float
    memory_limit_gb: Optional[float]
    memory_bandwidth_gb_s: float
    memory_channels: int
    memory_speed_mts: int
    cpu_arch: str
    cpu_microarch: str
    cpu_governor: Optional[str]
    cpu_freq_mhz: dict[str, float]
    vector_isa: str
    vector_width_bits: int
    fma_units_per_core: int
    peak_fp64_gflops: float
    interconnect: InterconnectInfo
    scratch_fs: str
    elpa_available: bool
    mkl_available: bool
    containerized: bool
    validation_warnings: list[str]

# =============================================================================
# Helper Functions
# =============================================================================

def parse_cpu_list(cpu_list_str: str) -> list[int]:
    """
    Parse CPU list string (e.g., '0-5,8,10-12') into a list of integers.
    Standard format used in sysfs (cpulist, shared_cpu_list).
    """
    ids: list[int] = []
    if not cpu_list_str or not cpu_list_str.strip():
        return ids
    for part in cpu_list_str.split(','):
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                ids.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                ids.append(int(part.strip()))
            except ValueError:
                continue
    return ids

def parse_memory_string(val: str) -> Optional[int]:
    """
    Parse memory string with units (e.g., '10G', '1024M', '500000K') into Megabytes (MB).
    Handles SLURM, PBS, and LSF formats robustly.
    """
    if not val:
        return None
    val = val.strip().upper()
    if val.endswith('B'):
        val = val[:-1]

    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?)$', val)
    if not match:
        try:
            raw = int(val)
            return raw if raw < 10000000 else raw // (1024*1024)
        except ValueError:
            return None

    num = float(match.group(1))
    unit = match.group(2)

    if unit == 'K':
        return int(num / 1024)
    elif unit == 'M' or unit == '':
        return int(num)
    elif unit == 'G':
        return int(num * 1024)
    elif unit == 'T':
        return int(num * 1024 * 1024)
    return int(num)

# =============================================================================
# Abstract Hardware Information Provider (ABC)
# =============================================================================

class HardwareInfoProvider(ABC):
    """Abstract interface for hardware queries, enabling dependency injection for testing."""

    @abstractmethod
    def get_logical_cores(self) -> int:
        ...

    @abstractmethod
    def get_physical_cores(self) -> int:
        ...

    @abstractmethod
    def is_hyperthreading_active(self) -> bool:
        ...

    @abstractmethod
    def get_vector_isa_and_width(self) -> dict[str, Union[str, int]]:
        ...

    @abstractmethod
    def get_fma_units_per_core(self) -> int:
        ...

    @abstractmethod
    def calculate_peak_fp64_gflops(self) -> float:
        ...

    @abstractmethod
    def get_cpu_governor(self, cpu_id: int = 0) -> Optional[str]:
        ...

    @abstractmethod
    def get_cpu_frequency_info(self) -> dict[str, float]:
        ...

    @abstractmethod
    def get_job_memory_limit_mb(self) -> Optional[int]:
        ...

    @abstractmethod
    def get_numa_topology_detailed(self) -> list[HardwareNUMANode]:
        ...

    @abstractmethod
    def get_cache_topology(self) -> list[CacheLevel]:
        ...

    @abstractmethod
    def get_total_mem_kb(self) -> int:
        ...

    @abstractmethod
    def get_scratch_filesystem_type(self) -> str:
        ...

    @abstractmethod
    def get_interconnect_info(self) -> InterconnectInfo:
        ...

    @abstractmethod
    def get_cpu_architecture(self) -> str:
        ...

    @abstractmethod
    def get_cpu_generation(self) -> str:
        ...

    @abstractmethod
    def get_system_type(self) -> str:
        ...

    @abstractmethod
    def get_memory_bandwidth_gb_s(self) -> float:
        ...

    @abstractmethod
    def is_containerized(self) -> bool:
        ...

    @abstractmethod
    def check_elpa_available(self) -> bool:
        ...

    @abstractmethod
    def check_mkl_available(self) -> bool:
        ...

    @abstractmethod
    def get_hardware_profile(self) -> HardwareProfile:
        ...

    def get_numa_node_count(self) -> int:
        """Return the number of NUMA nodes detected on the system."""
        return len(self.get_numa_topology_detailed())


# =============================================================================
# SysFS-backed Hardware Information Provider
# =============================================================================


def _get_threads_per_core(node_path: Path) -> int:
    """Read threads-per-core (SMT factor) from sysfs topology.

    Walks cpu*/topology/thread_siblings_list under the NUMA node path.
    Returns 1 if topology information is unavailable (treat as no SMT).
    """
    seen_topo: set = set()
    try:
        for cpu_dir in sorted(node_path.glob("cpu[0-9]*")):
            topo = cpu_dir / "topology" / "thread_siblings_list"
            if topo.exists():
                seen_topo.add(topo.read_text().strip())
        return max(1, len(seen_topo))
    except (OSError, ValueError, PermissionError):
        return 1


__all__ = [
    "CacheLevel",
    "HardwareInfoProvider",
    "HardwareNUMANode",
    "HardwareProfile",
    "InterconnectInfo",
    "_get_threads_per_core",
    "parse_cpu_list",
    "parse_memory_string",
]
