"""
Hardware Detection Module with Full NUMA, Cache, ISA, and Interconnect Awareness.
Designed for exascale-ready WIEN2k parallel optimization and Roofline model integration.

Final Key Improvements Applied:
- Fixed all string literal corruption, syntax typos, and variable naming errors.
- Replaced broken hex-bitmap cache parsing with robust sysfs 'shared_cpu_list' parsing.
- Implemented robust memory unit parsing (K/M/G/T) for scheduler limits.
- Added 'LC_ALL=C' enforcement for system commands to ensure consistent parsing across locales.
- Replaced custom TTL cache with standard `@lru_cache` for cleaner, thread-safe memoization.
- Enhanced Peak FP64 GFLOPS calculation to utilize max(base, boost) frequency for accurate Roofline modeling.
- Comprehensive English documentation and type hinting.
- Introduced HardwareInfoProvider ABC and SysFSHardwareInfo for testability via dependency injection.
"""

import os
import re
import json
import subprocess
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Union, TypedDict, Any
from functools import lru_cache

from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Structured Return Types for Type Safety & Static Analysis
# =============================================================================

class HardwareNUMANode(TypedDict):
    node_id: int
    cpus: str
    cpu_ids: List[int]
    mem_kb: int
    cores: int
    distance: Dict[int, int]

class CacheLevel(TypedDict):
    level: int
    size_kb: int
    type: str
    cores_sharing: List[int]

class InterconnectInfo(TypedDict):
    type: str          # e.g., infiniband, omni_path, ethernet, tcp
    provider: str      # e.g., mlx5, verbs, tcp, ucx
    speed_gbps: float
    latency_ns: float  # Estimated based on type
    numa_aware: bool

class HardwareProfile(TypedDict):
    physical_cores: int
    logical_cores: int
    hyperthreading: bool
    sockets: int
    cores_per_socket: int
    threads_per_core: int
    numa_nodes: List[HardwareNUMANode]
    cache_topology: List[CacheLevel]
    memory_total_gb: float
    memory_limit_gb: Optional[float]
    memory_bandwidth_gb_s: float
    memory_channels: int
    memory_speed_mts: int
    cpu_arch: str
    cpu_microarch: str
    cpu_governor: Optional[str]
    cpu_freq_mhz: Dict[str, float]
    vector_isa: str
    vector_width_bits: int
    fma_units_per_core: int
    peak_fp64_gflops: float
    interconnect: InterconnectInfo
    scratch_fs: str
    elpa_available: bool
    mkl_available: bool
    containerized: bool
    validation_warnings: List[str]

# =============================================================================
# Helper Functions
# =============================================================================

def parse_cpu_list(cpu_list_str: str) -> List[int]:
    """
    Parse CPU list string (e.g., '0-5,8,10-12') into a list of integers.
    Standard format used in sysfs (cpulist, shared_cpu_list).
    """
    ids = []
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
    def get_vector_isa_and_width(self) -> Dict[str, Union[str, int]]:
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
    def get_cpu_frequency_info(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def get_job_memory_limit_mb(self) -> Optional[int]:
        ...

    @abstractmethod
    def get_numa_topology_detailed(self) -> List[HardwareNUMANode]:
        ...

    @abstractmethod
    def get_cache_topology(self) -> List[CacheLevel]:
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

class SysFSHardwareInfo(HardwareInfoProvider):
    """
    Concrete implementation using sysfs, /proc, /sys and subprocess calls.
    All system I/O is performed through this class, making the module unit-testable
    by injecting mock providers.
    """

    @staticmethod
    def _run_cmd_safe(cmd: List[str], timeout: int = 5, force_c_locale: bool = False) -> Optional[str]:
        """Safely execute a shell command with timeout and stderr suppression."""
        env = os.environ.copy()
        if force_c_locale:
            env['LC_ALL'] = 'C'
        try:
            return subprocess.check_output(
                cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL, env=env
            ).strip()
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            logger.debug(f"Command failed or not found: {' '.join(cmd)}")
            return None

    @staticmethod
    def _parse_lscpu_flags() -> List[str]:
        """Extract CPU flags from lscpu."""
        raw = SysFSHardwareInfo._run_cmd_safe(["lscpu"], force_c_locale=True)
        if not raw:
            return []
        for line in raw.split('\n'):
            if line.strip().startswith("Flags:") or line.strip().startswith("CPU op-mode"):
                if ":" in line:
                    return line.split(':', 1)[1].strip().split()
        return []

    # --- Core CPU & Core Count Detection ---

    def get_logical_cores(self) -> int:
        return os.cpu_count() or 1

    def get_physical_cores(self) -> int:
        raw = self._run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
        if raw:
            try:
                data = json.loads(raw)
                fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
                sockets = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
                cores_per_socket = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
                return sockets * cores_per_socket
            except (json.JSONDecodeError, ValueError, StopIteration):
                pass

        raw = self._run_cmd_safe(["lscpu", "--parse=CPU,SOCKET,CORE"], force_c_locale=True)
        if raw:
            try:
                lines = [l for l in raw.split('\n') if l and not l.startswith('#')]
                unique_physical = set()
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3 and parts[1] != '-' and parts[2] != '-':
                        unique_physical.add((parts[1], parts[2]))
                if unique_physical:
                    return len(unique_physical)
            except Exception:
                pass

        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
            pairs = set()
            current_phys, current_core = None, None
            for line in content.split('\n'):
                if line.startswith('physical id'):
                    current_phys = line.split(':')[1].strip()
                elif line.startswith('core id'):
                    current_core = line.split(':')[1].strip()
                if current_phys is not None and current_core is not None:
                    pairs.add((current_phys, current_core))
            if pairs:
                return len(pairs)

            core_ids = set(re.findall(r'^core id\s*:\s*(\d+)', content, re.MULTILINE))
            if core_ids:
                return len(core_ids)
        except Exception as e:
            logger.debug(f"/proc/cpuinfo parsing failed: {e}")

        return self.get_logical_cores()

    def is_hyperthreading_active(self) -> bool:
        return self.get_logical_cores() > self.get_physical_cores()

    # --- ISA, Vector Width & FMA Detection ---

    def get_vector_isa_and_width(self) -> Dict[str, Union[str, int]]:
        flags = self._parse_lscpu_flags()
        flag_set = set(flags)

        if any(f.startswith("avx512") for f in flag_set):
            return {"isa": "avx512", "width_bits": 512}
        if "avx2" in flag_set or "avx" in flag_set:
            return {"isa": "avx2", "width_bits": 256}
        if "sve" in flag_set or "sve2" in flag_set:
            return {"isa": "sve", "width_bits": 128}
        if "neon" in flag_set:
            return {"isa": "neon", "width_bits": 128}
        if "sse4_2" in flag_set or "sse4_1" in flag_set:
            return {"isa": "sse4", "width_bits": 128}

        return {"isa": "scalar", "width_bits": 64}

    def get_fma_units_per_core(self) -> int:
        vector_info = self.get_vector_isa_and_width()
        isa = vector_info["isa"]

        if "avx512" in isa:
            return 2
        if "avx" in isa or "avx2" in isa:
            return 2
        if "neon" in isa or "sve" in isa:
            return 1
        return 0

    def calculate_peak_fp64_gflops(self) -> float:
        raw = self._run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
        sockets = 1
        cores_per_socket = 1

        if raw:
            try:
                data = json.loads(raw)
                fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
                sockets = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
                cores_per_socket = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
            except Exception:
                pass

        freq_info = self.get_cpu_frequency_info()
        base_freq = freq_info.get("base", 2000.0)
        max_freq = freq_info.get("max", 0.0)

        effective_freq = max(base_freq, max_freq)
        if effective_freq == 0.0:
            effective_freq = base_freq

        fma = self.get_fma_units_per_core()
        vec_width = self.get_vector_isa_and_width()["width_bits"]

        ops_per_core_per_cycle = fma * (vec_width / 64.0) * 2.0
        peak = sockets * cores_per_socket * effective_freq * 1e6 * ops_per_core_per_cycle / 1e9

        return round(peak, 2)

    # --- Frequency, Governor & Memory Limits ---

    def get_cpu_governor(self, cpu_id: int = 0) -> Optional[str]:
        path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor"
        try:
            return Path(path).read_text().strip()
        except FileNotFoundError:
            return None

    def get_cpu_frequency_info(self) -> Dict[str, float]:
        info = {"min": 0.0, "max": 0.0, "current": 0.0, "base": 0.0}
        base_path = Path("/sys/devices/system/cpu/cpu0/cpufreq")

        if base_path.exists():
            try:
                info["min"] = float((base_path / "scaling_min_freq").read_text().strip()) / 1000
                info["max"] = float((base_path / "scaling_max_freq").read_text().strip()) / 1000
                info["current"] = float((base_path / "scaling_cur_freq").read_text().strip()) / 1000
            except Exception as e:
                logger.debug(f"Frequency reading failed: {e}")

        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "cpu MHz" in line:
                        info["base"] = float(line.split(':')[1].strip())
                        break
        except Exception:
            info["base"] = info["max"] if info["max"] > 0 else 2000.0

        return info

    def get_job_memory_limit_mb(self) -> Optional[int]:
        sched_vars = [
            ("SLURM_MEM_PER_NODE", False),
            ("SLURM_MEM_PER_CPU", True),
            ("PBS_VMEM", False),
            ("LSB_MEMLIMIT", False),
        ]

        for env, per_cpu in sched_vars:
            val = os.getenv(env)
            if val:
                mb_val = parse_memory_string(val)
                if mb_val is not None:
                    if per_cpu:
                        cpus = int(os.getenv("SLURM_CPUS_ON_NODE", "1"))
                        mb_val *= cpus
                    return mb_val
        return None

    # --- NUMA Topology & Cache Hierarchy ---

    def get_numa_topology_detailed(self) -> List[HardwareNUMANode]:
        nodes = []
        try:
            online_content = Path("/sys/devices/system/node/online").read_text().strip()
            node_ids = []
            for part in online_content.split(","):
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    node_ids.extend(range(start, end + 1))
                else:
                    node_ids.append(int(part))

            for nid in node_ids:
                node_path = Path(f"/sys/devices/system/node/node{nid}")
                if not node_path.exists():
                    continue

                node_data: HardwareNUMANode = {
                    "node_id": nid, "cpus": "", "cpu_ids": [], "mem_kb": 0,
                    "cores": 0, "distance": {}
                }

                cpulist_path = node_path / "cpulist"
                if cpulist_path.exists():
                    cpu_range = cpulist_path.read_text().strip()
                    node_data["cpus"] = cpu_range
                    cpu_ids = parse_cpu_list(cpu_range)
                    node_data["cpu_ids"] = cpu_ids
                    node_data["cores"] = len(cpu_ids)

                meminfo_path = node_path / "meminfo"
                if meminfo_path.exists():
                    for line in meminfo_path.read_text().splitlines():
                        if "MemTotal:" in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                node_data["mem_kb"] = int(parts[-2])
                            break

                dist_path = node_path / "distance"
                if dist_path.exists():
                    distances = list(map(int, dist_path.read_text().strip().split()))
                    node_data["distance"] = {i: d for i, d in enumerate(distances) if i in node_ids}

                nodes.append(node_data)

        except Exception as e:
            logger.warning(f"NUMA topology detection failed, using fallback: {e}")

        if not nodes:
            phys = self.get_physical_cores()
            nodes = [HardwareNUMANode(
                node_id=0, cpus=f"0-{phys-1}", cpu_ids=list(range(phys)),
                mem_kb=self.get_total_mem_kb(), cores=phys, distance={0: 10}
            )]
        return nodes

    def get_cache_topology(self) -> List[CacheLevel]:
        caches = []
        base = Path("/sys/devices/system/cpu/cpu0/cache")
        if not base.exists():
            return caches

        try:
            for idx_dir in sorted(base.glob("index*")):
                if not idx_dir.is_dir():
                    continue

                level = int((idx_dir / "level").read_text().strip())
                size_str = (idx_dir / "size").read_text().strip().rstrip('K')
                cache_type = (idx_dir / "type").read_text().strip()

                shared_cores = []
                list_path = idx_dir / "shared_cpu_list"
                if list_path.exists():
                    shared_cores = parse_cpu_list(list_path.read_text().strip())

                caches.append(CacheLevel(
                    level=level, size_kb=int(size_str), type=cache_type,
                    cores_sharing=shared_cores
                ))
        except Exception as e:
            logger.debug(f"Cache topology detection failed: {e}")
        return caches

    # --- Memory, Scratch & Interconnect Detection ---

    def get_total_mem_kb(self) -> int:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
        except Exception:
            pass
        return 4 * 1024 * 1024

    def get_scratch_filesystem_type(self) -> str:
        scratch = os.getenv("SCRATCH", os.getenv("TMPDIR", "."))
        try:
            out = self._run_cmd_safe(["df", "-T", scratch], force_c_locale=True)
            if out:
                fstype = out.splitlines()[-1].split()[1].lower()
                if any(fs in fstype for fs in ["lustre", "gpfs", "beegfs", "nfs", "xfs", "ext4", "tmpfs"]):
                    return fstype
        except Exception:
            pass
        return "unknown"

    def get_interconnect_info(self) -> InterconnectInfo:
        interconnect: InterconnectInfo = {
            "type": "unknown", "provider": "unknown", "speed_gbps": 10.0,
            "latency_ns": 1000.0, "numa_aware": False
        }

        ib_raw = self._run_cmd_safe(["ibv_devinfo", "-l"])
        if ib_raw and "mlx" in ib_raw.lower():
            interconnect.update({
                "type": "infiniband", "provider": "mlx5/verbs",
                "speed_gbps": 100.0, "latency_ns": 1.5, "numa_aware": True
            })
            return interconnect

        fi_raw = self._run_cmd_safe(["fi_info", "-p", "verbs"])
        if fi_raw and "ofi" in fi_raw.lower():
            interconnect.update({
                "type": "omni_path", "provider": "ofi/psm2",
                "speed_gbps": 100.0, "latency_ns": 1.0, "numa_aware": True
            })
            return interconnect

        slurm_net = os.getenv("SLURM_NETWORK", "").lower()
        if "infiniband" in slurm_net or "ib" in slurm_net:
            interconnect.update({"type": "infiniband", "provider": "verbs", "numa_aware": True})
        elif "ethernet" in slurm_net or "tcp" in slurm_net:
            interconnect.update({"type": "ethernet", "provider": "tcp", "speed_gbps": 25.0, "latency_ns": 10.0})

        return interconnect

    def get_cpu_architecture(self) -> str:
        raw = self._run_cmd_safe(["lscpu"], force_c_locale=True)
        if raw:
            if "AMD" in raw:
                return "epyc" if "EPYC" in raw else "amd_ryzen"
            elif "Intel" in raw:
                return "xeon" if "Xeon" in raw else "intel_consumer"
            elif "aarch64" in raw or "ARM" in raw:
                return "arm_neoverse" if "Neoverse" in raw else "arm"
        return "unknown"

    def get_memory_bandwidth_gb_s(self) -> float:
        arch = self.get_cpu_architecture()
        channels = 8 if "epyc" in arch else (6 if "xeon" in arch else 4)
        base_bw = 50.0 if "epyc" in arch or "neoverse" in arch else 28.0
        return round(channels * base_bw, 1)

    # --- Environment & Library Checks ---

    def is_containerized(self) -> bool:
        indicators = [
            Path("/.dockerenv").exists(),
            Path("/run/.containerenv").exists(),
            Path("/singularity.d").exists(),
            bool(os.getenv("SINGULARITY_CONTAINER")),
            bool(os.getenv("CONTAINER_ID")),
        ]
        return any(indicators)

    def check_elpa_available(self) -> bool:
        wienroot = os.environ.get("WIENROOT", "/opt/codes/WIEN2k")
        paths = [Path(wienroot, "lib", "libelpa.a"), Path(wienroot, "lib", "libelpa.so")]
        return any(p.exists() for p in paths)

    def check_mkl_available(self) -> bool:
        return any(os.getenv(var) for var in ["MKLROOT", "MKL_LIB", "INTEL_MKL"])

    # --- Main Hardware Profile Aggregator ---

    def get_hardware_profile(self) -> HardwareProfile:
        sockets_raw = 1
        cores_per_socket_raw = self.get_physical_cores()

        lscpu_json = self._run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
        if lscpu_json:
            try:
                data = json.loads(lscpu_json)
                fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
                sockets_raw = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
                cores_per_socket_raw = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
            except Exception:
                pass

        threads_per_core = self.get_logical_cores() // (sockets_raw * cores_per_socket_raw) if (sockets_raw * cores_per_socket_raw) > 0 else 1

        mem_limit = self.get_job_memory_limit_mb()
        mem_total_kb = self.get_total_mem_kb()

        isa_info = self.get_vector_isa_and_width()
        peak_gflops = self.calculate_peak_fp64_gflops()

        cpu_arch = self.get_cpu_architecture()

        profile = HardwareProfile(
            physical_cores=self.get_physical_cores(),
            logical_cores=self.get_logical_cores(),
            hyperthreading=self.is_hyperthreading_active(),
            sockets=sockets_raw,
            cores_per_socket=cores_per_socket_raw,
            threads_per_core=threads_per_core,
            numa_nodes=self.get_numa_topology_detailed(),
            cache_topology=self.get_cache_topology(),
            memory_total_gb=mem_total_kb / (1024.0 * 1024.0),
            memory_limit_gb=mem_limit / 1024.0 if mem_limit else None,
            memory_bandwidth_gb_s=self.get_memory_bandwidth_gb_s(),
            memory_channels=8 if "epyc" in cpu_arch else 6,
            memory_speed_mts=4800 if "epyc" in cpu_arch else 3200,
            cpu_arch=cpu_arch,
            cpu_microarch=isa_info["isa"],
            cpu_governor=self.get_cpu_governor(),
            cpu_freq_mhz=self.get_cpu_frequency_info(),
            vector_isa=isa_info["isa"],
            vector_width_bits=isa_info["width_bits"],
            fma_units_per_core=self.get_fma_units_per_core(),
            peak_fp64_gflops=peak_gflops,
            interconnect=self.get_interconnect_info(),
            scratch_fs=self.get_scratch_filesystem_type(),
            elpa_available=self.check_elpa_available(),
            mkl_available=self.check_mkl_available(),
            containerized=self.is_containerized(),
            validation_warnings=[]
        )

        warnings = []
        if profile["hyperthreading"]:
            warnings.append("Hyper-threading detected. Set SLURM_HINT=nomultithread or OMP_PLACES=cores to avoid oversubscription.")
        if profile["memory_limit_gb"] and profile["memory_total_gb"] > 256:
            warnings.append("Memory limit detected. Ensure WIEN2k charge density fits within allocated RAM per node.")
        if profile["scratch_fs"] in ["nfs", "unknown"]:
            warnings.append("Non-local scratch filesystem detected. Performance may suffer during lapw0/lapw1 I/O heavy phases.")

        profile["validation_warnings"] = warnings

        return profile


# =============================================================================
# Module-Level Singleton & Backward-Compatibility Wrappers
# =============================================================================

_provider: HardwareInfoProvider = SysFSHardwareInfo()


def get_provider() -> HardwareInfoProvider:
    """Return the current module-level hardware info provider."""
    return _provider


def set_provider(provider: HardwareInfoProvider) -> None:
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


@lru_cache(maxsize=None)
def get_logical_cores() -> int:
    return _provider.get_logical_cores()


@lru_cache(maxsize=None)
def get_physical_cores() -> int:
    return _provider.get_physical_cores()


@lru_cache(maxsize=None)
def is_hyperthreading_active() -> bool:
    return _provider.is_hyperthreading_active()


@lru_cache(maxsize=None)
def get_vector_isa_and_width() -> Dict[str, Union[str, int]]:
    return _provider.get_vector_isa_and_width()


@lru_cache(maxsize=None)
def get_fma_units_per_core() -> int:
    return _provider.get_fma_units_per_core()


@lru_cache(maxsize=None)
def calculate_peak_fp64_gflops() -> float:
    return _provider.calculate_peak_fp64_gflops()


@lru_cache(maxsize=None)
def get_cpu_governor(cpu_id: int = 0) -> Optional[str]:
    return _provider.get_cpu_governor(cpu_id)


@lru_cache(maxsize=None)
def get_cpu_frequency_info() -> Dict[str, float]:
    return _provider.get_cpu_frequency_info()


@lru_cache(maxsize=None)
def get_job_memory_limit_mb() -> Optional[int]:
    return _provider.get_job_memory_limit_mb()


@lru_cache(maxsize=None)
def get_numa_topology_detailed() -> List[HardwareNUMANode]:
    return _provider.get_numa_topology_detailed()


@lru_cache(maxsize=None)
def get_cache_topology() -> List[CacheLevel]:
    return _provider.get_cache_topology()


@lru_cache(maxsize=None)
def get_total_mem_kb() -> int:
    return _provider.get_total_mem_kb()


@lru_cache(maxsize=None)
def get_scratch_filesystem_type() -> str:
    return _provider.get_scratch_filesystem_type()


@lru_cache(maxsize=None)
def get_interconnect_info() -> InterconnectInfo:
    return _provider.get_interconnect_info()


@lru_cache(maxsize=None)
def get_cpu_architecture() -> str:
    return _provider.get_cpu_architecture()


@lru_cache(maxsize=None)
def get_memory_bandwidth_gb_s() -> float:
    return _provider.get_memory_bandwidth_gb_s()


@lru_cache(maxsize=None)
def is_containerized() -> bool:
    return _provider.is_containerized()


@lru_cache(maxsize=None)
def check_elpa_available() -> bool:
    return _provider.check_elpa_available()


@lru_cache(maxsize=None)
def check_mkl_available() -> bool:
    return _provider.check_mkl_available()


@lru_cache(maxsize=None)
def get_hardware_profile() -> HardwareProfile:
    return _provider.get_hardware_profile()


@lru_cache(maxsize=None)
def get_numa_node_count() -> int:
    return _provider.get_numa_node_count()


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "HardwareNUMANode",
    "CacheLevel",
    "InterconnectInfo",
    "HardwareProfile",
    "parse_cpu_list",
    "parse_memory_string",
    "HardwareInfoProvider",
    "SysFSHardwareInfo",
    "get_provider",
    "set_provider",
    "get_logical_cores",
    "get_physical_cores",
    "is_hyperthreading_active",
    "get_vector_isa_and_width",
    "get_fma_units_per_core",
    "calculate_peak_fp64_gflops",
    "get_cpu_governor",
    "get_cpu_frequency_info",
    "get_job_memory_limit_mb",
    "get_numa_topology_detailed",
    "get_cache_topology",
    "get_total_mem_kb",
    "get_scratch_filesystem_type",
    "get_interconnect_info",
    "get_cpu_architecture",
    "get_memory_bandwidth_gb_s",
    "is_containerized",
    "check_elpa_available",
    "check_mkl_available",
    "get_hardware_profile",
    "get_numa_node_count",
]
