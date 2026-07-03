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
"""

import os
import re
import json
import subprocess
import logging
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

def _run_cmd_safe(cmd: List[str], timeout: int = 5, force_c_locale: bool = False) -> Optional[str]:
    """Safely execute a shell command with timeout and stderr suppression."""
    env = os.environ.copy()
    if force_c_locale:
        env['LC_ALL'] = 'C'  # Ensure English output for parsing
    try:
        return subprocess.check_output(
            cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL, env=env
        ).strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        logger.debug(f"Command failed or not found: {' '.join(cmd)}")
        return None

# =============================================================================
# Core CPU & Core Count Detection
# =============================================================================

@lru_cache(maxsize=None)
def get_logical_cores() -> int:
    """Retrieve total logical CPU cores from os.cpu_count()."""
    return os.cpu_count() or 1

@lru_cache(maxsize=None)
def get_physical_cores() -> int:
    """
    Determine physical core count with multi-method fallbacks.
    Priority: lscpu JSON -> lscpu CSV -> /proc/cpuinfo -> os.cpu_count()
    """
    # Method 1: Parse lscpu JSON output (most reliable)
    raw = _run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
    if raw:
        try:
            data = json.loads(raw)
            fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
            
            sockets = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
            cores_per_socket = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
            return sockets * cores_per_socket
        except (json.JSONDecodeError, ValueError, StopIteration):
            pass

    # Method 2: Parse lscpu --parse CSV format
    raw = _run_cmd_safe(["lscpu", "--parse=CPU,SOCKET,CORE"], force_c_locale=True)
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

    # Method 3: /proc/cpuinfo fallback
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

    return get_logical_cores()

@lru_cache(maxsize=None)
def is_hyperthreading_active() -> bool:
    """Check if SMT/Hyper-Threading is enabled."""
    return get_logical_cores() > get_physical_cores()

# =============================================================================
# ISA, Vector Width & FMA Detection
# =============================================================================

@lru_cache(maxsize=None)
def _parse_lscpu_flags() -> List[str]:
    """Extract CPU flags from lscpu."""
    raw = _run_cmd_safe(["lscpu"], force_c_locale=True)
    if not raw:
        return []
    for line in raw.split('\n'):
        if line.strip().startswith("Flags:") or line.strip().startswith("CPU op-mode"):
            if ":" in line:
                return line.split(':', 1)[1].strip().split()
    return []

@lru_cache(maxsize=None)
def get_vector_isa_and_width() -> Dict[str, Union[str, int]]:
    """Detect highest available vector ISA and register width."""
    flags = _parse_lscpu_flags()
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

@lru_cache(maxsize=None)
def get_fma_units_per_core() -> int:
    """Estimate FMA units per core based on microarchitecture."""
    vector_info = get_vector_isa_and_width()
    isa = vector_info["isa"]
    
    if "avx512" in isa:
        return 2
    if "avx" in isa or "avx2" in isa:
        return 2
    if "neon" in isa or "sve" in isa:
        return 1
    return 0

@lru_cache(maxsize=None)
def calculate_peak_fp64_gflops() -> float:
    """
    Calculate theoretical Peak FP64 GFLOPS for Roofline modeling.
    Uses max(base, boost) frequency to reflect true theoretical peak performance.
    """
    raw = _run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
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
            
    freq_info = get_cpu_frequency_info()
    base_freq = freq_info.get("base", 2000.0)
    max_freq = freq_info.get("max", 0.0)
    
    # Use the maximum of base and boost (max) frequency for theoretical peak
    effective_freq = max(base_freq, max_freq)
    if effective_freq == 0.0:
        effective_freq = base_freq

    fma = get_fma_units_per_core()
    vec_width = get_vector_isa_and_width()["width_bits"]

    # FP64 ops = fma_units * (vec_width/64) * 2 (multiply + add)
    ops_per_core_per_cycle = fma * (vec_width / 64.0) * 2.0
    peak = sockets * cores_per_socket * effective_freq * 1e6 * ops_per_core_per_cycle / 1e9

    return round(peak, 2)

# =============================================================================
# Frequency, Governor & Memory Limits
# =============================================================================

@lru_cache(maxsize=None)
def get_cpu_governor(cpu_id: int = 0) -> Optional[str]:
    """Retrieve current CPU frequency scaling governor."""
    path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor"
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return None

@lru_cache(maxsize=None)
def get_cpu_frequency_info() -> Dict[str, float]:
    """Extract min, max, base, and current CPU frequencies in MHz."""
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

@lru_cache(maxsize=None)
def get_job_memory_limit_mb() -> Optional[int]:
    """Parse scheduler environment variables for memory limits."""
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

# =============================================================================
# NUMA Topology & Cache Hierarchy
# =============================================================================

@lru_cache(maxsize=None)
def get_numa_topology_detailed() -> List[HardwareNUMANode]:
    """Build detailed NUMA topology from sysfs."""
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
        phys = get_physical_cores()
        nodes = [HardwareNUMANode(
            node_id=0, cpus=f"0-{phys-1}", cpu_ids=list(range(phys)),
            mem_kb=get_total_mem_kb(), cores=phys, distance={0: 10}
        )]
    return nodes

@lru_cache(maxsize=None)
def get_cache_topology() -> List[CacheLevel]:
    """Detect CPU cache hierarchy via sysfs using robust shared_cpu_list."""
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
            
            # Use shared_cpu_list instead of bitmap parsing for robustness
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

# =============================================================================
# Memory, Scratch & Interconnect Detection
# =============================================================================

@lru_cache(maxsize=None)
def get_total_mem_kb() -> int:
    """Read total system memory from /proc/meminfo."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 4 * 1024 * 1024  # 4GB safe fallback

@lru_cache(maxsize=None)
def get_scratch_filesystem_type() -> str:
    """Identify filesystem type of $SCRATCH, $TMPDIR, or CWD."""
    scratch = os.getenv("SCRATCH", os.getenv("TMPDIR", "."))
    try:
        out = _run_cmd_safe(["df", "-T", scratch], force_c_locale=True)
        if out:
            fstype = out.splitlines()[-1].split()[1].lower()
            if any(fs in fstype for fs in ["lustre", "gpfs", "beegfs", "nfs", "xfs", "ext4", "tmpfs"]):
                return fstype
    except Exception:
        pass
    return "unknown"

@lru_cache(maxsize=None)
def get_interconnect_info() -> InterconnectInfo:
    """Detect network interconnect type and estimated performance."""
    interconnect: InterconnectInfo = {
        "type": "unknown", "provider": "unknown", "speed_gbps": 10.0, 
        "latency_ns": 1000.0, "numa_aware": False
    }
    
    ib_raw = _run_cmd_safe(["ibv_devinfo", "-l"])
    if ib_raw and "mlx" in ib_raw.lower():
        interconnect.update({
            "type": "infiniband", "provider": "mlx5/verbs",
            "speed_gbps": 100.0, "latency_ns": 1.5, "numa_aware": True
        })
        return interconnect
        
    fi_raw = _run_cmd_safe(["fi_info", "-p", "verbs"])
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

@lru_cache(maxsize=None)
def get_cpu_architecture() -> str:
    """Detect CPU vendor and microarchitecture class."""
    raw = _run_cmd_safe(["lscpu"], force_c_locale=True)
    if raw:
        if "AMD" in raw:
            return "epyc" if "EPYC" in raw else "amd_ryzen"
        elif "Intel" in raw:
            return "xeon" if "Xeon" in raw else "intel_consumer"
        elif "aarch64" in raw or "ARM" in raw:
            return "arm_neoverse" if "Neoverse" in raw else "arm"
    return "unknown"

@lru_cache(maxsize=None)
def get_memory_bandwidth_gb_s() -> float:
    """Estimate memory bandwidth based on architecture."""
    arch = get_cpu_architecture()
    # EPYC has 8 (or 12 in Genoa) channels, Xeon typically 6 or 8
    channels = 8 if "epyc" in arch else (6 if "xeon" in arch else 4)
    base_bw = 50.0 if "epyc" in arch or "neoverse" in arch else 28.0
    return round(channels * base_bw, 1)

# =============================================================================
# Environment & Library Checks
# =============================================================================

@lru_cache(maxsize=None)
def is_containerized() -> bool:
    """Detect execution inside Docker, Singularity/Apptainer, or Podman."""
    indicators = [
        Path("/.dockerenv").exists(),
        Path("/run/.containerenv").exists(),
        Path("/singularity.d").exists(),
        bool(os.getenv("SINGULARITY_CONTAINER")),
        bool(os.getenv("CONTAINER_ID")),
    ]
    return any(indicators)

@lru_cache(maxsize=None)
def check_elpa_available() -> bool:
    """Verify ELPA library availability."""
    wienroot = os.environ.get("WIENROOT", "/opt/codes/WIEN2k")
    paths = [Path(wienroot, "lib", "libelpa.a"), Path(wienroot, "lib", "libelpa.so")]
    return any(p.exists() for p in paths)

@lru_cache(maxsize=None)
def check_mkl_available() -> bool:
    """Check for Intel MKL via environment variables."""
    return any(os.getenv(var) for var in ["MKLROOT", "MKL_LIB", "INTEL_MKL"])

# =============================================================================
# Main Hardware Profile Aggregator
# =============================================================================

@lru_cache(maxsize=None)
def get_hardware_profile() -> HardwareProfile:
    """Assemble comprehensive hardware profile for WIEN2k parallel configuration."""
    sockets_raw = 1
    cores_per_socket_raw = get_physical_cores()
    
    lscpu_json = _run_cmd_safe(["lscpu", "-J"], force_c_locale=True)
    if lscpu_json:
        try:
            data = json.loads(lscpu_json)
            fields = {x["field"].lower().strip(': '): x["data"] for x in data.get("lscpu", [])}
            sockets_raw = int(next((v for k, v in fields.items() if "socket(s)" in k and "per" not in k), 1))
            cores_per_socket_raw = int(next((v for k, v in fields.items() if "core(s) per socket" in k), 1))
        except Exception:
            pass
            
    threads_per_core = get_logical_cores() // (sockets_raw * cores_per_socket_raw) if (sockets_raw * cores_per_socket_raw) > 0 else 1

    mem_limit = get_job_memory_limit_mb()
    mem_total_kb = get_total_mem_kb()

    isa_info = get_vector_isa_and_width()
    peak_gflops = calculate_peak_fp64_gflops()

    profile = HardwareProfile(
        physical_cores=get_physical_cores(),
        logical_cores=get_logical_cores(),
        hyperthreading=is_hyperthreading_active(),
        sockets=sockets_raw,
        cores_per_socket=cores_per_socket_raw,
        threads_per_core=threads_per_core,
        numa_nodes=get_numa_topology_detailed(),
        cache_topology=get_cache_topology(),
        memory_total_gb=mem_total_kb / (1024.0 * 1024.0),
        memory_limit_gb=mem_limit / 1024.0 if mem_limit else None,
        memory_bandwidth_gb_s=get_memory_bandwidth_gb_s(),
        memory_channels=8 if "epyc" in get_cpu_architecture() else 6,
        memory_speed_mts=4800 if "epyc" in get_cpu_architecture() else 3200,
        cpu_arch=get_cpu_architecture(),
        cpu_microarch=isa_info["isa"],
        cpu_governor=get_cpu_governor(),
        cpu_freq_mhz=get_cpu_frequency_info(),
        vector_isa=isa_info["isa"],
        vector_width_bits=isa_info["width_bits"],
        fma_units_per_core=get_fma_units_per_core(),
        peak_fp64_gflops=peak_gflops,
        interconnect=get_interconnect_info(),
        scratch_fs=get_scratch_filesystem_type(),
        elpa_available=check_elpa_available(),
        mkl_available=check_mkl_available(),
        containerized=is_containerized(),
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


def get_numa_node_count() -> int:
    """Return the number of NUMA nodes detected on the system."""
    topology = get_numa_topology_detailed()
    return len(topology)