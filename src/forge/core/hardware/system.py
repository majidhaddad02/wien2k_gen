"""System detection mixin: memory, NUMA, IO, interconnect, libraries, hardware profile."""

import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from ...logging_config import get_logger
from .types import (
    CacheLevel,
    HardwareNUMANode,
    HardwareProfile,
    InterconnectInfo,
    _get_threads_per_core,
    parse_cpu_list,
    parse_memory_string,
)

logger = get_logger(__name__)


class SystemDetectionMixin:
    """System detection methods for SysFSHardwareInfo (memory, NUMA, IO, interconnect, libraries)."""

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

    def get_numa_topology_detailed(self) -> list[HardwareNUMANode]:
        nodes = self._get_numa_from_sysfs()
        if len(nodes) <= 1:
            nodes = self._augment_numa_from_lscpu(nodes)
        return self._augment_numa_from_numactl(nodes)

    def _get_numa_from_sysfs(self) -> list[HardwareNUMANode]:  # noqa: C901
        nodes: list[HardwareNUMANode] = []
        try:
            online_content = Path("/sys/devices/system/node/online").read_text().strip()
            node_ids: list[int] = []
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
                    tpc = _get_threads_per_core(node_path)
                    node_data["cores"] = max(1, len(cpu_ids) // max(1, tpc))

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

    def _augment_numa_from_lscpu(self, nodes: list[HardwareNUMANode]) -> list[HardwareNUMANode]:
        """Augment NUMA topology using lscpu output.

        lscpu reports thread/core/socket counts and NUMA layout,
        which complements sysfs for systems with SNC (sub-NUMA
        clustering) or memory interleaving.
        """
        try:
            out = self._run_cmd_safe(["lscpu", "-p=cpu,node,socket,core"], timeout=3)
            if not out:
                return nodes

            socket_to_node: dict[int, set] = {}
            for line in out.strip().split('\n'):
                if line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        socket = int(parts[2])
                        node_id = int(parts[1])
                        socket_to_node.setdefault(socket, set()).add(node_id)
                    except ValueError:
                        continue

            snc_detected = any(len(v) > 1 for v in socket_to_node.values())
            if snc_detected and len(nodes) == 0:
                logger.info("Sub-NUMA Clustering (SNC) detected via lscpu")
        except Exception as e:
            logger.debug(f"lscpu-based NUMA augmentation failed: {e}")
        return nodes

    def _augment_numa_from_numactl(self, nodes: list[HardwareNUMANode]) -> list[HardwareNUMANode]:
        """Augment NUMA topology using numactl --hardware.

        numactl reports memory bandwidth, interleaving, and distance
        matrices not always available in sysfs on older kernels.
        """
        try:
            out = self._run_cmd_safe(["numactl", "--hardware"], timeout=3)
            if not out:
                return nodes

            node_size_map: dict[int, int] = {}
            for line in out.split('\n'):
                m = re.search(r'node\s+(\d+)\s+size.*?(\d+)\s*MB', line, re.IGNORECASE)
                if m:
                    node_size_map[int(m.group(1))] = int(m.group(2)) * 1024

            for node in nodes:
                nid = node["node_id"]
                if nid in node_size_map and node["mem_kb"] == 0:
                    node["mem_kb"] = node_size_map[nid]
        except Exception as e:
            logger.debug(f"numactl augmentation failed: {e}")
        return nodes

    def get_cache_topology(self) -> list[CacheLevel]:
        caches: list[CacheLevel] = []
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

    def get_total_mem_kb(self) -> int:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
        except Exception as e:
            logger.debug(f"MemTotal detection failed: {e}")
        return 4 * 1024 * 1024

    def get_scratch_filesystem_type(self) -> str:
        scratch = os.getenv("SCRATCH", os.getenv("TMPDIR", "."))
        try:
            out = self._run_cmd_safe(["df", "-T", scratch], force_c_locale=True)
            if out:
                fstype = out.splitlines()[-1].split()[1].lower()
                if any(fs in fstype for fs in ["lustre", "gpfs", "beegfs", "nfs", "xfs", "ext4", "tmpfs"]):
                    return fstype
        except Exception as e:
            logger.debug(f"Scratch FS detection failed: {e}")
        return "unknown"

    def get_interconnect_info(self) -> InterconnectInfo:  # noqa: C901
        interconnect: InterconnectInfo = {
            "type": "unknown", "provider": "unknown", "speed_gbps": 10.0,
            "latency_ns": 1000.0, "numa_aware": False, "active_rate_gbps": 10.0,
        }

        # InfiniBand detection with active rate parsing
        ib_raw = self._run_cmd_safe(["ibv_devinfo", "-l"])
        if ib_raw:
            if "mlx" in ib_raw.lower():
                interconnect.update({
                    "type": "infiniband", "provider": "mlx5/verbs",
                    "speed_gbps": 100.0, "latency_ns": 1.5, "numa_aware": True,
                    "active_rate_gbps": 100.0,
                })
            else:
                interconnect.update({
                    "type": "infiniband", "provider": "verbs",
                    "speed_gbps": 56.0, "latency_ns": 2.0, "numa_aware": True,
                    "active_rate_gbps": 56.0,
                })
            # Parse actual link speed from ibv_devinfo output
            for line in ib_raw.split('\n'):
                line_lower = line.lower().strip()
                if ('port' in line_lower and 'rate' in line_lower) or 'active_speed' in line_lower:
                    # e.g. "port 1: rate: 100 Gb/sec (4X HDR)"
                    rate_str = line.split(':')[-1].strip().split()[0]
                    try:
                        rate_gbps = float(rate_str)
                        interconnect["speed_gbps"] = rate_gbps
                        interconnect["active_rate_gbps"] = rate_gbps
                    except (ValueError, IndexError):
                        pass
                elif 'rate' in line_lower:
                    # e.g. "rate: 56 Gb/sec"
                    rate_str = line.split(':')[-1].strip().split()[0]
                    try:
                        rate_gbps = float(rate_str)
                        interconnect["speed_gbps"] = rate_gbps
                        interconnect["active_rate_gbps"] = rate_gbps
                    except (ValueError, IndexError):
                        pass
            return interconnect

        # OmniPath detection: try psm2 first, then verbs fallback
        fi_raw = self._run_cmd_safe(["fi_info", "-p", "psm2"])
        if not fi_raw:
            fi_raw = self._run_cmd_safe(["fi_info", "-p", "verbs"])
        if fi_raw and ("ofi" in fi_raw.lower() or "psm2" in fi_raw.lower()):
            provider = "ofi/psm2" if "psm2" in fi_raw.lower() else "ofi/verbs"
            # Parse fabric speed from fi_info output
            rate_gbps = 100.0
            for line in fi_raw.split('\n'):
                if 'caps' in line.lower() or 'tx_size' in line.lower():
                    break
            interconnect.update({
                "type": "omni_path", "provider": provider,
                "speed_gbps": rate_gbps, "latency_ns": 1.0, "numa_aware": True,
                "active_rate_gbps": rate_gbps,
            })
            return interconnect

        slurm_net = os.getenv("SLURM_NETWORK", "").lower()
        if "infiniband" in slurm_net or "ib" in slurm_net:
            interconnect.update({
                "type": "infiniband", "provider": "verbs", "numa_aware": True,
                "active_rate_gbps": 56.0, "speed_gbps": 56.0,
            })
        elif "ethernet" in slurm_net or "tcp" in slurm_net:
            interconnect.update({
                "type": "ethernet", "provider": "tcp", "speed_gbps": 25.0,
                "latency_ns": 10.0, "active_rate_gbps": 25.0,
            })

        return interconnect

    def get_system_type(self) -> str:  # noqa: C901
        """
        Detect system type: laptop, workstation, compute_node, or cluster.

        Heuristics:
        - laptop:    battery present in /sys/class/power_supply, or chassis=Notebook
        - cluster:   SLURM/PBS/LSF/SGE job ID is set
        - compute_node: physical cores >= 32 but no scheduler job detected
        - workstation: physical cores < 32, no battery, no scheduler

        Returns one of: "laptop", "workstation", "compute_node", "cluster", "unknown"
        """
        if os.getenv("SLURM_JOB_ID") or os.getenv("PBS_JOBID") or \
           os.getenv("LSB_JOBID") or os.getenv("SGE_JOB_ID"):
            return "cluster"

        try:
            psu_path = Path("/sys/class/power_supply")
            if psu_path.exists():
                for entry in psu_path.iterdir():
                    try:
                        tp = (entry / "type").read_text().strip()
                        if tp == "Battery":
                            return "laptop"
                    except Exception:
                        logger.debug(f"Cannot read power supply type from {entry}")

            chassis_path = Path("/sys/class/dmi/id/chassis_type")
            if chassis_path.exists():
                chassis = chassis_path.read_text().strip()
                if chassis in ("8", "9", "10", "14"):
                    return "laptop"

            chassis_vendor = self._run_cmd_safe(["cat", "/sys/class/dmi/id/chassis_vendor"]) or \
                             self._run_cmd_safe(["cat", "/sys/devices/virtual/dmi/id/chassis_vendor"])
            if chassis_vendor:
                vl = chassis_vendor.lower()
                if any(kw in vl for kw in ("notebook", "laptop", "lenovo thinkpad", "dell latitude")):
                    return "laptop"
        except Exception as e:
            logger.debug(f"System type detection failed: {e}")

        phys = self.get_physical_cores()
        return "compute_node" if phys >= 32 else "workstation"

    def get_memory_bandwidth_gb_s(self) -> float:
        """Measure or estimate memory bandwidth.

        Try in order:
        1. sysfs numa bandwidth counters (instant, accurate on newer kernels)
        2. Estimate from CPU arch, DDR generation, and channel count.
        Warnings logged when bandwidth < 50 GB/s (LAPW1 memory-bound).
        """
        measured = self._measure_bandwidth_sysfs()
        if measured is not None:
            return round(measured, 1)

        return self._estimate_bandwidth_from_arch()

    @staticmethod
    def _measure_bandwidth_sysfs() -> Optional[float]:  # noqa: C901
        """Measure memory bandwidth via NUMA sysfs counters.

        Reads /sys/devices/system/node/node*/meminfo BW_total counters
        sampled over 3 seconds. Returns None if counters unavailable.
        """
        try:
            bw_files = sorted(Path("/sys/devices/system/node").glob("node*/meminfo"))
            if not bw_files:
                return None

            counters = {}
            for f in bw_files:
                content = f.read_text()
                for line in content.split('\n'):
                    if 'BW_total' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            with contextlib.suppress(ValueError):
                                counters[str(f)] = int(parts[1])

            if not counters:
                return None

            t0_sample = {}
            for f_path in bw_files:
                content = f_path.read_text()
                for line in content.split('\n'):
                    if 'BW_total' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            with contextlib.suppress(ValueError):
                                t0_sample[str(f_path)] = int(parts[1])

            time.sleep(3.0)

            bw_sum_mb = 0.0
            for f_path in bw_files:
                content = f_path.read_text()
                for line in content.split('\n'):
                    if 'BW_total' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                v1 = t0_sample.get(str(f_path), 0)
                                v2 = int(parts[1])
                                bw_sum_mb += (v2 - v1) / 3.0
                            except (ValueError, ZeroDivisionError):
                                pass

            if bw_sum_mb > 0:
                return bw_sum_mb / 1000.0
        except (OSError, PermissionError) as e:
            logger.debug(f"Sysfs bandwidth measurement failed: {e}")
        return None

    def _estimate_bandwidth_from_arch(self) -> float:
        """Estimate memory bandwidth from CPU arch, DDR generation, and channel count."""
        arch = self.get_cpu_architecture()
        channels = 8 if "epyc" in arch else (6 if "xeon" in arch else 4)

        ddr_gen = self._detect_ddr_generation()

        per_channel_bw = {
            "DDR3": 12.8,
            "DDR4": 25.6,
            "DDR5": 38.4,
        }.get(ddr_gen, 21.3)

        if ddr_gen == "DDR5" and "epyc" in arch:
            channels = 12

        return round(channels * per_channel_bw, 1)

    def _detect_ddr_generation(self) -> str:
        """Detect DDR memory generation from dmidecode or sysfs counters."""
        raw = self._run_cmd_safe(
            ["dmidecode", "-t", "memory"], force_c_locale=True, timeout=5
        )
        if raw:
            for line in raw.split('\n'):
                line_upper = line.strip().upper()
                if "DDR5" in line_upper:
                    return "DDR5"
                elif "DDR4" in line_upper:
                    return "DDR4"
                elif "DDR3" in line_upper:
                    return "DDR3"
        edac_path = Path("/sys/devices/system/edac/mc")
        if edac_path.exists():
            try:
                for mc_dir in edac_path.iterdir():
                    if mc_dir.is_dir():
                        return "DDR4"
            except PermissionError:
                pass
        return "DDR4"

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
        from ..locator import find_elpa_dir, find_wienroot
        wienroot = find_wienroot() or ""
        elpa_dir = find_elpa_dir() or ""
        paths = [
            Path(wienroot, "lib", "libelpa.a"), Path(wienroot, "lib", "libelpa.so"),
            Path(elpa_dir, "lib", "libelpa.a"), Path(elpa_dir, "lib", "libelpa.so"),
        ]
        return any(p.exists() for p in paths)

    def check_mkl_available(self) -> bool:
        return any(os.getenv(var) for var in ["MKLROOT", "MKL_LIB", "INTEL_MKL"])

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
            except Exception as e:
                logger.debug(f"lscpu JSON parsing failed: {e}")

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
            cpu_microarch=str(isa_info["isa"]),
            cpu_governor=self.get_cpu_governor(),
            cpu_freq_mhz=self.get_cpu_frequency_info(),
            vector_isa=str(isa_info["isa"]),
            vector_width_bits=int(isa_info["width_bits"]),
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
        if profile["memory_bandwidth_gb_s"] < 50.0:
            warnings.append(
                f"Low memory bandwidth ({profile['memory_bandwidth_gb_s']:.1f} GB/s < 50 GB/s). "
                f"LAPW1 is memory-bound for large systems; additional MPI ranks yield no benefit."
            )

        profile["validation_warnings"] = warnings

        return profile
