"""
GPU-Aware DFT Execution Support for WIEN2k, VASP, and Quantum ESPRESSO.
Provides GPU topology detection, usage recommendations, machine-file generation,
and mixed-precision configuration for accelerated DFT diagonalization.

Key Features:
- Multi-vendor GPU detection (NVIDIA via nvidia-smi, AMD via rocm-smi)
- Memory-aware threshold: only offload when GPU memory > nmat^2 * 16 bytes
- CUDA_VISIBLE_DEVICES round-robin assignment for hybrid MPI+GPU runs
- WIEN2k/VASP/QE specific machine-file and input-flag generation
- Mixed-precision recommendations based on Yu et al. (2021) ELPA-GPU benchmarks
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.topology import GPUInfo, Topology
from ..logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# GPU Detection
# =============================================================================

def detect_gpu() -> list[GPUInfo]:
    """
    Detect all GPUs in the system using vendor-specific tools.

    Priority: nvidia-smi (NVIDIA) -> rocm-smi (AMD) -> sysfs fallback.

    Returns:
        List of GPUInfo dataclass instances with name, memory, and topology data.
    """
    gpus: list[GPUInfo] = []

    nvidia_gpus = _detect_nvidia_gpus()
    if nvidia_gpus:
        return nvidia_gpus

    amd_gpus = _detect_amd_gpus()
    if amd_gpus:
        return amd_gpus

    fallback = _detect_sysfs_gpus()
    if fallback:
        return fallback

    return gpus


def _detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,uuid,pci.bus_id,index",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line_num, line in enumerate(result.stdout.strip().split("\n")):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            name = parts[0]
            memory_str = parts[1].replace("MiB", "").strip()
            memory_mb = int(memory_str) if memory_str.isdigit() else 0
            compute_cap = parts[2] if len(parts) > 2 else ""
            uuid = parts[3] if len(parts) > 3 else ""
            pci_bus = parts[4] if len(parts) > 4 else ""
            numa_affinity = _get_gpu_numa_affinity(line_num)

            gpus.append(GPUInfo(
                name=name,
                memory_mb=memory_mb,
                compute_capability=compute_cap,
                uuid=uuid,
                pci_bus=pci_bus,
                numa_affinity=numa_affinity,
            ))

        return gpus
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        logger.debug("nvidia-smi not available")
        return []


def _detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs using rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line_num, line in enumerate(result.stdout.strip().split("\n")):
            if not line.strip() or line.startswith("GPU"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue

            name = parts[1] if len(parts) > 1 else f"AMD GPU {line_num}"
            memory_mb = 0
            for part in parts:
                match = re.search(r"(\d+)\s*MB", part, re.IGNORECASE)
                if match:
                    memory_mb = int(match.group(1))
                    break

            numa_affinity = _get_gpu_numa_affinity(line_num)

            gpus.append(GPUInfo(
                name=name,
                memory_mb=memory_mb,
                compute_capability="",
                uuid=f"amd-{line_num}",
                pci_bus="",
                numa_affinity=numa_affinity,
            ))

        return gpus
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        logger.debug("rocm-smi not available")
        return []


def _detect_sysfs_gpus() -> list[GPUInfo]:
    """Detect GPUs via sysfs as a last-resort fallback."""
    gpus = []
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return gpus

    for card in sorted(drm_path.glob("card*")):
        if not card.is_dir():
            continue
        vendor_path = card / "device" / "vendor"
        if not vendor_path.exists():
            continue
        try:
            vendor = vendor_path.read_text().strip()
            if vendor not in ("0x10de", "0x1002"):
                continue
        except Exception:
            continue

        try:
            device_path = card / "device" / "device"
            device_id = device_path.read_text().strip() if device_path.exists() else ""
        except Exception:
            device_id = ""

        numa_affinity = _get_gpu_numa_affinity(len(gpus))

        gpus.append(GPUInfo(
            name=f"GPU-{device_id}" if device_id else f"GPU-{len(gpus)}",
            memory_mb=0,
            compute_capability="",
            uuid=str(card.name),
            pci_bus="",
            numa_affinity=numa_affinity,
        ))

    return gpus


def _get_gpu_numa_affinity(gpu_index: int) -> int:
    """Determine NUMA node affinity for a GPU from sysfs or pci topology."""
    pci_paths = [
        Path(f"/sys/class/drm/card{gpu_index}/device/numa_node"),
        Path("/sys/bus/pci/devices/*/numa_node"),
    ]

    for pattern in pci_paths:
        if "*" in str(pattern):
            matches = sorted(Path("/sys/bus/pci/devices").glob("*/numa_node"))
            for match in matches:
                try:
                    affinity = int(match.read_text().strip())
                    if affinity >= 0:
                        return affinity
                except Exception:
                    continue
        else:
            try:
                if pattern.exists():
                    affinity = int(pattern.read_text().strip())
                    if affinity >= 0:
                        return affinity
            except Exception:
                continue

    return -1


# =============================================================================
# GPU Interconnect Detection (NVLink / Infinity Fabric)
# =============================================================================

def detect_nvlink_active() -> bool:
    """
    Detect active NVLink connections via nvidia-smi.

    NVLink provides 300-900 GB/s GPU-to-GPU bandwidth (NVIDIA CUDA docs,
    NVLink Bridge Specification). Without NVLink, GPU communication falls
    back to PCIe (16-32 GB/s), a 10-50x reduction.

    Returns:
        True if NVLink is active on any GPU pair.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--capabilities"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "active" in output or "enabled" in output:
                nvlink_count = output.count("active")
                logger.info(f"NVLink detected: {nvlink_count} active links")
                return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "nv" in output and any(t in output for t in ["12", "18", "24"]):
                logger.info("NVLink detected via topology matrix")
                return True
    except Exception:
        pass

    return False


def detect_infinity_fabric_active() -> bool:
    """
    Detect AMD Infinity Fabric links between GPUs.

    Infinity Fabric is AMD's equivalent of NVLink, providing high-bandwidth
    GPU-to-GPU interconnects (up to 200 GB/s on MI250X). Detection uses
    rocm-smi or sysfs dxcore topology queries.

    Returns:
        True if Infinity Fabric links are active.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtopology"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "xgm" in output or "infinity_fabric" in output:
                logger.info("Infinity Fabric (xGMI) detected via rocm-smi")
                return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["rocm-smi", "--showlinkinfo"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C"},
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            if "active" in output and "link" in output:
                logger.info("Infinity Fabric links detected via rocm-smi")
                return True
    except Exception:
        pass

    return False


def get_gpu_interconnect_info() -> dict[str, Any]:
    """
    Detect GPU interconnect type and bandwidth.

    Combines NVLink (NVIDIA) and Infinity Fabric (AMD) detection to provide
    interconnect metadata for GPU-aware resource allocation decisions.

    Returns:
        dict with keys: nvlink_active, infinity_fabric_active,
        interconnect_type, bandwidth_estimate_gb_s.
    """
    nvlink = detect_nvlink_active()
    inf_fabric = detect_infinity_fabric_active()

    if nvlink:
        ic_type = "nvlink"
        bandwidth = 600.0  # NVLink 3.0: ~600 GB/s aggregate
    elif inf_fabric:
        ic_type = "infinity_fabric"
        bandwidth = 200.0  # Infinity Fabric: ~200 GB/s (MI250X)
    else:
        ic_type = "pcie"
        bandwidth = 32.0   # PCIe 4.0 x16

    return {
        "nvlink_active": nvlink,
        "infinity_fabric_active": inf_fabric,
        "interconnect_type": ic_type,
        "bandwidth_estimate_gb_s": bandwidth,
    }


# =============================================================================
# GPU Usage Recommendations
# =============================================================================

def get_gpu_recommendation(
    topo: Topology,
    nmat: int,
    nkpt: int,
    mode: str,
) -> dict[str, Any]:
    """
    Determine whether and how to use GPUs for a given DFT calculation.

    Decision rules:
    - nmat > 5000 AND GPU memory > nmat^2 * 16 bytes -> use GPU for diagonalization
    - nmat < 2000 -> GPU overhead hurts, stay on CPU
    - Hybrid mode: 1 GPU per MPI rank, round-robin assignment
    - GPU interconnect (NVLink/Infinity Fabric) enables multi-GPU scaling;
      PCIe-only GPUs are limited to 2-4 effective devices due to bandwidth

    Args:
        topo: Hardware topology with GPU information.
        nmat: Matrix dimension.
        nkpt: Number of k-points.
        mode: Parallelization mode ('mpi', 'hybrid', 'kpoint').

    Returns:
        Dictionary with keys: use_gpu, gpu_count, cuda_visible, gpu_per_mpi_rank,
        reason, recommended_library, interconnect_info.
    """
    gpu_topology = topo.gpu_topology
    if gpu_topology is None or not gpu_topology.gpus:
        return {
            "use_gpu": False,
            "gpu_count": 0,
            "cuda_visible": "",
            "gpu_per_mpi_rank": 0,
            "reason": "No GPUs detected in topology.",
            "recommended_library": "",
            "interconnect_info": get_gpu_interconnect_info(),
        }

    gpus = gpu_topology.gpus
    gpu_count = len(gpus)
    total_gpu_mem_mb = sum(g.memory_mb for g in gpus)
    ic_info = get_gpu_interconnect_info()

    matrix_mem_bytes = (nmat ** 2) * 16
    matrix_mem_mb = matrix_mem_bytes / (1024 * 1024)

    if nmat < 2000:
        return {
            "use_gpu": False,
            "gpu_count": gpu_count,
            "cuda_visible": "",
            "gpu_per_mpi_rank": 0,
            "reason": (
                f"Small matrix (nmat={nmat}): GPU kernel launch and data transfer "
                f"overhead exceeds benefit. Stay on CPU."
            ),
            "recommended_library": "",
            "interconnect_info": ic_info,
        }

    # NVLink/Infinity Fabric scaling check (NVIDIA CUDA docs, GPU Direct RDMA)
    has_highspeed = ic_info["nvlink_active"] or ic_info["infinity_fabric_active"]
    effective_gpu_limit = gpu_count if has_highspeed else min(gpu_count, 4)

    if nmat > 5000 and total_gpu_mem_mb > matrix_mem_mb:
        gpu_per_rank = 1 if mode == "hybrid" else max(1, effective_gpu_limit // max(1, topo.total_cores))
        cuda_visible = ",".join(str(i) for i in range(min(effective_gpu_limit, 8)))

        is_nvidia = any("nvidia" in g.name.lower() or g.compute_capability for g in gpus)
        library = "cuSOLVER" if is_nvidia else "MAGMA"

        ic_note = ""
        if not has_highspeed and gpu_count > 4:
            ic_note = (
                f" Limited to {effective_gpu_limit}/{gpu_count} GPUs: "
                f"PCIe-only interconnect ({ic_info['bandwidth_estimate_gb_s']:.0f} GB/s). "
                f"NVLink/Infinity Fabric not detected."
            )
        elif has_highspeed:
            ic_note = (
                f" {ic_info['interconnect_type']} interconnect "
                f"({ic_info['bandwidth_estimate_gb_s']:.0f} GB/s) enables "
                f"full {gpu_count}-GPU scaling."
            )

        return {
            "use_gpu": True,
            "gpu_count": gpu_count,
            "cuda_visible": f"CUDA_VISIBLE_DEVICES={cuda_visible}",
            "gpu_per_mpi_rank": gpu_per_rank,
            "reason": (
                f"Large matrix (nmat={nmat}, req {matrix_mem_mb:.0f} MB) fits in "
                f"GPU memory ({total_gpu_mem_mb} MB available). Recommend {library}.{ic_note}"
            ),
            "recommended_library": library,
            "interconnect_info": ic_info,
        }

    return {
        "use_gpu": False,
        "gpu_count": gpu_count,
        "cuda_visible": "",
        "gpu_per_mpi_rank": 0,
        "reason": (
            f"GPU available but matrix memory ({matrix_mem_mb:.0f} MB) exceeds "
            f"GPU memory ({total_gpu_mem_mb} MB) or nmat={nmat} < 5000 threshold."
        ),
        "recommended_library": "",
        "interconnect_info": ic_info,
    }


# =============================================================================
# GPU-Aware Machine File Generation
# =============================================================================

def generate_gpu_machines(
    topo: Topology,
    suggestion: dict[str, Any],
) -> str:
    """
    Generate GPU-aware parallel configuration for the target DFT code.

    Supports WIEN2k (.machines with 'gpu:' prefix), VASP (NCORE/KPAR),
    and Quantum ESPRESSO (pw.x GPU flags).

    Args:
        topo: Hardware topology.
        suggestion: Resource allocation suggestion including GPU recommendations.

    Returns:
        Configuration string ready to be written to the code-specific config file.
    """
    gpu_rec = suggestion.get("gpu_recommendation", {})
    if not gpu_rec.get("use_gpu"):
        return ""

    backend = suggestion.get("backend", "wien2k").lower()
    nodes = list(topo.nodes)
    cores_per_node = list(topo.cores_per_node)
    gpu_count = gpu_rec.get("gpu_count", 0)

    if backend == "wien2k":
        return _generate_wien2k_gpu_machines(nodes, cores_per_node, gpu_rec)
    elif backend in ("vasp", "vasp_gpu"):
        return _generate_vasp_gpu_input(gpu_count, suggestion)
    elif backend in ("quantum_espresso", "qe"):
        return _generate_qe_gpu_input(gpu_count, suggestion)
    else:
        logger.warning(f"GPU support not implemented for backend: {backend}")
        return ""


def _generate_wien2k_gpu_machines(
    nodes: list[str],
    cores_per_node: list[int],
    gpu_rec: dict[str, Any],
) -> str:
    """Generate .machines content with gpu: prefix for WIEN2k GPU runs."""
    lines = [
        "# WIEN2k GPU-Aware .machines",
        f"# GPUs available: {gpu_rec.get('gpu_count', 0)}",
        f"# CUDA_VISIBLE_DEVICES setting: {gpu_rec.get('cuda_visible', '')}",
        f"# GPU per MPI rank: {gpu_rec.get('gpu_per_mpi_rank', 1)}",
        "",
    ]

    gpu_per_rank = gpu_rec.get("gpu_per_mpi_rank", 1)
    gpu_index = 0

    for node, cores in zip(nodes, cores_per_node):
        if gpu_per_rank > 0 and gpu_index < gpu_rec.get("gpu_count", 0):
            lines.append(f"lapw1: {node}: {cores} gpu: {node}: {gpu_per_rank}")
            gpu_index += gpu_per_rank
        else:
            lines.append(f"lapw1: {node}: {cores}")

    lines.append("granularity: 1")
    lines.append("omp_global: 1")
    return "\n".join(lines)


def _generate_vasp_gpu_input(
    gpu_count: int,
    suggestion: dict[str, Any],
) -> str:
    """Generate VASP INCAR GPU parameters (NCORE, KPAR)."""
    total_cores = suggestion.get("recommended_total_cores", 1)
    nkpt = suggestion.get("nkpt", 1)

    ncore = max(1, total_cores // max(1, gpu_count))
    kpar = min(gpu_count, nkpt)
    if kpar <= 0:
        kpar = 1

    lines = [
        "# VASP GPU Configuration",
        f"NCORE = {ncore}",
        f"KPAR = {kpar}",
        f"# GPU count: {gpu_count}",
        "# Ensure VASP is compiled with -DGPU (NVIDIA) or -DOPENACC_GPU (AMD)",
    ]
    return "\n".join(lines)


def _generate_qe_gpu_input(
    gpu_count: int,
    suggestion: dict[str, Any],
) -> str:
    """Generate Quantum ESPRESSO GPU input flags."""
    lines = [
        "# Quantum ESPRESSO GPU Configuration",
        "# Add to pw.x or ph.x input:",
        f"# -ndiag {max(1, gpu_count)}",
        f"# -npool {max(1, gpu_count)}",
        "# Or set in input file:",
        "&CONTROL",
        "  use_gpu = .true.",
        "/",
    ]
    return "\n".join(lines)


# =============================================================================
# Mixed-Precision Configuration
# =============================================================================

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision DFT execution."""
    use_mixed: bool
    fp64_ops: list[str]
    fp32_ops: list[str]
    speedup_estimate: float


def get_mixed_precision_recommendation(
    backend: str,
    nmat: int,
) -> MixedPrecisionConfig:
    """
    Recommend mixed-precision strategy based on backend and matrix size.

    Research basis: mixed-precision benchmarks show that exchange-correlation
    integration tolerates FP32 while diagonalization requires FP64 for
    numerical stability. Speedup estimates from Yu et al. (2021),
    Comput. Phys. Commun. 262, 107808 (ELPA-GPU benchmarks).

    Args:
        backend: DFT code name ('wien2k', 'vasp', 'quantum_espresso').
        nmat: Matrix dimension.

    Returns:
        MixedPrecisionConfig with per-phase precision assignments.
    """
    backend_lower = backend.lower()

    if backend_lower == "wien2k":
        return MixedPrecisionConfig(
            use_mixed=nmat > 2000,
            fp64_ops=["lapw1_diag", "lapwso_diag", "lapw2"],
            fp32_ops=["lapw0_xc", "lapw1_setup", "mixer"],
            speedup_estimate=1.3,
        )

    if backend_lower in ("vasp", "vasp_gpu"):
        speedup = 1.5 if "gpu" in backend_lower else 1.2
        return MixedPrecisionConfig(
            use_mixed=nmat > 1000,
            fp64_ops=["diagonalization", "charge_density_mixing"],
            fp32_ops=["xc_potential", "fft_grid", "nonlocal_projectors"],
            speedup_estimate=speedup,
        )

    if backend_lower in ("quantum_espresso", "qe"):
        return MixedPrecisionConfig(
            use_mixed=nmat > 1500,
            fp64_ops=["diagonalization", "charge_density"],
            fp32_ops=["xc_functional", "fft", "structure_factor"],
            speedup_estimate=1.2,
        )

    return MixedPrecisionConfig(
        use_mixed=False,
        fp64_ops=[],
        fp32_ops=[],
        speedup_estimate=1.0,
    )
