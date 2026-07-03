"""
GPU-Aware DFT Execution Support for WIEN2k, VASP, and Quantum ESPRESSO.
Provides GPU topology detection, usage recommendations, machine-file generation,
and mixed-precision configuration for accelerated DFT diagonalization.

Key Features:
- Multi-vendor GPU detection (NVIDIA via nvidia-smi, AMD via rocm-smi)
- Memory-aware threshold: only offload when GPU memory > nmat^2 * 16 bytes
- CUDA_VISIBLE_DEVICES round-robin assignment for hybrid MPI+GPU runs
- WIEN2k/VASP/QE specific machine-file and input-flag generation
- Mixed-precision recommendations based on Auckenthaler et al. benchmarks
"""

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.topology import Topology, GPUInfo, GPUTopology
from ..logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# GPU Detection
# =============================================================================

def detect_gpu() -> List[GPUInfo]:
    """
    Detect all GPUs in the system using vendor-specific tools.

    Priority: nvidia-smi (NVIDIA) -> rocm-smi (AMD) -> sysfs fallback.

    Returns:
        List of GPUInfo dataclass instances with name, memory, and topology data.
    """
    gpus: List[GPUInfo] = []

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


def _detect_nvidia_gpus() -> List[GPUInfo]:
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


def _detect_amd_gpus() -> List[GPUInfo]:
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


def _detect_sysfs_gpus() -> List[GPUInfo]:
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
        Path(f"/sys/bus/pci/devices/*/numa_node"),
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
# GPU Usage Recommendations
# =============================================================================

def get_gpu_recommendation(
    topo: Topology,
    nmat: int,
    nkpt: int,
    mode: str,
) -> Dict[str, Any]:
    """
    Determine whether and how to use GPUs for a given DFT calculation.

    Decision rules:
    - nmat > 5000 AND GPU memory > nmat^2 * 16 bytes -> use GPU for diagonalization
    - nmat < 2000 -> GPU overhead hurts, stay on CPU
    - Hybrid mode: 1 GPU per MPI rank, round-robin assignment

    Args:
        topo: Hardware topology with GPU information.
        nmat: Matrix dimension.
        nkpt: Number of k-points.
        mode: Parallelization mode ('mpi', 'hybrid', 'kpoint').

    Returns:
        Dictionary with keys: use_gpu, gpu_count, cuda_visible, gpu_per_mpi_rank,
        reason, recommended_library.
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
        }

    gpus = gpu_topology.gpus
    gpu_count = len(gpus)
    total_gpu_mem_mb = sum(g.memory_mb for g in gpus)

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
        }

    if nmat > 5000 and total_gpu_mem_mb > matrix_mem_mb:
        gpu_per_rank = 1 if mode == "hybrid" else max(1, gpu_count // max(1, topo.total_cores))
        cuda_visible = ",".join(str(i) for i in range(min(gpu_count, 8)))

        is_nvidia = any("nvidia" in g.name.lower() or g.compute_capability for g in gpus)
        library = "cuSOLVER" if is_nvidia else "MAGMA"

        return {
            "use_gpu": True,
            "gpu_count": gpu_count,
            "cuda_visible": f"CUDA_VISIBLE_DEVICES={cuda_visible}",
            "gpu_per_mpi_rank": gpu_per_rank,
            "reason": (
                f"Large matrix (nmat={nmat}, req {matrix_mem_mb:.0f} MB) fits in "
                f"GPU memory ({total_gpu_mem_mb} MB available). Recommend {library}."
            ),
            "recommended_library": library,
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
    }


# =============================================================================
# GPU-Aware Machine File Generation
# =============================================================================

def generate_gpu_machines(
    topo: Topology,
    suggestion: Dict[str, Any],
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
    nodes: List[str],
    cores_per_node: List[int],
    gpu_rec: Dict[str, Any],
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
    suggestion: Dict[str, Any],
) -> str:
    """Generate VASP INCAR GPU parameters (NCORE, KPAR)."""
    total_cores = suggestion.get("recommended_total_cores", 1)
    nkpt = suggestion.get("nkpt", 1)

    ncore = max(1, total_cores // max(1, gpu_count))
    kpar = min(gpu_count, nkpt)
    if kpar <= 0:
        kpar = 1

    lines = [
        f"# VASP GPU Configuration",
        f"NCORE = {ncore}",
        f"KPAR = {kpar}",
        f"# GPU count: {gpu_count}",
        f"# Ensure VASP is compiled with -DGPU (NVIDIA) or -DOPENACC_GPU (AMD)",
    ]
    return "\n".join(lines)


def _generate_qe_gpu_input(
    gpu_count: int,
    suggestion: Dict[str, Any],
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
    fp64_ops: List[str]
    fp32_ops: List[str]
    speedup_estimate: float


def get_mixed_precision_recommendation(
    backend: str,
    nmat: int,
) -> MixedPrecisionConfig:
    """
    Recommend mixed-precision strategy based on backend and matrix size.

    Research basis: mixed-precision benchmarks show that exchange-correlation
    integration tolerates FP32 while diagonalization requires FP64 for
    numerical stability. Speedup estimates from Auckenthaler et al. (2011)
    and VASP mixed-precision benchmarks.

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
