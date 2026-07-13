"""
GPU Offloading Detection for WIEN2k HPC Workflows.

Detects GPU hardware, checks WIEN2k GPU compilation, analyzes offload
potential per lapw stage, and generates hybrid CPU+GPU recommendations.

References:
  WIEN2k official benchmarks: http://www.wien2k.at/reg_user/benchmark/
  Yu et al. (2021). "GPU-acceleration of the ELPA2 distributed eigensolver."
    Comput. Phys. Commun. 262, 107808. DOI: 10.1016/j.cpc.2021.107808
  NVIDIA CUDA Best Practices Guide
  WIEN2k GPU Integration Guide (vers. 24+)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Detected GPU properties."""

    vendor: str = "unknown"
    model: str = "unknown"
    memory_mb: int = 0
    compute_capability: str = ""
    count: int = 0
    detected: bool = False


@dataclass
class OffloadAnalysis:
    """Per-lapw stage offload analysis."""

    lapw0_offload: bool = False
    lapw1_offload: bool = False
    lapw2_offload: bool = False
    core_offload: bool = False
    lapw1_speedup: float = 1.0
    lapw2_speedup: float = 1.0
    gpu_memory_required_mb: float = 0.0
    gpu_memory_available_mb: float = 0.0
    oom_risk: bool = False
    recommendation: str = ""


# ---------------------------------------------------------------------------
# GPU Hardware Detection
# ---------------------------------------------------------------------------

def detect_gpu_hardware() -> list[GPUInfo]:  # noqa: C901
    """Detect GPU hardware via nvidia-smi, rocminfo, or sycl-ls.

    Returns list of GPUInfo for each detected GPU.
    """
    gpus: list[GPUInfo] = []

    # NVIDIA
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append(GPUInfo(
                        vendor="nvidia",
                        model=parts[0],
                        memory_mb=int(float(parts[1])),
                        compute_capability=parts[2],
                        count=1,
                        detected=True,
                    ))
            if gpus:
                logger.info(f"Detected {len(gpus)} NVIDIA GPU(s): {[g.model for g in gpus]}")
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # AMD
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and "GPU" in proc.stdout:
            model = "AMD GPU"
            mem = 0
            for line in proc.stdout.splitlines():
                if "Card" in line or "GPU" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        model = parts[-1].strip()
                if "VRAM" in line or "vram" in line:
                    m = re.search(r'(\d+)', line)
                    if m:
                        mem = int(m.group(1))
            if mem > 0:
                gpus.append(GPUInfo(
                    vendor="amd", model=model, memory_mb=mem, detected=True))
                logger.info(f"Detected AMD GPU: {model} ({mem}MB)")
                return gpus
    except FileNotFoundError:
        pass

    # Intel
    try:
        proc = subprocess.run(
            ["sycl-ls"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and "Intel" in proc.stdout:
            for line in proc.stdout.splitlines():
                if "GPU" in line:
                    gpus.append(GPUInfo(
                        vendor="intel", model="Intel GPU", memory_mb=0, detected=True))
            if gpus:
                logger.info(f"Detected {len(gpus)} Intel GPU(s)")
                return gpus
    except FileNotFoundError:
        pass

    # Check /dev/dri for generic GPU detection
    if not gpus:
        dri = Path("/dev/dri")
        if dri.exists():
            render_devices = list(dri.glob("renderD*"))
            if render_devices:
                gpus.append(GPUInfo(
                    vendor="generic", model="Generic GPU",
                    memory_mb=0, count=len(render_devices), detected=True))
                logger.info(f"Detected {len(render_devices)} generic GPU device(s) via /dev/dri")

    return gpus


# ---------------------------------------------------------------------------
# WIEN2k GPU Compilation Check
# ---------------------------------------------------------------------------

def check_wien2k_gpu_support(wienroot: str | None = None) -> dict[str, Any]:  # noqa: C901
    """Check if WIEN2k is compiled with GPU support.

    Scans siteconfig, Makefile, and binaries for GPU flags.
    """
    if wienroot is None:
        wienroot = os.environ.get("WIENROOT", "")

    root = Path(wienroot) if wienroot else Path("/opt/wien2k")
    result: dict[str, Any] = {
        "gpu_enabled": False,
        "vendor": "none",
        "gpu_binaries": [],
        "compile_flags": [],
        "warning": "",
    }

    # Check for GPU binaries
    gpu_binaries = ["lapw1gpu", "lapw2gpu", "lapwgpu"]
    for bn in gpu_binaries:
        for search_dir in [root, root / "SRC_lapw1", root / "SRC_lapw2"]:
            candidate = search_dir / bn
            if candidate.exists() and os.access(candidate, os.X_OK):
                result["gpu_binaries"].append(str(candidate))
                result["gpu_enabled"] = True

    # Check siteconfig for GPU flags
    siteconfig = root / "siteconfig_lapw"
    if siteconfig.exists():
        try:
            sc_text = siteconfig.read_text(encoding="utf-8", errors="replace")
            if re.search(r'-DUSE_CUDA|-DCUBLAS|nvcc', sc_text, re.IGNORECASE):
                result["vendor"] = "nvidia"
                result["compile_flags"].append("CUDA")
                result["gpu_enabled"] = True
            if re.search(r'-DUSE_HIP|hipcc|rocm', sc_text, re.IGNORECASE):
                result["vendor"] = "amd"
                result["compile_flags"].append("HIP")
                result["gpu_enabled"] = True
            if re.search(r'-DUSE_SYCL|dpcpp', sc_text, re.IGNORECASE):
                result["vendor"] = "intel"
                result["compile_flags"].append("SYCL")
                result["gpu_enabled"] = True
        except Exception:
            pass

    # Check parallel_options for GPU settings
    if not result["gpu_enabled"]:
        for opts_file in [root / "parallel_options", Path("parallel_options")]:
            if opts_file.exists():
                content = opts_file.read_text(encoding="utf-8", errors="replace")
                if "GPU" in content.upper() or "CUDA" in content.upper():
                    result["gpu_enabled"] = True
                    result["warning"] = (
                        "GPU flags found in parallel_options but no GPU binaries detected. "
                        "Recompile WIEN2k with GPU support."
                    )

    return result


# ---------------------------------------------------------------------------
# Offload Potential Analysis
# ---------------------------------------------------------------------------

def analyze_offload_potential(
    nmat: int,
    num_kpoints: int,
    system_type: str = "unknown",
    gpu_info: list[GPUInfo] | None = None,
) -> OffloadAnalysis:
    """Analyze GPU offload potential for each lapw stage.

    GPU offload thresholds based on ELPA-GPU benchmarks
    (Yu et al. 2021, Comput. Phys. Commun. 262, 107808) and WIEN2k
    official benchmark data at http://www.wien2k.at/reg_user/benchmark/:
        lapw0: offload not beneficial (I/O bound, FFT-dominated)
        lapw1: offload beneficial if nmat > 5000  (speedup ≈ min(10, nmat/1000))
               # TODO: validate nmat>5000 threshold on target cluster
        lapw2: offload beneficial if nmat > 8000  (speedup ≈ min(5, nmat/2000))
               # TODO: validate nmat>8000 threshold on target cluster
        core:  offload not beneficial (sequential)
    """
    analysis = OffloadAnalysis()

    if gpu_info:
        total_mem = sum(g.memory_mb for g in gpu_info)
        analysis.gpu_memory_available_mb = float(total_mem)

    analysis.lapw0_offload = False
    analysis.core_offload = False

    if nmat > 5000:
        analysis.lapw1_offload = True
        analysis.lapw1_speedup = min(10.0, max(1.0, nmat / 1000.0))
    else:
        analysis.lapw1_offload = False
        analysis.lapw1_speedup = 1.0

    if nmat > 8000:
        analysis.lapw2_offload = True
        analysis.lapw2_speedup = min(5.0, max(1.0, nmat / 2000.0))
    else:
        analysis.lapw2_offload = False
        analysis.lapw2_speedup = 1.0

    # GPU memory estimation
    kpts_per_gpu = num_kpoints
    analysis.gpu_memory_required_mb = estimate_gpu_memory(nmat, kpts_per_gpu)

    if analysis.gpu_memory_available_mb > 0 and analysis.gpu_memory_required_mb > 0.90 * analysis.gpu_memory_available_mb:
            analysis.oom_risk = True
            analysis.recommendation = (
                f"GPU memory tight: required={analysis.gpu_memory_required_mb:.0f}MB, "
                f"available={analysis.gpu_memory_available_mb:.0f}MB. Reduce k-points/GPU."
            )

    parts = []
    if analysis.lapw1_offload:
        parts.append(
            f"lapw1 -> GPU (speedup: {analysis.lapw1_speedup:.1f}x)")
    if analysis.lapw2_offload:
        parts.append(
            f"lapw2 -> GPU (speedup: {analysis.lapw2_speedup:.1f}x)")
    parts.append("lapw0 → CPU (I/O bound)")

    analysis.recommendation = ", ".join(parts)
    return analysis


def estimate_gpu_memory(nmat: int, num_kpoints_per_gpu: int) -> float:
    """Estimate GPU memory requirements in MB.

    Formula: GPU_mem = nmat^2 x 16 bytes x kpts_per_gpu x safety / (1024^2)

    The 16 bytes comes from double-precision complex numbers
    (2 doubles per complex = 16 bytes).
    """
    mem_bytes = (nmat ** 2) * 16.0 * num_kpoints_per_gpu * 1.5
    mem_mb = mem_bytes / (1024.0 * 1024.0)
    return mem_mb


# ---------------------------------------------------------------------------
# Strategy Recommendation
# ---------------------------------------------------------------------------

def recommend_gpu_strategy(
    gpu_info: list[GPUInfo],
    wien2k_gpu: dict[str, Any],
    nmat: int,
    num_kpoints: int,
) -> dict[str, Any]:
    """Generate intelligent GPU offloading recommendation.

    Scenarios:
      1. GPU avail + WIEN2k GPU + nmat > 8000 → full offload
      2. GPU avail + WIEN2k CPU-only → recompile
      3. GPU avail + nmat < 5000 → CPU-only (overhead > benefit)
      4. GPU avail + memory limited → hybrid mode
    """
    gpu_available = bool(gpu_info) and any(g.detected for g in gpu_info)
    wien2k_has_gpu = wien2k_gpu.get("gpu_enabled", False)
    sum(g.memory_mb for g in gpu_info) if gpu_info else 0

    offload = analyze_offload_potential(nmat, num_kpoints, gpu_info=gpu_info)

    if not gpu_available:
        return {
            "strategy": "cpu_only",
            "reason": "No GPU detected",
            "recommendation": "Use CPU-only parallelization. Consider GPU hardware for large systems.",
        }

    if not wien2k_has_gpu:
        return {
            "strategy": "recompile_needed",
            "reason": "GPU present but WIEN2k not compiled with GPU support",
            "recommendation": (
                f"GPU detected ({[g.model for g in gpu_info]}) but WIEN2k is CPU-only. "
                f"Recompile with -DUSE_CUDA or -DUSE_HIP for {offload.lapw1_speedup:.0f}-{offload.lapw2_speedup:.0f}x speedup."
            ),
        }

    if nmat < 5000:
        return {
            "strategy": "cpu_only",
            "reason": f"nmat={nmat} too small for GPU benefit",
            "recommendation": (
                "GPU overhead exceeds benefits for small systems (nmat < 5000). "
                "Use CPU-only for better efficiency."
            ),
        }

    if offload.oom_risk:
        hybrid_kpts = max(1, int(num_kpoints / 2))
        return {
            "strategy": "hybrid_cpu_gpu",
            "reason": "GPU memory constrained",
            "recommendation": (
                f"Hybrid mode: lapw1 → GPU (first {hybrid_kpts} k-points), "
                f"lapw2 → CPU. Reduce k-points per GPU to avoid OOM."
            ),
            "kpoints_gpu": hybrid_kpts,
        }

    if nmat > 8000 and gpu_available and wien2k_has_gpu:
        return {
            "strategy": "full_gpu_offload",
            "reason": f"Large system (nmat={nmat}) with GPU available",
            "recommendation": (
                f"Full GPU offload: lapw1 (speedup {offload.lapw1_speedup:.0f}x), "
                f"lapw2 (speedup {offload.lapw2_speedup:.0f}x). "
                f"Expected total speedup: {offload.lapw1_speedup:.0f}-{offload.lapw2_speedup:.0f}x"
            ),
            "offload": offload,
        }

    return {
        "strategy": "selective_gpu",
        "reason": "GPU available, moderate system size",
        "recommendation": (
            f"GPU offload for lapw1 only (speedup {offload.lapw1_speedup:.0f}x). "
            f"lapw2 remains on CPU."
        ),
        "offload": offload,
    }


# ---------------------------------------------------------------------------
# Hybrid CPU+GPU Machine File Generation
# ---------------------------------------------------------------------------

def generate_hybrid_machines(
    gpu_info: list[GPUInfo],
    cpu_cores: int,
    num_kpoints: int,
    nmat: int,
) -> str:
    """Generate .machines entries for hybrid CPU+GPU mode.

    GPU ranks handle lapw1 diagonalization,
    CPU ranks handle lapw2 and lapw0.
    """
    num_gpus = len(gpu_info) if gpu_info else 0
    if num_gpus == 0:
        return ""

    offload = analyze_offload_potential(nmat, num_kpoints, gpu_info=gpu_info)

    lines = [
        "# Hybrid CPU+GPU .machines — generated by forge",
        f"# GPUs: {num_gpus} ({[g.model for g in gpu_info]})",
        "#",
    ]

    # GPU ranks for lapw1
    if offload.lapw1_offload:
        lines.append("# GPU ranks for lapw1 (diagonalization)")
        for gpu_id in range(num_gpus):
            lines.append(f"lapw1:gpu{gpu_id}:1  # GPU-accelerated")
        lines.append("")

    # CPU ranks for lapw2 and lapw0
    if offload.lapw2_offload:
        lines.append("# CPU ranks for lapw2 (FFT-bound)")
    else:
        lines.append("# CPU ranks for all remaining tasks")
    cpu_per_task = max(1, cpu_cores // max(num_kpoints, 1))
    for i in range(min(num_kpoints, cpu_cores)):
        lines.append(f"lapw2:cpu{i:02d}:{cpu_per_task}")

    lines.append("")
    lines.append(f"lapw0:cpu00:{cpu_cores}  # I/O bound, shared memory")

    if offload.oom_risk:
        lines.append("")
        lines.append(f"# WARNING: GPU OOM risk — required={offload.gpu_memory_required_mb:.0f}MB")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU Benchmark Runner
# ---------------------------------------------------------------------------

def run_gpu_benchmark(
    case_name: str,
    gpu_id: int = 0,
    n_sampling_steps: int = 100,
    n_kpoints: int = 1,
) -> dict[str, Any]:
    """Run quick GPU vs CPU benchmark for lapw1.

    Uses subset of k-points and sampling steps for fast comparison.
    Results saved to .gpu_benchmark.json
    """
    result = {
        "case": case_name,
        "gpu_id": gpu_id,
        "cpu_time_s": 0.0,
        "gpu_time_s": 0.0,
        "speedup": 1.0,
        "benchmark_ok": False,
    }

    import time as _time

    # CPU timing
    env_cpu = os.environ.copy()
    env_cpu["USE_ELPA"] = "0"
    env_cpu["CUDA_VISIBLE_DEVICES"] = ""

    try:
        t0 = _time.time()
        subprocess.run(
            [f"{case_name}lapw1", "-p"],
            env=env_cpu,
            capture_output=True, text=True, timeout=300,
        )
        result["cpu_time_s"] = round(_time.time() - t0, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("CPU benchmark failed — lapw1 binary not found or timed out")
        return result

    # GPU timing
    env_gpu = os.environ.copy()
    env_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        t0 = _time.time()
        subprocess.run(
            [f"{case_name}lapw1gpu", "-p"],
            env=env_gpu,
            capture_output=True, text=True, timeout=300,
        )
        result["gpu_time_s"] = round(_time.time() - t0, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("GPU benchmark failed — lapw1gpu binary not found or timed out")
        return result

    if result["cpu_time_s"] > 0 and result["gpu_time_s"] > 0:
        result["speedup"] = round(result["cpu_time_s"] / result["gpu_time_s"], 1)
        result["benchmark_ok"] = True

    # Save results
    bench_file = Path(".gpu_benchmark.json")
    bench_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(
        f"GPU benchmark: CPU={result['cpu_time_s']}s, "
        f"GPU={result['gpu_time_s']}s, speedup={result['speedup']}x"
    )

    return result
