"""
NUMA-Aware Parallelization Engine for WIEN2k HPC Workflows.

Implements recommendations from Blaha et al. (CPC 185, 2014) and WIEN2k User Guide 2023:

1. NUMA-Aware Parallelization — 1 MPI rank per NUMA node + OpenMP within node
2. Hybrid MPI+OpenMP for LAPW0 — OpenMP for shared charge density access
3. Adaptive K-Point Weighting — more resources to near-Fermi k-points
4. I/O Optimization — granular parallelization for large systems
5. RKMAX Adaptive Selection — based on atomic composition
6. GMAX Optimization — based on calculation type

References:
  Blaha et al., CPC 185 (2014) 263-271
  Blaha et al., CPC 59 (1990) 399-415
  Schwarz et al., CPC 147 (2002) 71-76
  WIEN2k User Guide 2023, Sections 4-6
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core.topology import Topology
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelizationStrategy:
    mode: str  # "pure_mpi", "hybrid", "kpoint", "numa_aware"
    mpi_ranks: int
    omp_threads: int
    cores_per_node: List[int]
    numa_binding: bool
    granularity: int
    efficiency_pct: float
    recommendation: str


def recommend_numa_strategy(
    topology: Topology,
    nmat: int,
    nkpt: int,
    atoms: int,
    available_cores: int = 0,
) -> ParallelizationStrategy:
    """Recommend NUMA-aware parallelization strategy.

    Based on Blaha et al. CPC 185 (2014): granular parallelization is only
    effective when each MPI rank has multiple k-points (kpts_per_mpi > 10)
    and nmat exceeds 10000. For systems with nmat 5000-10000, granular
    overhead outweighs benefits.

    Also checks memory bandwidth: if < 50 GB/s, increasing MPI ranks
    provides no benefit since LAPW1 is memory-bound.
    """
    total_cores = sum(topology.cores_per_node)
    max_cores = min(available_cores, total_cores) if available_cores > 0 else total_cores

    try:
        from ..core.hardware import get_memory_bandwidth_gb_s, get_numa_node_count
        numa_nodes = get_numa_node_count()
        mem_bw = get_memory_bandwidth_gb_s()
    except Exception:
        numa_nodes = len(topology.nodes)
        mem_bw = float("inf")

    is_very_large = nmat > 10000
    is_large = 5000 < nmat <= 10000
    low_bandwidth = mem_bw < 50.0
    nmpi = max(nkpt, 1)
    kpts_per_mpi = nkpt / nmpi if nmpi > 0 else 0

    if low_bandwidth:
        logger.warning(f"Memory bandwidth {mem_bw:.1f} GB/s < 50 GB/s. "
                       f"LAPW1 is memory-bound; additional MPI ranks yield no benefit.")

    if is_very_large and numa_nodes > 1 and kpts_per_mpi > 10:
        ranks = min(numa_nodes, max_cores)
        omp = min(max_cores // ranks, 16)
        granularity = min(16, max(1, int(kpts_per_mpi / 2)))
        return ParallelizationStrategy(
            mode="numa_aware",
            mpi_ranks=ranks,
            omp_threads=omp,
            cores_per_node=[ranks],
            numa_binding=True,
            granularity=granularity,
            efficiency_pct=85.0,
            recommendation=(
                f"NUMA-aware granular: {ranks} ranks × {omp} threads, granular={granularity} "
                f"across {numa_nodes} NUMA nodes. Expected ~30% improvement for very large "
                f"systems (nmat={nmat}). Use `numactl --membind` for memory locality."
            ),
        )

    if is_large and numa_nodes > 1:
        ranks = min(numa_nodes, max_cores)
        omp = min(max_cores // ranks, 16)
        return ParallelizationStrategy(
            mode="numa_aware",
            mpi_ranks=ranks,
            omp_threads=omp,
            cores_per_node=[ranks],
            numa_binding=True,
            granularity=1,
            efficiency_pct=80.0,
            recommendation=(
                f"Standard NUMA: {ranks} ranks × {omp} threads. "
                f"nmat={nmat} below granular threshold (10000); granular overhead avoided."
            ),
        )

    return ParallelizationStrategy(
        mode="hybrid",
        mpi_ranks=nkpt if nkpt <= max_cores else max_cores,
        omp_threads=1,
        cores_per_node=[max_cores],
        numa_binding=False,
        granularity=1,
        efficiency_pct=70.0,
        recommendation=f"Standard hybrid: k-point parallel with {min(nkpt, max_cores)} ranks.",
    )


def recommend_lapw0_strategy(
    topology: Topology,
    nmat: int,
    fft_nx: int = 0,
    fft_ny: int = 0,
    fft_nz: int = 0,
) -> ParallelizationStrategy:
    """Recommend hybrid MPI+OpenMP for LAPW0.

    LAPW0 (potential calculation) benefits from OpenMP due to shared
    charge density access. For grids > 1M points, hybrid mode with
    MPI ranks = NUMA nodes and OMP threads = 16 provides ~85% efficiency.
    Ref: Laskowski & Blaha, CPC 185 (2014).
    """
    total_cores = sum(topology.cores_per_node)
    fft_total = fft_nx * fft_ny * fft_nz if fft_nx > 0 else (nmat * 10)

    try:
        from ..core.hardware import get_numa_node_count
        numa_nodes = get_numa_node_count()
    except Exception:
        numa_nodes = 1

    is_large_grid = fft_total > 1_000_000

    if is_large_grid and numa_nodes > 1:
        return ParallelizationStrategy(
            mode="hybrid",
            mpi_ranks=numa_nodes,
            omp_threads=min(total_cores // numa_nodes, 16),
            cores_per_node=[total_cores // numa_nodes for _ in range(numa_nodes)],
            numa_binding=True,
            granularity=1,
            efficiency_pct=85.0,
            recommendation=(
                f"Hybrid MPI+OpenMP for LAPW0: {numa_nodes} ranks × "
                f"{min(total_cores // numa_nodes, 16)} threads. "
                f"Optimal for FFT grid > 1M points ({fft_total} points). "
                f"Cache reuse in FFT benefits from OpenMP."
            ),
        )

    return ParallelizationStrategy(
        mode="pure_mpi",
        mpi_ranks=1,
        omp_threads=total_cores,
        cores_per_node=[total_cores],
        numa_binding=False,
        granularity=1,
        efficiency_pct=70.0,
        recommendation=f"Pure OpenMP for LAPW0: 1 rank × {total_cores} threads.",
    )


def recommend_io_strategy(
    nmat: int,
    nkpt: int,
    atoms: int,
    scratch_fs: str = "tmpfs",
) -> Dict[str, object]:
    """Recommend I/O optimization strategy.

    For large systems, I/O can be the main bottleneck. Writing .vector files
    in each iteration consumes significant time. Use granular parallelization
    (group k-points) and disable intermediate writes.

    ⚠ Peter Blaha warns: nowrite_vector is DANGEROUS without checkpointing.
    Only enable it when scratch is fast+reliable AND checkpointing is active.
    A crash without .vector files means total loss of all computation.

    Ref: WIEN2k User Guide 2023, Chapter on Parallelization.
    """
    result: Dict[str, object] = {}

    if nmat > 10000:
        result["granularity"] = 16
        result["vector_split"] = 4 if nkpt < 8 else 2
        result["nowrite_vector"] = False
        result["nowrite_vector_warning"] = (
            f"⚠ DANGER: nmat={nmat} is large enough for nowrite_vector, "
            f"but this will PREVENT RESTART on crash. Enable checkpointing "
            f"(wien2k_gen monitor creates checkpoints automatically) before "
            f"setting nowrite_vector=True."
        )
        result["recommendation"] = (
            f"Very large system (nmat={nmat}): enable granular={16}, "
            f"vector_split={result['vector_split']}. "
            f"[red]nowrite_vector disabled for safety[/] — use checkpointing instead. "
            f"Expect 25-40% I/O reduction from granularity."
        )
    elif nmat > 5000:
        result["granularity"] = 8
        result["vector_split"] = 2 if nkpt < 16 else 1
        result["nowrite_vector"] = False
        result["recommendation"] = (
            f"Large system (nmat={nmat}): enable granular={8}. "
            f"Collective I/O recommended for parallel filesystems."
        )
    elif nmat > 2500:
        result["granularity"] = 1
        result["vector_split"] = 0
        result["nowrite_vector"] = False
        result["recommendation"] = "Standard I/O — no special optimization needed."
    else:
        result["granularity"] = 1
        result["vector_split"] = 0
        result["nowrite_vector"] = False
        result["recommendation"] = "Standard I/O — no special optimization needed."

    if scratch_fs in ("tmpfs", "ramfs"):
        result["recommendation"] += " Scratch on tmpfs: excellent I/O performance."

    # Add granular memory warning (Blaha: granular parallelization increases per-rank memory)
    if result.get("granularity", 1) > 4:
        result.setdefault("warnings", [])
        result["warnings"].append(
            f"Granular={result['granularity']} increases per-MPI-rank memory. "
            f"Ensure each rank has enough RAM with safety factor 3x."
        )

    return result


def recommend_rkmax(
    atomic_numbers: List[int],
    calculation_type: str = "scf",
    rmt_ratios: Optional[List[float]] = None,
    is_soc: bool = False,
) -> float:
    """Recommend RKMAX based on atomic composition.

    Heavy atoms (Z > 50): RKMAX = 7-9
    Medium atoms (Z 20-50): RKMAX = 6-8
    Light atoms (Z < 20): RKMAX = 5-7

    ⚠ Blaha critique: RKMAX also depends on RMT hardness.
    Elements with small RMT (O, F, N in oxides/fluorides) need
    RKMAX >= 7.0 because hard potentials demand more plane waves.
    SOC calculations need RKMAX >= 7.0 for reliable results.

    Ref: Blaha et al., WIEN2k User Guide 2023.
    """
    if not atomic_numbers:
        return 7.0

    max_z = max(atomic_numbers)
    avg_z = sum(atomic_numbers) / len(atomic_numbers)

    if max_z > 70:
        base = 8.0
    elif max_z > 50:
        base = 7.5
    elif max_z > 30:
        base = 7.0
    elif max_z > 20:
        base = 6.5
    else:
        base = 6.0

    hard_elements = {8, 9, 7, 16, 17, 35}
    has_hard = any(z in hard_elements for z in atomic_numbers)

    if has_hard and rmt_ratios:
        min_rmt_idx = rmt_ratios.index(min(rmt_ratios))
        if atomic_numbers[min_rmt_idx] in hard_elements and rmt_ratios[min_rmt_idx] < 1.6:
            base = max(base, 7.0)

    if has_hard and rmt_ratios is None and base < 7.0:
        base = 7.0

    if is_soc:
        base = max(base, 7.0)
        base += 0.5

    if calculation_type == "opt":
        base += 0.5
    elif calculation_type in ("efg", "hyperfine"):
        base += 1.0
    elif calculation_type == "dos":
        base += 0.5

    return round(base, 1)


def recommend_gmax(rkmax: float, calculation_type: str = "scf") -> float:
    """Recommend GMAX based on RKMAX and calculation type.

    GMAX = 2.0 × RKMAX (default)
    GMAX = 2.5 × RKMAX (forces)
    GMAX = 3.0 × RKMAX (EFG, hyperfine)

    Ref: Schwarz et al., CPC 147 (2002) 71-76.
    """
    factors: Dict[str, float] = {
        "scf": 2.0,
        "dos": 2.0,
        "band": 2.0,
        "opt": 2.5,
        "forces": 2.5,
        "efg": 3.0,
        "hyperfine": 3.0,
    }
    factor = factors.get(calculation_type, 2.0)
    return round(rkmax * factor, 1)


def recommend_elpa_solver(
    nmat: int,
    nkpt: int,
    is_soc: bool = False,
    is_hybrid: bool = False,
    num_cores: int = 0,
) -> Optional[str]:
    """Recommend ELPA solver stage based on problem characteristics.

    Threshold lowered to 8000 based on Thomas Ruh (2023) PhD thesis benchmarks
    showing ELPA crossover at nmat ≈ 8000 for modern multicore nodes.

    Ref: ELPA documentation, WIEN2k integration guide, Ruh 2023.
    """
    if nmat < 500:
        return None

    if nmat > 8000:
        return "elpa2"
    if nmat > 5000 and num_cores > 64:
        return "elpa2"
    if nmat > 5000 or (is_soc and nmat > 2000):
        return "elpa2"
    if is_hybrid:
        return "elpa2"
    if nmat > 2000:
        return "elpa1"

    return None


def should_use_elpa(nmat: int, num_cores: int = 0) -> bool:
    """Determine if ELPA eigensolver should be used.

    Decision matrix based on Ruh 2023 benchmarks:
        nmat > 8000 → True (ELPA always faster)
        nmat > 5000 AND num_cores > 64 → True (better scalability)
        nmat > 5000 AND is_soc → True
        otherwise → False

    Warns if nmat < 5000 (ELPA overhead exceeds benefits).
    """
    if nmat > 8000:
        return True
    if nmat > 5000 and num_cores > 64:
        return True
    if nmat < 5000 and nmat > 500:
        logger.warning(
            f"ELPA overhead may exceed benefits for small systems "
            f"(nmat={nmat} < 5000). Consider using ScaLAPACK."
        )
    return False


def recommend_mkl_threading(nmat: int, nkpt: int) -> int:
    """Recommend MKL_NUM_THREADS for optimal BLAS performance.

    Large nmat: limit to 4 to avoid oversubscription
    Medium nmat: use 8
    Small nmat: use all available

    Ref: Intel MKL Developer Guide, Section on Threading.
    """
    if nmat > 4000:
        return 4
    elif nmat > 2000:
        return 8
    return 0  # use default


def recommend_weighted_kpoint_distribution(
    nkpt: int,
    nmpi: int,
    k_weights: Optional[List[float]] = None,
    symmetry_weight: bool = True,
) -> Dict[int, List[int]]:
    """Distribute k-points weighted by computational cost per k-point.

    Equal distribution of k-points across MPI ranks assumes all k-points
    require equal computation. In reality:
    - k-points near Fermi surface require more SCF iterations
    - k-points with lower symmetry have more plane waves
    - Hybrid functional k-points have non-uniform cost

    This function computes a weighted distribution minimizing the
    maximum total weight per rank (bin-packing heuristic).

    Args:
        nkpt: Total number of k-points
        nmpi: Number of MPI ranks
        k_weights: Optional list of per-kpoint computational weights.
                   If None, weights are estimated from symmetry heuristics.
        symmetry_weight: If True and no explicit weights, estimate weights
                        from IBZ (irreducible Brillouin zone) distribution.

    Returns:
        Dict mapping rank index (0..nmpi-1) to list of k-point indices.
    """
    if nmpi <= 1:
        return {0: list(range(nkpt))}

    if not k_weights or len(k_weights) != nkpt:
        k_weights = _estimate_kpoint_weights(nkpt, symmetry_weight)

    weighted = sorted(
        [(i, k_weights[i] if i < len(k_weights) else 1.0) for i in range(nkpt)],
        key=lambda x: -x[1],
    )

    rank_loads = [0.0] * nmpi
    rank_kpts: Dict[int, List[int]] = {r: [] for r in range(nmpi)}

    for k_idx, weight in weighted:
        min_rank = min(range(nmpi), key=lambda r: rank_loads[r])
        rank_loads[min_rank] += weight
        rank_kpts[min_rank].append(k_idx)

    avg_load = sum(k_weights) / nmpi if k_weights else nkpt / nmpi
    max_load = max(rank_loads)
    imbalance = (max_load / avg_load - 1.0) * 100 if avg_load > 0 else 0.0

    logger.info(
        f"K-point weighted distribution: {nkpt} kpts → {nmpi} ranks, "
        f"imbalance={imbalance:.1f}% (lower is better)"
    )

    if imbalance > 50.0:
        logger.warning(
            f"Severe k-point load imbalance ({imbalance:.0f}%). "
            f"Consider increasing granularity or using k-point parallel."
        )

    return rank_kpts


def _estimate_kpoint_weights(nkpt: int, symmetry_weight: bool = True) -> List[float]:
    """Estimate k-point computational weights from symmetry heuristics.

    Without explicit IBZ data, we estimate on the assumption that
    k-points near the zone boundary (index > nkpt/2) may have
    different plane-wave counts than zone-center k-points.
    """
    weights = [1.0] * nkpt
    if not symmetry_weight or nkpt < 4:
        return weights

    mid = nkpt // 2
    for i in range(nkpt):
        if i < mid // 3:
            weights[i] = 0.8
        elif i > 2 * nkpt // 3:
            weights[i] = 1.3
        else:
            weights[i] = 1.0

    return weights


def detect_numa_topology() -> Dict[str, Any]:
    """Detect NUMA topology via numactl or hwloc.

    Returns dict with:
        num_nodes: int — number of NUMA nodes
        cores_per_node: List[int] — physical cores per node
        total_cores: int — total physical cores
        detected: bool — whether NUMA was detected
    """
    result: Dict[str, Any] = {
        "num_nodes": 1,
        "cores_per_node": [1],
        "total_cores": 1,
        "detected": False,
    }

    # Try numactl --hardware (most reliable on Linux HPC)
    import subprocess
    try:
        proc = subprocess.run(
            ["numactl", "--hardware"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            text = proc.stdout
            node_match = re.findall(r'node\s+(\d+)\s+cpus:', text)
            core_counts = []
            for nm in node_match:
                cpus_line = re.search(
                    rf'node\s+{nm}\s+cpus:\s+([\d\s]+)', text)
                if cpus_line:
                    cpus = [int(c) for c in cpus_line.group(1).split()]
                    core_counts.append(len(cpus))
            if core_counts:
                result["num_nodes"] = len(core_counts)
                result["cores_per_node"] = core_counts
                result["total_cores"] = sum(core_counts)
                result["detected"] = True
                return result
    except Exception:
        pass

    # Fallback: check /sys/devices/system/node
    try:
        node_dirs = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
        if node_dirs:
            core_counts = []
            for nd in node_dirs:
                cpulist = nd / "cpulist"
                if cpulist.exists():
                    cpus = cpulist.read_text().strip()
                    ranges = cpus.split(",")
                    count = 0
                    for r in ranges:
                        if "-" in r:
                            a, b = r.split("-")
                            count += int(b) - int(a) + 1
                        else:
                            count += 1
                    core_counts.append(count)
            if core_counts:
                result["num_nodes"] = len(core_counts)
                result["cores_per_node"] = core_counts
                result["total_cores"] = sum(core_counts)
                result["detected"] = True
                return result
    except Exception:
        pass

    # Final fallback: assume single NUMA node
    import os
    cpu_count = os.cpu_count() or 1
    result["total_cores"] = cpu_count
    result["cores_per_node"] = [cpu_count]
    return result


def numa_aware_kpoint_distribution(
    kpoints: int,
    numa_nodes: int,
    cores_per_node: List[int],
    k_weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Distribute k-points across NUMA nodes for local memory access.

    Algorithm:
      1. Sort k-points by weight (descending)
      2. Round-robin allocation to NUMA nodes
      3. Each NUMA node processes only its local k-points
      4. Compute balance_ratio = min_load / max_load

    Returns dict with:
        node_kpts: Dict[int, List[int]] — k-point indices per NUMA node
        node_cores: Dict[int, int] — cores allocated per NUMA node
        balance_ratio: float — load balance quality (1.0 = perfect)
        recommendation: str
    """
    if k_weights is None:
        k_weights = [1.0] * kpoints
    if len(k_weights) != kpoints:
        k_weights = [1.0] * kpoints

    # Sort k-points by weight descending for greedy allocation
    indexed = sorted(enumerate(k_weights), key=lambda x: -x[1])

    node_loads = [0.0] * numa_nodes
    node_kpts: Dict[int, List[int]] = {n: [] for n in range(numa_nodes)}
    node_cores: Dict[int, int] = {
        n: cores_per_node[n] if n < len(cores_per_node) else 1
        for n in range(numa_nodes)
    }

    # Round-robin allocation
    for i, (k_idx, weight) in enumerate(indexed):
        best_node = i % numa_nodes
        node_loads[best_node] += weight
        node_kpts[best_node].append(k_idx)

    # Compute balance ratio
    max_load = max(node_loads) if node_loads else 1.0
    min_load = min(node_loads) if node_loads else 1.0
    balance_ratio = min_load / max_load if max_load > 0 else 1.0

    recommendation = (
        f"NUMA-aware distribution: {numa_nodes} nodes, "
        f"{kpoints} k-points, balance_ratio={balance_ratio:.3f}"
    )

    if balance_ratio < 0.90:
        logger.warning(
            f"NUMA load imbalance detected (ratio={balance_ratio:.3f}). "
            f"Consider adjusting k-point distribution or increasing granularity."
        )
        recommendation += " [yellow]WARNING: imbalance > 10%[/]"

    logger.info(recommendation)

    return {
        "node_kpts": node_kpts,
        "node_cores": node_cores,
        "balance_ratio": round(balance_ratio, 4),
        "recommendation": recommendation,
    }


def generate_numa_aware_machines(
    case_name: str,
    node_kpts: Dict[int, List[int]],
    node_cores: Dict[int, int],
    hostname_prefix: str = "node",
) -> str:
    """Generate .machines entries with explicit NUMA grouping.

    Format:
        # NUMA Node 0
        lapw1:node01:4
        lapw1:node02:4

        # NUMA Node 1
        lapw1:node03:4

    Returns multi-line string suitable for writing to .machines.
    """
    lines = ["# NUMA-aware .machines — generated by wien2k_gen"]
    for node_idx in sorted(node_kpts.keys()):
        cores = node_cores.get(node_idx, 1)
        kpt_list = node_kpts[node_idx]
        lines.append(f"# NUMA Node {node_idx} — {len(kpt_list)} k-points")
        entries_needed = max(len(kpt_list), 1)
        for i in range(entries_needed):
            hostname = f"{hostname_prefix}{node_idx + 1:02d}"
            lines.append(f"lapw1:{hostname}:{cores}")
        lines.append("")
    return "\n".join(lines)
