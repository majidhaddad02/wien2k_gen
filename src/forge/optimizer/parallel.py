"""
NUMA-Aware Parallelization Engine for WIEN2k HPC Workflows.

Implements recommendations from Laskowski & Blaha (CPC 185, 2014) and WIEN2k User Guide 2023:

1. NUMA-Aware Parallelization — 1 MPI rank per NUMA node + OpenMP within node
2. Hybrid MPI+OpenMP for LAPW0 — OpenMP for shared charge density access
3. Adaptive K-Point Weighting — more resources to near-Fermi k-points
4. I/O Optimization — granular parallelization for large systems
5. RKMAX Adaptive Selection — based on atomic composition
6. GMAX Optimization — based on calculation type

References:
  Laskowski & Blaha, CPC 185 (2014) 263-271
  Blaha et al., CPC 59 (1990) 399-415
  Schwarz et al., CPC 147 (2002) 71-76
  WIEN2k User Guide 2023, Sections 4-6
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.topology import Topology
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelizationStrategy:
    mode: str  # "pure_mpi", "hybrid", "kpoint", "numa_aware"
    mpi_ranks: int
    omp_threads: int
    cores_per_node: list[int]
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

    Based on Laskowski & Blaha, Comput. Phys. Commun. 185 (2014):
    granular parallelization is only
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
                f"NUMA-aware granular: {ranks} ranks x {omp} threads, granular={granularity} "
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
                f"Standard NUMA: {ranks} ranks x {omp} threads. "
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
                f"Hybrid MPI+OpenMP for LAPW0: {numa_nodes} ranks x "
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
        recommendation=f"Pure OpenMP for LAPW0: 1 rank x {total_cores} threads.",
    )


def recommend_io_strategy(
    nmat: int,
    nkpt: int,
    atoms: int,
    scratch_fs: str = "tmpfs",
) -> dict[str, object]:
    """Recommend I/O optimization strategy.

    For large systems, I/O can be the main bottleneck. Writing .vector files
    in each iteration consumes significant time. Use granular parallelization
    (group k-points) and disable intermediate writes.

    ⚠ WIEN2k Usersguide §4.5.8: nowrite_vector is marked DANGER and
    should only be enabled when scratch is fast+reliable AND checkpointing
    is active. A crash without .vector files means total loss of all
    computation.

    Ref: WIEN2k User Guide 2023, Chapter on Parallelization.
    """
    result: dict[str, object] = {}

    if nmat > 10000:
        result["granularity"] = 16
        result["vector_split"] = 4 if nkpt < 8 else 2
        result["nowrite_vector"] = False
        result["nowrite_vector_warning"] = (
            f"⚠ DANGER: nmat={nmat} is large enough for nowrite_vector, "
            f"but this will PREVENT RESTART on crash. Enable checkpointing "
            f"(forge monitor creates checkpoints automatically) before "
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


def recommend_rkmax(  # noqa: C901
    atomic_numbers: list[int],
    calculation_type: str = "scf",
    rmt_ratios: Optional[list[float]] = None,
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

    GMAX = 2.0 x RKMAX (default)
    GMAX = 2.5 x RKMAX (forces)
    GMAX = 3.0 x RKMAX (EFG, hyperfine)

    Ref: Schwarz et al., CPC 147 (2002) 71-76.
    """
    factors: dict[str, float] = {
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

    Threshold lowered to 8000 based on WIEN2k user guide recommendations
    (NMAT > 10000 beneficial for ScaLAPACK/ELPA) and official WIEN2k
    benchmarks at http://www.wien2k.at/reg_user/benchmark/.

    Ref: ELPA documentation, WIEN2k integration guide, WIEN2k benchmarks."""

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

    Decision matrix based on WIEN2k benchmarks
    (http://www.wien2k.at/reg_user/benchmark/):
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


def recommend_mkl_threading(nmat: int, nkpt: int) -> Optional[int]:
    """Recommend MKL_NUM_THREADS for optimal BLAS performance.

    Large nmat: limit to 4 to avoid oversubscription
    Medium nmat: use 8
    Small nmat: use all available (returns None)

    Ref: Intel MKL Developer Guide, Section on Threading.
    """
    if nmat > 4000:
        return 4
    elif nmat > 2000:
        return 8
    return None


def recommend_weighted_kpoint_distribution(
    nkpt: int,
    nmpi: int,
    k_weights: Optional[list[float]] = None,
    symmetry_weight: bool = True,
) -> dict[int, list[int]]:
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
    rank_kpts: dict[int, list[int]] = {r: [] for r in range(nmpi)}

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


def _estimate_kpoint_weights(nkpt: int, symmetry_weight: bool = True) -> list[float]:
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


def detect_numa_topology() -> dict[str, Any]:  # noqa: C901
    """Detect NUMA topology via numactl or hwloc.

    Returns dict with:
        num_nodes: int — number of NUMA nodes
        cores_per_node: List[int] — physical cores per node
        total_cores: int — total physical cores
        detected: bool — whether NUMA was detected
    """
    result: dict[str, Any] = {
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
        logger.debug("Suppressed exception in detect_numa_topology()", exc_info=True)

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
        logger.debug("Suppressed exception in detect_numa_topology()", exc_info=True)

    # Final fallback: assume single NUMA node
    import os
    cpu_count = os.cpu_count() or 1
    result["total_cores"] = cpu_count
    result["cores_per_node"] = [cpu_count]
    return result


def numa_aware_kpoint_distribution(
    kpoints: int,
    numa_nodes: int,
    cores_per_node: list[int],
    k_weights: Optional[list[float]] = None,
) -> dict[str, Any]:
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
    node_kpts: dict[int, list[int]] = {n: [] for n in range(numa_nodes)}
    node_cores: dict[int, int] = {
        n: cores_per_node[n] if n < len(cores_per_node) else 1
        for n in range(numa_nodes)
    }

    # Greedy bin-packing: assign each k-point to the least loaded node
    # This is Best Fit Decreasing (BFD), which minimises load variance
    # across NUMA nodes better than round-robin for heterogenous systems.
    for k_idx, weight in indexed:
        best_node = min(range(numa_nodes), key=lambda n: node_loads[n])
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
    node_kpts: dict[int, list[int]],
    node_cores: dict[int, int],
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
    lines = ["# NUMA-aware .machines — generated by forge"]
    for node_idx in sorted(node_kpts.keys()):
        cores = node_cores.get(node_idx, 1)
        kpt_list = node_kpts[node_idx]
        lines.append(f"# NUMA Node {node_idx} — {len(kpt_list)} k-points")
        entries_needed = max(len(kpt_list), 1)
        for _i in range(entries_needed):
            hostname = f"{hostname_prefix}{node_idx + 1:02d}"
            lines.append(f"lapw1:{hostname}:{cores}")
        lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Phase 2 — FFD (First Fit Decreasing) K-point Distribution
# ===========================================================================

def calculate_kpoint_weights(case_name: str) -> list[float]:  # noqa: C901
    """Calculate k-point weights from case.klist file.

    Reads k-point multiplicities and weights from WIEN2k klist.
    Normalizes weights so sum = 1.0

    Falls back to uniform weights if klist is unavailable or malformed.
    """

    weights = []
    klist_path = Path(f"{case_name}.klist")
    if not klist_path.exists():
        klist_path = Path(case_name) / f"{case_name}.klist"
    if not klist_path.exists():
        klist_files = sorted(Path(".").glob("*.klist"))
        if klist_files:
            klist_path = klist_files[0]

    try:
        content = klist_path.read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in content.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]

        if not lines:
            return [1.0]

        # Header line: nkpt OR nkpt type
        header_parts = lines[0].split()
        nkpt = 0
        if header_parts and header_parts[0].isdigit():
            nkpt = int(header_parts[0])

        if nkpt > 0:
            # Parse k-point lines: x y z weight
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        w = float(parts[3])
                        weights.append(w)
                    except ValueError:
                        continue

        if not weights:
            return [1.0] * max(nkpt, 1)

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return weights

    except Exception:
        return [1.0]


def ffd_kpoint_distribution(
    kpoint_weights: list[float],
    num_ranks: int,
) -> dict[str, Any]:
    """First Fit Decreasing (FFD) algorithm for balanced k-point assignment.

    Algorithm (greedy bin-packing):
      1. Sort k-points by weight descending
      2. Maintain rank_loads list (all zeros)
      3. For each k-point: assign to rank with minimum current load
      4. Update rank_loads
      5. Compute balance metrics

    Returns dict with:
        rank_kpts: Dict[int, List[int]] — k-point indices per rank
        rank_loads: List[float] — total weight per rank
        balance_ratio: float — min_load/max_load
        efficiency: float — parallel efficiency
        load_variance: float
        method: str — "ffd"
    """

    if num_ranks <= 0:
        return {
            "rank_kpts": {}, "rank_loads": [],
            "balance_ratio": 1.0, "efficiency": 1.0,
            "load_variance": 0.0, "method": "ffd",
        }

    # Sort descending
    indexed = sorted(enumerate(kpoint_weights), key=lambda x: -x[1])

    rank_loads = [0.0] * num_ranks
    rank_kpts: dict[int, list[int]] = {r: [] for r in range(num_ranks)}

    # Best Fit Decreasing (BFD) assignment: place each k-point on the
    # rank with the lowest current load, using heap for O(nkpt·log nrank).
    import heapq as _heapq
    heap: list[tuple] = [(0.0, r) for r in range(num_ranks)]
    for k_idx, weight in indexed:
        current_load, min_rank = _heapq.heappop(heap)
        rank_loads[min_rank] = current_load + weight
        rank_kpts[min_rank].append(k_idx)
        _heapq.heappush(heap, (rank_loads[min_rank], min_rank))

    metrics = calculate_balance_quality(rank_loads)
    metrics["rank_kpts"] = rank_kpts
    metrics["method"] = "ffd"

    logger.info(
        f"FFD distribution: {len(kpoint_weights)} k-points → {num_ranks} ranks, "
        f"balance_ratio={metrics['balance_ratio']:.3f}, "
        f"efficiency={metrics['efficiency']:.3f}"
    )

    return metrics


def calculate_balance_quality(rank_loads: list[float]) -> dict[str, float]:
    """Calculate load balance quality metrics.

    Returns:
        balance_ratio: min_load / max_load  (1.0 = perfect)
        efficiency: sum / (n x max)  (parallel efficiency)
        load_variance: variance of loads
        load_std: standard deviation
        max_load: maximum load value
        min_load: minimum load value
    """
    import math as _math

    n = len(rank_loads)
    if n == 0:
        return {"balance_ratio": 1.0, "efficiency": 1.0,
                "load_variance": 0.0, "load_std": 0.0,
                "max_load": 0.0, "min_load": 0.0}

    total = sum(rank_loads)
    max_val = max(rank_loads)
    min_val = min(rank_loads)
    mean = total / n

    balance_ratio = min_val / max_val if max_val > 0 else 1.0
    efficiency = total / (n * max_val) if max_val > 0 else 1.0

    variance = sum((x - mean) ** 2 for x in rank_loads) / n
    std = _math.sqrt(variance)

    return {
        "balance_ratio": round(balance_ratio, 4),
        "efficiency": round(efficiency, 4),
        "load_variance": round(variance, 6),
        "load_std": round(std, 4),
        "max_load": round(max_val, 4),
        "min_load": round(min_val, 4),
    }


def round_robin_distribution(
    kpoint_weights: list[float],
    num_ranks: int,
) -> dict[str, Any]:
    """Round Robin k-point distribution for comparison baseline."""
    if num_ranks <= 0:
        return {
            "rank_kpts": {}, "rank_loads": [],
            "balance_ratio": 1.0, "efficiency": 1.0,
            "load_variance": 0.0, "method": "round_robin",
        }

    rank_loads = [0.0] * num_ranks
    rank_kpts: dict[int, list[int]] = {r: [] for r in range(num_ranks)}

    for i, weight in enumerate(kpoint_weights):
        target_rank = i % num_ranks
        rank_loads[target_rank] += weight
        rank_kpts[target_rank].append(i)

    metrics = calculate_balance_quality(rank_loads)
    metrics["rank_kpts"] = rank_kpts
    metrics["method"] = "round_robin"

    return metrics


def compare_distribution_methods(
    kpoint_weights: list[float],
    num_ranks: int,
) -> dict[str, Any]:
    """Compare FFD vs Round-Robin and select the better method.

    Returns dict with ffd and round_robin results plus winner recommendation.
    """
    ffd_result = ffd_kpoint_distribution(kpoint_weights, num_ranks)
    rr_result = round_robin_distribution(kpoint_weights, num_ranks)

    ffd_ratio = ffd_result["balance_ratio"]
    rr_ratio = rr_result["balance_ratio"]
    improvement = (ffd_ratio - rr_ratio) / rr_ratio * 100 if rr_ratio > 0 else 0.0

    if ffd_ratio > rr_ratio:
        winner = "ffd"
    elif rr_ratio > ffd_ratio:
        winner = "round_robin"
    else:
        winner = "tie"

    logger.info(
        f"Distribution comparison: FFD ratio={ffd_ratio:.3f}, "
        f"RoundRobin ratio={rr_ratio:.3f}, "
        f"improvement={improvement:.1f}%, winner={winner}"
    )

    if ffd_ratio < 0.90:
        logger.warning(
            f"Poor k-point load balance (FFD ratio={ffd_ratio:.3f}). "
            f"Consider increasing k-point count or adjusting distribution."
        )
    if ffd_result["efficiency"] < 0.85:
        logger.warning(
            f"Low parallel efficiency ({ffd_result['efficiency']:.3f}). "
            f"Some ranks are idle."
        )

    return {
        "ffd": ffd_result,
        "round_robin": rr_result,
        "winner": winner,
        "improvement_pct": round(improvement, 1),
        "recommendation": (
            f"Use {winner.upper()} distribution: "
            f"balance_ratio={max(ffd_ratio, rr_ratio):.3f}, "
            f"improvement={improvement:.1f}%"
        ),
    }


def generate_ffd_machines(
    rank_kpts: dict[int, list[int]],
    rank_loads: list[float],
    num_ranks: int,
    hostname_prefix: str = "rank",
) -> str:
    """Generate .machines entries from FFD k-point assignment.

    Format:
        # FFD-optimized k-point distribution
        # Balance ratio: 0.97, Efficiency: 0.98

        lapw1:rank00:4  # k-points: 1,3,7,12 (weight: 0.25)
        lapw1:rank01:4  # k-points: 2,5,8,11 (weight: 0.24)
    """
    total = sum(rank_loads) if rank_loads else 1.0
    metrics = calculate_balance_quality(rank_loads)

    lines = [
        "# FFD-optimized k-point distribution (First Fit Decreasing)",
        f"# Balance ratio: {metrics['balance_ratio']:.2f}, "
        f"Efficiency: {metrics['efficiency']:.2f}",
        "# Generated by forge",
        "",
    ]

    for rank_idx in sorted(rank_kpts.keys()):
        kpts = rank_kpts[rank_idx]
        weight = rank_loads[rank_idx] if rank_idx < len(rank_loads) else 0.0
        kpt_str = ",".join(str(k + 1) for k in kpts[:8])
        if len(kpts) > 8:
            kpt_str += f",...+{len(kpts) - 8}more"
        hostname = f"{hostname_prefix}{rank_idx:02d}"
        lines.append(
            f"lapw1:{hostname}:4  "
            f"# k-points: {kpt_str} (weight: {weight / total:.3f})"
        )

    return "\n".join(lines)
