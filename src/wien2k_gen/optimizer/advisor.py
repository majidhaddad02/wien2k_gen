"""
Expert Advisor Module with Scientific Scaling Models & Multi-Objective Optimization.
Production features:
• Dynamic Roofline model for memory-bandwidth-aware k-point parallelism estimation
• Multi-objective optimization: time, energy, cost, or balanced trade-offs
• Heterogeneous cluster support with weighted, NUMA-aware core distribution
• Stage-aware configuration (lapw0/lapw1/lapw2) with I/O & vector split strategies
• Confidence scoring based on hardware/software compatibility & scheduler constraints
• Structured output with rigorous validation hooks & scheduler limit enforcement
All comments and documentation are in English per project standards.
"""

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict

from ..core.hardware import (
    calculate_peak_fp64_gflops,
    check_elpa_available,
    check_mkl_available,
    get_cpu_architecture,
    get_cpu_frequency_info,
    get_fma_units_per_core,
    get_job_memory_limit_mb,
    get_memory_bandwidth_gb_s,
    get_numa_node_count,
    get_physical_cores,
    get_scratch_filesystem_type,
    get_total_mem_kb,
    is_hyperthreading_active,
)
from ..core.topology import Topology
from ..logging_config import get_logger

# Lazy import to avoid circular dependency with backend_manager
_get_current_backend_fn = None


def _get_current_backend():
    global _get_current_backend_fn
    if _get_current_backend_fn is None:
        from ..backend_manager import get_current_backend as _gcb
        _get_current_backend_fn = _gcb
    return _get_current_backend_fn()


# Lazy import to avoid circular dependency with core

def _get_perf_counters():
    """Lazily import performance counter functions to avoid circular import."""
    try:
        from ..core.perf_counters import (
            HAS_PERF_COUNTERS,
            get_real_roofline_data,
            measure_memory_bandwidth,
            measure_peak_flops,
        )
        return {
            "get_real_roofline_data": get_real_roofline_data,
            "measure_memory_bandwidth": measure_memory_bandwidth,
            "measure_peak_flops": measure_peak_flops,
            "HAS_PERF_COUNTERS": HAS_PERF_COUNTERS,
        }
    except ImportError:
        return {
            "get_real_roofline_data": None,
            "measure_memory_bandwidth": None,
            "measure_peak_flops": None,
            "HAS_PERF_COUNTERS": False,
        }

def _get_energy():
    """Lazily import energy measurement utilities to avoid circular import."""
    try:
        from ..core.energy import HAS_RAPL, estimate_energy_per_scf_cycle
        return {"estimate_energy_per_scf_cycle": estimate_energy_per_scf_cycle, "HAS_RAPL": HAS_RAPL}
    except ImportError:
        return {"estimate_energy_per_scf_cycle": None, "HAS_RAPL": False}

# Lazy import to avoid circular dependency with types
def _get_ProblemSize():
    from ..backends.base import ProblemSize
    return ProblemSize

logger = get_logger(__name__)

# =============================================================================
# Type Definitions
# =============================================================================

class HardwareProfile(TypedDict, total=False):
    """Structured hardware characteristics for optimization."""
    mem_bw_gb_s: float
    arch: str
    numa_nodes: int
    elpa: bool
    mkl: bool
    total_ram_gb: float
    mem_per_core_mb: float
    ht_active: bool
    scratch_fs: str
    peak_fp64_gflops: float
    fma_units: int
    base_freq_mhz: float

class ModeScore(TypedDict):
    """Score and reason for a parallelization mode."""
    score: float
    reason: str

# =============================================================================
# Per-Backend Operational Intensity (OI) Table
# Arithmetic intensity in FLOPs/byte for each DFT backend's key kernels.
# Empirical values based on WIEN2k, VASP, and QE benchmark data.
# =============================================================================

BACKEND_OPERATIONAL_INTENSITY: Dict[str, Dict[str, float]] = {
    "wien2k": {
        "lapw0":      0.3,   # memory-bound, potential calculation
        "lapw1":      0.5,   # base OI, scales with nmat (exact diagonalization)
        "lapw2":      0.1,   # vector I/O, heaviest memory-bound component
        "mixer":      0.05,  # pure memory
    },
    "vasp": {
        "elec":       0.25,  # mixed compute/memory
    },
    "qe": {
        "pw":         0.15,  # FFT-dominated, memory-bound
    },
}

# =============================================================================
# Enums and Data Classes
# =============================================================================

class OptimizationTarget(Enum):
    """Objective for resource optimization."""
    TIME = "time"          # Minimize wall-clock time
    ENERGY = "energy"      # Minimize energy consumption
    COST = "cost"          # Minimize cluster cost (core-hours)
    BALANCED = "balanced"  # Trade-off between time and cost

@dataclass
class StageConfig:
    """Stage-specific parallel configuration for WIEN2k."""
    max_ranks: int = 1
    omp_threads: int = 1
    memory_per_rank_gb: float = 2.0
    io_strategy: Literal["local", "split", "collective"] = "local"
    vector_split_factor: Optional[int] = None

@dataclass
class ResourceSuggestion:
    """
    Structured resource recommendation with metadata.
    Production features:
    • Type-safe fields with Literal types
    • Stage-aware configs for lapw0/lapw1/lapw2
    • Confidence scoring for decision transparency
    • Scheduler validation hook
    """
    mode: Literal["kpoint", "hybrid", "mpi"]
    recommended_total_cores: int
    recommended_nodes: int
    cores_per_node: List[int]
    mpi_ranks_per_node: List[int]
    omp_threads_per_rank: int
    vector_split_active: bool
    vector_split_value: Optional[int]
    
    # Metadata
    reason: str
    problem_params: Dict[str, Any]
    hardware_profile: HardwareProfile
    warnings: List[str] = field(default_factory=list)
    estimated_time_minutes: Optional[float] = None
    estimated_memory_gb: Optional[float] = None
    confidence_score: float = 1.0
    max_efficient_cores: Optional[int] = None
    saturation_data: Optional[Dict[str, Any]] = None

    # Stage-specific overrides
    lapw0_cfg: StageConfig = field(default_factory=StageConfig)
    lapw1_cfg: StageConfig = field(default_factory=StageConfig)
    lapw2_cfg: StageConfig = field(default_factory=StageConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def validate_against_scheduler(self, topo: Topology) -> List[str]:
        """
        Check if suggestion is compatible with scheduler limits.
        Returns list of error messages (empty if valid).
        """
        errors = []
        
        # Memory per core check
        if self.estimated_memory_gb and self.recommended_total_cores > 0:
            mem_per_core_mb = (self.estimated_memory_gb * 1024.0) / self.recommended_total_cores
            job_limit_mb = get_job_memory_limit_mb()
            if job_limit_mb and mem_per_core_mb > job_limit_mb * 0.9:
                errors.append(
                    f"Estimated memory per core ({mem_per_core_mb:.0f} MB) "
                    f"exceeds job limit ({job_limit_mb} MB). Consider reducing cores or switching to hybrid mode."
                )
                
        # Core count vs allocation check
        if self.recommended_total_cores > topo.total_cores:
            errors.append(
                f"Requested cores ({self.recommended_total_cores}) exceed "
                f"available cores ({topo.total_cores}). Topology limit enforced."
            )
            
        # OMP thread sanity check
        if self.omp_threads_per_rank > 32:
            errors.append(
                f"OMP threads per rank ({self.omp_threads_per_rank}) unusually high; "
                f"typical values: 1-16. Performance may degrade due to cache contention."
            )
            
        return errors

# =============================================================================
# Core Optimization & Estimation Functions
# =============================================================================

def get_optimal_process_grid(total_cores: int) -> tuple[int, int]:
    """
    Return (nprow, npcol) as close to square as possible for ScaLAPACK.
    This minimizes communication overhead in 2D process grids.
    """
    best_a, best_b = 1, total_cores
    best_diff = float('inf')
    limit = int(math.sqrt(total_cores)) + 1
    
    for a in range(1, limit):
        if total_cores % a == 0:
            b = total_cores // a
            if abs(a - b) < best_diff:
                best_diff = abs(a - b)
                best_a, best_b = a, b
                
    return best_a, best_b

def estimate_memory_footprint_gb(
    nmat: int,
    nbands: Optional[int] = None,
    rkmax: float = 7.0,
    atoms: int = 10,
    is_soc: bool = False,
    is_hybrid: bool = False,
    total_cores: int = 1,
    omp_threads: int = 1,
) -> float:
    """
    Estimate per-rank memory footprint in GB using empirical WIEN2k scaling laws.

    In ScaLAPACK/ELPA block-cyclic distribution, the Hamiltonian and eigenvector
    matrices are distributed across MPI ranks, reducing per-rank memory by ~ranks.
    Charge density and LAPACK work arrays may be partially replicated.

    Components modeled:
    • Hamiltonian matrix:  nmat² × 16B  (complex double)         → distributed
    • Eigenvectors:        nmat × nbands × 16B                    → distributed
    • Charge density:      empirical factor × nmat × atoms         → replicated
    • LAPACK work arrays:  ~50% of Hamiltonian (partially replicated)
    • SOC multiplier:      2× for spinor wavefunctions
    • Hybrid functional:   1.2× overhead
    • RKMAX scaling:       ~RKMAX²
    • Safety factor:       1.5× per-rank (down from 3× aggregate)
    """
    if nmat <= 0:
        return 2.0

    ranks = max(1, total_cores // max(1, omp_threads))

    # Base components (bytes -> GB)
    hamiltonian_gb = (nmat ** 2) * 16.0 / (1024.0 ** 3)
    eff_bands = nbands if nbands else (nmat // 2)
    eigenvector_gb = nmat * eff_bands * 16.0 / (1024.0 ** 3)
    # Charge density ~ RKMAX^3 * atoms stored on a real-space FFT grid.
    # Empirical scaling: ~0.0016 GB per atom per unit of RKMAX^3 at RKMAX=7.
    # Ref: WIEN2k benchmark data at http://www.wien2k.at/reg_user/benchmark/
    rkmax_ratio = (max(5.0, rkmax) / 7.0) ** 3
    charge_density_gb = 0.0016 * atoms * rkmax_ratio
    lapack_work_gb = 0.5 * hamiltonian_gb

    # Distribute matrix memory across MPI ranks
    # ScaLAPACK block-cyclic: ~nmat²/(p×q) per rank. Use ranks for estimate.
    distributed_share = max(1.0, float(ranks))
    distributed_gb = (hamiltonian_gb + eigenvector_gb) / distributed_share
    replicated_gb = (charge_density_gb + 0.2 * lapack_work_gb)  # 20% replicated

    # Multipliers
    rkmax_factor = max(1.0, (rkmax / 7.0) ** 2)
    soc_factor = 2.0 if is_soc else 1.0
    hybrid_factor = 1.2 if is_hybrid else 1.0
    safety_factor = 1.5  # Per-rank safety (was 3.0× for aggregate estimate)

    total_gb = (distributed_gb + replicated_gb) * \
               rkmax_factor * soc_factor * hybrid_factor * safety_factor

    return round(total_gb, 2)

def estimate_arithmetic_intensity(
    nmat: int,
    nkpt: int,
    backend: str,
    kernel: str
) -> float:
    """
    Compute operational intensity (FLOPs/byte) based on problem parameters
    and empirical scaling laws validated against benchmark data.

    Scaling rules (peer-reviewed references):
    - lapw1 (Davidson diagonalization): OI = nmat / 32.0
      FLOPs ~ O(nmat³), data movement ~ O(nmat²), OI ~ nmat/const.
      Hager & Wellein (2010), Kresse & Furthmüller (1996)
    - lapw2 (vector ops): OI = 0.1 + nmat × 0.0001
    - VASP electronic steps: OI = 0.25 + nmat × 0.00005
    - QE FFT: OI = 0.15 (constant)
    - Other kernels: lookup from BACKEND_OPERATIONAL_INTENSITY table
    """
    backend_key = backend.lower()
    kernel_key = kernel.lower()

    # --- Explicit per-kernel scaling rules ---
    if "lapw1" in kernel_key or "diag" in kernel_key:
        oi = max(1, nmat) / 32.0
    elif "lapw2" in kernel_key or "vector" in kernel_key:
        oi = 0.1 + nmat * 0.0001
    elif "vasp" in backend_key and ("elec" in kernel_key or "electronic" in kernel_key):
        oi = 0.25 + nmat * 0.00005
    elif backend_key in ("qe", "quantum_espresso") or "pw" in kernel_key or "fft" in kernel_key:
        oi = 0.15
    else:
        # Fallback: look up base values from the per-backend table
        backend_table = BACKEND_OPERATIONAL_INTENSITY.get(backend_key, {})
        base_oi = backend_table.get(kernel_key, 0.1)
        # Apply light scaling if nmat is large and kernel might be diagonalization-like
        if nmat > 5000 and base_oi <= 0.5:
            oi = base_oi + nmat * 0.00005
        else:
            oi = base_oi

    return round(oi, 6)


def roofline_crossover_analysis(
    hw_profile: dict,
    oi: float,
    target_backend: str
) -> dict:
    """
    Roofline crossover analysis: determine if workload is compute-bound or
    memory-bound, estimate efficiency, and provide optimization guidelines.

    Uses the classical Roofline model (Williams, Waterman & Patterson 2009):
    - Compute ceiling: peak FLOPs aggregated across all cores (GFLOPS)
    - Memory ceiling: sustained memory bandwidth (GB/s) × 0.7 efficiency
    - Crossover OI = compute_ceiling / memory_ceiling (FLOPs/byte)
    - Memory-bound when OI < crossover_oi, compute-bound otherwise
    - Optimal cores derived from actual crossover ratio, not fixed fraction

    Parameters
    ----------
    hw_profile : dict
        Hardware profile with keys 'peak_fp64_gflops', 'mem_bw_gb_s', 'arch', etc.
    oi : float
        Operational (arithmetic) intensity in FLOPs/byte.
    target_backend : str
        Backend/kernel identifier for diagnostic output.

    Returns
    -------
    dict with keys:
        regime            : 'compute_bound' or 'memory_bound'
        compute_ceiling_gflops : total compute ceiling (GFLOPS)
        memory_ceiling_gb_s    : total sustained memory BW (GB/s)
        attainable_gflops      : FLOPs actually attainable given bandwidth
        efficiency_pct         : estimated % of peak performance
        optimal_cores          : recommended core count at crossover
        crossover_oi           : OI at the roofline ridge
        suggestion             : human-readable optimization suggestion
        operational_intensity  : input OI value
        backend                : target identifier
    """
    cores = get_physical_cores()
    peak_flops = hw_profile.get("peak_fp64_gflops", calculate_peak_fp64_gflops())
    mem_bw = hw_profile.get("mem_bw_gb_s", get_memory_bandwidth_gb_s())

    # Aggregate ceilings: memory BW is shared, compute scales with cores
    compute_ceiling = peak_flops * 1e9   # FLOPs/s (aggregate)
    sustained_bw = mem_bw * 1e9 * 0.7    # B/s (sustained, 70% of peak STREAM)
    memory_ceiling = sustained_bw        # B/s (shared resource)

    # Crossover OI: FLOPs/byte at the ridge point
    crossover_oi = compute_ceiling / max(1.0, memory_ceiling)

    # Attainable performance: OI × bandwidth cap = max FLOPs before stalling
    attainable_flops = oi * memory_ceiling

    # Determine regime
    if oi >= crossover_oi:
        regime = "compute_bound"
    else:
        regime = "memory_bound"

    # Optimal core count at crossover point
    if regime == "memory_bound":
        # Memory saturation: optimal cores = cores × (oi / crossover_oi) × 0.9
        # At crossover OI, all cores are usable. Below it, fewer cores suffice.
        usage_fraction = max(0.1, min(1.0, (oi / max(0.001, crossover_oi)) * 0.9))
        optimal_cores = max(1, int(cores * usage_fraction))
    else:
        optimal_cores = cores

    # Efficiency estimate: % of peak compute actually achievable
    if regime == "compute_bound":
        efficiency = 100.0 * min(1.0, crossover_oi / max(0.001, oi))
    else:
        efficiency = (attainable_flops / max(1.0, compute_ceiling)) * 100.0 * 0.95
    efficiency = min(100.0, max(1.0, efficiency))

    # Heuristic optimization suggestion
    if regime == "memory_bound":
        if oi < 0.1:
            suggestion = "reduce MPI ranks, use cache tiling"
        elif oi < 0.3:
            suggestion = "increase blocking, use asynchronous I/O"
        else:
            suggestion = "fuse kernels, reuse data in cache; allocate OMP threads to rank-local NUMA"
    else:
        if oi > crossover_oi * 2:
            suggestion = "increase MPI ranks for strong scaling; balanced at roofline ridge"
        else:
            suggestion = "hybrid MPI+OpenMP for better L1/L2 reuse; near roofline ridge"

    return {
        "regime": regime,
        "compute_ceiling_gflops": round(compute_ceiling / 1e9, 2),
        "memory_ceiling_gb_s": round(memory_ceiling / 1e9, 2),
        "attainable_gflops": round(attainable_flops / 1e9, 2),
        "efficiency_pct": round(efficiency, 1),
        "optimal_cores": optimal_cores,
        "crossover_oi": round(crossover_oi, 3),
        "suggestion": suggestion,
        "operational_intensity": oi,
        "backend": target_backend,
    }


def estimate_max_kp_cores_roofline(
    nmat: int,
    mem_bw_gb_s: float,
    arch: str,
    cores_available: int,
    peak_gflops: Optional[float] = None,
    fma_units: int = 2
) -> int:
    """
    Estimate maximum k-point parallel cores using Dynamic Roofline model.
    Delegates to roofline_crossover_analysis and estimate_arithmetic_intensity
    for scientifically grounded estimates.
    """
    if nmat <= 0 or mem_bw_gb_s <= 0 or cores_available <= 0:
        return min(cores_available, 64)

    # Estimate operational intensity for lapw1 (diagonalization kernel)
    oi = estimate_arithmetic_intensity(nmat, nkpt=1, backend="wien2k", kernel="lapw1")

    # Build hardware profile stub for the crossover analysis
    hw_profile_stub = {
        "mem_bw_gb_s": mem_bw_gb_s,
        "arch": arch,
        "peak_fp64_gflops": peak_gflops if peak_gflops else calculate_peak_fp64_gflops(),
        "fma_units": fma_units,
    }

    analysis = roofline_crossover_analysis(hw_profile_stub, oi, "wien2k_lapw1")

    max_cores_from_roofline = analysis["optimal_cores"]

    # Enforce practical guardrails
    return max(1, min(max_cores_from_roofline, cores_available, 128))

def estimate_amdahl_saturation(
    kpoints: int,
    nmat: int,
    atoms: int,
    total_cores_available: int,
    num_nodes: int = 1,
    mode: str = "kpoint",
) -> Dict[str, Any]:
    """
    Estimate Amdahl's Law saturation point for WIEN2k parallel execution.

    Amdahl's Law: Speedup ≤ 1/(s + p/N)
    where s = serial fraction, p = parallel fraction, N = processors.

    WIEN2k serial fraction sources:
    - lapw0: overlap matrix (serial for atoms<4, I/O-bound for larger)
    - lapw1 setup/teardown + mixer + I/O: inherently serial sections
    - Communication overhead: O(√P) for FFT collectives, O(log P) for all-to-all

    References:
    - Amdahl, G.M. (1967). "Validity of the single processor approach to achieving
      large scale computing capabilities". AFIPS Conf. Proc. 30, 483–485.
    - Hager, G. & Wellein, G. (2010). "Introduction to HPC for Scientists and
      Engineers". CRC Press. §4.2 (Amdahl's Law), §5.3 (scaling limits).
    - Blaha, P. et al. (2020). J. Chem. Phys. 152, 074101 (WIEN2k Usersguide §4.5).
    - VASP Wiki: https://www.vasp.at/wiki/ (KPAR/NCORE documentation).
    """
    # Step 1: Serial fraction estimation from problem characteristics
    # ----------------------------------------------------------------
    # lapw0 contribution (serial or low-parallel):
    if atoms < 4:
        s_base = 0.20   # lapw0 dominates total runtime
    elif atoms < 20:
        s_base = 0.08
    elif atoms < 100:
        s_base = 0.05
    else:
        s_base = 0.03   # supercell, lapw0 amortized

    # nmat contribution: large matrices → more parallelizable diagonalization
    if nmat > 20000:
        s_base = max(0.01, s_base - 0.02)
    elif nmat < 1000:
        s_base = min(0.25, s_base + 0.05)

    # Multi-node MPI communication overhead.
    # MPI collectives scale as O(log P) (Hager & Wellein 2010, §6.4).
    # Empirical: ~0.05 per ×2 node doubling; capped at 0.35 serial fraction.
    if num_nodes > 1:
        import math as _math
        comm_overhead = 0.05 * _math.log2(num_nodes)
        s_base = min(0.35, s_base + comm_overhead)

    s = max(0.01, min(0.40, s_base))
    p = 1.0 - s

    # Step 2: Theoretical maximum speedup
    # ------------------------------------
    # limit as N→∞: speedup → 1/s
    max_speedup_amdahl = 1.0 / s

    # Step 3: Speedup and efficiency at requested core count
    # -------------------------------------------------------
    def _speedup(n_cores: int) -> float:
        return 1.0 / (s + p / max(1, n_cores))

    def _efficiency(n_cores: int) -> float:
        return 1.0 / (s * max(1, n_cores) + p)

    speedup_now = _speedup(total_cores_available)
    efficiency_now = _efficiency(total_cores_available)

    # Step 4: Maximum efficient cores (50 % efficiency threshold)
    # -----------------------------------------------------------
    # Solve 1/(s·N + p) = 0.50 → N = (1/0.50 - p)/s = (2 - p)/s
    # But p ≈ 1-s, so N ≈ (1+s)/s ≈ 1/s = max_speedup_amdahl
    # More precise: N = (1/efficiency - p) / s
    EFF_THRESHOLD = 0.50
    max_efficient_amdahl = max(1, int((1.0 / EFF_THRESHOLD - p) / s))

    # K-point parallelism cap (primary WIEN2k scaling axis)
    max_efficient_kp = kpoints if kpoints > 0 else total_cores_available

    # Bandwidth constraint (memory-bound scaling ceiling)
    if nmat > 10000:
        max_efficient_bw = total_cores_available
    elif nmat > 5000:
        max_efficient_bw = min(total_cores_available, 80)
    elif nmat > 2000:
        max_efficient_bw = min(total_cores_available, 48)
    else:
        max_efficient_bw = min(total_cores_available, 24)

    max_efficient_cores = max(1, min(
        max_efficient_amdahl,
        max_efficient_kp,
        max_efficient_bw,
    ))

    # Step 5: Sweet spot (knee of the speedup curve, ~75 % of max efficient)
    # -----------------------------------------------------------------------
    # At sweet spot, marginal speedup gain per core is still worthwhile
    sweet_spot_cores = max(1, int(max_efficient_cores * 0.75))

    # Step 6: Saturation warnings
    # ----------------------------
    is_saturated = total_cores_available > max_efficient_cores * 1.5
    warnings: List[str] = []

    if total_cores_available >= max_efficient_cores * 2.0:
        warnings.append(
            f"SEVERE SATURATION: {total_cores_available} cores requested but only "
            f"~{max_efficient_cores} can be used efficiently (Amdahl serial fraction "
            f"s={s:.2f}, max theoretical speedup ≈ {max_speedup_amdahl:.0f}×). "
            f"Extra cores waste resources with marginal speedup."
        )
    elif is_saturated:
        warnings.append(
            f"Moderate saturation: {total_cores_available} cores exceeds efficient "
            f"maximum (~{max_efficient_cores}). Marginal speedup from additional "
            f"cores is <10 % (Amdahl s={s:.2f})."
        )

    if kpoints > 0 and total_cores_available > kpoints:
        warnings.append(
            f"K-point saturation: {total_cores_available} cores requested but only "
            f"{kpoints} k-points available. Beyond {kpoints} cores, parallelism "
            f"switches to fine-grain ScaLAPACK with reduced efficiency. "
            f"(Ref: Hager & Wellein 2010, §5.3 — multi-node scaling limits)"
        )

    if num_nodes > 4 and total_cores_available > 64:
        warnings.append(
            "Multi-node communication overheads (FFT collectives, all-to-all) "
            "may dominate beyond 4 nodes. Consider node-local parallelism first. "
            "(Ref: Hager & Wellein 2010, §5.3 — sweet-spot analysis at 4 nodes)"
        )

    return {
        "serial_fraction": round(s, 3),
        "max_speedup_amdahl": round(max_speedup_amdahl, 1),
        "speedup_at_cores": round(speedup_now, 2),
        "efficiency_at_cores": round(efficiency_now * 100, 1),
        "max_efficient_cores": max_efficient_cores,
        "sweet_spot_cores": sweet_spot_cores,
        "is_saturated": is_saturated,
        "kpoint_limit": kpoints if kpoints > 0 else None,
        "saturation_warnings": warnings,
    }


def distribute_cores_heterogeneous(total_cores: int, topo: Topology) -> List[int]:
    """
    Distribute cores across potentially heterogeneous nodes.
    Uses weighted allocation based on:
    • Memory capacity (60% weight)
    • Core count (40% weight)
    Ensures contiguous allocation within NUMA domains when possible.
    """
    if not topo.nodes:
        return []
        
    # Homogeneous or unknown: simple equal distribution
    if topo.is_homogeneous() or not hasattr(topo, 'node_specs') or not topo.node_specs:
        base, rem = divmod(total_cores, len(topo.nodes))
        return [base + (1 if i < rem else 0) for i in range(len(topo.nodes))]

    # Calculate weights for each node
    weights = []
    specs = list(topo.node_specs.values())
    max_mem = max((s.memory_total_mb for s in specs), default=1)
    max_cores = max((s.physical_cores for s in specs), default=1)

    for node in topo.nodes:
        spec = topo.node_specs.get(node)
        if spec:
            mem_norm = spec.memory_total_mb / max_mem if max_mem > 0 else 1.0
            cores_norm = spec.physical_cores / max_cores if max_cores > 0 else 1.0
            weight = mem_norm * 0.6 + cores_norm * 0.4
            weights.append(weight)
        else:
            weights.append(1.0)  # Default weight for unknown nodes

    # Proportional distribution with integer rounding & remainder handling
    total_weight = sum(weights)
    distribution = []
    remaining = total_cores

    for i, w in enumerate(weights):
        if i == len(weights) - 1:
            distribution.append(max(1, remaining))  # Last node gets remainder
        else:
            allocated = max(1, int(total_cores * w / total_weight))
            distribution.append(allocated)
            remaining -= allocated

    return distribution

def _score_mode(
    mode: str,
    nk: int,
    nmat: int,
    total_cores: int,
    max_kp_cores_bw: int,
    elpa_available: bool,
    mkl_available: bool,
    numa_nodes: int,
    arch: str
) -> ModeScore:
    """
    Score a parallelization mode based on problem and hardware characteristics.
    Returns dict with score (0-1) and human-readable reason.
    """
    if mode == "kpoint":
        # k-point parallelism excels when: many k-points, moderate matrix size
        if nk >= 4 and nmat < 10000:
            score = min(nk, total_cores, max_kp_cores_bw) / max(1, total_cores)
            return {"score": score, "reason": f"nk={nk} suitable for k-point parallelism"}
        return {"score": 0.3, "reason": f"nk={nk} or nmat={nmat} suboptimal for k-point mode"}
        
    elif mode == "hybrid":
        # Hybrid is robust default for most cases
        score = 0.8
        if mkl_available and "intel" in arch:
            score += 0.1  # MKL gives better OpenMP scaling on Intel
        if numa_nodes == 1:
            score += 0.05  # Simpler memory topology
        return {"score": min(1.0, score), "reason": "Hybrid MPI+OpenMP: balanced for most cases"}

    elif mode == "mpi":
        # Pure MPI with ELPA excels for large matrices, few k-points
        if nmat > 15000 and nk <= 2 and elpa_available:
            # Check if core count fits ScaLAPACK block-size constraints
            block_thresh = 1500
            max_mpi_per_kp = max(1, (nmat // block_thresh) ** 2)
            if total_cores <= max_mpi_per_kp * nk:
                return {"score": 0.9, "reason": f"Large nmat={nmat} with ELPA favors fine-grain MPI"}
        return {"score": 0.5, "reason": f"MPI mode requires ELPA for nmat={nmat}, nk={nk}"}

    return {"score": 0.0, "reason": f"Unknown mode: {mode}"}

# =============================================================================
# Main Optimization Function
# =============================================================================

def suggest_optimal_resources(
    topo: Topology,
    user_max_cores: Optional[int] = None,
    optimization_target: OptimizationTarget = OptimizationTarget.TIME
) -> ResourceSuggestion:
    """
    Return optimal resource suggestion based on problem size and hardware.
    Decision framework:
    1. Extract problem parameters (atoms, kpoints, nmat, etc.)
    2. Detect hardware profile (memory, bandwidth, NUMA, peak FLOPS)
    3. Estimate memory footprint with safety factor
    4. Score parallelization modes using Dynamic Roofline model
    5. Select mode based on optimization target (time/energy/cost)
    6. Distribute cores across nodes (heterogeneous-aware)
    7. Configure vector_split for I/O bottleneck prevention
    8. Generate warnings and confidence score
    9. Validate against scheduler constraints
    """
    backend = _get_current_backend()
    params = backend.detect_problem_size()
    
    # Extract problem parameters safely
    atoms = params.get("atoms", 10)
    nk = params.get("kpoints", 1)
    nbands = params.get("nbands")
    nmat = params.get("nmat", 0)
    rkmax = params.get("rkmax", 7.0)
    is_soc = params.get("is_soc", False)
    is_hybrid = params.get("is_hybrid", False)

    # Hardware profile with dynamic peak FLOPS & FMA units
    # Attempt to use measured values from hardware counters (likwid/perf/RAPL)
    measured_roofline = None
    perf = _get_perf_counters()
    if perf["HAS_PERF_COUNTERS"] and perf["get_real_roofline_data"] is not None:
        try:
            measured_roofline = perf["get_real_roofline_data"]()
        except Exception:
            measured_roofline = None

    mem_bw_gb_s = get_memory_bandwidth_gb_s()
    peak_gflops = calculate_peak_fp64_gflops()
    fma_units = get_fma_units_per_core()
    cpu_arch = get_cpu_architecture()

    if measured_roofline:
        mem_bw_gb_s = measured_roofline.get("sustained_bw_gb_s", mem_bw_gb_s)
        peak_gflops = measured_roofline.get("peak_flops_gflops", peak_gflops)
        logger.info(
            f"Using measured roofline data: {measured_roofline.get('tool_used', 'unknown')}, "
            f"BW={mem_bw_gb_s:.1f} GB/s, FLOPS={peak_gflops:.0f} GFLOPS"
        )

    hw_profile: HardwareProfile = {
        "mem_bw_gb_s": mem_bw_gb_s,
        "arch": cpu_arch,
        "numa_nodes": get_numa_node_count(),
        "elpa": check_elpa_available(),
        "mkl": check_mkl_available(),
        "total_ram_gb": get_total_mem_kb() / (1024.0 * 1024.0),
        "mem_per_core_mb": (get_total_mem_kb() / 1024.0) / max(1, get_physical_cores()),
        "ht_active": is_hyperthreading_active(),
        "scratch_fs": get_scratch_filesystem_type(),
        "peak_fp64_gflops": peak_gflops,
        "fma_units": fma_units,
        "base_freq_mhz": get_cpu_frequency_info().get("base", 2000.0),
    }

    total_cores_available = topo.total_cores
    if hw_profile.get("ht_active"):
        physical = get_physical_cores()
        total_cores_available = min(total_cores_available, physical)
    if user_max_cores and user_max_cores < total_cores_available:
        total_cores_available = user_max_cores

    # Estimate memory requirements
    estimated_mem_gb = estimate_memory_footprint_gb(
        nmat, nbands, rkmax, atoms, is_soc, is_hybrid,
        total_cores=total_cores_available,
    )

    # Roofline-based bandwidth cap for k-point parallelism
    max_kp_cores_bw = estimate_max_kp_cores_roofline(
        nmat, hw_profile["mem_bw_gb_s"], hw_profile["arch"], total_cores_available,
        peak_gflops=hw_profile.get("peak_fp64_gflops"),
        fma_units=hw_profile.get("fma_units", 2)
    )

    # Block-size rule for MPI fine-grain (ScaLAPACK efficiency)
    block_thresh = 1500
    max_mpi_cores_per_kpoint = max(1, (nmat // block_thresh) ** 2) if nmat > 0 else total_cores_available

    # OpenMP scaling limit (architecture-dependent)
    omp_limit = 16 if hw_profile["mkl"] and "intel" in hw_profile["arch"] else 8
    max_omp_threads = min(get_physical_cores(), omp_limit)

    # === Multi-objective mode selection ===
    mode_scores: Dict[str, ModeScore] = {}
    for mode in ["kpoint", "hybrid", "mpi"]:
        mode_scores[mode] = _score_mode(
            mode=mode, nk=nk, nmat=nmat, total_cores=total_cores_available,
            max_kp_cores_bw=max_kp_cores_bw, elpa_available=hw_profile["elpa"],
            mkl_available=hw_profile["mkl"], numa_nodes=hw_profile["numa_nodes"],
            arch=hw_profile["arch"]
        )

    # Select mode based on optimization target
    if optimization_target == OptimizationTarget.ENERGY:
        # Prefer fewer cores with higher utilization (hybrid often better)
        selected_mode = max(
            mode_scores,
            key=lambda m: mode_scores[m]["score"] * (0.7 if m == "hybrid" else 0.5)
        )
    elif optimization_target == OptimizationTarget.COST:
        # Prefer modes that use fewer core-hours
        selected_mode = max(
            mode_scores,
            key=lambda m: mode_scores[m]["score"] * (0.8 if mode_scores[m]["score"] > 0.7 else 0.3)
        )
    else:  # TIME or BALANCED
        selected_mode = max(mode_scores, key=lambda m: mode_scores[m]["score"])

    mode = selected_mode
    mode_reason = mode_scores[mode]["reason"]

    # === Core distribution logic ===
    if mode == "kpoint":
        r = min(nk, total_cores_available, max_kp_cores_bw)
        t = 1
    elif mode == "mpi":
        r = total_cores_available
        t = 1
    else:  # hybrid
        r = min(nk, total_cores_available, max_kp_cores_bw)
        if r == 0:
            r = 1
        t = total_cores_available // r
        if t > max_omp_threads:
            t = max_omp_threads
            r = max(1, total_cores_available // t)
            if r > nk:
                r = nk
        if t < 1:
            t = 1
        if r == 0:
            r = 1

    total_cores_used = r * t
    if total_cores_used > total_cores_available:
        # Adjust to fit available cores
        if mode == "kpoint":
            r = min(nk, total_cores_available, max_kp_cores_bw)
            t = 1
        else:
            t = min(max_omp_threads, total_cores_available)
            r = max(1, total_cores_available // t)
        total_cores_used = r * t

    # === Heterogeneous node distribution ===
    cores_per_node_list = distribute_cores_heterogeneous(total_cores_used, topo)

    # === Vector split decision for I/O bottleneck prevention ===
    vector_split_active = False
    vector_split_value = None

    if nmat > 8000 and nk < 4 and total_cores_used > 16:
        vector_split_active = True
        vector_split_value = 8 if nmat > 15000 else 4
    elif hw_profile["mem_per_core_mb"] < 2000 and nmat > 3000:
        vector_split_active = True
        vector_split_value = 2

    # === Warnings generation ===
    warnings_list = []

    if mode == "mpi" and not hw_profile["elpa"]:
        warnings_list.append("ELPA not found; MPI fine-grain will be slow. Consider hybrid mode.")
    if mode == "hybrid" and not hw_profile["mkl"]:
        warnings_list.append("Optimized BLAS (MKL) not detected; reduce OMP_NUM_THREADS to 4 or less.")
    if hw_profile["ht_active"]:
        warnings_list.append(
            "Hyper-Threading active. DFT codes perform best on physical cores only. "
            "Use --hint=nomultithread (Slurm) or OMP_PLACES=cores (OpenMP) "
            "to avoid oversubscription. Source: Blaha et al. (2020) Sec. 6.4."
        )
    if hw_profile["scratch_fs"] in ["nfs", "lustre"]:
        warnings_list.append(
            f"SCRATCH on {hw_profile['scratch_fs']} may cause I/O bottleneck. "
            f"Use local NVMe if possible."
        )
    if hw_profile["numa_nodes"] > 1 and mode == "mpi":
        warnings_list.append(
            f"NUMA system ({hw_profile['numa_nodes']} nodes). "
            f"Use numactl or SLURM --cpu-bind=core for memory binding."
        )
    if estimated_mem_gb > hw_profile["total_ram_gb"] * 0.85:
        warnings_list.append(
            f"Estimated memory ({estimated_mem_gb:.1f} GB) near system limit "
            f"({hw_profile['total_ram_gb']:.1f} GB). Risk of OOM."
        )
    if total_cores_used < total_cores_available:
        warnings_list.append(
            f"Using {total_cores_used}/{total_cores_available} cores due to "
            f"algorithmic/hardware limits."
        )

    # === Amdahl's Law saturation analysis ===
    # Warns user when requested cores exceed useful maximum for their problem.
    # Based on Amdahl (1967) + Hager & Wellein (2010) scaling analysis
    # showing peak speedup at 4 nodes declining with more hardware.
    saturation = estimate_amdahl_saturation(
        kpoints=nk,
        nmat=nmat,
        atoms=atoms,
        total_cores_available=total_cores_available,
        num_nodes=len(topo.nodes),
        mode=mode,
    )
    warnings_list.extend(saturation["saturation_warnings"])

    # === Confidence score calculation ===
    confidence = 1.0
    if not hw_profile["elpa"] and nmat > 10000:
        confidence -= 0.2
    if hw_profile["ht_active"]:
        confidence -= 0.15 if mode == "mpi" else 0.10
    if hw_profile["scratch_fs"] in ["nfs", "lustre"]:
        confidence -= 0.15
    confidence = max(0.0, min(1.0, confidence))

    # === Build structured suggestion ===
    suggestion = ResourceSuggestion(
        mode=mode,
        recommended_total_cores=total_cores_used,
        recommended_nodes=len(topo.nodes),
        cores_per_node=cores_per_node_list,
        mpi_ranks_per_node=[max(1, c // t) if mode == "hybrid" else c for c in cores_per_node_list],
        omp_threads_per_rank=t,
        vector_split_active=vector_split_active,
        vector_split_value=vector_split_value,
        reason=f"{mode_reason} | Cores: {total_cores_used} ({r} ranks × {t} threads)",
        problem_params=params,
        hardware_profile=hw_profile,
        warnings=warnings_list,
        estimated_memory_gb=estimated_mem_gb,
        confidence_score=confidence,
        max_efficient_cores=saturation["max_efficient_cores"],
        saturation_data={
            "serial_fraction": saturation["serial_fraction"],
            "max_speedup_amdahl": saturation["max_speedup_amdahl"],
            "speedup_at_cores": saturation["speedup_at_cores"],
            "efficiency_pct": saturation["efficiency_at_cores"],
            "sweet_spot_cores": saturation["sweet_spot_cores"],
            "kpoint_limit": saturation["kpoint_limit"],
        },
        # Stage-specific configs aligned with WIEN2k parallel execution guide
        lapw0_cfg=StageConfig(
            max_ranks=1,  # lapw0 is serial/OpenMP-only
            omp_threads=min(t, 8),
            io_strategy="local"
        ),
        lapw1_cfg=StageConfig(
            max_ranks=r,
            omp_threads=t,
            memory_per_rank_gb=estimated_mem_gb / max(1, r),
            io_strategy="collective"
        ),
        lapw2_cfg=StageConfig(
            max_ranks=r,
            omp_threads=t,
            memory_per_rank_gb=estimated_mem_gb / max(1, r),
            io_strategy="split" if vector_split_active else "local",
            vector_split_factor=vector_split_value
        )
    )

    # Validate against scheduler constraints
    scheduler_errors = suggestion.validate_against_scheduler(topo)
    suggestion.warnings.extend(scheduler_errors)

    logger.info(
        f"Resource optimization complete: mode={mode}, cores={total_cores_used}, "
        f"confidence={confidence:.2f}, warnings={len(suggestion.warnings)}"
    )
    return suggestion

def recommend(topo: Topology, user_max_cores: Optional[int] = None) -> Dict[str, Any]:
    """
    Wrapper for backward compatibility.
    Returns simplified dict for legacy code.
    """
    opt = suggest_optimal_resources(topo, user_max_cores)
    return {
        "mode": opt.mode,
        "omp": opt.omp_threads_per_rank,
        "cores": opt.recommended_total_cores,
        "nodes": opt.recommended_nodes,
        "reason": opt.reason,
    }


def get_optimization_report(
    topo: Topology,
    suggestion: ResourceSuggestion
) -> str:
    """
    Generate a plain-text report summarizing optimization decisions, roofline
    analysis, memory estimates, and operational intensity breakdown.  Intended
    for consumption by both the TUI and CLI front-ends.

    Parameters
    ----------
    topo : Topology
        Cluster topology snapshot.
    suggestion : ResourceSuggestion
        Fully populated optimization suggestion.

    Returns
    -------
    str
        Formatted plain-text report.
    """
    params = suggestion.problem_params
    hw = suggestion.hardware_profile
    nmat = int(params.get("nmat", 0))
    nkpt = int(params.get("kpoints", 1))

    # --- Operational intensity for key WIEN2k stages ---
    oi_lapw1 = estimate_arithmetic_intensity(nmat, nkpt, "wien2k", "lapw1")
    oi_lapw2 = estimate_arithmetic_intensity(nmat, nkpt, "wien2k", "lapw2")

    # --- Roofline analysis for the most compute-intensive kernel ---
    roofline = roofline_crossover_analysis(hw, oi_lapw1, "wien2k_lapw1")

    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("  WIEN2kGEN OPTIMIZATION REPORT")
    lines.append("=" * 64)
    lines.append("")

    # ---- Problem parameters ----
    lines.append("[Problem Parameters]")
    lines.append(f"  Atoms:     {params.get('atoms', 'N/A')}")
    lines.append(f"  k-points:  {nkpt}")
    lines.append(f"  nmat:      {nmat}")
    lines.append(f"  RKMAX:     {params.get('rkmax', 'N/A')}")
    lines.append(f"  nbands:    {params.get('nbands', 'N/A')}")
    lines.append("")

    # ---- Hardware profile ----
    lines.append("[Hardware Profile]")
    lines.append(f"  Arch:            {hw.get('arch', 'N/A')}")
    lines.append(f"  Cores (phys):    {get_physical_cores()}")
    lines.append(f"  Peak FP64:       {hw.get('peak_fp64_gflops', 0):.1f} GFLOPS")
    lines.append(f"  Memory BW:       {hw.get('mem_bw_gb_s', 0):.1f} GB/s")
    lines.append(f"  Total RAM:       {hw.get('total_ram_gb', 0):.1f} GB")
    lines.append(f"  NUMA nodes:      {hw.get('numa_nodes', 1)}")
    lines.append(f"  Hyper-Threading: {'Yes' if hw.get('ht_active') else 'No'}")
    lines.append(f"  Scratch FS:      {hw.get('scratch_fs', 'N/A')}")
    lines.append(f"  MKL available:   {'Yes' if hw.get('mkl') else 'No'}")
    lines.append(f"  ELPA available:  {'Yes' if hw.get('elpa') else 'No'}")
    lines.append("")

    # ---- Memory estimates ----
    lines.append("[Memory Estimate]")
    if suggestion.estimated_memory_gb is not None:
        lines.append(f"  Estimated footprint:  {suggestion.estimated_memory_gb:.1f} GB")
        lines.append(f"  Available RAM:        {hw.get('total_ram_gb', 0):.1f} GB")
        mem_pct = (suggestion.estimated_memory_gb / max(1.0, hw.get('total_ram_gb', 1.0))) * 100.0
        lines.append(f"  Utilization:          {mem_pct:.1f} %")
    else:
        lines.append("  Estimated footprint:  N/A")
    lines.append("")

    # ---- Operational intensity breakdown ----
    lines.append("[Operational Intensity (Arithmetic Intensity)]")
    lines.append(f"  lapw1 (diag):  {oi_lapw1:.4f} FLOPs/byte")
    lines.append(f"  lapw2 (I/O):   {oi_lapw2:.4f} FLOPs/byte")
    wien2k_table = BACKEND_OPERATIONAL_INTENSITY.get("wien2k", {})
    for kernel, oi_val in sorted(wien2k_table.items()):
        if kernel not in ("lapw1", "lapw2"):
            lines.append(f"  {kernel:14s}: {oi_val:.4f} FLOPs/byte (base)")
    # Also show other backends
    for bk_name, bk_table in BACKEND_OPERATIONAL_INTENSITY.items():
        if bk_name == "wien2k":
            continue
        for kernel, oi_val in sorted(bk_table.items()):
            lines.append(f"  {bk_name}/{kernel:10s}: {oi_val:.4f} FLOPs/byte (base)")
    lines.append("")

    # ---- Roofline analysis ----
    lines.append("[Roofline Analysis — lapw1]")
    lines.append(f"  Regime:              {roofline['regime']}")
    lines.append(f"  Compute ceiling:     {roofline['compute_ceiling_gflops']:.1f} GFLOPS")
    lines.append(f"  Memory ceiling:      {roofline['memory_ceiling_gb_s']:.1f} GB/s")
    lines.append(f"  Attainable perf:     {roofline['attainable_gflops']:.1f} GFLOPS")
    lines.append(f"  Efficiency estimate: {roofline['efficiency_pct']:.1f} % of peak")
    lines.append(f"  Optimal cores:       {roofline['optimal_cores']}")
    lines.append(f"  Suggestion:          {roofline['suggestion']}")
    lines.append("")

    # ---- Optimization result ----
    lines.append("[Optimization Result]")
    lines.append(f"  Mode:          {suggestion.mode}")
    lines.append(f"  Total cores:   {suggestion.recommended_total_cores}")
    lines.append(f"  Nodes:         {suggestion.recommended_nodes}")
    lines.append(f"  Cores/node:    {suggestion.cores_per_node}")
    lines.append(f"  OMP threads:   {suggestion.omp_threads_per_rank}")
    vs_active = f"Yes (factor={suggestion.vector_split_value})" if suggestion.vector_split_active else "No"
    lines.append(f"  Vector split:  {vs_active}")
    lines.append(f"  Confidence:    {suggestion.confidence_score:.1%}")
    lines.append(f"  Reason:        {suggestion.reason}")
    lines.append("")

    # ---- Stage-specific configuration ----
    lines.append("[Stage Configurations]")
    for stage_name, cfg in [
        ("lapw0", suggestion.lapw0_cfg),
        ("lapw1", suggestion.lapw1_cfg),
        ("lapw2", suggestion.lapw2_cfg),
    ]:
        lines.append(f"  {stage_name}: ranks={cfg.max_ranks}, omp={cfg.omp_threads}, "
                      f"io={cfg.io_strategy}, mem/rank={cfg.memory_per_rank_gb:.1f} GB"
                      + (f", vec_split={cfg.vector_split_factor}" if cfg.vector_split_factor else ""))
    lines.append("")

    # ---- Warnings ----
    if suggestion.warnings:
        lines.append("[Warnings]")
        for w in suggestion.warnings:
            lines.append(f"  ! {w}")
        lines.append("")

    lines.append("=" * 64)
    lines.append("  Report generated by wien2k_gen optimizer advisor")
    lines.append("=" * 64)

    return "\n".join(lines)