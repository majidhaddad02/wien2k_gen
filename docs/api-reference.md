# API Reference

## Table of Contents

1. [Core Modules](#core-modules)
   - [hardware](#wien2k_gencorehardware)
   - [topology](#wien2k_gencoretopology)
   - [scheduler](#wien2k_gencorescheduler)
   - [case_parser](#wien2k_gencorecase_parser)
   - [workflow_executor](#wien2k_gencoreworkflow_executor)
2. [Optimization Modules](#optimization-modules)
   - [advisor](#wien2k_genoptimizeradvisor)
   - [bayesian](#wien2k_genoptimizerbayesian)
   - [parallel](#wien2k_genoptimizerparallel)
   - [monitor](#wien2k_genoptimizermonitor)
   - [gpu_detector](#wien2k_genoptimizergpu_detector)
3. [ML Modules](#ml-modules)
   - [gnn_kpoint_predictor](#wien2k_genmlgnn_kpoint_predictor)
4. [Backend Modules](#backend-modules)
   - [wien2k](#wien2k_genbackendswien2k)
5. [Type Definitions](#type-definitions)

---

## Core Modules

### `wien2k_gen.core.hardware`

```python
from wien2k_gen.core.hardware import (
    get_physical_cores,
    get_logical_cores,
    is_hyperthreading_active,
    get_total_mem_kb,
    get_job_memory_limit_mb,
    get_numa_topology_detailed,
    get_cache_topology,
    get_cpu_frequency_info,
    get_cpu_architecture,
    get_cpu_generation,
    get_system_type,
    get_scratch_filesystem_type,
    get_interconnect_info,
    get_memory_bandwidth_gb_s,
    is_containerized,
    check_elpa_available,
    check_mkl_available,
    get_hardware_profile,
)
```

#### `get_physical_cores() -> int`
Detect physical CPU cores using `lscpu` JSON or `/proc/cpuinfo`.

#### `get_cpu_architecture() -> str`
Returns: `"xeon"`, `"epyc"`, `"amd_ryzen"`, `"intel_consumer"`, `"arm_neoverse"`, `"arm"`, `"unknown"`.

#### `get_cpu_generation() -> str`
Returns canonical generation: `"Xeon_SapphireRapids"`, `"EPYC_Genoa"`, `"EPYC_Milan"`, `"Neoverse_N1"`, `"Graviton3"`, etc.

#### `get_system_type() -> str`
Returns: `"laptop"`, `"workstation"`, `"compute_node"`, `"cluster"`, `"unknown"`.

#### `get_interconnect_info() -> List[Dict]`
```python
[{
    "type": "infiniband",
    "provider": "mlx5",
    "speed_gbps": 100.0,
    "latency_ns": 1.0,
    "active_rate_gbps": 100.0,
}]
```

#### `get_memory_bandwidth_gb_s() -> float`
Measures memory bandwidth from sysfs NUMA counters with STREAM benchmark fallback. Warns if < 50 GB/s.

#### `is_hyperthreading_active() -> bool`
True if logical cores > physical cores.

#### `get_hardware_profile() -> Dict`
Aggregate hardware summary with all fields above.

---

### `wien2k_gen.core.topology`

```python
from wien2k_gen.core.topology import (
    Topology, NUMANode, NodeSpec, GPUInfo,
    factorize_blacs_grid, blacs_grid_quality,
)
```

#### `Topology`
```python
Topology(
    nodes: List[str],
    cores_per_node: List[int],
    env_type: str,
    scheduler_hints: dict,
)
```

Key methods:
- `topo.total_cores` → int
- `topo.get_mpi_binding_hints()` → dict (openmpi, intel_mpi, mpich, mvapich, srun)
- `topo.split_load_balanced(total_cores: int)` → list of (node, cores) tuples
- `topo.get_optimal_mpi_distribution(mode: str)` → core distribution

#### `factorize_blacs_grid(total_cores: int) -> Tuple[int, int]`
Returns optimal 2D BLACS processor grid for ScaLAPACK/ELPA.

#### `blacs_grid_quality(rows: int, cols: int) -> float`
Quality score for BLACS grid (0-1). Penalizes 1D grids (Marek et al. 2014).

---

### `wien2k_gen.core.scheduler`

```python
from wien2k_gen.core.scheduler import detect, SchedulerHints, auto_detect_memory
```

#### `detect(max_cores=None, force_refresh=False) -> Topology`
Auto-detects scheduler (SLURM → PBS → LSF → SGE → local) and returns configured Topology.

#### `auto_detect_memory() -> str`
Returns memory string for the current scheduler (e.g., `"128G"` for SLURM).

---

### `wien2k_gen.core.case_parser`

```python
from wien2k_gen.core.case_parser import (
    CaseFileParser, CaseData, LDAUData,
    parse_case_directory, check_struct_quality,
    parse_crystal_structure, calculate_nn_distances,
    optimize_rmt, check_rmt_overlaps,
    recommend_final_rmt, generate_rmt_report,
)
```

#### `CaseData`
```python
@dataclass
class CaseData:
    case_name: str
    atoms: int
    atoms_inequiv: int
    kpoints: int
    nmat: int
    nbands: Optional[int]
    rkmax: float
    lmax: int
    gmax: float
    fft_nx: int; fft_ny: int; fft_nz: int
    is_soc: bool
    is_hybrid: bool
    is_spin_polarized: bool
    is_lda_u: bool
    is_eece: bool
    has_forces: bool
    ldau: LDAUData
    volume_bohr3: float
    lattice_vectors: list
    scf_iterations: int
    fermi_energy_ry: float
    total_energy_ry: float
    wien2k_version: str
```

#### `LDAUData`
```python
@dataclass
class LDAUData:
    u_ry: List[float]
    j_ry: List[float]
    ueff_ry: List[float]
    l_orbital: List[int]
    atoms: List[int]
    double_counting: str  # "AMF", "FLL", or "SIC"
    file_present: bool
```

#### `CaseFileParser`
```python
parser = CaseFileParser(Path("/path/to/case"))
data: CaseData = parser.parse_all()

# Parse individual files
nmat = CaseFileParser.parse_scf(Path("case.scf"))
in1  = CaseFileParser.parse_in1(Path("case.in1"))
ldau = CaseFileParser.parse_inm(Path("case.inm"))
```

#### `check_struct_quality(struct_path: Path) -> Dict[str, Any]`
Validates `.struct` file for RMT overlaps, small RMT warnings, and Wyckoff issues.
Returns `{"warnings": [...], "errors": [...], "rmt_data": [...]}`.

**RMT overlap thresholds:**
- overlap > 30% → errors (critical)
- overlap > 10% → warnings
- overlap > 0% → marginal warning

#### `parse_crystal_structure(struct_path: Path) -> Dict[str, Any]`
Full crystal structure parsing. Returns lattice, atoms (with coordinates, RMT, Z), spacegroup.

#### `calculate_nn_distances(structure: Dict[str, Any]) -> Dict[int, float]`
Nearest-neighbor distances via 3×3×3 supercell periodic image search. Returns atom_index → nn_distance (bohr).

#### `optimize_rmt(nn_distances, reduction_factor=0.95, min_rmt=2.5, max_rmt=4.0) -> Dict[int, float]`
Optimal RMT = `reduction_factor × (nn_distance / 2)`, clamped to [min_rmt, max_rmt].

#### `check_rmt_overlaps(rmts, structure, overlap_warning=0.95, overlap_critical=1.00) -> List[Dict]`
Detects RMT sphere overlaps. Returns list with atom_i, atom_j, rmt values, distance, overlap ratio, severity.

#### `recommend_final_rmt(optimal_rmts, overlaps, structure) -> Dict[int, float]`
Adjusts optimal RMT values proportionally to eliminate critical overlaps.

#### `generate_rmt_report(final_rmts, overlaps, nn_distances, structure) -> str`
Human-readable RMT optimization report with per-atom table and warnings.

---

### `wien2k_gen.core.workflow_executor`

```python
from wien2k_gen.core.workflow_executor import (
    WorkflowExecutor,
    detect_system_type,
    calculate_optimal_q0,
    select_mixing_strategy,
    restarted_pulay_mixing,
)
```

#### `detect_system_type(case_dir=".") -> str`
Detects system type from `.scf` band gap. Returns `"metal"`, `"semiconductor"`, `"insulator"`, or `"unknown"`.

**Thresholds:**
- gap > 0.5 eV → insulator
- 0.1 < gap ≤ 0.5 eV → semiconductor
- gap ≤ 0.1 eV or DOS(E_F) > 0 → metal

#### `calculate_optimal_q0(system_type, lattice_constant=10.0) -> float`
Smart Kerker q0 based on system type (Winkelmann et al. 2020, Phys. Rev. B 102, 195138).

**Formulas:**
- metal: `q₀ = 0.4 × (2π/a)`
- semiconductor: `q₀ = 0.15 × (2π/a)`
- insulator: `q₀ = 0.05 × (2π/a)`

#### `select_mixing_strategy(n_atoms: int, is_metallic: bool) -> str`
Selects mixing strategy based on system size and type.
Returns `"restarted_pulay_kerker"`, `"restarted_pulay"`, or `"broyden"`.

**Decision matrix (Pratapa & Suryanarayana, Chem. Phys. Lett. 635, 2015):**
- large (>50 atoms) + metal → restarted_pulay_kerker
- large (>50 atoms) + non-metal → restarted_pulay
- small (≤50 atoms) → broyden

#### `restarted_pulay_mixing(case_name="case", history_size=7, regularization=1e-10) -> None`
Implements restarted Pulay mixing for large systems. Writes `.pulay_history` and `.mixing_strategy` files.
Builds overlap matrix S_ij = ⟨R_i|R_j⟩ with Tikhonov regularization.

---

## Optimization Modules

### `wien2k_gen.optimizer.advisor`

```python
from wien2k_gen.optimizer.advisor import (
    suggest_optimal_resources,
    recommend,
    OptimizationTarget,
    ResourceSuggestion,
    estimate_memory_footprint_gb,
    estimate_amdahl_saturation,
    get_optimization_report,
    roofline_crossover_analysis,
)
```

#### `suggest_optimal_resources(topo, user_max_cores=None, optimization_target=OptimizationTarget.TIME) -> ResourceSuggestion`
Main optimization function. Returns ResourceSuggestion with `mode`, `recommended_total_cores`, `recommended_nodes`, `mpi_ranks_per_node`, `omp_threads_per_rank`, `warnings`, `confidence_score`.

#### `estimate_amdahl_saturation(kpoints, nmat, atoms, total_cores, num_nodes, mode="mpi") -> dict`
```python
{
    "serial_fraction": float,
    "max_speedup_amdahl": float,
    "speedup_at_cores": float,
    "efficiency_at_cores": float,
    "max_efficient_cores": int,
    "sweet_spot_cores": int,
    "is_saturated": bool,
    "saturation_warnings": List[str],
}
```

#### `estimate_memory_footprint_gb(nmat, nbands=None, rkmax=7.0, atoms=10, is_soc=False, is_hybrid=False, total_cores=1) -> float`
Memory requirement estimate in GB.

#### `roofline_crossover_analysis(peak_gflops, bandwidth_gb_s, operational_intensity) -> Dict`
Roofline model analysis with compute/memory-bound identification and crossover point.

---

### `wien2k_gen.optimizer.bayesian`

```python
from wien2k_gen.optimizer.bayesian import (
    matern_kernel,
    rbf_kernel_ard,
    compute_expected_improvement,
    compute_q_expected_improvement,
    latin_hypercube_sampling,
    add_physics_priors,
    define_search_space,
    bayesian_optimize_scf_params,
    load_warm_start_history,
    save_bo_history,
)
```

#### `matern_kernel(x1, x2, length_scales, nu=2.5) -> np.ndarray`
Matérn ν=2.5 kernel: `(1 + √5·r + 5r²/3) × exp(-√5·r)`. Preferred over RBF for non-smooth SCF convergence surfaces (Snoek et al. 2012, NIPS 25, 2951–2959).

#### `rbf_kernel_ard(x1, x2, length_scales) -> np.ndarray`
RBF kernel with Automatic Relevance Determination (per-dimension length scales).

#### `compute_expected_improvement(mu, sigma, best_y, xi=0.01) -> float`
Single-point Expected Improvement acquisition function. EI(x) = σ·(z·Φ(z) + φ(z)).

#### `compute_q_expected_improvement(mu, var, best_y, q=4, xi=0.01, n_mc_samples=500) -> np.ndarray`
q-batch Expected Improvement for parallel evaluation of up to q points simultaneously (Ginsbourger et al. 2010).

#### `latin_hypercube_sampling(bounds, n_samples, random_state=None) -> np.ndarray`
Uniform search space coverage for BO initialisation (McKay et al. 1979).

#### `add_physics_priors(structure, nmat=0, is_soc=False, is_metallic=False) -> Dict`
Returns parameter constraints based on element types: RKMAX ≥ 7.0 for O/F/N, mixing ≤ 0.3 for metals, kpt density ≥ 1000 for metals.

#### `define_search_space(structure) -> Dict`
Returns 5-parameter search space bounds: RKMAX [5,9], mixing [0.05,1], kpt_density [100,2000], GMAX [10,20], LMAX [8,12].

#### `bayesian_optimize_scf_params(structure, eval_objective, budget=20, initial_samples=10, kernel_type="matern", acquisition="EI", parallel_batch=4, warm_start=True, history_file=".bo_history.json") -> Dict`
Full Bayesian optimization loop with warm start, LHS initialisation, GP fitting, and EI/q-EI acquisition.

#### `load_warm_start_history(history_file=".bo_history.json") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]`
Loads previous BO results for warm starting.

#### `save_bo_history(history_file, evaluations) -> None`
Saves BO results to JSON for future warm starts.

---

### `wien2k_gen.optimizer.parallel`

```python
from wien2k_gen.optimizer.parallel import (
    recommend_elpa_solver,
    should_use_elpa,
    recommend_rkmax,
    recommend_gmax,
    recommend_io_strategy,
    recommend_weighted_kpoint_distribution,
    detect_numa_topology,
    numa_aware_kpoint_distribution,
    generate_numa_aware_machines,
    calculate_kpoint_weights,
    ffd_kpoint_distribution,
    calculate_balance_quality,
    round_robin_distribution,
    compare_distribution_methods,
    generate_ffd_machines,
    recommend_numa_strategy,
)
```

#### `recommend_elpa_solver(nmat, nkpt, is_soc=False, is_hybrid=False, num_cores=0) -> Optional[str]`
ELPA solver stage recommendation. Returns `"elpa2"`, `"elpa1"`, or `None`.

**Thresholds (based on WIEN2k benchmarks, wien2k.at/reg_user/benchmark/):**
- nmat > 8000 → elpa2
- nmat > 5000 + cores > 64 → elpa2
- nmat > 5000 or SOC+nmat>2000 → elpa2
- hybrid → elpa2
- nmat > 2000 → elpa1

#### `should_use_elpa(nmat, num_cores=0) -> bool`
Boolean ELPA recommendation with nmat < 5000 overhead warning.

#### `recommend_rkmax(atomic_numbers, calculation_type="scf", rmt_ratios=None, is_soc=False) -> float`
RKMAX recommendation with element hardness and SOC awareness.

**Hard element adjustments (Blaha JCP 2020):**
- O, F, N with small RMT → RKMAX ≥ 7.0
- SOC → RKMAX ≥ 7.0 + 0.5

#### `recommend_io_strategy(nmat, nkpt, atoms, scratch_fs="tmpfs") -> Dict`
I/O optimization with nowrite_vector safety. **nowrite_vector is disabled by default** — requires checkpointing for safety.

#### `detect_numa_topology() -> Dict[str, Any]`
Detects NUMA topology via numactl → sysfs fallback. Returns `num_nodes`, `cores_per_node`, `total_cores`, `detected`.

#### `numa_aware_kpoint_distribution(kpoints, numa_nodes, cores_per_node, k_weights=None) -> Dict`
Round-robin k-point allocation across NUMA nodes with balance ratio computation.

#### `calculate_kpoint_weights(case_name) -> List[float]`
Reads k-point multiplicities/weights from `case.klist`. Normalizes to sum=1.0.

#### `ffd_kpoint_distribution(kpoint_weights, num_ranks) -> Dict`
First Fit Decreasing (greedy bin-packing) k-point assignment. Returns `rank_kpts`, `rank_loads`, `balance_ratio`, `efficiency`, `load_variance`.

#### `calculate_balance_quality(rank_loads) -> Dict[str, float]`
Metrics: `balance_ratio` (min_load/max_load), `efficiency`, `load_variance`, `load_std`.

#### `round_robin_distribution(kpoint_weights, num_ranks) -> Dict`
Baseline comparison distribution.

#### `compare_distribution_methods(kpoint_weights, num_ranks) -> Dict`
Compares FFD vs Round-Robin, selects winner, reports improvement percentage.

#### `generate_ffd_machines(rank_kpts, rank_loads, num_ranks, hostname_prefix="rank") -> str`
Generates `.machines` format with per-rank k-point listing.

---

### `wien2k_gen.optimizer.monitor`

```python
from wien2k_gen.optimizer.monitor import (
    diagnose_charge_sloshing_root_cause,
    create_scf_checkpoint,
    restore_from_checkpoint,
    estimate_remaining_walltime,
    calculate_checkpoint_interval,
    perform_incremental_checkpoint,
    cleanup_old_checkpoints,
    resume_from_checkpoint,
    detect_charge_sloshing,
    detect_scf_divergence,
)
```

#### `diagnose_charge_sloshing_root_cause(dayfile_content, case_name="case", struct_path=None) -> dict`
Diagnoses root cause of charge sloshing and returns targeted remediation actions.

**Returns:**
```python
{
    "root_cause": str,      # "metallic", "symmetry_breaking", "core_overlap", "mixing_too_aggressive"
    "confidence": float,    # 0.0–1.0
    "indicators": dict,     # per-category evidence scores
    "actions": List[dict],  # ordered remediation steps with params
}
```

#### `create_scf_checkpoint(case_name, label="") -> str`
Saves SCF checkpoint: `.clmval`, `.clmsum`, `.broyd*` to timestamped directory.

#### `restore_from_checkpoint(case_name, checkpoint_dir=None) -> bool`
Restores SCF state from most recent checkpoint.

#### `estimate_remaining_walltime(job_id, scheduler="slurm") -> Dict`
Reads walltime from SLURM `scontrol` or PBS `qstat -f`. Returns `walltime_limit_sec`, `elapsed_sec`, `remaining_sec`, `remaining_pct`.

#### `calculate_checkpoint_interval(remaining_time_sec, time_per_cycle_sec=300.0) -> int`
Adaptive checkpoint interval (Daly 2006, J. Phys.: Conf. Ser. 46, 514-518):
- remaining < 20% → 5 cycles
- remaining < 50% → 10 cycles
- remaining ≥ 50% → 15 cycles

#### `perform_incremental_checkpoint(case_name, checkpoint_dir=".checkpoints", nowrite_vector=False, is_soc=False) -> Dict`
Copies only modified files. Returns `checkpoint_id`, `files_copied`, `size_mb`.

#### `cleanup_old_checkpoints(checkpoint_dir=".checkpoints", max_checkpoints=3) -> Dict`
Removes old checkpoints keeping last N. Warns if > 1GB.

#### `resume_from_checkpoint(case_name, checkpoint_id=None) -> Dict`
Restores checkpoint files and returns cycle tracking info.

---

### `wien2k_gen.optimizer.gpu_detector`

```python
from wien2k_gen.optimizer.gpu_detector import (
    detect_gpu_hardware,
    check_wien2k_gpu_support,
    analyze_offload_potential,
    estimate_gpu_memory,
    recommend_gpu_strategy,
    generate_hybrid_machines,
    run_gpu_benchmark,
    GPUInfo,
    OffloadAnalysis,
)
```

#### `detect_gpu_hardware() -> List[GPUInfo]`
Detects GPUs via nvidia-smi → rocm-smi → sycl-ls → /dev/dri fallback.
Returns `GPUInfo(vendor, model, memory_mb, compute_capability, count, detected)`.

#### `check_wien2k_gpu_support(wienroot=None) -> Dict`
Scans siteconfig, binaries, and parallel_options for GPU flags.
Returns `{"gpu_enabled": bool, "vendor": str, "gpu_binaries": [...], "compile_flags": [...]}`.

#### `analyze_offload_potential(nmat, num_kpoints, system_type="unknown", gpu_info=None) -> OffloadAnalysis`
Per-lapw-stage GPU offload analysis (see Yu et al. 2021, Comput. Phys. Commun. 262, 107808; WIEN2k benchmarks at wien2k.at/reg_user/benchmark/).
Returns speedup estimates, memory requirements, and OOM risk.

#### `estimate_gpu_memory(nmat, num_kpoints_per_gpu) -> float`
GPU memory estimate: `nmat² × 16 bytes × kpts × 1.5 / (1024²)` MB.

#### `recommend_gpu_strategy(gpu_info, wien2k_gpu, nmat, num_kpoints) -> Dict`
Intelligent GPU strategy recommendation with 4 scenarios:
1. Full GPU offload (nmat > 8000 + GPU+WIEN2k-GPU)
2. Hybrid CPU+GPU (memory constrained)
3. Recompile needed (GPU present but WIEN2k CPU-only)
4. CPU-only (small system or no GPU)

#### `generate_hybrid_machines(gpu_info, cpu_cores, num_kpoints, nmat) -> str`
Generates hybrid `.machines` with GPU ranks for lapw1 and CPU ranks for lapw2/lapw0.

#### `run_gpu_benchmark(case_name, gpu_id=0) -> Dict`
Runs CPU vs GPU lapw1 timing comparison. Returns speedup and saves to `.gpu_benchmark.json`.

---

## ML Modules

### `wien2k_gen.ml.gnn_kpoint_predictor`

```python
from wien2k_gen.ml.gnn_kpoint_predictor import (
    build_crystal_graph,
    predict_kpoints,
    _kpoint_fallback,
    CGCNNModel,
    GraphConvLayer,
)
```

#### `build_crystal_graph(positions, atomic_numbers, lattice_vectors, cutoff=8.0) -> Tuple[np.ndarray, ...]`
Builds crystal graph from atomic structure. Returns `(node_features, edge_index, edge_features)`.

**Node features** (per atom): atomic radius, electronegativity, covalent radius, valence electrons.
**Edge features** (per bond): distance in Å, bond type (0=covalent, 1=ionic, 2=metallic).

#### `predict_kpoints(structure, model_path=None, default_model_dir=None) -> Dict`
GNN-based k-point grid prediction.

**Returns:**
```python
{
    "grid": (nx, ny, nz),         # recommended k-point grid
    "confidence": float,           # 0.0–1.0
    "method": "GNN" | "fallback_mp_grid",
    "kpoint_density": int,         # k-points/Å⁻³
    "recommendation": str,
}
```

Confidence < 0.70 triggers MP grid fallback.

#### `_kpoint_fallback(structure, reason="") -> Dict`
Monkhorst-Pack grid fallback: `k_i = max(1, round(k0/|a_i|))`.
k0 = 30 for insulators/semiconductors, 40 for metals.

#### `CGCNNModel`
Lightweight 4-layer graph convolution network with residual connections, global mean+max pooling, and 2-layer MLP head. Pure NumPy — zero PyTorch dependency.

#### `GraphConvLayer`
Single graph convolution: `h_i' = σ(W_s·h_i + Σ W_n·h_j ⊙ EdgeMLP(e_ij))` with ReLU activation and degree normalization.

---

## Backend Modules

### `wien2k_gen.backends.wien2k`

```python
from wien2k_gen.backends.wien2k import Wien2kBackend, check_elpa_available
```

#### `Wien2kBackend`
- `detect_problem_size() -> dict` — Extract all problem parameters
- `_detect_wien2k_flags() -> Wien2kFlags` — Detect calculation type
- `get_execution_command(flags) -> str` — Build correct `run_lapw`/`runsp_lapw` command
- `write_machines(suggestion, topo, validate=True) -> str` — Generate `.machines` content
- `parse_dayfile(dayfile_path) -> dict` — Parse WIEN2k dayfile timings
- `_select_parallel_strategy(nmat, kpoints, is_hybrid, is_soc) -> dict` — Parallel strategy decision matrix
- `_build_machines_lines(...) -> str` — Strict WIEN2k `.machines` format compliance

---

## Type Definitions

### `wien2k_gen.types`

```python
from wien2k_gen.types import (
    BackendCode,         # WIEN2K, QUANTUM_ESPRESSO, VASP, CP2K
    CalculationType,     # SCF, SPIN_POLARIZED, SOC, SPIN_POLARIZED_SOC, LDA_U, HYBRID_FUNC, FORCES, EECE
    ExecutionMode,       # KPOINT, HYBRID, MPI, FINE_GRAIN
    Wien2kVersion,       # V19, V21, V23, V24, UNKNOWN
    Wien2kFlags,
    OptimizationTarget,  # TIME, ENERGY, COST, BALANCED
)
```

#### `Wien2kFlags`
```python
@dataclass
class Wien2kFlags:
    is_spin_polarized: bool = False
    is_soc: bool = False
    is_lda_u: bool = False
    is_hybrid: bool = False
    is_eece: bool = False
    has_forces: bool = False
    wien2k_version: str = ""

    def get_calculation_type() -> CalculationType
    def get_execution_command() -> str
```
