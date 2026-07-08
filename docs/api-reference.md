# API Reference

## Core Modules

### `wien2k_gen.core.hardware`

System hardware detection and profiling.

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
    get_cpu_governor,
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
Detect physical CPU cores using `lscpu` JSON or `/proc/cpuinfo`. Falls back to `os.cpu_count()`.

#### `get_cpu_architecture() -> str`
Returns one of: `"xeon"`, `"epyc"`, `"amd_ryzen"`, `"intel_consumer"`, `"arm_neoverse"`, `"arm"`, `"unknown"`.

#### `get_cpu_generation() -> str`
Returns canonical generation string: `"Xeon_SapphireRapids"`, `"EPYC_Genoa"`, `"EPYC_Milan"`, `"Neoverse_N1"`, `"Graviton3"`, etc.

#### `get_system_type() -> str`
Returns one of: `"laptop"`, `"workstation"`, `"compute_node"`, `"cluster"`, `"unknown"`.

Detection logic:
- **laptop:** Battery in `/sys/class/power_supply` or chassis type 8/9/10/14
- **cluster:** SLURM/PBS/LSF/SGE job ID is set
- **compute_node:** physical cores ≥ 32, no scheduler
- **workstation:** everything else

#### `get_interconnect_info() -> List[Dict]`
Returns interconnect details:
```python
[{
    "type": "infiniband",
    "provider": "mlx5",
    "speed_gbps": 100.0,
    "latency_ns": 1.0,
    "active_rate_gbps": 100.0,
}]
```

#### `is_hyperthreading_active() -> bool`
Returns `True` if logical cores > physical cores.

#### `get_hardware_profile() -> Dict`
Aggregate hardware summary with all fields above.

---

### `wien2k_gen.core.topology`

Execution environment topology.

```python
from wien2k_gen.core.topology import Topology, NUMANode, NodeSpec, GPUInfo
```

#### `Topology`
```python
Topology(
    nodes: List[str],          # hostname list
    cores_per_node: List[int], # cores per node
    env_type: str,             # "cluster" or "local"
    scheduler_hints: dict,     # {"mpi_launcher": "srun", ...}
)
```

Key methods:
- `topo.total_cores` → int
- `topo.get_mpi_binding_hints()` → `dict` with keys: `openmpi`, `intel_mpi`, `mpich`, `mvapich`, `srun`
- `topo.split_load_balanced(total_cores: int)` → list of (node, cores) tuples
- `topo.get_optimal_mpi_distribution(mode: str)` → core distribution

---

### `wien2k_gen.core.scheduler`

Scheduler environment detection.

```python
from wien2k_gen.core.scheduler import detect, SchedulerHints, auto_detect_memory
```

#### `detect(max_cores=None, force_refresh=False) -> Topology`
Auto-detects scheduler (SLURM → PBS → LSF → SGE → local) and returns configured Topology.

#### `auto_detect_memory() -> str`
Returns memory string for the current scheduler (e.g., `"128G"` for SLURM).

---

### `wien2k_gen.core.case_parser`

WIEN2k input file parser.

```python
from wien2k_gen.core.case_parser import CaseFileParser, CaseData, LDAUData, parse_case_directory
```

#### `CaseData`
```python
@dataclass
class CaseData:
    case_name: str           # case file stem
    atoms: int               # total atoms
    atoms_inequiv: int       # inequivalent atoms
    kpoints: int             # k-point count
    nmat: int                # basis set size (from .scf)
    nbands: Optional[int]    # number of bands (from .in1)
    rkmax: float             # plane-wave cutoff
    lmax: int                # maximum l for partial waves
    gmax: float              # Gmax for exchange-correlation
    fft_nx: int              # FFT grid X (from .in2)
    fft_ny: int
    fft_nz: int
    is_soc: bool             # spin-orbit coupling
    is_hybrid: bool          # hybrid functional
    is_spin_polarized: bool
    is_lda_u: bool           # LDA+U
    is_eece: bool            # onsite exact exchange
    has_forces: bool         # force calculation
    ldau: LDAUData           # U, J, Ueff per atom
    volume_bohr3: float      # unit cell volume
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
    u_ry: List[float]         # Hubbard U per atom (Ry)
    j_ry: List[float]         # Hund's J per atom (Ry)
    ueff_ry: List[float]      # Ueff = U - J per atom (Ry)
    l_orbital: List[int]      # orbital (2=d, 3=f)
    atoms: List[int]          # atom indices
    double_counting: str      # "AMF", "FLL", or "SIC"
    file_present: bool        # whether .inm was found
```

#### `CaseFileParser`
```python
parser = CaseFileParser(Path("/path/to/case"))

# Parse everything at once
data: CaseData = parser.parse_all()

# Parse individual files
nmat = CaseFileParser.parse_scf(Path("case.scf"))
in1  = CaseFileParser.parse_in1(Path("case.in1"))
ldau = CaseFileParser.parse_inm(Path("case.inm"))
```

#### `parse_case_directory(path=None) -> CaseData`
Convenience function equivalent to `CaseFileParser(path).parse_all()`.

---

### `wien2k_gen.config`

Configuration management.

```python
from wien2k_gen.config import load_config, get_config, AppConfig, ConfigManager
```

#### `load_config(file_path=None, cli_override=None) -> AppConfig`
Loads configuration with precedence: defaults → config file → env vars → CLI overrides.

#### `AppConfig`
```python
@dataclass
class AppConfig:
    wienroot: str
    scratch_dir: str
    backend: str          # "wien2k", "vasp", "quantum_espresso", "cp2k"
    max_cores: Optional[int]
    execution_mode: str   # "auto", "kpoint", "hybrid", "mpi", "fine_grain"
    log_level: str
    enable_gpu: bool
```

---

### `wien2k_gen.optimizer.advisor`

Resource optimization and recommendations.

```python
from wien2k_gen.optimizer.advisor import (
    suggest_optimal_resources,
    recommend,
    OptimizationTarget,
    ResourceSuggestion,
    estimate_memory_footprint_gb,
    estimate_amdahl_saturation,
    get_optimization_report,
)
```

#### `suggest_optimal_resources(topo, user_max_cores=None, optimization_target=OptimizationTarget.TIME) -> ResourceSuggestion`
Main optimization function. Returns a `ResourceSuggestion` with:
- `mode`: `"kpoint"`, `"hybrid"`, or `"mpi"`
- `recommended_total_cores`: int
- `recommended_nodes`: int
- `cores_per_node`: List[int]
- `mpi_ranks_per_node`: List[int]
- `omp_threads_per_rank`: int
- `warnings`: List[str]
- `confidence_score`: float
- `estimated_time_minutes`: Optional[float]
- `estimated_memory_gb`: Optional[float]

#### `estimate_amdahl_saturation(kpoints, nmat, atoms, total_cores_available, num_nodes, mode="mpi") -> dict`
Amdahl's Law saturation analysis. Returns:
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

---

### `wien2k_gen.backends.wien2k`

WIEN2k-specific backend.

```python
from wien2k_gen.backends.wien2k import Wien2kBackend
```

#### `Wien2kBackend`
- `detect_problem_size() -> dict` — Extract all problem parameters
- `_detect_wien2k_flags() -> Wien2kFlags` — Detect calculation type
- `get_execution_command(flags) -> str` — Build correct `run_lapw`/`runsp_lapw` command
- `write_machines(suggestion, topo, validate=True) -> str` — Generate `.machines` content
- `parse_dayfile(dayfile_path) -> dict` — Parse WIEN2k dayfile timings

---

### `wien2k_gen.types`

Type definitions and enums.

```python
from wien2k_gen.types import (
    BackendCode,         # WIEN2K, QUANTUM_ESPRESSO, VASP, CP2K
    CalculationType,     # SCF, SPIN_POLARIZED, SOC, SPIN_POLARIZED_SOC, LDA_U, HYBRID_FUNC, FORCES, EECE
    ExecutionMode,       # KPOINT, HYBRID, MPI, FINE_GRAIN
    Wien2kVersion,       # V19, V21, V23, V24, UNKNOWN
    Wien2kFlags,         # dataclass with is_spin_polarized, is_soc, is_lda_u, etc.
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
