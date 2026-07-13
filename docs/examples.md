# Real-World Examples

## Scenario 1: Single Workstation — NaCl (8 atoms, 1000 k-points)

**System:** 32-core AMD EPYC workstation, no scheduler

```bash
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000
run_lapw -p  # generates case.scf with :NMAT 2567
forge generate
```

**Generated `.machines`:**
```
# lapw0 is serial in WIEN2k — single core
lapw0: localhost: 1

# lapw1: k-point parallel
1: localhost: 32

# lapw2: k-point parallel
2: localhost: 32

granularity:1
extrafine:1
```

**Generated `parallel_options`:**
```bash
# Auto-generated parallel_options (forge)
# Topology: local | 32 cores | NUMA=2

USE_REMOTE=0
MPI_REMOTE=0
TASKSET=no
DELAY=0.1
SLEEPY=1
WIEN_MPIRUN=mpirun
OMP_NUM_THREADS=1
```

**Result:** All 32 cores used for k-point parallelism. NMAT=2567 is small enough that MPI fine-grain is unnecessary.

---

## Scenario 2: SLURM Cluster — Fe₂O₃ (80 atoms, 84 k-points)

**System:** 4 nodes × 64 cores, SLURM, InfiniBand HDR

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
forge generate
```

**Generated `.machines`:**
```
# lapw0: serial with 1 core
lapw0: node01: 1

# lapw1: hybrid MPI+OpenMP — 16 ranks × 4 threads
1: node01: 4  node02: 4  node03: 4  node04: 4
1: node01: 4  node02: 4  node03: 4  node04: 4
1: node01: 4  node02: 4  node03: 4  node04: 4
1: node01: 4  node02: 4  node03: 4  node04: 4

# lapw2: same — 16 ranks × 4 threads
2: node01: 4  node02: 4  node03: 4  node04: 4
2: node01: 4  node02: 4  node03: 4  node04: 4
2: node01: 4  node02: 4  node03: 4  node04: 4
2: node01: 4  node02: 4  node03: 4  node04: 4

granularity:1
extrafine:1
```

**Generated `parallel_options`:**
```bash
# Topology: slurm | 256 cores | NUMA=2
USE_REMOTE=0
MPI_REMOTE=0
TASKSET=no
WIEN_MPIRUN=srun --mpi=pmix --hint=nomultithread --cpu-bind=core
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
WIEN_GRANULARITY=1
```

**Why:** NMAT ≈ 15000 and atoms=80 — the matrix is large enough to benefit from ScaLAPACK. Hybrid mode gives 16 MPI ranks × 4 OMP threads = 64 cores per node. InfiniBand with `--hint=nomultithread` prevents HT oversubscription.

---

## Scenario 3: LDA+U Calculation — NiO (4 atoms, 256 k-points)

**Input files present:**
- `case.inorb` — LDA+U active
- `case.inm` — U=0.30 Ry, J=0.00 Ry for Ni (atom 1)

```bash
forge generate
```

**Detection output:**
```
Detected Calculation Type: LDA_U
Execution Command: run_lapw -p -orbc
LDA+U Parameters:
  Atom 1 (Ni): Ueff = 0.30 Ry (U=0.30, J=0.00, DC=AMF)
```

**Why:** The tool reads `.inm` to extract U and J per atom, then adjusts memory estimates for the double-counting scheme.

---

## Scenario 4: Spin-Polarized with SOC — FePt (2 atoms, 512 k-points)

```bash
runsp_lapw -p -so
```

**Auto-detected command:** `runsp_lapw -p -so`

**Why:** `case.inst` contains `SPIN` keyword and `case.inso` exists. The tool detects both and constructs the correct `runsp_lapw` command instead of `run_lapw`.

---

## Scenario 5: Reserved Cores — 128-core EPYC Genoa Workstation

```bash
forge generate --reserve-os-cores 4
```

**Output:**
```
Reserving 4 OS cores → using 124 of 128
Recommended cores: 124
```

**Generated `.machines`:**
```
1: localhost: 124
2: localhost: 124
```

---

## Scenario 6: SGE Cluster — SrTiO₃ (5 atoms, 512 k-points)

```bash
#$ -pe mpi 64              # 64 slots across nodes
#$ -l h_rt=24:00:00
forge generate --scheduler sge
```

**Detected:** SGE environment via `PE_HOSTFILE` → 64 slots across 4 nodes

**Generated `.machines`:**
```
1: node01: 16  node02: 16  node03: 16  node04: 16
2: node01: 16  node02: 16  node03: 16  node04: 16
```

---

## Scenario 7: GPU-Accelerated — Large Supercell

```bash
forge generate --gpu
```

**Output includes GPU-specific directives:**
- `$WIENROOT/lapw0_gpu` instead of `lapw0`
- CUDA-aware MPI hints in `parallel_options`
- Separate GPU memory budget in resource estimates

---

## Scenario 8: Dry-Run Preview

```bash
forge generate --dry-run
```

```
┌─ Configuration Preview ────────────────────────────────┐
│ Mode         hybrid                                    │
│ Total Cores  128                                       │
│ MPI Ranks    32                                        │
│ OMP Threads  4                                         │
│ Nodes        4 (node[01-04])                           │
│ Memory Est.  12.4 GB per rank                          │
│ Confidence   0.85                                      │
│                                                        │
│ Warnings:                                              │
│  • ELPA not found; MPI fine-grain may be slow          │
│  • NUMA system (2 nodes). Use --cpu-bind=core          │
└────────────────────────────────────────────────────────┘
```

---

## Scenario 9: JSON Export for Automation

```bash
forge generate --export config.json
```

**`config.json`:**
```json
{
  "status": "success",
  "mode": "hybrid",
  "recommended_total_cores": 128,
  "recommended_nodes": 4,
  "cores_per_node": [32, 32, 32, 32],
  "mpi_ranks_per_node": [8, 8, 8, 8],
  "omp_threads_per_rank": 4,
  "warnings": ["ELPA not found; MPI fine-grain may be slow"],
  "confidence_score": 0.85,
  "estimated_memory_gb": 48.2,
  "vector_split_active": false
}
```

---

## Physically Accurate `.machines` Logic

The tool makes physics-informed decisions based on problem parameters:

| Parameter | Source | Effect |
|-----------|--------|--------|
| **NMAT** | `case.scf` `:NMAT` field | Determines matrix diagonalization cost → mode selection |
| **nbands** | `case.in1` TOT/WFFIL format | Affects memory footprint per rank |
| **kpoints** | `case.klist` | Determines k-point parallel saturation limit |
| **atoms** | `case.struct` | Affects lapw0 serial fraction (Amdahl's Law) |
| **RKMAX** | `case.in0` | Impacts FFT grid size and memory |
| **GMAX** | `case.in1` / `case.in2` | Affects exchange-correlation computation |
| **U/J** | `case.inm` | LDA+U double-counting memory overhead |
| **SOC** | `case.inso` | 2× memory (complex wavefunctions) |
| **Hybrid** | `case.in0` HYBR keyword | 4-10× CPU cost (exact exchange) |
