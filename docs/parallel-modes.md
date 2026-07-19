# Parallel Execution Modes

forge supports four parallel execution modes borrowed from WIEN2k's own parallelization scheme. The optimizer automatically selects the best mode based on problem parameters and hardware topology. This document explains when each mode is appropriate and the physics behind the selection.

---

## Mode Summary

| Mode | MPI Ranks | OMP Threads | Best For | Communication |
|------|-----------|-------------|----------|---------------|
| **kpoint** | = k-points | 1 | Many k-points, small cells | Minimal (embarrassingly parallel) |
| **hybrid** | < k-points | > 1 | General purpose, multi-core nodes | Moderate |
| **mpi** | = cores | 1 | Large matrices (ELPA), few k-points | High (ScaLAPACK) |
| **fine_grain** | ≥ atoms | 1 | Very large systems, limited k-points | Very high (atom decomposition) |

---

## K-Point Parallel (`kpoint`)

### When It's Optimal
- **Many k-points** (≥ 4× core count) — enough independent tasks to fill all cores
- **Small unit cells** (atoms < 20) — lapw0 is cheap, Amdahl bottleneck is small
- **Rule of thumb:** `kpoints ≥ 2 × total_cores`

### Physics
Each k-point is diagonalized independently. This is the most efficient WIEN2k mode because communication is zero beyond summing eigenvalues at convergence time. Scaling is near-linear up to the k-point count.

### Memory
Each rank stores only its own k-point's matrices. Memory per rank ≈ NMAT² / ranks × 16 bytes.

### Limitations
Parallelism ceiling = number of k-points. With 4 k-points, you gain nothing beyond 4 cores.

---

## Hybrid MPI+OpenMP (`hybrid`)

### When It's Optimal
- **Moderate k-points** (4-64) — k-point pure mode saturates too quickly
- **Multi-core nodes** (≥ 16 cores/node) — OpenMP exploits shared memory
- **Multi-socket systems** — MPI across sockets, OpenMP within socket
- **NUMA systems** — OMP threads bound to local memory

### Physics
MPI distributes k-points across ranks; OpenMP parallelizes the ScaLAPACK diagonalization within each rank. The matrix (NMAT × NMAT) is split into blocks distributed across OMP threads using BLACS block-cyclic distribution.

### Configuration
```
Ranks per node ≈ min(kpoints, nodes × 4)
Threads per rank ≈ cores_per_node / ranks_per_node
```

### Memory
Per-rank memory includes full NMAT² blocks allocated per thread. The overhead is approximately 20% higher than pure MPI due to shared-memory allocation.

---

## MPI Fine-Grain (`mpi`)

### When It's Optimal
- **Large matrices** (NMAT > 5000) — ScaLAPACK/ELPA diagonalization benefits
- **Few k-points** (≤4) — k-point parallelism is useless
- **ELPA available** — delivers 2-4× speedup over ScaLAPACK

### Physics
The Hamiltonian/overlap matrix is distributed across all MPI ranks using 2D block-cyclic distribution. Diagonalization uses ScaLAPACK (PDSYEVD/PDSYEVX) or ELPA for the generalized eigenvalue problem H·c = ε·S·c.

### Configuration
```
Ranks = total_cores (1 rank per core, 1 thread per rank)
```

### Memory
Per-rank memory ≈ NMAT² / ranks × 16 bytes for the distributed matrix blocks. ELPA reduces this by 30% via optimized redistribution.

### Serial Bottleneck
lapw0 (potential generation) is serial in WIEN2k. For small systems (atoms < 10), lapw0 can consume 15-25% of total runtime regardless of MPI rank count. This is the Amdahl bottleneck.

---

## Fine-Grain Atom Decomposition (`fine_grain`)

### When It's Optimal
- **Very large systems** (atoms > 100) — per-atom parallelism pays off
- **Limited k-points** — cannot use k-point parallel
- **WIEN2k ≥ 23.x** — `WIEN_GRANULARITY` is required

### Physics
The Hamiltonian is decomposed atom-by-atom. Each MPI rank computes a subset of atoms' contribution to the full matrix. Requires MPI_Allreduce at each SCF step to assemble the global matrix.

### Configuration
```
Ranks per node ≥ atoms / 4
WIEN_GRANULARITY must be set in parallel_options
```

### Memory
Similar to MPI fine-grain but with proportional reduction from atom decomposition.

---

## Selection Algorithm

The optimizer makes mode selection decisions based on problem parameters:

- **kpoints ≥ 2×total_cores** and **atoms < 20** → kpoint mode (embarrassingly parallel)
- **5000 < nmat ≤ 10000** and **NUMA nodes > 1** → hybrid NUMA-aware
- **nmat > 10000** → fine-grain MPI with potential granularity
- **nmat > 8000** and **ELPA available** → ELPA2 solver
- **nmat < 500** → serial LAPACK (ELPA overhead exceeds benefit)

The final recommendation incorporates:
- User `--target` preference (time, energy, cost, balanced)
- Amdahl saturation analysis
- Memory bandwidth constraints (< 50 GB/s → memory-bound warning)
- GPU acceleration potential when detected

---

## Amdahl's Law Saturation

The optimizer estimates Amdahl saturation to prevent oversubscription:

```
amdahl_speedup = 1.0 / (serial_fraction + (1 - serial_fraction) / total_cores)
efficiency_percent = (amdahl_speedup / total_cores) * 100
```

The serial fraction is estimated from problem characteristics (atoms, scratch filesystem, node count) and clamped to [0.001, 0.99]. When efficiency drops below the target threshold, the tool warns that adding cores is counterproductive.

**References:**
- Amdahl, G. M. (1967). *AFIPS Conference Proceedings*, 30, 483-485.
