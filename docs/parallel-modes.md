# Parallel Execution Modes

wien2k_gen supports four parallel execution modes borrowed from WIEN2k's own parallelization scheme. The optimizer automatically selects the best mode based on problem parameters and hardware topology. This document explains when each mode is appropriate and the physics behind the selection.

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

The optimizer uses a scoring system:

<!-- TODO: verify scoring formula against code -->

```python
score(mode) = w_kpt × kpoint_efficiency +
              w_mat × matrix_parallelism +
              w_mem × memory_headroom +
              w_comm × communication_cost
```

Where:
- **kpoint_efficiency** = min(1.0, kpoints / total_cores)
- **matrix_parallelism** = min(1.0, nmat / nmat_threshold)
- **memory_headroom** = (available_ram - estimated_memory) / available_ram
- **communication_cost** = 1.0 for kpoint, 0.5 for hybrid, 0.2 for mpi, 0.1 for fine_grain

The mode with the highest weighted score wins, subject to the user's `--target` preference:

| Target | Preference |
|--------|-----------|
| `time` | Maximize throughput, prefer more parallelism |
| `energy` | Minimize communication (network + inter-node traffic) |
| `cost` | Minimize core-hours (efficiency > throughput) |
| `balanced` | Equal weight to all factors |

---

## Amdahl's Law Saturation

The optimizer uses Amdahl's Law to prevent oversubscription:

```
serial_fraction = MAX(
    lapw0_overhead(atoms),     # 0.15 for <4 atoms, 0.02 for >100
    iobw_penalty(scratch_fs),  # 0.05 for NFS, 0.01 for NVMe
    ncomm_overhead(nodes),     # 0.02 per additional node
)

max_speedup = 1.0 / serial_fraction
max_efficient_cores = kpoints × max_speedup / (max_speedup - 1 + kpoints/cores)
```

If the user requests more cores than `max_efficient_cores`, the tool warns of saturation.

**References:**
- Amdahl, G. M. (1967). *AFIPS Conference Proceedings*, 30, 483-485.
- Hager, G. & Wellein, G. (2010). *Introduction to High Performance Computing for Scientists and Engineers*. CRC Press.
- Blaha, P. et al. (2020). *J. Chem. Phys.* 152, 074101.
- Cebrián, J. M. et al. (2015). *Comput. Phys. Commun.* 201, 85-99.
