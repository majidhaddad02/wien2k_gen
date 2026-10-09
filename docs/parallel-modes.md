# Parallel Execution Modes

FORGE picks a WIEN2k parallel mode from NMAT, irreducible k-points, atom count, and NUMA so you do not copy a 32-rank k-point file onto a 4-k-point supercell.

Force a mode with `forge generate --mode`. Choices: `kpoint`, `hybrid`, `mpi`, `serial`, `fine_grain`. Tab-complete them from `completions/forge.bash`.

---

## Mode Summary

| Mode | MPI ranks | OMP threads | Best for | Communication |
|------|-----------|-------------|----------|---------------|
| **kpoint** | ~ min(cores, k-points) | 1 | Many k-points, small cells | Minimal |
| **hybrid** | < k-points | > 1 | Multi-core / NUMA nodes | Moderate |
| **mpi** | ~ cores | 1 | Large NMAT, few k-points, ELPA | High (ScaLAPACK) |
| **fine_grain** | >= atoms | 1 | Very large cells, WIEN2k 23+ | Very high |
| **serial** | 1 | 1 | Debug / tiny cells | None |

```bash
forge generate --mode kpoint
forge generate --mode hybrid --omp 4
forge generate --mode mpi --cores 128
forge generate --mode fine_grain
forge generate --mode serial --cores 1
```

---

## K-Point Parallel (`kpoint`)

Use when `kpoints >= 2 * total_cores` and the cell is small (atoms < 20). Each k-point is independent; scaling is near-linear up to the k-point count.

```bash
forge generate --mode kpoint --cores 32
forge generate --mode kpoint --max-cores 16 --dry-run
```

`extrafine: 1` is written only when `nkpt % n_ranks != 0`. Parallelism ceiling = number of k-points.

---

## Hybrid MPI+OpenMP (`hybrid`)

Use on multi-socket nodes when k-point pure mode saturates. MPI distributes k-points; OpenMP stays inside a NUMA domain.

```bash
forge generate --mode hybrid --omp 4
forge generate --mode hybrid --omp 8 --cores 256 --target time
```

`--target` is `time|memory|balanced|cost`.

`kpar:` is **not** this mode. `kpar` is only for hybrid-**functional** band parallel (`HYBR` in `case.in0`).

---

## MPI Fine-Grain (`mpi`)

Use when NMAT > 5000 and k-points are few (<=4). ELPA helps above ~8000.

```bash
forge generate --mode mpi --cores 128
forge generate --mode mpi --gpu --target time
```

lapw0 stays mostly serial — that is the Amdahl wall.

---

## Fine-Grain Atom Decomposition (`fine_grain`)

Use for atoms > 100, limited k-points, WIEN2k >= 23 (`WIEN_GRANULARITY`).

```bash
forge generate --mode fine_grain --target time
```

---

## Selection Algorithm

- **kpoints >= 2 x total_cores** and **atoms < 20** → kpoint
- **5000 < nmat <= 10000** and NUMA > 1 → hybrid
- **nmat > 10000** → fine-grain MPI
- **nmat > 8000** and ELPA present → ELPA2
- **nmat < 500** → serial LAPACK

User `--target` (time, memory, cost, balanced), Amdahl saturation, memory bandwidth, and `--gpu` all feed the final recommendation.

```bash
forge advise --case Si --cores 64 --target time
forge generate --ignore-saturation --cores 256
```

---

## Amdahl Saturation

```
amdahl_speedup = 1.0 / (serial_fraction + (1 - serial_fraction) / total_cores)
efficiency_percent = (amdahl_speedup / total_cores) * 100
```

Without `--ignore-saturation`, FORGE clamps auto-detected cores to `max_efficient_cores`.

Amdahl, G. M. (1967). *AFIPS*, 30, 483-485.
