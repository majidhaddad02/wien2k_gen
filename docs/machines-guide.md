# `.machines` File Guide

FORGE writes a `.machines` file that matches **this** case and **this** allocation: physical cores (not HT), Amdahl-capped ranks, NUMA-aware OpenMP, and hostnames from SLURM/PBS/LSF/SGE — not yesterday's node list.

A valid WIEN2k license is required. FORGE only writes files.

Tab-complete `forge generate` flags from `completions/forge.bash` / `completions/forge.zsh`.

---

## 1. What `.machines` Is

`.machines` tells WIEN2k which hosts run which program, and with how many cores. Parallel wrappers (`lapw1para`, `lapw2para`, `lapwsopara`, `lapwdmpara`) read it every SCF cycle.

| Line | Meaning |
|------|---------|
| `lapw0: hostname:N` | Potential generation |
| `1: hostname:N` | One MPI rank for `lapw1` |
| `2: hostname:N` | One MPI rank for `lapw2` |
| `lapw1:` / `lapw2:` | Stage lines (WIEN2k 23+) |
| `granularity: N` | Fine-grain grouping |
| `kpar: N` | **Only** for hybrid-functional band parallel |
| `omp_global: N` | OpenMP threads per rank |
| `omp_lapw0: N` | OpenMP for lapw0 |
| `extrafine: 1` | Remainder k-points when `nkpt % n_ranks != 0` |
| `lapw2_vector_split: N` | Split large vector I/O |

Physical cores per node are `max(lapw0, lapw1/1:, lapw2/2:)` — stages are sequential, so widths are **not** summed. Remainder ranks are not an OpenMP error.

---

## 2. Why Not Write It by Hand

| Decision | Manual | FORGE |
|----------|--------|-------|
| Host list | Typed names | Current scheduler allocation |
| Core count | Logical cores / guess | Physical cores, Amdahl cap, OS reserve |
| Mode | Copy last job | kpoint / hybrid / mpi / fine_grain from NMAT, k-points, atoms |
| lapw0 | Often over-allocated | Minimal cores from atom count |
| OpenMP | Forgotten or `OMP * MPI = explosion` | Bound to NUMA / socket |
| Memory | Hope | Estimate from NMAT, SOC, hybrid |
| `extrafine` | Always on or always off | Only if `nkpt % n_ranks != 0` |
| `kpar` | Copied into every file | Only `band_parallel` (hybrid functional) |
| Heterogeneous nodes | Broken | Scales ranks to core ratio |

Header written for audit:

```
# FORGE v0.1.0 | 2026-10-08T12:00:00Z
# Nodes: node01, node02 | Cores: [64, 64]
# Problem: atoms=8 kpts=84 nmat=2567 soc=False hybrid=False spin=False
# Est. memory: ~180 MB/core -> ~11.2 GB total
```

Edit afterwards with:

```bash
forge generate --manual
```

---

## 3. Generate Commands

```bash
forge generate
forge generate --dry-run
forge generate --mode kpoint
forge generate --mode hybrid --omp 4
forge generate --mode mpi --cores 128
forge generate --mode fine_grain
forge generate --reserve-os-cores 4
forge generate --max-cores 16
forge generate --gpu --target time
forge generate --export config.json
forge generate --overwrite
forge generate --ignore-saturation --cores 256
forge generate --recalibrate --target memory
```

`--target` is `time|memory|balanced|cost`.

---

## 4. How to Read a Generated File

### 4.1 K-point mode, many k-points

Silicon, 2 atoms, 1000 k-points, 32 cores:

```
lapw0: localhost:1

# K-point parallelization (nkpt=1000)
1: localhost:32
2: localhost:32
granularity: 1
omp_lapw0: 1
omp_mixer: 1
```

1000 is divisible by 32, so **no** `extrafine`. `1:` and `2:` are sequential stages; the node still uses 32 physical cores, not 64.

### 4.2 Remainder k-points (`extrafine`)

Copper, 512 k-points, 48 ranks:

```
lapw0: localhost:2

# K-point parallelization (nkpt=512)
1: localhost:48
2: localhost:48
granularity: 1
extrafine: 1
omp_lapw0: 2
omp_mixer: 1
```

`extrafine: 1` appears **only** when `kpoints % n_ranks != 0` (`512 % 48 != 0`). Without it, remainder k-points serialize at the end of each cycle. FORGE does not emit `extrafine` when the division is exact.

### 4.3 Hybrid MPI+OpenMP

Each `1: nodeXX:4` is one MPI rank using 4 cores. 16 ranks x 4 threads = 64 cores per node. Matching `parallel_options`:

```bash
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export WIEN_MPIRUN="srun --mpi=pmix --hint=nomultithread --cpu-bind=core"
```

```bash
forge generate --mode hybrid --omp 4 --cores 256
```

### 4.4 MPI / ELPA (few k-points, large NMAT)

```
lapw0: node01:4

# Fine-grain MPI with ELPA (nmat=18000, BLACS-aware)
lapw1: node01:40
lapw2: node01:20
lapw1: node02:40
lapw2: node02:20
granularity: 8
omp_global: 1
omp_lapw0: 4
omp_mixer: 1
lapw2_vector_split: 8
```

lapw1 and lapw2 cores on a node are sequential: physical cores = max(40, 20), not 60.

### 4.5 Fine-grain (WIEN2k 23+)

```bash
forge generate --mode fine_grain
```

`WIEN_GRANULARITY` in `parallel_options` must match `granularity:`. WIEN2k 19.x does not fully support this.

### 4.6 `kpar` (hybrid functional only)

`kpar: N` is written **only** when the strategy is `band_parallel` (hybrid functional, `HYBR` in `case.in0`). Ordinary k-point / hybrid-MPI files must not contain `kpar`.

```bash
forge generate --mode hybrid
# kpar appears only if the case is a hybrid functional
```

### 4.7 SOC / spin / LDA+U / GPU

SOC doubles memory (complex wavefunctions). The exec command becomes `runsp_lapw -p -so` when `.inst` and `.inso` exist. LDA+U (`case.inorb`) uses `run_lapw -p -orbc`. GPU:

```bash
forge generate --gpu
forge generate --gpu --gpu-mixed-precision
```

### 4.8 Heterogeneous nodes

FORGE scales ranks to the core ratio and can drop a node that would receive 0 cores. Topology caps live in `_allocate_node_cores`.

---

## 5. Copy-Paste Examples

```bash
forge generate --max-cores 4 --mode kpoint --dry-run
forge generate --max-cores 4 --mode kpoint
run_lapw -p -ec 0.0001 -cc 0.001 -i 20

forge generate --reserve-os-cores 2 --target time
run_lapw -p

forge advise --case Fe2O3 --cores 256
forge generate --mode hybrid --omp 4 --cores 256 --target time

forge generate --cores 256 --ignore-saturation
forge generate --recalibrate --target time

salloc -N 2 -n 64 -t 01:00:00 -p compute
cd $SLURM_SUBMIT_DIR/Fe
forge generate
runsp_lapw -p

forge generate --mode kpoint --dry-run --export kpoint.json
forge generate --mode hybrid --omp 4 --dry-run --export hybrid.json

export EDITOR=vim
forge generate --manual
```

---

## 6. `.machines` and `parallel_options` Must Agree

FORGE always writes both.

| `.machines` | `parallel_options` |
|-------------|--------------------|
| `omp_global: 4` | `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` |
| `granularity: 8` | `WIEN_GRANULARITY=8` |
| `kpar: N` | only with hybrid-functional band parallel |
| multi-node hosts | `WIEN_MPIRUN=srun ...` |
| GPU ranks | `WIEN_GPU=1` |

---

## 7. Checklist Before `run_lapw -p`

1. `forge generate --dry-run` looks sane.
2. Hostnames match `scontrol show hostnames` / `$PBS_NODEFILE`.
3. Per-node physical cores = max of sequential stage widths, not the sum.
4. `OMP_NUM_THREADS * MPI ranks per node <= physical cores`.
5. Estimated memory x nodes < requested `--mem`.
6. K-point mode: `extrafine: 1` only if `nkpt % n_ranks != 0`.
7. `kpar:` only for hybrid-functional band parallel.
8. Regenerated after k-mesh, SOC, node count, or NMAT changes.

---

## 8. Related Documents

- `docs/parallel-modes.md` — mode theory
- `docs/job-submission.md` — how to submit
- `docs/user-guide.md` — full flag list
- `docs/troubleshooting.md` — hostname, MPI, NUMA
