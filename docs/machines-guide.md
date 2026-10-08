# `.machines` File Guide

This guide teaches you how to generate, read, and use a WIEN2k `.machines` file with FORGE. It is written for users who currently write `.machines` by hand, and for users who have never seen the file.

A valid WIEN2k license is required to run the generated configuration. FORGE only writes files; it does not replace WIEN2k.

---

## 1. What `.machines` Is

`.machines` tells WIEN2k **which hosts run which program, and with how many cores**. WIEN2k parallel wrappers (`lapw1para`, `lapw2para`, `lapwsopara`, `lapwdmpara`) read this file every SCF cycle.

A typical file has four kinds of lines:

| Line | Meaning |
|------|---------|
| `lapw0: hostname:N` | Potential generation. Often serial or few cores. |
| `1: hostname:N` | One MPI rank for `lapw1` (Hamiltonian diagonalization). |
| `2: hostname:N` | One MPI rank for `lapw2` (charge density / vectors). |
| `granularity: N` | Fine-grain grouping of ranks. |
| `kpar: N` | k-point groups (hybrid / band-parallel). |
| `omp_global: N` | OpenMP threads per rank. |
| `omp_lapw0: N` | OpenMP threads for lapw0. |
| `extrafine: 1` | Extra k-point remainder handling. |
| `lapw2_vector_split: N` | Split large vector I/O. |

WIEN2k 23.x and 24.x also accept `lapw1:` / `lapw2:` stage lines. Older versions mainly use `1:` and `2:`.

---

## 2. Why FORGE Is Better Than Writing `.machines` by Hand

Manual `.machines` files fail in predictable ways. FORGE exists to remove those failures.

### 2.1 The manual workflow (what goes wrong)

A typical hand-written file looks like this:

```
lapw0: localhost:1
1: localhost:64
2: localhost:64
granularity:1
```

That file is easy to type and often **wrong**:

1. **You counted logical cores, not physical cores.** Hyper-Threading is on. DFT diagonalization saturates on physical cores. You oversubscribe, runtime goes up, not down.
2. **You gave lapw0 64 cores.** lapw0 is I/O / FFT bound and mostly serial. Extra cores do nothing except contend for memory bandwidth.
3. **You have 8 irreducible k-points and 64 ranks.** k-point parallel saturates at 8 ranks. The other 56 ranks idle or fight over the same k-points.
4. **You ignored NUMA.** 64 threads on a 2-socket node without binding cross the QPI/UPI link. Memory-bound lapw1 slows down 20-40%.
5. **You ignored NMAT.** A 2000-basis cell and a 20000-basis supercell need different modes. One file cannot be copied between them.
6. **You ignored SOC / hybrid / spin.** Complex wavefunctions double memory. Hybrid functionals change the parallel strategy. A copied file OOMs.
7. **You typed hostnames from yesterday's allocation.** SLURM gave you `node[17,22,41]`, your file still says `node01-04`. The job dies at `lapw1para`.
8. **You set `OMP_NUM_THREADS=64` and MPI ranks=64.** 4096 threads. The node melts. The queue kills you.

### 2.2 What FORGE does instead

| Decision | Manual | FORGE |
|----------|--------|-------|
| Host list | You type names | Reads SLURM/PBS/LSF/SGE allocation |
| Core count | Guess | Physical cores, Amdahl cap, OS reserve |
| Mode | Copy last job | kpoint / hybrid / mpi / fine_grain from NMAT, k-points, atoms |
| lapw0 | Often over-allocated | Minimal cores from atom count |
| lapw1 vs lapw2 split | 50/50 guess | 55-65% to lapw1 based on NMAT |
| OpenMP | Forgotten or too high | Bound to NUMA / socket |
| Memory | Hope | Estimates MB/core from NMAT, SOC, hybrid |
| ELPA | Unknown | Warns if NMAT > 5000 and ELPA missing |
| GPU | Not in the file | Optional hybrid CPU+GPU lines |
| Heterogeneous nodes | Broken | Scales ranks to core ratio |
| Saturation | Never considered | Clamps to max efficient cores |

The generated file also writes a header so you can audit the decision:

```
# FORGE v0.1.0 | 2026-10-08T12:00:00Z
# Nodes: node01, node02 | Cores: [64, 64]
# Problem: atoms=8 kpts=84 nmat=2567 soc=False hybrid=False spin=False
# Est. memory: ~180 MB/core -> ~11.2 GB total
```

You can still edit the file. FORGE is a starting point that is already physics-aware.

### 2.3 When you should still edit by hand

- You must pin a rank to a specific NIC or GPU id that FORGE cannot see.
- The cluster hostname in the allocation file is not the hostname WIEN2k should SSH/rsh to.
- You are debugging a single-rank serial comparison.

Use:

```bash
forge generate --manual
```

That generates the file, then opens it in `$EDITOR`.

---

## 3. One-Command Generation

From a WIEN2k case directory (`Si.struct`, `Si.in1`, ...):

```bash
forge generate
```

That command:

1. Detects the scheduler and the current node list.
2. Reads `.struct`, `.scf`, `.in1`, `.klist`, `.inso`, `.inorb`, `.inst`.
3. Chooses a parallel mode.
4. Writes `.machines` and `parallel_options`.

Preview without writing:

```bash
forge generate --dry-run
```

Force a mode:

```bash
forge generate --mode kpoint
forge generate --mode hybrid --omp 4
forge generate --mode mpi --cores 128
forge generate --mode fine_grain
```

Leave cores for the OS on a workstation:

```bash
forge generate --reserve-os-cores 4
```

Cap cores (useful inside a shared login node):

```bash
forge generate --max-cores 16
```

GPU-aware file (WIEN2k 24+ compiled with GPU):

```bash
forge generate --gpu --target time
```

Export a machine-readable summary:

```bash
forge generate --export config.json
```

---

## 4. How to Read a Generated File

### 4.1 Small semiconductor, many k-points (kpoint mode)

Silicon, 2 atoms, 1000 k-points, 32-core workstation:

```
# FORGE v0.1.0
# Mode: kpoint | Total cores: 32
# Problem: atoms=2 kpts=1000 nmat=2567 soc=False hybrid=False spin=False

lapw0: localhost:1

# K-point parallelization (nkpt=1000)
1: localhost:32
2: localhost:32
granularity: 1
omp_lapw0: 1
omp_mixer: 1
```

**How to read it**

- `lapw0` gets 1 core. The cell is tiny; potential generation is cheap.
- One `1:` line with 32 cores means 32 k-point workers for lapw1.
- The matching `2:` line does the same for lapw2.
- 1000 k-points / 32 ranks = 31.25, so WIEN2k can keep all ranks busy.
- `granularity: 1` means no extra fine-grain grouping.

**Manual mistake this avoids:** putting `1: localhost:1` 32 times, or giving lapw0 32 cores.

### 4.2 Metal, dense k-mesh (kpoint mode, extrafine)

Copper, FCC, 512 k-points, 48 cores, k-points not divisible by rank count:

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

`extrafine: 1` appears when `kpoints % n_ranks != 0`. Without it, remainder k-points serialize at the end of each cycle.

### 4.3 Hybrid MPI+OpenMP on a multi-socket node

Fe2O3, 80 atoms, 84 k-points, 4 nodes x 64 cores, NMAT around 15000:

```
lapw0: node01:8

# K-point parallelization (nkpt=84)
1: node01:4
1: node01:4
1: node01:4
1: node01:4
1: node02:4
1: node02:4
1: node02:4
1: node02:4
1: node03:4
1: node03:4
1: node03:4
1: node03:4
1: node04:4
1: node04:4
1: node04:4
1: node04:4
2: node01:4
2: node01:4
2: node01:4
2: node01:4
2: node02:4
2: node02:4
2: node02:4
2: node02:4
2: node03:4
2: node03:4
2: node03:4
2: node03:4
2: node04:4
2: node04:4
2: node04:4
2: node04:4
granularity: 1
omp_global: 4
omp_lapw0: 8
omp_mixer: 1
```

**How to read it**

- Each `1: nodeXX:4` is **one MPI rank using 4 cores** (OpenMP).
- 16 ranks x 4 threads = 64 cores per node.
- MPI crosses nodes and sockets; OpenMP stays inside a NUMA domain.
- lapw0 uses 8 OpenMP threads on `node01` only.

Matching `parallel_options` fragment:

```bash
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export WIEN_MPIRUN="srun --mpi=pmix --hint=nomultithread --cpu-bind=core"
```

**Manual mistake this avoids:** `1: node01:64` with `OMP_NUM_THREADS=1` on a 2-socket EPYC, which remote-accesses half the memory.

### 4.4 Large matrix, few k-points (mpi / ELPA)

A 120-atom oxide supercell, 4 k-points, NMAT = 18000, ELPA available:

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

**How to read it**

- k-point parallel is useless (only 4 k-points).
- The Hamiltonian is distributed with ScaLAPACK/ELPA.
- lapw1 gets more cores than lapw2 because diagonalization dominates.
- `lapw2_vector_split: 8` reduces vector-file I/O storms on shared filesystems.
- If ELPA is missing, FORGE writes a warning in the header instead of silently generating a slow file.

### 4.5 Very large cell (fine_grain)

200+ atoms, 2 k-points, WIEN2k 23+:

```
lapw0: node01:8

# Core parallelization (nmat=24000, large system)
1: node01:32
1: node02:32
1: node03:32
1: node04:32
2: node01:32
2: node02:32
2: node03:32
2: node04:32
granularity: 16
omp_lapw0: 8
omp_mixer: 1
lapw2_vector_split: 16
```

`WIEN_GRANULARITY` in `parallel_options` must match. WIEN2k 19.x does not fully support this.

### 4.6 Spin-orbit + spin-polarized

FePt, `FePt.inst` present, `FePt.inso` present:

```
# Problem: atoms=2 kpts=512 nmat=4100 soc=True hybrid=False spin=True

lapw0: localhost:1
1: localhost:32
2: localhost:32
granularity: 1
omp_lapw0: 1
omp_mixer: 1
```

The file itself looks similar. The important change is **memory**: SOC uses complex wavefunctions (about 2x). FORGE raises the memory estimate and the submit script `--mem`. The execution command becomes `runsp_lapw -p -so`, not `run_lapw -p`.

### 4.7 LDA+U

NiO with `NiO.inorb` and `NiO.inm`:

FORGE still writes a normal `.machines`, but:

- Detects `run_lapw -p -orbc`.
- Adds LDA+U double-counting memory overhead.
- Does not change k-point parallel unless NMAT is large.

### 4.8 GPU hybrid (WIEN2k 24+)

```bash
forge generate --gpu
```

Conceptual layout:

```
lapw0: cpu00:8
1: cpu00:16
1: gpu00:1
2: cpu00:16
granularity: 1
```

lapw0 stays on CPU (I/O bound). lapw1 offload is used when NMAT > 5000.

### 4.9 Heterogeneous nodes

One 128-core node and one 64-core node, total request 192 cores:

```
# Heterogeneous cluster — ranks scaled to core ratio
# node login01 excluded: 0 cores after rebalancing

lapw0: nodeA:4
1: nodeA:124
1: nodeB:64
2: nodeA:124
2: nodeB:64
granularity: 1
```

Manual files almost never get this ratio right. FORGE scales ranks to the core ratio and can exclude a node that would receive 0 cores.

---

## 5. Worked Examples You Can Copy

### Example A — First SCF on a laptop (debug)

```bash
cd ~/wien2k/Si
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 64
forge generate --max-cores 4 --mode kpoint --dry-run
forge generate --max-cores 4 --mode kpoint
run_lapw -p -ec 0.0001 -cc 0.001 -i 20
```

Use this to check that WIEN2k accepts the file before you spend queue time.

### Example B — Production Si on one workstation

```bash
cd ~/wien2k/Si
forge generate --reserve-os-cores 2 --target time
run_lapw -p
```

### Example C — Force hybrid because advise said so

```bash
forge advise --case Fe2O3 --cores 256
forge generate --mode hybrid --omp 4 --cores 256 --target time
```

### Example D — Ignore Amdahl cap (you measured that more cores still help)

```bash
forge generate --cores 256 --ignore-saturation
```

Without `--ignore-saturation`, FORGE may clamp 256 down to e.g. 96 if the serial fraction of lapw0 is high.

### Example E — Recalibrate hardware, then generate

```bash
forge generate --recalibrate --target time
```

This invalidates the cached Roofline measurement and re-runs STREAM-like probes.

### Example F — Inside an already allocated SLURM job

```bash
salloc -N 2 -n 64 -t 01:00:00 -p compute
cd $SLURM_SUBMIT_DIR/Fe
forge generate
runsp_lapw -p
```

Hostnames come from the live allocation, not from a stale file.

### Example G — Compare two modes before committing

```bash
forge generate --mode kpoint --dry-run --export kpoint.json
forge generate --mode hybrid --omp 4 --dry-run --export hybrid.json
```

Inspect both JSON files, then generate the winner without `--dry-run`.

### Example H — Open and tweak one line

```bash
export EDITOR=vim
forge generate --manual
```

Change only `omp_global` or a hostname, leave the rest.

---

## 6. `.machines` and `parallel_options` Must Agree

FORGE always writes both. If you edit one, edit the other.

| `.machines` | `parallel_options` |
|-------------|--------------------|
| `omp_global: 4` | `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` |
| `granularity: 8` | `WIEN_GRANULARITY=8` |
| multi-node hosts | `WIEN_MPIRUN=srun ...` or `mpirun -machinefile` |
| GPU ranks | `WIEN_GPU=1` |

Mismatch example that wastes a whole queue allocation:

```
# .machines
omp_global: 4

# parallel_options (forgotten)
export OMP_NUM_THREADS="1"
```

WIEN2k will start 4-core ranks that only use 1 thread.

---

## 7. Validation Checklist Before `run_lapw -p`

1. `forge generate --dry-run` looks sane (mode, cores, warnings).
2. Hostnames in `.machines` match `scontrol show hostnames` / `cat $PBS_NODEFILE`.
3. Sum of lapw1 cores is not larger than physical cores on that node.
4. `OMP_NUM_THREADS * MPI ranks per node <= physical cores`.
5. Estimated memory x nodes < requested `--mem`.
6. For kpoint mode: irreducible k-points >= MPI ranks, or `extrafine: 1` is present.
7. For mpi/fine_grain: ELPA warning is understood.
8. Calculation flags match input files (`-so`, `-orbc`, `-hf`, `-fc`).

---

## 8. Common Manual Errors vs FORGE Behavior

| Symptom | Manual cause | FORGE behavior |
|---------|--------------|----------------|
| Job hangs in lapw1para | Wrong hostname | Reads current allocation |
| Runtime slower with more cores | Oversubscription / Amdahl | Caps at max efficient cores |
| OOM killer | Ignored SOC/hybrid memory | Estimates from NMAT and flags |
| Idle ranks | More ranks than k-points | Switches to hybrid or mpi |
| QTL-B / crash after copy | File from another case | Regenerates from this `.struct`/`.scf` |
| InfiniBand unused | `mpirun` without binding | Sets `srun --cpu-bind=core` |
| One node idle | Repeated `localhost` on a cluster | Emits one line per allocated node |

---

## 9. Minimal Mental Model

```
case files (.struct .scf .klist .in1 ...)
        +
hardware + scheduler allocation
        +
mode (kpoint | hybrid | mpi | fine_grain)
        =
.machines  +  parallel_options
        |
        v
run_lapw -p   or   forge submit
```

If any input changes (new k-mesh, new node count, SOC turned on), regenerate. Do not reuse yesterday's `.machines`.

---

## 10. Related Documents

- `docs/parallel-modes.md` — when each of the four modes is valid
- `docs/job-submission.md` — how to submit the job that consumes `.machines`
- `docs/preprocessing-convergence.md` — converge RKmax / k-mesh **before** a long parallel run
- `docs/examples.md` — additional scenario dumps
- `docs/troubleshooting.md` — hostname, MPI, NUMA, HT failures
