# Job Submission Guide

FORGE can submit WIEN2k jobs to SLURM, PBS/Torque, LSF, and SGE, or run them locally. This guide covers **four usage models**, each with multiple copy-paste examples.

The four models:

| # | Model | Command | When to use |
|---|-------|---------|-------------|
| 1 | Auto submit | `forge submit` | Everyday production. Let FORGE read `.machines` and pick resources. |
| 2 | Generate-inside-allocation | `sbatch`/`qsub` script that calls `forge generate` then `run_lapw` | Hostnames must match the live node list. Best on multi-node jobs. |
| 3 | Dedicated SLURM helper | `forge_sbatch` | You need a reviewed SBATCH file, validation, preview, or job watching. |
| 4 | Interactive wizard / local | `forge_wizard` or local `run_lapw -p` | First-time setup, laptops, debug allocations, teaching. |

A fifth related path exists (`forge_sbatch generate` then `sbatch` by hand). It is treated as a variant of model 3.

---

## Prerequisites Common to All Models

```bash
# 1. WIEN2k case exists
ls Case.struct Case.in1 Case.in2

# 2. Optional but recommended: one SCF cycle so NMAT is known
#    Skip this only for a first-guess .machines
# run_lapw -i 1

# 3. FORGE on PATH
which forge
export WIENROOT=/path/to/WIEN2k
```

If `.machines` is missing:

- Interactive `forge submit` asks whether to generate it.
- Non-interactive `forge submit` fails unless you pass `--auto-generate`.
- `--no-auto-generate` always fails when the file is missing.

---

## Model 1 — `forge submit` (recommended default)

`forge submit` does four things:

1. Finds `.machines` (or generates it).
2. Parses ranks, OpenMP width, and node count from that file.
3. Builds a scheduler script (SLURM / PBS / LSF).
4. Submits it, unless `--dry-run`.

Resource auto-fill:

| Flag | Default if omitted |
|------|--------------------|
| `--nodes` | Number of hostnames in `.machines` |
| `--ntasks` | Rank count from `1:` / `lapw1:` lines |
| `--cpus-per-task` | `omp_global` (or max rank width) |
| `--mem` | Auto-detected node memory |
| `--time` | `24:00:00` |
| `--scheduler` | Auto-detect (`slurm`, `pbs`, `lsf`, `sge`) |
| `--job-name` | `wien2k_job` |

### Example 1.1 — SLURM, everything auto

```bash
cd ~/wien2k/Si
forge generate
forge submit --partition compute --time 48:00:00
```

Typical result:

```
Job ID: 1842203
Script: submit_job.sh
```

### Example 1.2 — Preview the script, do not submit

```bash
forge submit --partition compute --time 12:00:00 --dry-run
```

Prints the SBATCH body. No job is queued.

### Example 1.3 — Write the script to a chosen path

```bash
forge submit --partition compute --time 24:00:00 \
  --export si_scf.sh --dry-run
less si_scf.sh
```

Then submit yourself:

```bash
sbatch si_scf.sh
```

or let FORGE submit without `--dry-run`:

```bash
forge submit --partition compute --time 24:00:00 --export si_scf.sh
```

### Example 1.4 — Missing `.machines`, generate then submit

```bash
forge submit --auto-generate --partition compute --time 08:00:00
```

### Example 1.5 — Refuse to generate; fail fast in CI

```bash
forge submit --no-auto-generate --partition compute
# Error: No .machines file found. Run `forge generate` first, or pass --auto-generate.
```

### Example 1.6 — Override resources from the CLI

`.machines` says 32 ranks x 4 threads, but the partition only allows 16 tasks:

```bash
forge submit --partition short --ntasks 16 --cpus-per-task 4 --time 04:00:00
```

CLI flags win over the parsed file.

### Example 1.7 — Named job + dependency chain

Relaxation, then a denser SCF, then a band structure:

```bash
forge submit --job-name fe-relax --partition compute --time 24:00:00
# suppose Job ID = 9001

forge submit --job-name fe-scf --partition compute --time 48:00:00 \
  --dependency afterok:9001

forge submit --job-name fe-bands --partition compute --time 12:00:00 \
  --dependency afterok:9002
```

### Example 1.8 — PBS/Torque

```bash
forge submit --scheduler pbs --partition batch --time 48:00:00 --mem 64g
```

FORGE rewrites memory as PBS expects (`64gb`) and sets `nodes`/`ppn` from ranks x OpenMP.

### Example 1.9 — LSF

```bash
forge submit --scheduler lsf --partition normal --time 12:00:00 --job-name nio-ldau
```

Walltime `HH:MM:SS` is converted to LSF `HH:MM` when needed.

### Example 1.10 — Force SGE detection at generate time, then submit

```bash
forge generate --scheduler sge
forge submit --scheduler sge --partition all.q --time 24:00:00
```

### Example 1.11 — Spin-polarized Fe with SOC

```bash
cd examples/03_fe_magnetic
forge generate --mode hybrid --omp 4
forge submit --partition compute --time 48:00:00 --job-name fe-sp-so --mem 192G
```

FORGE's exec command becomes `runsp_lapw -p -so` because `Fe.inst` and (if present) `Fe.inso` are detected. You do not type the flags again in the submit command.

### Example 1.12 — JSON for a workflow manager

```bash
forge --json submit --partition compute --time 06:00:00
```

Use the `job_id` field to chain Nextflow / FireWorks / custom Python.

---

## Model 2 — Generate inside the allocation

Use this when **hostnames in `.machines` must be the hosts of this job**, not of the login node.

Pattern:

```
scheduler starts the job on the compute nodes
        ->
forge generate          # sees SLURM_NODELIST / PBS_NODEFILE
        ->
run_lapw -p
```

If you generate `.machines` on the login node, it often contains `localhost` or a debug node. The batch job then cannot start `lapw1para`.

### Example 2.1 — SLURM script, 2 nodes

Save as `run_si.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=si-scf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --hint=nomultithread
#SBATCH --cpu-bind=cores

module load wien2k/24.1
module load openmpi/4.1
export WIENROOT=/opt/WIEN2k_24.1
export SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$SCRATCH"

cd "$SLURM_SUBMIT_DIR"

forge generate --mode kpoint --target time
run_lapw -p -ec 0.0001 -cc 0.0001
```

Submit:

```bash
sbatch run_si.sbatch
```

### Example 2.2 — SLURM hybrid Fe, generate with explicit OMP

```bash
#!/bin/bash
#SBATCH --job-name=fe-hybrid
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd "$SLURM_SUBMIT_DIR"
forge generate --mode hybrid --omp 4 --cores 256
runsp_lapw -p
```

`--ntasks=64` and `--cpus-per-task=4` must match `--omp 4` and 4 nodes x 64 cores.

### Example 2.3 — PBS script

```bash
#!/bin/bash
#PBS -N cu-metal
#PBS -l nodes=2:ppn=32
#PBS -l walltime=24:00:00
#PBS -q batch

cd "$PBS_O_WORKDIR"
forge generate --scheduler pbs --mode kpoint
run_lapw -p
```

```bash
qsub run_cu.pbs
```

### Example 2.4 — LSF script

```bash
#!/bin/bash
#BSUB -J nio-u
#BSUB -n 64
#BSUB -W 24:00
#BSUB -q normal

cd "$LS_SUBCWD"
forge generate --scheduler lsf --mode hybrid --omp 4
run_lapw -p -orbc
```

```bash
bsub < run_nio.lsf
```

### Example 2.5 — Exclusive node, reserve OS cores

Some sites require `--exclusive` and still want a few cores for `sshd` / `slurmd`:

```bash
#!/bin/bash
#SBATCH --job-name=epyc-128
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --partition=compute

cd "$SLURM_SUBMIT_DIR"
forge generate --reserve-os-cores 4 --mode hybrid --omp 4
run_lapw -p
```

On a 128-core EPYC this yields 124 workers.

### Example 2.6 — GPU node

```bash
#!/bin/bash
#SBATCH --job-name=supercell-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

cd "$SLURM_SUBMIT_DIR"
forge generate --gpu --mode mpi --target time
run_lapw -p
```

### Example 2.7 — Checkpointed long SCF

```bash
#!/bin/bash
#SBATCH --job-name=big-scf
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --time=72:00:00
#SBATCH --partition=compute

export WIEN2KGEN_CHECKPOINT=1
cd "$SLURM_SUBMIT_DIR"
forge generate --mode fine_grain --target time
run_lapw -p -ec 0.00005 -cc 0.0001 -i 200
```

FORGE's checkpoint heuristic copies case files every 5 / 10 / 15 cycles depending on how long the SCF has already run.

### Example 2.8 — Why not generate on the login node?

```bash
# WRONG: on login node
forge generate          # writes 1: localhost:8
sbatch run.sbatch       # compute nodes ignore localhost or collide

# RIGHT: generate inside the job, as in 2.1
```

Exception: single-node jobs that `ssh` back to a known host list. That is rare. Prefer model 2.

---

## Model 3 — `forge_sbatch` (SLURM specialist)

`forge_sbatch` is a dedicated entry point: generate, validate, preview, submit.

Use it when:

- You must email a script to a reviewer before `sbatch`.
- CI must validate directives (`--time`, `--mem`, `--dependency`).
- You want `--watch` after submit.
- You need `--qos` or `--gres` that `forge submit` does not expose.

### Example 3.1 — Generate a script

```bash
forge_sbatch generate \
  --output si.sbatch \
  --job-name si-scf \
  --partition compute \
  --nodes 2 \
  --ntasks 64 \
  --cpus-per-task 1 \
  --mem 128G \
  --time 24:00:00 \
  --preview
```

`--ntasks 0` (the default) fills ntasks from topology detection.

### Example 3.2 — Dry-run print only

```bash
forge_sbatch generate --dry-run --partition compute --nodes 1 --time 04:00:00
```

### Example 3.3 — Validate an existing script

```bash
forge_sbatch validate si.sbatch
forge_sbatch validate si.sbatch --strict
```

`--strict` treats warnings as errors (bad time format, huge memory, missing bind flags).

### Example 3.4 — Preview with highlighting

```bash
forge_sbatch preview si.sbatch --highlight
```

### Example 3.5 — Submit and watch

```bash
forge_sbatch submit si.sbatch --watch
```

### Example 3.6 — GPU + QOS + dependency

```bash
forge_sbatch generate \
  -o nio-gpu.sbatch \
  -J nio-u \
  -p gpu \
  -N 1 \
  -n 16 \
  -c 8 \
  --gres gpu:a100:1 \
  --qos high \
  --time 12:00:00 \
  --mem 256G \
  --dependency afterok:1842203
```

### Example 3.7 — Validate then submit in a script

```bash
#!/bin/bash
set -euo pipefail
forge_sbatch generate -o job.sbatch -p compute -N 2 -n 64 -t 24:00:00 -J wien2k
forge_sbatch validate job.sbatch --strict
forge_sbatch submit job.sbatch
```

### Example 3.8 — JSON for automation

```bash
forge_sbatch --json generate -o job.sbatch -p compute -N 1 -t 02:00:00
```

---

## Model 4 — Wizard, TUI, and local / interactive runs

Use this model when there is no batch scheduler, or you are still choosing parameters.

### Example 4.1 — Interactive wizard

```bash
forge_wizard
```

Steps:

1. Topology detection.
2. WIENROOT and scratch check.
3. Target (time / memory / balanced / cost), max cores, memory limit.
4. Optional ELPA, Bayesian optimization, weighted k-points, RMT validation.
5. Review + write `.machines`.

Then run locally:

```bash
run_lapw -p
```

or submit with model 1.

### Example 4.2 — Local workstation, no scheduler

```bash
cd ~/wien2k/Cu
forge generate --reserve-os-cores 2 --mode kpoint
run_lapw -p
```

`forge submit` on a machine without SLURM/PBS/LSF will not have a batch provider. Run `run_lapw` directly.

### Example 4.3 — Interactive SLURM allocation (salloc)

```bash
salloc -N 1 -n 32 -t 02:00:00 -p debug
cd $SLURM_SUBMIT_DIR/Si
forge generate
run_lapw -p -i 30
exit
```

Good for debugging `.machines` in 30 minutes without writing a script.

### Example 4.4 — TUI

```bash
forge tui
```

Use when you want a terminal UI over generate / advise / diagnose rather than flags.

### Example 4.5 — Advise, then wizard, then submit

```bash
forge advise --case Fe --cores 128
forge_wizard
forge submit --partition compute --time 48:00:00
```

### Example 4.6 — Docker / Apptainer without host pip

```bash
docker run --rm -v "$PWD":/work -w /work forge:0.1.0 forge generate
```

WIEN2k itself is not in the image. Mount it or `module load` on the host, then submit with the site scheduler.

```bash
singularity exec forge.sif forge generate --dry-run
```

### Example 4.7 — Teaching / first Si example

```bash
cd examples/01_si_semiconductor
forge generate --max-cores 8 --mode kpoint --dry-run
# inspect
forge generate --max-cores 8 --mode kpoint
# if WIEN2k is installed:
# run_lapw -p
```

---

## Choosing a Model

```
Do you already have a reviewed SBATCH file or need --gres/--qos/--watch?
    yes -> Model 3  forge_sbatch
    no
        Are hostnames going to differ between login and compute?
            yes, multi-node cluster -> Model 2  generate inside the job
            no
                Is this a laptop, salloc, or first-time setup?
                    yes -> Model 4  wizard / local
                    no  -> Model 1  forge submit
```

Practical default on a SLURM cluster:

1. Converge RKmax and k-mesh on a small debug job (see `docs/preprocessing-convergence.md`).
2. `forge advise --case Case`.
3. `forge generate --mode ...` **or** put `forge generate` inside the batch script (model 2).
4. `forge submit` (model 1) or `sbatch` the model-2 script.

---

## Resource Flags Cheat Sheet

```bash
# Model 1
forge submit \
  --scheduler slurm|pbs|lsf|sge|auto \
  --partition NAME \
  --nodes N \
  --ntasks N \
  --cpus-per-task N \
  --time HH:MM:SS \
  --mem 64G \
  --job-name NAME \
  --dependency afterok:ID \
  --dry-run \
  --export path.sh \
  --auto-generate | --no-auto-generate

# Model 3
forge_sbatch generate \
  -o file.sbatch -J NAME -p PARTITION -N NODES -n TASKS -c CPUS \
  --mem 64G --time HH:MM:SS --dependency afterok:ID \
  --qos NAME --gres gpu:a100:1 --dry-run --preview
```

Time formats:

| Scheduler | Accepted |
|-----------|----------|
| SLURM | `HH:MM:SS`, `D-HH:MM:SS` |
| PBS | `HH:MM:SS` as walltime |
| LSF | `HH:MM` (FORGE converts from `HH:MM:SS`) |

Memory:

| Scheduler | Example |
|-----------|---------|
| SLURM | `64G`, `4000M` |
| PBS | FORGE writes `64gb` |
| LSF | passed through |

---

## After Submit

```bash
# SLURM
squeue -u $USER
scancel JOBID
tail -f slurm-JOBID.out

# PBS
qstat -u $USER
qdel JOBID

# LSF
bjobs
bkill JOBID
```

If the job dies in `lapw1para`:

1. `cat .machines` — do hostnames exist on this allocation?
2. `forge diagnostics`
3. See `docs/troubleshooting.md` and `docs/machines-guide.md` section 7.

If SCF oscillates rather than the parallel layer failing:

```bash
forge diagnose --log Case.scf
```

Then see `docs/preprocessing-convergence.md`.

---

## Related Documents

- `docs/machines-guide.md` — generate and read `.machines`
- `docs/parallel-modes.md` — kpoint / hybrid / mpi / fine_grain
- `docs/preprocessing-convergence.md` — RKmax, k-mesh, mixing before production
- `docs/user-guide.md` — full CLI reference
- `docs/troubleshooting.md` — scheduler and MPI failures
