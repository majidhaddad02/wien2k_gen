# Job Submission Guide

FORGE fills SLURM/PBS/LSF/SGE resources from `.machines` so `ntasks` equals MPI rank count, `cpus-per-task` equals OpenMP width, and the batch script does **not** `exec` the calculation (EXIT_CODE / hooks still run).

Tab-complete: `completions/forge.bash`, `completions/forge_sbatch.bash` (and zsh twins).

| # | Model | Command | When |
|---|-------|---------|------|
| 1 | Auto submit | `forge submit` | Everyday production |
| 2 | Generate inside allocation | `sbatch` script calls `forge generate` then `run_lapw` | Hostnames must be live |
| 3 | SLURM helper | `forge_sbatch` | Review, `--gres`/`--qos`, validate, `--watch` |
| 4 | Wizard / local | `forge_wizard` or `run_lapw -p` | Laptop, `salloc`, teaching |

---

## Prerequisites

```bash
ls Case.struct Case.in1 Case.in2
which forge
export WIENROOT=/path/to/WIEN2k
```

If `.machines` is missing:

- Interactive `forge submit` asks whether to generate.
- Non-interactive fails unless `--auto-generate`.
- `--no-auto-generate` always fails when missing.

---

## Model 1 — `forge submit`

1. Finds `.machines` (or generates it).
2. Parses ranks, OpenMP width, node count.
3. Builds a scheduler script.
4. Submits unless `--dry-run`.

| Flag | Default if omitted |
|------|--------------------|
| `--nodes` | Hostnames in `.machines` (`0` = auto) |
| `--ntasks` | Rank count from `1:` / `lapw1:` (`0` = auto) |
| `--cpus-per-task` | `omp_global` or max rank width (`0` = auto) |
| `--mem` | Auto-detected node memory |
| `--time` | `24:00:00` |
| `--scheduler` | Auto-detect |
| `--job-name` | `wien2k_job` |

CLI flags win over the parsed file.

```bash
cd ~/wien2k/Si
forge generate
forge submit --partition compute --time 48:00:00

forge submit --partition compute --time 12:00:00 --dry-run

forge submit --partition compute --time 24:00:00 --export si_scf.sh --dry-run
sbatch si_scf.sh
forge submit --partition compute --time 24:00:00 --export si_scf.sh

forge submit --auto-generate --partition compute --time 08:00:00
forge submit --no-auto-generate --partition compute

forge submit --partition short --ntasks 16 --cpus-per-task 4 --time 04:00:00

forge submit --job-name fe-relax --partition compute --time 24:00:00
forge submit --job-name fe-scf --partition compute --time 48:00:00 --dependency afterok:9001

forge submit --scheduler pbs --partition batch --time 48:00:00 --mem 64g
forge submit --scheduler lsf --partition normal --time 12:00:00 --job-name nio-ldau
forge generate --scheduler sge
forge submit --scheduler sge --partition all.q --time 24:00:00

cd examples/03_fe_magnetic
forge generate --mode hybrid --omp 4
forge submit --partition compute --time 48:00:00 --job-name fe-sp-so --mem 192G

forge --json submit --partition compute --time 06:00:00
```

`--ntasks` is the number of MPI ranks, not cores. Hybrid 16 ranks x 4 threads → `--ntasks 16 --cpus-per-task 4`.

Do not `exec` the WIEN2k command at the end of a hand-written wrapper if you still need EXIT_CODE or epilog hooks; FORGE's generated scripts do not.

---

## Model 2 — Generate inside the allocation

Hostnames in `.machines` must be the hosts of **this** job.

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
export WIENROOT=/opt/WIEN2k_24.1
cd "$SLURM_SUBMIT_DIR"
forge generate --mode kpoint --target time
run_lapw -p -ec 0.0001 -cc 0.0001
```

```bash
sbatch run_si.sbatch
```

Hybrid Fe (ntasks x cpus-per-task must match `--omp`):

```bash
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=4
forge generate --mode hybrid --omp 4 --cores 256
runsp_lapw -p
```

PBS:

```bash
#PBS -l nodes=2:ppn=32
cd "$PBS_O_WORKDIR"
forge generate --scheduler pbs --mode kpoint
run_lapw -p
```

LSF:

```bash
#BSUB -n 64
cd "$LS_SUBCWD"
forge generate --scheduler lsf --mode hybrid --omp 4
run_lapw -p -orbc
```

Exclusive node, leave OS cores:

```bash
#SBATCH --exclusive
forge generate --reserve-os-cores 4 --mode hybrid --omp 4
run_lapw -p
```

GPU:

```bash
#SBATCH --gres=gpu:a100:2
forge generate --gpu --mode mpi --target time
run_lapw -p
```

Wrong: `forge generate` on the login node (`1: localhost:8`) then `sbatch`. Right: generate inside the job.

---

## Model 3 — `forge_sbatch`

Use when you need `--gres`, `--qos`, validation, preview, or `--watch`. `forge submit` does not expose `--qos` / `--gres`.

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

forge_sbatch generate --dry-run --partition compute --nodes 1 --time 04:00:00
forge_sbatch validate si.sbatch
forge_sbatch validate si.sbatch --strict
forge_sbatch preview si.sbatch --highlight
forge_sbatch submit si.sbatch --watch
forge_sbatch submit si.sbatch --dry-run

forge_sbatch generate \
  -o nio-gpu.sbatch -J nio-u -p gpu -N 1 -n 16 -c 8 \
  --gres gpu:a100:1 --qos high --time 12:00:00 --mem 256G \
  --dependency afterok:1842203

forge_sbatch --json generate -o job.sbatch -p compute -N 1 -t 02:00:00
```

`--ntasks 0` fills ntasks from topology. `--backup` rotates an existing script (default on).

---

## Model 4 — Wizard and local

```bash
forge_wizard
run_lapw -p

cd ~/wien2k/Cu
forge generate --reserve-os-cores 2 --mode kpoint
run_lapw -p

salloc -N 1 -n 32 -t 02:00:00 -p debug
cd $SLURM_SUBMIT_DIR/Si
forge generate
run_lapw -p -i 30
```

On a machine without a batch scheduler, `forge submit` has no provider — run `run_lapw` directly.

---

## Related Documents

- `docs/machines-guide.md` — reading `.machines`
- `docs/user-guide.md` — every submit / sbatch flag
- `docs/troubleshooting.md` — MPI, hostname, NUMA
- `docs/workflow.md` — end-to-end path
