# User Guide

FORGE turns a WIEN2k case directory into a correct `.machines` file, a scheduler script, and a diagnosis of SCF / hardware bottlenecks. You do not count physical cores, guess OpenMP, or copy yesterday's hostnames.

Tab-complete every flag: `completions/forge.bash` and `completions/forge.zsh`. After `./install.sh`, open a new shell or `source ~/.local/opt/forge/env.sh`.

Related guides: `docs/machines-guide.md`, `docs/job-submission.md`, `docs/preprocessing-convergence.md`, `docs/workflow.md`.

---

## Quick Start

```bash
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000
run_lapw -p
forge advise --case case_name
forge diagnose --log case_name.scf
forge generate
forge submit --partition compute --time 48:00:00
```

---

## Global Flags

These apply to every `forge` subcommand. Place them **before** the command name.

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Increase verbosity (`-v`, `-vv`) |
| `-q`, `--quiet` | Suppress console output |
| `--json` | Print the command result as JSON |
| `--config PATH` | Custom config file |
| `--backend {wien2k,qe,vasp,cp2k}` | Override auto-detected backend |
| `--log-file PATH` | Redirect logs to a file |
| `--version` | Print `forge v0.1.0` |
| `--plain` | No Rich formatting (dumb terminals) |
| `--no-color` | Disable color |

```bash
forge -v generate --dry-run
forge -vv generate --mode hybrid
forge -q submit --partition compute --time 01:00:00 --dry-run
forge --json generate --dry-run
forge --json submit --partition compute --time 06:00:00
forge --config ~/.config/forge/config.json generate
forge --backend wien2k generate --dry-run
forge --backend qe generate --dry-run
forge --log-file /tmp/forge.log generate
forge --version
forge --plain diagnostics
forge --no-color advise --case Fe
forge --help
```

---

## `forge generate`

Write `.machines` and `parallel_options` from hardware, scheduler allocation, and case files.

`--target` choices are `time`, `memory`, `balanced`, `cost` (not `energy`). `--mode` choices are `mpi`, `hybrid`, `kpoint`, `serial`, `fine_grain`.

### `--nodes N`

Number of compute nodes. Default: detect from the scheduler.

```bash
forge generate --nodes 1
forge generate --nodes 4 --mode hybrid --omp 4
```

### `--cores N`

Total cores to allocate (Amdahl-clamped unless `--ignore-saturation`).

```bash
forge generate --cores 64
forge generate --cores 128 --mode mpi
```

### `--omp N`

OpenMP threads per MPI rank.

```bash
forge generate --omp 4
forge generate --mode hybrid --omp 8 --cores 256
```

### `--mode {mpi,hybrid,kpoint,serial,fine_grain}`

Force a parallel mode instead of auto-select.

```bash
forge generate --mode kpoint
forge generate --mode hybrid --omp 4
forge generate --mode mpi --cores 128
forge generate --mode fine_grain
forge generate --mode serial --cores 1
```

### `--target {time,memory,balanced,cost}`

Optimization target. Default: `time`.

```bash
forge generate --target time
forge generate --target memory --memory-limit 64
forge generate --target balanced
forge generate --target cost --max-cores 32
```

### `--max-cores N`

Hard cap on total cores.

```bash
forge generate --max-cores 16
forge generate --max-cores 124 --reserve-os-cores 4
```

### `--reserve-os-cores N`

Leave N cores for OS/daemons (e.g. 4 leaves 124 of 128).

```bash
forge generate --reserve-os-cores 4
forge generate --reserve-os-cores 2 --mode kpoint
```

### `--memory-limit GB`

Hard limit on memory per node in GB.

```bash
forge generate --memory-limit 64
forge generate --memory-limit 192 --mode hybrid --omp 4
```

### `--dry-run`

Print the plan; do not write files.

```bash
forge generate --dry-run
forge generate --mode kpoint --dry-run --export kpoint.json
```

### `--export PATH`

Write a configuration summary (JSON) to PATH.

```bash
forge generate --export config.json
forge generate --dry-run --export /tmp/plan.json
```

### `--overwrite`

Overwrite existing `.machines` without a prompt.

```bash
forge generate --overwrite
forge generate --overwrite --mode hybrid --omp 4
```

### `--scheduler`, `-S {slurm,pbs,lsf,sge,auto}`

Target scheduler. Default: `auto`.

```bash
forge generate --scheduler slurm
forge generate -S pbs --mode kpoint
forge generate --scheduler lsf
forge generate --scheduler sge
forge generate --scheduler auto
```

### `--gpu`

GPU-aware `.machines` (WIEN2k 24+ compiled with GPU).

```bash
forge generate --gpu
forge generate --gpu --target time --mode mpi
```

### `--gpu-mixed-precision`

Enable FP32/FP16 mixed precision with `--gpu`.

```bash
forge generate --gpu --gpu-mixed-precision
```

### `--manual`

After generation, open `.machines` in `$EDITOR`.

```bash
export EDITOR=vim
forge generate --manual
```

### `--ignore-saturation`

Do not clamp cores to Amdahl `max_efficient_cores`.

```bash
forge generate --cores 256 --ignore-saturation
```

### `--recalibrate`

Invalidate cached Roofline data and re-measure before generating.

```bash
forge generate --recalibrate --target time
```

---

## `forge submit`

Build a scheduler script from `.machines` and submit it.

Resource auto-fill when a flag is `0` / omitted:

| Flag | Default |
|------|---------|
| `--nodes` | Hostnames in `.machines` (`0` = auto) |
| `--ntasks` | Rank count from `1:` / `lapw1:` (`0` = auto) |
| `--cpus-per-task` | `omp_global` or max rank width (`0` = auto) |
| `--mem` | Auto-detected node memory |
| `--time` | `24:00:00` |
| `--scheduler` | Auto-detect |
| `--job-name` | `wien2k_job` |

`--auto-generate` and `--no-auto-generate` are mutually exclusive.

### `--scheduler`, `-S {slurm,pbs,lsf,sge,auto}`

```bash
forge submit --scheduler slurm --partition compute --time 24:00:00
forge submit -S pbs --partition batch --time 48:00:00 --mem 64g
forge submit --scheduler lsf --partition normal --time 12:00:00
forge submit --scheduler sge --partition all.q --time 24:00:00
forge submit --scheduler auto --partition compute
```

### `--partition`

```bash
forge submit --partition compute --time 48:00:00
forge submit --partition gpu --time 12:00:00 --mem 256G
```

### `--nodes N`

`0` (default) = auto from `.machines`.

```bash
forge submit --nodes 2 --partition compute --time 24:00:00
forge submit --nodes 0 --partition compute
```

### `--ntasks N`

`0` (default) = rank count from `.machines`.

```bash
forge submit --ntasks 32 --cpus-per-task 4 --partition short --time 04:00:00
forge submit --ntasks 0 --partition compute
```

### `--cpus-per-task N`

`0` (default) = OpenMP width from `.machines`.

```bash
forge submit --cpus-per-task 4 --ntasks 16 --partition compute --time 08:00:00
forge submit --cpus-per-task 1 --partition compute
```

### `--time HH:MM:SS`

```bash
forge submit --time 48:00:00 --partition compute
forge submit --time 04:00:00 --partition debug --dry-run
```

### `--mem`

```bash
forge submit --mem 64G --partition compute --time 24:00:00
forge submit --mem 192G --job-name fe-sp-so --partition compute
```

### `--job-name`

```bash
forge submit --job-name fe-relax --partition compute --time 24:00:00
forge submit --job-name si-scf --partition compute
```

### `--dependency`

```bash
forge submit --job-name fe-scf --dependency afterok:9001 --partition compute --time 48:00:00
forge submit --job-name fe-bands --dependency afterok:9002 --partition compute --time 12:00:00
```

### `--dry-run`

```bash
forge submit --partition compute --time 12:00:00 --dry-run
```

### `--export PATH`

```bash
forge submit --partition compute --time 24:00:00 --export si_scf.sh --dry-run
forge submit --partition compute --time 24:00:00 --export si_scf.sh
```

### `--auto-generate`

Generate `.machines` if missing, then submit.

```bash
forge submit --auto-generate --partition compute --time 08:00:00
```

### `--no-auto-generate`

Fail if `.machines` is missing (CI).

```bash
forge submit --no-auto-generate --partition compute
```

---

## `forge advise`

Roofline + Amdahl + NUMA advice. There is **no** `--verbose` on this command; use global `-v`. `--target` here is `time`, `energy`, `cost`, `balanced` (advice goal, distinct from `generate --target`).

### `--case`

```bash
forge advise --case Fe
forge advise --case La2CuO4 --cores 128
```

### `--nmat`

```bash
forge advise --case Fe --nmat 8000
```

### `--kpoints`

```bash
forge advise --case Si --kpoints 1000
```

### `--cores`

```bash
forge advise --case Fe --cores 128
forge advise --cores 64 --target time
```

### `--target {time,energy,cost,balanced}`

```bash
forge advise --case Fe --target time
forge advise --case Fe --target energy
forge advise --case Fe --target cost
forge advise --case Fe --target balanced
```

### `--ignore-saturation`

```bash
forge advise --case Fe --cores 256 --ignore-saturation
```

### `--recalibrate`

```bash
forge advise --case Fe --recalibrate
```

### `--json`

Command-local JSON (also available as global `--json`).

```bash
forge advise --case Fe --json
forge --json advise --case Fe
```

```bash
forge -v advise --case Fe --cores 128
```

---

## `forge diagnose`

SCF root-cause analysis (sloshing, QTL-B, divergence).

Positional `case` is optional.

```bash
forge diagnose Fe
forge diagnose --log Fe.scf
forge diagnose Fe --log Fe.scf
```

---

## `forge diagnostics`

Hardware and environment audit.

```bash
forge diagnostics
forge diagnostics --full
forge diagnostics --export report.json
forge diagnostics --full --export /tmp/hw.json
forge --json diagnostics
```

---

## `forge hardware`

Print detected topology; optionally recommend NUMA/hybrid/IO settings.

```bash
forge hardware
forge hardware --recommend
forge hardware -r --case Fe
forge hardware --case Si
```

---

## `forge analyze`

Parse an SCF / output log.

```bash
forge analyze --log Fe.scf
forge analyze --log Fe.scf --code wien2k
forge analyze --log OUTCAR --code vasp --export vasp.json
forge analyze --log pw.out --code qe --export qe.json
```

---

## `forge tui`

Interactive Textual UI.

```bash
forge tui
forge tui --compact
```

---

## `forge monitor`

Poll a running case (or list active jobs if `case` is omitted).

```bash
forge monitor
forge monitor Fe
forge monitor Fe --interval 5
forge monitor Fe --output Fe.scf --interval 2
```

---

## `forge run`

Execute a `workflow.yaml`.

```bash
forge run workflow.yaml
forge run workflow.yaml --auto-retry --max-retries 5
forge run workflow.yaml --no-retry
forge run workflow.yaml --poll 10
```

---

## `forge workflow`

Create, list, or visualize a DAG.

```bash
forge workflow create --case Si --steps scf,dos,band
forge workflow list
forge workflow visualize --case Si --output wf.yaml
forge workflow create --steps scf --output /tmp/wf.yaml
```

---

## `forge optimize`

Bayesian tuning of RKMAX, k-points, mixing.

```bash
forge optimize --case Fe
forge optimize --case Fe --budget 20
forge optimize --case Fe --target energy_convergence
forge optimize --case Fe --strategy gp_ei
forge optimize --case Fe --strategy bohb --budget 15
forge optimize --case Fe --simulated
forge optimize --case Fe --verbose
```

---

## `forge screen`

Query Materials Project.

```bash
forge screen --formula ABO3 --max 20
forge screen --elements Ti,O,Zr --max 50
forge screen --mp-id mp-149
forge screen --formula LiFePO4 --api-key "$MP_API_KEY" --output ./hits
```

---

## `forge predict`

GNN k-point grid prediction.

```bash
forge predict --case Fe
forge predict --struct Fe.struct
forge predict --case Fe --no-history
```

---

## `forge converge`

RKmax / k-mesh scans. `--tolerance` is in Ry.

```bash
forge converge --case Si --mode both
forge converge --case Si --mode rkmax --rkmax "6,7,8,9"
forge converge --case Si --mode kpoints --kpoints "6,6,6 8,8,8 10,10,10"
forge converge --case Si --tolerance 0.001 --mode both
```

---

## `forge history`

Query the execution SQLite store.

```bash
forge history --list
forge history --list --limit 20
forge history --show RUN_ID
forge history --similar-to Fe
forge history --export runs.csv --format csv
forge history --export runs.json --format json --backend wien2k
```

---

## `forge analyze-bands`

```bash
forge analyze-bands --case Si
forge analyze-bands --case Si --dos
forge analyze-bands --case Si --output bands.json --dos
```

---

## `forge calibrate`

Re-measure and cache hardware Roofline (bandwidth, FLOPS, cache).

```bash
forge calibrate
forge calibrate --json
```

---

## `forge benchmark`

Weak/strong scaling suite.

```bash
forge benchmark --type real
forge benchmark --type synthetic --max-cores 64
forge benchmark --type real --max-cores 64 --walltime 02:00:00 --output scaling.json
forge benchmark --skip-cleanup --scheduler slurm
forge benchmark -S pbs --type real
forge benchmark --scheduler lsf --max-cores 32
forge benchmark --scheduler auto
```

`--scheduler` choices: `slurm`, `pbs`, `lsf`, `auto` (no `sge`).

---

## `forge_sbatch`

Dedicated SLURM helper: generate / validate / preview / submit. Completions: `completions/forge_sbatch.bash` and `completions/forge_sbatch.zsh`.

Global on this binary: `-v`, `-q`, `--json`, `--backend`, `--config`, `--log-file`.

### `generate`

```bash
forge_sbatch generate --partition compute --time 24:00:00
forge_sbatch generate -o si.sbatch -J si-scf -p compute -N 2 -n 64 -c 1 --mem 128G -t 24:00:00 --preview
forge_sbatch generate --dry-run --partition compute --nodes 1 --time 04:00:00
forge_sbatch generate --qos high --gres gpu:a100:2 --dependency afterok:12345 -p gpu
forge_sbatch --json generate -o job.sbatch -p compute -N 1 -t 02:00:00
```

`--ntasks 0` (default) fills ntasks from topology. `--backup` is on by default.

### `validate`

```bash
forge_sbatch validate si.sbatch
forge_sbatch validate si.sbatch --strict
```

### `preview`

```bash
forge_sbatch preview si.sbatch
forge_sbatch preview si.sbatch --highlight
```

### `submit`

```bash
forge_sbatch submit si.sbatch
forge_sbatch submit si.sbatch --dry-run
forge_sbatch submit si.sbatch --watch
```

---

## `forge_wizard`

Interactive setup (topology, WIENROOT, target, optional ELPA / BO / FFD / RMT). Completions: `completions/forge_wizard.bash` and `completions/forge_wizard.zsh`.

```bash
forge_wizard
```

Then `run_lapw -p` or `forge submit`.

---

## Configuration File

Location: `~/.config/forge/config.json`.

Only keys that `AppConfig` actually loads:

```json
{
    "version": "1.2.0",
    "wienroot": "/opt/WIEN2k_24.1",
    "scratch_dir": "/scratch/user",
    "config_dir": "/home/user/.config/forge",
    "cache_dir": "/home/user/.cache/forge",
    "log_level": "INFO",
    "backend": "wien2k",
    "max_cores": 128,
    "timeout_sec": 300.0,
    "enable_tui": true,
    "quiet_mode": false,
    "dry_run": false,
    "custom_paths": {}
}
```

Unknown keys are ignored. Precedence: **CLI flags > environment > config file > defaults**.

Override the file with `--config`:

```bash
forge --config /tmp/forge.json generate --dry-run
```

---

## Interactive Wizard Steps

| Step | Description |
|------|-------------|
| 1. Topology | Cores, NUMA, memory, scheduler |
| 2. WIEN2k Setup | WIENROOT, scratch health |
| 3. Optimization | Target `time` / `memory` / `balanced` / `cost`, max cores, memory limit |
| 3.5 Advanced | ELPA (threshold 8000), Bayesian optimization, weighted k-points, struct validation |
| 4. Review | Advisor summary |
| 5. Generate | Write `.machines` and `parallel_options` |

---

## Mixing Strategies

| Strategy | When | Key parameter |
|----------|------|----------------|
| Broyden | <=50 atoms | Default WIEN2k |
| Kerker | Metals | q0 from system type + lattice constant |
| Restarted Pulay | >50 atoms | history_size=7, regularization=1e-10 |
| Pulay + Kerker | Large + metallic | Combined |

Kerker q0 (Winkelmann et al. 2020, PRB 102, 195138): metal `0.4 * 2pi/a`, semiconductor `0.15 * 2pi/a`, insulator `0.05 * 2pi/a`.

---

## Detected Execution Commands

| Input files | Command |
|-------------|---------|
| `case.struct` | `run_lapw -p` |
| `case.inst` | `runsp_lapw -p` |
| `case.inso` | `run_lapw -p -so` |
| `case.inst` + `case.inso` | `runsp_lapw -p -so` |
| `case.inorb` | `run_lapw -p -orbc` |
| `HYBR` in `case.in0` | `run_lapw -p -hf` |
| `case.ineece` | `run_lapw -p -eece` |
| `FOR` in `case.in2` | `run_lapw -p -fc` |

---

## WIEN2k Version Compatibility

| forge | 19.x | 21.x | 23.x | 24.x |
|-------|------|------|------|------|
| 0.1.0 | Partial | Partial | Yes | Yes |

WIEN2k 19 does not fully support `WIEN_GRANULARITY`. WIEN2k 24 is recommended for GPU offload.
