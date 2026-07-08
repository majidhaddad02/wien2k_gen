# User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Commands](#cli-commands)
3. [Interactive Wizard](#interactive-wizard)
4. [Textual TUI](#textual-tui)
5. [Configuration File](#configuration-file)
6. [Batch Script Usage](#batch-script-usage)

---

## Quick Start

```bash
# 1. Initialize a WIEN2k case (standard WIEN2k workflow)
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000

# 2. Run one SCF cycle to generate .scf (provides NMAT)
run_lapw -p

# 3. Auto-generate optimal .machines
wien2k_gen generate

# 4. Submit to scheduler
wien2k_gen submit --partition compute --time 48:00:00
```

---

## CLI Commands

### `generate` — Create `.machines` and `parallel_options`

```bash
wien2k_gen generate [options]
```

| Option | Description | Example |
|--------|-------------|---------|
| `--mode` | Force parallel mode: `kpoint`, `hybrid`, `mpi`, `fine_grain` | `--mode hybrid` |
| `--cores` | Total MPI cores to use | `--cores 64` |
| `--omp` | OpenMP threads per MPI rank | `--omp 4` |
| `--max-cores` | Hard cap on total cores | `--max-cores 124` |
| `--reserve-os-cores` | Reserve N cores for OS/daemons | `--reserve-os-cores 4` |
| `--target` | Optimization target: `time`, `energy`, `cost`, `balanced` | `--target time` |
| `--scheduler` | Force scheduler: `slurm`, `pbs`, `lsf`, `sge`, `auto` | `--scheduler slurm` |
| `--dry-run` | Print config without writing files | `--dry-run` |
| `--export` | Export config summary as JSON | `--export config.json` |
| `--gpu` | Enable GPU-aware configuration | `--gpu` |
| `--manual` | Open `.machines` in `$EDITOR` after generation | `--manual` |

#### Examples

**Standard auto-detect (most common):**
```bash
wien2k_gen generate
```

**Force hybrid mode with 4 threads per rank:**
```bash
wien2k_gen generate --mode hybrid --omp 4
```

**Reserve 4 cores for OS on a 128-core workstation:**
```bash
wien2k_gen generate --reserve-os-cores 4
```

**Dry-run to preview without writing:**
```bash
wien2k_gen generate --dry-run
```

### `submit` — Submit job to scheduler

```bash
wien2k_gen submit [options]
```

| Option | Description |
|--------|-------------|
| `--scheduler` | Target scheduler: `slurm`, `pbs`, `lsf`, `sge`, `auto` |
| `--partition` | Scheduler partition/queue name |
| `--nodes` | Number of nodes (default: auto) |
| `--ntasks` | Total MPI tasks (default: auto) |
| `--time` | Walltime: HH:MM:SS (default: 24:00:00) |
| `--mem` | Memory per node |
| `--job-name` | Job identifier |
| `--dependency` | Job dependency (e.g., `afterok:12345`) |
| `--dry-run` | Generate script without submitting |

### `benchmark` — Run scaling benchmarks

```bash
wien2k_gen benchmark --type real --max-cores 64 --output scaling.json
```

### `diagnostics` — System audit

```bash
wien2k_gen diagnostics
wien2k_gen diagnostics --json > hw_report.json
```

### `tui` — Launch full-featured Textual UI

```bash
wien2k_gen tui
```

---

## Interactive Wizard

```bash
wien2k_wizard
```

The wizard walks through:
1. Hardware topology detection and display
2. WIEN2k installation validation
3. Backend selection (WIEN2k, Quantum ESPRESSO)
4. Optimization strategy (time, energy, cost)
5. Pre-flight checks and confirmation

---

## Textual TUI

```bash
wien2k_gen tui
```

Features:
- Reactive dashboard with system topology
- Resource configuration tabs (advanced, resources, settings, submit)
- Real-time hardware monitoring
- One-click generation and submission
- Keyboard-driven navigation

---

## Configuration File

Location: `~/.config/wien2k_gen/config.json`

```json
{
    "wienroot": "/opt/WIEN2k_24.1",
    "scratch_dir": "/scratch/user",
    "backend": "wien2k",
    "max_cores": 128,
    "log_level": "INFO"
}
```

Precedence: **CLI flags > Environment variables > Config file > Defaults**

---

## Batch Script Usage

### SLURM

```bash
#!/bin/bash
#SBATCH --job-name=wien2k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=48:00:00
#SBATCH --partition=compute

# Generate .machines from inside the job allocation
wien2k_gen generate

# Run WIEN2k
run_lapw -p
```

### PBS/Torque

```bash
#!/bin/bash
#PBS -N wien2k
#PBS -l nodes=2:ppn=32
#PBS -l walltime=48:00:00

wien2k_gen generate
run_lapw -p
```

### SGE/GridEngine

```bash
#!/bin/bash
#$ -N wien2k
#$ -pe mpi 64
#$ -l h_rt=48:00:00

wien2k_gen generate --scheduler sge
run_lapw -p
```

---

## Supported Calculation Types

wien2k_gen auto-detects the correct WIEN2k execution command from input files:

| Input Files | Detected Calculation | Command |
|-------------|---------------------|---------|
| `case.struct` | Standard SCF | `run_lapw -p` |
| `case.struct` + `case.inst` | Spin-polarized | `runsp_lapw -p` |
| `case.struct` + `case.inso` | Spin-orbit coupling | `run_lapw -p -so` |
| `case.inst` + `case.inso` | Spin-polarized + SOC | `runsp_lapw -p -so` |
| `case.struct` + `case.inorb` | LDA/GGA+U | `run_lapw -p -orbc` |
| `case.struct` + `HYBR` in `case.in0` | Hybrid functional | `run_lapw -p -hf` |
| `case.struct` + `case.ineece` | Onsite exact exchange | `run_lapw -p -eece` |
| `case.struct` + `case.in2` FOR | Forces | `run_lapw -p -fc` |

### Multi-Flag Combinations

For spin-polarized LDA+U with SOC:
```bash
runsp_lapw -p -so -orbc
```

The tool constructs the command dynamically by priority: spin → SOC → LDA+U → hybrid → EECE → forces.

---

## WIEN2k Version Compatibility

| wien2k_gen | WIEN2k 19.x | WIEN2k 21.x | WIEN2k 23.x | WIEN2k 24.x |
|------------|-------------|-------------|-------------|-------------|
| 0.1.0      | Partial     | Partial     | Yes         | Yes         |

- **WIEN2k_19:** Does not fully support `WIEN_GRANULARITY` for fine-grain parallelism
- **WIEN2k_21:** Requires hybrid MPI+OpenMP for best performance on modern clusters (≥32 cores/node)
- **WIEN2k_23+:** Full support for all parallel modes including per-stage core assignment
