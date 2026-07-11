# User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Commands](#cli-commands)
3. [Interactive Wizard](#interactive-wizard)
4. [Performance Analysis](#performance-analysis)
5. [SCF Convergence Diagnostics](#scf-convergence-diagnostics)
6. [Configuration File](#configuration-file)
7. [Mixing Strategies](#mixing-strategies)
8. [Batch Script Usage](#batch-script-usage)
9. [Advanced Physics Options](#advanced-physics-options)

---

## Quick Start

```bash
# 1. Initialize a WIEN2k case (standard WIEN2k workflow)
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000

# 2. Run one SCF cycle to generate .scf (provides NMAT)
run_lapw -p

# 3. Analyze performance bottleneck and SCF health
wien2k_gen advise --case case_name
wien2k_gen diagnose --log case_name.scf

# 4. Auto-generate optimal .machines
wien2k_gen generate

# 5. Submit to scheduler
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

**With GPU-aware configuration:**
```bash
wien2k_gen generate --gpu --target time
```

---

### `advise` — Performance Bottleneck Analysis

```bash
wien2k_gen advise [options]
```

Provides Roofline model analysis + Amdahl's Law saturation + NUMA topology recommendations.

| Option | Description |
|--------|-------------|
| `--case` | Case name (reads `.scf` for NMAT, `.struct` for atoms) |
| `--cores` | Total cores to analyze |
| `--plain` | Simplify output (Persian/non-expert friendly) |
| `--verbose` | Show full backend trace and detailed calculations |

#### Examples
```bash
# Full English analysis
wien2k_gen advise --case Fe --cores 128

# Persian-friendly simplified output
wien2k_gen advise --case Si --plain

# Detailed debug mode
wien2k_gen advise --case La2CuO4 --verbose
```

The advise command displays:
- **Roofline Analysis**: Compute-bound vs memory-bound identification with crossover point
- **Amdahl Saturation**: Optimal core count before diminishing returns
- **NUMA Topology**: Memory bandwidth per socket, recommended MPI rank placement
- **Bottleneck Warnings**: Color-coded (red=critical, yellow=warning, green=optimal)

---

### `diagnose` — SCF Convergence Diagnostics

```bash
wien2k_gen diagnose [options]
```

Deep analysis of SCF convergence issues with root cause identification.

| Option | Description |
|--------|-------------|
| `--log` | Path to `.scf` or dayfile for analysis |
| `--case` | Case name (auto-locates `.scf` and `.inc`) |
| `--plain` | Simplify output (Persian/non-expert friendly) |

#### Diagnostics Performed

1. **SCF Metrics**: cycles completed, final energy, charge distance, convergence status
2. **Charge Sloshing Detection**: avg charge ratio, oscillation percentage, divergence type
3. **Root Cause Analysis**:
   - **Metallic** (band gap < 0.1 eV) → Kerker mixing + Methfessel-Paxton smearing
   - **Symmetry breaking** → Disable symmetry, reduce RMT
   - **Core overlap** (RMT ratio > 1.5) → Check RMT, adjust R0
   - **Aggressive mixing** → Reduce beta, increase PRATT cycles
4. **QTL-B Error Analysis**: linearization energy advice, GMAX recommendations, `init_lapw -b` check
5. **Divergence Detection**: catastrophic, monotonic drift, charge sloshing, stalled convergence

#### Examples
```bash
# Analyze SCF convergence
wien2k_gen diagnose --log Fe.scf

# Full case analysis with mixing history
wien2k_gen diagnose --case Fe

# Persian output
wien2k_gen diagnose --log Fe.scf --plain
```

Example output:
```
                    SCF Diagnostics: Fe.scf
┌────────────────────────┬──────────────────────────────────────┐
│ Cycles completed       │ 14                                   │
│ Final energy           │ -2545.123456 Ry                      │
│ Final charge distance  │ 0.000005                             │
│ Converged              │ True                                 │
│ Avg charge ratio       │ 1.892                                │
│ Diagnosis              │ Charge sloshing detected — see below │
└────────────────────────┴──────────────────────────────────────┘

╭──── Charge Sloshing Root Cause ────────────────────────────────────╮
│ Root cause: metallic (confidence: 0.90)                            │
│                                                                     │
│ Action 1: Set Kerker mixing (q0=0.251, beta=0.10)                  │
│ Action 2: Enable MP smearing (width=0.02 Ry)                       │
│ Action 3: Increase k-mesh density (factor 2.0)                     │
╰────────────────────────────────────────────────────────────────────╯
```

---

### `benchmark` — Performance Scaling

```bash
wien2k_gen benchmark --type real --max-cores 64 --output scaling.json
```

Runs weak/strong scaling benchmarks with uncertainty quantification and bottleneck identification.

---

### `diagnostics` — System Audit

```bash
wien2k_gen diagnostics
wien2k_gen diagnostics --json > hw_report.json
```

Includes GPU detection, ELPA availability, memory bandwidth, and WIEN2k compilation status.

---

## Interactive Wizard

```bash
wien2k_wizard
```

### Wizard Steps:

| Step | Description |
|------|-------------|
| **1. Topology** | Auto-detect hardware — cores, NUMA, memory, scheduler |
| **2. WIEN2k Setup** | Validate WIENROOT, check scratch health (fs type, free space) |
| **3. Optimization** | Select target (time/memory/balanced/cost), max cores, memory limit |
| **3.5 Advanced** | Physics options: ELPA (threshold 8000), Bayesian optimization, weighted k-points, struct validation |
| **4. Review** | Advisor recommendation summary with confidence score |
| **5. Generate** | Write `.machines` and `parallel_options`, optional manual review |

### Advanced Options (Step 3.5):

- **ELPA Solver**: Enable for large systems (nmat > 8000) based on WIEN2k benchmarks
- **Bayesian Optimization**: Automatically tune RKMAX and mixing parameters
- **Weighted K-points**: FFD bin-packing for load-balanced k-point distribution
- **Struct Validation**: Automatic RMT overlap detection with warnings

---

## Performance Analysis

### Roofline Model

The `advise` command computes:

- **Operational Intensity** (FLOP/byte) from nmat, kpoints, and FFT grid
- **Peak Performance** (GFLOP/s) from CPU architecture
- **Memory Bandwidth** (GB/s) from sysfs counters or STREAM benchmark

Output identifies whether the system is compute-bound or memory-bound, with the crossover point where additional cores stop providing speedup.

### Amdahl's Law Saturation

For any given parallel mode and core count:
- Serial fraction estimated from problem size
- Maximum theoretical speedup
- Sweet-spot core count before efficiency drops below 80%
- Saturation warnings when adding cores is counterproductive

---

## SCF Convergence Diagnostics

### Smart Kerker q0 (Winkelmann et al. 2020, PRB 102, 195138)

The system type is auto-detected from the band gap in `case.scf`:

| System Type | Band Gap (eV) | Kerker q0 Formula |
|-------------|---------------|-------------------|
| Metal | gap < 0.1 | `q₀ = 0.4 × (2π/a)` |
| Semiconductor | 0.1 ≤ gap ≤ 0.5 | `q₀ = 0.15 × (2π/a)` |
| Insulator | gap > 0.5 | `q₀ = 0.05 × (2π/a)` |

The lattice constant `a` is extracted from the `.struct` file. The q0 parameter controls the wavelength cutoff for charge density preconditioning.

### Charge Sloshing Root Causes

| Root Cause | Detection | Remediation |
|-----------|----------|-------------|
| **Metallic** | :GAP < 0.1 or FERMI keyword | Kerker mixing + MP smearing (0.02 Ry) + denser k-mesh |
| **Symmetry breaking** | "symmetry broken" in dayfile | Disable symmetry (`runsp_lapw`), reduce mixing to 0.05 |
| **Core overlap** | RMT ratio > 1.5 | Check RMT values, reduce R0 to 0.90, mixing to 0.02 |
| **Aggressive mixing** | mixing beta > 0.3 in `.inc` | Reduce beta to 0.05, increase PRATT to 3 cycles, try MSR1a |

---

## Mixing Strategies

| Strategy | When Applied | Algorithm | Reference |
|----------|-------------|-----------|-----------|
| **Broyden** | Small systems (≤50 atoms) | Default WIEN2k mixing | — |
| **Kerker** | Metallic systems | Preconditioned mixing with q0 control | Winkelmann et al. 2020 |
| **Restarted Pulay** | Large systems (>50 atoms) | history_size=7, Tikhonov reg=1e-10 | Pratapa & Suryanarayana 2015 |
| **Pulay + Kerker** | Large + metallic | Combined restart + preconditioning | — |

The mixing strategy is selected automatically by `_adjust_mixing()` in the workflow executor based on:
1. System type from `.scf` band gap
2. Number of atoms from `.struct`
3. Current SCF convergence state

---

## Configuration File

Location: `~/.config/wien2k_gen/config.json`

```json
{
    "wienroot": "/opt/WIEN2k_24.1",
    "scratch_dir": "/scratch/user",
    "backend": "wien2k",
    "max_cores": 128,
    "log_level": "INFO",
    "elpa_threshold": 8000,
    "use_bayesian_optimization": false,
    "use_ffd_distribution": true,
    "enable_auto_checkpoint": true,
    "max_checkpoints_to_keep": 3,
    "checkpoint_dir": ".checkpoints",
    "enable_gpu_detection": true,
    "enable_numa_aware_distribution": true,
    "pulay_history_size": 7,
    "pulay_regularization": 1e-10,
    "rmt_reduction_factor": 0.95,
    "min_rmt": 2.5,
    "max_rmt": 4.0,
    "metal_q0_factor": 0.4,
    "semiconductor_q0_factor": 0.15,
    "insulator_q0_factor": 0.05
}
```

Precedence: **CLI flags > Environment variables > Config file > Defaults**

---

## Advanced Physics Options

### Bayesian Optimization

The Bayesian hyperparameter optimizer tunes 5 parameters simultaneously:

| Parameter | Range | Type |
|-----------|-------|------|
| RKMAX | [5.0, 9.0] | Continuous |
| Mixing beta | [0.05, 1.0] | Continuous |
| K-point density | [100, 2000] | Integer |
| GMAX | [10.0, 20.0] | Continuous |
| LMAX APW | [8, 12] | Discrete |

Uses Matérn ν=2.5 kernel (Snoek et al. 2012) for modeling non-smooth SCF convergence surfaces. Supports q-batch Expected Improvement for parallel evaluation of up to 4 candidates simultaneously.

### GPU Offloading

GPU detection automatically identifies:
- NVIDIA GPUs via `nvidia-smi`
- AMD GPUs via `rocm-smi`
- Intel GPUs via `sycl-ls`
- Generic GPUs via `/dev/dri`

Offload analysis determines which lapw stages benefit from GPU:

| Stage | GPU Benefit | Threshold | Expected Speedup |
|-------|------------|----------|------------------|
| lapw0 | None | — | 1× (I/O bound) |
| lapw1 | High | nmat > 5000 | min(10, nmat/1000)× |
| lapw2 | Medium | nmat > 8000 | min(5, nmat/2000)× |
| core | None | — | 1× (sequential) |

GPU memory is estimated as `nmat² × 16 bytes × kpts × 1.5 / (1024²)` MB with 90% safety threshold.

### RMT Optimization (setrmt Algorithm)

Based on Blaha et al. (JCP 2020):
- Calculates nearest-neighbor distances via 3×3×3 supercell search
- Optimal RMT = 0.95 × (nn_distance / 2), clamped [2.5, 4.0] a.u.
- Detects RMT overlaps: warning >0.95, critical >1.00
- Auto-reduces overlapping RMT values proportionally

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

# Enable checkpointing for large jobs
export WIEN2KGEN_CHECKPOINT=1

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

---

## WIEN2k Version Compatibility

| wien2k_gen | WIEN2k 19.x | WIEN2k 21.x | WIEN2k 23.x | WIEN2k 24.x |
|------------|-------------|-------------|-------------|-------------|
| 0.1.0      | Partial     | Partial     | Yes         | Yes         |

- **WIEN2k_19:** Does not fully support `WIEN_GRANULARITY` for fine-grain parallelism. ELPA support introduced.
- **WIEN2k_21:** Requires hybrid MPI+OpenMP for best performance on modern clusters (≥32 cores/node). Experimental GPU support.
- **WIEN2k_23:** Full support for all parallel modes including per-stage core assignment. Enhanced fine-grain parallelism.
- **WIEN2k_24:** Full fine-grain + GPU offloading support. Recommended version.
