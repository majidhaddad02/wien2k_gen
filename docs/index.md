# FORGE (forge) — Documentation v0.1.0

Parallel configuration file generator, HPC job dispatcher, and SCF convergence optimizer for WIEN2k. Features automatic hardware topology detection, NUMA-aware resource allocation, Amdahl's Law saturation analysis, Roofline performance modeling, multi-scheduler integration (SLURM, PBS, LSF, SGE), Bayesian hyperparameter optimization, GNN-based k-point prediction, and GPU offloading detection.

> **Important:** WIEN2k is copyrighted by P. Blaha, K. Schwarz, and collaborators at TU Wien. A valid license is required. Visit [wien2k.at](http://www.wien2k.at/).

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | System requirements, pip install, from source, air-gapped HPC |
| [User Guide](user-guide.md) | CLI commands, wizard, interactive TUI, mixing strategies, convergence tools |
| [API Reference](api-reference.md) | Python module reference for scripting and automation |
| [Examples](examples.md) | Real-world `.machines` output for different scenarios |
| [Parallel Modes](parallel-modes.md) | kpoint, hybrid, mpi, fine-grain — when to use each |
| [Troubleshooting](troubleshooting.md) | Common errors, diagnostics, debugging |

---

## Quick Overview

```bash
pip install forge
forge generate                    # auto-detect everything
forge generate --mode hybrid      # force hybrid mode
forge generate --reserve-os-cores 4  # leave 4 cores for OS
forge submit --partition compute --time 48:00:00
forge advise --case Fe            # performance bottleneck analysis
forge diagnose --log case.scf     # SCF convergence diagnostics
forge_wizard                          # interactive wizard
```

---

## What It Detects

| What | How |
|------|-----|
| **Scheduler** | SLURM, PBS/Torque, LSF, SGE/GridEngine, or local |
| **Hardware** | Physical/logical cores, sockets, NUMA nodes, HT status |
| **CPU** | Architecture (Intel Xeon/AMD EPYC/ARM), generation (SapphireRapids, Genoa...), frequency |
| **Memory** | Total RAM, per-core, job limits from scheduler, bandwidth (sysfs counters + STREAM) |
| **Network** | InfiniBand (mlx5/psm2), OmniPath, Ethernet with speed |
| **MPI** | OpenMPI, Intel MPI, MPICH, MVAPICH with binding hints |
| **GPU** | NVIDIA (nvidia-smi), AMD (rocm-smi), Intel (sycl-ls), generic (/dev/dri) |
| **WIEN2k** | Version (19/21/23/24), spin polarization, SOC, LDA+U, hybrid, EECE, GPU compilation flags |
| **Input files** | `.struct` (atoms, volume, RMT), `.scf` (NMAT, energy, band gap), `.in1` (nbands, GMAX), `.inm` (U, J) |
| **System type** | Metal / semiconductor / insulator from band gap in `.scf` |
| **SCF convergence** | Charge sloshing root cause, QTL-B errors, divergence type, SCF cycle timing |

---

## Feature Categories

### HPC Resource Management
- Automatic hardware detection with NUMA topology
- Memory bandwidth profiling from hardware counters
- Amdahl's Law saturation analysis
- Roofline model compute/memory-bound identification
- SLURM/PBS/LSF job submission with auto-detected parameters

### Parallelization Strategies
- k-point, hybrid, MPI, and fine-grain parallel modes
- ELPA eigensolver recommendation (threshold 8000 — WIEN2k benchmarks)
- FFD k-point distribution for load balancing
- NUMA-aware k-point allocation
- GPU offloading detection with hybrid CPU+GPU `.machines` generation

### SCF Convergence Optimization
- Smart Kerker q0 based on system type (Winkelmann et al. 2020, PRB 102, 195138)
- Restarted Pulay mixing for large systems (Pratapa & Suryanarayana 2015)
- Automatic checkpointing with adaptive intervals (Daly 2006)
- Charge sloshing root cause diagnosis with targeted remediation
- QTL-B error analysis with specific fix recommendations

### ML & AI Assistance
- Bayesian optimization with Matérn ν=2.5 kernel (Snoek et al. 2012)
- q-batch Expected Improvement for parallel evaluation
- GNN-based k-point prediction (CGCNN architecture)
- Physics-informed parameter priors
- History-driven warm-start from execution database

### Structure Validation
- RMT sphere overlap detection and automated optimization
- Nearest-neighbor distance calculation (3×3×3 supercell)
- setrmt algorithm: optimal RMT from structure (Blaha JCP 2020)

---

## License & Citation

MIT License. See [LICENSE.md](../LICENSE.md).

If you use forge in your research:

- **WIEN2k:** Blaha, P. et al. (2020). *J. Chem. Phys.* 152, 074101.
- **Amdahl's Law:** Amdahl, G. M. (1967). *AFIPS Conference Proceedings*, 30, 483-485.
- **Roofline:** Williams, S. et al. (2009). *CACM*, 52(4), 65-76.
- **Bayesian Opt:** Snoek, J. et al. (2012). *NIPS*, 25, 2951-2959.
- **This tool:** See [CITATION.cff](../CITATION.cff)
