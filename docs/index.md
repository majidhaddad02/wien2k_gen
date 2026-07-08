# WIEN2k Generator (wien2k_gen) — Documentation v0.1.0

Parallel configuration file generator and HPC job dispatcher for WIEN2k and Quantum ESPRESSO. Features automatic hardware topology detection, NUMA-aware resource allocation, Amdahl's Law saturation analysis, and multi-scheduler integration (SLURM, PBS, LSF, SGE).

> **Important:** WIEN2k is copyrighted by P. Blaha, K. Schwarz, and collaborators at TU Wien. A valid license is required. Visit [wien2k.at](http://www.wien2k.at/).

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | System requirements, pip install, from source, air-gapped HPC |
| [User Guide](user-guide.md) | CLI commands, wizard, interactive TUI, workflows |
| [API Reference](api-reference.md) | Python module reference for scripting and automation |
| [Examples](examples.md) | Real-world `.machines` output for different scenarios |
| [Parallel Modes](parallel-modes.md) | kpoint, hybrid, mpi, fine-grain — when to use each |
| [Troubleshooting](troubleshooting.md) | Common errors, diagnostics, debugging |
| [Contributing](contributing.md) | Development setup, coding standards, adding backends |

---

## Quick Overview

```bash
pip install wien2k_gen
wien2k_gen generate                    # auto-detect everything
wien2k_gen generate --mode hybrid      # force hybrid mode
wien2k_gen generate --reserve-os-cores 4  # leave 4 cores for OS
wien2k_gen submit --partition compute --time 48:00:00
wien2k_wizard                          # interactive wizard
wien2k_gen tui                         # full-featured Textual TUI
```

---

## What It Detects

| What | How |
|------|-----|
| **Scheduler** | SLURM, PBS/Torque, LSF, SGE/GridEngine, or local |
| **Hardware** | Physical/logical cores, sockets, NUMA nodes, HT status |
| **CPU** | Architecture (Intel Xeon/AMD EPYC/ARM), generation (SapphireRapids, Genoa...), frequency |
| **Memory** | Total RAM, per-core, job limits from scheduler |
| **Network** | InfiniBand (mlx5/psm2), OmniPath, Ethernet with speed |
| **MPI** | OpenMPI, Intel MPI, MPICH, MVAPICH with binding hints |
| **WIEN2k** | Version (19/21/23/24), spin polarization, SOC, LDA+U, hybrid, EECE |
| **Input files** | `.struct` (atoms, volume), `.scf` (NMAT, energy), `.in1` (nbands, GMAX), `.inm` (U, J) |

---

## License & Citation

MIT License. See [LICENSE.md](../LICENSE.md).

If you use wien2k_gen in your research:

- **WIEN2k:** Blaha, P. et al. (2020). *J. Chem. Phys.* 152, 074101. DOI: [10.1063/1.5143061](https://doi.org/10.1063/1.5143061)
- **Amdahl's Law:** Amdahl, G. M. (1967). *AFIPS Conference Proceedings*, 30, 483-485.
- **This tool:** See [CITATION.cff](../CITATION.cff)
