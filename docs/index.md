# FORGE (forge) — Documentation v0.1.0

FORGE generates a hardware-aware `.machines` file, submits WIEN2k jobs to SLURM/PBS/LSF/SGE, and diagnoses SCF failures so you do not hand-tune ranks, OpenMP, or queue resources. It detects topology, picks a parallel mode from NMAT and k-points, and writes files WIEN2k actually sources.

A valid WIEN2k license is required. FORGE only writes configuration; it does not replace WIEN2k. See [wien2k.at](http://www.wien2k.at/).

Tab-complete every flag from `completions/` (`forge.bash`, `forge.zsh`, plus `forge_sbatch.*` and `forge_wizard.*`). After `./install.sh`, open a new shell or `source ~/.local/opt/forge/env.sh`.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Workflow](workflow.md) | End-to-end map from `init_lapw` to a finished job |
| [Installation](installation.md) | `install.sh`, source, Docker — not PyPI |
| [User Guide](user-guide.md) | Every `forge` command and flag with examples |
| [`.machines` Guide](machines-guide.md) | Generate, read, and validate `.machines` |
| [Job Submission](job-submission.md) | `forge submit`, generate-inside-job, `forge_sbatch`, wizard |
| [Parallel Modes](parallel-modes.md) | kpoint, hybrid, mpi, fine_grain |
| [Preprocessing Convergence](preprocessing-convergence.md) | RKmax, k-mesh, GMAX, RMT, mixing |
| [Troubleshooting](troubleshooting.md) | Common errors, diagnostics, debug |
| [API Reference](api-reference.md) | Python module reference |
| [ML Dataset](ml_dataset.md) | GNN / Bayesian / history pipeline |
| [File Tree](file-tree.md) | Complete repository file diagram |
| [Contributing](contributing.md) | Development setup |

---

## Quick Overview

```bash
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
./install.sh --yes
forge generate
forge generate --mode hybrid --omp 4
forge generate --reserve-os-cores 4
forge submit --partition compute --time 48:00:00
forge advise --case Fe
forge diagnose --log case.scf
forge_wizard
```

---

## What It Detects

| What | How |
|------|-----|
| Scheduler | SLURM, PBS/Torque, LSF, SGE/GridEngine, or local |
| Hardware | Physical/logical cores, sockets, NUMA, HT |
| CPU | Architecture and generation, frequency |
| Memory | Total RAM, per-core, job limits, bandwidth |
| Network | InfiniBand, OmniPath, Ethernet |
| MPI | OpenMPI, Intel MPI, MPICH, MVAPICH |
| GPU | NVIDIA, AMD, Intel, generic `/dev/dri` |
| WIEN2k | Version 19/21/23/24, spin, SOC, LDA+U, hybrid, EECE, GPU flags |
| Input files | `.struct`, `.scf`, `.in1`, `.inm`, `.klist`, `.inso`, `.inorb`, `.inst` |
| System type | Metal / semiconductor / insulator from band gap |
| SCF | Charge sloshing, QTL-B, divergence type |

---

## License and Citation

MIT License. See `LICENSE.md`.

- **WIEN2k:** Blaha, P. et al. (2020). *J. Chem. Phys.* 152, 074101.
- **Amdahl's Law:** Amdahl, G. M. (1967). *AFIPS*, 30, 483-485.
- **Roofline:** Williams, S. et al. (2009). *CACM*, 52(4), 65-76.
- **Bayesian Opt:** Snoek, J. et al. (2012). *NIPS*, 25, 2951-2959.
- **This tool:** See `CITATION.cff`
