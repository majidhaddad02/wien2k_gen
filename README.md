# WIEN2k Generator (wien2k_gen)

Production-grade WIEN2k parallel configuration file generator and HPC job dispatcher. Supports WIEN2k, VASP, Quantum ESPRESSO, and CP2K backends with SLURM integration.

## Quick Start

```bash
pip install wien2k_gen
wien2k_gen generate
wien2k_gen tui
```

## CLI Tools

- `wien2k_gen` — Full-featured CLI (generate, submit, benchmark, diagnostics, analyze, TUI)
- `wien2k_sbatch` — Dedicated SLURM batch job submission
- `wien2k_wizard` — Interactive configuration wizard

## Features

- Automatic hardware topology detection (NUMA, cache, ISA, interconnect)
- Multi-backend support (WIEN2k, VASP, Quantum ESPRESSO, CP2K)
- SLURM/PBS/LSF scheduler integration
- Roofline model-based resource optimization
- Interactive TUI via Textual
- Air-gapped HPC offline installation support
- Docker and Singularity/Apptainer containers
- SCF convergence monitoring and profiling

## Requirements

- Python >= 3.9
- Linux or macOS (HPC features require Linux)

## Development

```bash
make install
make test
make lint
```

## License

MIT License
