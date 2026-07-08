# WIEN2k Generator (wien2k_gen) v0.1.0

Parallel configuration file generator and HPC job dispatcher for WIEN2k density functional theory code. Features automatic hardware topology detection, NUMA-aware resource allocation, Amdahl's Law saturation analysis, and SLURM/PBS/LSF scheduler integration.

> **Note:** WIEN2k is a copyrighted code developed by P. Blaha, K. Schwarz, and collaborators at TU Wien. A valid WIEN2k license is required to use the generated configuration files. Visit [wien2k.at](http://www.wien2k.at/) for licensing information.

## WIEN2k Version Compatibility

| wien2k_gen | WIEN2k 19.x | WIEN2k 21.x | WIEN2k 23.x | WIEN2k 24.x |
|------------|-------------|-------------|-------------|-------------|
| 0.1.0      | Partial     | Partial     | Yes         | Yes         |

> **Known limitations:** WIEN2k_19 does not fully support fine-grain parallelism. WIEN2k_21 requires hybrid MPI+OpenMP for best performance on modern clusters. WIEN2k_23+ supports the `.machines` per-stage granularity syntax used by this tool.

## Citation

If you use wien2k_gen in your research, please cite:

- **WIEN2k primary reference:** Blaha, P., Schwarz, K., Tran, F., Laskowski, R., Madsen, G. K. H., & Marks, L. D. (2020). WIEN2k: An APW+lo program for calculating the properties of solids. *J. Chem. Phys.*, 152, 074101. DOI: [10.1063/1.5143061](https://doi.org/10.1063/1.5143061)
- **Amdahl's Law parallel scaling:** Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Conference Proceedings*, 30, 483-485.
- **This tool:** See [CITATION.cff](CITATION.cff) file.

## Quick Start

```bash
pip install wien2k_gen
wien2k_gen generate
wien2k_gen tui
```

## CLI Tools

| Command | Description |
|---------|------------|
| `wien2k_gen` | Full-featured CLI (generate, submit, benchmark, diagnostics, analyze, TUI) |
| `wien2k_sbatch` | Dedicated SLURM batch job submission |
| `wien2k_wizard` | Interactive configuration wizard |

## Features

- Automatic hardware topology detection (NUMA, cache, ISA, interconnect)
- Spin-polarization auto-detection (`run_lapw` vs `runsp_lapw`)
- Calculation flag detection: `-so`, `-orbc`, `-hf`, `-fc`, `-eece`
- SLURM/PBS/LSF scheduler integration
- Amdahl's Law saturation analysis for optimal core selection
- SCF convergence monitoring and profiling
- Air-gapped HPC offline installation support
- Docker and Singularity/Apptainer containers

## Supported Calculation Types

wien2k_gen automatically detects the correct WIEN2k execution command:

| Input Files | Detected Calculation | Command |
|-------------|---------------------|---------|
| `case.struct` | Standard SCF | `run_lapw -p` |
| `case.struct` + `case.inst` | Spin-polarized | `runsp_lapw -p` |
| `case.struct` + `case.inso` | Spin-orbit coupling | `run_lapw -p -so` |
| `case.inst` + `case.inso` | Spin-polarized + SOC | `runsp_lapw -p -so` |
| `case.struct` + `case.inorb` | LDA/GGA+U | `run_lapw -p -orbc` |
| `case.struct` + `HYBR` in `case.in0` | Hybrid functional | `run_lapw -p -hf` |
| `case.struct` + `case.ineece` | Onsite exact exchange | `run_lapw -p -eece` |

## Requirements

- Python >= 3.9
- Linux or macOS (HPC features require Linux)
- WIEN2k installation with valid license

## End-to-End Example

```bash
# 1. Initialize a WIEN2k case
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000

# 2. Run SCF to generate .scf file (provides NMAT for optimal parallelization)
run_lapw -p

# 3. Ask wien2k_gen for optimal parallel settings
wien2k_gen generate --target time

# 4. Submit to SLURM with auto-detected resources
wien2k_gen submit --partition compute --time 48:00:00

# Or use the interactive wizard
wien2k_wizard
```

## Parallel Execution Modes

| Mode | When to Use | Parallelism |
|------|-------------|-------------|
| **kpoint** | Many k-points (>4), small unit cells | k-point parallel, minimal communication |
| **hybrid** | General purpose, modern multi-core nodes | MPI + OpenMP mixed |
| **mpi** | Large matrices with ELPA, few k-points | Fine-grain ScaLAPACK/ELPA diagonalization |
| **fine_grain** | Very large systems, limited k-points | Atom-level decomposition via `WIEN_GRANULARITY` |

## Development

```bash
make install
make test
make lint
```

## License

MIT License. See [LICENSE.md](LICENSE.md).

**This project is not affiliated with or endorsed by the WIEN2k development team at TU Wien.**
