# FORGE — Flexible Orchestration for Robust Generation of Electronic-structure jobs (v0.1.0)

**FORGE** (forge) is a production-grade, HPC job dispatcher, and SCF convergence optimizer for WIEN2k density functional theory code. Features automatic hardware topology detection, NUMA-aware resource allocation, Amdahl's Law saturation analysis, Roofline performance modeling, multi-scheduler integration (SLURM, PBS, LSF), Bayesian hyperparameter optimization, and AI-assisted k-point prediction.

> **Note:** WIEN2k is a copyrighted code developed by P. Blaha, K. Schwarz, and collaborators at TU Wien. A valid WIEN2k license is required to use the generated configuration files. Visit [wien2k.at](http://www.wien2k.at/) for licensing information.

## WIEN2k Version Compatibility

| forge | WIEN2k 19.x | WIEN2k 21.x | WIEN2k 23.x | WIEN2k 24.x |
|------------|-------------|-------------|-------------|-------------|
| 0.1.0      | Partial     | Partial     | Yes         | Yes         |

> **Known limitations:** WIEN2k_19 does not fully support fine-grain parallelism. WIEN2k_21 requires hybrid MPI+OpenMP for best performance on modern clusters. WIEN2k_23+ supports the `.machines` per-stage granularity syntax and GPU offloading used by this tool.

## Citation

- **WIEN2k primary reference:** Blaha, P., Schwarz, K., Tran, F., Laskowski, R., Madsen, G. K. H., & Marks, L. D. (2020). WIEN2k: An APW+lo program for calculating the properties of solids. *J. Chem. Phys.*, 152, 074101.
- **Amdahl's Law:** Amdahl, G. M. (1967). *AFIPS Conference Proceedings*, 30, 483-485.
- **Roofline Model:** Williams, S., Waterman, A., & Patterson, D. (2009). *CACM*, 52(4), 65-76.
- **Bayesian Optimization:** Snoek, J., Larochelle, H., & Adams, R. P. (2012). *NIPS*, 25, 2951-2959.
- **This tool:** See [CITATION.cff](CITATION.cff) file.

## Quick Start

```bash
pip install forge
forge generate
forge submit --partition compute --time 48:00:00
forge_wizard                           # interactive configuration
```

## CLI Tools

| Command | Description |
|---------|------------|
| `forge` | Full-featured CLI (generate, submit, benchmark, diagnostics, analyze, advise, diagnose, TUI) |
| `forge generate` | Auto-detect topology and generate `.machines` + `parallel_options` |
| `forge submit` | Schedule and submit WIEN2k job to queue |
| `forge advise` | Roofline + Amdahl + NUMA performance analysis with hardware-aware recommendations |
| `forge diagnose` | SCF convergence diagnosis — charge sloshing root cause, QTL-B analysis, divergence detection |
| `forge benchmark` | Run weak/strong scaling benchmarks with uncertainty quantification |
| `forge_sbatch` | Dedicated SLURM batch job submission |
| `forge_wizard` | Interactive configuration wizard with advanced physics options |

## Features

### HPC Infrastructure
- Automatic hardware topology detection (NUMA, cache, ISA, interconnect)
- Memory bandwidth measurement via sysfs counters + STREAM benchmark integration
- Spin-polarization auto-detection (`run_lapw` vs `runsp_lapw`)
- Calculation flag detection: `-so`, `-orbc`, `-hf`, `-fc`, `-eece`
- SLURM/PBS/LSF/SGE scheduler integration
- Amdahl's Law saturation analysis for optimal core selection
- Roofline model analysis (compute-bound vs memory-bound bottleneck identification)
- Air-gapped HPC offline installation support
- Docker and Singularity/Apptainer containers

### Parallelization & Performance
- NUMA-aware resource allocation with `numactl` + `lscpu` integration
- Granular parallelism (`WIEN_GRANULARITY`) with 3× memory safety factor and OOM warnings
- ELPA eigensolver recommendation with threshold 8000 (see WIEN2k benchmarks: wien2k.at/reg_user/benchmark/)
- Weighted k-point distribution (First Fit Decreasing bin-packing algorithm)
- NUMA-aware k-point distribution with balance ratio scoring
- Hybrid MPI+OpenMP for LAPW0 FFT-dominated workloads
- GPU offloading detection and hybrid CPU+GPU `.machines` generation
- Weak/strong scaling bottleneck identification

### SCF Convergence & Physics
- Smart Kerker mixing q0 based on system type (Winkelmann et al. 2020, PRB 102, 195138):
  - metal: `q0 = 0.4 × 2π/a` | semiconductor: `0.15 × 2π/a` | insulator: `0.05 × 2π/a`
- Restarted Pulay mixing for large systems (>50 atoms) — Pratapa & Suryanarayana, Chem. Phys. Lett. 635, 69–74 (2015)
- Charge sloshing root cause diagnosis (metallic/symmetry/core-overlap/mixing)
- QTL-B error root cause analysis with targeted fixes
- SCF divergence detection (catastrophic/monotonic_drift/charge_sloshing/stalled)
- Automatic checkpointing with incremental file copy (heuristic interval, not Daly formula) — see Daly 2006, FGCS 22(3), 303-312 for optimal derivation
- Adaptive checkpoint intervals: <20 cycles → 5, <50 → 10, else → 15

### ML & AI-Assisted Optimization
- Bayesian hyperparameter optimization with Matérn ν=2.5 kernel (Snoek et al. 2012)
- Expected Improvement (EI) and q-batch EI acquisition functions (Monte Carlo joint posterior sampling — Ginsbourger et al. 2010)
- Latin Hypercube Sampling for uniform search space coverage
- Physics-informed priors (element-aware RKMAX/mixing/kpt constraints)
- GNN-based k-point grid prediction (CGCNN architecture, pure NumPy inference)
- History-driven warm-start from SQLite execution database

### Structure & Input Validation
- `.struct` RMT sphere overlap detection (warning >10%, critical >30%)
- Small RMT warnings for light hard elements (O, F, N)
- setrmt algorithm: automated RMT optimization from nearest-neighbor distances (Blaha JCP 2020)
- Wyckoff position / spacegroup heuristic warnings
- WIEN2k version detection 19.x–24.x with capability mapping

### UI & Output
- Rich-text interactive TUI with Textual framework
- Roofline + Amdahl bottleneck visualization with color-coded warnings
- Formatted RMT optimization reports with per-atom overlap tables

## Supported Calculation Types

| Input Files | Detected Calculation | Command |
|-------------|---------------------|---------|
| `case.struct` | Standard SCF | `run_lapw -p` |
| `case.struct` + `case.inst` | Spin-polarized | `runsp_lapw -p` |
| `case.struct` + `case.inso` | Spin-orbit coupling | `run_lapw -p -so` |
| `case.inst` + `case.inso` | Spin-polarized + SOC | `runsp_lapw -p -so` |
| `case.struct` + `case.inorb` | LDA/GGA+U | `run_lapw -p -orbc` |
| `case.struct` + `HYBR` in `case.in0` | Hybrid functional | `run_lapw -p -hf` |
| `case.struct` + `case.ineece` | Onsite exact exchange | `run_lapw -p -eece` |

## End-to-End Example

```bash
# 1. Initialize a WIEN2k case
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000

# 2. Run SCF to generate .scf file (provides NMAT for optimal parallelization)
run_lapw -p

# 3. Get performance advice before generating config
forge advise --case Fe

# 4. Diagnose any SCF convergence issues
forge diagnose --log Fe.scf

# 5. Auto-generate optimal .machines with all backend intelligence
forge generate --target time

# 6. Submit to SLURM with auto-detected resources
forge submit --partition compute --time 48:00:00

# Or use the interactive wizard with physics options
forge_wizard
```

## Parallel Execution Modes

| Mode | When to Use | Parallelism |
|------|-------------|-------------|
| **kpoint** | Many k-points (>4), small unit cells | k-point parallel, minimal communication |
| **hybrid** | General purpose, modern multi-core nodes | MPI + OpenMP mixed |
| **mpi** | Large matrices with ELPA (nmat > 8000), few k-points | Fine-grain ScaLAPACK/ELPA diagonalization |
| **fine_grain** | Very large systems (nmat > 15000), limited k-points | Atom-level decomposition via `WIEN_GRANULARITY` |

## Mixing Strategies

| Strategy | When Applied | Key Parameter |
|----------|-------------|---------------|
| **Broyden (default)** | Small systems (≤50 atoms) | Standard WIEN2k mixing |
| **Kerker** | Metallic systems | Smart q0 based on system type + lattice constant |
| **Restarted Pulay** | Large systems (>50 atoms) | history_size=7, regularization=1e-10 |
| **Restarted Pulay + Kerker** | Large + metallic | Combined Pulay restart + Kerker preconditioning |

## RKMAX Recommendations

| Element Type | Base RKMAX | Notes |
|-------------|-----------|-------|
| Heavy (Z > 70) | 8.0 | f-elements, actinides |
| Medium-heavy (Z > 50) | 7.5 | Transition metals, lanthanides |
| Medium (Z > 30) | 7.0 | First-row transition metals |
| Light (Z > 20) | 6.5 | p-block elements |
| Light hard (O, F, N) | 7.0 (min) | Requires high cutoff due to small RMT |
| With SOC | 7.0 (min) + 0.5 | Spin-orbit coupling demands high cutoff |
| Optimization | +0.5 | Forces require higher cutoff |
| EFG/Hyperfine | +1.0 | Maximum precision needed |

## Development

```bash
make install
make test
make lint
```

## License

MIT License. See [LICENSE.md](LICENSE.md).

**This project is not affiliated with or endorsed by the WIEN2k development team at TU Wien.**
