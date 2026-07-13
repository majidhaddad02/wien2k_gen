# WIEN2k Production Examples

Three realistic DFT calculations demonstrating `forge` capabilities
across different electronic-structure regimes.

## Examples

| # | System     | Type         | Key Features                                | Parallel Strategy |
|---|------------|--------------|---------------------------------------------|-------------------|
| 1 | **Si**     | Semiconductor | Indirect band gap, insulators               | k‑point parallel  |
| 2 | **Cu**     | Metal         | Dense k‑mesh, Fermi smearing, MSR1a mixing   | MPI-only          |
| 3 | **Fe**     | Magnetic      | Spin-polarized, LDA+U, SOC, Anderson/DIIS   | Hybrid MPI+OpenMP |

## Quick Start

```bash
# Container (recommended for HPC)
singularity build forge.sif Singularity.def
singularity exec forge.sif bash

# From source
pip install -e .
export WIENROOT=/opt/codes/WIEN2k

# Run an example
cd examples/01_si_semiconductor
forge run Si --ranks 64 --mode kpoint
```

## File Layout

Each example directory contains:

```
01_si_semiconductor/
  X.struct          WIEN2k crystal structure (lattice, atoms, symmetry)
  X.in1             SCF parameters (RKmax, energy params, k‑mesh)
  X.in2             Task specification (IFFT, convergence)
  X.inst (Fe only)  Initial spin moment for magnetic systems
  forge.yaml   Parallel & HPC configuration for forge
```

## Scaling Benchmarks

Expected scaling on typical HPC interconnects:

| System | Nodes | Ranks | Walltime (h) | Efficiency | Interconnect |
|--------|-------|-------|-------------|------------|--------------|
| Si     | 1     | 8     | 1.0         | 100%       | InfiniBand EDR |
| Si     | 4     | 32    | 0.33        | 83%        | InfiniBand EDR |
| Si     | 8     | 64    | 0.25        | 71%        | InfiniBand EDR |
| Cu     | 1     | 16    | 4.5         | 100%       | InfiniBand HDR |
| Cu     | 8     | 128   | 0.63        | 88%        | InfiniBand HDR |
| Cu     | 16    | 256   | 0.50        | 73%        | InfiniBand HDR |
| Fe     | 1     | 8     | 2.8         | 100%       | InfiniBand HDR |
| Fe     | 4     | 48    | 0.92        | 87%        | InfiniBand HDR |
| Fe     | 8     | 96    | 0.67        | 74%        | InfiniBand HDR |

## References

- **Si**: Blaha et al. 2020, WIEN2k User's Guide; Kresse & Furthmuller 1996
- **Cu**: Eyert 1996, Modified tetrahedron method (PRB 54, 4103)
- **Fe**: Pulay 1980, Convergence acceleration (Chem. Phys. Lett. 73, 393)
- **Parallel**: Marek et al. 2014, ELPA library; Hager & Wellein 2010
- **Mixing**: Anderson 1965 (JACM); Pulay 1980; Culler 1993 (DIIS)
