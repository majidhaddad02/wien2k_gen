# End-to-End WIEN2k Workflow with FORGE

This is the recommended path from a CIF/struct file to a finished SCF (or property) job. Details live in the specialized guides; this page is the map.

```
struct / init_lapw
    -> RMT check
    -> cheap SCF (mixing)
    -> RKmax scan
    -> k-mesh scan
    -> forge advise
    -> forge generate          (.machines)
    -> forge submit | sbatch   (4 models)
    -> forge diagnose          (if SCF misbehaves)
```

---

## 0. Install and environment

```bash
pip install forge
export WIENROOT=/path/to/WIEN2k
forge diagnostics
```

Air-gapped HPC: see `docs/installation.md`. Containers: `make docker` / Singularity.def.

---

## 1. Create the case

```bash
mkdir Si && cd Si
# obtain Si.struct
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 300
```

Flags you will replace later after convergence.

---

## 2. Validate geometry

Enable struct validation in `forge_wizard` or inspect RMT by hand. Overlaps > 10% must be fixed before any scan. See `docs/preprocessing-convergence.md` section 2.

---

## 3. Converge preprocessing parameters

```bash
forge generate --max-cores 8 --mode kpoint
run_lapw -p -i 15
forge diagnose --log Si.scf

forge converge --case Si --mode rkmax --rkmax "6,7,8,9"
forge converge --case Si --mode kpoints --kpoints "6,6,6 8,8,8 10,10,10"
```

Full recipes for Si, Cu, Fe: `docs/preprocessing-convergence.md` section 9.

---

## 4. Ask for a parallel plan

```bash
forge advise --case Si --cores 64
```

You get Roofline (compute vs memory bound), Amdahl saturation, and a suggested mode. Then:

```bash
forge generate --mode kpoint --target time
# or --mode hybrid --omp 4
# or --mode mpi
# or --mode fine_grain
```

How to read the file: `docs/machines-guide.md`.
Why not write it by hand: same file, section 2.

---

## 5. Submit

Pick one of four models (`docs/job-submission.md`):

| Situation | Model |
|-----------|--------|
| Normal SLURM/PBS/LSF | `forge submit --partition compute --time 24:00:00` |
| Multi-node, hostnames must be live | generate **inside** the batch script |
| Need `--gres`, `--qos`, validate/watch | `forge_sbatch` |
| Laptop / salloc / teaching | `forge_wizard` then `run_lapw -p` |

---

## 6. If the job fails or SCF oscillates

```bash
forge diagnose --log Si.scf
forge diagnostics
```

- Parallel/MPI/hostname: `docs/troubleshooting.md`, `docs/machines-guide.md` section 7.
- Charge sloshing / QTL-B: `docs/preprocessing-convergence.md` section 7.

Regenerate `.machines` after you change k-mesh, SOC, node count, or NMAT.

---

## 7. Examples in this repository

| Directory | System | Typical mode |
|-----------|--------|--------------|
| `examples/01_si_semiconductor` | Si | kpoint |
| `examples/02_cu_metal` | Cu | kpoint / mpi |
| `examples/03_fe_magnetic` | Fe | hybrid |

---

## Related Documents

| File | Role |
|------|------|
| `docs/machines-guide.md` | `.machines` tutorial |
| `docs/job-submission.md` | four submit models |
| `docs/preprocessing-convergence.md` | RKmax, k-mesh, mixing |
| `docs/parallel-modes.md` | mode theory |
| `docs/user-guide.md` | CLI reference |
| `docs/examples.md` | dumped `.machines` scenarios |
| `docs/troubleshooting.md` | errors |
| `docs/installation.md` | install |
| `CITATION.cff` | paper / software references (do not edit casually) |
