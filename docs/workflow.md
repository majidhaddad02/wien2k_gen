# End-to-End WIEN2k Workflow with FORGE

FORGE sits between `init_lapw` and the queue: it converges cheap parameters, chooses a parallel mode, writes `.machines`, and submits with ranks that match the file. This page is the map; details live in the specialized guides.

Tab-complete flags from `completions/forge.bash` / `completions/forge.zsh`.

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
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
./install.sh --yes
export WIENROOT=/path/to/WIEN2k
forge diagnostics
```

Air-gapped: `docs/installation.md` (`./install.sh --offline`). Containers: `make docker`.

---

## 1. Create the case

```bash
mkdir Si && cd Si
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 300
```

Replace RKmax and k-mesh after convergence.

---

## 2. Validate geometry

Enable struct validation in `forge_wizard` or inspect RMT by hand. Overlaps > 10% must be fixed before any scan. See `docs/preprocessing-convergence.md`.

---

## 3. Converge preprocessing parameters

```bash
forge generate --max-cores 8 --mode kpoint
run_lapw -p -i 15
forge diagnose --log Si.scf

forge converge --case Si --mode rkmax --rkmax "6,7,8,9"
forge converge --case Si --mode kpoints --kpoints "6,6,6 8,8,8 10,10,10"
```

Recipes: `docs/preprocessing-convergence.md`.

---

## 4. Ask for a parallel plan

```bash
forge advise --case Si --cores 64
```

Roofline (compute vs memory), Amdahl saturation, suggested mode. Then:

```bash
forge generate --mode kpoint --target time
forge generate --mode hybrid --omp 4
forge generate --mode mpi
forge generate --mode fine_grain
```

`--target` for generate is `time|memory|balanced|cost`. How to read the file: `docs/machines-guide.md`.

---

## 5. Submit

| Situation | Model |
|-----------|--------|
| Normal SLURM/PBS/LSF | `forge submit --partition compute --time 24:00:00` |
| Multi-node, live hostnames | generate **inside** the batch script |
| `--gres`, `--qos`, validate/watch | `forge_sbatch` |
| Laptop / salloc / teaching | `forge_wizard` then `run_lapw -p` |

See `docs/job-submission.md`.

---

## 6. If the job fails or SCF oscillates

```bash
forge diagnose --log Si.scf
forge diagnostics
forge diagnostics --full --export diag.json
```

- Parallel/MPI/hostname: `docs/troubleshooting.md`, `docs/machines-guide.md`.
- Charge sloshing / QTL-B: `docs/preprocessing-convergence.md`.

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
| `docs/installation.md` | `install.sh`, Docker, completions |
| `docs/machines-guide.md` | `.machines` tutorial |
| `docs/job-submission.md` | four submit models |
| `docs/preprocessing-convergence.md` | RKmax, k-mesh, mixing |
| `docs/parallel-modes.md` | mode theory |
| `docs/user-guide.md` | CLI flag reference |
| `docs/troubleshooting.md` | errors |
| `docs/api-reference.md` | Python API |
