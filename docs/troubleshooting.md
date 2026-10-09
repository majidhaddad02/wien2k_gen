# Troubleshooting Guide

FORGE's first response to a failed job is `forge diagnostics` plus `forge generate --dry-run`: they show whether the problem is WIENROOT, MPI, hostnames, or the `.machines` file — before you spend another queue allocation.

Tab-complete diagnostic flags from `completions/forge.bash` / `completions/forge.zsh`.

```bash
forge diagnostics
forge diagnostics --full
forge diagnostics --export diag.json
forge --json diagnostics
forge generate --dry-run
```

Covers CPU, memory, NUMA, scheduler, MPI, WIEN2k, scratch, and interconnect.

---

## Common Issues

### 1. "No WIEN2k installation found"

**Symptom:** `ConfigurationError: WIENROOT not set and cannot auto-detect WIEN2k`

```bash
export WIENROOT=/path/to/WIEN2k
forge generate
```

Or in `~/.config/forge/config.json`:

```json
{"wienroot": "/opt/WIEN2k_24.1"}
```

---

### 2. "mpirun not found"

```bash
which mpirun
module load openmpi/4.1
forge generate
```

Load the MPI module **before** FORGE so `WIEN_MPIRUN` in `parallel_options` is correct.

---

### 3. Hyper-Threading Warning

DFT saturates on physical cores. FORGE warns when HT is on.

SLURM:

```bash
#SBATCH --hint=nomultithread
#SBATCH --threads-per-core=1
```

Otherwise:

```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

```bash
forge generate --reserve-os-cores 4
```

---

### 4. NUMA Warning

```bash
#SBATCH --cpu-bind=cores
```

```bash
numactl --cpunodebind=0 --membind=0 mpirun -np 16 run_lapw -p
forge hardware --recommend --case Fe
```

---

### 5. Memory Near System Limit

- Reduce `--omp`
- Switch to `--mode mpi`
- Request more memory: `forge submit --mem 128G`
- Cap with `forge generate --memory-limit 64`

---

### 6. Scratch on Network Filesystem

```bash
export SCRATCH=/tmp
export TMPDIR=/local_scratch
```

---

### 7. K-Point Saturation

More ranks than irreducible k-points.

```bash
forge generate --cores 4 --mode kpoint
forge generate --mode hybrid --omp 4
```

`extrafine: 1` appears only when `nkpt % n_ranks != 0`. It does not create extra parallelism.

---

### 8. ELPA Not Found

NMAT > 8000 without ELPA is slow in mpi/fine_grain. Install ELPA, recompile WIEN2k (`siteconfig`), or:

```bash
forge generate --mode hybrid --omp 4
```

---

### 9. Duplicate Host Warning

FORGE deduplicates. If it persists:

```bash
echo $SLURM_JOB_NODELIST
hostname
```

Generate **inside** the allocation (`docs/job-submission.md` model 2), not on the login node.

---

### 10. `.machines` Validation Failed

```bash
cat .machines
ls -la .machines.bak.*
forge generate --overwrite --dry-run
forge generate --manual
```

Physical cores per node are `max(lapw0, lapw1, lapw2)`, not the sum. Remainder ranks are not an OpenMP error.

---

### 11. SGE — No `PE_HOSTFILE`

```bash
qconf -sp mpi
qsub -pe mpi 64 job.sh
forge generate --scheduler sge
```

---

### 12. `ModuleNotFoundError: No module named 'forge'`

There is no PyPI package. Install from this tree:

```bash
./install.sh --yes
# or, in a venv:
pip install -e .
export PATH="$HOME/.local/bin:$PATH"
forge --version
```

Air-gapped: `./install.sh --offline --yes`. See `docs/installation.md`.

---

### 13. Permission Denied Writing `.machines`

```bash
chmod u+w .machines
forge generate --overwrite
ls -la .
```

---

### 14. `forge submit` and missing `.machines`

```bash
forge submit --auto-generate --partition compute --time 08:00:00
forge submit --no-auto-generate --partition compute
```

`--auto-generate` and `--no-auto-generate` are mutually exclusive.

---

### 15. Wrong clone / install URL

The repository is `https://github.com/majidhaddad02/wien2k_gen.git`, not `.../forge`.

---

### 16. `generate --target energy` rejected

`forge generate --target` is `time|memory|balanced|cost`. Energy is an **advise** goal:

```bash
forge generate --target time
forge advise --case Fe --target energy
```

---

### 17. `advise --verbose` unknown

`advise` has no `--verbose`. Use the global flag:

```bash
forge -v advise --case Fe --cores 128
```

---

### 18. Batch script `exec` skips EXIT_CODE

Do not `exec run_lapw` at the end of a scheduler script. FORGE's generated scripts do not `exec`, so EXIT_CODE and epilog hooks still run.

---

## Debug Mode

```bash
export LOG_LEVEL=DEBUG
forge -vv generate --dry-run
forge --log-file /tmp/forge.log generate --dry-run
```

---

## Getting Help

1. `forge diagnostics --full --export diag.json`
2. Include atoms, k-points, NMAT
3. Paste `forge generate --dry-run`
4. File an issue at https://github.com/majidhaddad02/wien2k_gen/issues

## Related Documents

- `docs/machines-guide.md` — hostname, core-count, extrafine, kpar
- `docs/job-submission.md` — generate-inside-allocation vs login-node files
- `docs/preprocessing-convergence.md` — mixing, RKmax, k-mesh
- `docs/installation.md` — `install.sh`, offline, Docker
- `docs/workflow.md` — full path from `init_lapw` to submit
