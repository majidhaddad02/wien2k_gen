# Troubleshooting Guide

## Diagnostic Tool

Run the built-in diagnostic first whenever you encounter an issue:

```bash
forge diagnostics
forge diagnostics --json > diag.json
```

The diagnostic covers: CPU, memory, NUMA, scheduler, MPI, WIEN2k installation, scratch filesystem, and network interconnect.

---

## Common Issues

### 1. "No WIEN2k installation found"

**Symptom:** `ConfigurationError: WIENROOT not set and cannot auto-detect WIEN2k`

**Solution:**
```bash
export WIENROOT=/path/to/WIEN2k
forge generate
```

Or set permanently in `~/.config/forge/config.json`:
```json
{"wienroot": "/opt/WIEN2k_24.1"}
```

---

### 2. "mpirun not found"

**Symptom:** `subprocess.CalledProcessError: mpirun: command not found`

**Solution:**
```bash
which mpirun              # check if MPI is installed
module load openmpi/4.1   # or Intel MPI, MPICH, MVAPICH
forge generate
```

If using a module system, load the MPI module before running forge.

---

### 3. Hyper-Threading Warning

**Symptom:**
```
Warning: Hyper-Threading active. DFT codes perform best on physical cores only.
Use --hint=nomultithread (Slurm) or OMP_PLACES=cores (OpenMP) to avoid oversubscription.
```

**Solution:**

For SLURM:
```bash
#SBATCH --hint=nomultithread
#SBATCH --threads-per-core=1
```

For non-SLURM:
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

---

### 4. NUMA Warning — Unbalanced Memory Access

**Symptom:**
```
Warning: NUMA system (2 nodes). Use numactl or SLURM --cpu-bind=core for memory binding.
```

**Solution:**

For SLURM:
```bash
#SBATCH --cpu-bind=cores
```

For manual execution:
```bash
numactl --cpunodebind=0 --membind=0 mpirun -np 16 run_lapw -p
```

---

### 5. Memory Near System Limit

**Symptom:**
```
Warning: Estimated memory (48.2 GB) near system limit (64.0 GB). Risk of OOM.
```

**Solution:**
- Reduce `--omp` threads (fewer threads = less memory per rank)
- Switch to pure MPI mode: `--mode mpi`
- Request more memory: `forge submit --mem 128G`

---

### 6. Scratch on Network Filesystem

**Symptom:**
```
Warning: SCRATCH on nfs may cause I/O bottleneck. Use local NVMe if possible.
```

**Solution:**
- Set `SCRATCH=/tmp` or `SCRATCH=/dev/shm` (RAM disk)
- Set `TMPDIR=/local_scratch`
- Use `export SCRATCH=/local_scratch`

For SLURM:
```bash
#SBATCH --gres=scratch:100G    # request local scratch
export SCRATCH=$SLURM_SCRATCH
```

---

### 7. K-Point Saturation

**Symptom:**
```
Warning: Using 32 cores exceeds k-point count (4). Speedup capped at 4×.
Consider reducing cores or increasing k-point mesh.
```

**Solution:**
- Increase k-point mesh: re-run `init_lapw` with higher `-numk`
- Or reduce core count: `forge generate --cores 4`
- Or accept the warning — the run will still work, just won't use extra cores

---

### 8. ELPA Not Found

**Symptom:**
```
Warning: ELPA not found; MPI fine-grain will be slow. Consider hybrid mode.
```

**Solution:**

Install ELPA and recompile WIEN2k:
```bash
# Download and install ELPA
./configure --enable-openmp --with-mpi
make -j install

# Recompile WIEN2k with ELPA
cd $WIENROOT
./siteconfig  # set ELPA_LIBS path
make
```

---

### 9. Duplicate Host Warning

**Symptom:**
```
Warning: Duplicate hostnames detected in node list
```

**Cause:** SLURM or PBS returned a nodelist with duplicate entries (e.g., from hostname aliases).

**Solution:** The tool automatically deduplicates hostnames. If the warning persists, verify:
```bash
echo $SLURM_JOB_NODELIST    # should show unique hosts
hostname                     # should match scheduler's view
```

---

### 10. .machines Validation Failed

**Symptom:**
```
ValidationError: .machines syntax error at line 5
```

**Solution:**
- Check the generated `.machines` file manually:
```bash
cat .machines
```
- Compare with backup:
```bash
ls -la .machines.bak.*
```
- If the tool generated an invalid file, run diagnostics and file a GitHub issue with the diagnostic output.

---

### 11. SGE/GridEngine — No PE_HOSTFILE

**Symptom:** SGE detected but `PE_HOSTFILE` is missing from environment.

**Solution:** Ensure the parallel environment is configured:
```bash
qconf -sp mpi          # verify PE exists
qsub -pe mpi 64 job.sh # request PE with slot count
```

---

### 12. Module/Package Not Found

**Symptom:** `ModuleNotFoundError: No module named 'forge'`

**Solution:**
```bash
pip install forge
# or if installed from source:
pip install -e .
```

On HPC clusters, you may need:
```bash
pip install --user forge
export PATH=$HOME/.local/bin:$PATH
```

---

### 13. Permission Denied Writing .machines

**Symptom:** `PermissionError: [Errno 13] Permission denied: '.machines'`

**Solution:**
```bash
chmod u+w .machines           # if file exists
forge generate --overwrite
```

Check if the case directory is writable:
```bash
ls -la .
```

---

## Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
forge generate
```

This outputs step-by-step information about:
- Scheduler detection attempts
- Hardware topology parsing
- Problem size extraction (per-file parsing with values)
- Mode scoring and selection
- `.machines` generation logic
- Validation results

---

## Getting Help

If you cannot resolve an issue:

1. Run `forge diagnostics --json > diag.json`
2. Include the case-specific information (number of atoms, k-points, NMAT)
3. Paste the output of `forge generate --dry-run`
4. File an issue at: https://github.com/majidhaddad02/forge/issues
