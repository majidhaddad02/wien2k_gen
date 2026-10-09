# Real-World Examples

FORGE's generated `.machines` files are the examples: they already encode mode, ranks, OpenMP, and remainder handling. This page points at copy-paste commands; full file walkthroughs are in `docs/machines-guide.md`.

Tab-complete flags from `completions/forge.bash` / `completions/forge.zsh`.

---

## Workstation — Si, many k-points

```bash
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000
run_lapw -p
forge generate --mode kpoint
```

`extrafine` is **not** always written. It appears only when `nkpt % n_ranks != 0`. 1000 k-points on 32 ranks is exact, so there is no `extrafine` line.

Physical cores per node = max of sequential `lapw0` / `1:` / `2:` widths, not the sum.

---

## SLURM — Fe2O3 hybrid

```bash
forge advise --case Fe2O3 --cores 256
forge generate --mode hybrid --omp 4 --cores 256 --target time
forge submit --partition compute --time 48:00:00 --mem 192G --job-name fe2o3
```

`--ntasks` becomes the MPI rank count; `--cpus-per-task` becomes 4.

---

## LDA+U — NiO

```bash
forge generate
forge submit --partition compute --time 24:00:00 --job-name nio-u
```

Exec command is `run_lapw -p -orbc` when `case.inorb` exists.

---

## Spin + SOC — Fe

```bash
cd examples/03_fe_magnetic
forge generate --mode hybrid --omp 4
forge submit --partition compute --time 48:00:00 --job-name fe-sp-so --mem 192G
```

Exec command is `runsp_lapw -p -so`.

---

## Reserve OS cores

```bash
forge generate --reserve-os-cores 4
```

124 of 128 cores used. `1:` and `2:` both show 124 because the stages are sequential.

---

## SGE

```bash
forge generate --scheduler sge
forge submit --scheduler sge --partition all.q --time 24:00:00
```

---

## GPU

```bash
forge generate --gpu --mode mpi --target time
forge_sbatch generate -p gpu --gres gpu:a100:1 -n 16 -c 8 -t 12:00:00
```

---

## Compare modes without writing

```bash
forge generate --mode kpoint --dry-run --export kpoint.json
forge generate --mode hybrid --omp 4 --dry-run --export hybrid.json
```

---

## Related Documents

- `docs/machines-guide.md` — how to read the file
- `docs/job-submission.md` — four submit models
- `docs/user-guide.md` — every flag
- `docs/workflow.md` — end-to-end path
