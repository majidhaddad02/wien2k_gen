# Preprocessing Parameter Convergence

FORGE's `converge` and `diagnose` commands scan RKmax / k-mesh and explain charge sloshing **before** you spend a 48-hour parallel allocation on unconverged cutoffs.

Tab-complete `forge converge` and `forge diagnose` from `completions/forge.bash` / `completions/forge.zsh`.

Do not start a production SCF until the cheap parameters are converged. This guide covers the preprocessing knobs that dominate WIEN2k accuracy and stability:

| Parameter | File | What it controls | Typical first guess |
|-----------|------|------------------|---------------------|
| RKmax | `case.in1` | Plane-wave cutoff inside muffin-tin spheres | 7.0 |
| k-mesh | `case.klist` | Brillouin-zone sampling | 1000 k (metals more) |
| GMAX | `case.in2` | Charge-density FFT cutoff | 12–14 |
| RMT | `case.struct` | Sphere radii | setrmt / 0.95 x nn/2 |
| Mixing beta, q0 | `case.inm` / `.inc` | SCF charge mixing | system-dependent |
| Smearing | `case.int` | Metals only | MP 0.02 Ry |
| lmax / lnsmax | `case.in1` | Angular momentum | 10 / 4 |

FORGE automates the two most expensive scans (`forge converge`) and diagnoses mixing (`forge diagnose`). RMT, GMAX, and smearing still need a human check.

---

## 1. Order of Operations

Never scan everything at once. One parameter at a time, cheapest first:

```
1. Geometry / RMT           (seconds, no SCF)
2. Mixing that actually converges a coarse SCF
3. RKmax scan               (fixed coarse k-mesh)
4. k-mesh scan              (fixed converged RKmax)
5. GMAX check               (one or two extra SCFs)
6. Property-specific bump   (forces, EFG, SOC)
7. THEN forge generate + production job
```

If you densify the k-mesh first on RKmax=5, you waste queue time and still get a wrong energy.

Tolerance cheat sheet (total energy unless noted):

| Goal | Energy window | Extra |
|------|---------------|-------|
| Geometry / phase stability | 1 mRy / atom (~13 meV/atom) | — |
| Formation energies | 0.1–0.5 mRy / atom | same RKmax and k-mesh family |
| Forces / relaxation | 0.1 mRy / atom | RKmax +0.5 vs energy-only |
| Phonons | tighter than forces | — |
| EFG / hyperfine | 0.05 mRy / atom | RKmax +1.0 |
| Band gaps (qualitative) | 1 mRy | denser k near edges |
| Metals (DOS, EF) | 0.1 mRy | denser k, smearing test |

`forge converge --tolerance 0.001` is **0.001 Ry** on the CLI (about 13.6 meV). The internal finder treats the first point whose Delta-E falls below the threshold as converged.

---

## 2. RMT — Do This Before Any Scan

Overlapping spheres poison every later number.

```bash
forge_wizard          # step 3.5, enable struct validation
# or inspect Case.struct muffin-tin radii by hand
```

Rules FORGE uses (setrmt, Blaha JCP 2020):

- Nearest-neighbour search on a 3x3x3 supercell.
- Suggested RMT = 0.95 x (nn_distance / 2), clamped to [2.5, 4.0] a.u.
- Overlap > 10%: warning. Overlap > 30%: critical.

Light hard elements (O, F, N) want **small RMT** and therefore **higher RKmax** (minimum 7.0). Shrinking only oxygen without raising RKmax under-converges the O-2p.

Example — perovskite SrTiO3:

| Atom | Bad RMT | Why | Safer RMT |
|------|---------|-----|-----------|
| Sr | 2.50 | too small, wasted APW | 2.50–2.80 |
| Ti | 2.00 | too small | 1.90–2.10 is OK for Ti |
| O  | 2.20 | overlaps Ti | 1.60–1.80 |

After changing RMT, rerun `init_lapw`. Old `case.in1` linearization energies are invalid.

---

## 3. Automated Scans with `forge converge`

```bash
forge converge --case Si
```

Default:

- Mode `both` (k-points **and** RKmax)
- K-grids `2,2,2 4,4,4 6,6,6 8,8,8 10,10,10`
- RKmax `5,6,7,8,9,10`
- Tolerance `0.001` Ry
- Each point: `run_lapw -p -ec 0.0001 -cc 0.0001 -i 40` in a temp copy of the case

K-points only:

```bash
forge converge --case Si --mode kpoints --kpoints "4,4,4 6,6,6 8,8,8 12,12,12"
```

RKmax only:

```bash
forge converge --case Si --mode rkmax --rkmax "6,6.5,7,7.5,8,9"
```

Tighter energy window:

```bash
forge converge --case Fe --mode both --tolerance 0.0001
```

What the command does internally:

1. Copies the case into `kpt_NxNxN_*` or `rkmax_*` temp directories.
2. Rewrites `case.klist` / `case.in1`.
3. Optionally calls `init_lapw -b`.
4. Runs SCF, reads `:ENE` from `case.scf`.
5. Reports the first value whose energy change is below tolerance.
6. Deletes the temp directory.

You still need WIEN2k on PATH (`run_lapw`, ideally `init_lapw`).

---

## 4. RKmax Convergence

RKmax = R_MT x K_max. It is **not** a plane-wave cutoff in the VASP sense. Different atoms in the same cell share one RKmax; the smallest RMT sets the hardest requirement.

### 4.1 Starting values (FORGE table)

| Element type | Base RKmax |
|--------------|------------|
| Heavy (Z > 70) | 8.0 |
| Medium-heavy (Z > 50) | 7.5 |
| Medium (Z > 30) | 7.0 |
| Light (Z > 20) | 6.5 |
| Light hard (O, F, N) | 7.0 minimum |
| With SOC | +0.5 (floor 7.0) |
| Forces | +0.5 |
| EFG / hyperfine | +1.0 |

### 4.2 Worked example — silicon

Keep k-mesh fixed (e.g. 8x8x8) while scanning RKmax.

```bash
cd examples/01_si_semiconductor
forge generate --max-cores 8 --mode kpoint
forge converge --case Si --mode rkmax --rkmax "5,6,7,8,9" --tolerance 0.001
```

Illustrative table (your numbers will differ):

```
  Value     E_total (eV)     dE (meV)    Time (s)   SCF
  5.0       -215.112400      0.000       40         12
  6.0       -215.184100     71.700       55         11
  7.0       -215.191800      7.700       80         10
  8.0       -215.192400      0.600      130         10
  9.0       -215.192510      0.110      210         10
```

Converged at **8.0** for 1 meV; 7.0 is already fine for a 10 meV survey.

### 4.3 Worked example — Cu metal (harder)

Metals need both high RKmax **and** dense k. Scan RKmax on a **medium** mesh (e.g. 16x16x16), not on 4x4x4.

```bash
cd examples/02_cu_metal
forge converge --case Cu --mode rkmax --rkmax "6,7,8,9,10" --tolerance 0.0005
```

Expect slower decay than Si. Publish **the value where dE flattened**, not the last (most expensive) point unless you are doing EFG.

### 4.4 Worked example — Fe with SOC

```bash
cd examples/03_fe_magnetic
# floor 7.0 + 0.5 for SOC
forge converge --case Fe --mode rkmax --rkmax "7,7.5,8,8.5,9" --tolerance 0.001
```

SOC uses complex wavefunctions. Each step is roughly 2x a scalar-relativistic SCF. Do the RKmax scan **with** `-so` if the production job will use `-so`. A scalar scan underestimates the cutoff.

### 4.5 Manual RKmax loop (no FORGE converge)

```bash
for rk in 6 7 8 9; do
  mkdir -p rk_$rk
  cp Si.struct Si.in0 Si.in1 Si.in2 Si.inm Si.klist rk_$rk/ 2>/dev/null || true
  cd rk_$rk
  init_lapw -b -vxc 13 -ecut -6 -rkmax $rk -numk 1000
  forge generate --max-cores 16
  run_lapw -p -ec 0.0001 -cc 0.0001 -i 40
  grep :ENE Si.scf | tail -1
  cd ..
done
```

---

## 5. k-Point Convergence

WIEN2k cares about **irreducible** k-points after symmetry, not the Monkhorst–Pack triple you typed.

### 5.1 How dense?

| System | Starting `init_lapw -numk` | Comment |
|--------|----------------------------|---------|
| Insulator, large cell | 50–200 | Often already dense in IBZ |
| Semiconductor (Si, GaAs) | 500–2000 | Indirect gaps need more than Gamma |
| Simple metal (Cu, Al) | 2000–10000 | Fermi surface |
| Magnetic metal (Fe, Ni) | 2000+ | Moments converge slower than energy |
| 2D / slab | dense in-plane, 1 along c | — |
| Hybrid functional | fewer k, much more CPU | converge k **before** `-hf` |

### 5.2 FORGE command

```bash
forge converge --case Si --mode kpoints \
  --kpoints "4,4,4 6,6,6 8,8,8 10,10,10 12,12,12" \
  --tolerance 0.001
```

Keep RKmax **fixed** at the value from section 4.

Illustrative Si table at RKmax=7:

```
  Grid        irr k     E (eV)        dE (meV)
  4x4x4         8      -215.1801      —
  6x6x6        16      -215.1894      9.3
  8x8x8        29      -215.1912      1.8
  10x10x10     47      -215.1915      0.3
  12x12x12     72      -215.1916      0.1
```

Pick **8x8x8** for survey, **10x10x10** for publication energies.

### 5.3 Copper — tetrahedron + smearing

Energy vs k-mesh in metals oscillates. Always:

1. Use the same smearing for every grid.
2. Look at **DOS at EF** and magnetic moment (if any), not only total energy.
3. Prefer even grids or the mesh family you will use in production.

```bash
forge converge --case Cu --mode kpoints \
  --kpoints "8,8,8 12,12,12 16,16,16 20,20,20" \
  --tolerance 0.0005
```

If charge sloshes on coarse meshes, that is a mixing/smearing problem, not a reason to skip the scan. See section 7.

### 5.4 Hexagonal and tetragonal cells

Do **not** use NxNxN blindly.

- Hexagonal: NxNxM with M scaled by c/a.
- Tetragonal: NxNxM.
- Slab: NxNx1.

`forge converge` passes the triples you give it. It does not guess anisotropy.

### 5.5 Interaction with parallel `.machines`

k-point count **is** the parallel saturation limit in `kpoint` mode.

| irr k | Max useful kpoint ranks |
|-------|-------------------------|
| 8 | 8 |
| 29 | 29 |
| 512 | hundreds, then hybrid |

Converge k **before** `forge generate`. Generating `.machines` on a 2x2x2 test mesh, then switching to 20x20x20 without regenerating, leaves you in mpi/fine_grain mode that you no longer need (or the opposite).

---

## 6. GMAX, lmax, ecut

These are cheaper than RKmax/k-mesh but still real.

### GMAX (`case.in2`)

Charge density Fourier cutoff. Default ~12 is often enough. Raise to 14–16 when:

- Small RMT (O, F, H)
- Forces / phonons
- Hybrid functionals

A 2-point check (12 vs 14) is usually enough. If energy moves more than your tolerance, keep 14.

### lmax / lnsmax (`case.in1`)

Defaults (10 / 4) are standard. Increase lmax to 12 only for f-elements or EFG on heavy nuclei.

### Energy cutoff ecut (`init_lapw -ecut`)

Separates core and valence. Default -6.0 Ry is the usual valence window. Deep semicore (Ga 3d, Zn 3d, valence f) may need `-ecut -8` or local orbitals — that is a physics choice, not a FORGE scan.

---

## 7. Mixing, Smearing, and a Converged SCF

A parameter scan is meaningless if each point diverges.

### 7.1 Diagnose first

```bash
forge diagnose --log Case.scf
forge diagnose Case
```

You get cycle count, final energy, charge distance, charge-ratio trend, and a root cause.

### 7.2 Mixing map (FORGE)

| Situation | Strategy | Key knobs |
|-----------|----------|-----------|
| Small cell, <=50 atoms | Broyden (WIEN2k default) | beta ~ 0.2 |
| Metal | Kerker | q0 = 0.4 x 2pi/a |
| Semiconductor | Kerker weak | q0 = 0.15 x 2pi/a |
| Insulator | Kerker very weak / default | q0 = 0.05 x 2pi/a |
| Large cell, >50 atoms | Restarted Pulay | history 7, reg 1e-10 |
| Large + metal | Pulay + Kerker | both |

Lattice constant `a` comes from `.struct`.

### 7.3 Charge sloshing playbook

| Root cause | How it looks | Fix |
|------------|--------------|-----|
| Metallic | gap < 0.1 eV, oscillating :GAP | Kerker + MP smearing 0.02 Ry + denser k |
| Symmetry breaking | "symmetry broken" in dayfile | reduce beta to 0.05; check magnetic init |
| Core overlap | RMT ratio > 1.5 | fix RMT, then mixing |
| Aggressive mixing | beta > 0.3 | beta 0.05, PRATT 3, try MSR1a |

Catastrophic energy (> 1e5): restart from scratch, beta=0.02, PRATT=10, check RMT/RKmax.

Monotonic upward drift: beta x 0.3, PRATT=5, MSR1a.

### 7.4 Example — Cu metal mixing before k-scan

```bash
cd examples/02_cu_metal
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000
forge generate --max-cores 16
run_lapw -p -i 20
forge diagnose --log Cu.scf
```

If sloshing is metallic: set Kerker in `Cu.inm` / follow diagnose actions, rerun 20 cycles, **then** start `forge converge --mode kpoints`.

### 7.5 Example — Fe moment vs k-mesh

Magnetic moments often need a denser mesh than total energy.

After each k-grid:

```bash
grep :MMT Fe.scf | tail -1
```

Converge **moment to 0.01 muB** as well as energy. FORGE's energy-only finder will not do this for you.

---

## 8. Property-Specific Extra Convergence

| Property | Extra on top of energy-converged RKmax/k |
|----------|------------------------------------------|
| Forces (`-fc`) | RKmax +0.5, GMAX +2, denser k |
| Relaxation | same as forces; regenerate `.machines` if NMAT jumps |
| Phonon displacements | forces-level, identical k/RKmax for all displacements |
| EFG | RKmax +1.0, lmax 12, check GMAX |
| Hyperfine | same as EFG |
| SOC bands | RKmax floor 7.0+0.5, denser k near crossings |
| Hybrid (`-hf`) | converge k **without** HF first; then one HF confirmation |
| LDA+U | U is physics, not a cutoff; still converge k/RKmax at fixed U |

---

## 9. Full Recipes

### Recipe A — Si semiconductor, publication energy

```bash
cd examples/01_si_semiconductor
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 300

# cheap mixing check
forge generate --max-cores 8 --mode kpoint
run_lapw -p -i 15
forge diagnose --log Si.scf

# RKmax at medium k
forge converge --case Si --mode rkmax --rkmax "6,7,8,9" --tolerance 0.001
# suppose 7.0 is enough

# k-mesh at RKmax=7
forge converge --case Si --mode kpoints \
  --kpoints "6,6,6 8,8,8 10,10,10 12,12,12" --tolerance 0.001
# suppose 8x8x8 is enough

# production
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 1000
forge generate --mode kpoint --target time
forge submit --partition compute --time 08:00:00 --job-name si-prod
```

### Recipe B — Cu metal

```bash
cd examples/02_cu_metal
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.0 -numk 2000
forge generate --max-cores 16
run_lapw -p -i 25
forge diagnose Cu
# apply Kerker / MP if requested

forge converge --case Cu --mode rkmax --rkmax "7,8,9,10" --tolerance 0.0005
forge converge --case Cu --mode kpoints \
  --kpoints "12,12,12 16,16,16 20,20,20 24,24,24" --tolerance 0.0005

forge generate --mode kpoint --target time
forge submit --partition compute --time 24:00:00 --job-name cu-prod
```

### Recipe C — Fe magnetic + SOC

```bash
cd examples/03_fe_magnetic
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.5 -numk 2000
forge generate --mode hybrid --omp 4
runsp_lapw -p -i 30
forge diagnose --log Fe.scf

forge converge --case Fe --mode rkmax --rkmax "7,7.5,8,8.5,9" --tolerance 0.001
forge converge --case Fe --mode kpoints \
  --kpoints "8,8,8 12,12,12 16,16,16" --tolerance 0.001

# production with SOC only after scalar cutoffs are known
forge generate --mode hybrid --omp 4 --target time
forge submit --partition compute --time 48:00:00 --job-name fe-sp-so --mem 192G
```

### Recipe D — Large oxide, Pulay mixing, then fine_grain

```bash
# >50 atoms, few k-points
forge diagnose LaFeO3
# expect Restarted Pulay recommendation

forge converge --case LaFeO3 --mode rkmax --rkmax "6,7,8" --tolerance 0.001
# skip huge k-scan; cell is large, 2x2x2 or 4x4x4 may already be dense

forge generate --mode fine_grain --target time
# submit with model 2 so hostnames match the allocation
```

### Recipe E — Forces for relaxation

```bash
# after energy-level RKmax=7, k=8x8x8
init_lapw -b -vxc 13 -ecut -6 -rkmax 7.5 -numk 1000
forge generate
run_lapw -p -fc
```

Compare force on the first displaced atom against RKmax=8.0. If it moves more than 1 mRy/bohr, keep 8.0.

---

## 10. How FORGE Picks "Converged"

`find_converged_parameters`:

- Walks the scan in the order you listed.
- Returns the **first** point with `0 < delta_energy_meV < tolerance`.
- If none pass, returns the **last** (most expensive) point.

Implications:

1. List grids from coarse to fine.
2. The first point has `delta=0` by construction (no previous energy) and is never chosen as converged via the delta test.
3. Oscillatory metal energies can hit the window too early. Inspect the printed table, do not blindly trust the single number.

CLI `--tolerance` is in **Ry**. Internal comparison uses **meV**. If a printed delta looks inconsistent with your mental Ry scale, look at the table in eV/meV.

---

## 11. Bayesian Shortcut (Optional)

```bash
# enable in ~/.config/forge/config.json
# "use_bayesian_optimization": true
```

or the wizard advanced step. The optimizer jointly samples RKmax, mixing beta, k-density, GMAX, lmax with a Matérn 2.5 GP. Use it **after** you can run SCF at all, not as a substitute for a first coarse scan. It is extra walltime.

---

## 12. Checklist Before Production Submit

1. RMT overlaps checked; `init_lapw` rerun after any RMT change.
2. Coarse SCF converges (diagnose is green or yellow, not red).
3. RKmax scan done at **fixed** medium k-mesh; value recorded.
4. k-mesh scan done at **fixed** chosen RKmax; value recorded.
5. Metals: smearing and moment (if magnetic) recorded.
6. Property extras applied (forces +0.5 RKmax, SOC +0.5, EFG +1.0).
7. `forge generate` rerun **after** the final k-mesh (NMAT and nkpt changed).
8. `.machines` hostnames will match the production allocation (see `docs/job-submission.md`).

Skipping step 7 is the most common way to combine a careful convergence study with a terrible parallel layout.

---

## Related Documents

- `docs/machines-guide.md` — generate `.machines` after cutoffs are known
- `docs/job-submission.md` — four ways to submit the production job
- `docs/parallel-modes.md` — why nkpt changes the parallel mode
- `docs/user-guide.md` — mixing formulas and diagnose output
- `docs/troubleshooting.md` — QTL-B, OOM, divergence at runtime
