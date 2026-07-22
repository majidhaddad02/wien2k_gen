# Machine Learning Subsystems

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Feature Engineering](#feature-engineering)
4. [Feature Descriptions](#feature-descriptions)
5. [Model Architectures](#model-architectures)
6. [Training Pipeline](#training-pipeline)
7. [Prediction Flow](#prediction-flow)
8. [Hardware Features](#hardware-features)
9. [Dataset Generation](#dataset-generation)
10. [Model Persistence](#model-persistence)
11. [Limitations & Future Work](#limitations--future-work)

---

## Overview

The forge ML subsystem provides three independent models that predict, optimize, and guide WIEN2k DFT calculations:

| Model | Class | Purpose | Output |
|-------|-------|---------|--------|
| **RF (Random Forest)** | `SCFTimePredictor` | Predict SCF walltime and convergence probability | Hours ± uncertainty, difficulty class |
| **GNN (CGCNN)** | `CGCNNModel` | Predict optimal k-point grid from crystal structure | `(kx, ky, kz)` grid, confidence |
| **GP (Bayesian)** | `BayesianOptimizer` / `BayesianParameterTuner` | Optimize WIEN2k parameters via sequential BO | Best `(rkmax, kppra, mixing)` |

All three models use zero external ML dependencies at inference time:
- RF uses **scikit-learn** (optional; falls back to physics models).
- GNN uses **pure NumPy** (no PyTorch/TensorFlow).
- GP uses **custom NumPy** GP with Cholesky decomposition.

---

## Data Sources

### ExecutionHistory SQLite Schema

The training data for the RF model and GP warm-start comes from `~/.forge/history.db`.

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `run_id` | `TEXT PK` | UUID v4 unique run identifier |
| 2 | `timestamp` | `REAL` | Unix epoch at record creation |
| 3 | `backend` | `TEXT` | DFT engine (`"wien2k"`) |
| 4 | `mode` | `TEXT` | Parallelization (`"kpoint"`, `"hybrid"`, `"mpi"`) |
| 5 | `nmat` | `INTEGER` | LAPW basis size |
| 6 | `nkpt` | `INTEGER` | Number of irreducible k-points |
| 7 | `atoms` | `INTEGER` | Number of atoms in unit cell |
| 8 | `rkmax` | `REAL` | Plane-wave cutoff parameter |
| 9 | `total_cores` | `INTEGER` | Total cores used |
| 10 | `omp_threads` | `INTEGER` | OpenMP threads per MPI rank |
| 11 | `nodes_used` | `INTEGER` | Number of compute nodes |
| 12 | `walltime_sec` | `REAL` | Observed walltime (seconds) |
| 13 | `efficiency_pct` | `REAL` | Parallel efficiency percentage |
| 14 | `convergence_cycles` | `INTEGER` | SCF cycles to convergence |
| 15 | `memory_gb_used` | `REAL` | Peak memory usage (GB) |
| 16 | `node_list` | `TEXT` | JSON array of hostnames |
| 17 | `success` | `INTEGER` | Boolean (0=failed, 1=succeeded) |
| 18 | `tags` | `TEXT` | JSON array of string tags |
| 19 | `nbands` | `INTEGER` | Number of bands |
| 20 | `spacegroup` | `INTEGER` | Space group number (1–230) |
| 21 | `max_z` | `INTEGER` | Maximum atomic number |
| 22 | `avg_z` | `REAL` | Mean atomic number |
| 23 | `volume_bohr3` | `REAL` | Unit cell volume (Bohr³) |
| 24 | `is_soc` | `INTEGER` | Spin-orbit coupling active |
| 25 | `is_hybrid` | `INTEGER` | Hybrid functional active |
| 26 | `cpu_arch` | `TEXT` | CPU microarchitecture |
| 27 | `cpu_generation` | `TEXT` | CPU generation name |
| 28 | `peak_gflops` | `REAL` | DP FLOP/s per core |
| 29 | `mem_bandwidth_gbs` | `REAL` | DRAM bandwidth (GB/s) |
| 30 | `numa_nodes` | `INTEGER` | NUMA node count |
| 31 | `interconnect_type` | `TEXT` | `"infiniband"`, `"omni_path"`, `"ethernet"`, `"tcp"` |
| 32 | `scratch_fs` | `TEXT` | Scratch filesystem type |

**Indexes:** `backend`, `mode`, `nmat`, `nkpt`, `timestamp`, `success`, `walltime_sec`, `total_cores`, `(nmat, nkpt, backend)`, `cpu_arch`, `spacegroup`.

### Record Generation

Records are written via `ExecutionHistory.record()` (`history.py:345`). Each completed WIEN2k SCF run calls `record()` with an `ExecutionRecord` dataclass populated from:
- **Problem parameters**: parsed from `.struct` and `.in1` files (`nmat`, `nkpt`, `atoms`, `rkmax`).
- **Structural features**: volume from lattice vectors, `max_z`/`avg_z` from atom species, spacegroup from symmetry.
- **Hardware context**: detected at runtime in `capture_hardware_context()` (`ml_predict.py:496`), calling into `forge.core.hardware` detectors.
- **Timing**: wall clock from `time.monotonic()` around the `run_lapw` invocation.

---

## Feature Engineering

### RF (SCFTimePredictor) — 19 Features

| Index | Name | Type | Source | Description |
|-------|------|------|--------|-------------|
| 0 | `atoms` | `float` | `StructureFeatures.atoms` | Number of atoms |
| 1 | `nmat` | `float` | `ElectronicFeatures.nmat` | LAPW matrix dimension |
| 2 | `nbands` | `float` | `ElectronicFeatures.nbands` | Number of bands |
| 3 | `rkmax` | `float` | `ElectronicFeatures.rkmax` | Plane-wave cutoff |
| 4 | `nkpt` | `float` | `ElectronicFeatures.nkpt` | Irreducible k-points |
| 5 | `is_soc` | `float` | `ElectronicFeatures.is_soc` | Spin-orbit coupling (0/1) |
| 6 | `is_hybrid` | `float` | `ElectronicFeatures.is_hybrid` | Hybrid functional (0/1) |
| 7 | `spacegroup` | `float` | `StructureFeatures.spacegroup_number` | Space group 1–230 |
| 8 | `max_z` | `float` | `StructureFeatures.max_z` | Max atomic number |
| 9 | `avg_z` | `float` | `StructureFeatures.avg_z` | Mean atomic number |
| 10 | `volume_bohr3` | `float` | `StructureFeatures.volume_bohr3` | Unit cell volume (Bohr³) |
| 11 | `packing_fraction` | `float` | `StructureFeatures.packing_fraction` | Estimated atomic packing |
| 12 | `complexity` | `float` | Computed | `nmat^1.5 * nkpt * atoms * rkmax / 7.0 / 1e6` |
| 13 | `log_nmat_nkpt` | `float` | Computed | `ln(max(nmat * nkpt, 1.0))` |
| 14 | `peak_gflops` | `float` | `HardwareContext.peak_gflops` | DP GFLOPS per core |
| 15 | `mem_bandwidth_gbs` | `float` | `HardwareContext.mem_bandwidth_gbs` | DRAM bandwidth |
| 16 | `numa_nodes` | `float` | `HardwareContext.numa_nodes` | NUMA domain count |
| 17 | `interconnect_gbps` | `float` | Lookup table | Network speed per type |
| 18 | `scratch_is_local` | `float` | Lookup table | `1.0` if local, `0.0` if NFS |

Encoding function: `_encode_features()` at `ml_predict.py:296`.

### GNN (CGCNNModel) — Node & Edge Features

**Node features** (per atom, shape `(N_atoms, 4)`):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Atomic radius | Covalent radius (Å) |
| 1 | Electronegativity | Pauling scale |
| 2 | Group-based feature | Column/valence indicator (1.0–4.0) |
| 3 | Period | Row number in periodic table |

Source: `_ATOMIC_FEATURES` dict at `gnn_kpoint_predictor.py:42` (hardcoded lookup for 83 elements).

**Edge features** (per edge, shape `(N_edges, 2)`):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Distance | Inter-atomic distance in Å |
| 1 | Bond type | `0.0` (EN diff < 0.5), `1.0` (0.5 ≤ EN diff < 1.7), `2.0` (EN diff ≥ 1.7) |

**Graph topology**: Edges are generated by iterating over all atom pairs `(i, j)` with periodic images in `{-1, 0, 1}³`, including any pair within an 8 Å cutoff. Bidirectional edges are added for `i ≠ j`.

### GP (BayesianOptimizer) — 6-D Parameter Encoding

Used by the base `BayesianOptimizer` for parallel execution tuning.

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `total_cores / 256.0` | `[0.004, 1.0]` | Normalized core count |
| 1 | `omp_threads / 64.0` | `[0.016, 1.0]` | Normalized OpenMP threads |
| 2 | `mode=kpoint` | `{0, 1}` | One-hot: k-point parallel |
| 3 | `mode=hybrid` | `{0, 1}` | One-hot: hybrid MPI+OpenMP |
| 4 | `mode=mpi` | `{0, 1}` | One-hot: pure MPI |

Encoding function: `_encode_config()` at `sampling.py:51`. Total: **5 dimensions** (2 continuous + 3 one-hot categorical features).

### GP (BayesianParameterTuner) — 10-D Parameter Encoding

Used by the `BayesianParameterTuner` for tuning RKMAX, k-point density, and mixing.

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `total_cores / 256.0` | `[0.004, 1.0]` | Normalized core count |
| 1 | `omp_threads / 64.0` | `[0.016, 1.0]` | Normalized OpenMP threads |
| 2–4 | Mode one-hot | `{0, 1}` | `kpoint`, `hybrid`, `mpi` |
| 5 | `rkmax / 10.0` | `[0.5, 1.0]` | Clipped to `[5.0, 10.0]` |
| 6 | `kppra / 20000.0` | `[0.025, 1.0]` | Clipped to `[500, 20000]` |
| 7 | `mixing / 0.50` | `[0.1, 1.0]` | Clipped to `[0.05, 0.50]` |
| 8 | Interaction: `(rkmax/10)*(mixing/0.50)` | `[0.05, 1.0]` | `rkmax * mixing` cross-term |
| 9 | Interaction: `log(kppra)/log(20000)` | `[0.63, 1.0]` | Log-scaled kpt density |

Encoding function: `_encode_tuning_config()` at `bayesian_tuner.py:56`. Total: **10 dimensions** (5 base + 5 tuning).

---

## Feature Descriptions

### Why Each RF Feature Matters

| Feature | Reasoning for DFT Convergence Prediction |
|---------|------------------------------------------|
| `atoms` | Larger systems → more bands, more k-points, more CPU work |
| `nmat` | **Dominant term** — LAPW matrix dimension; diagonalization scales O(nmat³) |
| `nbands` | More bands → larger eigenvalue problem; correlates with metallicity |
| `rkmax` | Higher cutoff → denser plane-wave grid → larger nmat |
| `nkpt` | More k-points → linear scaling in computational work |
| `is_soc` | Spin-orbit coupling → doubles basis, ~4× walltime penalty |
| `is_hybrid` | Hybrid functional → Fock exchange, ~6× walltime penalty |
| `spacegroup` | High symmetry → fewer irreducible k-points → faster; P1 (`spacegroup=1`) → worst case |
| `max_z` | Heavy elements → harder pseudopotentials / deeper core states |
| `avg_z` | Average nuclear charge → rough indicator of electron count and basis size |
| `volume_bohr3` | Larger volume → lower k-point density needed (MP rule: k ∝ 1/a) |
| `packing_fraction` | Dense packing → metallic behavior; loose → molecular/insulating |
| `complexity` | Combined proxy: `nmat^1.5 * nkpt * atoms * rkmax / 7.0 / 1e6` |
| `log_nmat_nkpt` | Log-transform of problem size product; captures diminishing returns |
| `peak_gflops` | FLOP/s per core → speed of diagonalization/FFT (throughput) |
| `mem_bandwidth_gbs` | Memory bandwidth → ZGEEV / FFT3 are memory-bound |
| `numa_nodes` | NUMA domains → parallel scaling efficiency and memory locality |
| `interconnect_gbps` | Multi-node communication bandwidth (MPI_Allreduce cost) |
| `scratch_is_local` | Local SSD/NVMe vs NFS → I/O speed for scratch files (`.vec`, `.energy`) |

### Why Each GNN Feature Matters

**Node features** encode chemical identity that governs electronic structure:

| Feature | DFT Relevance |
|---------|---------------|
| Covalent radius | Bond lengths → Brillouin zone size (k ∝ 1/a) |
| Electronegativity | Band gap indicator; metallic vs insulating → k-point density needed |
| Group feature | Valence electron count → metallicity, band dispersion |
| Period | Core electron complexity; relativistic effects |

**Edge features** encode local geometry:

| Feature | DFT Relevance |
|---------|---------------|
| Distance | Short bonds → fast-varying wavefunctions → denser k-grid needed |
| Bond type (EN diff) | Ionic (large EN diff) → flat bands → sparse k-grid; covalent → dispersed bands → dense k-grid |

---

## Model Architectures

### RF: Random Forest Regressor

| Property | Value |
|----------|-------|
| Library | scikit-learn `RandomForestRegressor` (optional; fallback to physics) |
| `n_estimators` | 100 or 200 (GridSearchCV-tuned) |
| `max_depth` | 6, 8, 12, or `None` (GridSearchCV-tuned) |
| `min_samples_leaf` | 1, 3, or 5 (GridSearchCV-tuned) |
| `random_state` | 42 (reproducible) |
| `n_jobs` | `-1` (all cores) |
| Input dim | 19 |
| Output | Scalar SCF walltime (hours) |
| Ensemble prediction | Weighted blend: `0.3 * physics + 0.7 * ML` |
| Uncertainty | `abs(physics - ML) * 0.4` |
| Heuristics | Convergence prob, mixing recommendation, difficulty classification |

### GNN: CGCNN (Crystal Graph CNN)

| Property | Value |
|----------|-------|
| Framework | Pure NumPy (zero external deps) |
| Node embedding dim | 4 (atomic features) → 64 (hidden) |
| Graph conv layers | 4 layers |
| Conv equation | `h_i' = ReLU(W_s·h_i + mean_{j∈N(i)}(W_n·h_j · EdgeMLP(e_ij)) + b)` |
| Residual connections | Yes (skip from conv1 → conv2, conv2 → conv3, conv3 → conv4) |
| Global pooling | `concat(mean_pool, max_pool)` → shape `(128,)` |
| MLP head | Hidden 128 → ReLU → output 3 |
| Output | `(kx/12, ky/12, kz/12)` normalized k-point grid |
| Confidence | `1 / (1 + 3 * std(prediction))` |
| Optimizer | Adam (pure numpy): `lr=0.001`, `β₁=0.9`, `β₂=0.999` |
| Loss | MSE on normalized k-point grid |
| Gradient computation | Analytical backpropagation (chain rule through MLP + conv layers) |
| Parameters | `conv1: (4×64)` + `3 × conv_i: (64×64)` + `fc1: (128×128)` + `fc2: (128×3)` |

### GP: Gaussian Process

| Property | Value |
|----------|-------|
| Kernel | RBF-ARD (per-dim length scales) or Matern ν=2.5 |
| Kernel function | `k(x₁,x₂) = exp(-0.5 Σ_d (x₁_d - x₂_d)² / l_d²)` |
| Inference | Cholesky decomposition (`numpy.linalg.cholesky`) |
| Acquisition | Expected Improvement (EI) with `ξ=0.01` |
| Length scale optimization | Gradient ascent on marginal likelihood (50 steps, `lr=0.05`) |
| ARD bounds | `l_d ∈ [0.1, 10.0]` |
| Nugget | `1e-6` (numerical stability) |
| Initialization | Latin Hypercube Sampling + warm-start from `ExecutionHistory` |
| Transfer learning | Chemical similarity weighting via `_chemical_similarity()` |
| Multi-fidelity | 3 fidelity levels with cost/correlation factors |

---

## Training Pipeline

### RF Training

**Trigger**: `SCFTimePredictor.train_from_history()` (`ml_predict.py:119`)

1. Load records from `ExecutionHistory`: `SELECT * FROM execution_history WHERE success = 1 AND walltime_sec > 0 AND nmat > 0 LIMIT 1000`
2. Extract 19 features per record via `_encode_features_from_record()` (`ml_predict.py:332`)
3. Target: `walltime_sec / 3600.0` (hours)
4. Require ≥ 20 training records; otherwise fall back to physics model
5. Z-score normalize: `(X - μ) / σ` with `σ` clamped to ≥ `1e-8`
6. `GridSearchCV` with `cv=min(5, max(2, n//5))`:
   - `n_estimators ∈ {100, 200}`
   - `max_depth ∈ {6, 8, 12, None}`
   - `min_samples_leaf ∈ {1, 3, 5}`
7. Score: `neg_mean_absolute_error`
8. After CV, report cross-validated MAE and top-5 feature importances
9. Auto-cache to `~/.forge/models/scf_time_rf.pkl`

### GNN Training

**Trigger**: `get_trained_model()` → `CGCNNModel.train()` (`gnn_kpoint_predictor.py:347`)

1. Generate synthetic dataset: `generate_synthetic_dataset(200)` → 200 structures with known k-point grids
2. For each structure: build crystal graph (`build_crystal_graph`), encode target as `(kx/12, ky/12, kz/12)`
3. Train for `epochs=80` with `lr=0.001`
4. Each epoch: shuffle graphs, train one-by-one with Adam
5. Loss: MSE over normalized k-point grid via analytical backpropagation
6. Converges to loss < 0.01 in ~30-45 epochs on synthetic data
7. Auto-save to `~/.forge/models/gnn_kpoint_v1.npz`

### GP Training

**Trigger**: `BayesianOptimizer.__init__()` → `_warm_start()` (`core.py:399`)

1. Query `ExecutionHistory` for records matching `backend` with `success=True`, limit 200
2. Encode each record as 5-D vector via `_encode_config()`
3. Fit GP via Cholesky decomposition of RBF-ARD kernel
4. If `use_ard=True`: optimize length scales for 50 gradient steps on marginal likelihood
5. Online updates: each `update(record)` call appends to `(X, y)` and re-fits

---

## Prediction Flow

### RF: SCF Time Prediction

```
User: forge predict --struct Fe.struct
  │
  ├─► _extract_structure_features(Fe.struct)
  │     ├─ Parse lattice params → volume_bohr3
  │     ├─ Parse atom Z values → atoms, max_z, avg_z, ntype, packing_fraction
  │     └─ Return StructureFeatures
  │
  ├─► CaseFileParser → ElectronicFeatures
  │     ├─ .in1 → nmat, rkmax, nbands
  │     ├─ .klist → nkpt
  │     └─ .inop → is_soc, is_hybrid
  │
  ├─► capture_hardware_context()
  │     ├─ CPU arch, peak GFLOPS, memory BW, NUMA nodes
  │     ├─ Interconnect type → speed lookup (IB=100, Eth=25 Gbps)
  │     └─ Scratch FS → local/NFS flag
  │
  ├─► SCFTimePredictor.predict(struct, elec, hw)
  │     ├─ _encode_features() → 19-D vector
  │     ├─ If trained: X_scaled = (X - μ) / σ
  │     │     ml_time = model.predict(X_scaled)
  │     │     predicted = 0.3 * physics + 0.7 * ml
  │     │     uncertainty = abs(physics - ml) * 0.4
  │     ├─ Else: physics_estimate() only
  │     ├─ convergence_probability, mixing recommendation, difficulty
  │     └─ Return ConvergencePrediction
```

### GNN: K-point Grid Prediction

```
User: forge predict-kpoints --struct Fe.struct
  │
  ├─► Parse structure dict (atoms, lattice)
  │
  ├─► build_crystal_graph(positions, atomic_numbers, lattice_vectors)
  │     ├─ Node features: atom → _ATOMIC_FEATURES[Z] (4 values)
  │     ├─ Edge generation: pairs within 8 Å cutoff (periodic images)
  │     ├─ Edge features: (distance, bond_type via EN difference)
  │     └─ Return (node_feat, edge_index, edge_feat)
  │
  ├─► _get_or_create_model()
  │     ├─ Try ~/.forge/models/gnn_kpoint_v*.npz
  │     └─ Fallback: train from synthetic data (30-60 sec)
  │
  ├─► model.forward(node_feat, edge_index, edge_feat)
  │     ├─ conv1 → ReLU
  │     ├─ conv2 + residual → ReLU
  │     ├─ conv3 + residual → ReLU
  │     ├─ conv4 + residual → ReLU
  │     ├─ Global mean+max pool → concat → (1, 128)
  │     ├─ fc1 (128→128) → ReLU
  │     ├─ fc2 (128→3)
  │     └─ Output: (pred_kx/12, pred_ky/12, pred_kz/12)
  │
  ├─► Denormalize: kx = round(abs(pred[0]) * 12), ...
  │     confidence = 1/(1 + 3*std(pred))
  │
  ├─► If confidence < 0.60 → fallback to MP grid
  │     Return {grid, confidence, method, kpoint_density}
```

### GP: Bayesian Parameter Optimization

```
User: forge optimize --case Fe --target energy_convergence --budget 10
  │
  ├─► BayesianParameterTuner(case_name="Fe", budget=10)
  │     ├─ Init GP (ARD if use_ard)
  │     └─ No warm-start (fresh optimization)
  │
  ├─► For iteration 1..budget:
  │     ├─ _get_next_suggestion()
  │     │     ├─ If n_observations == 0: random sample
  │     │     └─ Else: GP.predict() on search grid
  │     │           ├─ compute_expected_improvement(mu, sigma, best_y)
  │     │           └─ argmax over 1000 LHS candidates
  │     │
  │     ├─ _objective_from_run(rkmax, kppra, mixing)
  │     │     ├─ Generate .in1 with specified rkmax
  │     │     ├─ Generate .inm with specified mixing
  │     │     ├─ Run run_lapw -p -ec 0.0001 -cc 0.0001 -NI
  │     │     ├─ Parse delta-energy from stdout
  │     │     └─ Return delta (lower is better)
  │     │
  │     ├─ Append (X_enriched, y) to observations
  │     ├─ GP.fit(X, y) ← Cholesky on RBF-ARD
  │     └─ Update best_y, best_x
  │
  └─► Return TunerResult with best (rkmax, kppra, mixing, delta_energy)
```

---

## Hardware Features

### HardwareProfile → ML Feature Mapping

The `HardwareProfile` TypedDict (`hardware/types.py:38`) provides system characteristics. Only a subset is used for ML:

| HardwareProfile Key | ML Feature | Model | Purpose |
|---------------------|------------|-------|---------|
| `peak_fp64_gflops` | `peak_gflops` | RF | CPU compute throughput for diagonalization cost |
| `memory_bandwidth_gb_s` | `mem_bandwidth_gbs` | RF | DRAM bandwidth for memory-bound operations (ZGEEV, FFT) |
| `numa_nodes` (count) | `numa_nodes` | RF, GP | NUMA topology affects memory locality and parallel scaling |
| `interconnect.type` | `interconnect_gbps` (lookup) | RF | IB=100, OP=100, Eth=25, TCP=10 Gbps |
| `interconnect.speed_gbps` | Not used directly | — | Measured link speed available but not in feature set |
| `scratch_fs` | `scratch_is_local` | RF | `1.0` if local (NVMe/SSD), `0.0` if NFS |
| `cpu_arch` | `cpu_arch` (string) | Stored in DB | Not used as numeric feature; for grouping/similarity |
| `cpu_microarch` | `cpu_generation` (string) | Stored in DB | Generation tag for historical comparisons |
| `physical_cores` | (implicit via `numa_nodes`) | RF, GP | Not a direct feature but affects default core estimates |

### Interconnect Speed Lookup

```python
# ml_predict.py:366 — _interconnect_speed_lookup()
{
    "infiniband": 100.0,   # Gbps (HDR/NDR)
    "omni_path":  100.0,   # Gbps
    "ethernet":    25.0,   # Gbps (100GbE)
    "tcp":         10.0,   # Gbps (fallback)
}
```

### Hardware Context Capture Flow

```
capture_hardware_context()                            # ml_predict.py:496
  ├─ get_cpu_architecture()         → cpu_arch
  ├─ get_cpu_generation()           → cpu_generation
  ├─ calculate_peak_fp64_gflops()   → peak_gflops  (ISA: AVX2/AVX-512/SSE)
  ├─ get_memory_bandwidth_gb_s()    → mem_bandwidth_gbs
  ├─ get_numa_node_count()          → numa_nodes
  ├─ get_interconnect_info()        → interconnect_type
  └─ get_scratch_filesystem_type()  → scratch_fs
```

---

## Dataset Generation

### Synthetic Dataset (GNN)

Generated by `generate_synthetic_dataset()` (`gnn_kpoint_predictor.py:524`).

**Specifications:**

| Property | Value |
|----------|-------|
| Samples | 200 |
| Templates | 15 structural archetypes |
| Seed | 42 (reproducible) |
| Lattice variation | `base * (0.9 + 0.2*U[0,1]) * U[0.85, 1.15]` |
| Coordinates | Fractional, `U[0,1)` random |
| MP k-point target | `k0/a`, `k0/b`, `k0/c` clamped to `[1, 12]` |

**Template Structures:**

| # | Label | Composition (Z:count) | Base Lattice (Å) |
|---|-------|-----------------------|-------------------|
| 1 | Fe₂O₃-type | Fe(26):2, O(8):3 | 5.0 × 5.0 × 14.0 |
| 2 | SiO₂-type | Si(14):1, O(8):2 | 5.4 × 5.4 × 5.4 |
| 3 | TiO₂-type | Ti(22):1, O(8):2 | 4.6 × 4.6 × 3.0 |
| 4 | Al₂O₃-type | Al(13):2, O(8):3 | 4.8 × 4.8 × 13.0 |
| 5 | BaTiO₃ | Ba(56):1, Ti(22):1, O(8):3 | 4.0 × 4.0 × 4.0 |
| 6 | PbZrO₃ | Pb(82):1, Zr(40):1, O(8):3 | 4.1 × 4.1 × 4.1 |
| 7 | CuO | Cu(29):1, O(8):1 | 4.6 × 3.4 × 5.1 |
| 8 | NiO | Ni(28):1, O(8):1 | 4.2 × 4.2 × 4.2 |
| 9 | Organic | C(6):4, H(1):4 | 6.0 × 8.0 × 10.0 |
| 10 | Au-fcc | Au(79):4 | 4.1 × 4.1 × 4.1 |
| 11 | Ag-fcc | Ag(47):4 | 4.1 × 4.1 × 4.1 |
| 12 | WO₃ | W(74):2, O(8):6 | 7.3 × 7.5 × 3.8 |
| 13 | Mn₂O₃ | Mn(25):2, O(8):3 | 5.0 × 8.5 × 5.0 |
| 14 | Co₃O₄ | Co(27):3, O(8):4 | 8.1 × 8.1 × 8.1 |
| 15 | UO₂ | U(92):2, O(8):4 | 5.4 × 5.5 × 5.5 |

**MP Rule for K-point Target:**

```python
k0 = 40 if has_metal else 30  # Metal detection: Z in {3,4,11,12,13,26,27,28,29,30}
kx = max(1, min(12, round(k0 / a)))  # k ∝ 1/lattice_constant
ky = max(1, min(12, round(k0 / b)))
kz = max(1, min(12, round(k0 / c)))
```

This matches the WIEN2k convention: `numk = k0 / lattice_constant`, where `k0 ≈ 30–40` for insulating/metallic systems respectively.

### Organic Accumulation (RF)

RF training data accumulates organically through normal usage:
1. Each `forge run` completion writes an `ExecutionRecord` to `history.db`
2. `SCFTimePredictor.train_from_history()` queries all successful records
3. Triggered explicitly by user: `forge train-ml`
4. Also auto-triggered if `predict_convergence()` finds no cached model and data ≥ 20 records
5. No synthetic data needed; validation comes from CV splits on real data

---

## Model Persistence

### File Paths

| Model | Default Path | Format | Variable |
|-------|-------------|--------|----------|
| RF + scaler | `~/.forge/models/scf_time_rf.pkl` | Python pickle | `_MODEL_CACHE_PATH` |
| GNN weights | `~/.forge/models/gnn_kpoint_v1.npz` | NumPy `.npz` | `_DEFAULT_MODEL_PATH` |
| GP history | `.bo_history.json` (cwd) | JSON | `history_file` param |
| Execution history | `~/.forge/history.db` | SQLite3 | `DEFAULT_HISTORY_DB` |

### RF Save/Load

```python
# Save: ml_predict.py:173
pickle.dump({
    "model": self._model,          # sklearn RandomForestRegressor
    "feature_names": self._feature_names,
    "scaler_mean": self._scaler_mean,
    "scaler_std": self._scaler_std,
}, f)

# Load: ml_predict.py:190
with open(_MODEL_CACHE_PATH, "rb") as f:
    data = pickle.load(f)
    self._model = data["model"]
    self._feature_names = data.get("feature_names", self._feature_names)
    self._scaler_mean = data.get("scaler_mean")
    self._scaler_std = data.get("scaler_std")
```

### GNN Save/Load

```python
# Save: gnn_kpoint_predictor.py:414
weights = {}
weights["conv1_W_self"] = self.conv1.W_self
weights["conv1_W_neigh"] = self.conv1.W_neigh
weights["conv1_W_edge"] = self.conv1.W_edge
weights["conv1_bias"] = self.conv1.bias
for i, conv in enumerate(self.convs):
    weights[f"convs_{i}_W_self"] = conv.W_self
    # ... (W_neigh, W_edge, bias)
weights["fc1_W"] = self.fc1_W
weights["fc1_b"] = self.fc1_b
weights["fc2_W"] = self.fc2_W
weights["fc2_b"] = self.fc2_b
np.savez(path, **weights)

# Load: gnn_kpoint_predictor.py:433
data = np.load(path)
# Auto-detect n_conv_layers by searching for "convs_{i}_W_self" keys
model = CGCNNModel(node_dim, hidden_dim, n_conv_layers, output_dim)
model.conv1.W_self = data["conv1_W_self"]
# ... restore all layer weights
```

### Versioning

- GNN: `gnn_kpoint_v1.npz` — versioned by filename. `_get_or_create_model()` loads the highest version: `sorted(model_dir.glob("gnn_kpoint_v*.npz"))[-1]`
- RF: Not versioned (overwrites `scf_time_rf.pkl` on retrain)
- GP: `.bo_history.json` is keyed by `case_name` via `HistoryFile` wrapper

### Retraining

- RF: Set env `FORGE_RF_RETRAIN=1` or call `train_from_history()` explicitly
- GNN: Set env `FORGE_GNN_RETRAIN=1` or pass `force_retrain=True` to `get_trained_model()`
- GP: Fresh `BayesianOptimizer` instance starts from scratch; `_warm_start()` seeds from history

---

## Limitations & Future Work

### Current Limitations

| Area | Limitation |
|------|------------|
| **RF data quality** | Cold-start: requires ≥ 20 records; physics fallback is crude (single-cycle heuristic, no MPI overhead model) |
| **RF features** | Missing: MPI rank-to-thread ratio, actual interconnect measured BW, GPU acceleration flag, iterative diagonalization method |
| **GNN dataset** | Synthetic only; 200 structures is small; no spin-polarized or non-collinear magnetic templates |
| **GNN architecture** | Single graph per structure (no line-graph or edge-update GNN); no magnetic moment or spin features |
| **GNN gradients** | Finite-difference gradient is slow and imprecise; limits training to small models |
| **GP acquisition** | EI only; no UCB, Thompson sampling, or entropy search; single-thread optimization |
| **GP parallel** | q-EI partially implemented but not integrated into main loop |
| **Categorical handling** | One-hot encoding with RBF kernel poorly handles categorical distances |
| **Transfer learning** | Chemical similarity weighting is basic (element overlap count); no learned embeddings |
| **Hardware drift** | No detection of hardware changes between predictions; stale hardware snapshots in DB |
| **Uncertainty calibration** | RF uncertainty is a heuristic (`abs(physics - ML) * 0.4`); GP uncertainty not calibrated either |
| **Model staleness** | No drift detection; no online retraining trigger |

### Future Work

| Enhancement | Description |
|-------------|-------------|
| **Online retraining** | Automatically retrain RF when new records exceed 20% of training set size |
| **Real GNN data** | Collect k-point convergence tests from real WIEN2k runs to replace synthetic data |
| **GNN gradient** | Implement analytic backpropagation for CGCNN (eliminate finite-difference training) |
| **Multi-task GNN** | Predict k-points + RKMAX + mixing simultaneously from structure |
| **GPU support** | CuPy-based GNN training; RF `n_jobs` already supports multi-core |
| **Bayesian RF** | Replace point prediction with full posterior (quantile regression forests) |
| **GP kernel** | Add periodic kernel for mode cycling, core count harmonics |
| **Multi-fidelity BO** | Integrate 3-fidelity levels (min, medium, tight convergence) into tuner |
| **Hardware embeddings** | Learn latent hardware representation rather than hand-crafted features |
| **Confidence calibration** | Platt scaling or isotonic regression for RF/GNN confidence scores |
| **Feature monitoring** | Detect feature drift between training and prediction distributions |
| **A/B testing** | Track which model (ML vs physics) gives more accurate predictions in production |
