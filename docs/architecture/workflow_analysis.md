# Computational Workflow Analysis

Investigation target from the prompt was:

User request -> configuration -> structure/case prep -> orchestration -> backend selection -> external execution -> parsing -> analysis -> optimization -> reporting.

The **implemented** `forge generate` path is shorter and config-centric. It does not run WIEN2k SCF itself.

## General lifecycle (`run_pipeline`)

Source: `src/forge/core/pipeline.py` function `run_pipeline`.

1. `get_current_backend()` (lazy import of `backend_manager`).
2. `backend.detect_problem_size()` — parse case inputs; on failure uses empty `ProblemSize`.
3. Resource suggestion: `optimizer.advisor.suggest_optimal_resources` unless the caller passed a `ResourceSuggestion`.
4. `preflight_check` — memory heuristic, SLURM env, NUMA, scratch FS, container warning. Strings starting with `ERROR: ` abort.
5. Dry-run: `backend.generate_input` returns a string.
6. Else `core.builder.build_auto` writes config (`atomic_write`), optionally validates.
7. Optional `utils.export.export_config`.
8. Returns `PipelineResult`. Exceptions become `success=False` with `validation_errors`.

Diagram: `execution_lifecycle.mmd`.

## WIEN2k

- Backend class: `forge.backends.wien2k.core.Wien2kBackend`.
- Config artifact: `.machines` (and related parallel files via builder / `utils.parallel_options`).
- Execution command default in CLI submit helper: `run_lapw -p` (`cli_commands._utils.get_exec_command`).
- Version sniff: `pipeline.detect_wien2k_version` uses `WIEN_VERSION`, `WIENROOT`, `run_lapw -v`.
- Submit: `forge submit` -> `submit.slurm.submit_slurm_job` or PBS/LSF providers; those call `sbatch` / `qsub` / `bsub`.
- Workflow DAG: `core.workflow` (SQLite provenance) + `core.workflow_executor.WorkflowExecutor` which may shell out to `forge generate` then `sbatch`/`qsub`/`bash`.
- CLI `forge run` loads YAML via `run_workflow_from_yaml`.

Diagram: `wien2k_execution.mmd`.

## Quantum ESPRESSO

- Class: `forge.backends.quantum_espresso.backend.QuantumEspressoBackend`.
- Loaded optionally in `_load_backends` from `.quantum_espresso.backend`.
- `config_generator.generate_qe_config`, `executor.py` uses `subprocess.Popen`.
- Selected when `--backend qe` or cwd matches `*.pw.in` / `*.in` (also matches many unrelated `.in` files; detection priority still prefers WIEN2k if `*.struct` exists).

Diagram: `qe_execution.mmd`.

## VASP and CP2K

- `forge.backends.vasp.VaspBackend` — INCAR parallel block, `vasp_std`/`gam`/`ncl`.
- `forge.backends.cp2k.CP2KBackend` — `.inp` parse, `cp2k.popt`/`psmp`.
- Both implement `Backend` ABC. Presence of code is not evidence they are exercised in default CLI generate, which goes through `get_current_backend()`.

## GPU

- `forge.backends.gpu_backend` is **not** registered in `_BACKENDS`. `generate.handle` imports it when `--gpu` is set.
- Separate from DFT backend selection.

## Optimization and prediction

- `forge advise` uses `CaseFileParser`, hardware counters, `scheduler.detect`; it does not always call `suggest_optimal_resources` for the printed path (JSON vs rich helpers in the same module).
- Bayesian tuner and GNN k-point predictor live under `optimizer/` and `ml/` and are reached from `optimize`, `predict`, `converge` commands (see each handler).

## Error handling

- CLI catches `FORGEError` vs generic `Exception` (`cli.py`).
- Pipeline catches all exceptions around the main try and returns `PipelineResult(success=False)`.
- Backend stubs raise `BackendError` on instantiate.
- Workflow executor retries with mixing adjustments (`_retry_with_adjustment`).

## What static analysis cannot prove

- Whether a cluster actually has `sbatch` or WIEN2k in PATH.
- Which optional backend is current at runtime.
- Dynamic `importlib` targets beyond the literal module strings in `_load_backends`.
