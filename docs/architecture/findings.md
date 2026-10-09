# Architecture Findings

Revision: `c3db6e711acfbab1abe84b29f98d909374d0ab9b`. Evidence is static unless noted. Confidence: **verified** = import or literal in source; **partial** = present but target ambiguous; **n/a** = not observed.

## 1. Application entry points

**Finding:** Three console scripts plus `python -m forge`.

**Evidence:** `pyproject.toml` lines 99–102 `[project.scripts]`; `src/forge/__main__.py` imports `forge.cli.main`; `cli.py` `if __name__ == "__main__"`.

**Confidence:** verified.

## 2. High-level architecture

**Finding:** CLI -> command handlers -> pipeline/advisor/submit/workflow. DFT codes are backends behind `forge.backends.Backend`.

**Evidence:** `cli.py` `get_handler`; `pipeline.run_pipeline`; `backends/__init__.py` `_load_backends`.

**Confidence:** verified.

## 3. Central orchestration modules

**Finding:** `forge.core.pipeline`, `forge.core.builder`, `forge.core.scheduler`, `forge.optimizer.advisor`, `forge.backends`.

**Evidence:** `generate.handle` calls `run_pipeline` and `detect`; `run_pipeline` calls `suggest_optimal_resources` and `build_auto`.

**Confidence:** verified for generate. Other commands may skip the pipeline.

## 4. Highest internal dependency counts

**Finding:** Highest in-degree production modules: `forge.logging_config` (66), `forge.core.topology` (27), `forge.config` (25), `forge.core.hardware` (21), `forge.cli_commands.base` (21), `forge.cli_commands._utils` (20), `forge.utils.atomic_write` (14), `forge.core.scheduler` (12).

**Evidence:** AST import graph in `module_dependencies.json`.

**Confidence:** verified for import edges. Lazy imports inside functions still count as import edges with context `function`.

## 5. Circular dependencies

**Finding:** {`forge.config`, `forge.core.hardware`, `forge.core.hardware.cpu`, `forge.core.hardware.detection`, `forge.core.hardware.system`, `forge.core.hardware.types`, `forge.core.hardware.wrapper`, `forge.core.locator`, `forge.exceptions`, `forge.logging_config`, `forge.types`}; {`forge.backend_manager`, `forge.backends`, `forge.backends.wien2k`, `forge.backends.wien2k.core`, `forge.optimizer.advisor`}; {`forge.core.workflow_executor`, `forge.optimizer`, `forge.optimizer.bayesian`, `forge.optimizer.bayesian.core`, `forge.optimizer.monitor`, `forge.optimizer.monitor.engine`}

**Evidence:** Tarjan SCC on unique production import edges excluding `TYPE_CHECKING`. Edge table: `dependency_report.md`. This run: 20 top-level edges and 13 function-local / conditional-function edges inside those components.

**Note:** Authors already break import-time cycles with function-local imports (`pipeline._get_current_backend`, `advisor._get_current_backend`, `builder._get_current_backend`, `logging_config` formatter). Those still appear as static edges. They are not proof of an import-time crash.

**Confidence:** verified that the listed import statements exist. Not verified that importing the package deadlocks.

## 6. Package dependencies

**Finding:** See `dependency_report.md` package table. Typical flow: `cli_commands` -> `core` / `optimizer` / `submit` / `backends`; `optimizer` -> `core`; `backends` -> `core` + `utils`.

**Confidence:** verified.

## 7. Backend registration and selection

**Finding:** In-code registry, not setuptools entry points.

**Evidence:** `backends/__init__.py` `_BACKENDS` filled in `_load_backends`. WIEN2k import is required; others `importlib` + stub. `auto_detect` uses filesystem globs. `BackendManager` is a shim (`backend_manager.py` module docstring).

**Confidence:** verified.

## 8. External programs

**Finding:** `subprocess.run` / `Popen` appear in scheduler submit modules, workflow executor, hardware probes, GPU detector, QE executor, wizard, scratch utilities, benchmarks.

**Evidence:** `external_execution.json`. Literal names in those files include `sbatch`, `squeue`, `qsub`, `qstat`, `bsub`, `scancel`, `qdel`, `bkill`, `run_lapw`, `lscpu`-class probes, `df`.

**Confidence:** verified that the call sites exist. Not verified that binaries exist at runtime.

**Not found as implementations:** a dedicated SGE submit module (flag exists). No VASP/CP2K executor module equivalent to `quantum_espresso/executor.py`.

## 9. Scheduler selection

**Finding:** Flag `auto` -> `_detect_scheduler()`. SLURM uses functional API; PBS/LSF use provider classes. `WorkflowExecutor._detect_scheduler` only distinguishes SLURM env vs PBS env vs default `"slurm"`.

**Evidence:** `cli_commands/_utils.py`, `cli_commands/submit.py`, `core/workflow_executor.py`.

**Confidence:** verified.

## 10. Errors

**Finding:** Typed `FORGEError` hierarchy formatted for UI/JSON in `cli.main`. Pipeline converts exceptions to `PipelineResult`. Backend stubs raise `BackendError`.

**Evidence:** `src/forge/cli.py` handlers; `exceptions.py`; `pipeline.py` except block.

**Confidence:** verified.

## 11. Tight coupling

**Finding:** Pipeline, builder, and advisor all talk to the current backend and topology. Duplicate `ResourceSuggestion` / `ProblemSize` types exist in `types.py`, `backends.base` TypedDicts, and fallback dataclasses inside `pipeline.py` / `builder.py`.

**Evidence:** try/except ImportError fallbacks in those files.

**Confidence:** verified that duplicates exist. Not labeled a bug.

## 12. Limited internal connections

**Finding:** No production module is isolated in the import graph. No two-segment package is isolated from other packages either (`dependency_report.md`). `cli_commands.tui` is registered but does not import Textual. `download_and_train.py` at repo root is outside the `forge` package.

**Confidence:** verified for imports.

## 13. Inconsistencies (observations, not defects)

- README still documents `forge tui` and Textual; handler reports TUI removed.
- `forge generate` argparse has no `--case` / `--task`, but `WorkflowExecutor._submit_node` runs `forge generate --case ... --task ...`.
- Pytest `addopts` includes `--numprocesses=auto` (needs pytest-xdist from `[dev]`).
- Optional extra names in stub errors say `pip install wien2k-gen[qe]` while the distribution name in pyproject is `forge`.

## 14. Relationships static analysis cannot establish

Dynamic backend instance, current scheduler binaries, Textual app (removed), runtime coverage, and most `self.*` calls across mixins.

## 15. Tests vs components

See `testing_report.md`. Strong static associations for advisor, builder, scheduler, WIEN2k backend, Bayesian, GNN. CLI dispatch has less dedicated test naming.

## 16. Runtime tracing that would help

- One integration run of `forge generate --dry-run` in `examples/01_si_semiconductor`.
- Logging around `_load_backends` and `auto_detect`.
- pytest-cov HTML after `pip install -e .[dev]` (not executed as part of claiming application test status).

Registered CLI commands in this revision: `advise`, `analyze`, `analyze-bands`, `benchmark`, `calibrate`, `converge`, `diagnose`, `diagnostics`, `generate`, `hardware`, `history`, `monitor`, `optimize`, `predict`, `run`, `screen`, `submit`, `tui`, `workflow`.
