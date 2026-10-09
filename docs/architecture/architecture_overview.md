# High-Level Software Architecture

Revision: `c3db6e711acfbab1abe84b29f98d909374d0ab9b` (dirty).

Companion diagram: `architecture_overview.mmd`. Solid arrows are verified by imports or direct calls. Dotted arrows in the mermaid file mark lazy imports (`importlib` / function-local import) that the source contains but that are not top-level.

## What FORGE is

FORGE (`forge` on PyPI metadata; repository `wien2k_gen`) is a Python HPC helper around DFT codes. The default backend is WIEN2k. Quantum ESPRESSO, VASP, and CP2K backends exist as classes implementing `forge.backends.base.Backend`.

It is **not** a DFT engine. It generates parallel config (`.machines`, INCAR blocks, QE flags), suggests resources, submits scheduler jobs, and diagnoses SCF logs.

## Layers (verified)

1. **Entry** — `forge.cli:main`, `forge.__main__`, `forge.cli_sbatch:run_sbatch_cli`, `forge.wizard:run_wizard` (`pyproject.toml` `[project.scripts]`).
2. **CLI commands** — `forge.cli_commands.*` register argparse subparsers and handlers.
3. **Config / logging / types** — `forge.config`, `forge.logging_config`, `forge.types`, `forge.exceptions`.
4. **Pipeline** — `forge.core.pipeline.run_pipeline` sequences detect -> advise -> preflight -> `build_auto` -> validate -> export.
5. **Topology / hardware** — `forge.core.scheduler.detect` and `forge.core.hardware.*`.
6. **Backends** — `forge.backends` registry; `forge.backend_manager` is a compatibility shim that re-exports it.
7. **Optimizer** — `forge.optimizer.advisor.suggest_optimal_resources` plus Bayesian, monitor, ML packages.
8. **Submit** — `forge.submit.slurm` (functions), `PBSSubmitProvider`, `LSFSubmitProvider`.
9. **UI** — Rich console helpers in `forge.ui.rich_ui`. The `tui` subcommand does not launch Textual.

## How backends are selected

`forge.backends._load_backends` (lazy, RLock) imports `Wien2kBackend` as required. VASP, QE, and CP2K are loaded with `importlib.import_module`; failures become stub classes that raise `BackendError` on use.

`auto_detect()` looks at cwd globs: `*.struct` (WIEN2k), `POSCAR*`/`INCAR*` (VASP), `*.pw.in`/`*.in` (QE), `*.inp` (CP2K). CLI `--backend` overrides via `load_config` / `BackendCode`.

## How schedulers are selected

`cli_commands._utils.resolve_scheduler` uses the `--scheduler` flag or `core.scheduler._detect_scheduler()`. Submit then branches on `slurm` vs `pbs`/`lsf` (`SUBMIT_PROVIDERS`). SGE is an argparse choice on some commands; there is no `submit/sge.py` implementation in this checkout.

## Legend for diagrams

- Solid arrow: verified static import or call.
- Dotted / dashed: lazy import, `importlib`, or subprocess boundary.
- Do not treat related filenames as dependencies.
