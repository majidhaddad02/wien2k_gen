# Testing and CI Report

Coverage numbers below are **static import associations**, not pytest-cov runtime coverage.

## Layout

- Unit-style modules: `tests/test_*.py`.
- Shared fixtures: `tests/conftest.py`, `tests/fixtures/`.
- Extra integration module: `tests/integration_test.py` (name does not match `python_files = test_*.py` in `pyproject.toml`, so pytest collection may skip it unless invoked explicitly).
- ReFrame: `tests/reframe/`.

Pytest markers in `pyproject.toml`: `slow`, `integration`, `hardware`.

## CI (`ci.yml`)

Job `lint-and-test` on Ubuntu, Python 3.9–3.12:

1. `pip install -e ".[dev,hpc]"`
2. `ruff check src/ tests/ --exit-zero` (lint does not fail the job)
3. `mypy src/`
4. `pytest --cov=forge --cov-fail-under=15`
5. `python -c "import forge; ..."`
6. Upload `coverage.xml`

## ReFrame workflow

`reframe_benchmark.yml` installs `reframe-hpc` and runs `tests/reframe/wien2k_gen_test.py` with tags `smoke` and `benchmark`. Failures fall back to a second untagged `reframe` invocation (`||`).

## Test-to-module map (static imports)

| Test | N forge imports | Modules |
|------|----------------:|---------|
| `tests/__init__.py` | 0 |  |
| `tests/codebase_map/test_extractor.py` | 0 |  |
| `tests/conftest.py` | 1 | forge.types |
| `tests/integration_test.py` | 6 | forge.cli, forge.config, forge.core.scheduler, forge.core.topology, forge.optimizer.advisor, forge.types |
| `tests/reframe/reframe_config.py` | 0 |  |
| `tests/reframe/wien2k_gen_test.py` | 0 |  |
| `tests/test_advisor.py` | 4 | forge.core.pipeline, forge.core.topology, forge.optimizer.advisor, forge.types |
| `tests/test_analysis.py` | 1 | forge.ui.analysis |
| `tests/test_auto_detect.py` | 3 | forge.backend_manager, forge.exceptions, forge.types |
| `tests/test_bayesian.py` | 2 | forge.optimizer.bayesian, forge.optimizer.history |
| `tests/test_bayesian_core.py` | 4 | forge.core.constants, forge.optimizer.bayesian, forge.optimizer.bayesian.core, forge.types |
| `tests/test_bohb_dpp.py` | 4 | forge.optimizer.bayesian.bohb, forge.optimizer.bayesian.dpp, forge.optimizer.bayesian.gp, forge.optimizer.bayesian.kernels |
| `tests/test_builder.py` | 4 | forge.backends.base, forge.backends.wien2k.core, forge.core.builder, forge.core.topology |
| `tests/test_case_parser.py` | 1 | forge.core.case_parser |
| `tests/test_completions.py` | 0 |  |
| `tests/test_config.py` | 1 | forge.config |
| `tests/test_convergence_opt.py` | 1 | forge.optimizer.convergence |
| `tests/test_diagnose.py` | 2 | forge.cli_commands, forge.core.case_parser |
| `tests/test_elpa_selector.py` | 2 | forge.backends.elpa_selector, forge.core.topology |
| `tests/test_gnn_kpoint_predictor.py` | 1 | forge.ml.gnn_kpoint_predictor |
| `tests/test_hardware.py` | 1 | forge.core.hardware |
| `tests/test_integration.py` | 3 | forge.backends.wien2k, forge.core.case_parser, forge.core.topology |
| `tests/test_new_features.py` | 3 | forge.core.hardware, forge.core.scheduler, forge.core.topology |
| `tests/test_parallel.py` | 2 | forge.core.topology, forge.optimizer.parallel |
| `tests/test_parallel_options.py` | 2 | forge.core.topology, forge.utils.parallel_options |
| `tests/test_perf_counters.py` | 4 | forge.cli_commands, forge.core, forge.core.topology, forge.types |
| `tests/test_robustness.py` | 4 | forge.core.case_parser, forge.core.topology, forge.optimizer.advisor, forge.utils.validation |
| `tests/test_scheduler.py` | 3 | forge.core.scheduler, forge.core.topology, forge.types |
| `tests/test_slurm.py` | 2 | forge.core.topology, forge.submit.slurm |
| `tests/test_submit.py` | 5 | forge.cli_commands, forge.core.topology, forge.submit.lsf, forge.submit.pbs, forge.types |
| `tests/test_topology.py` | 1 | forge.core.topology |
| `tests/test_types.py` | 1 | forge.types |
| `tests/test_utils.py` | 4 | forge.types, forge.utils.atomic_write, forge.utils.filelock, forge.utils.validation |
| `tests/test_wien2k_backend.py` | 9 | forge.backends, forge.backends.base, forge.backends.wien2k, forge.backends.wien2k.core, forge.backends.wien2k.parsers, forge.core.case_parser, forge.core.topology, forge.types, forge.ui.analysis |
| `tests/test_wien2k_flags.py` | 1 | forge.types |
| `tests/test_wizard.py` | 1 | forge.wizard |
