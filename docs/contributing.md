# Contributing Guide

## Development Setup

```bash
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
pip install -e ".[dev]"
pip install -r requirements-dev.txt
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_types.py -v

# Without coverage requirement
python -m pytest tests/ -v --no-cov

# Run slow/integration tests
python -m pytest tests/ -v -m "slow or integration"
```

## Code Conventions

- **Python 3.9+** — type hints everywhere
- **120 char** line limit
- **English** docstrings and comments
- **Google-style** docstrings for public API
- **`@cache`** decorator for expensive hardware detection functions
- **`try/except`** with graceful fallbacks for I/O operations
- **No external dependencies** beyond what's in `pyproject.toml`

## Architecture

```
src/wien2k_gen/
├── backends/          # DFT code backends (wien2k, vasp, qe, cp2k)
│   └── base.py        # Abstract base + ProblemSize, ResourceEstimate TypedDicts
├── core/              # Core detection and optimization
│   ├── hardware.py    # CPU, memory, NUMA, interconnect detection
│   ├── topology.py    # Topology representation and distribution
│   ├── scheduler.py   # SLURM, PBS, LSF, SGE detection
│   ├── builder.py     # Configuration build pipeline
│   ├── pipeline.py    # End-to-end orchestration
│   ├── case_parser.py # WIEN2k input file parser
│   └── constants.py   # Physical constants
├── optimizer/         # Resource optimization
│   ├── advisor.py     # suggest_optimal_resources()
│   ├── monitor.py     # SCF convergence monitoring
│   ├── profiler.py    # Auto-profiling
│   ├── history.py     # Execution history analysis
│   └── bayesian.py    # Bayesian optimization
├── submit/            # Scheduler job submission
│   ├── slurm.py
│   ├── pbs.py
│   └── lsf.py
├── utils/             # Utilities
│   ├── parallel_options.py  # parallel_options generation
│   ├── validation.py        # .machines validation
│   ├── diagnostic.py        # System diagnostics
│   └── ...
├── cli.py             # Rich CLI
├── wizard.py          # Interactive wizard
├── backend_manager.py # Backend registration
├── ui/                # User interfaces
│   ├── rich_ui.py     # Rich text UI utilities
│   ├── analysis.py    # UI analysis helpers
│   └── ...
├── types.py           # Type definitions and enums
├── config.py          # Configuration management
└── exceptions.py      # Custom exceptions
```

## Adding a New Backend

1. Create `src/wien2k_gen/backends/newcode.py`
2. Inherit from `Backend` (in `base.py`)
3. Implement required methods:
   - `detect_problem_size() -> ProblemSize`
   - `write_config(suggestion, topo) -> str`
   - `estimate_resources(params, topo) -> ResourceEstimate`
4. Register in `backend_manager.py`

## Adding a New Scheduler

1. Add `_detect_newscheduler()` in `core/scheduler.py`
2. Follow the existing pattern:
   - Return `None` if scheduler not active
   - Return dict with keys: `scheduler`, `nodes`, `cores_per_node`, `total_cores`, `cpus_per_task`, `hints`, `env_type`
3. Add to detector list in `detect()`:
   ```python
   detectors = [_detect_slurm, _detect_pbs, _detect_lsf, _detect_sge, _detect_newscheduler]
   ```
4. Add CLI choices where appropriate

## Writing Tests

- **Unit tests** in `tests/` — use `unittest.mock` for hardware/scheduler dependencies
- **Integration tests** in `tests/integration_test.py` — create temporary WIEN2k case directories
- **Fixtures** in `tests/conftest.py` and `tests/fixtures/`
- **Mark slow tests** with `@pytest.mark.slow`
- **Mark integration tests** with `@pytest.mark.integration`

## Pull Request Checklist

- [ ] All existing tests pass: `python -m pytest tests/`
- [ ] New tests added for new functionality
- [ ] Type hints on all new functions
- [ ] English docstrings on all public functions
- [ ] No new external dependencies without discussion
- [ ] README updated if user-facing change

## License

MIT License. See [LICENSE.md](../LICENSE.md). All contributions must be under the same license.
