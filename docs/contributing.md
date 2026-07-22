# Contributing Guide

## Development Setup

```bash
git clone https://github.com/majidhaddad02/forge.git
cd forge
pip install -e ".[dev]"
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
src/forge/
├── backends/              # DFT code backends
│   ├── base.py            # Abstract backend + ProblemSize, ResourceEstimate
│   ├── elpa_selector.py   # ELPA/ScaLAPACK solver selection
│   ├── gpu_backend.py     # GPU detection and strategy
│   ├── vasp.py            # VASP backend
│   ├── cp2k.py            # CP2K backend
│   ├── wien2k/            # WIEN2k backend
│   │   ├── core.py        # Main WIEN2k backend
│   │   └── parsers.py     # WIEN2k output parsers
│   └── quantum_espresso/  # Quantum ESPRESSO backend
│       ├── backend.py
    │       ├── config_generator.py  # QE parallel config generator
    │       ├── executor.py
│       └── parser.py
├── core/                  # Core detection and optimization
│   ├── hardware/          # Hardware detection (CPU, memory, NUMA)
│   │   ├── cpu.py         # CPU architecture detection
│   │   ├── detection.py   # Hardware feature detection
│   │   ├── system.py      # System-level info
│   │   ├── wrapper.py     # lscpu/likwid wrapper
│   │   └── types.py       # Hardware type definitions
│   ├── topology.py        # Topology representation and BLACS grid
│   ├── scheduler.py       # SLURM, PBS, LSF, SGE detection
│   ├── builder.py         # Configuration build pipeline
│   ├── pipeline.py        # End-to-end orchestration
│   ├── case_parser.py     # WIEN2k input file parser (in1, in2, inm, scf, struct)
│   ├── workflow.py        # DAG-based workflow representation
│   ├── workflow_executor.py  # DAG runtime engine (mixing, Kerker, CFP)
│   ├── constants.py       # Physical constants
│   ├── energy.py          # RAPL energy measurement
│   ├── perf_counters.py   # likwid-perfctr / perf stat interface
│   ├── electronic_structure.py  # Band structure analysis
│   ├── materials_project.py     # Materials Project integration
│   ├── terminal_monitor.py      # Terminal progress display
│   └── locator.py         # WIENROOT auto-detection
├── optimizer/             # Resource & parameter optimization
│   ├── advisor.py         # suggest_optimal_resources()
│   ├── parallel.py        # NUMA-aware parallelization engine
│   ├── convergence.py     # SCF convergence analysis
│   ├── history.py         # Execution history SQLite store
│   ├── profiler.py        # Auto-profiling
│   ├── bayesian_tuner.py  # Bayesian parameter tuning entry point
│   ├── ml_predict.py      # ML prediction integration
│   ├── gpu_detector.py    # GPU hardware detection
    │   ├── bayesian/          # Bayesian optimization subpackage
    │   │   ├── core.py        # BayesianOptimizer, multi-fidelity BO
    │   │   ├── gp.py          # GP with ARD, NLL, partial derivative
    │   │   ├── kernels.py     # RBF-ARD, Matérn ν=2.5 kernels
    │   │   ├── acquisition.py # EI, q-EI (Monte Carlo joint posterior)
    │   │   ├── elements.py    # Periodic table, chemical similarity
    │   │   ├── sampling.py    # Latin Hypercube Sampling
    │   │   ├── constraints.py # Memory/walltime constraint estimation
    │   │   ├── bohb.py        # BOHB with TPE/KDE multi-fidelity optimization
    │   │   └── dpp.py         # DPP batch selector (Cholesky greedy MAP)
    │   └── monitor/           # SCF convergence monitoring
│       ├── convergence.py # Charge sloshing, Durbin-Watson, FFT
│       ├── checkpoint.py  # SCF checkpoint/restore (heuristic, not Daly)
│       ├── engine.py      # Monitoring engine
│       └── types.py       # ConvergenceAnalysis type
├── ml/                    # Machine learning
│   ├── gnn_kpoint_predictor.py  # CGCNN k-point prediction (pure NumPy)
│   └── data_pipeline.py         # Materials Project dataset builder
├── submit/                # Scheduler job submission
│   ├── slurm.py
│   ├── pbs.py
│   └── lsf.py
├── utils/                 # Utilities
│   ├── parallel_options.py  # parallel_options generation
│   ├── validation.py        # .machines validation
│   ├── diagnostic.py        # System diagnostics
│   ├── export.py            # JSON export
│   ├── scratch.py           # Scratch filesystem helpers
│   ├── atomic_write.py      # Atomic file writes
│   ├── filelock.py          # File locking
│   └── subprocess_utils.py
├── cli_commands/          # CLI subcommands
│   ├── base.py            # Command registration framework
│   ├── _utils.py          # Shared CLI utilities
│   ├── generate.py        # forge generate
│   ├── submit.py          # forge submit
│   ├── advise.py          # forge advise
│   ├── diagnose.py        # forge diagnose
│   ├── benchmark.py       # forge benchmark
│   ├── optimize.py        # forge optimize
│   ├── predict.py         # forge predict
│   ├── diagnostics.py     # forge diagnostics
│   ├── monitor.py         # forge monitor
│   ├── screen.py          # forge screen
│   ├── workflow.py        # forge workflow
│   ├── history.py         # forge history
│   ├── run.py             # forge run
│   ├── converge.py        # forge converge
│   ├── tui.py             # forge tui
│   ├── hardware.py        # forge hardware
│   ├── analyze.py         # forge analyze
│   └── analyze_bands.py   # forge analyze_bands
├── ui/                    # User interfaces
│   ├── rich_ui.py         # Rich text UI utilities
│   └── analysis.py        # UI analysis helpers
├── benchmark/             # Benchmarking
│   ├── synthetic.py       # LogP + Amdahl synthetic benchmarks
│   ├── real.py            # Real-world scaling benchmarks
│   └── report.py          # Benchmark reporting and analysis
├── cli.py                 # Main CLI entry point
├── wizard.py              # Interactive wizard (Textual TUI)
├── backend_manager.py     # Backend registration
├── config.py              # Configuration management
├── types.py               # Type definitions and enums
└── exceptions.py          # Custom exceptions
```

## Adding a New Backend

1. Create `src/forge/backends/newcode.py`
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
