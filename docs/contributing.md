# Contributing Guide

FORGE is developed from the `wien2k_gen` source tree (not a PyPI package). A local editable install plus `completions/` lets you Tab-complete every flag while you hack.

Tab-complete: `source completions/forge.bash` (or zsh equivalent).

---

## Development Setup

```bash
git clone https://github.com/majidhaddad02/wien2k_gen.git
cd wien2k_gen
./install.sh --yes
# or:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make dev
source completions/forge.bash
```

There is no `pip install forge` from PyPI.

---

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_types.py -v
python -m pytest tests/ -v --no-cov
python -m pytest tests/ -v -m "slow or integration"
make test
```

---

## Code Conventions

- **Python 3.9+** — type hints everywhere
- **120 char** line limit
- **English** docstrings and comments
- **Google-style** docstrings for public API
- **`@cache`** for expensive hardware detection
- **`try/except`** with graceful I/O fallbacks
- **No new dependencies** beyond `pyproject.toml` without discussion

---

## Architecture

```
src/forge/
├── backends/              # DFT backends (wien2k, qe, vasp, cp2k)
├── core/                  # hardware, scheduler, pipeline, case parser
├── optimizer/             # advisor, parallel, Bayesian, history
├── ml/                    # GNN k-point predictor, data pipeline
├── submit/                # slurm, pbs, lsf
├── utils/                 # validation, parallel_options, diagnostics
├── cli_commands/          # one module per forge subcommand
├── cli.py                 # forge entry
├── cli_sbatch.py          # forge_sbatch entry
└── config.py              # AppConfig (wienroot, scratch_dir, max_cores, ...)
```

CLI commands: generate, submit, benchmark, diagnostics, hardware, analyze, tui, monitor, run, workflow, diagnose, optimize, screen, predict, advise, converge, history, analyze-bands, calibrate.

`.machines` rules that tests lock in:

- Physical cores per node = max of sequential stage widths, not the sum
- `ntasks` = rank count; `cpus_per_task` = max(`omp_global`, max rank width)
- `kpar:` only for `band_parallel` (hybrid functional)
- `extrafine: 1` only if `nkpt % n_ranks != 0`
- Scheduler scripts must not `exec` the calculation

---

## Adding a New Backend

1. Create `src/forge/backends/newcode.py`
2. Inherit from `Backend`
3. Implement `detect_problem_size`, `write_config`, `estimate_resources`
4. Register in `backend_manager.py`

---

## Adding a New Scheduler

1. Add `_detect_newscheduler()` in `core/scheduler.py`
2. Return `None` if inactive; else `scheduler`, `nodes`, `cores_per_node`, `total_cores`, `cpus_per_task`, `hints`, `env_type`
3. Append to the detector list in `detect()`
4. Add CLI choices (`generate`/`submit` use `slurm,pbs,lsf,sge,auto`; `benchmark` has no `sge`)

---

## Writing Tests

- Unit tests in `tests/` — mock hardware/scheduler
- Integration in `tests/integration_test.py`
- Fixtures in `tests/conftest.py` and `tests/fixtures/`
- `@pytest.mark.slow` / `@pytest.mark.integration`

---

## Docs

Keep `docs/api-reference.md` unchanged unless you are updating the Python API. User-facing CLI docs must list real argparse flags with examples. `--target` for `generate` is `time|memory|balanced|cost`; `advise --target` also allows `energy`. Clone URL is `wien2k_gen`, not `forge`.

---

## Pull Request Checklist

- [ ] `python -m pytest tests/` passes
- [ ] New tests for new behavior
- [ ] Type hints and English docstrings
- [ ] No new external dependencies without discussion
- [ ] Completions updated if you add a CLI flag (`completions/forge.bash` and `.zsh`)
- [ ] README / docs updated for user-facing changes

## License

MIT. See `LICENSE.md`.
