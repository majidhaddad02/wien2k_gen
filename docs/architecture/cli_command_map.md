# CLI Command Map

Packaging entry points from `pyproject.toml` `[project.scripts]`:

| Console script | Target |
|----------------|--------|
| `forge` | `forge.cli:main` |
| `forge_sbatch` | `forge.cli_sbatch:run_sbatch_cli` |
| `forge_wizard` | `forge.wizard:run_wizard` |

`python -m forge` uses `forge.__main__` which calls `forge.cli.main`.

Subcommands are registered in two places:

1. `cli_commands.__init__.register_all` calls each module's `register(subparsers)` (argparse).
2. Each module calls `register_command(name, handle)` which fills `cli_commands.base._registry`.
3. `forge.cli.main` dispatches with `get_handler(args.command)(args, cfg)`.

Do not assume a file in `cli_commands/` is a live command unless it appears below.

| Command | Handler | Source |
|---------|---------|--------|
| `advise` | `forge.cli_commands.advise.handle` | `src/forge/cli_commands/advise.py` |
| `analyze` | `forge.cli_commands.analyze.handle` | `src/forge/cli_commands/analyze.py` |
| `analyze-bands` | `forge.cli_commands.analyze_bands.handle` | `src/forge/cli_commands/analyze_bands.py` |
| `benchmark` | `forge.cli_commands.benchmark.handle` | `src/forge/cli_commands/benchmark.py` |
| `calibrate` | `forge.cli_commands.calibrate.handle` | `src/forge/cli_commands/calibrate.py` |
| `converge` | `forge.cli_commands.converge.handle` | `src/forge/cli_commands/converge.py` |
| `diagnose` | `forge.cli_commands.diagnose.handle` | `src/forge/cli_commands/diagnose.py` |
| `diagnostics` | `forge.cli_commands.diagnostics.handle` | `src/forge/cli_commands/diagnostics.py` |
| `generate` | `forge.cli_commands.generate.handle` | `src/forge/cli_commands/generate.py` |
| `hardware` | `forge.cli_commands.hardware.handle` | `src/forge/cli_commands/hardware.py` |
| `history` | `forge.cli_commands.history.handle` | `src/forge/cli_commands/history.py` |
| `monitor` | `forge.cli_commands.monitor.handle` | `src/forge/cli_commands/monitor.py` |
| `optimize` | `forge.cli_commands.optimize.handle` | `src/forge/cli_commands/optimize.py` |
| `predict` | `forge.cli_commands.predict.handle` | `src/forge/cli_commands/predict.py` |
| `run` | `forge.cli_commands.run.handle` | `src/forge/cli_commands/run.py` |
| `screen` | `forge.cli_commands.screen.handle` | `src/forge/cli_commands/screen.py` |
| `submit` | `forge.cli_commands.submit.handle` | `src/forge/cli_commands/submit.py` |
| `tui` | `forge.cli_commands.tui.handle` | `src/forge/cli_commands/tui.py` |
| `workflow` | `forge.cli_commands.workflow.handle` | `src/forge/cli_commands/workflow.py` |

The `tui` command is registered but its handler prints that the TUI was removed and returns `tui_removed` (`src/forge/cli_commands/tui.py`).

Global flags live on `forge.cli.create_parser`: `--verbose`, `--quiet`, `--json`, `--config`, `--backend`, `--log-file`, `--plain`, `--no-color`.
