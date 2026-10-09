"""Human-readable architecture reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _mod_by_name(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["module"]: m for m in analysis["modules"]}


def write_dependency_report(out_dir: Path, graphs: dict[str, Any], meta: dict[str, Any]) -> None:
    lines = [
        "# Module Import Dependency Report",
        "",
        f"Revision: `{meta['git'].get('commit')}` ({meta['git'].get('dirty')}).",
        "Edges are unique production-code imports resolved from AST. Standard-library and third-party imports are summarized separately.",
        "",
        "## Package-level dependencies",
        "",
        "| From package | To package | Unique module edges |",
        "|--------------|------------|--------------------:|",
    ]
    for e in graphs["package_deps"]:
        lines.append(f"| `{e['from']}` | `{e['to']}` | {e['count']} |")
    lines.extend(["", "## Highest in-degree (imported by many)", "", "| Module | In | Out |", "|--------|---:|----:|"])
    in_d = graphs["in_degree"]
    out_d = graphs["out_degree"]
    for name, n in graphs["hubs_in"]:
        lines.append(f"| `{name}` | {n} | {out_d.get(name, 0)} |")
    lines.extend(["", "## Highest out-degree (imports many)", "", "| Module | Out | In |", "|--------|----:|--:|"])
    for name, n in graphs["hubs_out"]:
        lines.append(f"| `{name}` | {n} | {in_d.get(name, 0)} |")
    lines.extend(["", "## Circular import components", ""])
    if graphs["cycles"]:
        lines.append(
            "Computed on unique production import edges **excluding** `type_checking` context. "
            "Function-local imports still count, because they execute if that function runs."
        )
        lines.append("")
        for i, c in enumerate(graphs["cycles"], 1):
            lines.append(f"{i}. " + ", ".join(f"`{x}`" for x in c))
        if graphs.get("cycle_edges"):
            lines.extend(["", "Edges in those components:", ""])
            lines.append("| From | To | Context | Location |")
            lines.append("|------|----|---------|----------|")
            for e in graphs["cycle_edges"]:
                loc = f"`{e.get('path','')}`:{e.get('lineno','')}"
                lines.append(f"| `{e['from']}` | `{e['to']}` | {e.get('context')} | {loc} |")
    else:
        lines.append("No strongly connected components with size > 1 among production modules (TYPE_CHECKING imports excluded).")
    if graphs["self_loops"]:
        lines.extend(["", "Self-imports:"] + [f"- `{x[0]}`" for x in graphs["self_loops"]])
    lines.extend(["", "## Isolated production modules", ""])
    if graphs["isolated"]:
        for m in graphs["isolated"]:
            lines.append(f"- `{m}`")
        lines.append("")
        lines.append(
            "Isolation means no *internal* production import edge. The module may still be reached via packaging entry points, lazy `__getattr__`, `importlib`, or tests."
        )
    else:
        lines.append("None at module level (every production module has at least one internal import edge).")
    lines.extend(["", "## Isolated packages (top two name segments, cross-package edges only)", ""])
    pkgs = graphs.get("isolated_packages") or []
    if pkgs:
        for p in pkgs:
            lines.append(f"- `{p}`")
        lines.append("")
        lines.append(
            "A package listed here has no import edge to or from a *different* top-level package. Submodules inside it may still import each other."
        )
    else:
        lines.append("None. Packages such as `forge.utils` / `forge.ml` import other `forge.*` packages.")
    lines.extend(["", "## Third-party imports (name counts)", ""])
    for name, n in list(graphs["third_party"].items())[:40]:
        lines.append(f"- `{name}` ({n})")
    lines.extend(["", "## Standard library imports (top)", ""])
    for name, n in list(graphs["stdlib"].items())[:30]:
        lines.append(f"- `{name}` ({n})")
    lines.append("")
    (out_dir / "dependency_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_cli_command_map_md(out_dir: Path, cli_map: list[dict[str, Any]]) -> None:
    lines = [
        "# CLI Command Map",
        "",
        "Packaging entry points from `pyproject.toml` `[project.scripts]`:",
        "",
        "| Console script | Target |",
        "|----------------|--------|",
        "| `forge` | `forge.cli:main` |",
        "| `forge_sbatch` | `forge.cli_sbatch:run_sbatch_cli` |",
        "| `forge_wizard` | `forge.wizard:run_wizard` |",
        "",
        "`python -m forge` uses `forge.__main__` which calls `forge.cli.main`.",
        "",
        "Subcommands are registered in two places:",
        "",
        "1. `cli_commands.__init__.register_all` calls each module's `register(subparsers)` (argparse).",
        "2. Each module calls `register_command(name, handle)` which fills `cli_commands.base._registry`.",
        "3. `forge.cli.main` dispatches with `get_handler(args.command)(args, cfg)`.",
        "",
        "Do not assume a file in `cli_commands/` is a live command unless it appears below.",
        "",
        "| Command | Handler | Source |",
        "|---------|---------|--------|",
    ]
    for r in cli_map:
        lines.append(f"| `{r['command']}` | `{r['handler']}` | `{r['path']}` |")
    lines.extend(
        [
            "",
            "The `tui` command is registered but its handler prints that the TUI was removed and returns `tui_removed` (`src/forge/cli_commands/tui.py`).",
            "",
            "Global flags live on `forge.cli.create_parser`: `--verbose`, `--quiet`, `--json`, `--config`, `--backend`, `--log-file`, `--plain`, `--no-color`.",
            "",
        ]
    )
    (out_dir / "cli_command_map.md").write_text("\n".join(lines), encoding="utf-8")


def write_architecture_overview_md(out_dir: Path, meta: dict[str, Any]) -> None:
    text = f"""# High-Level Software Architecture

Revision: `{meta['git'].get('commit')}` ({meta['git'].get('dirty')}).

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
"""
    (out_dir / "architecture_overview.md").write_text(text, encoding="utf-8")


def write_workflow_analysis(out_dir: Path) -> None:
    text = """# Computational Workflow Analysis

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
"""
    (out_dir / "workflow_analysis.md").write_text(text, encoding="utf-8")


def write_testing_report(out_dir: Path, tests_map: list[dict[str, str]]) -> None:
    lines = [
        "# Testing and CI Report",
        "",
        "Coverage numbers below are **static import associations**, not pytest-cov runtime coverage.",
        "",
        "## Layout",
        "",
        "- Unit-style modules: `tests/test_*.py`.",
        "- Shared fixtures: `tests/conftest.py`, `tests/fixtures/`.",
        "- Extra integration module: `tests/integration_test.py` (name does not match `python_files = test_*.py` in `pyproject.toml`, so pytest collection may skip it unless invoked explicitly).",
        "- ReFrame: `tests/reframe/`.",
        "",
        "Pytest markers in `pyproject.toml`: `slow`, `integration`, `hardware`.",
        "",
        "## CI (`ci.yml`)",
        "",
        "Job `lint-and-test` on Ubuntu, Python 3.9–3.12:",
        "",
        "1. `pip install -e \".[dev,hpc]\"`",
        "2. `ruff check src/ tests/ --exit-zero` (lint does not fail the job)",
        "3. `mypy src/`",
        "4. `pytest --cov=forge --cov-fail-under=15`",
        "5. `python -c \"import forge; ...\"`",
        "6. Upload `coverage.xml`",
        "",
        "## ReFrame workflow",
        "",
        "`reframe_benchmark.yml` installs `reframe-hpc` and runs `tests/reframe/wien2k_gen_test.py` with tags `smoke` and `benchmark`. Failures fall back to a second untagged `reframe` invocation (`||`).",
        "",
        "## Test-to-module map (static imports)",
        "",
        "| Test | N forge imports | Modules |",
        "|------|----------------:|---------|",
    ]
    for row in tests_map:
        mods = row["imported_forge_modules"].replace(";", ", ")
        lines.append(f"| `{row['test_path']}` | {row['n_targets']} | {mods} |")
    lines.append("")
    (out_dir / "testing_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_limitations(out_dir: Path, analysis: dict[str, Any], graphs: dict[str, Any]) -> None:
    n_calls = sum(len(m["calls"]) for m in analysis["modules"] if m["category"] == "production")
    n_verified = sum(
        1
        for m in analysis["modules"]
        if m["category"] == "production"
        for c in m["calls"]
        if c["confidence"] == "verified"
    )
    n_unresolved = sum(
        1
        for m in analysis["modules"]
        if m["category"] == "production"
        for c in m["calls"]
        if c["confidence"] == "unresolved"
    )
    syntax = [m for m in analysis["modules"] if m.get("syntax_error")]
    lines = [
        "# Static Analysis Limitations",
        "",
        "Python dynamic features prevent a complete runtime graph from AST alone.",
        "",
        f"- Production call sites extracted: {n_calls}.",
        f"- Resolved with verified confidence: {n_verified}.",
        f"- Unresolved: {n_unresolved}.",
        "",
        "## Intentionally not claimed",
        "",
        "- A function with no static callers is **not** unused. It may be a CLI handler, packaging entry, lazy `__getattr__` export, or plugin.",
        "- `forge.__init__.__getattr__` re-exports many names; those edges are not import statements in callers that use `from forge import detect`.",
        "- `importlib.import_module` in `backends._load_backends` is recorded as a call, not as a static import of VASP/QE/CP2K (those modules also have real `from .wien2k import` for the primary backend).",
        "- `self.method()` is resolved only when the enclosing class defines that method name.",
        "- Subprocess argv is not fully evaluated; see `external_execution.json` for call-site previews.",
        "",
        "## Syntax errors during parse",
        "",
    ]
    if syntax:
        for m in syntax:
            lines.append(f"- `{m['path']}`: {m['syntax_error']}")
    else:
        lines.append("None. All scanned `.py` files parsed.")
    lines.extend(
        [
            "",
            "## Isolated modules",
            "",
            "See `dependency_report.md`. Isolation is about internal imports only.",
            "",
            "## Renderers",
            "",
            "Graphviz `dot` is optional for `module_dependencies.svg`. `architecture_overview.svg` is always written as a small fallback drawing. mermaid-cli `mmdc` is optional. `render_status.json` records what this run actually rendered.",
            "",
        ]
    )
    (out_dir / "limitations.md").write_text("\n".join(lines), encoding="utf-8")


def write_findings(out_dir: Path, graphs: dict[str, Any], meta: dict[str, Any], cli_map: list[dict[str, Any]]) -> None:
    hubs = ", ".join(f"`{n}` ({k})" for n, k in graphs["hubs_in"][:8])
    cycles = graphs["cycles"]
    cycle_txt = (
        "; ".join("{" + ", ".join("`" + x + "`" for x in c) + "}" for c in cycles)
        if cycles
        else "No multi-node SCCs in the production import graph after excluding TYPE_CHECKING imports."
    )
    n_cycle_fn = sum(1 for e in graphs.get("cycle_edges") or [] if e.get("context") and "function" in str(e.get("context")))
    n_cycle_top = sum(1 for e in graphs.get("cycle_edges") or [] if e.get("context") == "top_level")
    text = f"""# Architecture Findings

Revision: `{meta['git'].get('commit')}`. Evidence is static unless noted. Confidence: **verified** = import or literal in source; **partial** = present but target ambiguous; **n/a** = not observed.

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

**Finding:** Highest in-degree production modules: {hubs}.

**Evidence:** AST import graph in `module_dependencies.json`.

**Confidence:** verified for import edges. Lazy imports inside functions still count as import edges with context `function`.

## 5. Circular dependencies

**Finding:** {cycle_txt}

**Evidence:** Tarjan SCC on unique production import edges excluding `TYPE_CHECKING`. Edge table: `dependency_report.md`. This run: {n_cycle_top} top-level edges and {n_cycle_fn} function-local / conditional-function edges inside those components.

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

Registered CLI commands in this revision: {", ".join("`" + r["command"] + "`" for r in cli_map)}.
"""
    (out_dir / "findings.md").write_text(text, encoding="utf-8")


def write_architecture_readme(out_dir: Path, meta: dict[str, Any], render_note: str) -> None:
    text = f"""# FORGE architecture map

Static reconstruction of `wien2k_gen` / FORGE. Generated `{meta['timestamp_utc']}` from git `{meta['git'].get('commit')}` ({meta['git'].get('dirty')}).

This directory does **not** replace `docs/workflow.md` or the user guides. It explains how Python modules connect.

## How to read diagrams

- **Solid arrows:** relationship found in source (import or named call).
- **Dashed / dotted:** lazy import, subprocess, or inferred only when labeled.
- **Node labels:** Python module or function names, or external binaries.
- **Verified vs partial vs unresolved:** see `limitations.md`.

Do not treat a diagram as a complete runtime trace.

## Files

| File | Contents |
|------|----------|
| `file_tree.md` | Checkout tree with categories |
| `file_inventory.json` / `.csv` | Machine-readable inventory |
| `architecture_overview.md` / `.mmd` / `.svg` | Layered system view |
| `module_dependencies.dot` / `.mmd` / `.json` | Import graph |
| `dependency_report.md` | Hubs, cycles, externals |
| `cli_command_map.md` / `.json` / `.csv` | Subcommand to handler |
| `cli_execution_flow.mmd` | CLI dispatch |
| `execution_lifecycle.mmd` | `run_pipeline` sequence |
| `wien2k_execution.mmd` / `qe_execution.mmd` / `scheduler_execution.mmd` | Backend/scheduler sequences |
| `call_graph.json` / `call_graph_focused.mmd` | Partial call graph |
| `test_architecture.mmd` / `test_module_mapping.csv` / `ci_pipeline.mmd` / `testing_report.md` | Tests and CI |
| `workflow_analysis.md` | Execution narrative |
| `findings.md` | Answers to the architecture questions |
| `limitations.md` | What AST cannot prove |
| `analysis_meta.json` | Timestamp and revision |
| `render_status.json` | Whether SVG was produced |

{render_note}

## Regenerating

From the repository root:

```
python tools/codebase_map/analyze.py --root . --output-dir docs/architecture
```

Optional:

```
dot -Tsvg docs/architecture/module_dependencies.dot -o docs/architecture/module_dependencies.svg
mmdc -i docs/architecture/architecture_overview.mmd -o docs/architecture/architecture_overview.svg
```

Graphviz and mermaid-cli are optional. Analysis uses only the Python standard library.

## After major changes, re-review

- `pyproject.toml` scripts
- `src/forge/cli.py` and `cli_commands/`
- `src/forge/backends/__init__.py`
- `src/forge/core/pipeline.py` and `builder.py`
- `src/forge/core/scheduler.py`
- `.github/workflows/ci.yml`

## Capability limits of the generator environment

The analysis script does not install FORGE, run DFT, or submit jobs. SVG rendering depends on local `dot`/`mmdc`.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_module_catalog(out_dir: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Python module catalog",
        "",
        "One-line descriptions from the module docstring first line, or empty if none. Based on implementation files, not directory names.",
        "",
        "| Module | Path | Category | Classes | Functions | Docstring |",
        "|--------|------|----------|--------:|----------:|-----------|",
    ]
    for m in sorted(analysis["modules"], key=lambda x: x["path"]):
        if not m["path"].endswith(".py"):
            continue
        doc = (m.get("docstring") or "").replace("|", "\\|")
        n_fn = len([f for f in m["functions"] if not f["is_nested"]])
        lines.append(
            f"| `{m['module']}` | `{m['path']}` | {m['category']} | {len(m['classes'])} | {n_fn} | {doc} |"
        )
    lines.append("")
    (out_dir / "module_catalog.md").write_text("\n".join(lines), encoding="utf-8")
