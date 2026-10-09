# FORGE architecture map

Static reconstruction of `wien2k_gen` / FORGE. Generated `2026-10-09T13:54:34Z` from git `c3db6e711acfbab1abe84b29f98d909374d0ab9b` (dirty).

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

`architecture_overview.svg` is a small fallback drawing from the analyzer. `module_dependencies.svg` requires Graphviz `dot` on PATH.

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
