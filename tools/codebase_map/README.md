# Codebase map toolkit

AST-only static analysis for the FORGE (`wien2k_gen`) repository. It does not import `forge`, run DFT jobs, or submit scheduler work.

## Requirements

Python 3.9+ standard library. Optional:

- Graphviz `dot` for `module_dependencies.svg`
- mermaid-cli `mmdc` for mermaid SVG (optional; a small `architecture_overview.svg` fallback is always written)

Missing Graphviz is recorded in `docs/architecture/render_status.json`. The run still succeeds.

## Commands

From the repository root:

```
python tools/codebase_map/analyze.py
```

Explicit paths:

```
python tools/codebase_map/analyze.py --root . --output-dir docs/architecture
```

Exclude extra directory names:

```
python tools/codebase_map/analyze.py --exclude-dir tmp
```

Optional rendering after a successful run:

```
dot -Tsvg docs/architecture/module_dependencies.dot -o docs/architecture/module_dependencies.svg
mmdc -i docs/architecture/architecture_overview.mmd -o docs/architecture/architecture_overview.svg
```

## Tests for this toolkit

```
python -m pytest tests/codebase_map -q
```

If pytest is not installed:

```
python tests/codebase_map/test_extractor.py -q
```

These tests parse tiny synthetic trees. They do not execute application modules.

## Layout

| File | Role |
|------|------|
| `analyze.py` | CLI entry |
| `extractor.py` | AST inventory, imports, calls, subprocess sites |
| `stdlib_modules.py` | Stdlib name set for classification |
| `reports.py` | Orchestrates writers |
| `gen_inventory.py` | File tree |
| `gen_graphs.py` | DOT / Mermaid |
| `gen_markdown.py` | Narrative reports |
| `config.json` | Default exclusions |

## Behaviour

- Repository-relative paths in all JSON/CSV/Markdown.
- Syntax errors are stored on the module record; analysis continues.
- Unresolved imports and calls are reported, not raised.
- Output is deterministic given the same tree (JSON keys sorted where practical).
- Git commit hash is recorded when `git` works.
