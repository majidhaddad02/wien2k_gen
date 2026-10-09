# Static Analysis Limitations

Python dynamic features prevent a complete runtime graph from AST alone.

- Production call sites extracted: 12084.
- Resolved with verified confidence: 3235.
- Unresolved: 8787.

## Intentionally not claimed

- A function with no static callers is **not** unused. It may be a CLI handler, packaging entry, lazy `__getattr__` export, or plugin.
- `forge.__init__.__getattr__` re-exports many names; those edges are not import statements in callers that use `from forge import detect`.
- `importlib.import_module` in `backends._load_backends` is recorded as a call, not as a static import of VASP/QE/CP2K (those modules also have real `from .wien2k import` for the primary backend).
- `self.method()` is resolved only when the enclosing class defines that method name.
- Subprocess argv is not fully evaluated; see `external_execution.json` for call-site previews.

## Syntax errors during parse

None. All scanned `.py` files parsed.

## Isolated modules

See `dependency_report.md`. Isolation is about internal imports only.

## Renderers

Graphviz `dot` is optional for `module_dependencies.svg`. `architecture_overview.svg` is always written as a small fallback drawing. mermaid-cli `mmdc` is optional. `render_status.json` records what this run actually rendered.
