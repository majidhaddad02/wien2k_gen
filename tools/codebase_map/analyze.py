#!/usr/bin/env python3
"""Static architecture analysis for FORGE / wien2k_gen.

Usage (from repository root):
    python tools/codebase_map/analyze.py
    python tools/codebase_map/analyze.py --root . --output-dir docs/architecture
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from codebase_map.extractor import (  # noqa: E402
    analyze_repository,
    classify_file,
    collect_python_files,
)
from codebase_map.reports import write_all_reports  # noqa: E402

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    ".tox",
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision(root: Path) -> dict[str, str]:
    info = {"commit": "unknown", "subject": "", "dirty": "unknown"}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
        ).strip()
        subject = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"], cwd=root, text=True, timeout=10
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, timeout=10
        ).strip()
        info["commit"] = commit
        info["subject"] = subject
        info["dirty"] = "dirty" if dirty else "clean"
    except Exception as e:
        info["error"] = str(e)
    return info


def walk_all_files(root: Path, exclude_dirs: set[str], exclude_suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in __import__("os").walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in exclude_suffixes:
                continue
            files.append(p)
    files.sort()
    return files


def file_inventory(
    root: Path,
    exclude_dirs: set[str],
    exclude_suffixes: set[str],
    summarize_dirs: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summarized: Counter[str] = Counter()
    for p in walk_all_files(root, exclude_dirs, exclude_suffixes):
        rel = p.relative_to(root).as_posix()
        skip = False
        for prefix in summarize_dirs:
            if rel.startswith(prefix.rstrip("/") + "/"):
                summarized[prefix] += 1
                skip = True
                break
        if skip:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "suffix": p.suffix,
                "category": classify_file(rel),
            }
        )
    for prefix, count in sorted(summarized.items()):
        rows.append(
            {
                "path": f"{prefix}/ (summarized {count} files)",
                "size_bytes": 0,
                "suffix": "",
                "category": "data",
            }
        )
    return rows


def scc(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    onstack: set[str] = set()
    result: list[list[str]] = []
    adj: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            result.append(sorted(comp))

    for n in nodes:
        if n not in indices:
            strongconnect(n)
    return result


def package_of(module: str) -> str:
    parts = module.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return module


def build_graphs(analysis: dict[str, Any]) -> dict[str, Any]:
    prod_modules = [
        m for m in analysis["modules"] if m["category"] == "production" and not m.get("syntax_error")
    ]
    test_modules = [
        m for m in analysis["modules"] if m["category"] == "test" and not m.get("syntax_error")
    ]
    prod_names = {m["module"] for m in prod_modules}

    unique_prod_edges: dict[tuple[str, str], dict[str, Any]] = {}
    unique_prod_edges_runtime: dict[tuple[str, str], dict[str, Any]] = {}
    for e in analysis["import_edges"]:
        if e["from"] in prod_names and e["to"] in prod_names:
            unique_prod_edges.setdefault((e["from"], e["to"]), e)
            if e.get("context") != "type_checking":
                unique_prod_edges_runtime.setdefault((e["from"], e["to"]), e)

    unique_test_edges: dict[tuple[str, str], dict[str, Any]] = {}
    for e in analysis["import_edges"]:
        src_cat = next((m["category"] for m in analysis["modules"] if m["module"] == e["from"]), "")
        if src_cat == "test":
            unique_test_edges.setdefault((e["from"], e["to"]), e)

    in_deg: Counter[str] = Counter()
    out_deg: Counter[str] = Counter()
    for a, b in unique_prod_edges_runtime:
        out_deg[a] += 1
        in_deg[b] += 1

    nodes = sorted(prod_names)
    components = scc(nodes, list(unique_prod_edges_runtime.keys()))
    cycles = [c for c in components if len(c) > 1]
    self_loops = [[a] for (a, b) in unique_prod_edges_runtime if a == b]
    cycle_edges = [
        unique_prod_edges_runtime[(a, b)]
        for (a, b) in unique_prod_edges_runtime
        if any(a in c and b in c for c in cycles)
    ]

    isolated = [m for m in nodes if in_deg[m] == 0 and out_deg[m] == 0]
    pkg_names = sorted({package_of(m) for m in nodes})
    pkg_in: Counter[str] = Counter()
    pkg_out: Counter[str] = Counter()
    for a, b in unique_prod_edges:
        pa = package_of(a)
        pb = package_of(b)
        if pa != pb:
            pkg_out[pa] += 1
            pkg_in[pb] += 1
    isolated_packages = [p for p in pkg_names if pkg_in[p] == 0 and pkg_out[p] == 0]

    third_party: Counter[str] = Counter()
    stdlib: Counter[str] = Counter()
    optional_like: list[dict[str, Any]] = []
    for m in prod_modules:
        for imp in m["imports"]:
            top = (imp.get("resolved") or imp.get("module") or "").split(".", 1)[0]
            if imp["category"] == "third_party":
                third_party[top] += 1
            elif imp["category"] == "stdlib":
                stdlib[top] += 1
            if "conditional" in imp["context"] or "function" in imp["context"]:
                optional_like.append(
                    {
                        "module": m["module"],
                        "import": imp.get("resolved") or imp.get("module"),
                        "context": imp["context"],
                        "lineno": imp["lineno"],
                        "category": imp["category"],
                    }
                )

    package_deps: Counter[tuple[str, str]] = Counter()
    for a, b in unique_prod_edges:
        pa = package_of(a)
        pb = package_of(b)
        if pa != pb:
            package_deps[(pa, pb)] += 1

    return {
        "production_modules": [m["module"] for m in prod_modules],
        "test_modules": [m["module"] for m in test_modules],
        "unique_production_edges": [
            {"from": a, "to": b, **{k: v for k, v in e.items() if k not in {"from", "to"}}}
            for (a, b), e in sorted(unique_prod_edges.items())
        ],
        "unique_test_edges": [{"from": a, "to": b} for (a, b) in sorted(unique_test_edges.keys())],
        "in_degree": dict(in_deg),
        "out_degree": dict(out_deg),
        "cycles": cycles,
        "cycle_edges": [
            {"from": e["from"], "to": e["to"], "context": e.get("context"), "path": e.get("path"), "lineno": e.get("lineno")}
            for e in cycle_edges
        ],
        "self_loops": self_loops,
        "isolated": isolated,
        "isolated_packages": isolated_packages,
        "third_party": dict(third_party.most_common()),
        "stdlib": dict(stdlib.most_common()),
        "lazy_or_conditional_imports": optional_like,
        "package_deps": [
            {"from": a, "to": b, "count": n} for (a, b), n in sorted(package_deps.items())
        ],
        "hubs_in": in_deg.most_common(15),
        "hubs_out": out_deg.most_common(15),
    }


def parse_register_command_literals(root: Path, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    import ast

    mapping: list[dict[str, Any]] = []
    for m in analysis["modules"]:
        if not m["module"].startswith("forge.cli_commands."):
            continue
        if m["module"] in {"forge.cli_commands", "forge.cli_commands._utils", "forge.cli_commands.base"}:
            continue
        path = root / m["path"]
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=m["path"])
        except SyntaxError:
            continue
        cmd_name: Optional[str] = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "register_command" and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                        cmd_name = arg0.value
        handle_imports = [
            imp.get("resolved")
            for imp in m["imports"]
            if imp["category"] == "internal" and imp.get("resolved")
        ]
        mapping.append(
            {
                "command": cmd_name or Path(m["path"]).stem.replace("_", "-"),
                "module": m["module"],
                "path": m["path"],
                "handler": f"{m['module']}.handle",
                "internal_imports": sorted(set(handle_imports)),
            }
        )
    mapping.sort(key=lambda r: r["command"] or "")
    return mapping


def collect_external_exec(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for m in analysis["modules"]:
        if m["category"] not in {"production", "script"}:
            continue
        for e in m["external_exec"]:
            rows.append({"module": m["module"], "path": m["path"], **e})
    return rows


def test_module_mapping(analysis: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for m in analysis["modules"]:
        if m["category"] != "test":
            continue
        targets = sorted(
            {
                (imp.get("resolved") or "")
                for imp in m["imports"]
                if imp["category"] == "internal" and (imp.get("resolved") or "").startswith("forge")
            }
        )
        rows.append(
            {
                "test_module": m["module"],
                "test_path": m["path"],
                "imported_forge_modules": ";".join(targets),
                "n_targets": str(len(targets)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def try_render_dot(dot_path: Path, svg_path: Path) -> dict[str, Any]:
    from shutil import which

    dot = which("dot")
    if not dot:
        return {"rendered": False, "reason": "graphviz 'dot' not found on PATH"}
    try:
        subprocess.check_call([dot, "-Tsvg", str(dot_path), "-o", str(svg_path)], timeout=60)
        return {"rendered": True, "tool": dot, "output": str(svg_path)}
    except Exception as e:
        return {"rendered": False, "reason": str(e)}


def try_render_mermaid(mmd_path: Path, svg_path: Path) -> dict[str, Any]:
    from shutil import which

    mmdc = which("mmdc")
    if not mmdc:
        return {"rendered": False, "reason": "mermaid-cli 'mmdc' not found on PATH"}
    try:
        subprocess.check_call([mmdc, "-i", str(mmd_path), "-o", str(svg_path)], timeout=60)
        return {"rendered": True, "tool": mmdc, "output": str(svg_path)}
    except Exception as e:
        return {"rendered": False, "reason": str(e)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Static architecture analysis for FORGE")
    parser.add_argument("--root", type=Path, default=None, help="Repository root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated reports (default: <root>/docs/architecture)",
    )
    parser.add_argument("--exclude-dir", action="append", default=[], help="Extra dir names to exclude")
    args = parser.parse_args(argv)

    tool_dir = Path(__file__).resolve().parent
    cfg = load_config(tool_dir / "config.json")
    root = (args.root or Path.cwd()).resolve()
    if not (root / "src" / "forge").is_dir() and (root / "wien2k_gen" / "src" / "forge").is_dir():
        root = (root / "wien2k_gen").resolve()
    out_dir = (args.output_dir or (root / "docs" / "architecture")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_dirs = set(cfg.get("exclude_dir_names") or []) | DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)
    exclude_suffixes = set(cfg.get("exclude_suffixes") or [".pyc", ".pyo", ".so", ".whl"])
    summarize_dirs = list(cfg.get("summarize_dirs") or [])

    git = git_revision(root)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    analysis = analyze_repository(root, exclude_dirs)
    graphs = build_graphs(analysis)
    inventory = file_inventory(root, exclude_dirs, exclude_suffixes, summarize_dirs)
    cli_map = parse_register_command_literals(root, analysis)
    external = collect_external_exec(analysis)
    tests_map = test_module_mapping(analysis)

    meta = {
        "timestamp_utc": timestamp,
        "git": git,
        "root": str(root),
        "python": sys.version.split()[0],
        "n_python_files": len(collect_python_files(root, exclude_dirs)),
        "n_inventory_rows": len(inventory),
        "n_production_modules": len(graphs["production_modules"]),
        "n_test_modules": len(graphs["test_modules"]),
        "n_production_import_edges": len(graphs["unique_production_edges"]),
        "n_cycles": len(graphs["cycles"]),
    }

    (out_dir / "analysis_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "file_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "file_inventory.csv", inventory, ["path", "size_bytes", "suffix", "category"])
    (out_dir / "module_dependencies.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "production_edges": graphs["unique_production_edges"],
                "package_deps": graphs["package_deps"],
                "cycles": graphs["cycles"],
                "cycle_edges": graphs.get("cycle_edges", []),
                "isolated": graphs["isolated"],
                "isolated_packages": graphs.get("isolated_packages", []),
                "hubs_in": graphs["hubs_in"],
                "hubs_out": graphs["hubs_out"],
                "third_party": graphs["third_party"],
                "stdlib": graphs["stdlib"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "call_graph.json").write_text(
        json.dumps(
            {
                "note": (
                    "Calls are statically extracted. Only Name() and some Attribute() "
                    "forms are resolved. Dynamic dispatch is unresolved."
                ),
                "modules": [
                    {
                        "module": m["module"],
                        "path": m["path"],
                        "category": m["category"],
                        "calls": m["calls"],
                    }
                    for m in analysis["modules"]
                    if m["category"] == "production"
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "cli_command_map.json").write_text(json.dumps(cli_map, indent=2) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "cli_command_map.csv",
        [{"command": r["command"], "module": r["module"], "path": r["path"], "handler": r["handler"]} for r in cli_map],
    )
    (out_dir / "external_execution.json").write_text(json.dumps(external, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "test_module_mapping.csv", tests_map)
    (out_dir / "modules.json").write_text(json.dumps(analysis["modules"], indent=2) + "\n", encoding="utf-8")

    render_status = write_all_reports(
        root=root,
        out_dir=out_dir,
        meta=meta,
        inventory=inventory,
        analysis=analysis,
        graphs=graphs,
        cli_map=cli_map,
        external=external,
        tests_map=tests_map,
    )

    fallback_overview = out_dir / "architecture_overview.svg"
    dot_svg = try_render_dot(out_dir / "module_dependencies.dot", out_dir / "module_dependencies.svg")
    mermaid_svg = try_render_mermaid(out_dir / "architecture_overview.mmd", out_dir / "architecture_overview.mmd.svg")
    render_status["module_dependencies_svg"] = dot_svg
    if mermaid_svg.get("rendered"):
        render_status["architecture_overview_svg"] = mermaid_svg
    elif fallback_overview.is_file():
        render_status["architecture_overview_svg"] = {
            "rendered": True,
            "tool": "stdlib_svg_fallback",
            "output": str(fallback_overview),
            "mermaid_cli": mermaid_svg,
        }
    else:
        render_status["architecture_overview_svg"] = mermaid_svg
    (out_dir / "render_status.json").write_text(json.dumps(render_status, indent=2) + "\n", encoding="utf-8")

    print(f"Analyzed {root}")
    print(f"Commit {git.get('commit')} ({git.get('dirty')})")
    print(f"Wrote reports to {out_dir}")
    print(f"Production modules: {meta['n_production_modules']}")
    print(f"Import edges: {meta['n_production_import_edges']}")
    print(f"Cycles: {meta['n_cycles']}")
    if not dot_svg.get("rendered"):
        print(f"DOT SVG not rendered: {dot_svg.get('reason')}")
    overview_status = render_status.get("architecture_overview_svg") or {}
    if not overview_status.get("rendered"):
        print(f"Overview SVG not rendered: {overview_status.get('reason')}")
    elif overview_status.get("tool") == "stdlib_svg_fallback":
        print("Overview SVG: stdlib fallback (mermaid-cli not on PATH)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
