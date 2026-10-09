"""Write all human and diagram artifacts under docs/architecture/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .gen_graphs import (
    write_architecture_overview_mmd,
    write_architecture_overview_svg,
    write_callgraph_focus,
    write_cli_flow_mmd,
    write_execution_lifecycle_mmd,
    write_module_dependencies_dot,
    write_module_dependencies_mmd,
    write_test_ci_mmd,
)
from .gen_inventory import write_file_tree
from .gen_markdown import (
    write_architecture_overview_md,
    write_architecture_readme,
    write_cli_command_map_md,
    write_dependency_report,
    write_findings,
    write_limitations,
    write_module_catalog,
    write_testing_report,
    write_workflow_analysis,
)


def write_all_reports(
    root: Path,
    out_dir: Path,
    meta: dict[str, Any],
    inventory: list[dict[str, Any]],
    analysis: dict[str, Any],
    graphs: dict[str, Any],
    cli_map: list[dict[str, Any]],
    external: list[dict[str, Any]],
    tests_map: list[dict[str, str]],
) -> dict[str, Any]:
    write_file_tree(out_dir, inventory, meta)
    write_module_dependencies_dot(out_dir, graphs)
    write_module_dependencies_mmd(out_dir, graphs)
    write_architecture_overview_mmd(out_dir)
    write_architecture_overview_svg(out_dir)
    write_architecture_overview_md(out_dir, meta)
    write_cli_flow_mmd(out_dir, cli_map)
    write_execution_lifecycle_mmd(out_dir)
    write_test_ci_mmd(out_dir)
    write_callgraph_focus(out_dir, analysis)
    write_dependency_report(out_dir, graphs, meta)
    write_cli_command_map_md(out_dir, cli_map)
    write_workflow_analysis(out_dir)
    write_testing_report(out_dir, tests_map)
    write_limitations(out_dir, analysis, graphs)
    write_findings(out_dir, graphs, meta, cli_map)
    write_module_catalog(out_dir, analysis)
    write_architecture_readme(
        out_dir,
        meta,
        render_note=(
            "`architecture_overview.svg` is a small fallback drawing from the analyzer. "
            "`module_dependencies.svg` requires Graphviz `dot` on PATH."
        ),
    )
    return {"markdown_and_diagram_sources": True}
