from __future__ import annotations

import argparse
from typing import Any

from rich.table import Table

from ..config import AppConfig
from ..utils.diagnostic import export_diagnostics_json, run_diagnostics
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("diagnostics", help="Collect system & environment health report")
    p.add_argument("--export", type=str, default=None, help="Save diagnostic report to path")
    p.add_argument("--full", action="store_true", help="Include verbose library & interconnect checks")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    report = run_diagnostics()
    if args.export:
        export_diagnostics_json(report, args.export)

    if getattr(args, 'json_output', False):
        return dict(report)

    table = Table(title="System Diagnostics Summary", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Hostname", report.get("hostname", "unknown"))
    table.add_row("OS", f"{report.get('os_info', {}).get('os', 'unknown')} {report.get('os_info', {}).get('release', '')}")
    table.add_row("Python", report.get('python_env', {}).get('python_version', 'unknown'))
    table.add_row("Warnings", str(len(report.get("warnings", []))), style="yellow" if report.get("warnings") else "green")
    table.add_row("Critical Errors", str(len(report.get("critical_errors", []))), style="red" if report.get("critical_errors") else "green")

    console.print(table)
    return {"hostname": report.get("hostname"), "warnings": len(report.get("warnings", [])), "errors": len(report.get("critical_errors", []))}


register_command("diagnostics", handle)
