from __future__ import annotations

import argparse
import json
from typing import Any

from rich.panel import Panel

from ..config import AppConfig
from ..ui.analysis import generate_report, parse_scf_log
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze", help="Parse SCF logs & generate performance reports")
    p.add_argument("--log", type=str, required=True, help="Path to SCF/output log file")
    p.add_argument("--code", type=str, choices=["wien2k", "vasp", "qe"], default=None, help="Force DFT code parser")
    p.add_argument("--export", type=str, default=None, help="Export analysis report to JSON")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    parsed = parse_scf_log(args.log, code_hint=args.code)
    report = generate_report(parsed, scaling_data=None, include_recommendations=True)

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

    if getattr(args, 'json_output', False):
        return report.to_dict()

    console.print(Panel(
        f"[green]✓ Analysis complete.[/]\n"
        f"Code: [cyan]{parsed.get('code', 'unknown')}[/] | "
        f"Converged: [{'green' if parsed.get('converged') else 'red'}]{parsed.get('converged')}[/]\n"
        f"Total Cycles: [bold]{parsed.get('total_cycles', 0)}[/]",
        border_style="green"
    ))
    return {"code": parsed.get("code"), "converged": parsed.get("converged"), "cycles": parsed.get("total_cycles")}


register_command("analyze", handle)
