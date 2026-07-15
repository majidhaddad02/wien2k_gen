from __future__ import annotations

import argparse
from typing import Any, cast

from rich.panel import Panel

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("converge", help="Run automated convergence testing (k-points, RKmax)")
    p.add_argument("--case", required=True, help="Case name or path")
    p.add_argument("--mode", choices=("kpoints", "rkmax", "both"), default="both", help="Convergence mode")
    p.add_argument("--tolerance", type=float, default=0.001, help="Tolerance in Ry")
    p.add_argument("--kpoints", default="2,2,2 4,4,4 6,6,6 8,8,8 10,10,10", help="K-point grids to test")
    p.add_argument("--rkmax", default="5,6,7,8,9,10", help="RKmax values to test")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    try:
        from ..optimizer.convergence import (
            find_converged_parameters,
            generate_convergence_report,
            run_kpoint_convergence,
            run_rkmax_convergence,
        )
    except ImportError as e:
        return {"error": f"Convergence module dependencies not available: {e}"}

    console.print(Panel(
        f"[cyan]Convergence Study[/]\nCase: [bold]{args.case}[/]\nMode: [bold]{args.mode}[/]\nTolerance: [bold]{args.tolerance} Ry[/]",
        border_style="blue"
    ))

    wien2k_cmd = {"run_lapw": "run_lapw", "lapw0": "lapw0", "lapw1": "lapw1", "lapw2": "lapw2"}
    results = {}

    if args.mode in ("kpoints", "both"):
        kpoint_grids = [cast(tuple[int, int, int], tuple(int(x) for x in g.split(","))) for g in args.kpoints.split()]
        console.print(f"[bold]Running k-point convergence with grids: {kpoint_grids}...[/bold]")
        results["kpoints"] = run_kpoint_convergence(args.case, kpoint_grids, wien2k_cmd)

    if args.mode in ("rkmax", "both"):
        rkmax_values = [float(x) for x in args.rkmax.split(",")]
        console.print(f"[bold]Running RKmax convergence with values: {rkmax_values}...[/bold]")
        results["rkmax"] = run_rkmax_convergence(args.case, rkmax_values, wien2k_cmd)

    for key, data in results.items():
        converged = find_converged_parameters(data, tolerance=args.tolerance * 1000.0)
        console.print(f"[green]{key}: converged at {converged.get('converged_value')}[/green]")

    report = generate_convergence_report({"results": list(results.values())})
    console.print(Panel(report, title="Convergence Report", border_style="green"))

    return {"status": "completed", "results": {k: v for k, v in results.items()}}


register_command("converge", handle)
