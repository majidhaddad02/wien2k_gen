from __future__ import annotations

import argparse
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("optimize", help="Bayesian auto-tuning of RKMAX, k-points, mixing")
    p.add_argument("--case", type=str, default="case", help="WIEN2k case name")
    p.add_argument("--budget", type=int, default=10, help="Max DFT runs (default: 10)")
    p.add_argument("--target", type=str, default="energy_convergence", help="Target metric")
    p.add_argument("--strategy", type=str, choices=["gp_ei", "bohb"], default="gp_ei",
                   help="Optimisation strategy: gp_ei (GP+EI, default) or bohb (BOHB+TPE)")
    p.add_argument("--simulated", action="store_true", help="Use simulated objective for testing")
    p.add_argument("--verbose", "-v", action="store_true", help="Show iteration details")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    try:
        from ..optimizer.bayesian_tuner import optimize_convergence_parameters
    except ImportError as e:
        return {"error": f"Bayesian tuner not available: {e}"}

    console.print(Panel(
        f"[cyan]Bayesian Parameter Optimization[/]\n"
        f"Case: [bold]{args.case}[/] | Budget: [bold]{args.budget} runs[/] | "
        f"Target: [bold]{args.target}[/] | Strategy: [bold]{args.strategy}[/]",
        border_style="magenta",
    ))

    result = optimize_convergence_parameters(
        case_name=args.case,
        budget=args.budget,
        use_simulated=args.simulated,
        verbose=args.verbose,
        strategy=args.strategy,
    )

    table = Table(title="Optimization Results", border_style="green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Optimal Value", style="green bold")
    table.add_column("Uncertainty", style="dim")
    table.add_row("RKMAX", f"{result.best_rkmax:.1f}", f"\u00b1{result.uncertainty_rkmax:.2f}")
    table.add_row("KPPRA", str(result.best_kppra), "-")
    table.add_row("Mixing \u03b2", f"{result.best_mixing:.4f}", f"\u00b1{result.uncertainty_mixing:.4f}")
    table.add_row("\u0394E (best)", f"{result.best_energy:.6f} Ry", "")
    table.add_row("Iterations", str(result.iterations), "")
    table.add_row("Converged", f"[{'green' if result.convergence_achieved else 'red'}]{result.convergence_achieved}[/]", "")
    console.print(table)

    if result.observations:
        hist_table = Table(title="Iteration History", border_style="dim")
        hist_table.add_column("#", style="dim")
        hist_table.add_column("RKMAX")
        hist_table.add_column("KPPRA")
        hist_table.add_column("Mixing")
        hist_table.add_column("\u0394E (Ry)")
        for obs in result.observations[-8:]:
            hist_table.add_row(
                str(obs["iteration"]), f"{obs['rkmax']:.1f}",
                str(obs["kppra"]), f"{obs['mixing']:.4f}",
                f"{obs['delta_energy']:.6f}",
            )
        console.print(hist_table)

    return {
        "rkmax": result.best_rkmax,
        "kppra": result.best_kppra,
        "mixing": result.best_mixing,
        "delta_energy": result.best_energy,
        "converged": result.convergence_achieved,
    }


register_command("optimize", handle)
