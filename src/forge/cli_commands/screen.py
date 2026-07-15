from __future__ import annotations

import argparse
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("screen", help="High-throughput screening via Materials Project")
    p.add_argument("--formula", type=str, default=None, help="Chemical formula (e.g. ABO3, LiFePO4)")
    p.add_argument("--elements", type=str, default=None, help="Comma-separated elements (e.g. Ti,O,Zr)")
    p.add_argument("--mp-id", type=str, default=None, help="Single Materials Project ID")
    p.add_argument("--max", type=int, default=50, help="Max materials (default: 50)")
    p.add_argument("--api-key", type=str, default=None, help="Materials Project API key")
    p.add_argument("--output", type=str, default=None, help="Output directory")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    try:
        from ..core.materials_project import screen_materials
    except ImportError as e:
        return {"error": f"Materials Project client not available: {e}"}

    elements_list = args.elements.split(",") if args.elements else None

    console.print(Panel(
        f"[cyan]Materials Project Screening[/]\n"
        f"Query: [bold]{args.formula or args.elements or args.mp_id}[/] | "
        f"Max: [bold]{getattr(args, 'max', 50)}[/]",
        border_style="blue",
    ))

    result = screen_materials(
        formula=args.formula,
        elements=elements_list,
        mp_id=args.mp_id,
        max_results=getattr(args, 'max', 50),
        api_key=args.api_key,
        output_dir=args.output,
    )

    table = Table(title="Screening Results", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total found", str(result.total_found))
    table.add_row("Downloaded", str(result.downloaded))
    table.add_row("Converted to .struct", str(result.converted))
    table.add_row("Errors", f"[{'red' if result.errors else 'green'}]{len(result.errors)}[/]")
    console.print(table)

    if result.materials[:10]:
        mat_table = Table(title="Materials (first 10)", border_style="dim")
        mat_table.add_column("MP-ID", style="cyan")
        mat_table.add_column("Formula", style="bold")
        mat_table.add_column("Band Gap (eV)")
        mat_table.add_column("E_above_hull (eV/at)")
        for mat in result.materials[:10]:
            gap_color = "green" if mat.band_gap > 0 else "red"
            mat_table.add_row(
                mat.mp_id, mat.formula,
                f"[{gap_color}]{mat.band_gap:.2f}[/{gap_color}]",
                f"{mat.energy_above_hull:.3f}",
            )
        console.print(mat_table)

    if result.errors:
        for err in result.errors[:5]:
            console.print(f"[dim]  - {err}[/dim]")

    return {"downloaded": result.downloaded, "converted": result.converted, "errors": len(result.errors)}


register_command("screen", handle)
