from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze-bands", help="Extract band structure and DOS data from WIEN2k output")
    p.add_argument("--case", required=True, help="Case name or path")
    p.add_argument("--output", help="Path to export JSON file")
    p.add_argument("--dos", action="store_true", help="Also parse DOS data")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    try:
        from ..core.electronic_structure import compute_band_gap, parse_band_structure, parse_dos
    except ImportError as e:
        return {"error": f"Electronic structure module dependencies not available: {e}"}

    case_path = Path(args.case)
    if case_path.is_dir():
        base_path = str(case_path)
        case_name = args.case
    else:
        base_path = str(case_path.parent) if case_path.parent != Path() else "."
        case_name = case_path.stem

    console.print(f"[cyan]Analyzing bands for case: [bold]{case_name}[/bold] in {base_path}[/cyan]")

    band_data = parse_band_structure(case_name, base_path)
    gap_info = compute_band_gap(band_data)

    table = Table(title="Band Structure Summary", border_style="cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("k-points", str(band_data.get("nkpt", 0)))
    table.add_row("Bands", str(band_data.get("nbnd", 0)))
    table.add_row("Spin-polarized", "Yes" if band_data.get("nspin", 1) > 1 else "No")
    table.add_row("Fermi energy", f"{band_data.get('fermi', 0):.4f} eV")
    table.add_row("Band gap", f"{gap_info.get('gap_ev', 0):.4f} eV")
    table.add_row("Direct gap", f"{gap_info.get('direct_gap_ev', 0):.4f} eV")
    console.print(table)

    result = {"case": case_name, "nkpt": band_data.get("nkpt"), "nbnd": band_data.get("nbnd"), "fermi_ev": band_data.get("fermi"), "gap_ev": gap_info.get("gap_ev")}

    if args.dos:
        dos_data = parse_dos(case_name, base_path)
        result["dos_n_energy"] = len(dos_data.get("energies", []))
        console.print(f"[green]DOS: {result['dos_n_energy']} energy points parsed.[/green]")

    if args.output:
        export_data = {
            "band_structure": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in band_data.items() if k not in ("k_points", "eigenvalues")},
            "gap": gap_info,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        console.print(f"[green]Exported to {args.output}[/green]")

    return result


register_command("analyze-bands", handle)
