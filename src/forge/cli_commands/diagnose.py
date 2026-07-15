from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("diagnose", help="Diagnose SCF convergence issues")
    p.add_argument("case", type=str, nargs="?", default=None, help="Case name to diagnose")
    p.add_argument("--log", type=str, default=None, help="Path to .scf or .output file")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:  # noqa: C901
    console = get_console()
    import re as _re

    scf_path = None
    if args.log:
        scf_path = Path(args.log)
    elif args.case:
        scf_path = Path(f"{args.case}.scf")
        if not scf_path.exists():
            scf_path = Path.cwd() / f"{args.case}.scf"
    else:
        matches = sorted(Path(".").glob("*.scf*"), key=lambda p: p.stat().st_mtime, reverse=True)
        scf_path = matches[0] if matches else None

    if not scf_path or not scf_path.exists():
        console.print("[red]No SCF output file found.[/red]")
        return {"error": "No SCF file found"}

    content = scf_path.read_text(encoding="utf-8", errors="replace")

    energy_matches = _re.findall(r":ENE\s*:\s*.*?(-?\d+\.\d+)", content)
    charge_matches = _re.findall(r":DIS\s*:\s*.*?(\d+\.\d+)", content)

    energies = [float(e) for e in energy_matches]
    charges = [float(c) for c in charge_matches]
    converged = "charge convergence" in content.lower() or "energy convergence" in content.lower()

    table = Table(title=f"SCF Diagnostics: {scf_path.name}", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Cycles completed", str(len(energies)))
    table.add_row("Final energy", f"{energies[-1]:.6f} Ry" if energies else "N/A")
    table.add_row("Final charge distance", f"{charges[-1]:.6f}" if charges else "N/A")
    table.add_row("Converged", f"[{'green' if converged else 'red'}]{converged}[/]")

    if len(charges) >= 3:
        ratios = []
        for i in range(1, len(charges)):
            if charges[i - 1] > 1e-12:
                ratios.append(charges[i] / charges[i - 1])
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        table.add_row("Avg charge ratio", f"{avg_ratio:.3f}")
        if avg_ratio > 1.01:
            table.add_row("Diagnosis", "[red]Divergent — reduce mixing beta or enable PRATT[/red]")
        elif avg_ratio < 0.95:
            table.add_row("Diagnosis", "[green]Converging monotonically[/green]")
        else:
            table.add_row("Diagnosis", "[yellow]Oscillatory — reduce mixing[/yellow]")

    if len(energies) >= 2:
        deltas = [abs(energies[i] - energies[i - 1]) for i in range(1, len(energies))]
        sign_changes = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0)
        oscillation_pct = sign_changes / max(1, len(deltas) - 1) * 100 if len(deltas) > 1 else 0
        table.add_row("Energy oscillations", f"{oscillation_pct:.1f}%")

    console.print(table)

    try:
        from ..optimizer.monitor import diagnose_charge_sloshing_root_cause
        diag = diagnose_charge_sloshing_root_cause(content, case_name=args.case or scf_path.stem)
        if diag["root_cause"] != "none":
            panel_title = "[red bold]Charge Sloshing Detected"
            cause_labels = {
                "metallic": "\u0641\u0644\u0632\u06cc \u2014 \u0633\u0637\u062d \u0641\u0631\u0645\u06cc \u067e\u06cc\u0686\u06cc\u062f\u0647 / Metallic \u2014 complex Fermi surface",
                "symmetry_breaking": "\u0634\u06a9\u0633\u062a \u062a\u0642\u0627\u0631\u0646 / Symmetry breaking",
                "core_overlap": "\u0647\u0645\u200c\u067e\u0648\u0634\u0627\u0646\u06cc \u0647\u0633\u062a\u0647 \u2014 RMT \u0646\u0627\u0645\u0646\u0627\u0633\u0628 / Core overlap \u2014 check RMT",
                "mixing_too_aggressive": "\u0646\u0631\u062e \u0645\u062e\u0644\u0648\u0637\u200c\u0633\u0627\u0632\u06cc \u0628\u0627\u0644\u0627 / Mixing too aggressive",
            }
            cause_label = cause_labels.get(diag["root_cause"], diag["root_cause"])
            body = f"[bold]Root Cause:[/] {cause_label}\n"
            body += f"[dim]Confidence: {diag['confidence']:.0%}[/]\n\n"
            for i, act in enumerate(diag["actions"], 1):
                body += f"  {i}. [cyan]{act['action']}[/] \u2014 {act['reason']}\n"
            console.print(Panel(body.strip(), title=panel_title, border_style="red"))
    except Exception:
        pass

    has_qtlb = any(p in content.lower() for p in ("qtl-b",))
    has_crash = any(p in content.lower() for p in ("lapw crashed", "segmentation fault"))
    has_not_conv = "not converged" in content.lower()

    if has_qtlb:
        qtlb_body = ""
        if "rkmax" in content.lower() or "kmax" in content.lower():
            qtlb_body += "\u2022 Reduce RKMAX by 0.5\u20131.0\n"
        if "overlap" in content.lower():
            qtlb_body += "\u2022 Reduce RMT values or check sphere overlap\n"
        if "linearization" in content.lower() or "ene" in content.lower():
            qtlb_body += "\u2022 Add more linearization energies in case.in1\n"
        qtlb_body += "\u2022 Increase GMAX to 2.5\u00d7RKMAX\n"
        qtlb_body += "\u2022 Check init_lapw \u2014b (non-default linearization)"
        console.print(Panel(qtlb_body, title="[red bold]QTL-B Error \u2014 Root Cause Analysis", border_style="red"))

    if has_crash:
        console.print(Panel(
            "\u2022 Check MPI stack (mpirun/srun) and memory limits\n"
            "\u2022 Look for OOM (Out of Memory) in SLURM output\n"
            "\u2022 Verify .machines file format matches WIEN2k version\n"
            "\u2022 Try running with fewer MPI ranks first",
            title="[red bold]LAPWx Crash Detected", border_style="red"
        ))

    try:
        from ..optimizer.convergence import detect_scf_divergence
        divergence = detect_scf_divergence(content, energy_values=energies if energies else None)
        if divergence["divergent"]:
            console.print(Panel(
                f"[bold]Type:[/] {divergence['divergence_type']}\n"
                f"[bold]Severity:[/] {divergence['severity']:.0%}\n\n"
                f"{divergence['recommended_action']}\n\n"
                f"[dim]Auto fix: beta={divergence['auto_mixing_params']['beta']}, "
                f"pratt={divergence['auto_mixing_params']['pratt_cycles']}, "
                f"msr1a={divergence['auto_mixing_params']['msr1a']}[/]",
                title="[red bold]SCF Divergence Analysis", border_style="red"
            ))
    except Exception:
        pass

    if has_not_conv and not converged:
        console.print(Panel(
            "1. Check if charge sloshing (see above)\n"
            "2. Reduce mixing beta to 0.05 and enable PRATT\n"
            "3. Increase NSTEPS in case.in2\n"
            "4. For metals: add TEMP 0.002 in case.in2 (Fermi smearing)\n"
            "5. Run [bold]forge optimize --simulated[/] to auto-tune RKMAX/KPPRA",
            title="[yellow bold]SCF Not Converged \u2014 Action Plan", border_style="yellow"
        ))

    if not any([has_qtlb, has_crash, has_not_conv, converged, diag.get("root_cause", "") != "none" if 'diag' in dir() else False]):
        console.print("[green]No critical issues detected. SCF appears healthy.[/green]")

    return {
        "converged": converged,
        "cycles": len(energies),
        "error_detected": has_qtlb or has_crash,
        "final_energy": energies[-1] if energies else 0.0,
    }


register_command("diagnose", handle)
