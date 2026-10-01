from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def _nmat_nkpt_from_struct(case_path: Path) -> tuple[int, int]:
    """Derive nmat/nkpt from a WIEN2k case.struct and sibling inputs.

    rkmax comes from ``{stem}.in1`` (default 7.0). nkpt comes from
    ``{stem}.klist*`` — the same suffix glob used by
    ``detect_problem_size()`` (``*.klist*``), scoped to this case stem
    so ``case.klist`` and ``case.klist_band`` both match.
    """
    from ..core.case_parser import CaseFileParser

    struct = CaseFileParser.parse_struct(case_path)
    rmts = struct.get("rmts") or []
    volume = float(struct.get("volume_bohr3") or 0.0)

    rkmax = 7.0
    in1_candidates = list(case_path.parent.glob(f"{case_path.stem}.in1"))
    if in1_candidates:
        in1_data = CaseFileParser.parse_in1(in1_candidates[0])
        rkmax = float(in1_data.get("rkmax", 7.0) or 7.0)

    if not rmts:
        real_nmat = 0
    else:
        real_nmat = int(CaseFileParser.estimate_nmat(rkmax, rmts, volume) or 0)

    klist_candidates = list(case_path.parent.glob(f"{case_path.stem}.klist*"))
    real_nkpt = 1
    if klist_candidates:
        klist_data = CaseFileParser.parse_klist(klist_candidates[0])
        real_nkpt = int(klist_data.get("kpoints", 1) or 1)

    return real_nmat, real_nkpt


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("history", help="Query and export execution history database")
    p.add_argument("--list", action="store_true", help="List recent history")
    p.add_argument("--show", help="Show details for a specific run_id")
    p.add_argument("--similar-to", help="Find similar runs to a given case")
    p.add_argument("--limit", type=int, default=10, help="Maximum records to return")
    p.add_argument("--export", help="Export history to CSV/JSON for ML training (provide output filename)")
    p.add_argument("--format", choices=["csv", "json"], default="csv",
                   help="Export format (default: csv)")
    p.add_argument("--backend", help="Filter export to specific backend (wien2k, qe, vasp, ...)")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:  # noqa: C901
    console = get_console()
    try:
        from ..optimizer.history import ExecutionHistory
    except ImportError as e:
        return {"error": f"History module dependencies not available: {e}"}

    with ExecutionHistory() as history:
        if args.export:
            output = None if args.export.lower() == "auto" else Path(args.export)
            out_path = history.export(
                output_path=output,
                fmt=args.format,
                backend=args.backend,
            )
            console.print(f"[green]Exported {args.format.upper()} → {out_path}[/green]")
            return {"exported": str(out_path)}

        if args.show:
            records = history.query({"run_id": args.show})
            if not records:
                console.print(f"[red]No record found for run_id: {args.show}[/red]")
                return {"found": False, "run_id": args.show}
            rec = records[0]
            table = Table(title=f"Run {args.show}", border_style="cyan")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Backend", rec.backend)
            table.add_row("Mode", rec.mode)
            table.add_row("Cores", str(rec.total_cores))
            table.add_row("Walltime", f"{rec.walltime_sec:.1f}s")
            table.add_row("Success", str(rec.success))
            table.add_row("NMAT", str(rec.nmat))
            table.add_row("k-points", str(rec.nkpt))
            console.print(table)
            return rec.to_dict() if hasattr(rec, 'to_dict') else {"run_id": rec.run_id}

        if args.similar_to:
            case_path = Path(args.similar_to)
            if case_path.exists() and case_path.suffix == ".struct":
                try:
                    real_nmat, real_nkpt = _nmat_nkpt_from_struct(case_path)
                except Exception:
                    real_nmat, real_nkpt = 0, 1
                if real_nmat == 0:
                    console.print(
                        "[yellow]Could not fully parse case file; "
                        "falling back to recent history.[/yellow]"
                    )
                    recs = history.query(limit=args.limit)
                else:
                    recs = history.get_similar(
                        nmat=real_nmat,
                        nkpt=real_nkpt,
                        backend=cfg.backend or "wien2k",
                        limit=args.limit,
                    )
            else:
                recs = history.query(limit=args.limit)
            if recs:
                table = Table(title=f"Similar Runs (limit={args.limit})", border_style="cyan")
                table.add_column("Run ID", style="cyan")
                table.add_column("Backend", style="green")
                table.add_column("Cores", style="green")
                table.add_column("Walltime", style="green")
                table.add_column("Success")
                for r in recs:
                    table.add_row(r.run_id[:8], r.backend, str(r.total_cores), f"{r.walltime_sec:.1f}s", "✓" if r.success else "✗")
                console.print(table)
            else:
                console.print("[yellow]No similar runs found.[/yellow]")
            return {"count": len(recs)}

        recs = history.query(limit=args.limit)
        if recs:
            table = Table(title="Execution History", border_style="cyan")
            table.add_column("Run ID", style="cyan")
            table.add_column("Backend")
            table.add_column("Cores")
            table.add_column("Walltime")
            table.add_column("Date")
            table.add_column("Status")
            for r in recs:
                ts = str(r.timestamp)[:10] if r.timestamp else "?"
                table.add_row(r.run_id[:8], r.backend, str(r.total_cores), f"{r.walltime_sec:.1f}s", ts, "✓" if r.success else "✗")
            console.print(table)
        else:
            console.print("[yellow]No execution history found.[/yellow]")

        return {"records": len(recs)}


register_command("history", handle)
