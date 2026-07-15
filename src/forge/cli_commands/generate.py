from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ..types import ExecutionMode
from ._utils import get_console, open_editor_for_manual_review
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("generate", help="Generate parallel configuration files")
    p.add_argument("--nodes", type=int, default=None, help="Number of compute nodes")
    p.add_argument("--cores", type=int, default=None, help="Total MPI cores to allocate")
    p.add_argument("--omp", type=int, default=None, help="OpenMP threads per rank")
    p.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in ExecutionMode],
        help="Parallel execution mode",
    )
    p.add_argument(
        "--target",
        type=str,
        choices=["time", "memory", "balanced", "cost"],
        default="time",
        help="Optimization target (time, memory, balanced, cost)",
    )
    p.add_argument("--max-cores", type=int, default=None, help="Hard limit on total cores to utilize")
    p.add_argument(
        "--reserve-os-cores",
        type=int,
        default=None,
        metavar="N",
        help="Reserve N cores for OS/daemons (e.g., 4 leaves 124 of 128)",
    )
    p.add_argument("--memory-limit", type=float, default=None, help="Hard limit on memory per node (GB)")
    p.add_argument("--dry-run", action="store_true", help="Generate config without writing to disk")
    p.add_argument("--export", type=str, default=None, help="Export configuration summary to path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .machines/INCAR without prompt")
    p.add_argument(
        "--scheduler",
        "-S",
        type=str,
        choices=["slurm", "pbs", "lsf", "sge", "auto"],
        default="auto",
        help="Target scheduler for job scripts (default: auto-detect)",
    )
    p.add_argument("--gpu", action="store_true", help="Enable GPU-aware configuration")
    p.add_argument("--gpu-mixed-precision", action="store_true", help="Enable FP32/FP16 mixed precision")
    p.add_argument(
        "--manual",
        action="store_true",
        help="After auto-generation, open .machines in $EDITOR for manual review/edit",
    )


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:  # noqa: C901
    console = get_console()

    from ..core.hardware import get_physical_cores
    from ..core.pipeline import run_pipeline
    from ..core.scheduler import detect as detect_topology

    max_cores = args.max_cores
    if args.reserve_os_cores is not None:
        phys = get_physical_cores()
        reserved_max = max(1, phys - args.reserve_os_cores)
        max_cores = min(max_cores, reserved_max) if max_cores else reserved_max
        console.print(f"[dim]Reserving {args.reserve_os_cores} OS cores → using {max_cores} of {phys}[/dim]")
    topo = detect_topology(max_cores=max_cores)

    suggestion: dict[str, Any] = {}
    if args.mode:
        suggestion["mode"] = ExecutionMode(args.mode)
    if args.cores:
        suggestion["recommended_total_cores"] = args.cores
    if args.omp:
        suggestion["omp_threads_per_rank"] = args.omp
    if args.memory_limit:
        suggestion["memory_limit_gb"] = args.memory_limit

    if args.gpu or args.gpu_mixed_precision:
        try:
            from ..backends.gpu_backend import (
                detect_gpu,
                get_gpu_recommendation,
                get_mixed_precision_recommendation,
            )
            gpus = detect_gpu()
            if gpus:
                gpu_rec = get_gpu_recommendation(topo, nmat=5000, nkpt=8, mode=suggestion.get("mode", "mpi"))
                suggestion["gpu_recommendation"] = gpu_rec
                if args.gpu_mixed_precision:
                    fp_rec = get_mixed_precision_recommendation("wien2k", nmat=5000)
                    suggestion["mixed_precision"] = fp_rec
                console.print(f"[green]GPU detection: {len(gpus)} device(s) found.[/green]")
            else:
                console.print("[yellow]No GPUs detected. Proceeding without GPU configuration.[/yellow]")
        except ImportError as e:
            console.print(f"[yellow]GPU backend not available: {e}[/yellow]")

    result = run_pipeline(
        topo=topo,
        user_suggestion=suggestion,
        dry_run=args.dry_run,
        export_path=args.export,
    )

    if getattr(args, "json_output", False):
        return {"status": "success" if result.success else "failed", "data": result.to_dict()}

    if result.success:
        if args.dry_run and result.dry_run_content:
            table = Table(title="Configuration Preview", border_style="cyan")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")

            table.add_row("Mode", str(suggestion.get("mode", "auto")))
            table.add_row("Total Cores", str(suggestion.get("recommended_total_cores", topo.total_cores)))

            max_eff = suggestion.get("max_efficient_cores")
            if max_eff:
                table.add_row("Max Efficient Cores", f"[yellow]{max_eff}[/yellow]")

            sat_data: dict[str, Any] = suggestion.get("saturation_data", {})
            if sat_data:
                eff = sat_data.get("efficiency_pct")
                if eff is not None:
                    table.add_row("Est. Efficiency", f"[dim]{eff:.0f}%[/dim]")
                sf = sat_data.get("serial_fraction")
                if sf is not None:
                    table.add_row("Serial Fraction (Amdahl)", f"[dim]s={sf:.3f}[/dim]")

            table.add_row("Dry-Run Content", f"[dim]{len(result.dry_run_content)} bytes generated[/dim]")

            console.print(table)
            console.print(Panel(result.dry_run_content, title="Generated Config", border_style="dim"))
        else:
            console.print(
                Panel(
                    f"[green]✓ Configuration generated successfully.[/]\nPath: [cyan]{result.config_path}[/]",
                    border_style="green",
                )
            )

            if getattr(args, "manual", False) and result.config_path:
                open_editor_for_manual_review(Path(result.config_path))

        if result.warnings:
            warn_table = Table(title="Warnings", show_header=False, box=None)
            for w in result.warnings:
                warn_table.add_row("[yellow]•[/]", f"[dim]{w}[/dim]")
            console.print(warn_table)
    else:
        console.print(
            Panel(f"[red]✗ Generation failed: {result.validation_errors}[/]", border_style="red")
        )

    return {"success": result.success, "path": result.config_path, "warnings": result.warnings}


register_command("generate", handle)
