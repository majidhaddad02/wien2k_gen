from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.panel import Panel

from ..config import AppConfig
from ..core.scheduler import auto_detect_memory
from ._utils import get_console, get_exec_command, resolve_scheduler
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("submit", help="Submit job to scheduler")
    p.add_argument(
        "--scheduler",
        "-S",
        type=str,
        choices=["slurm", "pbs", "lsf", "sge", "auto"],
        default="auto",
        help="Target scheduler (default: auto-detect)",
    )
    p.add_argument("--partition", type=str, default="", help="Scheduler partition/queue")
    p.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    p.add_argument("--ntasks", type=int, default=0, help="Total tasks (0 = auto from topology)")
    p.add_argument("--time", type=str, default="24:00:00", help="Walltime (HH:MM:SS)")
    p.add_argument("--mem", type=str, default=auto_detect_memory(), help="Memory per node")
    p.add_argument("--job-name", type=str, default="wien2k_job", help="Job identifier")
    p.add_argument("--dependency", type=str, default="", help="Job dependency (e.g., afterok:123)")
    p.add_argument("--dry-run", action="store_true", help="Generate script only, do not submit")
    p.add_argument("--export", type=str, default=None, help="Export script to path")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    from ..core.scheduler import detect as detect_topology
    from ..submit import SUBMIT_PROVIDERS
    from ..submit.slurm import SlurmDirectives, SlurmJobSpec, submit_slurm_job

    scheduler = resolve_scheduler(getattr(args, "scheduler", "auto"))
    topo = detect_topology(max_cores=args.ntasks or None)

    if scheduler == "slurm":
        directives = SlurmDirectives(
            job_name=args.job_name,
            partition=args.partition,
            nodes=args.nodes,
            ntasks=args.ntasks or topo.total_cores,
            cpus_per_task=1,
            mem_per_node=args.mem,
            time=args.time,
            dependency=args.dependency or None,
        )
        spec = SlurmJobSpec(
            topo=topo,
            exec_command=get_exec_command(),
            directives=directives,
            working_dir=Path.cwd(),
        )
        res = submit_slurm_job(
            spec=spec, dry_run=args.dry_run, script_path=Path(args.export) if args.export else None
        )

        if getattr(args, "json_output", False):
            return {
                "success": res.get("success"),
                "job_id": res.get("job_id"),
                "script_path": str(res.get("script_path", "")),
            }

        if res.get("success"):
            if args.dry_run:
                console.print(
                    Panel(
                        res.get("dry_run_content") or "Script content not available",
                        title="SBATCH Preview",
                        border_style="cyan",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[green]✓ Job submitted successfully.[/]\nJob ID: [bold cyan]{res.get('job_id')}[/]\nScript: [dim]{res.get('script_path')}[/]",
                        border_style="green",
                    )
                )
        else:
            console.print(
                Panel(f"[red]✗ Submission failed: {res.get('errors')}[/]", border_style="red")
            )
        return {
            "success": res.get("success"),
            "job_id": res.get("job_id"),
            "path": str(res.get("script_path", "")),
        }

    elif scheduler in ("pbs", "lsf"):
        provider_cls = SUBMIT_PROVIDERS.get(scheduler)
        if provider_cls:
            provider = provider_cls()
            pbs_res = provider.submit(
                topo=topo,
                exec_command="run_lapw -p",
                directives={
                    "job_name": args.job_name,
                    "queue": args.partition,
                    "nodes": args.nodes,
                    "walltime": args.time,
                    "mem" if scheduler == "pbs" else "memory": args.mem,
                },
                script_path=Path(args.export) if args.export else None,
                dry_run=args.dry_run,
            )
            if getattr(args, "json_output", False):
                return {
                    "success": pbs_res.get("success"),
                    "job_id": pbs_res.get("job_id"),
                    "script_path": str(pbs_res.get("script_path", "")),
                }
            if pbs_res.get("success"):
                console.print(
                    Panel(
                        f"[green]✓ Job submitted successfully.[/]\nJob ID: [bold cyan]{pbs_res.get('job_id')}[/]\nScript: [dim]{pbs_res.get('script_path')}[/]",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(f"[red]✗ Submission failed: {pbs_res.get('errors')}[/]", border_style="red")
                )
            return {
                "success": pbs_res.get("success"),
                "job_id": pbs_res.get("job_id"),
                "path": str(pbs_res.get("script_path", "")),
            }
        else:
            return {"success": False, "errors": [f"Scheduler provider '{scheduler}' not available."]}

    return {"success": False, "errors": [f"Unknown scheduler: {scheduler}"]}


register_command("submit", handle)
