from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from rich.panel import Panel

from ..config import AppConfig
from ..core.scheduler import auto_detect_memory
from ..utils.validation import parse_machines_file
from ._utils import get_console, get_exec_command, resolve_scheduler
from .base import register_command

MISSING_MACHINES_MSG = (
    "No .machines file found. Run `forge generate` first, or pass --auto-generate."
)
MACHINES_PROMPT = (
    "`.machines` not found — generate it now with `forge generate` before submitting? [y/N]"
)


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
    p.add_argument(
        "--ntasks",
        type=int,
        default=0,
        help="Total tasks (0 = auto from .machines allocation)",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=0,
        help="CPUs per task (0 = auto from .machines OMP setting)",
    )
    p.add_argument("--time", type=str, default="24:00:00", help="Walltime (HH:MM:SS)")
    p.add_argument("--mem", type=str, default=auto_detect_memory(), help="Memory per node")
    p.add_argument("--job-name", type=str, default="wien2k_job", help="Job identifier")
    p.add_argument("--dependency", type=str, default="", help="Job dependency (e.g., afterok:123)")
    p.add_argument("--dry-run", action="store_true", help="Generate script only, do not submit")
    p.add_argument("--export", type=str, default=None, help="Export script to path")
    gen_group = p.add_mutually_exclusive_group()
    gen_group.add_argument(
        "--auto-generate",
        action="store_true",
        help="Generate .machines via forge generate if missing, then submit",
    )
    gen_group.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Fail if .machines is missing instead of generating it",
    )


def _config_path() -> Path:
    try:
        from ..backend_manager import get_current_backend

        backend = get_current_backend()
        if hasattr(backend, "get_config_filename") and callable(backend.get_config_filename):
            return Path(backend.get_config_filename())
    except Exception:
        pass
    return Path(".machines")


def _is_interactive(args: argparse.Namespace) -> bool:
    return bool(sys.stdin.isatty()) and not getattr(args, "json_output", False)


def _fail(errors: list[str], *, warnings: Optional[list[str]] = None) -> dict[str, Any]:
    result: dict[str, Any] = {"success": False, "errors": errors}
    if warnings:
        result["warnings"] = warnings
    return result


def _prompt_generate(console: Any) -> bool:
    if hasattr(console, "input") and callable(console.input):
        answer = console.input(MACHINES_PROMPT)
    else:
        answer = input(MACHINES_PROMPT)
    return str(answer).strip().lower().startswith("y")


def _auto_generate_machines(topo: Any) -> dict[str, Any]:
    from ..core.pipeline import run_pipeline

    pipe = run_pipeline(topo)
    if not pipe.success:
        errors = list(getattr(pipe, "validation_errors", None) or [])
        if not errors:
            errors = ["Failed to auto-generate .machines"]
        return _fail(errors, warnings=list(getattr(pipe, "warnings", None) or []))

    sug = getattr(pipe, "suggestion", None)
    ntasks = 0
    cpus_per_task = 1
    if sug is not None:
        if hasattr(sug, "recommended_total_cores"):
            ntasks = int(sug.recommended_total_cores or 0)
            cpus_per_task = int(getattr(sug, "omp_threads_per_rank", 1) or 1)
        elif isinstance(sug, dict):
            ntasks = int(sug.get("recommended_total_cores") or 0)
            cpus_per_task = int(sug.get("omp_threads_per_rank") or 1)

    return {
        "success": True,
        "ntasks": ntasks or int(getattr(topo, "total_cores", 1) or 1),
        "cpus_per_task": cpus_per_task or 1,
        "warnings": list(getattr(pipe, "warnings", None) or []),
    }


def _resources_from_machines(machines_path: Path, topo: Any) -> dict[str, Any]:
    config, parse_errors = parse_machines_file(machines_path)
    real_ntasks = sum(config.get("cores_per_node", [])) or int(
        getattr(topo, "total_cores", 1) or 1
    )
    real_cpus_per_task = config.get("omp_global", 1) or 1
    return {
        "success": True,
        "ntasks": int(real_ntasks),
        "cpus_per_task": int(real_cpus_per_task),
        "warnings": list(parse_errors or []),
    }


def resolve_submit_resources(
    args: argparse.Namespace,
    topo: Any,
    console: Any,
) -> dict[str, Any]:
    machines_path = _config_path()
    warnings: list[str] = []

    if machines_path.exists():
        resolved = _resources_from_machines(machines_path, topo)
        warnings.extend(resolved.get("warnings") or [])
        ntasks = args.ntasks or resolved["ntasks"]
        cpus_per_task = args.cpus_per_task or resolved["cpus_per_task"]
        if warnings:
            for w in warnings:
                console.print(f"[yellow]Warning:[/] {w}")
        return {
            "success": True,
            "ntasks": ntasks,
            "cpus_per_task": cpus_per_task,
            "warnings": warnings,
        }

    auto_gen = bool(getattr(args, "auto_generate", False))
    no_auto = bool(getattr(args, "no_auto_generate", False))

    if no_auto:
        return _fail([MISSING_MACHINES_MSG])

    should_generate = auto_gen
    if not should_generate:
        if _is_interactive(args):
            if not _prompt_generate(console):
                return _fail([MISSING_MACHINES_MSG])
            should_generate = True
        else:
            return _fail([MISSING_MACHINES_MSG])

    generated = _auto_generate_machines(topo)
    if not generated.get("success"):
        return generated
    warnings.extend(generated.get("warnings") or [])
    ntasks = args.ntasks or generated["ntasks"]
    cpus_per_task = args.cpus_per_task or generated["cpus_per_task"]
    return {
        "success": True,
        "ntasks": ntasks,
        "cpus_per_task": cpus_per_task,
        "warnings": warnings,
    }


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    from ..core.scheduler import detect as detect_topology
    from ..submit import SUBMIT_PROVIDERS
    from ..submit.slurm import SlurmDirectives, SlurmJobSpec, submit_slurm_job

    scheduler = resolve_scheduler(getattr(args, "scheduler", "auto"))
    topo = detect_topology(max_cores=args.ntasks or None)

    resolved = resolve_submit_resources(args, topo, console)
    if not resolved.get("success"):
        errors = list(resolved.get("errors") or [MISSING_MACHINES_MSG])
        if not getattr(args, "json_output", False):
            console.print(Panel(f"[red]✗ {errors[0]}[/]", border_style="red"))
        return _fail(errors, warnings=resolved.get("warnings"))

    ntasks = resolved["ntasks"]
    cpus_per_task = resolved["cpus_per_task"]
    exec_command = get_exec_command()

    if scheduler == "slurm":
        directives = SlurmDirectives(
            job_name=args.job_name,
            partition=args.partition,
            nodes=args.nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem_per_node=args.mem,
            time=args.time,
            dependency=args.dependency or None,
        )
        spec = SlurmJobSpec(
            topo=topo,
            exec_command=exec_command,
            directives=directives,
            working_dir=Path.cwd(),
        )
        res = submit_slurm_job(
            spec=spec, dry_run=args.dry_run, script_path=Path(args.export) if args.export else None
        )

        payload = {
            "success": res.get("success"),
            "job_id": res.get("job_id"),
            "script_path": str(res.get("script_path", "")),
            "path": str(res.get("script_path", "")),
            "ntasks": ntasks,
            "cpus_per_task": cpus_per_task,
            "warnings": resolved.get("warnings") or [],
        }
        if getattr(args, "json_output", False):
            return payload

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
        return payload

    elif scheduler in ("pbs", "lsf"):
        provider_cls = SUBMIT_PROVIDERS.get(scheduler)
        if provider_cls:
            provider = provider_cls()
            pbs_res = provider.submit(
                topo=topo,
                exec_command=exec_command,
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
            payload = {
                "success": pbs_res.get("success"),
                "job_id": pbs_res.get("job_id"),
                "script_path": str(pbs_res.get("script_path", "")),
                "path": str(pbs_res.get("script_path", "")),
                "ntasks": ntasks,
                "cpus_per_task": cpus_per_task,
                "warnings": resolved.get("warnings") or [],
            }
            if getattr(args, "json_output", False):
                return payload
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
            return payload
        else:
            return {"success": False, "errors": [f"Scheduler provider '{scheduler}' not available."]}

    return {"success": False, "errors": [f"Unknown scheduler: {scheduler}"]}


register_command("submit", handle)
