from __future__ import annotations

import argparse
from typing import Any

from rich.panel import Panel

from ..config import AppConfig
from ._utils import get_console, resolve_scheduler
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("benchmark", help="Run empirical or synthetic benchmarks")
    p.add_argument("--type", choices=["real", "synthetic"], default="real", help="Benchmark type")
    p.add_argument("--max-cores", type=int, default=None, help="Maximum cores for scaling suite")
    p.add_argument("--walltime", type=str, default="02:00:00", help="Max runtime per run")
    p.add_argument("--output", type=str, default=None, help="Save results to JSON path")
    p.add_argument("--skip-cleanup", action="store_true", help="Retain temporary benchmark directories")
    p.add_argument(
        "--scheduler",
        "-S",
        type=str,
        choices=["slurm", "pbs", "lsf", "auto"],
        default="auto",
        help="Target scheduler (default: auto-detect)",
    )


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    from ..core.scheduler import detect as detect_topology

    topo = detect_topology(max_cores=args.max_cores)

    if args.type == "synthetic":
        from ..benchmark.synthetic import SyntheticWorkloadParams, generate_strong_scaling_suite

        problem: SyntheticWorkloadParams = {"atoms": 20, "kpoints": 8, "nmat": 1500, "is_hybrid": False}
        suite = generate_strong_scaling_suite(problem, topo, max_cores=args.max_cores)
        console.print(
            Panel(
                f"[green]✓ Synthetic benchmark complete. {len(suite)} runs generated.[/]",
                border_style="green",
            )
        )
        return {"type": "synthetic", "runs": len(suite), "data": [dict(s) for s in suite]}
    else:
        from ..benchmark.real import RealBenchmarkRunner

        scheduler = resolve_scheduler(getattr(args, "scheduler", "auto"))
        runner = RealBenchmarkRunner(
            {
                "backend": cfg.backend,
                "scheduler": scheduler,
                "walltime": args.walltime,
                "cleanup_after": not args.skip_cleanup,
            }
        )
        result = runner.run(topo)
        if getattr(args, "json_output", False):
            return dict(result)

        status_color = "green" if result["status"] == "success" else "red"
        console.print(
            Panel(
                f"[{status_color}]Benchmark {result['status']}[/]\n"
                f"Run ID: [cyan]{result['run_id']}[/]\n"
                f"Wall Time: [bold]{result['wall_time_sec']:.2f}s[/]",
                border_style=status_color,
            )
        )
        return {
            "run_id": result["run_id"],
            "status": result["status"],
            "wall_time": result["wall_time_sec"],
        }


register_command("benchmark", handle)
