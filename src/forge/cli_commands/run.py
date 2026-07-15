from __future__ import annotations

import argparse
from typing import Any

from rich.panel import Panel

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="Execute a WIEN2k workflow from YAML")
    p.add_argument("workflow_file", type=str, help="Path to workflow.yaml")
    p.add_argument("--auto-retry", action="store_true", default=True, help="Auto-retry on convergence failure")
    p.add_argument("--no-retry", action="store_true", help="Disable auto-retry")
    p.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts (default: 3)")
    p.add_argument("--poll", type=float, default=5.0, help="Job polling interval in seconds (default: 5)")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    try:
        from ..core.workflow_executor import run_workflow_from_yaml
    except ImportError as e:
        return {"error": f"Workflow executor not available: {e}"}

    auto_retry = not args.no_retry
    console.print(Panel(
        f"[cyan]Executing Workflow[/]\n"
        f"File: [bold]{args.workflow_file}[/]\n"
        f"Auto-retry: [{'green' if auto_retry else 'red'}]{auto_retry}[/]\n"
        f"Max retries: {args.max_retries}",
        border_style="blue",
    ))

    status = run_workflow_from_yaml(
        args.workflow_file,
        auto_retry=auto_retry,
        max_retries=args.max_retries,
        poll_interval=args.poll,
    )

    if status.state.value == "completed":
        console.print(f"[green]Workflow completed in {status.elapsed_total:.1f}s[/green]")
    else:
        console.print(f"[red]Workflow {status.state.value}: {len([e for e in status.events if 'Error' in e or 'Failed' in e])} errors[/red]")

    return {"status": status.state.value, "elapsed": status.elapsed_total, "events": status.events}


register_command("run", handle)
