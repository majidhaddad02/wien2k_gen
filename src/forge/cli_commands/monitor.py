from __future__ import annotations

import argparse
from typing import Any

from ..config import AppConfig
from ..core.terminal_monitor import launch_monitor, list_active_jobs
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("monitor", help="Monitor SCF convergence in real-time")
    p.add_argument("case", type=str, nargs="?", default=None, help="Case name to monitor (omit to list active jobs)")
    p.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds (default: 2)")
    p.add_argument("--output", type=str, default=None, help="SCF output file to parse")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()

    if getattr(args, 'json_output', False):
        return {"status": "monitor_skipped_in_json_mode"}

    if args.case:
        return launch_monitor(job_name=args.case, interval=args.interval)
    else:
        console.print(list_active_jobs())
        return {"status": "listed"}


register_command("monitor", handle)
