from __future__ import annotations

import argparse
from typing import Any

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tui", help="Launch interactive terminal UI")
    p.add_argument("--compact", action="store_true", help="Enable compact UI layout")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    console.print("[yellow]TUI has been removed. Use `forge monitor [case]` for live SCF monitoring.[/yellow]")
    return {"status": "tui_removed"}


register_command("tui", handle)
