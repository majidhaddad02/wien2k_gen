"""
Command base protocol and registry for CLI subcommands.

Each subcommand module defines:
    name: str          — the subparser name (e.g. "generate")
    help: str          — help text for argparse
    register(subparsers) — adds the parser
    handle(args, cfg) -> dict — executes the command
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from ..config import AppConfig

HandlerFunc = Callable[[Any, AppConfig], dict[str, Any]]


class CommandModule(Protocol):
    name: str
    help: str

    def register(self, subparsers: Any) -> None: ...

    def handle(self, args: Any, cfg: AppConfig) -> dict[str, Any]: ...


# Global registry — populated via __init__.py auto-discovery
_registry: dict[str, HandlerFunc] = {}


def register_command(name: str, handler: HandlerFunc) -> None:
    _registry[name] = handler


def get_handler(command_name: str) -> HandlerFunc | None:
    return _registry.get(command_name)
