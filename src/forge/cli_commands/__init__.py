"""
CLI command auto-discovery and registration.

Each submodule in this package defines:
    register(subparsers) — adds argparse subparser
    handle(args, cfg) -> dict — executes the command

Importing this package auto-registers all commands in the global registry.
"""

from __future__ import annotations

from . import (
    advise,
    analyze,
    analyze_bands,
    benchmark,
    converge,
    diagnose,
    diagnostics,
    generate,
    hardware,
    history,
    monitor,
    optimize,
    predict,
    run,
    screen,
    submit,
    tui,
    workflow,
)
from .base import get_handler


def register_all(subparsers) -> None:
    for mod in _ALL_MODULES:
        mod.register(subparsers)


_ALL_MODULES = (
    generate,
    submit,
    benchmark,
    diagnostics,
    hardware,
    analyze,
    tui,
    monitor,
    run,
    workflow,
    diagnose,
    optimize,
    screen,
    predict,
    advise,
    converge,
    history,
    analyze_bands,
)

__all__ = [
    "get_handler",
    "register_all",
]
