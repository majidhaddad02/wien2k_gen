"""
Command-Line Interface (CLI) Entry Point for FORGE.
Provides a production-grade, multi-command terminal interface for HPC/DFT workflow orchestration.
Supports configuration generation, job submission, benchmark execution, system diagnostics,
and TUI launch with structured logging, JSON output, and rigorous error handling.

Key Architecture Features:
• Subcommand-based routing: each sub-command lives in its own module under cli_commands/
• Global configuration & logging initialization before command dispatch
• Rich UI integration: Tables, Panels, and Progress Bars for human-readable output
• Structured exception handling with machine-readable JSON fallback & graceful degradation
• Terminal-aware console detection (--plain / --no-color for dumb terminals)
• Thread-safe execution context with signal-aware teardown
"""

import argparse
import json
import os
import signal
import sys
from typing import Any, Optional

from rich.console import Console

from .backend_manager import BackendManager
from .cli_commands import register_all as _register_commands
from .cli_commands._utils import set_console
from .cli_commands.base import get_handler
from .config import ensure_dirs, load_config
from .exceptions import (
    FORGEError,
    format_error_for_ui,
    log_exception_structured,
)
from .logging_config import get_logger, set_context, setup_logging
from .types import BackendCode
from .ui.rich_ui import detect_terminal_capabilities, get_plain_console, get_rich_console

logger = get_logger(__name__)
console = Console()

_term = os.environ.get("TERM", "")
_no_color = os.environ.get("NO_COLOR", "")
_is_dumb = _term in ("dumb", "vt100", "") or _no_color


def create_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with subcommands and global flags."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Production-grade WIEN2k parallel configuration & HPC job dispatcher",
        epilog=(
            "Examples:\n"
            "  forge generate --backend wien2k --cores 64 --target memory --dry-run\n"
            "  forge submit --partition gpu --time 48:00:00 --mem 64G --scheduler slurm\n"
            "  forge diagnostics --export report.json --full\n"
            "  forge tui\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    global_group = parser.add_argument_group("Global Options")
    global_group.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    global_group.add_argument("-q", "--quiet", action="store_true", help="Suppress console output")
    global_group.add_argument("--json", action="store_true", dest="json_output", help="Output results in JSON format")
    global_group.add_argument("--config", type=str, default=None, help="Path to custom config file")
    global_group.add_argument("--backend", type=str, choices=[b.value for b in BackendCode], help="Override auto-detected backend")
    global_group.add_argument("--log-file", type=str, default=None, help="Redirect logs to file")
    global_group.add_argument("--version", action="version", version="forge v0.1.0")
    global_group.add_argument("--plain", action="store_true", help="Use plain output (no Rich formatting, for dumb terminals)")
    global_group.add_argument("--no-color", action="store_true", dest="no_color", help="Disable colored output")

    subparsers = parser.add_subparsers(dest="command", help="Available workflow commands", required=True)
    _register_commands(subparsers)

    return parser


def main(argv: Optional[list[str]] = None) -> int:  # noqa: C901
    """
    CLI entry point with structured setup, dispatch, and error handling.
    Returns OS exit code: 0 (success), 1 (app error), 2 (CLI syntax error).
    """
    global console

    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    plain_mode = getattr(args, "plain", False) or getattr(args, "no_color", False)
    caps = detect_terminal_capabilities()

    if plain_mode or not caps.supports_color or _is_dumb:
        console = get_plain_console()
    else:
        console = get_rich_console()

    set_console(console)

    try:
        cfg = load_config(
            file_path=args.config,
            cli_override={
                "log_level": "DEBUG" if args.verbose > 0 else "ERROR" if args.quiet else "INFO",
                "quiet_mode": args.quiet,
                "backend": args.backend,
            },
        )
        ensure_dirs()
        setup_logging(config=cfg, verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
        set_context(cli="forge", user=os.environ.get("USER", "unknown"))
    except Exception as e:
        sys.stderr.write(f"Critical: Failed to initialize configuration/logging: {e}\n")
        return 2

    def _signal_handler(sig: int, frame: Any) -> None:
        logger.warning(f"Received signal {sig}. Cleaning up...")
        BackendManager.instance().reset()
        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    handler = get_handler(args.command)
    if not handler:
        parser.print_help()
        return 2

    try:
        logger.info(f"Executing command: {args.command}")
        result = handler(args, cfg)

        if args.json_output:
            print(json.dumps(result, indent=2, default=str))
        elif result and result.get("warnings"):
            for w in result["warnings"]:
                logger.warning(w)

        logger.info(f"Command '{args.command}' completed successfully.")
        return 0

    except FORGEError as e:
        log_exception_structured(e)
        if args.json_output:
            print(json.dumps({"error": e.to_dict()}, indent=2))
        else:
            console.print(format_error_for_ui(e), sep=" ", title="Error")
        return 1

    except Exception as e:
        logger.error(f"Unhandled exception in CLI dispatch: {e}", exc_info=True)
        if args.json_output:
            print(json.dumps({"error": {"message": str(e), "type": type(e).__name__}}, indent=2))
        else:
            console.print(f"[red]Unexpected Error:[/red] {e}\n[dim]Use -v or check logs for traceback.[/dim]")
        return 1


__all__ = [
    "create_parser",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
