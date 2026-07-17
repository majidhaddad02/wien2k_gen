"""
Shared utilities and global state for CLI command modules.
"""

import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

_console = Console()


def get_console() -> Console:
    return _console


def set_console(c: Console) -> None:
    global _console
    _console = c


def resolve_scheduler(flag: str) -> str:
    from ..core.scheduler import _detect_scheduler

    if flag and flag != "auto":
        return flag
    return _detect_scheduler()


def open_editor_for_manual_review(filepath: Path) -> None:
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", ""))
    if not editor:
        for fallback in ("nano", "vim", "vi"):
            if shutil.which(fallback):
                editor = fallback
                break
    if not editor:
        _console.print(
            f"[yellow]No editor found ($EDITOR unset, nano/vim not in PATH). "
            f"Edit manually: {filepath}[/yellow]"
        )
        return

    _console.print(f"[bold cyan]Opening {filepath.name} in {editor} for manual review...[/bold cyan]")
    _console.print("[dim](Save and exit to continue, or :q! to discard)[/dim]")
    try:
        subprocess.run([editor, str(filepath)], check=False)
        _console.print(f"[green]✓ Editor closed. Final config: {filepath}[/green]")
    except FileNotFoundError:
        _console.print(f"[yellow]Editor '{editor}' not found. Edit manually: {filepath}[/yellow]")
    except Exception as e:
        _console.print(f"[yellow]Editor error: {e}. Edit manually: {filepath}[/yellow]")


def get_exec_command() -> str:
    try:
        from ..backend_manager import get_current_backend

        backend = get_current_backend()
        params = backend.detect_problem_size()
        return str(params.get("exec_command", "run_lapw -p"))
    except Exception:
        return "run_lapw -p"
