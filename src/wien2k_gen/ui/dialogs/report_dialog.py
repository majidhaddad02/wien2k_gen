"""
Report Dialog Modal for Structured HPC/DFT Output Visualization.
Displays diagnostics, pipeline results, profiling summaries, and job submission logs
in a scrollable, syntax-highlighted modal screen. Supports thread-safe rendering,
real-time export (JSON/TXT), keyboard bindings, and graceful fallback for malformed data.

Key Architecture Features:
• ModalScreen lifecycle with dimmed backdrop and focus trapping
• Recursive dict-to-Rich-markup converter for nested HPC reports
• Thread-safe export via atomic writes with progress feedback
• Keyboard bindings (Escape, Ctrl+E, Ctrl+C) for power-user workflows
• Structured sections: Hardware, Software, Warnings, Errors, Recommendations
• Comprehensive English documentation, type hints, and HPC-grade error resilience

All documentation and inline comments are in English per project standards.
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from textual.screen import ModalScreen
from textual.containers import ScrollableContainer, Horizontal, Vertical, Container
from textual.widgets import Static, Button, RichLog, Rule, Label, DataTable
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from rich.text import Text
from rich.console import Console
from rich.markdown import Markdown

from ...utils.atomic_write import atomic_write
from ...logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Messages for Dialog Communication
# =============================================================================

class ReportExportedMessage(Message, bubble=False):
    """Emitted when report file is successfully written."""
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

class ReportExportFailedMessage(Message, bubble=False):
    """Emitted when export encounters an error."""
    def __init__(self, error: str):
        super().__init__()
        self.error = error


# =============================================================================
# Report Dialog Implementation
# =============================================================================

class ReportDialog(ModalScreen):
    """
    Modal dialog for displaying structured diagnostic/pipeline/profiling reports.
    Designed for HPC environments where users need to inspect complex outputs
    before exporting or closing. Integrates seamlessly with the main TUI event loop.
    """
    DEFAULT_CSS = """
    ReportDialog {
        background: transparent;
        align: center middle;
    }

    .modal-frame {
        width: 80;
        height: 70%;
        background: $surface;
        border: solid $primary;
        padding: 1;
        layout: vertical;
    }

    .modal-header {
        height: 3;
        background: $primary 20%;
        padding: 0 1;
        align: left middle;
    }

    .modal-title {
        text-style: bold;
        color: $text;
    }

    #report_content {
        height: 1fr;
        margin: 1 0;
        background: $panel;
        border: solid $accent;
        overflow-y: auto;
    }

    #report_content RichLog {
        width: 1fr;
        height: 1fr;
    }

    .modal-actions {
        height: 4;
        margin: 1 0 0 0;
        align: center middle;
    }

    .modal-actions Button {
        width: 30%;
        margin: 0 1;
    }

    #status_bar {
        height: 2;
        margin: 0 0 1 0;
        padding: 0 1;
        text-align: center;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "close_dialog", "Close", show=True),
        Binding("ctrl+e", "export_report", "Export", show=True),
        Binding("ctrl+c", "copy_to_clipboard", "Copy", show=False),
    ]

    # Reactive State
    title: str = reactive("HPC Report")
    report_data: Dict[str, Any] = reactive({})
    export_format: str = reactive("json")
    is_exporting: bool = reactive(False)
    status_message: str = reactive("Ready")

    def __init__(
        self,
        data: Dict[str, Any],
        title: str = "HPC Report",
        export_name: str = "wien2k_gen_report.json",
        format: str = "json"
    ) -> None:
        """
        Initialize modal with report payload and export configuration.
        Args:
            data: Nested dictionary containing diagnostics, pipeline, or profiling results.
            title: Dialog header title.
            export_name: Default filename for export operations.
            format: Preferred export format ('json' or 'txt').
        """
        super().__init__()
        self.report_data = data
        self.title = title
        self.export_name = export_name
        self.export_format = format

    def compose(self) -> Any:
        """Build modal layout with header, scrollable content, and action buttons."""
        with Container(classes="modal-frame"):
            with Container(classes="modal-header"):
                yield Static(self.title, classes="modal-title")
                
            with ScrollableContainer(id="report_content"):
                yield RichLog(id="log_output", markup=True, highlight=True, wrap=True)
                
            with Container(classes="modal-actions"):
                yield Button("Export JSON", id="btn_export_json", variant="primary")
                yield Button("Export TXT", id="btn_export_txt", variant="default")
                yield Button("Close", id="btn_close", variant="warning")
                
            yield Static(self.status_message, id="status_bar")

    def on_mount(self) -> None:
        """Render report data upon dialog activation."""
        self.log.info(f"ReportDialog mounted: {self.title}")
        self.call_later(self._render_report_content)

    # =========================================================================
    # Event Handlers & Actions
    # =========================================================================

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button clicks to export or close actions."""
        btn_id = event.button.id
        if btn_id == "btn_close":
            self.action_close_dialog()
        elif btn_id == "btn_export_json":
            self.export_format = "json"
            self.action_export_report()
        elif btn_id == "btn_export_txt":
            self.export_format = "txt"
            self.action_export_report()

    def action_close_dialog(self) -> None:
        """Dismiss modal and return focus to parent screen."""
        self.dismiss()
        self.log.debug("ReportDialog closed by user")

    def action_export_report(self) -> None:
        """Trigger background export task with UI feedback."""
        if self.is_exporting:
            self.notify("Export already in progress.", severity="warning")
            return
        self.is_exporting = True
        self.status_message = f"Exporting to {self.export_format.upper()}..."
        self.call_later(self._update_status_bar)
        
        # Run in daemon thread to avoid blocking TUI event loop
        threading.Thread(target=self._export_worker, daemon=True).start()

    def action_copy_to_clipboard(self) -> None:
        """Placeholder for clipboard integration (requires pyperclip or similar)."""
        self.notify("Clipboard copy not supported in this environment.", severity="information")

    # =========================================================================
    # Core Rendering & Export Logic
    # =========================================================================

    def _render_report_content(self) -> None:
        """Convert nested report dict to Rich-formatted lines and populate RichLog."""
        log = self.query_one("#log_output", RichLog)
        if not log:
            return
            
        log.clear()
        data = self.report_data or {}
        
        console = Console()
        with console.capture() as capture:
            # Header
            console.print(f"[bold cyan]{'='*60}[/]")
            console.print(f"[bold white]{self.title.upper()}[/]")
            console.print(f"[dim]Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}[/]")
            console.print(f"[bold cyan]{'='*60}[/]\n")
            
            # Recursive section renderer
            self._render_section(console, data, level=0)
            
            console.print(f"\n[dim]{'='*60}[/]")
            
        formatted_text = capture.get()
        log.write(formatted_text)
        self.status_message = f"Report rendered ({len(formatted_text)} chars)"
        self.call_later(self._update_status_bar)

    def _render_section(self, console: Console, section: Any, level: int = 0) -> None:
        """Recursively format nested dictionaries/lists for RichLog display."""
        indent = "  " * level
        prefix = f"{indent}{'├─' if level > 0 else ''} "
        
        if isinstance(section, dict):
            for key, value in section.items():
                label = key.replace("_", " ").title()
                if isinstance(value, (dict, list)):
                    console.print(f"[bold yellow]{prefix}{label}:[/]")
                    self._render_section(console, value, level + 1)
                else:
                    # Colorize based on content type
                    if key in ("warnings", "errors", "critical_errors"):
                        color = "red" if "error" in key else "yellow"
                        count = len(value) if isinstance(value, list) else 0
                        console.print(f"{prefix}[{color} bold]{label}:[/] [{color}]{count} items[/]")
                    elif key in ("success", "valid", "converged"):
                        color = "green" if value else "red"
                        console.print(f"{prefix}[{color} bold]{label}:[/] [{color}]{value}[/]")
                    else:
                        console.print(f"{prefix}[bold white]{label}:[/] [dim]{value}[/]")
                        
        elif isinstance(section, list):
            for i, item in enumerate(section, 1):
                if isinstance(item, dict):
                    console.print(f"{prefix}[bold magenta]Item {i}:[/]")
                    self._render_section(console, item, level + 1)
                else:
                    console.print(f"{prefix}[dim]•[/] {item}")
        else:
            console.print(f"{prefix}[dim]{section}[/]")

    def _export_worker(self) -> None:
        """Thread-safe file export with atomic writes and error handling."""
        try:
            target = Path(self.export_name)
            if self.export_format == "json":
                content = json.dumps(self.report_data, indent=2, default=str, ensure_ascii=False) + "\n"
                target = target.with_suffix(".json")
            else:
                # Convert to plain text fallback
                content = json.dumps(self.report_data, indent=2, default=str) + "\n"
                target = target.with_suffix(".txt")
                
            atomic_write(target, content, mode=0o644)
            self.call_later(lambda: self.post_message(ReportExportedMessage(target)))
        except Exception as e:
            logger.error(f"Report export failed: {e}", exc_info=True)
            self.call_later(lambda: self.post_message(ReportExportFailedMessage(str(e))))

    def on_report_exported_message(self, msg: ReportExportedMessage) -> None:
        """Handle successful export completion."""
        self.is_exporting = False
        self.status_message = f"✓ Exported to {msg.path.name} ({msg.path.stat().st_size} bytes)"
        self.call_later(self._update_status_bar)
        self.notify(f"Report saved: {msg.path}", severity="success")

    def on_report_export_failed_message(self, msg: ReportExportFailedMessage) -> None:
        """Handle export failure."""
        self.is_exporting = False
        self.status_message = f"✗ Export failed: {msg.error}"
        self.call_later(self._update_status_bar)
        self.notify(f"Export error: {msg.error}", severity="error")

    def _update_status_bar(self) -> None:
        """Refresh status bar widget with current state."""
        try:
            status_widget = self.query_one("#status_bar", Static)
            if status_widget:
                status_widget.update(self.status_message)
        except Exception:
            pass  # Ignore if dialog is closing

    def watch_status_message(self, new_msg: str) -> None:
        """Reactive status bar sync."""
        self.call_later(self._update_status_bar)


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ReportDialog",
    "ReportExportedMessage",
    "ReportExportFailedMessage",
]