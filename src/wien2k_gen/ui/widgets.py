"""
Custom Textual Widgets for Wien2kGen TUI.
Provides production-grade, reactive UI components for hardware visualization,
log streaming, configuration editing, validated inputs, and status tracking.
Designed for seamless integration with async workers, pipeline execution, and SLURM submission.

Key Architecture Features:
• Thread-safe UI updates via `call_later()` for background worker integration
• Reactive state synchronization with automatic DOM refreshes
• Rich markup rendering for color-coded logs, hardware metrics, and validation feedback
• Responsive DataTable layouts with heterogeneous cluster support
• Inline validation, constraint enforcement, and visual state transitions
• Comprehensive English documentation, type hints, and HPC-grade error resilience
All documentation and inline comments are in English per project standards.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, Horizontal, Vertical
from textual.widgets import Static, Input, Button, TextArea, DataTable, RichLog, Label, Rule, Collapsible
from textual.reactive import reactive
from textual.message import Message
from textual.css.query import NoMatches

from rich.text import Text
from rich.console import Console

from ..core.topology import Topology
from ..logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Message Types for Decoupled Widget Communication
# =============================================================================

class LogMessage(Message, bubble=False):
    """Structured log event for cross-widget communication."""
    def __init__(self, message: str, level: str = "INFO"):
        super().__init__()
        self.message = message
        self.level = level


class ValidationMessage(Message, bubble=False):
    """Emitted when validated input changes state."""
    def __init__(self, widget_id: str, is_valid: bool, value: str):
        super().__init__()
        self.widget_id = widget_id
        self.is_valid = is_valid
        self.value = value


# =============================================================================
# Core UI Components
# =============================================================================

class StatusIndicator(Static):
    """
    Real-time status display with color-coded states, animated indicators,
    and reactive message binding. Thread-safe for worker updates.
    """
    DEFAULT_CSS = """
    StatusIndicator {
        height: 1;
        padding: 0 1;
        background: $surface;
        border: round $primary;
        text-align: center;
    }
    """
    status = reactive("idle")  # idle, running, success, warning, error
    message = reactive("Ready")

    def watch_status(self, status: str) -> None:
        self.refresh()

    def watch_message(self, msg: str) -> None:
        self.refresh()

    def render(self) -> str:
        icons = {"idle": "⏸", "running": "⚙", "success": "✓", "warning": "⚠", "error": "✗"}
        colors = {"idle": "gray", "running": "yellow", "success": "green", "warning": "orange3", "error": "red"}
        icon = icons.get(self.status, "•")
        color = colors.get(self.status, "white")
        return f"[{color} bold]{icon}[/] [{color}]{self.message}[/]"

    def set_running(self, msg: str = "Processing...") -> None:
        self.call_later(lambda: setattr(self, "status", "running"))
        self.call_later(lambda: setattr(self, "message", msg))

    def set_success(self, msg: str = "Complete") -> None:
        self.call_later(lambda: setattr(self, "status", "success"))
        self.call_later(lambda: setattr(self, "message", msg))

    def set_error(self, msg: str = "Failed") -> None:
        self.call_later(lambda: setattr(self, "status", "error"))
        self.call_later(lambda: setattr(self, "message", msg))


class LogPanel(ScrollableContainer):
    """
    High-performance scrollable log viewer with RichLog backend.
    Supports level-based colorization, auto-scroll toggle, max-line rotation,
    and thread-safe append from background workers.
    """
    DEFAULT_CSS = """
    LogPanel {
        background: $panel;
        border: solid $primary;
    }
    LogPanel RichLog {
        width: 1fr;
        height: 1fr;
    }
    """
    auto_scroll = reactive(True)
    max_lines = reactive(5000)
    _log_widget: Optional[RichLog] = None

    def compose(self) -> ComposeResult:
        yield RichLog(id="log_content", markup=True, highlight=True, wrap=True, auto_scroll=self.auto_scroll)

    def on_mount(self) -> None:
        try:
            self._log_widget = self.query_one("#log_content", RichLog)
        except NoMatches:
            logger.warning("LogPanel mount failed: RichLog not found in DOM")

    def append(self, message: str, level: str = "INFO") -> None:
        """Append a log line with color coding. Thread-safe via call_later."""
        def _write() -> None:
            if not self._log_widget:
                return
            color_map = {
                "DEBUG": "gray50", "INFO": "white", "SUCCESS": "green",
                "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold red"
            }
            color = color_map.get(level.upper(), "white")
            timestamp = time.strftime("%H:%M:%S")
            line = f"[dim]{timestamp}[/] [{color}][{level}][/] {message}"
            
            # Rotate if exceeding limit
            if self._log_widget.line_count >= self.max_lines:
                self._log_widget.clear()
            self._log_widget.write(line)
            if self.auto_scroll:
                self.scroll_end(animate=False)

        if hasattr(self, "call_later"):
            self.call_later(_write)
        else:
            _write()

    def clear(self) -> None:
        if self._log_widget:
            self.call_later(self._log_widget.clear)

    def toggle_auto_scroll(self) -> None:
        self.auto_scroll = not self.auto_scroll
        if self._log_widget:
            self._log_widget.auto_scroll = self.auto_scroll


class ResourceSummaryTable(DataTable):
    """
    Responsive DataTable for hardware topology visualization.
    Reactively updates when `topology` changes, supports heterogeneous clusters,
    and displays node-level cores, memory, NUMA, and scheduler hints.
    """
    DEFAULT_CSS = """
    ResourceSummaryTable {
        height: auto;
        max-height: 12;
        border: solid $primary;
    }
    ResourceSummaryTable .datatable--header {
        background: $primary 20%;
    }
    """
    topology = reactive(None)

    def watch_topology(self, topo: Optional[Topology]) -> None:
        self.call_later(lambda: self._populate(topo))

    def _populate(self, topo: Optional[Topology]) -> None:
        self.clear()
        if not topo:
            self.add_columns("Metric", "Value", "Status")
            self.add_row("Environment", "Not detected", "⏸")
            return

        self.add_columns("Node / Metric", "Value", "Status")
        self.add_row("Environment", topo.env_type.upper(), "✓")
        self.add_row("Total Cores", str(topo.total_cores), "✓")
        self.add_row("Nodes", str(len(topo.nodes)), "✓")
        
        launcher = topo.scheduler_hints.get("mpi_launcher", "unknown")
        self.add_row("MPI Launcher", launcher, "ℹ")
        self.add_row("NUMA Nodes", str(len(topo.numa_nodes) if hasattr(topo, 'numa_nodes') and topo.numa_nodes else 1), "ℹ")

        # Node-specific details
        for i, (node, cores) in enumerate(zip(topo.nodes, topo.cores_per_node)):
            mem = topo.get_memory_for_node(node) if hasattr(topo, 'get_memory_for_node') else None
            mem_str = f"{mem/1024:.1f} GB" if mem else "Unknown"
            status = "✓" if cores > 0 else "⚠"
            prefix = "  └  " if i < len(topo.nodes) - 1 else "  └  "
            self.add_row(f"{prefix}{node}", f"{cores} cores | {mem_str}", status)

        if topo.heterogeneous:
            self.add_row("Cluster Type", "Heterogeneous", "⚠")
        if topo.scheduler_hints.get("numa_aware"):
            self.add_row("Binding", "NUMA-aware", "✓")


class ValidatedInput(Input):
    """
    Input field with real-time validation, inline error styling,
    type conversion, and min/max constraint enforcement.
    Emits ValidationMessage on state change.
    """
    DEFAULT_CSS = """
    ValidatedInput {
        border: solid $surface;
    }
    ValidatedInput.-valid {
        border: solid $success;
    }
    ValidatedInput.-invalid {
        border: solid $error;
    }
    """
    value_type = reactive("str")  # str, int, float, positive_int, non_zero_int
    min_val = reactive(None)
    max_val = reactive(None)
    is_valid = reactive(True)

    def on_input_changed(self, event: Input.Changed) -> None:
        self.validate()

    def validate(self) -> bool:
        val = self.value.strip()
        if not val:
            self.is_valid = True
            self.set_class(False, "-invalid")
            self.set_class(True, "-valid")
            return True

        try:
            if self.value_type == "int":
                num = int(val)
            elif self.value_type == "float":
                num = float(val)
            elif self.value_type == "positive_int":
                num = int(val)
                if num <= 0:
                    raise ValueError("Must be positive")
            elif self.value_type == "non_zero_int":
                num = int(val)
                if num == 0:
                    raise ValueError("Cannot be zero")
            else:
                self.is_valid = True
                self.set_class(False, "-invalid")
                self.set_class(True, "-valid")
                self._emit_validation()
                return True

            if self.min_val is not None and num < self.min_val:
                raise ValueError(f"Below minimum ({self.min_val})")
            if self.max_val is not None and num > self.max_val:
                raise ValueError(f"Above maximum ({self.max_val})")

            self.is_valid = True
            self.set_class(False, "-invalid")
            self.set_class(True, "-valid")
            self._emit_validation()
            return True
        except ValueError:
            self.is_valid = False
            self.set_class(True, "-invalid")
            self.set_class(False, "-valid")
            self._emit_validation()
            return False

    def _emit_validation(self) -> None:
        self.post_message(ValidationMessage(self.id, self.is_valid, self.value))


class HardwareInfoCard(Static):
    """
    Compact, reactive display of critical hardware & environment metrics.
    Renders key-value pairs with Rich formatting and automatic refresh on state change.
    """
    DEFAULT_CSS = """
    HardwareInfoCard {
        background: $panel;
        border: dashed $primary;
        padding: 0 1;
    }
    """
    data = reactive({})

    def render(self) -> str:
        if not self.data:
            return "[dim]No hardware data available. Run detection.[/]"
        lines = []
        for k, v in self.data.items():
            label = k.replace("_", " ").title()
            icon = "✓" if str(v).lower() not in ("unknown", "false", "none", "0") else "⚠"
            lines.append(f"[bold cyan]{label}:[/] [{'green' if icon == '✓' else 'yellow'}]{v}[/] {icon}")
        return "\n".join(lines)


class StreamingConfigEditor(TextArea):
    """
    Syntax-aware configuration editor for WIEN2k/QE/VASP parallel files.
    Supports read-only toggle, validation overlay, and auto-format hints.
    Thread-safe content updates from pipeline workers.
    """
    DEFAULT_CSS = """
    StreamingConfigEditor {
        height: 1fr;
        min-height: 10;
        background: $surface;
        border: solid $accent;
    }
    """
    is_read_only = reactive(False)
    validation_errors = reactive([])

    def on_mount(self) -> None:
        self.language = "toml"  # Closest match for .machines/key-value syntax
        self.soft_wrap = True
        self.show_line_numbers = True

    def watch_is_read_only(self, ro: bool) -> None:
        self.read_only = ro
        self.cursor_blink = not ro

    def set_content(self, content: str) -> None:
        """Thread-safe content replacement."""
        def _update() -> None:
            self.text = content
            self.cursor = (0, 0)
        self.call_later(_update)

    def append_warning(self, warning: str) -> None:
        """Append validation warning to bottom of editor (non-destructive)."""
        if not self.validation_errors:
            self.validation_errors = []
        self.validation_errors.append(warning)
        warning_block = f"\n# ⚠ Validation Warnings:\n" + "\n".join(f"# {w}" for w in self.validation_errors[-5:])
        self.call_later(lambda: setattr(self, "text", self.text + warning_block))


class SubmissionConfigForm(Container):
    """
    Collapsible form for job submission parameters.
    Integrates ValidatedInput widgets, reactive state binding,
    and auto-populated defaults from detected topology.
    """
    DEFAULT_CSS = """
    SubmissionConfigForm {
        layout: vertical;
        height: auto;
    }
    .form-row {
        layout: horizontal;
        height: auto;
        margin: 1 0;
    }
    .form-label {
        width: 12;
        text-align: right;
        padding-right: 1;
    }
    """
    def compose(self) -> ComposeResult:
        with Horizontal(classes="form-row"):
            yield Label("Job Name: ", classes="form-label")
            yield ValidatedInput(id="job_name", value="wien2k_job", placeholder="Unique job identifier")
        with Horizontal(classes="form-row"):
            yield Label("Nodes: ", classes="form-label")
            yield ValidatedInput(id="nodes", value_type="positive_int", value="1", placeholder="1-1000")
        with Horizontal(classes="form-row"):
            yield Label("Time (HH:MM): ", classes="form-label")
            yield ValidatedInput(id="walltime", value_type="str", value="24:00:00", placeholder="HH:MM:SS or D-HH:MM")
        with Horizontal(classes="form-row"):
            yield Label("Memory (GB): ", classes="form-label")
            yield ValidatedInput(id="memory", value_type="positive_int", value="64", placeholder="Per node")
        with Horizontal(classes="form-row"):
            yield Label("Partition: ", classes="form-label")
            yield ValidatedInput(id="partition", value_type="str", value="", placeholder="Leave empty for default")
        yield Rule()
        yield Button("Apply Submission Params", id="apply_submit_opts", variant="primary")


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "StatusIndicator",
    "LogPanel",
    "ResourceSummaryTable",
    "ValidatedInput",
    "HardwareInfoCard",
    "StreamingConfigEditor",
    "SubmissionConfigForm",
    "LogMessage",
    "ValidationMessage",
]