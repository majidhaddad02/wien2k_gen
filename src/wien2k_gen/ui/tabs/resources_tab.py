"""
Resources Tab – Hardware Detection & Parallel Allocation Interface.
Provides an interactive, reactive UI for topology discovery, core/memory allocation,
NUMA/SMT policy selection, and constraint validation. Designed for seamless integration
with async workers, scheduler detection, and the central optimization pipeline.

Key Architecture Features:
• Reactive state binding with automatic UI refreshes on topology changes
• Thread-safe hardware detection via app-level workers or local async dispatch
• Real-time validation of core/thread/memory constraints against detected limits
• NUMA-aware binding hints and SMT exclusion toggles for production HPC runs
• Structured error boundaries, fallback defaults, and comprehensive logging
• Modern Textual layout with collapsible advanced options and responsive grids
• Comprehensive English documentation, type hints, and HPC-grade resilience patterns
All documentation and inline comments are in English per project standards.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Grid, Container
from textual.widgets import Button, Input, Label, Static, Checkbox, Select, Rule, Switch, Collapsible
from textual.reactive import reactive
from textual.message import Message
from textual import on

# Project imports
from ...core.topology import Topology
from ...core.scheduler import detect as detect_topology
from ...core.hardware import get_physical_cores, get_job_memory_limit_mb, get_total_mem_kb
from ...optimizer.advisor import suggest_optimal_resources
from ...logging_config import get_logger
from ..widgets import (
    StatusIndicator,
    LogPanel,
    ResourceSummaryTable,
    ValidatedInput,
    HardwareInfoCard,
    ValidationMessage
)

logger = get_logger(__name__)


# =============================================================================
# Custom Messages for Tab-App Communication
# =============================================================================

class TopologyDetectedMessage(Message, bubble=True):
    """Emitted when hardware detection completes successfully."""
    def __init__(self, topology: Topology) -> None:
        super().__init__()
        self.topology = topology


class AllocationChangedMessage(Message, bubble=True):
    """Emitted when user modifies resource allocation parameters."""
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self.params = params


# =============================================================================
# Resources Tab Implementation
# =============================================================================

class ResourcesTab(Container):
    """
    Tab for hardware discovery, topology visualization, and parallel resource allocation.
    Integrates with scheduler detection, advisor optimization, and validation pipelines.
    """
    DEFAULT_CSS = """
    ResourcesTab {
        layout: vertical;
        height: 1fr;
        padding: 0 1;
    }
    #detection_header {
        height: auto;
        margin: 1 0;
        align: center middle;
    }
    #detection_controls {
        height: auto;
        margin: 1 0;
        align: center middle;
    }
    #detection_controls Button {
        width: 40%;
        margin: 0 2;
    }
    #topo_panel {
        height: auto;
        max-height: 14;
        margin: 1 0;
        border: solid $primary;
    }
    #hw_card {
        height: auto;
        margin: 1 0;
    }
    #alloc_form {
        height: auto;
        margin: 1 0;
        border: dashed $accent;
        padding: 1;
    }
    .alloc-row {
        height: auto;
        margin: 1 0;
        align: left middle;
    }
    .alloc-label {
        width: 14;
        text-align: right;
        padding-right: 1;
        text-style: bold;
    }
    .alloc-input {
        width: 20;
    }
    .alloc-toggle {
        height: auto;
        margin: 1 0;
    }
    #advanced_panel {
        height: auto;
        margin: 1 0;
    }
    #validation_box {
        height: auto;
        max-height: 8;
        margin: 1 0;
        padding: 0 1;
        background: $panel;
        border: solid $warning;
    }
    """

    # Reactive State
    topology: Optional[Topology] = reactive(None)
    is_detecting: bool = reactive(False)
    max_cores: int = reactive(0)
    nodes_count: int = reactive(1)
    omp_threads: int = reactive(1)
    memory_limit_mb: str = reactive("0")
    numa_binding: bool = reactive(True)
    disable_smt: bool = reactive(True)
    validation_state: List[str] = reactive([])

    def on_mount(self) -> None:
        """Initialize default values and trigger auto-detection if no topology exists."""
        self.log.info("ResourcesTab mounted. Initializing allocation UI...")
        self._apply_defaults()
        self.call_later(self._run_auto_detection)

    def compose(self) -> ComposeResult:
        """Build tab layout with detection controls, topology display, and allocation form."""
        yield Static("Hardware Detection & Resource Allocation", id="detection_header")
        
        with Horizontal(id="detection_controls"):
            yield Button("Detect Hardware", id="btn_detect", variant="primary")
            yield Button("Reset Defaults", id="btn_reset", variant="default")

        with Container(id="topo_panel"):
            yield ResourceSummaryTable(id="topo_table")

        yield HardwareInfoCard(id="hw_card")

        with Container(id="alloc_form"):
            yield Static("Parallel Resource Allocation", classes="title")
            
            with Horizontal(classes="alloc-row"):
                yield Label("Max Cores: ", classes="alloc-label")
                yield ValidatedInput(
                    id="inp_max_cores",
                    value_type="positive_int",
                    value="0",
                    classes="alloc-input",
                    placeholder="0 = auto"
                )
                
            with Horizontal(classes="alloc-row"):
                yield Label("Nodes: ", classes="alloc-label")
                yield ValidatedInput(
                    id="inp_nodes",
                    value_type="positive_int",
                    value="1",
                    classes="alloc-input"
                )
                
            with Horizontal(classes="alloc-row"):
                yield Label("OMP Threads: ", classes="alloc-label")
                yield ValidatedInput(
                    id="inp_omp",
                    value_type="positive_int",
                    value="1",
                    classes="alloc-input"
                )
                
            with Horizontal(classes="alloc-row"):
                yield Label("Mem Limit (MB): ", classes="alloc-label")
                yield ValidatedInput(
                    id="inp_mem",
                    value_type="non_zero_int",
                    value="0",
                    classes="alloc-input",
                    placeholder="0 = system limit"
                )

            yield Rule()
            
            with Horizontal(classes="alloc-toggle"):
                yield Label("NUMA Binding: ")
                yield Switch(id="sw_numa", value=True)
                yield Label("Disable SMT/HT: ")
                yield Switch(id="sw_smt", value=True)

        with Collapsible(title="Advanced Allocation Constraints", id="advanced_panel"):
            yield Static("Fine-grained tuning for scheduler limits, memory bands, and I/O strategies.")
            yield Checkbox("Force Single-Socket Allocation", id="chk_single_socket")
            yield Checkbox("Enable Vector Split for I/O Bottlenecks", id="chk_vector_split", value=True)
            yield Checkbox("Strict Scheduler Limit Enforcement", id="chk_strict_limits", value=True)

        yield Static("", id="validation_box")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    @on(Button.Pressed, "#btn_detect")
    def on_detect_pressed(self, event: Button.Pressed) -> None:
        """Route button clicks to detection logic."""
        self._run_detection()

    @on(Button.Pressed, "#btn_reset")
    def on_reset_pressed(self, event: Button.Pressed) -> None:
        """Route button clicks to reset logic."""
        self._apply_defaults()
        self.notify("Allocation reset to defaults.", severity="information")

    @on(Input.Changed)
    def on_validated_input_changed(self, event: Input.Changed) -> None:
        """Handle real-time validation and emit allocation updates."""
        self.call_later(self._validate_and_emit)

    @on(Switch.Changed)
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Toggle NUMA/SMT policies and revalidate."""
        self.call_later(self._validate_and_emit)

    @on(ValidationMessage)
    def on_validation_message(self, message: ValidationMessage) -> None:
        """Capture validation feedback from custom inputs."""
        self._validate_and_emit()

    # =========================================================================
    # Reactive Watchers
    # =========================================================================

    def watch_topology(self, topo: Optional[Topology]) -> None:
        """Update UI components when topology changes."""
        if not topo:
            return
            
        self.call_later(lambda: self.query_one("#topo_table").set_reactive("topology", topo))
        
        # Update hardware card
        hw_data = {
            "environment": topo.env_type.upper(),
            "total_cores": topo.total_cores,
            "nodes": len(topo.nodes),
            "launcher": topo.scheduler_hints.get("mpi_launcher", "unknown"),
            "mem_per_core": f"{(get_total_mem_kb() / 1024) / max(1, get_physical_cores()):.0f} MB"
        }
        self.call_later(lambda: self.query_one("#hw_card").set_reactive("data", hw_data))
        
        # Auto-populate limits if currently zero
        if self.max_cores == 0:
            self.max_cores = topo.total_cores
            self.nodes_count = max(1, len(topo.nodes))
            self.call_later(self._sync_inputs)
            
        self._validate_and_emit()

    def watch_validation_state(self, msgs: List[str]) -> None:
        """Render validation box with color-coded messages."""
        box = self.query_one("#validation_box", Static)
        if not msgs:
            box.update("[green]✓ All constraints satisfied.[/]")
            return
            
        lines = []
        for m in msgs:
            prefix = "⚠" if "warning" in m.lower() else "✗"
            color = "yellow" if prefix == "⚠" else "red"
            lines.append(f"[{color}]{prefix} {m}[/]")
        box.update("\n".join(lines))

    # =========================================================================
    # Core Logic: Detection, Validation & State Sync
    # =========================================================================

    def _run_auto_detection(self) -> None:
        """Run initial hardware detection on tab mount."""
        if self.topology:
            return
        self._run_detection()

    def _run_detection(self) -> None:
        """Execute scheduler detection asynchronously without blocking UI."""
        if self.is_detecting:
            self.notify("Detection already in progress.", severity="warning")
            return
            
        self.is_detecting = True
        btn = self.query_one("#btn_detect", Button)
        btn.disabled = True
        btn.label = "Detecting..."
        self.notify("Scanning hardware & scheduler environment...", severity="information")

        def _detect_task() -> None:
            try:
                topo = detect_topology(max_cores=None, force_refresh=True)
                self.call_later(lambda: self.post_message(TopologyDetectedMessage(topo)))
            except Exception as e:
                logger.error(f"Hardware detection failed: {e}", exc_info=True)
                self.call_later(lambda: self.notify(f"Detection failed: {e}", severity="error"))
            finally:
                self.call_later(self._on_detection_complete)

        # Run in daemon thread to avoid blocking Textual event loop
        threading.Thread(target=_detect_task, daemon=True).start()

    def _on_detection_complete(self) -> None:
        """Reset UI state after detection finishes."""
        self.is_detecting = False
        btn = self.query_one("#btn_detect", Button)
        btn.disabled = False
        btn.label = "Detect Hardware"

    def _apply_defaults(self) -> None:
        """Reset allocation inputs to safe defaults."""
        self.max_cores = 0
        self.nodes_count = 1
        self.omp_threads = 1
        self.memory_limit_mb = "0"
        self.numa_binding = True
        self.disable_smt = True
        self._sync_inputs()

    def _sync_inputs(self) -> None:
        """Sync reactive state with input widgets."""
        self.call_later(lambda: self.query_one("#inp_max_cores").update(str(self.max_cores)))
        self.call_later(lambda: self.query_one("#inp_nodes").update(str(self.nodes_count)))
        self.call_later(lambda: self.query_one("#inp_omp").update(str(self.omp_threads)))
        self.call_later(lambda: self.query_one("#inp_mem").update(self.memory_limit_mb))
        self.call_later(lambda: setattr(self.query_one("#sw_numa", Switch), "value", self.numa_binding))
        self.call_later(lambda: setattr(self.query_one("#sw_smt", Switch), "value", self.disable_smt))

    def _validate_and_emit(self) -> None:
        """Validate allocation constraints and emit message to app/pipeline."""
        warnings_list = []
        errors_list = []
        
        # Read current inputs
        try:
            max_c = int(self.query_one("#inp_max_cores").value or "0")
            nodes = max(1, int(self.query_one("#inp_nodes").value or "1"))
            omp = max(1, int(self.query_one("#inp_omp").value or "1"))
            mem = int(self.query_one("#inp_mem").value or "0")
        except ValueError:
            self.validation_state = ["Invalid numeric input detected."]
            return
            
        # Topology bounds check
        if self.topology:
            total_avail = self.topology.total_cores
            if max_c > 0 and max_c > total_avail:
                errors_list.append(f"Max cores ({max_c}) exceeds available ({total_avail}).")
            if nodes > len(self.topology.nodes):
                warnings_list.append(f"Requested nodes ({nodes}) exceeds allocated ({len(self.topology.nodes)}).")
                
        # Divisibility & Oversubscription check
        if max_c > 0 and max_c % omp != 0:
            warnings_list.append(f"Total cores ({max_c}) not divisible by OMP threads ({omp}). Ranks will be uneven.")
            
        # Memory limit check
        sys_limit_mb = get_job_memory_limit_mb() or (get_total_mem_kb() / 1024)
        if mem > 0 and mem > sys_limit_mb:
            errors_list.append(f"Memory limit ({mem} MB) exceeds job/system limit ({sys_limit_mb:.0f} MB).")
            
        # NUMA/SMT hints
        if self.disable_smt and self.topology and not self.topology.scheduler_hints.get("numa_aware"):
            warnings_list.append("SMT disabled but NUMA binding not detected. May underutilize cores on some clusters.")
            
        self.validation_state = errors_list + warnings_list
        
        if not errors_list:
            # Emit valid allocation to parent app/pipeline
            params = {
                "max_cores": max_c,
                "nodes": nodes,
                "omp_threads": omp,
                "memory_limit_mb": mem,
                "numa_binding": self.numa_binding,
                "disable_smt": self.disable_smt,
                "vector_split": self.query_one("#chk_vector_split", Checkbox).value,
                "strict_limits": self.query_one("#chk_strict_limits", Checkbox).value
            }
            self.post_message(AllocationChangedMessage(params))

    def get_allocation_params(self) -> Dict[str, Any]:
        """Return current allocation configuration for pipeline consumption."""
        return {
            "max_cores": self.max_cores,
            "nodes": self.nodes_count,
            "omp_threads": self.omp_threads,
            "memory_limit_mb": int(self.memory_limit_mb) if self.memory_limit_mb else 0,
            "numa_binding": self.numa_binding,
            "disable_smt": self.disable_smt,
        }


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ResourcesTab",
    "TopologyDetectedMessage",
    "AllocationChangedMessage",
]