"""
UI Package Initialization for Wien2kGen TUI.
Exports the main interactive application, CLI fallback entrypoints, shared UI components,
and analysis/worker modules. Designed for modular, type-safe integration with the core HPC pipeline.
"""

# =============================================================================
# Main TUI Application & CLI Fallbacks
# =============================================================================
from .interactive import Wien2kGenApp, launch_app
from .rich_ui import (
    print_banner,
    print_config_summary,
    print_error_report,
    launch_cli_mode,
    CLIWorkflowRunner,
)

# =============================================================================
# Reusable UI Components
# =============================================================================
from .widgets import (
    StatusIndicator,
    LogPanel,
    ResourceSummaryTable,
    ValidatedInput,
)

# =============================================================================
# Background Task Orchestrators
# =============================================================================
from .workers import (
    HPCWorkerOrchestrator,
    BackgroundTask,
    WorkerConfig,
    WorkerStartedMessage,
    WorkerCompletedMessage,
    WorkerFailedMessage,
)

# =============================================================================
# Log Analysis & Scaling Engine
# =============================================================================
from .analysis import (
    parse_scf_log,
    visualize_scaling,
    generate_report,
    SCFParseResult,
    ScalingMetrics,
    AnalysisReport,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    # App & CLI
    "Wien2kGenApp",
    "launch_app",
    "launch_cli_mode",
    "CLIWorkflowRunner",
    "print_banner",
    "print_config_summary",
    "print_error_report",
    # Widgets
    "StatusIndicator",
    "LogPanel",
    "ResourceSummaryTable",
    "ValidatedInput",
    # Workers
    "HPCWorkerOrchestrator",
    "BackgroundTask",
    "WorkerConfig",
    "WorkerStartedMessage",
    "WorkerCompletedMessage",
    "WorkerFailedMessage",
    # Analysis
    "parse_scf_log",
    "visualize_scaling",
    "generate_report",
    "SCFParseResult",
    "ScalingMetrics",
    "AnalysisReport",
]