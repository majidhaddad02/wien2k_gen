"""
UI Package — Terminal output formatting and SCF log analysis.
Provides Rich-based console formatting, analysis reports, and visualization.
"""

from .analysis import (
    AnalysisReport,
    ScalingMetrics,
    SCFParseResult,
    generate_report,
    parse_scf_log,
    visualize_scaling,
)
from .rich_ui import (
    CLIWorkflowRunner,
    launch_cli_mode,
    print_banner,
    print_diagnostics,
    print_error_panel,
    print_pipeline_result,
    print_table_from_dict,
    print_topology,
    print_warning_panel,
)

__all__ = [
    "AnalysisReport",
    "CLIWorkflowRunner",
    "SCFParseResult",
    "ScalingMetrics",
    "generate_report",
    "launch_cli_mode",
    "parse_scf_log",
    "print_banner",
    "print_diagnostics",
    "print_error_panel",
    "print_pipeline_result",
    "print_table_from_dict",
    "print_topology",
    "print_warning_panel",
    "visualize_scaling",
]
