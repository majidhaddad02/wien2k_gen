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
    print_error_panel,
    print_warning_panel,
    print_topology,
    print_pipeline_result,
    print_diagnostics,
    print_table_from_dict,
)

__all__ = [
    "CLIWorkflowRunner",
    "launch_cli_mode",
    "print_banner",
    "print_error_panel",
    "print_warning_panel",
    "print_topology",
    "print_pipeline_result",
    "print_diagnostics",
    "print_table_from_dict",
    "parse_scf_log",
    "visualize_scaling",
    "generate_report",
    "SCFParseResult",
    "ScalingMetrics",
    "AnalysisReport",
]
