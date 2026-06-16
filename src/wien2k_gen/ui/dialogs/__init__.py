"""
Dialog Package Initialization for Wien2kGen TUI.
Exports modal screens for help documentation, profile management, and structured report visualization.
Designed for seamless integration with the main interactive application and reactive state management.
"""

# =============================================================================
# Dialog Component Exports
# =============================================================================
from .help_dialog import HelpDialog
from .profile_dialog import ProfileDialog
from .report_dialog import ReportDialog

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
__all__ = [
    "HelpDialog",
    "ProfileDialog",
    "ReportDialog",
]