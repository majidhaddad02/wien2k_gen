"""
Tab Package Initialization for Wien2kGen TUI.
Exports the main interface tabs for resource allocation, environment settings,
job submission, and advanced tuning (Bayesian, ELPA, GPU, profiling).
Designed for modular integration with the main interactive app.

Submodules:
• resources_tab: Hardware detection, core/memory allocation, NUMA/SMT policies
• settings_tab: Backend selection, path management, UI/logging preferences, profile system
• submit_tab: SLURM configuration, script preview, job submission & tracking
• advanced_tab: Bayesian optimization, ELPA solver, GPU offload, performance profiling
"""

# =============================================================================
# Tab Component Exports
# =============================================================================
from .resources_tab import ResourcesTab
from .settings_tab import SettingsTab
from .submit_tab import SubmitTab
from .advanced_tab import AdvancedTab

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
# Controls `from wien2k_gen.ui.tabs import *` and provides clear IDE auto-completion boundaries.
# Only exports production-ready tab classes; internal helpers remain encapsulated.
__all__ = [
    "ResourcesTab",
    "SettingsTab",
    "SubmitTab",
    "AdvancedTab",
]