"""
Wien2kGen – Production-Grade Parallel Configuration & HPC Job Dispatcher.
Automates WIEN2k, VASP, and Quantum ESPRESSO setup with topology-aware optimization,
SLURM integration, benchmarking, and interactive TUI/CLI workflows.

Package Structure:
• core/: Hardware topology, scheduler detection, pipeline orchestration
• optimizer/: Resource advisor, SCF monitor, statistical profiler
• backends/: WIEN2k, VASP, QE, CP2K configuration generators
• submit/: SLURM script generation, job validation, submission tracking
• utils/: Atomic I/O, file locking, scratch staging, diagnostics, validation
• ui/: Textual TUI, Rich CLI fallback, async workers, analysis engine
• benchmark/: Synthetic simulation, real-cluster execution, calibration
• api/: REST API server and web dashboard for monitoring and job management
"""

__version__ = "0.1.0"
__author__ = "HPC Workflow Team"
__license__ = "MIT"

from typing import Any, List

# =============================================================================
# Lazy Import & API Exposure (PEP 562)
# =============================================================================

def __getattr__(name: str) -> Any:
    """
    Lazy attribute resolution for fast startup & circular-import prevention.
    Enables `from wien2k_gen import launch_tui, get_config, Topology, etc.` 
    without upfront import cost or dependency resolution at package load time.
    """
    # Core Scheduler & Topology
    if name == "detect":
        from .core.scheduler import detect
        return detect
    if name == "Topology":
        from .core.topology import Topology
        return Topology
        
    # Optimizer & Advisor
    if name == "suggest_optimal_resources":
        from .optimizer.advisor import suggest_optimal_resources
        return suggest_optimal_resources
    if name == "load_cached_suggestion":
        from .optimizer.advisor import load_cached_suggestion
        return load_cached_suggestion
    if name == "save_cached_suggestion":
        from .optimizer.advisor import save_cached_suggestion
        return save_cached_suggestion
    if name == "recommend":  # Alias for suggest_optimal_resources
        from .optimizer.advisor import suggest_optimal_resources
        return suggest_optimal_resources
        
    # Builder (Pipeline Execution)
    if name == "build_auto":
        from .core.builder import build_auto
        return build_auto
    if name == "build_mpi":
        from .core.builder import build_mpi
        return build_mpi
    if name == "build_hybrid":
        from .core.builder import build_hybrid
        return build_hybrid
    if name == "build_kpoint":
        from .core.builder import build_kpoint
        return build_kpoint
        
    # Backend Management (Aligned with backend_manager.py API)
    if name == "get_backend":
        from .backend_manager import get_backend
        return get_backend
    if name == "set_backend":
        from .backend_manager import set_backend
        return set_backend
    if name == "list_backends":
        from .backend_manager import list_backends
        return list_backends
    if name == "get_current_backend":
        from .backend_manager import get_backend
        return get_backend
        
    # Utils & Scratch
    if name == "setup_scratch":
        from .utils.scratch import setup_scratch
        return setup_scratch
    if name == "cleanup_scratch":
        from .utils.scratch import cleanup_scratch
        return cleanup_scratch
    if name == "write_parallel_options":
        from .utils.parallel_options import write_parallel_options
        return write_parallel_options
        
    # Interactive & Wizards
    if name == "run_wizard":
        from .wizard import run_wizard
        return run_wizard
    if name == "run_sbatch_wizard":
        from .wizard_sbatch import run_sbatch_wizard
        return run_sbatch_wizard
    if name == "launch_tui":
        from .ui.interactive import launch_app
        return launch_app
        
    # Config & Logging (Explicit access)
    if name == "get_config":
        from .config import get_config
        return get_config
    if name == "load_config":
        from .config import load_config
        return load_config
        
    # Types
    if name == "PipelineResult":
        from .types import PipelineResult
        return PipelineResult
    if name == "ResourceSuggestion":
        from .types import ResourceSuggestion
        return ResourceSuggestion
    if name == "TopologyData":
        from .types import TopologyData
        return TopologyData
    if name == "BackendCode":
        from .types import BackendCode
        return BackendCode
    if name == "ExecutionMode":
        from .types import ExecutionMode
        return ExecutionMode
    if name == "JobStatus":
        from .types import JobStatus
        return JobStatus
    if name == "OptimizationTarget":
        from .types import OptimizationTarget
        return OptimizationTarget
    if name == "CalculationType":
        from .types import CalculationType
        return CalculationType
    if name == "Wien2kVersion":
        from .types import Wien2kVersion
        return Wien2kVersion
    if name == "Wien2kFlags":
        from .types import Wien2kFlags
        return Wien2kFlags

    # History & Bayesian optimization
    if name == "ExecutionHistory":
        from .optimizer.history import ExecutionHistory
        return ExecutionHistory
    if name == "BayesianOptimizer":
        from .optimizer.bayesian import BayesianOptimizer
        return BayesianOptimizer
    if name == "suggest_from_history":
        from .optimizer.history import suggest_from_history
        return suggest_from_history

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> List[str]:
    """
    Support IDE auto-completion & dir() introspection for lazy attributes.
    """
    return list(globals().keys()) + [
        "detect", "Topology", "suggest_optimal_resources", "recommend",
        "build_auto", "build_mpi", "build_hybrid", "build_kpoint",
        "get_backend", "set_backend", "list_backends", "get_current_backend",
        "setup_scratch", "cleanup_scratch", "write_parallel_options",
        "run_wizard", "run_sbatch_wizard", "launch_tui",
        "get_config", "load_config",
        "PipelineResult", "ResourceSuggestion", "TopologyData",
        "BackendCode", "ExecutionMode", "JobStatus", "OptimizationTarget",
        "CalculationType", "Wien2kVersion", "Wien2kFlags",
        "ExecutionHistory", "BayesianOptimizer", "suggest_from_history",
        "__version__"
    ]


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "__version__",
    # Core
    "detect", "Topology",
    # Optimization
    "suggest_optimal_resources", "recommend",
    "load_cached_suggestion", "save_cached_suggestion",
    # Builders
    "build_auto", "build_mpi", "build_hybrid", "build_kpoint",
    # Backends
    "get_backend", "set_backend", "list_backends", "get_current_backend",
    # Utils
    "setup_scratch", "cleanup_scratch", "write_parallel_options",
    # Interactive
    "run_wizard", "run_sbatch_wizard", "launch_tui",
    # Config
    "get_config", "load_config",
    # Types
    "PipelineResult", "ResourceSuggestion", "TopologyData",
    "BackendCode", "ExecutionMode", "JobStatus", "OptimizationTarget",
    # History & Bayesian
    "ExecutionHistory", "BayesianOptimizer", "suggest_from_history",
]