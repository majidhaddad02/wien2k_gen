"""
Utility Package Initialization for HPC/DFT Workflows.
Provides robust, production-grade helpers for file I/O, process management,
configuration validation, scratch staging, diagnostics, and data export.

Submodules:
• atomic_write: Guaranteed atomic file operations with fsync & automatic cleanup
• diagnostic: System/hardware/software stack collection & structured reporting
• export: Multi-format serialization (JSON, YAML, TOML, CSV, TXT) with scientific encoders
• filelock: Process-safe locking with fcntl + atomic directory fallback for NFS/Lustre
• parallel_options: WIEN2k parallel execution config generation, parsing & validation
• scratch: Multi-node scratch selection, staging (sbcast/rsync), & safe cleanup
• subprocess_utils: Safe sync/async subprocess execution & process group (PGID) management
• validation: Rigorous .machines parsing, consistency checks, topology alignment & backup rotation

Designed for seamless integration with the forge pipeline, CLI wizards, and UI monitors.
"""

# =============================================================================
# Atomic File Operations
# =============================================================================
from .atomic_write import (
    atomic_write,
    atomic_write_json,
    atomic_write_list,
)

# =============================================================================
# System & Environment Diagnostics
# =============================================================================
from .diagnostic import (
    DiagnosticConfig,
    DiagnosticReport,
    export_diagnostics_json,
    run_diagnostics,
)

# =============================================================================
# Data Export & Serialization
# =============================================================================
from .export import (
    ExportConfig,
    ExportResult,
    export_config,
    export_multiple,
)

# =============================================================================
# Process-Safe File Locking
# =============================================================================
from .filelock import (
    DEFAULT_TIMEOUT,
    FileLock,
    LockAcquisitionError,
    LockTimeoutError,
    file_lock,
)

# =============================================================================
# WIEN2k Parallel Options Management
# =============================================================================
from .parallel_options import (
    DEFAULT_OPTIONS,
    ParallelOptionsDict,
    generate_parallel_options,
    parse_parallel_options,
    validate_parallel_options,
    write_parallel_options,
)

# =============================================================================
# Scratch Space & Multi-Node Staging
# =============================================================================
from .scratch import (
    ScratchConfig,
    ScratchResult,
    cleanup_scratch,
    configure_lustre_striping,
    setup_scratch,
)

# =============================================================================
# Subprocess Execution & Lifecycle Management
# =============================================================================
from .subprocess_utils import (
    ProcessResult,
    force_kill_process_group,
    run_async_command,
    run_command,
    stream_command_output,
    terminate_process_group,
)

# =============================================================================
# Configuration Validation & Backup
# =============================================================================
from .validation import (
    MachinesConfig,
    ValidationResult,
    backup_machines,
    parse_machines_file,
    validate_machines,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
# Controls `from forge.utils import *` and provides clear IDE auto-completion boundaries.
# Only exports production-ready interfaces; internal helpers remain encapsulated.
__all__ = [
    "DEFAULT_OPTIONS",
    "DEFAULT_TIMEOUT",
    "DiagnosticConfig",
    # diagnostic
    "DiagnosticReport",
    "ExportConfig",
    # export
    "ExportResult",
    # filelock
    "FileLock",
    "LockAcquisitionError",
    "LockTimeoutError",
    "MachinesConfig",
    # parallel_options
    "ParallelOptionsDict",
    # subprocess_utils
    "ProcessResult",
    "ScratchConfig",
    # scratch
    "ScratchResult",
    # validation
    "ValidationResult",
    # atomic_write
    "atomic_write",
    "atomic_write_json",
    "atomic_write_list",
    "backup_machines",
    "cleanup_scratch",
    "configure_lustre_striping",
    "export_config",
    "export_diagnostics_json",
    "export_multiple",
    "file_lock",
    "force_kill_process_group",
    "generate_parallel_options",
    "parse_machines_file",
    "parse_parallel_options",
    "run_async_command",
    "run_command",
    "run_diagnostics",
    "setup_scratch",
    "stream_command_output",
    "terminate_process_group",
    "validate_machines",
    "validate_parallel_options",
    "write_parallel_options",
]