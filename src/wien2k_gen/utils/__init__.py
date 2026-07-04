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

Designed for seamless integration with the wien2k_gen pipeline, CLI wizards, and UI monitors.
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
    DiagnosticReport,
    DiagnosticConfig,
    run_diagnostics,
    export_diagnostics_json,
)

# =============================================================================
# Data Export & Serialization
# =============================================================================
from .export import (
    ExportResult,
    ExportConfig,
    export_config,
    export_multiple,
)

# =============================================================================
# Process-Safe File Locking
# =============================================================================
from .filelock import (
    FileLock,
    LockAcquisitionError,
    LockTimeoutError,
    file_lock,
    DEFAULT_TIMEOUT,
)

# =============================================================================
# WIEN2k Parallel Options Management
# =============================================================================
from .parallel_options import (
    ParallelOptionsDict,
    parse_parallel_options,
    generate_parallel_options,
    validate_parallel_options,
    write_parallel_options,
    DEFAULT_OPTIONS,
)

# =============================================================================
# Scratch Space & Multi-Node Staging
# =============================================================================
from .scratch import (
    ScratchResult,
    ScratchConfig,
    setup_scratch,
    cleanup_scratch,
    configure_lustre_striping,
)

# =============================================================================
# Subprocess Execution & Lifecycle Management
# =============================================================================
from .subprocess_utils import (
    ProcessResult,
    run_command,
    run_async_command,
    stream_command_output,
    terminate_process_group,
    force_kill_process_group,
)

# =============================================================================
# Configuration Validation & Backup
# =============================================================================
from .validation import (
    ValidationResult,
    MachinesConfig,
    parse_machines_file,
    validate_machines,
    backup_machines,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
# Controls `from wien2k_gen.utils import *` and provides clear IDE auto-completion boundaries.
# Only exports production-ready interfaces; internal helpers remain encapsulated.
__all__ = [
    # atomic_write
    "atomic_write",
    "atomic_write_json",
    "atomic_write_list",
    # diagnostic
    "DiagnosticReport",
    "DiagnosticConfig",
    "run_diagnostics",
    "export_diagnostics_json",
    # export
    "ExportResult",
    "ExportConfig",
    "export_config",
    "export_multiple",
    # filelock
    "FileLock",
    "LockAcquisitionError",
    "LockTimeoutError",
    "file_lock",
    "DEFAULT_TIMEOUT",
    # parallel_options
    "ParallelOptionsDict",
    "parse_parallel_options",
    "generate_parallel_options",
    "validate_parallel_options",
    "write_parallel_options",
    "DEFAULT_OPTIONS",
    # scratch
    "ScratchResult",
    "ScratchConfig",
    "setup_scratch",
    "cleanup_scratch",
    "configure_lustre_striping",
    # subprocess_utils
    "ProcessResult",
    "run_command",
    "run_async_command",
    "stream_command_output",
    "terminate_process_group",
    "force_kill_process_group",
    # validation
    "ValidationResult",
    "MachinesConfig",
    "parse_machines_file",
    "validate_machines",
    "backup_machines",
]