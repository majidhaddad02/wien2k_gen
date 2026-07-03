"""
WIEN2k Parallel Options Generator & Parser Module.
Manages creation, validation, and persistence of `parallel_options` files
for robust MPI/OpenMP execution on modern HPC clusters.

Key Features:
• Parses legacy and modern `parallel_options` syntax (setenv/export)
• Auto-generates optimized directives based on topology, scheduler, and MPI vendor
• Disables inefficient SSH/remote calls for single-node or PMIX jobs
• Injects I/O throttling (DELAY/SLEEPY) to prevent shared filesystem storms
• Validates conflicting directives (e.g., TASKSET vs srun --cpu-bind)
• Atomic file writes with backup rotation and fallback safety
• Comprehensive type hints, structured logging, and HPC-grade error handling

All documentation and inline comments are in English per project standards.
"""

import os
import re
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TypedDict, Tuple

from ..core.topology import Topology
from ..core.hardware import get_interconnect_info, get_numa_node_count, is_containerized
from ..utils.atomic_write import atomic_write
from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Type Definitions
# =============================================================================

class ParallelOptionsDict(TypedDict, total=False):
    """
    Structured representation of WIEN2k parallel execution parameters.
    Matches variables recognized by run_lapw and lapwX scripts.
    """
    USE_REMOTE: str
    MPI_REMOTE: str
    TASKSET: str
    DELAY: float
    SLEEPY: int
    WIEN_MPIRUN: str
    OMP_GLOBAL: int
    KPAR: int
    # Legacy / backward compatibility
    RSH: str
    SSH: str
    # User-defined extensions
    custom: Dict[str, str]


# =============================================================================
# Constants & Default Profiles
# =============================================================================

# Production-tested defaults for modern HPC environments
DEFAULT_OPTIONS: Dict[str, Any] = {
    "USE_REMOTE": "0",       # Disable SSH/rsh for srun/mpirun/PMIX
    "MPI_REMOTE": "0",       # Disable legacy MPI remote spawning
    "TASKSET": "no",         # Let modern MPI launchers handle CPU binding
    "DELAY": 0.1,            # Prevent metadata storms on shared FS (NFS/Lustre/GPFS)
    "SLEEPY": 1,             # Reduce CPU spin-wait during I/O synchronization
    "WIEN_MPIRUN": "",       # Auto-detected if empty
}

# Deprecated or dangerous options that trigger warnings
DEPRECATED_KEYS = {"RSH", "SSH", "MPICH_RANK_REORDER_METHOD"}
DANGEROUS_VALUES = {
    "TASKSET": {"yes", "1"},  # Conflicts with --cpu-bind / KMP_AFFINITY
}


# =============================================================================
# Parsing & Normalization
# =============================================================================

def _normalize_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse a single line from parallel_options into (key, value) tuple.
    Supports:
      export VAR="value"
      export VAR=value
      setenv VAR value
    Ignores comments, empty lines, and malformed syntax.
    """
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("!"):
        return None
        
    # export VAR=value or export VAR="value"
    export_match = re.match(r'export\s+([A-Za-z_]\w*)\s*=\s*"?([^"\n]*)"?', line)
    if export_match:
        return export_match.group(1).upper(), export_match.group(2).strip()

    # setenv VAR value (csh/tcsh legacy)
    setenv_match = re.match(r'setenv\s+([A-Za-z_]\w*)\s+([^\s]+)', line)
    if setenv_match:
        return setenv_match.group(1).upper(), setenv_match.group(2).strip()

    return None


def parse_parallel_options(path: Union[str, Path]) -> Dict[str, str]:
    """
    Load and parse existing parallel_options file.
    Returns normalized dictionary of key-value pairs.
    Missing or unreadable files return empty dict (safe fallback).
    """
    target = Path(path)
    if not target.exists():
        return {}
        
    parsed: Dict[str, str] = {}
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            result = _normalize_line(line)
            if result:
                key, val = result
                parsed[key] = val
    except Exception as e:
        logger.warning(f"Failed to parse {target}: {e}")
        
    return parsed


# =============================================================================
# Generation & Auto-Optimization
# =============================================================================

def _detect_mpi_launcher(topo: Topology) -> str:
    """Auto-detect optimal MPI launcher command based on scheduler environment."""
    env = os.environ
    if env.get("SLURM_JOB_ID"):
        return "srun --mpi=pmix --hint=nomultithread --cpu-bind=core"
    if env.get("PBS_JOBID") or env.get("LSB_JOBID"):
        return "mpirun"
    if env.get("OMPI_COMM_WORLD_SIZE"):
        return "mpirun"
    return ""


def generate_parallel_options(
    topo: Topology,
    suggestion: Optional[Dict[str, Any]] = None,
    user_overrides: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate optimized parallel_options content string.
    Applies HPC best practices based on topology, scheduler, and MPI detection.
    
    Args:
        topo: Hardware/scheduler topology context.
        suggestion: Resource suggestion dict from optimizer (optional).
        user_overrides: Explicit key-value overrides from user/CLI.
        
    Returns:
        Formatted parallel_options file content ready for atomic write.
    """
    opts = DEFAULT_OPTIONS.copy()

    # 1. Scheduler & topology-aware tuning
    if topo.env_type in ("slurm", "pbs", "lsf"):
        opts["USE_REMOTE"] = "0"
        opts["MPI_REMOTE"] = "0"
        
    # 2. Container awareness (Singularity/Docker often break SSH/rsh)
    if is_containerized():
        opts["USE_REMOTE"] = "0"
        opts["TASKSET"] = "no"
        
    # 3. Auto-detect MPI launcher if not specified
    detected_launcher = _detect_mpi_launcher(topo)
    if detected_launcher and not opts.get("WIEN_MPIRUN"):
        opts["WIEN_MPIRUN"] = detected_launcher

    # 4. Apply suggestion hints (if provided)
    if suggestion:
        if suggestion.get("mode") == "kpoint":
            opts["KPAR"] = suggestion.get("kpar", 1)
        omp_val = suggestion.get("omp_threads_per_rank", 1)
        if omp_val > 1:
            opts["OMP_GLOBAL"] = omp_val
            
    # 5. Apply user overrides safely
    if user_overrides:
        for k, v in user_overrides.items():
            opts[k.upper()] = v

    # 6. Validate & sanitize before formatting
    warnings_list = validate_parallel_options(opts)
    if warnings_list:
        for w in warnings_list:
            logger.debug(f"Parallel options warning: {w}")

    # 7. Build formatted output with documentation comments
    lines = [
        "# ==============================================================================",
        "# Auto-generated parallel_options (wien2k_gen)",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"# Topology: {topo.env_type} | {topo.total_cores} cores | NUMA={get_numa_node_count()}",
        "# ==============================================================================",
        "#",
        "# USE_REMOTE  : Disable SSH/rsh for modern MPI launchers (srun/mpirun).",
        "# MPI_REMOTE  : Disable legacy MPI remote process spawning.",
        "# TASKSET     : 'no' allows MPI/OpenMP to manage CPU affinity natively.",
        "# DELAY       : Seconds between MPI rank startups to avoid metadata storms.",
        "# SLEEPY      : 1=reduce CPU spin-wait during I/O sync, 0=aggressive polling.",
        "# WIEN_MPIRUN : Explicit MPI launcher command (auto-detected if empty).",
        "# OMP_GLOBAL  : Global OMP threads per rank (override OMP_NUM_THREADS).",
        "# KPAR        : k-point parallelization factor (k-point mode only).",
        "#",
        "# =============================================================================="
    ]

    for key in sorted(opts.keys()):
        val = opts[key]
        if isinstance(val, float):
            val = f"{val:g}"
        lines.append(f'export {key}="{val}"')
        
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Validation & Safety Checks
# =============================================================================

def validate_parallel_options(opts: Dict[str, Any]) -> List[str]:
    """
    Validate parallel_options dictionary for conflicts, deprecated values,
    and HPC anti-patterns. Returns list of warning messages.
    """
    warnings_list: List[str] = []
    
    # Check deprecated keys
    for key in DEPRECATED_KEYS:
        if key in opts:
            warnings_list.append(f"Deprecated key '{key}' found. Modern MPI launchers ignore this.")

    # Check dangerous value combinations
    if opts.get("TASKSET", "").lower() in DANGEROUS_VALUES.get("TASKSET", set()):
        warnings_list.append(
            "TASKSET='yes' conflicts with modern CPU binding (--cpu-bind, KMP_AFFINITY, OMP_PLACES). "
            "Set TASKSET='no' for optimal performance."
        )

    # Validate numeric ranges
    try:
        delay = float(opts.get("DELAY", 0))
        if delay < 0:
            warnings_list.append("DELAY cannot be negative. Resetting to 0.")
            opts["DELAY"] = 0
        elif delay > 2.0:
            warnings_list.append("DELAY > 2.0 may cause excessive job startup latency.")
    except ValueError:
        warnings_list.append("DELAY must be a numeric value.")

    try:
        sleepy = int(opts.get("SLEEPY", 0))
        if sleepy not in (0, 1):
            warnings_list.append("SLEEPY must be 0 (aggressive) or 1 (yield).")
    except ValueError:
        warnings_list.append("SLEEPY must be an integer (0 or 1).")

    # Cross-check with environment
    if os.getenv("SLURM_HINT", "").lower() == "nomultithread" and str(opts.get("OMP_GLOBAL", "1")) == "1":
        warnings_list.append(
            "SLURM_HINT=nomultithread detected but OMP_GLOBAL=1. "
            "Verify if OpenMP parallelism is intentionally disabled."
        )

    return warnings_list


# =============================================================================
# Atomic Write & File Management
# =============================================================================

def write_parallel_options(
    path: Union[str, Path] = "parallel_options",
    topo: Optional[Topology] = None,
    suggestion: Optional[Dict[str, Any]] = None,
    user_overrides: Optional[Dict[str, str]] = None,
    backup: bool = True
) -> bool:
    """
    Generate, validate, and atomically write parallel_options file.
    Handles backup rotation, permission preservation, and error recovery.
    
    Args:
        path: Output file path (default: 'parallel_options').
        topo: Hardware/scheduler topology context.
        suggestion: Resource suggestion from optimizer.
        user_overrides: Explicit key-value overrides.
        backup: Create timestamped backup if file already exists.
        
    Returns:
        True on successful write, False on failure.
    """
    target = Path(path)

    # 1. Generate content
    try:
        content = generate_parallel_options(
            topo=topo or Topology(nodes=["localhost"], cores_per_node=[1]),
            suggestion=suggestion,
            user_overrides=user_overrides
        )
    except Exception as e:
        logger.error(f"Failed to generate parallel_options content: {e}")
        return False

    # 2. Backup existing file
    if backup and target.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = target.parent / f"{target.name}.bak.{ts}"
        try:
            shutil.copy2(target, backup_path)
            logger.debug(f"Backed up {target} to {backup_path}")
        except Exception as e:
            logger.warning(f"Backup failed for {target}: {e}")

    # 3. Atomic write
    try:
        success = atomic_write(target, content, mode=0o644)
        if success:
            logger.info(f"Written optimized parallel_options to {target}")
        return success
    except Exception as e:
        logger.error(f"Atomic write failed for {target}: {e}")
        return False


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ParallelOptionsDict",
    "parse_parallel_options",
    "generate_parallel_options",
    "validate_parallel_options",
    "write_parallel_options",
    "DEFAULT_OPTIONS",
]