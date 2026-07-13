"""
Job Submission & Scheduler Integration Package.
Provides production-grade SLURM, LSF, and PBS/Torque script generation, submission handling,
and cluster resource management with preemption resilience, NUMA awareness,
and HPC best practices.

Submodules:
• slurm: Advanced SBATCH template generation, job validation, atomic script writing,
  and direct `sbatch` submission with structured result tracking.
• lsf: LSF submit provider with #BSUB directive generation, bsub/bkill/bjobs integration,
  job array support, and optional jsrun for IBM Spectrum LSF.
• pbs: PBS/Torque submit provider with #PBS directive generation, qsub/qdel/qstat integration,
  job array support, and multi-node scratch synchronization.

Designed for seamless integration with the forge pipeline, UI wizards,
and automated HPC workflow dispatchers.
"""

# =============================================================================
# Core SLURM Integration Exports
# =============================================================================
# =============================================================================
# LSF Submit Provider Exports
# =============================================================================
from .lsf import (
    LSFDirectives,
    LSFJobSpec,
    LSFSubmitProvider,
    SubmitProvider,
    _check_lsf_limits,
    _validate_lsf_memory,
    _validate_lsf_time,
)

# =============================================================================
# PBS/Torque Submit Provider Exports
# =============================================================================
from .pbs import (
    PBSDirectives,
    PBSJobSpec,
    PBSSubmitProvider,
    _check_pbs_limits,
    _validate_pbs_memory,
    _validate_pbs_time,
)
from .slurm import (
    SlurmDirectives,
    SlurmJobSpec,
    SubmissionResult,
    _check_scheduler_limits,
    _validate_memory_string,
    _validate_time_format,
    generate_sbatch_script,
    submit_slurm_job,
)

# =============================================================================
# Provider Registry
# =============================================================================
# Maps scheduler backend keys to provider classes for dynamic dispatch.
SUBMIT_PROVIDERS = {
    "slurm": None,  # SLURM uses functional API; not yet class-based
    "lsf": LSFSubmitProvider,
    "pbs": PBSSubmitProvider,
    "torque": PBSSubmitProvider,
}


# =============================================================================
# Explicit Public API Declaration
# =============================================================================
# Controls `from forge.submit import *` and provides clear IDE auto-completion boundaries.
# Only exposes production-ready interfaces; internal helpers remain encapsulated.
__all__ = [
    "SUBMIT_PROVIDERS",
    # Data Structures — LSF
    "LSFDirectives",
    "LSFJobSpec",
    # Providers
    "LSFSubmitProvider",
    # Data Structures — PBS/Torque
    "PBSDirectives",
    "PBSJobSpec",
    "PBSSubmitProvider",
    # Data Structures — SLURM
    "SlurmDirectives",
    "SlurmJobSpec",
    # Shared
    "SubmissionResult",
    # Abstract Base
    "SubmitProvider",
    "_check_lsf_limits",
    "_check_pbs_limits",
    "_check_scheduler_limits",
    "_validate_lsf_memory",
    # Validation Utilities — LSF
    "_validate_lsf_time",
    "_validate_memory_string",
    "_validate_pbs_memory",
    # Validation Utilities — PBS
    "_validate_pbs_time",
    # Validation Utilities — SLURM
    "_validate_time_format",
    # Core Functions — SLURM
    "generate_sbatch_script",
    "submit_slurm_job",
]
