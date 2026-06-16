"""
Job Submission & Scheduler Integration Package.
Provides production-grade SLURM script generation, submission handling,
and cluster resource management with preemption resilience, NUMA awareness,
and HPC best practices.

Submodules:
• slurm: Advanced SBATCH template generation, job validation, atomic script writing,
  and direct `sbatch` submission with structured result tracking.

Designed for seamless integration with the wien2k_gen pipeline, UI wizards,
and automated HPC workflow dispatchers.
"""

# =============================================================================
# Core SLURM Integration Exports
# =============================================================================
from .slurm import (
    SlurmDirectives,
    SubmissionResult,
    SlurmJobSpec,
    generate_sbatch_script,
    submit_slurm_job,
    _validate_time_format,
    _validate_memory_string,
    _check_scheduler_limits,
)

# =============================================================================
# Explicit Public API Declaration
# =============================================================================
# Controls `from wien2k_gen.submit import *` and provides clear IDE auto-completion boundaries.
# Only exposes production-ready interfaces; internal helpers remain encapsulated.
__all__ = [
    # Data Structures
    "SlurmDirectives",
    "SubmissionResult",
    "SlurmJobSpec",
    # Core Functions
    "generate_sbatch_script",
    "submit_slurm_job",
    # Validation Utilities (exposed for CLI/UI integration)
    "_validate_time_format",
    "_validate_memory_string",
    "_check_scheduler_limits",
]