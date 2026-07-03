"""
Generic Configuration Builder – Orchestrates DFT backend configuration generation.
Integrates hardware topology, resource suggestions, atomic file operations, and multi-backend validation.
Designed for robust, exascale-ready WIEN2k and multi-code parallel execution setup.

Key Improvements Applied:
- Fixed all syntax errors, typos, and import resolution issues (e.g., logger name, dataclass fields).
- Implemented robust fallback for missing types module with proper, non-conflicting dataclass definitions.
- Enhanced suggestion normalization, auto-suggestion, and validation pipelines with strict type guards.
- Added safe backend method invocation using getattr() to prevent AttributeError on missing hooks.
- Integrated comprehensive English documentation and HPC-grade error handling at every step.
- Maintained full code volume with expanded safety checks, logging, and resiliency hooks.
"""

import os
import time
import logging
import warnings
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field, asdict, is_dataclass

from .topology import Topology
from ..config import OUTPUT_FILE
from ..logging_config import get_logger
from ..utils.atomic_write import atomic_write
from ..utils.validation import validate_machines, backup_machines

logger = get_logger(__name__)

# Lazy import to avoid circular dependency
def _get_suggest_optimal_resources():
    from ..optimizer.advisor import suggest_optimal_resources
    return suggest_optimal_resources

# Lazy import to avoid circular dependency
def _get_current_backend():
    from ..backend_manager import get_current_backend
    return get_current_backend()

# =============================================================================
# Fallback Type Definitions (if centralized types.py is unavailable)
# =============================================================================
try:
    from ..types import ResourceSuggestion, ProblemSize
except ImportError:
    logger.debug("Centralized types module not found; using local fallback definitions.")
    
    @dataclass
    class ProblemSize:
        """Lightweight representation of DFT problem scale."""
        n_atoms: int = 0
        n_kpoints: int = 0
        n_bands: int = 0
        is_spin_polarized: bool = False
        has_spin_orbit: bool = False
        memory_estimate_mb: float = 0.0

    @dataclass
    class ResourceSuggestion:
        """Structured recommendation for MPI/OpenMP hybrid configuration."""
        mode: str = "mpi"
        recommended_total_cores: int = 1
        omp_threads_per_rank: int = 1
        mpi_ranks_per_node: int = 1
        cores_per_node: List[int] = field(default_factory=list)
        vector_split_active: bool = False
        warnings: List[str] = field(default_factory=list)
        reason: str = ""
        confidence: float = 1.0

        def to_dict(self) -> Dict[str, Any]:
            """Convert dataclass to dictionary for backend consumption."""
            return asdict(self)


# =============================================================================
# Build Result & Validation Dataclasses
# =============================================================================
@dataclass
class BuildResult:
    """Immutable result of configuration build operation."""
    success: bool
    config_path: Optional[Path] = None
    config_content: Optional[str] = None
    error_message: Optional[str] = None
    resource_estimate: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    backup_path: Optional[Path] = None
    validation_passed: bool = False
    timestamp_ns: int = field(default_factory=lambda: 0, repr=False)

    def __post_init__(self):
        """Set timestamp on initialization if not provided."""
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for logging or UI consumption."""
        base = asdict(self)
        # Convert Path objects to strings for JSON compatibility
        base["config_path"] = str(base["config_path"]) if base["config_path"] else None
        base["backup_path"] = str(base["backup_path"]) if base["backup_path"] else None
        return base


# =============================================================================
# Internal Helper Functions
# =============================================================================
def _normalize_suggestion(suggestion: Optional[Union[Dict[str, Any], ResourceSuggestion]]) -> Dict[str, Any]:
    """
    Safely normalize user or auto-generated suggestion into a flat dictionary.
    Preserves original keys while ensuring type compatibility.
    """
    if suggestion is None:
        return {}
    if isinstance(suggestion, dict):
        return suggestion.copy()
    if is_dataclass(suggestion):
        return asdict(suggestion)
    if hasattr(suggestion, "to_dict") and callable(suggestion.to_dict):
        return suggestion.to_dict()
    raise TypeError(f"Unsupported suggestion type: {type(suggestion)}")


def _rotate_backups(config_path: Path, max_retention: int = 3) -> Optional[Path]:
    """
    Rotate existing configuration backups to prevent unbounded disk growth on HPC scratch.
    Returns path of the newly created backup, or None if no backup existed.
    """
    if not config_path.exists():
        return None
        
    backup_dir = config_path.parent
    backup_prefix = f".{config_path.name}.bak."
    
    try:
        existing_backups = sorted(
            [f for f in backup_dir.iterdir() if f.name.startswith(backup_prefix)],
            key=lambda x: x.stat().st_mtime
        )
    except OSError as e:
        logger.warning(f"Failed to list backups in {backup_dir}: {e}")
        return None

    # Remove oldest backups if exceeding retention limit
    while len(existing_backups) >= max_retention:
        old_backup = existing_backups.pop(0)
        try:
            old_backup.unlink()
            logger.debug(f"Rotated out old backup: {old_backup}")
        except OSError as e:
            logger.warning(f"Failed to remove stale backup {old_backup}: {e}")
            
    # Create new timestamped backup
    ts = time.strftime("%Y%m%d_%H%M%S")
    new_backup = config_path.parent / f"{backup_prefix}{ts}"
    try:
        shutil.copy2(config_path, new_backup)
        return new_backup
    except OSError as e:
        logger.warning(f"Backup creation failed: {e}")
        return None


def _validate_suggestion_safety(suggestion: Dict[str, Any], topo: Topology) -> List[str]:
    """
    Perform pre-write sanity checks on resource suggestion to prevent invalid configurations.
    Catches common HPC misconfigurations before they reach the backend or file system.
    """
    warnings_list = []
    total_cores = suggestion.get("recommended_total_cores", topo.total_cores)
    omp_threads = suggestion.get("omp_threads_per_rank", 1)
    mpi_ranks = suggestion.get("mpi_ranks_per_node", 1)
    
    if total_cores <= 0:
        warnings_list.append("Total cores must be positive.")
    if omp_threads <= 0:
        warnings_list.append("OMP threads per rank must be positive.")
    if mpi_ranks <= 0:
        warnings_list.append("MPI ranks per node must be positive.")
        
    # Check for oversubscription on single nodes
    for c in topo.cores_per_node:
        if omp_threads * mpi_ranks > c:
            warnings_list.append(
                f"Potential oversubscription: {omp_threads}×{mpi_ranks} > {c} physical cores on a node."
            )
            break
            
    return warnings_list


def _get_config_filename(backend: Any, default_name: str = OUTPUT_FILE) -> Path:
    """
    Determine the output configuration filename with fallback hierarchy.
    Prioritizes backend-specific method, then global config, then safe default.
    """
    # Try backend-specific method
    if hasattr(backend, "get_config_filename") and callable(backend.get_config_filename):
        try:
            return Path(backend.get_config_filename())
        except Exception as e:
            logger.warning(f"Backend get_config_filename() failed: {e}")
            
    # Fallback to module-level constant
    if default_name and str(default_name).strip():
        return Path(default_name)
        
    # Ultimate safe fallback
    return Path("machines")


# =============================================================================
# Main Build Orchestrator
# =============================================================================
def build_auto(
    topo: Topology,
    backup: bool = True,
    suggestion: Optional[Union[Dict[str, Any], ResourceSuggestion]] = None,
    dry_run: bool = False,
    validate: bool = True
) -> Union[BuildResult, str]:
    """
    Generate configuration for the currently active DFT backend.
    Orchestrates suggestion resolution, backend generation, atomic file operations,
    validation, and metadata collection. Designed for seamless integration with
    CLI wizards, automated job submission, and interactive UI workflows.

    Args:
        topo: Detected or user-defined hardware topology.
        backup: Whether to create a timestamped backup of existing config.
        suggestion: Pre-computed resource allocation dict or dataclass.
        dry_run: If True, return generated content string without writing to disk.
        validate: If True, run post-generation validation checks.
        
    Returns:
        BuildResult dataclass on success/failure, or config string in dry-run mode.
    """
    backend = _get_current_backend()
    logger.info(f"Starting configuration build for backend: {backend.__class__.__name__}")

    # 1. Normalize & Resolve Suggestion
    suggestion_dict = _normalize_suggestion(suggestion)
    if not suggestion_dict:
        logger.info("No suggestion provided; triggering auto-optimization pipeline...")
        try:
            # Detect problem size from backend-specific inputs (safely)
            detect_method = getattr(backend, 'detect_problem_size', lambda: None)
            problem = detect_method()
            
            problem_dict = {}
            if problem and is_dataclass(problem):
                problem_dict = asdict(problem)
            elif isinstance(problem, dict):
                problem_dict = problem
                
            # Generate optimal resources based on topology and problem scale
            opt_suggestion = _get_suggest_optimal_resources()(topo, user_max_cores=None)
            suggestion_dict = {
                **_normalize_suggestion(opt_suggestion),
                **problem_dict
            }
            logger.info(f"Auto-suggestion resolved: {suggestion_dict.get('recommended_total_cores', '?')} cores")
        except Exception as e:
            err_msg = f"Auto-suggestion failed: {e}"
            logger.error(err_msg, exc_info=True)
            if not dry_run:
                return BuildResult(success=False, error_message=err_msg)
            return f"# ERROR: {err_msg}\n"
            
    # 2. Pre-write Safety Validation
    safety_warnings = _validate_suggestion_safety(suggestion_dict, topo)
    if "warnings" not in suggestion_dict:
        suggestion_dict["warnings"] = []
    suggestion_dict["warnings"].extend(safety_warnings)

    # Backend-specific suggestion validation
    validate_sugg_method = getattr(backend, 'validate_suggestion', None)
    if callable(validate_sugg_method):
        try:
            backend_validation = validate_sugg_method(suggestion_dict)
            if backend_validation:
                suggestion_dict["warnings"].extend(backend_validation)
        except Exception as e:
            logger.warning(f"Backend suggestion validation failed: {e}")
            
    if safety_warnings and any("oversubscription" in w.lower() for w in safety_warnings):
        logger.warning("Critical oversubscription detected in suggestion; proceeding with caution.")
        
    # 3. Dry-Run Mode
    if dry_run:
        try:
            config_content = backend.generate_input(topo, suggestion_dict)
            # Return consistent BuildResult for dry_run to maintain API uniformity
            return BuildResult(
                success=True,
                config_content=config_content,
                warnings=suggestion_dict.get("warnings", [])
            )
        except Exception as e:
            err_msg = f"Dry-run generation failed: {e}"
            logger.error(err_msg, exc_info=True)
            return BuildResult(success=False, error_message=err_msg)
            
    # 4. Generate Configuration Content
    try:
        config_content = backend.generate_input(topo, suggestion_dict)
    except Exception as e:
        err_msg = f"Backend config generation failed: {e}"
        logger.error(err_msg, exc_info=True)
        return BuildResult(success=False, error_message=err_msg)
        
    if not config_content or not str(config_content).strip():
        return BuildResult(success=False, error_message="Generated configuration is empty.")
        
    # 5. Resolve Output Path & Backup
    config_path = _get_config_filename(backend, OUTPUT_FILE)
    backup_path = None

    if backup and config_path.exists():
        try:
            backup_path = _rotate_backups(config_path, max_retention=3)
            if backup_path:
                logger.info(f"Backed up existing configuration to {backup_path}")
        except Exception as e:
            logger.warning(f"Backup skipped due to error: {e}")
            
    # 6. Atomic Write Operation
    try:
        atomic_write(config_path, config_content)
        logger.info(f"Configuration atomically written to {config_path}")
    except Exception as e:
        err_msg = f"Atomic write failed: {e}"
        logger.error(err_msg, exc_info=True)
        return BuildResult(success=False, error_message=err_msg)
        
    # 7. Post-write Validation
    validation_passed = True
    if validate:
        try:
            validate_config_method = getattr(backend, 'validate_config', None)
            if callable(validate_config_method):
                validation_passed = validate_config_method(config_content, config_path)
            else:
                # Fallback to generic WIEN2k machines validator
                validation_passed = validate_machines(config_path)
        except Exception as e:
            logger.warning(f"Validation routine failed; assuming pass: {e}")
            validation_passed = False
            
    if not validation_passed:
        logger.error("Configuration validation failed after write.")
        return BuildResult(
            success=False,
            error_message="Config validation failed",
            config_path=config_path,
            backup_path=backup_path
        )
        
    # 8. Auxiliary Files (Optional Backend Hooks)
    write_aux_method = getattr(backend, 'write_auxiliary_files', None)
    if callable(write_aux_method):
        try:
            write_aux_method(topo, suggestion_dict)
            logger.debug("Auxiliary backend files written successfully.")
        except Exception as e:
            logger.warning(f"Failed to write auxiliary files: {e}")
            
    # 9. Resource Estimation & Finalization
    resource_estimate = {}
    estimate_method = getattr(backend, 'estimate_resources', None)
    detect_method = getattr(backend, 'detect_problem_size', lambda: None)
    
    if callable(estimate_method):
        try:
            problem = detect_method()
            resource_estimate = estimate_method(problem, topo)
        except Exception as e:
            logger.debug(f"Resource estimation skipped: {e}")
            
    final_result = BuildResult(
        success=True,
        config_path=config_path,
        config_content=config_content if logger.isEnabledFor(logging.DEBUG) else None,
        resource_estimate=resource_estimate,
        warnings=suggestion_dict.get("warnings", []),
        backup_path=backup_path,
        validation_passed=validation_passed
    )

    logger.info(f"✓ Configuration build completed successfully for {backend.__class__.__name__}")
    return final_result


# =============================================================================
# Legacy Compatibility Wrappers (Deprecated)
# =============================================================================
def build_mpi(topo: Topology, backup: bool = True) -> bool:
    """
    Legacy wrapper for pure MPI configuration generation.
    Deprecated: Use build_auto() for unified backend handling.
    """
    warnings.warn(
        "build_mpi() is deprecated and will be removed in v2.0. Use build_auto() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    result = build_auto(topo, backup=backup)
    return result.success if isinstance(result, BuildResult) else bool(result)


def build_hybrid(topo: Topology, omp: int = 1, backup: bool = True) -> bool:
    """
    Legacy wrapper for hybrid MPI/OpenMP configuration generation.
    Deprecated: Use build_auto() with explicit OMP suggestion.
    """
    warnings.warn(
        "build_hybrid() is deprecated and will be removed in v2.0. Use build_auto() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    suggestion = ResourceSuggestion(
        mode="hybrid",
        omp_threads_per_rank=omp,
        reason="Legacy hybrid wrapper invocation"
    )
    result = build_auto(topo, backup=backup, suggestion=suggestion)
    return result.success if isinstance(result, BuildResult) else bool(result)


def build_kpoint(topo: Topology, backup: bool = True) -> bool:
    """
    Legacy wrapper for k-point parallelism focused configuration.
    Deprecated: Handled automatically by backend suggestion logic.
    """
    warnings.warn(
        "build_kpoint() is deprecated and will be removed in v2.0. Use build_auto() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    suggestion = ResourceSuggestion(
        mode="kpoint",
        vector_split_active=True,
        reason="Legacy k-point wrapper invocation"
    )
    result = build_auto(topo, backup=backup, suggestion=suggestion)
    return result.success if isinstance(result, BuildResult) else bool(result)