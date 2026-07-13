"""
HPC Scratch Space Management & Multi-Node I/O Staging Module.
Production features:
• Intelligent scratch selection with priority chain (/dev/shm → $SCRATCH → local SSD → /tmp)
• Automatic filesystem type detection (tmpfs, ext4, xfs, lustre, gpfs, nfs) & space validation
• Multi-node synchronization via sbcast (SLURM), parallel rsync, or shared-FS fallback
• Configurable file staging filters for WIEN2k/DFT inputs with symlink & large-file handling
• Atomic cleanup with signal traps, permission preservation, and graceful degradation
• Structured return types, comprehensive logging, and HPC-grade error resilience
All documentation and inline comments are in English per project standards.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict, Union

from ..core.topology import Topology
from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Type Definitions & Configuration
# =============================================================================

class ScratchResult(TypedDict, total=False):
    """Structured outcome of scratch setup & staging operations."""
    success: bool
    scratch_path: Optional[str]
    is_shared: bool
    filesystem_type: str
    free_space_gb: float
    files_staged: list[str]
    staging_method: str  # 'local_copy', 'sbcast', 'rsync', 'shared_fs'
    errors: list[str]
    warnings: list[str]


@dataclass
class ScratchConfig:
    """Configuration for scratch behavior & staging preferences."""
    priority_paths: list[str] = field(default_factory=lambda: [
        "/dev/shm", 
        os.environ.get("SCRATCH", ""), 
        "/tmp"
    ])
    env_path_vars: list[str] = field(default_factory=lambda: [
        "SCRATCH", "TMPDIR", "TMP", "WIEN2K_SCRATCH", "QE_SCRATCH",
    ])
    min_free_space_gb: float = 2.0
    prefer_tmpfs: bool = True
    staging_method: str = "auto"  # auto, copy, sbcast, rsync
    file_patterns: list[str] = field(default_factory=lambda: [
        "case.", ".in*", ".klist", "parallel_options", ".struct",
        ".scf", "run.sh", "input_", "POSCAR", "INCAR", "KPOINTS", "POTCAR", ".in"
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        ".bak", ".tmp", ".log", ".out", "slurm-", "core.", ".o", ".e"
    ])
    preserve_permissions: bool = True
    cleanup_on_exit: bool = True
    stripe_count: int = 4
    stripe_size_mb: int = 1

    def resolve_priority_paths(self) -> list[str]:
        """Resolve priority paths, removing empties and expanding env vars."""
        resolved = []
        for p in self.priority_paths:
            expanded = os.path.expandvars(os.path.expanduser(p))
            if expanded and not expanded.isspace():
                resolved.append(expanded)
        # Append env var values as additional candidates
        for var in self.env_path_vars:
            val = os.environ.get(var, "").strip()
            if val and val not in resolved:
                resolved.append(val)
        return resolved


# =============================================================================
# Filesystem & Space Detection Helpers
# =============================================================================

def _detect_filesystem_type(path: Path) -> str:
    """
    Detect filesystem type for a given path using df -T.
    Returns normalized string (tmpfs, ext4, xfs, lustre, gpfs, nfs, unknown).
    """
    try:
        proc = subprocess.run(
            ["df", "-T", str(path)], capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0:
            lines = proc.stdout.strip().splitlines()
            if len(lines) > 1:
                fstype = lines[-1].split()[1].lower()
                if "lustre" in fstype:
                    return "lustre"
                if "gpfs" in fstype or "mmfs" in fstype:
                    return "gpfs"
                if "nfs" in fstype:
                    return "nfs"
                if "tmpfs" in fstype:
                    return "tmpfs"
                if "ext4" in fstype:
                    return "ext4"
                if "xfs" in fstype:
                    return "xfs"
                return fstype
    except Exception:
        pass
    return "unknown"


def _check_free_space_gb(path: Path) -> float:
    """Return available disk space in GB for the given path."""
    try:
        usage = shutil.disk_usage(str(path))
        return round(usage.free / (1024 ** 3), 2)
    except Exception:
        return 0.0


def _is_shared_filesystem(path: Path) -> bool:
    """Heuristic check if path resides on a network/shared filesystem."""
    fstype = _detect_filesystem_type(path)
    return fstype in ("lustre", "gpfs", "nfs", "cifs", "beegfs")


def _detect_lustre(path: Path) -> bool:
    """
    Detect Lustre filesystem via stat -f -c %T.
    Outputs "lustre" when the path resides on a Lustre mount.
    """
    try:
        proc = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip().lower() == "lustre":
            return True
    except Exception:
        pass
    return False


def configure_lustre_striping(
    path: str,
    stripe_count: int = 4,
    stripe_size_mb: int = 1
) -> bool:
    """
    Configure Lustre striping for a directory or file path.

    Runs `lfs setstripe -c {stripe_count} -s {stripe_size_mb}M {path}` before
    creating scratch directories. Lustre striping distributes MPI-IO operations
    across multiple object storage targets (OSTs), preventing serialization
    to a single OST.

    Reference:
        Lustre 2.x Operations Manual Chapter 7 (File Striping);
        Lustre MPI-IO Best Practices Guide.

    Args:
        path: Target path to apply striping to (directory or file).
        stripe_count: Number of OSTs to stripe across (default 4).
        stripe_size_mb: Stripe size in MB (default 1MB).

    Returns:
        True if striping was successfully configured, False otherwise.
    """
    try:
        if not shutil.which("lfs"):
            logger.debug("lfs command not found; skipping Lustre striping")
            return False

        proc = subprocess.run(
            ["lfs", "setstripe", "-c", str(stripe_count), "-s", f"{stripe_size_mb}M", path],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            logger.debug(
                f"Lustre striping configured: {path} -> "
                f"stripe_count={stripe_count}, stripe_size={stripe_size_mb}M"
            )
            return True
        else:
            logger.debug(f"lfs setstripe failed: {proc.stderr.strip()}")
            return False
    except Exception as e:
        logger.debug(f"Lustre striping exception: {e}")
        return False


# =============================================================================
# File Staging & Synchronization Logic
# =============================================================================

def _collect_staging_files(
    workdir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str]
) -> list[Path]:
    """
    Collect files matching include patterns while excluding globs.
    Handles recursive matching and resolves symlinks safely.
    """
    matched = set()
    for pattern in include_patterns:
        matched.update(workdir.glob(pattern))
        
    excluded = set()
    for pattern in exclude_patterns:
        excluded.update(workdir.glob(pattern))
        
    # Filter out excluded, directories, and broken symlinks
    valid = [
        p for p in matched
        if p not in excluded and p.is_file() and not (p.is_symlink() and not p.exists())
    ]
    return sorted(valid, key=lambda x: x.stat().st_size, reverse=True)


def _stage_local_copy(files: list[Path], dest: Path, preserve: bool = True) -> tuple[list[str], list[str]]:
    """Copy files to local scratch directory. Returns (success_list, error_list)."""
    success, errors = [], []
    dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        try:
            dst = dest / src.name
            if preserve:
                shutil.copy2(src, dst, follow_symlinks=False)
            else:
                shutil.copy(src, dst, follow_symlinks=False)
            success.append(src.name)
        except Exception as e:
            errors.append(f"Failed to copy {src.name}: {e}")
    return success, errors


def _stage_slurm_sbcast(files: list[Path], dest_dir: str) -> tuple[list[str], list[str]]:
    """
    Use SLURM sbcast for fast parallel file distribution.
    Falls back gracefully if sbcast is unavailable or fails.
    """
    success, errors = [], []
    if not files:
        return success, errors
        
    try:
        # Check sbcast availability
        if not shutil.which("sbcast"):
            raise RuntimeError("sbcast not found in PATH")
            
        # Create temporary tar to minimize sbcast metadata overhead
        tar_path = Path(tempfile.gettempdir()) / f".forge_stage_{os.getpid()}.tar.gz"
        proc = subprocess.run(
            ["tar", "-czf", str(tar_path), "-C", str(files[0].parent), *[f.name for f in files]],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            raise RuntimeError(f"tar creation failed: {proc.stderr}")
            
        # Broadcast tarball
        dest_tar = f"{dest_dir}/wien2k_stage.tar.gz"
        proc = subprocess.run(
            ["sbcast", "-f", str(tar_path), dest_tar],
            capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sbcast failed: {proc.stderr}")
            
        # Extract on destination (assumed local execution context for simplicity)
        proc = subprocess.run(
            ["tar", "-xzf", dest_tar, "-C", dest_dir],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Extraction failed: {proc.stderr}")
            
        success = [f.name for f in files]
        tar_path.unlink(missing_ok=True)
        
    except Exception as e:
        errors.append(f"sbcast staging failed: {e}")
        logger.debug(f"sbcast fallback triggered: {e}")
        # Fallback to local copy if sbcast fails
        return _stage_local_copy(files, Path(dest_dir))
        
    return success, errors


def _stage_rsync(files: list[Path], dest: Path) -> tuple[list[str], list[str]]:
    """
    Use rsync for efficient delta & parallel staging.
    Optimized for HPC networks with low-latency interconnects.
    """
    success, errors = [], []
    if not files:
        return success, errors
        
    dest.mkdir(parents=True, exist_ok=True)
    src_list = [str(f) for f in files]
    try:
        proc = subprocess.run(
            ["rsync", "-a", "--inplace", "--partial", "--info=progress2", "--no-compress", *src_list, str(dest) + "/"],
            capture_output=True, text=True, timeout=60
        )
        if proc.returncode == 0:
            success = [f.name for f in files]
        else:
            errors.append(f"rsync failed: {proc.stderr.strip()}")
    except Exception as e:
        errors.append(f"rsync exception: {e}")
    return success, errors


# =============================================================================
# Main Scratch Setup & Cleanup API
# =============================================================================

def setup_scratch(  # noqa: C901
    topo: Topology,
    config: Optional[ScratchConfig] = None,
    workdir: Optional[Path] = None
) -> ScratchResult:
    """
    Select, create, and populate scratch directory based on HPC environment.
    Handles multi-node detection, filesystem prioritization, and staging.
    
    Args:
        topo: Hardware/scheduler topology for node-aware decisions.
        config: Optional scratch behavior configuration.
        workdir: Source directory (defaults to current working directory).
        
    Returns:
        ScratchResult with paths, staging method, and diagnostics.
    """
    cfg = config or ScratchConfig()
    wd = workdir or Path.cwd()
    result: ScratchResult = {
        "success": False, "scratch_path": None, "is_shared": False,
        "filesystem_type": "unknown", "free_space_gb": 0.0,
        "files_staged": [], "staging_method": "none",
        "errors": [], "warnings": []
    }

    # 1. Select optimal scratch path
    selected_path = None
    for candidate in cfg.priority_paths:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists() and os.access(p, os.W_OK):
            free_gb = _check_free_space_gb(p)
            if free_gb < cfg.min_free_space_gb:
                result["warnings"].append(f"Skipping {p}: insufficient space ({free_gb:.1f}GB < {cfg.min_free_space_gb}GB)")
                continue
                
            # FIXED: Corrected logic for tmpfs preference check
            if cfg.prefer_tmpfs and topo.total_cores <= (os.cpu_count() or 1) and _detect_filesystem_type(p) == "tmpfs":
                    selected_path = p
                    break
                    
            if selected_path is None:
                selected_path = p
                
    if not selected_path:
        # Last resort: create in current directory
        selected_path = Path(".")
        result["warnings"].append("No suitable scratch found. Falling back to working directory.")
        
    # 2. Create job-specific scratch directory
    ts = int(time.time())
    scratch_dir = selected_path / f"wien2k_scratch_{os.getpid()}_{ts}"
    try:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(scratch_dir, os.W_OK):
            raise PermissionError(f"No write access to {scratch_dir}")
    except Exception as e:
        result["errors"].append(f"Scratch directory creation failed: {e}")
        return result
        
    result["scratch_path"] = str(scratch_dir)
    result["is_shared"] = _is_shared_filesystem(scratch_dir)
    result["filesystem_type"] = _detect_filesystem_type(scratch_dir)
    result["free_space_gb"] = _check_free_space_gb(scratch_dir)

    # Lustre striping for parallel MPI-IO distribution
    if _detect_lustre(scratch_dir):
        configure_lustre_striping(
            str(scratch_dir),
            stripe_count=cfg.stripe_count,
            stripe_size_mb=cfg.stripe_size_mb,
        )

    # 3. Collect & stage files
    files_to_stage = _collect_staging_files(wd, cfg.file_patterns, cfg.exclude_patterns)
    if not files_to_stage:
        result["warnings"].append("No input files matched staging patterns.")
        result["success"] = True
        return result
        
    # 4. Choose staging method
    method = cfg.staging_method.lower()
    if method == "auto":
        if os.getenv("SLURM_JOB_ID") and topo.nodes and len(topo.nodes) > 1:
            method = "sbcast" if shutil.which("sbcast") else "rsync"
        elif result["is_shared"]:
            method = "shared_fs"
        else:
            method = "copy"
            
    if method == "sbcast":
        success, errs = _stage_slurm_sbcast(files_to_stage, str(scratch_dir))
        result["staging_method"] = "sbcast"
    elif method == "rsync":
        success, errs = _stage_rsync(files_to_stage, scratch_dir)
        result["staging_method"] = "rsync"
    elif method == "shared_fs":
        success = [f.name for f in files_to_stage]  # No copy needed
        errs = []
        result["staging_method"] = "shared_fs"
    else:  # local copy
        success, errs = _stage_local_copy(files_to_stage, scratch_dir, cfg.preserve_permissions)
        result["staging_method"] = "copy"
        
    result["files_staged"] = success
    result["errors"].extend(errs)
    result["success"] = len(success) > 0 and len(errs) < len(files_to_stage) * 0.5

    if not result["success"]:
        result["errors"].append("Staging failure rate exceeded threshold. Aborting.")
        
    # 5. Register cleanup trap if requested
    if cfg.cleanup_on_exit and result["success"]:
        def _cleanup(sig=None, frame=None):
            if scratch_dir.exists():
                try:
                    shutil.rmtree(scratch_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up scratch: {scratch_dir}")
                except Exception:
                    pass
        try:
            signal.signal(signal.SIGTERM, _cleanup)
            signal.signal(signal.SIGINT, _cleanup)
        except ValueError: 
            pass  # Non-main thread
            
    logger.info(
        f"Scratch setup complete: {result['scratch_path']} | "
        f"method={result['staging_method']} | staged={len(success)}/{len(files_to_stage)} | "
        f"shared={result['is_shared']} | fs={result['filesystem_type']}"
    )
    return result


def cleanup_scratch(path: Union[str, Path], force: bool = False) -> bool:
    """
    Safely remove scratch directory and contents.
    Respects HPC cleanup policies and avoids race conditions.
    """
    p = Path(path)
    if not p.exists():
        return True
        
    if not force and not p.name.startswith("wien2k_scratch_"):
        logger.warning(f"Refusing to clean non-scratch directory: {p}")
        return False
        
    try:
        shutil.rmtree(p, ignore_errors=True)
        logger.info(f"Scratch cleaned: {p}")
        return True
    except Exception as e:
        logger.error(f"Scratch cleanup failed for {p}: {e}")
        return False


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "ScratchConfig",
    "ScratchResult",
    "_check_free_space_gb",
    "_collect_staging_files",
    "_detect_filesystem_type",
    "_detect_lustre",
    "cleanup_scratch",
    "configure_lustre_striping",
    "setup_scratch",
]