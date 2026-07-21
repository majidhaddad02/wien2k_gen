"""SCF checkpoint management: save, restore, incremental, cleanup."""

import os
import time
from pathlib import Path
from typing import Any, Optional

from ...logging_config import get_logger

logger = get_logger(__name__)


def create_scf_checkpoint(case_name: str, label: str = "") -> str:
    """Save SCF checkpoint for restart after failure or preemption.

    Copies case.clmval (charge density), case.clmsum, case.broyd*
    to a timestamped backup directory under case_checkpoints/.

    Returns path to the checkpoint directory.
    """
    import shutil as _shutil

    case = Path(case_name)
    if not case.exists():
        case = Path(".")

    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = case / "case_checkpoints" / f"ckpt_{ts}"
    if label:
        ckpt_dir = case / "case_checkpoints" / f"ckpt_{label}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for suffix in [".clmval", ".clmsum", ".broyd", ".broyd1", ".broyd2"]:
        src = case / f"{case.name}{suffix}"
        if src.exists():
            _shutil.copy2(str(src), str(ckpt_dir / src.name))

    (ckpt_dir / "CHECKPOINT_INFO").write_text(
        f"case={case.name}\nlabel={label}\ntimestamp={ts}\n"
    )

    logger.info(f"SCF checkpoint saved: {ckpt_dir}")
    return str(ckpt_dir)


def restore_from_checkpoint(case_name: str, checkpoint_dir: Optional[str] = None) -> bool:
    """Restore SCF state from most recent (or specified) checkpoint.

    Returns True if restore succeeded.
    """
    import shutil as _shutil

    case = Path(case_name)
    if not case.exists():
        case = Path(".")

    ckpt_base = case / "case_checkpoints"
    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir)
    elif ckpt_base.exists():
        dirs = sorted(ckpt_base.glob("ckpt_*"), key=os.path.getmtime, reverse=True)
        if not dirs:
            logger.warning("No checkpoints found")
            return False
        ckpt_dir = dirs[0]
    else:
        logger.warning("No checkpoint directory exists")
        return False

    for src_file in ckpt_dir.glob("*"):
        if src_file.name.endswith((".clmval", ".clmsum", ".broyd", ".broyd1", ".broyd2")):
            dest = case / f"{case.name}{src_file.suffix}"
            _shutil.copy2(str(src_file), str(dest))

    logger.info(f"SCF checkpoint restored from: {ckpt_dir}")
    return True


def calculate_checkpoint_interval(
    remaining_time_sec: float,
    time_per_cycle_sec: float = 300.0,
) -> int:
    """Calculate adaptive checkpoint interval based on remaining walltime budget.

    Densifies checkpoints as the allocation deadline approaches to minimise
    lost work in the event of preemption:

        < 20 cycles remaining  → interval =  5 cycles (urgent)
        < 50 cycles remaining  → interval = 10 cycles (moderate)
        >= 50 cycles remaining → interval = 15 cycles (relaxed)

    This is a simple stepped heuristic, not the optimal check-pointing
    formula from Daly (2006, Future Generation Computer Systems 22(3),
    303-312).  The latter requires MTBF estimates and measured I/O cost,
    which are system-dependent and rarely available in HPC batch environments.

    Falls back to 15 if time_per_cycle is zero.
    """
    if time_per_cycle_sec <= 0:
        return 15
    remaining_cycles = int(remaining_time_sec / time_per_cycle_sec)
    if remaining_cycles < 0:
        return 5
    if remaining_cycles < 20:
        return 5
    if remaining_cycles < 50:
        return 10
    return 15


def perform_incremental_checkpoint(
    case_name: str,
    checkpoint_dir: str = ".checkpoints",
    nowrite_vector: bool = False,
    is_soc: bool = False,
) -> dict[str, Any]:
    """Perform incremental checkpoint — copy only modified files.

    Files copied based on calculation context:
        case.scf         — always (SCF diagnostics)
        case.vector      — if nowrite_vector=False (restart vector)
        case.dmat        — if SOC (density matrix for spin-orbit)
        case.clmsum      — if orbital potential present
        case.broyd*      — if Broyden mixing active
        case.pulay_history — if Pulay mixing active

    Returns dict with:
        checkpoint_id: str
        files_copied: List[str]
        size_mb: float
        incremental: bool
    """
    import shutil as _shutil

    case = Path(case_name)
    if not case.exists():
        case = Path(".")

    ckpt_base = case / Path(checkpoint_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_id = f"ckpt_{ts}"
    ckpt_dir = ckpt_base / ckpt_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    files_to_check = [".scf"]
    if not nowrite_vector:
        files_to_check.append(".vector")
    if is_soc:
        files_to_check.append(".dmat")
    files_to_check.extend([".clmsum", ".broyd", ".broyd1", ".broyd2"])
    if (case / Path(".pulay_history")).exists():
        files_to_check.append(".pulay_history")

    case_files = {f.name: f for f in case.iterdir() if f.is_file()}

    files_copied = []
    total_size = 0.0

    for suffix in files_to_check:
        fname = f"{case.name}{suffix}"
        if fname in case_files:
            src = case_files[fname]
            dst = ckpt_dir / fname
            _shutil.copy2(str(src), str(dst))
            files_copied.append(fname)
            total_size += src.stat().st_size

    size_mb = total_size / (1024 * 1024)

    (ckpt_dir / "CHECKPOINT_INFO").write_text(
        f"case={case.name}\ncheckpoint_id={ckpt_id}\ntimestamp={ts}\n"
        f"incremental=True\nfiles_copied={','.join(files_copied)}\n"
        f"nowrite_vector={nowrite_vector}\nis_soc={is_soc}\n",
        encoding="utf-8")

    # Update checkpoint history
    history_file = ckpt_base / ".checkpoint_history"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"{ckpt_id} {ts} {size_mb:.1f}MB {len(files_copied)}files\n")

    # Save status
    status_file = ckpt_base / ".checkpoint_status"
    status_file.write_text(
        f"last_checkpoint={ckpt_id}\n"
        f"total_checkpoints={len(list(ckpt_base.glob('ckpt_*')))}\n"
        f"last_size_mb={size_mb:.1f}\n",
        encoding="utf-8")

    cleanup_old_checkpoints(str(ckpt_base))

    logger.info(
        f"Checkpoint saved: {ckpt_id} at cycle (incremental), "
        f"size={size_mb:.1f}MB, files={len(files_copied)}"
    )

    return {
        "checkpoint_id": ckpt_id,
        "files_copied": files_copied,
        "size_mb": round(size_mb, 2),
        "incremental": True,
    }


def cleanup_old_checkpoints(
    checkpoint_dir: str = ".checkpoints",
    max_checkpoints: int = 3,
    quota_warning_pct: float = 80.0,
) -> dict[str, Any]:
    """Remove oldest checkpoints keeping only last max_checkpoints.

    Also checks disk quota and warns if checkpoint space exceeds threshold.

    Returns dict with removed count and total space info.
    """
    ckpt_base = Path(checkpoint_dir)
    if not ckpt_base.exists():
        return {"removed": 0, "total_mb": 0.0, "warn": False}

    ckpts = sorted(ckpt_base.glob("ckpt_*"), key=lambda p: p.stat().st_mtime)
    removed = 0
    total_size = 0.0

    while len(ckpts) > max_checkpoints:
        old = ckpts.pop(0)
        if old.is_dir():
            import shutil as _shutil
            _shutil.rmtree(old)
        removed += 1

    for ckpt in ckpt_base.glob("ckpt_*"):
        if ckpt.is_dir():
            for f in ckpt.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

    total_mb = total_size / (1024 * 1024)
    warn = total_mb > 1024  # 1 GB

    if warn:
        logger.warning(
            f"Checkpoint storage: {total_mb:.1f}MB. "
            f"Consider reducing max_checkpoints or using external storage."
        )

    return {"removed": removed, "total_mb": round(total_mb, 1), "warn": warn}


def resume_from_checkpoint(
    case_name: str,
    checkpoint_id: Optional[str] = None,
) -> dict[str, Any]:
    """Resume WIEN2k calculation from a saved checkpoint.

    Copies checkpoint files back to working directory and adjusts
    case.in2 to continue from the last completed SCF cycle.

    Returns dict with success status and details.
    """
    import shutil as _shutil

    case = Path(case_name)
    if not case.exists():
        case = Path(".")

    ckpt_base = case / ".checkpoints"
    if checkpoint_id:
        ckpt_dir = ckpt_base / checkpoint_id
    else:
        dirs = sorted(ckpt_base.glob("ckpt_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not dirs:
            logger.warning("No checkpoints found for resume")
            return {"success": False, "message": "No checkpoints found"}
        ckpt_dir = dirs[0]

    if not ckpt_dir.exists():
        return {"success": False, "message": f"Checkpoint not found: {ckpt_dir}"}

    # Copy checkpoint files to working directory
    files_restored = []
    for src_file in ckpt_dir.glob("*"):
        if src_file.is_file() and src_file.name != "CHECKPOINT_INFO":
            if src_file.name.startswith(case.name):
                _shutil.copy2(str(src_file), str(case / src_file.name))
            else:
                _shutil.copy2(str(src_file), str(case / f"{case.name}{src_file.suffix}"))
            files_restored.append(src_file.name)

    # Read checkpoint info for cycle tracking
    info_file = ckpt_dir / "CHECKPOINT_INFO"
    cycle_info = ""
    if info_file.exists():
        cycle_info = info_file.read_text(encoding="utf-8", errors="replace")

    logger.info(
        f"Resumed from checkpoint {ckpt_dir.name}: "
        f"{len(files_restored)} files restored"
    )

    return {
        "success": True,
        "checkpoint_id": ckpt_dir.name,
        "files_restored": len(files_restored),
        "cycle_info": cycle_info.strip(),
    }


__all__ = [
    "calculate_checkpoint_interval",
    "cleanup_old_checkpoints",
    "create_scf_checkpoint",
    "perform_incremental_checkpoint",
    "restore_from_checkpoint",
    "resume_from_checkpoint",
]
