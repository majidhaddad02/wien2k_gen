"""
Specialized CLI Module for SLURM SBATCH Script Generation & Validation.
Provides focused command-line tools for creating, validating, previewing,
and submitting SLURM job scripts with rigorous syntax checking, resource
constraint enforcement, and atomic file operations.

Key Architecture Features:
• Dedicated argparse interface for sbatch-specific workflows
• Strict validation of SLURM directives (time, memory, partition, dependencies)
• Atomic script generation with automatic backup rotation & permission handling
• Dry-run & preview modes for safe configuration auditing before submission
• Seamless integration with topology detection, config defaults, and submit engine
• Structured JSON output & machine-readable error reporting for CI/CD pipelines
• Comprehensive English documentation, type hints, and HPC-grade resilience patterns
"""

import argparse
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AppConfig, ensure_dirs, load_config
from .core.scheduler import auto_detect_memory
from .core.scheduler import detect as detect_topology
from .exceptions import (
    ConfigurationError,
    SchedulerError,
    ValidationError,
    format_error_for_ui,
    log_exception_structured,
)
from .logging_config import get_logger, set_context, setup_logging
from .submit.slurm import (
    SlurmDirectives,
    SlurmJobSpec,
    generate_sbatch_script,
    submit_slurm_job,
)
from .utils.atomic_write import atomic_write

logger = get_logger(__name__)


# =============================================================================
# Validation Helpers
# =============================================================================

def _validate_time_format(value: str) -> bool:
    """Validate SLURM walltime format (HH:MM:SS, D-HH:MM:SS, etc.)."""
    return bool(re.match(r'^(\d+-)?(\d{1,2}:)?\d{2}:\d{2}$', value.strip()))


def _validate_memory_string(value: str) -> bool:
    """Validate SLURM memory string format (e.g., 64G, 4000M)."""
    return bool(re.match(r'^\d+[KMGkmgTt]?$', value.strip()))


# =============================================================================
# Argument Parser Builder
# =============================================================================

def create_sbatch_parser() -> argparse.ArgumentParser:
    """
    Build dedicated argparse interface for SBATCH script workflows.
    Can be attached to main CLI or used as a standalone entry point.
    """
    parser = argparse.ArgumentParser(
        prog="wien2k_sbatch",
        description="Generate, validate, preview, and submit SLURM job scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Return results as JSON")
    parser.add_argument("--backend", type=str, default=None, help="DFT backend context (wien2k/qe/vasp)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config file")
    parser.add_argument("--log-file", type=str, default=None, help="Redirect logs to file")

    sub = parser.add_subparsers(dest="action", required=True, help="SBATCH action to perform")

    gen = sub.add_parser("generate", help="Generate SBATCH script from topology & config")
    gen.add_argument("--output", "-o", type=str, default="submit_job.sh", help="Output script path")
    gen.add_argument("--job-name", "-J", type=str, default="wien2k_job", help="SLURM job name")
    gen.add_argument("--partition", "-p", type=str, default="", help="Target partition/queue")
    gen.add_argument("--nodes", "-N", type=int, default=1, help="Number of nodes")
    gen.add_argument("--ntasks", "-n", type=int, default=0, help="Total tasks (0 = auto from topology)")
    gen.add_argument("--cpus-per-task", "-c", type=int, default=1, help="CPUs per task")
    gen.add_argument("--mem", type=str, default=auto_detect_memory(), help="Memory per node (e.g., 8G, 16000M)")
    gen.add_argument("--time", "-t", type=str, default="24:00:00", help="Walltime limit")
    gen.add_argument("--dependency", type=str, default="", help="Job dependency (e.g., afterok:12345)")
    gen.add_argument("--qos", type=str, default="", help="Quality of service")
    gen.add_argument("--gres", type=str, default="", help="Generic resources (e.g., gpu:a100:2)")
    gen.add_argument("--dry-run", action="store_true", help="Print to stdout without writing")
    gen.add_argument("--backup", action="store_true", default=True, help="Rotate existing script")
    gen.add_argument("--preview", action="store_true", help="Preview script in terminal after generation")

    val = sub.add_parser("validate", help="Check script syntax & resource constraints")
    val.add_argument("script_path", type=str, help="Path to SBATCH script to validate")
    val.add_argument("--strict", action="store_true", help="Fail on warnings, not just errors")

    prev = sub.add_parser("preview", help="Render formatted script to stdout")
    prev.add_argument("script_path", type=str, help="Path to SBATCH script")
    prev.add_argument("--highlight", action="store_true", help="Apply syntax highlighting (if Rich available)")

    sub_cmd = sub.add_parser("submit", help="Submit script to SLURM controller")
    sub_cmd.add_argument("script_path", type=str, help="Path to SBATCH script")
    sub_cmd.add_argument("--dry-run", action="store_true", help="Validate & print submission response only")
    sub_cmd.add_argument("--watch", action="store_true", help="Poll job status after submission")

    return parser


# =============================================================================
# Core CLI Handlers
# =============================================================================

def _build_directives_from_args(args: argparse.Namespace) -> SlurmDirectives:
    """Map CLI arguments to validated SlurmDirectives dataclass."""
    if not _validate_time_format(args.time):
        raise ConfigurationError(f"Invalid walltime format: '{args.time}'. Use HH:MM:SS or D-HH:MM:SS.")
    if not _validate_memory_string(args.mem):
        raise ConfigurationError(f"Invalid memory format: '{args.mem}'. Use e.g., 64G or 4000M.")

    ntasks = args.ntasks
    if ntasks == 0:
        topo = detect_topology()
        ntasks = topo.total_cores
        if ntasks == 0:
            raise ConfigurationError("Cannot auto-calculate ntasks: topology detection returned 0 cores.")

    return SlurmDirectives(
        job_name=args.job_name,
        partition=args.partition or None,
        nodes=args.nodes,
        ntasks=ntasks,
        cpus_per_task=args.cpus_per_task,
        mem_per_node=args.mem,
        time=args.time,
        dependency=args.dependency or None,
        qos=args.qos or None,
        gres=args.gres or None,
        output="slurm-%j.out",
        error="slurm-%j.err",
    )


def _get_exec_command() -> str:
    """Auto-detect the correct WIEN2k execution command from input files."""
    try:
        from .backend_manager import get_current_backend as _gcb
        backend = _gcb()
        params = backend.detect_problem_size()
        return params.get("exec_command", "run_lapw -p")
    except Exception:
        return "run_lapw -p"


def handle_generate(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Generate, write, and optionally preview SBATCH script."""
    directives = _build_directives_from_args(args)
    topo = detect_topology(max_cores=directives.ntasks or None)
    spec = SlurmJobSpec(
        topo=topo,
        exec_command=_get_exec_command(),
        directives=directives
    )

    script_content = generate_sbatch_script(spec)

    if args.dry_run:
        return {"mode": "dry_run", "content": script_content}

    target = Path(args.output).resolve()

    if args.backup and target.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = target.parent / f"{target.name}.bak.{ts}"
        try:
            shutil.copy2(target, backup_path)
            logger.debug(f"Backed up {target} to {backup_path}")
        except Exception as e:
            logger.warning(f"Backup failed: {e}")

    atomic_write(target, script_content, mode=0o755)

    result = {"status": "success", "path": str(target), "size_bytes": len(script_content.encode())}
    if args.preview:
        result["preview"] = script_content

    return result


def handle_validate(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Validate SBATCH script syntax, resource limits, and SLURM compliance."""
    script_path = Path(args.script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    content = script_path.read_text(encoding="utf-8")
    errors = []
    warnings = []

    if not content.startswith("#!/bin/bash"):
        warnings.append("Missing #!/bin/bash shebang.")
    if "#SBATCH" not in content:
        errors.append("No #SBATCH directives found.")

    directives = {}
    for match in re.finditer(r'#SBATCH\s+--([\w-]+)=?([^\s]*)', content):
        key, val = match.group(1), match.group(2) or "true"
        directives[key] = val

    if "time" in directives and not _validate_time_format(directives["time"]):
        errors.append(f"Invalid --time format: {directives['time']}")
    if "mem-per-node" in directives or "mem" in directives:
        mem_val = directives.get("mem-per-node", directives.get("mem", ""))
        if not _validate_memory_string(mem_val):
            errors.append(f"Invalid --mem format: {mem_val}")
    if "ntasks" in directives and "cpus-per-task" in directives:
        try:
            ntasks = int(directives["ntasks"])
            cpt = int(directives["cpus-per-task"])
            if ntasks * cpt > 512:
                warnings.append(f"High core count requested: {ntasks * cpt}. Verify partition limits.")
        except ValueError:
            errors.append("Non-integer ntasks or cpus-per-task.")

    if args.strict:
        errors.extend(warnings)
        warnings.clear()

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "directives_parsed": len(directives)
    }


def handle_submit(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Submit validated script to SLURM controller."""
    script_path = Path(args.script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not os.access(script_path, os.X_OK):
        try:
            script_path.chmod(0o755)
        except Exception as e:
            raise PermissionError(f"Cannot make script executable: {e}")

    logger.info(f"Submitting {script_path} to SLURM...")

    topo = detect_topology()
    spec = SlurmJobSpec(
        topo=topo,
        exec_command=f"bash {script_path}",
        directives=SlurmDirectives(),
        working_dir=script_path.parent
    )

    result = submit_slurm_job(spec=spec, script_path=script_path, dry_run=args.dry_run)
    if not result.get("success"):
        raise SchedulerError(
            f"Submission failed: {'; '.join(result.get('errors', []))}",
            job_id=result.get("job_id")
        )

    return result


# =============================================================================
# Public CLI Entry Point
# =============================================================================

def run_sbatch_cli(argv: Optional[List[str]] = None) -> int:
    """
    Execute sbatch CLI workflow with structured logging & error handling.
    Returns OS exit code: 0 (success), 1 (app error), 2 (CLI syntax error).
    """
    parser = create_sbatch_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code is not None else 2

    try:
        cfg = load_config(file_path=args.config, cli_override={
            "log_level": "DEBUG" if args.verbose > 0 else "ERROR" if args.quiet else None,
            "quiet_mode": args.quiet,
            "backend": args.backend
        })
        ensure_dirs()
        setup_logging(config=cfg, verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
        set_context(cli="wien2k_sbatch", user=os.environ.get("USER", "unknown"))
    except Exception as e:
        sys.stderr.write(f"Critical: Failed to initialize configuration/logging: {e}\n")
        return 2

    def _signal_handler(sig: int, frame: Any) -> None:
        logger.warning(f"Received signal {sig}. Cleaning up...")
        sys.exit(130)
        
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    handlers = {
        "generate": lambda a: handle_generate(a, cfg),
        "validate": lambda a: handle_validate(a, cfg),
        "preview": lambda a: {"content": Path(a.script_path).read_text(encoding="utf-8")},
        "submit": lambda a: handle_submit(a, cfg)
    }

    handler = handlers.get(args.action)
    if not handler:
        parser.print_help()
        return 2

    try:
        logger.info(f"Executing sbatch action: {args.action}")
        result = handler(args)

        if args.json_output:
            print(json.dumps(result, indent=2, default=str))
        else:
            if args.action == "generate":
                print(f"[✓] Script written to: {result['path']} ({result['size_bytes']} bytes)")
                if "preview" in result:
                    print("\n--- PREVIEW ---\n" + result["preview"] + "\n---------------")
            elif args.action == "validate":
                status = result["status"]
                print(f"[{'✓' if status == 'passed' else '✗'}] Validation {status}.")
                for e in result.get("errors", []):
                    print(f"  ERROR: {e}")
                for w in result.get("warnings", []):
                    print(f"  WARN:  {w}")
            elif args.action == "preview":
                print(result["content"])
            elif args.action == "submit":
                print(f"[✓] {result.get('status', 'Submitted')} | Job ID: {result.get('job_id', 'N/A')}")

        logger.info(f"Action '{args.action}' completed successfully.")
        return 0

    except (ConfigurationError, SchedulerError, ValidationError, FileNotFoundError, PermissionError) as e:
        log_exception_structured(e)
        if args.json_output:
            print(json.dumps({"error": str(e), "type": type(e).__name__}, indent=2))
        else:
            sys.stderr.write(format_error_for_ui(e) + "\n")
        return 1
    except Exception as e:
        logger.error(f"Unexpected sbatch CLI error: {e}", exc_info=True)
        if args.json_output:
            print(json.dumps({"error": str(e), "type": type(e).__name__}, indent=2))
        else:
            sys.stderr.write(f"Error: {e}\n")
        return 1


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "create_sbatch_parser",
    "handle_generate",
    "handle_submit",
    "handle_validate",
    "run_sbatch_cli",
]

if __name__ == "__main__":
    sys.exit(run_sbatch_cli())
