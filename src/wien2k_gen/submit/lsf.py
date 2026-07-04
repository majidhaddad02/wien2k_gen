"""
LSF Job Submission Provider – Production-Grade Integration for IBM Spectrum LSF.
Implements the standard SubmitProvider interface with LSF-specific directives,
job array support, and optional jsrun integration for IBM Spectrum LSF on POWER.

Key Features:
• LSF-specific directives: #BSUB -n, #BSUB -W, #BSUB -M, #BSUB -R "span[hosts=...]"
• Job array via #BSUB -J "jobname[1-N]" with auto-indexing
• Environment variable detection: $LSB_JOBID, $LSB_MCPU_HOSTS, $LSB_DJOB_NUMPROC
• Submission via bsub, cancellation via bkill, status via bjobs
• Optional jsrun integration for IBM Spectrum LSF (#BSUB -o jsrun)
• Comprehensive type hints, English docstrings, and HPC-grade error handling
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass, field
import os
import re
import time
import datetime
import subprocess
import shutil
import logging

from ..core.topology import Topology
from ..logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Abstract Base Class – SubmitProvider
# =============================================================================

class SubmitProvider(ABC):
    """
    Abstract base class for all HPC scheduler submit providers.
    Defines the standard interface for script generation, job submission,
    cancellation, status querying, and environment-level job detection.

    Design principles:
    • Minimal required interface: 5 abstract methods for core functionality
    • Type-safe data contracts for submission results and job states
    • Scheduler-agnostic API with scheduler-specific directive encoding
    • Built-in support for dry-run, validation, and structured logging
    """

    @abstractmethod
    def generate_submit_script(
        self,
        topo: Topology,
        exec_command: str,
        directives: Optional[Dict[str, Any]] = None,
        modules_to_load: Optional[List[str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[Path] = None,
    ) -> str:
        """Generate a complete submission script with scheduler directives."""

    @abstractmethod
    def submit(
        self,
        topo: Topology,
        exec_command: str,
        directives: Optional[Dict[str, Any]] = None,
        script_path: Optional[Path] = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit job to the scheduler and return structured result."""

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running or pending job by its scheduler ID."""

    @abstractmethod
    def status(self, job_id: str) -> Dict[str, Any]:
        """Query the current status of a job by its scheduler ID."""

    @abstractmethod
    def get_job_id_from_env(self) -> Optional[str]:
        """Retrieve the current job ID from the scheduler environment variables."""


# =============================================================================
# LSF-Specific Type Definitions
# =============================================================================

class LSFDirectives(TypedDict, total=False):
    """Core LSF job directives with type-safe defaults."""
    job_name: str
    queue: str
    project: str
    nodes: int
    nprocs: int
    walltime: str           # e.g., "24:00"  (HH:MM)
    memory: str             # e.g., "64G" or "64000" (MB)
    span_hosts: str         # e.g., "span[hosts=1]" or "span[ptile=16]"
    rusage_mem: str         # e.g., "rusage[mem=64G]"
    output: str
    error: str
    job_array: str          # e.g., "1-100"
    email: str
    email_when: str         # e.g., "began,end"
    exclusive: bool
    gpu: str                # e.g., "num=1:mode=shared:j_exclusive=no"
    jsrun: bool             # IBM Spectrum LSF jsrun integration
    pre_exec: str
    post_exec: str


@dataclass
class LSFJobSpec:
    """
    Complete LSF job specification wrapper.
    Combines topology, execution command, LSF directives, and environment configuration.
    """
    topo: Topology
    exec_command: str
    directives: LSFDirectives = field(default_factory=dict)
    working_dir: Path = field(default_factory=Path.cwd)
    modules_to_load: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    scratch_enabled: bool = True
    preemption_grace_sec: int = 60
    dry_run: bool = False
    validate_constraints: bool = True
    stripe_count: int = 4
    stripe_size_mb: int = 1


# =============================================================================
# LSF Validation Utilities
# =============================================================================

def _validate_lsf_time(time_str: str) -> bool:
    """Check if time string matches LSF accepted format: HH:MM."""
    pattern = r'^\d{1,3}:\d{2}$'
    return bool(re.match(pattern, time_str))


def _validate_lsf_memory(mem_str: str) -> bool:
    """Validate LSF memory suffix format: digits optionally followed by G, M, K (case-insensitive)."""
    pattern = r'^\d+[GMKkmg]?$'
    return bool(re.match(pattern, mem_str))


def _check_lsf_limits(spec: LSFJobSpec) -> List[str]:
    """Validate job spec against common LSF queue limits and hardware constraints."""
    warnings_list = []
    directives = spec.directives

    if directives.get("walltime") and not _validate_lsf_time(directives["walltime"]):
        warnings_list.append(f"Invalid walltime format: {directives['walltime']}. Expected HH:MM.")

    if directives.get("memory") and not _validate_lsf_memory(directives["memory"]):
        warnings_list.append(f"Invalid memory format: {directives['memory']}. Expected e.g., 64G.")

    requested_cores = directives.get("nprocs", 1)
    if spec.topo.total_cores > 0 and requested_cores > spec.topo.total_cores:
        warnings_list.append(
            f"Requested cores ({requested_cores}) exceed available topology cores ({spec.topo.total_cores})."
        )

    return warnings_list


# =============================================================================
# LSF Submit Provider Implementation
# =============================================================================

class LSFSubmitProvider(SubmitProvider):
    """
    LSF (IBM Spectrum LSF) job submission provider.

    Generates LSF submission scripts with #BSUB directives, handles job submission
    via `bsub`, cancellation via `bkill`, and status queries via `bjobs`.

    Optional jsrun integration for IBM Spectrum LSF on POWER architectures
    is supported via the `#BSUB -o jsrun` directive.

    Usage:
        provider = LSFSubmitProvider()
        script = provider.generate_submit_script(topo, exec_command, directives)
        result = provider.submit(topo, exec_command, directives)
        provider.cancel("12345")
        status = provider.status("12345")
    """

    # =========================================================================
    # Script Generation
    # =========================================================================

    def generate_submit_script(
        self,
        topo: Topology,
        exec_command: str,
        directives: Optional[Dict[str, Any]] = None,
        modules_to_load: Optional[List[str]] = None,
        environment_vars: Optional[Dict[str, str]] = None,
        working_dir: Optional[Path] = None,
    ) -> str:
        """Generate a complete LSF submission script with #BSUB directives and execution body."""
        spec = LSFJobSpec(
            topo=topo,
            exec_command=exec_command,
            directives=directives or {},
            modules_to_load=modules_to_load or [],
            environment_vars=environment_vars or {},
            working_dir=working_dir or Path.cwd(),
        )
        return self._build_lsf_script(spec)

    def _build_lsf_script(self, spec: LSFJobSpec) -> str:
        """Assemble the complete LSF script with header, directives, and execution body."""
        header = self._build_header(spec)
        directives = self._format_bsub_directives(spec)
        body = self._build_execution_body(spec)
        return f"{header}\n\n{directives}\n\n{body}\n"

    def _build_header(self, spec: LSFJobSpec) -> str:
        """Generate the script header with metadata."""
        return f"""#!/bin/bash
# ==============================================================================
# Auto-generated LSF Submission Script (wien2k_gen v9.8.0)
# Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z
# Backend: {spec.topo.env_type.upper()} | Topology: {spec.topo.total_cores} cores
# Scheduler: IBM Spectrum LSF
# =============================================================================="""

    def _format_bsub_directives(self, spec: LSFJobSpec) -> str:
        """Format #BSUB directives with proper spacing and defaults."""
        lines = []
        directives = spec.directives

        # Job name
        job_name = directives.get("job_name", "wien2k_gen_job")
        array_spec = directives.get("job_array", "")
        if array_spec:
            lines.append(f'#BSUB -J "{job_name}[{array_spec}]"')
        else:
            lines.append(f'#BSUB -J "{job_name}"')

        # Queue
        if directives.get("queue"):
            lines.append(f"#BSUB -q {directives['queue']}")

        # Project / Account
        if directives.get("project"):
            lines.append(f"#BSUB -P {directives['project']}")

        # Number of processors
        nprocs = directives.get("nprocs", spec.topo.total_cores or 1)
        lines.append(f"#BSUB -n {nprocs}")

        # Walltime (LSF format: HH:MM)
        walltime = directives.get("walltime", "24:00")
        lines.append(f"#BSUB -W {walltime}")

        # Memory limit
        memory = directives.get("memory", "")
        if memory:
            lines.append(f"#BSUB -M {memory}")

        # Resource requirement strings
        rusage_mem = directives.get("rusage_mem", "")
        if rusage_mem:
            lines.append(f'#BSUB -R "rusage[mem={rusage_mem}]"')

        span_hosts = directives.get("span_hosts", f"span[hosts={directives.get('nodes', 1)}]")
        if span_hosts:
            lines.append(f'#BSUB -R "{span_hosts}"')

        # Exclusive node access
        if directives.get("exclusive"):
            lines.append("#BSUB -x")

        # GPU resource
        if directives.get("gpu"):
            lines.append(f'#BSUB -gpu "{directives["gpu"]}"')

        # jsrun integration for IBM Spectrum LSF
        if directives.get("jsrun"):
            lines.append("#BSUB -o jsrun")

        # Output and error files
        output = directives.get("output", "lsf-%J.out")
        error = directives.get("error", "lsf-%J.err")
        lines.append(f"#BSUB -o {output}")
        lines.append(f"#BSUB -e {error}")

        # Email notifications
        if directives.get("email"):
            lines.append(f"#BSUB -u {directives['email']}")
        if directives.get("email_when"):
            lines.append(f"#BSUB -N -B -N {directives['email_when']}")

        # Pre-execution and post-execution commands
        if directives.get("pre_exec"):
            lines.append(f"#BSUB -E \"{directives['pre_exec']}\"")
        if directives.get("post_exec"):
            lines.append(f"#BSUB -Ep \"{directives['post_exec']}\"")

        # Resource limit for cores per host
        cores_per_node = spec.topo.cores_per_node[0] if spec.topo.cores_per_node else nprocs
        lines.append(f'#BSUB -R "affinity[core({cores_per_node})]"')

        return "\n".join(lines)

    def _build_execution_body(self, spec: LSFJobSpec) -> str:
        """Construct the main execution body with environment setup and command invocation."""
        lines = []

        # Environment detection
        lines.append("# LSF Environment & Host Detection")
        lines.append('echo "[lsf_submit] Job ID: $LSB_JOBID"')
        lines.append('echo "[lsf_submit] Hosts: $LSB_MCPU_HOSTS"')
        lines.append('echo "[lsf_submit] MPI procs: ${LSB_DJOB_NUMPROC:-$LSB_MCPU_HOSTS}"')
        lines.append('echo "[lsf_submit] Hostname: $(hostname)"')
        lines.append("")

        # Module loading
        if spec.modules_to_load:
            lines.append("# Load required modules")
            lines.append(f"module load {' '.join(spec.modules_to_load)}")
            lines.append("")

        # Environment variables
        if spec.environment_vars:
            lines.append("# Set job environment variables")
            for key, value in spec.environment_vars.items():
                lines.append(f'export {key}="{value}"')
            lines.append("")

        # Working directory
        lines.append(f"cd {spec.working_dir} || exit 1")
        lines.append('echo "[lsf_submit] Working directory: $(pwd)"')
        lines.append("")

        # Scratch directory setup
        if spec.scratch_enabled:
            lines.extend(self._inject_scratch_setup())

        # Preemption handler
        lines.extend(self._inject_preemption_handler(spec))
        lines.append("")

        # jsrun launch command override
        if spec.directives.get("jsrun"):
            lines.extend(self._inject_jsrun_command(spec))
        else:
            # Standard MPI execution
            lines.append("# Execute calculation")
            lines.append(f'exec {spec.exec_command} "$@"')
            lines.append("EXIT_CODE=$?")
            lines.append("exit $EXIT_CODE")

        return "\n".join(lines)

    def _inject_scratch_setup(self) -> List[str]:
        """Generate LSF scratch directory setup for multi-host I/O staging."""
        return [
            "# ==============================================================================",
            "# Scratch & I/O Staging",
            "# ==============================================================================",
            'SCRATCH_BASE=$(mktemp -d -p /dev/shm 2>/dev/null || mktemp -d -p ${SCRATCH:-/scratch} 2>/dev/null || mktemp -d)',
            'export JOB_SCRATCH="$SCRATCH_BASE"',
            'export TMPDIR="$SCRATCH_BASE"',
            'echo "[lsf_submit] Scratch allocated at $SCRATCH_BASE on $(hostname)"',
            '',
            '# Lustre striping for parallel MPI-IO (case.vector writes)',
            'if command -v lfs &> /dev/null && [ "$(stat -f -c %T "$SCRATCH_BASE" 2>/dev/null)" = "lustre" ]; then',
            '    echo "[lsf_submit] Lustre detected: configuring striping on $SCRATCH_BASE"',
            '    lfs setstripe -c ${LSB_DJOB_NUMPROC:-4} -s 1M "$SCRATCH_BASE" 2>/dev/null || true',
            'fi',
            '',
            '# Multi-host scratch sync (if LSF assigned multiple hosts)',
            'if [ -n "$LSB_MCPU_HOSTS" ]; then',
            '    HOST_COUNT=$(echo "$LSB_MCPU_HOSTS" | wc -w)',
            '    if [ "$((HOST_COUNT / 2))" -gt 1 ]; then',
            '        echo "[lsf_submit] Multi-host job detected. Syncing scratch across nodes..."',
            '        for _host in $(echo "$LSB_MCPU_HOSTS" | tr " " "\\n" | grep -v "^[0-9]*$"); do',
            '            rsync -a "$SCRATCH_BASE/" "${_host}:$SCRATCH_BASE/" 2>/dev/null || true',
            '        done',
            '    fi',
            'fi',
            '',
            '# Cleanup on exit',
            'cleanup_scratch() {',
            '    echo "[lsf_submit] Cleaning up scratch on $(hostname)..."',
            '    rm -rf "$SCRATCH_BASE" 2>/dev/null || true',
            '}',
            'trap cleanup_scratch EXIT',
        ]

    def _inject_preemption_handler(self, spec: LSFJobSpec) -> List[str]:
        """Generate signal trap for graceful preemption and SIGTERM handling."""
        return [
            "# ==============================================================================",
            "# Preemption & Signal Resilience",
            "# ==============================================================================",
            "_preemption_handler() {",
            '    echo "[lsf_submit] Preemption / walltime signal received. Triggering clean exit..."',
            "    sync",
            f"    sleep {max(2, spec.preemption_grace_sec - 5)}",
            "    exit 143",
            "}",
            "trap _preemption_handler TERM USR2 XCPU",
        ]

    def _inject_jsrun_command(self, spec: LSFJobSpec) -> List[str]:
        """Generate jsrun-based launch command for IBM Spectrum LSF on POWER."""
        nprocs = spec.directives.get("nprocs", 1)
        nodes = spec.directives.get("nodes", 1)
        cpus_per_rs = max(1, nprocs // nodes) if nodes else nprocs

        return [
            "# ==============================================================================",
            "# jsrun Integration (IBM Spectrum LSF on POWER)",
            "# ==============================================================================",
            'echo "[lsf_submit] Launching via jsrun for IBM Spectrum LSF..."',
            f"jsrun --nrs=1 --rs_per_host=1 --tasks_per_rs={nprocs} --cpu_per_rs={cpus_per_rs} "
            f"--gpu_per_rs=0 --latency_priority=cpu-cpu --bind=packed:1 "
            f"{spec.exec_command}",
        ]

    # =========================================================================
    # Submission API
    # =========================================================================

    def submit(
        self,
        topo: Topology,
        exec_command: str,
        directives: Optional[Dict[str, Any]] = None,
        script_path: Optional[Path] = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit an LSF job or generate a script for review.

        Args:
            topo: Hardware topology for resource allocation.
            exec_command: The MPI/application command to execute.
            directives: LSF-specific directives (queue, nprocs, walltime, etc.).
            script_path: Output path for the LSF script. Defaults to `lsf_submit_<job_name>.sh`.
            dry_run: If True, return script content without writing or submitting.

        Returns:
            Dict with keys: success, job_id, script_path, dry_run_content, errors, warnings.
        """
        spec = LSFJobSpec(
            topo=topo,
            exec_command=exec_command,
            directives=directives or {},
            working_dir=kwargs.get("working_dir", Path.cwd()),
            modules_to_load=kwargs.get("modules_to_load", []),
            environment_vars=kwargs.get("environment_vars", {}),
            dry_run=dry_run,
            validate_constraints=kwargs.get("validate_constraints", True),
        )

        result: Dict[str, Any] = {
            "success": False,
            "job_id": None,
            "script_path": script_path or Path(f"lsf_submit_{spec.directives.get('job_name', 'job')}.sh"),
            "dry_run_content": None,
            "errors": [],
            "warnings": [],
        }

        # Validation
        if spec.validate_constraints:
            result["warnings"].extend(_check_lsf_limits(spec))

        # Generate script
        try:
            script_content = self._build_lsf_script(spec)
        except Exception as exc:
            result["errors"].append(f"Script generation failed: {exc}")
            return result

        # Dry-run mode
        if dry_run:
            result["dry_run_content"] = script_content
            result["success"] = True
            logger.info("LSF script generated in dry-run mode. Review before submission.")
            return result

        # Backup existing script
        if kwargs.get("backup", True) and result["script_path"].exists():
            try:
                backup_path = result["script_path"].with_suffix(".sh.bak")
                shutil.copy2(result["script_path"], backup_path)
                logger.debug(f"Backed up {result['script_path']} to {backup_path}")
            except Exception as exc:
                logger.warning(f"Backup failed: {exc}")

        # Write script
        try:
            result["script_path"].write_text(script_content, encoding="utf-8")
            result["script_path"].chmod(0o755)
            logger.info(f"LSF script written to {result['script_path']}")
        except Exception as exc:
            result["errors"].append(f"Failed to write script: {exc}")
            return result

        # Submit via bsub
        try:
            logger.info("Submitting job via bsub...")
            proc = subprocess.run(
                ["bsub", "<", str(result["script_path"])],
                capture_output=True, text=True, timeout=10, shell=True,
            )

            if proc.returncode == 0:
                match = re.search(r"Job <(\d+)>", proc.stdout)
                result["job_id"] = match.group(1) if match else None
                result["success"] = True
                logger.info(f"Job submitted successfully. Job ID: {result['job_id']}")
            else:
                result["errors"].append(f"bsub failed: {proc.stderr.strip()}")
                logger.error(f"Job submission failed: {proc.stderr.strip()}")

        except subprocess.TimeoutExpired:
            result["errors"].append("bsub command timed out. Check LSF controller connectivity.")
        except Exception as exc:
            result["errors"].append(f"Submission exception: {exc}")

        return result

    # =========================================================================
    # Job Management
    # =========================================================================

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running or pending LSF job.

        Args:
            job_id: The LSF job ID to cancel.

        Returns:
            True if cancellation command executed successfully, False otherwise.
        """
        try:
            proc = subprocess.run(
                ["bkill", str(job_id)],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                logger.info(f"Job {job_id} cancelled successfully.")
                return True
            else:
                logger.warning(f"bkill returned non-zero: {proc.stderr.strip()}")
                return "already finished" in proc.stderr.lower() or "not found" in proc.stderr.lower()
        except subprocess.TimeoutExpired:
            logger.error(f"bkill timed out for job {job_id}.")
            return False
        except FileNotFoundError:
            logger.error("bkill command not found. Is LSF installed?")
            return False
        except Exception as exc:
            logger.error(f"Failed to cancel job {job_id}: {exc}")
            return False

    def status(self, job_id: str) -> Dict[str, Any]:
        """
        Query the current status of an LSF job.

        Args:
            job_id: The LSF job ID to query.

        Returns:
            Dict with keys: job_id, state, queue, exit_code, running.
            State is one of: PEND, RUN, DONE, EXIT, PSUSP, USUSP, SSUSP, UNKWN.
        """
        result: Dict[str, Any] = {
            "job_id": job_id,
            "state": "UNKWN",
            "queue": "",
            "exit_code": None,
            "running": False,
            "raw_output": "",
        }

        try:
            proc = subprocess.run(
                ["bjobs", "-o", "jobid stat queue exit_code", "-noheader", str(job_id)],
                capture_output=True, text=True, timeout=10,
            )
            result["raw_output"] = proc.stdout.strip()

            if proc.returncode == 0 and proc.stdout.strip():
                parts = proc.stdout.strip().split()
                if len(parts) >= 2:
                    result["state"] = parts[1] if len(parts) > 1 else "UNKWN"
                    result["queue"] = parts[2] if len(parts) > 2 else ""
                    result["exit_code"] = parts[3] if len(parts) > 3 else None
                    result["running"] = result["state"] == "RUN"
            else:
                result["state"] = "UNKWN"

        except subprocess.TimeoutExpired:
            logger.error(f"bjobs timed out for job {job_id}.")
        except FileNotFoundError:
            logger.error("bjobs command not found. Is LSF installed?")
        except Exception as exc:
            logger.error(f"Failed to query status for job {job_id}: {exc}")

        return result

    def get_job_id_from_env(self) -> Optional[str]:
        """
        Retrieve the current job ID from LSF environment variables.

        Checks $LSB_JOBID first, then $LSF_JOBID as a fallback.

        Returns:
            Job ID string if running inside an LSF job, None otherwise.
        """
        return os.environ.get("LSB_JOBID") or os.environ.get("LSF_JOBID")
