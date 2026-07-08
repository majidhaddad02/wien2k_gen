"""
PBS/Torque Job Submission Provider – Production-Grade Integration for PBS Pro & Torque.
Implements the standard SubmitProvider interface with PBS-specific directives
for resource management, job arrays, and HPC cluster integration.

Key Features:
• PBS-specific directives: #PBS -l nodes, #PBS -l walltime, #PBS -l mem, #PBS -l ncpus
• Job array via #PBS -t 1-N with auto-indexing
• Environment variable detection: $PBS_JOBID, $PBS_NODEFILE, $PBS_NP
• Submission via qsub, cancellation via qdel, status via qstat
• Comprehensive type hints, English docstrings, and structured error handling
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import os
import re
import datetime
import subprocess
import shutil
import logging

from ..core.topology import Topology
from ..logging_config import get_logger
from .lsf import SubmitProvider

logger = get_logger(__name__)


# =============================================================================
# PBS-Specific Type Definitions
# =============================================================================

@dataclass
class PBSDirectives:
    """Core PBS job directives with type-safe defaults."""
    job_name: Optional[str] = None
    queue: Optional[str] = None
    account: Optional[str] = None
    nodes: Optional[int] = None
    ppn: Optional[int] = None               # processors per node
    ncpus: Optional[int] = None
    walltime: Optional[str] = None           # e.g., "24:00:00"
    mem: Optional[str] = None                # e.g., "64gb"
    pmem: Optional[str] = None               # memory per process
    job_array: Optional[str] = None          # e.g., "1-100"
    output: Optional[str] = None
    error: Optional[str] = None
    join_output: Optional[bool] = None
    email: Optional[str] = None
    email_events: Optional[str] = None       # e.g., "abe" (abort, begin, end)
    gpu: Optional[str] = None
    exclusive: Optional[bool] = None
    rerunnable: Optional[bool] = None

    def to_dict(self) -> Dict:
        result = {}
        for key, val in self.__dict__.items():
            if val is not None:
                result[key] = val
        return result


@dataclass
class PBSJobSpec:
    """
    Complete PBS job specification wrapper.
    Combines topology, execution command, PBS directives, and environment configuration.
    """
    topo: Topology
    exec_command: str
    directives: PBSDirectives = field(default_factory=PBSDirectives)
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
# PBS Validation Utilities
# =============================================================================

def _validate_pbs_time(time_str: str) -> bool:
    """Check if time string matches PBS accepted format: HH:MM:SS."""
    pattern = r'^\d{1,4}:\d{2}:\d{2}$'
    return bool(re.match(pattern, time_str))


def _validate_pbs_memory(mem_str: str) -> bool:
    """Validate PBS memory suffix format: digits followed by kb, mb, gb (case-insensitive)."""
    pattern = r'^\d+(kb|mb|gb|KB|MB|GB)?$'
    return bool(re.match(pattern, mem_str))


def _check_pbs_limits(spec: PBSJobSpec) -> List[str]:
    """Validate job spec against common PBS queue limits and hardware constraints."""
    warnings_list = []
    directives = spec.directives

    if directives.walltime and not _validate_pbs_time(directives.walltime):
        warnings_list.append(f"Invalid walltime format: {directives.walltime}. Expected HH:MM:SS.")

    if directives.mem and not _validate_pbs_memory(directives.mem):
        warnings_list.append(f"Invalid memory format: {directives.mem}. Expected e.g., 64gb.")

    requested_nodes = directives.nodes or 1
    ppn = directives.ppn or 1
    requested_cores = requested_nodes * ppn

    if spec.topo.total_cores > 0 and requested_cores > spec.topo.total_cores:
        warnings_list.append(
            f"Requested cores ({requested_cores}) exceed available topology cores ({spec.topo.total_cores})."
        )

    return warnings_list


# =============================================================================
# PBS Submit Provider Implementation
# =============================================================================

class PBSSubmitProvider(SubmitProvider):
    """
    PBS/Torque job submission provider.

    Generates PBS submission scripts with #PBS directives, handles job submission
    via `qsub`, cancellation via `qdel`, and status queries via `qstat`.

    Usage:
        provider = PBSSubmitProvider()
        script = provider.generate_submit_script(topo, exec_command, directives)
        result = provider.submit(topo, exec_command, directives)
        provider.cancel("12345.pbs-server")
        status = provider.status("12345.pbs-server")
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
        """Generate a complete PBS submission script with #PBS directives and execution body."""
        spec = PBSJobSpec(
            topo=topo,
            exec_command=exec_command,
            directives=PBSDirectives(**(directives or {})),
            modules_to_load=modules_to_load or [],
            environment_vars=environment_vars or {},
            working_dir=working_dir or Path.cwd(),
        )
        return self._build_pbs_script(spec)

    def _build_pbs_script(self, spec: PBSJobSpec) -> str:
        """Assemble the complete PBS script with header, directives, and execution body."""
        header = self._build_header(spec)
        directives = self._format_pbs_directives(spec)
        body = self._build_execution_body(spec)
        return f"{header}\n\n{directives}\n\n{body}\n"

    def _build_header(self, spec: PBSJobSpec) -> str:
        """Generate the script header with metadata."""
        return f"""#!/bin/bash
# ==============================================================================
# Auto-generated PBS Submission Script (wien2k_gen v0.1.0)
# Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}Z
# Backend: {spec.topo.env_type.upper()} | Topology: {spec.topo.total_cores} cores
# Scheduler: PBS/Torque
# =============================================================================="""

    def _format_pbs_directives(self, spec: PBSJobSpec) -> str:
        """Format #PBS directives with proper spacing and defaults."""
        lines = []
        directives = spec.directives

        job_name = directives.job_name or "wien2k_gen_job"
        lines.append(f"#PBS -N {job_name}")

        if directives.queue:
            lines.append(f"#PBS -q {directives.queue}")

        if directives.account:
            lines.append(f"#PBS -A {directives.account}")

        resource_parts = []

        nodes = directives.nodes or 1
        ppn = directives.ppn or (spec.topo.cores_per_node[0] if spec.topo.cores_per_node else 1)
        resource_parts.append(f"nodes={nodes}:ppn={ppn}")

        walltime = directives.walltime or "24:00:00"
        resource_parts.append(f"walltime={walltime}")

        if directives.mem:
            resource_parts.append(f"mem={directives.mem}")
        if directives.pmem:
            resource_parts.append(f"pmem={directives.pmem}")
        if directives.ncpus:
            resource_parts.append(f"ncpus={directives.ncpus}")
        if directives.gpu:
            resource_parts.append(f"ngpus={directives.gpu}")

        lines.append(f"#PBS -l {','.join(resource_parts)}")

        if directives.job_array:
            lines.append(f"#PBS -t {directives.job_array}")

        output = directives.output or "pbs-${PBS_JOBID}.out"
        if directives.join_output:
            lines.append(f"#PBS -j oe")
            lines.append(f"#PBS -o {output}")
        else:
            lines.append(f"#PBS -o {output}")
            error = directives.error or "pbs-${PBS_JOBID}.err"
            lines.append(f"#PBS -e {error}")

        if directives.email:
            lines.append(f"#PBS -M {directives.email}")
        if directives.email_events:
            lines.append(f"#PBS -m {directives.email_events}")

        if directives.exclusive:
            lines.append("#PBS -l place=excl")

        if directives.rerunnable if directives.rerunnable is not None else True:
            lines.append("#PBS -r y")
        else:
            lines.append("#PBS -r n")

        lines.append("#PBS -S /bin/bash")

        lines.append(f"#PBS -d {spec.working_dir}")

        return "\n".join(lines)

    def _build_execution_body(self, spec: PBSJobSpec) -> str:
        """Construct the main execution body with environment setup and command invocation."""
        lines = []

        lines.append("# PBS Environment & Host Detection")
        lines.append('echo "[pbs_submit] Job ID: $PBS_JOBID"')
        lines.append('echo "[pbs_submit] Nodfile: $PBS_NODEFILE"')
        lines.append('echo "[pbs_submit] Total procs: $PBS_NP"')
        lines.append('echo "[pbs_submit] Hostname: $(hostname)"')
        lines.append("")

        if spec.modules_to_load:
            lines.append("# Load required modules")
            lines.append(f"module load {' '.join(spec.modules_to_load)}")
            lines.append("")

        if spec.environment_vars:
            lines.append("# Set job environment variables")
            for key, value in spec.environment_vars.items():
                lines.append(f'export {key}="{value}"')
            lines.append("")

        lines.append(f"cd {spec.working_dir} || exit 1")
        lines.append('echo "[pbs_submit] Working directory: $(pwd)"')
        lines.append("")

        if spec.scratch_enabled:
            lines.extend(self._inject_scratch_setup())

        lines.extend(self._inject_preemption_handler(spec))
        lines.append("")

        lines.append("# Execute calculation")
        lines.append('echo "[pbs_submit] Launching: %s"' % spec.exec_command)
        lines.append(f'exec {spec.exec_command} "$@"')
        lines.append("EXIT_CODE=$?")
        lines.append("exit $EXIT_CODE")

        return "\n".join(lines)

    def _inject_scratch_setup(self) -> List[str]:
        """Generate PBS scratch directory setup with node synchronization."""
        return [
            "# ==============================================================================",
            "# Scratch & I/O Staging",
            "# ==============================================================================",
            'SCRATCH_BASE=$(mktemp -d -p /dev/shm 2>/dev/null || mktemp -d -p ${SCRATCH:-/scratch} 2>/dev/null || mktemp -d)',
            'export JOB_SCRATCH="$SCRATCH_BASE"',
            'export TMPDIR="$SCRATCH_BASE"',
            'echo "[pbs_submit] Scratch allocated at $SCRATCH_BASE on $(hostname)"',
            '',
            '# Lustre striping for parallel MPI-IO (case.vector writes)',
            'if command -v lfs &> /dev/null && [ "$(stat -f -c %T "$SCRATCH_BASE" 2>/dev/null)" = "lustre" ]; then',
            '    echo "[pbs_submit] Lustre detected: configuring striping on $SCRATCH_BASE"',
            '    lfs setstripe -c ${PBS_NCPUS:-4} -s 1M "$SCRATCH_BASE" 2>/dev/null || true',
            'fi',
            '',
            '# Multi-node scratch sync',
            'if [ -n "$PBS_NODEFILE" ] && [ "$(sort -u "$PBS_NODEFILE" | wc -l)" -gt 1 ]; then',
            '    echo "[pbs_submit] Multi-node job detected. Syncing scratch across compute nodes..."',
            '    for _host in $(sort -u "$PBS_NODEFILE"); do',
            '        [ "$_host" = "$(hostname)" ] && continue',
            '        rsync -a "$SCRATCH_BASE/" "${_host}:$SCRATCH_BASE/" 2>/dev/null || true',
            '    done',
            'fi',
            '',
            '# Cleanup on exit',
            'cleanup_scratch() {',
            '    echo "[pbs_submit] Cleaning up scratch on $(hostname)..."',
            '    rm -rf "$SCRATCH_BASE" 2>/dev/null || true',
            '}',
            'trap cleanup_scratch EXIT',
        ]

    def _inject_preemption_handler(self, spec: PBSJobSpec) -> List[str]:
        """Generate signal trap for graceful preemption handling."""
        return [
            "# ==============================================================================",
            "# Preemption & Signal Resilience",
            "# ==============================================================================",
            "_preemption_handler() {",
            '    echo "[pbs_submit] Preemption / walltime signal received. Triggering clean exit..."',
            "    sync",
            f"    sleep {max(2, spec.preemption_grace_sec - 5)}",
            "    exit 143",
            "}",
            "trap _preemption_handler TERM USR1",
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
        Submit a PBS job or generate a script for review.

        Args:
            topo: Hardware topology for resource allocation.
            exec_command: The MPI/application command to execute.
            directives: PBS-specific directives (queue, nodes, walltime, etc.).
            script_path: Output path for the PBS script. Defaults to `pbs_submit_<job_name>.sh`.
            dry_run: If True, return script content without writing or submitting.

        Returns:
            Dict with keys: success, job_id, script_path, dry_run_content, errors, warnings.
        """
        spec = PBSJobSpec(
            topo=topo,
            exec_command=exec_command,
            directives=PBSDirectives(**(directives or {})),
            working_dir=kwargs.get("working_dir", Path.cwd()),
            modules_to_load=kwargs.get("modules_to_load", []),
            environment_vars=kwargs.get("environment_vars", {}),
            dry_run=dry_run,
            validate_constraints=kwargs.get("validate_constraints", True),
        )

        result: Dict[str, Any] = {
            "success": False,
            "job_id": None,
            "script_path": script_path or Path(f"pbs_submit_{spec.directives.job_name or 'job'}.sh"),
            "dry_run_content": None,
            "errors": [],
            "warnings": [],
        }

        if spec.validate_constraints:
            result["warnings"].extend(_check_pbs_limits(spec))

        try:
            script_content = self._build_pbs_script(spec)
        except Exception as exc:
            result["errors"].append(f"Script generation failed: {exc}")
            return result

        if dry_run:
            result["dry_run_content"] = script_content
            result["success"] = True
            logger.info("PBS script generated in dry-run mode. Review before submission.")
            return result

        if kwargs.get("backup", True) and result["script_path"].exists():
            try:
                backup_path = result["script_path"].with_suffix(".sh.bak")
                shutil.copy2(result["script_path"], backup_path)
                logger.debug(f"Backed up {result['script_path']} to {backup_path}")
            except Exception as exc:
                logger.warning(f"Backup failed: {exc}")

        try:
            result["script_path"].write_text(script_content, encoding="utf-8")
            result["script_path"].chmod(0o755)
            logger.info(f"PBS script written to {result['script_path']}")
        except Exception as exc:
            result["errors"].append(f"Failed to write script: {exc}")
            return result

        try:
            logger.info("Submitting job via qsub...")
            proc = subprocess.run(
                ["qsub", str(result["script_path"])],
                capture_output=True, text=True, timeout=10,
            )

            if proc.returncode == 0:
                result["job_id"] = proc.stdout.strip()
                result["success"] = True
                logger.info(f"Job submitted successfully. Job ID: {result['job_id']}")
            else:
                result["errors"].append(f"qsub failed: {proc.stderr.strip()}")
                logger.error(f"Job submission failed: {proc.stderr.strip()}")

        except subprocess.TimeoutExpired:
            result["errors"].append("qsub command timed out. Check PBS server connectivity.")
        except Exception as exc:
            result["errors"].append(f"Submission exception: {exc}")

        return result

    # =========================================================================
    # Job Management
    # =========================================================================

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running or pending PBS job.

        Args:
            job_id: The PBS job ID to cancel (e.g., "12345.pbs-server").

        Returns:
            True if cancellation command executed successfully, False otherwise.
        """
        try:
            proc = subprocess.run(
                ["qdel", str(job_id)],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                logger.info(f"Job {job_id} cancelled successfully.")
                return True
            else:
                logger.warning(f"qdel returned non-zero: {proc.stderr.strip()}")
                return "Unknown Job Id" in proc.stderr or "finished" in proc.stderr.lower()
        except subprocess.TimeoutExpired:
            logger.error(f"qdel timed out for job {job_id}.")
            return False
        except FileNotFoundError:
            logger.error("qdel command not found. Is PBS/Torque installed?")
            return False
        except Exception as exc:
            logger.error(f"Failed to cancel job {job_id}: {exc}")
            return False

    def status(self, job_id: str) -> Dict[str, Any]:
        """
        Query the current status of a PBS job.

        Args:
            job_id: The PBS job ID to query.

        Returns:
            Dict with keys: job_id, state, queue, exit_code, running.
            State is one of: Q (queued), R (running), C (completed), H (held), E (exiting), UNKWN.
        """
        state_map = {
            "Q": "queued",
            "R": "running",
            "C": "completed",
            "H": "held",
            "E": "exiting",
            "T": "moved",
            "W": "waiting",
            "S": "suspended",
        }

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
                ["qstat", "-f", str(job_id)],
                capture_output=True, text=True, timeout=10,
            )
            result["raw_output"] = proc.stdout.strip()

            if proc.returncode == 0 and proc.stdout.strip():
                for line in proc.stdout.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("job_state"):
                        raw_state = stripped.split("=")[-1].strip()
                        result["state"] = state_map.get(raw_state, raw_state)
                        result["running"] = raw_state == "R"
                    elif stripped.startswith("queue"):
                        result["queue"] = stripped.split("=")[-1].strip()
                    elif stripped.startswith("exit_status"):
                        result["exit_code"] = int(stripped.split("=")[-1].strip())
            else:
                result["state"] = "UNKWN"

        except subprocess.TimeoutExpired:
            logger.error(f"qstat timed out for job {job_id}.")
        except FileNotFoundError:
            logger.error("qstat command not found. Is PBS/Torque installed?")
        except Exception as exc:
            logger.error(f"Failed to query status for job {job_id}: {exc}")

        return result

    def get_job_id_from_env(self) -> Optional[str]:
        """
        Retrieve the current job ID from PBS environment variables.

        Checks $PBS_JOBID.

        Returns:
            Job ID string if running inside a PBS job, None otherwise.
        """
        return os.environ.get("PBS_JOBID")
