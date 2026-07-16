"""
Live SCF monitor with Rich terminal display for HPC workflows.

Layers:
  1. Polling engine: reads .scf/.output files and SLURM/PBS job status
  2. Display: Rich Live panel showing energy, convergence, resources, events
  3. Multi-job: list active jobs and select one to watch

Usage (CLI):
  forge monitor              # list active jobs
  forge monitor Fe_scf       # watch specific job
  forge monitor --all        # dashboard of all running jobs
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class JobInfo:
    job_id: str
    name: str
    status: str = "PENDING"
    elapsed: str = "00:00:00"
    time_limit: str = "01:00:00"
    nodes: int = 1
    cores: int = 1
    output_file: str = ""


@dataclass
class SCFSnapshot:
    cycle: int = 0
    total_cycles: int = 40
    energy: float = 0.0
    delta_energy: float = 1.0
    charge_distance: float = 1.0
    converged: bool = False
    convergence_type: str = "unknown"
    walltime_per_cycle: float = 0.0
    estimated_remaining: str = "--:--:--"


@dataclass
class ResourceSnapshot:
    cpu_pct: float = 0.0
    mem_pct: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    io_mb_s: float = 0.0


@dataclass
class MonitorState:
    job: JobInfo = field(default_factory=JobInfo)
    scf: SCFSnapshot = field(default_factory=SCFSnapshot)
    resources: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    events: list[str] = field(default_factory=list)
    running: bool = True
    paused: bool = False


def detect_active_jobs(scheduler: Optional[str] = None) -> list[JobInfo]:  # noqa: C901
    """Detect running WIEN2k jobs from the batch scheduler."""
    scheduler = scheduler or _detect_scheduler()
    jobs: list[JobInfo] = []

    if scheduler == "slurm":
        import subprocess
        try:
            result = subprocess.run(
                ["squeue", "--me", "--noheader", "--format=%i %j %T %M %l %D %c"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                if any(kw in parts[1].lower() for kw in ("lapw", "wien2k", "scf")):
                    jobs.append(JobInfo(
                        job_id=parts[0], name=parts[1], status=parts[2],
                        elapsed=parts[3], time_limit=parts[4],
                        nodes=int(parts[5]), cores=int(parts[6]),
                    ))
        except Exception:
            logger.debug("Suppressed exception in detect_active_jobs()", exc_info=True)
    elif scheduler == "pbs":
        import subprocess
        try:
            result = subprocess.run(
                ["qstat", "-u", os.environ.get("USER", "")],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) < 6:
                    continue
                if any(kw in parts[1].lower() for kw in ("lapw", "wien2k", "scf")):
                    jobs.append(JobInfo(
                        job_id=parts[0], name=parts[1], status=parts[4],
                        elapsed=parts[3], time_limit=parts[5] if len(parts) > 5 else "01:00:00",
                    ))
        except Exception:
            logger.debug("Suppressed exception in detect_active_jobs()", exc_info=True)

    return jobs


def _detect_scheduler() -> str:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME"):
        return "slurm"
    if os.environ.get("PBS_JOBID"):
        return "pbs"
    if os.environ.get("LSB_JOBID"):
        return "lsf"
    return "local"


def _parse_scf_output(filepath: Path) -> SCFSnapshot:
    """Parse WIEN2k .scf output for current SCF cycle status."""
    snap = SCFSnapshot()
    if not filepath.exists():
        return snap

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return snap

    energy_matches = re.findall(r":ENE\s*:\s*.*?(-?\d+\.\d+)", content)
    if energy_matches:
        energies = [float(e) for e in energy_matches]
        snap.energy = energies[-1]
        if len(energies) >= 2:
            snap.delta_energy = abs(energies[-1] - energies[-2])

    charge_matches = re.findall(r":DIS\s*:\s*.*?(\d+\.\d+)", content)
    if charge_matches:
        snap.charge_distance = float(charge_matches[-1])

    cycle_matches = re.findall(r"\*\*.*?(?:cycle|lapw|ITERATION)\s*[:#]?\s*(\d+)", content, re.IGNORECASE)
    if cycle_matches:
        snap.cycle = int(cycle_matches[-1])

    if snap.charge_distance < 1e-5 and snap.delta_energy < 1e-5:
        snap.converged = True
        snap.convergence_type = "converged"
    elif snap.delta_energy < 1e-4 and snap.charge_distance < 1e-3:
        snap.convergence_type = "converging"

    return snap


def _parse_scf_events(filepath: Path) -> list[str]:
    """Extract recent events (lapw0/1/2 completion, errors) from dayfile."""
    events: list[str] = []
    dayfile = filepath.with_suffix(".dayfile")
    if not dayfile.exists():
        return events

    try:
        lines = dayfile.read_text(encoding="utf-8", errors="replace").splitlines()
        recent = lines[-8:] if len(lines) > 8 else lines
        for line in recent:
            line = line.strip()
            if "LAPW0" in line.upper() and "END" in line.upper():
                events.append(f"[dim]{line}[/dim]")
            elif "LAPW1" in line.upper() and ("END" in line.upper() or "CPU" in line.upper()):
                events.append(f"[cyan]{line}[/cyan]")
            elif "LAPW2" in line.upper() and ("END" in line.upper() or "CPU" in line.upper()):
                events.append(f"[blue]{line}[/blue]")
            elif "ERROR" in line.upper() or "FAIL" in line.upper():
                events.append(f"[red]{line}[/red]")
    except Exception:
        logger.debug("Suppressed exception in _parse_scf_events()", exc_info=True)

    return events


def _build_monitor_panel(state: MonitorState) -> Panel:
    """Build the Rich renderable for the live monitor display."""
    job = state.job
    scf = state.scf
    res = state.resources

    job_title = f"Job: {job.name} [{job.job_id}]" if job.job_id else f"Job: {job.name}"
    status_color = {"RUNNING": "green", "PENDING": "yellow", "COMPLETED": "cyan"}.get(job.status, "white")
    status_line = f"Status: [{status_color}]{job.status}[/{status_color}] ({job.elapsed} / {job.time_limit})"

    scf_pct = min(100, int(scf.cycle / max(scf.total_cycles, 1) * 100))
    conv_color = "green" if scf.converged else ("yellow" if scf.delta_energy < 1e-4 else "white")
    scf_line = (
        f"SCF: {scf.cycle}/{scf.total_cycles} iterations ({scf_pct}%) "
        f"[{'green' if scf_pct > 50 else 'yellow'}]" + "█" * (scf_pct // 5) + "░" * (20 - scf_pct // 5) + "[/]"
    )
    energy_line = (
        f"Energy: {scf.energy:+.6f} Ry  "
        f"ΔE: {scf.delta_energy:.6f} Ry  [{conv_color}]{'✓ Converged' if scf.converged else scf.convergence_type.title()}[/{conv_color}]"
    )

    resource_table = Table(show_header=False, box=None, padding=(0, 1))
    resource_table.add_column(style="dim")
    resource_table.add_column()
    resource_table.add_column(style="dim")
    resource_table.add_column()

    cpu_bar = "█" * int(res.cpu_pct // 5) + "░" * (20 - int(res.cpu_pct // 5))
    mem_bar = "█" * int(res.mem_pct // 5) + "░" * (20 - int(res.mem_pct // 5))
    resource_table.add_row(
        f"CPU: {res.cpu_pct:.0f}% ({job.cores} cores)", f"[cyan]{cpu_bar}[/cyan]",
        f"Mem: {res.mem_pct:.0f}% ({res.mem_used_gb:.0f}/{res.mem_total_gb:.0f} GB)", f"[magenta]{mem_bar}[/magenta]",
    )

    content = Group(
        f"[bold]{job_title}[/bold]",
        status_line,
        "",
        scf_line,
        energy_line,
        "",
        "[bold]Resources[/bold]",
        resource_table,
    )

    if state.events:
        events_text = "\n".join(state.events[-5:])
        content.renderables.append("")
        content.renderables.append("[bold]Recent Events[/bold]")
        content.renderables.append(events_text)

    return Panel(content, border_style="blue", padding=(1, 2))


def watch_job(
    job_name: str,
    output_path: Optional[str] = None,
    interval: float = 2.0,
    scheduler: Optional[str] = None,
) -> MonitorState:
    """Watch a specific job with live terminal display.

    Args:
        job_name: Job name or case name to monitor
        output_path: Path to .scf output file (auto-detected if None)
        interval: Polling interval in seconds
        scheduler: Force scheduler type (slurm, pbs, lsf) or auto-detect

    Returns:
        Final MonitorState when monitoring ends
    """
    console = Console()
    state = MonitorState(job=JobInfo(name=job_name, job_id="", status="RUNNING", output_file=""))

    output_file = Path(output_path) if output_path else Path(f"{job_name}.scf")
    if not output_file.exists():
        alternatives = sorted(Path(".").glob("*.scf*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if alternatives:
            output_file = alternatives[0]

    state.job.output_file = str(output_file)

    scheduler = scheduler or _detect_scheduler()

    with Live(_build_monitor_panel(state), console=console, refresh_per_second=4, screen=False) as live:
        while state.running:
            active_jobs = detect_active_jobs(scheduler)
            for j in active_jobs:
                if j.name == job_name or job_name in j.name:
                    state.job = j
                    break

            scf = _parse_scf_output(output_file)
            state.scf = scf

            state.events = _parse_scf_events(output_file)

            if scheduler == "slurm" and state.job.job_id:
                try:
                    import subprocess
                    subprocess.run(
                        ["sstat", "--format=TresUsageInAve", "--noheader", "-j", state.job.job_id],
                        capture_output=True, text=True, timeout=5,
                    )
                    # Parse TRES for CPU/memory usage (simplified)
                except Exception:
                    logger.debug("Suppressed exception", exc_info=True)

            live.update(_build_monitor_panel(state))

            if scf.converged:
                state.running = False
                break

            time.sleep(interval)

    return state


def list_active_jobs(scheduler: Optional[str] = None) -> Table:
    """Return a Rich Table of active WIEN2k jobs."""
    table = Table(title="Active WIEN2k Jobs", border_style="blue")
    table.add_column("Job ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Elapsed")
    table.add_column("Time Limit")
    table.add_column("Nodes", justify="right")
    table.add_column("Cores", justify="right")

    jobs = detect_active_jobs(scheduler)
    for j in jobs:
        status_color = {"RUNNING": "green", "PENDING": "yellow", "COMPLETING": "cyan"}.get(j.status, "white")
        table.add_row(
            j.job_id, j.name, f"[{status_color}]{j.status}[/{status_color}]",
            j.elapsed, j.time_limit, str(j.nodes), str(j.cores),
        )

    if not jobs:
        table.add_row("-", "No WIEN2k jobs found", "-", "-", "-", "-", "-")

    return table


def launch_monitor(job_name: Optional[str] = None, interval: float = 2.0) -> dict[str, Any]:
    """Entry point for the `forge monitor` CLI command.

    Args:
        job_name: Specific job to watch, or None to list all active jobs
        interval: Polling interval in seconds

    Returns:
        Status dictionary
    """
    if job_name is None:
        console = Console()
        table = list_active_jobs()
        console.print(table)
        return {"status": "listed", "jobs": len(detect_active_jobs())}

    state = watch_job(job_name, interval=interval)
    return {
        "status": "completed",
        "job": state.job.name,
        "converged": state.scf.converged,
        "final_energy": state.scf.energy,
        "cycles": state.scf.cycle,
    }
