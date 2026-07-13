"""
Command-Line Interface (CLI) Entry Point for FORGE.
Provides a production-grade, multi-command terminal interface for HPC/DFT workflow orchestration.
Supports configuration generation, job submission, benchmark execution, system diagnostics,
and TUI launch with structured logging, JSON output, and rigorous error handling.

Key Architecture Features:
• Subcommand-based routing (generate, submit, benchmark, diagnostics, analyze, tui)
• Global configuration & logging initialization before command dispatch
• Rich UI integration: Tables, Panels, and Progress Bars for human-readable output
• Structured exception handling with machine-readable JSON fallback & graceful degradation
• Multi-scheduler support (SLURM, PBS, LSF) with auto-detection
• Terminal-aware console detection (--plain / --no-color for dumb terminals)
• Thread-safe execution context, signal-aware teardown, and non-blocking I/O
• Comprehensive English documentation, type hints, and HPC-grade resilience patterns
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, cast

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backend_manager import BackendManager
from .benchmark.real import RealBenchmarkRunner
from .config import AppConfig, ensure_dirs, load_config
from .core.hardware import get_physical_cores
from .core.pipeline import run_pipeline
from .core.scheduler import _detect_scheduler, auto_detect_memory
from .core.scheduler import detect as detect_topology
from .core.terminal_monitor import launch_monitor, list_active_jobs
from .exceptions import (
    FORGEError,
    format_error_for_ui,
    log_exception_structured,
)
from .logging_config import get_logger, set_context, setup_logging
from .submit import SUBMIT_PROVIDERS
from .submit.slurm import SlurmDirectives, SlurmJobSpec, submit_slurm_job
from .types import BackendCode, ExecutionMode, OptimizationTarget, PipelineResult
from .ui.analysis import generate_report, parse_scf_log
from .ui.rich_ui import detect_terminal_capabilities, get_plain_console, get_rich_console
from .utils.diagnostic import export_diagnostics_json, run_diagnostics

logger = get_logger(__name__)
console = Console()

_term = os.environ.get("TERM", "")
_no_color = os.environ.get("NO_COLOR", "")
_is_dumb = _term in ("dumb", "vt100", "") or _no_color


# =============================================================================
# Argument Parser Construction
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with subcommands and global flags."""
    parser = argparse.ArgumentParser(
        prog="forge",
        description="Production-grade WIEN2k parallel configuration & HPC job dispatcher",
        epilog=(
            "Examples:\n"
            "  forge generate --backend wien2k --cores 64 --target memory --dry-run\n"
            "  forge submit --partition gpu --time 48:00:00 --mem 64G --scheduler slurm\n"
            "  forge diagnostics --export report.json --full\n"
            "  forge tui\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    global_group = parser.add_argument_group("Global Options")
    global_group.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    global_group.add_argument("-q", "--quiet", action="store_true", help="Suppress console output")
    global_group.add_argument("--json", action="store_true", dest="json_output", help="Output results in JSON format")
    global_group.add_argument("--config", type=str, default=None, help="Path to custom config file")
    global_group.add_argument("--backend", type=str, choices=[b.value for b in BackendCode], help="Override auto-detected backend")
    global_group.add_argument("--log-file", type=str, default=None, help="Redirect logs to file")
    global_group.add_argument("--version", action="version", version="forge v0.1.0")
    global_group.add_argument("--plain", action="store_true", help="Use plain output (no Rich formatting, for dumb terminals)")
    global_group.add_argument("--no-color", action="store_true", dest="no_color", help="Disable colored output")

    subparsers = parser.add_subparsers(dest="command", help="Available workflow commands", required=True)

    gen_p = subparsers.add_parser("generate", help="Generate parallel configuration files")
    gen_p.add_argument("--nodes", type=int, default=None, help="Number of compute nodes")
    gen_p.add_argument("--cores", type=int, default=None, help="Total MPI cores to allocate")
    gen_p.add_argument("--omp", type=int, default=None, help="OpenMP threads per rank")
    gen_p.add_argument("--mode", type=str, choices=[m.value for m in ExecutionMode], help="Parallel execution mode")
    gen_p.add_argument("--target", type=str, choices=[t.value for t in OptimizationTarget], default="time", help="Optimization target (time, memory, balanced, cost)")
    gen_p.add_argument("--max-cores", type=int, default=None, help="Hard limit on total cores to utilize")
    gen_p.add_argument("--reserve-os-cores", type=int, default=None, metavar="N", help="Reserve N cores for OS/daemons (e.g., 4 leaves 124 of 128)")
    gen_p.add_argument("--memory-limit", type=float, default=None, help="Hard limit on memory per node (GB)")
    gen_p.add_argument("--dry-run", action="store_true", help="Generate config without writing to disk")
    gen_p.add_argument("--export", type=str, default=None, help="Export configuration summary to path")
    gen_p.add_argument("--overwrite", action="store_true", help="Overwrite existing .machines/INCAR without prompt")
    gen_p.add_argument("--scheduler", "-S", type=str, choices=["slurm", "pbs", "lsf", "sge", "auto"], default="auto", help="Target scheduler for job scripts (default: auto-detect)")
    gen_p.add_argument("--gpu", action="store_true", help="Enable GPU-aware configuration")
    gen_p.add_argument("--gpu-mixed-precision", action="store_true", help="Enable FP32/FP16 mixed precision")
    gen_p.add_argument("--manual", action="store_true", help="After auto-generation, open .machines in $EDITOR for manual review/edit")

    sub_p = subparsers.add_parser("submit", help="Submit job to scheduler")
    sub_p.add_argument("--scheduler", "-S", type=str, choices=["slurm", "pbs", "lsf", "sge", "auto"], default="auto", help="Target scheduler (default: auto-detect)")
    sub_p.add_argument("--partition", type=str, default="", help="Scheduler partition/queue")
    sub_p.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    sub_p.add_argument("--ntasks", type=int, default=0, help="Total tasks (0 = auto from topology)")
    sub_p.add_argument("--time", type=str, default="24:00:00", help="Walltime (HH:MM:SS)")
    sub_p.add_argument("--mem", type=str, default=auto_detect_memory(), help="Memory per node")
    sub_p.add_argument("--job-name", type=str, default="wien2k_job", help="Job identifier")
    sub_p.add_argument("--dependency", type=str, default="", help="Job dependency (e.g., afterok:123)")
    sub_p.add_argument("--dry-run", action="store_true", help="Generate script only, do not submit")
    sub_p.add_argument("--export", type=str, default=None, help="Export script to path")

    bench_p = subparsers.add_parser("benchmark", help="Run empirical or synthetic benchmarks")
    bench_p.add_argument("--type", choices=["real", "synthetic"], default="real", help="Benchmark type")
    bench_p.add_argument("--max-cores", type=int, default=None, help="Maximum cores for scaling suite")
    bench_p.add_argument("--walltime", type=str, default="02:00:00", help="Max runtime per run")
    bench_p.add_argument("--output", type=str, default=None, help="Save results to JSON path")
    bench_p.add_argument("--skip-cleanup", action="store_true", help="Retain temporary benchmark directories")
    bench_p.add_argument("--scheduler", "-S", type=str, choices=["slurm", "pbs", "lsf", "auto"], default="auto", help="Target scheduler (default: auto-detect)")

    diag_p = subparsers.add_parser("diagnostics", help="Collect system & environment health report")
    diag_p.add_argument("--export", type=str, default=None, help="Save diagnostic report to path")
    diag_p.add_argument("--full", action="store_true", help="Include verbose library & interconnect checks")

    hw_p = subparsers.add_parser("hardware", help="Show hardware info and parallelization recommendations")
    hw_p.add_argument("--recommend", "-r", action="store_true", help="Show NUMA/hybrid/IO optimization advice")
    hw_p.add_argument("--case", type=str, default=None, help="Case name for problem-specific recommendations")

    ana_p = subparsers.add_parser("analyze", help="Parse SCF logs & generate performance reports")
    ana_p.add_argument("--log", type=str, required=True, help="Path to SCF/output log file")
    ana_p.add_argument("--code", type=str, choices=["wien2k", "vasp", "qe"], default=None, help="Force DFT code parser")
    ana_p.add_argument("--export", type=str, default=None, help="Export analysis report to JSON")

    tui_p = subparsers.add_parser("tui", help="Launch interactive terminal UI")
    tui_p.add_argument("--compact", action="store_true", help="Enable compact UI layout")

    mon_p = subparsers.add_parser("monitor", help="Monitor SCF convergence in real-time")
    mon_p.add_argument("case", type=str, nargs="?", default=None, help="Case name to monitor (omit to list active jobs)")
    mon_p.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds (default: 2)")
    mon_p.add_argument("--output", type=str, default=None, help="SCF output file to parse")

    run_p = subparsers.add_parser("run", help="Execute a WIEN2k workflow from YAML")
    run_p.add_argument("workflow_file", type=str, help="Path to workflow.yaml")
    run_p.add_argument("--auto-retry", action="store_true", default=True, help="Auto-retry on convergence failure")
    run_p.add_argument("--no-retry", action="store_true", help="Disable auto-retry")
    run_p.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts (default: 3)")
    run_p.add_argument("--poll", type=float, default=5.0, help="Job polling interval in seconds (default: 5)")

    wf_p = subparsers.add_parser("workflow", help="Generate workflow YAML templates")
    wf_p.add_argument("action", type=str, choices=["create", "list", "visualize"], help="Workflow action")
    wf_p.add_argument("--case", type=str, default="case", help="Case name")
    wf_p.add_argument("--steps", type=str, default="scf,dos,band", help="Comma-separated workflow steps")
    wf_p.add_argument("--output", type=str, default=None, help="Output YAML path")

    diag_scf = subparsers.add_parser("diagnose", help="Diagnose SCF convergence issues")
    diag_scf.add_argument("case", type=str, nargs="?", default=None, help="Case name to diagnose")
    diag_scf.add_argument("--log", type=str, default=None, help="Path to .scf or .output file")

    opt_p = subparsers.add_parser("optimize", help="Bayesian auto-tuning of RKMAX, k-points, mixing")
    opt_p.add_argument("--case", type=str, default="case", help="WIEN2k case name")
    opt_p.add_argument("--budget", type=int, default=10, help="Max DFT runs (default: 10)")
    opt_p.add_argument("--target", type=str, default="energy_convergence", help="Target metric")
    opt_p.add_argument("--simulated", action="store_true", help="Use simulated objective for testing")
    opt_p.add_argument("--verbose", "-v", action="store_true", help="Show iteration details")

    screen_p = subparsers.add_parser("screen", help="High-throughput screening via Materials Project")
    screen_p.add_argument("--formula", type=str, default=None, help="Chemical formula (e.g. ABO3, LiFePO4)")
    screen_p.add_argument("--elements", type=str, default=None, help="Comma-separated elements (e.g. Ti,O,Zr)")
    screen_p.add_argument("--mp-id", type=str, default=None, help="Single Materials Project ID")
    screen_p.add_argument("--max", type=int, default=50, help="Max materials (default: 50)")
    screen_p.add_argument("--api-key", type=str, default=None, help="Materials Project API key")
    screen_p.add_argument("--output", type=str, default=None, help="Output directory")

    pred_p = subparsers.add_parser("predict", help="Predict SCF convergence time before calculation")
    pred_p.add_argument("--case", type=str, default="case", help="WIEN2k case name")
    pred_p.add_argument("--struct", type=str, default=None, help="Path to .struct file")
    pred_p.add_argument("--no-history", action="store_true", help="Skip ML training from history")

    conv_p = subparsers.add_parser("converge", help="Run automated convergence tests")
    conv_p.add_argument("--case", type=str, required=True, help="Case name")
    conv_p.add_argument("--mode", type=str, choices=["kpoints", "rkmax", "both"], default="both", help="Parameter to converge (default: both)")
    conv_p.add_argument("--tolerance", type=float, default=0.001, help="Energy tolerance in Ry (default: 0.001)")
    conv_p.add_argument("--kpoints", type=str, default="2,2,2 4,4,4 6,6,6 8,8,8 10,10,10", help="Space-separated k-point grids (default: '2,2,2 4,4,4 6,6,6 8,8,8 10,10,10')")
    conv_p.add_argument("--rkmax", type=str, default="5,6,7,8,9,10", help="Comma-separated RKmax values (default: '5,6,7,8,9,10')")

    advise_p = subparsers.add_parser("advise", help="Get intelligent optimization advice (Roofline, Amdahl, NUMA)")
    advise_p.add_argument("--case", type=str, default="case", help="WIEN2k case name for problem-aware advice")
    advise_p.add_argument("--nmat", type=int, default=None, help="Override matrix size (detected from case if not given)")
    advise_p.add_argument("--kpoints", type=int, default=None, help="Override k-point count")
    advise_p.add_argument("--cores", type=int, default=None, help="Target total cores")
    advise_p.add_argument("--target", type=str, choices=["time", "energy", "cost", "balanced"], default="time", help="Optimization goal (default: time)")
    advise_p.add_argument("--plain", action="store_true", help="Show advice in simple language (non-expert mode)")
    advise_p.add_argument("--json", action="store_true", help="Export advice as JSON")

    hist_p = subparsers.add_parser("history", help="Query execution history database")
    hist_p.add_argument("--list", action="store_true", help="List past runs")
    hist_p.add_argument("--show", type=str, default=None, help="Show details of run ID")
    hist_p.add_argument("--similar-to", type=str, default=None, help="Find similar past cases by case path")
    hist_p.add_argument("--limit", type=int, default=10, help="Max results to display")

    bands_p = subparsers.add_parser("analyze-bands", help="Extract band structure and DOS data")
    bands_p.add_argument("--case", type=str, required=True, help="Case name")
    bands_p.add_argument("--output", type=str, default=None, help="Output file for band data (JSON)")
    bands_p.add_argument("--dos", action="store_true", help="Also parse DOS data")

    return parser


def _resolve_scheduler(flag: str) -> str:
    """Resolve --scheduler flag or auto-detect."""
    if flag and flag != "auto":
        return flag
    return _detect_scheduler()


# =============================================================================
# Command Handlers
# =============================================================================

def _open_editor_for_manual_review(filepath: Path) -> None:
    """Open a file in $EDITOR (or nano/vi fallback) for manual review."""
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", ""))
    if not editor:
        for fallback in ("nano", "vim", "vi"):
            if shutil.which(fallback):
                editor = fallback
                break
    if not editor:
        console.print("[yellow]No editor found ($EDITOR unset, nano/vim not in PATH). "
                       f"Edit manually: {filepath}[/yellow]")
        return

    console.print(f"[bold cyan]Opening {filepath.name} in {editor} for manual review...[/bold cyan]")
    console.print("[dim](Save and exit to continue, or :q! to discard)[/dim]")
    try:
        subprocess.run([editor, str(filepath)], check=False)
        console.print(f"[green]✓ Editor closed. Final config: {filepath}[/green]")
    except FileNotFoundError:
        console.print(f"[yellow]Editor '{editor}' not found. Edit manually: {filepath}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Editor error: {e}. Edit manually: {filepath}[/yellow]")


def _get_exec_command() -> str:
    """Auto-detect the correct WIEN2k execution command from input files."""
    try:
        from .backend_manager import get_current_backend
        backend = get_current_backend()
        params = backend.detect_problem_size()
        return params.get("exec_command", "run_lapw -p")  # type: ignore[return-value]
    except Exception:
        return "run_lapw -p"


def _handle_generate(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:  # noqa: C901
    """Execute pipeline configuration generation with Rich UI output."""
    max_cores = args.max_cores
    if args.reserve_os_cores is not None:
        phys = get_physical_cores()
        reserved_max = max(1, phys - args.reserve_os_cores)
        max_cores = min(max_cores, reserved_max) if max_cores else reserved_max
        console.print(f"[dim]Reserving {args.reserve_os_cores} OS cores → using {max_cores} of {phys}[/dim]")
    topo = detect_topology(max_cores=max_cores)
    
    suggestion: dict[str, Any] = {}
    if args.mode: 
        suggestion["mode"] = ExecutionMode(args.mode)
    if args.cores: 
        suggestion["recommended_total_cores"] = args.cores
    if args.omp: 
        suggestion["omp_threads_per_rank"] = args.omp
    if args.memory_limit:
        suggestion["memory_limit_gb"] = args.memory_limit

    if args.gpu or args.gpu_mixed_precision:
        try:
            from .backends.gpu_backend import (
                detect_gpu,
                get_gpu_recommendation,
                get_mixed_precision_recommendation,
            )
            gpus = detect_gpu()
            if gpus:
                gpu_rec = get_gpu_recommendation(topo, nmat=5000, nkpt=8, mode=suggestion.get("mode", "mpi"))
                suggestion["gpu_recommendation"] = gpu_rec
                if args.gpu_mixed_precision:
                    fp_rec = get_mixed_precision_recommendation("wien2k", nmat=5000)
                    suggestion["mixed_precision"] = fp_rec
                console.print(f"[green]GPU detection: {len(gpus)} device(s) found.[/green]")
            else:
                console.print("[yellow]No GPUs detected. Proceeding without GPU configuration.[/yellow]")
        except ImportError as e:
            console.print(f"[yellow]GPU backend not available: {e}[/yellow]")

    result: PipelineResult = run_pipeline(
        topo=topo,
        user_suggestion=suggestion,
        dry_run=args.dry_run,
        export_path=args.export
    )

    if args.json_output:
        return {"status": "success" if result.success else "failed", "data": result.to_dict()}

    if result.success:
        if args.dry_run and result.dry_run_content:
            table = Table(title="Configuration Preview", border_style="cyan")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")

            table.add_row("Mode", str(suggestion.get("mode", "auto")))
            table.add_row("Total Cores", str(suggestion.get("recommended_total_cores", topo.total_cores)))

            max_eff = suggestion.get("max_efficient_cores")
            if max_eff:
                table.add_row("Max Efficient Cores", f"[yellow]{max_eff}[/yellow]")

            sat_data: dict[str, Any] = suggestion.get("saturation_data", {})
            if sat_data:
                eff = sat_data.get("efficiency_pct")
                if eff is not None:
                    table.add_row("Est. Efficiency", f"[dim]{eff:.0f}%[/dim]")
                sf = sat_data.get("serial_fraction")
                if sf is not None:
                    table.add_row("Serial Fraction (Amdahl)", f"[dim]s={sf:.3f}[/dim]")

            table.add_row("Dry-Run Content", f"[dim]{len(result.dry_run_content)} bytes generated[/dim]")

            console.print(table)
            console.print(Panel(result.dry_run_content, title="Generated Config", border_style="dim"))
        else:
            console.print(Panel(f"[green]✓ Configuration generated successfully.[/]\nPath: [cyan]{result.config_path}[/]", border_style="green"))

            if getattr(args, "manual", False) and result.config_path:
                _open_editor_for_manual_review(Path(result.config_path))
            
        if result.warnings:
            warn_table = Table(title="Warnings", show_header=False, box=None)
            for w in result.warnings:
                warn_table.add_row("[yellow]•[/]", f"[dim]{w}[/dim]")
            console.print(warn_table)
    else:
        console.print(Panel(f"[red]✗ Generation failed: {result.validation_errors}[/]", border_style="red"))
        
    return {"success": result.success, "path": result.config_path, "warnings": result.warnings}


def _handle_submit(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Execute job submission to scheduler with multi-scheduler support."""
    scheduler = _resolve_scheduler(getattr(args, "scheduler", "auto"))
    topo = detect_topology(max_cores=args.ntasks or None)

    if scheduler == "slurm":
        directives = SlurmDirectives(
            job_name=args.job_name,
            partition=args.partition,
            nodes=args.nodes,
            ntasks=args.ntasks or topo.total_cores,
            cpus_per_task=1,
            mem_per_node=args.mem,
            time=args.time,
            dependency=args.dependency or None
        )
        spec = SlurmJobSpec(
            topo=topo,
            exec_command=_get_exec_command(),
            directives=directives,
            working_dir=Path.cwd()
        )
        res = submit_slurm_job(spec=spec, dry_run=args.dry_run, script_path=Path(args.export) if args.export else None)

        if args.json_output:
            return {"success": res.get("success"), "job_id": res.get("job_id"), "script_path": str(res.get("script_path", ""))}
            
        if res.get("success"):
            if args.dry_run:
                console.print(Panel(res.get("dry_run_content") or "Script content not available", title="SBATCH Preview", border_style="cyan"))
            else:
                console.print(Panel(f"[green]✓ Job submitted successfully.[/]\nJob ID: [bold cyan]{res.get('job_id')}[/]\nScript: [dim]{res.get('script_path')}[/]", border_style="green"))
        else:
            console.print(Panel(f"[red]✗ Submission failed: {res.get('errors')}[/]", border_style="red"))
        return {"success": res.get("success"), "job_id": res.get("job_id"), "path": str(res.get("script_path", ""))}

    elif scheduler in ("pbs", "lsf"):
        provider_cls = SUBMIT_PROVIDERS.get(scheduler)
        if provider_cls:
            provider = provider_cls()
            pbs_res = provider.submit(
                topo=topo,
                exec_command="run_lapw -p",
                directives={
                    "job_name": args.job_name,
                    "queue": args.partition,
                    "nodes": args.nodes,
                    "walltime": args.time,
                    "mem" if scheduler == "pbs" else "memory": args.mem,
                },
                script_path=Path(args.export) if args.export else None,
                dry_run=args.dry_run,
            )
            if args.json_output:
                return {"success": pbs_res.get("success"), "job_id": pbs_res.get("job_id"), "script_path": str(pbs_res.get("script_path", ""))}
            if pbs_res.get("success"):
                console.print(Panel(f"[green]✓ Job submitted successfully.[/]\nJob ID: [bold cyan]{pbs_res.get('job_id')}[/]\nScript: [dim]{pbs_res.get('script_path')}[/]", border_style="green"))
            else:
                console.print(Panel(f"[red]✗ Submission failed: {pbs_res.get('errors')}[/]", border_style="red"))
            return {"success": pbs_res.get("success"), "job_id": pbs_res.get("job_id"), "path": str(pbs_res.get("script_path", ""))}
        else:
            return {"success": False, "errors": [f"Scheduler provider '{scheduler}' not available."]}

    return {"success": False, "errors": [f"Unknown scheduler: {scheduler}"]}


def _handle_benchmark(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Run empirical or synthetic benchmark suite."""
    topo = detect_topology(max_cores=args.max_cores)
    
    if args.type == "synthetic":
        from .benchmark.synthetic import SyntheticWorkloadParams, generate_strong_scaling_suite
        problem: SyntheticWorkloadParams = {"atoms": 20, "kpoints": 8, "nmat": 1500, "is_hybrid": False}
        suite = generate_strong_scaling_suite(problem, topo, max_cores=args.max_cores)
        console.print(Panel(f"[green]✓ Synthetic benchmark complete. {len(suite)} runs generated.[/]", border_style="green"))
        return {"type": "synthetic", "runs": len(suite), "data": [dict(s) for s in suite]}
    else:
        scheduler = _resolve_scheduler(getattr(args, "scheduler", "auto"))
        runner = RealBenchmarkRunner({
            "backend": cfg.backend,
            "scheduler": scheduler,
            "walltime": args.walltime,
            "cleanup_after": not args.skip_cleanup
        })
        result = runner.run(topo)
        if args.json_output:
            return dict(result)
            
        status_color = "green" if result["status"] == "success" else "red"
        console.print(Panel(
            f"[{status_color}]Benchmark {result['status']}[/]\n"
            f"Run ID: [cyan]{result['run_id']}[/]\n"
            f"Wall Time: [bold]{result['wall_time_sec']:.2f}s[/]",
            border_style=status_color
        ))
        return {"run_id": result["run_id"], "status": result["status"], "wall_time": result["wall_time_sec"]}


def _handle_hardware(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Show hardware info with optional NUMA/hybrid/IO optimization recommendations."""
    from .core.hardware import (
        get_cpu_architecture,
        get_cpu_generation,
        get_numa_node_count,
        get_physical_cores,
        get_scratch_filesystem_type,
        get_system_type,
        get_total_mem_kb,
    )
    from .core.topology import Topology
    from .optimizer.parallel import (
        recommend_gmax,
        recommend_io_strategy,
        recommend_lapw0_strategy,
        recommend_mkl_threading,
        recommend_numa_strategy,
        recommend_rkmax,
    )

    cores = get_physical_cores()
    arch = get_cpu_architecture()
    generation = get_cpu_generation()
    sys_type = get_system_type()
    ram_gb = get_total_mem_kb() / (1024 * 1024)
    numa = get_numa_node_count()
    scratch = get_scratch_filesystem_type()

    table = Table(title="Hardware Profile", border_style="blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("CPU", f"{generation} ({arch})")
    table.add_row("Cores", str(cores))
    table.add_row("NUMA Nodes", str(numa))
    table.add_row("RAM", f"{ram_gb:.1f} GB")
    table.add_row("System Type", sys_type)
    table.add_row("Scratch FS", scratch)
    console.print(table)

    if args.recommend:
        nmat = 2000
        nkpt = 8
        atoms = 10
        if args.case:
            try:
                from .core.case_parser import CaseFileParser
                parser = CaseFileParser(args.case)
                data = parser.parse_all()
                nmat = data.nmat or 2000
                atoms = data.atoms or 10
                nkpt = data.kpoints or 8
            except Exception:
                console.print("[dim]Case parsing failed — using defaults[/dim]")

        rec_table = Table(title="Parallelization Recommendations", border_style="green")
        rec_table.add_column("Strategy", style="bold cyan")
        rec_table.add_column("Details", style="green")

        numa_rec = recommend_numa_strategy(
            Topology(nodes=["n1"], cores_per_node=[cores]), nmat, nkpt, atoms,
        )
        rec_table.add_row("NUMA-Aware", numa_rec.recommendation)

        lapw0_rec = recommend_lapw0_strategy(
            Topology(nodes=["n1"], cores_per_node=[cores]), nmat,
        )
        rec_table.add_row("LAPW0 (Hybrid)", lapw0_rec.recommendation)

        io_rec = recommend_io_strategy(nmat, nkpt, atoms, scratch)
        rec_table.add_row("I/O", str(io_rec.get("recommendation", "-")))

        rkmax_rec = recommend_rkmax([26], "scf")
        gmax_rec = recommend_gmax(rkmax_rec, "scf")
        rec_table.add_row("RKMAX/GMAX", f"RKMAX={rkmax_rec}, GMAX={gmax_rec} (SCF)")

        mkl_threads = recommend_mkl_threading(nmat, nkpt)
        rec_table.add_row("MKL Threads", str(mkl_threads) if mkl_threads else "use default")

        console.print(rec_table)
        return {"status": "displayed", "nmat": nmat, "nkpt": nkpt, "cores": cores}

    return {"cores": cores, "arch": arch, "generation": generation}


def _handle_diagnostics(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Collect & export system diagnostics."""
    report = run_diagnostics()
    if args.export:
        export_diagnostics_json(report, args.export)
        
    if args.json_output:
        return dict(report)
        
    table = Table(title="System Diagnostics Summary", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Hostname", report.get("hostname", "unknown"))
    table.add_row("OS", f"{report.get('os_info', {}).get('os', 'unknown')} {report.get('os_info', {}).get('release', '')}")
    table.add_row("Python", report.get('python_env', {}).get('python_version', 'unknown'))
    table.add_row("Warnings", str(len(report.get("warnings", []))), style="yellow" if report.get("warnings") else "green")
    table.add_row("Critical Errors", str(len(report.get("critical_errors", []))), style="red" if report.get("critical_errors") else "green")
    
    console.print(table)
    return {"hostname": report.get("hostname"), "warnings": len(report.get("warnings", [])), "errors": len(report.get("critical_errors", []))}


def _handle_analyze(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Parse SCF log & generate analysis report."""
    parsed = parse_scf_log(args.log, code_hint=args.code)
    report = generate_report(parsed, scaling_data=None, include_recommendations=True)
    
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
            
    if args.json_output:
        return report.to_dict()
        
    console.print(Panel(
        f"[green]✓ Analysis complete.[/]\n"
        f"Code: [cyan]{parsed.get('code', 'unknown')}[/] | "
        f"Converged: [{'green' if parsed.get('converged') else 'red'}]{parsed.get('converged')}[/]\n"
        f"Total Cycles: [bold]{parsed.get('total_cycles', 0)}[/]",
        border_style="green"
    ))
    return {"code": parsed.get("code"), "converged": parsed.get("converged"), "cycles": parsed.get("total_cycles")}


def _handle_tui(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Interactive terminal UI removed. Use `forge monitor` for live SCF display."""
    console.print("[yellow]TUI has been removed. Use `forge monitor [case]` for live SCF monitoring.[/yellow]")
    return {"status": "tui_removed"}


def _handle_monitor(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Monitor SCF convergence with live Rich terminal display."""
    if args.json_output:
        return {"status": "monitor_skipped_in_json_mode"}

    if args.case:
        return launch_monitor(job_name=args.case, interval=args.interval)
    else:
        console.print(list_active_jobs())
        return {"status": "listed"}


def _handle_run(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Execute a WIEN2k workflow from a YAML file."""
    try:
        from .core.workflow_executor import run_workflow_from_yaml
    except ImportError as e:
        return {"error": f"Workflow executor not available: {e}"}

    auto_retry = not args.no_retry
    console.print(Panel(
        f"[cyan]Executing Workflow[/]\n"
        f"File: [bold]{args.workflow_file}[/]\n"
        f"Auto-retry: [{'green' if auto_retry else 'red'}]{auto_retry}[/]\n"
        f"Max retries: {args.max_retries}",
        border_style="blue",
    ))

    status = run_workflow_from_yaml(
        args.workflow_file,
        auto_retry=auto_retry,
        max_retries=args.max_retries,
        poll_interval=args.poll,
    )

    if status.state.value == "completed":
        console.print(f"[green]Workflow completed in {status.elapsed_total:.1f}s[/green]")
    else:
        console.print(f"[red]Workflow {status.state.value}: {len([e for e in status.events if 'Error' in e or 'Failed' in e])} errors[/red]")

    return {"status": status.state.value, "elapsed": status.elapsed_total, "events": status.events}


def _handle_workflow(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Generate workflow YAML templates."""
    steps = [s.strip() for s in args.steps.split(",")]

    if args.action == "list":
        tasks = []
        prev = None
        for i, step in enumerate(steps):
            node_id = f"{args.case}_{step}"
            tasks.append((step, str(i + 1), prev if prev else "-"))
            prev = node_id

        table = Table(title=f"Workflow: {args.case}", border_style="cyan")
        table.add_column("Step", style="bold")
        table.add_column("Order")
        table.add_column("Depends On")
        for task in tasks:
            table.add_row(*task)
        console.print(table)
        return {"status": "listed", "steps": steps}

    yaml_content = f"# WIEN2k workflow generated by forge\ncase: {args.case}\nsteps: [{', '.join(steps)}]\nscheduler: auto\nauto_retry: true\nmax_retries: 3\npoll_interval: 5.0\n"
    output = args.output or f"{args.case}_workflow.yaml"
    with open(output, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    console.print(f"[green]Workflow template saved to {output}[/green]")
    console.print(f"  Steps: {', '.join(steps)}")
    return {"status": "created", "output": output, "steps": steps}


def _handle_diagnose(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:  # noqa: C901
    """SCF convergence diagnostics with root cause analysis.

    Connects backend intelligence (charge sloshing diagnosis, Bayesian tuning,
    QTL-B root cause) to the terminal UI.
    """
    import re as _re

    scf_path = None
    if args.log:
        scf_path = Path(args.log)
    elif args.case:
        scf_path = Path(f"{args.case}.scf")
        if not scf_path.exists():
            scf_path = Path.cwd() / f"{args.case}.scf"
    else:
        matches = sorted(Path(".").glob("*.scf*"), key=lambda p: p.stat().st_mtime, reverse=True)
        scf_path = matches[0] if matches else None

    if not scf_path or not scf_path.exists():
        console.print("[red]No SCF output file found.[/red]")
        return {"error": "No SCF file found"}

    content = scf_path.read_text(encoding="utf-8", errors="replace")

    energy_matches = _re.findall(r":ENE\s*:\s*.*?(-?\d+\.\d+)", content)
    charge_matches = _re.findall(r":DIS\s*:\s*.*?(\d+\.\d+)", content)

    energies = [float(e) for e in energy_matches]
    charges = [float(c) for c in charge_matches]
    converged = "charge convergence" in content.lower() or "energy convergence" in content.lower()

    table = Table(title=f"SCF Diagnostics: {scf_path.name}", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Cycles completed", str(len(energies)))
    table.add_row("Final energy", f"{energies[-1]:.6f} Ry" if energies else "N/A")
    table.add_row("Final charge distance", f"{charges[-1]:.6f}" if charges else "N/A")
    table.add_row("Converged", f"[{'green' if converged else 'red'}]{converged}[/]")

    if len(charges) >= 3:
        ratios = []
        for i in range(1, len(charges)):
            if charges[i - 1] > 1e-12:
                ratios.append(charges[i] / charges[i - 1])
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        table.add_row("Avg charge ratio", f"{avg_ratio:.3f}")
        if avg_ratio > 1.01:
            table.add_row("Diagnosis", "[red]Divergent — reduce mixing beta or enable PRATT[/red]")
        elif avg_ratio < 0.95:
            table.add_row("Diagnosis", "[green]Converging monotonically[/green]")
        else:
            table.add_row("Diagnosis", "[yellow]Oscillatory — reduce mixing[/yellow]")

    if len(energies) >= 2:
        deltas = [abs(energies[i] - energies[i - 1]) for i in range(1, len(energies))]
        sign_changes = sum(1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0)
        oscillation_pct = sign_changes / max(1, len(deltas) - 1) * 100 if len(deltas) > 1 else 0
        table.add_row("Energy oscillations", f"{oscillation_pct:.1f}%")

    console.print(table)

    # ── Charge Sloshing Root Cause Analysis ──
    try:
        from .optimizer.monitor import diagnose_charge_sloshing_root_cause
        diag = diagnose_charge_sloshing_root_cause(content, case_name=args.case or scf_path.stem)
        if diag["root_cause"] != "none":
            panel_title = "[red bold]Charge Sloshing Detected"
            cause_labels = {
                "metallic": "فلزی — سطح فرمی پیچیده / Metallic — complex Fermi surface",
                "symmetry_breaking": "شکست تقارن / Symmetry breaking",
                "core_overlap": "همپوشانی هسته — RMT نامناسب / Core overlap — check RMT",
                "mixing_too_aggressive": "نرخ مخلوط‌سازی بالا / Mixing too aggressive",
            }
            cause_label = cause_labels.get(diag["root_cause"], diag["root_cause"])
            body = f"[bold]Root Cause:[/] {cause_label}\n"
            body += f"[dim]Confidence: {diag['confidence']:.0%}[/]\n\n"
            for i, act in enumerate(diag["actions"], 1):
                body += f"  {i}. [cyan]{act['action']}[/] — {act['reason']}\n"
            console.print(Panel(body.strip(), title=panel_title, border_style="red"))
    except Exception:
        pass

    # ── QTL-B Root Cause Analysis ──
    has_qtlb = any(p in content.lower() for p in ("qtl-b",))
    has_crash = any(p in content.lower() for p in ("lapw crashed", "segmentation fault"))
    has_not_conv = "not converged" in content.lower()

    if has_qtlb:
        qtlb_body = ""
        if "rkmax" in content.lower() or "kmax" in content.lower():
            qtlb_body += "• Reduce RKMAX by 0.5–1.0\n"  # noqa: RUF001
        if "overlap" in content.lower():
            qtlb_body += "• Reduce RMT values or check sphere overlap\n"
        if "linearization" in content.lower() or "ene" in content.lower():
            qtlb_body += "• Add more linearization energies in case.in1\n"
        qtlb_body += "• Increase GMAX to 2.5×RKMAX\n"  # noqa: RUF001
        qtlb_body += "• Check init_lapw —b (non-default linearization)"
        console.print(Panel(qtlb_body, title="[red bold]QTL-B Error — Root Cause Analysis", border_style="red"))

    if has_crash:
        console.print(Panel(
            "• Check MPI stack (mpirun/srun) and memory limits\n"
            "• Look for OOM (Out of Memory) in SLURM output\n"
            "• Verify .machines file format matches WIEN2k version\n"
            "• Try running with fewer MPI ranks first",
            title="[red bold]LAPWx Crash Detected", border_style="red"
        ))

    # ── SCF Divergence Detection ──
    try:
        from .optimizer.convergence import detect_scf_divergence
        divergence = detect_scf_divergence(content, energy_values=energies if energies else None)
        if divergence["divergent"]:
            console.print(Panel(
                f"[bold]Type:[/] {divergence['divergence_type']}\n"
                f"[bold]Severity:[/] {divergence['severity']:.0%}\n\n"
                f"{divergence['recommended_action']}\n\n"
                f"[dim]Auto fix: beta={divergence['auto_mixing_params']['beta']}, "
                f"pratt={divergence['auto_mixing_params']['pratt_cycles']}, "
                f"msr1a={divergence['auto_mixing_params']['msr1a']}[/]",
                title="[red bold]SCF Divergence Analysis", border_style="red"
            ))
    except Exception:
        pass

    if has_not_conv and not converged:
        console.print(Panel(
            "1. Check if charge sloshing (see above)\n"
            "2. Reduce mixing beta to 0.05 and enable PRATT\n"
            "3. Increase NSTEPS in case.in2\n"
            "4. For metals: add TEMP 0.002 in case.in2 (Fermi smearing)\n"
            "5. Run [bold]forge optimize --simulated[/] to auto-tune RKMAX/KPPRA",
            title="[yellow bold]SCF Not Converged — Action Plan", border_style="yellow"
        ))

    if not any([has_qtlb, has_crash, has_not_conv, converged, diag.get("root_cause", "") != "none" if 'diag' in dir() else False]):
        console.print("[green]No critical issues detected. SCF appears healthy.[/green]")

    return {
        "converged": converged,
        "cycles": len(energies),
        "error_detected": has_qtlb or has_crash,
        "final_energy": energies[-1] if energies else 0.0,
    }


def _handle_optimize(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Bayesian auto-tuning of RKMAX, k-points, and mixing parameter."""
    try:
        from .optimizer.bayesian_tuner import optimize_convergence_parameters
    except ImportError as e:
        return {"error": f"Bayesian tuner not available: {e}"}

    console.print(Panel(
        f"[cyan]Bayesian Parameter Optimization[/]\n"
        f"Case: [bold]{args.case}[/] | Budget: [bold]{args.budget} runs[/] | "
        f"Target: [bold]{args.target}[/]",
        border_style="magenta",
    ))

    result = optimize_convergence_parameters(
        case_name=args.case,
        budget=args.budget,
        use_simulated=args.simulated,
        verbose=args.verbose,
    )

    table = Table(title="Optimization Results", border_style="green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Optimal Value", style="green bold")
    table.add_column("Uncertainty", style="dim")
    table.add_row("RKMAX", f"{result.best_rkmax:.1f}", f"±{result.uncertainty_rkmax:.2f}")
    table.add_row("KPPRA", str(result.best_kppra), "-")
    table.add_row("Mixing β", f"{result.best_mixing:.4f}", f"±{result.uncertainty_mixing:.4f}")
    table.add_row("ΔE (best)", f"{result.best_energy:.6f} Ry", "")
    table.add_row("Iterations", str(result.iterations), "")
    table.add_row("Converged", f"[{'green' if result.convergence_achieved else 'red'}]{result.convergence_achieved}[/]", "")
    console.print(table)

    if result.observations:
        hist_table = Table(title="Iteration History", border_style="dim")
        hist_table.add_column("#", style="dim")
        hist_table.add_column("RKMAX")
        hist_table.add_column("KPPRA")
        hist_table.add_column("Mixing")
        hist_table.add_column("ΔE (Ry)")
        for obs in result.observations[-8:]:
            hist_table.add_row(
                str(obs["iteration"]), f"{obs['rkmax']:.1f}",
                str(obs["kppra"]), f"{obs['mixing']:.4f}",
                f"{obs['delta_energy']:.6f}",
            )
        console.print(hist_table)

    return {
        "rkmax": result.best_rkmax,
        "kppra": result.best_kppra,
        "mixing": result.best_mixing,
        "delta_energy": result.best_energy,
        "converged": result.convergence_achieved,
    }


def _handle_screen(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """High-throughput screening via Materials Project API."""
    try:
        from .core.materials_project import screen_materials
    except ImportError as e:
        return {"error": f"Materials Project client not available: {e}"}

    elements_list = args.elements.split(",") if args.elements else None

    console.print(Panel(
        f"[cyan]Materials Project Screening[/]\n"
        f"Query: [bold]{args.formula or args.elements or args.mp_id}[/] | "
        f"Max: [bold]{args.max}[/]",
        border_style="blue",
    ))

    result = screen_materials(
        formula=args.formula,
        elements=elements_list,
        mp_id=args.mp_id,
        max_results=args.max,
        api_key=args.api_key,
        output_dir=args.output,
    )

    table = Table(title="Screening Results", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total found", str(result.total_found))
    table.add_row("Downloaded", str(result.downloaded))
    table.add_row("Converted to .struct", str(result.converted))
    table.add_row("Errors", f"[{'red' if result.errors else 'green'}]{len(result.errors)}[/]")
    console.print(table)

    if result.materials[:10]:
        mat_table = Table(title="Materials (first 10)", border_style="dim")
        mat_table.add_column("MP-ID", style="cyan")
        mat_table.add_column("Formula", style="bold")
        mat_table.add_column("Band Gap (eV)")
        mat_table.add_column("E_above_hull (eV/at)")
        for mat in result.materials[:10]:
            gap_color = "green" if mat.band_gap > 0 else "red"
            mat_table.add_row(
                mat.mp_id, mat.formula,
                f"[{gap_color}]{mat.band_gap:.2f}[/{gap_color}]",
                f"{mat.energy_above_hull:.3f}",
            )
        console.print(mat_table)

    if result.errors:
        for err in result.errors[:5]:
            console.print(f"[dim]  - {err}[/dim]")

    return {"downloaded": result.downloaded, "converted": result.converted, "errors": len(result.errors)}


def _handle_predict(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Predict SCF convergence time and probability before calculation."""
    try:
        from .optimizer.ml_predict import predict_convergence
    except ImportError as e:
        return {"error": f"ML predictor not available: {e}"}

    console.print(Panel(
        f"[cyan]SCF Convergence Prediction[/]\n"
        f"Case: [bold]{args.case}[/]",
        border_style="magenta",
    ))

    use_history = not args.no_history
    pred = predict_convergence(
        case_name=args.case,
        struct_path=args.struct,
        use_history=use_history,
    )

    table = Table(title=f"Prediction: {args.case}", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")
    table.add_row("Estimated SCF time", f"{pred.estimated_time_hours:.1f} ± {pred.time_uncertainty_hours:.1f} hours")
    table.add_row("Convergence probability", f"{pred.convergence_probability * 100:.0f}%")
    table.add_row("Estimated SCF cycles", str(pred.estimated_cycles))
    table.add_row("Recommended mixing", f"β = {pred.recommended_mixing:.3f}")
    difficulty_color = {"easy": "green", "moderate": "yellow", "hard": "red", "very_hard": "red"}
    table.add_row("Difficulty", f"[{difficulty_color.get(pred.convergence_difficulty, 'white')}]{pred.convergence_difficulty}[/]")
    console.print(table)

    if pred.feature_importance:
        sorted_features = sorted(pred.feature_importance.items(), key=lambda x: -x[1])[:5]
        fi_table = Table(title="Top Feature Importance", border_style="dim")
        fi_table.add_column("Feature", style="cyan")
        fi_table.add_column("Importance", style="green")
        for feat, imp in sorted_features:
            fi_table.add_row(feat, f"{imp:.3f}")
        console.print(fi_table)

    return {
        "estimated_hours": pred.estimated_time_hours,
        "uncertainty_hours": pred.time_uncertainty_hours,
        "convergence_probability": pred.convergence_probability,
        "recommended_mixing": pred.recommended_mixing,
        "difficulty": pred.convergence_difficulty,
    }


def _handle_advise(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Show intelligent optimization advice with Roofline, Amdahl, NUMA analysis.

    This is the UI layer that connects the backend intelligence to the user,
    providing Roofline, Amdahl, and NUMA analysis in a human-readable format.
    """
    from .core.case_parser import CaseFileParser
    from .core.hardware import (
        calculate_peak_fp64_gflops,
        get_cpu_architecture,
        get_memory_bandwidth_gb_s,
        get_numa_node_count,
        get_physical_cores,
    )

    plain = getattr(args, "plain", False)
    target = getattr(args, "target", "time")

    case_path = Path(args.case)
    case_data = None
    try:
        parser = CaseFileParser(case_path if case_path.exists() else None)
        case_data = parser.parse_all()
    except Exception:
        pass

    atoms = getattr(case_data, "atoms", 10) if case_data else 10
    nmat = args.nmat or (getattr(case_data, "nmat", 0) if case_data else 5000)
    kpoints = args.kpoints or (getattr(case_data, "kpoints", 0) if case_data else 8)

    if nmat == 0:
        nmat = 5000
    if kpoints == 0:
        kpoints = 8

    cores = args.cores or get_physical_cores()
    arch = get_cpu_architecture()
    mem_bw = get_memory_bandwidth_gb_s()
    peak_gflops = calculate_peak_fp64_gflops()
    numa_nodes = get_numa_node_count()

    from .core.scheduler import detect as detect_topology
    topo = detect_topology(max_cores=cores)

    if args.json:
        import json as _json
        result = _build_advice_dict(
            nmat=nmat, kpoints=kpoints, atoms=atoms, cores=cores,
            arch=arch, mem_bw=mem_bw, peak_gflops=peak_gflops,
            numa_nodes=numa_nodes, topo=topo, plain=plain, target=target,
        )
        console.print_json(_json.dumps(result))
        return result

    _print_advice_rich(
        nmat=nmat, kpoints=kpoints, atoms=atoms, cores=cores,
        arch=arch, mem_bw=mem_bw, peak_gflops=peak_gflops,
        numa_nodes=numa_nodes, topo=topo, plain=plain, target=target,
    )

    return {"status": "advice_displayed"}


def _build_advice_dict(nmat, kpoints, atoms, cores, arch, mem_bw,
                       peak_gflops, numa_nodes, topo, plain, target):
    """Build structured advice dictionary for JSON export."""
    from .optimizer.advisor import (
        estimate_amdahl_saturation,
        roofline_crossover_analysis,
    )
    roofline = roofline_crossover_analysis(
        {"mem_bw_gb_s": mem_bw, "arch": arch, "peak_fp64_gflops": peak_gflops},
        oi=0.15, target_backend="wien2k_lapw1",
    )
    amdahl = estimate_amdahl_saturation(kpoints, nmat, atoms, cores)

    return {
        "hardware": {
            "cpu_arch": arch,
            "cores": cores,
            "numa_nodes": numa_nodes,
            "memory_bandwidth_gb_s": mem_bw,
            "peak_fp64_gflops": peak_gflops,
        },
        "problem": {
            "nmat": nmat, "kpoints": kpoints, "atoms": atoms,
        },
        "roofline": {
            "regime": roofline["regime"],
            "efficiency_pct": roofline["efficiency_pct"],
            "optimal_cores": roofline["optimal_cores"],
            "suggestion": roofline["suggestion"],
        },
        "amdahl": amdahl,
    }


def _print_advice_rich(nmat, kpoints, atoms, cores, arch, mem_bw,
                       peak_gflops, numa_nodes, topo, plain, target):
    """Print rich terminal advice panel with Roofline + Amdahl + suggestions."""
    import os as _os

    from .optimizer.advisor import (
        estimate_amdahl_saturation,
        roofline_crossover_analysis,
    )
    _os.environ.setdefault("WIEN2K_NO_DETECT", "1")
    roofline = roofline_crossover_analysis(
        {"mem_bw_gb_s": mem_bw, "arch": arch, "peak_fp64_gflops": peak_gflops},
        oi=0.15, target_backend="wien2k_lapw1",
    )
    amdahl = estimate_amdahl_saturation(kpoints, nmat, atoms, cores)

    _lang = _build_plain_language() if plain else {}

    console.print(Panel(
        f"[cyan bold]WIEN2k Optimization Advisor[/]\n"
        f"System: {nmat}×{nmat} matrix, {kpoints} k-points, {atoms} atoms | "  # noqa: RUF001
        f"[dim]{arch} • {cores} cores • {numa_nodes} NUMA nodes • {mem_bw:.0f} GB/s mem[/]\n"
        f"[dim]Target: optimize for [bold]{target}[/][/]",
        border_style="cyan",
    ))

    bottleneck = None
    if roofline["regime"] == "memory_bound" and mem_bw < 100:
        label = "Memory Bandwidth"
        msg = ("حافظه کم آوردی — MPI بیشتر اضافه نکن، OpenMP بده" if plain
               else "LAPW1 is memory-hungry — extra MPI ranks won't help, use OpenMP instead")
        bottleneck = (f"[red]{label}[/]", "red", msg)
    elif isinstance(amdahl, dict) and amdahl.get("saturation_cores", cores) < max(cores * 0.6, 1):
        sat = amdahl["saturation_cores"]
        label = "Amdahl Saturation"
        msg = (f"قانون آمال میگه بیش از {sat} هسته بی‌فایده‌ست" if plain
               else f"More than {sat} cores won't improve performance (Amdahl's Law)")
        bottleneck = (f"[yellow]{label}[/]", "yellow", msg)

    if bottleneck:
        console.print(Panel(bottleneck[2], title=bottleneck[0], border_style=bottleneck[1]))

    table = Table(title=("تحلیل عملکرد" if plain else "Performance Analysis"), border_style="blue")
    table.add_column(("معیار" if plain else "Metric"), style="cyan")
    table.add_column(("مقدار" if plain else "Value"), style="green bold")
    table.add_column(("یعنی چه" if plain else "What This Means"), style="dim")

    regime_fa = "محدود به حافظه — MPI اضافی کمکی نمی‌کند" if roofline["regime"] == "memory_bound" else "محدود به پردازنده — MPI بیشتر جواب می‌دهد"
    regime_en = "Memory bottleneck — extra MPI won't help" if roofline["regime"] == "memory_bound" else "CPU-limited — more MPI will help"
    table.add_row(
        ("نوع گلوگاه" if plain else "Roofline Regime"),
        f"[{'red' if roofline['regime'] == 'memory_bound' else 'green'}]{roofline['regime'].replace('_', ' ').title()}[/]",
        (regime_fa if plain else regime_en),
    )
    table.add_row(
        ("کارایی Roofline" if plain else "Roofline Efficiency"),
        f"{roofline['efficiency_pct']:.0f}%", "",
    )
    table.add_row(
        ("تعداد هستهٔ بهینه" if plain else "Optimal Cores (Roofline)"),
        str(roofline["optimal_cores"]),
        ("با این تعداد بهترین کارایی رو می‌گیری" if plain else "Best efficiency at this core count"),
    )

    if isinstance(amdahl, dict):
        sat_cores = amdahl.get("saturation_cores", cores)
        eff = amdahl.get("efficiency_pct", 100.0)
        table.add_row(
            ("اشباع آمال" if plain else "Amdahl Saturation"),
            str(sat_cores),
            ("فراتر از این تعداد، بهبود محسوسی نداری" if plain else "Beyond this, speedup plateaus"),
        )
        table.add_row(
            ("کارایی آمال" if plain else "Amdahl Efficiency"),
            f"{eff:.0f}%", "",
        )
    else:
        sat_cores = cores
        eff = 100.0

    console.print(table)

    rec_table = Table(title=("پیشنهادات" if plain else "Recommendations"), border_style="green")
    rec_table.add_column("#", style="dim")
    rec_table.add_column(("اقدام" if plain else "Action"), style="cyan bold")
    rec_table.add_column(("دلیل" if plain else "Why"), style="dim")
    rec_table.add_column(("تأثیر" if plain else "Impact"), style="green")

    counter = 1
    if roofline["regime"] == "memory_bound":
        rec_table.add_row(str(counter),
            ("OpenMP رو زیاد کن، MPI کم کن" if plain else "Increase OpenMP threads, reduce MPI ranks"),
            ("پهنای باند حافظه اشباع شده" if plain else "Memory bandwidth saturated"),
            ("بالا" if plain else "HIGH"))
        counter += 1
        if omp := (cores // max(1, numa_nodes)):
            rec_table.add_row(str(counter),
                f"export OMP_NUM_THREADS={omp}",
                ("هر رتبهٔ MPI روی یک گرهٔ NUMA" if plain else "One MPI rank per NUMA node"),
                ("بالا" if plain else "HIGH"))
            counter += 1
    elif sat_cores < cores * 0.7:
        rec_table.add_row(str(counter),
            (f"حداکثر {sat_cores} هسته استفاده کن" if plain else f"Limit to {sat_cores} cores"),
            ("قانون آمال — بیش از این هدر رفته" if plain else "Amdahl's Law — more is wasted"),
            ("متوسط" if plain else "MEDIUM"))
        counter += 1

    if nmat > 5000:
        rec_table.add_row(str(counter),
            "lapw2_vector_split: 1" if not plain else "lapw2_vector_split رو فعال کن",
            ("کاهش I/O برای ماتریس بزرگ" if plain else "Large matrix I/O reduction"),
            ("متوسط" if plain else "MEDIUM"))
        counter += 1

    if kpoints > 1 and kpoints % cores != 0:
        rec_table.add_row(str(counter),
            ("تعداد k-point رو مضربی از هسته‌ها کن" if plain else "Set k-points to a multiple of core count"),  # noqa: RUF001
            ("توزیع نامتوازن بار" if plain else "Uneven load distribution"),
            ("متوسط" if plain else "MEDIUM"))
        counter += 1

    console.print(rec_table)
    console.print("\n[dim]➤ Run [bold]forge optimize --simulated[/] to auto-tune RKMAX/KPPRA/mixing[/]")
    console.print("[dim]➤ Run [bold]forge generate[/] to produce optimized .machines[/]")


_SIMPLE_LANGUAGE = {
    "memory_bound": "حافظه کم آوردی — MPI بیشتر اضافه نکن، OpenMP بده",
    "compute_bound": "پردازنده محدودیت داره — MPI بیشتر جواب میده",
    "rkmax": "اندازهٔ مجموعهٔ پایه (هرچی بیشتر = دقیق‌تر ولی کندتر)",
    "kppra": "تعداد k-point (نقاط نمونه‌برداری در فضای انرژی)",
    "mixing": "سرعت همگرایی SCF (کمتر = پایدارتر ولی کندتر)",
    "granularity": "تعداد k-point در هر گروه موازی",
    "elpa": "کتابخانهٔ قطری‌سازی سریع برای ماتریس‌های بزرگ",
    "nuam": "معماری حافظهٔ غیریکنواخت — دسترسی به حافظهٔ نزدیک سریع‌تره",
}


def _build_plain_language():
    return _SIMPLE_LANGUAGE


def _handle_converge(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Run automated convergence testing."""
    try:
        from .optimizer.convergence import (
            find_converged_parameters,
            generate_convergence_report,
            run_kpoint_convergence,
            run_rkmax_convergence,
        )
    except ImportError as e:
        return {"error": f"Convergence module dependencies not available: {e}"}

    console.print(Panel(
        f"[cyan]Convergence Study[/]\nCase: [bold]{args.case}[/]\nMode: [bold]{args.mode}[/]\nTolerance: [bold]{args.tolerance} Ry[/]",
        border_style="blue"
    ))

    wien2k_cmd = {"run_lapw": "run_lapw", "lapw0": "lapw0", "lapw1": "lapw1", "lapw2": "lapw2"}
    results = {}

    if args.mode in ("kpoints", "both"):
        kpoint_grids = [cast(tuple[int, int, int], tuple(int(x) for x in g.split(","))) for g in args.kpoints.split()]
        console.print(f"[bold]Running k-point convergence with grids: {kpoint_grids}...[/bold]")
        results["kpoints"] = run_kpoint_convergence(args.case, kpoint_grids, wien2k_cmd)

    if args.mode in ("rkmax", "both"):
        rkmax_values = [float(x) for x in args.rkmax.split(",")]
        console.print(f"[bold]Running RKmax convergence with values: {rkmax_values}...[/bold]")
        results["rkmax"] = run_rkmax_convergence(args.case, rkmax_values, wien2k_cmd)

    for key, data in results.items():
        converged = find_converged_parameters(data, tolerance=args.tolerance * 1000.0)
        console.print(f"[green]{key}: converged at {converged.get('converged_value')}[/green]")

    report = generate_convergence_report({"results": list(results.values())})
    console.print(Panel(report, title="Convergence Report", border_style="green"))

    return {"status": "completed", "results": {k: v for k, v in results.items()}}


def _handle_history(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Query execution history database."""
    try:
        from .optimizer.history import ExecutionHistory
    except ImportError as e:
        return {"error": f"History module dependencies not available: {e}"}

    with ExecutionHistory() as history:
        if args.show:
            records = history.query({"run_id": args.show})
            if not records:
                console.print(f"[red]No record found for run_id: {args.show}[/red]")
                return {"found": False, "run_id": args.show}
            rec = records[0]
            table = Table(title=f"Run {args.show}", border_style="cyan")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Backend", rec.backend)
            table.add_row("Mode", rec.mode)
            table.add_row("Cores", str(rec.total_cores))
            table.add_row("Walltime", f"{rec.walltime_sec:.1f}s")
            table.add_row("Success", str(rec.success))
            table.add_row("NMAT", str(rec.nmat))
            table.add_row("k-points", str(rec.nkpt))
            console.print(table)
            return rec.to_dict() if hasattr(rec, 'to_dict') else {"run_id": rec.run_id}

        if args.similar_to:
            case_path = Path(args.similar_to)
            if case_path.exists() and case_path.suffix == ".struct":
                recs = history.get_similar(nmat=5000, nkpt=4, backend=cfg.backend or "wien2k", limit=args.limit)
            else:
                recs = history.query(limit=args.limit)
            if recs:
                table = Table(title=f"Similar Runs (limit={args.limit})", border_style="cyan")
                table.add_column("Run ID", style="cyan")
                table.add_column("Backend", style="green")
                table.add_column("Cores", style="green")
                table.add_column("Walltime", style="green")
                table.add_column("Success")
                for r in recs:
                    table.add_row(r.run_id[:8], r.backend, str(r.total_cores), f"{r.walltime_sec:.1f}s", "✓" if r.success else "✗")
                console.print(table)
            else:
                console.print("[yellow]No similar runs found.[/yellow]")
            return {"count": len(recs)}

        recs = history.query(limit=args.limit)
        if recs:
            table = Table(title="Execution History", border_style="cyan")
            table.add_column("Run ID", style="cyan")
            table.add_column("Backend")
            table.add_column("Cores")
            table.add_column("Walltime")
            table.add_column("Date")
            table.add_column("Status")
            for r in recs:
                ts = str(r.timestamp)[:10] if r.timestamp else "?"
                table.add_row(r.run_id[:8], r.backend, str(r.total_cores), f"{r.walltime_sec:.1f}s", ts, "✓" if r.success else "✗")
            console.print(table)
        else:
            console.print("[yellow]No execution history found.[/yellow]")

        return {"records": len(recs)}


def _handle_analyze_bands(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    """Extract band structure and DOS data from WIEN2k output."""
    try:
        from .core.electronic_structure import compute_band_gap, parse_band_structure, parse_dos
    except ImportError as e:
        return {"error": f"Electronic structure module dependencies not available: {e}"}

    case_path = Path(args.case)
    if case_path.is_dir():
        base_path = str(case_path)
        case_name = args.case
    else:
        base_path = str(case_path.parent) if case_path.parent != Path() else "."
        case_name = case_path.stem

    console.print(f"[cyan]Analyzing bands for case: [bold]{case_name}[/bold] in {base_path}[/cyan]")

    band_data = parse_band_structure(case_name, base_path)
    gap_info = compute_band_gap(band_data)

    table = Table(title="Band Structure Summary", border_style="cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("k-points", str(band_data.get("nkpt", 0)))
    table.add_row("Bands", str(band_data.get("nbnd", 0)))
    table.add_row("Spin-polarized", "Yes" if band_data.get("nspin", 1) > 1 else "No")
    table.add_row("Fermi energy", f"{band_data.get('fermi', 0):.4f} eV")
    table.add_row("Band gap", f"{gap_info.get('gap_ev', 0):.4f} eV")
    table.add_row("Direct gap", f"{gap_info.get('direct_gap_ev', 0):.4f} eV")
    console.print(table)

    result = {"case": case_name, "nkpt": band_data.get("nkpt"), "nbnd": band_data.get("nbnd"), "fermi_ev": band_data.get("fermi"), "gap_ev": gap_info.get("gap_ev")}

    if args.dos:
        dos_data = parse_dos(case_name, base_path)
        result["dos_n_energy"] = len(dos_data.get("energies", []))
        console.print(f"[green]DOS: {result['dos_n_energy']} energy points parsed.[/green]")

    if args.output:
        export_data = {
            "band_structure": {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in band_data.items() if k not in ("k_points", "eigenvalues")},
            "gap": gap_info,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        console.print(f"[green]Exported to {args.output}[/green]")

    return result


# =============================================================================
# CLI Execution Engine
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:  # noqa: C901
    """
    CLI entry point with structured setup, dispatch, and error handling.
    Returns OS exit code: 0 (success), 1 (app error), 2 (CLI syntax error).
    """
    global console

    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    plain_mode = getattr(args, "plain", False) or getattr(args, "no_color", False)
    caps = detect_terminal_capabilities()

    if plain_mode or not caps.supports_color or _is_dumb:
        console = get_plain_console()
    else:
        console = get_rich_console()

    try:
        cfg = load_config(file_path=args.config, cli_override={
            "log_level": "DEBUG" if args.verbose > 0 else "ERROR" if args.quiet else "INFO",
            "quiet_mode": args.quiet,
            "backend": args.backend
        })
        ensure_dirs()
        setup_logging(config=cfg, verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
        set_context(cli="forge", user=os.environ.get("USER", "unknown"))
    except Exception as e:
        sys.stderr.write(f"Critical: Failed to initialize configuration/logging: {e}\n")
        return 2

    def _signal_handler(sig: int, frame: Any) -> None:
        logger.warning(f"Received signal {sig}. Cleaning up...")
        BackendManager.instance().reset()
        sys.exit(130)
        
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    handlers = {
        "generate": _handle_generate,
        "submit": _handle_submit,
        "benchmark": _handle_benchmark,
        "diagnostics": _handle_diagnostics,
        "hardware": _handle_hardware,
        "analyze": _handle_analyze,
        "tui": _handle_tui,
        "monitor": _handle_monitor,
        "run": _handle_run,
        "workflow": _handle_workflow,
        "diagnose": _handle_diagnose,
        "optimize": _handle_optimize,
        "screen": _handle_screen,
        "predict": _handle_predict,
        "advise": _handle_advise,
        "converge": _handle_converge,
        "history": _handle_history,
        "analyze-bands": _handle_analyze_bands,
    }

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help()
        return 2

    try:
        logger.info(f"Executing command: {args.command}")
        result = handler(args, cfg)
        
        if args.json_output:
            print(json.dumps(result, indent=2, default=str))
        elif result and result.get("warnings"):
            for w in result["warnings"]:
                logger.warning(w)
                
        logger.info(f"Command '{args.command}' completed successfully.")
        return 0
        
    except FORGEError as e:
        log_exception_structured(e)
        if args.json_output:
            print(json.dumps({"error": e.to_dict()}, indent=2))
        else:
            console.print(Panel(format_error_for_ui(e), title="Error", border_style="red"))
        return 1
        
    except Exception as e:
        logger.error(f"Unhandled exception in CLI dispatch: {e}", exc_info=True)
        if args.json_output:
            print(json.dumps({"error": {"message": str(e), "type": type(e).__name__}}, indent=2))
        else:
            console.print(f"[red]Unexpected Error:[/red] {e}\n[dim]Use -v or check logs for traceback.[/dim]")
        return 1


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "_detect_scheduler",
    "_resolve_scheduler",
    "create_parser",
    "main",
]

if __name__ == "__main__":
    sys.exit(main())
