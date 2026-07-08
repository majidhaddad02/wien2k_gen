"""
Command-Line Interface (CLI) Entry Point for Wien2kGen.
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

import os
import sys
import json
import time
import shutil
import argparse
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import load_config, get_config, AppConfig, ensure_dirs
from .logging_config import setup_logging, get_logger, set_context
from .backend_manager import get_backend, set_backend, list_backends, BackendManager
from .types import BackendCode, ExecutionMode, PipelineResult, TopologyData, OptimizationTarget
from .exceptions import (
    Wien2kGenError, format_error_for_ui, log_exception_structured,
    MissingInputError, BackendError, ConfigurationError
)
from .core.pipeline import run_pipeline
from .core.scheduler import detect as detect_topology, _detect_scheduler, auto_detect_memory
from .core.hardware import get_physical_cores
from .submit.slurm import submit_slurm_job, SlurmJobSpec, SlurmDirectives
from .submit import SUBMIT_PROVIDERS
from .ui.rich_ui import detect_terminal_capabilities, get_rich_console, get_plain_console
from .benchmark.real import RealBenchmarkRunner
from .utils.diagnostic import run_diagnostics, export_diagnostics_json
from .ui.interactive import launch_app
from .ui.analysis import parse_scf_log, generate_report

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
        prog="wien2k_gen",
        description="Production-grade WIEN2k parallel configuration & HPC job dispatcher",
        epilog=(
            "Examples:\n"
            "  wien2k_gen generate --backend wien2k --cores 64 --target memory --dry-run\n"
            "  wien2k_gen submit --partition gpu --time 48:00:00 --mem 64G --scheduler slurm\n"
            "  wien2k_gen diagnostics --export report.json --full\n"
            "  wien2k_gen tui\n"
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
    global_group.add_argument("--version", action="version", version="wien2k_gen v0.1.0")
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

    ana_p = subparsers.add_parser("analyze", help="Parse SCF logs & generate performance reports")
    ana_p.add_argument("--log", type=str, required=True, help="Path to SCF/output log file")
    ana_p.add_argument("--code", type=str, choices=["wien2k", "vasp", "qe"], default=None, help="Force DFT code parser")
    ana_p.add_argument("--export", type=str, default=None, help="Export analysis report to JSON")

    tui_p = subparsers.add_parser("tui", help="Launch interactive terminal UI")
    tui_p.add_argument("--compact", action="store_true", help="Enable compact UI layout")

    mon_p = subparsers.add_parser("monitor", help="Monitor SCF convergence in real-time")
    mon_p.add_argument("--case", type=str, required=True, help="Case name / path to monitor")
    mon_p.add_argument("--output", type=str, default=None, help="SCF output file to parse")
    mon_p.add_argument("--interval", type=int, default=10, help="Polling interval in seconds (default: 10)")

    conv_p = subparsers.add_parser("converge", help="Run automated convergence tests")
    conv_p.add_argument("--case", type=str, required=True, help="Case name")
    conv_p.add_argument("--mode", type=str, choices=["kpoints", "rkmax", "both"], default="both", help="Parameter to converge (default: both)")
    conv_p.add_argument("--tolerance", type=float, default=0.001, help="Energy tolerance in Ry (default: 0.001)")
    conv_p.add_argument("--kpoints", type=str, default="2,2,2 4,4,4 6,6,6 8,8,8 10,10,10", help="Space-separated k-point grids (default: '2,2,2 4,4,4 6,6,6 8,8,8 10,10,10')")
    conv_p.add_argument("--rkmax", type=str, default="5,6,7,8,9,10", help="Comma-separated RKmax values (default: '5,6,7,8,9,10')")

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
        return params.get("exec_command", "run_lapw -p")
    except Exception:
        return "run_lapw -p"


def _handle_generate(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Execute pipeline configuration generation with Rich UI output."""
    max_cores = args.max_cores
    if args.reserve_os_cores is not None:
        phys = get_physical_cores()
        reserved_max = max(1, phys - args.reserve_os_cores)
        max_cores = min(max_cores, reserved_max) if max_cores else reserved_max
        console.print(f"[dim]Reserving {args.reserve_os_cores} OS cores → using {max_cores} of {phys}[/dim]")
    topo = detect_topology(max_cores=max_cores)
    
    suggestion = {}
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
            from .backends.gpu_backend import detect_gpu, get_gpu_recommendation, get_mixed_precision_recommendation
            gpus = detect_gpu()
            if gpus:
                gpu_rec = get_gpu_recommendation(topo, nmat=5000, nkpt=8, mode=suggestion.get("mode", "mpi"))
                suggestion["gpu_recommendation"] = gpu_rec
                if args.gpu_mixed_precision:
                    fp_rec = get_mixed_precision_recommendation(topo, nmat=5000)
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

            sat_data = suggestion.get("saturation_data", {})
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


def _handle_submit(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
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
            return res
            
        if res.get("success"):
            if args.dry_run:
                console.print(Panel(res.get("dry_run_content", "Script content not available"), title="SBATCH Preview", border_style="cyan"))
            else:
                console.print(Panel(f"[green]✓ Job submitted successfully.[/]\nJob ID: [bold cyan]{res.get('job_id')}[/]\nScript: [dim]{res.get('script_path')}[/]", border_style="green"))
        else:
            console.print(Panel(f"[red]✗ Submission failed: {res.get('errors')}[/]", border_style="red"))
        return {"success": res.get("success"), "job_id": res.get("job_id"), "path": str(res.get("script_path", ""))}

    elif scheduler in ("pbs", "lsf"):
        provider_cls = SUBMIT_PROVIDERS.get(scheduler)
        if provider_cls:
            provider = provider_cls()
            res = provider.submit(
                topo=topo,
                exec_command="run_lapw -p",
                directives={
                    "job_name": args.job_name,
                    "queue" if scheduler == "pbs" else "queue": args.partition,
                    "nodes": args.nodes,
                    "walltime": args.time,
                    "mem" if scheduler == "pbs" else "memory": args.mem,
                },
                script_path=Path(args.export) if args.export else None,
                dry_run=args.dry_run,
            )
            if args.json_output:
                return res
            if res.get("success"):
                console.print(Panel(f"[green]✓ Job submitted successfully.[/]\nJob ID: [bold cyan]{res.get('job_id')}[/]\nScript: [dim]{res.get('script_path')}[/]", border_style="green"))
            else:
                console.print(Panel(f"[red]✗ Submission failed: {res.get('errors')}[/]", border_style="red"))
            return {"success": res.get("success"), "job_id": res.get("job_id"), "path": str(res.get("script_path", ""))}
        else:
            return {"success": False, "errors": [f"Scheduler provider '{scheduler}' not available."]}

    return {"success": False, "errors": [f"Unknown scheduler: {scheduler}"]}


def _handle_benchmark(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Run empirical or synthetic benchmark suite."""
    topo = detect_topology(max_cores=args.max_cores)
    
    if args.type == "synthetic":
        from .benchmark.synthetic import generate_strong_scaling_suite
        problem = {"atoms": 20, "kpoints": 8, "nmat": 1500, "is_hybrid": False}
        suite = generate_strong_scaling_suite(problem, topo, max_cores=args.max_cores)
        console.print(Panel(f"[green]✓ Synthetic benchmark complete. {len(suite)} runs generated.[/]", border_style="green"))
        return {"type": "synthetic", "runs": len(suite), "data": [s.to_dict() if hasattr(s, 'to_dict') else s for s in suite]}
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
            return result
            
        status_color = "green" if result["status"] == "success" else "red"
        console.print(Panel(
            f"[{status_color}]Benchmark {result['status']}[/]\n"
            f"Run ID: [cyan]{result['run_id']}[/]\n"
            f"Wall Time: [bold]{result['wall_time_sec']:.2f}s[/]",
            border_style=status_color
        ))
        return {"run_id": result["run_id"], "status": result["status"], "wall_time": result["wall_time_sec"]}


def _handle_diagnostics(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Collect & export system diagnostics."""
    report = run_diagnostics()
    if args.export:
        export_diagnostics_json(report, args.export)
        
    if args.json_output:
        return report
        
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


def _handle_analyze(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Parse SCF log & generate analysis report."""
    parsed = parse_scf_log(args.log, code_hint=args.code)
    report = generate_report(parsed, scaling_data=None, include_recommendations=True)
    
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(report.to_dict() if hasattr(report, 'to_dict') else report, f, indent=2, default=str)
            
    if args.json_output:
        return report.to_dict() if hasattr(report, 'to_dict') else report
        
    console.print(Panel(
        f"[green]✓ Analysis complete.[/]\n"
        f"Code: [cyan]{parsed.get('code', 'unknown')}[/] | "
        f"Converged: [{'green' if parsed.get('converged') else 'red'}]{parsed.get('converged')}[/]\n"
        f"Total Cycles: [bold]{parsed.get('total_cycles', 0)}[/]",
        border_style="green"
    ))
    return {"code": parsed.get("code"), "converged": parsed.get("converged"), "cycles": parsed.get("total_cycles")}


def _handle_tui(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Launch interactive TUI application."""
    if args.json_output:
        return {"status": "tui_launch_skipped_in_json_mode"}
    launch_app()
    return {"status": "tui_exited"}


def _handle_monitor(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Monitor SCF convergence in real-time."""
    try:
        from .optimizer.monitor import start_monitoring, get_monitor_status, stop_monitoring
        from .core.scheduler import detect as detect_topology
    except ImportError as e:
        return {"error": f"Monitor module dependencies not available: {e}"}

    topo = detect_topology()
    console.print(Panel(f"[cyan]Starting SCF Monitor[/]\nCase: [bold]{args.case}[/]\nInterval: {args.interval}s", border_style="blue"))

    output_path = args.output or f"{args.case}.scf"
    os.environ["WIEN2K_SCF_LOG"] = output_path
    
    try:
        start_monitoring(topo, check_interval=args.interval, daemon=False)
        while True:
            status = get_monitor_status()
            if not status.get("running"):
                break
            if status.get("events"):
                last_events = status.get("events", [])[-5:]
                for evt in last_events:
                    console.print(f"[dim]{evt}[/dim]")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        stop_monitoring()
        console.print("[yellow]Monitoring stopped by user.[/yellow]")
    
    return {"status": "monitoring_ended", "case": args.case}


def _handle_converge(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Run automated convergence testing."""
    try:
        from .optimizer.convergence import (
            run_kpoint_convergence, run_rkmax_convergence,
            find_converged_parameters, generate_convergence_report
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
        kpoint_grids = [tuple(int(x) for x in g.split(",")) for g in args.kpoints.split()]
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


def _handle_history(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Query execution history database."""
    try:
        from .optimizer.history import ExecutionHistory, ExecutionRecord
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
                ts = r.timestamp[:10] if r.timestamp else "?"
                table.add_row(r.run_id[:8], r.backend, str(r.total_cores), f"{r.walltime_sec:.1f}s", ts, "✓" if r.success else "✗")
            console.print(table)
        else:
            console.print("[yellow]No execution history found.[/yellow]")

        return {"records": len(recs)}


def _handle_analyze_bands(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Extract band structure and DOS data from WIEN2k output."""
    try:
        from .core.electronic_structure import parse_band_structure, parse_dos, compute_band_gap
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
    gap_info = compute_band_gap(case_name, base_path)

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

def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point with structured setup, dispatch, and error handling.
    Returns OS exit code: 0 (success), 1 (app error), 2 (CLI syntax error).
    """
    global console

    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code is not None else 2

    plain_mode = getattr(args, "plain", False) or getattr(args, "no_color", False)
    caps = detect_terminal_capabilities()

    if plain_mode or not caps.supports_color or _is_dumb:
        console = get_plain_console()
    else:
        console = get_rich_console()

    try:
        cfg = load_config(file_path=args.config, cli_override={
            "log_level": "DEBUG" if args.verbose > 0 else "ERROR" if args.quiet else None,
            "quiet_mode": args.quiet,
            "backend": args.backend
        })
        ensure_dirs()
        setup_logging(config=cfg, verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)
        set_context(cli="wien2k_gen", user=os.environ.get("USER", "unknown"))
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
        "analyze": _handle_analyze,
        "tui": _handle_tui,
        "monitor": _handle_monitor,
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
        
    except Wien2kGenError as e:
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
    "main",
    "create_parser",
    "_detect_scheduler",
    "_resolve_scheduler",
]

if __name__ == "__main__":
    sys.exit(main())
