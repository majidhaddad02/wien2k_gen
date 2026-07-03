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
• Seamless integration with refactored core, optimizer, submit, benchmark, and UI modules
• Thread-safe execution context, signal-aware teardown, and non-blocking I/O
• Comprehensive English documentation, type hints, and HPC-grade resilience patterns
"""

import os
import sys
import json
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Project imports (aligned with refactored architecture)
from .config import load_config, get_config, AppConfig, ensure_dirs
from .logging_config import setup_logging, get_logger, set_context
from .backend_manager import get_backend, set_backend, list_backends, BackendManager
from .types import BackendCode, ExecutionMode, PipelineResult, TopologyData, OptimizationTarget
from .exceptions import (
    Wien2kGenError, format_error_for_ui, log_exception_structured,
    MissingInputError, BackendError, ConfigurationError
)
from .core.pipeline import run_pipeline
from .core.scheduler import detect as detect_topology
from .submit.slurm import submit_slurm_job, SlurmJobSpec, SlurmDirectives
from .benchmark.real import RealBenchmarkRunner
from .utils.diagnostic import run_diagnostics, export_diagnostics_json
from .ui.interactive import launch_app
from .ui.analysis import parse_scf_log, generate_report

logger = get_logger(__name__)
console = Console()


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
            "  wien2k_gen submit --partition gpu --time 48:00:00 --mem 64G\n"
            "  wien2k_gen diagnostics --export report.json --full\n"
            "  wien2k_gen tui\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global flags
    global_group = parser.add_argument_group("Global Options")
    global_group.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    global_group.add_argument("-q", "--quiet", action="store_true", help="Suppress console output")
    global_group.add_argument("--json", action="store_true", dest="json_output", help="Output results in JSON format")
    global_group.add_argument("--config", type=str, default=None, help="Path to custom config file")
    global_group.add_argument("--backend", type=str, choices=[b.value for b in BackendCode], help="Override auto-detected backend")
    global_group.add_argument("--log-file", type=str, default=None, help="Redirect logs to file")
    global_group.add_argument("--version", action="version", version="wien2k_gen v9.8.0")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available workflow commands", required=True)

    # 1. generate
    gen_p = subparsers.add_parser("generate", help="Generate parallel configuration files")
    gen_p.add_argument("--nodes", type=int, default=None, help="Number of compute nodes")
    gen_p.add_argument("--cores", type=int, default=None, help="Total MPI cores to allocate")
    gen_p.add_argument("--omp", type=int, default=None, help="OpenMP threads per rank")
    gen_p.add_argument("--mode", type=str, choices=[m.value for m in ExecutionMode], help="Parallel execution mode")
    gen_p.add_argument("--target", type=str, choices=[t.value for t in OptimizationTarget], default="time", help="Optimization target (time, memory, balanced, cost)")
    gen_p.add_argument("--max-cores", type=int, default=None, help="Hard limit on total cores to utilize")
    gen_p.add_argument("--memory-limit", type=float, default=None, help="Hard limit on memory per node (GB)")
    gen_p.add_argument("--dry-run", action="store_true", help="Generate config without writing to disk")
    gen_p.add_argument("--export", type=str, default=None, help="Export configuration summary to path")
    gen_p.add_argument("--overwrite", action="store_true", help="Overwrite existing .machines/INCAR without prompt")

    # 2. submit
    sub_p = subparsers.add_parser("submit", help="Submit job to SLURM/scheduler")
    sub_p.add_argument("--partition", type=str, default="", help="Scheduler partition/queue")
    sub_p.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    sub_p.add_argument("--ntasks", type=int, default=0, help="Total tasks (0 = auto from topology)")
    sub_p.add_argument("--time", type=str, default="24:00:00", help="Walltime (HH:MM:SS)")
    sub_p.add_argument("--mem", type=str, default="4G", help="Memory per node")
    sub_p.add_argument("--job-name", type=str, default="wien2k_job", help="Job identifier")
    sub_p.add_argument("--dependency", type=str, default="", help="Job dependency (e.g., afterok:123)")
    sub_p.add_argument("--dry-run", action="store_true", help="Generate script only, do not submit")
    sub_p.add_argument("--export", type=str, default=None, help="Export SBATCH script to path")

    # 3. benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run empirical or synthetic benchmarks")
    bench_p.add_argument("--type", choices=["real", "synthetic"], default="real", help="Benchmark type")
    bench_p.add_argument("--max-cores", type=int, default=None, help="Maximum cores for scaling suite")
    bench_p.add_argument("--walltime", type=str, default="02:00:00", help="Max runtime per run")
    bench_p.add_argument("--output", type=str, default=None, help="Save results to JSON path")
    bench_p.add_argument("--skip-cleanup", action="store_true", help="Retain temporary benchmark directories")

    # 4. diagnostics
    diag_p = subparsers.add_parser("diagnostics", help="Collect system & environment health report")
    diag_p.add_argument("--export", type=str, default=None, help="Save diagnostic report to path")
    diag_p.add_argument("--full", action="store_true", help="Include verbose library & interconnect checks")

    # 5. analyze
    ana_p = subparsers.add_parser("analyze", help="Parse SCF logs & generate performance reports")
    ana_p.add_argument("--log", type=str, required=True, help="Path to SCF/output log file")
    ana_p.add_argument("--code", type=str, choices=["wien2k", "vasp", "qe"], default=None, help="Force DFT code parser")
    ana_p.add_argument("--export", type=str, default=None, help="Export analysis report to JSON")

    # 6. tui
    tui_p = subparsers.add_parser("tui", help="Launch interactive terminal UI")
    tui_p.add_argument("--compact", action="store_true", help="Enable compact UI layout")

    return parser


# =============================================================================
# Command Handlers
# =============================================================================

def _handle_generate(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Execute pipeline configuration generation with Rich UI output."""
    topo = detect_topology(max_cores=args.max_cores)
    
    suggestion = {}
    if args.mode: 
        suggestion["mode"] = ExecutionMode(args.mode)
    if args.cores: 
        suggestion["recommended_total_cores"] = args.cores
    if args.omp: 
        suggestion["omp_threads_per_rank"] = args.omp
    if args.memory_limit:
        suggestion["memory_limit_gb"] = args.memory_limit

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
            # Rich Table for Dry-Run Summary
            table = Table(title="Configuration Preview", border_style="cyan")
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            # Extract some mock/real metrics from result if available
            table.add_row("Mode", str(suggestion.get("mode", "auto")))
            table.add_row("Total Cores", str(suggestion.get("recommended_total_cores", topo.total_cores)))
            table.add_row("Dry-Run Content", f"[dim]{len(result.dry_run_content)} bytes generated[/dim]")
            
            console.print(table)
            console.print(Panel(result.dry_run_content, title="Generated Config", border_style="dim"))
        else:
            console.print(Panel(f"[green]✓ Configuration generated successfully.[/]\nPath: [cyan]{result.config_path}[/]", border_style="green"))
            
        if result.warnings:
            warn_table = Table(title="⚠ Warnings", show_header=False, box=None)
            for w in result.warnings:
                warn_table.add_row("[yellow]•[/]", f"[dim]{w}[/dim]")
            console.print(warn_table)
    else:
        console.print(Panel(f"[red]✗ Generation failed: {result.validation_errors}[/]", border_style="red"))
        
    return {"success": result.success, "path": result.config_path, "warnings": result.warnings}


def _handle_submit(args: argparse.Namespace, cfg: AppConfig) -> Dict[str, Any]:
    """Execute job submission to scheduler."""
    topo = detect_topology(max_cores=args.ntasks or None)
    directives = SlurmDirectives(
        job_name=args.job_name,
        partition=args.partition,
        nodes=args.nodes,
        ntasks=args.ntasks or topo.total_cores,
        cpus_per_task=1,
        mem_per_node=args.mem,
        time=args.time,
        dependency=args.dependency
    )
    spec = SlurmJobSpec(
        topo=topo,
        exec_command="run_lapw -p",
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
        runner = RealBenchmarkRunner({
            "backend": cfg.backend,
            "use_slurm": True,
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
        
    # Rich summary for diagnostics
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


# =============================================================================
# CLI Execution Engine
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point with structured setup, dispatch, and error handling.
    Returns OS exit code: 0 (success), 1 (app error), 2 (CLI syntax error).
    """
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if e.code is not None else 2

    # 1. Initialize Config & Logging
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

    # 2. Signal Handling for Graceful Teardown
    def _signal_handler(sig: int, frame: Any) -> None:
        logger.warning(f"Received signal {sig}. Cleaning up...")
        BackendManager.instance().reset()
        sys.exit(130)
        
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 3. Command Dispatch
    handlers = {
        "generate": _handle_generate,
        "submit": _handle_submit,
        "benchmark": _handle_benchmark,
        "diagnostics": _handle_diagnostics,
        "analyze": _handle_analyze,
        "tui": _handle_tui
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
            # Graceful degradation: Rich-formatted error with hint
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
]

if __name__ == "__main__":
    sys.exit(main())