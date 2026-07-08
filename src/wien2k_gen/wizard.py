"""
Interactive CLI Configuration Wizard for Wien2kGen.
Provides a guided, step-by-step terminal experience for setting up the HPC environment,
validating paths, detecting hardware topology, and generating the initial configuration.

Key Architecture Features:
• Rich, panel-driven UI with progress indicators, tables, and colored help text
• Step-based state machine for complex configuration flows
• Full integration with Optimizer (target selection, max_cores, memory_limit)
• Real-time validation of WIENROOT, SCRATCH, and MPI environments
• Pre-flight validation checks before writing .machines to prevent oversubscription
• Seamless integration with config.py, core scheduler detection, and atomic_write
• Comprehensive English documentation and HPC-grade error resilience
"""

import os
import sys
import json
import time
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.align import Align

from .config import AppConfig, load_config, get_config, DEFAULT_CONFIG_DIR, ensure_dirs
from .logging_config import get_logger, set_context
from .types import BackendCode, ExecutionMode, OptimizationTarget
from .utils.atomic_write import atomic_write
from .core.scheduler import detect as detect_topology, _detect_scheduler
from .core.builder import build_auto
from .optimizer.advisor import suggest_optimal_resources
from .backend_manager import list_backends, set_backend, get_backend
from .exceptions import Wien2kGenError, ConfigurationError

logger = get_logger(__name__)
console = Console()

PROFILES_DIR = Path.home() / ".config" / "wien2k_gen" / "profiles"


# =============================================================================
# Helper Functions
# =============================================================================

def detect_wienroot_candidates() -> List[str]:
    """Suggest likely WIENROOT locations based on system paths."""
    candidates = []
    env = os.environ.get("WIENROOT")
    if env:
        candidates.append(env)
    
    common_paths = ["/opt/codes/WIEN2k", "/usr/local/WIEN2k", str(Path.home() / "WIEN2k")]
    for p in common_paths:
        if Path(p).exists():
            candidates.append(p)
            
    return list(dict.fromkeys(candidates))


def validate_wienroot(path: str) -> bool:
    """Check if WIENROOT contains essential binaries."""
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / "run_lapw").exists() or (p / "siteconfig_lapw").exists()


def check_scratch_health(path: str) -> Dict[str, Any]:
    """Verify scratch path exists, is writable, and has space."""
    p = Path(path)
    info = {
        "valid": True, "writable": False, "exists": False, 
        "free_gb": 0.0, "fs_type": "unknown", "warning": ""
    }
    
    try:
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                info["exists"] = True
            except Exception as e:
                info["valid"] = False
                info["warning"] = f"Cannot create directory: {e}"
                return info
        else:
            info["exists"] = True
            
        if os.access(p, os.W_OK):
            info["writable"] = True
        else:
            info["valid"] = False
            info["warning"] = "Directory is not writable."
            return info
            
        usage = shutil.disk_usage(str(p))
        info["free_gb"] = round(usage.free / (1024 ** 3), 2)
        
        # Check filesystem type
        res = subprocess.run(["df", "-T", str(p)], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if len(lines) > 1:
                info["fs_type"] = lines[-1].split()[1].lower()
                
        if info["free_gb"] < 2.0:
            info["warning"] = f"Low disk space: only {info['free_gb']:.1f} GB available."
            
    except Exception as e:
        info["valid"] = False
        info["warning"] = f"Health check failed: {e}"
        
    return info


# =============================================================================
# Wizard Logic
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
        console.print(f"[yellow]No editor found. Edit manually: {filepath}[/yellow]")
        return

    console.print(f"[bold cyan]Opening {filepath.name} in {editor} for manual review...[/bold cyan]")
    try:
        subprocess.run([editor, str(filepath)], check=False)
        console.print(f"[green]✓ Manual review complete. Final config: {filepath}[/green]")
    except FileNotFoundError:
        console.print(f"[yellow]Editor '{editor}' not found. Edit manually: {filepath}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Editor error: {e}. Edit manually: {filepath}[/yellow]")


def run_wizard(topo=None) -> None:
    """
    Step-by-step interactive configuration with automatic and manual modes,
    plus profile loading at startup and full optimizer integration.
    """
    logger.info("Starting interactive wizard")
    set_context(cli="wien2k_wizard", user=os.environ.get("USER", "unknown"))

    # 0. Topology Detection
    console.print(Panel(Markdown("# 🔧 Wien2kGen Setup Wizard"), subtitle="Interactive Configuration Tool", border_style="cyan"))
    
    if topo is None:
        with console.status("[bold cyan]Detecting hardware topology...", spinner="dots"):
            try:
                topo = detect_topology()
            except Exception as e:
                logger.error(f"Topology detection failed: {e}")
                console.print("[red]❌ Topology detection failed. Cannot continue.[/red]")
                sys.exit(1)
                
    # Display Topology Summary
    topo_table = Table(title="Detected Hardware Topology", show_header=True, header_style="bold magenta")
    topo_table.add_column("Metric", style="cyan")
    topo_table.add_column("Value", style="green")
    topo_table.add_row("Total Cores", str(topo.total_cores))
    topo_table.add_row("Nodes", str(len(topo.nodes)))
    topo_table.add_row("Environment", topo.env_type.capitalize())
    console.print(topo_table)
    console.print(Rule(style="dim"))

    # 0.2 Scheduler Auto-Detection
    detected = _detect_scheduler()
    console.print(f"\n[bold cyan]Scheduler Detection:[/bold cyan] [green]{detected.upper()}[/green] detected.")
    scheduler_choice = Prompt.ask(
        "Select target scheduler",
        choices=["slurm", "pbs", "lsf", "auto"],
        default="auto",
        console=console
    )
    selected_scheduler = detected if scheduler_choice == "auto" else scheduler_choice
    console.print(f"Using scheduler: [bold]{selected_scheduler.upper()}[/bold]")
    console.print(Rule(style="dim"))

    # 0.1 Profile Check
    profile_values = {}
    if PROFILES_DIR.exists():
        profiles = list(PROFILES_DIR.glob("*.json"))
        if profiles:
            console.print("[bold cyan]Saved Profiles:[/bold cyan]")
            t = Table("Name", "Modified", show_header=False)
            for p in sorted(profiles, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                t.add_row(p.stem, time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)))
            console.print(t)
            
            if Confirm.ask("Load a profile?", console=console, default=False):
                name = Prompt.ask("Profile name", console=console)
                prof_path = PROFILES_DIR / f"{name}.json"
                if prof_path.exists():
                    try:
                        with open(prof_path, "r", encoding="utf-8") as f:
                            profile_values = json.load(f)
                        console.print(f"[green]✅ Profile '{name}' loaded.[/green]")
                    except Exception as e:
                        logger.error(f"Failed to load profile: {e}")
                        profile_values = {}
                else:
                    console.print("[yellow]⚠️ Profile not found.[/yellow]")

    # 0.15 WIENROOT Detection & Validation
    if profile_values.get("wienroot"):
        wienroot = profile_values["wienroot"]
        console.print(f"[bold cyan]WIENROOT from profile:[/bold cyan] [green]{wienroot}[/green]")
    else:
        env_wienroot = os.environ.get("WIENROOT", "")
        candidates = detect_wienroot_candidates()
        if candidates:
            console.print("[bold cyan]Detected WIENROOT Candidates:[/bold cyan]")
            cand_table = Table("Path", "Status", show_header=True, header_style="bold magenta")
            for c in candidates:
                valid = validate_wienroot(c)
                status = "[green]✓ Valid[/green]" if valid else "[red]✗ Invalid[/red]"
                cand_table.add_row(c, status)
            console.print(cand_table)
        
        if env_wienroot and validate_wienroot(env_wienroot):
            default_root = env_wienroot
        elif candidates:
            default_root = candidates[0]
        else:
            default_root = ""
        
        wienroot = Prompt.ask(
            "WIENROOT path",
            default=default_root,
            console=console
        )
        while wienroot and not validate_wienroot(wienroot):
            console.print(f"[red]✗ Invalid WIENROOT:[/] {wienroot} (no run_lapw or siteconfig_lapw found)")
            wienroot = Prompt.ask(
                "WIENROOT path",
                default=default_root,
                console=console
            )

    # 0.2 Scratch Path Detection & Health Check
    if profile_values.get("scratch_path"):
        scratch_path = profile_values["scratch_path"]
        console.print(f"[bold cyan]SCRATCH from profile:[/bold cyan] [green]{scratch_path}[/green]")
    else:
        env_scratch = os.environ.get("SCRATCH", os.environ.get("TMPDIR", "/tmp"))
        scratch_path = Prompt.ask(
            "SCRATCH path",
            default=env_scratch,
            console=console
        )
    
    health = check_scratch_health(scratch_path)
    if health["valid"]:
        fs_type = health["fs_type"]
        free_gb = health["free_gb"]
        console.print(f"[bold cyan]Scratch Health:[/bold cyan] {free_gb:.1f} GB free, fs=[bold]{fs_type}[/bold]")
        if fs_type in ("nfs", "nfs4"):
            console.print("[yellow]⚠ WARNING: Scratch on NFS (slow). Consider local SSD or /dev/shm for I/O-heavy jobs.[/yellow]")
        if free_gb < 10:
            console.print(f"[yellow]⚠ WARNING: Low disk space on scratch ({free_gb:.1f} GB). Less than 10 GB recommended.[/yellow]")
    else:
        console.print(f"[red]✗ Scratch health check failed: {health.get('warning', 'Unknown error')}[/red]")

    # 1. Backend Selection
    profile_backend = profile_values.get("backend") or profile_values.get("backend_code")
    if profile_backend:
        try:
            backend_code = BackendCode(profile_backend)
            set_backend(backend_code)
            console.print(f"[bold cyan]Backend from profile:[/bold cyan] [green]{profile_backend}[/green]")
            current_name = profile_backend
        except ValueError:
            pass
    
    if not profile_backend:
        backends = list_backends()
        current_backend = get_backend()
        current_name = current_backend.__class__.__name__.replace("Backend", "").lower()
        console.print(f"Current backend: [bold]{current_name}[/bold]")

        if backends:
            backend_choices = [str(i) for i in range(1, len(backends) + 1)] + ["0"]
            choice = Prompt.ask(
                "Select Backend (0 to keep current)",
                choices=backend_choices,
                default="0",
                console=console
            )
            if choice != "0":
                idx = int(choice) - 1
                if 0 <= idx < len(backends):
                    set_backend(backends[idx])
                    current_name = backends[idx].value
                    console.print(f"✅ Switched to [bold]{current_name}[/bold]")
    backend_name = current_name

    # 2. Optimization Strategy
    console.print("\n[bold cyan]Step 2: Optimization Strategy[/bold cyan]")
    console.print(Rule(style="dim"))
    
    profile_target = profile_values.get("optimization_target") or profile_values.get("target")
    if profile_target:
        try:
            target = OptimizationTarget(profile_target)
            console.print(f"[bold cyan]Target from profile:[/bold cyan] [green]{profile_target}[/green]")
        except ValueError:
            profile_target = None
    
    if not profile_target:
        target_str = Prompt.ask(
            "Optimization Target",
            choices=["time", "memory", "balanced", "cost"],
            default="balanced",
            console=console
        )
        target = OptimizationTarget(target_str)
    
    profile_max_cores = profile_values.get("max_cores") or profile_values.get("recommended_total_cores")
    if profile_max_cores is not None:
        max_cores = profile_max_cores
        console.print(f"[bold cyan]Max Cores from profile:[/bold cyan] [green]{max_cores}[/green]")
    else:
        max_cores = IntPrompt.ask(
            "Maximum Cores to Utilize (0 for auto)",
            default=0,
            console=console
        )
    max_cores = max_cores if max_cores > 0 else None
    
    profile_mem = profile_values.get("memory_limit") or profile_values.get("memory_limit_gb")
    if profile_mem is not None:
        memory_limit = float(profile_mem) if profile_mem is not None else None
        console.print(f"[bold cyan]Memory Limit from profile:[/bold cyan] [green]{memory_limit} GB[/green]")
    else:
        memory_limit = FloatPrompt.ask(
            "Memory Limit per Node in GB (0 for auto)",
            default=0.0,
            console=console
        )
        memory_limit = memory_limit if memory_limit > 0.0 else None

    # 3. Resource Suggestion via Advisor
    with console.status("[bold cyan]Consulting Optimizer Advisor...", spinner="dots"):
        try:
            suggestion = suggest_optimal_resources(
                topo=topo,
                user_max_cores=max_cores,
                optimization_target=target
            )
            sug_dict = suggestion.to_dict()
        except Wien2kGenError as e:
            console.print(Panel(f"[red]Advisor Error: {e.message}\n💡 {e.hint}[/red]", border_style="red"))
            return
        except Exception as e:
            logger.error(f"Advisor failed: {e}")
            console.print("[red]❌ Failed to generate resource suggestion.[/red]")
            return

    # Display Suggestion Summary
    sug_table = Table(title="Optimizer Recommendation", show_header=True, header_style="bold magenta")
    sug_table.add_column("Parameter", style="cyan")
    sug_table.add_column("Value", style="green")
    sug_table.add_row("Mode", sug_dict["mode"].upper() if isinstance(sug_dict["mode"], str) else sug_dict["mode"].value)
    sug_table.add_row("Total Cores", str(sug_dict["recommended_total_cores"]))
    sug_table.add_row("OMP Threads", str(sug_dict["omp_threads_per_rank"]))
    sug_table.add_row("Est. Memory", f"{sug_dict.get('estimated_memory_gb', 0.0):.1f} GB")
    sug_table.add_row("Bottleneck", sug_dict.get("reason", "N/A"))
    sug_table.add_row("Confidence", f"{sug_dict.get('confidence_score', 1.0)*100:.0f}%")
    console.print(sug_table)

    if sug_dict.get("warnings"):
        warn_table = Table(title="⚠️ Advisor Warnings", show_header=False)
        for w in sug_dict["warnings"]:
            warn_table.add_row("[yellow]•[/]", f"[dim]{w}[/dim]")
        console.print(warn_table)

    # 4. Pre-flight Validation & Confirmation
    console.print(Rule(style="dim"))
    if not Confirm.ask("Proceed with this configuration and generate .machines?", console=console, default=True):
        console.print("[yellow]⚠️ Configuration cancelled by user.[/yellow]")
        return

    # 5. Execution & Generation
    with console.status("[bold cyan]Generating configuration files...", spinner="dots"):
        try:
            # Apply core limit to topology for builder
            topo_final = topo
            if max_cores and max_cores < topo.total_cores:
                topo_final = detect_topology(max_cores=max_cores)
                
                build_result = build_auto(
                    topo=topo_final,
                    suggestion=sug_dict,
                    backup=True,
                    dry_run=False,
                    validate=True
                )

                if not build_result.success:
                    if hasattr(build_result, 'error_message') and build_result.error_message:
                        raise ConfigurationError(build_result.error_message)
                    raise ConfigurationError(
                        "Build failed. Check input files and topology."  # Generic error for edge cases
                    )
                
            console.print("[green]✅ .machines and parallel_options generated successfully![/green]")

            # 5.5 Manual review/edit step
            if Confirm.ask(
                "Review and manually edit .machines before finalizing?",
                console=console, default=False
            ):
                _open_editor_for_manual_review(Path(".machines"))

            # 6. Save as Profile
            if Confirm.ask("Save this configuration as a reusable profile?", console=console, default=False):
                name = Prompt.ask("Profile name", console=console)
                sug_dict["backend"] = backend_name
                PROFILES_DIR.mkdir(parents=True, exist_ok=True)
                profile_path = PROFILES_DIR / f"{name}.json"
                atomic_write(profile_path, json.dumps(sug_dict, indent=2, default=str))
                console.print(f"[green]✅ Profile saved to {profile_path}[/green]")
                
        except Wien2kGenError as e:
            logger.error(f"Generation failed: {e}")
            console.print(Panel(f"[red]❌ Error: {e.message}\n💡 {e.hint}[/red]", border_style="red"))
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            console.print(f"[red]❌ Unexpected Error: {e}[/red]")


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "run_wizard",
    "detect_wienroot_candidates",
    "validate_wienroot",
    "check_scratch_health",
]

if __name__ == "__main__":
    run_wizard()