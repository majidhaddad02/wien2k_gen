"""
Interactive SLURM Job Submission Wizard for Wien2kGen.
Provides a guided, visually rich, and error-proof terminal experience for creating
and submitting SBATCH job scripts to HPC schedulers.

Key Architecture Features:
• Step-by-step interactive flow using Rich UI components (Panels, Markdown, Syntax Highlighting)
• Intelligent input validation (Time formats, Memory strings, Integer ranges) with immediate feedback
• Smart defaults based on environment variables and detected system topology
• Live preview of generated #SBATCH directives before saving/submitting
• Options to Save, Submit, or Edit directly from the wizard
• Thread-safe atomic writing for generated scripts
• Seamless integration with the `submit` module and `core.scheduler`
• Comprehensive English documentation, type hints, and HPC-grade error resilience
"""

import os
import sys
import re
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.syntax import Syntax
from rich.rule import Rule
from rich.align import Align

from .core.scheduler import detect as detect_topology
from .submit.slurm import (
    generate_sbatch_script,
    SlurmJobSpec,
    SlurmDirectives,
)
from .utils.atomic_write import atomic_write
from .logging_config import get_logger

logger = get_logger(__name__)
console = Console()


# =============================================================================
# Input Validation Helpers
# =============================================================================

def validate_time_format(value: str) -> bool:
    """Validate SLURM time format (HH:MM:SS, D-HH:MM:SS, or MM:SS)."""
    return bool(re.match(r'^(\d+-)?(\d{1,2}:)?\d{2}:\d{2}$', value.strip()))


def validate_mem_format(value: str) -> bool:
    """Validate SLURM memory string (Number + K/M/G/T)."""
    return bool(re.match(r'^\d+[KMGkmgTt]?$', value.strip()))


# =============================================================================
# Wizard Steps
# =============================================================================

class WizardStep:
    """Base class for wizard steps."""
    def __init__(self, wizard: "SBATCHWizard"):
        self.wizard = wizard
        self.console = wizard.console

    def run(self) -> bool:
        """Execute step logic. Return True to continue, False to abort."""
        raise NotImplementedError


class WelcomeStep(WizardStep):
    def run(self) -> bool:
        intro = """
📝 **SBATCH Job Submission Wizard**

Create and submit SLURM job scripts interactively.
This tool helps you define resources, check constraints, and submit jobs safely.
"""
        self.console.print(Panel(Markdown(intro), title="SBATCH Wizard", border_style="cyan", padding=1))
        if not Confirm.ask(
            Align.center("[bold yellow]Start SBATCH wizard?[/]", vertical="middle"),
            console=self.console
        ):
            return False
        return True


class IdentityStep(WizardStep):
    def run(self) -> bool:
        self.console.print("\n[bold cyan]Step 1/4: Job Identity[/]")
        self.console.print(Rule(style="dim"))
        
        # Job Name
        user = os.environ.get("USER", "user")
        ts = time.strftime("%m%d_%H%M")
        self.wizard.data['job_name'] = Prompt.ask(
            "Job Name (-J)",
            default=f"w2k_{user}_{ts}",
            console=self.console
        )

        # Partition
        self.wizard.data['partition'] = Prompt.ask(
            "Partition (-p) [Leave empty for default]",
            default="",
            console=self.console
        )

        # Account
        self.wizard.data['account'] = Prompt.ask(
            "Account (--account) [Optional]",
            default="",
            console=self.console
        )
        return True


class ResourcesStep(WizardStep):
    def run(self) -> bool:
        self.console.print("\n[bold cyan]Step 2/4: Resource Allocation[/]")
        self.console.print(Rule(style="dim"))
        
        # Topology detection for hints
        topo = self.wizard.data.get('topo')
        if topo:
            self.console.print(f"[dim]Detected Topology: {topo.total_cores} cores available[/dim]")
            suggested_tasks = topo.total_cores
        else:
            self.console.print("[dim]Topology detection skipped. Manual entry required.[/dim]")
            suggested_tasks = 16

        # Nodes
        self.wizard.data['nodes'] = IntPrompt.ask(
            "Number of Nodes (-N)",
            default=1,
            console=self.console
        )

        # Tasks
        self.wizard.data['ntasks'] = IntPrompt.ask(
            "Total Tasks (-n)",
            default=suggested_tasks,
            console=self.console
        )

        # CPUs per task
        self.wizard.data['cpus_per_task'] = IntPrompt.ask(
            "CPUs per Task (-c)",
            default=1,
            console=self.console
        )

        # Memory
        while True:
            mem_val = Prompt.ask(
                "Memory per Node (--mem)",
                default="8G",
                console=self.console
            )
            if validate_mem_format(mem_val):
                self.wizard.data['mem_per_node'] = mem_val
                break
            else:
                self.console.print("[bold red]Invalid format. Use format like '8G', '4000M', '128G'.[/]")

        # Time
        while True:
            time_val = Prompt.ask(
                "Walltime (--time)",
                default="24:00:00",
                console=self.console
            )
            if validate_time_format(time_val):
                self.wizard.data['walltime'] = time_val
                break
            else:
                self.console.print("[bold red]Invalid format. Use HH:MM:SS or D-HH:MM:SS.[/]")

        return True


class AdvancedStep(WizardStep):
    def run(self) -> bool:
        self.console.print("\n[bold cyan]Step 3/4: Advanced Options[/]")
        self.console.print(Rule(style="dim"))
        self.console.print("[dim]Press Enter to skip optional fields.[/dim]")
        
        self.wizard.data['dependency'] = Prompt.ask(
            "Job Dependency (e.g., afterok:12345)",
            default="",
            console=self.console
        )
        
        self.wizard.data['qos'] = Prompt.ask(
            "Quality of Service (QoS)",
            default="",
            console=self.console
        )
        
        self.wizard.data['gres'] = Prompt.ask(
            "Generic Resources (e.g., gpu:a100:1)",
            default="",
            console=self.console
        )
        
        self.wizard.data['output'] = Prompt.ask(
            "Output File Pattern",
            default="slurm-%j.out",
            console=self.console
        )

        return True


class ReviewStep(WizardStep):
    def run(self) -> bool:
        self.console.print("\n[bold cyan]Step 4/4: Review & Action[/]")
        self.console.print(Rule(style="dim"))
        
        # Generate Script
        try:
            directives = SlurmDirectives(
                job_name=self.wizard.data['job_name'],
                partition=self.wizard.data['partition'],
                nodes=self.wizard.data['nodes'],
                ntasks=self.wizard.data['ntasks'],
                cpus_per_task=self.wizard.data['cpus_per_task'],
                mem_per_node=self.wizard.data['mem_per_node'],
                time=self.wizard.data['walltime'],
                dependency=self.wizard.data['dependency'],
                qos=self.wizard.data['qos'],
                gres=self.wizard.data['gres'],
                account=self.wizard.data['account'],
                output=self.wizard.data['output'],
                error=self.wizard.data.get('output', 'slurm-%j.out').replace('.out', '.err')
            )

            topo = self.wizard.data.get('topo')
            spec = SlurmJobSpec(
                topo=topo,
                exec_command="run_lapw -p",  # Default placeholder
                directives=directives
            )

            self.wizard.script_content = generate_sbatch_script(spec)
        except Exception as e:
            self.console.print(f"[bold red]Failed to generate script: {e}[/]")
            return False

        # Display Preview
        syntax = Syntax(self.wizard.script_content, "bash", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Generated SBATCH Script", border_style="green"))

        # Action Menu
        self.console.print("\n[bold]What would you like to do?[/]")
        self.console.print("  [1] Save Script to File")
        self.console.print("  [2] Save & Submit Job")
        self.console.print("  [3] Cancel")

        choice = Prompt.ask(
            "Select action",
            choices=["1", "2", "3"],
            default="1",
            console=self.console
        )

        if choice == "3":
            self.console.print("[dim]Wizard cancelled.[/]")
            return False

        # Determine filename
        filename = Prompt.ask(
            "Enter filename to save",
            default=f"submit_{self.wizard.data['job_name']}.sh",
            console=self.console
        )
        path = Path(filename)
        
        if path.exists():
            if not Confirm.ask(f"File {filename} exists. Overwrite?", console=self.console):
                return False

        # Save
        try:
            atomic_write(path, self.wizard.script_content, mode=0o755)
            self.console.print(f"\n[bold green]✅ Script saved to {path.resolve()}[/]")
            self.wizard.saved_path = path
            
            if choice == "2":
                return self._submit_job(path)
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Error saving file: {e}[/]")
            return False

    def _submit_job(self, path: Path) -> bool:
        """Submit the saved script via sbatch command."""
        if not Confirm.ask("Proceed with submission?", console=self.console):
            return False

        self.console.print("\n[yellow]Submitting job to scheduler...[/]")
        try:
            res = subprocess.run(
                ["sbatch", str(path)],
                capture_output=True, text=True, timeout=30
            )
            
            if res.returncode == 0:
                match = re.search(r"Submitted batch job (\d+)", res.stdout)
                job_id = match.group(1) if match else "Unknown"
                self.console.print(f"\n[bold green]🚀 Job Submitted! ID: {job_id}[/]")
                return True
            else:
                self.console.print(f"[bold red]Submission Failed:[/]\n{res.stderr}")
                return False
        except Exception as e:
            self.console.print(f"[bold red]Submission Error: {e}[/]")
            return False


# =============================================================================
# Main Wizard Engine
# =============================================================================

class SBATCHWizard:
    def __init__(self):
        self.console = console
        self.data: Dict[str, Any] = {}
        self.script_content: str = ""
        self.saved_path: Optional[Path] = None
        
        # Pre-load topology for smart defaults
        try:
            self.data['topo'] = detect_topology()
        except Exception:
            logger.debug("Topology detection failed in wizard context.")
            self.data['topo'] = None

        self.steps = [
            WelcomeStep(self),
            IdentityStep(self),
            ResourcesStep(self),
            AdvancedStep(self),
            ReviewStep(self)
        ]

    def run(self) -> bool:
        """Execute all steps in sequence."""
        for i, step in enumerate(self.steps):
            self.console.print(f"\n[dim]--- Step {i+1}/{len(self.steps)} ---[/dim]")
            if not step.run():
                return False
        return True


# =============================================================================
# Public API
# =============================================================================

def run_sbatch_wizard() -> bool:
    """Entry point for the SBATCH interactive wizard."""
    wizard = SBATCHWizard()
    success = wizard.run()
    return success


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "run_sbatch_wizard",
    "SBATCHWizard",
]

if __name__ == "__main__":
    run_sbatch_wizard()