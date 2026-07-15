from __future__ import annotations

import argparse
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("predict", help="Predict SCF convergence time before calculation")
    p.add_argument("--case", type=str, default="case", help="WIEN2k case name")
    p.add_argument("--struct", type=str, default=None, help="Path to .struct file")
    p.add_argument("--no-history", action="store_true", help="Skip ML training from history")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    try:
        from ..optimizer.ml_predict import predict_convergence
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
    table.add_row("Estimated SCF time", f"{pred.estimated_time_hours:.1f} \u00b1 {pred.time_uncertainty_hours:.1f} hours")
    table.add_row("Convergence probability", f"{pred.convergence_probability * 100:.0f}%")
    table.add_row("Estimated SCF cycles", str(pred.estimated_cycles))
    table.add_row("Recommended mixing", f"\u03b2 = {pred.recommended_mixing:.3f}")
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


register_command("predict", handle)
