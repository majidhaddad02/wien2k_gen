from __future__ import annotations

import argparse
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "calibrate",
        help="Re-measure and cache hardware roofline (bandwidth, FLOPS, cache)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print measured roofline data as JSON",
    )


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    from ..core.perf_counters import (
        _CALIBRATION_NOTICE,
        _PERF_CACHE_FILE,
        get_real_roofline_data,
        invalidate_perf_cache,
    )

    console = get_console()
    invalidate_perf_cache()
    console.print(f"[dim]{_CALIBRATION_NOTICE}[/dim]")
    data = get_real_roofline_data(use_cache=True)

    if getattr(args, "json_output", False) or getattr(args, "json", False):
        import json as _json
        console.print_json(_json.dumps(data, default=str))
        return {"status": "calibrated", "data": data, "cache": str(_PERF_CACHE_FILE)}

    table = Table(title="Hardware Roofline Calibration", border_style="cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Peak FP64", f"{data.get('peak_flops_gflops', 0):.2f} GFLOPS")
    table.add_row("Sustained BW", f"{data.get('sustained_bw_gb_s', 0):.2f} GB/s")
    table.add_row("Tool", str(data.get("tool_used", "fallback")))
    cache_bw = data.get("cache_bandwidth") or {}
    if cache_bw:
        table.add_row(
            "Cache BW L1/L2/L3",
            f"{cache_bw.get('l1', 0):.1f} / {cache_bw.get('l2', 0):.1f} / {cache_bw.get('l3', 0):.1f} GB/s",
        )
    table.add_row("Cache file", str(_PERF_CACHE_FILE))
    console.print(table)
    console.print(Panel(
        "Next [bold]forge generate[/]/[bold]advise[/] reuse this cache. "
        "Pass [bold]--recalibrate[/] or re-run this command after BIOS/firmware changes.",
        border_style="dim",
    ))
    return {"status": "calibrated", "data": data, "cache": str(_PERF_CACHE_FILE)}


register_command("calibrate", handle)
