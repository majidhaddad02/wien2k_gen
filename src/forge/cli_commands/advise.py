from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def _build_advice_dict(nmat, kpoints, atoms, cores, arch, mem_bw,
                       peak_gflops, numa_nodes, topo, target):
    from ..optimizer.advisor import (
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
                       peak_gflops, numa_nodes, topo, target):
    console = get_console()
    import os as _os

    from ..optimizer.advisor import (
        estimate_amdahl_saturation,
        roofline_crossover_analysis,
    )
    _os.environ.setdefault("WIEN2K_NO_DETECT", "1")
    roofline = roofline_crossover_analysis(
        {"mem_bw_gb_s": mem_bw, "arch": arch, "peak_fp64_gflops": peak_gflops},
        oi=0.15, target_backend="wien2k_lapw1",
    )
    amdahl = estimate_amdahl_saturation(kpoints, nmat, atoms, cores)

    console.print(Panel(
        f"[cyan bold]WIEN2k Optimization Advisor[/]\n"
        f"System: {nmat}\u00d7{nmat} matrix, {kpoints} k-points, {atoms} atoms | "
        f"[dim]{arch} \u2022 {cores} cores \u2022 {numa_nodes} NUMA nodes \u2022 {mem_bw:.0f} GB/s mem[/]\n"
        f"[dim]Target: optimize for [bold]{target}[/][/]",
        border_style="cyan",
    ))

    bottleneck = None
    if roofline["regime"] == "memory_bound" and mem_bw < 100:
        label = "Memory Bandwidth"
        msg = "LAPW1 is memory-hungry \u2014 extra MPI ranks won't help, use OpenMP instead"
        bottleneck = (f"[red]{label}[/]", "red", msg)
    elif isinstance(amdahl, dict) and amdahl.get("saturation_cores", cores) < max(cores * 0.6, 1):
        sat = amdahl["saturation_cores"]
        label = "Amdahl Saturation"
        msg = f"More than {sat} cores won't improve performance (Amdahl's Law)"
        bottleneck = (f"[yellow]{label}[/]", "yellow", msg)

    if bottleneck:
        console.print(Panel(bottleneck[2], title=bottleneck[0], border_style=bottleneck[1]))

    table = Table(title="Performance Analysis", border_style="blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")
    table.add_column("What This Means", style="dim")

    regime_en = "Memory bottleneck \u2014 extra MPI won't help" if roofline["regime"] == "memory_bound" else "CPU-limited \u2014 more MPI will help"
    table.add_row(
        "Roofline Regime",
        f"[{'red' if roofline['regime'] == 'memory_bound' else 'green'}]{roofline['regime'].replace('_', ' ').title()}[/]",
        regime_en,
    )
    table.add_row(
        "Roofline Efficiency",
        f"{roofline['efficiency_pct']:.0f}%", "",
    )
    table.add_row(
        "Optimal Cores (Roofline)",
        str(roofline["optimal_cores"]),
        "Best efficiency at this core count",
    )

    if isinstance(amdahl, dict):
        sat_cores = amdahl.get("saturation_cores", cores)
        eff = amdahl.get("efficiency_pct", 100.0)
        table.add_row(
            "Amdahl Saturation",
            str(sat_cores),
            "Beyond this, speedup plateaus",
        )
        table.add_row(
            "Amdahl Efficiency",
            f"{eff:.0f}%", "",
        )
    else:
        sat_cores = cores
        eff = 100.0

    console.print(table)

    rec_table = Table(title="Recommendations", border_style="green")
    rec_table.add_column("#", style="dim")
    rec_table.add_column("Action", style="cyan bold")
    rec_table.add_column("Why", style="dim")
    rec_table.add_column("Impact", style="green")

    counter = 1
    if roofline["regime"] == "memory_bound":
        rec_table.add_row(str(counter),
            "Increase OpenMP threads, reduce MPI ranks",
            "Memory bandwidth saturated",
            "HIGH")
        counter += 1
        if omp := (cores // max(1, numa_nodes)):
            rec_table.add_row(str(counter),
                f"export OMP_NUM_THREADS={omp}",
                "One MPI rank per NUMA node",
                "HIGH")
            counter += 1
    elif sat_cores < cores * 0.7:
        rec_table.add_row(str(counter),
            f"Limit to {sat_cores} cores",
            "Amdahl's Law \u2014 more is wasted",
            "MEDIUM")
        counter += 1

    if nmat > 5000:
        rec_table.add_row(str(counter),
            "lapw2_vector_split: 1",
            "Large matrix I/O reduction",
            "MEDIUM")
        counter += 1

    if kpoints > 1 and kpoints % cores != 0:
        rec_table.add_row(str(counter),
            "Set k-points to a multiple of core count",
            "Uneven load distribution",
            "MEDIUM")
        counter += 1

    console.print(rec_table)
    console.print("\n[dim]\u279b Run [bold]forge optimize --simulated[/] to auto-tune RKMAX/KPPRA/mixing[/]")
    console.print("[dim]\u279b Run [bold]forge generate[/] to produce optimized .machines[/]")


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("advise", help="Get intelligent optimization advice (Roofline, Amdahl, NUMA)")
    p.add_argument("--case", type=str, default="case", help="WIEN2k case name for problem-aware advice")
    p.add_argument("--nmat", type=int, default=None, help="Override matrix size (detected from case if not given)")
    p.add_argument("--kpoints", type=int, default=None, help="Override k-point count")
    p.add_argument("--cores", type=int, default=None, help="Target total cores")
    p.add_argument("--target", type=str, choices=["time", "energy", "cost", "balanced"], default="time", help="Optimization goal (default: time)")
    p.add_argument("--json", action="store_true", help="Export advice as JSON")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    console = get_console()
    from ..core.case_parser import CaseFileParser
    from ..core.hardware import (
        calculate_peak_fp64_gflops,
        get_cpu_architecture,
        get_memory_bandwidth_gb_s,
        get_numa_node_count,
        get_physical_cores,
    )

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

    from ..core.scheduler import detect as detect_topology
    topo = detect_topology(max_cores=cores)

    if getattr(args, "json", False):
        import json as _json
        result = _build_advice_dict(
            nmat=nmat, kpoints=kpoints, atoms=atoms, cores=cores,
            arch=arch, mem_bw=mem_bw, peak_gflops=peak_gflops,
            numa_nodes=numa_nodes, topo=topo, target=target,
        )
        console.print_json(_json.dumps(result))
        return result

    _print_advice_rich(
        nmat=nmat, kpoints=kpoints, atoms=atoms, cores=cores,
        arch=arch, mem_bw=mem_bw, peak_gflops=peak_gflops,
        numa_nodes=numa_nodes, topo=topo, target=target,
    )

    return {"status": "advice_displayed"}


register_command("advise", handle)
