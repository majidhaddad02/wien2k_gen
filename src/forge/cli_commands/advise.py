from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command

_SIMPLE_LANGUAGE = {
    "memory_bound": "\u062d\u0627\u0641\u0638\u0647 \u06a9\u0645 \u0622\u0648\u0631\u062f\u06cc \u2014 MPI \u0628\u06cc\u0634\u062a\u0631 \u0627\u0636\u0627\u0641\u0647 \u0646\u06a9\u0646\u060c OpenMP \u0628\u062f\u0647",
    "compute_bound": "\u067e\u0631\u062f\u0627\u0632\u0646\u062f\u0647 \u0645\u062d\u062f\u0648\u062f\u06cc\u062a \u062f\u0627\u0631\u0647 \u2014 MPI \u0628\u06cc\u0634\u062a\u0631 \u062c\u0648\u0627\u0628 \u0645\u06cc\u062f\u0647",
    "rkmax": "\u0627\u0646\u062f\u0627\u0632\u0647\u0654 \u0645\u062c\u0645\u0648\u0639\u0647\u0654 \u067e\u0627\u06cc\u0647 (\u0647\u0631\u0686\u06cc \u0628\u06cc\u0634\u062a\u0631 = \u062f\u0642\u06cc\u0642\u200c\u062a\u0631 \u0648\u0644\u06cc \u06a9\u0646\u062f\u062a\u0631)",
    "kppra": "\u062a\u0639\u062f\u0627\u062f k-point (\u0646\u0642\u0627\u0637 \u0646\u0645\u0648\u0646\u0647\u200c\u0628\u0631\u062f\u0627\u0631\u06cc \u062f\u0631 \u0641\u0636\u0627\u06cc \u0627\u0646\u0631\u0698\u06cc)",
    "mixing": "\u0633\u0631\u0639\u062a \u0647\u0645\u06af\u0631\u0627\u06cc\u06cc SCF (\u06a9\u0645\u062a\u0631 = \u067e\u0627\u06cc\u062f\u0627\u0631\u062a\u0631 \u0648\u0644\u06cc \u06a9\u0646\u062f\u062a\u0631)",
    "granularity": "\u062a\u0639\u062f\u0627\u062f k-point \u062f\u0631 \u0647\u0631 \u06af\u0631\u0648\u0647 \u0645\u0648\u0627\u0632\u06cc",
    "elpa": "\u06a9\u062a\u0627\u0628\u062e\u0627\u0646\u0647\u0654 \u0642\u0637\u0631\u06cc\u200c\u0633\u0627\u0632\u06cc \u0633\u0631\u06cc\u0639 \u0628\u0631\u0627\u06cc \u0645\u0627\u062a\u0631\u06cc\u0633\u200c\u0647\u0627\u06cc \u0628\u0632\u0631\u06af",
    "nuam": "\u0645\u0639\u0645\u0627\u0631\u06cc \u062d\u0627\u0641\u0638\u0647\u0654 \u063a\u06cc\u0631\u06cc\u06a9\u0646\u0648\u0627\u062e\u062a \u2014 \u062f\u0633\u062a\u0631\u0633\u06cc \u0628\u0647 \u062d\u0627\u0641\u0638\u0647\u0654 \u0646\u0632\u062f\u06cc\u06a9 \u0633\u0631\u06cc\u0639\u200c\u062a\u0631\u0647",
}


def _build_plain_language():
    return _SIMPLE_LANGUAGE


def _build_advice_dict(nmat, kpoints, atoms, cores, arch, mem_bw,
                       peak_gflops, numa_nodes, topo, plain, target):
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
                       peak_gflops, numa_nodes, topo, plain, target):
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

    _lang = _build_plain_language() if plain else {}

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
        msg = ("\u062d\u0627\u0641\u0638\u0647 \u06a9\u0645 \u0622\u0648\u0631\u062f\u06cc \u2014 MPI \u0628\u06cc\u0634\u062a\u0631 \u0627\u0636\u0627\u0641\u0647 \u0646\u06a9\u0646\u060c OpenMP \u0628\u062f\u0647" if plain
               else "LAPW1 is memory-hungry \u2014 extra MPI ranks won't help, use OpenMP instead")
        bottleneck = (f"[red]{label}[/]", "red", msg)
    elif isinstance(amdahl, dict) and amdahl.get("saturation_cores", cores) < max(cores * 0.6, 1):
        sat = amdahl["saturation_cores"]
        label = "Amdahl Saturation"
        msg = (f"\u0642\u0627\u0646\u0648\u0646 \u0622\u0645\u0627\u0644 \u0645\u06cc\u06af\u0647 \u0628\u06cc\u0634 \u0627\u0632 {sat} \u0647\u0633\u062a\u0647 \u0628\u06cc\u200c\u0641\u0627\u06cc\u062f\u0647\u200c\u0633\u062a" if plain
               else f"More than {sat} cores won't improve performance (Amdahl's Law)")
        bottleneck = (f"[yellow]{label}[/]", "yellow", msg)

    if bottleneck:
        console.print(Panel(bottleneck[2], title=bottleneck[0], border_style=bottleneck[1]))

    table = Table(title=("\u062a\u062d\u0644\u06cc\u0644 \u0639\u0645\u0644\u06a9\u0631\u062f" if plain else "Performance Analysis"), border_style="blue")
    table.add_column(("\u0645\u0639\u06cc\u0627\u0631" if plain else "Metric"), style="cyan")
    table.add_column(("\u0645\u0642\u062f\u0627\u0631" if plain else "Value"), style="green bold")
    table.add_column(("\u06cc\u0639\u0646\u06cc \u0686\u0647" if plain else "What This Means"), style="dim")

    regime_fa = "\u0645\u062d\u062f\u0648\u062f \u0628\u0647 \u062d\u0627\u0641\u0638\u0647 \u2014 MPI \u0627\u0636\u0627\u0641\u06cc \u06a9\u0645\u06a9\u06cc \u0646\u0645\u06cc\u200c\u06a9\u0646\u062f" if roofline["regime"] == "memory_bound" else "\u0645\u062d\u062f\u0648\u062f \u0628\u0647 \u067e\u0631\u062f\u0627\u0632\u0646\u062f\u0647 \u2014 MPI \u0628\u06cc\u0634\u062a\u0631 \u062c\u0648\u0627\u0628 \u0645\u06cc\u200c\u062f\u0647\u062f"
    regime_en = "Memory bottleneck \u2014 extra MPI won't help" if roofline["regime"] == "memory_bound" else "CPU-limited \u2014 more MPI will help"
    table.add_row(
        ("\u0646\u0648\u0639 \u06af\u0644\u0648\u06af\u0627\u0647" if plain else "Roofline Regime"),
        f"[{'red' if roofline['regime'] == 'memory_bound' else 'green'}]{roofline['regime'].replace('_', ' ').title()}[/]",
        (regime_fa if plain else regime_en),
    )
    table.add_row(
        ("\u06a9\u0627\u0631\u0627\u06cc\u06cc Roofline" if plain else "Roofline Efficiency"),
        f"{roofline['efficiency_pct']:.0f}%", "",
    )
    table.add_row(
        ("\u062a\u0639\u062f\u0627\u062f \u0647\u0633\u062a\u0647\u0654 \u0628\u0647\u06cc\u0646\u0647" if plain else "Optimal Cores (Roofline)"),
        str(roofline["optimal_cores"]),
        ("\u0628\u0627 \u0627\u06cc\u0646 \u062a\u0639\u062f\u0627\u062f \u0628\u0647\u062a\u0631\u06cc\u0646 \u06a9\u0627\u0631\u0627\u06cc\u06cc \u0631\u0648 \u0645\u06cc\u200c\u06af\u06cc\u0631\u06cc" if plain else "Best efficiency at this core count"),
    )

    if isinstance(amdahl, dict):
        sat_cores = amdahl.get("saturation_cores", cores)
        eff = amdahl.get("efficiency_pct", 100.0)
        table.add_row(
            ("\u0627\u0634\u0628\u0627\u0639 \u0622\u0645\u0627\u0644" if plain else "Amdahl Saturation"),
            str(sat_cores),
            ("\u0641\u0631\u0627\u062a\u0631 \u0627\u0632 \u0627\u06cc\u0646 \u062a\u0639\u062f\u0627\u062f\u060c \u0628\u0647\u0628\u0648\u062f \u0645\u062d\u0633\u0648\u0633\u06cc \u0646\u062f\u0627\u0631\u06cc" if plain else "Beyond this, speedup plateaus"),
        )
        table.add_row(
            ("\u06a9\u0627\u0631\u0627\u06cc\u06cc \u0622\u0645\u0627\u0644" if plain else "Amdahl Efficiency"),
            f"{eff:.0f}%", "",
        )
    else:
        sat_cores = cores
        eff = 100.0

    console.print(table)

    rec_table = Table(title=("\u067e\u06cc\u0634\u0646\u0647\u0627\u062f\u0627\u062a" if plain else "Recommendations"), border_style="green")
    rec_table.add_column("#", style="dim")
    rec_table.add_column(("\u0627\u0642\u062f\u0627\u0645" if plain else "Action"), style="cyan bold")
    rec_table.add_column(("\u062f\u0644\u06cc\u0644" if plain else "Why"), style="dim")
    rec_table.add_column(("\u062a\u0623\u062b\u06cc\u0631" if plain else "Impact"), style="green")

    counter = 1
    if roofline["regime"] == "memory_bound":
        rec_table.add_row(str(counter),
            ("OpenMP \u0631\u0648 \u0632\u06cc\u0627\u062f \u06a9\u0646\u060c MPI \u06a9\u0645 \u06a9\u0646" if plain else "Increase OpenMP threads, reduce MPI ranks"),
            ("\u067e\u0647\u0646\u0627\u06cc \u0628\u0627\u0646\u062f \u062d\u0627\u0641\u0638\u0647 \u0627\u0634\u0628\u0627\u0639 \u0634\u062f\u0647" if plain else "Memory bandwidth saturated"),
            ("\u0628\u0627\u0644\u0627" if plain else "HIGH"))
        counter += 1
        if omp := (cores // max(1, numa_nodes)):
            rec_table.add_row(str(counter),
                f"export OMP_NUM_THREADS={omp}",
                ("\u0647\u0631 \u0631\u062a\u0628\u0647\u0654 MPI \u0631\u0648\u06cc \u06cc\u06a9 \u06af\u0631\u0647\u0654 NUMA" if plain else "One MPI rank per NUMA node"),
                ("\u0628\u0627\u0644\u0627" if plain else "HIGH"))
            counter += 1
    elif sat_cores < cores * 0.7:
        rec_table.add_row(str(counter),
            (f"\u062d\u062f\u0627\u06a9\u062b\u0631 {sat_cores} \u0647\u0633\u062a\u0647 \u0627\u0633\u062a\u0641\u0627\u062f\u0647 \u06a9\u0646" if plain else f"Limit to {sat_cores} cores"),
            ("\u0642\u0627\u0646\u0648\u0646 \u0622\u0645\u0627\u0644 \u2014 \u0628\u06cc\u0634 \u0627\u0632 \u0627\u06cc\u0646 \u0647\u062f\u0631 \u0631\u0641\u062a\u0647" if plain else "Amdahl's Law \u2014 more is wasted"),
            ("\u0645\u062a\u0648\u0633\u0637" if plain else "MEDIUM"))
        counter += 1

    if nmat > 5000:
        rec_table.add_row(str(counter),
            "lapw2_vector_split: 1" if not plain else "lapw2_vector_split \u0631\u0648 \u0641\u0639\u0627\u0644 \u06a9\u0646",
            ("\u06a9\u0627\u0647\u0634 I/O \u0628\u0631\u0627\u06cc \u0645\u0627\u062a\u0631\u06cc\u0633 \u0628\u0632\u0631\u06af" if plain else "Large matrix I/O reduction"),
            ("\u0645\u062a\u0648\u0633\u0637" if plain else "MEDIUM"))
        counter += 1

    if kpoints > 1 and kpoints % cores != 0:
        rec_table.add_row(str(counter),
            ("\u062a\u0639\u062f\u0627\u062f k-point \u0631\u0648 \u0645\u0636\u0631\u0628\u06cc \u0627\u0632 \u0647\u0633\u062a\u0647\u200c\u0647\u0627 \u06a9\u0646" if plain else "Set k-points to a multiple of core count"),
            ("\u062a\u0648\u0632\u06cc\u0639 \u0646\u0627\u0645\u062a\u0648\u0627\u0632\u0646 \u0628\u0627\u0631" if plain else "Uneven load distribution"),
            ("\u0645\u062a\u0648\u0633\u0637" if plain else "MEDIUM"))
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
    p.add_argument("--plain", action="store_true", help="Show advice in simple language (non-expert mode)")
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

    from ..core.scheduler import detect as detect_topology
    topo = detect_topology(max_cores=cores)

    if getattr(args, "json", False):
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


register_command("advise", handle)
