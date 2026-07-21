from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.table import Table

from ..config import AppConfig
from ._utils import get_console
from .base import register_command


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("hardware", help="Show hardware info and parallelization recommendations")
    p.add_argument("--recommend", "-r", action="store_true", help="Show NUMA/hybrid/IO optimization advice")
    p.add_argument("--case", type=str, default=None, help="Case name for problem-specific recommendations")


def handle(args: argparse.Namespace, cfg: AppConfig) -> dict[str, Any]:
    from ..core.hardware import (
        get_cpu_architecture,
    )
    from ..core.hardware.wrapper import (
        get_cpu_generation,
        get_numa_node_count,
        get_physical_cores,
        get_scratch_filesystem_type,
        get_system_type,
        get_total_mem_kb,
    )
    from ..core.topology import Topology
    from ..optimizer.parallel import (
        recommend_gmax,
        recommend_io_strategy,
        recommend_lapw0_strategy,
        recommend_mkl_threading,
        recommend_numa_strategy,
        recommend_rkmax,
    )

    console = get_console()

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
                from ..core.case_parser import CaseFileParser
                parser = CaseFileParser(Path(args.case) if Path(args.case).exists() else None)
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


register_command("hardware", handle)
