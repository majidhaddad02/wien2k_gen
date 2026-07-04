"""
Benchmark Report Generator for WIEN2k Gen.

Produces speedup/efficiency charts and scaling analysis reports from benchmark
data collected by the synthetic and real benchmark runners. Supports both
matplotlib-based PDF/PNG output and plain-text summaries for headless HPC
environments.

References:
    Hager & Wellein 2010: Introduction to High Performance Computing
    Gustafson 1988: Reevaluating Amdahl's Law (CACM 31(5), 532-533)
"""

import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..logging_config import get_logger

logger = get_logger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.info(
        "matplotlib not available – benchmark reports will be text-only. "
        "Install matplotlib for PDF/PNG charts."
    )


@dataclass
class ScalingDataPoint:
    """Single data point in a strong/weak scaling series."""
    node_count: int
    ranks: int
    threads_per_rank: int
    runtime_seconds: float
    walltime_seconds: float
    efficiency_percent: float = 100.0
    memory_gb: float = 0.0
    io_volume_gb: float = 0.0
    interconnect: str = "unknown"
    notes: str = ""


@dataclass
class ScalingSeries:
    """Complete scaling series for one system/problem."""
    name: str
    description: str
    scaling_type: str  # "strong" or "weak"
    reference_point: str  # "1_node" or "1_core"
    data_points: List[ScalingDataPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def serial_speedup(self) -> List[float]:
        """Amdahl/Gustafson speedup relative to reference."""
        if not self.data_points:
            return []
        ref = self.data_points[0].runtime_seconds
        if ref <= 0:
            return [1.0] * len(self.data_points)
        return [ref / dp.runtime_seconds for dp in self.data_points]

    @property
    def ideal_speedup(self) -> List[float]:
        """Ideal linear speedup: N_ranks / N_ranks_ref."""
        if not self.data_points:
            return []
        ref_ranks = self.data_points[0].ranks if self.data_points else 1
        return [dp.ranks / ref_ranks for dp in self.data_points]

    @property
    def efficiency(self) -> List[float]:
        """Parallel efficiency: speedup / ideal_speedup * 100."""
        s = self.serial_speedup
        i = self.ideal_speedup
        return [min(100.0, 100.0 * s[j] / i[j]) if i[j] > 0 else 100.0 for j in range(len(s))]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_text_report(series_list: List[ScalingSeries]) -> str:
    """Generate a plain-text scaling report suitable for HPC logs."""

    lines = []
    sep = "=" * 80
    lines.append(sep)
    lines.append("  WIEN2k Gen – Benchmark Scaling Report")
    lines.append(sep)
    lines.append(f"  Generated: {__import__('datetime').datetime.now().isoformat()}")
    lines.append("")

    for series in series_list:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  System: {series.name}  ({series.scaling_type.upper()} scaling)")
        lines.append(f"  {series.description}")
        lines.append(f"{'─' * 60}")

        header = (
            f"{'Nodes':>6} {'Ranks':>6} {'Runtime(s)':>11} {'Speedup':>9} "
            f"{'Ideal':>7} {'Effic.%':>8} {'Mem(GB)':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for i, dp in enumerate(series.data_points):
            speedup = series.serial_speedup[i]
            ideal = series.ideal_speedup[i]
            eff = series.efficiency[i]
            lines.append(
                f"{dp.node_count:>6} {dp.ranks:>6} {dp.runtime_seconds:>11.1f} "
                f"{speedup:>8.2f}x {ideal:>6.2f}x {eff:>7.1f}% {dp.memory_gb:>7.1f}"
            )

        lines.append("")
        if series.metadata:
            lines.append("  Metadata:")
            for k, v in series.metadata.items():
                lines.append(f"    {k}: {v}")

    lines.append(f"\n{sep}")
    lines.append("  End of Report")
    lines.append(sep)
    return "\n".join(lines)


def generate_charts(
    series_list: List[ScalingSeries],
    output_dir: Union[str, Path],
    fname_prefix: str = "scaling",
    formats: Sequence[str] = ("pdf", "png"),
) -> List[Path]:
    """
    Generate speedup and efficiency charts as PDF/PNG.

    Returns list of generated file paths. Requires matplotlib.
    """
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not installed – skipping chart generation")
        return []

    output = Path(output_dir)
    _ensure_dir(output)
    generated: List[Path] = []

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # ── Speedup Chart ──────────────────────────────────────────────
    fig_speedup, ax_speedup = plt.subplots(figsize=(8, 5))
    max_ideal = 0

    for idx, series in enumerate(series_list):
        color = colors[idx % len(colors)]
        ranks = [dp.ranks for dp in series.data_points]
        speedup = series.serial_speedup
        ideal = series.ideal_speedup
        max_ideal = max(max_ideal, max(ideal) if ideal else 0)
        efficiency = series.efficiency

        ax_speedup.plot(
            ranks,
            speedup,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=color,
            label=f"{series.name} (actual)",
        )
        ax_speedup.plot(
            ranks,
            ideal,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            markersize=6,
            color=color,
            alpha=0.4,
            label=f"{series.name} (ideal)",
        )

        # Annotate efficiency on each point
        for i, (r, s) in enumerate(zip(ranks, speedup)):
            ax_speedup.annotate(
                f"{efficiency[i]:.0f}%",
                (r, s),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=7,
                color=color,
                ha="center",
            )

    ax_speedup.set_xlabel("MPI Ranks", fontsize=12, fontweight="bold")
    ax_speedup.set_ylabel("Speedup (×)", fontsize=12, fontweight="bold")
    ax_speedup.set_title("Strong Scaling Speedup", fontsize=14, fontweight="bold")
    ax_speedup.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_speedup.grid(True, alpha=0.3, linestyle="--")
    ax_speedup.set_xlim(left=0)

    # Diagonal reference
    if max_ideal > 0:
        ax_speedup.plot(
            [0, max_ideal],
            [0, max_ideal],
            "k-",
            alpha=0.15,
            linewidth=1,
            label="Linear",
        )

    fig_speedup.tight_layout()

    for fmt in formats:
        try:
            fpath = output / f"{fname_prefix}_speedup.{fmt}"
            fig_speedup.savefig(str(fpath), dpi=150, bbox_inches="tight")
            generated.append(fpath)
            logger.info(f"Speedup chart saved: {fpath}")
        except Exception as e:
            logger.error(f"Failed to save speedup chart ({fmt}): {e}")

    plt.close(fig_speedup)

    # ── Efficiency Chart ───────────────────────────────────────────
    fig_eff, ax_eff = plt.subplots(figsize=(8, 4))

    for idx, series in enumerate(series_list):
        color = colors[idx % len(colors)]
        ranks = [dp.ranks for dp in series.data_points]
        eff = series.efficiency

        ax_eff.plot(
            ranks,
            eff,
            marker="D",
            linestyle="-",
            linewidth=2,
            markersize=8,
            color=color,
            label=series.name,
        )
        # Annotate values
        for r, e in zip(ranks, eff):
            ax_eff.annotate(
                f"{e:.0f}%",
                (r, e),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                color=color,
                ha="center",
            )

    # 50% and 80% threshold lines
    ax_eff.axhline(y=80, color="green", linestyle=":", alpha=0.4, linewidth=1, label="80% target")
    ax_eff.axhline(y=50, color="red", linestyle=":", alpha=0.4, linewidth=1, label="50% floor")

    ax_eff.set_xlabel("MPI Ranks", fontsize=12, fontweight="bold")
    ax_eff.set_ylabel("Parallel Efficiency (%)", fontsize=12, fontweight="bold")
    ax_eff.set_title("Parallel Efficiency vs. MPI Ranks", fontsize=14, fontweight="bold")
    ax_eff.set_ylim(0, 105)
    ax_eff.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_eff.grid(True, alpha=0.3, linestyle="--")
    ax_eff.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    fig_eff.tight_layout()

    for fmt in formats:
        try:
            fpath = output / f"{fname_prefix}_efficiency.{fmt}"
            fig_eff.savefig(str(fpath), dpi=150, bbox_inches="tight")
            generated.append(fpath)
            logger.info(f"Efficiency chart saved: {fpath}")
        except Exception as e:
            logger.error(f"Failed to save efficiency chart ({fmt}): {e}")

    plt.close(fig_eff)

    return generated


def generate_report(
    series_list: List[ScalingSeries],
    output_dir: Union[str, Path],
    fname_prefix: str = "scaling",
    formats: Sequence[str] = ("pdf", "txt"),
) -> Dict[str, Any]:
    """
    Produce full benchmark report: text summary + charts (if matplotlib available).

    Returns dict with keys: 'text_report', 'chart_paths', 'output_dir'.
    """
    output = Path(output_dir)
    _ensure_dir(output)

    text_report = generate_text_report(series_list)
    text_path = output / f"{fname_prefix}_report.txt"
    text_path.write_text(text_report, encoding="utf-8")

    chart_paths: List[Path] = []
    chart_fmts = [f for f in formats if f not in ("txt", "text")]
    if chart_fmts and _MPL_AVAILABLE:
        chart_paths = generate_charts(
            series_list, output, fname_prefix, chart_fmts
        )

    logger.info(
        f"Benchmark report written to {output}/ ({len(chart_paths)} charts, 1 text)"
    )

    return {
        "text_report": text_report,
        "text_path": str(text_path),
        "chart_paths": [str(p) for p in chart_paths],
        "output_dir": str(output),
    }


def load_series_from_yaml(path: Union[str, Path]) -> List[ScalingSeries]:
    """Parse ScalingSeries from a wien2k_gen.yaml example file."""
    import yaml

    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    series_list = []
    benchmark = data.get("benchmark", {})
    parallel = data.get("parallel", {})
    calc = data.get("calculation", {})

    if not benchmark.get("nodes"):
        return series_list

    points = []
    nodes_list = benchmark["nodes"]
    runtimes = benchmark.get("runtime_seconds", [])
    efficiencies = benchmark.get("efficiency_percent", [])
    memory = benchmark.get(
        "memory_per_node_gb", [0] * len(nodes_list)
    )
    io_vol = benchmark.get("io_volume_gb", [0] * len(nodes_list))

    for i, n in enumerate(nodes_list):
        ranks = parallel.get("target_ranks", 1) * n // nodes_list[0] if nodes_list[0] > 0 else n
        dp = ScalingDataPoint(
            node_count=n,
            ranks=ranks,
            threads_per_rank=parallel.get("omp_threads", 1),
            runtime_seconds=runtimes[i] if i < len(runtimes) else 0,
            walltime_seconds=benchmark.get("walltime_seconds", runtimes)[i]
            if i < len(benchmark.get("walltime_seconds", runtimes))
            else 0,
            efficiency_percent=efficiencies[i]
            if i < len(efficiencies)
            else 100.0,
            memory_gb=memory if isinstance(memory, (int, float)) else (memory[i] if i < len(memory) else 0),
            io_volume_gb=io_vol if isinstance(io_vol, (int, float)) else (io_vol[i] if i < len(io_vol) else 0),
            interconnect=data.get("interconnect", "unknown"),
        )
        points.append(dp)

    series = ScalingSeries(
        name=calc.get("name", "unknown"),
        description=f"{calc.get('type', 'scf')} / {calc.get('xc', 'unknown')}",
        scaling_type="strong",
        reference_point="1_node",
        data_points=points,
        metadata={
            "lattice": data.get("structure", {}).get("lattice", ""),
            "spacegroup": data.get("structure", {}).get("spacegroup", ""),
            "kmesh": str(data.get("kmesh", {}).get("grid", "")),
            "rkmax": data.get("sphere", {}).get("rkmax", ""),
            "strategy": parallel.get("strategy", ""),
            "expected_speedup": parallel.get("expected_speedup", ""),
        },
    )
    series_list.append(series)
    return series_list


__all__ = [
    "ScalingDataPoint",
    "ScalingSeries",
    "generate_text_report",
    "generate_charts",
    "generate_report",
    "load_series_from_yaml",
]
