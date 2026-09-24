"""Unit tests for hardware counter detection and measurement bugs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from forge.core import perf_counters as pc


def test_detect_prefers_perf_over_sysfs() -> None:
    with patch.object(pc, "_which", side_effect=lambda name: name == "perf"):
        assert pc._detect_perf_tools() == "perf"


def test_detect_returns_perf_when_sysfs_missing() -> None:
    with patch.object(pc, "_which", side_effect=lambda name: name == "perf"):
        with patch.object(Path, "exists", return_value=False):
            assert pc._detect_perf_tools() == "perf"


def test_detect_sysfs_when_no_perf_tools() -> None:
    with patch.object(pc, "_which", return_value=False):
        with patch.object(Path, "exists", return_value=True):
            assert pc._detect_perf_tools() == "sysfs"


def test_check_counter_access_sysfs_returns_bool() -> None:
    with patch.object(pc, "_PERF_TOOL_AVAILABLE", "sysfs"):
        with patch.object(Path, "is_dir", return_value=True):
            with patch.object(pc.os, "access", return_value=True):
                assert pc._check_counter_access() is True


def test_peak_flops_scalar_fallback_uses_one_op() -> None:
    scalar_out = "  2000000000      fp_arith_inst_retired.scalar_double\n"

    def fake_run(cmd, timeout=30):
        event = cmd[3] if len(cmd) > 3 else ""
        if "256b_packed_double" in event:
            return None
        if "scalar_double" in event:
            return scalar_out
        return None

    with patch.object(pc, "_run_cmd_safe", side_effect=fake_run):
        gflops = pc._measure_peak_flops_perf(sample_sec=1.0)
    assert gflops == 2.0


def test_sysfs_bandwidth_none_without_numastat(tmp_path: Path) -> None:
    node = tmp_path / "node0"
    node.mkdir()
    (node / "meminfo").write_text(
        "Node 0 MemTotal: 1024 kB\nNode 0 MemFree: 512 kB\nNode 0 MemUsed: 512 kB\n"
    )
    with patch.object(Path, "glob", return_value=[node]):
        assert pc._measure_memory_bandwidth_sysfs(sample_sec=0.01) is None


def test_sysfs_bandwidth_from_numastat_delta(tmp_path: Path) -> None:
    node = tmp_path / "node0"
    node.mkdir()
    numastat = node / "numastat"
    state = {"n": 0}

    def write_stat() -> None:
        state["n"] += 250000
        numastat.write_text(f"numa_hit {state['n']}\nnuma_miss 0\n")

    write_stat()

    def fake_sleep(_seconds: float) -> None:
        write_stat()

    with patch.object(Path, "glob", return_value=[node]):
        with patch.object(pc.time, "sleep", side_effect=fake_sleep):
            with patch.object(pc.os, "sysconf", return_value=4096):
                bw = pc._measure_memory_bandwidth_sysfs(sample_sec=1.0)
    assert bw == 1.02


def test_cache_init_under_lock() -> None:
    old = pc.PerfCounterCache._instance
    pc.PerfCounterCache._instance = None
    try:
        cache = pc.PerfCounterCache()
        assert cache._initialized is True
        assert pc.PerfCounterCache() is cache
    finally:
        pc.PerfCounterCache._instance = old
