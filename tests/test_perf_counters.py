"""Unit tests for hardware counter detection and load-driven measurement."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

from forge.core import perf_counters as pc


def _idle_load_thread():
    """Return a finished dummy thread plus empty kernel metrics."""
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    return thread, {}


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

    with patch.object(pc, "_start_load_thread", return_value=_idle_load_thread()):
        with patch.object(pc, "_join_load_thread"):
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

    def fake_stream(_seconds: float, nbytes=None):
        write_stat()
        return {"bw_gb_s": 0.0, "bytes_moved": 0.0, "elapsed": 1.0}

    with patch.object(Path, "glob", return_value=[node]):
        with patch.object(pc, "_run_stream_kernel", side_effect=fake_stream):
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


def test_sysfs_bandwidth_invokes_stream_kernel(tmp_path: Path) -> None:
    node = tmp_path / "node0"
    node.mkdir()
    numastat = node / "numastat"
    numastat.write_text("numa_hit 0\nnuma_miss 0\n")
    called: dict[str, float] = {}

    def fake_stream(duration: float, nbytes=None):
        called["duration"] = duration
        numastat.write_text("numa_hit 1000\nnuma_miss 0\n")
        return {"bw_gb_s": 7.5, "bytes_moved": 7.5e9, "elapsed": duration}

    with patch.object(Path, "glob", return_value=[node]):
        with patch.object(pc, "_run_stream_kernel", side_effect=fake_stream):
            bw = pc._measure_memory_bandwidth_sysfs(sample_sec=0.5)
    assert called["duration"] == 0.5
    assert bw == 7.5


def test_perf_bandwidth_invokes_stream_kernel() -> None:
    called: dict[str, float] = {}

    def fake_stream(duration: float, nbytes=None):
        called["duration"] = duration
        return {"bw_gb_s": 11.0, "bytes_moved": 11e9, "elapsed": duration}

    with patch.object(pc, "_run_stream_kernel", side_effect=fake_stream):
        with patch.object(pc, "_run_cmd_safe", return_value=""):
            bw = pc._measure_memory_bandwidth_perf(sample_sec=0.4)
    assert called["duration"] == 0.4
    assert bw == 11.0


def test_perf_flops_invokes_fma_kernel() -> None:
    called: dict[str, float] = {}

    def fake_fma(duration: float):
        called["duration"] = duration
        return {"gflops": 9.0, "flops": 9e9, "elapsed": duration}

    with patch.object(pc, "_run_fma_kernel", side_effect=fake_fma):
        with patch.object(pc, "_run_cmd_safe", return_value=""):
            gflops = pc._measure_peak_flops_perf(sample_sec=0.3)
    assert called["duration"] == 0.3
    assert gflops == 9.0


def test_perf_cache_invokes_hierarchy_kernel() -> None:
    called: dict[str, float] = {}

    def fake_cache(duration: float):
        called["duration"] = duration
        return {"l1": 400.0, "l2": 150.0, "l3": 60.0, "elapsed": duration}

    with patch.object(pc, "_run_cache_hierarchy_kernel", side_effect=fake_cache):
        with patch.object(pc, "_run_cmd_safe", return_value=""):
            result = pc._perf_cache_events(sample_sec=0.3)
    assert called["duration"] == 0.3
    assert result["l1"] == 400.0
    assert result["l2"] == 150.0
    assert result["l3"] == 60.0


def test_flops_prefers_hardware_counters_over_python_kernel() -> None:
    packed = "  1000000000      fp_arith_inst_retired.256b_packed_double\n"

    def fake_fma(duration: float):
        return {"gflops": 1.2, "flops": 1.2e9, "elapsed": duration}

    with patch.object(pc, "_run_fma_kernel", side_effect=fake_fma):
        with patch.object(pc, "_run_cmd_safe", return_value=packed):
            gflops = pc._measure_peak_flops_perf(sample_sec=1.0)
    assert gflops == 4.0


def test_bandwidth_prefers_stream_kernel_over_counters() -> None:
    def fake_stream(duration: float, nbytes=None):
        return {"bw_gb_s": 18.5, "bytes_moved": 18.5e9, "elapsed": duration}

    with patch.object(pc, "_run_stream_kernel", side_effect=fake_stream):
        with patch.object(pc, "_run_cmd_safe", return_value="  1000      cache-misses\n  2000      cycles\n"):
            bw = pc._measure_memory_bandwidth_perf(sample_sec=1.0)
    assert bw == 18.5


def test_stream_kernel_nonzero_on_host() -> None:
    result = pc._run_stream_kernel(0.05, nbytes=4 * 1024 * 1024)
    assert result["elapsed"] >= 0.05
    assert result["bytes_moved"] > 0
    assert result["bw_gb_s"] > 0.1


def test_fma_kernel_nonzero_on_host() -> None:
    result = pc._run_fma_kernel(0.05)
    assert result["elapsed"] >= 0.05
    assert result["flops"] > 0
    assert result["gflops"] > 0.01


def test_repeated_stream_not_near_zero() -> None:
    values = [pc._run_stream_kernel(0.05, nbytes=4 * 1024 * 1024)["bw_gb_s"] for _ in range(2)]
    assert all(v > 0.1 for v in values)


def _isolated_cache(tmp_path: Path):
    old = pc.PerfCounterCache._instance
    pc.PerfCounterCache._instance = None
    cache_file = tmp_path / "perf_cache.json"
    return old, cache_file


def test_cache_ttl_keeps_value_after_five_minutes(tmp_path: Path, monkeypatch) -> None:
    old, cache_file = _isolated_cache(tmp_path)
    monkeypatch.setattr(pc, "_PERF_CACHE_FILE", cache_file)
    monkeypatch.setattr(pc, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pc, "_hardware_fingerprint", lambda: "fp-test")
    try:
        cache = pc.PerfCounterCache()
        cache.put("mem_bw_gb_s", 55.5)
        now = cache._cache["mem_bw_gb_s"]["_ts"]
        with patch.object(pc.time, "time", return_value=now + 6 * 60):
            assert cache.get("mem_bw_gb_s") == 55.5
        with patch.object(pc.time, "time", return_value=now + 8 * 24 * 3600):
            assert cache.get("mem_bw_gb_s") is None
    finally:
        pc.PerfCounterCache._instance = old


def test_cache_ttl_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_PERF_CACHE_TTL_SECONDS", "10")
    assert pc._cache_ttl_seconds() == 10
    old, cache_file = _isolated_cache(tmp_path)
    monkeypatch.setattr(pc, "_PERF_CACHE_FILE", cache_file)
    monkeypatch.setattr(pc, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pc, "_hardware_fingerprint", lambda: "fp-test")
    try:
        cache = pc.PerfCounterCache()
        cache.put("peak_flops_gflops", 100.0)
        now = cache._cache["peak_flops_gflops"]["_ts"]
        with patch.object(pc.time, "time", return_value=now + 11):
            assert cache.get("peak_flops_gflops") is None
    finally:
        pc.PerfCounterCache._instance = old


def test_fingerprint_mismatch_invalidates_regardless_of_ttl(tmp_path: Path, monkeypatch) -> None:
    cache_file = tmp_path / "perf_cache.json"
    cache_file.write_text(
        '{"_fingerprint": "old-fp", "mem_bw_gb_s": {"_data": 12.0, "_ts": 9999999999}}'
    )
    old = pc.PerfCounterCache._instance
    pc.PerfCounterCache._instance = None
    monkeypatch.setattr(pc, "_PERF_CACHE_FILE", cache_file)
    monkeypatch.setattr(pc, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pc, "_hardware_fingerprint", lambda: "new-fp")
    try:
        cache = pc.PerfCounterCache()
        assert cache.get("mem_bw_gb_s") is None
        assert cache._cache.get("_fingerprint") == "new-fp"
    finally:
        pc.PerfCounterCache._instance = old


def test_invalidate_clears_entries(tmp_path: Path, monkeypatch) -> None:
    old, cache_file = _isolated_cache(tmp_path)
    monkeypatch.setattr(pc, "_PERF_CACHE_FILE", cache_file)
    monkeypatch.setattr(pc, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(pc, "_hardware_fingerprint", lambda: "fp-test")
    try:
        cache = pc.PerfCounterCache()
        cache.put("roofline_data", {"peak_flops_gflops": 1.0})
        pc.invalidate_perf_cache()
        assert cache.get("roofline_data") is None
    finally:
        pc.PerfCounterCache._instance = old


class _DummyConsole:
    def print(self, *args, **kwargs) -> None:
        return None

    def print_json(self, *args, **kwargs) -> None:
        return None


def test_calibrate_command_invalidates_then_measures(monkeypatch) -> None:
    from argparse import Namespace

    from forge.cli_commands import calibrate as calibrate_cmd

    calls: list[str] = []

    def fake_invalidate() -> None:
        calls.append("invalidate")

    def fake_measure(use_cache: bool = True):
        calls.append("measure")
        return {
            "peak_flops_gflops": 80.0,
            "sustained_bw_gb_s": 40.0,
            "tool_used": "fallback",
            "cache_bandwidth": {},
        }

    monkeypatch.setattr("forge.core.perf_counters.invalidate_perf_cache", fake_invalidate)
    monkeypatch.setattr("forge.core.perf_counters.get_real_roofline_data", fake_measure)
    monkeypatch.setattr("forge.cli_commands.calibrate.get_console", lambda: _DummyConsole())
    result = calibrate_cmd.handle(Namespace(json_output=True, json=True), None)
    assert calls == ["invalidate", "measure"]
    assert result["status"] == "calibrated"
    assert result["data"]["peak_flops_gflops"] == 80.0


def test_generate_recalibrate_invalidates_cache(monkeypatch) -> None:
    from argparse import Namespace

    from forge.cli_commands import generate as generate_cmd
    from forge.core.topology import Topology
    from forge.types import PipelineResult

    calls: list[str] = []

    def fake_invalidate() -> None:
        calls.append("invalidate")

    monkeypatch.setattr("forge.core.perf_counters.invalidate_perf_cache", fake_invalidate)
    monkeypatch.setattr("forge.cli_commands.generate.get_console", lambda: _DummyConsole())
    monkeypatch.setattr("forge.core.hardware.get_physical_cores", lambda: 8)
    monkeypatch.setattr(
        "forge.core.scheduler.detect",
        lambda max_cores=None: Topology(nodes=["n1"], cores_per_node=[8]),
    )
    monkeypatch.setattr(
        "forge.core.pipeline.run_pipeline",
        lambda **kw: PipelineResult(success=True, warnings=[]),
    )
    args = Namespace(
        max_cores=None,
        reserve_os_cores=None,
        mode=None,
        cores=None,
        omp=None,
        memory_limit=None,
        gpu=False,
        gpu_mixed_precision=False,
        dry_run=True,
        export=None,
        ignore_saturation=False,
        recalibrate=True,
        json_output=False,
        manual=False,
    )
    generate_cmd.handle(args, None)
    assert "invalidate" in calls
