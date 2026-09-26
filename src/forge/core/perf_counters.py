"""
Hardware Performance Counter Integration Module.
Provides ACTUAL measured (not estimated) memory bandwidth and FLOPS
using Linux perf subsystem, likwid-perfctr, and sysfs NUMA counters.
Includes thread-safe disk caching with hardware fingerprint invalidation.

Measurements are load-driven: a STREAM triad, FP64 FMA loop, or sized
cache kernel runs during the sample window. Idle ``perf stat -- sleep``
or likwid ``-S`` samples of ambient activity are not used as the primary
result because they return near-zero on a typical CLI machine.

Measurement methods (in priority order):
1. likwid-perfctr CLI — FLOPS, bandwidth, cache misses (with synthetic load)
2. perf stat (Linux perf subsystem, with synthetic load)
3. /sys/devices/system/node/node*/numastat for NUMA-local bandwidth (with load)
4. Theoretical peak x efficiency factor fallback

All documentation and inline comments are in English per project standards.
"""

import contextlib
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Module-Level Graceful Degradation Flag
# =============================================================================

HAS_PERF_COUNTERS: bool = False
"""Flag indicating whether any hardware performance counter tools are available."""

_PERF_TOOL_AVAILABLE: Optional[str] = None
"""Name of the detected tool ("likwid", "perf", "sysfs") or None."""

_CACHE_DIR = Path.home() / ".forge"
_PERF_CACHE_FILE = _CACHE_DIR / "perf_cache.json"
_CACHE_TTL_SECONDS: int = 7 * 24 * 3600  # 7 days; fingerprint still invalidates
_CALIBRATION_NOTICE = (
    "Calibrating hardware profile — this happens once per machine and is cached at "
    "~/.forge/perf_cache.json; re-run with --recalibrate to refresh"
)


def _cache_ttl_seconds() -> int:
    """Return cache TTL. Override with FORGE_PERF_CACHE_TTL_SECONDS."""
    raw = os.environ.get("FORGE_PERF_CACHE_TTL_SECONDS", "")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            logger.debug("Invalid FORGE_PERF_CACHE_TTL_SECONDS=%r; using default", raw)
    return _CACHE_TTL_SECONDS

# =============================================================================
# Tool Detection (executed at module import)
# =============================================================================

def _which(executable: str) -> bool:
    """Return True if *executable* is found in PATH."""
    try:
        result = subprocess.run(
            ["which", executable],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def _detect_perf_tools() -> Optional[str]:
    """Detect available hardware counter tooling and return the best option."""
    if _which("likwid-perfctr"):
        return "likwid"
    if _which("perf"):
        return "perf"
    try:
        if Path("/sys/devices/system/node").exists():
            return "sysfs"
    except (PermissionError, OSError):
        logger.debug("Suppressed exception in _detect_perf_tools()", exc_info=True)
    return None


def _check_counter_access() -> bool:
    """Verify that the user has permission to read hardware counters."""
    try:
        if _PERF_TOOL_AVAILABLE == "perf":
            paranoid = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
            return int(paranoid) < 2
        if _PERF_TOOL_AVAILABLE == "likwid":
            result = subprocess.run(
                ["likwid-perfctr", "-C", "0", "-g", "MEM", "-m", "-O"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=15
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        if _PERF_TOOL_AVAILABLE == "sysfs":
            node = Path("/sys/devices/system/node")
            return node.is_dir() and os.access(node, os.R_OK)
    except (PermissionError, subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
        logger.debug("Suppressed exception in _check_counter_access()", exc_info=True)
    return False


try:
    _PERF_TOOL_AVAILABLE = _detect_perf_tools()
    if _PERF_TOOL_AVAILABLE:
        HAS_PERF_COUNTERS = bool(_check_counter_access())
except OSError:
    _PERF_TOOL_AVAILABLE = None
    HAS_PERF_COUNTERS = False

logger.debug(
    "Perf counters: tool=%s, access_ok=%s",
    _PERF_TOOL_AVAILABLE or "none",
    HAS_PERF_COUNTERS
)

# =============================================================================
# Helper: Safe Command Execution
# =============================================================================

def _run_cmd_safe(cmd: list[str], timeout: int = 30) -> Optional[str]:
    """Safely execute a shell command with timeout and stderr suppression."""
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        return subprocess.check_output(
            cmd, text=True, timeout=timeout,
            stderr=subprocess.DEVNULL, env=env
        ).strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        logger.debug("Command failed or not found: %s", " ".join(cmd))
        return None


# =============================================================================
# Load-driven synthetic kernels (STREAM / FMA / cache)
# =============================================================================
# Idle `perf stat -- sleep` / likwid `-S` / sysfs snapshots measure ambient
# machine noise, which is near-zero on a typical CLI run and poisons the
# roofline. These kernels generate real memory/FP traffic during the sample
# window. Bytes-moved (or FLOPs) / elapsed-time is the primary metric;
# hardware counters, when present, are a cross-check only.
# src/forge/benchmark/synthetic.py models DFT wall-time, not STREAM/FMA, so
# the kernels live here.

_STREAM_MAX_PER_ARRAY = 128 * 1024 * 1024
_STREAM_MIN_PER_ARRAY = 32 * 1024 * 1024


def _parse_size_to_bytes(text: str) -> Optional[int]:
    raw = text.strip().upper().replace("IB", "B").replace("I", "")
    match = re.match(r"^(\d+)\s*([KMG])?B?$", raw)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "K":
        value *= 1024
    elif unit == "M":
        value *= 1024 * 1024
    elif unit == "G":
        value *= 1024 * 1024 * 1024
    return value


def _cache_level_size_bytes(level: int) -> Optional[int]:
    """Return data-cache size for *level* (1/2/3) from sysfs, or None."""
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.is_dir():
        return None
    for idx in sorted(base.glob("index*")):
        try:
            lv = int((idx / "level").read_text().strip())
            if lv != level:
                continue
            typ = (idx / "type").read_text().strip().lower()
            if level == 1 and typ == "instruction":
                continue
            parsed = _parse_size_to_bytes((idx / "size").read_text())
            if parsed:
                return parsed
        except (OSError, ValueError):
            continue
    return None


def _stream_array_bytes() -> int:
    llc = _cache_level_size_bytes(3) or 32 * 1024 * 1024
    return int(min(max(4 * llc, _STREAM_MIN_PER_ARRAY), _STREAM_MAX_PER_ARRAY))


def _run_stream_kernel(duration: float, nbytes: Optional[int] = None) -> dict[str, float]:
    """STREAM triad ``c[i] = a[i] + scale*b[i]`` for *duration* seconds.

    Working set is larger than LLC so traffic goes to DRAM. Returns
    ``bw_gb_s``, ``bytes_moved``, ``elapsed``. Load-driven: this is the
    actual bytes/time, not an idle counter sample.
    """
    duration = max(0.05, float(duration))
    nbytes = int(nbytes) if nbytes else _stream_array_bytes()
    n = max(1024, nbytes // 8)
    try:
        import numpy as np
        a = np.empty(n, dtype=np.float64)
        b = np.empty(n, dtype=np.float64)
        c = np.empty(n, dtype=np.float64)
        a.fill(1.0)
        b.fill(2.0)
        scale = np.float64(1.0001)
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < duration:
            np.multiply(b, scale, out=c)
            np.add(a, c, out=c)
            iters += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
    except Exception:
        logger.debug("numpy STREAM unavailable; using array.array fallback", exc_info=True)
        import array as _array
        n = min(n, 2 * 1024 * 1024)
        a = _array.array("d", [1.0]) * n
        b = _array.array("d", [2.0]) * n
        c = _array.array("d", [0.0]) * n
        scale = 1.0001
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < duration:
            for i in range(n):
                c[i] = a[i] + scale * b[i]
            iters += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
    bytes_moved = float(iters * n * 8 * 3)
    bw_gb_s = bytes_moved / elapsed / 1e9
    return {"bw_gb_s": round(bw_gb_s, 2), "bytes_moved": bytes_moved, "elapsed": elapsed}


def _run_fma_kernel(duration: float) -> dict[str, float]:
    """Tight FP64 FMA loop on L1-resident data for *duration* seconds.

    Load-driven: hardware FP counters are sampled against real arithmetic,
    not an idle ``sleep``. Returns ``gflops``, ``flops``, ``elapsed``.
    """
    duration = max(0.05, float(duration))
    try:
        import numpy as np
        n = 4096
        x = np.ones(n, dtype=np.float64)
        y = np.full(n, 1.0000001, dtype=np.float64)
        s = np.float64(1.0000000001)
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < duration:
            x = x * s + y
            iters += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
        flops = float(iters * n * 2)
    except Exception:
        logger.debug("numpy FMA unavailable; using scalar fallback", exc_info=True)
        x = y = 1.0000001
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < duration:
            x = x * 1.0000000001 + y
            y = y * 1.0000000001 + x
            iters += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
        flops = float(iters * 4)
    gflops = flops / elapsed / 1e9
    return {"gflops": round(gflops, 2), "flops": flops, "elapsed": elapsed}


def _run_cache_kernel(duration: float, nbytes: int) -> dict[str, float]:
    """Sequential read/write of a *nbytes* working set for cache bandwidth."""
    duration = max(0.05, float(duration))
    n = max(64, int(nbytes) // 8)
    try:
        import numpy as np
        a = np.empty(n, dtype=np.float64)
        a.fill(1.0)
        t0 = time.perf_counter()
        iters = 0
        acc = 0.0
        while time.perf_counter() - t0 < duration:
            acc += float(a.sum())
            a += 0.0
            iters += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
        bytes_moved = float(iters * n * 8 * 2)
    except Exception:
        import array as _array
        n = min(n, 256 * 1024)
        a = _array.array("d", [1.0]) * n
        t0 = time.perf_counter()
        iters = 0
        acc = 0.0
        while time.perf_counter() - t0 < duration:
            acc += sum(a)
            iters += 1
        elapsed = max(time.perf_counter() - t0, 1e-9)
        bytes_moved = float(iters * n * 8)
    return {
        "bw_gb_s": round(bytes_moved / elapsed / 1e9, 2),
        "bytes_moved": bytes_moved,
        "elapsed": elapsed,
        "acc": acc,
    }


def _run_cache_hierarchy_kernel(duration: float) -> dict[str, float]:
    """L1/L2/L3-sized access loops, split across *duration*."""
    slice_t = max(0.05, float(duration) / 3.0)
    l1 = _cache_level_size_bytes(1) or 32 * 1024
    l2 = _cache_level_size_bytes(2) or 256 * 1024
    l3 = _cache_level_size_bytes(3) or 8 * 1024 * 1024
    r1 = _run_cache_kernel(slice_t, max(4096, l1 // 2))
    r2 = _run_cache_kernel(slice_t, max(l1 * 2, min(l2 // 2, 512 * 1024)))
    r3 = _run_cache_kernel(slice_t, max(l2 * 2, min(l3 // 2, 4 * 1024 * 1024)))
    return {
        "l1": r1["bw_gb_s"],
        "l2": r2["bw_gb_s"],
        "l3": r3["bw_gb_s"],
        "elapsed": r1["elapsed"] + r2["elapsed"] + r3["elapsed"],
    }


def _start_load_thread(fn, *args, **kwargs):
    """Run *fn* on a daemon thread; returns (thread, result_dict)."""
    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            got = fn(*args, **kwargs) or {}
            if isinstance(got, dict):
                result.update(got)
        except Exception:
            logger.debug("Synthetic load kernel failed", exc_info=True)

    thread = threading.Thread(target=_runner, name="forge-perf-load", daemon=True)
    thread.start()
    return thread, result


def _join_load_thread(thread: threading.Thread, timeout: float) -> None:
    thread.join(timeout=max(0.1, timeout))


def _prefer_kernel_metric(
    kernel: Optional[float],
    counter: Optional[float],
    label: str,
) -> Optional[float]:
    """Combine load-driven kernel timing with hardware-counter cross-check.

    Hardware counters sampled during a real kernel are preferred for FLOPS
    (Python cannot approach peak FP64). For bandwidth, STREAM bytes/time is
    the primary signal; counters only fill in if the kernel result is missing.
    """
    if label == "gflops":
        if counter is not None and counter > 0:
            if kernel is not None and kernel > 0:
                logger.debug("Load-driven %s: kernel=%.3f counters=%.3f", label, kernel, counter)
            return round(float(counter), 2)
        if kernel is not None and kernel > 0:
            return round(float(kernel), 2)
        return None
    if kernel is not None and kernel > 0:
        if counter is not None and counter > 0:
            logger.debug("Load-driven %s: kernel=%.3f counters=%.3f", label, kernel, counter)
        return round(float(kernel), 2)
    if counter is not None and counter > 0:
        return round(float(counter), 2)
    return None


# =============================================================================
# Hardware Fingerprint for Cache Invalidation
# =============================================================================

@cache
def _hardware_fingerprint() -> str:
    """
    Generate a stable hash based on CPU model, core count, and NUMA topology.
    Used to invalidate cached measurements when hardware changes.
    """
    components = []
    with contextlib.suppress(Exception):
        components.append(Path("/proc/cpuinfo").read_text().split("\n")[0])
    try:
        cpu_model = _run_cmd_safe(["lscpu"], timeout=5)
        if cpu_model:
            components.append(cpu_model)
    except Exception:
        logger.debug("Suppressed exception in _hardware_fingerprint()", exc_info=True)
    try:
        for node_dir in sorted(Path("/sys/devices/system/node").glob("node*")):
            meminfo = node_dir / "meminfo"
            if meminfo.exists():
                components.append(meminfo.read_text())
    except Exception:
        logger.debug("Suppressed exception in _hardware_fingerprint()", exc_info=True)

    fingerprint = hashlib.sha256(
        "|".join(components).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return fingerprint

# =============================================================================
# Abstract Perf Counter Interface
# =============================================================================

class PerfCounterInterface:
    """
    Abstract interface for hardware counter access.

    Provides unified access to:
    - likwid-perfctr (FLOPS, bandwidth, cache misses)
    - Linux perf stat (cycles, instructions, cache events)
    - sysfs NUMA memory counters
    - Theoretical fallback calculations
    """

    @staticmethod
    def is_likwid_available() -> bool:
        """Return True if likwid-perfctr is installed and accessible."""
        return _PERF_TOOL_AVAILABLE == "likwid" and HAS_PERF_COUNTERS

    @staticmethod
    def is_perf_available() -> bool:
        """Return True if Linux perf subsystem is accessible."""
        return _PERF_TOOL_AVAILABLE == "perf" and HAS_PERF_COUNTERS

    @staticmethod
    def is_sysfs_available() -> bool:
        """Return True if sysfs NUMA counters are readable."""
        return _PERF_TOOL_AVAILABLE == "sysfs" and HAS_PERF_COUNTERS

    @staticmethod
    def list_available_tools() -> list[str]:
        """Return list of detected hardware counter tools."""
        tools = []
        if PerfCounterInterface.is_likwid_available():
            tools.append("likwid")
        if PerfCounterInterface.is_perf_available():
            tools.append("perf")
        if PerfCounterInterface.is_sysfs_available():
            tools.append("sysfs")
        return tools

# =============================================================================
# PerfCounterCache — Thread-Safe Disk Cache
# =============================================================================

class PerfCounterCache:
    """
    Thread-safe disk cache for hardware performance measurements.

    Cached at ``~/.forge/perf_cache.json`` with:
    - Hardware fingerprint hash for invalidation when hardware changes
    - Configurable TTL (default 7 days, ``FORGE_PERF_CACHE_TTL_SECONDS``) per key
    - Atomic write via temporary file + rename
    Fingerprint mismatch always discards the cache, independent of TTL.
    """

    _lock = threading.Lock()
    _instance: Optional["PerfCounterCache"] = None

    def __new__(cls) -> "PerfCounterCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._cache: dict[str, Any] = {}
            self._fingerprint = _hardware_fingerprint()
            self._load()
            self._initialized = True

    def _load(self) -> None:
        """Load cache from disk, discarding entries with mismatched fingerprint."""
        try:
            if _PERF_CACHE_FILE.exists():
                raw = json.loads(_PERF_CACHE_FILE.read_text())
                if isinstance(raw, dict) and raw.get("_fingerprint") == self._fingerprint:
                    self._cache = raw
                else:
                    logger.debug("Cache fingerprint mismatch; discarding old cache")
                    self._cache = {"_fingerprint": self._fingerprint}
            else:
                self._cache = {"_fingerprint": self._fingerprint}
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to load perf cache: %s", e)
            self._cache = {"_fingerprint": self._fingerprint}

    def _save(self) -> None:
        """Atomically write cache to disk."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = _PERF_CACHE_FILE.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(self._cache, indent=2))
            tmp_path.rename(_PERF_CACHE_FILE)
        except OSError as e:
            logger.debug("Failed to save perf cache: %s", e)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts = entry.get("_ts", 0)
            if time.time() - ts > _cache_ttl_seconds():
                return None
            return entry.get("_data")

    def put(self, key: str, data: Any) -> None:
        """Store a value in the cache with current timestamp."""
        with self._lock:
            self._cache[key] = {"_data": data, "_ts": time.time()}
            self._save()

    def invalidate(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache = {"_fingerprint": self._fingerprint}
            self._save()

    def refresh_fingerprint(self) -> bool:
        """
        Recompute hardware fingerprint; returns True if it changed
        (causing cache invalidation).
        """
        new_fp = _hardware_fingerprint()
        if new_fp != self._fingerprint:
            self._fingerprint = new_fp
            self.invalidate()
            return True
        return False

# =============================================================================
# Frequency-Dependent Fallback Calculations
# =============================================================================

@cache
def _get_cpu_freq_mhz() -> float:
    """Return current CPU frequency in MHz from sysfs or /proc/cpuinfo."""
    try:
        path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if path.exists():
            return float(path.read_text().strip()) / 1000.0
    except Exception:
        logger.debug("Suppressed exception in _get_cpu_freq_mhz()", exc_info=True)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "cpu MHz" in line:
                    return float(line.split(":")[1].strip())
    except Exception:
        logger.debug("Suppressed exception in _get_cpu_freq_mhz()", exc_info=True)
    return 2000.0


def calculations_from_cpu_frequency() -> dict[str, float]:
    """
    Estimate bandwidth and FLOPS from CPU frequency and known microarchitecture
    parameters. Used as fallback when no hardware counter tools are available.

    Returns
    -------
    dict with keys "bw_gb_s" and "gflops".
    """
    freq_mhz = _get_cpu_freq_mhz()

    raw = _run_cmd_safe(["lscpu"], timeout=5)
    core_count = os.cpu_count() or 1
    sockets = 1
    cores_per_socket = core_count

    if raw:
        for line in raw.split("\n"):
            low = line.lower().strip()
            if "socket(s)" in low and "per" not in low:
                with contextlib.suppress(ValueError):
                    sockets = int(line.split(":")[-1].strip())
            elif "core(s) per socket" in low:
                with contextlib.suppress(ValueError):
                    cores_per_socket = int(line.split(":")[-1].strip())

    # Estimate memory bandwidth: ~50 GB/s per socket for modern EPYC/Xeon
    bw_gb_s = sockets * 50.0 * 0.7

    # Estimate peak FP64: assumed 2 FMA units x 256-bit x 2 ops/cycle
    # AVX2 ceiling: 2 * (256/64) * 2 = 16 FLOPs/cycle/core
    ops_per_cycle = 16.0
    gflops = sockets * cores_per_socket * freq_mhz * 1e6 * ops_per_cycle / 1e9

    return {"bw_gb_s": round(bw_gb_s, 2), "gflops": round(gflops, 2)}

# =============================================================================
# Memory Bandwidth Measurement
# =============================================================================

def _measure_memory_bandwidth_likwid(sample_sec: float = 2.0) -> Optional[float]:
    """
    Measure sustained memory bandwidth using likwid-perfctr MEM group.

    Load-driven: a STREAM triad kernel runs concurrently so likwid samples
    DRAM traffic rather than idle/ambient counters. Kernel bytes/time is
    preferred; likwid is a cross-check.
    Returns GB/s or None on failure.
    """
    thread, load = _start_load_thread(_run_stream_kernel, sample_sec)
    output = None
    try:
        cmd = ["likwid-perfctr", "-C", "0", "-g", "MEM", "-m", "-O", "-S", str(sample_sec)]
        output = _run_cmd_safe(cmd, timeout=int(sample_sec) + 15)
    finally:
        _join_load_thread(thread, sample_sec + 5)

    total = 0.0
    if output:
        for line in output.split("\n"):
            match = re.search(r"Memory bandwidth\s+\[GB/s\]\s+([\d.]+)", line, re.IGNORECASE)
            if match:
                total += float(match.group(1))
            match = re.search(r"L3 memory bandwidth\s+\[GB/s\]\s+([\d.]+)", line, re.IGNORECASE)
            if match:
                total += float(match.group(1))
        if total <= 0:
            for line in output.split("\n"):
                parts = line.strip().split(",")
                for part in parts:
                    part = part.strip()
                    if part and re.match(r"^[\d.]+$", part):
                        total += float(part)

    counter_bw = round(total, 2) if total > 0 else None
    return _prefer_kernel_metric(load.get("bw_gb_s"), counter_bw, "mem_bw")


def _measure_memory_bandwidth_perf(sample_sec: float = 3.0) -> Optional[float]:
    """
    Measure memory bandwidth with a STREAM triad under ``perf stat``.

    Load-driven: idle ``perf stat -- sleep`` samples ambient noise and
    usually returns None or a near-zero value. The STREAM kernel runs on a
    background thread for the sample window; bytes-moved/elapsed is the
    primary result, with cache-miss counters as a cross-check.
    Returns GB/s estimate or None on failure.
    """
    thread, load = _start_load_thread(_run_stream_kernel, sample_sec)
    output = None
    try:
        cmd = [
            "perf", "stat",
            "-e", "cycles,instructions,cache-references,cache-misses",
            "-a", "--", "sleep", str(sample_sec)
        ]
        output = _run_cmd_safe(cmd, timeout=int(sample_sec) + 15)
    finally:
        _join_load_thread(thread, sample_sec + 5)

    counter_bw = None
    if output:
        cache_misses = 0
        cycles = 0
        for line in output.split("\n"):
            match = re.search(r"([\d.,]+)\s+cache-misses", line)
            if match:
                cache_misses = int(match.group(1).replace(",", "").replace(".", ""))
            match = re.search(r"([\d.,]+)\s+cycles", line)
            if match:
                cycles = int(match.group(1).replace(",", "").replace(".", ""))
        if cache_misses > 0 and cycles > 0:
            counter_bw = round((cache_misses * 64.0) / sample_sec / 1e9, 2)

    return _prefer_kernel_metric(load.get("bw_gb_s"), counter_bw, "mem_bw")


def _read_numa_traffic_pages(node_dir: Path) -> Optional[int]:
    """Return local+remote NUMA page hits from numastat, or None."""
    numastat = node_dir / "numastat"
    if not numastat.exists():
        return None
    hits = 0
    found = False
    try:
        for line in numastat.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            if parts[0] in ("numa_hit", "numa_miss"):
                hits += int(parts[1])
                found = True
    except (OSError, ValueError):
        logger.debug("Suppressed exception in _read_numa_traffic_pages()", exc_info=True)
        return None
    return hits if found else None


def _measure_memory_bandwidth_sysfs(sample_sec: float = 3.0) -> Optional[float]:
    """
    Estimate NUMA memory traffic from sysfs numastat page counters.

    Load-driven: a STREAM triad runs between the before/after snapshots so
    the delta is traffic we generated, not ambient NUMA activity from
    unrelated processes. Kernel bytes/time is preferred; numastat is a
    cross-check. meminfo is a capacity snapshot, not a bandwidth counter.
    Returns GB/s or None when traffic cannot be measured.
    """
    node_paths = sorted(Path("/sys/devices/system/node").glob("node*"))
    if not node_paths:
        return None

    def _read_snapshot() -> dict[int, int]:
        snap: dict[int, int] = {}
        for node_dir in node_paths:
            try:
                nid = int(node_dir.name.replace("node", ""))
            except ValueError:
                continue
            pages = _read_numa_traffic_pages(node_dir)
            if pages is not None:
                snap[nid] = pages
        return snap

    snapshot_before = _read_snapshot()
    if not snapshot_before:
        return None
    load = _run_stream_kernel(sample_sec)
    snapshot_after = _read_snapshot()

    total_delta = 0
    for nid, before in snapshot_before.items():
        delta = snapshot_after.get(nid, before) - before
        if delta > 0:
            total_delta += delta

    counter_bw = None
    if total_delta > 0:
        page_size = os.sysconf("SC_PAGE_SIZE") or 4096
        counter_bw = round((total_delta * page_size) / sample_sec / 1e9, 2)

    return _prefer_kernel_metric(load.get("bw_gb_s"), counter_bw, "mem_bw")


def measure_memory_bandwidth(use_cache: bool = True) -> float:  # noqa: C901
    """
    Measure sustained memory bandwidth in GB/s.

    Priority:
    1. likwid-perfctr MEM group
    2. perf stat cache-miss estimation
    3. sysfs NUMA memory counters
    4. Frequency-based theoretical fallback

    Parameters
    ----------
    use_cache : bool
        If True (default), cache the result in PerfCounterCache (7-day TTL).

    Returns
    -------
    float
        Sustained memory bandwidth in GB/s.
    """
    cache_key = "mem_bw_gb_s"

    if use_cache:
        cached = PerfCounterCache().get(cache_key)
        if cached is not None and isinstance(cached, (int, float)):
            logger.debug("Returning cached memory bandwidth: %.2f GB/s", cached)
            return float(cached)

    bw: Optional[float] = None
    tool_used = "fallback"

    if PerfCounterInterface.is_likwid_available():
        bw = _measure_memory_bandwidth_likwid()
        if bw is not None:
            tool_used = "likwid"

    if bw is None and PerfCounterInterface.is_perf_available():
        bw = _measure_memory_bandwidth_perf()
        if bw is not None:
            tool_used = "perf"

    if bw is None and PerfCounterInterface.is_sysfs_available():
        try:
            bw = _measure_memory_bandwidth_sysfs()
            if bw is not None:
                tool_used = "sysfs"
        except Exception as e:
            logger.debug("sysfs bandwidth measurement failed: %s", e)

    if bw is None:
        bw = calculations_from_cpu_frequency()["bw_gb_s"]
        tool_used = "fallback"

    result = round(float(bw), 2)
    logger.info("Memory bandwidth: %.2f GB/s (tool=%s)", result, tool_used)

    if use_cache:
        PerfCounterCache().put(cache_key, result)

    return result

# =============================================================================
# Peak FLOPS Measurement
# =============================================================================

def _measure_peak_flops_likwid(sample_sec: float = 2.0) -> Optional[float]:
    """
    Measure peak FP64 FLOPS using likwid-perfctr FLOPS_DP group.

    Load-driven: an FP64 FMA kernel runs concurrently so FLOPS_DP samples
    real arithmetic, not idle core 0. Kernel FLOPs/time is preferred.
    Returns GFLOPS or None on failure.
    """
    thread, load = _start_load_thread(_run_fma_kernel, sample_sec)
    output = None
    try:
        cmd = ["likwid-perfctr", "-C", "0", "-g", "FLOPS_DP", "-m", "-O", "-S", str(sample_sec)]
        output = _run_cmd_safe(cmd, timeout=int(sample_sec) + 15)
    finally:
        _join_load_thread(thread, sample_sec + 5)

    total = 0.0
    if output:
        for line in output.split("\n"):
            match = re.search(r"(?:DP|FP64)\s+\[?M?FLOPS/s\]?\s+([\d.]+)", line, re.IGNORECASE)
            if match:
                total += float(match.group(1))
            match = re.search(r"([\d.]+)\s+(?:M?FLOPS/s)", line)
            if match and "DP" in line:
                total += float(match.group(1))
    counter_gflops = round(total, 2) if total > 0 else None
    return _prefer_kernel_metric(load.get("gflops"), counter_gflops, "gflops")


def _measure_peak_flops_perf(sample_sec: float = 3.0) -> Optional[float]:
    """
    Measure peak FP64 FLOPS from perf stat fp_arith counters.

    Load-driven: a tight FMA loop runs during the sample window. Idle
    ``perf stat -- sleep`` would count near-zero ``fp_arith_inst_retired``.
    Hardware counters sampled during the kernel are preferred for FLOPS;
    the Python FMA loop is a fallback when counters are unavailable.
    Returns GFLOPS or None on failure.
    """
    thread, load = _start_load_thread(_run_fma_kernel, sample_sec)
    packed_event = "fp_arith_inst_retired.256b_packed_double"
    scalar_event = "fp_arith_inst_retired.scalar_double"
    ops_per_instr = 4
    output = None
    try:
        cmd = ["perf", "stat", "-e", packed_event, "-a", "--", "sleep", str(sample_sec)]
        output = _run_cmd_safe(cmd, timeout=int(sample_sec) + 15)
        if not output:
            ops_per_instr = 1
            cmd = ["perf", "stat", "-e", scalar_event, "-a", "--", "sleep", str(sample_sec)]
            output = _run_cmd_safe(cmd, timeout=int(sample_sec) + 15)
    finally:
        _join_load_thread(thread, sample_sec + 5)

    counter_gflops = None
    if output:
        count = 0
        for line in output.split("\n"):
            match = re.search(r"([\d.,]+)\s+fp_arith", line)
            if match:
                count = int(match.group(1).replace(",", "").replace(".", ""))
                break
        if count > 0:
            counter_gflops = round((count * ops_per_instr) / sample_sec / 1e9, 2)

    return _prefer_kernel_metric(load.get("gflops"), counter_gflops, "gflops")


def calculate_peak_fp64_gflops_fallback() -> float:
    """Fallback theoretical peak FP64 GFLOPS calculation."""
    from .hardware import calculate_peak_fp64_gflops as _hw_peak
    try:
        return _hw_peak()
    except Exception:
        return calculations_from_cpu_frequency()["gflops"]


def measure_peak_flops(use_cache: bool = True) -> float:
    """
    Measure actual peak FP64 FLOPS in GFLOPS.

    Priority:
    1. likwid-perfctr FLOPS_DP group
    2. perf stat fp_arith_inst_retired events
    3. Theoretical peak from hardware.py

    Parameters
    ----------
    use_cache : bool
        If True (default), cache the result in PerfCounterCache (7-day TTL).

    Returns
    -------
    float
        Measured peak FP64 GFLOPS.
    """
    cache_key = "peak_flops_gflops"

    if use_cache:
        cached = PerfCounterCache().get(cache_key)
        if cached is not None and isinstance(cached, (int, float)):
            logger.debug("Returning cached peak FLOPS: %.2f GFLOPS", cached)
            return float(cached)

    gflops: Optional[float] = None
    tool_used = "fallback"

    if PerfCounterInterface.is_likwid_available():
        gflops = _measure_peak_flops_likwid()
        if gflops is not None:
            tool_used = "likwid"

    if gflops is None and PerfCounterInterface.is_perf_available():
        gflops = _measure_peak_flops_perf()
        if gflops is not None:
            tool_used = "perf"

    if gflops is None:
        gflops = calculate_peak_fp64_gflops_fallback()
        tool_used = "fallback"

    result = round(float(gflops), 2)
    logger.info("Peak FLOPS: %.2f GFLOPS (tool=%s)", result, tool_used)

    if use_cache:
        PerfCounterCache().put(cache_key, result)

    return result

# =============================================================================
# Cache Bandwidth Measurement (L1/L2/L3)
# =============================================================================

def _parse_likwid_cache_bw(output: str) -> dict[str, float]:
    """Parse likwid MEM_DP output for L1/L2/L3 bandwidth values."""
    result: dict[str, float] = {}
    for line in output.split("\n"):
        for cache_name, key in [("L1", "l1"), ("L2", "l2"), ("L3", "l3")]:
            match = re.search(
                rf"{cache_name}\s+(?:data\s+)?(?:cache\s+)?bandwidth\s+\[GB/s\]\s+([\d.]+)",
                line, re.IGNORECASE
            )
            if match:
                result[key] = float(match.group(1))
                break
    return result


def _perf_cache_events(sample_sec: float = 2.0) -> dict[str, float]:
    """
    Estimate L1/L2/L3 cache bandwidth from perf stat cache counters.

    Load-driven: sized L1/L2/L3 access loops run during the sample window
    so cache events are from a real working set, not idle sleep.
    Kernel bandwidth is preferred; counters fill any missing levels.
    Returns dict with keys "l1", "l2", "l3" in GB/s.
    """
    result: dict[str, float] = {}
    thread, load = _start_load_thread(_run_cache_hierarchy_kernel, sample_sec)
    events = [
        "L1-dcache-loads",
        "L1-dcache-load-misses",
        "LLC-loads",
        "LLC-load-misses",
    ]
    output = None
    try:
        cmd = ["perf", "stat", "-e", ",".join(events), "-a", "--", "sleep", str(sample_sec)]
        output = _run_cmd_safe(cmd, timeout=int(sample_sec) + 15)
    finally:
        _join_load_thread(thread, sample_sec + 5)

    for key in ("l1", "l2", "l3"):
        val = load.get(key)
        if isinstance(val, (int, float)) and val > 0:
            result[key] = round(float(val), 2)

    if not output:
        return result

    loads = {"L1": 0, "L1_misses": 0, "LLC": 0, "LLC_misses": 0}
    for line in output.split("\n"):
        for key, pattern in [
            ("L1", r"([\d.,]+)\s+L1-dcache-loads\b"),
            ("L1_misses", r"([\d.,]+)\s+L1-dcache-load-misses\b"),
            ("LLC", r"([\d.,]+)\s+LLC-loads\b"),
            ("LLC_misses", r"([\d.,]+)\s+LLC-load-misses\b"),
        ]:
            match = re.search(pattern, line)
            if match:
                loads[key] = int(match.group(1).replace(",", "").replace(".", ""))

    l1_hits = max(0, loads["L1"] - loads["L1_misses"])
    l1_accesses = loads["L1"]
    if l1_accesses > 0:
        result.setdefault("l1", round((l1_accesses * 64.0) / sample_sec / 1e9, 2))
        result.setdefault("l1_hit", round((l1_hits * 64.0) / sample_sec / 1e9, 2))
    if loads["L1_misses"] > 0:
        result.setdefault("l2", round((loads["L1_misses"] * 64.0) / sample_sec / 1e9, 2))
    if loads["LLC"] > 0:
        result.setdefault("l3", round((loads["LLC"] * 64.0) / sample_sec / 1e9, 2))

    return result


def measure_cache_bandwidth(use_cache: bool = True) -> dict[str, float]:
    """
    Measure L1/L2/L3 cache bandwidths in GB/s.

    Uses likwid MEM_DP group or perf stat cache events.
    These values are needed for accurate operational intensity calculations
    in the roofline model.

    Parameters
    ----------
    use_cache : bool
        If True (default), cache the result in PerfCounterCache (7-day TTL).

    Returns
    -------
    dict
        Keys: "l1", "l2", "l3" with values in GB/s.
    """
    cache_key = "cache_bw"

    if use_cache:
        cached = PerfCounterCache().get(cache_key)
        if cached is not None and isinstance(cached, dict):
            logger.debug("Returning cached cache bandwidths")
            return cached

    result: dict[str, float] = {}
    tool_used = "none"

    if PerfCounterInterface.is_likwid_available():
        thread, load = _start_load_thread(_run_cache_hierarchy_kernel, 2.0)
        output = None
        try:
            cmd = ["likwid-perfctr", "-C", "0", "-g", "MEM_DP", "-m", "-O", "-S", "2"]
            output = _run_cmd_safe(cmd, timeout=20)
        finally:
            _join_load_thread(thread, 7.0)
        parsed = _parse_likwid_cache_bw(output) if output else {}
        for key in ("l1", "l2", "l3"):
            kbw = load.get(key)
            if isinstance(kbw, (int, float)) and kbw > 0:
                parsed[key] = round(float(kbw), 2)
        if parsed:
            result = parsed
            tool_used = "likwid"

    if not result and PerfCounterInterface.is_perf_available():
        result = _perf_cache_events()
        if result:
            tool_used = "perf"

    if not result:
        # Conservative fallback estimates
        logger.debug("Cache bandwidth measurement unavailable; using fallback estimates")
        result = {"l1": 500.0, "l2": 200.0, "l3": 80.0}
        tool_used = "fallback"

    logger.info("Cache bandwidth (tool=%s): L1=%.1f L2=%.1f L3=%.1f GB/s",
                 tool_used,
                 result.get("l1", 0.0),
                 result.get("l2", 0.0),
                 result.get("l3", 0.0))

    if use_cache:
        PerfCounterCache().put(cache_key, result)

    return result

# =============================================================================
# Combined Roofline Data
# =============================================================================

def get_real_roofline_data(use_cache: bool = True) -> dict[str, Any]:
    """
    Combine measured peak FLOPS and sustained memory bandwidth into a single
    roofline data dict. Cached for 7 days by default (fingerprint-invalidated).

    Returns
    -------
    dict
        Keys: "peak_flops_gflops", "sustained_bw_gb_s", "measured_date",
        "tool_used", "cache_bandwidth", "has_perf_counters".
    """
    cache_key = "roofline_data"

    if use_cache:
        cached = PerfCounterCache().get(cache_key)
        if cached is not None and isinstance(cached, dict):
            logger.debug("Returning cached roofline data")
            return cached

    maybe_notify_calibration(cached=False)
    peak = measure_peak_flops(use_cache=False)
    bw = measure_memory_bandwidth(use_cache=False)

    active_tools = PerfCounterInterface.list_available_tools()
    tool_used = active_tools[0] if active_tools else "fallback"

    cache_bw: dict[str, float] = {}
    try:
        cache_bw = measure_cache_bandwidth(use_cache=False)
    except Exception as e:
        logger.debug("Cache bandwidth measurement failed in roofline: %s", e)

    result = {
        "peak_flops_gflops": peak,
        "sustained_bw_gb_s": bw,
        "measured_date": datetime.now().isoformat(),
        "tool_used": tool_used,
        "cache_bandwidth": cache_bw,
        "has_perf_counters": HAS_PERF_COUNTERS,
    }

    if use_cache:
        PerfCounterCache().put(cache_key, result)

    return result

# =============================================================================
# Explicit Public API Declaration
# =============================================================================

def maybe_notify_calibration(cached: bool) -> None:
    """Log the one-shot calibration notice when a fresh measurement will run."""
    if not cached:
        logger.info(_CALIBRATION_NOTICE)


def invalidate_perf_cache() -> None:
    """Clear ~/.forge/perf_cache.json so the next measurement is fresh."""
    PerfCounterCache().invalidate()


__all__ = [
    "HAS_PERF_COUNTERS",
    "PerfCounterCache",
    "PerfCounterInterface",
    "_CALIBRATION_NOTICE",
    "calculations_from_cpu_frequency",
    "get_real_roofline_data",
    "invalidate_perf_cache",
    "measure_cache_bandwidth",
    "measure_memory_bandwidth",
    "measure_peak_flops",
]
