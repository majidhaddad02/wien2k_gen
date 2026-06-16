"""
Synthetic Benchmark & Workload Simulation Module.
Provides robust tools for generating synthetic performance data,
simulating DFT workloads (WIEN2k/QE/VASP), and validating optimization models
(Roofline, Scaling, MPI Overhead) without executing expensive binaries.

Key Architecture Features:
• DFT Workload Simulation: Models complexity of lapw0/lapw1/lapw2 (O(N^3), O(N^2), I/O).
• Roofline Model Integration: Theoretical limits based on Peak FLOPS & Bandwidth.
• MPI/Network Latency Injection: Amdahl's Law & LogP model simulation.
• Noise Generation: Stochastic variance for realistic benchmark data.
• Structured output (`BenchmarkResult`) compatible with `analysis.py` & `optimizer/profiler.py`.
• Strong & Weak Scaling suite generators for automated regression testing.
• Comprehensive English documentation, type hints, and HPC-grade error resilience.

All documentation and inline comments are in English per project standards.
"""

import os
import time
import math
import random
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, TypedDict
from dataclasses import dataclass, field, asdict

# Project imports
from ..core.topology import Topology
from ..core.hardware import (
    get_hardware_profile,
    get_physical_cores,
    get_memory_bandwidth_gb_s,
    calculate_peak_fp64_gflops,
)
from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Type Definitions for Synthetic Data Structures
# =============================================================================

class SyntheticWorkloadParams(TypedDict, total=False):
    """Parameters defining a synthetic DFT problem."""
    atoms: int
    kpoints: int
    nmat: int
    is_hybrid: bool
    is_soc: bool
    complexity_class: str  # 'small', 'medium', 'large', 'exascale'


class BenchmarkResult(TypedDict, total=False):
    """Structured result from a synthetic benchmark run."""
    run_id: str
    problem_params: SyntheticWorkloadParams
    hardware_profile: Dict[str, Any]
    theoretical_time_sec: float
    simulated_time_sec: float
    speedup: float
    efficiency_percent: float
    bottleneck: str  # 'cpu', 'memory', 'io', 'mpi'
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class SimulationConfig:
    """Configuration for the synthetic simulation engine."""
    noise_level: float = 0.05  # 5% variance in timing
    mpi_latency_us: float = 1.5  # InfiniBand default ~1.5us
    mpi_bandwidth_gb_s: float = 100.0
    io_penalty_factor: float = 1.0
    enable_roofline: bool = True


# =============================================================================
# DFT Complexity & Workload Models
# =============================================================================

def _calculate_flop_count(nmat: int, atoms: int, kpoints: int, is_hybrid: bool) -> float:
    """
    Estimate total FLOP count for a DFT calculation.
    Approximations:
    • lapw1 (Hamiltonian): O(N^3) per k-point (dominant diagonalization).
    • lapw0 (Potential): O(N^2).
    • Mixer/Post-processing: O(N^2).
    • Hybrid: Adds expensive exact exchange calculation O(N^3).
    """
    # Base cost for lapw1 (diagonalization)
    # 20 FLOPs per element is a conservative estimate for full diagonalization
    flop_lapw1 = (nmat ** 3) * kpoints * 20.0
    
    # Lapw0 cost scales with grid size (approx proportional to atoms)
    flop_lapw0 = (nmat ** 2) * atoms * 100.0

    total_flops = flop_lapw1 + flop_lapw0

    # Hybrid functionals increase cost significantly (often 5-10x)
    if is_hybrid:
        total_flops *= 5.0
        
    return total_flops


def _calculate_memory_traffic_gb(nmat: int, atoms: int, kpoints: int) -> float:
    """Estimate memory traffic in GB for data movement."""
    # Hamiltonian storage & read/write
    # 16 bytes per element (complex double)
    mat_size_gb = (nmat ** 2) * 16.0 / (1024 ** 3)
    traffic = mat_size_gb * kpoints * 5.0  # Multiple passes over memory
    traffic += (atoms * 0.001) * kpoints  # Atomic data traffic
    return traffic


def _calculate_mpi_messages(nmat: int, cores: int, mode: str) -> int:
    """Estimate number of MPI messages exchanged during parallel run."""
    if mode == "kpoint":
        return kpoints  # Minimal communication (map-reduce)
    if mode == "mpi":
        # ScaLAPACK diagonalization communication ~ O(log P) or O(sqrt P) matrix blocks
        # Heuristic: increases with cores and matrix size
        return int(cores * math.log2(max(2, cores)) * (nmat / 1000.0))
    return cores


# =============================================================================
# Core Simulation Engine
# =============================================================================

class WorkloadSimulator:
    """
    Simulates DFT execution time based on hardware topology and problem size.
    Uses Roofline model and MPI overhead estimation to predict performance.
    """
    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        # Cache hardware profile to avoid repeated syscalls
        self.hw_profile = get_hardware_profile()

    def simulate_run(
        self,
        topo: Topology,
        problem: SyntheticWorkloadParams,
        mode: str = "hybrid"
    ) -> BenchmarkResult:
        """
        Execute synthetic simulation.
        Returns predicted time, bottleneck analysis, and scaling metrics.
        """
        nmat = problem.get("nmat", 1000)
        atoms = problem.get("atoms", 10)
        kpoints = problem.get("kpoints", 1)
        is_hybrid = problem.get("is_hybrid", False)

        # 1. Calculate Workload Characteristics
        total_flops = _calculate_flop_count(nmat, atoms, kpoints, is_hybrid)
        total_traffic_gb = _calculate_memory_traffic_gb(nmat, atoms, kpoints)
        
        # 2. Hardware Limits (from Profile)
        peak_flops = self.hw_profile.get("peak_fp64_gflops", 100.0) * 1e9
        mem_bw = self.hw_profile.get("memory_bandwidth_gb_s", 50.0) * 1e9
        
        # 3. Parallelization Factors
        total_cores = topo.total_cores
        if total_cores == 0: 
            total_cores = 1
        
        # Amdahl's Law / Efficiency scaling
        # Serial fraction decreases with problem size (larger problems parallelize better)
        serial_fraction = 0.05 / (1.0 + math.log10(max(1, nmat)))
        parallel_efficiency = 1.0 / (serial_fraction + (1.0 - serial_fraction) / total_cores)
        
        # 4. Compute Theoretical Time (Roofline)
        if self.config.enable_roofline:
            # Arithmetic Intensity (FLOPs/Byte)
            ai = total_flops / max(1.0, total_traffic_gb) * 1e9
            
            # Crossover point (Hardware limit)
            crossover = mem_bw / peak_flops
            
            if ai < crossover:
                # Memory Bound
                time_sec = (total_traffic_gb * 1e9) / mem_bw
                bottleneck = "memory"
            else:
                # Compute Bound
                time_sec = total_flops / peak_flops
                bottleneck = "cpu"
        else:
            # Simple estimation
            time_sec = total_flops / (peak_flops * total_cores * parallel_efficiency)
            bottleneck = "hybrid"

        # 5. Adjust for MPI/IO Overhead
        # MPI Latency penalty (LogP model approximation)
        msg_count = _calculate_mpi_messages(nmat, total_cores, mode)
        mpi_penalty_sec = (msg_count * self.config.mpi_latency_us) * 1e-6
        time_sec += mpi_penalty_sec

        # I/O Penalty (Lapw2 bottleneck simulation)
        # If memory bound and high core count, I/O contention is likely
        if bottleneck == "memory" and kpoints < total_cores:
            time_sec *= (1.0 + self.config.io_penalty_factor * 0.2)
            bottleneck = "io"

        # 6. Scale by Efficiency
        # In the Roofline model, we calculated single-node or saturated time.
        # We need to divide by cores * efficiency for parallel speedup approximation.
        if total_cores > 1:
            time_sec = time_sec / (total_cores * parallel_efficiency)

        # 7. Add Noise (Stochastic variance)
        noise = 1.0 + random.gauss(0, self.config.noise_level)
        simulated_time = time_sec * max(0.01, noise)  # Ensure non-negative
        
        # Generate Run ID
        run_id = hashlib.md5(f"{nmat}{kpoints}{mode}{total_cores}".encode()).hexdigest()[:8]

        # Construct Result
        result: BenchmarkResult = {
            "run_id": run_id,
            "problem_params": problem,
            "hardware_profile": {k: v for k, v in self.hw_profile.items() if isinstance(v, (str, int, float, bool))},
            "theoretical_time_sec": round(time_sec, 4),
            "simulated_time_sec": round(simulated_time, 4),
            "speedup": 0.0,  # Calculated relative to base later
            "efficiency_percent": round(parallel_efficiency * 100, 2),
            "bottleneck": bottleneck,
            "metadata": {
                "flops_estimated": total_flops,
                "traffic_gb": total_traffic_gb,
                "parallel_efficiency": parallel_efficiency,
                "mpi_penalty_ms": mpi_penalty_sec * 1000
            },
            "timestamp": time.time()
        }
        
        return result


# =============================================================================
# Benchmark Suite Generators
# =============================================================================

def generate_strong_scaling_suite(
    base_problem: SyntheticWorkloadParams,
    topo: Topology,
    max_cores: Optional[int] = None
) -> List[BenchmarkResult]:
    """
    Generate a suite of results simulating strong scaling.
    Fixed problem size, increasing core count.
    """
    sim = WorkloadSimulator()
    results = []
    
    # Test points: Powers of 2 up to max_cores
    cores_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if max_cores:
        cores_to_test = [c for c in cores_to_test if c <= max_cores]
        
    base_time = 0.0

    for n_cores in cores_to_test:
        # Create a sub-topology with fixed node count but reduced cores
        # to simulate scaling on the same hardware
        sub_topo_data = asdict(topo)
        sub_topo_data['total_cores'] = n_cores
        
        # Distribute cores per node (simplified assumption)
        # In reality, this depends on the specific node allocation
        c_per_node = sub_topo_data.get('cores_per_node', [n_cores])
        sub_topo_data['cores_per_node'] = [min(n_cores, c) for c in c_per_node]
        if not sub_topo_data['cores_per_node']:
            sub_topo_data['cores_per_node'] = [n_cores]
            
        # Reconstruct Topology object (assuming dataclass compatibility)
        try:
            sub_topo = Topology(**sub_topo_data)
        except TypeError:
            # Fallback for strict topology classes
            sub_topo = topo
            
        res = sim.simulate_run(sub_topo, base_problem)
        
        if base_time == 0:
            base_time = res["simulated_time_sec"]
        
        # Calculate speedup relative to 1-core (or first) run
        res["speedup"] = round(base_time / max(0.001, res["simulated_time_sec"]), 2)
        results.append(res)
        
    return results


def generate_weak_scaling_suite(
    base_problem: SyntheticWorkloadParams,
    topo: Topology,
    scaling_factor: int = 4
) -> List[BenchmarkResult]:
    """
    Generate a suite simulating weak scaling.
    Increase problem size proportional to core count to maintain load balance.
    """
    sim = WorkloadSimulator()
    results = []
    
    # Simulate 1 node, 2 nodes, 4 nodes...
    node_counts = [1, 2, 4]
    base_nmat = base_problem.get("nmat", 500)

    for nodes in node_counts:
        # Scale problem: Increase atoms/Nmat to keep density constant
        # Nmat scales roughly as Volume^(1/3) or Atoms^(1/3)
        scaled_problem = base_problem.copy()
        scaled_problem["atoms"] = base_problem.get("atoms", 10) * nodes
        scaled_problem["nmat"] = int(base_nmat * (nodes ** 0.33))
        
        res = sim.simulate_run(topo, scaled_problem)
        results.append(res)
        
    return results


# =============================================================================
# Explicit Public API
# =============================================================================

__all__ = [
    "SyntheticWorkloadParams",
    "BenchmarkResult",
    "SimulationConfig",
    "WorkloadSimulator",
    "generate_strong_scaling_suite",
    "generate_weak_scaling_suite",
    "_calculate_flop_count",
    "_calculate_memory_traffic_gb",
]