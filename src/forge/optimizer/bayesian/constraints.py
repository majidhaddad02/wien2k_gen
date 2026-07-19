"""Physics-based constraint estimation for Bayesian optimization."""

import math


def _estimate_memory_gb_for_config(nmat: int, total_cores: int, omp_threads: int = 1) -> float:
    """
    Per-rank memory estimate for a given MPI/OpenMP configuration.

    In ScaLAPACK/ELPA block-cyclic distribution, the Hamiltonian matrix
    is distributed across MPI ranks (NOT total cores), so per-rank memory
    scales as nmat²/ranks. In hybrid mode, ranks = total_cores // omp_threads.

    Args:
        nmat: Hamiltonian matrix size.
        total_cores: Total CPU cores (MPI ranks x OpenMP threads).
        omp_threads: OpenMP threads per MPI rank (1 for pure MPI/kpoint).

    Returns:
        Estimated per-rank memory in GB.
    """
    ranks = max(1, total_cores // max(1, omp_threads))
    # Aggregate matrix memory, then divide by ranks for block-cyclic distribution
    aggregate_gb = (float(nmat) ** 2.0) * 16.0 / (1024.0 ** 3)
    per_rank_gb = aggregate_gb / float(ranks)
    # Small per-rank overhead for communication buffers + replicated data
    comm_overhead = 0.5  # GB per rank for MPI buffers, charge density copies
    safety = 1.5  # Per-rank safety factor (was 3.0x on aggregate)
    return (per_rank_gb + comm_overhead) * safety


def _estimate_walltime_min_for_config(nmat: int, nkpt: int, total_cores: int) -> float:
    """
    Rough walltime estimate for a given config using empirical scaling.

    Args:
        nmat: Hamiltonian matrix size.
        nkpt: Number of k-points.
        total_cores: Total CPU cores.

    Returns:
        Estimated walltime in minutes.
    """
    baseline_work = (float(nmat) ** 3.0) * float(max(1, nkpt))
    baseline_time_sec = baseline_work / (5000.0 ** 3.0 * 4.0) * 3600.0
    effective_cores = float(max(1, total_cores))
    return baseline_time_sec / effective_cores / 60.0


def _sigmoid_feasibility(estimated: float, max_allowed: float, slope: float = 10.0) -> float:
    """
    Soft feasibility probability via sigmoid.

    P(feasible) ≈ 1 / (1 + exp(slope * (estimated/max_allowed - 1)))

    Args:
        estimated: Estimated resource consumption.
        max_allowed: Maximum resource budget.
        slope: Steepness of the transition.

    Returns:
        Probability between 0 and 1.
    """
    if max_allowed <= 0:
        return 0.0
    ratio = estimated / max_allowed
    return float(1.0 / (1.0 + math.exp(slope * (ratio - 1.0))))


__all__ = ["_estimate_memory_gb_for_config", "_estimate_walltime_min_for_config", "_sigmoid_feasibility"]
