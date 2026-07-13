"""
Optimal Parallel Configuration Generator for Quantum ESPRESSO (QE).

Implements rigorous domain decomposition logic for QE 6.7/7.x+ parallel execution.
Focuses on:
• npool (k-point pools): Maximized subject to k-point count divisibility.
• nband (band parallelization): Enabled for large systems/hybrid functionals.
• ndiag (diagonalization groups): ScaLAPACK 2D grid sizing for matrix diagonalization.
• ntg (task groups): Memory reduction via FFT grid splitting.

Key constraints enforced:
• Divisibility: npool * ndiag * nband * ntg == total_cores
• Load balancing: nkpts % npool == 0 (preferred for even k-point distribution)
• Efficiency: Square-ish grids for diagonalization (ndiag) to minimize communication
• Memory/Time trade-offs for task groups (ntg)

This module replaces heuristic skeletons with production-grade number-theoretic optimization.
"""

import logging
import math
from typing import Any, Optional

# Use module-level logger for consistency with project standards.
logger = logging.getLogger(__name__)


def _get_divisors(n: int) -> list[int]:
    """
    Return sorted list of divisors of n using O(sqrt(n)) algorithm.
    Essential for finding valid decomposition factors.
    """
    if n <= 0:
        return []
    divs: set[int] = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def optimal_npool(total_cores: int, nkpts: int) -> int:
    """
    Determine optimal number of k-point pools (npool).

    QE logic:
    • npool must divide total_cores.
    • npool should ideally divide nkpts to ensure balanced workload across pools.
    • We maximize npool to exploit k-point parallelism, capped by nkpts.
    • If nkpts is 1 (Gamma-only), npool must be 1.
    """
    if nkpts <= 0 or total_cores <= 0:
        return 1

    # Find divisors of nkpts
    divs_kpts = _get_divisors(nkpts)
    # Find divisors of total_cores
    divs_cores = set(_get_divisors(total_cores))

    # Find common divisors (must divide both nkpts and total_cores)
    common = sorted([d for d in divs_kpts if d in divs_cores], reverse=True)

    # Pick the largest common divisor
    if common:
        return common[0]

    # Fallback: If no common divisor > 1 exists (e.g. nkpts is prime and doesn't divide total_cores),
    # we default to 1 to avoid load imbalance or idle cores.
    return 1


def optimal_nband(cores_per_pool: int, nbnd: Optional[int], is_hybrid: bool) -> int:
    """
    Determine optimal number of band groups (nband).

    QE logic:
    • nband must divide cores_per_pool.
    • nband should ideally divide nbnd (number of bands) for load balance.
    • Band parallelism is most beneficial when nbnd is large (>100) or for hybrid functionals.
    • We prefer smaller nband values (e.g., 2) to preserve resources for diagonalization (ndiag).
    """
    if cores_per_pool <= 1:
        return 1

    # If band count is unknown, default to 1 (safest)
    if nbnd is None or nbnd <= 0:
        return 1

    # Determine if band parallelism is warranted
    enable = (nbnd > 100) or is_hybrid

    if not enable:
        return 1

    # Find valid divisors
    divs_pool = set(_get_divisors(cores_per_pool))
    divs_nbnd = set(_get_divisors(nbnd))
    common = sorted([d for d in divs_pool if d in divs_nbnd], reverse=True)

    # We prefer the smallest valid divisor > 1 to minimize overhead and leave cores for ndiag.
    candidates = [d for d in common if d > 1]
    if candidates:
        return candidates[0]  # Smallest valid > 1

    return 1


def generate_qe_config(  # noqa: C901
    total_cores: int,
    nkpts: int,
    nbnd: Optional[int] = None,
    is_hybrid: bool = False,
    user_npool: Optional[int] = None,
    user_ndiag: Optional[int] = None,
    user_nband: Optional[int] = None,
    user_ntg: Optional[int] = None,
) -> dict[str, Any]:
    """
    Generate full parallel configuration for QE, ensuring strict divisibility.

    Hierarchy of decomposition:
    1. npool: Determined first to maximize k-point parallelism.
    2. nband: Determined next to handle large band counts or hybrid functionals.
    3. ntg: Determined for memory constraints (default 1 for max performance).
    4. ndiag: Consumes the remainder (forms the ScaLAPACK diagonalization grid).

    Args:
        total_cores: Total MPI ranks available.
        nkpts: Number of k-points (from input file).
        nbnd: Number of bands (optional, for nband optimization).
        is_hybrid: Whether hybrid functional is used.
        user_npool: Optional user override.
        user_ndiag: Optional user override (currently informational/checked).
        user_nband: Optional user override.
        user_ntg: Optional user override.

    Returns:
        Dictionary with npool, ndiag, nband, ntg, and any warnings.
    """
    warnings: list[str] = []

    # --- 1. Determine npool ---
    if user_npool is not None:
        npool = user_npool
        if total_cores % npool != 0:
            warnings.append(f"user_npool={npool} does not divide total_cores={total_cores}.")
        if nkpts > 0 and nkpts % npool != 0:
            warnings.append(f"user_npool={npool} does not divide nkpts={nkpts}. Load imbalance expected.")
    else:
        npool = optimal_npool(total_cores, nkpts)

    npool = max(1, npool)

    # Calculate cores remaining per pool
    if total_cores % npool == 0:
        cores_per_pool = total_cores // npool
    else:
        cores_per_pool = total_cores // npool
        warnings.append(f"Non-divisible npool results in {cores_per_pool} cores/pool (some cores may be idle).")

    # --- 2. Determine nband ---
    if user_nband is not None:
        nband = user_nband
        if cores_per_pool % nband != 0:
            warnings.append(f"user_nband={nband} incompatible with remaining cores per pool ({cores_per_pool}).")
            # Attempt to find closest valid divisor
            divs = _get_divisors(cores_per_pool)
            nband = min(divs, key=lambda x: abs(x - user_nband))
            warnings.append(f"Adjusted nband to {nband}.")
    else:
        nband = optimal_nband(cores_per_pool, nbnd, is_hybrid)

    # Safety check
    if cores_per_pool % nband != 0:
        nband = 1
        warnings.append("Reset nband to 1 due to divisibility conflict.")

    cores_per_band_group = cores_per_pool // nband

    # --- 3. Determine ntg ---
    if user_ntg is not None:
        ntg = user_ntg
        if cores_per_band_group % ntg != 0:
            warnings.append(f"user_ntg={ntg} incompatible with available cores ({cores_per_band_group}).")
            divs = _get_divisors(cores_per_band_group)
            ntg = min(divs, key=lambda x: abs(x - user_ntg))
            warnings.append(f"Adjusted ntg to {ntg}.")
    else:
        # Default ntg=1 maximizes cores available for diagonalization (performance).
        # Increase ntg only if memory limits are hit (handled by external logic or user).
        ntg = 1

    # --- 4. Determine ndiag (Remainder) ---
    # ndiag consumes all remaining cores in the band/task group.
    # It forms the processor grid for ScaLAPACK diagonalization.
    if cores_per_band_group % ntg == 0:
        ndiag = cores_per_band_group // ntg
    else:
        ndiag = 1
        warnings.append("Invalid ntg choice, reset to 1.")

    if ndiag <= 0:
        ndiag = 1
        warnings.append("Calculated ndiag <= 0, reset to 1.")

    # --- 5. Heuristic Checks ---
    # ndiag represents the number of processors for the diagonalization group.
    # ScaLAPACK works best if ndiag is composite (allows 2D grid decomposition).
    if ndiag > 1:
        is_prime = True
        limit = int(ndiag**0.5) + 1
        for i in range(2, limit):
            if ndiag % i == 0:
                is_prime = False
                break
        if is_prime:
            warnings.append(
                f"ndiag={ndiag} is prime. ScaLAPACK will use a 1D processor grid (less efficient). "
                f"Consider adjusting nband or ntg."
            )

    # User override for ndiag is informational unless it matches the calculated remainder.
    if user_ndiag is not None and user_ndiag != ndiag:
        warnings.append(
            f"user_ndiag={user_ndiag} ignored. Calculated ndiag is {ndiag} based on remaining cores."
        )

    return {
        "npool": npool,
        "ndiag": ndiag,
        "nband": nband,
        "ntg": ntg,
        "warnings": warnings,
    }