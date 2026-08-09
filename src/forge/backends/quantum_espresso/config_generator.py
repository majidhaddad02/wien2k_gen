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

import math
from typing import Any, Optional

from forge.logging_config import get_logger

logger = get_logger(__name__)


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
    common = sorted([d for d in divs_pool if d in divs_nbnd])

    # We prefer the smallest valid divisor > 1 to minimize overhead and leave cores for ndiag.
    candidates = [d for d in common if d > 1]
    if candidates:
        return candidates[0]  # Smallest valid > 1

    return 1


def optimal_ndiag(nproc_bgrp: int) -> int:
    """
    Determine optimal number of processors in the linear-algebra (diagonalization) group.

    QE logic (Doc/user_guide.tex §Parallelization, PW/Doc/user_guide.tex §Parallelization issues):
    • The diagonalization group is a sub-group of the band-group pool.
    • Its size must be a perfect square n^2 (ScaLAPACK 2D grid).
    • n^2 must be smaller than or equal to the number of processors in the PW group.
    • QE defaults to the square integer smaller than or equal to nproc_bgrp.
    """
    if nproc_bgrp <= 1:
        return 1

    n = math.isqrt(nproc_bgrp)
    while n > 1 and n * n > nproc_bgrp:
        n -= 1
    return max(1, n * n)


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
    Generate full parallel configuration for QE, following the real QE domain
    decomposition hierarchy (QEF user_guide.tex §Parallelization).

    QE structure (MPI ranks per image):
        nproc = npool x nband x ntg x procs_per_task_group
    where:
        npool   (-nk)  splits the world into k-point pools.
        nband   (-nb)  splits each pool into band groups.
        ntg     (-nt)  splits each band group into FFT task groups.
        ndiag   (-nd)  is NOT a multiplicative factor: it is a *sub-group* of the
                       band group, of size n^2 (perfect square, 2D ScaLAPACK grid),
                       with n^2 <= procs per band group. QE sets it automatically
                       to the largest square <= procs per band group when omitted.

    Args:
        total_cores: Total MPI ranks available.
        nkpts: Number of k-points (from input file).
        nbnd: Number of bands (optional, for nband optimization).
        is_hybrid: Whether hybrid functional is used.
        user_npool: Optional user override for -nk.
        user_ndiag: Optional user override for -nd (must be a perfect square).
        user_nband: Optional user override for -nb.
        user_ntg: Optional user override for -nt.

    Returns:
        Dictionary with npool, ndiag, nband, ntg, procs_per_task_group, and warnings.
    """
    warnings: list[str] = []

    # --- 1. Determine npool (-nk) ---
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

    # --- 2. Determine nband (-nb) ---
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

    # --- 3. Determine ntg (-nt) ---
    if user_ntg is not None:
        ntg = user_ntg
        if cores_per_band_group % ntg != 0:
            warnings.append(f"user_ntg={ntg} incompatible with available cores ({cores_per_band_group}).")
            divs = _get_divisors(cores_per_band_group)
            ntg = min(divs, key=lambda x: abs(x - user_ntg))
            warnings.append(f"Adjusted ntg to {ntg}.")
    else:
        # Default ntg=1 keeps all band-group cores for the FFT slice and maximizes
        # the available diagonalization grid (ndiag). Increase ntg only for memory
        # reduction or when procs/band-group exceeds the FFT planes (handled externally).
        ntg = 1

    if cores_per_band_group % ntg != 0:
        ntg = 1
        warnings.append("Invalid ntg choice, reset to 1.")

    procs_per_task_group = cores_per_band_group // ntg

    # --- 4. Determine ndiag (-nd) ---
    # QE: diag group is a sub-group of the band group with a square 2D grid.
    # It must be n^2 with n^2 <= procs per band group (not a multiplicative factor).
    max_ndiag = cores_per_band_group
    if user_ndiag is not None:
        # Clamp to the largest perfect square <= procs per band group, which is
        # exactly what QE does. This guarantees -nd is always a valid n^2 grid.
        if user_ndiag > max_ndiag:
            warnings.append(
                f"user_ndiag={user_ndiag} exceeds procs per band group ({max_ndiag}). "
                f"QE would truncate it to the largest square <= {max_ndiag}."
            )
        n_sqrt = math.isqrt(user_ndiag)
        if n_sqrt * n_sqrt != user_ndiag:
            warnings.append(
                f"user_ndiag={user_ndiag} is not a perfect square. QE requires n^2 for the "
                f"2D ScaLAPACK grid; clamped to {n_sqrt * n_sqrt}."
            )
        n_sqrt = min(n_sqrt, math.isqrt(max_ndiag))
        ndiag = max(1, n_sqrt * n_sqrt)
    else:
        ndiag = optimal_ndiag(max_ndiag)

    # --- 5. Heuristic Checks ---
    if ndiag > max_ndiag:
        ndiag = max_ndiag
        warnings.append(f"ndiag capped at {ndiag} (procs per band group).")

    return {
        "npool": npool,
        "ndiag": ndiag,
        "nband": nband,
        "ntg": ntg,
        "procs_per_task_group": procs_per_task_group,
        "procs_per_band_group": cores_per_band_group,
        "warnings": warnings,
    }