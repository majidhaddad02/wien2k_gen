"""Latin Hypercube Sampling and parameter encoding/decoding."""

from typing import Any, Optional

import numpy as np

_CATEGORICAL_MODES = ["kpoint", "hybrid", "mpi"]


def latin_hypercube_sampling(
    bounds: list[tuple[float, float]],
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Latin Hypercube Sampling for uniform coverage of the search space.

    Divides each dimension into n_samples equal intervals and samples
    one point randomly from each interval, then shuffles to produce
    uncorrelated samples.

    Preferred over random sampling for BO initialisation as it guarantees
    coverage of the entire parameter space (McKay et al. 1979).

    Args:
        bounds: [(low, high), ...] per dimension.
        n_samples: Number of samples to generate.
        random_state: Seed for reproducibility.

    Returns:
        Array of shape (n_samples, len(bounds)) with LHS samples.
    """
    rng = np.random.RandomState(random_state)
    dims = len(bounds)
    samples = np.zeros((n_samples, dims), dtype=np.float64)

    for d in range(dims):
        low, high = bounds[d]
        interval_width = (high - low) / n_samples
        for i in range(n_samples):
            lower = low + i * interval_width
            samples[i, d] = lower + rng.uniform(0.0, interval_width)

    for d in range(dims):
        rng.shuffle(samples[:, d])

    return samples


def _encode_config(mode: str, total_cores: int, omp_threads: int) -> np.ndarray:
    """
    Encode a configuration into a numeric feature vector.

    Encoding scheme:
    - total_cores:     normalised by dividing by 256.0
    - omp_threads:     normalised by dividing by 64.0
    - mode:            one-hot encoded (3 categories -> 3 features)

    Args:
        mode: Parallelisation mode ('kpoint', 'hybrid', 'mpi').
        total_cores: Total CPU cores.
        omp_threads: OpenMP threads per rank.

    Returns:
        Feature vector of shape (5,).
    """
    vec = np.zeros(2 + len(_CATEGORICAL_MODES), dtype=np.float64)
    vec[0] = float(total_cores) / 256.0
    vec[1] = float(omp_threads) / 64.0

    if mode in _CATEGORICAL_MODES:
        vec[2 + _CATEGORICAL_MODES.index(mode)] = 1.0

    return vec


def _decode_config(vec: np.ndarray, min_cores: int, max_cores: int) -> dict[str, Any]:
    """
    Decode a feature vector back to a configuration dictionary.

    Returns:
        Dict with keys 'mode', 'total_cores', 'omp_threads'.
    """
    total_cores = max(min_cores, min(max_cores, round(vec[0] * 256.0)))
    omp_threads = max(1, min(64, round(vec[1] * 64.0)))

    one_hot = vec[2:2 + len(_CATEGORICAL_MODES)]
    mode_idx = int(np.argmax(one_hot))
    mode = _CATEGORICAL_MODES[mode_idx]

    return {
        "mode": mode,
        "total_cores": total_cores,
        "omp_threads": omp_threads,
    }


__all__ = ["_CATEGORICAL_MODES", "_decode_config", "_encode_config", "latin_hypercube_sampling"]
