"""Bayesian optimization for WIEN2k SCF parameters."""

from .acquisition import (
    _DEFAULT_EI_THRESHOLD,
    compute_expected_improvement,
    compute_q_expected_improvement,
)
from .bohb import BOHBOptimizer
from .constraints import (
    _estimate_memory_gb_for_config,
    _estimate_walltime_min_for_config,
    _sigmoid_feasibility,
)
from .core import (
    _FIDELITY_CORRELATION,
    _FIDELITY_COST,
    BayesianOptimizer,
    MultiFidelityBayesianOptimizer,
    _params_dict,
    add_physics_priors,
    bayesian_optimize_scf_params,
    define_search_space,
    load_warm_start_history,
    save_bo_history,
)
from .dpp import DPPBatchSelector
from .elements import _chemical_similarity
from .gp import _GaussianProcess, _GaussianProcessARD
from .kernels import _EPS, _NUGGET, matern_kernel, rbf_kernel, rbf_kernel_ard
from .sampling import _CATEGORICAL_MODES, _decode_config, _encode_config, latin_hypercube_sampling

__all__ = [
    "_CATEGORICAL_MODES",
    "_DEFAULT_EI_THRESHOLD",
    "_EPS",
    "_FIDELITY_CORRELATION",
    "_FIDELITY_COST",
    "_NUGGET",
    "BOHBOptimizer",
    "BayesianOptimizer",
    "DPPBatchSelector",
    "MultiFidelityBayesianOptimizer",
    "_GaussianProcess",
    "_GaussianProcessARD",
    "_chemical_similarity",
    "_decode_config",
    "_encode_config",
    "_estimate_memory_gb_for_config",
    "_estimate_walltime_min_for_config",
    "_params_dict",
    "_sigmoid_feasibility",
    "add_physics_priors",
    "bayesian_optimize_scf_params",
    "compute_expected_improvement",
    "compute_q_expected_improvement",
    "define_search_space",
    "latin_hypercube_sampling",
    "load_warm_start_history",
    "matern_kernel",
    "rbf_kernel",
    "rbf_kernel_ard",
    "save_bo_history",
]
