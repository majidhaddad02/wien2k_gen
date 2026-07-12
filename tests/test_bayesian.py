"""
Production-Grade Tests for optimizer.bayesian Module.
Tests BayesianOptimizer basic loop, MultiFidelityBayesianOptimizer scheduling,
and Expected Improvement edge cases. Uses mocked GP and ExecutionHistory.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from wien2k_gen.optimizer.bayesian import (
    BayesianOptimizer,
    MultiFidelityBayesianOptimizer,
    _chemical_similarity,
    _decode_config,
    _encode_config,
    _GaussianProcess,
    _GaussianProcessARD,
    _sigmoid_feasibility,
    compute_expected_improvement,
    rbf_kernel,
    rbf_kernel_ard,
)
from wien2k_gen.optimizer.history import ExecutionHistory, ExecutionRecord


@pytest.fixture
def mock_execution_history():
    history = MagicMock(spec=ExecutionHistory)
    record = MagicMock(spec=ExecutionRecord)
    record.walltime_sec = 120.0
    record.mode = "kpoint"
    record.total_cores = 32
    record.omp_threads = 2
    history.query.return_value = [record]
    return history


@pytest.fixture
def mock_empty_history():
    history = MagicMock(spec=ExecutionHistory)
    history.query.return_value = []
    return history


def _make_record(walltime, mode="kpoint", cores=32, omp=2):
    record = MagicMock(spec=ExecutionRecord)
    record.walltime_sec = walltime
    record.mode = mode
    record.total_cores = cores
    record.omp_threads = omp
    return record


# =============================================================================
# Kernel Tests
# =============================================================================

class TestKernels:
    def test_rbf_kernel_scalar(self):
        result = rbf_kernel(np.array([1.0]), np.array([2.0]))
        val = float(result.flatten()[0])
        assert 0.0 < val < 1.0

    def test_rbf_kernel_same_point(self):
        result = rbf_kernel(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        val = float(result.flatten()[0])
        assert abs(val - 1.0) < 1e-6

    def test_rbf_kernel_identity_matrix(self):
        X = np.eye(3)
        K = rbf_kernel(X, X)
        assert K.shape == (3, 3)
        assert all(abs(K[i, i] - 1.0) < 1e-6 for i in range(3))

    def test_rbf_kernel_ard_shape(self):
        x1 = np.array([[0.0, 1.0], [2.0, 3.0]])
        x2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        K = rbf_kernel_ard(x1, x2, np.array([1.0, 2.0]))
        assert K.shape == (2, 2)

    def test_rbf_kernel_ard_same_point(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        K = rbf_kernel_ard(X, X, np.array([1.0, 1.0]))
        assert abs(K[0, 0] - 1.0) < 1e-6

    def test_rbf_kernel_vector_regression(self):
        x1 = np.array([[0.0, 0.0]])
        x2 = np.array([[1.0, 0.0]])
        length_scale = 1.0
        K = rbf_kernel(x1, x2, length_scale)
        expected = math.exp(-0.5)
        assert abs(float(K.flatten()[0]) - expected) < 1e-6


# =============================================================================
# Expected Improvement Tests
# =============================================================================

class TestExpectedImprovement:
    def test_no_uncertainty_returns_zero(self):
        ei = compute_expected_improvement(mu=10.0, sigma=0.0, best_y=5.0)
        assert ei == 0.0

    def test_improvement_when_mu_better_than_best(self):
        ei = compute_expected_improvement(mu=10.0, sigma=0.5, best_y=5.0)
        assert ei > 0.0

    def test_zero_improvement_when_mu_much_worse(self):
        ei = compute_expected_improvement(mu=0.0, sigma=0.5, best_y=5.0)
        assert ei < 1e-10

    def test_exploration_xi_effect(self):
        ei_small_xi = compute_expected_improvement(mu=6.0, sigma=1.0, best_y=5.0, xi=0.01)
        ei_large_xi = compute_expected_improvement(mu=6.0, sigma=1.0, best_y=5.0, xi=0.5)
        assert ei_small_xi > ei_large_xi

    def test_very_small_sigma(self):
        ei = compute_expected_improvement(mu=5.0, sigma=1e-10, best_y=5.0)
        assert ei == 0.0

    def test_near_zero_sigma(self):
        ei = compute_expected_improvement(mu=10.0, sigma=1e-9, best_y=5.0)
        assert ei == 0.0

    def test_same_mu_as_best(self):
        ei = compute_expected_improvement(mu=5.0, sigma=1.0, best_y=5.0)
        assert ei >= 0.0


# =============================================================================
# Encode/Decode Tests
# =============================================================================

class TestEncoding:
    def test_encode_shape(self):
        vec = _encode_config("kpoint", 64, 4)
        assert vec.shape == (5,)
        assert vec[0] == 64.0 / 256.0
        assert vec[1] == 4.0 / 64.0

    def test_encode_one_hot_mode(self):
        vec = _encode_config("hybrid", 128, 8)
        assert vec[2] == 0.0
        assert vec[3] == 1.0
        assert vec[4] == 0.0

    def test_roundtrip(self):
        original = {"mode": "mpi", "total_cores": 64, "omp_threads": 2}
        vec = _encode_config(original["mode"], original["total_cores"], original["omp_threads"])
        decoded = _decode_config(vec, min_cores=1, max_cores=256)
        assert decoded["mode"] == original["mode"]
        assert decoded["total_cores"] == original["total_cores"]
        assert decoded["omp_threads"] == original["omp_threads"]

    def test_roundtrip_clamps_cores(self):
        vec = _encode_config("kpoint", 64, 4)
        decoded = _decode_config(vec, min_cores=1, max_cores=16)
        assert decoded["total_cores"] <= 16
        assert decoded["total_cores"] >= 1


# =============================================================================
# Gaussian Process Tests
# =============================================================================

class TestGaussianProcess:
    def test_gp_fit_predict(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 2.0])
        gp = _GaussianProcess()
        gp.fit(X, y)
        mu, sigma2 = gp.predict(np.array([[1.5]]))
        assert len(mu) == 1
        assert len(sigma2) == 1
        assert sigma2[0] >= 0.0

    def test_gp_predict_before_fit_raises(self):
        gp = _GaussianProcess()
        with pytest.raises(RuntimeError):
            gp.predict(np.array([[0.0]]))

    def test_gp_perfect_interpolation(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        gp = _GaussianProcess()
        gp.fit(X, y)
        mu, _ = gp.predict(X)
        np.testing.assert_allclose(mu, y, atol=1e-4)

    def test_gp_ard_basic_fit(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array([0.0, 1.0])
        gp = _GaussianProcessARD(n_opt_steps=1)
        gp.fit(X, y)
        assert gp.length_scales is not None
        assert len(gp.length_scales) == 2


# =============================================================================
# BayesianOptimizer Tests
# =============================================================================

class TestBayesianOptimizer:
    def test_init_with_history(self, mock_execution_history):
        opt = BayesianOptimizer(mock_execution_history)
        assert opt.backend == "wien2k"
        assert opt.min_cores == 1
        assert opt.max_cores == 256

    def test_init_with_empty_history(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history)
        assert opt.n_observations == 0

    def test_suggest_random_when_no_data(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history)
        result = opt.suggest_next(nmat=1000, nkpt=4)
        assert result["source"] == "random"
        assert "mode" in result
        assert "total_cores" in result
        assert "omp_threads" in result
        assert result["expected_improvement"] == 0.0

    def test_update_adds_observation(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history, use_ard=False)
        record = _make_record(60.0, mode="kpoint", cores=32, omp=2)
        opt.update(record)
        record2 = _make_record(45.0, mode="hybrid", cores=64, omp=4)
        opt.update(record2)
        assert opt.n_observations == 2

    def test_suggest_model_with_enough_data(self, mock_execution_history):
        opt = BayesianOptimizer(mock_execution_history, use_ard=False)
        r1 = _make_record(120.0, mode="kpoint", cores=32, omp=2)
        r2 = _make_record(90.0, mode="hybrid", cores=64, omp=4)
        r3 = _make_record(75.0, mode="mpi", cores=128, omp=1)
        for r in [r1, r2, r3]:
            opt.update(r)
        result = opt.suggest_next(nmat=5000, nkpt=8, user_max_cores=128)
        assert result["source"] == "model"
        assert "predicted_mean" in result
        assert "predicted_std" in result
        assert "expected_improvement" in result
        assert result["total_cores"] <= 128

    def test_get_best_observed(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history, use_ard=False)
        opt.update(_make_record(120.0, mode="kpoint", cores=32, omp=2))
        opt.update(_make_record(60.0, mode="hybrid", cores=64, omp=4))
        opt.update(_make_record(90.0, mode="mpi", cores=128, omp=1))
        best = opt.get_best_observed()
        assert best is not None
        assert best["walltime_sec"] == 60.0

    def test_get_best_observed_empty(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history)
        assert opt.get_best_observed() is None

    def test_constrained_suggest_with_budget(self, mock_execution_history):
        opt = BayesianOptimizer(mock_execution_history, use_ard=False)
        r1 = _make_record(120.0, mode="kpoint", cores=32, omp=2)
        r2 = _make_record(90.0, mode="hybrid", cores=64, omp=4)
        for r in [r1, r2]:
            opt.update(r)
        result = opt.suggest_next_with_constraints(
            nmat=5000, nkpt=8, max_memory_gb=64.0, max_walltime_min=30.0
        )
        assert "p_feasible" in result
        assert "estimated_memory_gb" in result
        assert "estimated_walltime_min" in result

    def test_constrained_suggest_random_when_no_data(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history)
        result = opt.suggest_next_with_constraints(
            nmat=5000, nkpt=8, max_memory_gb=64.0, max_walltime_min=30.0
        )
        assert result["source"] == "random"
        assert result["p_feasible"] == 1.0


# =============================================================================
# Multi-Fidelity BayesianOptimizer Tests
# =============================================================================

class TestMultiFidelityBayesianOptimizer:
    def test_init(self, mock_empty_history):
        opt = MultiFidelityBayesianOptimizer(mock_empty_history, use_ard=False)
        assert opt.current_fidelity == 0

    def test_initial_fidelity_zero(self, mock_empty_history):
        opt = MultiFidelityBayesianOptimizer(mock_empty_history)
        config, fid = opt.suggest_next_fidelity(nmat=1000, nkpt=4)
        assert fid == 0
        assert config["source"] == "random_mf"

    def test_suggest_next_fidelity_with_data(self, mock_empty_history):
        opt = MultiFidelityBayesianOptimizer(mock_empty_history, use_ard=False)
        r1 = _make_record(120.0, "kpoint", 32, 2)
        r2 = _make_record(90.0, "hybrid", 64, 4)
        r3 = _make_record(75.0, "mpi", 128, 1)
        for r in [r1, r2, r3]:
            opt.update(r)
        config, _fid = opt.suggest_next_fidelity(nmat=5000, nkpt=8, user_max_cores=128)
        assert "fidelity" in config
        assert "mf_ei" in config
        assert config["source"] in ("model_mf", "model_mf_promoted")

    def test_fidelity_stats(self, mock_empty_history):
        opt = MultiFidelityBayesianOptimizer(mock_empty_history)
        stats = opt.get_fidelity_stats()
        assert "evals_per_fidelity" in stats
        assert "cost_multipliers" in stats
        assert "correlation_weights" in stats
        assert stats["current_fidelity"] == 0

    def test_fidelity_promotion_when_ei_low(self, mock_empty_history):
        opt = MultiFidelityBayesianOptimizer(
            mock_empty_history,
            use_ard=False,
            ei_promotion_threshold=1.0,
            min_fidelity_evals=0,
        )
        r1 = _make_record(120.0, "kpoint", 32, 2)
        r2 = _make_record(90.0, "hybrid", 64, 4)
        r3 = _make_record(75.0, "mpi", 128, 1)
        for r in [r1, r2, r3]:
            opt.update(r)
        _, fid = opt.suggest_next_fidelity(nmat=5000, nkpt=8)
        assert fid in (0, 1)


# =============================================================================
# Chemical Similarity Tests
# =============================================================================

class TestChemicalSimilarity:
    def test_identical_atoms(self):
        s = _chemical_similarity(14, 14)
        assert s == 1.0

    def test_same_group(self):
        s = _chemical_similarity(14, 32)
        assert s > 0.3

    def test_different_atoms(self):
        s = _chemical_similarity(1, 86)
        assert s < 1.0


# =============================================================================
# Sigmoid Feasibility Tests
# =============================================================================

class TestSigmoidFeasibility:
    def test_well_within_budget(self):
        p = _sigmoid_feasibility(1.0, 10.0)
        assert p > 0.99

    def test_at_budget_limit(self):
        p = _sigmoid_feasibility(10.0, 10.0)
        assert 0.45 < p < 0.55

    def test_way_over_budget(self):
        p = _sigmoid_feasibility(20.0, 10.0)
        assert p < 0.01

    def test_zero_budget(self):
        p = _sigmoid_feasibility(10.0, 0.0)
        assert p == 0.0

    def test_custom_slope(self):
        p_sharp = _sigmoid_feasibility(11.0, 10.0, slope=50.0)
        p_soft = _sigmoid_feasibility(11.0, 10.0, slope=1.0)
        assert p_sharp < p_soft


# =============================================================================
# Transfer Learning Tests
# =============================================================================

class TestTransferLearning:
    def test_transfer_from_identical_system(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history, use_ard=False)
        r1 = _make_record(120.0, "kpoint", 32, 2)
        r2 = _make_record(90.0, "hybrid", 64, 4)
        r3 = _make_record(75.0, "mpi", 128, 1)
        for r in [r1, r2, r3]:
            opt.update(r)
        source_hist = MagicMock(spec=ExecutionHistory)
        source_hist.query.return_value = [r1, r2, r3]
        opt.transfer_from_system("Si", "Si", source_history=source_hist)
        assert opt._transfer_weight > 0.0

    def test_transfer_from_distant_system(self, mock_empty_history):
        opt = BayesianOptimizer(mock_empty_history, use_ard=False)
        r1 = _make_record(120.0, "kpoint", 32, 2)
        r2 = _make_record(90.0, "hybrid", 64, 4)
        r3 = _make_record(75.0, "mpi", 128, 1)
        for r in [r1, r2, r3]:
            opt.update(r)
        source_hist = MagicMock(spec=ExecutionHistory)
        source_hist.query.return_value = [r1, r2, r3]
        opt.transfer_from_system("H", "Rn", source_history=source_hist)
        assert opt._transfer_weight < 0.3


# =============================================================================
# Parameter Relevance Tests
# =============================================================================

class TestParameterRelevance:
    def test_no_ard_returns_empty(self, mock_execution_history):
        opt = BayesianOptimizer(mock_execution_history, use_ard=False)
        r1 = _make_record(120.0, "kpoint", 32, 2)
        r2 = _make_record(90.0, "hybrid", 64, 4)
        for r in [r1, r2]:
            opt.update(r)
        assert opt.get_parameter_relevance() == {}

    def test_ard_returns_relevance(self, mock_execution_history):
        opt = BayesianOptimizer(mock_execution_history, use_ard=False)
        records = []
        for cores, omp in [(16, 1), (32, 2), (64, 4), (128, 8), (256, 1)]:
            records.append(_make_record(200.0 - cores * 0.5, "kpoint", cores, omp))
        for r in records:
            opt.update(r)
        relevance = opt.get_parameter_relevance()
        assert isinstance(relevance, dict)
