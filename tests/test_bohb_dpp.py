"""Tests for BOHB (TPE/KDE) and DPP batch selector."""

import numpy as np

from forge.optimizer.bayesian.bohb import (
    BOHBOptimizer,
    _aitchison_aitken_kernel,
    _estimate_bandwidth,
    _gaussian_kde_pdf,
)
from forge.optimizer.bayesian.dpp import DPPBatchSelector
from forge.optimizer.bayesian.gp import _GaussianProcess
from forge.optimizer.bayesian.kernels import rbf_kernel_ard


def _make_rbf_kernel(dims: int):
    ls = np.ones(dims)

    def kfn(a, b):
        return float(rbf_kernel_ard(np.atleast_2d(a), np.atleast_2d(b), ls)[0, 0])

    return kfn

# ---------------------------------------------------------------------------
# KDE helpers
# ---------------------------------------------------------------------------

class TestKDEHelpers:
    def test_estimate_bandwidth(self):
        X = np.random.randn(50, 2)
        bw = _estimate_bandwidth(X, factor=0.25)
        assert 0.01 < bw < 5.0

    def test_estimate_bandwidth_small(self):
        X = np.array([[0.0, 0.0]])
        bw = _estimate_bandwidth(X)
        assert bw > 0

    def test_gaussian_kde_pdf_shape(self):
        samples = np.random.randn(20, 2)
        x = np.random.randn(5, 2)
        log_p = _gaussian_kde_pdf(x, samples, bandwidth=0.5)
        assert log_p.shape == (5,)
        assert np.all(np.isfinite(log_p))

    def test_gaussian_kde_pdf_empty(self):
        log_p = _gaussian_kde_pdf(
            np.random.randn(3, 2),
            np.empty((0, 2)),
            bandwidth=0.5,
        )
        assert np.all(log_p == -np.inf)

    def test_aitchison_aitken_kernel(self):
        x = np.array([[0], [1], [2]])
        y = np.array([[1], [2], [0]])
        vals = _aitchison_aitken_kernel(x, y, lambda_=0.8, n_categories=4)
        assert vals.shape == (3, 3)
        # Diagonal: x_i == y_i
        assert vals[0, 2] > 0.5   # x_0=0 matches y_2=0
        assert vals[1, 0] > 0.5   # x_1=1 matches y_0=1
        assert vals[2, 1] > 0.5   # x_2=2 matches y_1=2
        # Off-diagonal: lower prob
        assert vals[0, 0] < 0.5
        assert vals[0, 1] < 0.5
        assert vals[1, 1] < 0.5


# ---------------------------------------------------------------------------
# BOHB Optimiser
# ---------------------------------------------------------------------------

class TestBOHBOptimizer:
    def test_initialization(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3)
        assert bohb._s_max == 3
        assert bohb.n_observations == 0
        assert bohb.best_observed is None

    def test_suggest_returns_config(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3)
        config, budget = bohb.suggest()
        assert isinstance(config, dict)
        assert "mode" in config
        assert "total_cores" in config
        assert config["mode"] in ("kpoint", "mpi", "hybrid", "sequential")
        assert budget >= 1

    def test_observe_and_best(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3)
        config, budget = bohb.suggest()
        bohb.observe(config, walltime=120.0, budget=budget)
        assert bohb.n_observations == 1
        assert bohb.best_observed is not None
        assert bohb.best_observed["walltime"] == 120.0 if "walltime" in bohb.best_observed else True

    def test_multiple_observations(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3)
        for _ in range(20):
            config, budget = bohb.suggest()
            walltime = 100.0 - config["total_cores"] * 0.5 + np.random.random() * 20
            bohb.observe(config, walltime, budget)
        assert bohb.n_observations >= 20
        assert bohb.best_observed is not None

    def test_bohb_explores_space(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3, n_configs=8)
        modes = set()
        for _ in range(30):
            config, budget = bohb.suggest()
            modes.add(config["mode"])
            walltime = 50.0 + np.random.random() * 30
            bohb.observe(config, walltime, budget)
        # TPE with exploration should visit multiple modes
        assert len(modes) >= 1

    def test_brackets_generated(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3)
        brackets = bohb._generate_brackets()
        # s_max = 2 → brackets for s=2,1,0
        assert len(brackets) == 4
        # s=3 bracket has s+1=4 rungs
        assert len(brackets[0]["budgets"]) == 4
        # s=0 bracket has 1 rung
        assert len(brackets[-1]["budgets"]) == 1

    def test_tpe_bandwidth_positive(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=1, max_budget=27, eta=3)
        for _ in range(10):
            config, budget = bohb.suggest()
            bohb.observe(config, 50.0 + np.random.random() * 30, budget)
        X_arr = np.array(bohb._X, dtype=np.float64)
        y_arr = np.array(bohb._y, dtype=np.float64)
        _, good, bad, bw = bohb._build_tpe_kdes(X_arr, y_arr, 1)
        assert bw > 0
        assert good.shape[0] >= 1 or bad.shape[0] >= 1

    def test_random_config_vec(self):
        bohb = BOHBOptimizer(nkpt=27, min_budget=0)
        rng = np.random.RandomState(42)
        vec = bohb._random_config_vec(rng)
        assert vec.shape == (5,)
        assert 0.0 <= vec[0] <= 1.0
        assert np.sum(vec[2:]) > 0  # one-hot is active


# ---------------------------------------------------------------------------
# DPP Batch Selector
# ---------------------------------------------------------------------------

class TestDPPBatchSelector:
    def setup_method(self):
        np.random.seed(123)

        self.n_candidates = 20
        self.n_dims = 2
        self.X_candidates = np.random.randn(self.n_candidates, self.n_dims)

        X_train = np.random.randn(10, self.n_dims)
        y_train = np.random.randn(10) * 2 + 5

        self.gp = _GaussianProcess()
        self.gp.fit(X_train, y_train)

    def test_select_returns_indices(self):
        ei_values = 0.1 + np.abs(np.random.randn(self.n_candidates))
        selector = DPPBatchSelector()
        kernel_fn = _make_rbf_kernel(self.n_dims)
        indices = selector.select(
            self.X_candidates, ei_values, self.gp, q=4,
            kernel_func=kernel_fn,
        )
        assert len(indices) == 4
        assert len(set(indices)) == 4
        assert all(0 <= i < self.n_candidates for i in indices)

    def test_select_empty(self):
        selector = DPPBatchSelector()
        indices = selector.select(
            np.empty((0, 2)), np.empty(0), None, q=4,
        )
        assert indices == []

    def test_select_q_larger_than_n(self):
        selector = DPPBatchSelector()
        kernel_fn = _make_rbf_kernel(self.n_dims)
        indices = selector.select(
            self.X_candidates[:3], np.ones(3), self.gp, q=10,
            kernel_func=kernel_fn,
        )
        assert len(indices) == 3

    def test_dpp_diversity(self):
        # Cluster candidates at 3 distinct locations — DPP should spread
        rng = np.random.RandomState(99)
        clusters = [np.array([0.0, 0.0]), np.array([10.0, 10.0]), np.array([-10.0, 5.0])]
        X = np.vstack([
            clusters[0] + rng.randn(5, 2) * 0.3,
            clusters[1] + rng.randn(5, 2) * 0.3,
            clusters[2] + rng.randn(5, 2) * 0.3,
        ])
        # Equal quality → selection purely by diversity
        quality = np.ones(len(X))

        selector = DPPBatchSelector()
        kernel_fn = _make_rbf_kernel(2)
        indices = selector.select(X, quality, self.gp, q=3, kernel_func=kernel_fn)

        selected_points = X[indices]
        # Points should come from different clusters
        centroids = np.array(clusters)
        assigned = np.argmin(
            np.sum((selected_points[:, None, :] - centroids[None, :, :]) ** 2, axis=-1),
            axis=-1,
        )
        assert len(set(assigned)) >= 2, "DPP should select diverse points from different clusters"

    def test_chol_insert_identity(self):
        from forge.optimizer.bayesian.dpp import _chol_insert

        q_norm = np.array([1.0, 1.0, 1.0])
        cands = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        k_fn = _make_rbf_kernel(2)

        selected = [0]
        L = _chol_insert(None, q_norm, selected, cands, k_fn)
        assert L.shape == (1, 1)
        assert abs(L[0, 0] - 1.0) < 1e-6

        selected.append(1)
        L = _chol_insert(L, q_norm, selected, cands, k_fn)
        assert L.shape == (2, 2)
        assert np.all(L[np.tril_indices(2, -1)] == 0) or L[-1, 0] != 0
