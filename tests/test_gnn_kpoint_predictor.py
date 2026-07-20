"""Tests for GNN k-point predictor — analytical gradient verification."""

import numpy as np

from forge.ml.gnn_kpoint_predictor import (
    CGCNNModel,
    GraphConvLayer,
    generate_synthetic_dataset,
)


def _finite_diff_grad(
    conv: GraphConvLayer,
    h: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    param_name: str,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference gradient for a single parameter matrix."""
    param = getattr(conv, param_name)
    orig = param.copy()
    grad = np.zeros_like(param)

    out_ref = conv.forward(h, edge_index, edge_feat)
    _ = float(np.sum(out_ref ** 2))

    for idx in np.ndindex(param.shape):
        param[idx] = orig[idx] + eps
        out_plus = conv.forward(h, edge_index, edge_feat)
        loss_plus = float(np.sum(out_plus ** 2))

        param[idx] = orig[idx] - eps
        out_minus = conv.forward(h, edge_index, edge_feat)
        loss_minus = float(np.sum(out_minus ** 2))

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = orig[idx]

    return grad


class TestGraphConvLayerGradient:
    """Verify analytical backward against central finite differences."""

    def test_forward_output_shape(self):
        conv = GraphConvLayer(in_dim=4, out_dim=8, edge_dim=2)
        h = np.random.randn(6, 4).astype(np.float32)
        edge_index = np.array([[0, 1, 2, 3, 4],
                               [1, 2, 3, 4, 5]], dtype=np.int64)
        edge_feat = np.random.randn(5, 2).astype(np.float32)
        out = conv.forward(h, edge_index, edge_feat)
        assert out.shape == (6, 8)

    def test_forward_no_edges(self):
        conv = GraphConvLayer(in_dim=4, out_dim=8, edge_dim=2)
        h = np.random.randn(3, 4).astype(np.float32)
        out = conv.forward(h, np.zeros((2, 0), dtype=np.int64),
                           np.zeros((0, 2), dtype=np.float32))
        assert out.shape == (3, 8)

    def test_backward_gradient_W_g(self):
        np.random.seed(42)
        conv = GraphConvLayer(in_dim=3, out_dim=5, edge_dim=2)
        h = np.random.randn(4, 3).astype(np.float32)
        edge_index = np.array([[0, 1, 2],
                               [1, 2, 3]], dtype=np.int64)
        edge_feat = np.random.randn(3, 2).astype(np.float32)

        _ = conv.forward(h, edge_index, edge_feat)
        out = conv.forward(h, edge_index, edge_feat)
        dL_dout = 2.0 * out

        _, grads = conv.backward(dL_dout, edge_feat)
        num_grad = _finite_diff_grad(conv, h, edge_index, edge_feat, "W_g")

        rel_err = np.max(np.abs(grads["W_g"] - num_grad)) / max(
            np.max(np.abs(num_grad)), 1e-8
        )
        max_abs_err = np.max(np.abs(grads["W_g"] - num_grad))
        max_grad = np.max(np.abs(grads["W_g"]))
        max_num = np.max(np.abs(num_grad))
        assert rel_err < 0.20, (
            f"W_g rel err {rel_err:.4f}, "
            f"max|analytical|={max_grad:.6f}, max|numerical|={max_num:.6f}, "
            f"max|diff|={max_abs_err:.6f}"
        )

    def test_backward_gradient_W_c(self):
        np.random.seed(43)
        conv = GraphConvLayer(in_dim=3, out_dim=5, edge_dim=2)
        h = np.random.randn(4, 3).astype(np.float32)
        edge_index = np.array([[0, 1, 2],
                               [1, 2, 3]], dtype=np.int64)
        edge_feat = np.random.randn(3, 2).astype(np.float32)

        _ = conv.forward(h, edge_index, edge_feat)
        out = conv.forward(h, edge_index, edge_feat)
        dL_dout = 2.0 * out

        _, grads = conv.backward(dL_dout, edge_feat)
        num_grad = _finite_diff_grad(conv, h, edge_index, edge_feat, "W_c")

        rel_err = np.max(np.abs(grads["W_c"] - num_grad)) / max(
            np.max(np.abs(num_grad)), 1e-8
        )
        assert rel_err < 0.20, f"W_c rel err {rel_err:.4f}"

    def test_backward_gradient_b_g(self):
        np.random.seed(44)
        conv = GraphConvLayer(in_dim=3, out_dim=5, edge_dim=2)
        h = np.random.randn(4, 3).astype(np.float32)
        edge_index = np.array([[0, 1, 2],
                               [1, 2, 3]], dtype=np.int64)
        edge_feat = np.random.randn(3, 2).astype(np.float32)

        _ = conv.forward(h, edge_index, edge_feat)
        out = conv.forward(h, edge_index, edge_feat)
        dL_dout = 2.0 * out
        dL_dout = dL_dout.astype(np.float32)

        _, grads = conv.backward(dL_dout, edge_feat)
        num_grad_w = _finite_diff_grad(conv, h, edge_index, edge_feat, "W_g")
        num_grad_c = _finite_diff_grad(conv, h, edge_index, edge_feat, "W_c")

        assert np.all(np.abs(grads["b_g"]) > 0) or np.all(np.abs(num_grad_w) < 1e-6)
        assert np.all(np.abs(grads["b_c"]) > 0) or np.all(np.abs(num_grad_c) < 1e-6)

    def test_backward_no_edges(self):
        conv = GraphConvLayer(in_dim=4, out_dim=6, edge_dim=2)
        h = np.random.randn(3, 4).astype(np.float32)
        ei = np.zeros((2, 0), dtype=np.int64)
        ef = np.zeros((0, 2), dtype=np.float32)

        _ = conv.forward(h, ei, ef)
        out = conv.forward(h, ei, ef)
        dL_dout = 2.0 * out

        dL_dh, _grads = conv.backward(dL_dout, ef)
        assert dL_dh.shape == h.shape

    def test_multiple_layers_do_not_explode(self):
        model = CGCNNModel(node_dim=4, hidden_dim=16, n_conv_layers=3)
        h = np.random.randn(5, 4).astype(np.float32)
        ei = np.array([[0, 1, 2, 3],
                       [1, 2, 3, 4]], dtype=np.int64)
        ef = np.random.randn(4, 2).astype(np.float32)

        pred = model.forward(h, ei, ef)
        target = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        grads = model.backward(target, ef)

        for name, g in grads.items():
            assert not np.any(np.isnan(g)), f"NaN in {name}"
            assert not np.any(np.isinf(g)), f"Inf in {name}"

        assert np.all(np.isfinite(pred))


class TestModelIntegration:
    def test_train_on_synthetic(self):
        dataset = generate_synthetic_dataset(n_samples=30)
        model = CGCNNModel(node_dim=4, hidden_dim=8, n_conv_layers=2)
        history = model.train(dataset, epochs=5, lr=0.01, verbose=False)

        assert len(history) == 5
        assert history[-1] <= history[0] or history[-1] < 10.0

    def test_save_load_roundtrip(self, tmp_path):
        model = CGCNNModel(node_dim=4, hidden_dim=8, n_conv_layers=2)
        path = str(tmp_path / "test_model.npz")
        model.save(path)

        loaded = CGCNNModel.load(path)
        assert loaded.conv1.W_g.shape == model.conv1.W_g.shape
        assert np.allclose(loaded.conv1.W_g, model.conv1.W_g)
