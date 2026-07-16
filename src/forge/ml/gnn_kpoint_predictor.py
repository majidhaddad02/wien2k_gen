"""
Crystal Graph Neural Network for K-point Grid Prediction.

Implements CGCNN-style graph convolution (Xie & Grossman 2018) using only numpy.
Zero-dependency inference — no PyTorch, no TensorFlow.

Architecture:
  1. Build crystal graph from structure (nodes=atoms, edges=bonds up to cutoff)
  2. 4-layer graph convolution with residual connections
  3. Global mean+max pooling
  4. 2-layer MLP head -> k-point grid prediction + confidence score

Training pipeline:
  - Synthetic dataset from Monkhorst-Pack rules + heuristic chemistry knowledge
  - Adam optimizer (pure numpy implementation)
  - MAE loss on normalized k-point grid
  - Weights saved to .npz for portable deployment

References:
  Choudhary & DeCost (2019) "Atomistic Line Graph Neural Network"
  Xie & Grossman (2018) "Crystal Graph Convolutional Neural Networks"
  Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Atomic feature maps
# ---------------------------------------------------------------------------

_ATOMIC_FEATURES: dict[int, list[float]] = {
    1:  [0.31,  2.20,  1.0,  1],    2:  [0.28,  0.00,  1.0,  2],
    3:  [1.28,  0.98,  1.0,  1],    4:  [0.96,  1.57,  1.5,  2],
    5:  [0.84,  2.04,  2.0,  3],    6:  [0.76,  2.55,  2.5,  4],
    7:  [0.71,  3.04,  3.0,  5],    8:  [0.66,  3.44,  3.5,  6],
    9:  [0.57,  3.98,  4.0,  7],   10:  [0.58,  0.00,  1.0,  8],
    11: [1.66,  0.93,  1.0,  1],   12: [1.41,  1.31,  1.5,  2],
    13: [1.21,  1.61,  2.0,  3],   14: [1.11,  1.90,  2.5,  4],
    15: [1.07,  2.19,  3.0,  5],   16: [1.05,  2.58,  3.5,  6],
    17: [1.02,  3.16,  4.0,  7],   18: [1.06,  0.00,  1.0,  8],
    19: [2.03,  0.82,  1.0,  1],   20: [1.76,  1.00,  1.5,  2],
    21: [1.44,  1.36,  2.0,  3],   22: [1.32,  1.54,  2.5,  4],
    23: [1.22,  1.63,  2.5,  5],   24: [1.18,  1.66,  2.5,  6],
    25: [1.17,  1.55,  2.5,  7],   26: [1.17,  1.83,  2.5,  8],
    27: [1.16,  1.88,  2.5,  9],   28: [1.15,  1.91,  2.5,  10],
    29: [1.17,  1.90,  2.5,  11],  30: [1.25,  1.65,  2.5,  12],
    31: [1.26,  1.81,  2.5,  3],   32: [1.22,  2.01,  2.5,  4],
    33: [1.21,  2.18,  2.5,  5],   34: [1.22,  2.55,  2.5,  6],
    35: [1.23,  2.96,  2.5,  7],   36: [1.24,  3.00,  2.5,  8],
    37: [2.20,  0.82,  1.0,  1],   38: [2.15,  0.95,  1.5,  2],
    39: [1.80,  1.22,  2.0,  3],   40: [1.55,  1.33,  2.5,  4],
    41: [2.08,  1.60,  2.5,  5],   42: [2.09,  2.16,  2.5,  6],
    44: [2.07,  2.20,  2.5,  8],   45: [1.80,  2.28,  2.5,  9],
    46: [1.62,  2.20,  2.5,  10],  47: [1.53,  1.93,  2.5,  11],
    48: [1.49,  1.69,  2.5,  12],  49: [1.56,  1.78,  2.5,  3],
    50: [1.46,  1.96,  2.5,  4],   51: [1.40,  2.05,  2.5,  5],
    52: [1.44,  2.10,  2.5,  6],   53: [1.40,  2.66,  2.5,  7],
    54: [1.31,  2.60,  2.5,  8],   56: [2.15,  0.89,  1.0,  2],
    57: [1.95,  1.10,  2.0,  3],   58: [1.85,  1.12,  2.0,  4],
    64: [1.75,  1.20,  2.5,  3],   72: [1.50,  1.30,  2.5,  4],
    73: [2.00,  1.50,  2.5,  5],   74: [2.10,  2.36,  2.5,  6],
    75: [2.05,  1.90,  2.5,  7],   76: [2.00,  2.20,  2.5,  8],
    77: [1.90,  2.20,  2.5,  9],   78: [1.83,  2.28,  2.5,  10],
    79: [1.59,  2.54,  2.5,  11],  80: [1.65,  2.00,  2.5,  12],
    82: [1.54,  2.33,  2.5,  4],   83: [1.56,  2.02,  2.5,  5],
    90: [1.80,  1.30,  2.5,  4],   92: [1.75,  1.38,  2.5,  6],
}

_NUM_ATOMIC_FEATURES = 4

# ---------------------------------------------------------------------------
# Crystal graph builder
# ---------------------------------------------------------------------------

def build_crystal_graph(  # noqa: C901
    positions: list[tuple[float, float, float]],
    atomic_numbers: list[int],
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    cutoff: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build crystal graph from atomic structure.

    Args:
        positions: fractional coordinates [(x1,y1,z1), ...]
        atomic_numbers: [Z1, Z2, ...]
        lattice_vectors: (a_vec, b_vec, c_vec) in cartesian
        cutoff: maximum bond distance in Angstrom

    Returns:
        node_features:  (N, 4)  — atomic features per atom
        edge_index:     (2, E)  — source->target atom pairs
        edge_features:  (E, 2)  — distance + bond_type per edge
    """
    n_atoms = len(atomic_numbers)
    if n_atoms == 0:
        return (
            np.zeros((0, _NUM_ATOMIC_FEATURES)),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, 2)),
        )

    node_features = np.zeros((n_atoms, _NUM_ATOMIC_FEATURES))
    for i, z in enumerate(atomic_numbers):
        feats = _ATOMIC_FEATURES.get(z, [1.0, 1.5, 2.0, 1])
        node_features[i] = feats[:_NUM_ATOMIC_FEATURES]

    a, b, c_vec = lattice_vectors
    cart = np.zeros((n_atoms, 3))
    for i, (x, y, z) in enumerate(positions):
        cart[i] = x * np.array(a) + y * np.array(b) + z * np.array(c_vec)

    edges_src, edges_dst = [], []
    edge_feats = []
    cutoff_sq = cutoff ** 2

    for i in range(n_atoms):
        for j in range(i, n_atoms):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        offset = di * np.array(a) + dj * np.array(b) + dk * np.array(c_vec)
                        dist_vec = cart[j] + offset - cart[i]
                        dist_sq = float(np.dot(dist_vec, dist_vec))
                        if 0 < dist_sq <= cutoff_sq:
                            dist = math.sqrt(dist_sq)
                            zi, zj = atomic_numbers[i], atomic_numbers[j]
                            en_i = _ATOMIC_FEATURES.get(zi, [1.0, 1.5])[1]
                            en_j = _ATOMIC_FEATURES.get(zj, [1.0, 1.5])[1]
                            en_diff = abs(en_i - en_j)
                            bond_type = 0.0 if en_diff < 0.5 else (1.0 if en_diff < 1.7 else 2.0)

                            edges_src.append(i)
                            edges_dst.append(j)
                            edge_feats.append([dist, bond_type])
                            if j != i:
                                edges_src.append(j)
                                edges_dst.append(i)
                                edge_feats.append([dist, bond_type])

    if not edges_src:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 2))
    else:
        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
        edge_features = np.array(edge_feats)

    return node_features.astype(np.float32), edge_index, edge_features.astype(np.float32)


# ---------------------------------------------------------------------------
# Graph convolution layer
# ---------------------------------------------------------------------------

class GraphConvLayer:
    """A single graph convolution layer.

    h_i' = sigma( W_s*h_i + sum_{j in N(i)} W_n*h_j * EdgeMLP(e_ij) )
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 2):
        scale = math.sqrt(2.0 / in_dim)
        self.W_self = np.random.randn(in_dim, out_dim).astype(np.float32) * scale * 0.5
        self.W_neigh = np.random.randn(in_dim, out_dim).astype(np.float32) * scale * 0.5
        self.W_edge = np.random.randn(edge_dim, out_dim).astype(np.float32) * scale * 0.5
        self.bias = np.zeros(out_dim, dtype=np.float32)

    @property
    def in_dim(self) -> int:
        return self.W_self.shape[0]

    @property
    def out_dim(self) -> int:
        return self.W_self.shape[1]

    def forward(self, h: np.ndarray, edge_index: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
        n = h.shape[0]
        self_msg = h @ self.W_self
        neigh_msg = np.zeros((n, self.out_dim), dtype=np.float32)

        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            edge_mlp = edge_feat @ self.W_edge
            for e in range(edge_index.shape[1]):
                d = dst[e]
                s = src[e]
                neigh_msg[d] += h[s] @ self.W_neigh * edge_mlp[e]

        deg = np.maximum(np.bincount(edge_index[1], minlength=n) if edge_index.shape[1] > 0 else np.ones(n), 1).reshape(-1, 1)
        neigh_msg /= deg

        out = self_msg + neigh_msg + self.bias
        return np.maximum(out, 0.0)


class AdamOptimizer:
    """Adam optimizer in pure numpy for GNN training."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: dict[int, dict[str, np.ndarray]] = {}
        self.v: dict[int, dict[str, np.ndarray]] = {}

    def step(self, param_id: int, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_id not in self.m:
            self.m[param_id] = {}
            self.v[param_id] = {}
        if param_name not in self.m[param_id]:
            self.m[param_id][param_name] = np.zeros_like(param)
            self.v[param_id][param_name] = np.zeros_like(param)

        self.t += 1
        self.m[param_id][param_name] = self.beta1 * self.m[param_id][param_name] + (1 - self.beta1) * grad
        self.v[param_id][param_name] = self.beta2 * self.v[param_id][param_name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[param_id][param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id][param_name] / (1 - self.beta2 ** self.t)

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# CGCNN Model
# ---------------------------------------------------------------------------

class CGCNNModel:
    """Lightweight CGCNN for k-point grid prediction — pure numpy."""

    def __init__(
        self,
        node_dim: int = 4,
        hidden_dim: int = 64,
        n_conv_layers: int = 4,
        output_dim: int = 3,
    ):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.conv1 = GraphConvLayer(node_dim, hidden_dim)
        self.convs = [GraphConvLayer(hidden_dim, hidden_dim) for _ in range(n_conv_layers - 1)]
        scale = math.sqrt(2.0 / hidden_dim)
        self.fc1_W = np.random.randn(hidden_dim * 2, 128).astype(np.float32) * scale * 0.5
        self.fc1_b = np.zeros(128, dtype=np.float32)
        self.fc2_W = np.random.randn(128, output_dim).astype(np.float32) * scale * 0.5
        self.fc2_b = np.zeros(output_dim, dtype=np.float32)

    @property
    def n_conv_layers(self) -> int:
        return len(self.convs) + 1

    def forward(self, x: np.ndarray, edge_index: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
        h = self.conv1.forward(x, edge_index, edge_feat)
        for conv in self.convs:
            h_res = conv.forward(h, edge_index, edge_feat)
            h = h + h_res

        mean_pool = np.mean(h, axis=0, keepdims=True)
        max_pool = np.max(h, axis=0, keepdims=True)
        pooled = np.concatenate([mean_pool, max_pool], axis=-1)

        out = pooled @ self.fc1_W + self.fc1_b
        out = np.maximum(out, 0.0)
        out = out @ self.fc2_W + self.fc2_b
        return out.flatten()

    def _all_parameters(self) -> list[tuple[int, str, np.ndarray]]:
        params: list[tuple[int, str, np.ndarray]] = [
            (0, "W_self", self.conv1.W_self),
            (0, "W_neigh", self.conv1.W_neigh),
            (0, "W_edge", self.conv1.W_edge),
            (0, "bias", self.conv1.bias),
        ]
        for i, conv in enumerate(self.convs):
            params.extend([
                (i + 1, "W_self", conv.W_self),
                (i + 1, "W_neigh", conv.W_neigh),
                (i + 1, "W_edge", conv.W_edge),
                (i + 1, "bias", conv.bias),
            ])
        params.extend([
            (99, "W", self.fc1_W),
            (99, "b", self.fc1_b),
            (100, "W", self.fc2_W),
            (100, "b", self.fc2_b),
        ])
        return params

    def _apply_gradients(self, grads: dict[str, np.ndarray], optimizer: AdamOptimizer) -> None:
        idx = 0
        for conv_list, prefix in [([self.conv1], "conv1"), (self.convs, "convs")]:
            for ci, conv in enumerate(conv_list):
                for name in ["W_self", "W_neigh", "W_edge", "bias"]:
                    key = f"{prefix}_{ci}_{name}" if conv_list is self.convs else f"{prefix}_{name}"
                    if key in grads:
                        new_val = optimizer.step(idx, key, getattr(conv, name), grads[key])
                        setattr(conv, name, new_val)
                    idx += 1
        for name, attr in [("W", "fc1_W"), ("b", "fc1_b"), ("W", "fc2_W"), ("b", "fc2_b")]:
            key = f"fc_{name}"
            if key in grads:
                new_val = optimizer.step(idx, key, getattr(self, attr), grads[key])
                setattr(self, attr, new_val)
            idx += 1

    def train(
        self,
        dataset: list[dict[str, Any]],
        epochs: int = 100,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> list[float]:
        """Train on a list of structure dicts with known k-point grids.

        Each dict must have: atoms, lattice, kpoints=(nx, ny, nz)
        """
        graphs = []
        targets = []

        for entry in dataset:
            atoms = entry.get("atoms", [])
            if not atoms:
                continue
            positions = [(a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0)) for a in atoms]
            atomic_numbers = [int(a.get("z_num", 1)) for a in atoms]
            lattice = entry.get("lattice", {})
            lat_vecs = (
                (lattice.get("a", 10.0), 0.0, 0.0),
                (0.0, lattice.get("b", 10.0), 0.0),
                (0.0, 0.0, lattice.get("c", 10.0)),
            )

            nf, ei, ef = build_crystal_graph(positions, atomic_numbers, lat_vecs)
            if nf.shape[0] == 0:
                continue

            kpts = entry.get("kpoints", (4, 4, 4))
            target = np.array([kpts[0] / 12.0, kpts[1] / 12.0, kpts[2] / 12.0], dtype=np.float32)
            graphs.append((nf, ei, ef))
            targets.append(target)

        if len(graphs) == 0:
            logger.warning("No valid training graphs; skipping training")
            return []

        targets = np.array(targets, dtype=np.float32)
        optimizer = AdamOptimizer(lr=lr)
        history: list[float] = []

        for epoch in range(epochs):
            perm = np.random.permutation(len(graphs))
            epoch_loss = 0.0

            for idx in perm:
                node_feat, edge_idx, edge_feat = graphs[idx]
                target = targets[idx]

                pred = self.forward(node_feat, edge_idx, edge_feat)
                error = pred - target
                nf_loss = float(np.sum(error ** 2))

                grads = _compute_gradients(self, node_feat, edge_idx, edge_feat, error)
                self._apply_gradients(grads, optimizer)
                epoch_loss += nf_loss

            avg_loss = epoch_loss / max(len(graphs), 1)
            history.append(avg_loss)
            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"GNN epoch {epoch+1}/{epochs} — loss={avg_loss:.6f}")

        return history

    def save(self, path: str) -> None:
        """Save model weights to .npz file."""
        weights: dict[str, np.ndarray] = {}
        weights["conv1_W_self"] = self.conv1.W_self
        weights["conv1_W_neigh"] = self.conv1.W_neigh
        weights["conv1_W_edge"] = self.conv1.W_edge
        weights["conv1_bias"] = self.conv1.bias
        for i, conv in enumerate(self.convs):
            weights[f"convs_{i}_W_self"] = conv.W_self
            weights[f"convs_{i}_W_neigh"] = conv.W_neigh
            weights[f"convs_{i}_W_edge"] = conv.W_edge
            weights[f"convs_{i}_bias"] = conv.bias
        weights["fc1_W"] = self.fc1_W
        weights["fc1_b"] = self.fc1_b
        weights["fc2_W"] = self.fc2_W
        weights["fc2_b"] = self.fc2_b
        np.savez(path, **weights)
        logger.info(f"GNN model saved to {path}")

    @staticmethod
    def load(path: str) -> CGCNNModel:
        """Load model weights from .npz file."""
        data = np.load(path)
        n_conv_layers = 0
        while f"convs_{n_conv_layers}_W_self" in data:
            n_conv_layers += 1
        n_conv_layers = max(n_conv_layers, 1)
        output_dim = data["fc2_b"].shape[0]
        hidden_dim = data["conv1_W_self"].shape[1]

        model = CGCNNModel(
            node_dim=data["conv1_W_self"].shape[0],
            hidden_dim=hidden_dim,
            n_conv_layers=n_conv_layers,
            output_dim=output_dim,
        )
        model.conv1.W_self = data["conv1_W_self"]
        model.conv1.W_neigh = data["conv1_W_neigh"]
        model.conv1.W_edge = data["conv1_W_edge"]
        model.conv1.bias = data["conv1_bias"]
        for i in range(n_conv_layers - 1):
            model.convs[i].W_self = data[f"convs_{i}_W_self"]
            model.convs[i].W_neigh = data[f"convs_{i}_W_neigh"]
            model.convs[i].W_edge = data[f"convs_{i}_W_edge"]
            model.convs[i].bias = data[f"convs_{i}_bias"]
        model.fc1_W = data["fc1_W"]
        model.fc1_b = data["fc1_b"]
        model.fc2_W = data["fc2_W"]
        model.fc2_b = data["fc2_b"]
        logger.info(f"GNN model loaded from {path}")
        return model


def _compute_gradients(
    model: CGCNNModel,
    node_feat: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    error: np.ndarray,
) -> dict[str, np.ndarray]:
    """Finite-difference gradient approximation for GNN training.

    Each parameter is perturbed by +eps/-eps and the change in loss
    (with respect to the error vector) gives the gradient.
    Uses central difference: grad = (f(x+eps) - f(x-eps)) / (2*eps).
    """
    eps = 1e-4
    grads: dict[str, np.ndarray] = {}

    def _loss_for_pred(pred: np.ndarray) -> float:
        return float(np.sum((pred - error) ** 2))

    def _param_grad(param: np.ndarray) -> np.ndarray:
        flat_param = param.ravel()
        flat_grad = np.zeros_like(flat_param)
        for idx in range(flat_param.shape[0]):
            orig = flat_param[idx]
            flat_param[idx] = orig + eps
            loss_plus = _loss_for_pred(model.forward(node_feat, edge_index, edge_feat))
            flat_param[idx] = orig - eps
            loss_minus = _loss_for_pred(model.forward(node_feat, edge_index, edge_feat))
            flat_param[idx] = orig
            flat_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        return flat_grad.reshape(param.shape)

    for conv_list, prefix in [([model.conv1], "conv1"), (model.convs, "convs")]:
        for ci, conv in enumerate(conv_list):
            for name in ["W_self", "W_neigh", "W_edge", "bias"]:
                key = f"{prefix}_{ci}_{name}" if conv_list is model.convs else f"{prefix}_{name}"
                grads[key] = _param_grad(getattr(conv, name))

    grads["fc_W1"] = _param_grad(model.fc1_W)
    grads["fc_b1"] = _param_grad(model.fc1_b)
    grads["fc_W2"] = _param_grad(model.fc2_W)
    grads["fc_b2"] = _param_grad(model.fc2_b)
    return grads


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_samples: int = 200) -> list[dict[str, Any]]:
    """Generate synthetic crystal structures with MP-rule k-point grids.

    Each sample has:
      - atoms: list of {x, y, z, z_num}
      - lattice: {a, b, c, alpha, beta, gamma} in Angstrom
      - kpoints: (kx, ky, kz) from Monkhorst-Pack rule

    Covers common WIEN2k materials: metals, oxides, perovskites, organics.
    """
    rng = np.random.RandomState(42)
    dataset: list[dict[str, Any]] = []

    # Template structures for diversity
    templates = [
        {"composition": [(26, 2), (8, 3)], "lattice_base": (5.0, 5.0, 14.0), "label": "Fe2O3-type"},
        {"composition": [(14, 1), (8, 2)], "lattice_base": (5.4, 5.4, 5.4), "label": "SiO2-type"},
        {"composition": [(22, 1), (8, 2)], "lattice_base": (4.6, 4.6, 3.0), "label": "TiO2-type"},
        {"composition": [(13, 2), (8, 3)], "lattice_base": (4.8, 4.8, 13.0), "label": "Al2O3-type"},
        {"composition": [(56, 1), (22, 1), (8, 3)], "lattice_base": (4.0, 4.0, 4.0), "label": "BaTiO3"},
        {"composition": [(82, 1), (40, 1), (8, 3)], "lattice_base": (4.1, 4.1, 4.1), "label": "PbZrO3"},
        {"composition": [(29, 1), (8, 1)], "lattice_base": (4.6, 3.4, 5.1), "label": "CuO"},
        {"composition": [(28, 1), (8, 1)], "lattice_base": (4.2, 4.2, 4.2), "label": "NiO"},
        {"composition": [(6, 4), (1, 4)], "lattice_base": (6.0, 8.0, 10.0), "label": "organic"},
        {"composition": [(79, 4)], "lattice_base": (4.1, 4.1, 4.1), "label": "Au-fcc"},
        {"composition": [(47, 4)], "lattice_base": (4.1, 4.1, 4.1), "label": "Ag-fcc"},
        {"composition": [(74, 2), (8, 6)], "lattice_base": (7.3, 7.5, 3.8), "label": "WO3"},
        {"composition": [(25, 2), (8, 3)], "lattice_base": (5.0, 8.5, 5.0), "label": "Mn2O3"},
        {"composition": [(27, 3), (8, 4)], "lattice_base": (8.1, 8.1, 8.1), "label": "Co3O4"},
        {"composition": [(92, 2), (8, 4)], "lattice_base": (5.4, 5.5, 5.5), "label": "UO2"},
    ]

    for _ in range(n_samples):
        tpl = templates[rng.randint(0, len(templates))]
        comp = tpl["composition"]
        base = tpl["lattice_base"]

        a = base[0] * (0.9 + 0.2 * rng.random())
        b = base[1] * (0.9 + 0.2 * rng.random())
        c = base[2] * (0.9 + 0.2 * rng.random())
        scale = rng.uniform(0.85, 1.15)
        a, b, c = a * scale, b * scale, c * scale

        atoms = []
        for z, count in comp:
            for _ in range(count):
                atoms.append({
                    "x": float(rng.random()),
                    "y": float(rng.random()),
                    "z": float(rng.random()),
                    "z_num": z,
                    "name": str(z),
                })

        has_metal = any(a["z_num"] in {3, 4, 11, 12, 13, 26, 27, 28, 29, 30} for a in atoms)
        k0 = 40 if has_metal else 30
        kx = max(1, min(12, round(k0 / a)))
        ky = max(1, min(12, round(k0 / b)))
        kz = max(1, min(12, round(k0 / c)))

        dataset.append({
            "atoms": atoms,
            "lattice": {"a": a, "b": b, "c": c, "alpha": 90, "beta": 90, "gamma": 90},
            "kpoints": (kx, ky, kz),
            "label": tpl["label"],
        })

    return dataset


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = Path.home() / ".forge" / "models"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "gnn_kpoint_v1.npz"


def get_trained_model(force_retrain: bool = False) -> CGCNNModel:
    """Load pre-trained GNN model or train a new one from synthetic data.

    Args:
        force_retrain: Ignore saved weights and re-train from scratch.

    Returns:
        Trained CGCNNModel ready for inference.
    """
    if not force_retrain and _DEFAULT_MODEL_PATH.exists():
        try:
            return CGCNNModel.load(str(_DEFAULT_MODEL_PATH))
        except Exception as e:
            logger.warning(f"Failed to load saved model: {e} — retraining")

    logger.info("Training GNN on synthetic dataset (this may take 30-60 seconds)...")
    model = CGCNNModel()
    dataset = generate_synthetic_dataset(200)
    loss_history = model.train(dataset, epochs=80, lr=0.001, verbose=True)

    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(_DEFAULT_MODEL_PATH))
    logger.info(f"GNN trained. Final loss: {loss_history[-1]:.6f}, weights saved to {_DEFAULT_MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def predict_kpoints(
    structure: dict[str, Any],
    model_path: str | None = None,
    default_model_dir: str | None = None,
) -> dict[str, Any]:
    """Predict optimal k-point grid from crystal structure.

    Returns dict with:
        grid: Tuple[int, int, int]  — recommended k-point grid
        confidence: float           — prediction confidence (0-1)
        method: str                 — "GNN" or "fallback"
        kpoint_density: int         — recommended density (k-points/Ang^{-3})
    """
    atoms = structure.get("atoms", [])
    if not atoms:
        return _kpoint_fallback(structure, "Empty structure")

    positions = [(a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0)) for a in atoms]
    atomic_numbers = [int(a.get("z_num", 1)) for a in atoms]
    lattice = structure.get("lattice", {})
    lattice_vectors = (
        (lattice.get("a", 10.0), 0.0, 0.0),
        (0.0, lattice.get("b", 10.0), 0.0),
        (0.0, 0.0, lattice.get("c", 10.0)),
    )

    node_feat, edge_idx, edge_feat = build_crystal_graph(
        positions, atomic_numbers, lattice_vectors
    )

    if node_feat is None or node_feat.shape[0] == 0:
        return _kpoint_fallback(structure, "Failed to build graph")

    model = _get_or_create_model(model_path, default_model_dir)

    try:
        prediction = model.forward(node_feat, edge_idx, edge_feat)
    except Exception as e:
        logger.warning(f"GNN inference failed: {e}")
        return _kpoint_fallback(structure, f"Inference error: {e}")

    nx = max(1, round(abs(prediction[0]) * 12))
    ny = max(1, round(abs(prediction[1]) * 12))
    nz = max(1, round(abs(prediction[2]) * 12))

    pred_std = float(np.std(prediction))
    confidence = min(1.0, 1.0 / (1.0 + pred_std * 3.0))

    volume = lattice["a"] * lattice["b"] * lattice["c"]
    density = int(nx * ny * nz / volume * 1000) if volume > 0 else 500

    if confidence < 0.60:
        logger.info(f"GNN confidence={confidence:.2f} < 0.6 — falling back to MP grid")
        return _kpoint_fallback(structure, f"Low confidence ({confidence:.2f})")

    return {
        "grid": (nx, ny, nz),
        "confidence": round(confidence, 3),
        "method": "GNN",
        "kpoint_density": density,
        "recommendation": f"GNN predicts {nx}x{ny}x{nz} grid (density ~{density} kpts/Ang^3)",
    }


def _get_or_create_model(model_path: str | None = None, default_dir: str | None = None) -> CGCNNModel:
    """Load pre-trained model or create a heuristically-initialized one."""
    if model_path and Path(model_path).exists():
        try:
            return CGCNNModel.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

    if default_dir:
        model_dir = Path(default_dir)
        if model_dir.exists():
            model_files = sorted(model_dir.glob("gnn_kpoint_v*.npz"))
            if model_files:
                try:
                    return CGCNNModel.load(str(model_files[-1]))
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_files[-1]}: {e}")

    if _DEFAULT_MODEL_PATH.exists() and not os.environ.get("FORGE_GNN_RETRAIN"):
        try:
            return CGCNNModel.load(str(_DEFAULT_MODEL_PATH))
        except Exception:
            logger.debug("Suppressed exception in _get_or_create_model()", exc_info=True)

    return get_trained_model()


def _kpoint_fallback(structure: dict[str, Any], reason: str = "") -> dict[str, Any]:
    """Fallback: Monkhorst-Pack grid based on lattice constants."""
    lattice = structure.get("lattice", {})
    a, b, c = lattice.get("a", 10.0), lattice.get("b", 10.0), lattice.get("c", 10.0)
    k0 = 30
    atoms = structure.get("atoms", [])
    has_metal = any(a.get("z_num", 0) in {3, 4, 11, 12, 13, 26, 27, 28, 29, 30} for a in atoms)
    if has_metal:
        k0 = 40

    nx = max(1, round(k0 / a))
    ny = max(1, round(k0 / b))
    nz = max(1, round(k0 / c))

    return {
        "grid": (nx, ny, nz),
        "confidence": 0.5,
        "method": "fallback_mp_grid",
        "kpoint_density": int(nx * ny * nz / (a * b * c) * 1000),
        "recommendation": f"Fallback Monkhorst-Pack {nx}x{ny}x{nz} grid (reason: {reason})",
    }
