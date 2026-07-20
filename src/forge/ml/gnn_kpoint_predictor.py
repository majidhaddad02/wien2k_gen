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

def build_crystal_graph(
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

    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    c_arr = np.array(c_vec, dtype=np.float64)

    offsets = np.array(
        [di * a_arr + dj * b_arr + dk * c_arr
         for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)],
        dtype=np.float64,
    )

    edges_src, edges_dst = [], []
    edge_feats = []
    cutoff_sq = cutoff ** 2

    for i in range(n_atoms):
        cart_i = cart[i]
        for j in range(i, n_atoms):
            cart_j = cart[j]
            diff_vectors = cart_j + offsets - cart_i
            dist_sq = np.sum(diff_vectors ** 2, axis=1)
            valid_mask = (dist_sq > 0) & (dist_sq <= cutoff_sq)
            valid_indices = np.where(valid_mask)[0]
            for o_idx in valid_indices:
                dist = math.sqrt(float(dist_sq[o_idx]))
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
    """Crystal Graph Convolution layer — Xie & Grossman (PRL 120, 145301, 2018).

    Per-edge gate/core decomposition with full analytical backprop::

        z_ij = [v_i; v_j; e_ij]                      // concat (2*in_dim + edge_dim)
        g_ij = sigma(z_ij @ W_g + b_g)               // gate (sigmoid)
        c_ij = softplus(z_ij @ W_c + b_c)            // core
        m_i  = sum_{j in N(i)} g_ij * c_ij           // scatter-add
        v_i' = softplus(v_i + m_i)                   // residual update

    Backward is closed-form O(1) relative to forward — no finite differences.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 2):
        cat_dim = in_dim * 2 + edge_dim
        scale_g = math.sqrt(2.0 / cat_dim)
        scale_c = math.sqrt(2.0 / cat_dim)
        self.W_g = np.random.randn(cat_dim, out_dim).astype(np.float32) * scale_g * 0.5
        self.b_g = np.zeros(out_dim, dtype=np.float32)
        self.W_c = np.random.randn(cat_dim, out_dim).astype(np.float32) * scale_c * 0.5
        self.b_c = np.zeros(out_dim, dtype=np.float32)
        self._in_dim = in_dim
        self._edge_dim = edge_dim
        self._has_proj = in_dim != out_dim
        if self._has_proj:
            proj_scale = math.sqrt(2.0 / in_dim)
            self.W_proj = np.random.randn(in_dim, out_dim).astype(np.float32) * proj_scale * 0.5
        else:
            self.W_proj = np.eye(in_dim, dtype=np.float32)

    @property
    def in_dim(self) -> int:
        return self._in_dim

    @property
    def out_dim(self) -> int:
        return self.W_g.shape[1]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.maximum(x, -50.0).astype(np.float64))).astype(np.float32)

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.minimum(x, 50.0).astype(np.float64))).astype(np.float32)

    def forward(self, h: np.ndarray, edge_index: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
        n = h.shape[0]
        d = self.out_dim
        has_edges = edge_index.shape[1] > 0

        h_proj = h @ self.W_proj

        if has_edges:
            src = edge_index[0]
            dst = edge_index[1]

            z = np.concatenate([h[src], h[dst], edge_feat], axis=-1)

            pre_g = z @ self.W_g + self.b_g
            pre_c = z @ self.W_c + self.b_c

            gate = self._sigmoid(pre_g)
            core = self._softplus(pre_c)

            msg = gate * core

            m = np.zeros((n, d), dtype=np.float32)
            np.add.at(m, dst, msg)

            m_plus_h = h_proj + m
        else:
            src = np.array([], dtype=np.int64)
            dst = np.array([], dtype=np.int64)
            z = np.zeros((0, self._in_dim * 2 + self._edge_dim), dtype=np.float32)
            pre_g = np.zeros((0, d), dtype=np.float32)
            pre_c = np.zeros((0, d), dtype=np.float32)
            gate = np.zeros((0, d), dtype=np.float32)
            core = np.zeros((0, d), dtype=np.float32)
            msg = np.zeros((0, d), dtype=np.float32)
            m = np.zeros((n, d), dtype=np.float32)
            m_plus_h = h_proj

        out = self._softplus(m_plus_h)

        self._cache = {
            "h_input": h,
            "h_proj": h_proj,
            "src": src,
            "dst": dst,
            "z": z,
            "pre_g": pre_g,
            "pre_c": pre_c,
            "gate": gate,
            "core": core,
            "msg": msg,
            "m": m,
            "m_plus_h": m_plus_h,
            "has_edges": has_edges,
            "n": n,
        }
        return out

    # ------------------------------------------------------------------
    # Analytical backward pass
    # ------------------------------------------------------------------

    def backward(self, dL_dout: np.ndarray, edge_feat: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cache = self._cache
        d = self.out_dim
        has_edges = cache["has_edges"]

        # softplus'(x) = sigmoid(x)
        ds = dL_dout * self._sigmoid(cache["m_plus_h"])
        dL_dh_proj = ds.copy()
        dm = ds.copy()

        grads: dict[str, np.ndarray] = {}
        grads["b_g"] = np.zeros(d, dtype=np.float32)
        grads["b_c"] = np.zeros(d, dtype=np.float32)
        grads["W_g"] = np.zeros_like(self.W_g)
        grads["W_c"] = np.zeros_like(self.W_c)

        # Projection backward: h_proj = h @ W_proj
        grads["W_proj"] = cache["h_input"].T @ dL_dh_proj
        dL_dh = dL_dh_proj @ self.W_proj.T

        if has_edges:
            dm_dst = dm[cache["dst"]]

            dgate = dm_dst * cache["core"]
            dcore = dm_dst * cache["gate"]

            # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            dpre_g = dgate * cache["gate"] * (1.0 - cache["gate"])
            # softplus'(x) = sigmoid(x)
            dpre_c = dcore * self._sigmoid(cache["pre_c"])

            grads["b_g"] = np.sum(dpre_g, axis=0)
            grads["b_c"] = np.sum(dpre_c, axis=0)
            grads["W_g"] = cache["z"].T @ dpre_g
            grads["W_c"] = cache["z"].T @ dpre_c

            dz = dpre_g @ self.W_g.T + dpre_c @ self.W_c.T

            split1 = self._in_dim
            split2 = self._in_dim * 2
            dsrc = dz[:, :split1]
            ddst = dz[:, split1:split2]

            dL_dsrc = np.zeros_like(cache["h_input"])
            np.add.at(dL_dsrc, cache["src"], dsrc)
            dL_ddst = np.zeros_like(cache["h_input"])
            np.add.at(dL_ddst, cache["dst"], ddst)
            dL_dh = dL_dh + dL_dsrc + dL_ddst

        return dL_dh, grads


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
        n = x.shape[0]

        h = self.conv1.forward(x, edge_index, edge_feat)
        conv_caches = [self.conv1._cache]
        h_list = [h]

        for conv in self.convs:
            h_res = conv.forward(h, edge_index, edge_feat)
            conv_caches.append(conv._cache)
            h = h + h_res
            h_list.append(h)

        mean_pool = np.mean(h, axis=0, keepdims=True)
        max_pool = np.max(h, axis=0, keepdims=True)
        pooled = np.concatenate([mean_pool, max_pool], axis=-1)

        fc1_out = pooled @ self.fc1_W + self.fc1_b
        fc1_act = np.maximum(fc1_out, 0.0)
        fc2_out = fc1_act @ self.fc2_W + self.fc2_b

        self._training_cache = {
            "n": n,
            "h_list": h_list,
            "conv_caches": conv_caches,
            "mean_pool": mean_pool,
            "max_pool": max_pool,
            "pooled": pooled,
            "fc1_out": fc1_out,
            "fc1_act": fc1_act,
        }

        return fc2_out.flatten()

    def backward(self, target: np.ndarray, edge_feat: np.ndarray) -> dict[str, np.ndarray]:
        cache = self._training_cache
        n_nodes = cache["n"]
        h_list = cache["h_list"]
        _ = cache["conv_caches"]
        grads: dict[str, np.ndarray] = {}

        # Prediction from cache
        pred = cache["fc1_act"] @ self.fc2_W + self.fc2_b
        pred = pred.flatten()

        # L = sum((pred - target)^2)
        dL_dpred = 2.0 * (pred - target).astype(np.float32)
        dL_dfc2 = dL_dpred.reshape(1, -1)

        grads["fc2_W"] = cache["fc1_act"].T @ dL_dfc2
        grads["fc2_b"] = dL_dpred

        dL_dfc1_act = dL_dfc2 @ self.fc2_W.T
        dL_dfc1 = dL_dfc1_act * (cache["fc1_out"] > 0).astype(np.float32)

        grads["fc1_W"] = cache["pooled"].T @ dL_dfc1
        grads["fc1_b"] = dL_dfc1.flatten()

        dL_dpooled = dL_dfc1 @ self.fc1_W.T
        hidden_dim = self.hidden_dim
        dL_dmean = dL_dpooled[:, :hidden_dim]
        dL_dmax = dL_dpooled[:, hidden_dim:]

        h_final = h_list[-1]
        dL_dh = np.full_like(h_final, dL_dmean / n_nodes)

        max_vals = cache["max_pool"]
        mask = (h_final == max_vals).astype(np.float32)
        tied_counts = np.sum(mask, axis=0, keepdims=True)
        dL_dh = dL_dh + mask * (dL_dmax / n_nodes) / np.maximum(tied_counts, 1.0)

        dL_dh = dL_dh.astype(np.float32)

        for idx in range(len(self.convs) - 1, -1, -1):
            conv = self.convs[idx]
            dL_dh_res, conv_grads = conv.backward(dL_dh, edge_feat)
            key_prefix = f"convs_{idx}"
            for k, v in conv_grads.items():
                grads[f"{key_prefix}_{k}"] = v
            dL_dh = dL_dh + dL_dh_res

        _, conv1_grads = self.conv1.backward(dL_dh, edge_feat)
        for k, v in conv1_grads.items():
            grads[f"conv1_{k}"] = v

        return grads

    def _all_parameters(self) -> list[tuple[int, str, np.ndarray]]:
        params: list[tuple[int, str, np.ndarray]] = [
            (0, "W_g", self.conv1.W_g),
            (0, "b_g", self.conv1.b_g),
            (0, "W_c", self.conv1.W_c),
            (0, "b_c", self.conv1.b_c),
            (0, "W_proj", self.conv1.W_proj),
        ]
        for i, conv in enumerate(self.convs):
            params.extend([
                (i + 1, "W_g", conv.W_g),
                (i + 1, "b_g", conv.b_g),
                (i + 1, "W_c", conv.W_c),
                (i + 1, "b_c", conv.b_c),
                (i + 1, "W_proj", conv.W_proj),
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
                for name in ["W_g", "b_g", "W_c", "b_c", "W_proj"]:
                    key = f"{prefix}_{ci}_{name}" if conv_list is self.convs else f"{prefix}_{name}"
                    if key in grads:
                        new_val = optimizer.step(idx, key, getattr(conv, name), grads[key])
                        setattr(conv, name, new_val)
                    idx += 1
        for attr, key in [("fc1_W", "fc1_W"), ("fc1_b", "fc1_b"), ("fc2_W", "fc2_W"), ("fc2_b", "fc2_b")]:
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
                nf_loss = float(np.sum((pred - target) ** 2))

                grads = self.backward(target, edge_feat)
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
        weights["conv1_W_g"] = self.conv1.W_g
        weights["conv1_b_g"] = self.conv1.b_g
        weights["conv1_W_c"] = self.conv1.W_c
        weights["conv1_b_c"] = self.conv1.b_c
        weights["conv1_W_proj"] = self.conv1.W_proj
        for i, conv in enumerate(self.convs):
            weights[f"convs_{i}_W_g"] = conv.W_g
            weights[f"convs_{i}_b_g"] = conv.b_g
            weights[f"convs_{i}_W_c"] = conv.W_c
            weights[f"convs_{i}_b_c"] = conv.b_c
            weights[f"convs_{i}_W_proj"] = conv.W_proj
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
        while f"convs_{n_conv_layers}_W_g" in data:
            n_conv_layers += 1
        n_conv_layers = max(n_conv_layers, 1)
        output_dim = data["fc2_b"].shape[0]
        hidden_dim = data["conv1_W_g"].shape[1]

        model = CGCNNModel(
            node_dim=data["conv1_W_g"].shape[0],
            hidden_dim=hidden_dim,
            n_conv_layers=n_conv_layers,
            output_dim=output_dim,
        )
        model.conv1.W_g = data["conv1_W_g"]
        model.conv1.b_g = data["conv1_b_g"]
        model.conv1.W_c = data["conv1_W_c"]
        model.conv1.b_c = data["conv1_b_c"]
        if "conv1_W_proj" in data:
            model.conv1.W_proj = data["conv1_W_proj"]
        for i in range(n_conv_layers - 1):
            model.convs[i].W_g = data[f"convs_{i}_W_g"]
            model.convs[i].b_g = data[f"convs_{i}_b_g"]
            model.convs[i].W_c = data[f"convs_{i}_W_c"]
            model.convs[i].b_c = data[f"convs_{i}_b_c"]
            if f"convs_{i}_W_proj" in data:
                model.convs[i].W_proj = data[f"convs_{i}_W_proj"]
        model.fc1_W = data["fc1_W"]
        model.fc1_b = data["fc1_b"]
        model.fc2_W = data["fc2_W"]
        model.fc2_b = data["fc2_b"]
        logger.info(f"GNN model loaded from {path}")
        return model


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
_PACKAGE_MODEL_PATH = Path(__file__).parent / "gnn_kpoint_v1.npz"


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


def _get_or_create_model(model_path: str | None = None, default_dir: str | None = None) -> CGCNNModel:  # noqa: C901
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

    if _PACKAGE_MODEL_PATH.exists() and not os.environ.get("FORGE_GNN_RETRAIN"):
        try:
            return CGCNNModel.load(str(_PACKAGE_MODEL_PATH))
        except Exception:
            pass

    if _DEFAULT_MODEL_PATH.exists() and not os.environ.get("FORGE_GNN_RETRAIN"):
        try:
            return CGCNNModel.load(str(_DEFAULT_MODEL_PATH))
        except Exception:
            pass

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
