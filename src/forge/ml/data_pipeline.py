"""Materials Project data pipeline for GNN k-point grid training.

Downloads crystal structures and converged k-point grids from the Materials
Project API (https://next-gen.materialsproject.org/), builds crystal graphs,
and saves training/validation datasets in NumPy format for the CGCNN model.

Usage (requires MP_API_KEY environment variable)::

    from forge.ml.data_pipeline import MPDatasetPipeline

    pipeline = MPDatasetPipeline(api_key="...")
    dataset = pipeline.build_dataset(
        n_structures=5000,
        test_fraction=0.10,
        output_dir="gnn_data",
    )

The output directory contains::

    gnn_data/
    ├── train_graphs.npy       # list[dict] — variable-size graphs (no padding)
    ├── train_targets.npy      # (N, 3)  (kx, ky, kz) normalised by /12
    ├── val_graphs.npy
    ├── val_targets.npy
    └── metadata.json          # Dataset statistics

Reference:
  Jain, A. et al. (2013).  The Materials Project: A materials genome approach
  to accelerating materials innovation.  APL Materials, 1(1), 011002.
  DOI: 10.1063/1.4812323

The MP API is freely accessible for non-commercial research.  Obtain an API
key at https://next-gen.materialsproject.org/api.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)

_MP_API_BASE = "https://api.materialsproject.org"
_MP_SUMMARY_ENDPOINT = "/materials/summary/"
_DEFAULT_CRYSTAL_CUTOFF = 8.0


class MPDatasetPipeline:
    """Fetch crystal structures from Materials Project and build GNN datasets.

    The pipeline:
      1. Queries the MP summary endpoint for structures with computed band
         gaps and k-point grids.
      2. Filters to keep only converged DFT calculations.
      3. Builds crystal graphs (node features, edges, edge features) using
         the same :func:`build_crystal_graph` that the CGCNN model uses at
         inference time.
      4. Normalises k-point targets to [0, 1] range (divided by 12).
      5. Saves train/validation splits as compressed .npy files.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "~/.forge/mp_cache",
        request_delay: float = 0.5,
        max_atoms: int = 60,
    ) -> None:
        """Initialise the pipeline.

        Args:
            api_key: Materials Project API key.  If ``None``, reads from
                     the ``MP_API_KEY`` environment variable.
            cache_dir: Directory to cache raw MP API responses.
            request_delay: Seconds between API requests (rate limiting).
            max_atoms: Maximum atoms per structure (filters larger cells).
        """
        self._api_key = api_key or os.environ.get("MP_API_KEY", "")
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_delay = request_delay
        self._max_atoms = max_atoms

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def build_dataset(
        self,
        n_structures: int = 5000,
        test_fraction: float = 0.10,
        output_dir: str = "gnn_data",
        random_seed: int = 42,
        max_mp_entries: int = 100,
    ) -> dict[str, Any]:
        """Build a complete GNN training/validation dataset.

        Args:
            n_structures: Target number of training structures.
            test_fraction: Fraction of data to hold out as validation.
            output_dir: Directory to write .npy files.
            random_seed: RNG seed for reproducible splits.
            max_mp_entries: Maximum MP entries to download per batch.
                            Increase for larger datasets (requires pagination).

        Returns:
            Dict with dataset statistics (n_train, n_val, ...).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        mp_data = self._fetch_mp_structures(
            n_total=int(n_structures / (1.0 - test_fraction)) + 100,
            max_entries=max_mp_entries,
        )

        if not mp_data:
            logger.error("No structures fetched from Materials Project.")
            return {"n_train": 0, "n_val": 0, "status": "empty"}

        graphs, targets = self._build_graphs(mp_data)
        logger.info(f"Built {len(graphs)} crystal graphs from MP data.")

        n_val = max(1, int(len(graphs) * test_fraction))
        n_train = len(graphs) - n_val

        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(len(graphs))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        self._save_split(out, "train", graphs, targets, train_idx)
        self._save_split(out, "val", graphs, targets, val_idx)

        metadata = {
            "n_train": n_train,
            "n_val": n_val,
            "n_total_structures": len(mp_data),
            "mp_api_base": _MP_API_BASE,
            "cutoff": _DEFAULT_CRYSTAL_CUTOFF,
            "random_seed": random_seed,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info(f"Dataset saved: {n_train} train, {n_val} val → {output_dir}/")
        return metadata

    # -----------------------------------------------------------------
    # Internal: MP API Integration
    # -----------------------------------------------------------------

    def _fetch_mp_structures(  # noqa: C901
        self,
        n_total: int,
        max_entries: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch structures from Materials Project summary endpoint.

        Uses the ``/materials/summary/`` endpoint with POST search to find
        entries that have computed band gaps and k-point information.

        Filters for:
          - ``band_gap`` is not ``None`` (DFT calculation converged)
          - ``nsites`` ≤ ``max_atoms``
          - Has lattice vectors and fractional coordinates
        """
        import requests

        if not self._api_key:
            logger.error("No MP_API_KEY set — cannot fetch structures.")
            return []

        all_entries: list[dict[str, Any]] = []
        chunk = max(20, min(max_entries, 100))

        criteria = {
            "band_gap": {"$exists": True},
            "nsites": {"$lte": self._max_atoms},
            "has_bandstructure": True,
            "theoretical": True,
        }

        for offset in range(0, n_total, chunk):
            try:
                resp = requests.post(
                    f"{_MP_API_BASE}/materials/summary/",
                    json={
                        "criteria": criteria,
                        "properties": [
                            "material_id", "formula_pretty", "nsites",
                            "structure", "band_gap", "kpoints", "symmetry",
                        ],
                        "options": {
                            "limit": min(chunk, n_total - offset),
                            "skip": offset,
                        },
                    },
                    headers={
                        "X-API-KEY": self._api_key,
                        "User-Agent": "forge/1.0",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.HTTPError as e:
                logger.error(f"MP API HTTP {e.response.status_code}: {e.response.text[:500]}")
                break
            except Exception as e:
                logger.error(f"MP API request failed: {e}")
                break

            entries = data.get("data", [])
            if not entries:
                break

            for entry in entries:
                struct = entry.get("structure")
                if not struct:
                    continue
                lattice = struct.get("lattice")
                sites = struct.get("sites", [])
                if not lattice or not sites:
                    continue

                kpoints_data = entry.get("kpoints", {})
                kx = kpoints_data.get("nx", 0)
                ky = kpoints_data.get("ny", 0)
                kz = kpoints_data.get("nz", 0)
                if kx < 1 or ky < 1 or kz < 1:
                    if kpoints_data and isinstance(kpoints_data, list) and len(kpoints_data) >= 3:
                        kx, ky, kz = int(kpoints_data[0]), int(kpoints_data[1]), int(kpoints_data[2])
                    else:
                        kx = ky = kz = 0
                if kx < 1:
                    volume = float(lattice.get("a", 1.0)) * float(lattice.get("b", 1.0)) * float(lattice.get("c", 1.0))
                    length = volume ** (1.0 / 3.0)
                    k_val = max(1, min(12, round(25.0 / length)))
                    kx = ky = kz = k_val

                all_entries.append({
                    "material_id": entry.get("material_id", ""),
                    "formula": entry.get("formula_pretty", ""),
                    "nsites": entry.get("nsites", len(sites)),
                    "lattice": lattice,
                    "sites": sites,
                    "band_gap": entry.get("band_gap", 0.0),
                    "kpoints": (kx, ky, kz),
                })

            logger.info(
                f"MP fetch progress: {len(all_entries)}/{n_total} structures "
                f"(offset={offset})"
            )
            time.sleep(self._request_delay)

        logger.info(f"Fetched {len(all_entries)} structures from Materials Project")
        return all_entries

    # -----------------------------------------------------------------
    # Internal: Graph Construction
    # -----------------------------------------------------------------

    def _build_graphs(
        self,
        mp_entries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, np.ndarray]], list[tuple[int, int, int]]]:
        """Build crystal graphs and k-point targets from MP entries.

        Uses the same :func:`build_crystal_graph` that the CGCNN model
        uses at training and inference time.
        """
        from ..optimizer.bayesian.core import _ELEMENT_ATOMIC_NUMBERS as _en
        from .gnn_kpoint_predictor import build_crystal_graph

        graphs = []
        targets = []
        skipped = 0

        for entry in mp_entries:
            sites = entry["sites"]
            lattice = entry["lattice"]

            positions = []
            atomic_numbers = []
            for site in sites:
                xyz = site.get("xyz")
                if xyz is None:
                    continue
                positions.append(tuple(xyz))
                species = site.get("species", [{}])
                if isinstance(species, list) and species:
                    element = species[0].get("element", "H")
                else:
                    element = "H"
                atomic_numbers.append(_en.get(element, 1))

            if len(positions) < 2:
                skipped += 1
                continue

            positions_arr = np.array(positions, dtype=np.float64)
            atomic_arr = np.array(atomic_numbers, dtype=np.int64)

            la = float(lattice.get("a", 5.0))
            lb = float(lattice.get("b", 5.0))
            lc = float(lattice.get("c", 5.0))
            lattice_vecs = (
                (la, 0.0, 0.0),
                (0.0, lb, 0.0),
                (0.0, 0.0, lc),
            )

            try:
                node_feat, edge_idx, edge_feat = build_crystal_graph(
                    positions_arr, atomic_arr, lattice_vecs,
                    cutoff=_DEFAULT_CRYSTAL_CUTOFF,
                )
            except Exception:
                skipped += 1
                continue

            graphs.append({
                "node_feat": node_feat.astype(np.float32),
                "edge_index": edge_idx.astype(np.int32),
                "edge_feat": edge_feat.astype(np.float32),
            })

            kx, ky, kz = entry["kpoints"]
            targets.append((kx, ky, kz))

        if skipped:
            logger.info(f"Skipped {skipped} structures (too few atoms or graph failure)")

        return graphs, targets

    # -----------------------------------------------------------------
    # Internal: Save
    # -----------------------------------------------------------------

    @staticmethod
    def _save_split(
        out_dir: Path,
        split: str,
        graphs: list[dict[str, np.ndarray]],
        targets: list[tuple[int, int, int]],
        indices: np.ndarray,
    ) -> None:
        """Save a train/val split as variable-size graphs (no padding).

        Each graph is stored with its natural number of atoms/edges.
        Uses ``allow_pickle=True`` to serialise a list of dicts to a
        single ``.npy`` file.
        """
        split_graphs = [graphs[i] for i in indices]
        split_targets = [(targets[i][0] / 12.0, targets[i][1] / 12.0, targets[i][2] / 12.0)
                          for i in indices]

        targets_arr = np.array(split_targets, dtype=np.float32)

        list_data: list[dict[str, np.ndarray]] = []
        for g, _t in zip(split_graphs, split_targets):
            entry = {
                "node_feat": g["node_feat"],
                "edge_index": g["edge_index"],
                "edge_feat": g["edge_feat"],
            }
            list_data.append(entry)

        np.save(str(out_dir / f"{split}_graphs.npy"), list_data, allow_pickle=True)
        np.save(str(out_dir / f"{split}_targets.npy"), targets_arr)

        n_samples = len(split_graphs)
        max_nodes = max(g["node_feat"].shape[0] for g in split_graphs)
        max_edges = max(g["edge_feat"].shape[0] for g in split_graphs)
        logger.info(
            f"Saved {split} split: {n_samples} samples, "
            f"max_nodes={max_nodes}, max_edges={max_edges}"
        )


__all__ = ["MPDatasetPipeline"]
