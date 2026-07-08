"""
Workflow Provenance System for WIEN2k Computational Pipelines.

Provides a lightweight DAG-based workflow engine inspired by AiiDA/FireWorks
patterns but designed for simplicity and zero-dependency deployment. Uses
SQLite as the persistence backend with WAL mode for concurrent access.

Key Features:
- WorkflowNode: dataclass representing a single computational step with
  full provenance metadata (timing, parameters, parent/child DAG edges).
- WorkflowDAG: Directed Acyclic Graph orchestrator supporting
  topological sort, ready-node detection, and visualisation.
- WorkflowStore: SQLite-backed CRUD store with WAL mode, statistics,
  and workflow listing.
- Pre-built WIEN2k templates: SCF, convergence, and band structure.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Domain Enumerations
# =============================================================================


class NodeStatus(str, Enum):
    """Canonical statuses for a WorkflowNode lifecycle."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Core Data Model
# =============================================================================


@dataclass
class WorkflowNode:
    """
    Single computational step within a provenance-tracked DAG.

    Attributes
    ----------
    node_id : str
        UUID v4 identifier for this node.
    name : str
        Human-readable label (e.g. ``"lapw0"``, ``"lapw1"``).
    status : NodeStatus
        Current lifecycle state.
    start_time : Optional[float]
        Unix timestamp when execution began.
    end_time : Optional[float]
        Unix timestamp when execution finished.
    input_files : List[str]
        Files consumed by this step (e.g. ``["case.struct", "case.in1"]``).
    output_files : List[str]
        Files produced by this step (e.g. ``["case.output1", "case.scf"]``).
    parameters : Dict[str, Any]
        Computational parameters (kpoints, RKmax, atoms, etc.).
    parent_ids : List[str]
        Upstream node IDs forming the DAG edges.  A node is *ready* when
        all parents have status ``COMPLETED``.
    metadata : Dict[str, Any]
        Auxiliary provenance (user, machine, wien2k version, commit hash).
    """
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Elapsed wall-clock seconds, or *None* if not yet finished."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def is_ready(self) -> bool:
        """A node is ready when it is *pending* (has not started yet)."""
        return self.status == NodeStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowNode:
        """Rehydrate a node from a dictionary (supports legacy fields)."""
        node = cls(
            node_id=data.get("node_id", ""),
            name=data.get("name", ""),
            status=NodeStatus(data.get("status", "pending")),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            input_files=data.get("input_files", []),
            output_files=data.get("output_files", []),
            parameters=data.get("parameters", {}),
            parent_ids=data.get("parent_ids", []),
            metadata=data.get("metadata", {}),
        )
        return node

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkflowNode):
            return NotImplemented
        return self.node_id == other.node_id


# =============================================================================
# Directed Acyclic Graph Orchestrator
# =============================================================================


class WorkflowDAG:
    """
    Lightweight DAG that orchestrates WorkflowNode instances.

    Tracks parent/child relationships, performs topological sorting,
    identifies nodes that are ready to execute, and emits textual
    visualisations.

    Parameters
    ----------
    workflow_id : Optional[str]
        External identifier; auto-generated UUID if omitted.
    name : str
        Human-readable label for the entire workflow.
    """
    def __init__(self, workflow_id: Optional[str] = None, name: str = "") -> None:
        self.workflow_id: str = workflow_id or uuid.uuid4().hex
        self.name: str = name
        self._nodes: Dict[str, WorkflowNode] = {}
        self._children: Dict[str, List[str]] = defaultdict(list)

    # ---- Node Management ----------------------------------------------------

    def add_node(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
        input_files: Optional[List[str]] = None,
        output_files: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new node in the DAG.

        Returns the generated ``node_id`` which callers should retain for
        subsequent status updates or child-node wiring.
        """
        parent_ids = parents or []
        _validate_parents_exist(parent_ids, set(self._nodes.keys()))

        node = WorkflowNode(
            name=name,
            parameters=parameters or {},
            parent_ids=parent_ids,
            input_files=input_files or [],
            output_files=output_files or [],
            metadata=metadata or {},
        )
        self._nodes[node.node_id] = node
        for pid in parent_ids:
            self._children[pid].append(node.node_id)
        logger.debug("Added node %s (%s) with %d parent(s)", node.node_id, name, len(parent_ids))
        return node.node_id

    def get_node(self, node_id: str) -> WorkflowNode:
        """Retrieve a node by its identifier."""
        _validate_node_exists(node_id, self._nodes)
        return self._nodes[node_id]

    def iter_nodes(self) -> Iterator[WorkflowNode]:
        """Yield every node in the DAG."""
        yield from self._nodes.values()

    # ---- Status Lifecycle ---------------------------------------------------

    def set_node_status(
        self,
        node_id: str,
        status: NodeStatus,
        outputs: Optional[List[str]] = None,
    ) -> None:
        """
        Transition a node's status, optionally recording output files.

        Automatically stamps *start_time* when entering RUNNING and
        *end_time* when entering COMPLETED or FAILED.
        """
        node = self.get_node(node_id)
        now = time.time()
        node.status = status
        if status == NodeStatus.RUNNING and node.start_time is None:
            node.start_time = now
        if status in (NodeStatus.COMPLETED, NodeStatus.FAILED) and node.end_time is None:
            node.end_time = now
        if outputs is not None:
            node.output_files = list(outputs)
        logger.debug("Node %s -> %s (duration=%.1fs)", node_id, status.value, node.duration or 0)

    def get_ready_nodes(self) -> List[WorkflowNode]:
        """
        Return *pending* nodes whose upstream parents are all COMPLETED.

        Nodes with zero parents are immediately ready.
        """
        ready: List[WorkflowNode] = []
        for node in self._nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            if all(
                self._nodes[pid].status == NodeStatus.COMPLETED
                for pid in node.parent_ids
            ):
                ready.append(node)
        return ready

    # ---- Topological Sort ---------------------------------------------------

    def get_execution_order(self) -> List[str]:
        """
        Return node IDs in topological (Kahn) order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        in_degree: Dict[str, int] = {nid: len(node.parent_ids) for nid, node in self._nodes.items()}
        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: List[str] = []

        adj: Dict[str, List[str]] = defaultdict(list)
        for nid, node in self._nodes.items():
            for pid in node.parent_ids:
                adj[pid].append(nid)

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in adj.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            remaining = set(self._nodes) - set(order)
            raise ValueError(f"Cycle detected involving nodes: {remaining}")
        return order

    # ---- Serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire DAG to a JSON-friendly dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowDAG:
        """Reconstruct a DAG from a serialized dictionary."""
        dag = cls(workflow_id=data.get("workflow_id"), name=data.get("name", ""))
        for node_data in data.get("nodes", []):
            node = WorkflowNode.from_dict(node_data)
            dag._nodes[node.node_id] = node
            for pid in node.parent_ids:
                dag._children[pid].append(node.node_id)
        return dag

    # ---- Visualization ------------------------------------------------------

    def visualize(self, fmt: str = "ascii") -> str:
        """
        Render the DAG structure.

        Parameters
        ----------
        fmt : str
            ``"ascii"`` for tree-style terminal output,
            ``"mermaid"`` for Mermaid.js markdown.

        Returns
        -------
        str
            Formatted representation.
        """
        if fmt == "mermaid":
            return self._visualize_mermaid()
        return self._visualize_ascii()

    def _visualize_ascii(self) -> str:
        """ASCII tree representation of the DAG."""
        lines: List[str] = [f"WorkflowDAG: {self.name} ({self.workflow_id})"]
        lines.append("-" * 60)

        root_ids = [nid for nid, node in self._nodes.items() if not node.parent_ids]
        visited: set[str] = set()

        def _render_subtree(nid: str, prefix: str, is_last: bool) -> None:
            if nid in visited:
                return
            visited.add(nid)
            node = self._nodes[nid]
            connector = "└── " if is_last else "├── "
            status_icon = _status_icon(node.status)
            lines.append(f"{prefix}{connector}{status_icon} {node.name} [{node.node_id[:8]}]")
            children = self._children.get(nid, [])
            for i, child in enumerate(children):
                extension = "    " if is_last else "│   "
                _render_subtree(child, prefix + extension, i == len(children) - 1)

        for i, rid in enumerate(root_ids):
            _render_subtree(rid, "", i == len(root_ids) - 1)

        # Show orphaned nodes (should not happen in a well-formed DAG)
        orphaned = [nid for nid in self._nodes if nid not in visited]
        for nid in orphaned:
            node = self._nodes[nid]
            lines.append(f"  ? {node.name} [{nid[:8]}] (orphaned)")

        return "\n".join(lines)

    def _visualize_mermaid(self) -> str:
        """Mermaid.js flowchart representation."""
        lines = ["```mermaid", "graph TD"]
        for nid, node in self._nodes.items():
            short = nid[:8]
            label = f"{node.name}<br/>{node.status.value}"
            shape_map = {
                NodeStatus.PENDING: f'{short}["{label}"]',
                NodeStatus.RUNNING: f'{short}("{label}")',
                NodeStatus.COMPLETED: f'{short}[("{label}")]',
                NodeStatus.FAILED: f'{short}{{"{label}"}}',
            }
            lines.append(f"    {shape_map.get(node.status, shape_map[NodeStatus.PENDING])}")
            for pid in node.parent_ids:
                lines.append(f"    {pid[:8]} --> {short}")
        lines.append("```")
        return "\n".join(lines)

    # ---- Properties ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"<WorkflowDAG id={self.workflow_id[:8]} nodes={len(self._nodes)} name='{self.name}'>"


# =============================================================================
# SQLite-Backed Persistence Store
# =============================================================================


class WorkflowStore:
    """
    SQLite-backed CRUD store for WorkflowDAG instances.

    Uses WAL journal mode for concurrent read/write access and stores
    node data as JSON blobs.  All public methods accept a ``db_path``
    parameter so the store can be used with multiple database files.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file (created if absent).
    """
    def __init__(self, db_path: Path = Path("wien2k_workflows.db")) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode, foreign keys, and row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create schema if it does not exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id  TEXT PRIMARY KEY,
                    name         TEXT NOT NULL DEFAULT '',
                    created_at   REAL NOT NULL,
                    updated_at   REAL NOT NULL,
                    dag_json     TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_workflows_name
                    ON workflows(name);
            """)
            conn.commit()

    # ---- Core CRUD ----------------------------------------------------------

    def save_dag(self, dag: WorkflowDAG) -> str:
        """
        Persist a WorkflowDAG (insert or update).

        Returns the workflow_id.
        """
        now = time.time()
        dag_json = json.dumps(dag.to_dict(), default=str)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT workflow_id FROM workflows WHERE workflow_id = ?",
                (dag.workflow_id,),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE workflows SET name = ?, updated_at = ?, dag_json = ? WHERE workflow_id = ?",
                    (dag.name, now, dag_json, dag.workflow_id),
                )
            else:
                conn.execute(
                    "INSERT INTO workflows (workflow_id, name, created_at, updated_at, dag_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (dag.workflow_id, dag.name, now, now, dag_json),
                )
            conn.commit()

        logger.info("Saved workflow %s (%d nodes)", dag.workflow_id[:8], len(dag))
        return dag.workflow_id

    def load_dag(self, workflow_id: str) -> WorkflowDAG:
        """
        Load a previously saved WorkflowDAG.

        Raises
        ------
        KeyError
            If the workflow_id is not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dag_json FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Workflow '{workflow_id}' not found in store")

        data = json.loads(row["dag_json"])
        return WorkflowDAG.from_dict(data)

    def delete_workflow(self, workflow_id: str) -> bool:
        """Remove a workflow.  Returns True if it existed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted workflow %s", workflow_id[:8])
        return deleted

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        Return summary metadata for every stored workflow.

        Each summary dictionary contains:
        - workflow_id
        - name
        - created_at
        - updated_at
        - node_count
        - completed_count
        - failed_count
        """
        summaries: List[Dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT workflow_id, name, created_at, updated_at, dag_json FROM workflows ORDER BY updated_at DESC"
            ).fetchall()

        for row in rows:
            dag_data = json.loads(row["dag_json"])
            nodes = dag_data.get("nodes", [])
            completed = sum(1 for n in nodes if n.get("status") == "completed")
            failed = sum(1 for n in nodes if n.get("status") == "failed")
            summaries.append({
                "workflow_id": row["workflow_id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "node_count": len(nodes),
                "completed_count": completed,
                "failed_count": failed,
            })
        return summaries

    # ---- Statistics & Analytics ---------------------------------------------

    def get_statistics(self, workflow_id: str) -> Dict[str, Any]:
        """
        Compute aggregate statistics for a workflow.

        Returns a dictionary with the following keys:
        - total_nodes
        - completed / failed / pending / running counts
        - total_wall_time_sec (sum of completed-node durations)
        - avg_duration_sec
        - max_duration_sec
        - max_duration_node (name of the slowest node)
        - success_rate (completed / total)
        - bottlenecks (list of node names whose duration exceeds avg + 2σ)
        """
        dag = self.load_dag(workflow_id)
        nodes = list(dag.iter_nodes())

        counts = {s.value: 0 for s in NodeStatus}
        durations: List[Tuple[str, float]] = []

        for node in nodes:
            counts[node.status.value] += 1
            if node.duration is not None:
                durations.append((node.name, node.duration))

        total = len(nodes)
        total_wall = sum(d for _, d in durations)
        avg_dur = (total_wall / len(durations)) if durations else 0.0
        max_dur = max((d for _, d in durations), default=0.0)
        max_node = max(durations, key=lambda x: x[1])[0] if durations else ""

        # Simple bottleneck detection: nodes > avg + 2 * stddev
        if len(durations) > 1 and avg_dur > 0:
            variance = sum((d - avg_dur) ** 2 for _, d in durations) / len(durations)
            stddev = variance ** 0.5
            threshold = avg_dur + 2 * stddev
            bottlenecks = [name for name, d in durations if d > threshold]
        else:
            bottlenecks = []

        return {
            "total_nodes": total,
            "completed": counts["completed"],
            "failed": counts["failed"],
            "pending": counts["pending"],
            "running": counts["running"],
            "total_wall_time_sec": round(total_wall, 2),
            "avg_duration_sec": round(avg_dur, 2),
            "max_duration_sec": round(max_dur, 2),
            "max_duration_node": max_node,
            "success_rate": round(counts["completed"] / total, 4) if total else 0.0,
            "bottlenecks": bottlenecks,
        }


# =============================================================================
# Pre-Built WIEN2k Workflow Templates
# =============================================================================


def create_wien2k_workflow(
    case_name: str = "case",
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowDAG:
    """
    Build the standard WIEN2k SCF pipeline:

        init_lapw → run_lapw → save_lapw

    Each stage is a separate WorkflowNode so provenance is tracked
    at the granularity of individual WIEN2k programs.

    Parameters
    ----------
    case_name : str
        Root name for input/output files (e.g. ``"TiC"``).
    parameters : Optional[Dict]
        Overrides for default SCF parameters (kpoints, RKmax, etc.).
    metadata : Optional[Dict]
        User, machine, and WIEN2k version provenance.
    """
    params = parameters or {}
    meta = metadata or {}

    dag = WorkflowDAG(name=f"{case_name}_scf")

    init_id = dag.add_node(
        name="init_lapw",
        parameters=params,
        input_files=[f"{case_name}.struct"],
        output_files=[
            f"{case_name}.in0",
            f"{case_name}.in1",
            f"{case_name}.in2",
            f"{case_name}.inm",
            f"{case_name}.inst",
            f"{case_name}.klist",
            f"{case_name}.outputd",
        ],
        metadata=meta,
    )

    run_id = dag.add_node(
        name="run_lapw",
        parameters=params,
        parents=[init_id],
        input_files=[
            f"{case_name}.in0",
            f"{case_name}.in1",
            f"{case_name}.in2",
            f"{case_name}.inm",
            f"{case_name}.klist",
        ],
        output_files=[
            f"{case_name}.scf",
            f"{case_name}.output1",
            f"{case_name}.output2",
            f"{case_name}.energy",
            f"{case_name}.dayfile",
        ],
        metadata=meta,
    )

    dag.add_node(
        name="save_lapw",
        parameters=params,
        parents=[run_id],
        input_files=[f"{case_name}.scf", f"{case_name}.energy"],
        output_files=[f"{case_name}_saved.scf", f"{case_name}_saved.energy"],
        metadata=meta,
    )

    return dag


def create_convergence_workflow(
    kpoint_grids: List[Tuple[int, int, int]],
    rkmax_values: List[float],
    case_name: str = "case",
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowDAG:
    """
    Build a convergence-study DAG that independently varies k-point mesh
    and RKmax, then collects and compares total energies.

    Structure::

        [splitter]
         ├── kpoint_4x4x4_rk6  →  ...
         ├── kpoint_6x6x6_rk6  →  ...
         └── kpoint_6x6x6_rk7  →  ...
         [merger]

    Each leaf node runs an SCF cycle with a unique (kpoints, RKmax) pair.

    Parameters
    ----------
    kpoint_grids : List[Tuple[int, int, int]]
        List of k-point meshes to test.
    rkmax_values : List[float]
        List of RKmax values to test.
    case_name : str
        Base name for input/output files.
    metadata : Optional[Dict]
        Provenance metadata.
    """
    meta = metadata or {}
    dag = WorkflowDAG(name=f"{case_name}_convergence")

    splitter_id = dag.add_node(
        name="convergence_splitter",
        parameters={"kpoint_grids": kpoint_grids, "rkmax_values": rkmax_values},
        metadata=meta,
    )

    scf_ids: List[str] = []
    for kgrid in kpoint_grids:
        for rk in rkmax_values:
            label = f"kpoint_{kgrid[0]}x{kgrid[1]}x{kgrid[2]}_rk{rk:.0f}".replace(".", "p")
            scf_params = {
                "kpoints": list(kgrid),
                "RKmax": rk,
                "case": case_name,
            }
            scf_id = dag.add_node(
                name=label,
                parameters=scf_params,
                parents=[splitter_id],
                input_files=[f"{case_name}.struct"],
                output_files=[f"{label}.scf", f"{label}.energy"],
                metadata=meta,
            )
            scf_ids.append(scf_id)

    dag.add_node(
        name="convergence_merger",
        parameters={},
        parents=scf_ids,
        output_files=["convergence_report.csv", "convergence_plot.png"],
        metadata=meta,
    )

    return dag


def create_band_structure_workflow(
    case_name: str = "case",
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowDAG:
    """
    Build the standard WIEN2k band-structure pipeline:

        SCF (init_lapw→run_lapw) → kgen → lapw1 → spaghetti

    Parameters
    ----------
    case_name : str
        Root name for input/output files.
    parameters : Optional[Dict]
        Overrides for computational parameters.
    metadata : Optional[Dict]
        Provenance metadata.
    """
    params = parameters or {}
    meta = metadata or {}
    dag = WorkflowDAG(name=f"{case_name}_bands")

    init_id = dag.add_node(
        name="init_lapw",
        parameters=params,
        input_files=[f"{case_name}.struct"],
        output_files=[
            f"{case_name}.in0", f"{case_name}.in1", f"{case_name}.in2",
            f"{case_name}.inm", f"{case_name}.inst", f"{case_name}.klist",
        ],
        metadata=meta,
    )

    scf_id = dag.add_node(
        name="run_lapw_scf",
        parameters=params,
        parents=[init_id],
        input_files=[
            f"{case_name}.in0", f"{case_name}.in1", f"{case_name}.in2",
            f"{case_name}.klist",
        ],
        output_files=[f"{case_name}.scf", f"{case_name}.energy"],
        metadata=meta,
    )

    kgen_id = dag.add_node(
        name="kgen",
        parameters={"task": "bandstructure", **params},
        parents=[scf_id],
        input_files=[f"{case_name}.struct", f"{case_name}.scf"],
        output_files=[f"{case_name}.kgen", f"{case_name}.klist_band"],
        metadata=meta,
    )

    lapw1_id = dag.add_node(
        name="lapw1_bands",
        parameters=params,
        parents=[kgen_id],
        input_files=[
            f"{case_name}.in1", f"{case_name}.in0", f"{case_name}.klist_band",
        ],
        output_files=[f"{case_name}.energy_band", f"{case_name}.output1_band"],
        metadata=meta,
    )

    spaghetti_id = dag.add_node(
        name="spaghetti",
        parameters=params,
        parents=[lapw1_id],
        input_files=[
            f"{case_name}.energy_band", f"{case_name}.klist_band",
            f"{case_name}.struct",
        ],
        output_files=[
            f"{case_name}.spaghetti_ene", f"{case_name}.spaghetti_ps",
            f"{case_name}.band.agr",
        ],
        metadata=meta,
    )

    return dag


# =============================================================================
# Internal Helpers
# =============================================================================


def _status_icon(status: NodeStatus) -> str:
    """Return a single-character icon for the given node status."""
    return {
        NodeStatus.PENDING: "○",
        NodeStatus.RUNNING: "◉",
        NodeStatus.COMPLETED: "●",
        NodeStatus.FAILED: "✗",
    }.get(status, "?")


def _validate_parents_exist(parent_ids: List[str], existing: set) -> None:
    """Raise ValueError if any parent is not already in the DAG."""
    missing = set(parent_ids) - existing
    if missing:
        raise ValueError(
            f"Parent node(s) not found in DAG: {missing}. "
            f"Add parent nodes before referencing them."
        )


def _validate_node_exists(node_id: str, nodes: Dict[str, WorkflowNode]) -> None:
    """Raise KeyError if the node does not exist."""
    if node_id not in nodes:
        raise KeyError(f"Node '{node_id}' not found in the DAG.")


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "NodeStatus",
    "WorkflowDAG",
    "WorkflowNode",
    "WorkflowStore",
    "create_band_structure_workflow",
    "create_convergence_workflow",
    "create_wien2k_workflow",
]
