"""
SQLite-Based Execution History Store for Tracking & Learning from Past Runs.
Production features:
• Persistent SQLite database at ~/.forge/history.db with auto-created schema
• ExecutionRecord dataclass capturing full run metadata (timing, resources, success)
• Flexible filtering via query() with arbitrary column-value constraints
• Similarity search (get_similar) matching by problem size and backend
• Best-configuration retrieval (get_best_config) based on walltime ranking
• Aggregate statistics (get_statistics) with averages, counts, and mode distributions
• Thread-safe with RLock, connection pooling, and context-manager protocol
• Historical suggestion engine (suggest_from_history) for warm-start configs
• Efficiency metric (compute_efficiency) normalized across problem sizes
All documentation and inline comments are in English per project standards.
"""

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from ..logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_HISTORY_DIR = Path.home() / ".forge"
DEFAULT_HISTORY_DB = DEFAULT_HISTORY_DIR / "history.db"
MAX_POOL_SIZE = 5

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecutionRecord:
    """
    Complete record of a single WIEN2k parallel execution run.
    Captures problem parameters, resource allocation, timing, and outcome.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    backend: str = "wien2k"
    mode: str = "hybrid"
    nmat: int = 0
    nkpt: int = 0
    atoms: int = 0
    rkmax: float = 7.0
    total_cores: int = 1
    omp_threads: int = 1
    nodes_used: int = 1
    walltime_sec: float = 0.0
    efficiency_pct: float = 0.0
    convergence_cycles: int = 0
    memory_gb_used: float = 0.0
    node_list: list[str] = field(default_factory=list)
    success: bool = False
    tags: list[str] = field(default_factory=list)
    # --- Structural features (extracted from WIEN2k input) ---
    nbands: int = 0
    spacegroup: int = 1
    max_z: int = 26
    avg_z: float = 26.0
    volume_bohr3: float = 100.0
    is_soc: bool = False
    is_hybrid: bool = False
    # --- Hardware context at run time ---
    cpu_arch: str = ""
    cpu_generation: str = ""
    peak_gflops: float = 0.0
    mem_bandwidth_gbs: float = 0.0
    numa_nodes: int = 1
    interconnect_type: str = ""
    scratch_fs: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with JSON-safe types."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRecord":
        """Deserialize from dictionary with proper type coercion."""
        record = cls()
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return record


# =============================================================================
# Schema Definition
# =============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS execution_history (
    run_id              TEXT PRIMARY KEY,
    timestamp           REAL NOT NULL,
    backend             TEXT NOT NULL,
    mode                TEXT NOT NULL,
    nmat                INTEGER NOT NULL,
    nkpt                INTEGER NOT NULL,
    atoms               INTEGER NOT NULL,
    rkmax               REAL NOT NULL,
    total_cores         INTEGER NOT NULL,
    omp_threads         INTEGER NOT NULL,
    nodes_used          INTEGER NOT NULL,
    walltime_sec        REAL NOT NULL,
    efficiency_pct      REAL NOT NULL DEFAULT 0.0,
    convergence_cycles  INTEGER NOT NULL DEFAULT 0,
    memory_gb_used      REAL NOT NULL DEFAULT 0.0,
    node_list           TEXT NOT NULL DEFAULT '[]',
    success             INTEGER NOT NULL DEFAULT 0,
    tags                TEXT NOT NULL DEFAULT '[]',
    -- Structural features
    nbands              INTEGER NOT NULL DEFAULT 0,
    spacegroup          INTEGER NOT NULL DEFAULT 1,
    max_z               INTEGER NOT NULL DEFAULT 26,
    avg_z               REAL NOT NULL DEFAULT 26.0,
    volume_bohr3        REAL NOT NULL DEFAULT 100.0,
    is_soc              INTEGER NOT NULL DEFAULT 0,
    is_hybrid           INTEGER NOT NULL DEFAULT 0,
    -- Hardware context at run time
    cpu_arch            TEXT NOT NULL DEFAULT '',
    cpu_generation      TEXT NOT NULL DEFAULT '',
    peak_gflops         REAL NOT NULL DEFAULT 0.0,
    mem_bandwidth_gbs   REAL NOT NULL DEFAULT 0.0,
    numa_nodes          INTEGER NOT NULL DEFAULT 1,
    interconnect_type   TEXT NOT NULL DEFAULT '',
    scratch_fs          TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_history_backend ON execution_history(backend);
CREATE INDEX IF NOT EXISTS idx_history_mode ON execution_history(mode);
CREATE INDEX IF NOT EXISTS idx_history_nmat ON execution_history(nmat);
CREATE INDEX IF NOT EXISTS idx_history_nkpt ON execution_history(nkpt);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON execution_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_history_success ON execution_history(success);
CREATE INDEX IF NOT EXISTS idx_history_walltime ON execution_history(walltime_sec);
CREATE INDEX IF NOT EXISTS idx_history_cores ON execution_history(total_cores);
CREATE INDEX IF NOT EXISTS idx_history_problem ON execution_history(nmat, nkpt, backend);
CREATE INDEX IF NOT EXISTS idx_history_cpu_arch ON execution_history(cpu_arch);
CREATE INDEX IF NOT EXISTS idx_history_spacegroup ON execution_history(spacegroup);
"""

_NEW_COLUMNS_MIGRATION = [
    ("nbands",              "INTEGER NOT NULL DEFAULT 0"),
    ("spacegroup",          "INTEGER NOT NULL DEFAULT 1"),
    ("max_z",               "INTEGER NOT NULL DEFAULT 26"),
    ("avg_z",               "REAL NOT NULL DEFAULT 26.0"),
    ("volume_bohr3",        "REAL NOT NULL DEFAULT 100.0"),
    ("is_soc",              "INTEGER NOT NULL DEFAULT 0"),
    ("is_hybrid",           "INTEGER NOT NULL DEFAULT 0"),
    ("cpu_arch",            "TEXT NOT NULL DEFAULT ''"),
    ("cpu_generation",      "TEXT NOT NULL DEFAULT ''"),
    ("peak_gflops",         "REAL NOT NULL DEFAULT 0.0"),
    ("mem_bandwidth_gbs",   "REAL NOT NULL DEFAULT 0.0"),
    ("numa_nodes",          "INTEGER NOT NULL DEFAULT 1"),
    ("interconnect_type",   "TEXT NOT NULL DEFAULT ''"),
    ("scratch_fs",          "TEXT NOT NULL DEFAULT ''"),
]

_COLUMN_LIST = [
    "run_id", "timestamp", "backend", "mode", "nmat", "nkpt", "atoms",
    "rkmax", "total_cores", "omp_threads", "nodes_used", "walltime_sec",
    "efficiency_pct", "convergence_cycles", "memory_gb_used",
    "node_list", "success", "tags",
    "nbands", "spacegroup", "max_z", "avg_z", "volume_bohr3",
    "is_soc", "is_hybrid",
    "cpu_arch", "cpu_generation", "peak_gflops", "mem_bandwidth_gbs",
    "numa_nodes", "interconnect_type", "scratch_fs",
]


def _row_to_record(row: tuple) -> ExecutionRecord:
    """Convert a database row tuple to an ExecutionRecord instance."""
    def _safe_bool(val): return bool(val) if val is not None else False
    def _safe_json(val): return json.loads(val) if isinstance(val, str) and val else val if val else []
    return ExecutionRecord(
        run_id=row[0],
        timestamp=row[1],
        backend=row[2],
        mode=row[3],
        nmat=row[4],
        nkpt=row[5],
        atoms=row[6],
        rkmax=row[7],
        total_cores=row[8],
        omp_threads=row[9],
        nodes_used=row[10],
        walltime_sec=row[11],
        efficiency_pct=row[12],
        convergence_cycles=row[13],
        memory_gb_used=row[14],
        node_list=_safe_json(row[15]),
        success=_safe_bool(row[16]),
        tags=_safe_json(row[17]),
        nbands=row[18] if len(row) > 18 and row[18] else 0,
        spacegroup=row[19] if len(row) > 19 and row[19] else 1,
        max_z=row[20] if len(row) > 20 and row[20] else 26,
        avg_z=row[21] if len(row) > 21 and row[21] else 26.0,
        volume_bohr3=row[22] if len(row) > 22 and row[22] else 100.0,
        is_soc=_safe_bool(row[23] if len(row) > 23 else False),
        is_hybrid=_safe_bool(row[24] if len(row) > 24 else False),
        cpu_arch=row[25] if len(row) > 25 and row[25] else "",
        cpu_generation=row[26] if len(row) > 26 and row[26] else "",
        peak_gflops=row[27] if len(row) > 27 and row[27] else 0.0,
        mem_bandwidth_gbs=row[28] if len(row) > 28 and row[28] else 0.0,
        numa_nodes=row[29] if len(row) > 29 and row[29] else 1,
        interconnect_type=row[30] if len(row) > 30 and row[30] else "",
        scratch_fs=row[31] if len(row) > 31 and row[31] else "",
    )


# =============================================================================
# ExecutionHistory Class
# =============================================================================

class ExecutionHistory:
    """
    Thread-safe SQLite-backed store for WIEN2k execution records.
    Provides connection pooling, a context-manager interface, and a rich
    query API for analysing past runs and guiding future configurations.

    Usage:
        with ExecutionHistory() as history:
            history.record(ExecutionRecord(...))
            similar = history.get_similar(nmat=5000, nkpt=4, backend="wien2k")
            stats = history.get_statistics()
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the history store.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.forge/history.db.
        """
        if db_path is None:
            db_path = DEFAULT_HISTORY_DB
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._connections: list[sqlite3.Connection] = []
        self._closed = False
        self._init_db()

    def _init_db(self) -> None:
        """Create database directory and schema on first access."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = self._get_connection()
            with conn:
                conn.executescript(_SCHEMA_SQL)
                self._migrate_schema(conn)
            logger.debug(f"Database schema initialized at {self.db_path}")
        finally:
            if conn:
                self._return_connection(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Add new columns to existing databases (idempotent)."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(execution_history)")}
        for col_name, col_def in _NEW_COLUMNS_MIGRATION:
            if col_name not in existing:
                try:
                    conn.execute(f"ALTER TABLE execution_history ADD COLUMN {col_name} {col_def}")
                    logger.debug(f"Migrated: added column {col_name} to execution_history")
                except sqlite3.OperationalError as e:
                    logger.debug(f"Migration skip for {col_name}: {e}")

    def _get_connection(self) -> sqlite3.Connection:
        """Obtain a connection from the pool or create a new one."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ExecutionHistory has been closed")
            if self._connections:
                conn = self._connections.pop()
                try:
                    conn.execute("SELECT 1")
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    conn = self._create_connection()
                return conn
            return self._create_connection()

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=10.0,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool or close if pool is full."""
        with self._lock:
            if len(self._connections) < MAX_POOL_SIZE:
                self._connections.append(conn)
            else:
                with suppress(Exception):
                    conn.close()

    @contextmanager
    def _conn(self):
        """Context manager for safe connection acquisition and return."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "ExecutionHistory":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close all pooled connections and mark as closed."""
        with self._lock:
            self._closed = True
            for conn in self._connections:
                with suppress(Exception):
                    conn.close()
            self._connections.clear()

    # =========================================================================
    # Public CRUD API
    # =========================================================================

    def record(self, record: ExecutionRecord) -> str:
        """
        Insert a new execution record into the database.

        Args:
            record: The ExecutionRecord to persist.

        Returns:
            The run_id of the inserted record.
        """
        sql = f"""
            INSERT OR REPLACE INTO execution_history
                ({', '.join(_COLUMN_LIST)})
            VALUES ({', '.join(['?'] * len(_COLUMN_LIST))})
        """
        values = (
            record.run_id,
            record.timestamp,
            record.backend,
            record.mode,
            record.nmat,
            record.nkpt,
            record.atoms,
            record.rkmax,
            record.total_cores,
            record.omp_threads,
            record.nodes_used,
            record.walltime_sec,
            record.efficiency_pct,
            record.convergence_cycles,
            record.memory_gb_used,
            json.dumps(record.node_list),
            int(record.success),
            json.dumps(record.tags),
            record.nbands,
            record.spacegroup,
            record.max_z,
            record.avg_z,
            record.volume_bohr3,
            int(record.is_soc),
            int(record.is_hybrid),
            record.cpu_arch,
            record.cpu_generation,
            record.peak_gflops,
            record.mem_bandwidth_gbs,
            record.numa_nodes,
            record.interconnect_type,
            record.scratch_fs,
        )
        with self._conn() as conn:
            conn.execute(sql, values)
            conn.commit()
        logger.debug(f"Recorded execution {record.run_id} (mode={record.mode}, cores={record.total_cores})")
        return record.run_id

    def query(self, filters: Optional[dict[str, Any]] = None,
              order_by: Optional[str] = None,
              limit: Optional[int] = None) -> list[ExecutionRecord]:
        """
        Flexible parameterized query against execution history.

        Args:
            filters: Dictionary of column-value pairs for WHERE clause.
                     Supports lists for IN (...) queries.  Supports '__lt',
                     '__gt', '__lte', '__gte' suffixes for comparisons.
            order_by: Column name with optional 'ASC'/'DESC' suffix (e.g. 'walltime_sec DESC').
            limit: Maximum number of rows to return.

        Returns:
            List of matching ExecutionRecord instances.
        """
        sql = "SELECT * FROM execution_history"
        params: list[Any] = []

        if filters:
            clauses: list[str] = []
            for key, value in filters.items():
                comparison = "="
                col = key
                for suffix in ("__lt", "__gt", "__lte", "__gte"):
                    if key.endswith(suffix):
                        col = key[:-len(suffix)]
                        suffix_map = {"__lt": "<", "__gt": ">", "__lte": "<=", "__gte": ">="}
                        comparison = suffix_map[suffix]
                        break

                if col not in _COLUMN_LIST:
                    logger.warning(f"Unknown column '{col}' in query filter; skipping")
                    continue

                if isinstance(value, (list, tuple)):
                    placeholders = ", ".join(["?"] * len(value))
                    clauses.append(f"{col} IN ({placeholders})")
                    params.extend(value)
                else:
                    clauses.append(f"{col} {comparison} ?")
                    params.append(value)

            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

        if order_by:
            sql += f" ORDER BY {order_by}"
        else:
            sql += " ORDER BY timestamp DESC"

        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        with self._conn() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [_row_to_record(tuple(row)) for row in rows]

    def get_similar(
        self,
        nmat: int,
        nkpt: int,
        backend: str,
        limit: int = 10,
        require_success: bool = True,
    ) -> list[ExecutionRecord]:
        """
        Find historically similar runs based on problem size and backend.

        Similarity is defined by proximity in (nmat, nkpt) space within
        a tolerance band, ordered by closest match first.

        Args:
            nmat: Hamiltonian matrix size.
            nkpt: Number of k-points.
            backend: DFT backend name (e.g. 'wien2k', 'vasp').
            limit: Maximum number of results to return.
            require_success: If True, only return successful runs.

        Returns:
            List of ExecutionRecord instances ordered by similarity.
        """
        nmat_low = int(nmat * 0.5)
        nmat_high = int(nmat * 2.0)
        nkpt_low = max(1, int(nkpt * 0.5))
        nkpt_high = max(1, int(nkpt * 2.0) + 1)

        sql = """
            SELECT * FROM execution_history
            WHERE backend = ?
              AND nmat BETWEEN ? AND ?
              AND nkpt BETWEEN ? AND ?
        """
        params: list[Any] = [backend, nmat_low, nmat_high, nkpt_low, nkpt_high]

        if require_success:
            sql += " AND success = 1"

        # Score: absolute distance in normalised (nmat, nkpt) space
        sql += """
            ORDER BY
                ABS(nmat - ?) * 1.0 / MAX(1, nmat) +
                ABS(nkpt - ?) * 1.0 / MAX(1, nkpt) ASC
            LIMIT ?
        """
        params.extend([nmat, nkpt, int(limit)])

        with self._conn() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [_row_to_record(tuple(row)) for row in rows]

    def get_best_config(
        self,
        nmat: int,
        nkpt: int,
        backend: str,
    ) -> Optional[ExecutionRecord]:
        """
        Retrieve the historically best configuration for a given problem.

        "Best" is defined as the successful run with the lowest walltime
        among records whose nmat and nkpt are within a factor of 2.

        Args:
            nmat: Hamiltonian matrix size.
            nkpt: Number of k-points.
            backend: DFT backend name.

        Returns:
            The best ExecutionRecord, or None if no similar successful run exists.
        """
        similar = self.get_similar(nmat, nkpt, backend, limit=50, require_success=True)
        if not similar:
            return None

        best = min(similar, key=lambda r: r.walltime_sec if r.walltime_sec > 0 else float("inf"))
        return best

    def get_statistics(self) -> dict[str, Any]:
        """
        Compute aggregate statistics over all recorded runs.

        Returns:
            Dictionary with keys:
            - total_runs, successful_runs, failed_runs
            - avg_walltime_sec, min_walltime_sec, max_walltime_sec
            - avg_cores, avg_omp, avg_nmat
            - mode_distribution (count per parallelisation mode)
            - backend_distribution (count per backend)
            - common_node_count (most frequent node count)
        """
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM execution_history").fetchone()[0]
            successful = conn.execute(
                "SELECT COUNT(*) FROM execution_history WHERE success = 1"
            ).fetchone()[0]

            stats: dict[str, Any] = {
                "total_runs": total,
                "successful_runs": successful,
                "failed_runs": total - successful,
            }

            if total == 0:
                return stats

            row = conn.execute("""
                SELECT
                    AVG(walltime_sec), MIN(walltime_sec), MAX(walltime_sec),
                    AVG(total_cores), AVG(omp_threads), AVG(nmat),
                    AVG(efficiency_pct), AVG(convergence_cycles)
                FROM execution_history
            """).fetchone()
            stats["avg_walltime_sec"] = round(row[0], 2) if row[0] else 0.0
            stats["min_walltime_sec"] = round(row[1], 2) if row[1] else 0.0
            stats["max_walltime_sec"] = round(row[2], 2) if row[2] else 0.0
            stats["avg_cores"] = round(row[3], 1) if row[3] else 0.0
            stats["avg_omp"] = round(row[4], 1) if row[4] else 0.0
            stats["avg_nmat"] = round(row[5], 1) if row[5] else 0.0
            stats["avg_efficiency_pct"] = round(row[6], 2) if row[6] else 0.0
            stats["avg_convergence_cycles"] = round(row[7], 1) if row[7] else 0.0

            # Mode distribution
            mode_rows = conn.execute("""
                SELECT mode, COUNT(*) as cnt
                FROM execution_history
                GROUP BY mode ORDER BY cnt DESC
            """).fetchall()
            stats["mode_distribution"] = {r[0]: r[1] for r in mode_rows}

            # Backend distribution
            backend_rows = conn.execute("""
                SELECT backend, COUNT(*) as cnt
                FROM execution_history
                GROUP BY backend ORDER BY cnt DESC
            """).fetchall()
            stats["backend_distribution"] = {r[0]: r[1] for r in backend_rows}

            # Most common node count
            node_row = conn.execute("""
                SELECT nodes_used, COUNT(*) as cnt
                FROM execution_history
                GROUP BY nodes_used ORDER BY cnt DESC LIMIT 1
            """).fetchone()
            stats["common_node_count"] = node_row[0] if node_row else 1

            return stats

    def delete_older_than(self, days: int) -> int:
        """
        Purge records older than the specified number of days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of deleted records.
        """
        cutoff = time.time() - (days * 86400.0)
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM execution_history WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount


# =============================================================================
# Helper Functions
# =============================================================================

def compute_efficiency(
    walltime: float,
    total_cores: int,
    nmat: int,
    nkpt: int,
) -> float:
    """
    Compute a normalised efficiency metric for comparing runs of different sizes.

    The metric accounts for the O(N^3) scaling of Hamiltonian diagonalisation
    and the linear scaling with k-point count.  Lower is better.
    A value near 1.0 indicates excellent efficiency for the given problem size.

    Formula:
        workload = nmat^3 * nkpt
        efficiency = workload / (walltime * total_cores)  [normalised]

    Args:
        walltime: Wall-clock duration in seconds.
        total_cores: Total CPU cores used.
        nmat: Hamiltonian matrix dimension.
        nkpt: Number of k-points.

    Returns:
        Normalised efficiency score (float, higher is better).
    """
    if walltime <= 0 or total_cores <= 0 or nmat <= 0:
        return 0.0

    workload = float(nmat) ** 3.0 * float(max(1, nkpt))
    efficiency = workload / (walltime * float(total_cores))

    # Normalise to a reference: nmat=5000, nkpt=4 on 64 cores taking 3600 s
    ref_workload = 5000.0 ** 3.0 * 4.0
    ref_efficiency = ref_workload / (3600.0 * 64.0)
    normalised = efficiency / ref_efficiency if ref_efficiency > 0 else efficiency

    return round(min(normalised, 1000.0), 4)


def suggest_from_history(
    nmat: int,
    nkpt: int,
    backend: str,
    topo_cores: int,
) -> dict[str, Any]:
    """
    Derive a configuration suggestion from historically successful runs.

    Analyses past runs with similar problem sizes and returns a recommended
    mode, core count, and OMP thread count based on walltime performance
    and success rate.

    Args:
        nmat: Hamiltonian matrix size.
        nkpt: Number of k-points.
        backend: DFT backend name.
        topo_cores: Total cores available in the current topology.

    Returns:
        Dictionary with keys:
        - suggested_mode: str ('kpoint', 'hybrid', 'mpi')
        - suggested_cores: int
        - suggested_omp: int
        - confidence: float (0.0-1.0, based on quantity and recency of data)
        - source: str ('history', 'default')
        - num_relevant_records: int
    """
    db_path = DEFAULT_HISTORY_DB
    if not db_path.exists():
        return {
            "suggested_mode": "hybrid",
            "suggested_cores": max(1, topo_cores),
            "suggested_omp": min(4, max(1, topo_cores // 2)),
            "confidence": 0.0,
            "source": "default",
            "num_relevant_records": 0,
        }

    with ExecutionHistory(db_path) as history:
        candidates = history.get_similar(nmat, nkpt, backend, limit=100, require_success=True)

        if not candidates:
            # Fall back to any successful run for this backend
            candidates = history.query(
                filters={"backend": backend, "success": True},
                order_by="timestamp DESC",
                limit=50,
            )

        if not candidates:
            return {
                "suggested_mode": "hybrid",
                "suggested_cores": max(1, topo_cores),
                "suggested_omp": min(4, max(1, topo_cores // 2)),
                "confidence": 0.0,
                "source": "default",
                "num_relevant_records": 0,
            }

        # Count mode popularity weighted by recency and efficiency
        now = time.time()
        mode_scores: dict[str, float] = {"kpoint": 0.0, "hybrid": 0.0, "mpi": 0.0}

        for rec in candidates:
            age_days = max(1, (now - rec.timestamp) / 86400.0)
            recency_weight = 1.0 / (1.0 + max(0, age_days - 30) / 30.0)
            efficiency_weight = max(0.01, rec.efficiency_pct / 100.0) if rec.efficiency_pct > 0 else 0.5
            walltime_weight = 1.0 / max(1.0, rec.walltime_sec / 3600.0)

            weight = recency_weight * efficiency_weight * walltime_weight
            mode_scores[rec.mode] += weight

        best_mode = max(mode_scores, key=mode_scores.get)

        # Extract core and OMP from best-mode runs
        mode_runs = [r for r in candidates if r.mode == best_mode]
        if mode_runs:
            # Weight cores by inverse walltime
            weighted_cores = sum(
                r.total_cores / max(1.0, r.walltime_sec / 60.0) for r in mode_runs
            )
            weighted_omp = sum(
                r.omp_threads / max(1.0, r.walltime_sec / 60.0) for r in mode_runs
            )
            weight_sum = sum(1.0 / max(1.0, r.walltime_sec / 60.0) for r in mode_runs)
            avg_cores = int(weighted_cores / max(weight_sum, 0.001))
            avg_omp = int(weighted_omp / max(weight_sum, 0.001))
        else:
            avg_cores = topo_cores
            avg_omp = 1

        suggested_cores = max(1, min(avg_cores, topo_cores))
        suggested_omp = max(1, min(avg_omp, 64))

        # Ensure product does not exceed available cores
        if suggested_cores * suggested_omp > topo_cores:
            suggested_cores = max(1, topo_cores // suggested_omp)

        confidence = min(
            1.0,
            len(candidates) / 20.0 + 0.1,  # More data = higher confidence
        )

        return {
            "suggested_mode": best_mode,
            "suggested_cores": suggested_cores,
            "suggested_omp": suggested_omp,
            "confidence": round(confidence, 3),
            "source": "history",
            "num_relevant_records": len(candidates),
        }


# =============================================================================
# Explicit Public API Declaration
# =============================================================================

__all__ = [
    "DEFAULT_HISTORY_DB",
    "DEFAULT_HISTORY_DIR",
    "ExecutionHistory",
    "ExecutionRecord",
    "compute_efficiency",
    "suggest_from_history",
]
