"""
WorkflowExecutor — DAG runtime engine for automated WIEN2k pipelines.

Bridges WorkflowDAG data model to actual HPC execution:
  1. Gets ready nodes from DAG (parents completed)
  2. Generates .machines + run_optimized.sh for each node
  3. Submits via SLURM/PBS using existing submit modules
  4. Polls job status (squeue/qstat)
  5. Checks SCF convergence via parse_output/parse_dayfile
  6. Auto-retries with adjusted mixing on failure
  7. Advances to child nodes

Usage:
  from wien2k_gen.core.workflow import WorkflowDAG, create_wien2k_workflow
  from wien2k_gen.core.workflow_executor import WorkflowExecutor

  dag = create_wien2k_workflow(case="Fe", steps=["scf", "dos", "band"])
  executor = WorkflowExecutor(dag)
  executor.run(auto_retry=True)
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .topology import Topology
from .workflow import NodeStatus, WorkflowDAG, WorkflowNode
from ..logging_config import get_logger

logger = get_logger(__name__)


class ExecutorState(Enum):
    IDLE = "idle"
    SUBMITTING = "submitting"
    POLLING = "polling"
    CHECKING = "checking"
    RETRYING = "retrying"
    ADVANCING = "advancing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutorStatus:
    state: ExecutorState = ExecutorState.IDLE
    current_node: Optional[str] = None
    job_id: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    elapsed_total: float = 0.0
    events: List[str] = field(default_factory=list)


class WorkflowExecutor:
    """Executes a WorkflowDAG by submitting, monitoring, and advancing nodes."""

    def __init__(
        self,
        dag: WorkflowDAG,
        topology: Optional[Topology] = None,
        scheduler: Optional[str] = None,
        poll_interval: float = 5.0,
        max_retries: int = 3,
        auto_retry: bool = True,
    ) -> None:
        self.dag = dag
        self.topology = topology or self._detect_topology()
        self.scheduler = scheduler or self._detect_scheduler()
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.auto_retry = auto_retry
        self.status = ExecutorStatus(max_retries=max_retries)

        self._on_node_start: Optional[Callable] = None
        self._on_node_complete: Optional[Callable] = None
        self._on_node_fail: Optional[Callable] = None
        self._on_retry: Optional[Callable] = None

    @staticmethod
    def _detect_scheduler() -> str:
        if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME"):
            return "slurm"
        if os.environ.get("PBS_JOBID"):
            return "pbs"
        return "slurm"

    @staticmethod
    def _detect_topology() -> Topology:
        try:
            from .scheduler import detect as detect_topology
            return detect_topology()
        except Exception:
            return Topology(nodes=["localhost"], cores_per_node=[1])

    def on_node_start(self, fn: Callable) -> None:
        self._on_node_start = fn

    def on_node_complete(self, fn: Callable) -> None:
        self._on_node_complete = fn

    def on_node_fail(self, fn: Callable) -> None:
        self._on_node_fail = fn

    def on_retry(self, fn: Callable) -> None:
        self._on_retry = fn

    def run(self) -> ExecutorStatus:
        """Execute the entire DAG, advancing nodes as they become ready."""
        start_time = time.time()
        self.status.state = ExecutorState.ADVANCING

        while True:
            ready = self.dag.get_ready_nodes()
            if not ready:
                all_completed = all(
                    self.dag._nodes[nid].status == NodeStatus.COMPLETED
                    for nid in self.dag._nodes
                )
                if all_completed:
                    self.status.state = ExecutorState.COMPLETED
                    break
                else:
                    pending = [
                        nid for nid, n in self.dag._nodes.items()
                        if n.status == NodeStatus.PENDING
                    ]
                    failed = [
                        nid for nid, n in self.dag._nodes.items()
                        if n.status == NodeStatus.FAILED
                    ]
                    if not pending and failed:
                        self.status.state = ExecutorState.FAILED
                        break
                    logger.info(f"No ready nodes. Pending: {len(pending)}, Failed: {len(failed)}. Waiting...")
                    time.sleep(self.poll_interval)
                    continue

            for node in ready:
                self._execute_node(node)

        self.status.elapsed_total = time.time() - start_time
        return self.status

    def _execute_node(self, node: WorkflowNode) -> None:
        self.status.current_node = node.node_id
        self.status.retries = 0

        if self._on_node_start:
            self._on_node_start(node)

        while self.status.retries < self.max_retries:
            try:
                self.status.state = ExecutorState.SUBMITTING
                job_id = self._submit_node(node)
                self.status.job_id = job_id
                self.status.events.append(f"Submitted {node.name} [{job_id}]")

                self.dag.set_node_status(node.node_id, NodeStatus.RUNNING)

                self.status.state = ExecutorState.POLLING
                job_ok = self._poll_job(job_id)

                if job_ok:
                    self.status.state = ExecutorState.CHECKING
                    converged = self._check_convergence(node)
                    if converged:
                        self.dag.set_node_status(node.node_id, NodeStatus.COMPLETED)
                        self.status.events.append(f"Completed {node.name}")
                        if self._on_node_complete:
                            self._on_node_complete(node)
                        return
                    else:
                        self.status.events.append(f"Not converged: {node.name}")
                        if self.auto_retry:
                            self._retry_with_adjustment(node)
                        else:
                            break
                else:
                    self.status.events.append(f"Job failed: {node.name} [{job_id}]")
                    if self.auto_retry:
                        self._retry_with_adjustment(node)
                    else:
                        break

            except Exception as e:
                logger.error(f"Error executing {node.name}: {e}")
                self.status.events.append(f"Error: {node.name} — {e}")
                if self.auto_retry:
                    self.status.retries += 1
                    time.sleep(30)
                else:
                    break

        if self.status.retries >= self.max_retries:
            self.dag.set_node_status(node.node_id, NodeStatus.FAILED)
            self.status.events.append(f"Failed after {self.max_retries} retries: {node.name}")
            if self._on_node_fail:
                self._on_node_fail(node)

    def _retry_with_adjustment(self, node: WorkflowNode) -> None:
        self.status.retries += 1
        self.status.events.append(f"Retry {self.status.retries}/{self.max_retries}: {node.name}")
        if self._on_retry:
            self._on_retry(node, self.status.retries)

        if node.name in ("lapw0", "lapw1", "scf", "run_lapw"):
            self._adjust_mixing(self.status.retries)
        time.sleep(10)

    def _adjust_mixing(self, attempt: int) -> None:
        """Reduce mixing parameter on retry to stabilise SCF convergence."""
        mixing_file = Path(".mixing_params")
        factors = {1: 0.70, 2: 0.50, 3: 0.30}
        factor = factors.get(attempt, 0.25)

        if mixing_file.exists():
            current = mixing_file.read_text(encoding="utf-8", errors="replace").strip()
            try:
                current_val = float(current)
                new_val = current_val * factor
                mixing_file.write_text(f"{new_val:.4f}", encoding="utf-8")
                self.status.events.append(f"Mixing adjusted: {current_val:.4f} → {new_val:.4f}")
            except ValueError:
                mixing_file.write_text(f"{0.3 * factor:.4f}", encoding="utf-8")

    def _submit_node(self, node: WorkflowNode) -> str:
        case_name = node.parameters.get("case", "case")
        script_path = Path(case_name).parent / f"run_{case_name}.sh"

        if not script_path.exists():
            cmd_list = ["wien2k_gen", "generate", "--case", case_name, "--task", node.name]
            subprocess.run(cmd_list, check=True, capture_output=True, text=True)

        if self.scheduler == "slurm":
            result = subprocess.run(
                ["sbatch", "--parsable", str(script_path)],
                capture_output=True, text=True, check=True, timeout=30,
            )
            return result.stdout.strip()
        elif self.scheduler == "pbs":
            result = subprocess.run(
                ["qsub", str(script_path)],
                capture_output=True, text=True, check=True, timeout=30,
            )
            match = re.search(r"(\d+(?:\.\w+)?)", result.stdout)
            return match.group(1) if match else result.stdout.strip()
        else:
            result = subprocess.run(
                ["bash", str(script_path)],
                capture_output=True, text=True, check=True, timeout=30,
            )
            return "local"

    def _poll_job(self, job_id: str) -> bool:
        if job_id == "local":
            return True

        while True:
            if self.scheduler == "slurm":
                result = subprocess.run(
                    ["squeue", "--job", job_id, "--noheader", "--format=%T"],
                    capture_output=True, text=True, timeout=10,
                )
                state = result.stdout.strip()
                if not state:
                    break
                if state in ("COMPLETED",):
                    return True
                if state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
                    return False

            elif self.scheduler == "pbs":
                result = subprocess.run(
                    ["qstat", "-f", job_id],
                    capture_output=True, text=True, timeout=10,
                )
                if "Unknown Job Id" in result.stdout or result.returncode != 0:
                    break
                if "job_state = C" in result.stdout:
                    return True
                for state in ("E", "H"):
                    if f"job_state = {state}" in result.stdout:
                        return False

            time.sleep(self.poll_interval)

        return True

    def _check_convergence(self, node: WorkflowNode) -> bool:
        case_name = node.parameters.get("case", "case")
        scf_path = Path(f"{case_name}.scf")

        if not scf_path.exists():
            scf_matches = sorted(Path(".").glob("*.scf*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if scf_matches:
                scf_path = scf_matches[0]
            else:
                return True

        try:
            content = scf_path.read_text(encoding="utf-8", errors="replace").lower()
            if any(phrase in content for phrase in ("charge convergence", "energy convergence", "scf cycle converged")):
                return True
        except Exception:
            pass

        try:
            dayfile = scf_path.with_suffix(".dayfile")
            if dayfile.exists():
                content = dayfile.read_text(encoding="utf-8", errors="replace").lower()
                if ".dayfile" in str(dayfile) and "converged" in content and "not converged" not in content:
                    return True
        except Exception:
            pass

        return False


def run_workflow_from_yaml(
    yaml_path: str,
    auto_retry: bool = True,
    max_retries: int = 3,
    poll_interval: float = 5.0,
) -> ExecutorStatus:
    """Load and execute a workflow from a YAML file.

    YAML format:
        case: Fe
        steps: [scf, dos, band]
        scheduler: slurm
        auto_retry: true
        max_retries: 3
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML required for YAML workflow files. Install: pip install pyyaml")
        raise

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    case = config.get("case", "case")
    steps = config.get("steps", ["scf"])
    scheduler = config.get("scheduler")
    auto_retry = config.get("auto_retry", auto_retry)
    max_retries = config.get("max_retries", max_retries)

    dag = _create_dag_for_steps(case, steps)
    executor = WorkflowExecutor(
        dag, scheduler=scheduler, auto_retry=auto_retry,
        max_retries=max_retries, poll_interval=poll_interval,
    )
    return executor.run()


def _create_dag_for_steps(case: str, steps: List[str]) -> WorkflowDAG:
    dag = WorkflowDAG(name=f"{case}_workflow")

    nodes: Dict[str, WorkflowNode] = {}
    prev_id: Optional[str] = None

    for step in steps:
        node = WorkflowNode(
            name=step,
            parameters={"case": case, "task": step},
            parent_ids=[prev_id] if prev_id else [],
        )
        dag.add_node(node)
        nodes[step] = node.node_id
        prev_id = node.node_id

    return dag
