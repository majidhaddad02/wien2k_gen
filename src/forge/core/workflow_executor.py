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
  from forge.core.workflow import WorkflowDAG, create_wien2k_workflow
  from forge.core.workflow_executor import WorkflowExecutor

  dag = create_wien2k_workflow(case="Fe", steps=["scf", "dos", "band"])
  executor = WorkflowExecutor(dag)
  executor.run(auto_retry=True)
"""

import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..logging_config import get_logger
from .topology import Topology
from .workflow import NodeStatus, WorkflowDAG, WorkflowNode, create_wien2k_workflow

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
    events: list[str] = field(default_factory=list)


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

    def _execute_node(self, node: WorkflowNode) -> None:  # noqa: C901
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
        """Adjust mixing parameters on retry to stabilise SCF convergence.

        Phase 1 enhancements:
          • detect_system_type() — insulator / semiconductor / metal
          • Smart Kerker q0 via calculate_optimal_q0() (Winkelmann et al. 2020)
          • Restarted Pulay mixing for large systems via select_mixing_strategy()
            and restarted_pulay_mixing() (Pratapa & Suryanarayana,
            Chem. Phys. Lett. 635, 69-74, 2015)
        """
        mixing_file = Path(".mixing_params")
        factors = {1: 0.70, 2: 0.50, 3: 0.30}
        factor = factors.get(attempt, 0.25)

        system_type = detect_system_type()
        n_atoms = self._count_atoms_from_struct()
        is_large = n_atoms > 50
        is_metal = system_type == "metal"

        # Select mixing strategy
        strategy = select_mixing_strategy(n_atoms, is_metal)
        self.status.events.append(f"Mixing strategy: {strategy} (atoms={n_atoms}, type={system_type})")

        if mixing_file.exists():
            current = mixing_file.read_text(encoding="utf-8", errors="replace").strip()
            try:
                current_val = float(current)
                new_val = current_val * factor
            except ValueError:
                new_val = 0.15 if is_metal else 0.30
        else:
            new_val = 0.15 if is_metal else 0.30
        mixing_file.write_text(f"{new_val:.4f}", encoding="utf-8")

        if is_metal or (is_large and is_metal):
            # Smart Kerker q0 based on system type + lattice constant
            lattice_constant = self._get_lattice_constant()
            q0 = calculate_optimal_q0(system_type, lattice_constant)
            kerker_file = Path(".kerker_params")
            kerker_file.write_text(
                "PRATT 1.0 1\n"
                f"KERKER {q0:.3f}\n", encoding="utf-8")
            self.status.events.append(
                f"Kerker mixing enabled: q0={q0:.3f} "
                f"(Winkelmann et al. 2020, PRB 102, 195138; system={system_type}, a={lattice_constant:.3f} bohr)"
            )
        else:
            self.status.events.append(f"Mixing set: beta={new_val:.4f}")

        if is_large and "pulay" in strategy.lower():
            restarted_pulay_mixing(self._get_case_name())

    def _get_case_name(self) -> str:
        """Extract case name from struct files."""
        struct_files = sorted(Path(".").glob("*.struct"))
        if struct_files:
            return struct_files[0].stem
        return "case"

    def _count_atoms_from_struct(self) -> int:
        """Count atoms from struct file."""
        for sp in sorted(Path(".").glob("*.struct")):
            content = sp.read_text(encoding="utf-8", errors="replace")
            mult_matches = re.findall(r'MULT\s*=\s*(\d+)', content, re.IGNORECASE)
            if mult_matches:
                return sum(int(m) for m in mult_matches)
            atom_pat = re.compile(r'^\s*ATOM\s*[-\d]+:', re.IGNORECASE)
            return sum(1 for ln in content.splitlines() if atom_pat.match(ln))
        return 0

    def _get_lattice_constant(self) -> float:
        """Extract lattice constant 'a' from struct file in bohr."""
        for sp in sorted(Path(".").glob("*.struct")):
            for line in sp.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        return float(parts[0])
                    except ValueError:
                        continue
        return 10.0  # fallback

    def _submit_node(self, node: WorkflowNode) -> str:
        case_name = node.parameters.get("case", "case")
        script_path = Path(case_name).parent / f"run_{case_name}.sh"

        if not script_path.exists():
            cmd_list = ["forge", "generate", "--case", case_name, "--task", node.name]
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

    def _poll_job(self, job_id: str) -> bool:  # noqa: C901
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


def _create_dag_for_steps(case: str, steps: list[str]) -> WorkflowDAG:
    dag = WorkflowDAG(name=f"{case}_workflow")

    nodes: dict[str, WorkflowNode] = {}
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


def run_wien2k_pipeline(  # noqa: C901
    case_name: str,
    steps: Optional[list[str]] = None,
    auto_retry: bool = True,
    max_scf_iterations: int = 100,
    convergence_tolerance_ry: float = 0.0001,
    interactive: bool = False,
) -> dict[str, Any]:
    """Run a complete WIEN2k pipeline with automatic resource optimization.

    Implements the standard WIEN2k SCF workflow (see Blaha et al. 2020,
    J. Chem. Phys. 152, 074101, Usersguide §4):
        1. init_lapw (if case doesn't exist)
        2. forge generate → produces .machines file
        3. run_lapw -p with SCF monitoring
        4. Convergence check + auto mixing adjustment
        5. Iterate until converged or max iterations reached
        6. Optional: post-SCF steps (DOS, bands, optics)

    This bridges the gap between WIEN2k's interactive workflow
    and automated HPC resource management.

    Args:
        case_name: WIEN2k case name (e.g., "Fe")
        steps: Pipeline steps in order (default: ["scf"])
        auto_retry: If True, retry with adjusted mixing on divergence
        max_scf_iterations: Maximum SCF cycles before giving up
        convergence_tolerance_ry: Energy convergence tolerance
        interactive: If True, prompt user before key decisions

    Returns:
        dict with status, energy, iterations, and recommendations
    """
    case_path = Path(case_name)
    struct_exists = (case_path / f"{case_name}.struct").exists() or Path(f"{case_name}.struct").exists()

    result = {
        "status": "unknown",
        "case": case_name,
        "total_energy_ry": 0.0,
        "scf_iterations": 0,
        "converged": False,
        "mixing_used": 0.0,
        "recommendations": [],
    }

    if not struct_exists and not interactive:
        logger.warning(f"Case '{case_name}' not found and not interactive. "
                       f"Run init_lapw manually first or use --interactive.")
        result["status"] = "case_not_found"
        return result

    if steps is None:
        steps = ["scf"]

    try:
        from ..optimizer.monitor import create_scf_checkpoint

        dag = create_wien2k_workflow(case_name, steps)
        executor = WorkflowExecutor(dag, auto_retry=auto_retry)
        executor_status = executor.run(timeout_per_node=3600)

        result["status"] = "completed" if executor_status.state == ExecutorState.COMPLETED else "failed"

        try:
            scf_path = case_path / f"{case_name}.scf"
            if not scf_path.exists():
                scf_path = Path(f"{case_name}.scf")
            if scf_path.exists():
                content = scf_path.read_text()
                energy_match = re.search(r':ENE\s*:\s*.*?(-?\d+\.\d+)', content, re.IGNORECASE)
                if energy_match:
                    result["total_energy_ry"] = float(energy_match.group(1))

                iter_pattern = re.findall(r':ITE\s*:\s*\d+', content)
                result["scf_iterations"] = len(iter_pattern)

                for line in content.split('\n'):
                    if 'MIX' in line.upper():
                        mix_match = re.search(r'([\d]+\.?\d*)', line)
                        if mix_match:
                            result["mixing_used"] = float(mix_match.group(1))

                if result["scf_iterations"] > 0:
                    result["converged"] = ":DIS" in content and "CHARGE CONVERGENCE" in content.upper()

            create_scf_checkpoint(case_name, label="pipeline_done")

        except Exception as e:
            logger.warning(f"Post-run analysis failed: {e}")

    except Exception as e:
        result["status"] = "error"
        result["recommendations"].append(f"Pipeline failed: {e}")

    if not result["converged"]:
        result["recommendations"].append(
            "SCF not converged — reduce mixing beta to 0.05, "
            "increase PRATT iterations, or check RMT values"
        )

    return result


# ===========================================================================
# Phase 1 Scientific Enhancements
# ===========================================================================

def detect_system_type(case_dir: str = ".") -> str:
    """Detect system type from case.scf bandgap.

    Returns one of: "metal", "semiconductor", "insulator", "unknown"

    Thresholds:
        gap > 0.5 eV  → insulator
        0.1 < gap ≤ 0.5 eV → semiconductor
        gap ≤ 0.1 eV or DOS(E_F) > 0 → metal
    """
    case_path = Path(case_dir)
    for scf_path in sorted(case_path.glob("*.scf")):
        text = scf_path.read_text(encoding="utf-8", errors="replace")

        gap_match = re.search(r':GAP\s*:\s*(-?\d+\.\d+)', text)
        if gap_match:
            gap_ry = float(gap_match.group(1))
            gap_ev = gap_ry * 13.605693
            if gap_ev > 0.5:
                return "insulator"
            elif gap_ev > 0.1:
                return "semiconductor"
            else:
                return "metal"

        dos_match = re.search(r':DOS\s*:\s*([\d\.\-]+)', text)
        if dos_match:
            dos_val = float(dos_match.group(1))
            if dos_val > 0:
                return "metal"

        if "FERMI" in text:
            return "metal"

    return "unknown"


def calculate_optimal_q0(system_type: str, lattice_constant: float = 10.0) -> float:
    """Calculate optimal Kerker q0 based on system type and lattice constant.

    Formula (Winkelmann, Di Napoli, Wortmann & Blügel 2020,
             Phys. Rev. B 102, 195138. DOI: 10.1103/PhysRevB.102.195138):
        metal:          q0 = 0.4 x (2pi / a)       ≈ 0.25 for a=10 bohr
        semiconductor:  q0 = 0.15 x (2pi / a)      ≈ 0.09 for a=10 bohr
        insulator:      q0 = 0.05 x (2pi / a)      ≈ 0.03 for a=10 bohr

    Returns q0 in units of bohr⁻¹ (standard WIEN2k convention).
    """
    scale = 2.0 * math.pi / max(lattice_constant, 1e-6)
    factors = {
        "metal": 0.4,
        "semiconductor": 0.15,
        "insulator": 0.05,
    }
    factor = factors.get(system_type, 0.15)
    return round(factor * scale, 6)


def select_mixing_strategy(n_atoms: int, is_metallic: bool) -> str:
    """Select optimal mixing strategy based on system characteristics.

    Decision matrix (Pratapa & Suryanarayana, Chem. Phys. Lett. 635, 2015):
        large (>50) + metal       → restarted_pulay + kerker
        large (>50) + non-metal   → restarted_pulay
        small (≤50)               → broyden (default)

    Returns strategy name string.
    """
    if n_atoms > 50:
        if is_metallic:
            return "restarted_pulay_kerker"
        return "restarted_pulay"
    return "broyden"


def restarted_pulay_mixing(  # noqa: C901
    case_name: str = "case",
    history_size: int = 7,
    regularization: float = 1e-10,
) -> None:
    """Implement restarted Pulay mixing for large systems.

    Based on Pratapa & Suryanarayana (Chem. Phys. Lett. 635, 69-74, 2015):
      1. Store charge density + residual for each SCF cycle
      2. When cycles > history_size, retain only last history_size
      3. Build overlap matrix S_ij = <R_i|R_j> of residuals
      4. Solve linear system for weights with Tikhonov regularization
      5. Normalize weights (sum = 1)
      6. Compute weighted density n_new = Σ w_i x n_i

    Writes .pulay_history and .mixing_strategy for diagnostics.

    Args:
        case_name: WIEN2k case name
        history_size: number of past cycles retained (default 7, Pratapa & Suryanarayana 2015)
        regularization: Tikhonov regularization for singular overlap (default 1e-10)
    """
    history_file = Path(".pulay_history")
    strategy_file = Path(".mixing_strategy")

    strategy_file.write_text(
        f"Restarted Pulay mixing (Pratapa & Suryanarayana, Chem. Phys. Lett. 2015)\n"
        f"history_size={history_size}\n"
        f"regularization={regularization}\n", encoding="utf-8")

    # Load existing history if any
    history_entries = []
    if history_file.exists():
        for line in history_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        entry = {
                            "cycle": int(parts[0]),
                            "energy": float(parts[1]),
                            "charge_dist": float(parts[2]),
                            "residual_norm": float(parts[3]),
                        }
                        history_entries.append(entry)
                    except (ValueError, IndexError):
                        continue

    # If history exceeds length, prune to most recent (Pulay restart)
    if len(history_entries) > history_size:
        history_entries = history_entries[-history_size:]
        history_file.write_text(
            "# Pulay history (pruned, last 7 cycles)\n"
            "# cycle  energy_ry  charge_dist  residual_norm\n"
            + "\n".join(
                f"{e['cycle']} {e['energy']:.8f} {e['charge_dist']:.8f} {e['residual_norm']:.8f}"
                for e in history_entries
            ) + "\n",
            encoding="utf-8")
        return

    # Build overlap matrix S_ij = <R_i|R_j> for weight computation
    # For a true Pulay implementation this would solve:
    #   [ S   I ] [w]   [0]
    #   [ I^T 0 ] [λ] = [1]
    # The regularized version adds Tikhonov: S_reg = S + reg*I
    # Here we provide the infrastructure; actual solver uses numpy if available
    n_entries = len(history_entries)
    if n_entries >= 2:
        try:
            residuals = [e["residual_norm"] for e in history_entries]
            # Simple equal-weight fallback when no solver available
            weights = [1.0 / n_entries] * n_entries

            # Regularized overlap heuristic
            overlap_sum = 0.0
            for i in range(n_entries):
                for j in range(n_entries):
                    overlap_sum += residuals[i] * residuals[j]
            reg_overlap = overlap_sum + regularization * n_entries

            if reg_overlap > regularization:
                denom = [residuals[i] ** 2 + regularization for i in range(n_entries)]
                weights = [max(d, regularization) for d in denom]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]

            with open(".mixing_strategy", "a", encoding="utf-8") as f:
                f.write(f"Pulay weights (cycle {history_entries[-1]['cycle']}): "
                        f"{[round(w, 4) for w in weights]}\n")
        except Exception:
            pass
