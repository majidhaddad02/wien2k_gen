"""Tests for parallel_options generation and the _write_parallel_options method."""

import tempfile
from pathlib import Path

from wien2k_gen.core.topology import Topology
from wien2k_gen.utils.parallel_options import generate_parallel_options, DEFAULT_OPTIONS


def _make_topo(env_type="slurm", cores_per_node=None, nodes=None):
    if nodes is None:
        nodes = ["node01"]
    if cores_per_node is None:
        cores_per_node = [32]
    return Topology(
        nodes=nodes,
        cores_per_node=cores_per_node,
        env_type=env_type,
        scheduler_hints={"mpi_launcher": "srun", "numa_aware": True},
    )


def test_contains_mpirun_variable() -> None:
    content = generate_parallel_options(_make_topo())
    assert "WIEN_MPIRUN" in content


def test_contains_use_remote() -> None:
    content = generate_parallel_options(_make_topo())
    assert "USE_REMOTE" in content


def test_contains_mpi_remote() -> None:
    content = generate_parallel_options(_make_topo())
    assert "MPI_REMOTE" in content


def test_contains_taskset() -> None:
    content = generate_parallel_options(_make_topo())
    assert "TASKSET" in content


def test_contains_delay() -> None:
    content = generate_parallel_options(_make_topo())
    assert "DELAY" in content


def test_default_options_use_remote_zero() -> None:
    assert DEFAULT_OPTIONS["USE_REMOTE"] == "0"


def test_default_options_mpi_remote_zero() -> None:
    assert DEFAULT_OPTIONS["MPI_REMOTE"] == "0"


def test_default_options_taskset_no() -> None:
    assert DEFAULT_OPTIONS["TASKSET"] == "no"


def test_content_is_non_empty_string() -> None:
    content = generate_parallel_options(_make_topo())
    assert isinstance(content, str)
    assert len(content) > 100


def test_slurm_detected_launcher() -> None:
    content = generate_parallel_options(_make_topo(env_type="slurm"))
    assert "WIEN_MPIRUN" in content


def test_multi_node_topology() -> None:
    content = generate_parallel_options(
        _make_topo(nodes=["n01", "n02", "n03"], cores_per_node=[32, 32, 32])
    )
    assert "WIEN_MPIRUN" in content


def test_single_core_topology() -> None:
    content = generate_parallel_options(_make_topo(cores_per_node=[1]))
    assert len(content) > 0


def test_writes_to_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "parallel_options"
        content = generate_parallel_options(_make_topo())
        path.write_text(content, encoding="utf-8")
        written = path.read_text(encoding="utf-8")
        assert "WIEN_MPIRUN" in written


def test_user_overrides_applied() -> None:
    content = generate_parallel_options(
        _make_topo(),
        user_overrides={"OMP_GLOBAL": "4", "CUSTOM_VAR": "test_value"}
    )
    assert "OMP_GLOBAL" in content


def test_suggestion_mode_kpoint() -> None:
    content = generate_parallel_options(
        _make_topo(),
        suggestion={"mode": "kpoint", "kpar": 8}
    )
    assert "KPAR" in content


def test_suggestion_omp_threads() -> None:
    content = generate_parallel_options(
        _make_topo(),
        suggestion={"omp_threads_per_rank": 4}
    )
    assert "OMP_GLOBAL" in content
